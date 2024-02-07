import numpy as np
import pandas as pd
import umap

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import convert_path_names
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec
from src.jguides_2024.task_event.jguidera_dio_trials import (DioWellDATrials, DioWellDDTrials)
from src.jguides_2024.task_event.jguidera_task_performance import reward_outcomes_to_int
from src.jguides_2024.utils.df_helpers import unpack_df_columns
from src.jguides_2024.utils.kernel_helpers import ExponentialKernel
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals
from src.jguides_2024.utils.set_helpers import check_membership


class UmapContainer:
    """Run UMAP on time series data and store input, parameters, and results"""

    def __init__(self,
                 input_array,
                 n_neighbors=15,
                 n_components=2,
                 input_information=None):
        self.input_array = input_array  # [time, cells]
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.input_information = input_information
        self.embedding, self.reducer = self._run_umap()

    def _run_umap(self):

        reducer = umap.UMAP(n_neighbors=self.n_neighbors,
                            n_components=self.n_components)
        embedding = reducer.fit_transform(self.input_array)

        return embedding, reducer

    def make_embedding_df(self, index_name):

        check_membership([index_name], self.input_information.keys(), "passed index name", "available index names")
        df = pd.DataFrame(self.embedding, columns=[str(x) for x in np.arange(0, self.n_components)],
                          index=self.input_information[index_name])
        df.index.name = index_name

        self.embedding_df = df


class AnnotatedIdxs():

    def __init__(self,
                 nwb_file_name,
                 epochs,
                 external_time_vector,
                 annotations=("trial", "reward_delay"),  # types of annotations to add
                 trial_parameter_name="no_shift",  # for trial tables
                 reward_delay_duration=2,
                 expected_value_params=None):
        self.annotations = annotations
        self.nwb_file_name = nwb_file_name
        self.epochs = epochs
        self.external_time_vector = external_time_vector
        self._check_inputs()

        if "trial" in annotations:
            self.trial_parameter_name = trial_parameter_name
            self.epoch_trial_idxs = self._get_epoch_trial_idxs()
            self.trial_times = self._get_trial_times()
            self.trial_idxs, self.trial_idxs_collapsed = self._get_trial_idxs()
            self.trial_path_names = self._get_trial_path_names()
            self.trial_idxs_by_path, self.trial_idxs_by_path_collapsed = self._get_trial_idxs_by_path()
            if expected_value_params is not None:
                self.expected_value_params = expected_value_params
                (self.trial_overall_expected_value_by_path,
                 self.trial_overall_expected_value_by_path_kernel) = self._get_trial_overall_expected_value_by_path()
                (self.path_expected_value,
                 self.path_expected_value_kernel) = self._get_trial_path_expected_value()

        if "reward_delay" in annotations:
            self.reward_delay_duration = reward_delay_duration
            (self.reward_delay_idxs,
             self.reward_delay_idxs_by_performance,
             self.reward_delay_idxs_by_performance_collapsed) = self._get_reward_delay_idxs()

    def _check_inputs(self):
        valid_annotations = ["trial", "reward_delay"]
        if not all([x in valid_annotations for x in self.annotations]):
            raise Exception(f"All elements of annotations must be in {valid_annotations}")

    def _trials_key(self, epoch):
        return {"nwb_file_name": self.nwb_file_name,
                "epoch": epoch,
                "trial_parameter_name": self.trial_parameter_name}

    def _get_trials_entry(self, epoch):
        return (DioWellDATrials & self._trials_key(epoch)).fetch1()

    def _get_epoch_trial_idxs(self):
        num_epoch_trials = [len(self._get_trials_entry(epoch)["trial_start_times"])
                            for epoch in self.epochs]
        epoch_trials_idxs_start_end = np.concatenate(([0], np.cumsum(num_epoch_trials)))
        epoch_trials_idxs = [np.arange(start_idx, end_idx) for start_idx, end_idx
                             in zip_adjacent_elements(epoch_trials_idxs_start_end)]

        return {k:v for k,v in zip(self.epochs, epoch_trials_idxs)}

    def _get_trial_times(self):
        return np.concatenate([np.asarray(list(zip(self._get_trials_entry(epoch)["trial_start_times"],
                                   self._get_trials_entry(epoch)["trial_end_times"])))
                              for epoch in self.epochs])  # trial start/end times

    def _get_trial_idxs(self):
        trial_idxs = [event_times_in_intervals(event_times=self.external_time_vector,
                                               valid_time_intervals=[trial_times])[0]
                      for trial_times in self.trial_times]
        return trial_idxs, np.concatenate(trial_idxs)

    def _get_trial_path_names(self):
        return np.concatenate([np.asarray(convert_path_names(list(zip(self._get_trials_entry(epoch)["trial_start_well_names"],
                                                      self._get_trials_entry(epoch)["trial_end_well_names"]))))
                               for epoch in self.epochs])

    def _get_trial_idxs_by_path(self):

        trial_idxs_by_path = {path_name: [event_times_in_intervals(event_times=self.external_time_vector,
                                                                   valid_time_intervals=[self.trial_times[path_idx]])[0]
                                          for path_idx in np.where(self.trial_path_names == path_name)[0]]
                              # for each trial on path
                              for path_name in np.unique(self.trial_path_names)}  # for each path
        trial_idxs_by_path_collapsed = {k: np.concatenate(v) for k, v in trial_idxs_by_path.items()}

        return trial_idxs_by_path, trial_idxs_by_path_collapsed

    def _get_reward_delay_idxs(self):

        x = [unpack_df_columns((DioWellDDTrials & self._trials_key(epoch)).fetch1_dataframe(),
                               ["trial_end_well_arrival_times", "trial_end_performance_outcomes"])
            for epoch in self.epochs]

        arrival_times, performance_outcomes = [np.concatenate(x_i) for x_i in list(zip(*x))]

        reward_delay_idxs_by_performance = dict()
        for performance_outcome in set(performance_outcomes):
            trial_arrival_times = arrival_times[performance_outcomes == performance_outcome]
            reward_delay_idxs_by_performance[performance_outcome] = [
                event_times_in_intervals(event_times=self.external_time_vector,
                                         valid_time_intervals=[delay_interval])[0]
                for delay_interval in zip(trial_arrival_times, trial_arrival_times + self.reward_delay_duration)]

        return (np.concatenate([np.concatenate(v) for v in reward_delay_idxs_by_performance.values()]),
                reward_delay_idxs_by_performance,
                {k: np.concatenate(v) for k, v in reward_delay_idxs_by_performance.items()})

    def _check_expected_value_inputs(self):
        for expected_input in ["kernel_num_samples", "kernel_tau"]:
            if expected_input not in self.expected_value_params:
                raise Exception(f"{expected_input} must be defined in expected_value_params")

    def _get_reward_outcomes_int(self):

        reward_outcomes = np.concatenate([self._get_trials_entry(epoch)["trial_end_reward_outcomes"]
            for epoch in self.epochs])  # reward outcomes across epochs

        return np.asarray(reward_outcomes_to_int(reward_outcomes))  # convert to integers

    def _get_expected_value_kernel(self):
        return ExponentialKernel(self.expected_value_params["kernel_num_samples"],
                                 self.expected_value_params["kernel_tau"],
                                 kernel_center=0,
                                 kernel_symmetric=False,
                                 kernel_offset=1,
                                 density=True)

    def _mask_invalid_expected_values(self, trial_values):

        # Set value to nan for samples without enough previous samples
        trial_values[:self.expected_value_params[
            "kernel_num_samples"]] = np.nan

        return trial_values

    def _get_trial_overall_expected_value_by_path(self):

        # Check inputs
        self._check_expected_value_inputs()
        reward_outcomes_int = self._get_reward_outcomes_int()
        kernel = self._get_expected_value_kernel()
        trial_values = kernel.convolve(reward_outcomes_int, mode="full")[0:len(reward_outcomes_int)]
        trial_values = self._mask_invalid_expected_values(trial_values)  # mask values when too few trials to estimate

        # Separate by path
        trial_values_by_path = {path_name: trial_values[np.where(self.trial_path_names == path_name)[0]]
                                for path_name in np.unique(self.trial_path_names)}

        return trial_values_by_path, kernel

    def _get_trial_path_expected_value(self):

        # Check inputs
        self._check_expected_value_inputs()
        reward_outcomes_int = self._get_reward_outcomes_int()
        kernel = self._get_expected_value_kernel()
        path_value_dict = dict()
        for path_name in set(self.trial_path_names):  # for paths
            path_idxs = self.trial_path_names == path_name
            path_reward_outcomes_int = reward_outcomes_int[path_idxs]
            trial_values = kernel.convolve(path_reward_outcomes_int, mode="full")[0:len(path_reward_outcomes_int)]
            path_value_dict[path_name] = self._mask_invalid_expected_values(
                trial_values)  # mask values when not enough trials to estimate

        return path_value_dict, kernel


def embed_target_region(nwb_file_name,
                        epochs_id,
                        brain_region,
                        brain_region_units_param_name,
                        curation_name,
                        res_time_bins_pool_cohort_param_name,
                        res_epoch_spikes_sm_param_name,
                        zscore_fr=False,
                        n_neighbors=15,
                        n_components=2,
                        populate_tables=True,
                        ):
    # Get time vector and firing rates

    # ...Make key for querying tables
    key = {"nwb_file_name": nwb_file_name, "epochs_id": epochs_id, "brain_region": brain_region,
           "brain_region_units_param_name": brain_region_units_param_name,
           "curation_name": curation_name, "res_time_bins_pool_cohort_param_name": res_time_bins_pool_cohort_param_name,
           "res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name, "zscore_fr": zscore_fr}

    # ...Get data across epochs
    dfs = FRVec().firing_rate_across_sort_groups_epochs(key, populate_tables=populate_tables)

    # ...Make map from epoch to idxs in concatenated firing rates
    # epoch_idx_map = {epoch: np.where(dfs.epoch_vector == epoch)[0] for epoch in np.unique(dfs.epoch_vector)}

    # Embed
    input_information = {"nwb_file_name": nwb_file_name,
                         "epoch_vector": dfs.epoch_vector,
                         "time": np.asarray(dfs.fr_concat_df.index)}
    umap_container = UmapContainer(
        input_array=dfs.fr_concat_df.to_numpy(), input_information=input_information, n_neighbors=n_neighbors,
        n_components=n_components)

    # Add df with embedding and time information
    umap_container.make_embedding_df("time")

    return umap_container
#
