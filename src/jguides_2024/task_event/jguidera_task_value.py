import copy

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, ComputedBase, CovDigmethBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDATrials, DioWellDATrialsParams, \
    DioWellDDTrials, DioWellDDTrialsParams
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellTrials
from src.jguides_2024.task_event.jguidera_task_performance import reward_outcomes_to_int
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPool, ResTimeBinsPoolSel
from src.jguides_2024.utils.df_helpers import df_from_data_list, df_filter_columns
from src.jguides_2024.utils.kernel_helpers import Kernel
from src.jguides_2024.utils.plot_helpers import get_fig_axes
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import unpack_single_element

nd
DioWellTrials
ResTimeBinsPool

schema = dj.schema("jguidera_task_value")


@schema
class TrialExpecValParams(dj.Manual):
    definition = """
    # Parameters for TrialExpecVal
    trial_expec_val_param_name : varchar(200)
    ---
    trial_expec_val_params : blob
    """

    def insert1(self, key, **kwargs):
        # Check parameters

        # Check that all required params passed
        check_membership(["kernel_params", "val_model"], key["trial_expec_val_params"])

        # Check that kernel params valid
        Kernel(key["trial_expec_val_params"]["kernel_params"])  # params checked when create kernel

        if "skip_duplicates" not in kwargs:
            kwargs["skip_duplicates"] = True

        super().insert1(key, **kwargs)

    def insert_defaults(self, **kwargs):
        # Single paths, no kernel
        val_model = "single_path"
        kernel_type = "exponential"
        kernel_num_samples = 4
        kernel_tau = 2

        kernel_params = {"kernel_type": kernel_type, "kernel_num_samples": kernel_num_samples, "kernel_tau": kernel_tau}
        kernel_text = "_".join([str(x) for x in [kernel_type, kernel_num_samples, kernel_tau]])
        expec_val_param_name = f"{val_model}_{kernel_text}"
        expec_val_params = {"kernel_params": kernel_params, "val_model": val_model}

        self.insert1({"trial_expec_val_param_name": expec_val_param_name, "trial_expec_val_params": expec_val_params})


@schema
class TrialExpecValSel(SelBase):
    definition = """
    # Selection from upstream tables for TrialExpecVal
    -> DioWellTrials
    -> TrialExpecValParams
    """

    class DioWellDATrials(dj.Part):
        definition = """
        # Achieves upstream dependence on DioWellDATrials
        -> TrialExpecValSel
        -> DioWellDATrials
        """

    def _get_potential_keys(self, key_filter=None):
        return [k for k in super()._get_potential_keys() if len(DioWellDATrials & k) > 0]


@schema
class TrialExpecVal(ComputedBase):
    definition = """
    # Expected value for trials
    -> TrialExpecValSel
    ---
    -> nd.common.AnalysisNwbfile
    trial_expec_val_object_id : varchar(40)
    """

    @staticmethod
    def _covariate_name():
        return "trial_end_value"

    def make(self, key):

        # Get params
        params = (TrialExpecValParams & key).fetch1("trial_expec_val_params")

        # Get kernel
        kernel = Kernel(params["kernel_params"]).kernel

        # Get trials information
        # ...Set departure to arrival trials param name to no shift
        dio_well_da_trials_param_name = DioWellDATrialsParams().lookup_no_shift_param_name()
        da_trials_key = {**key, **{"dio_well_da_trials_param_name": dio_well_da_trials_param_name}}
        # ...Get information
        trial_start_epoch_trial_numbers, path_names, trial_end_reward_outcomes = (
                DioWellDATrials & da_trials_key).fetch1(
            "trial_start_epoch_trial_numbers", "path_names", "trial_end_reward_outcomes")

        # Convert reward outcomes to integer
        trial_end_reward_outcomes = np.asarray(reward_outcomes_to_int(trial_end_reward_outcomes))

        # Get expected value for each trial
        if "val_model" == "single_path":
            pass

        data_list = []
        for path_name in np.unique(path_names):  # for paths
            idxs = np.where(path_names == path_name)[0]
            path_trial_end_reward_outcomes = trial_end_reward_outcomes[idxs]
            values = kernel.convolve(path_trial_end_reward_outcomes, mode="full")[0:len(path_trial_end_reward_outcomes)]
            for idx, trial_value, reward_outcome in zip(idxs, values, path_trial_end_reward_outcomes):
                data_list.append((trial_start_epoch_trial_numbers[idx], trial_value, path_name, reward_outcome))
        df = df_from_data_list(data_list, [
            "trial_start_epoch_trial_number", self._covariate_name(), "path_name", "trial_end_reward_outcome"])

        insert_analysis_table_entry(self, [df], key)

    def fetch1_dataframe(
            self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="trial_start_epoch_trial_number"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def plot_results(self):

        df = self.fetch1_dataframe()

        path_names = np.unique(df.path_name)

        # Initialize figure
        fig, axes = get_fig_axes(
            num_rows=len(path_names), num_columns=1, subplot_width=5, subplot_height=2.5)

        for idx, (path_name, ax) in enumerate(zip(path_names, axes)):

            df_subset = df_filter_columns(df, {"path_name": path_name})

            for x, plot_marker, color in zip(
                    ["trial_end_reward_outcome", "trial_end_value"], ["o", "x"], ["black", "red"]):
                ax.plot(df_subset[x], plot_marker, label=x, color=color)
                ax.plot(df_subset[x], label=x, color=color)

            ax.set_title(path_name)

            if idx == 0:
                ax.legend()


@schema
class TimeExpecValSel(SelBase):
    definition = """
    # Selection from upstream tables for TimeExpecVal
    -> ResTimeBinsPool
    -> TrialExpecVal
    """

    def _get_potential_keys(self, key_filter=None):

        # Get key filter if not passed
        if key_filter is None:
            key_filter = dict()

        # Restrict to 100ms epoch time bins
        valid_res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms")
        # ...If time bins param passed, check that 100ms epoch time bins
        if "res_time_bins_pool_param_name" in key_filter:
            if key_filter["res_time_bins_pool_param_name"] != valid_res_time_bins_pool_param_name:
                raise Exception(f"res_time_bins_pool_param_name must be {valid_res_time_bins_pool_param_name}")
        # ...Update key filter with time bins param
        key_filter.update({"res_time_bins_pool_param_name": valid_res_time_bins_pool_param_name})

        return super()._get_potential_keys(key_filter)

    @staticmethod
    def _covariate_name():
        return "trial_end_value"


@schema
class TimeExpecVal(CovDigmethBase):
    definition = """
    # Expected value at times
    -> TimeExpecValSel
    ---
    -> nd.common.AnalysisNwbfile
    time_expec_val_object_id : varchar(40)
    """

    def make(self, key):

        # Get expected values for next possible reward at each time bin center. This is the value associated
        # with the END of the upcoming delay period if NOT in a delay period, or else the current delay period
        # if in a delay period.
        # Using epoch trial numbers, get trial end values at each time bin center. nan for values that are outside
        # of epoch trials or for which an expected value does not exist (can happen at end of session when
        # no "next well")

        # Get time bins
        time_bins_df = (ResTimeBinsPool & key).fetch1_dataframe()

        # Match each time bin center to trials that span consecutive events, defined for each well as post delay
        # or departure, whichever comes first

        # Get trial numbers and trial times for current dio well trials table entry
        table_entry = (DioWellTrials & key)
        # Get epoch trial numbers, omitting last since no associated "post delay or departure" trial
        epoch_trial_nums = table_entry.fetch1("epoch_trial_numbers")[:-1]
        # Get "post delay or departure" trial intervals
        trial_times = table_entry.get_post_delay_or_departure_trials()
        if len(epoch_trial_nums) != len(trial_times):
            raise Exception(f"different number of trial numbers and trial times")

        # Find the index of the trial interval in which each time falls
        time_bin_centers = time_bins_df.time_bin_centers.values
        idxs = [unpack_single_element(
            np.where(np.prod(np.asarray(trial_times) - x, axis=1) <= 0)[0], tolerate_no_entry=True,
            return_no_entry=np.nan) for x in time_bin_centers]

        # Get the epoch trial number that each time falls within
        time_vector_epoch_trial_nums = [np.nan if np.isnan(idx) else epoch_trial_nums[idx] for idx in idxs]

        # Get trial expected values
        trial_expec_val_df = (TrialExpecVal & key).fetch1_dataframe()

        trial_end_values = [
            np.nan if np.logical_or(np.isnan(x), x not in trial_expec_val_df.index) else
            trial_expec_val_df.trial_end_value.loc[x] for x in time_vector_epoch_trial_nums]

        df = pd.DataFrame.from_dict({"time_bin_centers": time_bin_centers, "trial_end_value": trial_end_values,
                                     "epoch_trial_nums": time_vector_epoch_trial_nums})

        insert_analysis_table_entry(self, [df], key)

    @staticmethod
    def _covariate_name():
        return TrialExpecVal()._covariate_name()

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time_bin_centers"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def fetch1_dataframe_exclude(self, exclusion_params=None, object_id_name=None, restore_empty_nwb_object=True,
                                 df_index_name="time_bin_centers"):

        df = super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

        # Exclude as indicated
        if exclusion_params is not None:

            # Copy params to avoid changing outside function
            exclusion_params = copy.deepcopy(exclusion_params)

            # Check exclusion type passed if want to exclude
            if "exclusion_type" not in exclusion_params:
                raise Exception(f"exclusion_type must be in exclusion_params")

            # Check exclusion type valid
            exclusion_type = exclusion_params["exclusion_type"]
            check_membership([exclusion_type], ["paths", None])

            # Restrict to trials on particular paths
            if exclusion_type == "paths":

                task_period = exclusion_params.pop("task_period", None)

                # Get key for querying table with path trial times
                key = self.fetch1("KEY")
                param_name = DioWellDDTrialsParams().lookup_no_shift_param_name()
                key.update({"dio_well_dd_trials_param_name": param_name})

                # Get times by paths
                path_times = (DioWellDDTrials & key).times_by_paths(
                    df.index, exclusion_params["path_names"], task_period)

                # Restrict to times on paths
                df = df.loc[path_times]

        return df

    @staticmethod
    def get_range():
        return np.asarray([0, 1])

    def plot_results(self, ax=None):
        """
        Plot values along with trial information for context
        :param ax: optional, matplotlib axis object
        """

        # Get axis if not passed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))

        # Get results
        df = self.fetch1_dataframe()

        # Plot values
        ax.plot(df.index, df.trial_end_value, '.', color="gray")

        # Plot dio well departure to departure trial information for context
        (DioWellDDTrials & self.fetch1("KEY")).plot_results(ax)


def populate_jguidera_task_value(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_task_value"
    upstream_schema_populate_fn_list= []
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_task_value():
    schema.drop()
