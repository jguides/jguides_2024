import copy
import os
from collections import namedtuple

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase, \
    AcrossFRVecTypeTableSelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    insert1_print, get_cohort_test_entry, get_table_name, \
    get_epochs_id, get_default_param
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_delay_duration
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.embedding.umap_wrappers import embed_target_region
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec, \
    populate_jguidera_firing_rate_vector
from src.jguides_2024.metadata.jguidera_brain_region import CurationSet, BrainRegionCohort
from src.jguides_2024.metadata.jguidera_epoch import EpochCohortParams, EpochCohort, EpochsDescription
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.position_and_maze.jguidera_maze import RewardWellPathColor, MazePathWell, RewardWellColor
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptInterp
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams, ResEpochSpikesSm, \
    ResEpochSpikesSmDs
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, EpsUnitsSel, BrainRegionUnitsParams
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDATrials, DioWellArrivalTrials, \
    DioWellArrivalTrialsParams, \
    DioWellADTrialsParams, DioWellADTrials, DioWellDDTrials, DioWellDDTrialsParams, DioWellTrials
from src.jguides_2024.task_event.jguidera_task_performance import PerformanceOutcomeColors
from src.jguides_2024.task_event.jguidera_task_value import TimeExpecVal
from src.jguides_2024.time_and_trials.jguidera_relative_time_at_well import RelTimeWell, RelTimeDelay, \
    RelTimeWellPostDelay
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams, \
    ResTimeBinsPoolSel
from src.jguides_2024.utils.df_helpers import df_filter_columns
from src.jguides_2024.utils.dict_helpers import add_defaults
from src.jguides_2024.utils.plot_helpers import return_n_cmap_colors
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import unpack_single_element, find_spans_increasing_list

# Needed for table definitions:
ResTimeBinsPoolCohortParams
BrainRegionUnits
nd

schema = dj.schema("jguidera_firing_rate_vector_embedding")


"""
Notes on design of FRVecEmb tables:
 - We want the selection table to contain all primary keys of FRVec except for epoch and res_time_bins_pool_param_name
 (nwb_file_name, brain_region, brain_region_units_param_name, curation name, res_epoch_spikes_sm_param_name,
 zscore_fr), and to contain epochs_id and res_time_bins_pool_cohort_param_name (time bins across epochs)
 as a primary key.
 Rationale: we want to look at embedding of firing rate vectors within and optionally
 across epochs (and each epoch has an associated res_time_bins_pool_param_name, which we want to allow
 to be different across epochs for maximal flexibility)
"""


@schema
class FRVecEmbParams(SecKeyParamsBase):
    definition = """
    # Parameters for FRVecEmb
    fr_vec_emb_param_name : varchar(40)
    ---
    n_neighbors : smallint unsigned
    n_components : smallint unsigned
    """

    def _default_params(self):
        return [[15, 2], [15, 3]]


@schema
class FRVecEmbSel(AcrossFRVecTypeTableSelBase):
    definition = """
    # Selection from upstream tables for FRVecEmb
    -> ResTimeBinsPoolCohortParams  # nwb_file_name, epochs_id, res_time_bins_cohort_param_name
    -> BrainRegionUnits  # nwb_file_name, brain_region, brain_region_units_param_name, curation_name
    res_epoch_spikes_sm_param_name : varchar(40)
    -> FRVecEmbParams
    zscore_fr : bool
    """

    # Overwrite parent class method so can add fr_vec_emb_param_name and restrict smoothing kernel
    def _get_potential_keys(self, key_filter=None):

        valid_spikes_sm_param_names = [ResEpochSpikesSmParams().lookup_param_name([x]) for x in [.1, .2]]
        keys = [k for k in super()._get_potential_keys(key_filter) if k["res_epoch_spikes_sm_param_name"] in
                 valid_spikes_sm_param_names]
        return [{**key, **fr_vec_emb_param_key} for key in keys for fr_vec_emb_param_key in
                FRVecEmbParams().fetch("KEY")]

    def insert_epochs(self, nwb_file_name, epochs, res_time_bins_pool_param_names=None,
                      brain_region_units_param_name=None, res_epoch_spikes_sm_param_name=None, zscore_fr=False,
                      fr_vec_emb_param_name=None, curation_set_name=None, brain_region_cohort_name=None,
                      verbose=True, debug_mode=False):

        # Get inputs if not passed
        if res_time_bins_pool_param_names is None:
            res_time_bins_pool_param_names = [ResTimeBinsPoolSel().lookup_param_name_from_shorthand(
                "epoch_100ms")] * len(epochs)
        if brain_region_units_param_name is None:
            # For now, use "all runs" brain_region_units_param_name if multiple epochs, or runX if single epoch
            min_epoch_mean_firing_rate = .1
            if len(epochs) == 1:
                brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(
                    nwb_file_name, unpack_single_element(epochs), min_epoch_mean_firing_rate=min_epoch_mean_firing_rate)
            else:
                brain_region_units_param_name = BrainRegionUnitsParams().lookup_runs_param_name(
                    nwb_file_name, min_epoch_mean_firing_rate=min_epoch_mean_firing_rate)
        if fr_vec_emb_param_name is None:
            fr_vec_emb_param_name = FRVecEmbParams().lookup_param_name([15, 3])
        if res_epoch_spikes_sm_param_name is None:
            res_epoch_spikes_sm_param_name = ResEpochSpikesSmParams().lookup_param_name([.1])
        if curation_set_name is None:
            curation_set_name = get_default_param("curation_set_name")
        if brain_region_cohort_name is None:
            brain_region_cohort_name = get_default_param("brain_region_cohort_name")

        # Assemble key to query tables
        key = {"nwb_file_name": nwb_file_name, "epochs_id": get_epochs_id(epochs),
               "curation_set_name": curation_set_name,
               "brain_region_cohort_name": brain_region_cohort_name,
               "brain_region_units_param_name": brain_region_units_param_name,
               "res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name,
               "zscore_fr": zscore_fr,
               "fr_vec_emb_param_name": fr_vec_emb_param_name}

        # Insert into upstream tables with non-specific key
        if verbose:
            print(f"Inserting into tables upstream of FRVecEmb ...")
        EpochCohortParams().insert_from_epochs(nwb_file_name, epochs)
        ResTimeBinsPoolCohortParams().insert_entry(
            nwb_file_name, epochs, res_time_bins_pool_param_names * len(epochs), use_full_param_name=True)

        # Get restricted time bins cohort param name and add to key
        res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name(
            pool_param_names=res_time_bins_pool_param_names)
        key.update({
               "res_time_bins_pool_cohort_param_name": res_time_bins_pool_cohort_param_name})

        # Loop through entries in tables with brain regions and units, insert into upstream tables with specific key,
        # and insert entry if corresponding entries in table with firing rate vector exist
        brain_regions = (BrainRegionCohort & key).fetch1("brain_regions")
        epochs_description = (EpochsDescription & key).fetch1("epochs_description")
        if verbose:
            print(f"In FRVecEmbSel. Found the following brain regions to try to insert for fr vector embedding: "
                  f"{brain_regions}. key: {key}")
        for brain_region in brain_regions:

            print(f"On {brain_region}...")

            key.update({"brain_region": brain_region, "curation_name": (
                    CurationSet & key).get_curation_name(brain_region, epochs_description)})

            # Make key for populating upstream tables for each epoch
            upstream_key = copy.deepcopy(key)

            # Loop through epochs and populate upstream tables
            for epoch in epochs:

                # Update key for populating upstream tables
                upstream_key.update({"epoch": epoch})

                # Populate epochs units table
                EpsUnitsSel().insert_defaults(key_filter=upstream_key, populate_tables=True)

                # Populate time restricted smoothed spikes tables
                ResEpochSpikesSm().populate_(key=upstream_key)
                ResEpochSpikesSmDs().populate_(key=upstream_key)

                # Populate firing rate vector table
                FRVec().populate_(key=upstream_key)

            # Get number of entries in firing rate vector table and raise error if more than expected (from
            # underspecified key), or print out if fewer than needed to insert into FRVecEmbSel
            num_entries_fr_vec = np.asarray([len(FRVec & {**key, **x}) for x in
                        ResTimeBinsPoolCohortParams().get_cohort_params(key)])

            # ...raise error if more entries than expected
            if any(num_entries_fr_vec > 1):
                raise Exception(f"multiple entries found in FRVec; key underspecified")

            # ...print out if fewer entries than needed to insert into FRVecEmbSel
            if not all(num_entries_fr_vec == 1):
                print(f"not all entries available in FRVec to insert epochs into FRVecEmbSel. key: {key}")
                if debug_mode:
                    raise Exception
                continue

            # Insert into FRVecEmbSel
            # ...make key for insertion
            insert_key = {k: key[k] for k in self.primary_key}
            # ...print out that inserting
            if verbose:
                print(f"Inserting {insert_key} into FRVecEmbSel...")
            # ...insert
            self.insert1(insert_key)

    def insert_epochs_test(self):

        # Insert an "across epochs" entry for testing
        # Use for test, first two entries where there is same nwb_file_name and res_time_bins_pool_param_names
        num_epochs = 2
        fr_vec_emb_param_name = FRVecEmbParams().fetch("fr_vec_emb_param_name")[0]
        res_epoch_spikes_sm_param_name = ResEpochSpikesSmParams().lookup_param_name([.2])
        test_entry_obj = get_cohort_test_entry(
            FRVec & {"res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name}, col_vary="epoch",
            num_entries=num_epochs)
        # Insert if found valid set
        if test_entry_obj is not None:
            key = test_entry_obj.same_col_vals_map
            res_time_bins_pool_param_names = [key["res_time_bins_pool_param_name"]]
            res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name(
                pool_param_names=res_time_bins_pool_param_names)
            epochs = test_entry_obj.target_vals
            key.update({"res_time_bins_pool_cohort_param_name": res_time_bins_pool_cohort_param_name,
                        "epochs_id": get_epochs_id(epochs),
                        "fr_vec_emb_param_name": fr_vec_emb_param_name})
            key.pop("res_time_bins_pool_param_name")
            # Ensure upstream tables have required entry
            ResTimeBinsPoolCohortParams().insert_entry(
                key["nwb_file_name"], epochs, res_time_bins_pool_param_names*num_epochs, use_full_param_name=True)
            self.insert1(key)
        else:
            print(f"Could not insert test entry into {get_table_name(self)}")

    # Override parent class method so can insert two epoch test case
    def insert_defaults(self, key_filter=None):
        super().insert_defaults(key_filter=key_filter)
        self.insert_epochs_test()

    @staticmethod
    def _fr_vec_table():
        return FRVec


@schema
class FRVecEmb(ComputedBase):
    definition = """
    # UMAP embedding of firing rate vectors
    -> FRVecEmbSel
    ---
    -> nd.common.AnalysisNwbfile
    embedding_object_id : varchar(40)
    epoch_vector_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> FRVecEmb
        -> FRVec
        """

    def make(self, key):

        # Get params
        params = copy.deepcopy(key)
        params.update((FRVecEmbParams & key).fetch1())
        params.pop("fr_vec_emb_param_name")
        params.update({"populate_tables": False})
        umap_container = embed_target_region(**params)

        # Insert into main table
        insert_analysis_table_entry(
            self, [umap_container.embedding_df.reset_index(),
                   umap_container.input_information["epoch_vector"].reset_index()], key)

        # Insert into part table
        for k in ResTimeBinsPoolCohortParams().get_keys_with_cohort_params(key):
            insert1_print(self.Upstream, {**key, **k})

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        if df_index_name is None:  # restore index
            df_index_name = "time"
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    @classmethod
    def _plot_invalid(cls, embedding, invalid_bool, ax, plot_components, **plot_params):
        plot_params_copy = copy.deepcopy(plot_params)
        cval_invalid = plot_params.pop("cval_invalid", .9)
        alpha_invalid = plot_params.pop("alpha_invalid", .01)
        plot_params_copy.update({"alpha": alpha_invalid})
        plot_params_copy.update({"s": 1})
        cls._plot_single_color(embedding, invalid_bool, ax, plot_components, [cval_invalid]*3, **plot_params_copy)

    @staticmethod
    def _plot_single_color(embedding, bool_, ax, plot_components, cval, **plot_params):
        plot_params_copy = copy.deepcopy(plot_params)
        plot_params_copy.update({"color": cval})
        plot_line = plot_params_copy.pop("plot_line", False)

        plot_embedding_scatter(embedding.loc[bool_], ax, plot_components, **plot_params_copy)

        if plot_line:
            spans, _ = find_spans_increasing_list(np.where(bool_)[0])
            for x1, x2 in spans:
                plot_embedding_line(embedding.iloc[x1:x2], ax, plot_components, **plot_params_copy)

    def _get_target_epoch(self, target_epoch):
        # Allow target_epoch to be None if single epoch. In this case, use single epoch as target
        if target_epoch is None:
            key = self.fetch1("KEY")
            target_epoch = unpack_single_element((ResTimeBinsPoolCohortParams & key).fetch1("epochs"))
        return target_epoch

    @staticmethod
    def _get_ax(plot_components, ax):

        # Get axis if none passed (depends on number of embedding components)
        if ax is None:
            if len(plot_components) == 2:
                fig, ax = plt.subplots(figsize=(3, 3))
            else:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection='3d')

        return ax

    def plot_by_ppt(self, target_epoch=None, plot_components=("0", "1"), ax=None, **plot_params):
        # Color dots in scatter by fraction of path traversed

        # Get inputs if not passed
        target_epoch = self._get_target_epoch(target_epoch)
        ax = self._get_ax(plot_components, ax)

        # Hard code
        # Resolution for coloring covariate
        bin_width = .01
        # Color map for covariate
        cmap_name = "jet"

        # Get total number of covariate bins
        num_ppt_bins = int(np.ceil(unpack_single_element(np.diff(Ppt.get_range()) / bin_width)))

        # Get digitized ppt
        key = self.fetch1("KEY")
        PptInterp.populate(key)
        cov_dig = pd.concat([(PptInterp & {**key, **k}).digitized_ppt(bin_width=bin_width)
                             for k in ResTimeBinsPoolCohortParams().get_keys_with_cohort_params(key)],
                            axis=0).digitized_ppt

        # Get embedding and epochs corresponding to embedding samples
        dfs = self.fetch1_dataframes()

        # Define idxs inside epoch and covariate period as valid (we will color these); all others are invalid
        valid_bool = np.logical_and(np.ndarray.flatten(dfs.epoch_vector.values) == target_epoch,
                                    cov_dig.values <= num_ppt_bins)
        invalid_bool = np.invert(valid_bool)

        # Plot invalid points for context
        self._plot_invalid(dfs.embedding, invalid_bool, ax, plot_components, **plot_params)

        # Plot valid points: ppt in epoch
        color_list = return_n_cmap_colors(cmap=plt.cm.get_cmap(cmap_name), num_colors=num_ppt_bins)[
            np.asarray(cov_dig[valid_bool] - 1, dtype=int)]
        plot_params.update({"color": color_list})
        plot_embedding_scatter(dfs.embedding.loc[valid_bool], ax, plot_components, **plot_params)

    def _plot_by_covariate(
            self, covariate_table, digitized_fn_name, digitized_col_name, cmap_name="jet", bin_width=.01,
            target_epoch=None, plot_components=("0", "1"), ax=None, exclusion_params=None, **plot_params):
        # Color dots in scatter by covariate, optionally plotting line as well

        plot_params_copy = copy.deepcopy(plot_params)
        plot_line = plot_params_copy.pop("plot_line", False)

        # Get inputs if not passed
        target_epoch = self._get_target_epoch(target_epoch)
        ax = self._get_ax(plot_components, ax)

        # Get total number of covariate bins
        num_covariate_bins = int(np.ceil(unpack_single_element(np.diff(covariate_table.get_range()) / bin_width)))

        # Get digitized covariate, applying restrictions
        key = self.fetch1("KEY")
        cov_dig = pd.concat([getattr(covariate_table & {**key, **k}, digitized_fn_name)(
            bin_width=bin_width, exclusion_params=exclusion_params)
                             for k in ResTimeBinsPoolCohortParams().get_keys_with_cohort_params(key)],
                            axis=0)[digitized_col_name]

        # Get embedding and epochs corresponding to embedding samples
        dfs = self.fetch1_dataframes()

        # Apply same restrictions to embedding and epochs vector times by taking subset of these times
        # in digitized covariate times
        epoch_vector = dfs.epoch_vector.loc[cov_dig.index]
        embedding = dfs.embedding.loc[cov_dig.index]

        # Define idxs inside epoch and covariate period as valid (we will color these); all others are invalid
        valid_bool = np.logical_and(np.ndarray.flatten(epoch_vector.values) == target_epoch,
                                    cov_dig.values <= num_covariate_bins)
        invalid_bool = np.invert(valid_bool)

        # Plot invalid points for context
        self._plot_invalid(embedding, invalid_bool, ax, plot_components, **plot_params_copy)

        # Plot valid points: covariate in epoch
        color_list = return_n_cmap_colors(cmap=plt.cm.get_cmap(cmap_name), num_colors=num_covariate_bins + 1)[
            np.asarray(cov_dig - 1, dtype=int)]  # note that this computation is fairly fast
        plot_params_copy.update({"color": color_list[valid_bool]})
        plot_embedding_scatter(embedding.loc[valid_bool], ax, plot_components, **plot_params_copy)

        if plot_line:
            spans, _ = find_spans_increasing_list(np.where(valid_bool)[0])
            for x1, x2 in spans:
                plot_embedding_line(embedding.iloc[x1:x2], ax, plot_components, **plot_params_copy)

        # Plot large dots for samples in a trial, if indicated
        if exclusion_params is not None:
            if "highlight_trial_num" in exclusion_params:
                valid_bool = self._get_trial_bool(exclusion_params, valid_bool, embedding.index)
                # Update dot size to large and get list of colors corresponding to trial
                plot_params_copy.update(
                    {"s": self._highlight_trial_dot_size(), "edgecolors": "black", "color": color_list[valid_bool]})
                plot_embedding_scatter(embedding.loc[valid_bool], ax, plot_components, **plot_params_copy)
                plot_embedding_line(embedding.loc[valid_bool], ax, plot_components, **plot_params_copy)

    @staticmethod
    def _highlight_trial_dot_size():
        return 50

    def _get_epoch_trial_times(self):
        # Only allow this plotting if one epoch passed. Otherwise, unclear which epoch to plot trials for
        key = EpochCohort().add_epoch_to_key(self.fetch1("KEY"))
        return (DioWellTrials & key).epoch_trial_times()

    def _get_trial_bool(self, exclusion_params, valid_bool, time_vector):
        epoch_trial_times = self._get_epoch_trial_times()
        trial_bool = event_times_in_intervals_bool(
            time_vector, [epoch_trial_times[exclusion_params["highlight_trial_num"]]])
        return np.logical_and(valid_bool, trial_bool)

    def plot_by_relative_time_in_delay(
            self, target_epoch=None, plot_components=("0", "1"), ax=None, exclusion_params=None, **plot_params):
        # Color dots in scatter by fraction of time in delay period passed
        cmap_name = "jet_r"
        self._plot_by_covariate(
            RelTimeDelay, "digitized", "digitized_relative_time_in_delay", cmap_name=cmap_name,
            target_epoch=target_epoch, plot_components=plot_components, ax=ax, exclusion_params=exclusion_params,
            **plot_params)

    def plot_by_relative_time_at_well(self, target_epoch=None, plot_components=("0", "1"), ax=None, **plot_params):
        # Color dots in scatter by fraction of time at well passed
        self._plot_by_covariate(
            RelTimeWell, "digitized", "digitized_relative_time_at_well",
            target_epoch=target_epoch, plot_components=plot_components, ax=ax, **plot_params)

    def plot_by_relative_time_at_well_post_delay(
            self, target_epoch=None, plot_components=("0", "1"), ax=None, **plot_params):
        # Color dots in scatter by fraction of time at well passed, after delay
        self._plot_by_covariate(
            RelTimeWellPostDelay, "digitized", "digitized_relative_time_at_well_post_delay",
            target_epoch=target_epoch, plot_components=plot_components, ax=ax, **plot_params)

    def plot_by_expected_value(
            self, target_epoch=None, plot_components=("0", "1"), ax=None, **plot_params):
        cmap_name = "jet"
        self._plot_by_covariate(
            TimeExpecVal, "digitized", f"digitized_{TimeExpecVal._covariate_name()}", cmap_name=cmap_name,
            target_epoch=target_epoch, plot_components=plot_components, ax=ax, **plot_params)

    def plot_by_path(self, target_epoch=None, plot_components=("0", "1"), ax=None, task_period="path", **plot_params):

        # Check inputs
        check_membership(
            [task_period], ["path", "delay", "well_post_delay"], "passed task period", "valid task periods")

        # Get inputs if not passed
        target_epoch = self._get_target_epoch(target_epoch)
        ax = self._get_ax(plot_components, ax)

        # Get exclusion params if passed
        exclusion_params = plot_params.pop("exclusion_params", None)
        # ...check that exclusion type within those that is currently accounted for
        if exclusion_params is not None:
            check_membership(
                exclusion_params["exclusion_types"], [
                    "stay_trial", "leave_trial", "paths", "epoch_trial_numbers"], "exclusion type",
                "valid exclusion types")

        # Get map from path name to color
        color_map = RewardWellPathColor().get_color_map()

        # Get embedding and epochs corresponding to embedding samples
        dfs = self.fetch1_dataframes()

        # Define key
        key = {**self.fetch1("KEY"), **{"epoch": target_epoch}}

        # Define path names to include if indicated
        path_names = MazePathWell().get_rewarded_path_names(key["nwb_file_name"], target_epoch)  # default
        if exclusion_params is not None:
            if "paths" in exclusion_params["exclusion_types"]:
                path_names = exclusion_params["path_names"]

        # Loop through potentially rewarded paths and color embedding for each
        valid_bool_list = []  # keep track of valid across paths so can define invalid as everything else
        for path_name in path_names:
            trial_feature_map = {"path_names": [path_name]}

            # Define trial intervals on current path
            if task_period == "path":
                trial_intervals = (DioWellDATrials & key).trial_intervals(trial_feature_map, as_dict=True)

            elif task_period in ["delay", "well_post_delay"]:
                key.update(DioWellDDTrialsParams().lookup_no_shift_param_name(as_dict=True))
                dd_trials_df = (DioWellDDTrials & key).fetch1_dataframe()[
                    ["path_names", "trial_end_well_arrival_times", "trial_end_times",
                     "trial_start_epoch_trial_numbers"]]
                df_subset = df_filter_columns(dd_trials_df, {"path_names": path_name})
                well_arrival_times = df_subset.trial_end_well_arrival_times

                if task_period == "delay":
                    trial_intervals = list(zip(well_arrival_times, well_arrival_times + get_delay_duration()))

                elif task_period == "well_post_delay":
                    start_times = well_arrival_times + get_delay_duration()
                    trial_intervals = list(zip(start_times, df_subset.trial_end_times))

                trial_intervals = {k: v for k, v in zip(
                    df_subset["trial_start_epoch_trial_numbers"], trial_intervals)}

                # Exclude stay and leave trials as indicated
                if exclusion_params is not None:

                    stay_trials_bool = (df_subset.trial_end_times - df_subset.trial_end_well_arrival_times).values >= \
                                       get_delay_duration()

                    if "stay_trial" in exclusion_params["exclusion_types"]:
                        valid_bool = np.invert(stay_trials_bool)

                    elif "leave_trial" in exclusion_params["exclusion_types"]:
                        valid_bool = stay_trials_bool

                    trial_intervals = np.asarray(trial_intervals)[valid_bool]

            # Exclude epoch trial numbers if indicated
            if "epoch_trial_numbers" in exclusion_params["exclusion_types"]:
                trial_intervals = {
                    k: v for k, v in trial_intervals.items() if k not in exclusion_params["epoch_trial_numbers"]}

            # Get just trial intervals (leave out epoch trial numbers)
            trial_intervals_ = list(trial_intervals.values())

            # Define idxs inside epoch and trial intervals defined above as valid (we will color these)
            valid_bool = np.logical_and(np.ndarray.flatten(dfs.epoch_vector.values) == target_epoch,
                                        event_times_in_intervals_bool(dfs.embedding.index.values, trial_intervals_))
            valid_bool_list.append(valid_bool)

            # Plot path samples
            self._plot_single_color(dfs.embedding, valid_bool, ax, plot_components, color_map[path_name], **plot_params)

            # Plot markers for estimate of when rat first crossed each maze junction
            first_crossed_junction_df = (Ppt & key).get_first_crossed_junction_df()
            for junction_num, marker in zip([0, 1], ["x", "o"]):
                df_subset = df_filter_columns(first_crossed_junction_df, {"junction_number": junction_num})
                # ...restrict to valid epoch trial numbers
                df_subset = df_subset.loc[list(trial_intervals.keys())]
                idxs = [np.argmin(abs(dfs.embedding.index - x)) for x in df_subset.time_estimate]
                junction_params = {"marker": marker, "s": 80, "alpha": 1}
                self._plot_single_color(
                    dfs.embedding.iloc[idxs], valid_bool[idxs], ax, plot_components, color_map[path_name],
                    **junction_params)

        # Plot points that were not in epoch, or in epoch but not along any of the potentially rewarded paths,
        # for context
        invalid_bool = np.sum(np.asarray(valid_bool_list), axis=0) == 0
        self._plot_invalid(dfs.embedding, invalid_bool, ax, plot_components, **plot_params)

    def plot_by_well(self, target_epoch=None, plot_components=("0", "1"), ax=None, task_period="delay", **plot_params):

        # Check inputs
        check_membership([task_period], ["delay", "well_post_delay"], "passed task period", "valid task periods")
        # Get inputs if not passed
        target_epoch = self._get_target_epoch(target_epoch)
        ax = self._get_ax(plot_components, ax)

        # Get map from well name to color
        key = {**self.fetch1("KEY"), **{"epoch": target_epoch}}
        contingency = TaskIdentification().get_contingency(key["nwb_file_name"], key["epoch"])
        color_map = (RewardWellColor & {"contingency": contingency}).get_color_map()

        # Get embedding and epochs corresponding to embedding samples
        dfs = self.fetch1_dataframes()

        # Loop through potentially rewarded wells and color embedding for each
        well_names = MazePathWell().get_well_names(key["nwb_file_name"], target_epoch, rewarded_wells=True)
        valid_bool_list = []  # keep track of valid so can define invalid as everything else
        for well_name in well_names:
            trial_feature_map = {"well_names": [well_name]}
            if task_period == "well_post_delay":
                trials_key = {"dio_well_ad_trials_param_name": DioWellADTrialsParams().lookup_param_name(
                    [get_delay_duration(), 0])}
                trial_intervals = (DioWellADTrials & {**key, **trials_key}).trial_intervals(trial_feature_map)
            elif task_period == "delay":
                trials_key = {
                    "dio_well_arrival_trials_param_name": DioWellArrivalTrialsParams().lookup_delay_param_name()}
                trial_intervals = (
                        DioWellArrivalTrials & {**key, **trials_key}).trial_intervals(trial_feature_map)

            # Define idxs inside epoch and well period trials as valid (we will color these)
            valid_bool = np.logical_and(np.ndarray.flatten(dfs.epoch_vector.values) == target_epoch,
                                        event_times_in_intervals_bool(dfs.embedding.index.values, trial_intervals))
            valid_bool_list.append(valid_bool)

            # Plot path samples
            self._plot_single_color(dfs.embedding, valid_bool, ax, plot_components, color_map[well_name], **plot_params)

        # Plot points that were not in epoch, or in epoch but not in any of the potentially rewarded well periods,
        # for context
        invalid_bool = np.sum(np.asarray(valid_bool_list), axis=0) == 0
        self._plot_invalid(dfs.embedding, invalid_bool, ax, plot_components, **plot_params)

    def plot_by_performance(
            self, target_epoch=None, plot_components=("0", "1"), ax=None, task_period="delay", **plot_params):

        # Check inputs
        check_membership([task_period], ["delay", "well_post_delay"], "passed task period", "valid task periods")

        # Get inputs if not passed
        target_epoch = self._get_target_epoch(target_epoch)
        ax = self._get_ax(plot_components, ax)

        # Get map from performance outcome to color
        color_map = PerformanceOutcomeColors().get_performance_outcome_color_map()

        # Get embedding and epochs corresponding to embedding samples
        dfs = self.fetch1_dataframes()

        # Get exclusion params if passed
        exclusion_params = plot_params.pop("exclusion_params", None)

        # Loop through performance outcomes and color embedding for each
        # Restrict wells to potentially rewarded in target epoch
        key = {**self.fetch1("KEY"), **{"epoch": target_epoch}}
        well_names = MazePathWell().get_well_names(key["nwb_file_name"], target_epoch, rewarded_wells=True)
        trial_feature_map = {"well_names": well_names}
        valid_bool_list = []  # keep track of valid so can define invalid as everything else
        for performance_outcome, color in color_map.items():
            trial_feature_map.update({"performance_outcomes": [performance_outcome]})
            if task_period == "well_post_delay":
                trials_key = {"dio_well_ad_trials_param_name": DioWellADTrialsParams().lookup_param_name(
                    [get_delay_duration(), 0])}
                trial_intervals = (DioWellADTrials & {**key, **trials_key}).trial_intervals(trial_feature_map)
            elif task_period == "delay":
                trials_key = {"dio_well_arrival_trials_param_name": DioWellArrivalTrialsParams().lookup_delay_param_name()}
                trial_intervals = (
                        DioWellArrivalTrials & {**key, **trials_key}).trial_intervals(trial_feature_map)

            # Define idxs inside epoch and trials as valid (we will color these)
            valid_bool = np.logical_and(np.ndarray.flatten(dfs.epoch_vector.values) == target_epoch,
                                        event_times_in_intervals_bool(dfs.embedding.index.values, trial_intervals))

            # Append list with valid bools for eventually plotting invalid samples for context
            valid_bool_list.append(valid_bool)

            # Plot samples
            self._plot_single_color(dfs.embedding, valid_bool, ax, plot_components, color, **plot_params)

            # Plot large dots for samples in a trial, if indicated
            if exclusion_params is not None:
                if "highlight_trial_num" in exclusion_params:
                    valid_bool = self._get_trial_bool(exclusion_params, valid_bool, dfs.embedding.index)
                    # Update dot size to large
                    plot_params_copy = copy.deepcopy(plot_params)  # avoid overwriting plot_params in loop
                    plot_params_copy.update({"s": self._highlight_trial_dot_size(), "edgecolors": "black"})
                    self._plot_single_color(dfs.embedding, valid_bool, ax, plot_components, color, **plot_params_copy)

        # Plot points that were not in epoch, or in epoch but not in any of the potentially rewarded well periods,
        # for context
        invalid_bool = np.sum(np.asarray(valid_bool_list), axis=0) == 0
        self._plot_invalid(dfs.embedding, invalid_bool, ax, plot_components, **plot_params)

    def plot(self, target_epoch=None, plot_components=("0", "1"), ax=None, **plot_params):
        # Color dots in scatter gray

        # Get inputs if not passed
        target_epoch = self._get_target_epoch(target_epoch)
        ax = self._get_ax(plot_components, ax)

        # Get embedding and epochs corresponding to embedding samples
        dfs = self.fetch1_dataframes()

        # Define idxs inside epoch as valid
        valid_bool = np.ndarray.flatten(dfs.epoch_vector.values) == target_epoch

        # Plot only valid points
        self._plot_single_color(dfs.embedding, valid_bool, ax, plot_components, [.9]*3, **plot_params)


def _plot_embedding(embedding, ax, plot_components=("0", "1"), plot_type="scatter", **kwargs):

    # Check inputs
    check_membership([plot_type], ["scatter", "line"])

    # Define plot function based on plot type
    plot_fn_map = {"scatter": "scatter", "line": "plot"}
    plot_fn = plot_fn_map[plot_type]

    # Define default params based on plot type
    if plot_type == "scatter":
        default_params = {"s": 1, "alpha": .4, "marker": "o", "axis_off": True}
    elif plot_type == "line":
        default_params = {"alpha": .8, "linewidth": 1, "axis_off": True}

    # Add params if not passed
    plot_params = copy.deepcopy(kwargs)  # copy so dont change outside function
    plot_params = add_defaults(plot_params, default_params, add_nonexistent_keys=True)
    axis_off = plot_params.pop("axis_off")

    # Convert params from line to scatter or vice versa
    markersize_s_ratio = 1/3
    if plot_type == "line":
        if "s" in plot_params:
            plot_params["markersize"] = plot_params.pop("s")*markersize_s_ratio
        if "c" in plot_params:
            color = plot_params.pop("c")
            if len(np.shape(color)) > 1:
                plot_params["color"] = "gray"
            else:
                plot_params["color"] = color
        if "edgecolors" in plot_params:
            plot_params.pop("edgecolors")
    elif plot_type == "scatter":
        if "markersize" in plot_params:
            plot_params["s"] = plot_params.pop("markersize")*(1/markersize_s_ratio)
        if "color" in plot_params:
            plot_params["c"] = plot_params.pop("color")

    # Get x and y plot components
    x = embedding[plot_components[0]]
    y = embedding[plot_components[1]]

    # Reverse axes if indicated
    reverse_axes = plot_params.pop("reverse_axes", None)
    if reverse_axes is not None:
        valid_reverse_axes = ["x", "y", "z"]
        check_membership(reverse_axes, valid_reverse_axes, "passed axes to reverse", "valid axes to reverse")
        if "x" in reverse_axes:
            x = -x
        if "y" in reverse_axes:
            y = -y

    # Apply azimuth and elevation
    for param_name in ["azim", "elev"]:
        param_val = plot_params.pop(param_name, None)
        if param_val is not None:
            setattr(ax, param_name, param_val)

    # Get subset of plot params that are allowed by scatter plot function
    invalid_params = ["alpha_invalid"]
    plot_params_ = {k: v for k, v in plot_params.items() if k not in invalid_params}

    # If three points being plotted, "c" gets interpreted as each sample corresponding to point, rather than the single
    # color to apply to all points. We address this here
    if len(x) == 3:
        plot_params_.update({"c": np.tile(plot_params_["c"], (len(x), 1))})

    # 2d plot
    if len(plot_components) == 2:
        getattr(ax, plot_fn)(x, y, **plot_params_)

    # 3d plot
    elif len(plot_components) == 3:
        z = embedding[plot_components[2]]
        if reverse_axes is not None:
            if "z" in reverse_axes:
                z = -z
        getattr(ax, plot_fn)(x, y, z, **plot_params_)

    # Remove axis if indicated
    if axis_off:
        ax.axis('off')


def plot_embedding_scatter(embedding, ax, plot_components=("0", "1"), **kwargs):
    _plot_embedding(embedding, ax, plot_components, "scatter", **kwargs)


def plot_embedding_line(embedding, ax, plot_components=("0", "1"), **kwargs):
    _plot_embedding(embedding, ax, plot_components, "line", **kwargs)


def populate_jguidera_firing_rate_vector_embedding(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_firing_rate_vector_embedding"
    upstream_schema_populate_fn_list = [populate_jguidera_firing_rate_vector]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


class EmbeddingLegend:

    def __init__(self):
        self.annotations_map = self._get_legend_annotations_map()

    def _get_legend_annotations_map(self):
        # Path to figures for legend
        figures_path = "/home/jguidera/Src/presentation_png/cosyne_2023_png"
        fork_maze_path = os.path.join(figures_path, "fork_maze_paths.png")
        Annotation = namedtuple("YText", "title cbar_lim im_path")
        return {"ppt": Annotation("Path\nfraction", ["0", "1"], None),
                "delay_period": Annotation("Time in\ndelay (s)", ["0", "2"], None),
                "paths": Annotation("Path identity", None, fork_maze_path),
                "linear position_and_maze": Annotation("Linear\nposition_and_maze", None, None)}

    def get_legend_annotation(self, plot_var):
        return self._get_legend_annotations_map()[plot_var]


def drop_jguidera_firing_rate_vector_embedding():
    schema.drop()
