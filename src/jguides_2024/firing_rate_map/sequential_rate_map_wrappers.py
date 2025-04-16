import copy
import itertools

import numpy as np
import pandas as pd
import scipy as sp

from src.jguides_2024.datajoint_nwb_utils.datajoint_fr_table_helpers import get_smoothed_fr_table_map
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_unit_name, abbreviate_path_name, \
    populate_multiple_flexible_key, \
    abbreviate_path_names, make_params_string
from src.jguides_2024.firing_rate_map.jguidera_ppt_firing_rate_map import (FrmapPupt,
                                                                           FrmapPuptSm)
from src.jguides_2024.firing_rate_map.jguidera_well_arrival_firing_rate_map import (FrmapWellArrival,
                                                                                    FrmapWellArrivalSm,
                                                                                    STFrmapWellArrivalSm,
                                                                                    STFrmapWellArrival,
                                                                                    FrmapUniqueWellArrival,
                                                                                    FrmapUniqueWellArrivalSm)
from src.jguides_2024.metadata.jguidera_brain_region import SortGroupTargetedLocation, CurationSet, \
    get_brain_region_from_targeted_location
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.position_and_maze.jguidera_position import IntervalLinearizedPositionRescaled
from src.jguides_2024.position_and_maze.jguidera_position_stop import StopLikeWellArrival
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt
from src.jguides_2024.spikes.jguidera_spikes import (EpochSpikeTimesRelabelParams,
                                                     )
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits
from src.jguides_2024.task_event.jguidera_dio_event import DioEvents, ProcessedDioEvents, PumpDiosComplete
from src.jguides_2024.task_event.jguidera_dio_trials import (DioWellDDTrials,
                                                             DioWellDATrials,
                                                             DioWellArrivalTrials,
                                                             DioWellArrivalTrialsParams,
                                                             DioWellDepartureTrials, populate_jguidera_dio_trials,
                                                             DioWellDATrialsParams)
from src.jguides_2024.task_event.jguidera_statescript_event import StatescriptEvents, ProcessedStatescriptEvents
from src.jguides_2024.task_event.jguidera_task_performance import (AlternationTaskPerformance)
from src.jguides_2024.utils.df_helpers import df_filter_columns, df_filter_columns_isin, df_filter1_columns, \
    df_from_data_list
from src.jguides_2024.utils.plot_helpers import plot_spanning_line, get_gridspec_ax_maps, get_plot_idx_map, format_ax
from src.jguides_2024.utils.plot_helpers import save_figure
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals as in_intervals
from src.jguides_2024.utils.string_helpers import format_optional_var
from src.jguides_2024.utils.vector_helpers import unpack_single_element, check_vectors_equal, merged_combinations, \
    vectors_finite_idxs, unpack_single_vector, overlap

# Tables called with eval (do not remove):
DioWellDATrialsParams

def populate_pre_firing_rate_map_tables(key=None, tolerate_error=False):
    populate_multiple_flexible_key(tables=[StatescriptEvents,
                                           DioEvents,
                                           PumpDiosComplete,
                                           ProcessedDioEvents,
                                           ProcessedStatescriptEvents,
                                           AlternationTaskPerformance,
                                           DioWellDATrials,
                                           DioWellDDTrials,
                                           DioWellArrivalTrials,
                                           DioWellDepartureTrials,
                                           IntervalLinearizedPositionRescaled,
                                           Ppt],
                                   key=key,
                                   tolerate_error=tolerate_error)
    EpochSpikeTimesRelabelParams().insert_defaults(key_filter=key)


def populate_frmap_pupt_tables(key=None, tolerate_error=False):
    populate_pre_firing_rate_map_tables(key, tolerate_error)
    populate_multiple_flexible_key(tables=[FrmapPupt, FrmapPuptSm], key=key, tolerate_error=tolerate_error)


def populate_firing_rate_map_wa_tables(key=None,
                                       tolerate_error=False):
    populate_multiple_flexible_key(tables=[FrmapWellArrival,
                                           FrmapWellArrivalSm,
                                           STFrmapWellArrival,
                                           STFrmapWellArrivalSm],
                                   key=key,
                                   tolerate_error=tolerate_error)


def populate_firing_rate_map_uwa_tables(key=None,
                                        tolerate_error=False):
    populate_multiple_flexible_key(tables=[FrmapUniqueWellArrival, FrmapUniqueWellArrivalSm],
                                   key=key,
                                   tolerate_error=tolerate_error)


# def populate_frmap_well_departure_tables(key=None, tolerate_error=False):
#     populate_multiple_flexible_key(tables=[FrmapWellDeparture,
#                                            FrmapWellDepartureSm],
#                                    key=key,
#                                    tolerate_error=tolerate_error)
#
#
# def populate_frmap_unique_well_departure_tables(key=None, tolerate_error=False):
#     FrmapUniqueWellDepartureParams().insert_defaults()
#     FrmapUniqueWellDepartureSmParams().insert_defaults()
#     populate_multiple_flexible_key(tables=[FrmapUniqueWellDeparture,
#                                            FrmapUniqueWellDepartureSm],
#                                    key=key,
#                                    tolerate_error=tolerate_error)


def populate_firing_rate_map_wad_tables(key=None, tolerate_error=False):
    pass
    # FrmapWellArrivalDepartureSm().populate_(**kwargs)
    # FrmapUniqueWellArrivalDepartureSm().populate_(**kwargs)


def populate_firing_rate_map_slwa_tables(key=None, tolerate_error=False):
    populate_multiple_flexible_key(tables=[StopLikeWellArrival,
                                           # StopLikeWellArrivalTrials,
                                           # FrmapStopLikeWellArrival,
                                           # FrmapStopLikeWellArrivalSm,
                                           ],
                                   key=key,
                                   tolerate_error=tolerate_error)


def populate_firing_rate_map_tables(key=None, tolerate_error=False):
    populate_pre_firing_rate_map_tables(key, tolerate_error)
    populate_firing_rate_map_wa_tables(key, tolerate_error)
    populate_firing_rate_map_uwa_tables(key, tolerate_error)
    # populate_frmap_well_departure_tables(key, tolerate_error)
    # populate_frmap_unique_well_departure_tables(key, tolerate_error)
    populate_firing_rate_map_slwa_tables(key, tolerate_error)
    populate_firing_rate_map_wad_tables(key, tolerate_error)


class SmoothedFiringRateMaps:
    def __init__(self,
                 nwb_file_name,
                 epochs,
                 trial_align_events,
                 trial_param_names,  # one for each entry in trial_align_events
                 brain_regions,
                 brain_region_units_param_name="0.1_runs_target_region",
                 curation_set_name="runs_analysis_v2",
                 position_info_param_name="default",
                 populate_tables=True,
                 verbose=False):
        # Get units across epochs (object)
        self.unit_name_df = BrainRegionUnits().get_unit_name_df(
            nwb_file_name, brain_region_units_param_name, curation_set_name, brain_regions=brain_regions)
        # Store parameters
        self.parameters = {"nwb_file_name": nwb_file_name,
                           "epochs": epochs,
                           "trial_align_events": trial_align_events,
                           "trial_param_names": trial_param_names,
                           "brain_regions": brain_regions,
                           "brain_region_units_param_name": brain_region_units_param_name,
                           "curation_set_name": curation_set_name,
                           "position_info_param_name": position_info_param_name,  # relevant for space-based rate maps
                           }
        # Get inputs
        self._get_inputs()
        # Check inputs
        self._check_inputs()
        # Get information related to firing rate maps
        self.tables_info = self._get_tables_info()
        # Populate tables if indicated
        self._populate_tables(populate_tables)
        # Get dataframe with firing rate maps
        self.fr_df = self._get_fr_df(verbose)

    def _get_inputs(self):
        # Get inputs if not passed
        default_trial_param_names_map = {"pupt": "0_0",
                                         "well_arrival": DioWellArrivalTrialsParams().lookup_param_name([0, 2]),
                                         "unique_well_arrival": DioWellArrivalTrialsParams().lookup_param_name([0, 2]),
                                         "well_departure": "-2_0",
                                         "unique_well_departure": "-2_0",
                                         "stop_like_well_arrival": "-1_1"}
        if self.parameters["trial_param_names"] is None:
            self.parameters["trial_param_names"] = [default_trial_param_names_map[x]
                                                    for x in self.parameters["trial_align_events"]]

    def _check_inputs(self):

        # Check that same number of trial align events and parameters for these events
        if len(self.parameters["trial_align_events"]) != len(self.parameters["trial_param_names"]):
            raise Exception(
                f"Same number of trial align events and trial param names must be passed (these correspond 1:1)")

        # Check that trial_align_events valid
        valid_trial_align_events = list(get_smoothed_fr_table_map().index)  # smoothed fr table information
        invalid_trial_align_events = [trial_align_event for trial_align_event in
                                      self.parameters["trial_align_events"] if trial_align_event
                                      not in valid_trial_align_events]
        if len(invalid_trial_align_events) > 0:
            raise Exception(
                f"Valid trial_align_events include {valid_trial_align_events}. The following invalid "
                f"trial_align_events were passed: {invalid_trial_align_events}")

    def _get_tables_info(self):
        # Get information from tables for each event to which to align time_and_trials

        # Get map that connects smoothed firing rate map tables to other information
        fr_table_map = get_smoothed_fr_table_map()

        # First, check that passed parameters exist in params tables
        for trial_align_event, trial_param_name in zip(self.parameters['trial_align_events'],
                                                       self.parameters[
                                                           'trial_param_names']):  # trial align events
            trials_params_table_name = fr_table_map.loc[trial_align_event]["trial_params_table_name"]
            trials_params_table = eval(trials_params_table_name)
            table_column_param_name = fr_table_map.loc[trial_align_event]["trial_param_name"]
            if len(trials_params_table & {table_column_param_name: trial_param_name}) == 0:
                raise Exception(f"No entry in {trials_params_table_name} for param type "
                                f"{table_column_param_name} with passed value: {trial_param_name}")

        # Get tables information
        tables_info_dict = dict()
        for trial_align_event, trial_param_name in zip(self.parameters['trial_align_events'],
                                                       self.parameters['trial_param_names']):

            (trial_feature_name, trials_table_name, trials_params_table_name,
             trials_param_column_name, fr_map_smoothed_table_name,
             map_type) = [fr_table_map.loc[trial_align_event][x] for x in ["trial_feature_name",
                                                                           "trial_table_name",
                                                                           "trial_params_table_name",
                                                                           "trial_param_name",
                                                                           "fr_map_smoothed_table_name",
                                                                           "map_type"]]  # filter for relevant entries

            # Obtain additional time-related information for time-aligned maps
            trial_duration, trial_start_time_shift, trial_end_time_shift = None, None, None  # initialize
            if map_type == "temporal":
                trials_params_table = eval(trials_params_table_name)
                trial_duration = trials_params_table().trial_duration(trial_param_name)
                trial_start_time_shift, trial_end_time_shift = trials_params_table().trial_shifts(
                    trial_param_name)

            # Store firing rate map information
            tables_info_dict[trial_align_event] = {
                "fr_map_smoothed_table_name": fr_map_smoothed_table_name,
                "map_type": map_type,
                "trial_feature_name": trial_feature_name,
                "trials_table_name": trials_table_name,
                "trials_params_table_name": trials_params_table_name,
                "trials_param_column_name": trials_param_column_name,
                "trial_duration": trial_duration,
                "trial_start_time_shift": trial_start_time_shift,
                "trial_end_time_shift": trial_end_time_shift}

        return tables_info_dict

    def _populate_tables(self, populate_tables):

        # Return if not populating tables
        if not populate_tables:
            return

        # Get relationship between brain regions and sort groups
        targeted_location_sort_group_map = SortGroupTargetedLocation().return_targeted_location_sort_group_map(
            self.parameters['nwb_file_name'])
        targeted_locations = list(targeted_location_sort_group_map.keys())

        for trial_align_event, trial_param_name in zip(
                self.parameters['trial_align_events'], self.parameters['trial_param_names']):  # trial align events

            # Populate trials tables (also populates things upstream)
            table_info = self.tables_info[trial_align_event]
            trials_table = eval(self.tables_info[trial_align_event]["trials_table_name"])
            key = {"nwb_file_name": self.parameters['nwb_file_name'],
                   table_info['trials_param_column_name']: trial_param_name,
                   "position_info_param_name": self.parameters["position_info_param_name"]}
            populate_jguidera_dio_trials(key)

            # Get smoothed FR table and populate for relevant sort groups, epochs, paths
            fr_map_smoothed_table = eval(table_info['fr_map_smoothed_table_name'])  # smoothed FR table
            for targeted_location in targeted_locations:  # target locations (brain regions)
                targeted_location_sg_ids = [
                    sg_id for sg_id in targeted_location_sort_group_map[targeted_location]
                    if sg_id in np.unique(self.unit_name_df.sort_group_id)]  # restrict to sort group ids in unit_name_df
                for epoch in self.parameters['epochs']:  # epochs
                    key["epoch"] = epoch  # update key
                    for sort_group_id in targeted_location_sg_ids:  # sort groups IDs for current target location
                        key["sort_group_id"] = sort_group_id  # update key
                        # Define path names: relevant for space-based firing rate maps (map_type == "spatial")
                        trial_feature_values = [
                            "none"]  # default. Use str instead of None because hard to work with None in searches etc.
                        trial_feature_name = table_info["trial_feature_name"]
                        if trial_feature_name != "none":
                            trials_df = (trials_table & key).fetch1_dataframe()
                            trial_feature_name_ = trial_feature_name
                            if trial_feature_name == "well_name":
                                trial_feature_name_ = "well_names"
                            trial_feature_values = set(trials_df[trial_feature_name_])
                        for trial_feature_value in trial_feature_values:
                            key[trial_feature_name] = trial_feature_value
                            # Populate smoothed firing rate table
                            fr_map_smoothed_table.populate(key)

    def _get_fr_df(self, verbose):

        # Get relationship between brain regions and sort groups
        targeted_location_sort_group_map = SortGroupTargetedLocation().return_targeted_location_sort_group_map(
            self.parameters['nwb_file_name'])
        targeted_locations = list(targeted_location_sort_group_map.keys())

        # Assemble fr df: each row has smoothed firing rate from one unit for a single trial alignment event,
        # epoch, and maze path. Must get these out of smoothed firing rate table
        df_list = []
        for trial_align_event, trial_param_name in zip(
                self.parameters['trial_align_events'], self.parameters['trial_param_names']):  # trial align events

            table_info = self.tables_info[trial_align_event]
            trials_table = eval(self.tables_info[trial_align_event]["trials_table_name"])

            # Make key for querying tables
            key = {"nwb_file_name": self.parameters['nwb_file_name'],
                   table_info['trials_param_column_name']: trial_param_name,
                   "position_info_param_name": self.parameters["position_info_param_name"]}
            fr_map_smoothed_table = eval(table_info['fr_map_smoothed_table_name'])  # smoothed FR table

            for targeted_location in targeted_locations:  # target locations (brain regions)

                if verbose:
                    print(f"on target location {targeted_location}...")

                targeted_location_sg_ids = [
                    sg_id for sg_id in targeted_location_sort_group_map[targeted_location]
                    if sg_id in np.unique(self.unit_name_df.sort_group_id)]  # restrict to sort group ids in unit_name_df

                for epoch in self.parameters['epochs']:  # epochs

                    if verbose:
                        print(f"on epoch {epoch}...")

                    key["epoch"] = epoch  # update key

                    # Add curation_set_name to key
                    brain_region = get_brain_region_from_targeted_location(targeted_location)
                    epochs_description = EpochsDescription().lookup_epochs_description(
                        self.parameters["nwb_file_name"], [epoch])
                    curation_name = (CurationSet & {
                        "nwb_file_name": self.parameters["nwb_file_name"],
                        "brain_region_cohort_name": "all_targeted",
                        "curation_set_name": self.parameters["curation_set_name"]}).get_curation_name(
                        brain_region, epochs_description)
                    key.update({"curation_name": curation_name})

                    for sort_group_id in targeted_location_sg_ids:  # sort groups IDs for current target location
                        key["sort_group_id"] = sort_group_id  # update key
                        valid_units = np.unique(df_filter_columns(self.unit_name_df, {
                            "sort_group_id": sort_group_id}).unit_id)  # units from current sort group

                        # Define values of feature of trials used to form subsets of trials with which rate maps
                        # were formed, if relevant
                        trial_feature_name = table_info["trial_feature_name"]
                        trial_feature_values = [
                            "none"]  # default. Use string instead of None because hard to work with None in searches.
                        if trial_feature_name != "none":
                            trial_feature_values = set((trials_table & key).fetch1_dataframe(strip_s=True)[trial_feature_name])

                        for trial_feature_value in trial_feature_values:

                            key[trial_feature_name] = trial_feature_value

                            # Reset index (unit_id) since does not uniquely identify units once have concatenated
                            # across sort groups and epochs
                            sg_df = (fr_map_smoothed_table & key).fetch1_dataframe().loc[
                                valid_units].reset_index()

                            # Add information to sg_df: unique unit identifier, brain region, epoch, trial alignment
                            # event, path, peak firing location
                            sg_df["unit_name"] = [
                                get_unit_name(sort_group_id, unit_id) for unit_id in sg_df[
                                    "unit_id"]]  # add column with sort group id and unit id (uniquely identifies unit)
                            sg_df["targeted_location"] = [targeted_location] * len(
                                sg_df)  # add column with brain region
                            sg_df["epoch"] = [epoch] * len(sg_df)  # add column with epoch
                            sg_df["trial_align_event"] = [trial_align_event] * len(
                                sg_df)  # add column with trial align event
                            sg_df[trial_feature_name] = trial_feature_value
                            sg_df["peak_idx"] = [
                                np.where(rate_map == np.nanmax(rate_map))[0][0] for rate_map in sg_df[
                                    "smoothed_rate_map"].values]  # add column with bin where firing rate peaks

                            # Define columns in sg_df to keep
                            keep_column_names = [
                                "unit_name", "epoch", trial_feature_name, "targeted_location",
                                "trial_align_event", "smoothed_rate_map", "peak_idx"]
                            bin_centers_column_name = unpack_single_element([x for x in sg_df.columns
                                                                             if "bin_centers" in x])
                            bin_edges_column_name = unpack_single_element([x for x in sg_df.columns
                                                                           if "bin_edges" in x])
                            keep_column_names += [
                                bin_centers_column_name, bin_edges_column_name]  # bin centers and edges
                            sg_df_subset = sg_df[keep_column_names]

                            # Rename bin centers/edges columns so that consistent across firing rate map types
                            sg_df_subset.rename(columns={bin_centers_column_name: "bin_centers",
                                                         bin_edges_column_name: "bin_edges"}, inplace=True)
                            df_list.append(sg_df_subset)  # keep relevant columns

        fr_df = pd.concat(df_list)

        # Add zscored firing rate (zscoring within each firing rate map individually)
        fr_df["zscored_smoothed_rate_map"] = [sp.stats.zscore(x) for x in fr_df["smoothed_rate_map"]]

        return fr_df

    def get_fr_df_subset(self,
                         trial_align_events=None,
                         targeted_locations=None,
                         epochs=None,
                         trial_feature_name=None,
                         trial_feature_values=None,
                         min_trial_firing_rate=0,
                         min_trial_firing_rate_epoch=None):

        # Filter for entries that contain a certain trial alignment event, targeted location, epoch, and
        # trial feature value
        df_filter = {k: v for k, v in zip(["trial_align_event", "targeted_location", "epoch", trial_feature_name],
                                          [trial_align_events, targeted_locations, epochs, trial_feature_values])
                     if v is not None}
        fr_df = df_filter_columns_isin(self.fr_df, df_filter)

        # Filter for units at or above average trial firing rate threshold
        # Default is to search within each epoch individually. If min_trial_firing_rate_epoch is passed,
        # find valid units in this epoch, then take only these units across all epochs
        if min_trial_firing_rate_epoch is None:
            valid_bool = [np.mean(x) >= min_trial_firing_rate for x in fr_df["smoothed_rate_map"]]
            return fr_df[valid_bool]
        else:
            reference_fr_df = df_filter_columns(fr_df, {"epoch": min_trial_firing_rate_epoch})
            valid_bool = [np.mean(x) >= min_trial_firing_rate for x in reference_fr_df["smoothed_rate_map"]]
            valid_unit_identifiers = reference_fr_df[valid_bool]["unit_name"].values
            return df_filter_columns_isin(fr_df, {"unit_name": valid_unit_identifiers})

    def get_rate_map_correlation_entry(self, epoch_pair, label_name_pair, targeted_location, segment_name, zscore):
        # There should be a single entry in zscored_rate_map_correlation df corresponding to a targeted_location,
        # two epochs and two path names, but user may not know which epoch/path is 1 vs. 2 in dataframe. This
        # finds and returns the entry.

        # Check inputs
        if len(epoch_pair) != 2:
            raise Exception(f"epoch_pair must have exactly two elements")
        if len(label_name_pair) != 2:
            raise Exception(f"label_name_pair must have exactly two elements")

        # Try both orderings of path names and ensure at most one is present
        if len(np.unique(epoch_pair)) == 2 and len(np.unique(label_name_pair)) == 2:
            valid_idx_combinations = [[0, 1, 0, 1],
                                     [1, 0, 0, 1],
                                     [0, 1, 1, 0],
                                     [1, 0, 1, 0]]
        elif len(np.unique(epoch_pair)) == 1 and len(np.unique(label_name_pair)) == 1:
            valid_idx_combinations = [[0, 0, 0, 0]]
        elif len(np.unique(epoch_pair)) == 1:
            valid_idx_combinations = [[0, 1, 0, 0],
                                      [1, 0, 0, 0]]
        elif len(np.unique(label_name_pair)) == 1:
            valid_idx_combinations = [[0, 0, 0, 1],
                                      [0, 0, 1, 0]]

        df_subsets = [df_filter1_columns(self.rate_map_correlation,
                                         {"label_name_1": label_name_pair[idx_1],
                                          "label_name_2": label_name_pair[idx_2],
                                          "epoch_1": epoch_pair[idx_3],
                                          "epoch_2": epoch_pair[idx_4],
                                          "targeted_location": targeted_location,
                                          "segment_name": segment_name,
                                          "zscore": zscore
                                          },
                                         tolerate_no_entry=True)
                      for idx_1, idx_2, idx_3, idx_4 in valid_idx_combinations]
        nonempty_df_subsets = [x for x in df_subsets if len(x) > 0]

        if len(nonempty_df_subsets) == 0:
            return []

        return unpack_single_element(nonempty_df_subsets)

    def get_rate_map_correlation_by_unit_entry(
            self, epoch_pair, label_name_pair, targeted_location, segment_name, zscore, similarity_metric):
        # There should be a single entry in rate_map_correlation df corresponding to a targeted_location,
        # two epochs and two path names, but user may not know which epoch/path is 1 vs. 2 in dataframe. This
        # finds and returns the entry.

        # Check inputs
        if len(epoch_pair) != 2:
            raise Exception(f"epoch_pair must have exactly two elements")
        if len(label_name_pair) != 2:
            raise Exception(f"label_name_pair must have exactly two elements")

        # Try both orderings of path names and ensure at most one is present
        if len(np.unique(epoch_pair)) == 2 and len(np.unique(label_name_pair)) == 2:
            valid_idx_combinations = [[0, 1, 0, 1],
                                     [1, 0, 0, 1],
                                     [0, 1, 1, 0],
                                     [1, 0, 1, 0]]
        elif len(np.unique(epoch_pair)) == 1 and len(np.unique(label_name_pair)) == 1:
            valid_idx_combinations = [[0, 0, 0, 0]]
        elif len(np.unique(epoch_pair)) == 1:
            valid_idx_combinations = [[0, 1, 0, 0],
                                      [1, 0, 0, 0]]
        elif len(np.unique(label_name_pair)) == 1:
            valid_idx_combinations = [[0, 0, 0, 1],
                                      [0, 0, 1, 0]]

        df_subsets = [df_filter_columns(self.rate_map_correlation_by_unit,
                                         {"label_name_1": label_name_pair[idx_1],
                                          "label_name_2": label_name_pair[idx_2],
                                          "epoch_1": epoch_pair[idx_3],
                                          "epoch_2": epoch_pair[idx_4],
                                          "targeted_location": targeted_location,
                                          "segment_name": segment_name,
                                          "zscore": zscore,
                                          "similarity_metric": similarity_metric
                                          })
                      for idx_1, idx_2, idx_3, idx_4 in valid_idx_combinations]
        nonempty_df_subsets = [x for x in df_subsets if len(x) > 0]

        if len(nonempty_df_subsets) == 0:
            return []

        return unpack_single_element(nonempty_df_subsets)

    def add_rate_map_correlation(self,
                                 epoch_label_comparison_settings=None,
                                 targeted_locations=None,
                                 trial_align_events=None):

        label_meta_name = unpack_single_element([x for x in ["path_name", "well_name"] if x in self.fr_df.columns])

        # Define epochs and labels whose firing rate maps to compare if not passed
        if epoch_label_comparison_settings is None:
            label_names = set(self.fr_df[label_meta_name])
            different_label_name_pairs = list(itertools.combinations(label_names, r=2))
            same_label_name_pairs = [(x, x) for x in label_names]
            different_epoch_pairs = list(itertools.combinations(self.parameters["epochs"], r=2))
            same_epoch_pairs = [(x, x) for x in self.parameters["epochs"]]
            across_label_comparison_settings = merged_combinations(same_epoch_pairs, different_label_name_pairs)
            across_epoch_comparison_settings = merged_combinations(different_epoch_pairs, same_label_name_pairs)
            epoch_label_comparison_settings = across_label_comparison_settings + across_epoch_comparison_settings

        # Define targeted_locations if not passed
        if targeted_locations is None:
            targeted_locations = set(self.fr_df["targeted_location"])

        # Define trial_align_event if not passed
        if trial_align_events is None:
            trial_align_events = set(self.fr_df["trial_align_event"])

        # Find correlation between subset of zscored rate maps (only part of path or delay period)
        data_list = []
        for trial_align_event in trial_align_events:
            for epoch_1, epoch_2, label_name_1, label_name_2 in epoch_label_comparison_settings:
                for targeted_location_idx, targeted_location in enumerate(targeted_locations):

                    # Only proceed if both rate maps for the current epochs/path names exist
                    if not all([len(df_filter_columns(self.fr_df, {
                        "trial_align_event": trial_align_event,
                        "targeted_location": targeted_location,
                        label_meta_name: label_name,
                        "epoch": epoch})) > 0
                                for epoch, label_name in [
                                    (epoch_1, label_name_1), (epoch_2, label_name_2)]]):
                        continue

                    # Get df entry for each rate map in the pair
                    comparison_dfs = [df_filter_columns(self.fr_df, {
                        "trial_align_event": trial_align_event, "targeted_location": targeted_location,
                        label_meta_name: label_name, "epoch": epoch}).set_index(
                        "unit_name")
                                      for epoch, label_name in [(epoch_1, label_name_1), (epoch_2, label_name_2)]]

                    # Check that same unit id order across dfs
                    check_vectors_equal([np.asarray(x.index) for x in comparison_dfs])

                    # Get firing rates in form [unrolled_rate_map_1, unrolled_rate_map_2]
                    bin_centers = unpack_single_vector(np.concatenate([
                        comparison_df["bin_centers"] for comparison_df in comparison_dfs]))

                    # TODO: code segments
                    segments = []
                    if "unique_well_arrival" in self.parameters["trial_align_events"]:
                        segments = [[-.5, .5], [-1, 1], [0, 2]]
                    if "pupt" in self.parameters["trial_align_events"]:
                        segments += [[0, 1], [0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1],
                                    [0, .1], [.1, .2], [.2, .3], [.3, .4], [.4, .5], [.5, .6], [.6, .7], [.7, .8],
                                    [.8, .9], [.9, 1]]

                    for x1, x2 in segments:

                        segment_name = f"{x1}_to_{x2}"

                        valid_bool = np.logical_and(bin_centers >= x1, bin_centers < x2)

                        for zscore in [True, False]:

                            if zscore:
                                column_name = "zscored_smoothed_rate_map"
                            else:
                                column_name = "smoothed_rate_map"

                            fr_unwrapped = [
                                np.ndarray.flatten(np.vstack(x[column_name])[:, valid_bool])
                                for x in comparison_dfs]

                            # Only consider places where both firing rate maps finite
                            valid_idxs = vectors_finite_idxs(fr_unwrapped)

                            # Calculate correlation between zscored rate maps
                            if np.sum(valid_idxs) == 0:
                                corr_coef = np.nan
                            else:
                                corr_coef = sp.stats.pearsonr(
                                    fr_unwrapped[0][valid_idxs], fr_unwrapped[1][valid_idxs])[0]

                            data_list.append(
                                (corr_coef, trial_align_event, epoch_1, epoch_2, label_name_1, label_name_2,
                                 targeted_location, segment_name, zscore))

        self.rate_map_correlation = df_from_data_list(data_list, [
            "zscored_fr_map_corr", "trial_align_event", "epoch_1", "epoch_2", "label_name_1", "label_name_2",
            "targeted_location", "segment_name", "zscore"])

    def add_rate_map_correlation_by_unit(self,
                                 epoch_label_comparison_settings=None,
                                 targeted_locations=None,
                                 trial_align_events=None):

        label_meta_name = unpack_single_element([x for x in ["path_name", "well_name"] if x in self.fr_df.columns])

        # Define epochs and labels whose firing rate maps to compare if not passed
        if epoch_label_comparison_settings is None:
            label_names = set(self.fr_df[label_meta_name])
            different_label_name_pairs = list(itertools.combinations(label_names, r=2))
            same_label_name_pairs = [(x, x) for x in label_names]
            different_epoch_pairs = list(itertools.combinations(self.parameters["epochs"], r=2))
            same_epoch_pairs = [(x, x) for x in self.parameters["epochs"]]
            across_label_comparison_settings = merged_combinations(same_epoch_pairs, different_label_name_pairs)
            across_epoch_comparison_settings = merged_combinations(different_epoch_pairs, same_label_name_pairs)
            epoch_label_comparison_settings = across_label_comparison_settings + across_epoch_comparison_settings

            # FOR NOW, JUST COMPARE ACROSS EPOCHS FOR EFFICIENCY
            epoch_label_comparison_settings = across_epoch_comparison_settings

        # Define targeted_locations if not passed
        if targeted_locations is None:
            targeted_locations = set(self.fr_df["targeted_location"])

        # Define trial_align_event if not passed
        if trial_align_events is None:
            trial_align_events = set(self.fr_df["trial_align_event"])

        # Find correlation between subset of zscored rate maps (only part of path or delay period)
        data_list = []
        for trial_align_event in trial_align_events:
            for epoch_1, epoch_2, label_name_1, label_name_2 in epoch_label_comparison_settings:
                for targeted_location_idx, targeted_location in enumerate(targeted_locations):

                    # Only proceed if both rate maps for the current epochs/path names exist
                    if not all([len(df_filter_columns(self.fr_df, {
                        "trial_align_event": trial_align_event,
                        "targeted_location": targeted_location,
                        label_meta_name: label_name,
                        "epoch": epoch})) > 0
                                for epoch, label_name in [
                                    (epoch_1, label_name_1), (epoch_2, label_name_2)]]):
                        continue

                    # Get df entry for each rate map in the pair
                    comparison_dfs = [df_filter_columns(self.fr_df, {
                        "trial_align_event": trial_align_event, "targeted_location": targeted_location,
                        label_meta_name: label_name, "epoch": epoch}).set_index(
                        "unit_name")
                                      for epoch, label_name in [(epoch_1, label_name_1), (epoch_2, label_name_2)]]

                    # Check that same unit id order across dfs, then get unit_ids (index)
                    unit_names = unpack_single_vector([np.asarray(x.index) for x in comparison_dfs])

                    # Get firing rates in form [unrolled_rate_map_1, unrolled_rate_map_2]
                    bin_centers = unpack_single_vector(np.concatenate([
                        comparison_df["bin_centers"] for comparison_df in comparison_dfs]))

                    segments = [[0, 1], [0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1],
                                [0, .1], [.1, .2], [.2, .3], [.3, .4], [.4, .5], [.5, .6], [.6, .7], [.7, .8],
                                [.8, .9], [.9, 1]]

                    for x1, x2 in segments:

                        segment_name = f"{x1}_to_{x2}"

                        valid_bool = np.logical_and(bin_centers >= x1, bin_centers < x2)

                        for zscore in [True, False]:

                            if zscore:
                                column_name = "zscored_smoothed_rate_map"
                            else:
                                column_name = "smoothed_rate_map"

                            for unit_name in unit_names:

                                fr_unwrapped = [
                                    np.asarray(x.loc[unit_name][column_name])[valid_bool]
                                    for x in comparison_dfs]

                                # Only consider places where both firing rate maps finite
                                valid_idxs = vectors_finite_idxs(fr_unwrapped)

                                # Loop through similarity metrics
                                for similarity_metric in ["correlation", "overlap"]:

                                    if similarity_metric == "correlation":
                                        fn = sp.stats.pearsonr

                                    elif similarity_metric == "overlap":
                                        fn = overlap

                                    # Can't compute overlap with negative values; negative values possible with z
                                    # scoring. So skip overlap if z score True
                                    if zscore and similarity_metric == "overlap":
                                        continue

                                    # Calculate similarity between single unit rate maps
                                    # ...skip if not enough valid values
                                    if np.sum(valid_idxs) < 3 and similarity_metric == "correlation":
                                        continue
                                    elif np.sum(valid_idxs) < 1 and similarity_metric == "overlap":
                                        continue
                                    similarity_measure = fn(fr_unwrapped[0][valid_idxs], fr_unwrapped[1][valid_idxs])

                                    if similarity_metric == "correlation":  # sp.stats.pearsonr returns r and p, take r
                                        similarity_measure = similarity_measure[0]

                                    data_list.append(
                                        (similarity_measure, trial_align_event, epoch_1, epoch_2, label_name_1, label_name_2,
                                         targeted_location, segment_name, zscore, unit_name, similarity_metric))

        self.rate_map_correlation_by_unit = df_from_data_list(data_list, [
            "similarity_measure", "trial_align_event", "epoch_1", "epoch_2", "label_name_1", "label_name_2",
            "targeted_location", "segment_name", "zscore", "unit_name", "similarity_metric"])

    def plot_sequential_rate(self,
                             trial_align_events=None,
                             targeted_locations=None,
                             epochs=None,
                             trial_feature_values=None,
                             min_trial_firing_rate=0,
                             min_trial_firing_rate_epoch=None,
                             zscore_units=True,
                             # True to zscore firing rate of within individual firing rate maps
                             reference_epoch=None,  # order units from this epoch
                             reference_trial_align_event=None,
                             # order units in the firing rate map with this alignment
                             reference_trial_feature_values=None,
                             # order units along this path. If None, align within each path separately
                             cmap="Greys",
                             clim=None,
                             subplot_width=1,
                             subplot_height=1,
                             mega_row_gap_factor=.1,
                             mega_column_gap_factor=.1,
                             fontsize1=None,
                             plot_epoch_text=True,
                             save_fig=False,
                             tolerate_empty_plot=True):

        # Define inputs if not passed
        if trial_align_events is None:
            trial_align_events = self.parameters['trial_align_events']
        if targeted_locations is None:
            targeted_locations = set(self.fr_df["targeted_location"])
        if epochs is None:
            epochs = np.sort(list(set(self.fr_df["epoch"])))
        if reference_epoch is None:
            reference_epoch = epochs[0]
        if reference_trial_align_event is None:
            reference_trial_align_event = self.parameters["trial_align_events"][
                0]  # first trial alignment event
        if reference_trial_feature_values is None:
            reference_trial_feature_values = ["none"]
        if fontsize1 is None:
            fontsize1 = 14

        # Define rate map to use
        rate_map_name = "smoothed_rate_map"
        if zscore_units:
            rate_map_name = f"zscored_{rate_map_name}"

        # Apply trial firing rate threshold
        fr_df = self.get_fr_df_subset(min_trial_firing_rate=min_trial_firing_rate,
                                      min_trial_firing_rate_epoch=min_trial_firing_rate_epoch)

        for trial_align_event in trial_align_events:

            # If trial_feature_values not passed, use all that occur for current trial align event
            trial_feature_name = self.tables_info[trial_align_event]["trial_feature_name"]
            if trial_feature_values is None:
                trial_feature_values = set(self.get_fr_df_subset(trial_align_events=[trial_align_event])[trial_feature_name])
            table_info = self.tables_info[trial_align_event]  # tables info for event to which trials aligned

            # Initialize one big plot for current trial alignment event
            # ...If temporal firing rate maps, scale width so that time has same length unit across plots
            if table_info["map_type"] == "temporal":
                subplot_width *= table_info['trial_duration'] / 3  # make plots wider by this factor
            # Define what we iterate over in rows/columns
            mega_row_iterables = epochs
            mega_column_iterables = targeted_locations
            row_iterables = reference_trial_feature_values
            column_iterables = trial_feature_values
            gs_dict, ax_dict, fig = get_gridspec_ax_maps(
                mega_row_iterables, mega_column_iterables, row_iterables, column_iterables, fig=None,
                 sharex=False, sharey=False, subplot_width=subplot_width, subplot_height=subplot_height,
                 mega_row_gap_factor=mega_row_gap_factor,
                 mega_column_gap_factor=mega_column_gap_factor, wspace=None, hspace=None, width_ratios=None,
                 height_ratios=None,
                 constrained_layout=True)
            plot_idx_map = get_plot_idx_map(mega_row_iterables,
                             mega_column_iterables,
                             row_iterables,
                             column_iterables)

            # Plot
            for targeted_location in targeted_locations:
                for epoch in epochs:
                    for trial_feature_value in trial_feature_values:
                        for reference_trial_feature_value in reference_trial_feature_values:
                            # Set reference path None, set to current trial feature value (e.g. path)
                            use_reference_trial_feature_value = copy.deepcopy(reference_trial_feature_value)  # default
                            if reference_trial_feature_value == "none":
                                use_reference_trial_feature_value = copy.deepcopy(trial_feature_value)
                            # Get unit ordering
                            unit_order = df_filter_columns(fr_df, {"targeted_location": targeted_location,
                                                                   "epoch": reference_epoch,
                                                                   "trial_align_event": reference_trial_align_event,
                                                                   trial_feature_name: use_reference_trial_feature_value}).sort_values(
                                "peak_idx")[
                                "unit_name"].values  # order of the form: ["sg_id_unit_id", ...]
                            if len(unit_order) == 0:
                                if tolerate_empty_plot:
                                    continue
                                raise Exception(
                                    f"unit_order is empty. Make sure reference_epoch, targeted_location, "
                                    f"reference_trial_align_event, use_reference_trial_feature_value are what you expect")

                            # Get axis
                            plot_key = (epoch, targeted_location, reference_trial_feature_value, trial_feature_value)
                            ax = ax_dict[plot_key]
                            plot_idx_obj = plot_idx_map[plot_key]
                            fr_df_subset = df_filter_columns(fr_df, {"targeted_location": targeted_location,
                                                                     "epoch": epoch,
                                                                     "trial_align_event": trial_align_event,
                                                                     trial_feature_name: trial_feature_value})
                            epoch_text_y = .5  # initialize
                            if len(fr_df_subset) > 0:  # if firing rate maps
                                # Apply unit ordering
                                # ...First update unit order to contain only units in subset of firing rate df
                                # There can be units in the unit_order not in the firing rate df subset if the
                                # unit order is formed off of an epoch different from the epoch used to get the
                                # firing rate df subset, and a minimum spike threshold was applied when getting the
                                # firing rate df subset
                                unit_order_subset = [x for x in unit_order
                                                     if x in fr_df_subset["unit_name"].values]
                                rate_map_arr = np.vstack(fr_df_subset.set_index("unit_name").loc[
                                                             unit_order_subset][rate_map_name])
                                # Use bin centers as x values for plot. Check that same across df rows
                                bin_centers = np.vstack(fr_df_subset["bin_centers"])
                                check_vectors_equal(bin_centers)
                                bin_centers = bin_centers[0, :]
                                img = ax.pcolormesh(bin_centers, np.arange(0, np.shape(rate_map_arr)[0]),rate_map_arr,
                                              cmap=cmap)
                                epoch_text_y = len(rate_map_arr) / 2  # update epoch text y position_and_maze
                                # Plot vertical lines at 0s and 2s if temporal firing rate map
                                if table_info["map_type"] == "temporal":
                                    potential_lines = np.asarray([0, 2])
                                    for x in \
                                            in_intervals(potential_lines, [[table_info['trial_start_time_shift'],
                                                                            table_info['trial_end_time_shift']]])[
                                                1]:  # restrict to lines within trial period
                                        plot_spanning_line(span_data=[0, np.shape(rate_map_arr)[0]], constant_val=x,
                                                           ax=ax, span_axis="y", linewidth=1, color="#880000")

                            # Axis
                            # ...ylims
                            ylim = [0, np.shape(rate_map_arr)[0]]
                            # ...y ticks on first column and first row/mega row
                            yticks = None
                            yticklabels = []
                            if (plot_idx_obj.mega_row_idx == 0 and
                                plot_idx_obj.row_idx == 0 and
                                plot_idx_obj.column_idx == 0 and
                                len(fr_df_subset) > 0):  # can only set if rate map above
                                yticks = ylim
                                yticklabels = yticks
                            # ...x ticks
                            # If temporal firing rate map, use trial time relative to event
                            xticks = None
                            xticklabels = []
                            if table_info["map_type"] == "temporal":
                                xticks = np.arange(table_info['trial_start_time_shift'],
                                                   table_info['trial_end_time_shift'] + 1, 2)
                                # x tick labels if first row/column
                                if (plot_idx_obj.row_idx == 0 and
                                    plot_idx_obj.column_idx == 0):
                                    xticklabels = xticks
                            # ...y label: reference_trial_feature_value if not "none", if first column
                            ylabel = None
                            if (plot_idx_obj.column_idx == 0 and
                                reference_trial_feature_value != "none"):
                                ylabel = abbreviate_path_name(reference_trial_feature_value)
                            # ...Title: path name (if not "none"), if first row
                            title = ""
                            title_color = "black"
                            if (plot_idx_obj.row_idx == 0 and
                                trial_feature_value != "none"):
                                    title = abbreviate_path_name(trial_feature_value)
                                    # TODO: make color setting below an option
                                    if trial_feature_value == use_reference_trial_feature_value:
                                        title_color = "red"
                            # ...Suptitle: target region -- seems to not be working...
                            if (plot_idx_obj.row_idx == 0 and
                                plot_idx_obj.column_idx == 0 and
                                plot_idx_obj.mega_row_idx == 0):
                                # fig.suptitle(targeted_location)
                                pass
                            # ...Apply axis changes
                            format_ax(ax, ylim=ylim,
                                      xticks=xticks, xticklabels=xticklabels,
                                      yticks=yticks, yticklabels=yticklabels,
                                      ylabel=ylabel,
                                      title=title, title_color=title_color,
                                      fontsize=fontsize1)
                            # ...Remove top and right plot border
                            for border in ['top', 'right']:
                                ax.spines[border].set_visible(False)
                            # ...clim
                            if clim is not None:
                                img.set_clim(clim)
                            # ...Text for epoch if indicated, and more than one epoch, and first column/mega column
                            # and first row in each mega row
                            if (plot_epoch_text and
                                len(epochs) > 1 and
                                plot_idx_obj.row_idx == 0 and
                                plot_idx_obj.column_idx == 0 and
                                plot_idx_obj.mega_column_idx == 0):
                                # Use red text if epoch is same as reference epoch (used to order units)
                                color = "black"  # default text color
                                if epoch == reference_epoch:
                                    color = "red"
                                # Move x position_and_maze of epoch text based on plot width
                                xlims = ax.get_xlim()
                                ax.text(xlims[0] - unpack_single_element(np.diff(xlims))*.3,
                                        epoch_text_y, f"ep{int(epoch)}", fontsize=fontsize1, ha="center",
                                        color=color)

            # Save figure
            clim_text = format_optional_var(clim, prepend_underscore=True, leading_text="clim")
            trial_feature_value_text = "_".join(abbreviate_path_names(trial_feature_values))
            epochs_text = "_".join([str(x) for x in epochs])
            target_region_text = "_".join(targeted_locations)
            trial_param_name_text= ""  # default
            if table_info["map_type"] == "temporal":
                trial_param_name_text = "_" + make_params_string([table_info["trial_start_time_shift"],
                                                        table_info["trial_end_time_shift"]])
            save_figure(fig,
                        f"{self.parameters['nwb_file_name'].split('_.')[0]}_"
                        f"ep{reference_epoch}_{reference_trial_align_event}_aligned_{trial_feature_value_text}_"
                        f"{epochs_text}_{target_region_text}_{trial_align_event}{trial_param_name_text}{clim_text}",
                        save_fig=save_fig)


