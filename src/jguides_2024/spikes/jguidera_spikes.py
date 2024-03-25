# This module defines tables related to processing spikes data

import copy

import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd
from spyglass.common import IntervalList
from spyglass.spikesorting.v0.spikesorting_curation import SortInterval, CuratedSpikeSorting

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase, PartBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import (insert_analysis_table_entry,
                                                                          get_schema_table_names_from_file,
                                                                          insert1_print,
                                                                          populate_insert, add_param_defaults,
                                                                          get_curation_name,
                                                                          fetch_entries_as_dict,
                                                                          get_table_key_names, get_key_filter,
                                                                          delete_,
                                                                          get_table_curation_names_for_key,
                                                                          get_unit_name)
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.metadata.jguidera_premaze_durations import PremazeDurations
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.time_and_trials.jguidera_time_bins import EpochTimeBins, EpochTimeBinsParams
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals

# Needed for table definitions:
TaskIdentification
EpochTimeBinsParams
EpochTimeBins
nd

schema_name = "jguidera_spikes"
schema = dj.schema(schema_name)


# TODO (table drop): if remake table, add upstream selection table so can pair specific entries in CuratedSpikeSorting
# with specific entries in TaskIdentification
@schema
class EpochSpikeTimes(ComputedBase):
    definition = """
    # Table with spike times within an epoch
    -> CuratedSpikeSorting
    -> TaskIdentification
    ---
    -> nd.common.AnalysisNwbfile
    epoch_spike_times_object_id : varchar(40)
    """

    def make(self, key):

        # Get entry from CuratedSpikeSorting
        curated_spike_sorting_entry = (CuratedSpikeSorting & key).fetch_nwb()

        # Check that only one entry in CuratedSpikeSorting for key
        if len(curated_spike_sorting_entry) != 1:
            raise Exception(f"Should have found exactly one entry in CuratedSpikeSorting for key but found "
                            f"{len(curated_spike_sorting_entry)}")
        curated_spike_sorting_entry = curated_spike_sorting_entry[0]  # take single entry in table

        # Initialize empty df (used in case of no units)
        epoch_spikes_df = pd.DataFrame(columns=["unit_id", "epoch_spike_times"])

        # Make df with spike times if units exist
        if "units" in curated_spike_sorting_entry:
            # Get sort interval
            sort_interval = (SortInterval & key).fetch1("sort_interval")

            # Get epoch valid times and check that all fall within sort interval allowing for some margin,
            # then proceed to find spikes from curated units within epoch valid times
            # Get interval list name for epoch
            interval_list_name = EpochIntervalListName().get_interval_list_name(key["nwb_file_name"], key["epoch"])
            epoch_valid_times = (IntervalList & {"nwb_file_name": key["nwb_file_name"],
                                                 "interval_list_name": interval_list_name}).fetch1(
                "valid_times")  # epoch valid times
            # Define max tolerated amount that sort interval bounds can be outside epoch interval
            margin_tolerance = 1  # seconds

            # Account for premaze duration if this was excluded during creation of sorting interval
            premaze_duration = 0
            if "no premaze" in key["sort_interval_name"]:
                if len(PremazeDurations & key) > 1:
                    raise Exception(f"More than one premaze duration found")
                if len(PremazeDurations & key) == 1:
                    premaze_duration = (PremazeDurations & key).fetch1("premaze_duration")
            if all(np.logical_and(t_range[0] + margin_tolerance + premaze_duration >= sort_interval[0],
                                  t_range[1] - margin_tolerance <= sort_interval[1]) for t_range in
                   epoch_valid_times):  # if all epoch valid times within sort interval
                epoch_spikes_dict = {
                    unit_id: spike_times[np.asarray([np.logical_and(
                        spike_times > t_range[0], spike_times < t_range[1]) for t_range in epoch_valid_times]).sum(
                        axis=0) > 0] for unit_id, spike_times in curated_spike_sorting_entry[
                        "units"]["spike_times"].items()}
                epoch_spikes_df = pd.DataFrame.from_dict({"unit_id": list(epoch_spikes_dict.keys()),
                                                          "epoch_spike_times": list(epoch_spikes_dict.values())})

        # Insert into table
        insert_analysis_table_entry(self, [epoch_spikes_df], key)

    def in_time_intervals(self, time_intervals):
        unit_df = self.fetch1_dataframe()
        return zip(*[(unit_id, event_times_in_intervals(event_times=epoch_spike_times.values[0],
                                                        valid_time_intervals=[interval])[1])
                     for interval in time_intervals
                     for unit_id, epoch_spike_times in unit_df.iterrows()])

    def cleanup(self, safemode=True):
        valid_curation_ids = [3]
        invalid_curation_ids = set(self.fetch("curation_id")) - set(valid_curation_ids)
        for invalid_curation_id in invalid_curation_ids:
            # To be able to delete entries in EpochSpikeTimes, must first delete entries in EpochSpikeTimesRelabelParams
            # (since EpochSpikeTimesRelabel cohort table depends on EpochSpikeTimes and has parts table -- must delete
            # entries in main table before can delete entries in part table. Note the proper place to apply our key for
            # EpochSpikeTimes to have the inteded effects in deleting entries from EpochSpikeTimesRelabel is the
            # EpochSpikeTimesRelabelParams, and NOT EpochSpikeTimesRelabel (the main table here does not share primary
            # key attributes with EpochSpikeTimes, whereas EpochSpikeTimesRelabelParams does).
            (EpochSpikeTimesRelabelParams & {"curation_id": invalid_curation_id}).delete()
            # Now delete entries in EpochSpikeTimes
            table_entries = (self & {"curation_id": invalid_curation_id})
            print(f"Will delete entries: {table_entries.fetch('KEY')}")
            table_entries.delete(safemode=safemode)

    # Extend parent class method so can restrict curation_id
    def populate_(self, key=None, tolerate_error=False, populated_tables=None, recursive=False):
        # Define key if not passed
        if key is None:
            key = dict()

        # Make copy of key so as not to alter key outside this function
        key_ = copy.deepcopy(key)

        # Define curation ID we want
        target_curation_id = 3

        # If curation ID in key, check that is desired
        if "curation_id" in key_:
            if key["curation_id"] != target_curation_id:
                raise Exception(f"curation_id must be {target_curation_id} when running populate_ on EpochSpikeTimes, "
                                f"but is {key['curation_id']}")

        # Otherwise add desired curation ID to key
        if "curation_id" not in key_:
            key_.update({"curation_id": target_curation_id})

        super().populate_(key_, tolerate_error, populated_tables, recursive)

    def delete_(self, key, safemode=True):
        delete_(self, [EpochSpikeTimesRelabelParams], key, safemode)


@schema
class EpochSpikeTimesRelabelParams(SecKeyParamsBase):
    definition = """
    # Consolidate primary keys for entries in EpochSpikeTimes
    nwb_file_name : varchar(40)
    epoch : int
    sort_group_id : int
    curation_name : varchar(80)
    ---
    sort_interval_name : varchar(80)
    curation_id : int
    preproc_params_name : varchar(100)
    team_name : varchar(50)
    sorter : varchar(50)
    sorter_params_name : varchar(50)
    artifact_removed_interval_list_name : varchar(1000)
    """

    def insert_defaults(self, **kwargs):
        # Define key if not passed
        key_filter = get_key_filter(kwargs)

        # Get default params if not in key
        key_filter = add_param_defaults(key_filter, target_keys=["sorter"], add_nonexistent_keys=True)

        # Set curation ID to 3
        key_filter.update({"curation_id": 3})

        # Need to make sure that sorts corresponding to single epochs only get spikes
        # assigned to those epochs.
        # Approach: include all table entries that do not correspond to single epochs
        # For table entries that correspond to single epochs, exclude those
        # done on an epoch (indicated by sort interval name) different from that
        # in which spikes are found (epoch in table key)
        # Note that currently requires considerable manual parsing; would be better
        # if this workflow had less hard code
        # Get table entries corresponding to single epoch sort (have pos in sort interval name)
        table_entries = fetch_entries_as_dict(EpochSpikeTimes & key_filter)
        pos_table_entries = [k for k in table_entries if "pos" in k["sort_interval_name"]]
        # Get starting interval list name for these sort intervals
        starting_interval_list_names = [x["sort_interval_name"].split("valid times")[0] + "valid times" for x in
                                        pos_table_entries]
        # get nwb file names for these table entries
        pos_nwb_file_names = [x["nwb_file_name"] for x in pos_table_entries]
        # Get map from starting interval list name and nwb file name to epoch
        starting_interval_list_names_map = {(interval_list_name, nwb_file_name): EpochIntervalListName().get_epoch(
            nwb_file_name, interval_list_name) for interval_list_name, nwb_file_name in
            set(list(zip(starting_interval_list_names, pos_nwb_file_names)))}
        # use map to find epoch corresponding to starting interval list name / nwb file name
        # for each table entry
        pos_epochs = [starting_interval_list_names_map[(interval_list_name, nwb_file_name)] for
                      interval_list_name, nwb_file_name in zip(
                starting_interval_list_names, pos_nwb_file_names)]
        # Exclude table entries for which above epoch isnt same as epoch in key
        valid_pos_table_entries = [x for x, y in zip(pos_table_entries, pos_epochs) if x["epoch"] == y]
        # Define valid table entries as non single epoch entries and single epoch entries for which epoch
        # on which sort performed matches that we are finding spikes within
        non_pos_table_entries = [k for k in table_entries if "pos" not in k["sort_interval_name"]]
        valid_table_entries = non_pos_table_entries + valid_pos_table_entries

        # Loop through valid entries in epoch spikes table for key
        for key in valid_table_entries:
            # Get curation_name: reflects combination of desired secondary keys. Note that we intentionally exclude
            # secondary keys that we do not expect or want to take on multiple values
            key.update({"curation_name": get_curation_name(key["sort_interval_name"], key["curation_id"])})
            # Insert into table
            self.insert1({k: key[k] for k in get_table_key_names(self)}, skip_duplicates=True)

    def delete_(self, key, safemode=True):
        # Add curation_name if not present but params to define it are. Helps ensure only relevant entries for key
        # are deleted
        key = copy.deepcopy(key)
        for curation_name in get_table_curation_names_for_key(self, key):
            key.update({"curation_name": curation_name})
            delete_(self, [EpochSpikeTimesRelabel], key, safemode)


def convert_df_index_unit_name(unit_id_df, sort_group_id, unit_ids=None):
    # Convert df index from unit_id to unit_name

    # Make copy of df to avoid altering outside function
    df = copy.deepcopy(unit_id_df)
    # Check that unit_id is index
    if df.index.name != "unit_id":
        raise Exception(f"df index must be unit_id")
    # Take subset of unit ids if passed
    if unit_ids is None:
        unit_ids = df.index
    df_subset = df.loc[unit_ids]
    sort_group_unit_ids = [get_unit_name(sort_group_id, unit_id) for unit_id in df_subset.index]
    df_subset.index = sort_group_unit_ids
    df_subset.index.name = "unit_name"
    return df_subset


@schema
class EpochSpikeTimesRelabel(ComputedBase):
    definition = """
    # Relabeled entries in EpochSpikeTimes
    -> EpochSpikeTimesRelabelParams
    """

    class RelabelEntries(PartBase):
        definition = """
        # Relabeled entries in EpochSpikeTimes
        -> EpochSpikeTimesRelabel
        -> EpochSpikeTimes
        ---
        -> nd.common.AnalysisNwbfile
        epoch_spike_times_object_id : varchar(100)
        """

        def in_time_intervals(self, time_intervals):
            unit_df = self.fetch1_dataframe()
            return zip(*[(unit_id, event_times_in_intervals(
                event_times=epoch_spike_times.values[0], valid_time_intervals=[interval])[1])
                         for interval in time_intervals for unit_id, epoch_spike_times in unit_df.iterrows()])

        def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="unit_id"):
            return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

        def spike_times_across_sort_groups(self, key, sort_group_unit_ids_map):
            return pd.concat([convert_df_index_unit_name(
                (self & {**key, **{"sort_group_id": sort_group_id}}).fetch1_dataframe(), sort_group_id, unit_ids)
                for sort_group_id, unit_ids in sort_group_unit_ids_map.items()], axis=0)

    def make(self, key):
        # Only populate tables if spike times entry available
        relabel_params = (EpochSpikeTimesRelabelParams & key).fetch1()
        if len((EpochSpikeTimes & relabel_params)) == 0:
            print(f"No entry in EpochSpikeTimes for {relabel_params}; cannot populate EpochSpikeTimesRelabel")
            return

        # Insert into main table
        insert1_print(self, key)

        # Insert into part table
        table_entry = (EpochSpikeTimes & relabel_params).fetch1()
        insert1_print(self.RelabelEntries, {**table_entry, **key})

    def delete_(self, key, safemode=True):
        # Add curation_name if not present but params to define it are. Helps ensure only relevant entries for key
        # are deleted
        from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikeCountsSel
        key = copy.deepcopy(key)
        for curation_name in get_table_curation_names_for_key(self, key):
            key.update({"curation_name": curation_name})
            delete_(self, [EpochMeanFiringRate, ResEpochSpikeCountsSel], key, safemode)


@schema
class EpochMeanFiringRate(ComputedBase):
    definition = """
    # Mean firing rate of units during epoch
    -> EpochSpikeTimesRelabel
    ---
    -> nd.common.AnalysisNwbfile
    epoch_mean_firing_rate_object_id : varchar(40)
    """

    def make(self, key):

        from src.jguides_2024.utils.point_process_helpers import calculate_average_event_rate

        # Get epoch valid times
        epoch_interval_list_name = EpochIntervalListName().get_interval_list_name(key["nwb_file_name"], key["epoch"])
        epoch_interval_list = (IntervalList & {"nwb_file_name": key["nwb_file_name"],
                                               "interval_list_name": epoch_interval_list_name}).fetch1("valid_times")

        # Get epoch spike times
        epoch_spike_times_df = (EpochSpikeTimesRelabel.RelabelEntries & key).fetch1_dataframe()  # epoch spike times
        mean_firing_rate_list = [calculate_average_event_rate(event_times=epoch_spike_times,
                                                        valid_time_intervals=epoch_interval_list)
                                for epoch_spike_times in epoch_spike_times_df["epoch_spike_times"]]
        mean_firing_rate_df = pd.DataFrame.from_dict({"unit_id": epoch_spike_times_df.index,
                                                      "mean_firing_rate": mean_firing_rate_list})

        # Insert into table
        insert_analysis_table_entry(self, [mean_firing_rate_df], key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="unit_id"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def delete_(self, key, safemode=True):
        # Add curation_name if not present but params to define it are. Helps ensure only relevant entries for key
        # are deleted

        key = copy.deepcopy(key)
        from src.jguides_2024.spikes.jguidera_unit import EpsUnits
        for curation_name in get_table_curation_names_for_key(self, key):
            key.update({"curation_name": curation_name})
            delete_(self, [EpsUnits], key, safemode)


def _get_kernel_standard_deviations():
    return [.1]


def populate_jguidera_spikes(key=None, tolerate_error=False):
    schema_name = "jguidera_spikes"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_spikes():
    schema.drop()
