from collections import namedtuple

import datajoint as dj
import numpy as np
from spyglass.common import IntervalList
from spyglass.spikesorting import (SpikeSortingRecording, CuratedSpikeSorting, WaveformParameters,
                                   AutomaticCurationParameters, MetricParameters, SortInterval)

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert, split_curation_name
from src.jguides_2024.metadata.jguidera_brain_region import SortGroupTargetedLocation
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch
from src.jguides_2024.time_and_trials.define_interval_list import NewIntervalList
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.utils.vector_helpers import check_all_unique

schema = dj.schema("jguidera_spikesorting")


# TODO (feature): if remake table, have only iterables in secondary key (sort_group_ids) and others in primary key
@schema
class SpikeSortingRecordingCohortParams(SecKeyParamsBase):
    definition = """
    # Specifies groups of entries from SpikeSortingRecording
    spike_sorting_recording_cohort_param_name : varchar(1000)
    ---
    sort_group_ids : blob
    nwb_file_name : varchar(40)
    sort_interval_name : varchar(80)
    preproc_params_name : varchar(40)
    team_name : varchar(40)
    """

    def insert_entry(self, nwb_file_name, sort_interval_name, preproc_params_name, sort_group_ids):
        # Populate SpikeSortingRecordingCohortParams

        key = {"nwb_file_name": nwb_file_name,
               "sort_interval_name": sort_interval_name,
               "preproc_params_name": preproc_params_name}

        # Only populate if all sort groups available, and above key restricts to a set of unique sort groups
        available_sort_group_ids = (SpikeSortingRecording & key).fetch("sort_group_id")
        check_all_unique(available_sort_group_ids)  # if this fails, must make key more restrictive
        if not all(np.isin(sort_group_ids, available_sort_group_ids)):
            print(f"Not all sort groups available for key {nwb_file_name}, {sort_interval_name}. "
                  f"Not populating SpikeSortingRecordingCohortParams. "
                  f"Desired sort group ids: {sort_group_ids} "
                  f"Available sort_group_ids: {available_sort_group_ids}")
            return

        # Populate table with parameters for spike sorting recording cohorts
        for ssr_entry in (SpikeSortingRecording & key).proj():
            column_names = ["nwb_file_name", "sort_interval_name", "preproc_params_name", "team_name"]
            secondary_key_subset_map = {
                "sort_group_ids": sort_group_ids, "nwb_file_name": nwb_file_name, "sort_interval_name":
                 sort_interval_name, "preproc_params_name": ssr_entry["preproc_params_name"],
                 "team_name": ssr_entry["team_name"]}
            spike_sorting_recording_cohort_param_name = self.make_param_name(secondary_key_subset_map)
            params_entry = {**{"spike_sorting_recording_cohort_param_name": spike_sorting_recording_cohort_param_name,
                               "sort_group_ids": sort_group_ids},
                            **{column_name: ssr_entry[column_name] for column_name in column_names}}
            self.insert1(params_entry, skip_duplicates=True)

    def make_param_name(self, secondary_key_subset_map):
        return self._make_param_name(
                secondary_key_subset_map, separating_character="_", tolerate_non_unique=True)


@schema
class SpikeSortingRecordingCohort(ComputedBase):
    definition = """
    # Groups of entries in SpikeSortingRecording
    -> SpikeSortingRecordingCohortParams
    """

    class CohortEntries(dj.Part):
        definition = """
        # Entries from SpikeSortingRecording
        -> master
        -> SpikeSortingRecording
        ---
        recording_path : varchar(200)
        sort_interval_list_name : varchar(200)
        """

    def make(self, key):

        ssr_cohort_params = (SpikeSortingRecordingCohortParams & key).fetch1()

        # Only populate tables if all sort groups available
        available_sort_group_ids = (SpikeSortingRecording & {"nwb_file_name": ssr_cohort_params["nwb_file_name"],
                                                             "sort_interval_name":
                                                                 ssr_cohort_params["sort_interval_name"]}).fetch(
            "sort_group_id")
        check_all_unique(available_sort_group_ids)
        if not all(np.isin(ssr_cohort_params["sort_group_ids"], available_sort_group_ids)):
            print(f"Not all sort groups available for key {key}. Not populating SpikeSortingRecordingCohort.")
            return
        # Insert into main table
        self.insert1(key)
        print('Populated SpikeSortingRecordingCohort for cohort '
              '{spike_sorting_recording_cohort_param_name}'.format(**key))
        # Insert into parts table
        for sort_group_id in ssr_cohort_params["sort_group_ids"]:
            table_entry = (SpikeSortingRecording & {**ssr_cohort_params,
                                                    **{"sort_group_id": sort_group_id}}).fetch1()

            entry_key = {**table_entry, **key}
            SpikeSortingRecordingCohort.CohortEntries.insert1(entry_key, skip_duplicates=True)
            print('Added an entry to SpikeSortingRecording.CohortEntries for cohort '
                  '{spike_sorting_recording_cohort_param_name}'.format(**entry_key))


def get_default_metric_params_name():
    max_spikes_for_nn = 20000
    return get_metric_params_name(max_spikes_for_nn)


def get_metric_params_name(max_spikes_for_nn):
    # Metrics
    # Insert metrics params with max spikes
    version_flag = "_v2"
    metric_params_name = f"peak_offset_num_spikes_{max_spikes_for_nn}spikes{version_flag}"
    starting_params = (MetricParameters & {
        "metric_params_name": f"peak_offset_num_spikes{version_flag}"}).fetch1()
    starting_params.update({"metric_params_name": metric_params_name})
    starting_params["metric_params"]["nn_isolation"]["max_spikes"] = max_spikes_for_nn
    starting_params["metric_params"]["nn_noise_overlap"]["max_spikes"] = max_spikes_for_nn
    MetricParameters.insert1(starting_params, skip_duplicates=True)
    return metric_params_name


def get_waveform_params_name(max_spikes_per_unit, n_jobs):
    # Insert whitened waveform params with specified number of max spikes and jobs
    waveform_params_name = f"default_whitened_{max_spikes_per_unit}spikes_{n_jobs}jobs"
    WaveformParameters().insert_default()
    starting_params = (WaveformParameters &
                       {"waveform_params_name": "default_whitened"}).fetch1()
    starting_params["waveform_params_name"] = waveform_params_name
    starting_params["waveform_params"].update({"max_spikes_per_unit": max_spikes_per_unit,
                                               "n_jobs": n_jobs})
    WaveformParameters.insert1(starting_params, skip_duplicates=True)
    return waveform_params_name


def get_default_waveform_params_name():
    max_spikes_per_unit = 20000
    n_jobs = 20
    return get_waveform_params_name(max_spikes_per_unit, n_jobs)


def get_default_automatic_curation_params_name():
    noise_threshold = .03
    isi_violation_threshold = .0025
    peak_offset_threshold = 2
    return get_automatic_curation_params_name(noise_threshold, isi_violation_threshold, peak_offset_threshold)


def get_automatic_curation_params_name(noise_threshold, isi_violation_threshold, peak_offset_threshold):
    AutomaticCurationParameters().insert_default()
    auto_curation_params_name = f"noise{noise_threshold}_isi{isi_violation_threshold}_offset{peak_offset_threshold}"
    label_params = {'nn_noise_overlap': ['>', noise_threshold, ['noise', 'reject']],
                    'isi_violation': ['>', isi_violation_threshold, ['noise', 'reject']],
                    'peak_offset': ['>', peak_offset_threshold, ['noise', 'reject']]}
    AutomaticCurationParameters().insert1({'auto_curation_params_name': auto_curation_params_name,
                                           'merge_params': {}, 'label_params': label_params}, skip_duplicates=True)
    return auto_curation_params_name


def return_spikesorting_params():
    parameter_set_dict = {
        k: {} for k in ["sorter_params_name", "preproc_params_name", "artifact", "waveform_params_name"]}

    # Different params across regions
    for region in ["CA1"]:
        parameter_set_dict["sorter_params_name"][region] = "franklab_tetrode_hippocampus_30KHz"
        parameter_set_dict["preproc_params_name"][region] = "franklab_tetrode_hippocampus_min_seg"
        from src.jguides_2024.spike_sorting_curation.jguidera_artifact import ArtifactDetectionAcrossSortGroupsParams  # local import to avoid circular import
        parameter_set_dict["artifact"][region] = ArtifactDetectionAcrossSortGroupsParams().get_default_param_name()
    for region in ["mPFC", "OFC", "Cortex"]:
        parameter_set_dict["sorter_params_name"][region] = "franklab_probe_ctx_30KHz_115rad"
        parameter_set_dict["preproc_params_name"][region] = "default_min_seg"
        from src.jguides_2024.spike_sorting_curation.jguidera_artifact import return_global_artifact_detection_params
        parameter_set_dict["artifact"][region] = list(return_global_artifact_detection_params().keys())[0]

    # Same param for all regions
    for region in ["CA1", "mPFC", "OFC", "Cortex"]:
        parameter_set_dict["waveform_params_name"][region] = get_default_waveform_params_name()

    return parameter_set_dict


def print_sort_groups_CuratedSpikeSorting(nwb_file_names=None,
                                          curation_id=1,
                                          sort_interval_name="raw data valid times no premaze no home",
                                          sorter="mountainsort4"):
    if nwb_file_names is None:
        from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_names
        nwb_file_names = get_jguidera_nwbf_names(high_priority=True, highest_priority=False)
    for nwb_file_name in nwb_file_names:
        sort_group_ids = np.sort((CuratedSpikeSorting & {"nwb_file_name": nwb_file_name,
                                                         "curation_id": curation_id,
                                                         "sort_interval_name": sort_interval_name,
                                                        "sorter": sorter}).fetch("sort_group_id"))
        all_sort_group_ids = (SortGroupTargetedLocation & {"nwb_file_name": nwb_file_name}).fetch("sort_group_id")
        missing_sort_group_ids = [x for x in all_sort_group_ids if x not in sort_group_ids]
        print(f"Sort group ids in CuratedSpikeSorting for {nwb_file_name}, sort interval {sort_interval_name}, "
              f"curation_id {curation_id}: {sort_group_ids}")
        print(f"Number of unique sort group ids: {len(np.unique(sort_group_ids))}")
        print(f"Sort group ids in SortGroupTargetedLocation not yet in CuratedSpikeSorting: {missing_sort_group_ids}\n")


def targeted_region_filter_parameter_set_name_map():
    return {"OFC": "franklab_default_cortex",
            "mPFC": "franklab_default_cortex",
            "CA1": "franklab_default_hippocampus"}


class DefineSortInterval:
    def __init__(self, starting_interval_list_names, nwb_file_name, NO_PREMAZE, NO_HOME, NO_SLEEP):
        self.starting_interval_list_names = starting_interval_list_names
        self.nwb_file_name = nwb_file_name
        self.NO_PREMAZE = NO_PREMAZE
        self.NO_HOME = NO_HOME
        self.NO_SLEEP = NO_SLEEP

    @staticmethod
    # Define sort interval name as interval list name and vice versa
    def convert_sort_interval_name_interval_list_name(x):
        return x

    @classmethod
    def get_epochs_for_sort_interval_name(cls, nwb_file_name, sort_interval_name):
        # Get epochs encompassed by a given sort interval

        # Get interval list name from sort interval name
        interval_list_name = cls.convert_sort_interval_name_interval_list_name(sort_interval_name)

        # Get starting interval list names from interval list name
        return NewIntervalList.get_epochs_for_new_interval_list_name(nwb_file_name, interval_list_name)

    @classmethod
    def get_epochs_for_curation_name(cls, nwb_file_name, curation_name):
        # Get epochs encompassed by a given curation name
        # Approach: split curation name into sort interval name and curation ID,
        # and get epochs encompassed by sort interval name (since curation ID not relevant
        # for this)
        sort_interval_name, curation_id = split_curation_name(curation_name)
        return cls.get_epochs_for_sort_interval_name(nwb_file_name, sort_interval_name)

    @classmethod
    def define_sort_interval_as_interval_list(cls, interval_list_name,
                                              interval_list,
                                              nwb_file_name):
        sort_interval_name = cls.convert_sort_interval_name_interval_list_name(interval_list_name)
        sort_interval = np.asarray([interval_list[0][0], interval_list[-1][1]])
        SortInterval.insert1({'nwb_file_name': nwb_file_name,
                              'sort_interval_name': sort_interval_name,
                              'sort_interval': sort_interval}, skip_duplicates=True)
        return sort_interval_name, sort_interval

    def get_sort_interval_obj(self):
        # Populate premaze durations table in case needed
        from src.jguides_2024.metadata.jguidera_premaze_durations import PremazeDurations
        PremazeDurations().insert_defaults()

        # Define sort interval
        from src.jguides_2024.time_and_trials.define_interval_list import NewIntervalList
        obj = NewIntervalList(
            starting_interval_list_names=self.starting_interval_list_names,
            nwb_file_name=self.nwb_file_name,
            NO_PREMAZE=self.NO_PREMAZE,
            NO_HOME=self.NO_HOME,
            NO_SLEEP=self.NO_SLEEP)
        IntervalList.insert1(
            {"nwb_file_name": self.nwb_file_name, "interval_list_name": obj.new_interval_list_name,
             "valid_times": obj.new_interval_list}, skip_duplicates=True)

        # Define sort interval
        sort_interval_name, sort_interval = self.define_sort_interval_as_interval_list(
            obj.new_interval_list_name, obj.new_interval_list, self.nwb_file_name)
        return namedtuple("SortIntervalQuantities", "interval_list_name sort_interval_name")(
            obj.new_interval_list_name, sort_interval_name)


# Define valid times for sort
def define_sort_intervals(targeted_location, nwb_file_name, curation_set_name="runs_analysis_v1"):
    """
    Define sort intervals based on brain region, for full recording day sorts
    Current approach: sort cortical electrodes across concatenated run and sleeps; sort HPc
    electrodes across concatenated runs except for some nwb files in which hippocampal tetrodes
    are unstable
    :param targeted_location: str, targeted brain region (e.g. CA1, OFC, mPFC)
    :param nwb_file_name: str, name of nwb file
    :return: interval lists
    """

    # Define sort interval params based on brain region and nwb_file_name

    # Define for analysis on runs sessions
    if curation_set_name == "runs_analysis_v1":
        # Case 1: concatenated run and sleep (default)
        starting_interval_list_names_list = [[
            "raw data valid times"]]  # use these interval lists and make changes as indicated by flags
        NO_PREMAZE = True  # True to exclude periods when rat being carried to maze
        NO_HOME = True  # True to exclude home epochs
        NO_SLEEP = False  # True to exclude sleep epochs

        # Case 2: concatenated run (default for HPc)
        if targeted_location == "CA1":
            NO_SLEEP = True

        # Case 3: single runs (unstable HPc)
        unstable_hpc_nwb_file_names = ["peanut20201107_.nwb", "peanut20201108_.nwb", "peanut20201109_.nwb"]
        if nwb_file_name in unstable_hpc_nwb_file_names and targeted_location == "CA1":
            starting_interval_list_names_list = [
                [EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch)]
                for epoch in (RunEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch")]

    else:
        raise Exception(f"No code written for case where curation_set_name {curation_set_name}")

    return [DefineSortInterval(
        starting_interval_list_names, nwb_file_name, NO_PREMAZE, NO_HOME, NO_SLEEP).get_sort_interval_obj()
            for starting_interval_list_names in starting_interval_list_names_list]


# # Delete unwanted artifact removed interval lists
# from spyglass.spikesorting import ArtifactRemovedIntervalList
# interval_list_names = (ArtifactRemovedIntervalList).fetch("artifact_removed_interval_list_name")
# delete_interval_list_names = [x for x in interval_list_names if "^" not in x and "no sleep" in x and "no home" in x]
# for x in delete_interval_list_names:
#     print(f"DELETING artifact_removed_interval_list_name: {x}")
#     (ArtifactRemovedIntervalList & {"artifact_removed_interval_list_name": x}).delete()


def populate_jguidera_spikesorting(key=None, tolerate_error=False):
    schema_name = "jguidera_spikesorting"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_spikesorting():
    schema.drop()


