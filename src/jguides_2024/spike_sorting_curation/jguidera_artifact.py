import copy
import warnings
from functools import reduce

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import spikeinterface as si
import spyglass as nd
from spyglass.common import IntervalList
from spyglass.spikesorting import (ArtifactDetectionSelection,
                                   SpikeSortingRecording, ArtifactRemovedIntervalList,
                                   ArtifactDetectionParameters)
from spyglass.utils.nwb_helper_fn import get_valid_intervals

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import make_param_name
from src.jguides_2024.spike_sorting_curation.jguidera_spikesorting import SpikeSortingRecordingCohort, \
    SpikeSortingRecordingCohortParams
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.plot_helpers import format_ax
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import check_all_unique

schema = dj.schema("jguidera_artifact")


# For visualizing artifacts
def _scale_factors():
    # Return default scale factors for visualizing artifacts
    return {"scale_1": 10,
            "scale_2": 1000}


def _plot_above_threshold_bool(ax,
                               plot_timestamps,
                               above_thresh_bool,
                               num_traces_above_zscore_thresh=None,
                               num_traces_above_amplitude_thresh=None,
                               zscore_trace_thresh=None,
                               traces_list=None,
                               scale_1=None,
                               scale_2=None):
    if scale_1 is None:
        scale_1 = _scale_factors()["scale_1"]
    if scale_2 is None:
        scale_2 = _scale_factors()["scale_2"]
    if num_traces_above_zscore_thresh is not None:
        ax.plot(plot_timestamps,
                num_traces_above_zscore_thresh * scale_1,
                color="green", linewidth=5, label=f"num_traces_above_zscore_thresh x {scale_1}")
        if zscore_trace_thresh is not None and traces_list is not None:  # color green trace values above zscore thresh
            for trace, thresh in zip(traces_list, zscore_trace_thresh):
                artifact_idxs = trace > thresh
                ax.plot(plot_timestamps[artifact_idxs], trace[artifact_idxs], color="green", linewidth=1, alpha=.5)
    if num_traces_above_amplitude_thresh is not None:
        ax.plot(plot_timestamps,
                num_traces_above_amplitude_thresh * scale_1,
                color="brown", linewidth=1, label=f"num_traces_above_amplitude_thresh x {scale_1}")
    ax.plot(plot_timestamps, above_thresh_bool * scale_2,
            color="red", linewidth=4, label=f"above_thresh_bool x {scale_2}")


def _plot_min_num_traces_thresh(ax, x_extent, min_num_traces_above_thresh, num_traces, scale_1=None):
    if scale_1 is None:
        scale_1 = _scale_factors()["scale_1"]
    for (label, plot_y), color in zip({"min_num_traces_above_thresh": min_num_traces_above_thresh,
                                       "number contacts": num_traces}.items(), ["blue", "black"]):
        ax.plot(x_extent, [plot_y * scale_1] * 2, color=color, label=f"{label} x {scale_1}")


def _plot_traces(ax, traces_list, plot_timestamps):
    for trace_idx, trace in enumerate(traces_list):
        label = None
        if trace_idx == 0:
            label = "trace"
        ax.plot(plot_timestamps, trace, color="black", alpha=.2, label=label)


def _plot_thresholds(ax, x_extent, amplitude_thresh=None, zscore_trace_thresh=None):
    if amplitude_thresh is not None:
        ax.plot(x_extent, [amplitude_thresh] * 2, color="purple", label="amplitude threshold")
    if zscore_trace_thresh is not None:
        for thresh_idx, thresh in enumerate(zscore_trace_thresh):
            label = None
            if thresh_idx == 0:
                label = "zscore trace thresh"
            ax.plot(x_extent, [thresh] * 2,
                    color="crimson", label=label, alpha=.5)


def _plot_thresh_traces_wrapper(traces,
                                plot_trace_idxs,
                                plot_timestamps,
                                above_thresh_bool,
                                min_num_traces_above_thresh,
                                num_traces_above_zscore_thresh=None,
                                num_traces_above_amplitude_thresh=None,
                                amplitude_thresh=None,
                                zscore_trace_thresh=None,
                                scale_1=None,
                                scale_2=None,
                                ax=None,
                                title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 3))
    x_extent = [plot_timestamps[0], plot_timestamps[-1]]
    traces_list = [traces[:, idx] for idx in plot_trace_idxs]
    # Plot number contacts above threshold
    _plot_above_threshold_bool(ax,
                               plot_timestamps,
                               above_thresh_bool,
                               num_traces_above_zscore_thresh,
                               num_traces_above_amplitude_thresh,
                               [zscore_trace_thresh[idx] for idx in plot_trace_idxs],
                               traces_list,
                               scale_1,
                               scale_2)
    # Plot minimum number of traces above threshold and total number of traces
    num_traces = np.shape(traces)[1]
    _plot_min_num_traces_thresh(ax, x_extent, min_num_traces_above_thresh, num_traces, scale_1)
    # Plot traces
    _plot_traces(ax, traces_list, plot_timestamps)
    # Plot thresholds
    _plot_thresholds(ax, x_extent, amplitude_thresh, zscore_trace_thresh)
    format_ax(ax, xlabel="time (s)", ylabel="number channels", title=title)
    ax.legend()


def _visualize_artifacts_wrapper(traces,
                                 valid_timestamps,
                                 artifact_intervals,
                                 above_thresh_bool,
                                 min_num_traces_above_thresh,
                                 num_traces_above_zscore_thresh,
                                 num_traces_above_amplitude_thresh,
                                 amplitude_thresh,
                                 zscore_trace_thresh,
                                 expand_artifact_interval_plot=0,
                                 scale_1=10,
                                 scale_2=1000,
                                 plot_trace_idxs_all_time=None,
                                 plot_trace_idxs_artifacts=None,
                                 downsample_factor=50):  # for plot of all time. 1 = no downsampling)

    from src.jguides_2024.utils.vector_helpers import index_if_not_none, expand_interval
    if plot_trace_idxs_all_time is None:
        plot_trace_idxs_all_time = [0]
    if plot_trace_idxs_artifacts is None:
        plot_trace_idxs_artifacts = np.arange(0, np.shape(traces)[1])

    # Plot each artifact, full sampling rate
    for artifact_idx, artifact_interval in enumerate(artifact_intervals):
        plot_interval = expand_interval(artifact_interval, expand_artifact_interval_plot)
        fig, ax = plt.subplots(figsize=(15, 4))
        valid_bool = event_times_in_intervals_bool(valid_timestamps, [plot_interval])
        _plot_thresh_traces_wrapper(traces[valid_bool, :],
                                    plot_trace_idxs_artifacts,
                                    valid_timestamps[valid_bool],
                                    above_thresh_bool[valid_bool],
                                    min_num_traces_above_thresh,
                                    index_if_not_none(num_traces_above_zscore_thresh, valid_bool),
                                    index_if_not_none(num_traces_above_amplitude_thresh, valid_bool),
                                    amplitude_thresh,
                                    zscore_trace_thresh,
                                    scale_1,
                                    scale_2,
                                    ax,
                                    title=f"Artifact interval #{artifact_idx}")

    # Plot all time, downsampled for tractability
    fig, ax = plt.subplots(figsize=(20, 4))
    title = f"downsampled {downsample_factor}x"
    _plot_thresh_traces_wrapper(traces[::downsample_factor],
                                plot_trace_idxs_all_time,
                                valid_timestamps[::downsample_factor],
                                above_thresh_bool[::downsample_factor],
                                min_num_traces_above_thresh,
                                num_traces_above_zscore_thresh[::downsample_factor],
                                num_traces_above_amplitude_thresh[::downsample_factor],
                                amplitude_thresh,
                                zscore_trace_thresh,
                                scale_1,
                                scale_2,
                                ax,
                                title=title)


def load_ssr_traces(ssr_key, sort_group_ids, zscore=False, verbose=True):

    """
    Load voltage traces from SpikeSortingRecording table
    :param ssr_key: dictionary, specifies a single entry in SpikeSortingRecording
    :param sort_group_ids: list, load traces for these sort groups
    :param zscore: boolean, if True return zscored trace, default is False
    :param verbose: if True, print statement with progress
    :return: traces: array with time in rows and electrodes in columns
    :return: valid_timestamps: vector with timestamps relative to recording start (so begins at zero)
    """
    # Load spikesorting recording traces from each sort group, and return these along with timestamps (note that these
    # should be same across sort groups, so just returns from one sort group)
    traces = []
    for sort_group_id in sort_group_ids:
        if verbose:
            print(f"Loading traces for sort group {sort_group_id}...")
        # Get recording object
        ssr_key.update({"sort_group_id": sort_group_id})
        recording_path = (SpikeSortingRecording & ssr_key).fetch1('recording_path')
        recording = si.load_extractor(recording_path)

        # Get timestamps (important to do BEFORE concatenating recording segments, as concatenation disrupts
        # timestamp information)
        timestamps = SpikeSortingRecording._get_recording_timestamps(recording)

        # Concatenate recording segments and get traces
        if recording.get_num_segments() > 1 and isinstance(recording, si.AppendSegmentRecording):
            recording = si.concatenate_recordings(recording.recording_list)
        elif recording.get_num_segments() > 1 and isinstance(recording, si.BinaryRecordingExtractor):
            recording = si.concatenate_recordings([recording])
        sg_traces = recording.get_traces()

        # z score traces if indicated (doing here instead of at end hopefully saves memory from avoiding having
        # all traces and zscored traces as defined variables)
        if zscore:
            sg_traces = sp.stats.zscore(sg_traces)

        # Store
        traces.append(sg_traces)

    # Concatenate traces across sort groups
    traces = np.concatenate(traces, axis=1)  # concatenate across sort groups

    # Return traces and timestamps
    return traces, timestamps


def detect_artifacts_across_sort_groups(sort_group_ids,
                                        ssr_key,
                                        proportion_above_thresh,
                                        removal_window_ms,
                                        zscore_thresh=None,
                                        amplitude_thresh=None,
                                        verbose=True):
    # ssr_key: dictionary to narrow SpikeSortingRecording to desired entries.
    # Keys nwb_file_name and sort_interval_name may be sufficient in most cases.

    def _apply_min_num_traces_thresh(traces_, thresh, min_num_traces_above_thresh_):
        num_traces_above_thresh = np.sum(traces_ >= thresh, axis=1)
        above_thresh_bool_ = num_traces_above_thresh >= min_num_traces_above_thresh_
        return above_thresh_bool_, num_traces_above_thresh

    if verbose:
        print(f"Detecting artifacts across sort groups for {ssr_key}...")

    # Check inputs well defined
    if zscore_thresh is None and amplitude_thresh is None:
        raise Exception(f"Either z score or amplitude threshold must be defined")

    # Load traces and timestamps
    traces, valid_timestamps = load_ssr_traces(ssr_key, sort_group_ids, verbose=verbose)

    # Detect threshold crossings
    num_traces = np.shape(traces)[1]  # number of traces
    min_num_traces_above_thresh = np.ceil(
        proportion_above_thresh * num_traces)  # at or above this number of contacts above threhsold --> artifact
    if zscore_thresh is not None:
        zscore_trace_thresh = zscore_thresh * traces.std(0)  # convert z score to voltage for each trace
        above_zscore_thresh_bool, num_traces_above_zscore_thresh = _apply_min_num_traces_thresh(traces,
                                                                                                zscore_trace_thresh,
                                                                                                min_num_traces_above_thresh)
    if amplitude_thresh is not None:
        above_amplitude_thresh_bool, num_traces_above_amplitude_thresh = _apply_min_num_traces_thresh(traces,
                                                                                                      amplitude_thresh,
                                                                                                      min_num_traces_above_thresh)
    if zscore_thresh is not None and amplitude_thresh is not None:
        above_thresh_bool = np.logical_or(above_zscore_thresh_bool,
                                          above_amplitude_thresh_bool)
    elif amplitude_thresh is not None:
        above_thresh_bool = amplitude_thresh
    elif zscore_thresh is not None:
        above_thresh_bool = zscore_trace_thresh

    if np.sum(above_thresh_bool) == 0:  # no artifacts detected
        recording_interval = np.asarray([[valid_timestamps[0],
                                          valid_timestamps[-1]]])
        artifact_times_empty = np.asarray([])
        print("No artifacts detected.")
        return recording_interval, artifact_times_empty

    # Get timestamps within each artifact removal window and corresponding indices
    half_removal_window_s = removal_window_ms * (1 / 1000) * (1 / 2)
    above_thresh_times = valid_timestamps[above_thresh_bool]  # timestamps of artifact threshold crossings
    artifact_times = []
    artifact_indices = []
    for a in above_thresh_times:
        a_bool = np.logical_and(valid_timestamps > (a - half_removal_window_s),
                                valid_timestamps <= (a + half_removal_window_s))
        a_times = np.copy(valid_timestamps[a_bool])
        a_indices = np.argwhere(a_bool)
        artifact_times.append(a_times)
        artifact_indices.append(a_indices)
    all_artifact_times = reduce(np.union1d, artifact_times)
    all_artifact_indices = reduce(np.union1d, artifact_indices)

    # Turn artifact detected times into intervals
    # Sort artifact timestamps if not increasing
    if not np.all(all_artifact_times[:-1] <= all_artifact_times[1:]):
        warnings.warn("Warning: sorting artifact timestamps; all_artifact_times was not strictly increasing")
        all_artifact_times = np.sort(all_artifact_times)
    # Get recording sampling frequency. Get from recording object for first available sort group
    recording_path = (SpikeSortingRecording & {**ssr_key, **{"sort_group_id": sort_group_ids[0]}}).fetch1('recording_path')
    recording = si.load_extractor(recording_path)
    sampling_frequency = recording.get_sampling_frequency()
    artifact_intervals = get_valid_intervals(
        all_artifact_times, sampling_frequency, 1.5, .000001)

    # Turn all artifact detected times into -1 to easily find non-artifact intervals
    valid_timestamps_copy = copy.deepcopy(valid_timestamps)
    valid_timestamps_copy[all_artifact_indices] = -1
    artifact_removed_valid_times = get_valid_intervals(valid_timestamps_copy[valid_timestamps_copy != -1],
                                                       sampling_frequency, 1.5, 0.000001)

    if verbose:
        # Print performance
        artifact_percent_of_times = 100 * len(all_artifact_times) / len(valid_timestamps)
        print(f"{len(artifact_intervals)} artifact intervals detected; \
              {artifact_percent_of_times} % of recording valid_timestamps removed as artifact")
        # Visualize artifacts
        _visualize_artifacts_wrapper(traces,
                                     valid_timestamps,
                                     artifact_intervals,
                                     above_thresh_bool,
                                     min_num_traces_above_thresh,
                                     num_traces_above_zscore_thresh,
                                     num_traces_above_amplitude_thresh,
                                     amplitude_thresh,
                                     zscore_trace_thresh,
                                     expand_artifact_interval_plot=0,  # s
                                     scale_1=10,
                                     scale_2=1000,
                                     plot_trace_idxs_all_time=[0],
                                     plot_trace_idxs_artifacts=np.arange(0, num_traces),
                                     downsample_factor=50)
    return artifact_removed_valid_times, artifact_intervals


@schema
class ArtifactDetectionAcrossSortGroupsParams(SecKeyParamsBase):
    definition = """
    # Parameters for detecting artifacts across sort groups
    artifact_detection_across_sort_groups_param_name: varchar(200)
    ---
    proportion_above_thresh : decimal(10,5) unsigned
    amplitude_thresh : decimal(10,5) unsigned
    zscore_thresh : decimal(10,5) unsigned
    removal_window_ms : decimal(10,5) unsigned
    """

    def _default_params(self):
        return return_global_artifact_detection_params(as_dict=False)

    # Override parent class method so can return name for just first artifact detection param set
    def get_default_param_name(self):
        return self.lookup_param_name(self._default_params()[0])

    # Override parent class method so can prepend "group" to params, and so can use underscore as
    # separating character
    def _make_param_name(self, secondary_key_subset_map, separating_character=None, tolerate_non_unique=False):
        # Check that separating character either None or underscore, then set to underscore
        check_membership([separating_character], [None, "_"])
        separating_character = "_"
        # Prepend group to param name and return
        return "group_" + super()._make_param_name(secondary_key_subset_map, separating_character, tolerate_non_unique)

    def insert_defaults(self, **kwargs):
        global_artifact_detection_params = return_global_artifact_detection_params()
        for artifact_params_name, artifact_params in global_artifact_detection_params.items():
            params = {**{"artifact_detection_across_sort_groups_param_name": "group_" + artifact_params_name},
                      **artifact_params}
            self.insert1(params, skip_duplicates=True)


@schema
class ArtifactDetectionAcrossSortGroupsSelection(SelBase):
    definition = """
    # Selection from upstream tables for detecting artifacts across sort groups
    -> SpikeSortingRecordingCohort
    -> ArtifactDetectionAcrossSortGroupsParams
    """

    def insert_defaults(self, nwb_file_name=None, sort_interval_name=None):
        # Define ssr cohort entries
        if len([x is None for x in [nwb_file_name, sort_interval_name]]) == 1:
            raise Exception(f"Currently, both nwb_file_name and sort_interval must either both be specified, or "
                            f"neither must be specified")
        if nwb_file_name is not None and sort_interval_name is not None:
            spike_sorting_recording_cohort_param_name = (
                    SpikeSortingRecordingCohortParams() & {"nwb_file_name": nwb_file_name,
                                                           "sort_interval_name": sort_interval_name}).fetch1(
                "spike_sorting_recording_cohort_param_name")
            ssr_cohort_entries = [{"spike_sorting_recording_cohort_param_name":
                                   spike_sorting_recording_cohort_param_name}]
        else:
            ssr_cohort_entries = SpikeSortingRecordingCohort.proj()
        # Populate ArtifactDetectionAcrossSortGroupsSelection with combinations of entries from upstream tables
        # Restrict to default artifact detection across sort groups param
        table_subset = ArtifactDetectionAcrossSortGroupsParams & {
            ArtifactDetectionAcrossSortGroupsParams().meta_param_name():
                ArtifactDetectionAcrossSortGroupsParams().get_default_param_name()}
        for artifact_params_entry in table_subset.proj():
            for ssr_cohort_entry in ssr_cohort_entries:
                self.insert1({**artifact_params_entry, **ssr_cohort_entry}, skip_duplicates=True)


@schema
class ArtifactDetectionAcrossSortGroups(ComputedBase):
    definition = """
    # Detected artifacts across sort groups
    -> ArtifactDetectionAcrossSortGroupsSelection
    ---
    artifact_times: longblob # np array of artifact intervals
    artifact_removed_valid_times: longblob # np array of valid artifact-free intervals
    """

    def make(self, key):
        # Detect artifacts
        artifact_params = (ArtifactDetectionAcrossSortGroupsParams & key).fetch1()
        sort_group_ids = (SpikeSortingRecordingCohort.CohortEntries & key).fetch("sort_group_id")
        check_all_unique(sort_group_ids)
        ssr_key = {k: check_return_single_element((SpikeSortingRecordingCohort.CohortEntries & key).fetch(k)).single_element
                   for k in SpikeSortingRecording.primary_key if k != "sort_group_id"}
        key['artifact_removed_valid_times'], key['artifact_times'] = detect_artifacts_across_sort_groups(
            sort_group_ids=sort_group_ids,
            ssr_key=ssr_key,
            proportion_above_thresh=float(artifact_params["proportion_above_thresh"]),
            removal_window_ms=float(artifact_params["removal_window_ms"]),
            zscore_thresh=float(artifact_params["zscore_thresh"]),
            amplitude_thresh=float(artifact_params["amplitude_thresh"]))
        # Insert artifact times and valid times into lab tables for each sort group
        print("Inserting into ArtifactDetectionSelection, ArtifactRemovedIntervalList, and IntervalList...")
        for sort_group_id in sort_group_ids:
            interval_list_name_sg = (f"{ssr_key['nwb_file_name']}_{sort_group_id}_"
                                    f"{ssr_key['sort_interval_name']}_{ssr_key['preproc_params_name']}_{ssr_key['team_name']}_"
                                    f"{artifact_params['artifact_detection_across_sort_groups_param_name']}_artifact_removed_valid_times")
            key_subset = {**ssr_key,
                          **{"artifact_params_name": key["artifact_detection_across_sort_groups_param_name"]}}
            selection_key = {**key_subset, **{"custom_artifact_detection": 1}}
            artifact_removed_key = {**key_subset, **{k: key[k] for k in ["artifact_removed_valid_times",
                                                                         "artifact_times"]},
                                    **{"artifact_removed_interval_list_name": interval_list_name_sg}}
            ArtifactDetectionParameters.insert1({"artifact_params_name": key["artifact_detection_across_sort_groups_param_name"],
                                                 "artifact_params": {k:artifact_params[k] for k in artifact_params.keys()
                                                                     if k != "artifact_detection_across_sort_groups_param_name"}}, skip_duplicates=True)
            selection_key["sort_group_id"] = sort_group_id
            artifact_removed_key["sort_group_id"] = sort_group_id
            ArtifactDetectionSelection.insert1(selection_key, skip_duplicates=True)
            ArtifactRemovedIntervalList.insert1(artifact_removed_key, skip_duplicates=True)
            IntervalList.insert1({'nwb_file_name': ssr_key['nwb_file_name'],
                                  'interval_list_name': interval_list_name_sg,
                                  'valid_times': key['artifact_removed_valid_times']},
                                 skip_duplicates=True)
        # Insert into table
        print("Inserting into ArtifactDetectionAcrossSortGroups...")
        self.insert1(key)
        print(f"Added an entry to ArtifactDetectionAcrossSortGroups for {key}")


def return_global_artifact_detection_params(as_dict=True):
    param_sets = [[.25, 500, 8, 2], [.8, 2000, 8, 1]]
    # Return as dictionary if indicated
    if as_dict:
        return {make_param_name(param_set, separating_character="_"):
                                                {k: v for k, v in zip(
                ["proportion_above_thresh", "amplitude_thresh", "zscore_thresh", "removal_window_ms"], param_set)}
                                            for param_set in param_sets}
    # Otherwise return param sets
    return param_sets


def populate_ArtifactDetectionParameters():
    global_artifact_detection_params = return_global_artifact_detection_params()
    for artifact_params_name, artifact_params in global_artifact_detection_params.items():
        for prefix in ["", "group_"]:
            nd.spikesorting.ArtifactDetectionParameters.insert1(
                {"artifact_params_name": prefix + artifact_params_name, "artifact_params": artifact_params},
                skip_duplicates=True)


