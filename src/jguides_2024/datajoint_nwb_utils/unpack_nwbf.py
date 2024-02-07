"""This module contains helper functions to unpack data in an nwb file"""

import numpy as np

from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import get_epoch_time_interval


def get_epoch_interval_nwbf(nwbf, epoch):

    nwb_file_name = f"{nwbf.fields['session_id'].replace('_', '')}_.nwb"

    return get_epoch_time_interval(nwb_file_name, epoch)


def get_epoch_timestamps_nwbf(
        nwbf, epoch, num_samples_estimation=10000000, expected_sampling_rates=[20000, 30000],
        error_tolerance_expected_vs_estimated_fs=.5, tolerance_distance_to_epoch_bounds=.002):
    """
    Get timestamps within epoch.
    :param nwbf: nwb file
    :param epoch: epoch number
    :param num_samples_estimation: number of data samples used to estimate data sampling rate
    :param expected_sampling_rates: expected data sampling rates
    :param error_tolerance_expected_vs_estimated_fs: maximum tolerated min difference between estimated sampling rate
           and expected sampling rates
    :param tolerance_distance_to_epoch_bounds: maximum tolerated difference between min/max sample times and
           epoch bounds, in seconds
    :return: timestamps within epoch
    """

    def _get_ptp_times(nwbf, start_idx, end_idx):
        return nwbf.fields["processing"]["sample_count"].data_interfaces[
                   "sample_count"].fields["timestamps"][start_idx:end_idx + 1] / (10 ** 9)  # convert from ns to seconds

    # Estimate sampling rate (fs) from data
    # estimate fs from data (using last samples because some recordings have jumps in timestamps at beginning)
    num_samples_estimation = np.min(
        [num_samples_estimation, len(nwbf.fields["acquisition"]["e-series"].fields["timestamps"])])
    estimated_fs = 1 / np.mean(np.diff(nwbf.fields["acquisition"]["e-series"].fields["timestamps"][
                                       -num_samples_estimation:]))

    # Verify it is close to an expected sampling rate
    if not any([expected_fs - estimated_fs < error_tolerance_expected_vs_estimated_fs for expected_fs in
                expected_sampling_rates]):
        raise Exception(
            f"Estimated fs not close enough to any expected fs: {expected_sampling_rates}. Estimated fs: {estimated_fs}")

    # Identify timestamps falling within epoch interval (note: must use sparse timestamps vector because full vector
    # too large to load it seems)
    # Approach: find sample times (taken at full sampling rate) in sparse_timestamps that are just outside of epoch
    # interval, and use these to select a tractable range of timestamps and sample counts in the full nwbf
    # Make sparse timestamps array (full array too large to work with)
    nth_sample = int(np.round(estimated_fs))  # to sparsify samples, take every nth_sample
    sparse_timestamps = nwbf.fields["processing"]["sample_count"].data_interfaces["sample_count"].fields["timestamps"][
                        ::nth_sample] / (
                                    10 ** 9)  # note that do not index into parent array using above idxs because slow

    # Define indices in full sample count data corresponding to sparse timestamps
    idxs_parent_array = np.arange(0, len(
        nwbf.fields["processing"]["sample_count"].data_interfaces["sample_count"].fields["timestamps"]), nth_sample)

    # Add final sample to sparse timestamps array if not present. Important for last epoch, which could end after
    # last nth sample above
    final_timestamp = nwbf.fields["processing"]["sample_count"].data_interfaces["sample_count"].fields["timestamps"]
    if final_timestamp not in sparse_timestamps:
        sparse_timestamps = np.append(sparse_timestamps, final_timestamp)  # udpate sparse timestamps
        idxs_parent_array = np.append(idxs_parent_array,
                                      len(sparse_timestamps) - 1)  # update array with corresponding idxs in full data

    # Get start and end time of epoch
    epoch_interval = get_epoch_interval_nwbf(nwbf=nwbf,
                                             epoch=epoch)

    # First, find timestamps that span epoch (just outside epoch interval) if these exist, using sparse timestamps
    # array. This helps speed up subsequent search for entries in full timestamps that are within epoch.
    # Begin by finding times in sparse_timestamps that are just outside of epoch interval
    before_epoch_idxs = np.where(sparse_timestamps < epoch_interval[0])[0]
    after_epoch_idxs = np.where(sparse_timestamps > epoch_interval[1])[0]

    # Start timestamps at these "just outside epoch" places if they exist.
    # If they do not, start timestamps at places closest to epoch boundaries (endpoints of timestamps).
    if len(before_epoch_idxs) > 0:
        start_idx = before_epoch_idxs[-1]
    else:
        start_idx = 0
    if len(after_epoch_idxs) > 0:
        end_idx = after_epoch_idxs[0]
    else:
        end_idx = -1
    flank_epoch_idxs = [idxs_parent_array[start_idx],
                        idxs_parent_array[end_idx]]

    # Now narrow to timestamps within epoch
    ptp_times_around_and_in_epoch = _get_ptp_times(nwbf,
                                                   start_idx=flank_epoch_idxs[0],
                                                   end_idx=flank_epoch_idxs[1])  # times in and around epoch
    valid_idxs = np.where(np.logical_and(ptp_times_around_and_in_epoch >= epoch_interval[0],
                                         ptp_times_around_and_in_epoch <= epoch_interval[1]))[0]  # time idxs in epoch
    ptp_times = ptp_times_around_and_in_epoch[valid_idxs]  # times in epoch

    if any([abs(t1 - t2) > tolerance_distance_to_epoch_bounds for t1, t2 in
            zip([ptp_times[0], ptp_times[-1]], epoch_interval)]):  # check that times not too far from epoch bounds
        raise Exception(f"Sample time start or end too far from epoch interval")

    # Use times identified above to take a subset of full sample times (both ptp and trodes) that span the epoch
    return {"ptp": ptp_times,
            "trodes": np.round((
           nwbf.fields["processing"]["sample_count"].data_interfaces["sample_count"].fields["data"][
           flank_epoch_idxs[0]:flank_epoch_idxs[1]]) * (
                       1000 / estimated_fs))[valid_idxs],
            "trodes_sample_count": nwbf.fields["processing"]["sample_count"].data_interfaces["sample_count"].fields["data"][
           flank_epoch_idxs[0]:flank_epoch_idxs[1]][valid_idxs]}  # convert from samples to ms and round to be able to
                                               # match to statescript times which are whole numbers


def get_epoch_dios_nwbf(nwbf, epoch, dio_name):

    # Get DIO event values and times
    dio_events_ptp = np.asarray(nwbf.fields["processing"]["behavior"]["behavioral_events"].fields[
                                    "time_series"][dio_name].fields["data"])
    dio_times_ptp = np.asarray(nwbf.fields["processing"]["behavior"]["behavioral_events"].fields[
                                   "time_series"][dio_name].fields["timestamps"])

    # Filter for DIO events during epoch and with value of one
    epoch_interval = get_epoch_interval_nwbf(nwbf=nwbf,
                                             epoch=epoch)  # get epoch interval
    epoch_filter = np.logical_and(dio_times_ptp >= epoch_interval[0],
                                  dio_times_ptp <= epoch_interval[1])  # create filter for epoch
    dio_value_filter = dio_events_ptp == 1  # create filter for DIOs with value one
    epoch_dio_events_ptp = dio_events_ptp[np.logical_and(epoch_filter, dio_value_filter)]
    epoch_dio_times_ptp = dio_times_ptp[np.logical_and(epoch_filter, dio_value_filter)]

    return epoch_dio_events_ptp, epoch_dio_times_ptp
