import numpy as np


def identify_trodes_camera_pause_at_epoch_start(camera_times, epoch_start_time):
    """
    The purpose of this function is to identify when Trodes camera module pauses for about half a second
    at the start of recordings.
    Why this is helpful: these pauses leads to a single large difference in consecutive camera timestamps at the
    beginning of epochs. Functionally, this is not problematic for recordings wherein behavior starts seconds
    after the recording starts. However, even in these cases, it is helpful to identify these events if one
    hopes to check remaining camera times for uniform spacing.
    :param camera_times: array-like with times of camera samples
    :return: boolean corresponding ot camera_times where True denotes putative Trodes camera module pause
    """

    if len(np.shape(camera_times)) > 1:  # check that camera_times is 1D array
        raise Exception(f"camera_times must be 1D array")

    # *** HARD CODED VALUES ***
    timestamps_diff_threshold = .3
    early_recording_cutoff = 5
    # *************************

    if any(camera_times < epoch_start_time):
        raise Exception(f"Camera times cannot be before epoch start time")

    # Find large jumps in camera timestamps
    # append False to return vector length of camera_times
    large_camera_time_diff = np.asarray(
        list(np.diff(camera_times) > timestamps_diff_threshold) + [False])

    # Find where elapsed time less than a threshold ("early recording", where jump happens)
    early_recording = np.cumsum(camera_times - epoch_start_time) < early_recording_cutoff  # seconds

    return np.logical_and(large_camera_time_diff, early_recording)


