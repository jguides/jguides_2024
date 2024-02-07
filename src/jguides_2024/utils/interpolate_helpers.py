import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals
from src.jguides_2024.utils.vector_helpers import series_finite_spans


def interpolate_at_average_sampling_rate(t_original, y_original, error_tolerance=.001, verbose=False):
    """
    Interpolate vector y_original at average of original sampling rate.
    This is useful if you have a vector sampled at somewhat uneven sampling rate, and require samples
    to be evenly spaced in time.
    :param t_original: times of samples (values in  y_original)
    :param y_original: samples
    :param error_tolerance: fraction of average time between samples that difference between average
                            time between samples and maximum time between samples can deviate from without raising
                            error. Use to ensure that not performing this interpolation operation on a vector with
                            very uneven time spacing between samples.
    :param verbose: if True, plot original and interpolated samples
    :return: t_new: times of interpolation. y_new: interpolated samples
    """

    # First, check that maximum distance between samples doesnt deviate too much from average
    from src.jguides_2024.utils.vector_helpers import check_uniform_spacing
    check_uniform_spacing(x=t_original,
                          error_tolerance=error_tolerance)

    # Interpolate at average sampling rate
    t_new = np.linspace(np.min(t_original), np.max(t_original), len(t_original))  # define evenly spaced times
    y_new = np.interp(x=t_new, xp=t_original, fp=y_original)  # interpolate

    # Plot old and new samples
    if verbose:
        fig, ax = plt.subplots(figsize=(20,3))
        ax.plot(t_original, y_original, 'o', color="gray")
        ax.plot(t_new, y_new, '.', color="orange", alpha=.5)
        ax.legend(labels=("original", "new"))
        _ = ax.set_ylabel("y")
        _ = ax.set_xlabel("t")

    return t_new, y_new


def downsample(t_original,
               y_original,
               downsample_time_bin_width,
               verbose=False):
    """
    Downsample a vector y_original using fs = 1/downsample_time_bin_width
    :param t_original: timestamps of vector to be downsampled
    :param y_original: vector to be downsampled
    :param downsample_time_bin_width: space between down samples
    :return: time stamps (t_new) and values of downsampled vector
    """

    t_new = np.arange(t_original[0], t_original[-1], downsample_time_bin_width)
    y_new = np.interp(x=t_new, xp=t_original, fp=y_original)
    if verbose:
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(t_original, y_original, '.')
        ax.plot(t_new, y_new, '.')
        ax.legend(("original", "downsampled"))
    return t_new, y_new


def interpolate_finite_intervals(x, new_index, verbose=False):
    x_finite_subsets = series_finite_spans(x)
    finite_x_interp = pd.Series([np.nan] * len(new_index), index=new_index)
    for x_subset in x_finite_subsets:
        valid_idxs, new_index_subset = event_times_in_intervals(
            new_index, [[x_subset.index[0], x_subset.index[-1]]])
        finite_x_interp.iloc[valid_idxs] = np.interp(new_index_subset, x_subset.index, x_subset.values)
    if verbose:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.scatter(x.index, x.values, label="original data", alpha=.5)
        ax.scatter(finite_x_interp.index, finite_x_interp.values,
                   label="interpolated finite subsets", alpha=.5)
        for subset_idx, x_subset in enumerate(x_finite_subsets):
            label = None
            if subset_idx == 0:
                label = "finite subsets of original data"
            ax.plot(x_subset, 'x', color="black", label=label)
        ax.legend()
    return finite_x_interp


def nearest_neighbor_interpolation(
        original_series, new_index, interpolate_outside_original_index=False):  # TODO: test this function
    # For each sample in new_index, find closest sample in original_series
    # Approach: for each sample in new_index, use closest sample in original_series. This must be the sample
    # in original_series that comes before or after the sample in new_index.

    new_index = np.asarray(new_index)  # convert to array

    original_index = original_series.index

    # indices of first value in original_index that is larger than each value in new_index
    after_idxs = np.searchsorted(original_index, new_index)

    # indices of first value in original_index that is smaller than each value in new_index
    before_idxs = after_idxs - 1

    # Assign idxs so can interpolate outside original index values (will mask these later if
    # interpolate_outside_original_index is False)
    before_idxs[before_idxs < 0] = 0
    after_idxs[after_idxs > len(original_index) - 1] = len(original_index) - 1

    # For each sample in new_index, determine whether sample in original_series that's before or after is closer
    # distance between values in new_index and after/before idxs above, stacked
    before_after_idx_dist = abs(np.vstack((original_index[before_idxs],
                                           original_index[after_idxs])) - new_index)
    # indices where new_index value closer to preceding index in original_index
    before_idx_bool = np.argmin(before_after_idx_dist, axis=0) == 0
    # indices where new_index value closer to subsequent index in original_index
    after_idx_bool = np.argmin(before_after_idx_dist, axis=0) == 1
    valid_before_idxs = before_idxs[before_idx_bool]
    valid_after_idxs = after_idxs[after_idx_bool]
    if len(valid_before_idxs) + len(valid_after_idxs) != len(new_index):
        raise Exception(f"Identified indices of closest matches in original_index to new_index should have "
                        f"length equal to length of new_index, but does not")
    valid_idxs = np.sort(np.concatenate((valid_before_idxs, valid_after_idxs)))

    # Apply idxs found above
    new_series = pd.Series(original_series.iloc[valid_idxs].values, index=new_index)

    # Mask samples outside original index if indicated
    if not interpolate_outside_original_index:
        invalid_idxs = np.logical_or(new_index < np.min(original_index),
                                     new_index > np.max(original_index))
        new_series.iloc[invalid_idxs] = np.nan

    return new_series