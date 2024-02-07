import sys
import time
from multiprocessing import Pool

import numpy as np
import psutil
import scipy.stats

from src.jguides_2024.utils.vector_helpers import vector_midpoints


# NOTE: this function must be defined at the highest level of the module to avoid a local object pickling error
def evaluate_gaussian(sample_times, mew, kernel_sd):
    return scipy.stats.norm.pdf(x=sample_times, loc=mew, scale=kernel_sd)


def convolve_point_process(event_times, kernel_sd, fs=None, sample_times=None):
    """
    Convolve point process with a Gaussian kernel
    :param event_times: point process events
    :param kernel_sd: standard deviation of Gaussian kernel used to smooth point process
    :param fs: sampling rate
    :return convolved_events: convolved point process events
    """

    if sample_times is None and fs is None:
        raise Exception(f"Either times at which to sample convolved point process or rate at which to sample"
                        f"convolved point process must be passed.")

    t1 = time.process_time()  # time process

    # Define sample times if not passed
    if sample_times is None:
        sample_times = np.arange(np.min(event_times),
                             np.max(event_times) + 1 / fs,
                             1 / fs)  # array with times at which to sample Gaussian
    # Sample Gaussian centered at each event
    with Pool(3) as p:
        gaussian_at_events_list = p.starmap(evaluate_gaussian,
                                            zip([sample_times] * len(event_times),
                                            event_times,
                                            [kernel_sd] * len(event_times)))  # shape: events x samples
    # Sum Gaussians across events
    convolved_events = np.sum(gaussian_at_events_list, axis=0)

    print(f"{time.process_time() - t1 : 2f}s elapsed")

    return convolved_events, sample_times


def convolve_point_process_efficient(event_times, kernel_sd, time_bins, error_tolerance=.00001, verbose=False):
    """
    Convolve point process with a Gaussian kernel, using an efficient approach. In short,
    histogram spikes in time bins, then put precomputed Gaussian around bins with spikes.
    Finally, sum these Gaussians.
    :param event_times: point process events
    :param kernel_sd: standard deviation of Gaussian kernel used to smooth point process
    :param time_bins: vector of bins for histogram of spike times
    :param error_tolerance: maximum tolerated difference between average time bin width and max time bin width
    :return convolved_events: convolved point process events):
    """

    # Return array of zeros if no event times
    if len(event_times) == 0:
        return np.zeros(len(time_bins) - 1), vector_midpoints(time_bins)

    # If fewer than two time bins, cannot compute sampling rate, and therefore cannot convolve
    if len(time_bins) < 2:
        raise Exception(f"Need at least two time bins to convolve")

    # Define sampling rate
    time_bin_diff = np.diff(time_bins)
    if any(abs(np.unique(time_bin_diff) - np.mean(
            time_bin_diff)) > error_tolerance):  # check that sampling rate fairly constant
        raise Exception(f"Sample time differences deviate from average more than error tolerance")
    fs = np.mean(time_bin_diff)

    # Precompute Gaussian centered at zero
    mew = 0
    one_half_times = np.arange(fs, kernel_sd * 10, fs)  # one half of the times (excluding center)
    kernel_sample_times = np.concatenate((-one_half_times[::-1], [mew], one_half_times))
    kernel = evaluate_gaussian(sample_times=kernel_sample_times, mew=mew, kernel_sd=kernel_sd)
    kernel_center_idx = np.argmin(abs(kernel_sample_times))

    # Split data into fewest partitions where processing each respects available memory
    mem = psutil.virtual_memory().available
    data_element_size = sys.getsizeof(kernel[0])/2  # note that sys may overestimate size if used directly on array
    num_time_bins = len(time_bins) - 1
    arr_size = data_element_size * len(event_times) * num_time_bins  # size of one array element times number of spikes times number of time bins
    num_partitions = int(np.ceil(arr_size / (mem * .9)))
    partition_len = int(np.ceil(len(event_times) / num_partitions))
    partitioned_event_times = [event_times[idx:idx + partition_len]
                               for idx in range(0, len(event_times), partition_len)]
    if verbose:
        print(f"split data into {num_partitions} partitions so that processing respects memory")

    # Smooth spike times in partitions
    arr_sum = np.zeros((num_partitions, num_time_bins))  # array for sum across each partition array
    for partition_idx, event_times_part in enumerate(partitioned_event_times):  # for group of spike times
        # Histogram spikes
        event_counts, _ = np.histogram(event_times_part, time_bins)  # spike counts in sample bins

        # Plop Gaussians at spike times
        # ...initialize max size array for summing gaussians across events (not all rows used if more than one
        # spike in a time bin)
        arr = np.zeros((len(event_times_part), num_time_bins))
        # ...store final arr column idx for use below
        final_arr_col_idx = np.shape(arr)[1] - 1
        # ...loop through unique numbers of spikes found in bins (e.g. 1, 2, ...)
        row_counter = 0  # counter for filling rows in array
        for num_spikes in np.unique(event_counts[event_counts > 0]):  # unique numbers of spikes found in bins
            event_bin_idxs = np.where(event_counts == num_spikes)[0]
            for event_bin_idx in event_bin_idxs:
                # Define idxs into array
                start_idx_arr = np.max([event_bin_idx - kernel_center_idx, 0])
                end_idx_arr = np.min([event_bin_idx + kernel_center_idx + 1, np.shape(arr)[1]])  # 1 accounts for slice
                # Define idxs into kernel, accounting for cases where cant use full kernel because event
                # happened relatively early or late
                start_idx_kernel = 0  # initialize
                end_idx_kernel = len(kernel)  # initialize
                # ...account for case where kernel start idx would be negative
                # note that kernel center idx is equivalent to number of samples on either side of kernel
                if event_bin_idx - kernel_center_idx < 0:
                    start_idx_kernel = abs(event_bin_idx - kernel_center_idx)
                # ...account for case where kernel end would be past array end (in terms of columns)
                if event_bin_idx + kernel_center_idx > final_arr_col_idx:
                    end_idx_kernel = len(kernel) - (event_bin_idx + kernel_center_idx - final_arr_col_idx)
                # Put kernel corresponding to current spike into array
                arr[row_counter, start_idx_arr:end_idx_arr] = kernel[start_idx_kernel:end_idx_kernel] * num_spikes
                row_counter += 1
        arr_sum[partition_idx, :] = np.sum(arr, axis=0)
    return np.sum(arr_sum, axis=0), vector_midpoints(time_bins)


"""
# Script for testing convolve_point_process_efficient: 
event_times = [1.1, 5.09, 8.8]
time_bins = np.arange(0, 10, .1)
kernel_sd = .1
x1, x2 = convolve_point_process_efficient(event_times, kernel_sd, time_bins, verbose=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 2))
plt.plot(x2, x1, '.-', label="convolved")
plt.plot(event_times, [1]*len(x), 'o', label="events")
plt.legend()
"""

