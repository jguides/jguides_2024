import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.jguides_2024.utils.array_helpers import min_positive_val_arr
from src.jguides_2024.utils.interval_helpers import check_intervals_list
from src.jguides_2024.utils.plot_helpers import plot_intervals
from src.jguides_2024.utils.stats_helpers import mean_squared_error


def event_times_in_intervals_bool(event_times, valid_time_intervals):
    """
    Filter event times for those within valid_intervals
    :param event_times: array-like with times of events
    :param valid_time_intervals: nested list with intervals for valid times
    :return: boolean indicating indices in event_times within valid_time_intervals
    """
    # If no valid_time_intervals, return boolean of False same length as event_times
    if len(valid_time_intervals) == 0:
        return np.asarray([False]*len(event_times))
    # Otherwise, return boolean indicating which event times are within valid time intervals
    # (note that in case of empty valid_time_intervals, this returns False which is not what we
    # want)
    return np.sum(np.asarray([np.logical_and(event_times >= t1, event_times <= t2)
                              for t1, t2 in valid_time_intervals]), axis=0) > 0


def event_times_in_intervals(event_times, valid_time_intervals):
    """
    Filter event times for those within valid_intervals
    :param event_times: array-like with times of events
    :param valid_time_intervals: nested list with intervals for valid times
    :return: array with indices (from np.where) in original event_times of valid event times
    :return: array with valid event times
    """
    event_times = np.asarray(event_times)
    valid_bool = event_times_in_intervals_bool(event_times, valid_time_intervals)
    return np.where(valid_bool)[0], event_times[valid_bool]


def bins_in_intervals_bool(valid_intervals, bin_centers, bin_edges=None):
    bin_starts, bin_ends = copy.deepcopy(bin_centers), copy.deepcopy(bin_centers)  # default
    if bin_edges is not None:
        bin_starts, bin_ends = list(map(np.asarray, list(zip(*bin_edges))))
    return np.sum(np.vstack([event_times_in_intervals_bool(x, valid_intervals)
                             for x in [bin_centers, bin_starts, bin_ends]]), axis=0) == 3


def bins_in_intervals(valid_intervals, bin_centers, bin_edges=None, verbose=False):
    valid_bool = bins_in_intervals_bool(valid_intervals, bin_centers, bin_edges)
    valid_bin_centers = bin_centers[valid_bool]
    # Plot if indicated
    if verbose:
        fig, ax = plt.subplots(figsize=(12, 2))
        plot_intervals(valid_intervals, label="valid_intervals", ax=ax)
        ax.plot(valid_bin_centers, [1]*len(valid_bin_centers), 'x', color="red", label="valid_bin_centers")
    return valid_bin_centers

def calculate_average_event_rate(event_times, valid_time_intervals):
    """
    Calculate average event rate during a time interval
    :param event_times: array-like with times of events
    :param valid_time_intervals: nested list with intervals for valid times
    :return:
    """
    if len(np.shape(valid_time_intervals)) != 2 or np.shape(valid_time_intervals)[1] != 2:
        raise Exception(f"valid_intervals must be n by 2 array, but has shape {np.shape(valid_time_intervals)}")
    _, valid_event_times = event_times_in_intervals(event_times,
                                                    valid_time_intervals)
    return len(valid_event_times)/np.sum(np.diff(valid_time_intervals))


def ideal_poisson_error(observed_counts,
                        num_ideal_trials=100):
    observed_counts_arr = np.tile(observed_counts, (num_ideal_trials, 1))
    return mean_squared_error(np.random.poisson(observed_counts_arr), observed_counts_arr)


def not_small_diff_bool(x, diff_threshold):
    if len(x) == 0:
        return np.asarray([])
    x_diff = np.diff(x)
    valid_bool = x_diff > diff_threshold
    return (np.concatenate(([True], valid_bool)) *
                 np.concatenate((valid_bool, [True])))  # True if x value is NOT part of small gap pair


def get_event_times_relative_to_trial_start(event_times, trial_intervals, trial_start_time_shift=0):
    """
    Get time of each event from start of trial in which event occurs, and express relative to unshifted trials.
    For example, if interested in spike times (these are the event times) during well arrival trials (these
    provide the trial times) with minus/plus one second shift, express spike times on domain: -1 to 1 second
    around well arrival
    :param event_times: list of event times (e.g. spike times)
    :param trial_intervals: list of trial start/end times ([[trial start, trial end], ...])
    :param trial_start_time_shift: how much the start times of trials was shifted relative to an event of interest. We
           add this to event times relative to trials to express these in terms of the time of the event of interest
    :return: event times relative to trial start
    """
    # Check trial intervals well defined
    check_intervals_list(trial_intervals)
    # Get event times within trial intervals (reduces size of computation below)
    event_times_in_trials_idxs, event_times_trials = event_times_in_intervals(event_times, trial_intervals)
    # Get time of each event from start of each trial. In array below, event times vary along columns
    # and trial start times vary along rows
    event_times_rel_all_trials_start = np.tile(event_times_trials, (len(trial_intervals), 1)) - \
                                       np.tile(trial_intervals[:, 0], (len(event_times_trials), 1)).T
    # For each spike time, take the smallest positive (positive -> after trial start) relative time
    event_times_rel_trial_start = min_positive_val_arr(event_times_rel_all_trials_start, axis=0) + \
                                  trial_start_time_shift
    return event_times_rel_trial_start, event_times_in_trials_idxs


def get_full_event_times_relative_to_trial_start(event_times, trial_intervals, trial_start_time_shift=0):
    # Ensure trial intervals ias array
    trial_intervals = np.asarray(trial_intervals)
    # Initialize vector same length as event times
    relative_times = np.asarray([np.nan]*len(event_times))
    # Get event times relative to trial start, for event times within trials
    valid_relative_times, valid_idxs = get_event_times_relative_to_trial_start(event_times, trial_intervals,
                                                                               trial_start_time_shift)
    relative_times[valid_idxs] = valid_relative_times
    return pd.Series(relative_times, index=event_times)