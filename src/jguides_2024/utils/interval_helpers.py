import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spyglass.common.common_interval import interval_list_intersect

from src.jguides_2024.utils.list_helpers import check_lists_same_length
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import check_monotonic_increasing, check_all_unique, \
    unpack_single_element


def check_n_by_2(arr, error_message="arr must be two dimensional"):
    if len(np.shape(arr)) != 2:
        raise Exception(error_message)
    if np.shape(arr)[1] != 2:
        raise Exception(error_message)


def check_intervals_list(intervals, require_monotonic_increasing=True):
    # Require intervals to be n by 2 dimensional
    check_n_by_2(intervals,
                 error_message=f"arr must have dimension n x 2 (should contain a list of start/stop times in rows)")
    # Check interval starts before interval ends
    check_interval_start_before_end(*zip(*intervals))
    # Require monotonic increasing intervals if indicated
    if require_monotonic_increasing:
        check_monotonic_increasing(np.concatenate(intervals))


def unzip_intervals_list_as_arr(intervals):
    check_intervals_list(intervals, require_monotonic_increasing=True)  # check inputs
    intervals_start, intervals_end = list(zip(*intervals))
    return np.asarray(intervals_start), np.asarray(intervals_end)


def merge_close_intervals_bool(intervals, merge_threshold):
    intervals_start, intervals_end = unzip_intervals_list_as_arr(intervals)
    valid_bool = intervals_start[1:] - intervals_end[:-1] > merge_threshold  # if nth idx is False, merge n and n+1 entry
    start_bool = np.concatenate(([True], valid_bool))
    end_bool = np.concatenate((valid_bool, [True]))
    return start_bool, end_bool


def merge_close_intervals(intervals, merge_threshold):
    start_bool, end_bool = merge_close_intervals_bool(intervals, merge_threshold)
    intervals_start, intervals_end = unzip_intervals_list_as_arr(intervals)
    return list(zip(intervals_start[start_bool],
                    intervals_end[end_bool]))


def apply_merge_close_intervals(intervals_1, merge_threshold, intervals_2):
    """
    Find intervals that are "close" and merge. Apply the preserved indices to a separate array.
    :param intervals_1: n x 2 array with intervals
    :param merge_threshold: merge rows in intervals that are separated by less than or equal to this
    :param intervals_2: n x 2 array with intervals to which to apply merges found with intervals_1
    :return: intervals_2 after merging corresponding rows that were merged in intervals_1
    """
    if np.shape(intervals_1) != np.shape(intervals_2):
        raise Exception(f"intervals_1 and intervals_2 must have same shape")
    start_bool, end_bool = merge_close_intervals_bool(intervals_1, merge_threshold)
    intervals_2_start, intervals_2_end = unzip_intervals_list_as_arr(intervals_2)
    return list(zip(intervals_2_start[start_bool],
                    intervals_2_end[end_bool]))


def merge_overlapping_intervals(interval_list, verbose=False):
    # Check that intervals have form [smaller number, larger number]
    check_interval_start_before_end(*list(zip(*interval_list)))
    # Check that interval starts monotonic increasing
    check_monotonic_increasing(list(zip(*interval_list))[0])
    # Merge overlapping intervals
    nonoverlapping_intervals = []  # initialize list to store nonoverlapping intervals
    previous_valid_start, previous_end = interval_list[
        0]  # initialize variables for tracking start and end of previous interval
    for start, end in interval_list[1:]:  # for intervals after the first
        if start > previous_end:  # if start of current interval after end of last interval
            nonoverlapping_intervals.append((previous_valid_start, previous_end))  # add last valid interval to list
            previous_valid_start = start  # update previous valid start
        previous_end = np.max([previous_end, end])  # update previous end
    nonoverlapping_intervals.append((previous_valid_start, previous_end))  # add final interval to list

    # Plot original intervals and merged intervals if indicated
    if verbose:
        fig, ax = plt.subplots(figsize=(12, 2))
        for list_idx, (list_, color) in enumerate(zip([interval_list, nonoverlapping_intervals], ["black", "red"])):
            for interval in list_:
                ax.plot(interval, [list_idx]*2, color=color)
    return nonoverlapping_intervals


def check_interval_start_before_end(start_times, end_times):
    # Check that interval end times after interval start times
    if any(np.asarray(end_times) - np.asarray(start_times) < 0):
        raise Exception(f"All interval end times must be after interval start times")


def match_intervals(intervals_1, intervals_2, require_monotonic_increasing=True):
    """
    # Find which interval (in terms of index) in intervals_1 each interval in interval_2 is within
    (consider intervals_1 to be closed)
    :param intervals_1: array or list with intervals
    :param intervals_2: array or list with intervals
    :param require_monotonic_increasing: True to require each set of intervals monotonic increasing
    :return: list where each index corresponds to the same indexed interval in intervals_2, and the value
             is the index of the interval in intervals_1 that the intervals_2 interval is within
    """
    # If none, mark with np.nan

    # FOR TESTING:
    # intervals_1 = np.asarray([(1, 2), (4, 5), (6, 10)])
    # intervals_2 = np.asarray([(1, 2), (2, 5), (7, 8), (8, 10)])

    # Convert to array
    intervals_1 = np.asarray(intervals_1)
    intervals_2 = np.asarray(intervals_2)

    # Check intervals well defined
    for intervals in [intervals_1, intervals_2]:
        check_intervals_list(intervals, require_monotonic_increasing)

    # Repeat intervals_1 in columns
    intervals_1_rep = np.tile(intervals_1, (1, len(intervals_2)))

    # Repeat intervals_2 in rows (one row for each interval in intervals_1) and columns
    # (one column for each interval in intervals_2)
    intervals_2_rep = np.hstack([np.tile(x, (len(intervals_1), 1)) for x in intervals_2])

    # Take difference of above arrays
    # Even numbered columns correspond to interval starts and odd numbered columns
    # correspond to interval ends
    diff_arr = intervals_1_rep - intervals_2_rep

    # Even columns (interval starts): nonpositive if intervals_1 starts before or at intervals_2 starts
    valid_starts_bool = diff_arr[:, 0::2] <= 0

    # Odd columns (interval ends): nonnegative if intervals_1 ends after or at intervals_2 starts
    valid_ends_bool = diff_arr[:, 1::2] >= 0

    # Need both conditions above to be True if intervals_2 within intervals_1
    valid_bool = valid_starts_bool * valid_ends_bool
    rows, cols = np.where(valid_bool)

    # Ensure each interval in intervals_2 matches no more than one interval in intervals_1. This holds if and only if
    # each column value above is unique
    check_all_unique(cols)

    # Initialize array for interval_1 index match for each interval in intervals_2. We will populate this
    # with interval_1 matches for each interval in interval_2
    interval_1_idx_matches = np.asarray([np.nan] * len(intervals_2))
    interval_1_idx_matches[cols] = [int(x) for x in rows]  # ensure int type since may use an index
    # TODO: figure out why even with line above, seems like values are as float

    return interval_1_idx_matches


def match_samples_to_intervals(samples, intervals):
    """
    Return index in intervals corresponding to the interval in intervals that each sample in samples fell within
    :param samples: vector
    :param intervals: list of intervals
    :return: list with indices in intervals denoting interval in which each element in samples fell within
    """
    # Check inputs
    check_intervals_list(intervals)
    idxs = np.searchsorted(np.concatenate(intervals), samples)
    if not all(idxs % 2 == 1):
        raise Exception(f"At least one element in samples is not within any interval in intervals")
    return np.asarray((idxs - 1)/2).astype(int)


def intersect_two_intervals(interval_1, interval_2):
    if interval_1[0] > interval_2[1] or interval_2[0] > interval_1[1]:
        return []
    return [np.max([interval_1[0], interval_2[0]]), np.min([interval_1[1], interval_2[1]])]


def intersect_intervals(intervals):
    interval_1 = intervals[0]
    for interval_2 in intervals[1:]:
        interval_1 = intersect_two_intervals(interval_1, interval_2)
    return interval_1


def combine_intervals(intervals, operation):
    if operation == "intersection":
        return [intersect_intervals(intervals)]  # put in list so that same format as union: [[x1, x2], ...]
    elif operation == "union":
        return merge_overlapping_intervals(intervals)


def get_interval_lists_union(interval_lists, verbose=False):
    # Join intervals across lists and sort by start
    concatenated_intervals = np.concatenate(interval_lists)
    interval_starts = list(list(zip(*concatenated_intervals))[0])
    sort_idxs = np.argsort(interval_starts)
    concatenated_intervals = concatenated_intervals[sort_idxs]
    interval_lists_union = merge_overlapping_intervals(concatenated_intervals)

    if verbose:
        fig, ax = plt.subplots(figsize=(12, 1))
        plot_interval_lists = interval_lists + [interval_lists_union]
        colors = ["black"] * len(interval_lists) + ["green"]
        for interval_list_idx, interval_list in enumerate(plot_interval_lists):
            for x in interval_list:
                ax.plot(x, [interval_list_idx] * 2, 'o-', color=colors[interval_list_idx])

    return interval_lists_union


def combine_interval_lists(interval_list_1, interval_list_2, operation, verbose=False):
    if operation == "intersection":
        return interval_list_intersect(np.asarray(interval_list_1), np.asarray(interval_list_2))
    elif operation == "union":
        return get_interval_lists_union([interval_list_1, interval_list_2], verbose=verbose)


class CombineIntervalLists:

    """
    :param interval_lists: list of interval lists: [list_1, list_2,...] where list_n is like [[x1, x2],...]
    """

    """
    # FOR TESTING:
    interval_lists = {0: [[-1, 4], [0, 10], [12, 15], [19, 25]],
                      1: [[-2, 0]],
                      2: [[40, 50]]}
    combination_interval_list_names = [[0, 0], [1, 0], [1, 0]]
    combination_sources = [["original", "original"], ["original", "new"], ["new", "original"]]
    combination_operations = ["intersection", "intersection", "intersection"]
    obj = CombineIntervalLists(interval_lists)
    obj.get_combined_intervals(combination_interval_list_names, combination_sources, combination_operations, verbose=False)
    """


    def __init__(self, interval_lists):
        self.interval_lists = interval_lists
        self._check_inputs()

    def _check_inputs(self):
        # Check that interval_lists as dictionary
        if not isinstance(self.interval_lists, dict):
            raise Exception(f"interval_lists must be a dictionary, but is {type(self.interval_lists)}")

    def convert_check_combination_params(self, combination_interval_list_names, combination_sources,
                                         combination_operations):
        # Check params and convert to array if not None

        # Case 1: No interval list names to combine passed: check that combination source and operation are also None,
        # and only a single interval list
        if combination_interval_list_names is None:
            for variable_name, variable in {"combination_sources": combination_sources,
                                            "combination_operations": combination_operations}.items():
                if variable is not None:
                    raise Exception(f"{variable_name} must be None if combination_interval_list_names is None, "
                                    f"but is {variable}")
            if len(self.interval_lists) != 1:
                raise Exception(f"Can only have one interval list if specifying to not combine any interval lists"
                                f" (combination_interval_list_names is None), but there are "
                                f"{len(self.interval_lists)} items in interval_lists")

        # Cases 2 and 3
        else:
            # First convert to array
            combination_interval_list_names = np.asarray(combination_interval_list_names)
            combination_sources = np.asarray(combination_sources)
            combination_operations = np.asarray(combination_operations)

            # Case 2: List with interval lists is empty: raise error
            if len(combination_interval_list_names) == 0:
                raise Exception(f"No interval list names passed")

            # Case 3: More than one interval list passed
            else:
                # ...Check data shape well defined
                list(map(check_n_by_2, [combination_interval_list_names, combination_sources]))
                check_lists_same_length([combination_interval_list_names, combination_sources, combination_operations])
                # ...Check interval sources valid
                valid_combination_sources_list = ["original", "new"]
                check_membership(np.concatenate(combination_sources), valid_combination_sources_list,
                                 set_1_name="combination_sources", set_2_name="valid_combination_sources_list")
                # ...Check combination_operations valid
                valid_combination_operations = ["intersection", "union"]
                check_membership(combination_operations, valid_combination_operations, set_1_name="combination_operations",
                                 set_2_name="valid_combination_operations")
                # ...Check combination_interval_list_names valid. Here, handle two cases: 1) interval list names for
                # ORIGINAL interval lists, and 2) interval list names for NEW interval list
                # First, define map from source to valid combination interval list names
                # For new interval lists (source="new"), note that new interval lists created through combinations are
                # stored with combination number as key. So keys can take on values from 0 to one less than the
                # total number of combinations
                valid_names_map = {"original": list(self.interval_lists.keys()),
                                   "new": np.arange(0, len(combination_interval_list_names) - 1)}
                # Now handle the two cases
                flat_names = np.concatenate(combination_interval_list_names)
                flat_sources = np.concatenate(combination_sources)
                for source, valid_names in valid_names_map.items():
                    check_membership(flat_names[flat_sources == source], valid_names, "combination_interval_list_names",
                                     "valid combination_interval_list_names")

        return combination_interval_list_names, combination_sources, combination_operations

    def get_combined_intervals(self, combination_interval_list_names, combination_sources,
                               combination_operations, verbose=False):

        """
        Perform arbitrary union / intersection operations on multiple interval_lists as well as the
        resulting combined interval_lists
        :param combination_interval_list_names: n by 2 array. In each row is (index of first interval list, index of
           second interval list) for a combination of two interval lists. If None, return the single interval list in
           interval_lists
        :param combination_sources: n by 2 array. In each row is (source for first interval list, source for second
           interval list) for a combination of two interval lists. Sources must be "original", which denotes
           interval_lists, or "new", which denotes the results of previous interval list combinations. Must be None
           if combination_interval_list_names is None
        :param combination_operations: vector of length n. Each entry is the operation to perform for a combination.
           Must be None if combination_interval_list_names is None
        :param verbose: if True, print statements
        :return: list with intervals resulting from interval combinations
        """
        # Convert inputs to array and check valid
        combination_interval_list_names, combination_sources, combination_operations = \
            self.convert_check_combination_params(combination_interval_list_names,
                                                  combination_sources,
                                                  combination_operations)

        # If no interval list names to combine passed, simply return single interval list
        if combination_interval_list_names is None:
            return unpack_single_element(list(self.interval_lists.values()))

        # Otherwise, combine interval_lists as indicated
        interval_lists_map = {"original": self.interval_lists,
                              "new": dict()}
        combined_intervals = []  # default
        for combination_num, (interval_list_names, sources, operation) in enumerate(zip(combination_interval_list_names,
                                                                                  combination_sources,
                                                                                  combination_operations)):
            # Define the two interval_lists to combine in current step: these are lists of intervals whose name is
            # specified by combination_interval_list_names, from either the passed interval lists
            # (interval_source = "original") or new interval lists resulting from previous interval list
            # combinations (interval_source = "new")
            interval_lists_1_2 = [interval_lists_map[source][interval_list_name]
                                  for interval_list_name, source
                                  in zip(interval_list_names, sources)]
            # Combine intervals
            combined_intervals = combine_interval_lists(*interval_lists_1_2, operation=operation, verbose=verbose)
            # Store result to allow for future use
            interval_lists_map["new"][combination_num] = combined_intervals
            # Print progress
            if verbose:
                if combination_num == 0:
                    print(f"We start off with the following interval_lists: {self.interval_lists}\n")
                print(f"On combination {combination_num}. Interval selection params are:")
                for k, v in {"interval_list_names": interval_list_names,
                             "sources": sources}.items():
                    print(f"{k}: {v}")
                print(f"The following interval_lists were combined with {operation}: {interval_lists_1_2}")
                print(f"After {operation}, we have: {combined_intervals}")

        # Return result of final combination of intervals
        return combined_intervals


def fill_trial_values(trial_values, trials_valid_bools, invalid_value=None, index=None):
    # Fill a vector with trial values at particular idxs. This is useful for constructing a time series with a
    # categorical variable that varies across trials

    # Check inputs
    # Check valid bools same length, as they should refer to full vector we want to fill
    trials_valid_bools = np.asarray(trials_valid_bools)
    # Check same number of trial values and trials in trials valid bool
    if len(trial_values) != len(trials_valid_bools):
        raise Exception(f"trial_values and trials_valid_bools should have same length")

    # Initialize vector to return as pandas series
    filled_trial_values_vector = pd.Series([invalid_value] * np.shape(trials_valid_bools)[1], index=index)
    # Fill vector with trial values according to trials valid bools
    for trial_value, valid_bool in zip(trial_values, trials_valid_bools):
        filled_trial_values_vector[valid_bool] = trial_value
    return filled_trial_values_vector