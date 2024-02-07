import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from src.jguides_2024.utils.check_well_defined import check_one_none


# Note that cannot import from plot_helpers (will get circular import)


def expand_interval(interval, expand_amount=None, expand_factor=None):
    # Check inputs well defined
    if len(interval) != 2:
        raise Exception(f"interval must have two entries")
    check_one_none([expand_amount, expand_factor], ["expand_amount", "expand_factor"])
    if expand_amount is None:
        if expand_factor is None:
            expand_factor = .1
        expand_amount = unpack_single_element(np.diff(interval))*expand_factor
    return np.asarray([interval[0] - expand_amount,
                       interval[-1] + expand_amount])


def remove_repeat_elements(x, keep_first=True):
    """
    Remove consecutive elements in array
    :param x: array-like.
    :param keep_first: default True. If True, keep first element in a stretch of repeats. Otherwise, keep last.
    :return: x after removing consecutive elements.
    :return keep_x_idxs: kept indices in x after removing consecutive elements.
    """

    x = np.asarray(x)  # ensure array
    if len(x) == 0:  # if no values in array
        return x, []
    if len(x.shape) != 1:  # check that 1D array-like passed
        raise Exception(f"Must pass 1D array-like. Shape of passed item: {x.shape}")
    if keep_first:  # keep first element in repeats
        keep_x_idxs = np.concatenate((np.array([0]),
                                  np.where(x[1:] != x[:-1])[0] + 1))
    else:  # keep last element in repeats
        keep_x_idxs = np.concatenate((np.where(x[1:] != x[:-1])[0],
                              np.array([len(x) - 1])))
    return x[keep_x_idxs], keep_x_idxs


def repeat_elements_idxs(x, slice_idxs=True, as_dict=False):

    # Add one to end index if slice indices
    val = 0
    if slice_idxs:
        val = 1

    # Define indices of start/stop of same value in x (slice indices if indicated)
    idxs = list(zip(remove_repeat_elements(x)[1], remove_repeat_elements(x, keep_first=False)[1] + val))

    # If indicated, return dictionary where keys are indices and values are kept values in x
    if as_dict:
        return {j: k for j, k in zip(idxs, remove_repeat_elements(x)[0])}

    # Otherwise return in list
    return idxs


def check_vectors_equal(vectors, error_message=None):
    if error_message is None:
        error_message = "Vectors should be equal but are not"
    if not all([all(v1 == v2) for v1, v2 in zip(vectors[:-1], vectors[1:])]):
        raise Exception(error_message)


def unpack_single_vector(vectors, error_message=None):
    check_vectors_equal(vectors, error_message=error_message)
    return vectors[0]


def vector_midpoints(x):
    if len(np.shape(x)) != 1:
        raise Exception(f"x must be 1D array-like")
    return x[:-1] + np.diff(x)/2


def check_uniform_spacing(x,
                          valid_diff_mask=None,
                          error_tolerance=None):
    """
    Check that spacing between elements of x is uniform within a margin of error
    :param x: array-like, check its spacing
    :param valid_diff_mask: array-like, same length as x, True to include corresponding element in diff(x),
                                 False to exclude corresponding element in diff(x)
    :param error_tolerance: fraction of average distance between samples that difference between average
                            distance between samples and maximum distance between samples can deviate from without
                            raising error
    """

    if error_tolerance is None:
        error_tolerance = .001
    if valid_diff_mask is None:  # include all idxs if mask not passed
        valid_diff_mask = [True]*(len(x) - 1)
    x_diff = np.diff(x)[valid_diff_mask]
    epsilon = np.median(x_diff)*error_tolerance  # max tolerated sample time diff deviation from average
    if abs(np.max(abs(x_diff)) - np.median(x_diff)) > epsilon:
        raise Exception(f"Difference between sample times deviates from average by more than tolerance")


def convert_inf_to_nan(x):
    x = np.asarray(x)
    x[np.isinf(x)] = np.nan
    return x


def rescale_1d(x, y):
    """
    Rescale values in y to have the same minimum and maximum as x
    :param x: vector
    :param y: vector
    :return: vector with rescaled x
    """

    mult_factor, shift_factor = find_scale_factors(x, y)

    return y*mult_factor + shift_factor


def find_scale_factors(x, y):
    """
    Find factors such that minimum and maximum of y are equal to that of x
    after multiplying min and max of y by mult_factor and adding shift_factor.
    :param x: vector
    :param y: vector
    :return mult_factor, shift_factor
    """

    mult_factor = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
    shift_factor = np.max(x) - np.max(y) * mult_factor

    return mult_factor, shift_factor


def vectors_finite_idxs(arr):
    """
    Determine entries at which all vectors are finite
    :param arr: array with vectors in rows, or list with vectors
    :return: boolean indicating indices at which all vectors are finite
    """

    if len(np.unique([len(x) for x in arr])) != 1:
        raise Exception(f"Elements of passed list must all be same length")

    return np.isfinite(np.sum(np.asarray(arr), axis=0))


def none_to_string_none(x):
    return np.asarray(["none" if x_i is None else x_i for x_i in x])


def return_constant_vector(constant_value, vector_length):
    return np.asarray([constant_value]*vector_length)


def check_range_within(x, max_tolerated_range):
    range_x = np.nanmax(x) - np.nanmin(x)
    if range_x > max_tolerated_range:
        raise Exception(f"Elements of x have range {range_x}, which is larger "
                        f"than maximum tolerated range {max_tolerated_range}")


def check_length(x, expected_length):
    if len(x) != expected_length:
        raise Exception(f"Vector length should be length {expected_length}, but is {len(x)}")


def unpack_single_element(x, tolerate_no_entry=False, return_no_entry=None):
    if tolerate_no_entry:
        if len(x) == 0:
            return return_no_entry
        return unpack_single_element(x, tolerate_no_entry=False)
    check_length(x, expected_length=1)  # first check only one element
    return x[0]


def check_in_range(p, r):
    if len(r) != 2:
        raise Exception(f"Range must have two elements")
    if not np.logical_and(p >= np.min(r),
                          p <= np.max(r)):
        raise Exception(f"{p} not in range {r}")


def check_all_whole(x):
    if not all([x_i == int(x_i) for x_i in x]):  # return True if all entries in x are whole numbers
        raise Exception(f"Not all entries in x are whole numbers")


def check_all_unique(x, tolerate_error=False):
    passed_check = True  # default
    if len(np.unique(x)) != len(x):
        passed_check = False
    # If tolerating error, return result of check
    if tolerate_error:
        return passed_check
    # Otherwise, raise exception if failed check
    if not passed_check:
        raise Exception(f"Not all elements unique")


def find_spans_increasing_list(x, max_diff=1, slice_idxs=False, verbose=False):
    """
    Find spans of increasing elements whose difference does not exceed a threshold. INCLUDES spans with
     a single element.
    Example: if x is [0,3,5,6,7], then [(0,0),(3,3),(5,7)] will be returned.

    :param x: list of numbers
    :param max_diff: if difference between elements in x are greater than this amount, define new span
    :param slice_idxs: if True, return spans_x_idxs as slice idxs: (start, stop + 1)
    :return spans: list of tuples with (start, stop) values in x.
    :return spans_x_idxs: list of tuples with (start_idx, stop_idx), or (start_idx, stop_idx + 1) if
            slice_idxs is True, in x
    """

    # Return empty lists if x empty
    if len(x) == 0:
        return [], []

    # Check that x is increasing
    check_monotonic_increasing(x)

    # Find spans
    x_diff = np.diff(x)
    breakpoint_idxs = np.where(x_diff > max_diff)[0]
    spans_stop_idx = np.concatenate((breakpoint_idxs, np.asarray([len(x) - 1])))
    spans_start_idx = np.concatenate((np.asarray([0]), breakpoint_idxs + 1))
    spans = list(zip(x[spans_start_idx], x[spans_stop_idx]))

    # Add one to stop idxs if want slice idxs
    if slice_idxs:
        spans_stop_idx = spans_stop_idx + 1

    # Make span idxs
    spans_x_idxs = list(zip(spans_start_idx, spans_stop_idx))

    # Plot if indicated
    if verbose:
        fig, ax = plt.subplots()
        ax.plot(x, color="black")
        ax.scatter(spans_start_idx, x[spans_start_idx], color='red', s=5)
        ax.scatter(spans_stop_idx, x[spans_stop_idx], color='blue', s=5)
        ax.set_ylabel("integer")
        ax.set_xlabel("index of integer in list")

    return spans, spans_x_idxs


def series_finite_spans(x):
    """
    Return spans of finite elements in x
    :param x: series
    :return: List with contiguous intervals in x containing finite elements
    """

    return [x.iloc[i_1:i_2 + 1] for i_1, i_2 in
            find_spans_increasing_list(np.where(np.isfinite(x))[0])[0]]


def return_unique_finite_elements(x):
    return np.unique(x)[np.isfinite(np.unique(x))]


def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between vectors x and y
    :param x: vector
    :param y: vector
    :return: distance between x and y
    """

    return np.sqrt(np.sum((np.asarray(x) - np.asarray(y))**2))


def vectors_euclidean_distance(vectors, squareform=True):
    euc_dist = sp.spatial.distance.pdist(vectors, metric="euclidean")  # compressed distance matrix
    if squareform:
        euc_dist = sp.spatial.distance.squareform(euc_dist)
    return euc_dist


def vectors_cosine_similarity(vectors, squareform=True):
    cos_sim = sp.spatial.distance.pdist(vectors, metric="cosine")  # compressed distance matrix
    if squareform:
        cos_sim = sp.spatial.distance.squareform(cos_sim)
    return 1 - cos_sim  # important to subtract from one AFTER squareform since squareform assumes distance matrix


def cosine_similarity_one_to_many(reference_vector, comparison_vectors):
    return (np.matmul(comparison_vectors, reference_vector)/
                     (np.linalg.norm(comparison_vectors, axis=1)*np.linalg.norm(reference_vector)))


def cosine_similarity_vector_groups(vectors_by_group):
    if np.unique([len(np.shape(arr)) for arr in vectors_by_group]) != 2:
        raise Exception(f"All arrays must be 2D")
    if len(np.unique([np.shape(arr)[1] for arr in vectors_by_group])) != 1:
        raise Exception(f"All arrays must have same second dimension")

    def _arr_idxs(m, i, j):
        return list(map(int, m * i + j - ((i + 2) * (i + 1)) / 2))

    # Compute cosine similarity on concatenated vectors across groups
    vectors = np.concatenate(vectors_by_group)
    vectors_cosine = sp.spatial.distance.pdist(vectors, metric='cosine')

    # Match entries to group pair
    group_endpoints = [0] + list(np.cumsum(list(map(len, vectors_by_group))))
    vectors_idxs_by_group = [np.arange(x1, x2) for x1, x2 in zip(group_endpoints[:-1],
                                                                 group_endpoints[
                                                                 1:])]  # start and end idxs of groups in vectors
    num_groups = len(vectors_by_group)
    num_vectors = len(vectors)
    arr_idxs_by_group_pair = dict()
    for g1 in np.arange(0, num_groups):
        for g2 in np.arange(0, g1 + 1):
            g1_idxs = vectors_idxs_by_group[g1]
            g2_idxs = vectors_idxs_by_group[g2]
            pairs = np.asarray(list(itertools.product(g2_idxs, g1_idxs)))
            pairs = pairs[np.ndarray.flatten(np.diff(pairs)) > 0]
            if len(pairs) > 0:
                i, j = zip(*list(pairs))  # take only pairs where elements different
                arr_idxs_by_group_pair[(g1, g2)] = _arr_idxs(num_vectors, np.asarray(i), np.asarray(j))
    arr_idxs_unwrapped = np.concatenate(list(arr_idxs_by_group_pair.values()))
    check_all_unique(arr_idxs_unwrapped)
    check_length(arr_idxs_unwrapped, len(vectors_cosine))
    return {group: vectors_cosine[idxs] for group, idxs in arr_idxs_by_group_pair.items()}


class InterpolateDigitizeVector:

    def __init__(self, original_v, new_t, bin_edges):
        self.bin_edges = bin_edges
        self.vector = pd.Series(np.interp(new_t, original_v.index, original_v.values),
                                index=new_t)
        self.digitized_vector = np.digitize(self.vector.values,
                                            bin_edges)


def overlap_finite_samples(x, y, normalized=None):
    valid_bool = vectors_finite_idxs([x, y])  # indices where both vectors finite
    return overlap(x[valid_bool], y[valid_bool], normalized)


def overlap(x, y, normalized=None):
    # Get inputs if not passed
    if normalized is None:
        normalized = False

    z = list(x) + list(y)
    if not all(np.isfinite(z)):
        raise Exception(f"All elements must be finite")
    if not all(np.asarray(z) >= 0):
        raise Exception(f"All elements must be nonnegative")

    # First normalize each curve to have area of one, if indicated
    if normalized:
        x = x / np.sum(x)
        y = y / np.sum(y)

    return 2 * np.sum(np.min(np.vstack((x, y)), axis=0)) / (np.sum(x) + np.sum(y))


def truncate_symmetrically(x, desired_len):
    num_idxs_exclude = int((len(x) - desired_len)/2)
    x_truncated = x[num_idxs_exclude:-num_idxs_exclude]
    if len(x_truncated) != desired_len:
        raise Exception(f"Could not truncate vector symmetrically to match desired length: {desired_len}")
    return x_truncated


def index_if_not_none(x, idxs):
    if x is None:
        return None
    return x[idxs]


def index_vectors(vectors, idx):
    return [np.asarray(x)[idx] for x in vectors]


def check_monotonic_increasing(x):
    if not all(np.diff(x) >= 0):
        raise Exception(f"x must be monotonic increasing")


def min_max(x):
    return np.min(x), np.max(x)


def fraction_in_interval(x, interval):
    if len(interval) != 2:
        raise Exception(f"Interval must have exactly two elements")
    if not all(np.logical_and(x - interval[0] >= 0,
               x - interval[1] <= 0)):
        raise Exception(f"All elements of x must be within interval")
    return (x - interval[0])/np.diff(interval)[0]


def vector_soft_match(x1, x2, epsilon=.0000001):
    # Intended to be used to find matching values (close enough within a margin of error)
    # in two float indices. Returns the indices of the matching values in each vector (e.g. each float index).
    dist = abs(np.tile(x1, (len(x2), 1)) -
               np.tile(x2, (len(x1), 1)).T)
    valid_x2, valid_x1 = np.where(dist < epsilon)
    # Check that no value in one index matched to two values in another index
    check_all_unique(valid_x1)
    check_all_unique(valid_x2)
    return valid_x1, valid_x2


def match_increasing_elements(x1, x2, check_x2_dense_in_x1=True):
    """
    # Return element in x2 that is closest match to each element in x1
    # Approach: find idxs in x2 where each element in x1 would have to be inserted
    # to preserve order in x2
    :param x1: array
    :param x2: array
    :param check_x2_dense_in_x1: True to require that at least one element in x2 between each element in x1
    :return: closest match in x2 to each element in x1
    """

    # Check x1 and x2 increasing
    for x in [x1, x2]:
        check_monotonic_increasing(x)

    # For each x1 element, find idxs of larger, closest x2 elements
    post_match_idxs = np.searchsorted(x2, x1)  # x2 idxs such that placing x1 elements before these preserves x2 order

    # For each x1 element, find idxs of larger, closest x2 elements
    pre_match_idxs = post_match_idxs - 1
    valid_post_match_bool = post_match_idxs < len(x2)

    # Check that x2 dense in x1 if indicated.
    # Approach: if x2 dense in x1, valid_post_match_bool either should either be: 1) all True,
    # indicating for each x1 element there is a larger x2 element, or
    # 2) last entry is False and all others True, indicating for each x1 element except last there is a larger x2
    # element.
    if check_x2_dense_in_x1:
        if not all(valid_post_match_bool[:-1]):
            raise Exception(f"x2 should be dense in x1")
    post_match_idxs = post_match_idxs[valid_post_match_bool]  # exclude idxs outside x2

    # Time distance between each x1 event and closest earlier x2 event
    dist_pre = abs(x1 - x2[pre_match_idxs])

    # Time distance between each x1 event and closest later x2 event
    dist_post = abs(x1[:len(post_match_idxs)] - x2[post_match_idxs])

    # To be able to concatenate difference arrays above so that each row has one x1 event, must add difference
    # value (inf) to dist_post if there was no x2 element larger than largest x1 element
    dist_post = np.asarray(list(dist_post) + [np.inf] * np.sum(np.invert(valid_post_match_bool)))

    return np.min(np.vstack((dist_pre, dist_post)), axis=0)  # closest match in x2 to each element in x1


def linspace(start, end, approximate_bin_width):
    return np.linspace(start, end, int(np.round((1/approximate_bin_width) + 1)))


def vector_diff_arr(x1, x2):
    # Difference between each element in x1 (rows) and each element in x2 (columns)
    return np.tile(x1, (len(x2), 1)).T - np.tile(x2, (len(x1), 1))


def merged_combinations(v1, v2):
    """
    Put each combination of elements in v1 and v2 in a merged list
    :param v1: first vector
    :param v2: second return
    :return: mreged combinations of v1, v2 elements
    """
    return [list(x) + list(y) for x in v1 for y in v2]  # np.concatenate seem to force single datatype


def check_vectors_close(list_vectors, epsilon, error_message=None):
    # Define error message if not passed
    if error_message is None:
        error_message = f"vectors are not within {epsilon}, so will not join"
    # Stack vectors into an array, with one vector in each row
    arr = np.vstack(list_vectors)
    # If at any sample the largest distance between vector values is greater than epsilon, raise error
    if any(np.max(arr, axis=0) - np.min(arr, axis=0) > epsilon):
        raise Exception(error_message)


def state_durations(x):
    bout_idxs = repeat_elements_idxs(x)
    bout_durations = np.concatenate(np.diff(bout_idxs))
    bout_states = remove_repeat_elements(x)[0]
    return pd.DataFrame.from_dict({"state": bout_states,
                                   "duration": bout_durations})


def grouped_state_durations(states, durations):
    return {state: durations[states == state] for state in np.unique(states)}


def histogram_disjoint_intervals(x, bin_edges_list, tolerate_overlapping_edges=False):

    # Check inputs
    # Check each set of bin edges monotonic increasing
    for bin_edges in bin_edges_list:
        check_monotonic_increasing(bin_edges)
    # Ensure bin edges strictly non-overlapping. Note that shared edges across two bin sets could
    # lead to double counted events with np.histogram.
    bin_edges_starts = np.asarray([x[0] for x in bin_edges_list])
    bin_edges_ends = np.asarray([x[-1] for x in bin_edges_list])
    if any(bin_edges_starts[1:] - bin_edges_ends[:-1] < 0) and not tolerate_overlapping_edges:
        raise Exception(f"entries in bin_edges list must be strictly increasing, "
                        f"since otherwise events in x could be double counted")

    return np.concatenate([np.histogram(x, bins=bin_edges)[0] for bin_edges in bin_edges_list])


def unique_in_order(x):
    # This is a useful function for working with arrays or lists with text entries and want
    # unique elements in same order as in array/list. When using np.unique or set on these, it
    # seems order of elements can change

    # Check inputs
    num_dims = len(np.shape(x))
    valid_num_dims = [1, 2]
    if num_dims not in valid_num_dims:
        raise Exception(f"x must have number of dimensions in: {valid_num_dims} but has {num_dims} dimensions")

    unique_elements = []
    for x_i in x:
        if not tuple(x_i) in [tuple(z) for z in unique_elements]:
            unique_elements.append(x_i)
    return unique_elements