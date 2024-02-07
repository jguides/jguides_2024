import itertools
import multiprocessing as mp

import numpy as np

from src.jguides_2024.utils.df_helpers import df_from_data_list
from src.jguides_2024.utils.for_loop_helpers import print_iteration_progress
from src.jguides_2024.utils.parallelization_helpers import show_error
from src.jguides_2024.utils.string_helpers import format_optional_var


def _get_setting_bool(half_setting):
    # Return boolean indicating which entries in df match half of setting at all characteristics
    return np.prod(np.vstack([getattr(df, k) == v for k, v in zip(characteristics, half_setting)]), axis=0)


def apply_setting_bools(setting):
    # Index separating one half of setting from the other. Each half has a value for each of the characteristics
    middle_idx = int(len(setting) / 2)

    # Create a mask to pick out all PAIRS of df idxs that each match their half of setting at all characteristics
    # Combine with passed mask (valid_bool) via "and" operation
    # note that * below performs element-wise multiplication
    b1 = _get_setting_bool(setting[:middle_idx])
    b2 = _get_setting_bool(setting[middle_idx:])
    mask = np.outer(b1, b2)  # asymmetric metric
    # Symmetric metric: get other ordering of combination also
    # if np.sum(mask) > 1:
    #     raise Exception
    if all(setting == np.asarray([1.0, 14, 'handle_well_to_right_well_leave_trial', 1.0, 14,
       'handle_well_to_right_well_leave_trial'])):
        raise Exception
    if symmetric_metric:
        mask = mask + np.outer(b2, b1) > 0  # "or" operation
    mask = mask * valid_bool  # "and" operation with passed mask

    # Get indices at which mask has True
    row_idxs, column_idxs = np.where(mask)

    # Index into array using indices
    arr_subset = arr[row_idxs, column_idxs]

    # Return mean and variance of all array entries corresponding to desired df pairs, along with metadata and
    # other useful quantities (len(arr_subset): how many pairs were found)
    return (*setting, np.asarray(row_idxs), np.asarray(column_idxs), np.mean(arr_subset), np.var(arr_subset),
            len(arr_subset))


def parse_matrix(df_, characteristics_, arr_, valid_bool_, settings=None, metric_name=None,
                 symmetric_metric_=True, debug_mode=False, verbose=True):

    """
    This function uses parallelization to efficiently find the mean and variance of groups of entries
    in an array, under the following conditions. The array should contain the output of some function
    that takes in two variables. Row and columns of the array correspond to "settings" of "characteristics".
    The function finds the average and variance of groups of entries with the same setting of
    characteristics.

    The approach to parsing differs if the metric is symmetric (order of function arguments does not matter)
    or asymmetric (order of function arguments matters), since in the first case, we want to group settings
    for which the first and second set of paired characteristics are identical and just in reverse order,
    whereas in the latter case, we do not.

    To motivate this function, consider the following illustrative example:
    We have a time series of firing rate vectors from neural recordings in a rat, with each vector
    occurring when the rat is on a particular
    path in a maze, and in a particular spatial bin on that path. The set of path identity and spatial bin
    forms a "setting" of the path identity and spatial bin "characteristics". We are interested in finding
    the average euclidean distance between vectors with certain settings of these characteristics. We can
    find this efficiently by applying the euclidean distance function to all possible pairs of
    vectors using built-in python functions. However, we are then left with the challenge of parsing the resulting
    distance matrix: grouping entries that correspond to the same setting of path and spatial bin, and then
    finding the average euclidean distance within each setting. This function provides an efficient way to do
    this parsing, using parallelization.

    :param df_: Pandas df. Contains values of characteristics at each row and column index in the array.
    :param characteristics_: List of characteristics to consider. Correspond to columns in df_.
    :param arr_: Result of applying metric to values with same index as rows of df_.
    :param valid_bool_: A priori mask that should be applied when parsing array.
    :param settings: A priori settings of characteristics with which to find groups of df entries and
           their average and variance.
    :param metric_name: string, name of metric, used to store results of parsing in a new df.
    :param symmetric_metric: string, whether metric is symmetric, influences how we parse df.
    :param verbose: boolean, True to print out statements, False to not.
    :param debug_mode: boolean, True to run code without parallelization
    :return: df with mean and variance of groups of array entries corresponding to one setting of characteristics,
             along with corresponding metadata (characteristic values, row and column idxs in matrix, number of
             array entries found for the setting).
    """

    # We declare variables as global, which allows us to not have to pass these to helper functions. Empirically,
    # this saves a lot of time when iterating over many items in the for loop in parallel
    global df, characteristics, arr, valid_bool, symmetric_metric
    df = df_
    characteristics = characteristics_
    arr = arr_
    valid_bool = valid_bool_
    symmetric_metric = symmetric_metric_

    # Get all possible settings of vector characteristics, if settings not passed
    # Note that settings defined below are not required to be in df. This may be useful for some analysis
    # where we want to be able to query df even if no entries exist for some setting.
    if settings is None:
        # Symmetric metric
        if symmetric_metric:
            half_settings = list(
                itertools.product(*[np.unique(getattr(df, x).values) for x in characteristics]))  # cartesian product
            # Get combinations of pairs of half settings
            # We have to add to these cases where items in the pair are the same
            settings_combinations = np.asarray(list(itertools.combinations(half_settings, r=2)) + [
                (x, x) for x in half_settings], dtype=object)
            # Reshape to merge members of a pair
            x1, x2, x3 = np.shape(settings_combinations)
            settings = np.reshape(settings_combinations, (x1, x2 * x3))
        # Asymmetric metric
        else:
            settings = list(itertools.product(*[np.unique(getattr(df, x).values) for x in characteristics * 2])
                            )  # cartesian product

    # Get matrix entries corresponding to each setting
    print(f"Parsing square matrix of size {np.shape(arr)} for {len(settings)} settings of characteristics...")

    results = []

    def append_result(x):
        results.append(x)

    # Go through loop without parallelization if want to debug
    if debug_mode:
        for idx, setting in enumerate(settings):
            results.append(apply_setting_bools(setting))

    pool = mp.Pool(mp.cpu_count())
    for idx, setting in enumerate(settings):
        if verbose:
            print_iteration_progress(idx, len(settings), target_num_print_statements=10)
        pool.apply_async(
            apply_setting_bools, args=(setting, ), callback=append_result, error_callback=show_error)
    pool.close()
    pool.join()  # waits until all processes done before running next line

    # Store results in dataframe
    metric_name = format_optional_var(metric_name, append_underscore=True)  # add underscore to metric name if passed
    column_names = list(np.concatenate([[f"{x}_{y}" for x in characteristics] for y in [1, 2]])) + [
        "row_idxs", "column_idxs", f"{metric_name}mean", f"{metric_name}variance", "num_samples"]
    # Convert column name dtype to str since cannot write np.str_ type to analysis nwb file
    column_names = [str(x) for x in column_names]
    return df_from_data_list(results, column_names)


"""
# For testing:
# Toy data
characteristics = ['x_pair_int', 'epoch', 'label']
df = pd.DataFrame.from_dict({"x_pair_int": [1, 2, 3, 1], "epoch": [2, 2, 2, 2,],
                             "label": ["center_well_to_handle_well", "center_well_to_handle_well",
                             "handle_well_to_handle_well", "center_well_to_handle_well"],
                            "diff_vector": [[0, 2, 4], [0, 0, 0], [1, 1, 1], [0, 0, 1]], })
setting = np.asarray([1, 2, "center_well_to_handle_well", 2, 2, "center_well_to_handle_well"], dtype="object")

# Check getting setting bool
middle_idx = int(len(setting)/2)
def _get_setting_bool(half_setting):
    return np.prod(np.vstack([getattr(df, k) == v for k, v in zip(characteristics, half_setting)]), axis=0)
b1 = _get_setting_bool(setting[:middle_idx])
b2 = _get_setting_bool(setting[middle_idx:])
print("first boolean:", b1)
print("second boolean:", b2)
mask_1 = np.outer(b1, b2)
print("outer product #1:\n", mask_1)
mask_2 = np.outer(b2, b1)
print("outer product #2:\n", mask_2)
mask_3 = mask_1 + mask_2 > 0
print(mask_3)

# To verify all settings are unique:
check_all_unique(["_".join([str(y) for y in x]) for x in settings])
"""