import copy

import numpy as np


def check_array_1d(arr):
    arr = np.asarray(arr)
    if np.prod(np.shape(arr)) != np.max(np.shape(arr)):
        raise Exception(f"Array is not 1D. Shape of array: {np.shape(arr)}")


def nan_array(num_rows, num_columns):
    arr = np.zeros((num_rows, num_columns))
    arr[:] = np.nan
    return arr


def array_to_list(arr):
    return [x for x in arr]


def array_to_tuple_list(arr):
    return [tuple(x) for x in arr]


def min_positive_val_arr(arr, axis=0):
    # Return the smallest positive value along an axis of an array
    arr = arr.astype('float')
    arr[arr < 0] = np.inf
    return np.min(arr, axis=axis)


def stack_adjacent_columns(arr, stack_num_columns):
    # stack_num_columns: number of adjacent columns to stack
    if len(np.shape(arr)) != 2:
        raise Exception(f"Must pass 2D array")
    num_groups, remainder = divmod(np.shape(arr)[1], stack_num_columns)
    if remainder != 0:
        raise Exception("Must be able to divide number of columns by div_factor without remainder")
    return np.asarray([np.ndarray.flatten(np.asarray(arr[:, i*stack_num_columns:(i + 1)*stack_num_columns]).T)
                        for i in np.arange(0, num_groups)]).T


def average_adjacent_columns(arr, average_num_columns, average_function=np.mean):
    # average_num_columns: number of adjacent columns to average
    if len(np.shape(arr)) != 2:
        raise Exception(f"Must pass 2D array")
    num_groups, remainder = divmod(np.shape(arr)[1], average_num_columns)
    if remainder != 0:
        raise Exception("Must be able to divide number of columns by div_factor without remainder")
    return np.asarray([np.mean(arr[:, i*average_num_columns:(i + 1)*average_num_columns], axis=1)
                        for i in np.arange(0, num_groups)]).T


def mask_upper_diagonal(arr, mask_value=0):
    mask = np.zeros_like(arr, dtype=bool)
    mask[np.tril_indices_from(mask)] = True
    arr[mask] = mask_value
    return arr


def check_dimensionality(arr, expected_num_dimensions, raise_error=True, return_outcome=False):
    num_dimensions = len(np.shape(arr))
    outcome = num_dimensions == expected_num_dimensions
    if raise_error and not outcome:
        raise Exception(f"arr should have dimensionality {expected_num_dimensions} "
                        f"but has has dimensionality {num_dimensions}")
    if return_outcome:
        return outcome


def cartesian_product(x1, x2):
    arr = np.zeros((len(x1), len(x2)), dtype=object)
    for idx_1, x_1 in enumerate(x1):
        for idx_2, x_2 in enumerate(x2):
            arr[idx_1, idx_2] = (x_1, x_2)
    return arr


def check_arrays_equal(arrays):
    ref_arr = arrays[0]
    if not all([(arr == ref_arr).all() for arr in arrays[1:]]):
        raise Exception(f"arrays not all equal")


def on_off_diagonal_ratio(arr):

    # Return ratio of array value along diagonal to average off diagonal, in each row of a symmetric matrix

    # Check that array is symmetric (tolerate nans)
    arr_copy = copy.deepcopy(arr)
    arr_copy[np.isnan(arr_copy)] = 0
    if not (arr_copy.transpose() == arr_copy).all():
        raise Exception(f"Passed array must be symmetric (tolerating nans)")

    # Find metric along diagonal in each row
    diagonal_val = np.diagonal(arr)

    # Find average metric off diagonal in each row
    masked_arr = copy.deepcopy(arr)
    np.fill_diagonal(masked_arr, np.nan)  # nan out diagonal
    off_diagonal_val = np.nanmean(masked_arr, axis=1)

    # Find ratio of metric along diagonal to average off diagonal in each row
    return diagonal_val / off_diagonal_val
