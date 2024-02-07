import itertools
from collections import namedtuple

import numpy as np


def check_single_element(x):
    """
    Check list for existence of a single unique element
    :param x: list with elements
    """
    unique_elements = np.unique(list(x))
    if len(unique_elements) != 1:
        raise Exception(f"Should have found one unique element in list but found {len(unique_elements)}")


def check_single_element(x, tolerate_error=False, error_message=None):
    unique_elements = np.unique(list(x))
    passed_check = True
    if len(unique_elements) != 1:
        passed_check = False
    if not tolerate_error and not passed_check:
        if error_message is None:
            error_message = f"Should have found one unique element in list but found {len(unique_elements)}"
        raise Exception(error_message)
    return passed_check


def check_return_single_element(x, tolerate_error=False, error_message=None):
    """
    Check list for existence of a single unique element and return if exists, otherwise raise error
    :param x: list with elements
    :return: single unique element (if list had one)
    """

    passed_check = check_single_element(x, tolerate_error, error_message)
    single_element = None
    if passed_check:
        single_element = x[0]
    # Return both whether passed check and single element (if passed check), because if just returned single element
    # (or some other variable z (e.g. None) if multiple elements), could not distinguish [z] from [x1, x2, ...]

    return namedtuple("SingleElement", "passed_check single_element")(passed_check, single_element)


def check_lists_same_length(lists, lists_description="Lists"):
    var_lengths = np.unique(list(map(len, lists)))
    if len(var_lengths) != 1:
        raise Exception(f"{lists_description} must all have same length, but set of lengths is: {var_lengths}")


def duplicate_inside_elements(x, num_duplicates=2):
    return ([x[0]]
            + list(itertools.chain.from_iterable([[x_i]*num_duplicates
                                                  for idx, x_i in enumerate(x)
                                                  if idx not in [0, len(x) - 1]]))
            + [x[-1]])


def duplicate_elements(x, num_duplicates=2):
    return list(itertools.chain.from_iterable([[x_i]*num_duplicates for x_i in x]))


def remove_duplicate_combinations(x):
    return list(set(map(tuple, map(sorted, x))))


def return_n_empty_lists(n, as_array=False):
    if as_array:
        return tuple([np.asarray([]) for _ in np.arange(0, n)])
    return tuple([[] for _ in np.arange(0, n)])


def zip_adjacent_elements(x):
    return list(zip(x[:-1], x[1:]))


def unzip_adjacent_elements(x):
    # Performs inverse operation of zip_adjacent_elements
    x_1, x_2 = zip(*x)
    return np.concatenate((x_1, [x_2[-1]]))


def append_multiple_lists(variables, lists):
    # Check inputs
    if len(variables) != len(lists):
        raise Exception(f"number of lists to be appended must be same as number of variables")
    appended_lists = []
    for variable, list_ in zip(variables, lists):
        list_.append(variable)
        appended_lists.append(list_)
    return appended_lists


def check_alternating_elements(x, element_1, element_2):
    # Check inputs
    if element_1 == element_2:
        raise Exception(f"elements 1 and 2 cannot be the same")
    x = np.asarray(x)  # convert to array
    x1_idxs = np.where(x == element_1)[0]  # where array has first element
    x2_idxs = np.where(x == element_2)[0]  # where array has second element
    # Check that x contains only elements 1 and 2
    if len(x1_idxs) + len(x2_idxs) != len(x):
        raise Exception(f"passed x contains elements other than {element_1} and {element_2}")
    # Return if one element in x
    if len(x) == 1:
        return
    # Check that idxs alternating
    # To do so, must index into idxs lists above. Use shortest length across these lists to avoid
    # indexing error.
    end_idx = np.min(list(map(len, [x1_idxs, x2_idxs])))
    if abs(check_return_single_element(x1_idxs[:end_idx] - x2_idxs[:end_idx]).single_element) != 1:
        raise Exception(f"List does not contain alternating {element_1} and {element_2}")
    # Check that last idx not same as second to last (escapes check above)
    if len(x) > 1:
        if x[-1] == x[-2]:
            raise Exception(f"List does not contain alternating {element_1} and {element_2}")


def element_frequency(x, elements=None):
    if elements is None:
        elements = set(x)
    return {element: np.sum(x == element) for element in elements}


def check_in_list(x, list_, x_name="x"):
    if x not in list_:
        raise Exception(f"{x_name} should be in {list_} but is {x}")

