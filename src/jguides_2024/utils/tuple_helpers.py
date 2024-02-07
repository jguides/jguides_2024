import numpy as np


def key_value_tuple_to_dict(list_tuples):
    keys, values = zip(*list_tuples)
    return {key: np.asarray(values)[keys == key]
                    for key in np.unique(keys)}


def index_list_of_tuples(list_tuples, valid_bool):
    return [t for idx, t in enumerate(list_tuples) if valid_bool[idx]]


def tuples_index_list(list_tuples, list_to_index):
    """
    Convert list of tuples with indices to list of tuples with values of a list at those indices
    :param list_tuples: list with tuples that contain indices to list_to_index
    :param list_to_index: list to be indexed with entries from tuples
    :return: list of tuples containing values from list_to_index
    """
    return [tuple([list_to_index[y] for y in x]) for x in list_tuples]


def unzip_as_array(list_tuples):
    # Unzip a list of tuples into arrays
    return list(map(np.asarray, list(zip(*list_tuples))))


def unzip_as_array(list_tuples):
    # Unzip a list of tuples into lists
    return list(map(list, list(zip(*list_tuples))))


def unzip_as_list(list_tuples):
    x, y = zip(*list_tuples)
    return list(x), list(y)


def reverse_pair(pair):
    # Check that pair passed
    if len(pair) != 2:
        raise Exception(f"pair must have exactly two elements")
    return (pair[1], pair[0])


def add_reversed_pairs(pairs):
    reversed_pairs = [reverse_pair(x) for x in pairs]
    # Raise error if any of reversed pairs composed of different elements are in pairs (in future if this is expected,
    # can update code to pass flag to
    # allow this)
    unique_element_reversed_pairs = [x for x in reversed_pairs if x[0] != x[1]]
    if any([x in pairs for x in unique_element_reversed_pairs]):
        raise Exception(f"at least one reversed pair already in passed pairs")
    return list(pairs) + reversed_pairs