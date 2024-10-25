import collections.abc
import copy
import itertools
import typing
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np

from src.jguides_2024.utils.check_well_defined import failed_check
from src.jguides_2024.utils.list_helpers import check_return_single_element, check_single_element
from src.jguides_2024.utils.set_helpers import check_membership, check_set_equality
from src.jguides_2024.utils.vector_helpers import unpack_single_element, check_all_unique, check_vectors_equal


def invert_dict(dictionary, unpack_single_elements=False):
    inverted_dict = {new_k: [k for k,v in dictionary.items() if v == new_k]
                       for new_k in set(dictionary.values())}
    # Unpack single elements (requires all value lists have length one)
    if unpack_single_elements:
        return {k: unpack_single_element(v) for k, v in inverted_dict.items()}
    return inverted_dict


def pairs_keys_same_value(dictionary):
    return list(itertools.chain.from_iterable([list(itertools.combinations([k for k,
                                             v in dictionary.items() if v == target_v], r=2))
                                             for target_v in set(dictionary.values())]))


def pairs_keys_different_value(dictionary):
    from src.jguides_2024.utils.list_helpers import remove_duplicate_combinations
    return remove_duplicate_combinations(list(itertools.chain.from_iterable([[(k, target_k) for k,
                                             v in dictionary.items() if v != target_v]
                                             for target_k, target_v in dictionary.items()])))


def merge_dicts(dict_list):
    # Merge dictionaries with no shared keys
    merged_dict = dict()
    for d in dict_list:
        if any([k in merged_dict.keys() for k in d.keys()]):
            raise Exception(f"Keys in passed dictionaries must all be unique")
        merged_dict.update(d)
    return merged_dict


def merge_dicts_lists(dict_list):
    # Merge lists that are values of passed dictionaries
    merged_dict = dict()
    for d in dict_list:
        for k, v in d.items():
            if not isinstance(v, list):
                raise Exception(f"dictionary values must all be lists")
            if k not in merged_dict:
                merged_dict[k] = v
            else:
                for x in v:
                    if x not in merged_dict[k]:
                        merged_dict[k].append(x)
    return merged_dict


def return_n_empty_dicts(n):
    return tuple([dict() for _ in np.arange(0, n)])


def add_pandas_index_to_dict(dictionary, pandas_obj):
    # pandas_obj should be pandas series or dataframe
    return {**dictionary, **{pandas_obj.index.name: pandas_obj.index}}


def find_keys_for_list_value(dict_, target_element):
    # Check inputs
    if not isinstance(dict_, dict):
        raise Exception(f"dict_ must be a dictionary but is {type(dict_)}")
    if not all([isinstance(x, Iterable) for x in dict_.values()]):
        raise Exception(f"All values in dict_ must be iterables")
    return [k for k, v in dict_.items() if target_element in v]


def find_key_for_list_value(dict_, target_element):
    keys = find_keys_for_list_value(dict_, target_element)
    if len(keys) == 0:
        return None
    return unpack_single_element(keys)


def dict_comprehension(keys, values, tolerate_non_unique_keys=False):
    if not tolerate_non_unique_keys:
        check_all_unique(keys)
    return {k: v for k, v in zip(keys, values)}


def dict_comprehension_repeated_keys(keys, values, sort_values=False):
    dictionary = {k: np.asarray(values)[keys == k] for k in np.unique(keys)}
    # Sort values within each key if indicated
    if sort_values:
        dictionary = {k: np.sort(v) for k, v in dictionary.items()}
    return dictionary


def dict_subset(dictionary, key_subset):
    check_membership(key_subset, dictionary.keys(), "passed keys", "available keys")
    return {k: dictionary[k] for k in key_subset}


def dict_drop_keys(dictionary, drop_keys):
    check_membership(drop_keys, dictionary.keys(), "passed keys to drop", "available keys to drop")
    return {k: v for k, v in dictionary.items() if k not in drop_keys}


def dict_list(value_lists, key_names):
    # Return a list with dictionaries
    # Each entry in value_lists should be a list with value for the key name with corresponding index in key_names
    return [{k: v for k, v in zip(key_names, x)} for x in list(zip(*value_lists))]


def unpack_dict(dictionary):
    """
    Return keys and values in separate lists
    :param dictionary: dictionary
    :return: list with keys, list with values
    """
    return list(dictionary.keys()), list(dictionary.values())


def unpack_dicts(dictionaries):
    """
    Get keys and values in separate lists for each dictionary in passed list of dictionaries, then concatenate
    lists of keys and separately lists of values
    :param dictionaries: list of dictionaries
    :return: list with concatenated keys across dictionaries, list with concatenated values across dictionaries
    """
    return list(map(np.concatenate, list(zip(*[unpack_dict(dictionary) for dictionary in dictionaries]))))


def sort_dict_by_keys(dictionary):
    return {k: dictionary[k] for k in sorted(dictionary)}


def sort_dict_by_values(dictionary, as_ordered_dict=False):
    """
    Note that newer python versions preserve order in key insertion, so dont need as_ordered_dict to be True
    with newer python version
    :param dictionary: dictionary
    :param as_ordered_dict: boolean, if True then return OrderedDict
    :return:
    """
    sorted_tuples = sorted(dictionary.items(), key=lambda x: x[1])
    if as_ordered_dict:
        sorted_dict = OrderedDict()
        for k, v in sorted_tuples:
            sorted_dict[k] = v
        return sorted_dict
    return {k: v for k, v in sorted_tuples}


def get_set_of_values_for_set_of_keys(dicts, mark_omit=True):
    # Walk through each dictionary in turn, and append to a list the values not yet in the list for a given key
    return_values = []
    for dict_idx, dict_ in enumerate(dicts):
        for key, value in dict_.items():
            if value not in [other_dict[key] if key in other_dict else None for other_dict in dicts[:dict_idx]]:
                return_values.append(value)
            else:
                if mark_omit:
                    return_values.append("_")

    return return_values


def check_shared_key_value(dicts, key, tolerate_error=False):
    # Check that dictionary have same value at a key
    return check_single_element(
        [x[key] for x in dicts], tolerate_error=tolerate_error,
        error_message=f"dictionaries do not match at key '{key}'")


def return_shared_key_value(dicts, key, tolerate_error=False):
    # Check that dictionaries have same value at a key, then return that value
    obj = check_return_single_element(
        [x[key] for x in dicts], tolerate_error=tolerate_error,
        error_message=f"dictionaries do not match at key '{key}'")
    if not obj.passed_check and tolerate_error:
        return obj.passed_check
    return check_return_single_element(
        [x[key] for x in dicts], tolerate_error=tolerate_error,
        error_message=f"dictionaries do not match at key '{key}'").single_element


def check_same_values_at_shared_keys(dicts, tolerate_error=False):
    # Check that dictionaries have same value at shared keys
    shared_keys = set.intersection(*[set(x.keys()) for x in dicts])
    passed_check = True
    for key in shared_keys:
        passed_check *= check_shared_key_value(dicts, key, tolerate_error)
    return bool(passed_check)


def add_defaults(dictionary, defaults, add_nonexistent_keys=False, replace_none=True, require_match=False):
    # If indicated, require that keys in dictionary that are also in defaults have same value as in defaults
    if require_match:
        check_same_values_at_shared_keys([defaults, dictionary])
    # Add default key value pairs to dict if key doesnt exist, if indicated
    if add_nonexistent_keys:
        for k, v in defaults.items():
            if k not in dictionary:
                dictionary[k] = v
    # Replace None with default value if indicated
    if replace_none:
        for k, v in dictionary.items():
            if v is None and k in defaults:
                dictionary[k] = defaults[k]
    return dictionary


def replace_vals(dictionary, replace_dictionary):
    """
    Replace values in a dictionary
    :param dictionary: dictionary in which to replace values
    :param replace_dictionary: map from the values to replace, to their replacements
    :return: dictionary with values replaced
    """
    new_dictionary = copy.deepcopy(dictionary)
    for k, v in dictionary.items():
        if isinstance(v, typing.Hashable):
            if v in replace_dictionary:
                new_dictionary[k] = replace_dictionary[v]
    return new_dictionary


def restrict_dictionary_list(dictionary_list, restriction_dict=None):
    if restriction_dict is None:
        return dictionary_list
    return [x for x in dictionary_list if all([restriction_dict[k] == v for k, v in x.items() if k in
                                               restriction_dict])]


def check_dict_equality(dictionaries, tolerate_nonequal_nan=False, tolerate_error=False, issue_warning=True):
    d1 = dictionaries[0]
    passed_check = True  # initialize
    for idx, d2 in enumerate(dictionaries[1:]):
        # Check that same keys
        passed_check *= check_set_equality(
            d1.keys(), d2.keys(), f"dictionary 0 in passed list", f"dictionary {idx + 1} in passed list",
            tolerate_error, issue_warning)
        # Check that at each key, same values (package d2[k] == v in iterable so can cover case
        # where v is iterable, and when v is not iterable)
        matching_vals_bool = [np.asarray([d2[k] == v]).all() for k, v in d1.items()]
        # tolerate non-equal np.nan if indicated (np.nan not always equal)
        if tolerate_nonequal_nan:
            for idx in np.where(np.invert(matching_vals_bool))[0]:
                if all([np.isnan(list(x.values())[idx]) for x in [d1, d2]]):
                    matching_vals_bool[idx] = True
        if not all(matching_vals_bool):
            passed_check *= False
            failed_check(tolerate_error, f"dictionaries 0 and {idx + 1} do not have matching values for keys",
                         issue_warning)
    return passed_check


def check_return_single_dict(dictionaries):
    check_dict_equality(dictionaries)
    return dictionaries[0]


def check_equality(x_list):
    if all([isinstance(x, dict) for x in x_list]):
        check_dict_equality(x_list)
    elif all([isinstance(x, str) for x in x_list]):  # important to come before sequence check since string is sequence
        check_single_element(x_list)
    elif all([isinstance(x, collections.abc.Sequence) for x in x_list]):  # meant to cover arrays (or lists)
        check_vectors_equal(np.asarray(x_list))  # convert to array since check fails on lists
    else:
        raise Exception(f"Only cases covered in check_equality are all elements of type: dictionary, sequence, or string")


def make_keys(primary_features, all_features):
    keys = []
    for feature_name, features in all_features.items():
        key = copy.deepcopy(primary_features)
        for feature in features:
            key.update({feature_name: feature})
            keys.append(copy.deepcopy(key))
    return keys


def compare_keys(dict_1, dict_2):
    print("keys in dict_1 not in dict_2:")
    print([x for x in dict_1.keys() if x not in list(dict_2.keys())])
    print("keys in dict_2 not in dict_1:")
    print([x for x in dict_2 if x not in list(dict_1.keys())])


def remove_repeat_dicts(dict_list):
    new_dict_list = []
    for x in dict_list:
        if x not in new_dict_list:
            new_dict_list.append(x)
    return new_dict_list