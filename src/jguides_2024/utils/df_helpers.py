import numpy as np
import pandas as pd
import sklearn

from src.jguides_2024.utils.dict_helpers import dict_comprehension, merge_dicts, check_dict_equality
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.set_helpers import check_set_equality
from src.jguides_2024.utils.vector_helpers import check_vectors_close, check_vectors_equal, \
    unpack_single_element, check_all_unique


def df_filter_columns(df, key, column_and=True):
    x = np.asarray([df[k] == v if not pd.isnull(v) else pd.isnull(df[k])
                    for k, v in key.items()]).sum(axis=0)  # use pd.isnull to avoid error with object datatype
    if column_and:
        return df[x == len(key)]
    else:
        return df[x > 0]


def df_filter_columns_greater_than(df, key, column_and=True):
    num_valid_columns = np.asarray([df[k] > v for k, v in key.items()]).sum(axis=0)  # num columns in key meeting condition
    if column_and:
        return df[num_valid_columns == len(key)]
    else:
        return df[num_valid_columns > 0]


def df_filter_columns_greater_than_equals_to(df, key, column_and=True):
    num_valid_columns = np.asarray([df[k] >= v for k, v in key.items()]).sum(axis=0)  # num columns in key meeting condition
    if column_and:
        return df[num_valid_columns == len(key)]
    else:
        return df[num_valid_columns > 0]


def df_filter_columns_less_than(df, key, column_and=True):
    num_valid_columns = np.asarray([df[k] < v for k, v in key.items()]).sum(axis=0)  # num columns in key meeting condition
    if column_and:
        return df[num_valid_columns == len(key)]
    else:
        return df[num_valid_columns > 0]


def df_filter1_columns(df, key, tolerate_no_entry=False, verbose=False, unpack_entry=False):
    df_subset = df_filter_columns(df, key)
    # Print more information if indicated
    error_message = f"Should have found exactly one entry in df for key, but found {len(df_subset)}"
    if np.logical_or(len(df_subset) > 1,
                     not tolerate_no_entry and len(df_subset) == 0):
        if verbose:
            error_message += f"\n key: {key} \n df columns: {df_subset.columns}"
        raise Exception(error_message)
    # Return single entry unpacked if indicated, otherwise return as is
    if unpack_entry and len(df_subset) > 0:
        return df_subset.iloc[0]
    return df_subset


def df_pop(df, key, column, tolerate_no_entry=False, verbose=False):
    df_subset = df_filter1_columns(df, key, tolerate_no_entry, verbose)
    if len(df_subset) == 0:  # empty df
        return df_subset
    return df_subset.iloc[0][column]


def df_filter_columns_isin(df, key):
    if len(key) == 0:  # if empty key
        return df
    return df[np.sum(np.asarray([df[k].isin(v) for k, v in key.items()]), axis=0) == len(key)]
    # Alternate code: df[df[list(df_filter)].isin(df_filter).all(axis=1)]


def df_filter_columns_contains(df, target_column, target_str):
    return df[df[target_column].str.contains(target_str)]


def zip_df_columns(df, column_names=None):
    if column_names is None:
        column_names = df.columns
    # Return iterable of tuples with column values for each row
    return zip(*[df[column_name] for column_name in column_names])


def unpack_df_columns(df, column_names, pop_single_entries=False):
    # Check inputs
    if not isinstance(df, pd.DataFrame):
        raise Exception(f"passed argument for df is not a pandas dataframe. Instead, it is: {type(df)}")
    # TODO (feature): Turning series into tuple seems to get rid of series. So should find alternative strategy for this function.
    # For dfs with single row, this works: df_subset[["fold_lens", "y_test", "y_test_predicted"]].to_numpy()[0]
    # Should see if can change to make work with any number of rows.
    if len(df) == 1 and pop_single_entries:
        return df[column_names].to_numpy()[0]
    return tuple([np.asarray(x) for x in zip(*zip_df_columns(df, column_names))])


def df_filter_index(df, valid_intervals):
    from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
    return df[event_times_in_intervals_bool(event_times=df.index, valid_time_intervals=valid_intervals)]


def df_column_lists_to_array(df, column_name):
    return np.vstack(df[column_name].values)


def dfs_column_lists_to_array(dfs, column_name):
    return np.vstack([df_column_lists_to_array(df, column_name) for df in dfs])


def check_df_indices_close(dfs, epsilon=.0001):
    # Check indices from dfs close enough
    check_vectors_close([df.index for df in dfs], epsilon=epsilon,
                        error_message=f"indices from dfs are not within {epsilon}")


def join_dfs_float_index(dfs, epsilon=.0001):
    # epsilon: tolerate at most this difference between the values of one index and a second index
    # Check one name for index across dfs
    df_index_name = check_return_single_element([df.index.name for df in dfs]).single_element
    # Check indices from dfs close enough and extract one if so
    check_df_indices_close(dfs, epsilon)
    df_index = dfs[0].index
    # First merge dfs normally; if join on index works, length should be unchanged
    merged_dfs = pd.concat(dfs)
    if len(merged_dfs) == len(dfs[0]):
        return merged_dfs
    # If above did not join on axis, remove index, join, then add index back
    else:
        merged_dfs = pd.concat([df.reset_index().drop(columns=df_index_name) for df in dfs], axis=1)
        merged_dfs[df_index_name] = df_index  # put index column back
        return merged_dfs.set_index(df_index_name)


def digitize_df(df, digitize_column_names, bins):
    return pd.DataFrame.from_dict({**{column_name: np.digitize(df[column_name], bins=bins)
             for column_name in digitize_column_names}, **{df.index.name: df.index}}).set_index(df.index.name)


def group_dfs_by_column_value(df, group_by_column):
    return {column_value: df.loc[df[group_by_column] == column_value,
                                 df.columns != group_by_column].to_numpy()
            for column_value in set(df[group_by_column])}


def zscore_df_columns(df):
    """
    Z score each column of df using sklearn's preprocessing.scale function and return
    as a df
    :param df: pandas dataframe
    :return: pandas dataframe with z scored columns
    """
    return pd.DataFrame(sklearn.preprocessing.scale(df.to_numpy()),
                        columns=df.columns,
                        index=df.index)


def copy_df_columns(copy_from_df, copy_to_df, column_names, allow_overwrite=False):
    # Check inputs
    if len(copy_from_df) != len(copy_to_df):
        raise Exception(f"copy_from_df and copy_to_df must be same length")
    if not allow_overwrite and any([k in copy_to_df for k in column_names]):
        raise Exception(f"Some of column_names already exist in copy_to_df, which isnt allowed when allow_overwrite is False")
    for k in column_names:
        copy_to_df[k] = copy_from_df[k]
    return copy_to_df


def get_empty_df(column_names):
    return pd.DataFrame.from_dict({k: [] for k in column_names})


def unique_df_column_sets(df, column_names, as_dict=False):
    # Return unique settings of column values
    df_tuples = set([tuple(x) for x in df[column_names].to_numpy()])
    # Return each unique setting as dictionary if indicated ([{column_name: value, ...}, {column_name: value, ...}]
    if as_dict:
        return [dict_comprehension(column_names, v) for v in df_tuples]
    return df_tuples


def convert_categorical(df, column_name, categories):
    # Convert df column to categorical datatype
    # Check that all values in specified column are in passed categories
    missing_vals = [x for x in df[column_name] if x not in categories]
    if len(missing_vals) > 0:
        raise Exception(f"The following values were in the df column '{column_name}' but not in passed categories: "
                        f"{categories}: {missing_vals}")
    df[column_name] = pd.Categorical(df[column_name], categories=categories)
    return df


def df_from_data_list(data_list, entry_names, empty_df_has_cols=False):
    # If nothing in data_list and want to return empty df with columns, return empty df with columns
    if empty_df_has_cols and len(data_list) == 0:
        return pd.DataFrame([], columns=entry_names)

    return pd.DataFrame.from_dict({k: v for k, v in zip(entry_names, zip(*data_list))})


def add_column_to_df(df, column, column_name):
    df[column_name] = column
    return df


def check_single_df(dfs, tolerate_error=False):
    passed_check = all([df.equals(dfs[0]) for df in dfs[1:]])
    if not tolerate_error and not passed_check:
        raise Exception(f"list of dfs must contain identical dfs")
    return passed_check


def unpack_single_df(dfs):
    check_single_df(dfs)
    return dfs[0]


def check_same_index(pandas_objs):
    check_vectors_equal([x.index for x in pandas_objs])


def dfs_same_values(dfs, tolerate_error=False):
    df1 = dfs[0]
    passed_check = True
    for df2 in dfs[1:]:
        passed_check *= check_set_equality(df1.columns, df2.columns, "df1 columns", "df2 columns", tolerate_error)
        passed_check *= all(
            [(np.vstack(df1[x]) == np.vstack(df2[x])).all() for x in df1.columns])


def df_filter1_columns_symmetric(df, column_set_1, column_set_2, tolerate_no_entry=False):

    # Check inputs
    if not all([isinstance(x, dict) for x in [column_set_1, column_set_2]]):
        raise Exception(f"column_sets must be dictionaries but this was not the case for at least one of "
                        f"column_set_1 and column_set_2")

    def _get_df_key(d1, d2):
        # Add "1" suffix to each key in c1 and "2" suffix to each key in c2,
        # and put all resulting keys into a single dictionary
        return merge_dicts([{f"{k}_{idx + 1}": v for k, v in d.items()} for idx, d in enumerate([d1, d2])])

    # Find df entries that match no matter which way the column sets are arranged (with the first set having 1's as
    # suffix and the second set having 2's as suffix, or vice versa)
    # If column sets are identical, we have just one unique ordering of column sets. If they are different, we have two
    # ...Case where column sets the same (default)
    unique_column_set_orders = [[column_set_1, column_set_2]]
    # ...Case where column sets not the same
    if not check_dict_equality([column_set_1, column_set_2], tolerate_error=True, issue_warning=False):
        unique_column_set_orders += [[column_set_2, column_set_1]]
    # ...Get df entries for each unique ordering of column sets
    df_entries = [df_filter_columns(df, _get_df_key(d1, d2)) for d1, d2 in unique_column_set_orders]

    # Check that no more than one df entry found. Return single entry. Tolerate none found if indicated.
    nonempty_df_entries = [x for x in df_entries if len(x) > 0]  # restrict list to non-empty df subsets
    # If no df entries found, return empty df or raise error as indicated
    if len(nonempty_df_entries) == 0:
        # Return empty df if tolerating no entry
        if tolerate_no_entry:
            return df_entries[0]  # all members of df_entries are empty dfs, so just return the first
        # Otherwise raise exception
        raise Exception(f"No df entries found")
    # If one df entry found, return entry
    elif len(nonempty_df_entries) == 1:
        return unpack_single_element(nonempty_df_entries)
    # If more than one df entry found, raise error
    else:
        raise Exception(f"Found more than one df entry")


def restore_df_columns(new_df, original_df, restore_column_names, shared_column_name):
    # Add column names in restore_column_names to new_df by searching in original_df
    # for entries with same value at shared_column_name. Requires a single value at
    # a restore_column_name for a given value at shared_column_name. Example use case:
    # restore nwb_file_name and epochs_description in a df with nwb_file_name_epochs_description

    # Make a map from unique values at shared_column_name to values at eaech entry in restore_column_names
    column_df = df_from_data_list([(
        shared_column_val, restore_column_name,
        check_return_single_element(df_filter_columns(original_df, {shared_column_name: shared_column_val})[
                                        restore_column_name].values).single_element)
        for shared_column_val in np.unique(new_df[shared_column_name]) for restore_column_name in restore_column_names],
        ["shared_column_val", "restore_column_name", "restore_column_val"])

    # Use map to restore columns
    for restore_column_name in restore_column_names:
        new_df[restore_column_name] = [
            df_pop(column_df, {"shared_column_val": shared_column_val, "restore_column_name": restore_column_name},
                   "restore_column_val")
            for shared_column_val in new_df[shared_column_name]]
    return new_df


def match_two_dfs(x1, x2, max_tolerated_diff):
    """
    "Match" indices across two dataframes (also works with series), and return dfs at these indices
    :param x1: pandas df or series
    :param x2: pandas df or series
    :param max_tolerated_diff: maximum tolerated difference between corresponding indices across the two dfs or series
    :return: df or series at "matching" indices
    """

    # Find indices in x1 such that if elements of x2 were placed before the indices,
    # the order of x1 would be preserved
    match_idxs = np.searchsorted(x1.index, x2.index)

    # Remove repeated indices
    valid_match_idxs_idxs = np.where(
        np.diff(match_idxs) != 0)[0] + 1  # indices of adjacent repeated indices, plus one (keep second in pair)
    valid_match_idxs_idxs = np.asarray([0] + list(valid_match_idxs_idxs))  # add first index
    valid_position_idxs = match_idxs[valid_match_idxs_idxs]

    # Ensure no repeated indices
    check_all_unique(valid_position_idxs)

    # Apply indices found above
    x1 = x1.iloc[valid_position_idxs]
    x2 = x2.iloc[valid_match_idxs_idxs]

    # Check that corresponding indices across vectors not too far off
    if any(abs(x1.index - x2.index) > max_tolerated_diff):
        raise Exception(f"At at least one index, matched x1 and x2 indices differ by more "
                        f"than acceptable tolerance: {max_tolerated_diff}")

    # Return matched dataframes or series
    return x1, x2