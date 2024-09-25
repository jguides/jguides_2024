import copy
import os
from collections import namedtuple
from collections.abc import Iterable

import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd
from datajoint import DataJointError
from matplotlib import pyplot as plt
from networkx import NetworkXError
from spyglass.common import (Session, IntervalList)
from spyglass.utils.dj_helper_fn import fetch_nwb as fetch_nwb_

from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_environments, get_jguidera_nwbf_names
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.df_helpers import zip_df_columns
from src.jguides_2024.utils.dict_helpers import dict_comprehension, add_defaults, remove_repeat_dicts, \
    merge_dicts_lists
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.save_load_helpers import get_file_contents
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.string_helpers import camel_to_snake_case, snake_to_camel_case, strip_string, \
    strip_trailing_s, \
    get_string_prior_to_dunder, remove_leading_dunder, replace_chars
from src.jguides_2024.utils.vector_helpers import (unpack_single_element, find_spans_increasing_list,
                                                   check_all_unique, none_to_string_none)

"""
Note that datajoint tables used in this module should be imported locally to avoid a circular import error,
since functions defined here are meant to be capable of being imported in any module with datajoint tables.
"""


def special_fetch1(table, attribute, key=None, tolerate_no_entry=False):

    # "Loaded" special_fetch1. Allows user to tolerate no entry and return None in this case, or receive
    # informative error message if not one entry

    # Get inputs if not passed
    if key is None:
        key = dict()

    # If tolerating no entry, return single entry or None if no entry,
    if tolerate_no_entry:
        return fetch1_tolerate_no_entry((table & key), attribute)

    # Otherwise, return single entry or give informative error message if no entry
    check_single_table_entry(table, key)
    return (table & key).fetch1(attribute)


def fetch1_tolerate_no_entry(table, attribute=None):

    """
    Fetch a single entry from a datajoint table; return None if no entry found
    :param table: datajoint table (or table subset)
    :param attribute: column for datajoint table to retrieve
    :return: table column
    """

    if len(table) > 0:  # if entry in table
        if attribute is None:
            return table.fetch1()
        return table.fetch1(attribute)
    else:
        return None


def fetch_nwb(table):

    return fetch_nwb_(table, (nd.common.AnalysisNwbfile, 'analysis_file_abs_path'))


def fetch1_dataframe_tolerate_no_entry(table_subset, object_name=None):

    """
    Fetch a single entry from a datajoint table subset; return None if no entry
    :param table_subset: subset of datajoint table
    :param object_name is necessary for fetch1_dataframe method used by tables that have analysis nwb files,
            but not those that dont.
    :return: single table entry or None if no table entries
    """

    if len(table_subset) > 0:  # if entry in table
        if object_name is None:
            return table_subset.fetch1_dataframe()
        return table_subset.fetch1_dataframe(object_name)
    else:
        return None


def fetch1_dataframe(table, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):

    # Get name of object id if not passed. Only possible if the table has a single object id
    if object_id_name is None:
        object_id_name = get_table_object_id_name(table, leave_out_object_id=True)

    # Get nwb data
    entries = table.fetch_nwb()

    # Check that only one entry
    if len(entries) != 1:
        raise Exception(f"Should have found exactly one entry but found {len(entries)}")

    # Take first entry, and desired object
    df = entries[0][object_id_name]

    # Restore altered empty nwb_object if indicated
    if restore_empty_nwb_object:
        df = handle_empty_nwb_object(df, from_empty=False)

    # Set df index if passed
    if df_index_name is not None:
        df = df.set_index(df_index_name)

    return df


def fetch1_dataframe_from_table_entry(table, strip_s=False, index_column=None, column_subset=None):

    # Get table entry as dictionary
    table_entry = table.fetch1()

    # Get subset of columns if indicated
    if column_subset is not None:
        table_entry = {k: table_entry[k] for k in column_subset}

    # Remove trailing s from column names if indicated
    if strip_s:
        table_entry = {strip_trailing_s(k): v for k, v in table_entry.items()}

    # Convert dictionary to dataframe
    df = pd.DataFrame.from_dict(table_entry)

    # Set dataframe index if indicated
    if index_column is not None:
        if strip_s:
            index_column = strip_trailing_s(index_column)
        df = df.set_index(index_column)

    # Return dataframe
    return df


def fetch_dataframe(table_subset, table_column_names=None):

    """
    Return multiple rows in datajoint table within single dataframe
    :param table_subset: datajoint table
    :return: dataframe
    """

    # Define inputs if not passed
    if table_column_names is None:
        table_column_names = get_table_column_names(table_subset)  # all columns of table

    return pd.DataFrame({k: v for k, v in zip(
        table_column_names, list(zip(*fetch_iterable_array(table_subset, table_column_names))))})


def fetch1_dataframes(table):

    return namedtuple("dfs", " ".join(table.get_object_id_name(leave_out_object_id=True)))(
        *[table.fetch1_dataframe(object_id_name=object_id_name)
          for object_id_name in table.get_object_id_name(leave_out_object_id=True)])


def check_no_attribute_duplicates(table_subset,
                                  attribute):

    """
    Check that no attribute repeated in subset of datajoint table
    :param table_subset: subset of datajoint table
    :param attribute: string, attribute
    """

    attribute_vals = table_subset.fetch(attribute)
    if len(attribute_vals) != len(np.unique(attribute_vals)):
        raise Exception(f"{attribute} has repeat entries in table subset")


def get_pos_valid_times_interval_names(nwb_file_name):

    pos_valid_times_interval_list_names = []
    for n in (IntervalList & {"nwb_file_name": nwb_file_name}).fetch("interval_list_name"):
        if len(np.asarray(n.split(" "))) == 4:
            if all(np.asarray(n.split(" "))[[0, 2, 3]] == np.asarray(["pos", "valid", "times"])):
                pos_valid_times_interval_list_names.append(n)

    return pos_valid_times_interval_list_names


def check_nwb_file_name(nwb_file_name):

    if nwb_file_name not in get_jguidera_nwbf_names(highest_priority=False, high_priority=False):
        raise Exception(f"nwb file name {nwb_file_name} not recognized")


def fetch_entries_as_dict(table, column_names=None):

    valid_column_names = get_table_column_names(table)
    if column_names is None:
        column_names = valid_column_names
    check_membership(column_names, valid_column_names, "passed column names", "available column names")

    return [{k: v for k, v in zip(column_names, table_entry_values)} for table_entry_values in
            fetch_iterable_array(table, column_names)]


def check_attributes_in_key(necessary_attributes, key):

    missing_attributes = [attribute not in key.keys() for attribute in
                          necessary_attributes]  # boolean indicating missing attributes in above list

    if any(missing_attributes):  # ensure required attributes in key
        raise Exception(f"Attributes {necessary_attributes[missing_attributes]} missing from passed key")


def get_table_column_names(table):

    return list(table.fetch().dtype.fields.keys())


def get_table_object_id_name(table, leave_out_object_id=False, unpack_single_object_id=True, tolerate_none=False):

    object_id_names = [x for x in get_table_column_names(table) if "object_id" in x]

    if len(object_id_names) == 0 and tolerate_none:
        return None

    if leave_out_object_id:
        object_id_names = [x.replace("_object_id", "") for x in object_id_names]

    if len(object_id_names) == 1 and unpack_single_object_id:
        return unpack_single_element(object_id_names)

    return object_id_names


def get_table_key_names(table, secondary_key_names_subset=None):

    return list(table.primary_key) + get_table_secondary_key_names(table, secondary_key_names_subset)


def get_table_secondary_key_names(table, secondary_key_names_subset=None):

    # Get secondary key names in table
    secondary_key_names = [k for k in get_table_column_names(table) if k not in table.primary_key]

    # Restrict to certain secondary key column names if passed
    # First check that passed secondary key column name subset valid
    if secondary_key_names_subset is not None:
        check_membership(secondary_key_names_subset, secondary_key_names, "passed secondary key names",
                         "available secondary key names")
        return secondary_key_names_subset

    return secondary_key_names


def fetch_iterable_array(table, attributes):

    # Return as matrix
    return np.reshape(np.asarray((table).fetch(*attributes), dtype=object).T, (-1, len(attributes)))


def make_param_name(param_values, separating_character=None, tolerate_non_unique=False, replace_double_quotes=True):

    # Get inputs if not passed
    if separating_character is None:
        separating_character = "^"

    # Convert param values to string
    param_values = [str(x) for x in param_values]

    # If indicated, check that separating character not in any param values, otherwise cannot parse param name easily
    if any([separating_character in x for x in param_values]) and not tolerate_non_unique:
        raise Exception(f"separating character ({separating_character}) found in at least one param values; this is "
                        f"not allowed")

    # Define param name
    param_name = separating_character.join(param_values)

    # Strip double quotes from param names to avoid mysql error when querying a table with string with double quotes
    if replace_double_quotes:
        param_name = param_name.replace('"', ".")

    return param_name


def make_params_string(param_values):

    return make_param_name(param_values, separating_character="_", tolerate_non_unique=True)


def get_params_table_name(table=None, table_name=None):

    check_one_none([table, table_name])

    if table is not None:
        table_name = get_table_name(table)

    return f"{table_name}Params"


def plot_datajoint_table_rate_map(fr_table, key, ax, color, label=None):

    def _column_name(column_keyword, fr_table, key):
        return unpack_single_element([column_name for column_name in (fr_table & key).fetch1_dataframe().columns
                                      if column_keyword in column_name])

    check_attributes_in_key(necessary_attributes=np.asarray(["nwb_file_name", "epoch", "sort_group_id", "unit_id"]),
                            key=key)  # require certain attributes passed
    fr_df = (fr_table & key).fetch1_dataframe()
    bin_centers_name =_column_name("bin_centers", fr_table, key)
    rate_map_name = _column_name("rate_map", fr_table, key)

    # Initialize plot if not passed
    if ax is None:
        fig, ax = plt.subplots()

    # Plot
    ax.plot(np.asarray(fr_df.loc[key["unit_id"]][bin_centers_name]),
            np.asarray(fr_df.loc[key["unit_id"]][rate_map_name]), '.-', color=color, label=label)
    ax.set_title(f"{key['nwb_file_name']}, epoch {key['epoch']}, "
                 f"\n sort group {key['sort_group_id']}, unit {key['unit_id']}")


def handle_empty_nwb_object(nwb_object, from_empty):

    # Cannot save empty df out to analysis nwb file. Handle this case below.
    # Since unclear whether this applies to other datatype, raise error if case not yet accounted for

    # Case 1: dataframe
    # Empty df to one that can be saved: add a row with nans, and add a column to signal empty dataframe
    # Saved df restored to empty df: remove row with nans, remove column that had signaled empty dataframe.
    if isinstance(nwb_object, pd.DataFrame):
        empty_df_tag = "EMPTY_DF"  # add as column to empty df to be able to identify altered previously empty dfs
        # FROM EMPTY DF
        if from_empty and len(nwb_object) == 0:  # convert empty df
            for column in nwb_object.columns:
                nwb_object[column] = [np.nan]  # add row with nan
            nwb_object[empty_df_tag] = [np.nan]  # add flag so can know that empty df
        # TO EMPTY DF
        else:  # restore empty df
            # Identify altered empty df as one that has empty_df_tag as column, and has a single row with all nans
            if empty_df_tag in nwb_object and len(nwb_object) == 1 and all(np.isnan(nwb_object.iloc[0].values)):
                nwb_object.drop(labels=["EMPTY_DF"], axis=1, inplace=True)
                nwb_object.drop([0], inplace=True)

    # Case 2: dictionary
    elif isinstance(nwb_object, (dict, list)):
        return nwb_object

    # All other cases
    else:
        raise Exception(f"Need to write code to handle empty nwb object of type {type(nwb_object)}")

    return nwb_object


def create_analysis_nwbf(key, nwb_objects, nwb_object_names):

    # Make copy of key to avoid altering key outside function
    key = copy.deepcopy(key)

    # Get nwb file name with which to create file. If nwb_file_name in key, use this. Otherwise,
    # use recording_set_name
    if "nwb_file_name" in key:
        nwb_file_name = key["nwb_file_name"]
    elif "recording_set_name" in key:
        # Take first nwb file name
        from src.jguides_2024.metadata.jguidera_epoch import RecordingSet  # local import to avoid circular import error
        nwb_file_name = (RecordingSet & key).fetch1("nwb_file_names")[0]
    elif "train_test_epoch_set_name" in key:
        # Take first nwb file name
        from src.jguides_2024.metadata.jguidera_epoch import TrainTestEpochSet
        nwb_file_name = (TrainTestEpochSet & key).fetch1("nwb_file_names")[0]
    else:
        raise Exception(f"nwb_file_name, recording_set_name, or train_test_epoch_set_name must be in key to"
                        f" define nwb file name for making analysis nwb file")

    # Create analysis nwb file
    key['analysis_file_name'] = nd.common.AnalysisNwbfile().create(nwb_file_name)
    nwb_analysis_file = nd.common.AnalysisNwbfile()

    # Check that objects all dfs (code currently assumes this in defining table_name)
    if not all([isinstance(x, pd.DataFrame) for x in nwb_objects]):
        raise Exception(f"create_analysis_nwbf currently assumes all objects dfs")

    for nwb_object_name, nwb_object in zip(nwb_object_names, nwb_objects):
        key[nwb_object_name] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=nwb_object,
            table_name=f"pandas_table_{nwb_object_name}")
    nwb_analysis_file.add(nwb_file_name=nwb_file_name, analysis_file_name=key['analysis_file_name'])

    return key


def insert_analysis_table_entry(table, nwb_objects, key, nwb_object_names=None, convert_empty_nwb_object=True,
                                reset_index=False, replace_none_col_names=None):

    # Reset index in any dfs in nwb_objects if indicated (useful because currently index does not get stored
    # in analysis nwb file). Default is not True because if reset index when there is none, adds column called "index"
    if reset_index:
        nwb_objects = [x.reset_index() if isinstance(x, pd.DataFrame) else x for x in nwb_objects]

    # Convert None to "none" in specified df cols since None cannot be stored in analysis nwb file currently
    if replace_none_col_names is not None:
        for nwb_object in nwb_objects:
            if isinstance(nwb_object, pd.DataFrame):
                for col_name in replace_none_col_names:
                    nwb_object[col_name] = none_to_string_none(nwb_object[col_name])

    # Get nwb object names if not passed
    if nwb_object_names is None:
        nwb_object_names = table.get_object_id_name(unpack_single_object_id=False)

    # Convert nwb objects that are empty pandas dfs to something that can be saved out if indicated
    if convert_empty_nwb_object:
        nwb_objects = [handle_empty_nwb_object(x, from_empty=True) for x in nwb_objects]

    # Insert into table
    key = create_analysis_nwbf(key=key, nwb_objects=nwb_objects, nwb_object_names=nwb_object_names)
    table.insert1(key, skip_duplicates=True)  # insert into table
    print(f'Populated {table.table_name} for {key}')


def check_nwb_file_inserted(nwb_file_name):

    if len((Session & {'nwb_file_name': nwb_file_name})) == 0:
        raise Exception("nwb file not in Session table")


def check_nwb_files_inserted(nwb_file_names):

    for nwb_file_name in nwb_file_names:
        check_nwb_file_inserted(nwb_file_name)


def trial_duration_from_params_table(trials_table, column_name, param_name):

    trial_start_time_shift, trial_end_time_shift = (
            trials_table & {column_name: param_name}).fetch1("trial_start_time_shift", "trial_end_time_shift")

    return int(trial_end_time_shift - trial_start_time_shift)


def format_path_name(path_name):

    return path_name.replace("_to_", "-").replace("_well", "")


def abbreviate_path_name(path_name):

    return "-".join([x[0].upper() for x in format_path_name(path_name).split("-")])


def abbreviate_path_names(path_names):

    return [abbreviate_path_name(x) for x in path_names]


def abbreviate_environment(environment):

    # Convert environment name to upper case initials
    # Check inputs
    valid_environments = get_environments()
    if environment not in valid_environments:
        raise Exception(f"environment must be in {valid_environments} but is {environment}")
    return "".join([x[0].upper() for x in camel_to_snake_case(environment).split("_")])


def delete_(main_table, dependent_tables, key, safemode=True):

    print(f"deleting entries from {get_table_name(main_table)} and listed dependent tables for key: {key}")
    for table in dependent_tables:
        if hasattr(table, "delete_"):
            table().delete_(key, safemode=safemode)
        else:
            delete_flexible_key(table, key, safemode)

    delete_flexible_key(main_table, key, safemode)


def delete_flexible_key(table, key=None, safemode=True):

    if key is None:
        table.delete(safemode=safemode)
    else:
        # Restrict key to primary key entries in table to avoid mysql error
        (table & {k: key[k] for k in table.primary_key if k in key}).delete(safemode=safemode)


def drop_(table_instances):

    # Drop a series of tables
    for table_instance in table_instances:
        # TODO (feature): allow recursive use of this function. Requires breaking out from function once all downstream
        #  tables dropped. Can use below code in part.
        # if hasattr(table_instance, "drop_"):
        #     table_instance.drop_()
        try:
            table_instance.drop()
        except NetworkXError as e:
            print(e)


def delete_multiple_flexible_key(tables, key):

    # Require key because dont want to accidentally delete all entries from a table

    for table in tables:
        delete_flexible_key(table, key)


def populate_flexible_key(table, key=None, tolerate_error=False, error_message=None, verbose=False):

    # Get inputs if not passed
    if error_message is None:
        # Use table name function instead of attribute because function returns name in camel case (more readable)
        error_message = f"Populating {get_table_name(table)} failed"
        if key is not None:
            error_message += f" for key {key}"

    if verbose:
        print(f"populating {get_table_name(table)}...")

    # Populate whole table if key not passed
    if key is None:
        # Tolerate error
        if tolerate_error:
            try:
                table.populate()
            except:
                print(error_message)
        # Do not tolerate error
        else:
            table.populate()

    # Otherwise populate subset of table with key
    else:
        # Tolerate error
        if tolerate_error:
            try:
                table.populate(key)
            except:
                print(error_message)
        # Do not tolerate error
        else:
            table.populate(key)


def populate_multiple_flexible_key(tables, key=None, tolerate_error=False, error_message=None):

    for table in tables:
        populate_flexible_key(table, key, tolerate_error, error_message)


def get_unit_name(sort_group, unit_id):

    return f"{sort_group}_{unit_id}"


def split_unit_name(formated_sort_group_unit_id):

    return list(map(int, formated_sort_group_unit_id.split("_")))


def split_unit_names(formated_sort_group_unit_ids):

    return list(zip(*[split_unit_name(x) for x in formated_sort_group_unit_ids]))


def format_nwb_file_name(nwb_file_name):

    return nwb_file_name.split("_")[0]


def get_next_int_id(table):

    # If no integer id entries in table yet, return zero
    int_ids = table.fetch("int_id")
    if all([x is None for x in int_ids]):  # returns True both when no int_ids and when all int_ids are None
        return 0
    # Otherwise, define next int_id as one more than last number in stretch of consecutive integers from zero
    int_ids = list(map(int, [x for x in int_ids if x is not None]))  # convert to int in case string datatype
    last_int = find_spans_increasing_list(np.sort(int_ids), max_diff=1)[0][0][-1]

    return last_int + 1


def check_int_id(table, int_id):

    # Check that all current int ids in table valid: no duplicates
    check_all_unique(table.fetch("int_id"))

    # Check that passed int_id valid: should be one more than greatest integer after stretch of consecutive
    # integers from zero
    expected_int_id = get_next_int_id(table)
    if int_id != expected_int_id:
        raise Exception(f"int_id must be {expected_int_id} but is {int_id}")


# For inverse operation to below, use EpochCohort table
def get_epochs_id(epochs):

    if not isinstance(epochs, Iterable):
        raise Exception(f"epochs must be iterable but is type {type(epochs)}")
    if len(epochs) == 0:
        raise Exception(f"epochs must not be empty")

    return make_param_name(np.sort(epochs), separating_character="_")


def check_epochs_id(epochs_id, epochs):

    expected_epochs_id = get_epochs_id(epochs)
    if expected_epochs_id != epochs_id:
        raise Exception(f"Passed epochs correspond to epochs_id of {expected_epochs_id}, but {epochs_id} was passed")


def intersect_tables(tables, key_filter=None):

    # Returns intersection of table entries using primary keys
    # Approach: intersect each table after the first with the first

    # Restrict intersection using key_filter if passed
    table_intersection = tables[0]
    for x in tables[1:]:
        table_intersection = table_intersection.proj() * x.proj()
    if key_filter is not None:
        return table_intersection & key_filter

    return table_intersection


def convert_long_table_name(long_table_name):

    x = strip_string(long_table_name.split(".")[-1], "`", strip_start=True, strip_end=True)
    if is_part_table_name(long_table_name):
        x = strip_string(x.replace("__", "."), ".", strip_start=True)

    return snake_to_camel_case(x)


def get_upstream_table_names(table):

    return [convert_long_table_name(table_name) for table_name in table.parents()]


def get_downstream_table_names(table):

    return [convert_long_table_name(table_name) for table_name in table.children()]


def get_table_name(table):

    return snake_to_camel_case(table.table_name)


def is_part_table_name(long_table_name):

    x = strip_string("".join(long_table_name.split(".")[1:]), "`", strip_start=True, strip_end=True)
    if x[:2] == "__":  # remove first dunder, which I believe indicates imported table
        x = x[2:]

    return len(x.split("__")) == 2  # part table has an additional dunder


def get_part_table_names(long_table_names):

    return [x for x in long_table_names if is_part_table_name(x)]


def get_parent_table_name_from_part_table_name(part_table_name):

    return "".join(part_table_name.split("__")[:-1]) + "`"


def get_parent_table_names(table_names):

    return [get_parent_table_name_from_part_table_name(x) for x in get_part_table_names(table_names)]


def insert1_print(table, key):

    table.insert1(key, skip_duplicates=True)
    print(f"Added entry to {get_table_name(table)} for {key}")


def insert_manual_table_test_entry(table):

    # Insert "test" entry into manual table
    # Get keys that could be inserted
    potential_keys = table._get_potential_keys()
    # Exit if no potential keys with which to insert test entry
    if len(potential_keys) == 0:
        print(f"No potential entries from which to insert test entry for table "
              f"{get_table_name(table)}; exiting")
        return
    # Otherwise, insert first of the potential keys
    test_entry = potential_keys[0]
    # (note we dont want to use insert1_print here b/c that prints out message that we inserted even for duplicates)
    table.insert1(test_entry, skip_duplicates=True)


def convert_array_none(x):

    if len(x) == 1:
        if unpack_single_element(x) == "none":
            return None
    return x


def table_name_from_table_type(table_name, table_type):

    # Check table name: require non-empty and table type string (table_type) appears at most once
    split_table_name = table_name.split(table_type)
    if len(split_table_name) not in [1, 2]:
        raise Exception(f"table name empty or has {table_type} appearing more than once. table name: {table_name}")
    return split_table_name[0]


def table_name_from_selection_table_name(table_name):

    return table_name_from_table_type(table_name, "Sel")


def table_name_from_param_name_table_name(table_name):

    return table_name_from_table_type(table_name, "ParamName")


def table_name_from_params_table_name(table_name):

    return table_name_from_table_type(table_name, "Params")


def get_main_table_name(table_name):

    for fn in [table_name_from_selection_table_name, table_name_from_param_name_table_name,
               table_name_from_params_table_name]:
        table_name = fn(table_name)

    return table_name


def selection_table_name_from_table_name(table_name):

    # Get selection table name from main table name or from selection table name
    # If selection table passed, converts to main table first
    return f"{table_name_from_selection_table_name(table_name)}Sel"


def get_meta_param_name(table, verbose=False):

    # Get what param name is called in table
    # Three acceptable cases, in order of preference:
    # Case 1: param name is main table name in snake case with "param_name" added (preferred)
    # Case 2: param name is sole primary key with "param_name" (non-preferred)
    # Case 3: param name is sole primary key (non-preferred)

    # Only possible if table has already been initialized
    try:

        non_preferred_param_name = False  # default

        # Case 1
        # Get name of table
        table_name = get_table_name(table)
        # Convert name of selection or param name table to name of corresponding main table
        main_table_name = get_main_table_name(table_name)
        # Expected param name is name of table (in snake case) with param_name at end
        param_name = f"{camel_to_snake_case(main_table_name, group_uppercase=True)}_param_name"
        # Check that expected param name found in table primary key
        passed_check = check_membership([param_name], table.primary_key, "list with expected param name",
                                        f"{table_name} primary key", tolerate_error=True)

        # Case 2
        if not passed_check:
            param_name = unpack_single_element([x for x in table.primary_key if "param_name" in x], tolerate_no_entry=True)
            non_preferred_param_name = True

        # Case 3
        if param_name is None:
            param_name = unpack_single_element(table.primary_key)
            non_preferred_param_name = True

        # Signal non-preferred param name if indicated
        if verbose and non_preferred_param_name:
            print(f"warning! non preferred param name for {get_table_name(table)}: {param_name}")

        # Return table param name
        return param_name

    except DataJointError:
        print(f"DataJointError when getting meta param name; this occurs when table not yet defined in schema")


def package_secondary_key(table, secondary_key_values, secondary_key_names_subset=None):
    secondary_key_names = get_table_secondary_key_names(table, secondary_key_names_subset)
    SecondaryKey = namedtuple("SecondaryKey", secondary_key_names)
    secondary_key = SecondaryKey(*secondary_key_values)
    return secondary_key._asdict()


def get_entry_secondary_key(table_subset):

    secondary_key_values = [table_subset.fetch1(k) for k in get_table_secondary_key_names(table_subset)]

    return package_secondary_key(table_subset, secondary_key_values)


def get_entry_primary_key(table_subset):

    return {k: v for k, v in table_subset.fetch1() if k in table_subset.primary_key}


def get_non_param_name_primary_key_names(table):

    return [x for x in table.primary_key if x != get_meta_param_name(table)]


def check_single_table_entry(table, key=None):

    if key is None:
        key = dict()
    num_table_entries = len(table & key)
    if num_table_entries != 1:
        raise Exception(f"Exactly one entry should have been found in {get_table_name(table)} for key, but "
                        f"{num_table_entries} were. Key: {key}")


def unique_table_column_sets(table, column_names=None, key_filter=None, as_dict=False):

    # Return unique settings of a desired set of columns

    # Get column names if not passed
    if column_names is None:
        column_names = table.primary_key

    # Restrict table if indicated
    if key_filter is not None:
        table = (table & key_filter)
    # Get values for passed columns in each row of table
    column_values = list(zip(*[table.fetch(x) for x in column_names]))
    # Return unique settings of these (as dictionary if indicated, otherwise as list)
    unique_column_sets = np.asarray(list(set(column_values)), dtype=object)
    if as_dict:
        return [{k: v for k, v in zip(column_names, unique_column_set)} for unique_column_set in unique_column_sets]
    return unique_column_sets


def get_cohort_test_entry(table, col_vary, num_entries, verbose=True):

    # Column_vary: name of column whose value can vary across rows

    # Check that column set to vary is a primary key of table
    if col_vary not in table.primary_key:
        raise Exception(f"Column set to vary is not a primary key of table")
    # Define columns requiring same value across rows as all those except the column whose value can vary across rows
    col_same = [x for x in table.primary_key if x != col_vary]
    col_vary_vals = table.fetch(col_vary)
    # Get unique settings of "same value" columns
    unique_col_same_vals = unique_table_column_sets(table, col_same)
    # Loop over these and find all row idxs where "same value" columns have the given values
    # If there are at least as many of these as we want entries, stop and take the specified number of entries
    col_same_vals = list(zip(*[table.fetch(x) for x in col_same]))
    target_vals = None  # initialize
    for x in unique_col_same_vals:
        valid_idxs = np.where(np.prod(col_same_vals == x, axis=1))[0]
        # Take first few to fulfill desired number of entries
        target_idxs = valid_idxs[:num_entries]
        if len(valid_idxs) > num_entries - 1:
            target_vals = col_vary_vals[target_idxs]
            break

    # Signal no test entry if indicated
    if verbose and target_vals is None:
        print(f"Could not find cohort table test entry based on {get_table_name(table)} "
              f"because could not find {num_entries} entries "
              f"varying along {col_vary} with all other values the same")

    # If no test entry, return None
    if target_vals is None:
        return None

    # Otherwise return target vals and same col vals for test entry
    TestEntry = namedtuple("TestEntry", "target_vals same_col_vals_map")
    return TestEntry(target_vals, dict_comprehension(col_same, x))


# It may be the case that inserting duplicate entries into a manual table is slow); functions below helpful in this case
def narrow_candidate_keys(table, candidate_keys):

    # Narrow to keys not yet in table
    return np.asarray(candidate_keys)[valid_candidate_keys_bool(table, candidate_keys)]


def valid_candidate_keys_bool(table, candidate_keys):

    # Return indices of keys not yet in table
    return [idx for idx, candidate_key in enumerate(candidate_keys) if
            len(table & {k: candidate_key[k] for k in table.primary_key}) == 0]


def get_schema_table_names(schema=None, schema_name=None):

    # Get tables in a schema, not including parts tables
    # Get from datajoint memory. Here, tables listed in alphabetical order. Useful if some tables in
    # schema missing from file

    check_one_none([schema, schema_name])
    if schema is None:
        schema = dj.schema(schema_name)

    return np.unique([snake_to_camel_case(get_string_prior_to_dunder(remove_leading_dunder(x)))
                      for x in schema.list_tables()])


def get_schema_table_names_from_file(schema_name, schema_path=None):
    # Get tables in a schema, not including parts tables
    # Get from file. Here, tables listed in order of file. Useful if want this ordering

    # Get jguidera schema path if not passed
    if schema_path is None:
        schema_path = get_jguidera_schema_dir(schema_name)

    # Get contents of file with schema
    file_contents = np.asarray(get_file_contents(
        f"{schema_name.replace('.', '/')}.py", schema_path).split("\n"))  # split lines

    # Identify lines with table definitions as those below lines with the schema decorator. Exclude comments
    table_idxs = [idx + 1 for idx, x in enumerate(file_contents) if "@schema" in x and x[0] != "#"]

    # Extract table names from these lines
    table_lines = file_contents[table_idxs]
    return [x.replace("class ", "").split("(")[0] for x in table_lines]


def get_project_dir():
    return "/home/jguidera/Src/jguides_2024/"


def get_module_dir():
    return "src/jguides_2024"


def get_module_path():
    return os.path.join(get_project_dir(), get_module_dir())


def get_module_subdirs():
    module_path = get_module_path()
    return [x for x in os.listdir(module_path) if not x.startswith("_")
                  and os.path.isdir(os.path.join(module_path, x))]


def get_schema_names():

    # Get schema names
    # IMPORTANT NOTE: only set up to get schema names with "one level of nesting", i.e.
    # in a subfolder of the module path

    module_path = get_module_path()
    module_dir = get_module_dir()
    subdirs = get_module_subdirs()
    target_start_string = "jguidera"

    schema_names = [
        f"{module_dir.replace('/', '.')}.{child_dir}.{x}".replace("..", ".") for child_dir in subdirs
        for x in os.listdir(f"{module_path}/{child_dir}")
        if x[:len(target_start_string)] == target_start_string]

    return [x.replace(".py", "") for x in schema_names]


def get_jguidera_schema_dir(schema_name):
    # Get directory in which a jguidera schema lives
    # IMPORTANT NOTE: only set up to get schema names with "one level of nesting", i.e.
    # in a subfolder of the module path

    module_path = get_module_path()
    subdirs = [
        x for x in os.listdir(module_path) if not x.startswith("_") and os.path.isdir(os.path.join(module_path, x))]

    return unpack_single_element(
        [os.path.join(module_path, child_dir) for child_dir in subdirs for x in os.listdir(
            f"{module_path}/{child_dir}") if x == f"{schema_name}.py"])


def get_import_statements(schema_names=None):

    # Use all possible schema names if none passed
    if schema_names is None:
        schema_names = get_schema_names()

    # Loop through schema names and make statements like "from schema_name import table_1, table_2, ..."
    import_text = []
    for schema_name in schema_names:
        table_names_str = ", ".join(get_schema_table_names_from_file(schema_name, schema_path=get_project_dir()))
        import_text.append(f"from {schema_name} import {table_names_str}")

    return import_text


def get_default_param(param_name):
    return get_param_defaults_map()[param_name]


def get_param_defaults_map():

    # Local import to avoid circular import error
    os.chdir("/home/jguidera/Src/jguides_2024/")
    from src.jguides_2024.position_and_maze.jguidera_ppt import PptParams
    from src.jguides_2024.spikes.jguidera_unit import EpsUnitsParams
    from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
    default_curation_id = 3

    return {"position_info_param_name": "default",
          "linearization_param_name": "default",
          "sorter": "mountainsort4",
          "curation_id": default_curation_id,
          "eps_units_param_name": EpsUnitsParams().lookup_param_name([.1]),
            "brain_region_cohort_name": "all_targeted",
            "curation_set_name": "runs_analysis_v1",
          "ppt_param_name": PptParams().get_default_param_name(),
            # "unit_subset_type": "all",  # TODO: remove if can; ideally use defaults stored with fns.
            "zscore_fr": 0,
            "time_rel_wa_fr_vec_param_name": "none",
            "path_fr_vec_param_name": "none",
            "res_epoch_spikes_sm_param_name": ResEpochSpikesSmParams().lookup_param_name([.1]),
            "team_name": "JG_DG"}


def add_param_defaults(key, target_keys=None, add_nonexistent_keys=False, replace_none=True):
    # Add default params to datajoint table

    param_defaults = get_param_defaults_map()
    # Restrict to target keys if passed
    if target_keys is not None:
        param_defaults = {k: v for k, v in param_defaults.items() if k in target_keys}

    return add_defaults(key, param_defaults, add_nonexistent_keys, replace_none)


def get_valid_position_info_param_names():
    return ["default", "default_decoding"]


def get_curation_name(sort_interval_name, curation_id):
    return make_param_name([sort_interval_name, curation_id], "_")


def split_curation_name(curation_name):

    split_curation_name = curation_name.split("_")
    # Check that only two components separated by underscore (first will be taken as sort_interval_name,
    # second as curation_id)
    if len(split_curation_name) > 2:
        raise Exception(f"curation_name should have a single underscore")

    return split_curation_name[0], int(split_curation_name[1])


def populate_insert(table, **kwargs):

    populated_tables = []
    if hasattr(table, "populate_"):
        populated_tables = table().populate_(**kwargs)
    if hasattr(table, "insert_defaults"):
        # Pass as key_filter: 1) key_filter if in kwargs, otherwise 2) key if in kwargs, otherwise 3) None
        key_filter = None  # default
        if "key_filter" in kwargs:
            key_filter = kwargs["key_filter"]
        elif "key" in kwargs:
            key_filter = kwargs["key"]
        table().insert_defaults(key_filter=copy.deepcopy(key_filter))  # copy to avoid changing outside function

    return populated_tables


def convert_path_names(path_names):
    return [convert_path_name(path_name) for path_name in path_names]


def convert_path_name(path_name):

    """
    Convert path name in tuple form, e.g. ("handle_well", "center_well"), to string form, e.g.
    "handle_well_to_center_well" Necessary to store edge names in string format when saving to nwb files, which use
    hd5f, which cannot handle nested data structures within dataframes.
    :param path_name: string with name of path or tuple with (start well name, end well name)
    :return: path name in string form (if tuple form passed), or in tuple form (if string form passed)
    """

    if isinstance(path_name, tuple):
        return "_to_".join(path_name)
    elif isinstance(path_name, str):
        return tuple(path_name.split("_to_"))


def make_table_param_names(table, secondary_key_splits):

    # Check all items in secondary key split are in table secondary key
    secondary_key = get_table_secondary_key_names(table)
    check_membership(np.ndarray.flatten(np.asarray(secondary_key_splits)), secondary_key, "secondary key split members",
                     f"{get_table_name(table)} secondary key")

    # Return param name for each split
    return [table._make_insert_param_name({k: secondary_key[k] for k in x}) for x in secondary_key_splits]


def get_num_param_name(param_name):
    return f"num_{param_name}"


def get_param_name_separating_character(table_name=None):

    if table_name is None:
        return None

    separating_characters = {"TrialsPoolSel": "$",
                             "TrialsPoolCohortParams": "@",
                             "ResTimeBinsPoolSel": "#",
                             "ResTimeBinsPoolCohortParams": "*",  # pools from ResTimeBinsPool
                             "TrainTestSplitPoolSel": "+",  # depends on ResTimeBinsPoolCohort
                             "XInterpPoolSel": "*",  # depends on ResTimeBinsPool, pulls from tables like PptInterp (which also depends on ResTimeBinsPool)
                             "XInterpPoolCohortParams": "|",
    }

    return separating_characters[table_name]


def get_virtual_module(schema_name):
    # Can use the following to mimic table import. Tables are attributes of virtual module. Can delete from these.

    return dj.create_virtual_module("virtual_module", schema_name)


def get_key(table_subset, key):

    if key is None:
        if len(table_subset) != 1:
            raise Exception(f"Trying to get cohort params for members of a cohort in ResTimeBinsPoolCohortParams. "
                            f"No key passed so expected single table entry, but have {len(table_subset)}")
        key = table_subset.fetch1("KEY")

    return key


def get_key_filter(k):
    # Get key_filter in cases where: key_filter is None, key_filter in kwargs (as dictionary, or as None)
    # Note that case where k is dictionary that is filter_key is NOT covered here; so this function should not be used
    # in this case

    if not isinstance(k, dict) and k is not None:
        raise Exception(f"input should be dictionary with key filter or None")
    if isinstance(k, dict):
        if "key_filter" in k:
            if k["key_filter"] is None:
                return dict()
            return copy.deepcopy(k["key_filter"])
        return dict()
    if k is None:
        return dict()


def add_upstream_res_set_params(kwargs):
    # When populating a table using a key that applies to downstream tables, it is ideal to infer
    # restrictions to the current table from the key where possible, so we dont populate the current
    # table with irrelevant entries. This is the case with the following cascading keys, from upstream
    # to downstream: "res_set_param_name", "res_time_bins_pool_param_name", and
    # "res_time_bins_pool_cohort_param_name". This function infers upstream params from downstream params
    # in this case.
    # For now, we implement this only for cohorts whose members all have the same res_time_bins_pool_param_
    # name. In future, could expand to cover cases where cohort has multiple of these. This would require
    # making multiple keys from passed key, and looping through.

    # Local imports to avoid circular import error
    from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams, ResTimeBinsPoolSel

    key = kwargs.pop("key", None)

    if key is not None:
        # Infer res_time_bins_pool_param_name (helpful for inferring res_set_param_name)
        if "res_time_bins_pool_cohort_param_name" in key and "res_time_bins_pool_param_name" not in key:
            table_subset = (ResTimeBinsPoolCohortParams & key)
            if len(table_subset) == 1:  # one cohort represented by key
                res_time_bins_pool_param_names = table_subset.fetch1("res_time_bins_pool_param_names")
                res_time_bins_pool_param_name = check_return_single_element(
                    res_time_bins_pool_param_names).single_element
                if res_time_bins_pool_param_name is not None:
                    key.update({"res_time_bins_pool_param_name": res_time_bins_pool_param_name})

        # Infer res_set_param_name
        if "res_time_bins_pool_param_name" in key and "res_set_param_name" not in key:
            try:
                # requires res_set_param_name used to construct res_time_bins_pool_param_name, and that same
                # across all instances (useful setup for if expand this code to cases where more than one
                # res_time_bins_pool_param_name; currently the above requires single member cohort so could have
                # left out the loop over entries in ResTimeBinsPoolSel)
                res_set_param_name = (ResTimeBinsPoolSel & key).get_res_set_param_name()
                key.update({"res_set_param_name": res_set_param_name})
            except:
                pass

    kwargs["key"] = key  # put key back into kwargs

    return kwargs


def get_relationship_text(x1, x2):

    if x1 == x2:
        return "same"
    return "different"


def get_relationship_texts(x):
    """
    Return list with relationship between pairs in a tuple, like ["same", "different", ...]
    :param x: list of tuples like [(x1, x2), (x3, x4), ...]
    :return: list with "same" and/or "different"
    """

    return np.asarray([get_relationship_text(x1, x2) for x1, x2 in x])


def fetch1_dataframes_across_epochs(table, key, axis=0):
    # Concatenate dfs across epochs

    # Assemble keys, one for each epoch/res_epoch_time_bins_pool_param_name pair
    from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams
    keys = ResTimeBinsPoolCohortParams().get_keys_with_cohort_params(key)
    # Get dfs across epochs
    dfs_objs = [(table & key).fetch1_dataframes() for key in keys]
    # Concatenate dfs across epochs (units in columns, time in rows)
    object_id_names = table.get_object_id_name(leave_out_object_id=True)
    dfs_concat = {object_id_name: pd.concat([getattr(dfs_obj, object_id_name) for dfs_obj in dfs_objs], axis=axis)
                  for object_id_name in object_id_names}
    # Add a df with epochs corresponding to samples. Since returning one epoch df per dfs object, we require
    # that all dfs in the object have the same number of samples
    epoch_df = pd.DataFrame.from_dict({"epoch": np.concatenate([
        [key["epoch"]]*check_return_single_element(
            [len(getattr(dfs_obj, object_id_name)) for object_id_name in object_id_names]).single_element
        for key, dfs_obj in zip(keys, dfs_objs)])})
    dfs_concat.update({"epoch_vector": epoch_df})
    return namedtuple("dfs_concat", dfs_concat)(**dfs_concat)


def replace_param_name_chars(param_name):

    replace_char_map = {"[": "", "]": "", "' '": "_", "'": ""}  # order matters here

    return replace_chars(param_name, replace_char_map)


def _get_idx_column_name(column_name):
    return f"{column_name}_idx"


def preserve_df_row_idx(df, column_name):

    # Check that column doesnt already exist with name we will give to column with index
    idx_column_name = _get_idx_column_name(column_name)
    if idx_column_name in df:
        raise Exception(f"column already exists in df with name {column_name}")
    df[idx_column_name] = [x.index.values for x in df[column_name]]

    return df


def restore_df_row_idx(df, column_name, idx_column_name=None, drop_idx_column=True):

    if idx_column_name is None:
        idx_column_name = _get_idx_column_name(column_name)
    df[column_name] = [pd.Series(x, index=y) for x, y in zip_df_columns(df, [column_name, idx_column_name])]

    if drop_idx_column:
        df.drop(columns=[idx_column_name], inplace=True)

    return df


def scrappy_clean(sel_table, main_table, safemode=True):

    bad_keys = []
    keys = sel_table.fetch("KEY")
    for key in keys:
        if len(main_table & key) == 0:
            bad_keys.append(key)
    print(f"Found {len(bad_keys)} keys of {len(keys)} in {get_table_name(sel_table)} but not main table. Deleting...")
    for key in bad_keys:
        sel_table().delete_(key, safemode)


def get_table_curation_names_for_key(table, key):

    from src.jguides_2024.spikes.jguidera_spikes import EpochSpikeTimesRelabel

    # Get curation names relevant to a key
    if "curation_name" in key:
        return [key["curation_name"]]
    else:
        if "sort_interval_name" in key and "curation_id" in key:
            return [get_curation_name(key["sort_interval_name"], key["curation_id"])]
        if "sort_interval_name" in key:
            return [curation_name for curation_name in set(table.fetch("curation_name")) if
                              split_curation_name(curation_name)[0] == key["sort_interval_name"]]
        if "curation_id" in key:
            return [curation_name for curation_name in set(table.fetch("curation_name")) if
                              split_curation_name(curation_name)[1] == key["curation_id"]]
        else:
            return np.unique((EpochSpikeTimesRelabel & key).fetch("curation_name"))


# TODO (feature): move to class
def get_boot_params(boot_set_name):
    # Return parameters for bootstrapping

    if boot_set_name in [
        "default", "default_rat_cohort", "collapse_maze_segments", "collapse_maze_segments_rat_cohort",
        "brain_region_diff", "brain_region_diff_rat_cohort", "stay_leave_diff", "stay_leave_diff_rat_cohort",
        "stay_leave_diff_brain_region_diff", "stay_leave_diff_brain_region_diff_rat_cohort",
        "relationship_div", "relationship_div_rat_cohort",
        "same_different_outbound_path_correct_diff", "same_different_outbound_path_correct_diff_rat_cohort",
        "same_different_outbound_path_correct_diff_brain_region_diff",
        "same_different_outbound_path_correct_diff_brain_region_diff_rat_cohort",
    ]:
        num_bootstrap_samples = 1000
        average_fn = np.mean
        alphas = (.05, .01, .001, .0001)

    elif boot_set_name in ["relationship_div_median", "relationship_div_rat_cohort_median"]:
        num_bootstrap_samples = 1000
        average_fn = np.median
        alphas = (.05, .01, .001, .0001)

    else:
        raise Exception(f"boot_set_name {boot_set_name} not accounted for in code")

    return namedtuple("boot_params", "num_bootstrap_samples average_fn alphas")(
            num_bootstrap_samples, average_fn, alphas)


def return_table_key(table, key, tolerate_missing_columns=False):

    # Check that all columns in table passed in key, unless indicated otherwise
    table_columns = table.primary_key
    missing_columns = [k not in key for k in table_columns]
    if not tolerate_missing_columns and np.any(missing_columns):
        raise Exception(
            f"The following keys in table were not passed in key: {table_columns[missing_columns]}")

    # Return subset of entries in key that correspond to columns in table
    return {k: key[k] for k in table_columns if k in key}


class UpstreamEntries:

    def __init__(self):
        self.upstream_entries = dict()

    def update_upstream_entries(self, table, key):

        table_name = get_table_name(table)
        upstream_table_key = return_table_key(table, key, tolerate_missing_columns=True)
        upstream_table_entries = list((table & upstream_table_key).fetch("KEY"))
        if table_name not in self.upstream_entries:
            self.upstream_entries[table_name] = []
        self.upstream_entries[table_name] += upstream_table_entries

    def merge_tracker(self, table):
        # Merge tracker from another table
        self.upstream_entries = merge_dicts_lists([self.upstream_entries, table.upstream_obj.upstream_entries])

    def remove_repeat_entries(self):
        return {k: remove_repeat_dicts(v) for k, v in self.upstream_entries.items()}



