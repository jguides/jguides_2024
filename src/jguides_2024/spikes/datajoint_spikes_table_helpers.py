import copy

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name, get_unit_name
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.plot_helpers import format_ax
from src.jguides_2024.utils.vector_helpers import unpack_single_vector


def plot_smoothed_spikes_table_result(table, key=None, unit_id=None, ax=None):
    # Get key if not passed
    if key is None:
        key = table.fetch1()

    # Get firing rates
    fr_df = (table & key).fetch1_dataframe()

    # If unit_id not passed, use unit_id with max firing rate
    if unit_id is None:
        total_fr_map = {unit_id: np.mean(x) for unit_id, x in fr_df.firing_rate.items()}
        unit_id = max(total_fr_map, key=total_fr_map.get)

    # Initialize plot
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    # Plot unit firing rate
    unit_fr = pd.Series(fr_df.loc[unit_id].firing_rate, index=fr_df.loc[unit_id].sample_times)
    ax.plot(unit_fr, '.', color="black")

    # Title
    title = f"{format_nwb_file_name(key['nwb_file_name'])}, ep{key['epoch']}, " \
        f"{get_unit_name(key['sort_group_id'], unit_id)}, " \
        f"smooth param: {key['epoch_spikes_sm_param_name']}"
    format_ax(ax=ax, title=title)


def spikes_table_data_across_sort_groups(spikes_table, column_name, time_column_name, key, sort_group_unit_ids_map,
                                         sort_group_id_label_map=None, label_name=None, zscore=False,
                                         populate_tables=True):
    # Return dataframe with firing rate like data across sort groups and corresponding time vector

    # Check that sort_group_id_label_map not empty (if it is, error results)
    if sort_group_id_label_map is not None:
        if len(sort_group_id_label_map) == 0:
            raise Exception(f"No entries in sort_group_id_label_map")

    # Make copy of key to avoid altering the passed key
    key = copy.deepcopy(key)

    # Stack spikes data across units
    dfs = []
    for sort_group_id, unit_ids in sort_group_unit_ids_map.items():  # sort groups
        # Update key
        key.update({"sort_group_id": sort_group_id})
        # Populate table if indicated
        if populate_tables:
            spikes_table.populate(key)
        # Append dataframe to list if not empty
        df = (spikes_table & key).fetch1_dataframe()
        # TODO: check whether this change fine, or whether need unit_ids check after df len check
        if len(df.loc[unit_ids]) > 0:
            dfs.append(df.loc[unit_ids])
    binned_data_dict = {column_name: list(np.vstack([np.vstack(df[column_name]) for df in dfs]))}

    # Define unit labels for stacked spikes data
    unit_id_arr = np.asarray(list(map(str, np.concatenate([df[column_name].index
                                                           for df in dfs]))), dtype=object)
    sort_group_id_arr = np.concatenate([[str(sort_group_id)] * len(unit_ids)
                                        for sort_group_id, unit_ids in sort_group_unit_ids_map.items()])
    underscore_arr = np.asarray(["_"] * len(sort_group_id_arr), dtype=object)
    unit_names = sort_group_id_arr + underscore_arr + unit_id_arr
    binned_data_dict.update({"unit_name": unit_names})

    # Define additional labels for stacked spikes data (e.g. brain region) if indicated
    if sort_group_id_label_map is not None and label_name is not None:
        extra_labels = list(np.concatenate([[sort_group_id_label_map[sort_group_id]] * len(unit_ids)
                                            for sort_group_id, unit_ids in sort_group_unit_ids_map.items()]))
        binned_data_dict.update({label_name: extra_labels})

    # Convert stacked spikes data to dataframe
    df_concat = pd.DataFrame.from_dict(binned_data_dict).set_index("unit_name")

    # Z score spikes data if indicated
    if zscore:
        df_concat[column_name] = [sp.stats.zscore(x) for x in df_concat[column_name]]

    # Get time vector corresponding to spikes data
    # Approach: get time vector for each unit, check that all are equal, then return the common time vector
    time_vectors = list(np.vstack([np.vstack(df[time_column_name]) for df in dfs]))
    time_vector = unpack_single_vector(time_vectors, error_message="Time vectors not same across units")

    return df_concat, time_vector


def firing_rate_across_sort_groups(table, key, sort_group_unit_ids_map, sort_group_id_label_map=None,
                                   label_name=None, populate_tables=True):

    column_name = "firing_rate"
    time_column_name = "sample_times"
    zscore = None
    if "zscore_fr" in key:
        zscore = key["zscore_fr"]

    return spikes_table_data_across_sort_groups(spikes_table=table,
                                                column_name=column_name,
                                                time_column_name=time_column_name,
                                                key=key,
                                                sort_group_unit_ids_map=sort_group_unit_ids_map,
                                                sort_group_id_label_map=sort_group_id_label_map,
                                                label_name=label_name,
                                                zscore=zscore,
                                                populate_tables=populate_tables)


def firing_rate_across_sort_groups_epochs(table, epochs, sort_group_unit_ids_map, key=None, keys=None,
                                          sort_group_id_label_map=None, label_name=None, populate_tables=True,
                                          verbose=False):
    # Helper function to be called within tables as a method
    # Concatenate dataframes with firing rate across sort groups and epochs

    # User can restrict table with either a single key to be used across all epochs, or a list of keys corresponding
    # to epochs. Ensure params for only one method passed
    check_one_none([key, keys])
    # If keys passed, check that same length as epochs
    if keys is not None:
        if len(keys) != len(epochs):
            raise Exception(f"Length of passed keys must be same as number of epochs")

    # If single key passed, copy to avoid changing outside function and convert to keys
    if key is not None:
        keys = [copy.deepcopy(key)]*len(epochs)

    df_list = []
    time_vector_list = []
    for epoch, key in zip(epochs, keys):
        if verbose:
            print(f"On epoch {epoch}...")
        # Update key with current epoch
        key.update({"epoch": epoch})
        # Get firing rate across sort groups within current epoch
        df, time_vector = firing_rate_across_sort_groups(
            table, key, sort_group_unit_ids_map, sort_group_id_label_map, label_name, populate_tables)
        # Reset index (name) since unit names can be shared across across epochs
        df = df.reset_index()
        # Add epoch to dataframe
        df["epoch"] = [epoch] * len(df)
        # Update list with dataframes across epochs
        df_list.append(df)
        # Update list with time vectors across epochs
        time_vector_list.append(time_vector)
    # Concatenate firing rate dataframes across epochs
    fr_df = pd.concat(df_list, axis=0)
    # Make separate dataframe with time vector in each epoch
    time_vector_df = pd.DataFrame.from_dict(
        {"epoch": epochs, "time_vector": time_vector_list}).set_index("epoch")

    return fr_df, time_vector_df
