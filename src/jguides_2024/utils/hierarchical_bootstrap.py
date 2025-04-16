"""
This module contains functions to perform a hierarchical procedure for finding the average value
and average difference of a specified quantity from bootstrap samples, when data contains multiple
levels, as described in "Application of the hierarchical bootstrap to multi-level data in
neuroscience", Saravanan et al. 2021.

USER WARNINGS:
- The average difference bootstrap code here is only appropriate if you want to resample two groups separately
(e.g. all cells in one region, then another). Example: difference between resampled metric across units in a brain area
that's present in three subjects, and resampled metric across units in a brain are that's present in five subjects.
It is NOT appropriate if you want to resample the paired difference within subjects. Example: within subject
difference between resampled metric in two brain areas. Think carefully about the overall statistical picture
you think is appropriate to assume, and whether the current bootstrap average difference code achieves that or not,
before using the current code.

Notes about design:
- Avoided using functions within segment of code that uses multiprocessing, since potential concern
this could slow things down (have not rigorously tested).
- Avoided passing variables to segment of code that uses multiprocessing, since could slow things down (qualitatively
appeared to be the case).
"""


import itertools
import multiprocessing as mp
from collections import namedtuple

import numpy as np
import pandas as pd

from src.jguides_2024.utils.df_helpers import df_pop, df_filter_columns, df_from_data_list, unique_df_column_sets
from src.jguides_2024.utils.for_loop_helpers import print_iteration_progress
from src.jguides_2024.utils.parallelization_helpers import show_error
from src.jguides_2024.utils.stats_helpers import recursive_resample, return_confidence_interval
from src.jguides_2024.utils.vector_helpers import unpack_single_element


def append_result(x):
    results.append(x)


def _boot():
    # Get average value, using single bootstrap sample
    data_list = []
    for column_set in ave_group_column_sets:  # get a separate bootstrap sample for each set of conditions
        df_subset = df_filter_columns(df, column_set)
        # Get bootstrap sample
        resample_df = recursive_resample(df_subset, resample_levels, resample_quantity)
        data_list.append(tuple(list(column_set.values()) + [average_fn(resample_df[resample_quantity])]))
    boot_ave_df = df_from_data_list(data_list, ave_group_column_names + ["boot_ave"], empty_df_has_cols=True)

    # Get average differences for pairs, using bootstrap sample
    boot_ave_diff_df = None
    if ave_diff_column_name is not None:
        boot_ave_diff_df = df_from_data_list([tuple(list(column_set.values()) + list(x_pair) + [unpack_single_element(np.diff(
            [df_pop(boot_ave_df, {**column_set, **{ave_diff_column_name: x}}, "boot_ave", verbose=True) for x in x_pair]))])
                                              for x_pair in x_pairs for column_set in ave_diff_group_column_sets],
                                             ave_diff_group_column_names + ave_diff_pair_column_names + ["boot_ave_diff"])

    # Return dfs
    return boot_ave_df, boot_ave_diff_df


def get_conf_df(df, column_names, quantity, alphas):
    conf_df = df_from_data_list([tuple([df_key[k] for k in column_names] + [alpha] + list(return_confidence_interval(
                                df_filter_columns(df, df_key)[quantity].values, alpha)))
    for df_key in unique_df_column_sets(df, column_names, as_dict=True) for alpha in alphas], column_names + [
        "alpha", "lower_conf", "upper_conf"], empty_df_has_cols=True)
    # Add column with significance
    conf_df["significant"] = [x*y > 0 for x, y in zip(conf_df.lower_conf, conf_df.upper_conf)]
    return conf_df


def hierarchical_bootstrap(
        df_, resample_levels_, resample_quantity_, ave_group_column_names_, ave_diff_group_column_names_=None,
        ave_diff_column_name_=None, num_bootstrap_samples_=1000, average_fn_=np.mean, alphas=(.05, .01, .001, .0001),
        debug_mode=False):
    """
    Bootstrap data with hierarchical structure

    :param df_: dataframe with quantity to bootstrap resample and characteristics
    :param resample_levels_: levels at which to resample data
    :param resample_quantity_: quantity to bootstrap resample
    :param ave_group_column_names_: perform resampling separately on groups with a unique set of values at these
                                    columns for example, if "brain_region" is the only entry in
                                    ave_group_column_names, perform resampling separately for each brain region
    :param ave_diff_group_column_names_: find average difference between two conditions within groups with a
                                         unique set of values at these columns
    :param ave_diff_column_name_: find average difference between groups at this value
    :param num_bootstrap_samples_: number of bootstrap samples
    :param average_fn_: function for finding average (e.g. np.mean)
    :param alphas: p values
    :param debug_mode: bool: True to run code without multiprocessing so can more easily debug
    :return:
    """

    print("Performing hierarchical bootstrap (standard)...")

    # Check inputs
    if np.sum([x is None for x in [ave_diff_group_column_names_, ave_diff_column_name_]]) == 1:
        raise Exception(f"Either ave_diff_group_column_names_ and ave_diff_column_name_ must both be None, or must "
                        f"both not be None")

    # Note that declaring variables as global allows us to not pass to function in for loop, and this
    # saves time when iterating over many items in the for loop
    global df, resample_levels, resample_quantity, ave_group_column_names, ave_diff_group_column_names, \
        ave_diff_column_name, num_bootstrap_samples, average_fn, ave_group_column_sets, ave_diff_group_column_sets, \
        x_pairs, results, ave_diff_pair_column_names
    df = df_
    resample_levels = resample_levels_
    resample_quantity = resample_quantity_
    ave_group_column_names = ave_group_column_names_
    ave_diff_group_column_names = ave_diff_group_column_names_
    ave_diff_column_name = ave_diff_column_name_
    num_bootstrap_samples = num_bootstrap_samples_
    average_fn = average_fn_

    # Get column sets for groups for bootstrapped average
    ave_group_column_sets = unique_df_column_sets(df, ave_group_column_names, as_dict=True)

    # Define variables for bootstrapped average difference
    if ave_diff_column_name is not None:
        # Get column sets for groups for bootstrapped average difference
        ave_diff_group_column_sets = unique_df_column_sets(df, ave_diff_group_column_names, as_dict=True)
        # Define pairs for which to find bootstrapped average difference
        x_pairs = list(itertools.combinations(np.unique(df[ave_diff_column_name]), r=2))
        # ...raise error if no pairs found
        if len(x_pairs) == 0:
            raise Exception(f"No pairs found for which to find bootstrapped average difference")
        # Define name of column with pairs
        ave_diff_pair_column_names = [f"{ave_diff_column_name}_{x}" for x in [1, 2]]

    # Get bootstrap distributions
    # run without multiprocessing if want to debug
    if debug_mode:
        _boot()
        raise Exception
    # run multiprocessing
    pool = mp.Pool(mp.cpu_count())
    results = []
    for idx in range(0, num_bootstrap_samples):
        pool.apply_async(_boot, args=(), callback=append_result, error_callback=show_error)
    pool.close()
    pool.join()  # waits until all processes done before running next line

    # Unpack results
    boot_ave_dfs, boot_ave_diff_dfs = zip(*results)

    # Concatenate average and average difference dfs across bootstrap iterations
    boot_ave_dfs_concat = pd.concat(boot_ave_dfs, axis=0)
    boot_ave_diff_dfs_concat = None  # initialize
    if ave_diff_column_name is not None:
        boot_ave_diff_dfs_concat = pd.concat(boot_ave_diff_dfs, axis=0)

    # Get confidence intervals from bootstrap distribution for average
    ave_conf_df = get_conf_df(boot_ave_dfs_concat, ave_group_column_names, "boot_ave", alphas)

    # Determine p value category for average difference
    ave_diff_conf_df = None  # initialize
    if ave_diff_column_name is not None:
        ave_diff_conf_df = get_conf_df(
            boot_ave_diff_dfs_concat, ave_diff_group_column_names + ave_diff_pair_column_names, "boot_ave_diff", alphas)

    # Get average values (no bootstrap)
    ave_df = df_from_data_list([tuple(list(column_set.values()) + [average_fn(df_filter_columns(
        df, column_set)[resample_quantity])]) for column_set in ave_group_column_sets],
                               ave_group_column_names + [f"ave_{resample_quantity}"], empty_df_has_cols=True)
    # Add averages to df with confidence bounds on average
    ave_conf_df = pd.merge(ave_df, ave_conf_df)

    return namedtuple("boot_dfs", "boot_ave_df boot_ave_diff_df ave_conf_df ave_diff_conf_df")(
        boot_ave_dfs_concat, boot_ave_diff_dfs_concat, ave_conf_df, ave_diff_conf_df)