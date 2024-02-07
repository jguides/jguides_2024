# Parallelized loading of glm results

import copy
import multiprocessing as mp

from src.jguides_2024.glm.jguidera_el_net import ElNet
from src.jguides_2024.utils.df_helpers import df_from_data_list
from src.jguides_2024.utils.parallelization_helpers import show_error


def append_result(x):
    if x is not None:
        data_list.append(x)


def _get_dfs(glm_restriction_idx, unit_name, df_row, key):

    table_subset = (ElNet & key)
    if len(table_subset) == 0 and tolerate_missing_units:
        return None
    dfs = table_subset.fetch1_dataframes()

    return (glm_restriction_idx, unit_name, df_row.brain_region,
            dfs.results_folds_merged_df, dfs.folds_df, dfs.fit_params_df, dfs.log_likelihood)


def get_glm_results_df(arg_sets, tolerate_missing_units_, debug_mode=True):

    global tolerate_missing_units, data_list

    tolerate_missing_units = copy.deepcopy(tolerate_missing_units_)

    data_list = []

    pool = mp.Pool(mp.cpu_count())
    # Iterate over restriction conditions and units
    # If want to debug, do in for loop
    if debug_mode:
        for arg_set in arg_sets:
            _get_dfs(*arg_set)

    # Otherwise do using parallelization
    for arg_set in arg_sets:
        pool.apply_async(
            _get_dfs, args=arg_set, callback=append_result, error_callback=show_error)

    pool.close()
    pool.join()  # waits until all processes done before running next line

    return df_from_data_list(
        data_list, ["glm_restriction_idx", "unit_name", "brain_region", "results_folds_merged_df", "folds_df",
                    "fit_params", "log_likelihood"])
