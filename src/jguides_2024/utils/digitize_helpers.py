import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.jguides_2024.utils.interpolate_helpers import interpolate_finite_intervals
from src.jguides_2024.utils.make_bins import make_bin_edges
from src.jguides_2024.utils.vector_helpers import vector_midpoints


def digitize_indexed_variable(
        indexed_variable, bin_edges=None, new_index=None, exclude_nan=True, right=False, verbose=False):
    """
    Performs two transformations: 1) reindex vector (optional), 2) digitize vector
    :param indexed_variable:
    :param bin_edges:
    :param new_index:
    :param exclude_nan: if True, interpolate finite intervals (and not over nan)
    :param verbose:
    :return:
    """

    if bin_edges is None:
        indexed_variable_range = [np.min(indexed_variable), np.max(indexed_variable)]
        bin_edges = make_bin_edges(x=indexed_variable_range,
                                   bin_width=np.diff(indexed_variable_range)[0] / 20)

    if new_index is None:
        new_index = indexed_variable.index

    if exclude_nan:
        reindexed_variable = interpolate_finite_intervals(indexed_variable, new_index, verbose)
    else:
        reindexed_variable = pd.Series(np.interp(new_index, indexed_variable.index, indexed_variable.values),
                                       index=new_index)

    digitized_reindexed_variable = pd.Series(np.digitize(reindexed_variable, bin_edges, right=right),
                                             index=reindexed_variable.index)

    if verbose:
        fig, ax = plt.subplots(figsize=(15, 2))
        ax.plot(indexed_variable, 'o', color="gray", label="original variable")
        ax.plot(reindexed_variable, '|', markersize=10, label="reindexed variable")
        # Plot reindexed digitized variable. Add nan to bin centers for cases where variable was outside bin edges
        bin_centers_plus_nan = np.concatenate((vector_midpoints(bin_edges),
                                               np.asarray([np.nan])))
        ax.plot(digitized_reindexed_variable.index,
                bin_centers_plus_nan[list(map(int, digitized_reindexed_variable - 1))], '*', color="red",
                alpha=.5, markersize=10, label="digitized reindexed variable")
        ax.legend()
        for bin_edge in bin_edges:
            ax.plot(indexed_variable.index, [bin_edge] * len(indexed_variable), color="gray")

    return digitized_reindexed_variable
