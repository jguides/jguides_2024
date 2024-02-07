import copy

import matplotlib.pyplot as plt
import numpy as np


def exclude_intervals(starting_interval_list, exclude_interval_list):
    # Exclude intervals

    # Check that interval list consists of pairs of values
    if np.shape(starting_interval_list)[1] != 2:
        raise Exception(
            f"interval_list must be a list with pairs of values, i.e. shape should be n by 2. Instead, shape is: "
            f"{np.shape(starting_interval_list)}")

    # Exclude periods in exclude_interval_list from starting_interval_list
    new_interval_list = []  # initialize list for starting_interval_list after removing periods in exclude_interval_list
    for interval in starting_interval_list:  # for each valid interval
        valid_interval = list(copy.deepcopy(interval))  # initialize list to store valid interval
        for exclude_interval in exclude_interval_list:  # for each interval we want to exclude
            if np.logical_and(valid_interval[0] > exclude_interval[0],
                              valid_interval[1] < exclude_interval[
                                  1]):  # if valid interval contained in exclusion interval
                valid_interval = None
                break
            elif np.logical_and(valid_interval[1] > exclude_interval[0],
                                valid_interval[1] < exclude_interval[
                                    1]):  # if end of valid interval within exclusion interval
                valid_interval[1] = exclude_interval[0]  # set end of valid interval to start of exclusion interval
            elif np.logical_and(valid_interval[0] > exclude_interval[0],
                                valid_interval[0] < exclude_interval[
                                    1]):  # if start of valid interval within exclusion interval
                valid_interval[0] = exclude_interval[1]  # set start of valid interval to end of exclusion interval
        if valid_interval is not None:  # append valid interval
            new_interval_list.append(valid_interval)

    # Plot result for checking
    from src.jguides_2024.utils.plot_helpers import plot_intervals
    for xlims in [None] + list(starting_interval_list):  # for full set of intervals and for each starting interval
        fig, ax = plt.subplots(figsize=(35, 2))
        _ = plot_intervals(starting_interval_list, ax, color="black", label="original intervals")
        _ = plot_intervals(exclude_interval_list, ax, val_list=[2] * len(exclude_interval_list),
                           color="red", label="excluded intervals")
        _ = plot_intervals(new_interval_list, ax, val_list=[3] * len(new_interval_list), color="green",
                           label="new intervals")
        ax.set_ylim([0, 4])
        ax.set_xlim(xlims)
        ax.axes.get_yaxis().set_visible(False)
        ax.legend()

    return new_interval_list
