import matplotlib.pyplot as plt
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_meta_param_name
from src.jguides_2024.position_and_maze.datajoint_position_table_helpers import fetch1_IntervalPositionInfo
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.digitize_helpers import digitize_indexed_variable
from src.jguides_2024.utils.make_bins import make_bin_edges
from src.jguides_2024.utils.plot_helpers import return_n_cmap_colors


def trials_container_default_time_vector(nwb_file_name, epochs, new_index=None):

    # Use position_and_maze data as index if none passed
    time_vector = new_index  # default
    if new_index is None:
        time_vector = np.concatenate(
            [fetch1_IntervalPositionInfo(nwb_file_name, epoch).index.values for epoch in epochs])

    return time_vector


def digitized_rt_df_all_time(
        table, nwb_file_name, epoch, param_name, new_index=None, bin_width=None, bin_edges=None, verbose=False):

    # Check inputs
    check_one_none([bin_width, bin_edges], ["bin_width", "bin_edges"])

    # Make bin edges if not passed
    if bin_edges is None:
        key = {"nwb_file_name": nwb_file_name,
               "epoch": epoch,
               get_meta_param_name(table): param_name}
        trial_shifts = (table & key).trial_shifts()
        bin_edges = make_bin_edges(x=trial_shifts,
                                   bin_width=bin_width)

    # Get time relative to trial start and reindex in this process
    rel_time_df = table().rt_df_all_time(nwb_file_name,
                                        epoch,
                                        param_name,
                                        new_index=new_index)

    # Digitize relative time
    rel_time_df["digitized_relative_time_in_trial"] = digitize_indexed_variable(
        indexed_variable=rel_time_df.relative_time_in_trial,
        bin_edges=bin_edges,
        verbose=verbose)

    return rel_time_df


def get_rt_colors(rt_vals, trial_duration, bin_width, cmap_name=None, invalid_color=None):

    # Get inputs if not passed
    if cmap_name is None:
        cmap_name = "viridis"
    if invalid_color is None:
        invalid_color = .9

    # Get a color for each ppt bin
    cmap_colors = return_n_cmap_colors(cmap=plt.get_cmap(cmap_name), num_colors=trial_duration/bin_width)

    # Append color for invalid ppt
    cmap_colors = np.vstack((cmap_colors, np.tile(invalid_color, (1, 4))))

    # Use digitized ppt to index into colors at each ppt sample
    return cmap_colors[np.asarray(rt_vals - 1, dtype=int)]

