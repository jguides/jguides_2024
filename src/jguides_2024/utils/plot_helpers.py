import os
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from statannotations.Annotator import Annotator

from src.jguides_2024.utils.cd_make_if_nonexistent import cd_make_if_nonexistent
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.dict_helpers import return_n_empty_dicts, add_defaults
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.stats_helpers import return_confidence_interval
from src.jguides_2024.utils.tuple_helpers import reverse_pair
from src.jguides_2024.utils.vector_helpers import unpack_single_element


def format_ax(ax,
              xlabel=None,
              ylabel=None,
              zlabel=None,
              title=None,
              title_color=None,
              xlim=None,
              ylim=None,
              zlim=None,
              xticks=None,
              xticklabels=None,
              xticklabels_rotation=0,
              ticklabels_fontsize=None,
              yticks=None,
              yticklabels=None,
              zticks=None,
              zticklabels=None,
              tick_width=None,
              spine_width=None,
              fontsize=20,
              spines_off_list=np.asarray(["right", "top"]),
              axis_off_list=None):
    """
    Format axis of plot.
    :param ax: axis object.
    :param xlabel: string. x label.
    :param ylabel: string. y label.
    :param title: string. Title.
    :param title_color: title color.
    :param xlim: list. x limits.
    :param ylim: list. y limits.
    :param zlim: list. z limits.
    :param xticks: list. x ticks.
    :param xticklabels: list. x tick labels.
    :param xticklabels_rotation: float, rotation of x tick labels in degrees
    :param ticklabels_fontsize: float, font size for x tick labels
    :param yticks: list. y ticks.
    :param yticklabels: list. y tick labels.
    :param zticks: list. z ticks.
    :param zticklabels: list. z tick labels.
    :param tick_width: float. Thickness of ticks.
    :param spine_width: float. Thickness of spines.
    :param fontsize: number. font size.
    :param spines_off_list: list. Remove these spines.
    :return:
    """

    # Define inputs if not passed
    if title_color is None:
        title_color = "black"

    # Check inputs
    if axis_off_list is not None:
        check_membership(axis_off_list, ["x", "y"], "axis_off_list", "valid axis_off_list")

    AxSetting = namedtuple("AxSetting", "setting ax_method args")
    plot_setting_objs = [AxSetting(*x) for x in
                         [(xlabel, "set_xlabel", {"xlabel": xlabel, "fontsize": fontsize}),
                          (ylabel, "set_ylabel", {"ylabel": ylabel, "fontsize": fontsize}),
                          (zlabel, "set_zlabel", {"zlabel": zlabel, "fontsize": fontsize}),
                          (title, "set_title",  {"label": title, "fontsize": fontsize, "color": title_color}),
                          (fontsize, "tick_params",  {"labelsize": fontsize}),
                          (tick_width, "tick_params", {"width": tick_width}),
                          (xticks, "set_xticks", {"ticks": xticks}),
                          (yticks, "set_yticks", {"ticks": yticks}),
                          (zticks, "set_zticks", {"ticks": zticks}),
                          (xticklabels, "set_xticklabels", {"labels": xticklabels,
                                                            "fontsize": ticklabels_fontsize,
                                                            "rotation": xticklabels_rotation}),
                          (yticklabels, "set_yticklabels", {"labels": yticklabels,
                                                            "fontsize": ticklabels_fontsize}),
                          (zticklabels, "set_zticklabels", {"labels": zticklabels,
                                                            "fontsize": ticklabels_fontsize}),
                          (xlim, "set_xlim", {"left": xlim}),
                            (ylim, "set_ylim", {"bottom": ylim}),
                            (zlim, "set_zlim", {"bottom": zlim})]]
    for obj in plot_setting_objs:
        if obj.setting is not None:
            getattr(ax, obj.ax_method)(**obj.args)

    if spine_width is not None:
        # Note that if spine width set, spines seem to persist when removing axes below. Define spines to remove
        # so as to avoid this --> actually this doesnt seem to help...
        spines = ['top', 'bottom', 'right', 'left']  # default is all spines
        # TODO (feature): delete the below and comment above if no longer needed
        if axis_off_list is not None:
            for axis, spine in {"x": "bottom", "y": "left"}.items():
                if axis in axis_off_list:
                    spines.remove(spine)
        for spine in spines:
            ax.spines[spine].set_linewidth(spine_width)
    for spine in spines_off_list:
        ax.spines[spine].set_visible(False)
    # Remove axes
    if axis_off_list is not None:
        if "y" in axis_off_list:
            ax.get_yaxis().set_visible(False)
        if "x" in axis_off_list:
            ax.get_xaxis().set_visible(False)


def get_gridspec_ax_maps(mega_row_iterables, mega_column_iterables, row_iterables, column_iterables,
                         num_mega_rows_top=0, num_mega_rows_bottom=0, num_mega_columns_left=0,
                         num_mega_columns_right=0, fig=None,
                         sharex=False, sharey=False, subplot_width=2, subplot_height=2, mega_row_gap_factor=None,
                         mega_column_gap_factor=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None,
                         constrained_layout=True, add_subplot_args=None):
    # Note that could not figure out how to get axis sharing (x and y) to work, nor constrained_layout

    # Get inputs if not passed
    if mega_row_gap_factor is None:
        mega_row_gap_factor = .2
    if mega_column_gap_factor is None:
        mega_column_gap_factor = .2
    if add_subplot_args is None:
        add_subplot_args = dict()

    # Define gaps between mega rows and between mega columns
    total_num_mega_rows = len(mega_row_iterables) + num_mega_rows_top + num_mega_rows_bottom
    total_num_mega_columns = len(mega_column_iterables) + num_mega_columns_left + num_mega_columns_right
    mega_row_gap = mega_row_gap_factor/total_num_mega_rows
    mega_column_gap = mega_column_gap_factor/total_num_mega_columns

    def _get_stack_values(idx, num_mega_iterables_before, total_num_mega_iterables, plot_gap):
        shift = num_mega_iterables_before / total_num_mega_iterables
        return [idx / total_num_mega_iterables + shift + plot_gap,
                (1 + idx) / total_num_mega_iterables + shift - plot_gap]

    # Initialize figure
    # ...Define number of plot components to get figure size
    num_mega_rows, num_mega_columns, num_rows, num_columns = list(
        map(len, [mega_row_iterables, mega_column_iterables, row_iterables, column_iterables]))
    # ...Get figure if not passed
    if fig is None:
        fig = plt.figure(figsize=get_figsize(
            num_mega_rows * num_rows, num_mega_columns * num_columns, subplot_width, subplot_height),
            constrained_layout=constrained_layout)

    # Define total number of mega rows
    total_num_mega_rows = len(mega_row_iterables) + num_mega_rows_top + num_mega_rows_bottom

    # Make map from iterables to gridspec object and to ax
    ax_dict, gs_dict = return_n_empty_dicts(2)
    for mega_row_idx, mega_row_val in enumerate(mega_row_iterables):
        for mega_column_idx, mega_column_val in enumerate(mega_column_iterables):

            # Populate gridspec dictionary with mega columns/rows
            # For row stack values, go top to bottom.
            row_stack_values = list(total_num_mega_rows - np.asarray(_get_stack_values(
                mega_row_idx, num_mega_rows_top, total_num_mega_rows, mega_row_gap))[::-1])
            column_stack_values = _get_stack_values(  # left to right
                mega_column_idx, num_mega_columns_left, total_num_mega_columns, mega_column_gap)
            stack_names = ["bottom", "top", "left", "right"]
            stack_args = {k: v for k, v in zip(stack_names, row_stack_values + column_stack_values)}
            gs_dict[(mega_row_val, mega_column_val)] = GridSpec(
                num_rows, num_columns, wspace=wspace, hspace=hspace, width_ratios=width_ratios,
                height_ratios=height_ratios, **stack_args)

            # Populate dictionary with row and column iterables
            for row_idx, row_val in enumerate(row_iterables):
                for column_idx, column_val in enumerate(column_iterables):
                    # Get 3 digit integer that defines subplot
                    if num_rows*num_columns == 1:
                        subplot_ints = gs_dict[(mega_row_val, mega_column_val)][:]
                    elif num_rows == 1:
                        subplot_ints = gs_dict[(mega_row_val, mega_column_val)][column_idx]
                    elif num_columns == 1:
                        subplot_ints = gs_dict[(mega_row_val, mega_column_val)][row_idx]
                    else:
                        subplot_ints = gs_dict[(mega_row_val, mega_column_val)][row_idx, column_idx]
                    # Define shared axes if indicated
                    share_ax_map = {"rows": (mega_row_val, mega_column_val, 0, column_idx),
                                    "columns": (mega_row_val, mega_column_val, row_idx, 0),
                                    True: (mega_row_val, mega_column_val, 0, 0)}
                    share_args = dict()
                    for share_keyword, x in zip(["sharex", "sharey"],
                                                [sharex, sharey]):
                        if x in share_ax_map:
                            if share_ax_map[x] in ax_dict:
                                share_args[share_keyword] = ax_dict[share_ax_map[x]]
                    ax_dict[(mega_row_val, mega_column_val, row_val, column_val)] = fig.add_subplot(
                        subplot_ints, **{**share_args, **add_subplot_args})
    return gs_dict, ax_dict, fig


def get_plot_idx_map(mega_row_iterables, mega_column_iterables, row_iterables, column_iterables):
    PlotIdx = namedtuple("PlotIdx", "mega_row_idx mega_column_idx row_idx column_idx")
    plot_idx_map = dict()
    for mega_row_idx, mega_row in enumerate(mega_row_iterables):
        for mega_column_idx, mega_column in enumerate(mega_column_iterables):
            for row_idx, row in enumerate(row_iterables):
                for column_idx, column in enumerate(column_iterables):
                    plot_idx_map[(mega_row, mega_column, row, column)] = PlotIdx(mega_row_idx, mega_column_idx, row_idx,
                                                                                 column_idx)

    return plot_idx_map


def get_figsize(num_rows, num_columns, subplot_width, subplot_height):
    return (subplot_width*num_columns, subplot_height*num_rows)


def get_fig_axes(num_rows=None, num_columns=None, num_subplots=None, sharex=False, sharey=False, figsize=None,
                 subplot_width=None, subplot_height=None, remove_empty_subplots=True, gridspec_kw=None):
    # Check inputs not under or overspecified
    if np.sum([x is None for x in [num_rows, num_columns, num_subplots]]) != 1:
        raise Exception(f"Exactly two of num_rows, num_columns, num_subplots must be passed")
    if figsize is not None and (subplot_width is not None or subplot_height is not None):
        raise Exception(f"If figsize is passed, subplot_width and subplot_height should be None")

    # Define missing parameters
    if num_subplots is None:
        num_subplots = num_rows*num_columns
    if num_rows is None:
        num_rows = int(np.ceil(num_subplots/num_columns))
    elif num_columns is None:
        num_columns = int(np.ceil(num_subplots/num_rows))

    # If rows or columns is one, remove extra subplots if indicated
    if remove_empty_subplots and (num_rows == 1 or num_columns == 1) and num_rows * num_columns < num_subplots:
        if num_rows == 1:
            num_columns = num_subplots
        elif num_columns == 1:
            num_rows = num_subplots

    # Get figure size if not passed, using above params
    if figsize is None and subplot_width is not None and subplot_height is not None:
        figsize = get_figsize(num_rows, num_columns, subplot_width, subplot_height)

    # Return subplots
    return plt.subplots(num_rows, num_columns, sharex=sharex, sharey=sharey, figsize=figsize, gridspec_kw=gridspec_kw)


def get_num_subplots(axes):
    if single_axis(axes):
        return 1
    elif axes.ndim == 1:  # one row or one column of subplots
        return len(axes)
    else:
        return np.prod(np.shape(axes))


def single_axis(axes):
    return hasattr(axes, 'plot')


def get_ax_for_layout(axes, plot_num, layout="left_right"):
    """
    Return ax from axes if arranging plots left to right then top to bottom, or top to bottom then left to right
    :param axes: array with axis objects
    :param plot_num: plot number
    :param layout: "left_right" to arrange plots left to right then top to bottom. "top_bottom" to arrange plots top
           to bottom then left to right.
    :return: current axis given plot number
    """
    # Check layout valid
    check_membership([layout], ["left_right", "top_bottom"], "layout", "valid layouts")

    if single_axis(axes):  # single axis object
        return axes
    if len(np.shape(axes)) == 1:  # # one row or one column of subplots
        return axes[plot_num]
    elif len(np.shape(axes)) == 2:  # 2D panel of subplots
        num_rows, num_columns = np.shape(axes)
        if layout == "left_right":
            row, col = divmod(plot_num, num_columns)  # find row/column for current plot
        elif layout == "top_bottom":
            col, row = divmod(plot_num, num_rows)  # find row/column for current plot
        return axes[row, col]  # get axis for current plot
    else:
        raise Exception(f"axes do not conform to expected cases")


def plot_distribution_confidence_interval(x, bins=None, alpha=.05, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if bins is None:
        bins = np.linspace(np.min(x), np.max(x), 20)
    confidence_interval = return_confidence_interval(x, alpha)
    hist_counts_bins = ax.hist(x, bins=bins, color="gray")
    for c in confidence_interval:
        ax.plot([c]*2, [0, np.max(hist_counts_bins[0])], color="black")
    return hist_counts_bins


def get_default_plot_save_dir():
    return "/home/jguidera/to_sync"


def save_figure(fig, file_name, figure_type=None, dpi=2000, save_dir=None, save_fig=True):

    # Save figure

    # Setting so that font shows up in svg
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"

    # Get inputs if not passed
    if figure_type is None:
        figure_type = ".svg"

    if save_fig:
        # Get inputs if not passed
        if save_dir is None:
            save_dir = get_default_plot_save_dir()
        current_dir = os.getcwd()  # change back to this directory after saving
        cd_make_if_nonexistent(save_dir)
        file_name += figure_type
        print(f"Saving {file_name} in {os.getcwd()}")
        fig.savefig(file_name, transparent=True, dpi=dpi,
                    bbox_inches="tight")  # transparent=True gets rid of white background
        os.chdir(current_dir)  # change back to directory


def plot_histogram(x, bins=None, color="gray", vertical_line_x=None, arrow_at_median=True, ax=None):
    from src.jguides_2024.utils.make_bins import make_bin_edges
    if ax is None:
        fig, ax = plt.subplots()
    if bins is None:
        bins = make_bin_edges(x, bin_width=(np.max(x) - np.min(x))/20)
    counts = ax.hist(x, bins=bins, color=color)[0]
    line_extent = [0, np.max(counts) * 1.1]
    if vertical_line_x is not None:
        ax.plot([vertical_line_x] * 2, line_extent, color="black", linewidth=2)
    if arrow_at_median and len(x) != 0:
        # Only works if all data within bins
        # if np.nanmax(x) > bins[-1]:
        #     print("cannot put arrow at median with current code because x falls outside bins")
        x_median = np.nanmedian(x)
        ax.arrow(x_median,
                 counts[np.digitize([x_median], bins=bins) - 1] * 1.2,
                 0, -.0001, head_width=0.05, head_length=line_extent[1] * .05, overhang=0,
                 color="black")  # arrow at median


def plot_spanning_line(
        span_data, constant_val, ax, span_axis="x", linewidth=1, color="black", linestyle="solid", alpha=1,
        zorder=None):
    if span_axis not in ["x", "y"]:
        raise Exception("span_axis must be either x or y")
    data_range = [np.min(span_data), np.max(span_data)]
    line_val = [constant_val] * 2
    plot_x = data_range
    plot_y = line_val
    if span_axis == "y":
        plot_x, plot_y = plot_y, plot_x
    ax.plot(plot_x, plot_y, linewidth=linewidth, color=color, linestyle=linestyle, alpha=alpha, zorder=zorder)


def path_name_to_plot_string(path_name):
    well1, well2 = path_name.split("_to_")
    return f"{well1.split('_well')[0]}-{well2.split('_well')[0]}"


def plot_true_ranges(ax, valid_bool, x_vals=None, y_val=None, **kwargs):
    from src.jguides_2024.utils.vector_helpers import find_spans_increasing_list

    # If x values passed, check that same length as valid bool
    if x_vals is not None:
        if len(x_vals) != len(valid_bool):
            raise Exception(f"x_vals must be same length as valid_bool")
    # Otherwise define
    else:
        x_vals = np.arange(0, len(valid_bool))

    # Define y val as y axis maximum value if not passed
    if y_val is None:
        y_val = ax.get_ylim()[-1]

    span_idxs = find_spans_increasing_list(np.where(valid_bool)[0])[0]

    for x1, x2 in span_idxs:
        ax.plot([x_vals[x1], x_vals[x2]], [y_val] * 2, **kwargs)


def plot_intervals(interval_list, ax=None, val_list=None, interval_axis="x", color="black", label=None):

    """
    Plot intervals
    :param interval_list: list of start/stop of each interval
    :param ax: matplotlib axis object
    :param val_list: list same length as interval_list with values to plot on x or y axis
    :param interval_axis: str, either "x" or "y", if "x" intervals along x-axis, if "y" intervals along y-axis
    :param color: color of intervals
    :param label: name of intervals
    :return ax: matplotlib axis object
    """

    # Check interval_axis valid
    check_membership([interval_axis], ["x", "y"])

    # Define axis if not passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 2))

    # Define y values if not passed
    if val_list is None:
        val_list = [1] * len(interval_list)

    # Check that as many values as intervals
    if len(interval_list) != len(val_list):
        raise Exception("must be as many values in val_list as intervals in interval_list")

    # Plot intervals
    for idx, (interval, y_val) in enumerate(zip(interval_list, val_list)):

        # Define input order so that intervals span specified axis
        plot_x = interval
        plot_y = [y_val] * 2
        if interval_axis == "y":
            plot_x, plot_y = reverse_pair([plot_x, plot_y])

        # Define label
        label_ = None
        if idx == 0:
            label_ = label

        # Plot interval
        ax.plot(plot_x, plot_y, "o-", color=color, label=label_)

    # Return axis
    return ax


def add_colorbar(img, fig, ax, cbar_location="left", orientation="vertical",
                 ticks=None, ticklabels=None, size="5%", pad_factor=.05, fontsize=None):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    # Scale pad based on figure width (cannot do if subfigure object passed)
    if not isinstance(fig, matplotlib.figure.SubFigure):
        pad_factor *= fig.get_size_inches()[0]

    cax = divider.append_axes(cbar_location, size=size, pad=pad_factor)

    cbar = fig.colorbar(img, ax=[ax], cax=cax, orientation=orientation, ticks=ticks)

    # Tick labels
    if ticklabels is not None:  # must be iterable to add
        if orientation == "vertical":
            cbar.ax.set_yticklabels(ticklabels)
        else:  # horizontal
            cbar.ax.set_xticklabels(ticklabels)

    # Tick fontsize
    cbar.ax.tick_params(labelsize=fontsize)

    # Return color bar
    return cbar


def get_cmap_colors(cmap_name):
    return matplotlib.cm.get_cmap(cmap_name).colors


def return_n_cmap_colors(num_colors=10, cmap=plt.get_cmap("jet"), squeeze_lower_range=None, squeeze_upper_range=None):
    # Get number of unused colors
    num_squeezed_out_colors = 0  # default
    if squeeze_upper_range is not None:
        num_squeezed_out_colors += squeeze_upper_range
    if squeeze_lower_range is not None:
        num_squeezed_out_colors += squeeze_lower_range
    # Define total number of colors to get
    total_num_colors = num_colors + num_squeezed_out_colors
    colors = np.asarray([cmap(n/total_num_colors) for n in np.arange(0, total_num_colors)])
    # Squeeze out colors if indicated
    if squeeze_lower_range is not None:
        colors = colors[squeeze_lower_range:]
    if squeeze_upper_range is not None:
        colors = colors[:-squeeze_upper_range]
    return colors


def plot_text_color(text_list, colors, fig_ax_list=None, file_name_save="text_colors", save_fig=False):
    # Initialize axis if not passed
    if fig_ax_list is None:
        fig_ax_list = plt.subplots()
    # Unpack figure and axis
    fig, ax = fig_ax_list
    for idx, (text, color) in enumerate(zip(text_list, colors)):
        ax.text(0, idx, text, color=color)
    ax.set_ylim([-1, idx + 1])
    ax.set_xlim([-1, 1])
    ax.axis("off")
    save_figure(fig, file_name_save, save_fig=save_fig)


def plot_colors(colors):
    text_list = [str(x) for x in np.arange(0, len(colors))]
    plot_text_color(text_list, colors)


def plot_heatmap(arr, clim=None, scale_clim=1, fig_ax_list=None, figsize=(15, 5), plot_color_bar=True, xlabel=None,
                 ylabel=None, cbar_location="right", cbar_ticks=None, cbar_ticklabels=None, cbar_fontsize=None,
                 edgecolors=None, axis_off=False, zorder=None, **kwargs):

    # Plot heat map

    # Define clim if not passed
    if clim is None:
        clim = [np.nanmin(arr), np.nanmax(arr)]

    # Define color bar ticks as min and max values if not passed
    if cbar_ticks is None:
        cbar_ticks = clim

    # Initialize figure if not passed
    if fig_ax_list is None:
        fig_ax_list = plt.subplots(figsize=figsize)

    # Get figure and axis
    fig, ax = fig_ax_list

    img = ax.pcolormesh(arr, cmap="Greys", edgecolors=edgecolors, zorder=zorder)
    img.set_clim(np.asarray(clim)*scale_clim)

    # Add color bar if indicated
    if plot_color_bar:
        add_colorbar(img, fig, ax, cbar_location=cbar_location, ticks=cbar_ticks, ticklabels=cbar_ticklabels,
                     fontsize=cbar_fontsize)

    # If df and no x and y labels passed, add x and y labels given by df index and column names
    if isinstance(arr, pd.DataFrame):
        if xlabel is None and arr.index.name is not None:
            xlabel = arr.columns.name
        if ylabel is None and arr.columns.name is not None:
            ylabel = arr.index.name

    # Format axis
    format_ax(ax=ax, xlabel=xlabel, ylabel=ylabel, **kwargs)

    # Remove axis if indicated
    if axis_off:
        ax.axis("off")


def get_ticklabels(ticks, label_every_n=None, idxs=None, include_endpoints=False, round_n=None):
    # Two methods of defining tick labels: 1) label every n ticks, or 2) label ticks corresponding to passed idxs

    # If no method specified, default to labeling every tick
    if label_every_n is None and idxs is None:
        label_every_n = 1

    # Check that one or no method passed
    check_one_none((label_every_n, idxs))

    # Round to nearest nth decimal if indicated (do before converting ticks to strings)
    if round_n is not None:
        ticks = [np.round(tick, round_n) for tick in ticks]

    # Method 1: label every n ticks
    if idxs is None:
        ticklabels = [str(tick) if idx%label_every_n == 0 else "" for idx, tick in enumerate(ticks)]

    # Method 2: label ticks specified by passed idxs
    else:
        ticklabels = [str(ticks[idx]) if idx in idxs else "" for idx in np.arange(0, len(ticks))]

    # Add endpoints if indicated
    if include_endpoints:
        for idx in [0, -1]:
            ticklabels[idx] = str(ticks[idx])

    # Return tick labels
    return ticklabels


def add_jitter(x, jitter_extent=.1, unpack_single_value=True):
    x = np.asarray(x)
    jittered_x = x + np.random.uniform(-jitter_extent/2, jitter_extent/2, len(x))
    if unpack_single_value and len(jittered_x) == 1:
        return jittered_x[0]


def tint_color(color, num_tints):
    # Return color along with tints (lighter shades of the color)

    # Convert color to array
    color = np.asarray(color)
    # Check that color has three values
    if len(color) != 3:
        raise Exception(f"color should be array with 3 values denoting RGB setting. color: {color}")
    return np.asarray([color + (1 - color) * tint_factor for tint_factor in np.arange(0, 1, 1 / num_tints)])


def plot_violin_or_box(df, hue, y, x, test="Mann-Whitney", comparison_pairs=None, fig_ax_list=None, figsize=(4, 4),
                       plot_type="box",
                       cut=0, font_scale=12, legend_fontsize=12, ticklabels_fontsize=12, showfliers=False,
                       order=None, order_names=None, show_legend=True, violin_colors=None,
                       ylim=None, yticks=None, yticklabels=None, file_name_save=None, save_fig=False):
    # Wrapper for seaborn violin or box plot

    # Check inputs
    if comparison_pairs is not None:
        if len(comparison_pairs) < 2:
            raise Exception(f"Need at least two items in comparison_pairs")
    if save_fig and file_name_save is None:
        raise Exception(f"file_name_save must be passed if save_fig is True")
    check_membership([plot_type], ["box", "violin"])  # check plot_type valid

    # Unpack figure / axis if passed, otherwise initialize
    if fig_ax_list is None:
        fig_ax_list = plt.subplots(figsize=figsize)
    fig, ax = fig_ax_list

    # Collect plot params
    plot_params = {"data": df, "hue": hue, "y": y, "x": x, "order": order}
    if plot_type == "violin":
        plot_params.update({"cut": cut, "font_scale": font_scale})
    elif plot_type == "box":
        plot_params.update({"palette": violin_colors, "showfliers": showfliers})

    # Plot
    if plot_type == "box":
        sns.boxplot(ax=ax, **plot_params)
    elif plot_type == "violin":
        sns.violinplot(ax=ax, **plot_params)

    # Color each plot and add legend for "hue" argument  # TODO (feature): make specific to violin if doesnt work for box
    if violin_colors is None:  # define colors if not passed
        violin_colors = ["gray"] * len(ax.findobj(PolyCollection))
    # Ghost handles; useful for specifying legend colors below
    ghost_handles = [plt.Rectangle((0, 0), 0, 0, facecolor=violin_color, edgecolor="black")
                           for violin_color in violin_colors]
    for violin_plot, violin_color in zip(ax.findobj(PolyCollection), violin_colors):
        violin_plot.set_facecolor(violin_color)

    # Add statistical comparison of passed condition pairs to plot
    if comparison_pairs is not None:
        if len(comparison_pairs) > 0:
            annotator = Annotator(ax, comparison_pairs, **plot_params)
            annotator.configure(test=test).apply_and_annotate()

    # Legend if indicated (requires names of ordered elements passed)
    if show_legend and order_names is not None:
        # Get number of hues (number of unique settings of df column with name given by hue)
        num_hues = len(set(df[hue]))
        ax.legend(handles=[tuple(ghost_handles[:num_hues]),
                           tuple(ghost_handles[num_hues:])], labels=order_names,
                  title="", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                  fontsize=legend_fontsize,
                  loc='upper left', bbox_to_anchor=(1.03, 1))

    # Format plot
    # ...Set left spine to same range as yticks if yticks passed
    if yticks is not None:
        ax.spines["left"].set_bounds(np.min(yticks), np.max(yticks))
    format_ax(ax=ax, ylim=ylim, xlabel="", ylabel=y, yticks=yticks, yticklabels=yticklabels, axis_off_list=["x"],
              tick_width=1.5, spine_width=1.5, ticklabels_fontsize=ticklabels_fontsize)
    ax.get_xaxis().set_visible(False)
    sns.despine(
        bottom=True)  # note that ax.get_xaxis().set_visible(False) seemed to not work with seaborn or matplotlib plot
    ax.yaxis.set_label_coords(-0.2, .48)
    save_figure(fig, file_name_save, save_fig=save_fig)


def plot_ave_conf(ax, x_vals, mean, conf_bounds, **plot_params):
    """
    Plot average and confidence intervals
    :param ax: axis object
    :param x_vals: x values
    :param mean: average values
    :param conf_bounds: [(lower_1, upper_1), (lower_2, upper_2), ...]
    :param plot_params: dictionary with addition parameters for plot
    """
    # Initialize figure if axis not passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))
    # Get alpha
    alpha = plot_params.pop("alpha", 1)
    fill_alpha = plot_params.pop("fill_alpha", .3)
    # Get color
    color = plot_params.pop("color", "black")
    # Plot average
    # add label if passed
    label = plot_params.pop("label", None)
    ax.plot(x_vals, mean, color=color, alpha=alpha, zorder=10, label=label)
    # Confidence bounds
    lower_bound, upper_bound = zip(*conf_bounds)
    # ...Plot confidence bounds on average if indicated
    plot_conf_bounds = plot_params.pop("plot_conf_bounds", False)
    if plot_conf_bounds:
        for plot_y in [lower_bound, upper_bound]:
            ax.plot(x_vals, plot_y, color=color, alpha=alpha, zorder=10)
    # ...Fill between confidence bounds
    ax.fill_between(x_vals, y1=lower_bound, y2=upper_bound, color=color, zorder=9, alpha=fill_alpha)  # color="white"
    # Format plot
    format_ax(ax=ax, **plot_params)


def get_centered_axis_val(val, vals, val_separation, extra_shift=0, jitter_extent=0):
    idx = unpack_single_element(np.where(np.asarray(vals) == val)[0])  # where brain region falls in list
    shift = idx*val_separation - (len(vals) - 1)*val_separation/2
    return add_jitter([shift + extra_shift], jitter_extent)


def plot_errorbar(lower_conf, upper_conf, ave_val, x_val, ax, plot_average=True, **plot_params):
    # Add default plot params if not passed
    plot_params = add_defaults(
        plot_params, {"markersize": 5, "capsize": 2, "linewidth": 2, "solid_capstyle": "projecting",
                      "ave_line_extent": 1}, add_nonexistent_keys=True)

    # Plot average using horizontal line
    if plot_average:
        params = {k: v for k, v in plot_params.items() if k in ["color", "markersize"]}
        shift = plot_params.pop("ave_line_extent")
        x_vals = [x_val - shift, x_val + shift]
        ax.plot(x_vals, [ave_val] * 2, "-", **params)

    # Plot error bars
    errors = abs(np.asarray([lower_conf, upper_conf]) - ave_val)
    params = {k: v for k, v in plot_params.items() if k in ["color", "solid_capstyle", "capsize", "linewidth"]}
    ax.errorbar(x_val, ave_val, np.asarray([errors]).T, **params)