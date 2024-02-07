import matplotlib.pyplot as plt
import numpy as np

from src.jguides_2024.utils.plot_helpers import format_ax
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals


def plot_spike_times(spike_times_list, min_spike_count=None, scale_spike_idxs=1, ax=None, xlims=None, colors=None,
                     alpha=.5, linewidths=None, yticks_show_unit_count=True, title=None,
                     use_eventplot=True, figsize=(20, 5)):
    """
    Plot spike times
    :param min_spike_count: exclude rows in spike_times_list with fewer than this number of spikes. Default is None
           which defaults to zero
    :param spike_times_list: nested list with spike times
    :param scale_spike_idxs: scale spike indices (along y axis) by this much
    :param ax: figure axis on which to generate plot, optional
    :param xlims: x limits, optional
    :param alpha: transparency of spikes markers, optional
    :param linewidths: widths of lines if using eventplot, optional
    :param yticks_show_unit_count: boolean, if True show total unit count on y axis, default is True
    :param title: str, optional, title
    :param use_eventplot: boolean, if True use eventplot, otherwise use "|" in scatter which puts lines closer
    together but will overlap them along the direction of the y axis even despite scaling y idxs with scale_spike_idxs.
    The scatter plot is useful for exploring the data.
    :param figsize: tuple, figure size, optional
    """

    # Get inputs if not passed
    if min_spike_count is None:
        min_spike_count = 0

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if use_eventplot:
        # Narrow events to x lims if passed
        if xlims is not None:
            spike_times_list = [event_times_in_intervals(x, [xlims])[1] for x in spike_times_list]
        # Exclude rows with fewer than min_spike_count
        spike_times_list = [x for x in spike_times_list if len(x) >= min_spike_count]
        ax.eventplot(spike_times_list, lineoffsets=np.arange(0, len(spike_times_list)) * scale_spike_idxs,
                     colors=colors, linewidths=linewidths)
    else:
        for neuron_idx, spike_times in enumerate(spike_times_list):
            ax.scatter(spike_times, [neuron_idx * scale_spike_idxs] * len(spike_times), marker='|', color=colors,
                       alpha=alpha)

    if xlims is not None:
        ax.set_xlim(xlims)

    # y ticks
    unit_count = len(spike_times_list)
    if yticks_show_unit_count:
        yticks = [unit_count]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
    if scale_spike_idxs != 1:  # remove y ticks if scaled spike y position_and_maze
        ax.set_yticks([])
    format_ax(ax=ax, title=title, ylim=[0, unit_count])