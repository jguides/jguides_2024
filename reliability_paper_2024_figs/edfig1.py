import os

import matplotlib.pyplot as plt
import numpy as np

# Import custom datajoint tables
analysis_dir = '/home/jguidera/Src/jguides_2024'
os.chdir(analysis_dir)
from src.jguides_2024.utils.vector_helpers import unpack_single_element
from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_reliability_paper_nwb_file_names, get_plot_marker
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescriptions, RunEpoch
from src.jguides_2024.task_event.jguidera_task_performance import AlternationTaskPerformanceStatistics
from src.jguides_2024.utils.plot_helpers import plot_spanning_line, format_ax, save_figure

# Define plotting params
epoch_gap = .2
fontsize = 30
ticklabels_fontsize = 23
markersize = 8
linewidth = 2
marker_alpha = .6
line_alpha = .6
subplot_width = 10
subplot_height = 3
save_fig = True

# Get nwb file names
nwb_file_names_map = get_reliability_paper_nwb_file_names(as_dict=True)

# Initialize figure
fig, ax = plt.subplots(figsize=(subplot_width, subplot_height))

# Define expected total number of epochs in each of the three days on Haight post-acquisition rotation
num_epochs_per_day = [
    8, 8, 6]

key = dict()
cum_x_vals = []
xticklabels = []
for subject_id_idx, (subject_id, nwb_file_names) in enumerate(nwb_file_names_map.items()):

    # initialize x value
    x1 = 0
    x1_label = 0

    for nwb_file_name, num_epochs in zip(nwb_file_names, num_epochs_per_day):

        # Get valid single contingency epochs for this nwb file
        epochs = EpochsDescriptions().get_epochs(nwb_file_name, "valid_single_contingency_runs")

        # Get performance during these epochs
        key = {"nwb_file_name": nwb_file_name}
        performance = [(AlternationTaskPerformanceStatistics & {**key, **{"epoch": epoch}}).fetch1("percent_correct")
                       for epoch in epochs]

        # Define x values
        # ...Get all run epochs
        all_run_epochs = (RunEpoch() & {"nwb_file_name": nwb_file_name}).fetch("epoch")
        # ...Remove run epochs with preceeding run rather than sleep sessions (i.e. view split run sessions as a single
        # run session)
        all_run_epochs = [all_run_epochs[0]] + [
            epoch for epoch, diff in zip(all_run_epochs[1:], np.diff(all_run_epochs)) if diff > 1]
        # ...Define x value as run session number
        x_vals = np.asarray([unpack_single_element(np.where(all_run_epochs == epoch)[0]) for epoch in epochs]) + x1

        # Plot performance, connecting contiguous sessions in the day with a line
        # ...Markers for sessions
        ax.plot(x_vals, performance, get_plot_marker(nwb_file_name=nwb_file_name), color="black", markersize=markersize,
                alpha=marker_alpha)
        # ...Lines between contiguous sessions
        for x_val_1, x_val_2, p1, p2 in zip(x_vals[:-1], x_vals[1:], performance[:-1], performance[1:]):
            if x_val_2 - x_val_1 == 1:
                ax.plot([x_val_1, x_val_2], [p1, p2], '-', linewidth=linewidth, color="black", alpha=line_alpha)

        # Store x tick values / labels
        if subject_id_idx == 0:
            cum_x_vals += list(x_vals)
            xticklabels += [x1_label + 1] + [""] * (num_epochs - 1)  # add one so not zero indexed to x tick labels

        # Update x value
        x1 += num_epochs + epoch_gap  # total possible epochs plus gap factor
        x1_label += num_epochs + 1

# Add dotted line for "chance" performance
plot_spanning_line(ax=ax, span_axis="x", span_data=cum_x_vals, constant_val=1 / 3, linestyle="dotted")

# Format plot
xlim = [cum_x_vals[0] - 1, cum_x_vals[-1] + 1]
yticks = [0, .2, .4, .6, .8, 1]
format_ax(ax=ax, ylim=[0, 1.05], xlabel="Session", ylabel="Fraction\ncorrect", xlim=xlim, xticks=cum_x_vals,
          xticklabels=xticklabels,
          yticks=yticks, yticklabels=yticks, ticklabels_fontsize=ticklabels_fontsize, fontsize=fontsize)

# Save figure
file_name = f"performance_summary_single_contingency"
save_figure(fig, file_name, save_fig=save_fig)
