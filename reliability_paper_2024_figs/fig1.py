import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Rectangle

# Import custom datajoint tables
os.chdir("/home/jguidera/Src/jguides_2024/")
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellArrivalTrials, DioWellArrivalTrialsParams, DioWellDDTrials
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsParams, BrainRegionUnits
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.utils.plot_helpers import get_figsize, plot_spanning_line, save_figure, format_ax
from src.jguides_2024.spikes.plot_spike_times import plot_spike_times
from src.jguides_2024.datajoint_nwb_utils.datajoint_fr_table_wrappers import order_units_by_trial_segments_fr
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionColor
from src.jguides_2024.utils.vector_helpers import expand_interval
from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import format_brain_region, get_nwb_file_name_epoch_text
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_unit_name
from src.jguides_2024.spikes.jguidera_spikes import EpochSpikeTimesRelabel


# Plot trial raster with units from each brain region

# *** PARAMETERS ***
nwb_file_name = "J1620210606_.nwb"
epoch = 2
brain_region_cohort_name = "all_targeted"
# Order units
order_units_by = "trial_segments_fr"
df_names = ["FrmapPuptSm", "FrmapUniqueWellArrivalSm", "FrmapWADSmWT"]
time_bin_width = .1  # used to order units by peak path firing rate
min_epoch_mean_firing_rate = .1  # minimum epoch average firing rate threshold
# Trial
path_name = MazePathWell().get_rewarded_path_names(nwb_file_name, epoch)[2]  # plot a trial along this path
well_name = "_".join(path_name.split("_")[-2:])
path_trial_num = 5
# Bars to denote task period
use_cmaps = False
task_period_colors_map = {"path traversal": "lightsteelblue", "delay": "lightgray", "reward": "wheat"}
# Raster
alpha = 1
linewidths = None
# Axis
time_relative_to_trial_start = True
subplot_width, subplot_height = 5, 1.5
wspace = .25
hspace = 0
height_ratios = [1, 10]
tick_width = 3  # width of axis ticks
fontsize = 26
label_fontsize = fontsize * 1
title_fontsize = fontsize * 1.2
save_fig = False
# ******************

# Hard code
dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_delay_param_name()
position_info_param_name = "default"
rectangle_height = 2  # height of rectangles that denote task period

# Get brain region units param name
brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(nwb_file_name, epoch,
                                                                                        min_epoch_mean_firing_rate)

# Define key for querying tables
key = {"nwb_file_name": nwb_file_name, "epoch": epoch, "brain_region_cohort_name": brain_region_cohort_name,
       "brain_region_units_param_name": brain_region_units_param_name,
       "dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name,
       "position_info_param_name": position_info_param_name}

# Define brain regions
brain_regions_ = (BrainRegionCohort & key).fetch1("brain_regions")
# order the brain regions
ordered_brain_regions = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]
brain_regions = [x for x in ordered_brain_regions if x in brain_regions_]
if len(brain_regions) != len(brain_regions_):
    raise Exception(f"brain regions from BrainRegionCohort did not match ordered_brain_regions")

# Get plot x ranges
trials_map = (DioWellDDTrials & key).trial_intervals(trial_feature_map={"path_names": [path_name]}, as_dict=True)
trial_num = list(trials_map.keys())[path_trial_num]
plot_x_range = trials_map[trial_num]

# Initialzie plot
num_rows = 2
num_columns = len(brain_regions)
figsize = get_figsize(2, num_columns, subplot_width, subplot_height)
fig = plt.figure(figsize=figsize)
# Initialize axes using gridspec so can place subplots very close together
gs = gridspec.GridSpec(num_rows, num_columns, wspace=wspace, hspace=hspace, height_ratios=height_ratios)
ax_map = {k: [fig.add_subplot(gs[row_idx, column_idx]) for column_idx in np.arange(0, num_columns)]
          for row_idx, k in enumerate(["task_progression", "raster"])}
raster_fig_axes_lists = [(fig, ax_map["raster"])]
task_fig_axes_lists = [(fig, ax_map["task_progression"])]

# Get unit order
unit_order_path_names = [path_name]
unit_order = order_units_by_trial_segments_fr(
    nwb_file_name, epoch, df_names, well_name, path_name, dio_well_arrival_trials_param_name)
unit_order = [get_unit_name(*x) for x in unit_order]  # convert to unit name

# Get colors for brain regions
color_map = BrainRegionColor().return_brain_region_color_map(brain_regions)

# Plot rasters
global_xlims = np.asarray(plot_x_range) - plot_x_range[0]
for ax, brain_region in zip(ax_map["raster"], brain_regions):
    key.update({"brain_region": brain_region})
    sort_group_unit_ids_map = (BrainRegionUnits & key).fetch1("sort_group_unit_ids_map")
    spike_times_df = EpochSpikeTimesRelabel.RelabelEntries().spike_times_across_sort_groups(key,
                                                                                            sort_group_unit_ids_map)
    spike_times_list = spike_times_df.loc[[x for x in unit_order if x in spike_times_df.index]].epoch_spike_times.values
    # Convert spike times to time in trial
    spike_times_list = [x - plot_x_range[0] for x in spike_times_list]
    plot_spike_times(spike_times_list, ax=ax, colors=color_map[brain_region], xlims=global_xlims, alpha=alpha,
                     linewidths=linewidths)

# Plot shaded bars to denote task periods
# get delay period interval
wa_trial_interval = (DioWellArrivalTrials & key).trial_intervals(as_dict=True)[trial_num]
boundaries = np.asarray([plot_x_range[0]] + list(wa_trial_interval) + [plot_x_range[1]]) - plot_x_range[
    0]  # time relative to interval start
for ax in ax_map["task_progression"]:
    for (task_period, color), (x_start, width) in zip(task_period_colors_map.items(),
                                                      zip(boundaries, np.diff(boundaries))):
        ax.add_patch(Rectangle((x_start, 0), width, rectangle_height, color=color))
    # Remove axis
    ax.axis("off")
    # Set x lims
    ax.set_xlim(global_xlims)

# Mark task periods with vertical lines in raster
boundaries = np.asarray(wa_trial_interval) - plot_x_range[0]
for ax in ax_map["raster"]:
    span_data = expand_interval(ax.get_ylim(), expand_factor=-.02)
    for boundary in boundaries:
        plot_spanning_line(ax=ax, span_axis="y", span_data=span_data, constant_val=boundary, zorder=0, color="gray",
                           linewidth=3, alpha=.3)
    # Set x lims
    ax.set_xlim(global_xlims)

# Add title with brain region
for idx, (ax, brain_region) in enumerate(zip(ax_map["task_progression"], brain_regions)):
    xlabel, ylabel = "", ""
    if idx == 0:
        xlabel = "Time (s)"
        ylabel = "Unit"
    format_ax(ax=ax, title=format_brain_region(brain_region), xlabel=xlabel, ylabel=ylabel, fontsize=title_fontsize)

# Format raster plot axis
for idx, ax in enumerate(ax_map["raster"]):
    xlabel, ylabel = "", ""
    if idx == 0:
        xlabel = "Time (s)"
        ylabel = "Unit"
    format_ax(ax=ax, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

# Save figure
file_name = f"raster_full_trial_{get_nwb_file_name_epoch_text(nwb_file_name, epoch)}_trial{trial_num}"
save_figure(fig, file_name, save_fig=save_fig)