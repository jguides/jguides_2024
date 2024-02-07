import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import custom datajoint tables
analysis_dir = '/home/jguidera/Src/jguides_2024'
os.chdir(analysis_dir)
from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_thesis_nwb_file_names, format_brain_region, \
    get_subject_id_shorthand, get_subject_id
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionColor, BrainRegionCohort
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsParams
from src.jguides_2024.utils.df_helpers import df_from_data_list, df_filter_columns
from src.jguides_2024.utils.hierarchical_bootstrap import hierarchical_bootstrap
from src.jguides_2024.utils.plot_helpers import format_ax, save_figure
from src.jguides_2024.utils.stats_helpers import random_sample
from src.jguides_2024.utils.pca_wrappers import PCAContainer


# Define parameters
nwb_file_names = get_thesis_nwb_file_names()
zscore_fr = False
curation_set_name = "hpc_runs_ctx_runs_sleeps"
brain_region_cohort_name = "all_targeted"
res_epoch_spikes_sm_param_name = "0.1"
min_epoch_mean_firing_rate = .1
cumulative_variance_explained_threshold = .8

# Hard code
res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms")

# Define key for querying tables
key = {"zscore_fr": zscore_fr, "res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name,
      "res_time_bins_pool_param_name": res_time_bins_pool_param_name, "curation_set_name": curation_set_name,
      "brain_region_cohort_name":brain_region_cohort_name}


# Plot first two PCs of PCA of firing rate vectors from each brain region

# Parameters
nwb_file_name = "J1620210606_.nwb"
epoch = 8
save_fig = False

# Define key for querying table
key.update({"nwb_file_name": nwb_file_name, "epoch": epoch})
key["brain_region_units_param_name"] = BrainRegionUnitsParams().lookup_single_epoch_param_name(
    nwb_file_name, epoch, min_epoch_mean_firing_rate=min_epoch_mean_firing_rate)

# Define brain regions
brain_regions = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]

# Initialize figure
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

# Loop through brain regions and plot
for idx, (brain_region, ax) in enumerate(zip(brain_regions, axes)):

    key.update({"brain_region": brain_region})

    # Get firing rate data
    if len(FRVec & key) == 0:
        raise Exception(f"No entry found in FRVec for {key}")
    df_concat, time_vector = (FRVec & key).firing_rate_across_sort_groups()
    spikes_df = pd.DataFrame(np.vstack(df_concat["firing_rate"]).T, columns=df_concat.index, index=time_vector).dropna(
        axis=1)

    # Get PCA container
    pca_container = PCAContainer(all_features_df=spikes_df)

    # Get color for brain region
    color = BrainRegionColor().get_brain_region_color(brain_region)

    # Plot
    pca_container.plot_pca_output_2D(fig_ax_list=[fig, ax], color=color, alpha=.2)

    # Format axis
    xlabel, ylabel = "", ""
    if idx == 0:
        xlabel = "PC 1"
        ylabel = "PC 2"
    format_ax(ax=ax, xlabel=xlabel, ylabel=ylabel)

    # Print number of PCs and fraction of variance captured by first 2 PCs
    var_expl = np.cumsum(pca_container.pca_obj.explained_variance_ratio_)[1]
    print(f"{brain_region} total PCs: {np.shape(pca_container.input_array)[1]}. "
          f"Variance explained by first 2 PCs: {var_expl}")

# Save figure
file_name_save = f"pca_{format_nwb_file_name(nwb_file_name)}_ep{epoch}"
save_figure(fig, file_name_save, save_fig=save_fig)


# Get quantities related to cumulative variance explained by principal components
# in a PCA of firing rate vectors

print("Making cum_var_exp_df...")

data_list = []
# Loop through nwb file names
for nwb_file_name in nwb_file_names:
    key.update({"nwb_file_name": nwb_file_name})
    # Loop through single contingency run epochs
    for epoch in RunEpoch().get_single_contingency_epochs(nwb_file_name):
        key.update({"epoch": epoch})

        # Get brain regions
        key["brain_region_units_param_name"] = BrainRegionUnitsParams().lookup_single_epoch_param_name(
            nwb_file_name, epoch, min_epoch_mean_firing_rate=min_epoch_mean_firing_rate)
        brain_regions = (BrainRegionCohort & key).fetch1("brain_regions")

        # Find cumulative variance explained and where it exceeds predefined threshold across brain regions
        for brain_region in brain_regions:

            key.update({"brain_region": brain_region})

            # Get firing rate data
            if len(FRVec & key) == 0:
                raise Exception(f"No entry found in FRVec for {key}")
            df_concat, time_vector = (FRVec & key).firing_rate_across_sort_groups()
            spikes_df = pd.DataFrame(np.vstack(df_concat["firing_rate"]).T, columns=df_concat.index,
                                     index=time_vector).dropna(axis=1)

            # Get PCA container
            pca_container = PCAContainer(all_features_df=spikes_df)

            # Get cumulative variance explained
            cumulative_variance_explained = np.cumsum(pca_container.pca_obj.explained_variance_ratio_)

            # Get where cumulative variance explained threshold reached or exceeded, and the cumulative variance
            # explained at this point
            # ...component number
            component_num = pca_container.get_first_component_above_explained_variance_ratio(
                cumulative_variance_explained_threshold)
            # ...store total number of components
            num_components = pca_container.pca_obj.n_components
            # ...express number of components as fraction of components
            fraction_components = component_num / num_components
            above_thresh_cum_expl_var_ratio = pca_container.get_cumulative_explained_variance_ratio()[component_num]
            data_list.append((nwb_file_name, epoch, brain_region, cumulative_variance_explained, component_num,
                              num_components, fraction_components,
                              above_thresh_cum_expl_var_ratio))

# Store in dataframe
cum_var_exp_df = df_from_data_list(data_list,
                                   ["nwb_file_name", "epoch", "brain_region", "cumulative_variance_explained",
                                    "component_num", "num_components",
                                    "fraction_components", "above_thresh_cum_expl_var_ratio"])


# Hierarchical bootstrap
resample_levels = ["nwb_file_name", "epoch"]
resample_quantity = "fraction_components"
ave_group_column_names = ["brain_region"]
ave_diff_group_column_names = []
ave_diff_column_name = "brain_region"
num_bootstrap_samples = 1000
average_fn = np.mean

print("\nGetting bootstrap_dfs...")

bootstrap_dfs = hierarchical_bootstrap(
    cum_var_exp_df, resample_levels, resample_quantity, ave_group_column_names, ave_diff_group_column_names,
    ave_diff_column_name, num_bootstrap_samples, average_fn)


# Get number of principal components required to explain at least 80% variance in each brain region / epoch,
# subsampling units, proceeding from a minimum size in increments of one up to size of recorded population
min_units = 5  # smallest number of units to subsample

print(f"Getting cum_var_exp_subsample_df...")

data_list = []
# Loop through nwb file names
for nwb_file_name in nwb_file_names:
    key.update({"nwb_file_name": nwb_file_name})
    print(f"on {nwb_file_name}...")
    # Loop through single contingency run epochs
    for epoch in RunEpoch().get_single_contingency_epochs(nwb_file_name)[1:2]:
        key.update({"epoch": epoch})
        print(f"on epoch {epoch}...")
        # Get brain regions
        key["brain_region_units_param_name"] = BrainRegionUnitsParams().lookup_single_epoch_param_name(
            nwb_file_name, epoch, min_epoch_mean_firing_rate=min_epoch_mean_firing_rate)
        brain_regions = (BrainRegionCohort & key).fetch1("brain_regions")

        # Find cumulative variance explained and where it exceeds predefined threshold across brain regions
        for brain_region in brain_regions:
            key.update({"brain_region": brain_region})
            print(f"on {brain_region}...")
            # Get firing rate data
            if len(FRVec & key) == 0:
                print(f"Warning! no entry found in FRVec for {key}. Continuing...")
                continue
            df_concat, time_vector = (FRVec & key).firing_rate_across_sort_groups()
            spikes_df = pd.DataFrame(
                np.vstack(df_concat["firing_rate"]).T, columns=df_concat.index, index=time_vector).dropna(axis=1)
            # Take random sample
            for n in np.arange(min_units, len(spikes_df.columns) + 1):
                sample_column_names = random_sample(spikes_df.columns, n, replace=False)
                # Get PCA container
                pca_container = PCAContainer(all_features_df=spikes_df[sample_column_names])
                # Get cumulative variance explained
                cumulative_variance_explained = np.cumsum(pca_container.pca_obj.explained_variance_ratio_)
                # Get where cumulative variance explained threshold reached or exceeded, and the
                # cumulative variance explained at this point
                component_num = pca_container.get_first_component_above_explained_variance_ratio(
                    cumulative_variance_explained_threshold)  # component number
                num_components = pca_container.pca_obj.n_components  # store total number of components
                data_list.append((nwb_file_name, epoch, brain_region, component_num, num_components))
cum_var_exp_subsample_df = df_from_data_list(data_list, [
    "nwb_file_name", "epoch", "brain_region", "component_num", "num_components"])


# Example cumulative explained variance ratio plot

# Define parameters
ex_cum_exp_var_ratio_nwb_file_name = nwb_file_names[0]
ex_cum_exp_var_ratio_epoch = 2
# Examples of relationship between num units and num PCs to explain at least 80% variance in single
# behavioral sessions
num_examples_per_row = 3
examples = [(x, 4) for x in nwb_file_names]
# Overall figure dimensions
fig_width = 8
fig_height = len(examples) * 1.4
# Save figure
save_fig = False

# Hard code
ylim = [0, 1]
spine_width = 1.5
tick_width = 1.5
fontsize0 = 22
fontsize1 = 15
fontsize2 = 15
fontsize3 = 14
brain_regions = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]

# Initialize main figure
main_fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
# Initialize global figures
left_right_subfigs = main_fig.subfigures(1, 2, width_ratios=[1, 1.5], wspace=.1)

# Plot example cumulative explained variance ratio

# Get df subset
cum_var_exp_df_subset = df_filter_columns(cum_var_exp_df, {
    "nwb_file_name": ex_cum_exp_var_ratio_nwb_file_name, "epoch": ex_cum_exp_var_ratio_epoch}).set_index("brain_region")

# Initialize counter for plots
plot_counter = 0

# Loop through brain regions and plot
for brain_region_idx, brain_region in enumerate(brain_regions):
    df_subset = cum_var_exp_df_subset.loc[brain_region]
    ax = left_right_subfigs[0].add_subplot(len(brain_regions), 1, brain_region_idx + 1)
    plot_counter += 1
    # Plot variance explained as a function of number of components
    plot_x = np.arange(1, len(df_subset.cumulative_variance_explained) + 1)
    ax.plot(plot_x, df_subset.cumulative_variance_explained,
            color=BrainRegionColor().get_brain_region_color(brain_region))
    # Mark where cumulative variance explained threshold reached or exceeded
    ax.plot([df_subset.component_num] * 2, ylim, color="black")
    ax.plot([plot_x[0], plot_x[-1]], [df_subset.above_thresh_cum_expl_var_ratio] * 2, color="black", linestyle="--")

    # Format axis
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    xticks = [x for x in plot_x if x in [20, 50, 150]]
    xlabel, ylabel = None, None
    yticklabels = yticks
    if brain_region_idx == 0:
        xlabel = "Number PCs"
        ylabel = "Cumulative\nvariance explained"
    format_ax(ax=ax, xlabel=xlabel, ylabel=ylabel, ylim=ylim, xlim=[np.min(plot_x), np.max(plot_x)], xticks=xticks,
              yticks=yticks,
              yticklabels=yticklabels, spine_width=spine_width, tick_width=tick_width, spines_off_list=[],
              fontsize=fontsize1,
              ticklabels_fontsize=fontsize2)

    # Plot cumulative variance explained as a function of fraction of components
    ax2 = ax.twiny()
    fraction_components = np.arange(1, len(df_subset.cumulative_variance_explained) + 1) / len(
        df_subset.cumulative_variance_explained)
    plot_x = fraction_components
    color = BrainRegionColor().get_brain_region_color(brain_region)
    ax2.plot(plot_x, df_subset.cumulative_variance_explained, color=color, zorder=10,
             linewidth=2)
    xlabel = None
    # title
    title = format_brain_region(brain_region)
    if brain_region_idx == 0:
        xlabel = "Fraction PCs"
    format_ax(ax=ax2, xlabel=xlabel, xlim=[np.min(plot_x), np.max(plot_x)], ylim=ylim, spines_off_list=[],
              spine_width=spine_width,
              tick_width=tick_width, fontsize=fontsize1, ticklabels_fontsize=fontsize3)
    ax2.set_title(title, fontsize=fontsize0, color=color)

# Make map from number of brain regions available for rat and which column
# gets title
title_position_map = {3: 1, 2: 1, 1: 0}

# Plot subsampling examples

# Define axes
axes = left_right_subfigs[1].subplots(len(examples), len(brain_regions))

# Loop through examples
for idx, (nwb_file_name, epoch) in enumerate(examples):

    # Initialize counter for plots, which we use to set plot column number
    plot_counter = 0
    no_plot_counter = 0

    # Define number of brain regions for this subject
    num_brain_regions = len(
        np.unique(df_filter_columns(cum_var_exp_subsample_df, {"nwb_file_name": nwb_file_name}).brain_region))

    # Loop through brain regions
    for brain_region_idx, brain_region in enumerate(brain_regions):
        df_subset = df_filter_columns(cum_var_exp_subsample_df,
                                      {"brain_region": brain_region, "nwb_file_name": nwb_file_name})

        # Continue if brain region not available
        if len(df_subset) == 0:
            axes[idx, len(brain_regions) - 1 - no_plot_counter].axis("off")
            no_plot_counter += 1
            continue

        # Otherwise plot
        ax = axes[idx, plot_counter]  # get axis
        color = BrainRegionColor().get_brain_region_color(brain_region)
        for _, df_row in df_subset.iterrows():
            ax.plot(df_row.num_components, df_row.component_num, ".", color=color, alpha=.6)

        # Format axis
        ax.set_xlim([0, np.nanmax(df_subset.num_components) * 1.1])
        xlabel, ylabel, title = None, None, None
        if brain_region_idx == 0 and idx == 0:
            xlabel = "Number units"
            ylabel = "Number PCs >=\n80% var exp"
        if plot_counter == title_position_map[num_brain_regions]:
            title = get_subject_id_shorthand(get_subject_id(nwb_file_name))
        format_ax(ax=ax, xlabel=xlabel, ylabel=ylabel, title=title, fontsize=14)
        plot_counter += 1

# Save figure
nwb_file_names_text = "_".join([format_nwb_file_name(x) for x in nwb_file_names])
file_name_save = f"pca_summary_single_contingency_{nwb_file_names_text}"
save_figure(main_fig, file_name_save, save_fig=save_fig)