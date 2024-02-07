import os
import pprint
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

# Import custom datajoint tables
os.chdir("/home/jguidera/Src/jguides_2024/")
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_epochs_id
from src.jguides_2024.utils.df_helpers import df_from_data_list, df_pop
from src.jguides_2024.embedding.embedding_params import EmbeddingParams
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector_embedding import FRVecEmb, FRVecEmbParams, FRVecEmbSel
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams
from src.jguides_2024.spikes.jguidera_unit import EpsUnitsParams, BrainRegionUnitsParams, BrainRegionUnits
from src.jguides_2024.utils.plot_helpers import get_gridspec_ax_maps, get_plot_idx_map, save_figure
from src.jguides_2024.utils.vector_helpers import unpack_single_element

# Define parameters
curation_set_name = "runs_analysis_v1"
brain_regions = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]
brain_region_cohort_name = "all_targeted"
min_epoch_mean_firing_rate = .1
kernel_sd = .1
zscore_fr = False
n_neighbors = 15
n_components = 3
min_num_units = 50
populate_tables = False
# Plot layout params
subplot_width = 2
subplot_height = 2
# Define alpha and s
# For task progression plots:
s_task_progression = subplot_width * 1.5
# For path context plots:
alpha_path_context = .09
s_path_context = subplot_width * 2
PlotParams = namedtuple("PlotParams", "plot_fn_names kwargs_list time_bin_shorthands")
wa_exclusion_params = {"exclusion_type": "leave_trial"}
# Save figure
save_fig = False

# Store parameters
plot_params_map = {
    "task_period": PlotParams(
        ["plot_by_ppt", "plot_by_relative_time_in_delay", "plot_by_relative_time_at_well_post_delay"],
        [{"s": s_task_progression}, {"s": s_task_progression, "exclusion_params": wa_exclusion_params},
         {"s": s_task_progression}],
        ["epoch_100ms", "epoch_100ms", "epoch_100ms"]),
    "context": PlotParams(
        ["plot_by_path", "plot_by_path", "plot_by_path"],
        [{"task_period": "path", "alpha": alpha_path_context, "s": s_path_context},
         {"task_period": "delay", "alpha": alpha_path_context, "s": s_path_context,
          "exclusion_params": wa_exclusion_params},
         {"task_period": "well_post_delay", "alpha": alpha_path_context, "s": s_path_context}],
        ["epoch_100ms", "epoch_100ms", "epoch_100ms"]),
}

# Define plot layout parameters
# Plot across epochs (rows), task periods (columns), and brain regions (mega columns)
plot_components = [str(x) for x in [0, 1, 2]]  # embedding dimensions
mega_column_iterables = list(plot_params_map.keys())
row_iterables = np.arange(0, np.max(
    [len(x.plot_fn_names) for x in plot_params_map.values()]))  # correspond to coloring schemes
mega_row_gap_factor = 0
mega_column_gap_factor = 0
wspace = 0
hspace = 0
# Axis
axis_off = True

# Define nwb_file_name, epochs for paper 1 embedding figures
column_names = ["subject_id", "nwb_file_name", "epochs", "task_description"]
subject_df = df_from_data_list([
    ("J16", "J1620210606_.nwb", [8], "centerAlternation"),
    ("J16", "J1620210606_.nwb", [6], "centerAlternation_env2"),
    ("J16", "J1620210606_.nwb", [10], "handleAlternation"),
    ("mango", "mango20211207_.nwb", [6], "centerAlternation"),
    ("june", "june20220420_.nwb", [10], "centerAlternation"),
    ("peanut", "peanut20201108_.nwb", [8], "centerAlternation"),
    ("fig", "fig20211109_.nwb", [8], "centerAlternation"),
], column_names)

# Get quantities
res_epoch_spikes_sm_param_name = ResEpochSpikesSmParams().lookup_param_name([kernel_sd])
fr_vec_emb_param_name = FRVecEmbParams().lookup_param_name([n_neighbors, n_components])

for _, subject_df_subset in subject_df.iterrows():
    # Get quantities
    epochs = subject_df_subset.epochs
    nwb_file_name = subject_df_subset.nwb_file_name
    epochs_id = get_epochs_id(epochs)
    epochs_description = (EpochsDescription & {"nwb_file_name": nwb_file_name, "epochs_id": epochs_id}).fetch1(
        "epochs_description")

    mega_row_iterables = epochs

    # Define brain region units param name based on epochs
    # If one epoch, use units from that epoch
    if len(epochs) == 1:
        brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(
            nwb_file_name, int(unpack_single_element(epochs)), min_epoch_mean_firing_rate)
    # Otherwise, use all units active across run epochs in the day
    else:
        brain_region_units_param_name = BrainRegionUnitsParams().lookup_param_name(
            [EpsUnitsParams().lookup_param_name([min_epoch_mean_firing_rate]), "runs", "all", None, None])

    # Make key to FRVecEmb table
    key = {"nwb_file_name": nwb_file_name, "epochs_id": epochs_id,
           "brain_region_units_param_name": brain_region_units_param_name,
           "res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name,
           "fr_vec_emb_param_name": fr_vec_emb_param_name, "zscore_fr": zscore_fr}

    # Get subset of user defined brain regions that are in brain region cohort
    brain_regions_ = [x for x in brain_regions if x in (
            BrainRegionCohort & {"nwb_file_name": nwb_file_name,
                                 "brain_region_cohort_name": brain_region_cohort_name}).fetch1("brain_regions")]
    # Update brain regions to include only those with at least a certain number of units
    curation_set_df = (CurationSet & key).fetch1_dataframe()
    brain_regions_ = [x for x in brain_regions_ if
                      BrainRegionUnits().get_num_units(
                          {**key, **{"brain_region": x, "curation_name": df_pop(curation_set_df, {
                              "brain_region": x, "epochs_description": epochs_description},
                                                                                "curation_name")}}) >= min_num_units]

    column_iterables = brain_regions_

    # Populate tables if indicated
    if populate_tables:

        # Loop through time bins settings
        time_bin_shorthands = np.unique(np.concatenate([x.time_bin_shorthands for x in plot_params_map.values()]))
        for time_bin_shorthand in time_bin_shorthands:

            # Insert into embedding selection table
            # ...get time bins pool param names
            res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand(
                time_bin_shorthand)
            res_time_bins_pool_param_names = (ResTimeBinsPoolCohortParams & {
                "nwb_file_name": nwb_file_name, "epochs_id": epochs_id,
                "res_time_bins_pool_cohort_param_name": res_time_bins_pool_cohort_param_name}).fetch1(
                "res_time_bins_pool_param_names")
            # ...insert into selection table
            FRVecEmbSel().insert_epochs(
                nwb_file_name, epochs, res_time_bins_pool_param_names, brain_region_units_param_name,
                fr_vec_emb_param_name=fr_vec_emb_param_name, curation_set_name=curation_set_name)

            # Populate
            curation_set_map = (CurationSet & key).fetch1_dataframe()
            for brain_region in brain_regions_:
                curation_name = df_pop(curation_set_map,
                                       {"brain_region": brain_region, "epochs_description": epochs_description},
                                       "curation_name")
                key.update({"curation_name": curation_name, "brain_region": brain_region})
                FRVecEmb().populate(key)

    # Plot
    add_subplot_args = dict()
    if len(plot_components) == 3:
        add_subplot_args.update({"projection": "3d"})

    # Set background to dark
    with plt.style.context("dark_background"):

        # Initialize figure
        gs_map, ax_map, fig = get_gridspec_ax_maps(
            mega_row_iterables, mega_column_iterables, row_iterables, column_iterables, subplot_width=subplot_width,
            subplot_height=subplot_height, mega_row_gap_factor=mega_row_gap_factor,
            mega_column_gap_factor=mega_column_gap_factor,
            wspace=wspace, hspace=hspace, add_subplot_args=add_subplot_args)
        plot_idx_map = get_plot_idx_map(mega_row_iterables, mega_column_iterables, row_iterables, column_iterables)

        for coloring_scheme, params_obj in plot_params_map.items():
            for target_epoch in epochs:
                for row_idx in row_iterables:
                    for brain_region in brain_regions_:
                        # Print keys with different brain regions so can store in reverse axes params if desired
                        if row_idx == 0:
                            pprint.pprint(key)
                        epochs_description = EpochsDescription().get_single_run_description(nwb_file_name, target_epoch)
                        curation_name = df_pop((CurationSet & key).fetch1_dataframe(), {
                            "brain_region": brain_region, "epochs_description": epochs_description}, "curation_name")
                        key.update(
                            {"brain_region": brain_region, "curation_name": curation_name,
                             "res_time_bins_pool_cohort_param_name":
                                 ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand(
                                     params_obj.time_bin_shorthands[row_idx])})

                        # Get axis
                        ax = ax_map[(target_epoch, coloring_scheme, row_idx, brain_region)]

                        # Plot
                        plot_fn_name = params_obj.plot_fn_names[row_idx]
                        kwargs = dict()
                        if params_obj.kwargs_list is not None:
                            kwargs = params_obj.kwargs_list[row_idx]

                        # Apply embeddding params
                        kwargs = EmbeddingParams().apply_embedding_params(key, kwargs)

                        # Add information about whether to keep or remove axis
                        kwargs["axis_off"] = axis_off

                        # Plot embedding
                        getattr(FRVecEmb & key, plot_fn_name)(target_epoch, ax=ax, plot_components=plot_components,
                                                              **kwargs)

                        # Remove reverse axes param
                        if "reverse_axes" in kwargs:  # remove reverse axes param
                            del kwargs["reverse_axes"]

                        # Apply zoom (should come after plotting)
                        kwargs = EmbeddingParams().apply_embedding_params(key, kwargs, ax, ["zoom"])

        # Save figure
        file_name = f"embedding_{format_nwb_file_name(key['nwb_file_name'])}_eps{key['epochs_id']}_sm" + \
                    f"{key['res_epoch_spikes_sm_param_name']}_{brain_region_units_param_name}_{fr_vec_emb_param_name}"
        save_figure(fig, file_name, save_fig=save_fig, dpi=800, figure_type=".png")