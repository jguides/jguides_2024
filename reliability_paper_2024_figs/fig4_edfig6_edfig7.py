import copy
import os

import matplotlib
import matplotlib.pyplot as plt

# Import custom tables
analysis_dir = '/home/jguidera/Src/jguides_2024'
os.chdir(analysis_dir)
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity_ave import \
    FRDiffVecCosSimCovNnAveSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import PathWellPopSummBase
from src.jguides_2024.utils.df_helpers import df_pop
from src.jguides_2024.utils.plot_helpers import save_figure
from reliability_paper_2024_figs.population_reliability_plot_helpers import fr_vec_plot, _get_recording_set_names, \
    _update_boot_set_name, _get_remove_axis_empty_plot, _get_single_epochs_plot_params


# Additional functions

def _plot_legend(brain_region_vals, relationship_vals, save_fig=False, file_name=None):

    # Get map from brain region value to color
    colors_df = PathWellPopSummBase()._get_colors_df("brain_region_val", "relationship_val")

    # Initialize figure
    width_scale_factor = .53
    height_scale_factor = .2
    fig, ax = plt.subplots(
        figsize=(len(brain_region_vals) * width_scale_factor, len(relationship_vals) * height_scale_factor))

    # Plot
    for brain_region_val_idx, brain_region_val in enumerate(brain_region_vals):
        for relationship_val_idx, relationship_val in enumerate(relationship_vals[::-1]):
            color = df_pop(colors_df, {"brain_region_val": brain_region_val, "relationship_val": relationship_val},
                           "color")
            ax.add_patch(matplotlib.patches.Rectangle(
                [brain_region_val_idx, relationship_val_idx], 1, 1, facecolor=color))
    ax.set_xlim([0, len(brain_region_vals)])
    ax.set_ylim([0, len(relationship_vals)])
    ax.axis("off")

    # Save figure if indicated
    if file_name is None:
        file_name = "brain_region_relationship_colors_legend"
    save_figure(fig, file_name, save_fig=save_fig)


def _get_brain_region_vals(boot_set_name, dmPFC_OFC_only=None):

    # Define brain region values
    brain_region_vals = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]
    if dmPFC_OFC_only:
        brain_region_vals = ["mPFC_targeted", "OFC_targeted"]
    if "brain_region_diff" in boot_set_name:
        brain_region_vals = ["OFC_targeted_mPFC_targeted"]

    return brain_region_vals


def nn_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, single_epochs_plot, dmPFC_OFC_only, params):

    # Make copy of param to avoid changing outsisde of function
    params = copy.deepcopy(params)

    # Get quantities based on params
    subject_ids = params.pop("subject_ids", None)
    recording_set_names = _get_recording_set_names(rat_cohort, boot_set_name, dmPFC_OFC_only, subject_ids)
    boot_set_name = _update_boot_set_name(boot_set_name, rat_cohort)
    brain_region_vals = _get_brain_region_vals(boot_set_name, dmPFC_OFC_only)
    remove_axis_empty_plot = _get_remove_axis_empty_plot(brain_regions_separate)
    show_ave_conf, show_single_epoch = _get_single_epochs_plot_params(single_epochs_plot)

    # Define firing rate difference vector cosine similarity cov ave table param name
    if plot_type == "default":
        fr_diff_vec_cos_sim_wa_nn_ave_param_name = "1^0.25^mask_duration_10^potentially_rewarded_trial_stay_trial"
        fr_diff_vec_cos_sim_ppt_nn_ave_param_name = "1^0.05^mask_duration_10^potentially_rewarded_trial_stay_trial"
    elif plot_type == "correct_trial":
        fr_diff_vec_cos_sim_wa_nn_ave_param_name = "1^0.25^mask_duration_10^potentially_rewarded_trial_stay_trial"
        fr_diff_vec_cos_sim_ppt_nn_ave_param_name = "1^0.05^mask_duration_10^potentially_rewarded_trial_stay_trial"
    else:
        raise Exception(f"plot_type not recognized")

    # Update params
    table_names = ["FRDiffVecCosSimPptNnAveSumm", "FRDiffVecCosSimWANnAveSumm"]
    params.update({
        "table_names": table_names, "recording_set_names": recording_set_names, "boot_set_name": boot_set_name,
        "fr_diff_vec_cos_sim_wa_nn_ave_param_name": fr_diff_vec_cos_sim_wa_nn_ave_param_name,
        "fr_diff_vec_cos_sim_ppt_nn_ave_param_name": fr_diff_vec_cos_sim_ppt_nn_ave_param_name,
        "show_ave_conf": show_ave_conf,
        "show_single_epoch": show_single_epoch, "remove_axis_empty_plot": remove_axis_empty_plot, })

    # Plot
    keys = FRDiffVecCosSimCovNnAveSummBase._get_multiplot_params(**params)
    FRDiffVecCosSimCovNnAveSummBase().multiplot(recording_set_names, table_names, brain_regions_separate,
                                                brain_region_vals, keys, rat_cohort, **params)


# Trial average


# 1) Trial average, single rats (Figure 4, Extended Data Figure 6)

# Parameters
plot_type = "trial_average_raw"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = True
median = False
single_epochs_plot = False
dmPFC_OFC_only = False
# Table names and accompanying params
table_names = ["PathAveFRVecSumm", "TimeRelWAAveFRVecSumm"]
label_names = ["path", "end_well"]
relationship_vals_list = [None, None]
# Processing/saving
populate_tables = False
make_plot = True
save_fig = False

metric_names = ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

wspace = .25
yticklabels = None  # causes y tick labels to be placed on all plots
params = {"populate_tables": populate_tables, "save_fig": save_fig, "wspace": wspace, "yticklabels": yticklabels}

if make_plot:
    for metric_name, vector_type in zip(metric_names, vector_types):
        fr_vec_plot(
            plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
            metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# 2) Trial average, relationship ratio, rat cohort (Figure 4)

# Parameters
plot_type = "trial_average"
boot_set_name = "relationship_div"
rat_cohort = True
brain_regions_separate = True
median = True
single_epochs_plot = False
dmPFC_OFC_only = False
# Table names and accompanying params
table_names = ["PathAveFRVecSumm", "TimeRelWAAveFRVecSumm"]
label_names = ["path", "end_well"]
relationship_vals_list = [None, None]
# Processing/saving
populate_tables = False
make_plot = True
save_fig = False

metric_names = ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

wspace = .25
yticklabels = None  # causes y tick labels to be placed on all plots
params = {"populate_tables": populate_tables, "save_fig": save_fig, "wspace": wspace, "yticklabels": yticklabels}

if make_plot:
    for metric_name, vector_type in zip(metric_names, vector_types):
        fr_vec_plot(
            plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
            metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Single trial


# 3) Single trial, example rat, y zoom (Figure 4)

# Parameters
plot_type = "single_trial_rat1"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = False
median = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm", "TimeRelWAFRVecSTAveSumm"]
label_names = ["path", "path"]
relationship_vals_list = [None, ["same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = False
make_plot = True
save_fig = True

metric_names =  ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

params = {"populate_tables": populate_tables, "save_fig": save_fig, "subject_ids": ["J16"], "ylim": [-.1, .75]}

if make_plot:
    for metric_name, vector_type in zip(metric_names, vector_types):
        fr_vec_plot(
            plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
            metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# 4) Single trial, brain region difference, rat cohort (Figure 4)

# Parameters
plot_type = "single_trial"
boot_set_name = "brain_region_diff"
rat_cohort = True
brain_regions_separate = False
median = False
single_epochs_plot = False
dmPFC_OFC_only = False
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm", "TimeRelWAFRVecSTAveSumm"]
label_names = ["path", "path"]
relationship_vals_list = [None, ["same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = False
save_fig = False

metric_names =  ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

params = {"populate_tables": populate_tables, "save_fig": save_fig}

for metric_name, vector_type in zip(metric_names, vector_types):
    fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# 5) Single trial, single rats (Extended Data Figure 7)

# Parameters
plot_type = "single_trial"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = False
median = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm", "TimeRelWAFRVecSTAveSumm"]
label_names = ["path", "path"]
relationship_vals_list = [None, ["same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = False
make_plot = True
save_fig = False

metric_names = ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

params = {"populate_tables": populate_tables, "save_fig": save_fig}

if make_plot:
    for metric_name, vector_type in zip(metric_names, vector_types):
        fr_vec_plot(
            plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
            metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# 6) Single trial, single rats, single epochs (Extended Data Figure 7)

# Parameters
plot_type = "single_trial"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = False
median = False
single_epochs_plot = True
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm", "TimeRelWAFRVecSTAveSumm"]
label_names = ["path", "path"]
relationship_vals_list = [None, ["same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = False
make_plot = True
save_fig = False

metric_names = ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

params = {"populate_tables": populate_tables, "save_fig": save_fig}

if make_plot:
    for metric_name, vector_type in zip(metric_names, vector_types):
        fr_vec_plot(
            plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
            metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Nearest neighbor cosine similarity

# 7) Single rats (Extended Data Figure 7)

# Parameters
plot_type = "default"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Processing/saving
populate_tables = False
make_plot = True
save_fig = False

params = {"populate_tables": populate_tables, "save_fig": save_fig, "mega_column_gap_factor": .04, "ylim": [0, .75], "yticks": [0, .5],
          "yticklabels": None}

# Plot
if make_plot:
    nn_plot(plot_type, boot_set_name, rat_cohort, brain_regions_separate, single_epochs_plot, dmPFC_OFC_only, params)


# 8) Single rats, single epochs

# Parameters
plot_type = "default"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = False
single_epochs_plot = True
dmPFC_OFC_only = True
# Processing/saving
populate_tables = False
make_plot = True
save_fig = False

params = {"populate_tables": populate_tables, "save_fig": save_fig, "mega_column_gap_factor": .04, "ylim": [0, .75],
          "yticks": [0, .5], "yticklabels": None}

# Plot
if make_plot:
    nn_plot(plot_type, boot_set_name, rat_cohort, brain_regions_separate, single_epochs_plot, dmPFC_OFC_only, params)


# Levels of generality legends

# Params
make_plot = False

if make_plot:

    # Path
    brain_region_vals = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]
    relationship_vals = [
        "same_path_even_odd_trials", "same_turn_even_odd_trials", "different_turn_well_even_odd_trials"]
    fig = _plot_legend(brain_region_vals, relationship_vals, True, "brain_region_relationship_colors_legend_path")

    # Reward well
    brain_region_vals = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]
    relationship_vals = ["same_end_well_even_odd_stay_trials", "different_end_well_even_odd_stay_trials"]
    fig = _plot_legend(brain_region_vals, relationship_vals, True, "brain_region_relationship_colors_legend_well")

    # Path relationship ratio
    brain_region_vals = ["mPFC_targeted"]
    relationship_vals = ["same_turn_even_odd_trials_same_path_even_odd_trials",
                         "different_turn_well_even_odd_trials_same_path_even_odd_trials"]
    _plot_legend(brain_region_vals, relationship_vals, True, "brain_region_relationship_colors_legend_path_div")

    # Reward well relationship ratio
    brain_region_vals = ["mPFC_targeted"]
    relationship_vals = ["different_end_well_even_odd_stay_trials_same_end_well_even_odd_stay_trials"]
    _plot_legend(brain_region_vals, relationship_vals, True, "brain_region_relationship_colors_legend_well_div")


# Extra

# Trial average, relationship ratio, single rats

# Parameters
plot_type = "trial_average"
boot_set_name = "relationship_div"
rat_cohort = False
brain_regions_separate = True
median = True
single_epochs_plot = False
dmPFC_OFC_only = False
# Table names and accompanying params
table_names = ["PathAveFRVecSumm", "TimeRelWAAveFRVecSumm"]
label_names = ["path", "end_well"]
relationship_vals_list = [None, None]
# Processing/saving
populate_tables = False
make_plot = False
save_fig = False

metric_names = ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

wspace = .25
yticklabels = None  # causes y tick labels to be placed on all plots
params = {"populate_tables": populate_tables, "save_fig": save_fig, "wspace": wspace, "yticklabels": yticklabels}

if make_plot:
    for metric_name, vector_type in zip(metric_names, vector_types):
        fr_vec_plot(
            plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
            metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Single trial, brain region difference, single rats

# Parameters
plot_type = "single_trial"
boot_set_name = "brain_region_diff"
rat_cohort = False
brain_regions_separate = False
median = False
single_epochs_plot = False
dmPFC_OFC_only = False
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm", "TimeRelWAFRVecSTAveSumm"]
label_names = ["path", "path"]
relationship_vals_list = [None, ["same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = False
make_plot = False
save_fig = False

metric_names =  ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]

params = {"populate_tables": populate_tables, "save_fig": save_fig}

if make_plot:
    for metric_name, vector_type in zip(metric_names, vector_types):
        fr_vec_plot(
            plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
            metric_name, vector_type, table_names, label_names, relationship_vals_list, params)

