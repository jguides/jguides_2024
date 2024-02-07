import os

# Import custom datajoint tables
analysis_dir = '/home/jguidera/Src/jguides_2024'
os.chdir(analysis_dir)
from reliability_paper_2024_figs.population_reliability_plot_helpers import _add_medium_sized_plot_params, \
    _add_outbound_path_x_params, fr_vec_plot, _add_medium_long_sized_plot_params


# Path traversal

# Single trial, same outbound path different outbound path, correct time_and_trials,
# example rat with y lim zoom

# Parameters
plot_type = "same_different_outbound_path_correct_rat2"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = True
median = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm"]
label_names = ["path"]
relationship_vals_list = [["outbound_correct_correct_trials", "same_path_outbound_correct_correct_trials"]]
# Processing/saving
populate_tables = False
save_fig = False

metric_names = ["euclidean_distance", "cosine_similarity"]
vector_types = ["vec", "diff_vec"]
params = {"populate_tables": populate_tables, "save_fig": save_fig, "path_fr_vec_param_name":
          "correct_incorrect_trials", "ylim": [-.1, .5], "subject_ids": ["mango"]}
params = _add_medium_sized_plot_params(params)
params = _add_outbound_path_x_params(params)

for metric_name, vector_type in zip(metric_names, vector_types):
    fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Single trial, same outbound path different outbound path difference, correct trials, example rat with y lim zoom

# Parameters
plot_type = "same_different_outbound_path_correct_rat2"
boot_set_name = "same_different_outbound_path_correct_diff"
rat_cohort = False
brain_regions_separate = True
median = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm"]
label_names = ["path"]
relationship_vals_list = [["outbound_correct_correct_trials_same_path_outbound_correct_correct_trials"]]
# Processing/saving
populate_tables = False
save_fig = False

metric_names = ["euclidean_distance", "cosine_similarity"]
vector_types = ["vec", "diff_vec"]
ylims_list = [[-.1, .4], [-.1, .3]]
params = {"populate_tables": populate_tables, "save_fig": save_fig, "path_fr_vec_param_name":
          "correct_incorrect_trials", "subject_ids": ["mango"]}
params = _add_medium_sized_plot_params(params)
params = _add_outbound_path_x_params(params)

params.update({"mega_row_gap_factor": .1})

for metric_name, vector_type, ylims in zip(metric_names, vector_types, ylims_list):
    params.update({"ylim": ylims, "yticks": [0, ylims[-1]]})
    fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Single trial, same outbound path vs. different outbound path, correct trials, brain region difference, rat cohort

# Parameters
plot_type = "same_different_outbound_path_correct_diff"
boot_set_name = "same_different_outbound_path_correct_diff_brain_region_diff"
rat_cohort = True
brain_regions_separate = False
median = False
single_epochs_plot = False
dmPFC_OFC_only = False
# Table names and accompanying params
table_names = ["PathFRVecSTAveSumm"]
label_names = ["path"]
relationship_vals_list = [["outbound_correct_correct_trials_same_path_outbound_correct_correct_trials"]]
# Processing/saving
populate_tables = False
save_fig = False

params = {"populate_tables": populate_tables, "save_fig": save_fig, "path_fr_vec_param_name":
          "correct_incorrect_trials", "xlim": [0, .5], "ylim": [-.1, .3], "yticks": [0, .3]}
params = _add_medium_long_sized_plot_params(params)
params = _add_outbound_path_x_params(params)

params.update({"yticklabels": None})  # causes y tick labels to be placed on all plots

for metric_name, vector_type in zip(metric_names, vector_types):
    fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Delay period

# Single trial, stay leave difference, example rat with y lim zoom

# Parameters
plot_type = "single_trial_stay_leave_rat2"
boot_set_name = "stay_leave_diff"
rat_cohort = False
brain_regions_separate = True
median = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["TimeRelWAFRVecSTAveSumm"]
label_names = ["path"]
relationship_vals_list = [["same_path_stay_leave_trials_same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = True
save_fig = False

ylim = [-.2, .35]
yticks = [-.2, 0, .2]

params = {"populate_tables": populate_tables, "save_fig": save_fig, "ylim": ylim, "yticks": yticks, "xticklabels": [],
          "subject_ids": ["mango"]}
params = _add_medium_sized_plot_params(params)

for metric_name, vector_type in zip(metric_names, vector_types):
    fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Single trial, stay leave, example rat

# Parameters
plot_type = "single_trial_stay_leave"
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = True
median = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["TimeRelWAFRVecSTAveSumm"]
label_names = ["path"]
relationship_vals_list = [["same_path_stay_leave_trials", "same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = True
save_fig = False

metric_names = ["cosine_similarity", "euclidean_distance"]
vector_types = ["diff_vec", "vec"]
ylims = [[-.15, .7], [-.15, .45]]
yticks = [0, .2, .4, .6]

params = {"populate_tables": populate_tables, "save_fig": save_fig, "yticks": yticks, "subject_ids": ["mango"]}
params = _add_medium_sized_plot_params(params)
params.update({"yticklabels": None})  # causes y tick labels to be placed on all plots

for metric_name, vector_type, ylim in zip(metric_names, vector_types, ylims):
    params.update({"ylim": ylim, })
    fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params)


# Single trial, stay leave difference, brain region difference, rat cohort

# Parameters
plot_type = "single_trial_stay_leave_brain_region_diff"
boot_set_name = "stay_leave_diff_brain_region_diff"
rat_cohort = True
brain_regions_separate = False
median = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Table names and accompanying params
table_names = ["TimeRelWAFRVecSTAveSumm"]
label_names = ["path"]
relationship_vals_list = [["same_path_stay_leave_trials_same_path_stay_stay_trials"]]
# Processing/saving
populate_tables = False
save_fig = False

ylim =  [-.25, .1]
yticks = [-.2, 0]
params = {"populate_tables": populate_tables, "save_fig": save_fig, "ylim": ylim, "yticks": yticks, "xticklabels": []}
params = _add_medium_long_sized_plot_params(params)

for metric_name, vector_type in zip(metric_names, vector_types):
    fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params)