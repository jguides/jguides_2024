import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

# Import custom datajoint tables
analysis_dir = '/home/jguidera/Src/jguides_2024'
os.chdir(analysis_dir)
from src.jguides_2024.utils.plot_helpers import save_figure
from src.jguides_2024.utils.string_helpers import format_bool
from src.jguides_2024.glm.analysis_glm import GLMContainer
from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_reliability_paper_nwb_file_names
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name
from src.jguides_2024.time_and_trials.jguidera_condition_trials import ConditionTrialsParams
from src.jguides_2024.time_and_trials.jguidera_cross_validation_pool import TrainTestSplitPoolSel
from src.jguides_2024.glm.jguidera_el_net import ElNetParams, ElNet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescriptions, EpochsDescription, EpochCohort, RecordingSet
from src.jguides_2024.glm.jguidera_measurements_interp_pool import XInterpPoolCohortParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams, ResTimeBinsPoolSel
from src.jguides_2024.utils.df_helpers import df_from_data_list, df_filter1_columns, df_filter_columns, df_pop
from src.jguides_2024.utils.save_load_helpers import unpickle_file, pickle_file


# Set file limit
import subprocess

# Define the new file limit
new_limit = 20480

try:
    subprocess.run(["ulimit", "-n", str(new_limit)], check=True, shell=True)
    print(f"File limit set to {new_limit}")
except subprocess.CalledProcessError as e:
    print(f"Error setting file limit: {e}")


# Define additional functions

def _get_param_names(condition_name, time_bins_shorthand, cross_validation_table_name, covariates_shorthand):

    # Get time bins param name
    res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand(
        time_bins_shorthand)

    # Get cross validation param name (train_test_split_pool_param_name)
    condition_trials_param_name = ConditionTrialsParams().lookup_param_name([condition_name])
    trials_pool_param_name = ResTimeBinsPoolSel().lookup_trials_pool_param_name_from_shorthand(time_bins_shorthand)
    source_table_key = {"trials_pool_param_name": trials_pool_param_name,
                        "condition_trials_param_name": condition_trials_param_name}
    train_test_split_pool_param_name = TrainTestSplitPoolSel().lookup_param_name(
        source_table_name=cross_validation_table_name, source_table_key=source_table_key)

    # Get covariate param name
    x_interp_pool_cohort_param_name = XInterpPoolCohortParams().lookup_param_name_from_shorthand(covariates_shorthand)

    return namedtuple(
        "ParamNames",
        "res_time_bins_pool_cohort_param_name train_test_split_pool_param_name x_interp_pool_cohort_param_name")(
        res_time_bins_pool_cohort_param_name, train_test_split_pool_param_name, x_interp_pool_cohort_param_name)


def get_unit_names_map(nwb_file_name, epochs_id, test_mode=False):

    # Return map from brain region and glm restriction index to particular unit names

    unit_names_map = None

    if nwb_file_name == "J1620210607_.nwb" and epochs_id == "10":
        unit_names_map = {("CA1", 0): ["8_11", "0_6", "12_25"], ("CA1", 1): ["15_18", "0_1"],
                          ("mPFC", 0): ["28_34", "29_24", "27_27"], ("mPFC", 1): ["29_59", "29_7"],
                          ("OFC", 0): ["22_99", "23_4", "23_22"], ("OFC", 1): ["23_83", "25_43"]}

    elif nwb_file_name == "J1620210606_.nwb" and epochs_id == "8":
        unit_names_map = {("CA1", 0): ["12_17", "15_14", "9_2"], ("CA1", 1): ["10_15", "0_6"],
                          ("mPFC", 0): ["26_11", "29_32", "27_49"], ("mPFC", 1): ["27_37", "29_5"],
                          ("OFC", 0): ["24_50", "24_110", "23_32"], ("OFC", 1): ["25_32", "24_68"]}

    # Define smaller subset of units for faster plotting while testing code
    if test_mode:
        unit_names_map = {("CA1", 0): ["12_17", "15_14", ], ("CA1", 1): ["10_15", "0_6"],
                          ("mPFC", 0): ["26_11", "29_32", ], ("mPFC", 1): ["27_37", "29_5"],
                          }

    # Raise error if unit names map not defined
    if unit_names_map is None:
        raise Exception(f"unit_names_map not defined for {nwb_file_name}, epochs_id {epochs_id}")

    # Return unit names map
    return unit_names_map


# Make restriction sets: includes GLM params and prediction similarity params in desired groupings

data_list = []

# path traversal
name = "path"
condition_name = "path_names"
time_bins_shorthand = "path_20ms"
cross_validation_table_name = "LOOCTTrainTestSplit"
covariates_shorthand = "ppt_intercept"
param_names_obj = _get_param_names(condition_name, time_bins_shorthand, cross_validation_table_name, covariates_shorthand)
prediction_similarity_restrictions = {"covariate_restrictions": {"ppt": [0, 1]},
                                           "restrict_to_non_overlapping_space": False}
data_list.append((
    name, param_names_obj.res_time_bins_pool_cohort_param_name, param_names_obj.train_test_split_pool_param_name,
    param_names_obj.x_interp_pool_cohort_param_name, condition_name, prediction_similarity_restrictions))

# delay
name = "delay"
condition_name = "well_names"
time_bins_shorthand = "delay_stay_20ms"
cross_validation_table_name = "LOOCTTrainTestSplit"
covariates_shorthand = "delay_intercept"
param_names_obj = _get_param_names(condition_name, time_bins_shorthand, cross_validation_table_name, covariates_shorthand)
prediction_similarity_restrictions = {"covariate_restrictions": {},
                                           "restrict_to_non_overlapping_space": False}
data_list.append((
    name, param_names_obj.res_time_bins_pool_cohort_param_name, param_names_obj.train_test_split_pool_param_name,
    param_names_obj.x_interp_pool_cohort_param_name, condition_name, prediction_similarity_restrictions))

# Create restriction df
column_names = ["name", "res_time_bins_pool_cohort_param_name", "train_test_split_pool_param_name", "x_interp_pool_cohort_param_name",
                "condition", "prediction_similarity_restrictions"]
glm_restriction_df = df_from_data_list(data_list, column_names)
print(glm_restriction_df)


# Pickle out data so faster to load

# Define params
min_epoch_mean_firing_rate = .1
min_num_spikes = 1
res_time_bins_pool_cohort_param_names = [ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand("path_20ms")]
el_net_param_name = ElNetParams().fetch1("el_net_param_name")
brain_region_cohort_name = "all_targeted"
populate_tables = False

# Define path to file
save_dir = "/nimbus/jguidera/glm/"

# Get keys corresponding to populated entries
populated_keys = ElNet().print_populated(return_keys=True, populate_tables=populate_tables)

# Load or create data
# Indicate whether to load just data to generate group summary plots (group_data_quick_load = True), or rawer data
# (group_data_quick_load = False)
group_data_quick_load = False
# Indicate whether to restrict to single contingency used in example unit plots (fast to load)
restrict_single_contingency = False
# Indicate whether to create only missing files and not load data
create_only = True  # True to only create missing files

if group_data_quick_load:
    os.chdir(save_dir)
    cohort_coeff_plot_data = unpickle_file("cohort_coeff_plot_data", save_dir)
    df_concat = cohort_coeff_plot_data["df_concat"]

else:
    # Restrict to single contingency case (faster to load)
    if restrict_single_contingency:
        populated_keys = [x for x in populated_keys if
                          "J1620210606_.nwb" == x["nwb_file_name"] and x["epochs_id"] == "8"]

    nwb_file_names = get_reliability_paper_nwb_file_names()
    keys = []
    for nwb_file_name in nwb_file_names:
        epochs_descriptions = (EpochsDescriptions() & {"nwb_file_name": nwb_file_name,
                                                       "epochs_descriptions_name": "valid_single_contingency_runs"}).fetch1(
            "epochs_descriptions")
        for epochs_description in epochs_descriptions:
            epochs_id = (EpochsDescription & {"nwb_file_name": nwb_file_name,
                                              "epochs_description": epochs_description}).fetch1("epochs_id")
            key = {"nwb_file_name": nwb_file_name, "epochs_id": epochs_id}
            # only add if elnet fully populated with this key
            if key in populated_keys:
                keys.append(key)

    # Text for file name
    min_fr_text = f"_fr{min_epoch_mean_firing_rate}"
    min_spikes_text = f"_minspks{min_num_spikes}"

    # Loop through keys
    data_list = []
    for key in keys:
        # Define file name
        epoch = (EpochCohort & key).get_epoch()
        file_name_save = "glm_" + format_nwb_file_name(
            key["nwb_file_name"]) + f"_ep{epoch}" + min_spikes_text + min_fr_text

        # Create file if does not exist
        save_path = os.path.join(save_dir, file_name_save)
        if not os.path.exists(save_path):
            print(f"\n{save_path} does not exist. Creating...")
            epochs = (EpochCohort & key).fetch1("epochs")
            glm_container = GLMContainer(
                key["nwb_file_name"], epochs, glm_restriction_df, el_net_param_name,
                brain_region_cohort_name=brain_region_cohort_name,
                sigmas=None, similarity_metrics=None, verbose=True, tolerate_missing_units=False,
                min_epoch_mean_firing_rate=min_epoch_mean_firing_rate, min_num_spikes=min_num_spikes)
            pickle_file(glm_container, file_name_save, save_dir)
        else:
            if create_only:
                continue

        # Load file
        print(f"\nLoading data for {key}...")
        glm_container = unpickle_file(file_name_save, save_dir)

        # Store data
        data_list.append((key["nwb_file_name"], key["epochs_id"], glm_container))
    glm_container_df = df_from_data_list(data_list, ["nwb_file_name", "epochs_id", "glm_container"])


# Check that all epochs present if not restricting to a single contingency and using fully loaded data
all_epochs_present = False
if not restrict_single_contingency and not group_data_quick_load:
    for recording_set_name in RecordingSet().get_recording_set_names(
            recording_set_names_types=["Haight_rotation_single_nwb_files"]):
        nwb_file_names, epochs_descriptions_names = (
                    RecordingSet() & {"recording_set_name": recording_set_name}).fetch1(
            "nwb_file_names", "epochs_descriptions_names")
        for nwb_file_name, epochs_descriptions_name in zip(nwb_file_names, epochs_descriptions_names):
            epochs_descriptions = (EpochsDescriptions() & {
                "nwb_file_name": nwb_file_name, "epochs_descriptions_names": epochs_descriptions_names}).fetch1(
                "epochs_descriptions")
            for epochs_description in epochs_descriptions:
                epochs_id = (EpochsDescription() & {"nwb_file_name": nwb_file_name,
                                                    "epochs_description": epochs_description}).fetch1("epochs_id")
                df_filter1_columns(glm_container_df, {"nwb_file_name": nwb_file_name, "epochs_id": epochs_id})
    all_epochs_present = True

    # Print epochs that are present
    for nwb_file_name in np.unique(glm_container_df.nwb_file_name):
        epochs = np.unique([(EpochCohort & {"nwb_file_name": nwb_file_name, "epochs_id": x}).get_epoch()
                            for x in df_filter_columns(glm_container_df, {'nwb_file_name': nwb_file_name}).epochs_id])
        print(f"{nwb_file_name} epochs: {epochs}")


# Extended data fig. 4: single cell examples and ensemble summary of generalized representation coefficients
if not group_data_quick_load:
    show_legend = False
    plot_color_bar = False
    nwb_file_name = "J1620210606_.nwb"
    epochs_id = "8"
    sigma = 5  # samples. e.g. 20 ms bins x 5 samples = 100ms. bin size in: glm_container.glm_params
    similarity_metrics = ["correlation"]
    test_mode = False
    save_fig = False

    unit_names_map = get_unit_names_map(nwb_file_name, epochs_id)

    glm_container = df_pop(glm_container_df, {"nwb_file_name": nwb_file_name, "epochs_id": epochs_id}, "glm_container")

    # Define the number of rows in each subfigure for a combination of brain region and glm restriction
    # index as the maximum number of such rows for any combination
    num_rows = np.max([len(x) for x in unit_names_map.values()])
    # Get the number of brain regions and glm restriction idxs represented so can define figure size
    # accordingly
    brain_regions = [x for x in ["mPFC", "OFC", "CA1"] if x in [x[0] for x in unit_names_map.keys()]]
    num_brain_regions = len(brain_regions)
    glm_restriction_idxs = np.unique([x[1] for x in unit_names_map.keys()])
    num_glm_restriction_idxs = len(glm_restriction_idxs)

    height_ratios = np.asarray(
        [np.max([len(v) for k, v in unit_names_map.items() if k[1] == glm_restriction_idx]) for glm_restriction_idx in
         glm_restriction_idxs])
    height_ratios = height_ratios + np.asarray(
        [0, .1])  # for some reason, figure looks slightly imbalanced without this correction

    # Define global figure that spans all subfigures
    # define horizontal spacing between plots
    glm_restriction_hspace = .05
    units_hspace = 0

    # Define size of entire plot
    fig_width = 8 * num_brain_regions
    fig_height = 2.4 * num_rows * num_glm_restriction_idxs

    global_fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    subfigs_glm_restriction_idxs = global_fig.subfigures(nrows=num_glm_restriction_idxs, ncols=1,
                                                         hspace=glm_restriction_hspace, height_ratios=height_ratios)

    for glm_restriction_idx_idx, glm_restriction_idx in enumerate(glm_restriction_idxs):
        # Define subfigures, each corresponding to brain region
        # Use spacing between brain region columns proportional to number of conditions so similar aspect ratios
        # across condition types
        num_conditions = len(glm_container.conditions_df.loc[
                                 glm_container._get_glm_restriction_val(glm_restriction_idx, "condition")].condition_vals)
        wspace = .3 / num_conditions
        glm_restriction_idx_fig = subfigs_glm_restriction_idxs[glm_restriction_idx_idx]
        subfigs_brain_regions = glm_restriction_idx_fig.subfigures(nrows=1, ncols=num_brain_regions, wspace=wspace)

        for brain_region_idx, brain_region in enumerate(brain_regions):
            # Define subfigures, each corresponding to a glm restriction idx
            units_fig = subfigs_brain_regions[brain_region_idx]
            # Define subfigures for unit names
            unit_names = unit_names_map[(brain_region, glm_restriction_idx)]
            subfigs_units = units_fig.subfigures(nrows=len(unit_names), ncols=1, hspace=units_hspace)
            for unit_name, fig in zip(unit_names, subfigs_units):
                glm_container.nnls.plot_fr_pred_sim_glm_coeff(glm_restriction_idx=glm_restriction_idx, main_fig=fig,
                                                              sigma=sigma, unit_names=[unit_name],
                                                              show_legend=show_legend,
                                                              similarity_metrics=similarity_metrics,
                                                              plot_color_bar=plot_color_bar)

    # Save figure if indicated
    cbar_text = format_bool(plot_color_bar, "cbar")
    save_figure(global_fig,
                f"glm_single_unit_examples_{format_nwb_file_name(nwb_file_name)}_ep{epochs_id}_sigma{sigma}_{cbar_text}",
                save_fig=save_fig)


# Figure 3: single cell examples and ensemble summary of generalized representation coefficients
if not group_data_quick_load:
    show_legend = False
    plot_color_bar = False
    nwb_file_name = "J1620210606_.nwb"
    epochs_id = "8"
    sigma = 5  # samples. e.g. 20 ms bins x 5 samples = 100ms. bin size in: glm_container.glm_params
    similarity_metrics = ["correlation"]
    test_mode = False
    save_fig = False

    unit_names_map = get_unit_names_map(nwb_file_name, epochs_id)

    glm_container = df_pop(glm_container_df, {"nwb_file_name": nwb_file_name, "epochs_id": epochs_id}, "glm_container")

    # Define the number of rows in each subfigure for a combination of brain region and glm restriction
    # index as the maximum number of such rows for any combination
    num_rows = np.max([len(x) for x in unit_names_map.values()])
    # Get the number of brain regions and glm restriction idxs represented so can define figure size
    # accordingly
    brain_regions = [x for x in ["mPFC", "OFC", "CA1"] if x in [x[0] for x in unit_names_map.keys()]]
    num_brain_regions = len(brain_regions)
    glm_restriction_idxs = np.unique([x[1] for x in unit_names_map.keys()])
    num_glm_restriction_idxs = len(glm_restriction_idxs)

    height_ratios = np.asarray(
        [np.max([len(v) for k, v in unit_names_map.items() if k[1] == glm_restriction_idx]) for glm_restriction_idx in
         glm_restriction_idxs])
    height_ratios = height_ratios + np.asarray(
        [0, .1])  # for some reason, figure looks slightly imbalanced without this correction

    # Define global figure that spans all subfigures
    # define horizontal spacing between plots
    glm_restriction_hspace = .05
    units_hspace = 0

    # Define size of entire plot
    fig_width = 8 * num_brain_regions
    fig_height = 2.4 * num_rows * num_glm_restriction_idxs

    global_fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    subfigs_glm_restriction_idxs = global_fig.subfigures(nrows=num_glm_restriction_idxs, ncols=1,
                                                         hspace=glm_restriction_hspace, height_ratios=height_ratios)

    for glm_restriction_idx_idx, glm_restriction_idx in enumerate(glm_restriction_idxs):
        # Define subfigures, each corresponding to brain region
        # Use spacing between brain region columns proportional to number of conditions so similar aspect
        # ratios across condition types
        num_conditions = len(glm_container.conditions_df.loc[glm_container._get_glm_restriction_val(
            glm_restriction_idx, "condition")].condition_vals)
        wspace = .3 / num_conditions
        glm_restriction_idx_fig = subfigs_glm_restriction_idxs[glm_restriction_idx_idx]
        subfigs_brain_regions = glm_restriction_idx_fig.subfigures(nrows=1, ncols=num_brain_regions, wspace=wspace)

        for brain_region_idx, brain_region in enumerate(brain_regions):
            # Define subfigures, each corresponding to a glm restriction idx
            units_fig = subfigs_brain_regions[brain_region_idx]
            # Define subfigures for unit names
            unit_names = unit_names_map[(brain_region, glm_restriction_idx)]
            subfigs_units = units_fig.subfigures(nrows=len(unit_names), ncols=1, hspace=units_hspace)
            for unit_name, fig in zip(unit_names, subfigs_units):
                glm_container.nnls.plot_fr_pred_sim_glm_coeff(glm_restriction_idx=glm_restriction_idx, main_fig=fig,
                                                              sigma=sigma, unit_names=[unit_name],
                                                              show_legend=show_legend,
                                                              similarity_metrics=similarity_metrics,
                                                              plot_color_bar=plot_color_bar)

    # Save figure if indicated
    cbar_text = format_bool(plot_color_bar, "cbar")
    save_figure(
        global_fig,
        f"glm_single_unit_examples_{format_nwb_file_name(nwb_file_name)}_ep{epochs_id}_sigma{sigma}_{cbar_text}",
        save_fig=save_fig)