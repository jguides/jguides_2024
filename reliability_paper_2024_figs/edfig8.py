import copy
import os

# Import custom datajoint tables
analysis_dir = '/home/jguidera/Src/jguides_2024'
os.chdir(analysis_dir)
from reliability_paper_2024_figs.population_reliability_plot_helpers import _update_boot_set_name, \
    _get_remove_axis_empty_plot, _get_single_epochs_plot_params
from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_ordered_subject_ids
from src.jguides_2024.edeno_decoder.jguidera_edeno_decoder_error import EdenoDecodeErrSumm
from src.jguides_2024.metadata.jguidera_epoch import TrainTestEpochSet


def _get_brain_region_vals(boot_set_name, dmPFC_OFC_only=None):
    # Define brain region values
    brain_region_vals = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]
    if dmPFC_OFC_only:
        brain_region_vals = ["mPFC_targeted", "OFC_targeted"]
    if "brain_region_diff" in boot_set_name:
        brain_region_vals = ["mPFC_targeted_OFC_targeted"]
    return brain_region_vals


def decoding_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, single_epochs_plot, dmPFC_OFC_only, params):

    # Make copy of param to avoid changing outsisde of function
    params = copy.deepcopy(params)

    # Get quantities based on params
    subject_ids = params.pop("subject_ids", None)
    boot_set_name = _update_boot_set_name(boot_set_name, rat_cohort)
    brain_region_vals = _get_brain_region_vals(boot_set_name, dmPFC_OFC_only)
    remove_axis_empty_plot = _get_remove_axis_empty_plot(brain_regions_separate)
    show_ave_conf, show_single_epoch = _get_single_epochs_plot_params(single_epochs_plot)

    # Define mega row iterables
    # ...Single rats
    subject_ids = get_ordered_subject_ids()
    if boot_set_name == "brain_region_diff":
        subject_ids = [x for x in subject_ids if x not in ["peanut", "fig"]]
    train_test_epoch_set_names = [TrainTestEpochSet().get_Haight_single_contingency_rotation_set_name([subject_id])
                                  for subject_id in subject_ids]
    # ...Rat cohot
    if rat_cohort:
        train_test_epoch_set_names = [TrainTestEpochSet().lookup_rat_cohort_set_name()]

    edeno_decode_param_names = [
        "ppt_default^looct_path_default^sorted_spikes_default^gpu_default^nimbus",
        "wa_stay^looct_path_default^sorted_spikes_default^gpu_default^nimbus"]
    edeno_decode_err_summ_bin_param_name = "epoch_average_uniform_10"
    edeno_decode_err_param_name = "absolute_original_scale^0.01"

    # Update params
    params.update({
        "edeno_decode_param_names": edeno_decode_param_names, "train_test_epoch_set_names": train_test_epoch_set_names,
        "edeno_decode_err_summ_bin_param_name": edeno_decode_err_summ_bin_param_name,
        "edeno_decode_err_param_name": edeno_decode_err_param_name,
        "rat_cohort": rat_cohort, "plot_type": plot_type,
        "boot_set_name": boot_set_name, "show_ave_conf": show_ave_conf,
        "show_single_epoch": show_single_epoch, "remove_axis_empty_plot": remove_axis_empty_plot})

    # Plot
    keys = EdenoDecodeErrSumm._get_multiplot_params(**params)
    EdenoDecodeErrSumm().multiplot(train_test_epoch_set_names, edeno_decode_param_names, brain_regions_separate,
                                   brain_region_vals, keys, **params)


# Single rats

# Parameters
plot_type = ""
boot_set_name = "default"
rat_cohort = False
brain_regions_separate = False
single_epochs_plot = False
dmPFC_OFC_only = True
# Processing/saving
populate_tables = False
save_fig = True

params = {"populate_tables": populate_tables, "save_fig": save_fig}
params.update({"mega_row_gap_factor": .18})

decoding_plot(
    plot_type, boot_set_name, rat_cohort, brain_regions_separate, single_epochs_plot, dmPFC_OFC_only, params)


# Brain region difference, rat cohort
plot_type = ""
boot_set_name = "brain_region_diff"
rat_cohort = True
brain_regions_separate = False
single_epochs_plot = False
dmPFC_OFC_only = False
# Processing/saving
populate_tables = False
save_fig = True

params = {"populate_tables": populate_tables, "save_fig": save_fig}

decoding_plot(
    plot_type, boot_set_name, rat_cohort, brain_regions_separate, single_epochs_plot, dmPFC_OFC_only, params)


# Extra

# Brain region difference, single rats
plot_type = ""
boot_set_name = "brain_region_diff"
rat_cohort = False
brain_regions_separate = False
single_epochs_plot = False
dmPFC_OFC_only = False
# Processing/saving
populate_tables = False
make_plot = False
save_fig = False

params = {"populate_tables": populate_tables, "save_fig": save_fig}

if make_plot:
    decoding_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, single_epochs_plot, dmPFC_OFC_only, params)


