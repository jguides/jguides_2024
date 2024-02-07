import copy

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_ordered_subject_ids
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import CovariateFRVecAveSummBase
from src.jguides_2024.metadata.jguidera_epoch import RecordingSet


def _get_recording_set_names(rat_cohort, boot_set_name, dmPFC_OFC_only, subject_ids=None):
    # Define recording set names

    # Single rats
    if subject_ids is None:
        subject_ids = get_ordered_subject_ids()
    recording_set_names = [RecordingSet().get_Haight_single_contingency_rotation_set_name([subject_id])
                           for subject_id in subject_ids]

    # Rat cohot
    if rat_cohort:
        recording_set_names = [RecordingSet().lookup_rat_cohort_set_name()]

    if "brain_region_diff" in boot_set_name and not rat_cohort:
        # Remove the rat fig if brain region diff and single rat plots since fig only has one brain region (OFC)
        recording_set_names = [x for x in recording_set_names if "fig" not in x]
        # Remove peanut also if only dmPFC/OFC difference
        if dmPFC_OFC_only:
            recording_set_names = [x for x in recording_set_names if "peanut" not in x]

    # Return recording set names
    return recording_set_names


def _update_boot_set_name(boot_set_name, rat_cohort, median=False):
    # Update boot set name based on params
    if rat_cohort:
        boot_set_name = boot_set_name + "_rat_cohort"
    if median:
        boot_set_name += "_median"
    return boot_set_name


def _get_brain_region_vals(boot_set_name, dmPFC_OFC_only=None):
    # Define brain region values
    brain_region_vals = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]
    if dmPFC_OFC_only:
        brain_region_vals = ["mPFC_targeted", "OFC_targeted"]
    if "brain_region_diff" in boot_set_name:
        brain_region_vals = ["OFC_targeted_mPFC_targeted"]
    return brain_region_vals


def _get_single_epochs_plot_params(single_epochs_plot):
    # Plot mean and conf, or single epochs
    show_ave_conf = True
    show_single_epoch = False
    if single_epochs_plot:
        show_ave_conf = False
        show_single_epoch = True
    return show_ave_conf, show_single_epoch


def _get_metric_processing_name(metric_name):
    # Define metric processing name based on metric name
    metric_processing_name = "none"
    if metric_name == "euclidean_distance":
        metric_processing_name = "one_minus_on_off_diagonal_ratio"
    return metric_processing_name


def _get_remove_axis_empty_plot(brain_regions_separate):
    # Define whether to remove axis on empty plot
    if brain_regions_separate:
        return True
    return False


def fr_vec_plot(
        plot_type, boot_set_name, rat_cohort, brain_regions_separate, median, single_epochs_plot, dmPFC_OFC_only,
        metric_name, vector_type, table_names, label_names, relationship_vals_list, params):
    # Make copy of param to avoid changing outsisde of function
    params = copy.deepcopy(params)

    # Get quantities based on params
    subject_ids = params.pop("subject_ids", None)
    recording_set_names = _get_recording_set_names(rat_cohort, boot_set_name, dmPFC_OFC_only, subject_ids)
    boot_set_name = _update_boot_set_name(boot_set_name, rat_cohort, median)
    brain_region_vals = _get_brain_region_vals(boot_set_name, dmPFC_OFC_only)
    remove_axis_empty_plot = _get_remove_axis_empty_plot(brain_regions_separate)
    show_ave_conf, show_single_epoch = _get_single_epochs_plot_params(single_epochs_plot)
    metric_processing_name = _get_metric_processing_name(metric_name)

    # Update params
    params.update({
        "metric_name": metric_name, "metric_processing_name": metric_processing_name, "vector_type": vector_type,
        "rat_cohort": rat_cohort, "plot_type": plot_type, "table_names": table_names, "label_names": label_names,
        "relationship_vals_list": relationship_vals_list,
        "recording_set_names": recording_set_names, "boot_set_name": boot_set_name, "show_ave_conf": show_ave_conf,
        "show_single_epoch": show_single_epoch, "remove_axis_empty_plot": remove_axis_empty_plot})

    # Plot
    keys = CovariateFRVecAveSummBase._get_multiplot_params(**params)
    CovariateFRVecAveSummBase().multiplot(recording_set_names, table_names, brain_regions_separate, brain_region_vals,
                                          keys, **params)


def _add_medium_sized_plot_params(params, extra_gaps=False):
    subplot_width = 1.5
    subplot_height = 1.3
    mega_row_gap_factor = .1
    wspace = None
    fontsize = 15
    ticklabels_fontsize = 13

    if extra_gaps:
        mega_row_gap_factor = .2
        wspace = .25

    params.update({"wspace": wspace, "subplot_width": subplot_width, "subplot_height": subplot_height,
                   "mega_row_gap_factor": mega_row_gap_factor,
                   "fontsize": fontsize, "ticklabels_fontsize": ticklabels_fontsize})
    return params


def _add_medium_long_sized_plot_params(params):
    params = _add_medium_sized_plot_params(params)
    params.update({"subplot_width": params["subplot_width"] * 1.8})
    return params


def _add_outbound_path_x_params(params):
    xlim = [0, .5]
    params.update({"xlim": xlim, "xticks": xlim, "xticklabels": None})
    return params