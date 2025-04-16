import copy
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_subject_id
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    PopulationAnalysisSelBase, PathWellFRVecSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, ParamsBase, SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import delete_, insert_analysis_table_entry, \
    insert1_print, get_table_secondary_key_names, get_table_name
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_nwb_file_name_epochs_description
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, RecordingSet
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.utils.df_helpers import df_filter_columns, df_from_data_list, df_pop, get_empty_df
from src.jguides_2024.utils.dict_helpers import add_defaults
from src.jguides_2024.utils.hierarchical_bootstrap import hierarchical_bootstrap
from src.jguides_2024.utils.plot_helpers import get_fig_axes, format_ax, get_ax_for_layout
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.tuple_helpers import reverse_pair
from src.jguides_2024.utils.vector_helpers import unpack_single_element, unique_in_order
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import extract_from_path_name


# This function must be defined in module and not be a method of a class in order for parallelization to work
def _get_dfs(table_key):

    # Accept single argument as input instead of multiple so that can use multiprocessing "map function",
    # which only works with one input

    table, key = table_key

    # Get data for this table entry
    table_subset = (table & key)
    df = table_subset.fetch1_performance_df(drop_column_names=None)

    # To match columns in other summary tables:

    # ...Add columns
    for x in ["brain_region", "nwb_file_name", "epochs_description",
              "brain_region_units_param_name"]:
        df[x] = key[x]
    df["subject_id"] = [get_subject_id(x) for x in df["nwb_file_name"]]

    # ...Add fusion columns
    # nwb file name plus epochs description
    df["nwb_file_name_epochs_description"] = get_nwb_file_name_epochs_description(
        key['nwb_file_name'], key['epochs_description'])

    # ...Rename columns
    rename_columns = {"label_pair_name": "relationship"}
    for x, y in rename_columns.items():
        df[y] = df[x]
    df.drop(columns=list(rename_columns.keys()), inplace=True)

    # Get x vals reflecting covariate based on x, if relevant (cases where decoding variable other than x, e.g.
    # correct vs. incorrect at each path progression bin, i.e. x value)
    bin_centers_map = (table._fr_vec_table() & key).get_bin_centers_map()["x"]
    df["x_val"] = bin_centers_map.loc[df.x].values

    # ...Reorder columns
    other_summary_table_columns = [
        "brain_region", "subject_id", "nwb_file_name", "epochs_description", "x_val", "val",
        "brain_region_units_param_name", "relationship", "original_eps_labels",
        "nwb_file_name_epochs_description"]
    additional_columns = ['label_train_test_order', 'epoch_train_test_order',
                          'total_num_train_trials', 'total_num_test_trials',
                          'num_train_trials_per_iteration', 'num_test_trials_per_iteration']
    return df[other_summary_table_columns + additional_columns]


def get_eps_labels(epochs_1, epochs_2, labels_1=None, labels_2=None):
    labels_none = all([x is None for x in np.concatenate((labels_1, labels_2))])
    no_labels_none = all([x is not None for x in np.concatenate((labels_1, labels_2))])

    if not np.logical_or(labels_none, no_labels_none):
        raise Exception(f"label_1 and label_2 must both be None or neither must be None")

    epochs_1_text = "&".join([str(x) for x in epochs_1])
    epochs_2_text = "&".join([str(x) for x in epochs_2])
    epochs_text = "^".join([epochs_1_text, epochs_2_text])
    if labels_none:
        return epochs_text

    labels_1_text = "&".join([str(x) for x in labels_1])
    labels_2_text = "&".join([str(x) for x in labels_2])
    labels_text = "^".join([labels_1_text, labels_2_text])
    return "^^".join([epochs_text , labels_text])


class DecodeCovFRVecParamsBase(SecKeyParamsBase):

    def get_params(self):
        return self.fetch1(unpack_single_element(get_table_secondary_key_names(self)))

    def get_valid_bin_nums(self, **kwargs):
        raise Exception(f"Must overwrite in child class")


class DecodeCovFRVecSelBase(PopulationAnalysisSelBase):

    def _get_param_name_map(self, key_filter, brain_region_units_cohort_types):
        # Define summary table param name based on key parameters

        params_table = self._get_params_table()()

        # Map from particular boot set names (not necessarily all possible) to valid param names
        relationship_div_decode_cov_fr_vec_param_names = [
            "LDA_path_progression_loocv_correct_stay_trials",
            "LDA_time_in_delay_loocv_correct_stay_trials_pdaw_expand"]
        boot_set_name_param_name_map = {
            "relationship_div_median": relationship_div_decode_cov_fr_vec_param_names,
            "relationship_div_rat_cohort_median": relationship_div_decode_cov_fr_vec_param_names}

        recording_set_names_boot_set_names = [

             # LATE IN LEARNING

             # Rat cohort
             (RecordingSet().lookup_rat_cohort_set_name(), boot_set_name)
             for boot_set_name in self._default_cohort_boot_set_names()] + [

             # Non rat cohort
             (recording_set_name, boot_set_name) for recording_set_name in
             RecordingSet().get_recording_set_names(
                 key_filter, ["Haight_rotation"])
             for boot_set_name in self._default_noncohort_boot_set_names()] + [

            # EARLY IN LEARNING

            # Non cohort
            (recording_set_name, boot_set_name)
            for recording_set_name in RecordingSet().get_recording_set_names(
                key_filter, ["first_day_learning_single_epoch"])
            for boot_set_name in self._default_noncohort_boot_set_names()
        ]

        param_name_map = dict()
        for recording_set_name, boot_set_name in recording_set_names_boot_set_names:
            for brain_region_units_cohort_type in brain_region_units_cohort_types:

                # Get summary table param names
                params_table_keys = params_table.fetch("KEY")
                param_names = [
                    key[params_table.meta_param_name()] for key in params_table_keys if all([(
                            params_table & key).get_params()[x] == y for x, y in zip([
                        "boot_set_name", "brain_region_units_cohort_type"], [
                        boot_set_name, brain_region_units_cohort_type])])]

                # Get decode covariate fr vec param names
                decode_cov_fr_vec_params_table = self._get_main_table()._upstream_table()()._get_params_table()()
                decode_cov_fr_vec_meta_param_name = decode_cov_fr_vec_params_table.meta_param_name()
                decode_cov_fr_vec_param_names = decode_cov_fr_vec_params_table.fetch(
                    decode_cov_fr_vec_meta_param_name)

                # Restrict decode_cov_fr_vec_param_names based on boot_set_name
                if boot_set_name in boot_set_name_param_name_map:
                    decode_cov_fr_vec_param_names = [
                        x for x in decode_cov_fr_vec_param_names if x in boot_set_name_param_name_map[boot_set_name]]

                for decode_cov_fr_vec_param_name in decode_cov_fr_vec_param_names:

                    # Get param name map key
                    kwargs = {
                        "recording_set_name": recording_set_name,
                        decode_cov_fr_vec_meta_param_name: decode_cov_fr_vec_param_name,
                        "brain_region_units_cohort_type": brain_region_units_cohort_type}
                    param_name_map_key = self._format_param_name_map_key(**kwargs)

                    if param_name_map_key not in param_name_map:
                        param_name_map[param_name_map_key] = []

                    param_name_map[param_name_map_key] += list(param_names)

        return param_name_map

    def _format_param_name_map_key(self, **kwargs):

        # Get decode_cov_fr_vec_param_name if not passed
        decode_cov_fr_vec_params_table = self._get_main_table()._upstream_table()()._get_params_table()()
        decode_cov_fr_vec_meta_param_name = decode_cov_fr_vec_params_table.meta_param_name()
        if decode_cov_fr_vec_meta_param_name not in kwargs:
            kwargs[decode_cov_fr_vec_meta_param_name] = kwargs["key"][decode_cov_fr_vec_meta_param_name]

        return tuple([
            kwargs[x] for x in [
                "recording_set_name", "brain_region_units_cohort_type",
                decode_cov_fr_vec_meta_param_name]])

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + ["brain_region_diff", "relationship_div_median"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + [
            "brain_region_diff_rat_cohort", "relationship_div_rat_cohort_median"]

    def insert1(self, key, **kwargs):

        # Only populate if no entry in main table. Alternative is to set skip_insertion to True for part table,
        # but would prevent catching undesired duplicate entries
        table_key = {k: v for k, v in key.items() if k in self.primary_key}  # just check for match at primary key

        if len(self & table_key) == 0:

            # Concatenate performance dfs across entries
            df_concat = self._get_main_table()()._upstream_table()().get_concat_metric_df(key["upstream_keys"])

            # Raise error if df is empty and this is not expected
            expected_empty_df_concat_keys = [
                {'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'recording_set_name': 'fig_Haight_single_contingency_rotation', 'brain_region_cohort_name': 'all_targeted', 'curation_set_name': 'runs_analysis_v1', 'upstream_keys': [{'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run1_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run1'}, {'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run2_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run2'}, {'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run3_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run3'}, {'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run4_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run4'}, {'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run5_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run5'}, {'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run6_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run6'}, {'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run7_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run7'}, {'nwb_file_name': 'fig20211108_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run8_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run8'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run1_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run1'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run2_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run2'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run3_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run3'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run4_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run4'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run5_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run5'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run6_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run6'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run7_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run7'}, {'nwb_file_name': 'fig20211109_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run8_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run8'}, {'nwb_file_name': 'fig20211110_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run1_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run1'}, {'nwb_file_name': 'fig20211110_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run2_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run2'}, {'nwb_file_name': 'fig20211110_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run3_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run3'}, {'nwb_file_name': 'fig20211110_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run4_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run4'}, {'nwb_file_name': 'fig20211110_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run5_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run5'}, {'nwb_file_name': 'fig20211110_.nwb', 'brain_region': 'OFC_targeted', 'brain_region_units_param_name': '0.1_run6_rand_target_region_50_0', 'curation_name': 'raw data valid times no premaze no home_3', 'res_time_bins_pool_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb', 'time_rel_wa_dig_param_name': '0.25', 'time_rel_wa_dig_single_axis_param_name': '0^2', 'res_epoch_spikes_sm_param_name': '0.1', 'decode_time_rel_wa_fr_vec_param_name': 'SVC_stay_leave_loocv_stay_leave_trials_pre_departure', 'zscore_fr': 0, 'time_rel_wa_fr_vec_param_name': 'stay_leave_trials_pre_departure', 'epochs_description': 'run6'}], 'decode_time_rel_wa_fr_vec_summ_param_name': 'default^0.1_1_rand_target_region_50_single_iteration_0'},
            ]

            if len(df_concat) == 0 and key not in expected_empty_df_concat_keys:
                raise Exception(f"df_concat is empty. This is not expected")

            # Insert into main table
            key = insert_analysis_table_entry(self, [df_concat], key, skip_insertion=True)
            super().insert1(key, skip_duplicates=True)

            # Insert into part table
            upstream_keys = key.pop("upstream_keys")
            key_subset = {k: v for k, v in key.items() if k not in ["analysis_file_name", "df_concat_object_id"]}
            for upstream_key in upstream_keys:
                self.Upstream.insert1({**key_subset, **upstream_key})

    def delete_(self, key, safemode=True):
        # If recording set name not in key but components that determine it are, then
        # find matching recording set names given the components, to avoid deleting irrelevant
        # entries
        key = copy.deepcopy(key)  # make copy of key to avoid changing outside function
        recording_set_names = RecordingSet().get_matching_recording_set_names(key)
        for recording_set_name in recording_set_names:
            key.update({"recording_set_name": recording_set_name})
            delete_(self, [], key, safemode)


class DecodeCovFRVecBase(ComputedBase):

    @staticmethod
    def _fr_vec_table():
        raise Exception(f"Must overwrite in child class")

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)

    def _get_vector_df_subset_quantities(self, vector_df, x, labels, decode_var):

        # Restrict df if indicated
        valid_bool = np.ones(len(vector_df))
        # ...label
        if labels is not None:
            valid_bool *= np.asarray([x in labels for x in vector_df["label"]])
        # ...x
        if not np.isnan(x):
            valid_bool *= vector_df["x"] == x
        # ...Apply restriction
        iloc_idxs = np.where(valid_bool)[0]
        df_subset = vector_df.iloc[iloc_idxs]

        # Define vectors
        vectors = np.vstack(df_subset.vector)

        # Define classes
        if decode_var in ["path_progression", "path_progression_collapse_path", "time_in_delay"]:
            classes = np.asarray(df_subset.x)
        elif decode_var in [
            "stay_leave",
            "correct_incorrect", "previous_correct_incorrect",
            "outbound_path", "path", "previous_path",
            "well", "destination_well"]:
            classes = np.asarray(df_subset.label)
        else:
            raise Exception(f"decode_var {decode_var} not accounted for")

        # Define epoch trial numbers
        epoch_trial_numbers = df_subset.epoch_trial_number

        return self._vector_df_subset_quantities(vectors, classes, epoch_trial_numbers, iloc_idxs)

    @staticmethod
    def _vector_df_subset_quantities(vectors, classes, epoch_trial_numbers, iloc_idxs):
        return namedtuple("VectorDfSubsetQuantities", "vectors classes epoch_trial_numbers iloc_idxs")(
            vectors, classes, epoch_trial_numbers, iloc_idxs)

    @staticmethod
    def get_confusion_matrix(test_classes, predicted_classes):

        from sklearn.metrics import confusion_matrix

        # Compute confusion matrix
        classes = np.unique(np.concatenate((test_classes, predicted_classes)))
        cmat = confusion_matrix(test_classes, predicted_classes, labels=classes, normalize="true")
        if len(set([len(x) for x in cmat])) != 1:
            raise Exception

        return cmat, classes

    @staticmethod
    def train_test(train_vectors, train_classes, test_vectors, classifier_name):

        # Initialize model
        if classifier_name == "linear_discriminant_analysis":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
        elif classifier_name == "SVC":
            from sklearn.svm import SVC
            clf = SVC(kernel="linear")
        else:
            raise Exception(f"classifier_name {classifier_name} not recognized")

        # Train model using training vectors and classes
        clf.fit(train_vectors, train_classes)

        # Predict classes based on test vectors
        return clf.predict(test_vectors)

    @staticmethod
    def _get_min_num_samples_train_test(
            classifier_name, cross_validation_method, cross_validate):
        # Define minimum number of classes that must be present to proceed with decoding

        # If LDA: require at least two instances of class for training (e.g. correct vs. incorrect)
        # across every x (e.g. path progression bin), since LDA estimates covariance which
        # requires at least 2 samples
        min_num_classes = 1
        if classifier_name == "linear_discriminant_analysis":
            min_num_classes = 2

        # If using leave one out cross validation, require one additional class so that
        # there are always previous min_num_classes available for training after
        # holding out any given trial for testing.
        if cross_validation_method == "loocv" and cross_validate:
            min_num_classes += 1

        return min_num_classes

    def _get_train_test_data(self, epochs, key, vector_df, verbose=False):

        # Get parameters
        params_table_subset = (self._get_params_table() & key)
        decode_cov_fr_vec_params = params_table_subset.get_params()
        decode_cov_fr_vec_params_meta_name = params_table_subset.meta_param_name()
        decode_cov_fr_vec_params_name = params_table_subset.fetch1(decode_cov_fr_vec_params_meta_name)
        cov_fr_vec_meta_param_name = self._fr_vec_table()()._get_params_table()().meta_param_name()
        cov_fr_vec_param_name = decode_cov_fr_vec_params[cov_fr_vec_meta_param_name]
        decode_var = decode_cov_fr_vec_params["decode_var"]
        classifier_name = decode_cov_fr_vec_params["classifier_name"]
        cross_validation_method = decode_cov_fr_vec_params["cross_validation_method"]
        cross_validation_always = decode_cov_fr_vec_params.pop("cross_validation_always", False)

        nwb_file_name = key["nwb_file_name"]

        # Define order of train and test epochs
        epoch_train_test_orders = np.arange(0, len(np.unique(epochs)))

        # Define epochs
        if len(epochs) == 1:
            epoch_1 = epoch_2 = unpack_single_element(epochs)
            train_epochs = test_epochs = epochs_1 = epochs_2 = [epoch_1]
            epoch_train_test_order = 0
        else:
            epoch_1, epoch_2 = epochs
            # Raise error until fully code up multiple epochs case
            raise Exception(f"Code currently not equipped to run with multiple epochs")

        # Define x to iterate over separately. If we are decoding task progression, want to
        # iterate over all x together. Otherwise, want to iterate over each x (e.g.
        # decoding correct and incorrect at each path progression bin)
        if decode_var in [
            "path_progression", "path_progression_collapse_path", "time_in_delay"]:
            iterate_x_list = [np.nan]
        elif decode_var in [
            "stay_leave",
            "correct_incorrect", "previous_correct_incorrect",
            "path", "previous_path", "outbound_path",
            "well", "destination_well"]:
            iterate_x_list = np.unique(vector_df.x)
            # Check that no nans in list with x
            if np.sum(np.isnan(iterate_x_list) > 0):
                raise Exception(f"nans in x, this is unexpected")
        else:
            raise Exception(f"Defining x to iterate over. Case with decode_var = {decode_var} not accounted for.")

        # Get train/test data for decoding
        num_possible_iterations = 0
        num_skipped_iterations = 0
        data_list = []

        # NON PATH NAME PAIR TYPES
        if decode_var in [
            "path", "previous_path", "well", "destination_well", "path_progression_collapse_path"]:

            label_pair_name = label_train_test_order = None

            cov_fr_vec_params_table = self._fr_vec_table()()._get_params_table()()
            cov_fr_vec_meta_param_name = cov_fr_vec_params_table.meta_param_name()
            cov_fr_vec_params_table_subset = (
                    cov_fr_vec_params_table & {cov_fr_vec_meta_param_name: cov_fr_vec_param_name})

            # Define labels
            if decode_var in ["path", "previous_path"]:
                labels = MazePathWell().get_rewarded_path_names_across_epochs(
                    nwb_file_name, epochs)

            elif decode_var in ["well", "destination_well"]:
                labels = MazePathWell().get_well_names_across_epochs(
                    nwb_file_name, epochs, rewarded_wells=True)

            elif decode_var in ["path_progression_collapse_path"]:
                labels = [""]

            else:
                raise Exception(f"Case for {decode_var} not covered in defining labels")

            labels = cov_fr_vec_params_table_subset.update_labels(labels)

            # Drop incorrect trials if decode_cov_fr_vec_params_name indicates only considering correct trials
            if decode_cov_fr_vec_params_name in [
                "LDA_path_loocv_correct_stay_trials",
                "LDA_path_loocv_correct_stay_trials_pdaw_expand",
                "LDA_well_loocv_correct_stay_trials_pdaw_expand",
                "LDA_destination_well_loocv_correct_stay_trials",
                "LDA_path_progression_collapse_path_loocv_correct_stay_trial",
            ]:
                labels = [x for x in labels if "_correct_trial" in x]
            else:
                raise Exception(f"Finding valid labels. {decode_cov_fr_vec_params_name} not accounted for")

            # Cross validation (if indicated by variable or if only one label such that train/test label same)
            if cross_validation_always or len(labels) == 1:

                cross_validate = cross_validation_always

                # Define minimum number of classes required to proceed
                min_num_classes = self._get_min_num_samples_train_test(
                    classifier_name, cross_validation_method, cross_validate)

                # Loop through x
                for x in iterate_x_list:

                    # Determine which labels have a sufficient number of samples
                    valid_labels = []
                    for label in labels:

                        # Do not filter for x if nan (nan signifies not filtering; see above where
                        # iterate_x_list defined)
                        df_subset = df_filter_columns(vector_df, {"label": label})
                        if not np.isnan(x):
                            df_subset = df_filter_columns(vector_df, {"x": x})

                        # Skip label if insufficient samples
                        num_possible_iterations += 1
                        if len(df_subset) < min_num_classes:
                            if verbose:
                                print(f"On x = {x}. Insufficient trials for labels {label}. Skipping...")
                            num_skipped_iterations += 1
                            raise Exception(f"check this code")
                        else:
                            valid_labels.append(label)

                    # Store
                    labels_1 = labels_2 = train_labels = test_labels = valid_labels
                    data_list.append(
                        (label_pair_name, label_train_test_order, epoch_train_test_order,
                         epochs_1, epochs_2, labels_1, labels_2,
                         train_epochs, test_epochs, train_labels, test_labels,
                         x, cross_validate))
            else:
                raise Exception(f"case where decode var is {decode_var} and not cross validation is"
                                f" not accounted for")

        # PATH NAME PAIR TYPES (E.G. SAME PATH, SAME TURN PATH)
        elif decode_var in [
            "path_progression", "time_in_delay", "correct_incorrect", "previous_correct_incorrect", "outbound_path",
            "stay_leave"
        ]:

            # Define condition comparisons based on param name
            valid_names = None
            if cov_fr_vec_param_name in [
                "prev_correct_incorrect_trials",
                "correct_incorrect_stay_trials", "correct_incorrect_stay_trials_pdaw", "stay_leave_trials_pre_departure"]:

                if decode_var in ["path_progression", "time_in_delay"]:
                    valid_names = [
                        "same_path_correct_correct_stay_trials", "same_turn_correct_correct_stay_trials",
                        "inbound_correct_correct_stay_trials", "outbound_correct_correct_stay_trials",
                        "different_turn_well_correct_correct_stay_trials"]

                elif decode_var in ["outbound_path"]:
                    valid_names = ["outbound_correct_correct_stay_trials"]

                elif decode_var in ["correct_incorrect"]:
                    valid_names = ["same_path_correct_incorrect_stay_trials"]

                elif decode_var in ["previous_correct_incorrect"]:
                    valid_names = ["same_path_prev_correct_incorrect_trials"]

                elif decode_var in ["stay_leave"]:
                    valid_names = ["same_path_stay_leave_trials"]

            elif cov_fr_vec_param_name in ["correct_incorrect_trials"]:

                if decode_var in ["correct_incorrect"]:
                    valid_names = ["same_path_correct_incorrect_trials"]

            # Raise error if valid names not defined (params not accounted for)
            if valid_names is None:
                raise Exception(
                    f"Pair of cov_fr_vec_param_name {cov_fr_vec_param_name} and decode_var {decode_var} "
                    f"not accounted for in code to define condition comparisons")

            # Get map from path name pair types to path name pairs
            include_reversed_pairs = False
            path_name_pair_types_map = dict()
            for epoch in epochs:
                path_name_pair_types_map.update(MazePathWell.get_path_name_pair_types_map(
                    nwb_file_name=nwb_file_name, epoch=epoch, valid_names=valid_names,
                    include_reversed_pairs=include_reversed_pairs))
            # ...Raise error if map empty
            if len(path_name_pair_types_map) == 0:
                raise Exception(f"path_name_pair_types_map is empty. This is not expected. Check that "
                                f"valid_names={valid_names} is accounted for in "
                                f"MazePathWell.get_path_name_pair_types_map")

            # Loop through path name pair types
            for label_pair_name, label_pairs in path_name_pair_types_map.items():

                # Loop through label pairs for this path name pair type (e.g. instances of same turn paths)
                for label_1, label_2 in label_pairs:
                    labels_1 = [label_1]
                    labels_2 = [label_2]

                    # Define train/test orders
                    label_train_test_orders = np.arange(0, len(np.unique([label_1, label_2])))

                    # If leave one out cross validation in all cases, no need to reverse train/test labels
                    if cross_validation_always and cross_validation_method == "loocv":
                        label_train_test_orders = [0]

                    # Loop through label train/test orders
                    for label_train_test_order in label_train_test_orders:

                        # Loop through epoch train/test orders
                        for epoch_train_test_order in epoch_train_test_orders:

                            # Define train/test epochs and labels based on order

                            # EPOCHS
                            reverse = epoch_train_test_order == 1
                            train_epoch, test_epoch = reverse_pair([epoch_1, epoch_2], reverse)
                            train_epochs = [train_epoch]
                            test_epochs = [test_epoch]

                            # LABELS
                            reverse = label_train_test_order == 1
                            train_label, test_label = reverse_pair([label_1, label_2], reverse)
                            train_labels = [train_label]
                            test_labels = [test_label]

                            # Indicate whether or not to cross validate
                            # ...Define whether same train and test label and train and test epoch
                            same_train_test_labels_epochs = np.logical_and(
                                set(train_labels) == set(test_labels), set(train_epochs) == set(test_epochs))
                            cross_validate = np.logical_or(same_train_test_labels_epochs, cross_validation_always)

                            # Define minimum number of classes required to proceed
                            min_num_classes = self._get_min_num_samples_train_test(
                                classifier_name, cross_validation_method, cross_validate)

                            # Loop through x
                            for x in iterate_x_list:

                                # Continue to next iteration if insufficient samples for either label
                                insufficient_trials = False
                                for label in [label_1, label_2]:

                                    # Do not filter for x if nan (nan signifies not filtering; see above where
                                    # iterate_x_list defined)
                                    df_subset = df_filter_columns(vector_df, {"label": label})
                                    if not np.isnan(x):
                                        df_subset = df_filter_columns(df_subset, {"x": x})

                                    if len(set(df_subset.epoch_trial_number)) < min_num_classes:
                                        insufficient_trials = True

                                num_possible_iterations += 1
                                if insufficient_trials:
                                    if verbose:
                                        print(f"On x {x}. {label_1} {label_2} pair does not have sufficient trials. "
                                              f"Skipping...")
                                        print("num trials:", len(df_subset))
                                    num_skipped_iterations += 1
                                    continue

                                # Store
                                data_list.append(
                                    (label_pair_name, label_train_test_order, epoch_train_test_order,
                                     epochs_1, epochs_2, labels_1, labels_2,
                                     train_epochs, test_epochs, train_labels, test_labels,
                                     x, cross_validate))

        else:
            raise Exception(f"{decode_var} not accounted for")

        return df_from_data_list(data_list, [
                "label_pair_name", "label_train_test_order", "epoch_train_test_order",
                "epochs_1", "epochs_2", "labels_1", "labels_2",
                "train_epochs", "test_epochs", "train_labels", "test_labels",
                "x", "cross_validate"]), num_possible_iterations, num_skipped_iterations

    def get_cmat_df(self, key):

        # Get df with confusion matrices
        # Separate function for this to be able to easily check make function computations outside of populate

        # Get params
        params_table = self._get_params_table()
        decode_cov_fr_vec_params = (params_table & key).get_params()
        decode_var = decode_cov_fr_vec_params["decode_var"]
        classifier_name = decode_cov_fr_vec_params["classifier_name"]
        cross_validation_method = decode_cov_fr_vec_params["cross_validation_method"]
        cross_validation_always = decode_cov_fr_vec_params.pop("cross_validation_always", False)

        # Define epochs (for now coded for one epoch)
        epochs = (EpochsDescription & key).fetch1("epochs")

        # Get firing rate vectors and covariate
        vector_df_name = "vector_df"
        concat_df = self._fr_vec_table()().concat_across_epochs(key, vector_df_name)

        # Restrict firing rate vectors to those at valid x
        # ...Get valid bin numbers
        valid_bin_nums = params_table().get_valid_bin_nums(key=key)
        # ...Restrict
        vector_df = concat_df[concat_df.isin({"x": valid_bin_nums}).x]

        # Alter labels if indicated
        table_name = get_table_name(self)
        if np.logical_or(table_name == "DecodeTimeRelWAFRVec" and decode_var == "well",
                         table_name == "DecodePathFRVec" and decode_var == "destination_well"):
            vector_df.label = [extract_from_path_name(x, "end_well_plus_descriptor") for x in vector_df.label]
        elif decode_var in ["path_progression_collapse_path"]:
            vector_df.label = [extract_from_path_name(x, "descriptor") for x in vector_df.label]
        # Do not alter
        elif decode_var in [
            "correct_incorrect", "previous_correct_incorrect", "stay_leave", "outbound_path", "path",
            "path_progression", "time_in_delay"]:
            pass
        else:
            raise Exception(f"Altering labels. Case decode_var = {decode_var} not accounted for. "
                            f"set(vector_df.label): {set(vector_df.label)}")

        # Get train/test data to loop through
        train_test_df, num_possible_iterations, num_skipped_iterations = self._get_train_test_data(
            epochs, key, vector_df)

        # Loop through train/test data
        data_list = []
        for _, df_row in train_test_df.iterrows():

            # Define train/test sets for decoding, and decode either 1) using cross validation or
            # 2) not using cross validation

            # CROSS VALIDATION
            if df_row.cross_validate:

                # Leave one out cross validation
                if cross_validation_method == "loocv":

                    # Get data for training and testing

                    # ...If same train and test labels and epochs, get one set of data
                    same_train_test_labels_epochs = np.logical_and(
                        set(df_row.train_labels) == set(df_row.test_labels),
                        set(df_row.train_epochs) == set(df_row.test_epochs))
                    if same_train_test_labels_epochs:
                        data = self._get_vector_df_subset_quantities(
                            vector_df, df_row.x, df_row.train_labels, decode_var)

                    # ...If not same train and test labels but doing cross validation, combine sets of
                    # data for train and test
                    elif cross_validation_always:

                        train_data = self._get_vector_df_subset_quantities(
                            vector_df, df_row.x, df_row.train_labels, decode_var)
                        test_data = self._get_vector_df_subset_quantities(
                            vector_df, df_row.x, df_row.test_labels, decode_var)
                        new_fields = [np.concatenate(
                                [getattr(d, x) for d in [train_data, test_data]])
                                for x in train_data._fields]
                        data = self._vector_df_subset_quantities(*new_fields)

                    # ...Otherwise raise an exception
                    else:
                        raise Exception(f"case not accounted for")

                    # Test each trial having trained on remaining trials
                    # We also need to track corresponding iloc_idxs so we can store

                    # ...Initialize variables
                    test_classes = []
                    predicted_labels = []
                    train_iloc_idxs = []
                    test_iloc_idxs = []

                    # ...Find unique epoch numbers
                    unique_epoch_trial_numbers = np.unique(data.epoch_trial_numbers)

                    # ...Loop through each trial number, hold one out, train on the remaining, and
                    # decode
                    for epoch_trial_number in unique_epoch_trial_numbers:

                        # Define boolean for train and test
                        test_bool = data.epoch_trial_numbers == epoch_trial_number
                        train_bool = np.invert(test_bool)

                        # Define train vectors and classes
                        train_vectors_ = data.vectors[train_bool]
                        train_classes_ = data.classes[train_bool]

                        # Define test vectors and classes
                        test_vectors_ = data.vectors[test_bool]
                        test_classes_ = data.classes[test_bool]

                        # Store test classes
                        test_classes += list(test_classes_)

                        # Track iloc idxs corresponding to data for training and testing
                        train_iloc_idxs.append(data.iloc_idxs[train_bool])
                        test_iloc_idxs.append(data.iloc_idxs[test_bool])

                        # Predict and store
                        predicted_labels += list(self.train_test(
                            train_vectors_, train_classes_, test_vectors_, classifier_name))

                    cmat, cmat_classes = self.get_confusion_matrix(test_classes, predicted_labels)

                    # Keep track of data used for training and testing

                    # Define number of trials used for training and testing, in total and for each
                    # training iteration
                    total_num_train_trials = total_num_test_trials = len(
                        np.unique(data.epoch_trial_numbers))
                    # subtract one since leave one out cross validation
                    num_train_trials_per_iteration = total_num_train_trials - 1
                    num_test_trials_per_iteration = 1

                    # Define all classes used for training and testing
                    all_train_classes = all_test_classes = data.classes

                # Raise exception if cross validation method not accounted for in code
                else:
                    raise Exception(
                        f"cross_validation_method {cross_validation_method} not accounted for in code")

            # 2) NO CROSS VALIDATION. Train and test on sets of data defined by train_label and
            # test_label
            else:

                # Get train vectors and classes
                train_data = self._get_vector_df_subset_quantities(
                    vector_df, df_row.x, df_row.train_labels, decode_var)

                # Get test vectors and classes
                test_data = self._get_vector_df_subset_quantities(vector_df, df_row.x, df_row.test_labels, decode_var)

                # Compute confusion matrix
                predicted_labels = self.train_test(
                    train_data.vectors, train_data.classes, test_data.vectors, classifier_name)
                cmat, cmat_classes = self.get_confusion_matrix(test_data.classes, predicted_labels)

                # Define iloc idxs
                train_iloc_idxs = train_data.iloc_idxs
                test_iloc_idxs = test_data.iloc_idxs

                # Define number of trials used for training
                total_num_train_trials = num_train_trials_per_iteration = len(
                    np.unique(train_data.epoch_trial_numbers))
                # Define number of trials tested
                total_num_test_trials = num_test_trials_per_iteration = len(
                    np.unique(test_data.epoch_trial_numbers))

                # Define all classes used for training and testing
                all_train_classes = train_data.classes
                all_test_classes = test_data.classes

            # Define text indicating epochs and labels used for training and testing
            # ...Original
            original_eps_labels = get_eps_labels(df_row.epochs_1, df_row.epochs_2, df_row.labels_1, df_row.labels_2)
            # ...After possibly reversing order as indicated
            eps_labels = get_eps_labels(
                df_row.train_epochs, df_row.test_epochs, df_row.train_labels, df_row.test_labels)

            # Store
            data_list.append(
                (df_row.x, df_row.label_pair_name, original_eps_labels, eps_labels, df_row.epoch_train_test_order,
                 df_row.label_train_test_order, df_row.train_labels, df_row.test_labels, df_row.train_epochs,
                 df_row.test_epochs, train_iloc_idxs, test_iloc_idxs, total_num_train_trials, total_num_test_trials,
                 num_train_trials_per_iteration, num_test_trials_per_iteration,
                 all_train_classes, all_test_classes, cmat_classes, cmat))

        # Define names for df columns
        df_column_names = ["x", "label_pair_name", "original_eps_labels", "eps_labels", "epoch_train_test_order",
                           "label_train_test_order", "train_labels", "test_labels", "train_epochs", "test_epochs",
                           "train_iloc_idxs", "test_iloc_idxs", "total_num_train_trials", "total_num_test_trials",
                           "num_train_trials_per_iteration", "num_test_trials_per_iteration",
                           "all_train_classes", "all_test_classes", "confusion_matrix_classes", "confusion_matrix"]

        # Create df
        cmat_df = df_from_data_list(data_list, df_column_names)

        # Raise exception if cmat_df has a different number of entries than expected based on number of
        # skipped entries (due to limited data)
        if len(cmat_df) == 0:
            total_num_test_labels = 0
        else:
            total_num_test_labels = len(np.concatenate(np.vstack(cmat_df.test_labels)))
        if total_num_test_labels + num_skipped_iterations != num_possible_iterations:
            raise Exception(f"cmat_df does not have expected length and number of test labels. total_num_test_labels: "
                            f"{total_num_test_labels} "
                            f"num_skipped_iterations: {num_skipped_iterations} num_possible_iterations: "
                            f"{num_possible_iterations}")

        # Create empty df with expected columns if df empty
        if len(cmat_df) == 0:
            cmat_df = get_empty_df(df_column_names)

        # Convert None to "none" in df so can save as nwbf
        none_columns = self._cmat_df_none_cols()
        for column_name in none_columns:
            if np.sum(cmat_df[column_name] == "none") > 0:
                raise Exception(f"the string none already present in cmat_df columnn {column_name}")
            cmat_df[column_name] = ["none" if x is None else x for x in cmat_df[column_name]]

        # Return df
        return cmat_df

    @staticmethod
    def _cmat_df_none_cols():
        return ["label_pair_name", "label_train_test_order"]

    def make(self, key):

        # Get df with confusion matrices
        cmat_df = self.get_cmat_df(key)

        # Drop train and test iloc_idxs, as these columns may have ragged arrays which unfortunately cant be stored in
        # nwb file
        cmat_df.drop(columns=["train_iloc_idxs", "test_iloc_idxs"], inplace=True)

        # Convert datatype of columns with arrays with characters, which cannot save in hdf5
        # (get broadcasting error). Tried converting to object datatype or storing as list in df but these
        # gave same error.
        for column_name in self._char_column_names():
            if any([isinstance(x, str) for y in cmat_df[column_name] for x in y]):
                parse_char = self._parse_char()
                if any([parse_char in x for x in cmat_df[column_name]]):
                    raise Exception(f"Cannot reformat data because parse_char is in data")
                cmat_df[column_name] = [parse_char.join(x) for x in cmat_df[column_name]]

        # Insert into main table
        insert_analysis_table_entry(self, [cmat_df], key)

        # Insert into part table
        for epoch in (EpochsDescription & key).fetch1("epochs"):
            key.update({"epoch": epoch})
            insert1_print(self.Upstream, key)

    @staticmethod
    def _parse_char():
        return "_PARSESPACE_"

    @staticmethod
    def _char_column_names():
        return ["train_labels", "test_labels", "confusion_matrix_classes", "all_train_classes", "all_test_classes"]

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):

        # Get df
        cmat_df = super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

        # Restore character arrays, which had to save out as single string with parse character to be hdf5 compatible
        column_names = self._char_column_names()
        for column_name in column_names:

            str_bool = [isinstance(x, str) for x in cmat_df[column_name]]

            if np.sum(str_bool) not in [0, len(str_bool)]:
                raise Exception(f"Expect {column_name} rows to have all strings or no strings")

            if np.sum(str_bool) == len(str_bool):
                cmat_df[column_name] = [
                    x.split(self._parse_char()) for x in cmat_df[column_name]]

        # Return df
        return cmat_df

    def plot_ave_confusion_matrices(self):

        # Take average of confusion matrices across label pair instances

        df = self.fetch1_dataframe()

        label_pair_names = np.unique(df["label_pair_name"])

        x_list = np.unique(df.x)

        data_list = []
        for x in x_list:
            for label_pair_name in label_pair_names:
                df_subset = df_filter_columns(df, {"label_pair_name": label_pair_name, "x": x})

                confusion_matrix_ave = np.mean(np.stack(df_subset.confusion_matrix), axis=0)

                # Store
                data_list.append((x, label_pair_name, confusion_matrix_ave))

        cmat_ave_df = df_from_data_list(data_list, ["x", "label_pair_name", "confusion_matrix_ave"])

        # Plot
        num_rows = len(label_pair_names)
        num_columns = len(x_list)

        subplot_width = 2
        subplot_height = subplot_width*num_rows

        # Initialize figures
        fig, axes = get_fig_axes(num_rows, num_columns, subplot_width=subplot_width, subplot_height=subplot_height)

        plot_num = 0
        for label_pair_name_idx, label_pair_name in enumerate(label_pair_names):
            for x_idx, x in enumerate(x_list):
                ax = get_ax_for_layout(axes, plot_num)
                plot_num += 1
                confusion_matrix_ave = df_pop(cmat_ave_df, {
                    "x": x, "label_pair_name": label_pair_name}, "confusion_matrix_ave")

                ax.imshow(confusion_matrix_ave)

                format_ax(ax, title=label_pair_name.replace("_", " "), fontsize=6)

    def plot_performance(self, **kwargs):
        # Plot performance

        # Get performance information
        df = self.fetch1_performance_df()

        # Average across label pair instances (e.g. same turn paths), train/test order, and label pair order (e.g.
        # whether path 1 or path 2 comes first)
        ave_across_column_names = [
            "original_eps_labels", "eps_labels", "train_labels", "test_labels", "epoch_train_test_order",
            "train_epochs", "test_epochs", "label_train_test_order",
            "total_num_train_trials", "total_num_test_trials",
            "num_train_trials_per_iteration", "num_test_trials_per_iteration", "confusion_matrix_class"]
        groupby_cols = [x for x in df.columns if x not in ave_across_column_names + ["val"]]
        # To use groupby, convert columns with lists to string
        for col in groupby_cols:
            if type(df[col][0]) in [np.ndarray, list]:
                df[col] = ["_".join([str(x) for x in y]) for y in df[col]]
        df = df.groupby(groupby_cols).mean(numeric_only=True).reset_index()

        # Plot

        # Initialize figure
        ax = kwargs.pop("ax", None)
        if ax is None:
            fig, ax = plt.subplots()

        # Get label pair names
        label_pair_names = np.unique(df.label_pair_name)

        # Loop through labels and plot average performance value at each x
        for label_pair_name in label_pair_names:
            vals = df_filter_columns(df, {"label_pair_name": label_pair_name}).set_index("x").sort_index()["val"]
            ax.plot(vals)

        # Format axis
        ax.legend(label_pair_names)
        key = self.fetch1("KEY")
        title = f"{key['nwb_file_name']} {key['epochs_description']} {key['brain_region']}"
        format_ax(ax, title=title)

    def fetch1_performance_df(self, drop_column_names=(
            "confusion_matrix", "train_label", "test_label", "train_epoch",
            "test_epoch", "all_train_classes", "all_test_classes")):

        # Return df with performance information by test class label, separately for each class, and
        # optionally without columns

        df = self.fetch1_dataframe()

        if drop_column_names is None:
            drop_column_names = []
        keep_df_columns = [x for x in df.columns if x not in drop_column_names]

        data_list = []
        for _, df_row in df.iterrows():
            diag = np.diagonal(df_row.confusion_matrix)
            data_list.append([df_row[x] for x in keep_df_columns] + [diag])

        df_column_names = keep_df_columns + ["val"]
        df = df_from_data_list(data_list, df_column_names)

        if len(df) == 0:
            df = get_empty_df(df_column_names)

        meta_param_name = self._get_params_table()().meta_param_name()

        param_name = self.fetch1(meta_param_name)

        # If decoding task progression, flatten df so that each x has its own row
        # If decoding correct/incorrect, flatten df so that each correct/incorrect condition has its own row
        confusion_matrix_class_name = None
        if param_name in [
            "LDA_path_progression_loocv_correct_stay_trials",
            "LDA_path_progression_collapse_path_loocv_correct_stay_trial",
            "LDA_time_in_delay_loocv_correct_stay_trials",
            "LDA_time_in_delay_loocv_correct_stay_trials_pdaw_expand",
        ]:
            confusion_matrix_class_name = "x"

        elif param_name in [
            "SVC_correct_incorrect_loocv",
            "SVC_previous_correct_incorrect_loocv",
            "SVC_correct_incorrect_loocv_stay_trials",
            "SVC_correct_incorrect_loocv_stay_trials_pdaw",
            "SVC_correct_incorrect_loocv_stay_trials_pdaw_expand",
            "SVC_stay_leave_loocv_stay_leave_trials_pre_departure",
            "LDA_outbound_path_loocv_correct_stay_trials",
            "LDA_outbound_path_loocv_correct_stay_trials_pdaw_expand",
            "LDA_path_loocv_correct_stay_trials",
            "LDA_well_loocv_correct_stay_trials_pdaw_expand",
            "LDA_path_loocv_correct_stay_trials_pdaw_expand",
            "LDA_destination_well_loocv_correct_stay_trials",
        ]:
            confusion_matrix_class_name = "confusion_matrix_class"

        if confusion_matrix_class_name is not None:
            keep_df_columns = [x for x in df.columns if x not in [
                confusion_matrix_class_name, "val", "confusion_matrix_classes"]]
            data_list = []
            for _, df_row in df.iterrows():
                cols = [getattr(df_row, z) for z in keep_df_columns]
                for confusion_matrix_class, val in zip(df_row.confusion_matrix_classes, df_row.val):
                    data_list.append(cols + [confusion_matrix_class, val])

            df_column_names = keep_df_columns + [confusion_matrix_class_name, "val"]
            df = df_from_data_list(data_list, df_column_names)

            if len(df) == 0:
                df = get_empty_df(df_column_names)

        else:
            raise Exception(
                f"{meta_param_name} {param_name} not accounted for")

        return df

    def get_concat_metric_df(self, keys, debug_mode=False):

        print(f"Concatenating table entries from {self.table_name} across {len(keys)} keys...")

        # Iterate over restriction conditions and units
        # If want to debug, do in for loop
        if debug_mode:
            df_list = []
            for key in keys:
                df_list.append(_get_dfs((self, key)))

        # Otherwise do using parallelization
        else:
            import multiprocessing as mp

            tables_keys = [(self, key) for key in keys]

            pool = mp.Pool(mp.cpu_count())
            df_list = pool.map(_get_dfs, tables_keys)
            pool.close()
            pool.join()

        # Return concatenated entries
        print(f"Returning df with concatenated entries...")
        df_concat = None
        if len(df_list) > 0:
            return pd.concat(df_list)
        return df_concat


class DecodeCovFRVecSummBase(PathWellFRVecSummBase):

    @staticmethod
    def _upstream_table():
        raise Exception(f"This method must be overwritten in child class")

    def _get_relationship_div_column_params(self, **kwargs):

        upstream_table = self._upstream_table()()
        decode_cov_fr_vec_params = (upstream_table._get_params_table() & kwargs["key"]).get_params()
        cov_fr_vec_meta_param_name = upstream_table._fr_vec_table()()._get_params_table()().meta_param_name()
        cov_fr_vec_param_name = decode_cov_fr_vec_params[cov_fr_vec_meta_param_name]
        decode_var = decode_cov_fr_vec_params["decode_var"]

        decode_meta_param_name = upstream_table._get_params_table()().meta_param_name()
        decode_param_name = (upstream_table._get_params_table()() & kwargs["key"]).fetch1(decode_meta_param_name)

        params = None
        if cov_fr_vec_param_name in ["correct_incorrect_stay_trials", "correct_incorrect_stay_trials_pdaw"] \
                and decode_param_name in [
            "LDA_path_progression_loocv_correct_stay_trials", "LDA_time_in_delay_loocv_correct_stay_trials_pdaw_expand"
        ]:

            if decode_var in ["path_progression", "time_in_delay"]:
                params = {
                    "denominator_column_name": "same_path_correct_correct_stay_trials", "numerator_column_names": [
                    "same_turn_correct_correct_stay_trials", "inbound_correct_correct_stay_trials",
                        "outbound_correct_correct_stay_trials", "different_turn_well_correct_correct_stay_trials"]}

        if params is None:
            raise Exception(
                f"combination of {cov_fr_vec_param_name}, {decode_param_name}, {decode_var} not accounted for")

        return params

    def make(self, key):

        df_concat = (self._get_selection_table() & key).fetch1_dataframe()

        # Define columns to average across
        average_across_cols = ["label_train_test_order", "epoch_train_test_order"]

        # Define columns to drop
        drop_columns = [
            "total_num_train_trials", "total_num_test_trials", "num_train_trials_per_iteration",
            "num_test_trials_per_iteration",
            "nwb_file_name", "epochs_description"
        ]
        # Add to columns to drop if 'none', otherwise average across
        column_name = "label_train_test_order"
        if all(df_concat[column_name] == "none"):
            drop_columns.append(column_name)
            average_across_cols.remove(column_name)

        # Drop columns
        df_concat.drop(columns=drop_columns, inplace=True)

        # Average across
        groupby_cols = [x for x in df_concat.columns if x not in average_across_cols + ["val"]]
        metric_df = df_concat.groupby(groupby_cols).mean().reset_index().drop(
            columns=average_across_cols)

        # Hierarchical bootstrap and sample mean

        # ...Define bootstrap params as indicated
        params_table_subset = (self._get_params_table()() & key)
        bonferroni_num_tests = len(set(metric_df.x_val))
        bootstrap_params = params_table_subset.get_boot_params(bonferroni_num_tests)
        boot_set_name = params_table_subset.get_params()["boot_set_name"]

        # average values
        if boot_set_name in ["default", "default_rat_cohort"]:
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = [
                "nwb_file_name_epochs_description", "original_eps_labels", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = ["x_val", "brain_region", "relationship"]
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        # average difference values across brain regions
        elif boot_set_name in ["brain_region_diff", "brain_region_diff_rat_cohort"]:

            target_column_name = "brain_region"
            pairs_order = self._get_brain_region_order_for_pairs()
            vals_index_name = self._get_vals_index_name()
            eps_labels_resample_col_name = "original_eps_labels"
            exclude_columns = None
            metric_df, resample_levels, ave_group_column_names = self._get_boot_diff_params(
                target_column_name, metric_df, pairs_order, vals_index_name, boot_set_name,
                eps_labels_resample_col_name, exclude_columns, debug_mode=False)

        # average ratio of values across same path and same turn or different turn path relationships for
        # path traversals and delay period
        elif boot_set_name in self._get_params_table()()._valid_relationship_div_boot_set_names():

            # First redefine metric_df to reflect ratio of val for same path and same turn or different turn
            # relationship
            kwargs = {"key": key}
            div_params = self._get_relationship_div_column_params(**kwargs)
            # ...Define pairs of relationships
            target_column_name = "relationship"
            target_column_pairs = [
                (x, div_params["denominator_column_name"]) for x in div_params["numerator_column_names"]]
            # ...Define function for computing metric on relationship pairs
            metric_pair_fn = self.metric_pair_div
            # ...Define columns at which to allow different values across members of pair. Here, these are columns
            # that contain information about the context (path or well) rat is on, as we want to compare metrics
            # across different contexts (paths or wells)
            exclude_columns = ["original_eps_labels"]
            # ...Get df with paired metric
            # tolerate nonoverlapping column sets if early in learning sessions
            tolerate_nonoverlapping_column_sets = False
            if np.logical_and(
                    key["recording_set_name"] in RecordingSet().get_recording_set_names(
                    dict(), ["first_day_learning_single_epoch"]),
                    key["decode_path_fr_vec_param_name"] == "LDA_path_progression_loocv_correct_stay_trials"):
                tolerate_nonoverlapping_column_sets = True
            metric_df = self.get_paired_metric_df(
                metric_df, target_column_name, target_column_pairs, metric_pair_fn, exclude_columns=exclude_columns,
            tolerate_nonoverlapping_column_sets=tolerate_nonoverlapping_column_sets)

            # Define parameters for bootstrap
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = ["nwb_file_name_epochs_description", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            vals_index_name =  self._get_vals_index_name()
            ave_group_column_names = [
                vals_index_name, "x_val", "brain_region", self._get_joint_column_name(target_column_name)] + \
                                     self._get_pair_column_names(target_column_name)
            ave_group_column_names = unique_in_order(ave_group_column_names)
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        # Raise exception if boot set name not accounted for in code
        else:
            raise Exception(f"Have not written code for boot_set_name {boot_set_name}")

        # Perform bootstrap
        print("Boostrapping...")
        boot_results = hierarchical_bootstrap(
            metric_df, resample_levels, "val", ave_group_column_names,
            num_bootstrap_samples_=bootstrap_params.num_bootstrap_samples, average_fn_=bootstrap_params.average_fn,
            alphas=bootstrap_params.alphas)

        # Store dfs with results together to save out below
        # ...df with metric values
        results_dict = {"metric_df": metric_df}
        # ...bootstrap results. Convert results that are None to dfs
        for x in ["ave_conf_df", "boot_ave_df"]:
            results_dict[x] = pd.DataFrame()
            if getattr(boot_results, x) is not None:
                results_dict[x] = getattr(boot_results, x)

        # Insert into main table
        insert_analysis_table_entry(self, list(results_dict.values()), key)

    def _get_param_names_obj(self, **kwargs):

        # summary table
        params_table = self._get_params_table()()
        param_name = params_table.get_param_name(**kwargs)
        meta_param_name = params_table.meta_param_name()

        # upstream table
        upstream_table = self._upstream_table()()
        upstream_params_table = upstream_table._get_params_table()()
        upstream_meta_param_name = upstream_params_table.meta_param_name()
        upstream_param_name = kwargs["decode_cov_fr_vec_param_name"]

        Params = namedtuple(
            "Params", "meta_param_name param_name upstream_meta_param_name upstream_param_name")

        return Params(meta_param_name, param_name, upstream_meta_param_name, upstream_param_name)

    def _get_vals_index_name(self):
        return "x_val"

    def _get_val_lims(self, **kwargs):
        # Get a set range for value, e.g. for use in plotting value on same range across plots
        params_table = self._get_params_table()()
        boot_set_name = self.get_upstream_param("boot_set_name")
        scale_ymax = 1
        if boot_set_name in params_table._valid_brain_region_diff_boot_set_names():
            ymax = .6
            return [-ymax, ymax*scale_ymax]
        ymax = 1
        return [0, ymax*scale_ymax]

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        # Return default params
        return params

    def _get_val_text(self):
        return "Decode\naccuracy"

    def _get_val_ticks(self):
        return [0, .5, 1]

    def _get_yticklabels(self, ticks=None):
        return ["0", "", "1"]

    def get_ordered_relationships(self):

        return [
            'different_turn_well_correct_correct_stay_trials', 'inbound_correct_correct_stay_trials',
            'outbound_correct_correct_stay_trials', 'same_path_correct_correct_stay_trials',
            'same_turn_correct_correct_stay_trials'
        ]

    # Override in children classes where relationship exists
    def _get_relationship_meta_name(self):

        params_table = self._get_params_table()()
        boot_set_name = (params_table & self.fetch1("KEY")).get_params()["boot_set_name"]
        if boot_set_name in params_table._valid_relationship_div_boot_set_names() + \
            params_table._valid_stay_leave_diff_boot_set_names() + \
            params_table._valid_stay_leave_diff_brain_region_diff_boot_set_names() + \
            params_table._valid_same_different_outbound_path_correct_diff_boot_set_names() + \
            params_table._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():
            return self._get_joint_column_name("relationship")

        return "relationship"

    @staticmethod
    def _get_multiplot_params(**kwargs):
        # Define params for plotting multiple table entries. One param set per table entry.

        # Check that loop inputs passed
        check_membership([
            "table_names", "label_names", "relationship_vals_list", "recording_set_names",
            "decode_cov_fr_vec_param_names"], kwargs)

        # Make copy of kwargs to serve as base of each key
        kwargs = copy.deepcopy(kwargs)

        # Remove iterables so that not in individual keys
        table_names = kwargs.pop("table_names")
        label_names = kwargs.pop("label_names")
        relationship_vals_list = kwargs.pop("relationship_vals_list")
        recording_set_names = kwargs.pop("recording_set_names")
        decode_cov_fr_vec_param_names = kwargs.pop("decode_cov_fr_vec_param_names")

        if len(np.unique([len(x) for x in [
            table_names, label_names, relationship_vals_list, decode_cov_fr_vec_param_names]])) != 1:
            raise Exception(f"These must all be same length: "
                            f"table_names, label_names, relationship_vals_list, decode_cov_fr_vec_param_names")

        param_sets = []
        # Loop through table names (and corresponding label names and relationship vals)
        for table_name, label_name, relationship_vals, decode_cov_fr_vec_param_name in zip(
                table_names, label_names, relationship_vals_list, decode_cov_fr_vec_param_names):

            # Make copy of kwargs so that updates dont carry over from one for loop iteration to the next
            key = copy.deepcopy(kwargs)

            # Add table_name, label name and relationship vals to key
            key.update(
                {"table_name": table_name, "label_name": label_name, "relationship_vals": relationship_vals,
                 "decode_cov_fr_vec_param_name": decode_cov_fr_vec_param_name
                 })

            # Add to key param names for several firing rate vector covariate tables
            # ...Get table
            table = get_table(table_name)
            # ...Add default params
            default_params = table().get_default_table_entry_params()
            key = add_defaults(key, default_params, add_nonexistent_keys=True)
            # ...Get param names for several firing rate vector covariate tables
            obj = table()._get_param_names_obj(**key)
            # ...Update key with params not present in table
            default_cov_params = {
                obj.meta_param_name: obj.param_name, obj.upstream_meta_param_name: obj.upstream_param_name}
            key = add_defaults(key, default_cov_params, add_nonexistent_keys=True)

            # Loop through recording set names
            for recording_set_name in recording_set_names:
                key.update({"recording_set_name": recording_set_name})

                # Append param set to list
                param_sets.append(copy.deepcopy(key))

        # Return param sets

        return param_sets