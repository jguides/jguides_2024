import copy

import datajoint as dj
import numpy as np
import spyglass as nd
import matplotlib.pyplot as plt

from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecAveSelBase, CovariateFRVecAveSummSelBase, CovariateFRVecAveSummParamsBase, PathWellPopSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import delete_, insert_analysis_table_entry, \
    insert1_print
from src.jguides_2024.firing_rate_vector.jguidera_path_firing_rate_vector import PathFRVec, PathFRVecParams
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, RecordingSet
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.position_and_maze.jguidera_ppt import PptParams
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptDigParams
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, BrainRegionUnitsCohortType
from src.jguides_2024.utils.df_helpers import df_filter_columns, df_from_data_list, df_pop
from src.jguides_2024.utils.plot_helpers import format_ax, get_fig_axes
from src.jguides_2024.utils.vector_helpers import unpack_single_vector

schema = dj.schema("jguidera_path_firing_rate_vector_decode")


# These imports are called with eval or used in table definitions (do not remove):
EpochsDescription
PptParams
PptDigParams
BrainRegionUnits
ResEpochSpikesSmParams
PathFRVec
nd
BrainRegionUnitsCohortType
PathFRVecParams
BrainRegionCohort
CurationSet


@schema
class DecodePathFRVecParams(SecKeyParamsBase):
# class DecodePathFRVecParams(dj.Manual):  # use when initially generating table; if not cannot update table later
    definition = """
    # Parameters for DecodePathFRVec
    decode_path_fr_vec_param_name : varchar(100)
    ---
    decode_path_fr_vec_params : blob
    """

    def insert_defaults(self, **kwargs):
        decode_path_fr_vec_param_name = "LDA_path_progression_even_odd_correct_stay_trials"
        decode_path_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "path_progression",
            "path_fr_vec_param_name": "even_odd_correct_incorrect_stay_trials"}
        self.insert1({
            "decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
            "decode_path_fr_vec_params": decode_path_fr_vec_params})


class DecodePathFRVecSelBase(CovariateFRVecAveSelBase):

    @staticmethod
    def _fr_vec_table():
        return PathFRVec


@schema
class DecodePathFRVecSel(DecodePathFRVecSelBase):
# class DecodePathFRVecSel(dj.Manual):  # use when initially generating table; if not cannot update table later

    definition = """
    # Selection from upstream tables for DecodePathFRVec 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> DecodePathFRVecParams
    zscore_fr : bool
    path_fr_vec_param_name : varchar(40)
    """

    # Override parent class method to further restrict potentials keys and limit path_fr_vec_param_name
    # to those defined in params for a given decode_path_fr_vec_param_name

    def _get_potential_keys(self, key_filter=None, populate_tables=False):

        key_filter = {"ppt_dig_param_name": "0.0625", "res_epoch_spikes_sm_param_name": "0.1", "zscore_fr": 0}

        potential_keys = []
        for param_name, params in DecodePathFRVecParams.fetch():
            key_filter.update({"decode_path_fr_vec_param_name": param_name,
                          "path_fr_vec_param_name": params["path_fr_vec_param_name"]})
            potential_keys += super()._get_potential_keys(key_filter, populate_tables)

        return potential_keys

    def delete_(self, key, safemode=True):
        delete_(self, [DecodePathFRVec], key, safemode)

    def _get_cov_fr_vec_param_names(self):
        return ["correct_incorrect_trials", "even_odd_correct_incorrect_stay_trials"]


@schema
class DecodePathFRVec(ComputedBase):
    definition = """
    # Decode covariate along paths
    -> DecodePathFRVecSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> DecodePathFRVec
        -> PathFRVec
        """

    @staticmethod
    def _fr_vec_table():
        return PathFRVec

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)

    @staticmethod
    def _get_vectors_classes(vector_df, label):
        df_subset = df_filter_columns(vector_df, {"label": label})
        return np.vstack(df_subset.vector), np.asarray(df_subset.x)

    @staticmethod
    def get_confusion_matrix(train_vectors, train_classes, test_vectors, test_classes, classifier_name):

        from sklearn.metrics import confusion_matrix

        # Initialize model
        if classifier_name == "linear_discriminant_analysis":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
        else:
            raise Exception(f"classifier_name {classifier_name} not recognized")

        # Train model using training vectors and classes
        clf.fit(train_vectors, train_classes)

        # Predict classes based on test vectors
        predicted_classes = clf.predict(test_vectors)

        # Compute confusion matrix
        classes = np.unique(np.concatenate((test_classes, predicted_classes)))
        cmat = confusion_matrix(test_classes, predicted_classes, labels=classes, normalize="true")
        if len(set([len(x) for x in cmat])) != 1:
            raise Exception

        return cmat, classes

    def make(self, key):

        # Define condition comparisons based on param name

        decode_cov_fr_vec_params = (self._get_params_table() & key).fetch1("decode_path_fr_vec_params")
        cov_fr_vec_param_name = decode_cov_fr_vec_params["path_fr_vec_param_name"]

        if cov_fr_vec_param_name == "even_odd_correct_incorrect_stay_trials":
            valid_names = [
                "same_path_even_odd_correct_stay_trials", "same_turn_even_odd_correct_stay_trials",
                "inbound_even_odd_correct_stay_trials", "outbound_even_odd_correct_stay_trials",
                "different_turn_well_even_odd_correct_stay_trials"]
        else:
            raise Exception(f"{cov_fr_vec_param_name} not accounted for in code to define condition comparisons")

        epoch = (EpochsDescription & key).get_epoch()
        path_name_pair_types_map = MazePathWell.get_path_name_pair_types_map(
            nwb_file_name=key["nwb_file_name"], epoch=epoch, valid_names=valid_names)

        # Define number of train/test orders based on param name
        num_train_test_orders = 1
        if "even_odd" in cov_fr_vec_param_name:
            num_train_test_orders = 2

        # Get valid bin numbers
        valid_bin_nums = (PptDigParams & key).get_valid_bin_nums()

        # Get firing rate vectors and covariate
        vector_df_name = "vector_df"
        concat_df = self._fr_vec_table()().concat_across_epochs(key, vector_df_name)

        # Decode and get confusion matrices
        data_list = []
        for label_pair_name, label_pairs in path_name_pair_types_map.items():
            for label_1, label_2 in label_pairs:
                for train_test_order in np.arange(0, num_train_test_orders):

                    # First order
                    train_label = label_1
                    test_label = label_2
                    # Second order: reverse train and test classes
                    if train_test_order == 1:
                        train_label = label_2
                        test_label = label_1

                    # Omit invalid x
                    vector_df = concat_df[concat_df.isin({"x": valid_bin_nums}).x]

                    # Require at least two instances of class if LDA
                    min_num_classes = 1
                    classifier_name = decode_cov_fr_vec_params["classifier_name"]
                    if classifier_name == "linear_discriminant_analysis":
                        min_num_classes = 2

                    # Continue if invalid for decoding
                    invalid_for_decode = False
                    for label in [label_1, label_2]:
                        df_subset = df_filter_columns(vector_df, {"label": label})
                        if any([np.sum(df_subset.x == x) < min_num_classes for x in np.unique(df_subset.x)]):
                            invalid_for_decode = True
                    if invalid_for_decode:
                        continue

                    # Get train vectors and classes
                    train_vectors, train_classes = self._get_vectors_classes(vector_df, train_label)

                    # Get test vectors and classes
                    test_vectors, test_classes = self._get_vectors_classes(vector_df, test_label)

                    cmat, cmat_classes = self.get_confusion_matrix(
                        train_vectors, train_classes, test_vectors, test_classes, classifier_name)

                    original_labels_name = f"{label_1}^{label_2}"
                    data_list.append((label_pair_name, original_labels_name, train_test_order, train_label, test_label,
                                      train_classes, test_classes, cmat_classes, cmat))

        df_column_names = ["label_pair_name", "original_labels_name", "train_test_order", "train_label", "test_label",
                           "train_classes", "test_classes", "confusion_matrix_classes", "confusion_matrix"]

        cmat_df = df_from_data_list(data_list, df_column_names)

        # Insert into main table
        insert_analysis_table_entry(self, [cmat_df], key)

        # Insert into part table
        for epoch in (EpochsDescription & key).fetch1("epochs"):
            key.update({"epoch": epoch})
            insert1_print(self.Upstream, key)

    def plot_ave_confusion_matrices(self):

        # Take average of confusion matrices across label pair instances

        df = self.fetch1_dataframe()

        label_pair_names = np.unique(df["label_pair_name"])

        data_list = []
        for label_pair_name in label_pair_names:
            df_subset = df_filter_columns(df, {"label_pair_name": label_pair_name})
            cmat_classes = unpack_single_vector(df_subset.confusion_matrix_classes.values)
            confusion_matrix_ave = np.mean(np.stack(df_subset.confusion_matrix), axis=0)

            # Store
            data_list.append((label_pair_name, confusion_matrix_ave))

        cmat_ave_df = df_from_data_list(data_list, ["label_pair_name", "confusion_matrix_ave"])

        # Plot
        num_rows = 1
        num_columns = len(label_pair_names)

        subplot_width = 2
        subplot_height = subplot_width*num_columns

        # Initialize figures
        fig, axes = get_fig_axes(num_rows, num_columns, subplot_width=subplot_width, subplot_height=subplot_height)

        for label_pair_name, ax in zip(label_pair_names, axes):
            confusion_matrix_ave = df_pop(cmat_ave_df, {"label_pair_name": label_pair_name}, "confusion_matrix_ave")

            ax.imshow(confusion_matrix_ave)

            format_ax(ax, title=label_pair_name.replace("_", " "), fontsize=6)

    def fetch1_performance_df(self):

        # Return df with performance information by test class label, separately for each class, and
        # without confusion matrix and train/test classes

        df = self.fetch1_dataframe()

        drop_column_names = ["confusion_matrix", "train_classes", "test_classes", "train_label", "test_label"]
        keep_df_columns = [x for x in df.columns if x not in drop_column_names]

        data_list = []
        for _, df_row in df.iterrows():
            diag = np.diagonal(df_row.confusion_matrix)
            data_list.append([df_row[x] for x in keep_df_columns] + [diag])

        df = df_from_data_list(data_list, keep_df_columns + ["val"])

        decode_path_fr_vec_param_name = self.fetch1("decode_path_fr_vec_param_name")

        if decode_path_fr_vec_param_name == "LDA_path_progression_even_odd_correct_stay_trials":

            # Flatten df so that each x has its own row
            keep_df_columns = [x for x in df.columns if x not in ["confusion_matrix_classes", "val"]]
            data_list = []
            for _, df_row in df.iterrows():
                cols = [getattr(df_row, z) for z in keep_df_columns]
                for x, val in zip(df_row.confusion_matrix_classes, df_row.val):
                    data_list.append(cols + [x, val])

            df = df_from_data_list(data_list, keep_df_columns + ["x", "val"])

        else:
            raise Exception(f"decode_path_fr_vec_param_name {decode_path_fr_vec_param_name} not accounted for")

        return df


"""
Notes on DecodePathFRVecSumm table setup:
- We want to combine entries across DecodePathFRVec, across nwb_file_names, epochs_description, 
and brain_region. For this reason, we want DecodePathFRVecSummSel to have all primary keys of DecodePathFRVec
except for nwb_file_name, epochs_description, brain_region, brain_region_units_param_name, and 
curation_name. 
  To specify the nwb_file_names and corresponding epochs_descriptions we want to combine across, we use recording_set.
  To specify the brain regions we want to combine across, we use brain_region_cohort. 
  To specify curation_name, we use curation_set_name.
  To specify brain region unit information, we use BrainRegionUnitsCohortType
- We include BrainRegionUnitsCohortType in DecodePathFRVecSummParams so that we can stay within the
limit on number of primary keys
"""


@schema
class DecodePathFRVecSummParams(CovariateFRVecAveSummParamsBase):
    definition = """
    # Parameters for DecodePathFRVecSumm
    decode_path_fr_vec_summ_param_name : varchar(200)
    ---
    metric_processing_name : varchar(40)  # describes additional processing on metric
    label_name : varchar(40)
    boot_set_name : varchar(120)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    def _boot_set_names(self):
        return super()._boot_set_names() + self._valid_brain_region_diff_boot_set_names()


@schema
class DecodePathFRVecSummSel(CovariateFRVecAveSummSelBase):
    definition = """
    # Selection from upstream tables for DecodePathFRVecSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> DecodePathFRVecParams
    zscore_fr : bool
    -> PathFRVecParams
    -> DecodePathFRVecSummParams
    ---
    upstream_keys : mediumblob
    """

    def _default_cov_fr_vec_param_names(self):
        return DecodePathFRVecSel()._get_cov_fr_vec_param_names()

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + [
            "brain_region_diff"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + [
            "brain_region_diff_rat_cohort"]

    def delete_(self, key, safemode=True):
        # If recording set name not in key but components that determine it are, then
        # find matching recording set names given the components, to avoid deleting irrelevant
        # entries
        key = copy.deepcopy(key)  # make copy of key to avoid changing outside function
        recording_set_names = RecordingSet().get_matching_recording_set_names(key)
        for recording_set_name in recording_set_names:
            key.update({"recording_set_name": recording_set_name})
            delete_(self, [], key, safemode)


@schema
class DecodePathFRVecSumm(PathWellPopSummBase):
# class DecodePathFRVecSumm(dj.Computed):  # use to initialize table
    definition = """
    # Summary of decodes of covariate along paths
    -> DecodePathFRVecSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> DecodePathFRVecSumm
        -> BrainRegionCohort
        -> CurationSet
        -> DecodePathFRVec
        """

    def make(self, key):
        raise Exception

    @staticmethod
    def _upstream_table():
        return DecodePathFRVec

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "even_odd_correct_incorrect_stay_trials"

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        params.update({"mask_duration": self._upstream_table()()._get_params_table()._default_mask_duration()})

        # Return default params
        return params
