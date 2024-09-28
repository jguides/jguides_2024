import copy
from collections import namedtuple

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_subject_id, get_val_pairs
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecAveSelBase, PathWellPopSummBase, \
    PopulationAnalysisParamsBase, PopulationAnalysisSelBase, PathWellFRVecSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import delete_, insert_analysis_table_entry, \
    insert1_print, get_table_secondary_key_names
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_nwb_file_name_epochs_description
from src.jguides_2024.firing_rate_vector.jguidera_path_firing_rate_vector import PathFRVec, PathFRVecParams
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, RecordingSet
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.position_and_maze.jguidera_ppt import PptParams
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptDigParams
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, BrainRegionUnitsCohortType
from src.jguides_2024.utils.df_helpers import df_filter_columns, df_from_data_list, df_pop
from src.jguides_2024.utils.hierarchical_bootstrap import hierarchical_bootstrap
from src.jguides_2024.utils.plot_helpers import format_ax, get_fig_axes
from src.jguides_2024.utils.vector_helpers import unpack_single_vector, unpack_single_element

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
        decode_path_fr_vec_param_name = "LDA_path_progression_loocv_correct_stay_trials"
        decode_path_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "path_progression",
            "path_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv"}
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
        for param_name, params in DecodePathFRVecParams().fetch():
            key_filter.update({"decode_path_fr_vec_param_name": param_name,
                          "path_fr_vec_param_name": params["path_fr_vec_param_name"]})
            potential_keys += super()._get_potential_keys(key_filter, populate_tables)

        return potential_keys

    def delete_(self, key, safemode=True):
        delete_(self, [DecodePathFRVec], key, safemode)

    def _get_cov_fr_vec_param_names(self):
        return ["correct_incorrect_stay_trials"]


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
    def _get_vector_df_subset_quantities(vector_df, label):
        iloc_idxs = np.where(vector_df["label"] == label)[0]
        df_subset = vector_df.iloc[iloc_idxs]

        vectors = np.vstack(df_subset.vector)
        classes = np.asarray(df_subset.x)
        epoch_trial_numbers = df_subset.epoch_trial_number

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
        else:
            raise Exception(f"classifier_name {classifier_name} not recognized")

        # Train model using training vectors and classes
        clf.fit(train_vectors, train_classes)

        # Predict classes based on test vectors
        return clf.predict(test_vectors)

    def get_cmat_df(self, key):

        # Get df with confusion matrices
        # Separate function for this to be able to easily check make function computations outside of populate

        # Get params
        decode_cov_fr_vec_params = (self._get_params_table() & key).fetch1("decode_path_fr_vec_params")
        cov_fr_vec_param_name = decode_cov_fr_vec_params["path_fr_vec_param_name"]
        decode_var = decode_cov_fr_vec_params["decode_var"]
        cross_validation_method = decode_cov_fr_vec_params["cross_validation_method"]

        # Define condition comparisons based on param name
        valid_names = None
        if cov_fr_vec_param_name == "correct_incorrect_stay_trials":
            if decode_var == "path_progression":
                valid_names = [
                    "same_path_correct_correct_stay_trials", "same_turn_correct_correct_stay_trials",
                    "inbound_correct_correct_stay_trials", "outbound_correct_correct_stay_trials",
                    "different_turn_well_correct_correct_stay_trials"]

        # Raise error if valid names not defined (params not accounted for)
        if valid_names is None:
            raise Exception(f"{cov_fr_vec_param_name} not accounted for in code to define condition comparisons")

        # Define epochs (for now coded for one epoch)
        epochs = (EpochsDescription & key).fetch1("epochs")

        # Get map from path name pair types to path name pairs
        include_reversed_pairs = False
        path_name_pair_types_map = dict()
        for epoch in epochs:
            path_name_pair_types_map.update(MazePathWell.get_path_name_pair_types_map(
                nwb_file_name=key["nwb_file_name"], epoch=epoch, valid_names=valid_names,
                include_reversed_pairs=include_reversed_pairs))

        # Get valid bin numbers
        valid_bin_nums = (PptDigParams & key).get_valid_bin_nums()

        # Get firing rate vectors and covariate
        vector_df_name = "vector_df"
        concat_df = self._fr_vec_table()().concat_across_epochs(key, vector_df_name)

        # Omit invalid x
        vector_df = concat_df[concat_df.isin({"x": valid_bin_nums}).x]

        # Define order of train and test epochs
        epoch_train_test_orders = np.arange(0, len(np.unique(epochs)))

        # Define epochs
        if len(epochs) == 1:
            epoch_1 = epoch_2 = unpack_single_element(epochs)
        else:
            epoch_1, epoch_2 = epochs
            # Raise error until fully code up multiple epochs case
            raise Exception(f"Code currently not equipped to run with multiple epochs")

        # Decode and get confusion matrices
        # TODO: figure out if want to keep train_test_order in df. Should either
        #  note which using df entries should averaged using train/test order here,
        #  or later on.
        data_list = []
        for label_pair_name, label_pairs in path_name_pair_types_map.items():
            for label_1, label_2 in label_pairs:
                for label_train_test_order in np.arange(0, len(np.unique([label_1, label_2]))):
                    for epoch_train_test_order in epoch_train_test_orders:

                        # Define train/test epochs and labels based on order

                        # EPOCHS
                        # First order
                        train_epoch = epoch_1
                        test_epoch = epoch_2
                        # Second order: reverse train and test
                        if epoch_train_test_order == 1:
                            train_epoch = epoch_2
                            test_epoch = epoch_1

                        # LABELS
                        # First order
                        train_label = label_1
                        test_label = label_2
                        # Second order: reverse train and test
                        if label_train_test_order == 1:
                            train_label = label_2
                            test_label = label_1

                        # If LDA: require at least two instances of class for training.
                        # This means if using leave one out cross validation (for same train/test
                        # label), require three total classes. Otherwise require two total classes.
                        min_num_classes = 1
                        classifier_name = decode_cov_fr_vec_params["classifier_name"]
                        if classifier_name == "linear_discriminant_analysis":
                            if label_1 == label_2 and cross_validation_method == "loocv":
                                min_num_classes = 3
                            else:
                                min_num_classes = 2

                        # Continue if invalid for decoding
                        invalid_for_decode = False
                        for label in [label_1, label_2]:
                            df_subset = df_filter_columns(vector_df, {"label": label})
                            if len(df_subset) == 0:
                                invalid_for_decode = True
                            if any([np.sum(df_subset.x == x) < min_num_classes for x in np.unique(df_subset.x)]):
                                invalid_for_decode = True
                        if invalid_for_decode:
                            continue

                        # Define train/test sets for decoding

                        # Use cross validation if train and test labels are the same
                        if train_label == test_label and train_epoch == test_epoch:

                            # Leave one out cross validation
                            if cross_validation_method == "loocv":

                                data = self._get_vector_df_subset_quantities(vector_df, label)

                                unique_epoch_trial_numbers = np.unique(data.epoch_trial_numbers)

                                # Test each trial having trained on remaining trials
                                # We also need to track corresponding iloc_idxs so we can store
                                test_classes = []
                                predicted_labels = []
                                train_iloc_idxs = []
                                test_iloc_idxs = []
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

                                # Define number of trials used for training and testing, in total and for each
                                # training iteration
                                total_num_train_trials = total_num_test_trials = len(np.unique(data.epoch_trial_numbers))
                                num_train_trials_per_iteration = total_num_train_trials - 1  # subtract one since leave one out cross validation
                                num_test_trials_per_iteration = 1

                            else:
                                raise Exception(
                                    f"cross_validation_method {cross_validation_method} not accounted for in code")

                        # Otherwise train using all available data (one train/test set)
                        else:
                            # Get train vectors and classes
                            train_data = self._get_vector_df_subset_quantities(vector_df, train_label)

                            # Get test vectors and classes
                            test_data = self._get_vector_df_subset_quantities(vector_df, test_label)

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

                        original_eps_labels = get_eps_labels(epoch_1, epoch_2, label_1, label_2)
                        eps_labels = get_eps_labels(train_epoch, test_epoch, train_label, test_label)
                        data_list.append(
                            (label_pair_name, original_eps_labels, eps_labels, epoch_train_test_order,
                             label_train_test_order, train_label, test_label, train_epoch, test_epoch,
                             train_iloc_idxs, test_iloc_idxs, total_num_train_trials, total_num_test_trials,
                             num_train_trials_per_iteration, num_test_trials_per_iteration, cmat_classes, cmat))

        df_column_names = ["label_pair_name", "original_eps_labels", "eps_labels", "epoch_train_test_order",
                           "label_train_test_order", "train_label", "test_label", "train_epoch", "test_epoch",
                           "train_iloc_idxs", "test_iloc_idxs", "total_num_train_trials", "total_num_test_trials",
                           "num_train_trials_per_iteration", "num_test_trials_per_iteration",
                           "confusion_matrix_classes", "confusion_matrix"]

        return df_from_data_list(data_list, df_column_names)

    def make(self, key):

        # Get df with confusion matrices
        cmat_df = self.get_cmat_df(key)

        # Drop train and test iloc_idxs, as these columns may have ragged arrays which unfortunately cant be stored in
        # nwb file
        cmat_df.drop(columns=["train_iloc_idxs", "test_iloc_idxs"], inplace=True)

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

    def plot_performance(self):
        # Plot performance

        # Get performance information
        df = self.fetch1_performance_df()

        # Average across label pair instances (e.g. same turn paths), train/test order, and label pair order (e.g.
        # whether path 1 or path 2 comes first)
        ave_across_column_names = [
            "original_labels_name", "epoch_train_test_order", "label_train_test_order",
            "total_num_train_trials", "total_num_test_trials",
            "num_train_trials_per_iteration", "num_test_trials_per_iteration"]
        groupby_cols = [x for x in df.columns if x not in ave_across_column_names + ["val"]]
        df = df.groupby(groupby_cols).mean(numeric_only=True).reset_index()
        # appears that character columns already dropped, and so requires only specifying numerical columns to drop
        df = df.drop(columns=["epoch_train_test_order", "label_train_test_order"])

        # Plot

        # Initialize figure
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
            "confusion_matrix", "train_classes", "test_classes", "train_label", "test_label", "train_epoch",
            "test_epoch")):

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

        df = df_from_data_list(data_list, keep_df_columns + ["val"])

        decode_path_fr_vec_param_name = self.fetch1("decode_path_fr_vec_param_name")

        if decode_path_fr_vec_param_name == "LDA_path_progression_loocv_correct_stay_trials":

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

    def get_concat_metric_df(self, keys):

        df_concat = None
        for key in keys:

            # Get data for this table entry
            df = (self & key).fetch1_performance_df(drop_column_names=None)

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
            rename_columns = {"x": "x_val", "label_pair_name": "relationship"}
            for x, y in rename_columns.items():
                df[y] = df[x]
            df.drop(columns=list(rename_columns.keys()), inplace=True)

            # ...Reorder columns
            other_summary_table_columns = [
                "brain_region", "subject_id", "nwb_file_name", "epochs_description", "x_val", "val",
                 "brain_region_units_param_name", "relationship", "original_eps_labels",
                "nwb_file_name_epochs_description"]
            additional_columns = ['label_train_test_order', 'epoch_train_test_order',
             'total_num_train_trials', 'total_num_test_trials',
             'num_train_trials_per_iteration', 'num_test_trials_per_iteration']
            df = df[other_summary_table_columns + additional_columns]

            # Concatenate to previous dfs
            if df_concat is None:
                df_concat = df
            else:
                df_concat = pd.concat((df_concat, df))

        return df_concat


def get_eps_labels(epoch_1, epoch_2, label_1, label_2):
    return "^".join([str(x) for x in [epoch_1, epoch_2, label_1, label_2]])


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
class DecodePathFRVecSummParams(PopulationAnalysisParamsBase):
# class DecodePathFRVecSummParams(dj.Manual):  # use when initially generating table; if not cannot update table later
    definition = """
    # Parameters for DecodePathFRVecSumm
    decode_path_fr_vec_summ_param_name : varchar(200)
    ---
    decode_path_fr_vec_summ_params : blob
    """

    # Had to shorten name to comply with mysql requirements
    class BRUCohortType(dj.Part):
        definition = """
        # Achieves dependence on BrainRegionUnitsCohortType
        -> BrainRegionUnitsCohortType
        -> DecodePathFRVecSummParams
        """

    def insert_defaults(self, **kwargs):

        # Get names of brain_region_units_cohort_type
        brain_region_units_cohort_types = self._default_brain_region_units_cohort_types()

        # Get names for bootstrap param sets
        boot_set_names = self._boot_set_names()

        for boot_set_name in boot_set_names:

            for brain_region_units_cohort_type in brain_region_units_cohort_types:

                param_name = f"{boot_set_name}^{brain_region_units_cohort_type}"

                decode_path_fr_vec_summ_params = {
                    "boot_set_name": boot_set_name, "brain_region_units_cohort_type": brain_region_units_cohort_type}

                # Insert into table
                self.insert1({"decode_path_fr_vec_summ_param_name": param_name,
                              "decode_path_fr_vec_summ_params": decode_path_fr_vec_summ_params}, skip_duplicates=True)

    def _boot_set_names(self):
        return super()._boot_set_names() + self._valid_brain_region_diff_boot_set_names()

    def get_params(self):
        return self.fetch1("decode_path_fr_vec_summ_params")


@schema
class DecodePathFRVecSummSel(PopulationAnalysisSelBase):
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

    def _get_param_name_map(self, key_filter, brain_region_units_cohort_types):
        # Define summary table param name based on recording set name.
        # To do this, define map from recording set name and brain_region_units_cohort_type to summary table param names

        params_table = self._get_params_table()()

        recording_set_names_boot_set_names = [

             # Rat cohort
             (RecordingSet().lookup_rat_cohort_set_name(), boot_set_name)
             for boot_set_name in self._default_cohort_boot_set_names()] + [

             # Non rat cohort
             (recording_set_name, boot_set_name) for recording_set_name in
             RecordingSet().get_recording_set_names(
                 key_filter, ["Haight_rotation"])
             for boot_set_name in self._default_noncohort_boot_set_names()]

        param_name_map = dict()
        for recording_set_name, boot_set_name in recording_set_names_boot_set_names:
            for brain_region_units_cohort_type in brain_region_units_cohort_types:
                param_name_map_key = self._format_param_name_map_key(recording_set_name, brain_region_units_cohort_type)
                if param_name_map_key not in param_name_map:
                    param_name_map[param_name_map_key] = []

                params_table_keys = params_table.fetch("KEY")
                param_names = [
                    key[params_table.meta_param_name()] for key in params_table_keys if all([(
                            params_table & key).get_params()[x] == y for x, y in zip([
                        "boot_set_name", "brain_region_units_cohort_type"], [
                        boot_set_name, brain_region_units_cohort_type])])]

                param_name_map[param_name_map_key] += list(param_names)

        return param_name_map

    @staticmethod
    def _format_param_name_map_key(recording_set_name, brain_region_units_cohort_type):
        return (recording_set_name, brain_region_units_cohort_type)

    def _get_param_name_map_key(self, key, brain_region_units_cohort_type):
        # Make key to param name map given a set of parameters
        return self._format_param_name_map_key(key["recording_set_name"], brain_region_units_cohort_type)

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + ["brain_region_diff"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + ["brain_region_diff_rat_cohort"]

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
class DecodePathFRVecSumm(PathWellFRVecSummBase):
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

        # Concatenate performance dfs across entries

        upstream_keys = (self._get_selection_table() & key).fetch1("upstream_keys")

        df_concat = self._upstream_table()().get_concat_metric_df(upstream_keys)

        # Drop columns
        drop_columns = [
            "total_num_train_trials", "total_num_test_trials", "num_train_trials_per_iteration",
            "num_test_trials_per_iteration"]
        df_concat.drop(columns=drop_columns, inplace=True)

        # Average across train/test order
        average_across_cols = ["label_train_test_order", "epoch_train_test_order"]
        groupby_cols = [x for x in df_concat.columns if x not in average_across_cols + ["val"]]
        metric_df = df_concat.groupby(groupby_cols).mean().reset_index().drop(
            columns=average_across_cols)

        # Hierarchical bootstrap and sample mean

        # ...Define bootstrap params as indicated
        params_table_subset = (self._get_params_table()() & key)
        bootstrap_params = params_table_subset.get_boot_params()
        boot_set_name = params_table_subset.get_params()["boot_set_name"]

        # average values
        if boot_set_name in ["default", "default_rat_cohort"]:
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = ["nwb_file_name_epochs_description", "original_eps_labels", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = ["x_val", "brain_region"]
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

        # Raise exception if boot set name not accounted for in code
        else:
            raise Exception(f"Have not written code for boot_set_name {boot_set_name}")

        # Perform bootstrap
        boot_results = hierarchical_bootstrap(
            metric_df, resample_levels, "val", ave_group_column_names,
            num_bootstrap_samples_=bootstrap_params.num_bootstrap_samples, average_fn_=bootstrap_params.average_fn,
            alphas=bootstrap_params.alphas, debug_mode=False)

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

        # Insert into parts tables
        for upstream_key in (self._get_selection_table() & key).fetch1("upstream_keys"):
            self.Upstream.insert1({**key, **upstream_key})

    @staticmethod
    def _upstream_table():
        return DecodePathFRVec

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "correct_incorrect_stay_trials"

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        # Return default params
        return params

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

    def _get_x_lims(self):
        return [0, 10]

    def _get_xticks(self):
        return [0, .5, 1]

    def _get_val_lims(self, **kwargs):
        # Get a set range for value, e.g. for use in plotting value on same range across plots
        params_table = self._get_params_table()()
        boot_set_name = self.get_upstream_param("boot_set_name")
        if boot_set_name in params_table._valid_brain_region_diff_boot_set_names():
            return [-.5, .5]
        return [0, 1]

    def _get_x_text(self):
        return "Path fraction"

    def _get_val_text(self):
        return "Decode performance"

    def _get_vals_index_name(self):
        return "x_val"

    def get_ordered_relationships(self):

        return [
            'different_turn_well_correct_correct_stay_trials', 'inbound_correct_correct_stay_trials',
            'outbound_correct_correct_stay_trials', 'same_path_correct_correct_stay_trials',
            'same_turn_correct_correct_stay_trials'
        ]
