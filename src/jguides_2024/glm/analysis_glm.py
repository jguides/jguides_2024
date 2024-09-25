import copy
import inspect
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import rgb2hex
from scipy.stats import chi2_contingency
from statannotations.Annotator import Annotator

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import format_brain_region
from src.jguides_2024.datajoint_nwb_utils.datajoint_fr_table_wrappers import PlotSTFRMap
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import \
    abbreviate_path_name as abbreviate_condition_name, get_epochs_id, \
    format_nwb_file_name
from src.jguides_2024.glm.get_glm_results_df import get_glm_results_df
from src.jguides_2024.glm.glm_helpers import get_glm_file_name_base, get_glm_params_text
from src.jguides_2024.glm.jguidera_el_net import ElNetParams
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionColor, BrainRegionCohort
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsParams, BrainRegionUnits
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellArrivalTrialsParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams
from src.jguides_2024.utils.df_helpers import df_filter1_columns, df_filter_columns, zip_df_columns, df_pop, \
    unpack_df_columns, \
    df_filter_columns_greater_than, df_filter_columns_isin, convert_categorical, df_from_data_list, \
    unique_df_column_sets
from src.jguides_2024.utils.dict_helpers import invert_dict, dict_comprehension, unpack_dicts, unpack_dict, \
    add_defaults
from src.jguides_2024.utils.hierarchical_bootstrap import hierarchical_bootstrap
from src.jguides_2024.utils.list_helpers import element_frequency, check_return_single_element
from src.jguides_2024.utils.plot_helpers import get_fig_axes, get_ax_for_layout, plot_heatmap, format_ax, \
    get_gridspec_ax_maps, \
    get_plot_idx_map, get_ticklabels, save_figure, return_n_cmap_colors, plot_violin_or_box
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.smooth_helpers import smooth_intervals
from src.jguides_2024.utils.string_helpers import format_bool
from src.jguides_2024.utils.vector_helpers import overlap_finite_samples, unpack_single_element, check_all_unique

"""
Currently set up to load results across units and models.
"""


class GLMContainer:

    def __init__(self, nwb_file_name, epochs, glm_restriction_df,  # varies across models
                 el_net_param_name,
                 curation_set_name="runs_analysis_v1", brain_region_cohort_name="all_targeted",
                 min_epoch_mean_firing_rate=.1, min_num_spikes=1, sigmas=None, similarity_metrics=None,
                 tolerate_missing_units=False, verbose=True):
        self.nwb_file_name = nwb_file_name
        self.epochs = epochs
        self.epochs_id = get_epochs_id(epochs)
        self.glm_restriction_df = glm_restriction_df
        self.el_net_param_name = el_net_param_name
        self.prediction_similarity_params = {"sigmas": sigmas, "similarity_metrics": similarity_metrics}
        self._get_inputs()
        self.glm_params = self._get_glm_params()
        self.units_params = self._get_units_params(
            curation_set_name, brain_region_cohort_name, min_epoch_mean_firing_rate)
        self.brain_regions = self._get_brain_regions()
        self.unit_name_df = self._get_unit_name_df()
        self.conditions_df = self._get_conditions_df()
        self.results_df = self._get_results_df(tolerate_missing_units)
        self.brain_region_color_map = self._get_brain_region_color_map()
        self._keep_elnet_df_columns = self._get_keep_elnet_df_columns()
        self.condition_spike_count_df = self._get_condition_spike_count_df()
        self.similarity_df = self._get_similarity_df(verbose)
        self.same_condition_val_average_similarity_df = self._get_same_condition_val_average_similarity_df()
        self.similarity_arr_df, self.similarity_arr_df_metadata_columns = self._get_similarity_arr_df()
        self.nnls = self._create_nnls(min_num_spikes)

    def _get_inputs(self):
        # Get inputs to class if not passed

        prediction_similarity_defaults = {"sigmas": [3, 5, 10], "similarity_metrics": [
            "correlation", "overlap", "normalized_overlap"]}
        add_defaults(self.prediction_similarity_params, prediction_similarity_defaults)

    def _get_glm_params(self):
        # Extract and store glm params based on datajoint table param names
        # Require a single time bin width across passed res_time_bins_pool_cohort_param_name

        time_bin_width = check_return_single_element([
            (ResTimeBinsPoolCohortParams &
             {"nwb_file_name": self.nwb_file_name, "epochs_id": self.epochs_id,
              "res_time_bins_pool_cohort_param_name": res_time_bins_pool_cohort_param_name}).get_time_bin_width()
            for res_time_bins_pool_cohort_param_name in self.glm_restriction_df[
                "res_time_bins_pool_cohort_param_name"]]).single_element
        params = (ElNetParams & {"el_net_param_name": self.el_net_param_name}).fetch1_params()
        return {"time_bin_width": time_bin_width, "alpha": params["alpha"], "L1_wt": params["L1_wt"]}

    def _get_units_params(self, curation_set_name, brain_region_cohort_name, min_epoch_mean_firing_rate):
        # Get brain_region_units_param_name from passed params

        if len(self.epochs) == 1:
            brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(
                self.nwb_file_name, unpack_single_element(self.epochs), min_epoch_mean_firing_rate
            )

        else:
            raise Exception(f"Need to write code to get brain_region_units_param_name for multi epoch case")

        return {"curation_set_name": curation_set_name, "brain_region_cohort_name": brain_region_cohort_name,
                "min_epoch_mean_firing_rate": min_epoch_mean_firing_rate, "brain_region_units_param_name":
                    brain_region_units_param_name}

    def _get_brain_regions(self):
        return (BrainRegionCohort & {
            "nwb_file_name": self.nwb_file_name,
            "brain_region_cohort_name": self.units_params["brain_region_cohort_name"]}).fetch1("brain_regions")

    def _get_unit_name_df(self):
        epochs_description = (EpochsDescription & {
            x: getattr(self, x) for x in ["nwb_file_name", "epochs_id"]}).fetch1("epochs_description")
        return BrainRegionUnits().get_unit_name_df(
            self.nwb_file_name, self.units_params["brain_region_units_param_name"],
            self.units_params["curation_set_name"], self.units_params["brain_region_cohort_name"],
            epochs_description)

    def _get_file_name_base(self, glm_restriction_idx=None):
        glm_restriction_name = None
        if glm_restriction_idx is not None:
            glm_restriction_name = self.glm_restriction_df.iloc[glm_restriction_idx].name
        return get_glm_file_name_base(self.nwb_file_name, self.epochs, self.glm_params["time_bin_width"],
                                      self.glm_params["alpha"], self.glm_params["L1_wt"], glm_restriction_name)

    def _get_glm_text_base(self, use_glm_param_names=None):
        # Turn specified glm param names from self.glm_params into text
        # Provides a way to string together a subset of glm params

        # Get inputs if not passed
        if use_glm_param_names is None:
            use_glm_param_names = [x.name for x in inspect.signature(get_glm_params_text).parameters.values()]
        # Get text with indicated glm params
        glm_params_text = get_glm_params_text(**{k: v for k, v in self.glm_params.items() if k in use_glm_param_names})
        # Return the above, concatenated
        return f"{format_nwb_file_name(self.nwb_file_name)}_{get_epochs_id(self.epochs)}_{glm_params_text}"

    def _convert_categorical(self, df):
        # Convert certain df columns to categorical datatype so have order information
        return convert_categorical(df, column_name="brain_region", categories=self.brain_regions)

    def _get_key(self, glm_restriction_idx=None, unit_name=None):
        # Return key for querying tables

        # Initialize key
        key = {"nwb_file_name": self.nwb_file_name, "epochs_id": self.epochs_id,
               "el_net_param_name": self.el_net_param_name,
               "curation_set_name": self.units_params["curation_set_name"]}

        # Add information related to glm restriction if glm restriction index passed
        if glm_restriction_idx is not None:
            glm_restriction = self.glm_restriction_df.loc[glm_restriction_idx]
            key.update({"res_time_bins_pool_cohort_param_name": glm_restriction.res_time_bins_pool_cohort_param_name,
                        "x_interp_pool_cohort_param_name": glm_restriction.x_interp_pool_cohort_param_name,
                        "train_test_split_pool_param_name": glm_restriction.train_test_split_pool_param_name})

        # Add information specific to unit if indicated
        if unit_name is not None:
            df_row = self.unit_name_df.loc[unit_name]
            for column_name in ["sort_group_id", "unit_id", "curation_name"]:
                key.update({column_name: df_row[column_name]})

        # Return key
        return key

    def _get_conditions_df(self):
        # Map from condition to instances and train/test order
        # Conditions include 1) potentially rewarded path names (for path traversal) and 2) potentially
        # rewarded well names (for delay period)

        conditions = ["path_names", "well_names"]
        condition_vals_list = [
            MazePathWell.get_same_turn_path_names_across_epochs(
                self.nwb_file_name, self.epochs, rewarded_paths=True, collapse=True, collapsed_path_order="left_right"),
            MazePathWell.get_well_names_across_epochs(
                self.nwb_file_name, self.epochs, rewarded_wells=True, well_order="left_right")]

        # Define ordering of conditions on 2D grid of similarity measures
        train_orders = [{x: idx for idx, x in enumerate(condition_vals)} for condition_vals in condition_vals_list]

        # Reverse order of train relative to test so that same condition indices fall along main diagonal when making
        # cross condition similarity matrix
        test_orders = [{k: v for k, v in zip(list(train_order.keys())[::-1],  # pair keys with values in reverse order
                                                  train_order.values())}
                        for train_order in train_orders]

        return pd.DataFrame.from_dict(
            {"condition": conditions, "condition_vals": condition_vals_list, "train_order": train_orders,
             "test_order": test_orders}).set_index("condition")

    def _get_results_df(self, tolerate_missing_units):
        arg_sets = [
            (glm_restriction_idx, unit_name, df_row, self._get_key(glm_restriction_idx, unit_name)) for
            glm_restriction_idx in self.glm_restriction_df.index for unit_name, df_row in self.unit_name_df.iterrows()]
        df = get_glm_results_df(arg_sets, tolerate_missing_units)
        if len(df) == 0:
            raise Exception(f"no data found")
        return self._convert_categorical(df)

    @staticmethod
    def _get_keep_elnet_df_columns():
        # Define information from elastic net df to hold onto in dataframe with prediction similarity results
        return ["unit_name", "brain_region", "glm_restriction_idx"]

    def _get_condition_spike_count_df(self):

        # Get spike counts during condition instances, so can restrict analyses based on spike count as indicated

        data_list = []
        for _, df_row in self.results_df.iterrows():
            folds_df = df_row.folds_df
            for test_condition_val, folds_df_row in folds_df.iterrows():
                data_list.append((df_row.unit_name, df_row.glm_restriction_idx, test_condition_val,
                                  np.sum(folds_df_row.y_test)))

        return df_from_data_list(
            data_list, ["unit_name", "glm_restriction_idx", "test_condition_val", "num_spikes"])

    def _get_brain_region_color_map(self):
        # Get map from target region to colors
        return BrainRegionColor().return_brain_region_color_map(self.brain_regions)

    def _get_glm_restriction_val(self, glm_restriction_idx, column_name):
        return self.glm_restriction_df.loc[glm_restriction_idx][column_name]

    def _get_glm_restriction_text(self, glm_restriction_idx):

        glm_restriction = self.glm_restriction_df.loc[glm_restriction_idx]

        spatial_overlap_text = format_bool(
            glm_restriction.prediction_similarity_restrictions["restrict_to_non_overlapping_space"],
            "no spatial overlap")

        def round_interval(x):
            return [np.round(x_i, 2) for x_i in x]

        covariate_restrictions_text = "_".join([f"{covariate_type}{round_interval(valid_range)}"
                                                for covariate_type, valid_range in
                                                glm_restriction.prediction_similarity_restrictions[
                                                    "covariate_restrictions"].items()])

        return f"{glm_restriction['name']} {covariate_restrictions_text} {spatial_overlap_text}"

    def _get_similarity_df(self, verbose):
        # Find similarity between actual and predicted firing rate

        data_list = []
        for _, df_row in self.results_df.iterrows():

            if verbose:
                print(f"on unit: {df_row.unit_name}, glm_restriction_idx: {df_row.glm_restriction_idx}...")

            # Get restrictions for later use
            prediction_similarity_restrictions = self.glm_restriction_df.loc[
                df_row.glm_restriction_idx].prediction_similarity_restrictions  # get restrictions

            # Get train and test conditions for this GLM run
            # ...Default is to include all pairs of train and test conditions in results
            train_conditions = np.unique(df_row.results_folds_merged_df["train_condition"])
            test_conditions = np.unique(df_row.results_folds_merged_df["test_condition"])
            train_test_conditions = [(train_condition, test_condition) for train_condition in train_conditions
                                     for test_condition in test_conditions]
            # ...Restrict to certain train / test pairs as indicated
            if "train_test_conditions" in prediction_similarity_restrictions:
                train_test_conditions = [x for x in train_test_conditions if x in
                                         prediction_similarity_restrictions["train_test_conditions"]]

            # Get similarity df
            for sigma in self.prediction_similarity_params["sigmas"]:
                for train_condition, test_condition in train_test_conditions:
                    # Unpack variables
                    df_subset = df_filter1_columns(
                        df_row.results_folds_merged_df,
                        {"train_condition": train_condition, "test_condition": test_condition})
                    y_test_predicted = df_subset.y_test_predicted.iloc[0]
                    y_test = df_row.folds_df.loc[test_condition].y_test
                    # Restrict based on covariate values as indicated
                    valid_bool = [True] * len(y_test)  # default
                    # ...apply restriction
                    y_test = y_test[valid_bool]
                    y_test_predicted = y_test_predicted[valid_bool]
                    # Get actual and predicted firing rates
                    fold_intervals = df_row.folds_df.loc[test_condition].fold_intervals
                    fr_list = [
                        smooth_intervals(y / self.glm_params["time_bin_width"], fold_intervals, sigma)
                        for y in [y_test, y_test_predicted]]
                    smoothed_values_list = [x.values for x in fr_list]

                    # Calculate similarity of actual and predicted firing rates if train and test conditions
                    # each had at least one spike. Otherwise set similarity metric to nan. Note that predicted firing
                    # rate can be very slightly non-zero even if spike counts were zero
                    # (TODO: figure out if this is just numerical imprecision of if there is a better explanation),
                    #  which can affect especially correlation calculation and give
                    # positive correlations that dont make sense. So helpful to enforce here only calculating when
                    # data from train and test sets.

                    # Loop through similarity metrics
                    for similarity_metric in self.prediction_similarity_params["similarity_metrics"]:

                        # Calculate metric if spikes from train and test set, otherwise set to nan
                        similarity_val = np.nan  # initialize
                        if all([np.sum(df_row.folds_df.loc[condition].y_test) > 0 for condition in
                                [train_condition, test_condition]]):
                            if similarity_metric == "correlation":
                                similarity_val = np.corrcoef(*smoothed_values_list)[0, 1]
                            elif similarity_metric == "overlap":
                                similarity_val = overlap_finite_samples(*fr_list)
                            elif similarity_metric == "normalized_overlap":
                                similarity_val = overlap_finite_samples(*fr_list, normalized=True)

                        # Store in list with identifiers
                        # ...Keep select identifiers from elastic net container
                        elnet_df_vals = [getattr(df_row, k) for k in self._keep_elnet_df_columns]
                        data_list.append(
                            tuple(elnet_df_vals + [
                                train_condition, test_condition, sigma, similarity_metric, similarity_val]))

        return df_from_data_list(
            data_list, self._keep_elnet_df_columns + [
                "train_condition", "test_condition", "sigma", "similarity_metric", "similarity_val"])

    def _restricted_similarity_df(self, condition):
        # Restrict train / test conditions in similarity_df (one pair per row) to those in conditions_df (potentially
        # rewarded paths / wells)
        valid_condition_vals = self.conditions_df.loc[condition].condition_vals
        return df_filter_columns_isin(
            self.similarity_df, {"test_condition": valid_condition_vals, "train_condition": valid_condition_vals})

    def _get_similarity_arr_df(self):
        # Place similarity measures into array in desired order (so that "same condition" pairs on main diagonal)
        # Restrict to potentially rewarded paths, wells

        # Get unique sets of identifiers in similarity df, except for similarity value and train and test conditions
        column_names = [
            x for x in self.similarity_df.columns if x not in ["similarity_val", "train_condition", "test_condition"]]
        similarity_keys = unique_df_column_sets(self.similarity_df, column_names, as_dict=True)

        # Put similarity measurements into arrays (store these as dfs)
        data_list = []
        for similarity_key in similarity_keys:

            # Initialize array with nans for similarity measures
            condition = self._get_glm_restriction_val(similarity_key["glm_restriction_idx"], "condition")
            train_order, test_order = unpack_df_columns(
                self.conditions_df.loc[[condition]], ["train_order", "test_order"], pop_single_entries=True)
            similarity_arr = np.empty((len(test_order), len(train_order)))
            similarity_arr[:] = np.nan

            # Get subset of similarity df corresponding to potentially rewarded paths / wells
            similarity_df_subset = self._restricted_similarity_df(condition)
            similarity_df_subset = df_filter_columns(similarity_df_subset, similarity_key)

            # Fill array
            # ...Get unique train/test conditions
            train_test_conditions = set(list(zip_df_columns(
                similarity_df_subset, ["train_condition", "test_condition"])))

            # Initialize vectors for rows (train conditions) and columns (test conditions) of array
            column_names = np.asarray([None]*len(set(similarity_df_subset["train_condition"])))
            row_names = np.asarray([None]*len(set(similarity_df_subset["test_condition"])))
            for train_condition, test_condition in train_test_conditions:
                test_condition_idx = test_order[test_condition]
                train_condition_idx = train_order[train_condition]
                similarity_arr[test_condition_idx, train_condition_idx] = df_pop(
                    similarity_df_subset, {"train_condition": train_condition, "test_condition": test_condition},
                    "similarity_val")

                # Update column and row names
                column_names[test_condition_idx] = test_condition
                row_names[train_condition_idx] = train_condition

            # Convert array to dataframe
            similarity_arr = pd.DataFrame(similarity_arr, columns=column_names, index=row_names)

            # Store similarity array and similarity key
            data_list.append(tuple(list(similarity_key.values()) + [similarity_arr]))

        # Store in dataframe
        df = df_from_data_list(data_list, list(similarity_key.keys()) + ["similarity_arr"])

        # Convert certain df columns to categorical
        df = self._convert_categorical(df)

        # Keep track of "metadata columns": all those except for ones computed here
        metadata_columns = [x for x in df.columns if x != "similarity_arr"]

        return df, metadata_columns

    def _get_same_condition_val_average_similarity_df(self):
        # For each unit and in each glm restriction, get average similarity metric across condition values,
        # restricted to potentially rewarded paths / wells

        # Get variable sets to iterate over
        column_names = ["unit_name", "brain_region", "glm_restriction_idx", "sigma", "similarity_metric"]
        df_keys = unique_df_column_sets(self.similarity_df, column_names=column_names, as_dict=True)

        data_list = []
        for df_key in df_keys:

            # Get subset of similarity df corresponding to potentially rewarded paths / wells
            condition = self._get_glm_restriction_val(df_key["glm_restriction_idx"], "condition")
            similarity_df_subset = self._restricted_similarity_df(condition)

            # Restrict using df_key
            similarity_df_subset = df_filter_columns(similarity_df_subset, df_key)

            # Filter for cases where train and test conditions are same
            valid_bool = similarity_df_subset["test_condition"] == similarity_df_subset["train_condition"]
            similarity_df_subset = similarity_df_subset[valid_bool]

            # Check that condition values all represented in df
            condition = self.glm_restriction_df.loc[df_key["glm_restriction_idx"]].condition
            check_membership(
                self.conditions_df.loc[condition].condition_vals, similarity_df_subset.test_condition.values,
                "expected conditions", "conditions present in similarity_df_subset")

            # Take average of similarity values within units
            similarity_vals = similarity_df_subset.set_index("unit_name").similarity_val
            for unit_name in np.unique(similarity_vals.index):
                data_list.append(tuple(list(df_key.values()) + [np.nanmean(similarity_vals.loc[unit_name])]))

        return df_from_data_list(data_list, column_names + ["average_similarity_val"])

    def _create_nnls(self, min_num_spikes):
        return self._Nnls(self, min_num_spikes)

    class _Nnls:

        """
        Non-negative least squares analysis

        Terminology:
        Condition: A task element that can take on multiple values. Example: "well_name" can take on the values "left",
                   "right", and "center".
        Condition value: A value taken on by a condition. Example: "left" is a value of the condition "well_name".
        Condition combination: a subset of condition values.
        k: Number of values in a combination.
        k-combination: A combination of size k.
        k set size: How many values k can take on for a condition. Equal to the number of values in the condition.
                    Example: there are three values for the condition "well_name", so k_set_size for "well_name" is
                    three.
        set of k combinations: Set of combinations of size k. Example: the set of k=2 combinations for "well_name"
                               includes {"left", "right"}, {"left", "center"}, and {"right", "center"}.
        Basis: A set of basis vectors. One basis per condition. One basis vector for each possible combination of
               condition values. Vector represents perfect similarity among each pair of the condition values
               in the combination, and perfect dissimilarity among all other pairs of condition values.
               Example: for the condition "well_name" and k=2, there are three basis vectors: one representing perfect
               similarity between the two wells in each of the three sets of two wells, and perfect dissimilarity
               within each of the other pairs of wells. For example, one of the bases corresponds to perfect
               similarity between "right" and "left" wells and perfect dissimilarity for all other pairs of wells.
        Cross condition similarity matrix: A matrix with outcomes of applying a similarity metric (e.g. correlation)
                                           on predicted and actual firing rates for models with a particular train
                                           condition (matrix rows) and test condition (matrix columns). This is stored
                                           in the parent class as "similarity_arr" (within "similarity_arr_df").
        Coefficients: Reflect contribution of basis vectors to reconstructing observed cross condition similarity
                      matrix. Fit with non-negative least squares.
        """

        def __init__(self, parent_obj, min_num_spikes):
            self.parent_obj = parent_obj
            self.min_num_spikes = min_num_spikes
            self.basis_obj = self._Basis()
            self.k_order_plot_num_maps = self._get_k_order_plot_num_maps()
            self.condition_coefficient_df = self._get_condition_coefficient_df()
            self.basis_estimation_df = self._get_basis_estimation_df()
            self.best_k_summary = self._get_best_k_summary()
            self.best_k_summary_stats = self._get_best_k_summary_stats()
            self.k_colors_df = self._get_k_colors_df()
            # TODO: could be good to have an object for coefficient types when computing metrics (maybe even for all
            #   metrics) with names, abbreviations, ylim

        class _Basis:
            """
            Basis for cross context generalization non-negative least squares analysis
            """

            def __init__(self):
                # Define k set sizes accounted for in the current code
                self.valid_k_set_sizes = [3, 4]
                self.basis_df = self._get_basis_df()

            @staticmethod
            def _get_k_set(k_set_size):

                # Get k set

                return np.arange(1, k_set_size + 1)

            def _get_basis_df(self):

                # Define map from size of set of k to:
                # 1) k_set: set of possible subset sizes
                # 2) num_k_combinations: number of combinations for each value of k
                # 3) basis vectors
                # 4) map from k to basis vector idxs

                # Map from k set size to set of k
                k_sets_map = {k_set_size: self._get_k_set(k_set_size) for k_set_size in self.valid_k_set_sizes}
                # Number of k-combinations for each k
                num_k_combinations = [np.asarray([sp.special.comb(k_set_size, k) for k in k_set]).astype(int)
                                      for k_set_size, k_set in k_sets_map.items()]
                # Map from k set size to matrix with basis vectors in columns
                basis_vectors_map = {
                    4: np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                                   [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                                   [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                                   [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                                   [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
                                   [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T,
                    3: np.asarray([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 1, 1, 0, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 0, 1, 1, 0],
                                   [1, 0, 1, 0, 0, 0, 1, 0, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1]]).T}

                # Map from k set size to map from k to column indices of basis vectors for k-combinations
                k_idxs_maps = [{k: np.where(np.sum(basis_vectors_map[k_set_size], axis=0) == (k ** 2))[0]
                                for k in k_sets_map[k_set_size]} for k_set_size in self.valid_k_set_sizes]

                # Return as dataframe
                return pd.DataFrame.from_dict(
                    {"k_set_size": self.valid_k_set_sizes, "k_set": k_sets_map.values(),
                     "num_k_combinations": num_k_combinations, "basis_vectors": [
                        basis_vectors_map[k] for k in self.valid_k_set_sizes], "k_idxs_map": k_idxs_maps}).set_index(
                    "k_set_size")

            def plot_basis_vectors(self, k_set_sizes=None, subplot_width=2, subplot_height=2, num_columns=5, pad=None,
                                   save_fig=False):
                # Plot basis vectors

                # Get inputs if not passed
                if k_set_sizes is None:
                    k_set_sizes = self.basis_df.k_set_size.values

                for k_set_size in k_set_sizes:
                    basis_vectors = self.basis_df.loc[k_set_size].basis_vectors
                    fig, axes = get_fig_axes(num_columns=num_columns, num_subplots=np.shape(basis_vectors)[1],
                                             subplot_width=subplot_width, subplot_height=subplot_height)
                    for idx, basis_vector in enumerate(basis_vectors.T):
                        ax = get_ax_for_layout(axes, idx)
                        new_shape = list(map(int, [np.sqrt(len(basis_vector))] * 2))
                        plot_heatmap(np.reshape(basis_vector, new_shape).T, fig_ax_list=[fig, ax], clim=[0, 1],
                                     plot_color_bar=False, edgecolors="silver")
                        format_ax(ax=ax, title=idx, fontsize=18)
                        ax.axis("off")
                    fig.tight_layout(pad=pad)
                    if save_fig:
                        save_figure(fig, f"basis_vectors_{k_set_size}", save_fig=save_fig)

        def _get_k_set_size(self, condition):
            # Return k set size for a given condition, where k set size is number of condition values
            return len(self.parent_obj.conditions_df.loc[condition].condition_vals)

        def _get_k_order_plot_num_maps(self):
            # Get map from k set size to map from ordering of k to plot number. This allows sorting bars
            # with group fractions by prominence of "k"

            basis_df = self.basis_obj.basis_df
            plot_order_map = [
                {v: idx for idx, v in enumerate(list(itertools.permutations(k_set))[::-1])}
                for k_set_size, k_set in basis_df["k_set"].items()]

            return pd.DataFrame.from_dict({"k_order_plot_num_map": plot_order_map,
                                           "k_set_size": basis_df.index}).set_index("k_set_size")

        def _get_condition_coefficient_df(self):
            # For each condition, get map from column index of each basis vector to condition instances
            # explained by basis vector

            data_list = []
            for condition in self.parent_obj.conditions_df.index:
                k_set_size = self._get_k_set_size(condition)
                basis_arr = self.basis_obj.basis_df.loc[k_set_size].basis_vectors
                column_idxs = np.arange(0, np.shape(basis_arr)[1])  # basis array column idxs
                for column_idx in column_idxs:  # basis array column indices
                    basis_vector = basis_arr[:, column_idx]
                    # Find row and column idxs where basis vector (in n by n matrix form) has value of 1
                    # ...Reshape the basis vector into n by n matrix
                    arr_shape = [int(np.sqrt(len(basis_vector)))] * 2
                    reshaped_basis_vector = np.reshape(basis_vector, arr_shape).T
                    # ...Find row/column pairs where value is one
                    row_idxs, column_idxs = np.where(reshaped_basis_vector)
                    # Consider entries above along diagonal to identify which conditions are represented as similar
                    # by basis. Rationale: if conditions are represented as similar by basis, the diagonal entries
                    # corresponding to these conditions must have value of one
                    conditions_df_subset = self.parent_obj.conditions_df.loc[condition]
                    train_names = np.asarray(
                        [unpack_single_element(invert_dict(conditions_df_subset.train_order)[idx])
                         for idx in column_idxs])  # columns
                    test_names = np.asarray(
                        [unpack_single_element(invert_dict(conditions_df_subset.test_order)[idx])
                         for idx in row_idxs])  # rows
                    # ...Keep diagonal entries: cases where row and column values are the same
                    condition_vals = train_names[train_names == test_names]
                    # ...Define basis vector coefficient name by joining abbreviated condition instances explained
                    # by basis vector
                    coefficient_name = "_".join(list(map(abbreviate_condition_name, condition_vals)))
                    data_list.append((condition, column_idx, condition_vals, coefficient_name))

            return df_from_data_list(data_list, ["condition", "column_idx", "condition_vals", "coefficient_name"])

        def _get_coeff_ecs(self, unit_name, glm_restriction_idx, coeff, min_num_spikes):
            # Get "ecs" ("explainable condition scaled") coefficients: coefficients
            # multiplied by the number of conditions they can explain that meet a minimum spikes threshold,
            # divided by total number of conditions meeting spikes threshold

            # Find condition instances used in nnls (limited to potentially rewarded paths / wells)
            # where unit fired at least one spike
            condition = self.parent_obj._get_glm_restriction_val(glm_restriction_idx, "condition")
            filter_key = {"unit_name": [unit_name], "glm_restriction_idx": [glm_restriction_idx],
                          "test_condition_val": self.parent_obj.conditions_df.loc[condition].condition_vals}
            spike_count_df_subset = df_filter_columns_isin(
                self.parent_obj.condition_spike_count_df, filter_key).set_index("test_condition_val")
            valid_bool = spike_count_df_subset.num_spikes >= min_num_spikes
            spike_condition_vals = spike_count_df_subset.index[valid_bool]

            # Loop through coefficients and scale each by number of valid condition instances they explain
            condition_coefficient_df_subset = df_filter_columns(
                self.condition_coefficient_df, {"condition": condition}).set_index("column_idx")

            # Check that coefficients represented no more than once in df subset
            check_all_unique(condition_coefficient_df_subset.index)

            # Loop through coefficients and scale by number of explainable condition values
            coeff_ecs = np.asarray([np.nan] * len(coeff))  # initialize
            for column_idx, df_row in condition_coefficient_df_subset.iterrows():
                # Take intersection of condition instances for which at least one spike from unit, and
                # condition instances that can be explained by current basis vector
                valid_condition_vals = set(spike_condition_vals) and set(df_row.condition_vals)
                coeff_ecs[column_idx] = coeff[column_idx] * len(valid_condition_vals)

            # Divide all scaled coefficients by total number of condition instances for which at least
            # one spike
            return coeff_ecs / len(spike_condition_vals)

        def _get_metrics(self, condition, coeff_, coeff_text=""):
            # Get metrics

            k_set_size = self._get_k_set_size(condition)

            # Sum coefficients for k-combinations within each k then sort sum across k (convert to df
            # so can easily sort)
            k_coeff_map = {
                k: np.nansum(coeff_[idxs]) for k, idxs in self.basis_obj.basis_df.loc[k_set_size].k_idxs_map.items()}
            k_coeff_df = pd.DataFrame.from_dict(
                {"k": k_coeff_map.keys(), "coeff_sum": k_coeff_map.values()}).set_index("k").sort_values(
                ["coeff_sum"], ascending=False)
            # TODO: will sort work if there are nans?

            # Use above ordering to get "plot number" where higher number means the ks which represent
            # greater generalization across conditions contribute more to cross condition similarity matrix
            # reconstruction
            plot_num = self.k_order_plot_num_maps.loc[k_set_size].k_order_plot_num_map[tuple(k_coeff_df.index)]

            # Define "best k" as that with highest coefficient sum
            best_k = k_coeff_df.index[0]

            # Find the absolute SUM of coefficients for k-combinations for the best k. Multiply this by -1
            # for latter half of set of k, so that can order cells by how generalized their firing is across
            # conditions (i.e. this requires that we reverse the ordering by magnitude for one half of the groups)
            mult_factor = 1  # default
            if best_k >= np.floor(k_set_size / 2):  # latter half of groups
                mult_factor = -1  # so that can reverse order
            best_k_coeff_sum_mult = k_coeff_df.loc[best_k].coeff_sum * mult_factor  # apply multiplication factor

            # Find FRACTION of summed coefficients corresponding to k-combinations for each k, and again
            # multiply by factor that lets us order cells by how generalized their firing is across conditions
            coeff_sum = np.nansum(list(k_coeff_map.values()))  # summed coefficients across all k
            best_k_coeff_prop_mult = (k_coeff_map[best_k] / coeff_sum) * -mult_factor

            # Revise keys in above map to include "k" and add passed text so we can unpack as separate metrics
            k_coeff_map = {f"k{k}coeff_sum": v for k, v in k_coeff_map.items()}

            # Return metrics in dictionary
            def add_coeff_text(text_list):
                return [f"{x}_{coeff_text}" for x in text_list]
            metric_names = add_coeff_text(
                list(k_coeff_map.keys()) + [
                    "plot_num", "best_k", "best_k_coeff_sum_mult", "best_k_coeff_prop_mult", "coeff_sum"])
            metrics = list(k_coeff_map.values()) + [
                plot_num, best_k, best_k_coeff_sum_mult, best_k_coeff_prop_mult, coeff_sum]
            return dict_comprehension(metric_names, metrics)

        def _get_basis_estimation_df(self):
            # 1) Estimate contribution of basis vectors representing extremes of degree of generalization
            # across conditions to reconstructing observed cross condition similarity matrix
            # 2) Compute metrics on the results

            # Make separate dfs for different glm restrictions, since glm restrictions can have different
            # numbers of condition instances, which results in different numbers of metric values
            similarity_arr_df = self.parent_obj.similarity_arr_df
            metrics_df_list = []  # metrics dfs across glm restrictions
            # Loop through GLM restrictions
            glm_restriction_idxs = list(set(similarity_arr_df.glm_restriction_idx))
            for glm_restriction_idx in glm_restriction_idxs:
                df_subset = df_filter_columns(similarity_arr_df, {"glm_restriction_idx": glm_restriction_idx})
                data_list = []
                for idx, (_, df_row) in enumerate(df_subset.iterrows()):
                    # 1) Estimate contribution of basis vectors to observed vector using nonnegative least squares:
                    # reconstruct observed cross condition similarity values through positive linear combination of
                    # basis vectors

                    # Get observed cross condition similarity values
                    obs_b = np.ndarray.flatten(df_row.similarity_arr.to_numpy().T)
                    # Convert nan (expected in cases of no cell firing) to zero
                    # TODO (p2): check that nans always cases of no cell firing
                    obs_b = np.asarray([x if not np.isnan(x) else 0 for x in obs_b])

                    # Get basis vectors for current condition
                    # ...First get condition corresponding to current restriction index
                    condition = self.parent_obj._get_glm_restriction_val(df_row.glm_restriction_idx, "condition")
                    # ...Use condition to get k set size
                    k_set_size = self._get_k_set_size(condition)
                    # ...Use k set size to get basis vectors
                    basis_df_subset = self.basis_obj.basis_df.loc[k_set_size]
                    basis_vectors = basis_df_subset.basis_vectors

                    # Fit basis vector coefficients
                    coeff, residual = sp.optimize.nnls(basis_vectors, obs_b)

                    # 2) Get the same set of metrics for three conditions:
                    # - Coefficients as they are
                    # - Coefficients scaled how many conditions they possibly apply to:
                    coeff_cs = np.asarray([np.nan] * len(coeff))  # initialize; cs stands for "condition scaled"
                    for k, idxs in basis_df_subset.k_idxs_map.items():
                        coeff_cs[idxs] = coeff[idxs] * k
                    # - Coefficients scaled by how many "explainable conditions" (conditions during which the cell
                    # fired at least one spike) they apply to, normalized by the number of "explainable conditions";
                    # "ecs" stands for "explainable condition scaled"
                    coeff_ecs = self._get_coeff_ecs(df_row.unit_name, glm_restriction_idx, coeff, self.min_num_spikes)
                    # ...Get metrics for each of these cases
                    coeff_list = [coeff, coeff_cs, coeff_ecs]  # list with coefficient types
                    coeff_text = ["", "cs", "ecs"]  # descriptor of coefficient type
                    metrics_dict_list = [self._get_metrics(condition, coeff_, coeff_text)
                                         for coeff_, coeff_text in zip(coeff_list, coeff_text)]
                    # ...Add additional metrics
                    coefficient_names = df_filter_columns(
                        self.condition_coefficient_df, {"condition": condition}).coefficient_name.values
                    metrics_dict_list.append({
                        **{"obs_b": obs_b, "max_obs_b": np.max(obs_b)}, **{
                            f"coeff_{coeff_text}": pd.Series(coeff_, index=coefficient_names)
                            for coeff_text, coeff_ in zip(coeff_text, coeff_list)}})
                    # ...String together metric names and values across dictionaries
                    metric_names, metrics = unpack_dicts(metrics_dict_list)

                    # Copy metadata columns from similarity_arr_df
                    identifiers = {k: getattr(df_row, k) for k in self.parent_obj.similarity_arr_df_metadata_columns}

                    # Store results
                    data_list.append(tuple(list(identifiers.values()) + list(metrics)))  # identifiers and metrics

                    # Ensure metric names not changing across df rows, so that can use last instance of metric names
                    # for all df rows
                    if idx > 0:
                        if not all(metric_names == previous_metric_names):
                            raise Exception(f"metric_names should not change while iterating over df rows")
                    previous_metric_names = copy.deepcopy(metric_names)

                # Return results and metrics in dataframe
                column_names = list(identifiers.keys()) + list(metric_names)
                # Convert certain columns to categorical
                df = df_from_data_list(data_list, column_names).set_index("unit_name")
                metrics_df_list.append(self.parent_obj._convert_categorical(df))

            return pd.DataFrame.from_dict({"glm_restriction_idx": glm_restriction_idxs,
                                           "metrics_df": metrics_df_list}).set_index("glm_restriction_idx")

        def _get_best_k_summary(self):
            # Get count and frequency of best_k across units within a target region

            df_tuple_column_names = ["brain_region", "glm_restriction_idx", "similarity_metric", "sigma"]
            # TODO: get rid of extra underscore in best_k_ upstream and pull all these column names from
            #  a more logical source
            best_k_names = ["best_k_", "best_k_cs", "best_k_ecs"]
            data_list = []
            for glm_restriction_idx, metrics_df in self.basis_estimation_df.metrics_df.items():
                # Get k set for current glm restriction
                # TODO: may want helper function to go from glm_restriction_idx or condition to k_set or
                #  another k related quantity
                condition = self.parent_obj._get_glm_restriction_val(glm_restriction_idx, "condition")
                k_set_size = self._get_k_set_size(condition)
                # ...Use k set size to get basis vectors
                k_set = self.basis_obj.basis_df.loc[k_set_size].k_set[::-1]
                # Iterate over sets of df column values
                df_keys = unique_df_column_sets(metrics_df, df_tuple_column_names, as_dict=True)
                for df_key in df_keys:
                    df_subset = df_filter_columns(metrics_df, df_key)
                    # Ensure units represented only once
                    check_all_unique(df_subset.index)
                    for best_k_name in best_k_names:
                        # Get counts of best_k across units
                        best_ks, best_k_counts = unpack_dict(element_frequency(df_subset[best_k_name], elements=k_set))
                        # Convert to frequency
                        best_k_freq = np.asarray(best_k_counts) / np.sum(best_k_counts)
                        # Store in df
                        best_k_df = pd.DataFrame.from_dict(
                            {"best_k": best_ks, "count": best_k_counts, "frequency": best_k_freq}).set_index("best_k")
                        # Store
                        data_list.append(
                            tuple([glm_restriction_idx] + list(df_key.values()) + [best_k_name, best_k_df]))

            entry_names = ["glm_restriction_idx"] + list(df_key.keys()) + ["best_k_name", "best_k_df"]

            return df_from_data_list(data_list, entry_names)

        def _get_best_k_summary_stats(self):

            df_tuple_column_names = ["best_k_name", "sigma", "similarity_metric"]
            df_keys = unique_df_column_sets(self.best_k_summary, df_tuple_column_names, as_dict=True)

            data_list = []
            for df_key in df_keys:
                df_subset = df_filter_columns(self.best_k_summary, df_key)
                for glm_restriction_idx in set(df_subset.glm_restriction_idx):
                    df_restriction_subset = df_filter_columns(
                        df_subset, {"glm_restriction_idx": glm_restriction_idx}).set_index("brain_region")
                    brain_region_pairs = list(itertools.combinations(df_restriction_subset.index, r=2))
                    for brain_region_pair in brain_region_pairs:
                        contingency_table = pd.concat(
                            [df_restriction_subset.loc[brain_region].best_k_df["count"] for brain_region in
                             brain_region_pair], axis=1).values.T
                        # Can only perform chi2 test if all categories have at least one count across brain regions
                        if any(np.sum(contingency_table, axis=0) == 0):
                            print(f"cannot calculate chi2 statistic for {brain_region_pair} because at least one"
                                  f"category has zero counts in both brain regions")
                            continue
                        _, p, _, _ = chi2_contingency(contingency_table)
                        data_list.append(tuple(list(df_key.values()) + [glm_restriction_idx, brain_region_pair, p]))
            column_names = df_tuple_column_names + ["glm_restriction_idx", "brain_region_pair", "p"]

            return df_from_data_list(data_list, column_names)

        def _get_k_colors_df(self):
            # Match k to colors

            squeeze_lower_range = 5
            squeeze_upper_range = 17

            data_list = []
            k_color_maps = {"path_names": {1: "Reds", 2: "Greens", 3: "Blues", 4: "Purples"},
                             "well_names": {1: "Reds", 2: "Greens", 3: "Blues"}}

            for condition, k_color_map in k_color_maps.items():
                squeeze_upper_range_change_map = {k: 0 for k in list(k_color_map.values())}  # initialize
                squeeze_upper_range_change_map.update({"Purples": -12, "Reds": 4})  # update
                k_set_size = self._get_k_set_size(condition)
                df_subset = self.basis_obj.basis_df.loc[k_set_size]
                for k, num_combinations in zip_df_columns(df_subset, ["k_set", "num_k_combinations"]):
                    cmap_name = k_color_map[k]
                    colors = return_n_cmap_colors(
                        num_colors=num_combinations, cmap=plt.get_cmap(cmap_name),
                        squeeze_lower_range=squeeze_lower_range,
                        squeeze_upper_range=squeeze_upper_range + squeeze_upper_range_change_map[cmap_name])[::-1]
                    for color, idx in zip(colors, df_subset.k_idxs_map[k]):
                        data_list.append((condition, k, idx, color))

            return df_from_data_list(data_list, ["condition", "k", "column_idx", "color"])

        def plot_coeff(self, brain_regions=None, glm_restriction_idx=None, sigma=None, similarity_metric=None,
                       num_columns=2, subplot_width=3, subplot_height=2, max_plots=5):
            # Plot nnls coefficients and similarity array for single units

            # Get inputs if not passed
            if brain_regions is None:
                brain_regions = self.parent_obj.brain_regions
            if glm_restriction_idx is None:
                glm_restriction_idx = self.parent_obj.glm_restriction_df.index[0]
            if sigma is None:
                sigma = self.parent_obj.prediction_similarity_params["sigmas"][0]
            if similarity_metric is None:
                similarity_metric = self.parent_obj._get_default_param("similarity_metric")

            # Plot
            plot_counter = 0
            for brain_region in brain_regions:
                df_subset = df_filter_columns(
                    self.basis_estimation_df.iloc[glm_restriction_idx].metrics_df, {"brain_region": brain_region})
                unit_names = df_subset.index
                for unit_name_idx, unit_name in enumerate(unit_names):

                    # Continue if reached plot number limit
                    if max_plots is not None:
                        if plot_counter > max_plots:
                            continue
                        plot_counter += 1

                    # Get data
                    filter_key = {"unit_name": unit_name, "glm_restriction_idx": glm_restriction_idx,
                                  "sigma": sigma, "similarity_metric": similarity_metric}
                    basis_estimation_df_subset = df_filter_columns(df_subset.reset_index(), filter_key)

                    # Continue if no result for this unit (can happen if nans upstream)
                    if len(basis_estimation_df_subset) == 0:
                        continue

                    coeff = basis_estimation_df_subset["coeff_ecs"].iloc[
                        0]  # unpack here instead of above to preserve series
                    similarity_arr = df_pop(self.parent_obj.similarity_arr_df, filter_key, "similarity_arr")

                    # Plot coefficients
                    fig, axes = plt.subplots(1, 2, figsize=(subplot_width * num_columns, subplot_height))
                    ax = axes[0]
                    ax.plot(coeff, 'o-', color=self.parent_obj.brain_region_color_map[brain_region])
                    xticks = np.arange(0, len(coeff))
                    xticklabels = get_ticklabels(xticks, label_every_n=5)
                    format_ax(ax=ax, xticks=xticks, xticklabels=xticklabels, ylim=[0, 1], title=unit_name)

                    # Plot similarity array
                    ax = axes[1]
                    plot_heatmap(similarity_arr, fig_ax_list=[fig, ax], clim=[0, 1], plot_color_bar=False)

        def _initialize_plot_bar_pie(self, glm_restriction_idxs, similarity_metrics, subplot_width, subplot_height,
                                     mega_row_gap_factor, mega_column_gap_factor, wspace, hspace,
                                     plot_brain_regions=None):

            # Initialize figure
            # Get unique combinations of quantities of interest
            # ...First, filter df using inputs
            # TODO: may be worth putting getting of basis tuples into separate helper function
            basis_tuples = []
            for glm_restriction_idx in glm_restriction_idxs:
                basis_estimation_df_subset = df_filter_columns_isin(
                    self.basis_estimation_df.loc[glm_restriction_idx].metrics_df,
                    {"similarity_metric": similarity_metrics})
                basis_tuples += list(set(list(zip_df_columns(basis_estimation_df_subset,
                                           ["glm_restriction_idx", "similarity_metric", "brain_region"]))))
            tuples_unpacked = list(map(np.unique, list(zip(*basis_tuples))))
            glm_restriction_idxs, similarity_metrics, brain_regions = tuples_unpacked

            # Order brain regions if indicated
            if plot_brain_regions is not None:
                brain_regions = [x for x in plot_brain_regions if x in brain_regions]

            mega_row_iterables = similarity_metrics
            mega_column_iterables = brain_regions
            row_iterables = glm_restriction_idxs
            column_iterables = [0]
            gs_map, ax_map, fig = get_gridspec_ax_maps(
                mega_row_iterables, mega_column_iterables, row_iterables, column_iterables,
                subplot_width=subplot_width, subplot_height=subplot_height, mega_row_gap_factor=mega_row_gap_factor,
                mega_column_gap_factor=mega_column_gap_factor, wspace=wspace, hspace=hspace, sharey=True)
            plot_idx_map = get_plot_idx_map(mega_row_iterables, mega_column_iterables, row_iterables, column_iterables)

            return (basis_tuples, gs_map, ax_map, fig, plot_idx_map)

        def _get_title_bar_pie(
                self, plot_idx_map, plot_key, glm_restriction_idx, similarity_metric, brain_region, include_title):

            title = ""
            if not include_title:
                return title
            if plot_idx_map[plot_key].column_idx == 0:
                glm_restriction_text = self.parent_obj._get_glm_restriction_text(glm_restriction_idx)
                title = f"{glm_restriction_text}, {similarity_metric}"
            if plot_idx_map[plot_key].row_idx == 0 and plot_idx_map[plot_key].mega_row_idx == 0:
                title = f"{brain_region}\n{title}"

            return title

        def plot_coeff_bar(self,
                           max_obs_b_threshold=0,  # only plot units with max observed b at least this much
                           sort_by=("best_k_ecs", "coeff_sum_ecs"),
                           quantity="coeff_ecs",
                           glm_restriction_idxs=None,
                           similarity_metrics=None,
                           plot_brain_regions=None,
                           sigma=None,
                           break_between_groups=True,
                           group_gap_factor=.05,
                           subplot_width=4,
                           subplot_height=2,
                           mega_row_gap_factor=.1,
                           mega_column_gap_factor=.05,
                           wspace=None,
                           hspace=None,
                           include_title=True,
                           include_xticklabels=False,
                           plot_scale_bar=False,
                           axis_off_list=(),
                           include_xticks=False,
                           tick_width=None,
                           spine_width=None,
                           save_fig=False):
            # Bar plot of nnls coefficients (or scaled coefficients)

            # Check inputs
            valid_sort_by = ["best_k", "best_k_cs", "best_k_ecs"]
            if break_between_groups and sort_by[0] not in valid_sort_by:
                raise Exception(f"If want to put break in plot around groups, first sort must be in {valid_sort_by}, "
                                f"but is {sort_by[0]}")

            # Get inputs if not passed
            plot_params = self.parent_obj._get_default_params(
                {"glm_restriction_idxs": glm_restriction_idxs, "similarity_metrics": similarity_metrics,
                 "sigma": sigma})

            # y limits and y ticks
            if "_cs" in quantity:
                ylim = [0, 4]  # TODO: this is temporary; get in better way
                yticks = np.arange(0, 4.2, .2)
            else:
                ylim = [0, 1]  # TODO: pull appropriate y limit from object for coeff sets
                yticks = [0, .2, .4, .6, .8, 1]

            # Get plot vars
            (basis_tuples, gs_map, ax_map, fig, plot_idx_map) = self._initialize_plot_bar_pie(
                plot_params["glm_restriction_idxs"], plot_params["similarity_metrics"], subplot_width, subplot_height,
                mega_row_gap_factor, mega_column_gap_factor, wspace, hspace, plot_brain_regions)

            for basis_tuple_idx, (glm_restriction_idx, similarity_metric, brain_region) in enumerate(basis_tuples):
                # Get plot objects
                plot_key = (similarity_metric, brain_region, glm_restriction_idx, 0)
                ax = ax_map[plot_key]

                # Get subset of df
                df_subset = df_filter_columns(
                    self.basis_estimation_df.loc[glm_restriction_idx].metrics_df, {
                        "brain_region": brain_region, "glm_restriction_idx": glm_restriction_idx,
                        "similarity_metric": similarity_metric, "sigma": plot_params["sigma"]})

                # Apply max obs b threshold
                df_subset = df_filter_columns_greater_than(df_subset, {"max_obs_b": max_obs_b_threshold})

                # Sort units as indicated
                df_sorted = df_subset.sort_values(list(sort_by))

                # Pull quantities of interest
                coeff_arr = np.vstack(df_sorted[quantity]).T  # units in columns, basis vector coefficients in rows
                unit_names = df_sorted.index

                # If break in plot between "best group" groups indicated, define size of break using number of units
                # that will be plotted
                group_gap = 0  # default
                if break_between_groups:
                    group_gap = group_gap_factor * len(unit_names)  # gap between groups

                # If first sort is by "best group" groups, define map from "best group" group to Series whose index is
                # units and values are boolean indicating whether unit has this "best group" as its best group
                coeff_arr_bool_map = {"all": [True] * np.shape(coeff_arr)[1]}  # default
                if "best_k" in sort_by[0]:
                    best_ks = df_sorted[sort_by[0]]
                    coeff_arr_bool_map = {group: best_ks == group for group in set(best_ks)}

                # Make bar plot. Approach: loop through groups as defined in coeff_arr_bool_map
                # and plot bars for units in each
                xticks = []  # keep track of x ticks
                condition = self.parent_obj._get_glm_restriction_val(glm_restriction_idx, "condition")
                for group_idx, (group_name, valid_bool) in enumerate(coeff_arr_bool_map.items()):
                    coeff_arr_subset = coeff_arr[:, valid_bool]  # units in columns, basis vector coefficients in rows
                    x_vals = np.where(valid_bool)[0] + group_idx * group_gap
                    for idx, row in enumerate(coeff_arr_subset):  # basis vector coefficients
                        color = df_pop(self.k_colors_df, {"condition": condition, "column_idx": idx}, "color")
                        ax.bar(x_vals, row, bottom=np.sum(coeff_arr_subset[:idx, :], axis=0), color=color)
                    xticks += list(x_vals)

                # Title (shorter if saving figure)
                title = None
                if plot_idx_map[plot_key].row_idx == 0:
                    title = self._get_title_bar_pie(plot_idx_map, plot_key, glm_restriction_idx, similarity_metric,
                                                    brain_region, include_title)
                    if save_fig:
                        title = format_brain_region(brain_region)

                # y label
                ylabel = ""
                if plot_idx_map[plot_key].mega_column_idx == 0 and plot_idx_map[plot_key].row_idx == 0:
                    ylabel = "Reliability metric"

                # Set x tick labels to unit names (and adjust tick label font size) if indicated
                xticklabels = []  # default
                fontsize = 28
                ticklabels_fontsize = 26
                title_fontsize = 40
                if include_xticklabels:
                    xticklabels = unit_names
                    ticklabels_fontsize = 7

                # x ticks
                if not include_xticks:  # overwrite with empty list if dont want to include
                    xticks = []

                # y tick labels
                yticklabels = []  # default
                if plot_idx_map[plot_key].mega_column_idx == 0:
                    yticklabels = [x if x in [0, .2, .4, .6, .8, 1] else "" for x in yticks]

                # Format axis
                ax.set_title(title, fontsize=title_fontsize)
                format_ax(ax=ax, ylabel=ylabel, fontsize=fontsize, xticks=xticks,
                          xticklabels=xticklabels, xticklabels_rotation=90, ticklabels_fontsize=ticklabels_fontsize,
                          yticks=yticks, yticklabels=yticklabels,
                          ylim=ylim, axis_off_list=axis_off_list,
                          tick_width=tick_width, spine_width=spine_width)

                # Plot scale bar if indicated
                if plot_scale_bar:
                    x_start, x_stop = ax.get_xlim()
                    x_extent = x_stop - x_start
                    shift_factor = .01 * x_extent
                    x_val = x_start + shift_factor
                    scalebar_start_y = .5
                    scalebar_extent = .5
                    scalebar_y_vals = [scalebar_start_y, scalebar_start_y + scalebar_extent]
                    ax.plot([x_val] * 2, scalebar_y_vals, linewidth=4, color="black")
                    ax.text(x_val - abs(shift_factor) * 5, scalebar_start_y + scalebar_extent / 3, str(scalebar_extent),
                            fontsize=fontsize, rotation=90)

            fig.tight_layout()
            similarity_metric_text = "_".join(similarity_metrics)
            figure_name = f"glm_nnls_bar_{self.parent_obj._get_glm_text_base()}_{similarity_metric_text}_sigma{sigma}"
            save_figure(fig, figure_name, save_fig=save_fig)

        def plot_coeff_pie(self,
                           glm_restriction_idxs=None,
                           max_obs_b_threshold=0,
                           similarity_metrics=None,
                           sigma=None,
                           best_k_name="best_k_ecs",
                           subplot_width=2,
                           subplot_height=2,
                           mega_row_gap_factor=.1,
                           mega_column_gap_factor=None,
                           wspace=None,
                           hspace=None,
                           include_title=True,
                           save_fig=False):
            # Pie chart with best group frequency among units

            # Get inputs if not passed
            plot_params = self.parent_obj._get_default_params({"glm_restriction_idxs": glm_restriction_idxs,
                                                    "similarity_metrics": similarity_metrics,
                                                     "sigma": sigma})

            # Initialize figure
            (basis_tuples, gs_map, ax_map, fig, plot_idx_map) = self._initialize_plot_bar_pie(
                plot_params["glm_restriction_idxs"],
               plot_params["similarity_metrics"],
               subplot_width,
               subplot_height,
               mega_row_gap_factor,
               mega_column_gap_factor,
               wspace,
               hspace)

            for glm_restriction_idx, similarity_metric, brain_region in basis_tuples:
                plot_key = (similarity_metric, brain_region, glm_restriction_idx, 0)
                ax = ax_map[plot_key]
                # Get subset of df
                df_subset = df_filter_columns(self.basis_estimation_df.loc[glm_restriction_idx].metrics_df,
                                              {"brain_region": brain_region,
                                               "glm_restriction_idx": glm_restriction_idx,
                                               "similarity_metric": similarity_metric,
                                               "sigma": plot_params["sigma"]})
                # Apply max obs b threshold
                df_subset = df_filter_columns_greater_than(df_subset, {"max_obs_b": max_obs_b_threshold})

                # Get groups
                condition = self.parent_obj._get_glm_restriction_val(glm_restriction_idx, "condition")
                k_set_size = self._get_k_set_size(condition)
                # ...Use k set size to get basis vectors
                k_set = self.basis_obj.basis_df.loc[k_set_size].k_set[::-1]  # reverse order in pie chart

                # Get frequency at which each k is best according to best_k_name
                best_k_frequency_map = element_frequency(df_subset[best_k_name],
                                                             elements=k_set)
                k_frequencies = [best_k_frequency_map[k] for k in k_set]

                # Get colors for groups
                colors = []
                for k in k_set:
                    k_colors_df_subset = df_filter_columns(self.k_colors_df, {"condition": condition,
                                                                              "k": k})
                    middle_idx = int(np.median(k_colors_df_subset["column_idx"].values))
                    colors.append(df_pop(k_colors_df_subset, {"column_idx": middle_idx}, "color"))
                # Pie
                ax.pie(k_frequencies, colors=colors, startangle=90, counterclock=False)
                # Title
                title = self._get_title_bar_pie(plot_idx_map, plot_key, glm_restriction_idx, similarity_metric,
                                                brain_region, include_title)
                format_ax(ax=ax, title=title, fontsize=8)

            figure_name = f"glm_nnls_pie_{self.parent_obj._get_glm_text_base()}_sigma{sigma}"
            save_figure(fig, figure_name, save_fig=save_fig)

        def plot_fr(
                self, glm_restriction_idx=None, brain_regions=None,
                firing_rate_bounds_type="IQR", fr_ppt_smoothed_param_name="0.05", fr_wa_smoothed_param_name="0.1",
                average_function=np.nanmedian, unit_names=None, main_fig=None, max_units_plot=3, plot_color_bar=False,
                populate_tables=True):
            # For individual units, plot single trial firing rate, average firing rate during conditions,
            # matrix with across path decoding goodness metric (similarity of actual and predicted smoothed
            # firing rates), and bar plot with nnls decomposition of that matrix as linear combination
            # of basis vectors representing different levels of path generalization

            # Get inputs if not passed
            if brain_regions is None:
                brain_regions = self.parent_obj.brain_regions
            if glm_restriction_idx is None:
                glm_restriction_idx = self.parent_obj.glm_restriction_df.index[0]
            if unit_names is None:
                unit_names = set(
                    df_filter_columns_isin(self.parent_obj.similarity_df, {"brain_region": brain_regions})[
                        "unit_name"])

            # Define time periods restriction
            glm_restriction = self.parent_obj.glm_restriction_df.iloc[glm_restriction_idx]
            if glm_restriction["name"] == "path":
                restrict_time_periods = ["path"]
            elif glm_restriction["name"] == "delay":
                restrict_time_periods = ["well_arrival"]

            # Define dio well arrival trials param name if indicated
            dio_well_arrival_trials_param_name = None  # default
            if glm_restriction["name"] == "delay":
                dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_delay_param_name()

            # Hard code params
            # Parameters for single trial firing rate maps
            height_ratios = np.asarray([1, 1])
            plot_performance_outcomes = False
            plot_environment_marker = False
            sharey = "columns"

            unit_counter = 0
            for unit_name in unit_names:

                # Continue if reached max unit limit
                unit_counter += 1
                if unit_counter > max_units_plot:
                    continue

                # Initialize figure if not passed
                if main_fig is None:
                    plot_width = 2 * self._get_k_set_size(
                        self.parent_obj.glm_restriction_df.iloc[glm_restriction_idx].condition)
                    figsize = (plot_width, 3)
                    main_fig = plt.figure(figsize=figsize, constrained_layout=True)

                # Initialize subfigures within main figure
                plot_obj = PlotSTFRMap(self.parent_obj.nwb_file_name, self.parent_obj.epochs, restrict_time_periods)

                # Plot firing rate maps
                plot_obj.plot_single_trial_firing_rate_map(
                    unit_names=[unit_name],
                    firing_rate_bounds_type=firing_rate_bounds_type,
                    plot_performance_outcomes=plot_performance_outcomes,
                    plot_environment_marker=plot_environment_marker,
                    fr_ppt_smoothed_param_name=fr_ppt_smoothed_param_name,
                    fr_wa_smoothed_param_name=fr_wa_smoothed_param_name,
                    dio_well_arrival_trials_param_name=dio_well_arrival_trials_param_name,
                    average_function=average_function, populate_tables=populate_tables, height_ratios=height_ratios,
                    sharey=sharey, fig=main_fig, plot_color_bar=plot_color_bar, suppress_labels=False)

        def plot_fr_pred_sim_glm_coeff(
                self, glm_restriction_idx=None, similarity_metrics=None, sigma=None, brain_regions=None,
                firing_rate_bounds_type="IQR", fr_ppt_smoothed_param_name="0.05", fr_wa_smoothed_param_name="0.1",
                average_function=np.nanmedian, unit_names=None, main_fig=None,
                label_coeff_threshold=.1,  # only put coeff name in legend if at least this much (plot coeff regardless)
                show_legend=True, coeff_text=True, max_units_plot=3, populate_tables=True, plot_color_bar=False,
                save_fig=False):
            # For individual units, plot single trial firing rate, average firing rate during conditions,
            # matrix with across path decoding goodness metric (similarity of actual and predicted smoothed
            # firing rates), and bar plot with nnls decomposition of that matrix as linear combination
            # of basis vectors representing different levels of path generalization

            # Get inputs if not passed
            if brain_regions is None:
                brain_regions = self.parent_obj.brain_regions
            if glm_restriction_idx is None:
                glm_restriction_idx = self.parent_obj.glm_restriction_df.index[0]
            if similarity_metrics is None:
                similarity_metrics = [self.parent_obj._get_default_param("similarity_metric")]
            if sigma is None:
                sigma = self.parent_obj._get_default_param("sigma")
            if unit_names is None:
                unit_names = set(
                    df_filter_columns_isin(self.parent_obj.similarity_df, {"brain_region": brain_regions})[
                        "unit_name"])

            # Define time periods restriction
            glm_restriction = self.parent_obj.glm_restriction_df.iloc[glm_restriction_idx]
            if glm_restriction["name"] == "path":
                restrict_time_periods = ["path"]
            elif glm_restriction["name"] == "delay":
                restrict_time_periods = ["well_arrival"]

            # Define dio well arrival trials param name if indicated
            dio_well_arrival_trials_param_name = None  # default
            if glm_restriction["name"] == "delay":
                dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_delay_param_name()

            # Hard code params
            # Parameters for single trial firing rate maps
            height_ratios = np.asarray([1, 1])
            plot_performance_outcomes = False
            plot_environment_marker = False
            sharey = "columns"
            # GLM params
            quantity = "coeff_ecs"
            fontsize1 = 15
            fontsize2 = 13
            fontsize3 = 11

            unit_counter = 0
            for unit_name in unit_names:

                # Continue if reached max unit limit
                unit_counter += 1
                if unit_counter > max_units_plot:
                    continue

                # Initialize "main figure" spanning the figures described above:

                # First subplot corresponds to firing rates; second subplot to GLM results
                if main_fig is None:
                    plot_width = 2 * self._get_k_set_size(
                        self.parent_obj.glm_restriction_df.iloc[glm_restriction_idx].condition)
                    figsize = (plot_width, 3)
                    main_fig = plt.figure(figsize=figsize, constrained_layout=True)

                # Initialize subfigures within main figure
                plot_obj = PlotSTFRMap(self.parent_obj.nwb_file_name, self.parent_obj.epochs, restrict_time_periods)
                # ...Determine width between subfigures
                wspace = 0
                subfigs = main_fig.subfigures(
                    nrows=1, ncols=2, width_ratios=[len(plot_obj.restrictions)/3, .65], wspace=wspace)

                # Plot firing rate maps
                # ...get subfigure
                fig = subfigs[0]
                # ...call plotting code
                plot_obj.plot_single_trial_firing_rate_map(
                    unit_names=[unit_name],
                    firing_rate_bounds_type=firing_rate_bounds_type,
                    plot_performance_outcomes=plot_performance_outcomes,
                    plot_environment_marker=plot_environment_marker,
                    fr_ppt_smoothed_param_name=fr_ppt_smoothed_param_name,
                    fr_wa_smoothed_param_name=fr_wa_smoothed_param_name,
                    dio_well_arrival_trials_param_name=dio_well_arrival_trials_param_name,
                    average_function=average_function, populate_tables=populate_tables, height_ratios=height_ratios,
                    sharey=sharey, fig=fig, suppress_labels=False, plot_color_bar=plot_color_bar)

                # Plot prediction similarity array and fit coefficients for each similarity metric

                # First, initialize figure within bottom row of main_fig
                fig2 = subfigs[1]
                # allow extra rows and columns so can squish this set of plots to be similar height and width
                # as first set of plots above
                num_extra_rows_one_side = 1
                num_extra_columns = 0
                # define width and height between subfigures
                wspace, hspace = .5, .3
                gs = gridspec.GridSpec(
                    1 + num_extra_rows_one_side*2, 2 * len(similarity_metrics) + num_extra_columns, width_ratios=
                    [4, 1] * len(similarity_metrics) + [1]*num_extra_columns,
                    height_ratios=[1]*num_extra_rows_one_side + [5] + [1]*num_extra_rows_one_side,
                    wspace=wspace, hspace=hspace)

                # Now loop through similarity metrics
                for similarity_metric_idx, similarity_metric in enumerate(similarity_metrics):

                    # Get basis vector coefficients now so can skip plotting if not present
                    df_subset = df_filter_columns(
                        self.basis_estimation_df.loc[glm_restriction_idx].metrics_df,
                        {"similarity_metric": similarity_metric, "sigma": sigma})

                    # Continue if unit not in df subset
                    if unit_name not in df_subset.index:
                        continue

                    # Plot prediction similarity array
                    # ...get coefficients
                    coeff_vector = df_subset.loc[unit_name][quantity]
                    # ...get axis
                    ax = fig2.add_subplot(gs[num_extra_rows_one_side:-num_extra_rows_one_side,
                                          similarity_metric_idx * len(similarity_metrics)])
                    # ...set x and y labels to nothing if after first plot
                    xlabel, ylabel = None, None  # default text is used
                    if similarity_metric_idx > 0:
                        xlabel, ylabel = "", ""
                    self.parent_obj.plot_prediction_similarity(
                        fig=fig2, ax=ax, unit_name=unit_name, glm_restriction_idx=glm_restriction_idx,
                        similarity_metric=similarity_metric, plot_color_bar=False, title="FR correlation",
                        xlabel=xlabel, ylabel=ylabel, fontsize=fontsize1, ticklabels_fontsize=fontsize2)

                    # Plot fit coefficients for basis vectors in reconstructing similarity array (stacked bar)
                    # ...get axis
                    ax = fig2.add_subplot(gs[num_extra_rows_one_side:-num_extra_rows_one_side,
                                          similarity_metric_idx *len(similarity_metrics) + 1])
                    condition = self.parent_obj._get_glm_restriction_val(glm_restriction_idx, "condition")
                    # ...get values (and number of them) for this condition so can use to abbreviate
                    # long coefficient names
                    k_set_size = self._get_k_set_size(condition)
                    for idx, (coeff_name, coeff) in enumerate(coeff_vector.items()):  # basis vector coefficients
                        column_idx = unpack_single_element(np.where(coeff_vector.index == coeff_name)[0])
                        color = df_pop(self.k_colors_df, {"condition": condition, "column_idx": column_idx}, "color")
                        bar_start = np.sum(coeff_vector[:column_idx])
                        # ...if coeff exceeds threshold for labeling, get an abbreviated version of coefficient
                        # name to include in legend if indicated, and/or as text next to corresponding coefficient
                        # in bar as indicated
                        label = ""  # initialize
                        if coeff >= label_coeff_threshold:  # update label if coeff exceeds threshold for labeling
                            label = coeff_name.replace("_", " ")
                            # abbreviate label name if long
                            # ...if coefficient describes set of all condition values, abbreviate as "all"
                            # get condition values corresponding to this coefficient
                            condition_values = self.condition_coefficient_df.set_index(
                                "coefficient_name").loc[coeff_name].condition_vals
                            if len(condition_values) == k_set_size:
                                label = "all"
                            # ...text for coeff if indicated
                            if coeff_text:
                                ax.text(.5, bar_start + coeff/5, label, fontsize=fontsize3)
                        ax.bar(0, coeff, bottom=bar_start, color=color, label=label)
                    # ...remove x axis
                    ax.axes.get_xaxis().set_visible(False)
                    # ...y ticks
                    yticks = [0, .2, .4, .6, .8, 1]
                    yticklabels = [x if x in [0, 1] else "" for x in yticks]
                    # ...Legend. Reverse label order to match bar segments
                    if show_legend:
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=8,
                                  bbox_to_anchor=(4.5, 1))
                    # ...title
                    title = "Reliability\nmetric"
                    title_fontsize = fontsize1
                    # ...apply formatting
                    format_ax(ax=ax, ylim=[0, 1], yticks=yticks, yticklabels=yticklabels, ticklabels_fontsize=fontsize2,
                              ylabel=ylabel, fontsize=title_fontsize, title=title)

                # Save figure
                file_name_save = f"nnls_{format_nwb_file_name(self.parent_obj.nwb_file_name)}_eps" \
                    f"{self.parent_obj.epochs_id}_{glm_restriction['name']}_{unit_name}_"
                save_figure(main_fig, file_name_save, save_fig=save_fig)

        def print_group_colors_as_hex(self):
            for x in self.k_colors_df["color"]:
                print(rgb2hex(x))

        def _get_coefficient_names(self, condition):
            return df_filter_columns(self.condition_coefficient_df, {"condition": condition}).coefficient_name.values

        def plot_coefficient_names(self, glm_restriction_idx, subplot_width=1, subplot_height=1, fontsize=8,
                                   save_fig=False):
            # Make a standalone legend with coefficient names
            # Get names of coefficients for this glm restriction
            condition = self.parent_obj._get_glm_restriction_val(glm_restriction_idx, "condition")
            coefficient_names = self._get_coefficient_names(condition)

            # Initialize plot
            fig, ax = plt.subplots(figsize=(subplot_width, subplot_height))

            # Make legend for coefficient names
            for idx, coeff_name in enumerate(coefficient_names):
                # Replace underscore with space in coefficient name
                coeff_name = coeff_name.replace("_", " ")
                # Get color for coefficient
                color = df_pop(self.k_colors_df, {"condition": condition, "column_idx": idx}, "color")
                ax.plot(0, 0, label=coeff_name, color=color, linewidth=12)
            # Format
            ax.legend(loc="center", fontsize=fontsize, frameon=False)
            ax.axis("off")

            # Save figure
            restriction_name = self.parent_obj._get_glm_restriction_val(glm_restriction_idx, "name")
            save_figure(fig, f"nnls_{restriction_name}_coeff_legend", save_fig=save_fig)

    def _get_default_params_map(self):
        return {"glm_restriction_idxs": self.nnls.basis_estimation_df.index, "similarity_metrics": ["correlation"],
                "similarity_metric": self.prediction_similarity_params["similarity_metrics"][0],
                "sigma": self.prediction_similarity_params["sigmas"][0], "brain_regions": self.brain_regions,
                "min_num_spikes": self.nnls.min_num_spikes}

    def _get_default_params(self, plot_params):
        # Update dictionary of params with default params if None
        default_params_map = self._get_default_params_map()
        return {k: v if v is not None else default_params_map[k] for k, v in plot_params.items()}

    def _get_default_param(self, param_name):
        return self._get_default_params_map()[param_name]

    def plot_similarity_violins(self, similarity_metric=None, sigma=None, brain_regions=None, min_num_spikes=None,
                                include_stats=False, fig_ax_list=None, show_legend=True, save_fig=False):
        # Make violin plots of similarity values separately for brain regions and GLM restriction periods
        # (pooling across restriction conditions and units; note that averaging across restriction conditions
        # would average over very high and low values across conditions for sparsely firing, reliable cells (e.g.
        # place cells)). Averaging across conditions is not done here so that the full distribution of reliability
        # across conditions and units can be seen.

        # Get inputs if not passed
        params = {"brain_regions": brain_regions, "similarity_metric": similarity_metric, "sigma": sigma,
                  "min_num_spikes": min_num_spikes}
        params = self._get_default_params(params)
        brain_regions = params["brain_regions"]
        similarity_metric = params["similarity_metric"]

        figsize = (4, 4)
        font_scale = 12
        legend_fontsize = 12

        test = "Mann-Whitney"
        ylim = [-.5, 1.6]
        yticks = np.asarray([np.round(x, 1) for x in np.arange(-.7, 1.1, .1)])
        yticks[yticks == 0] = 0  # for some reason 0 showing up with negative in front. This addresses that
        yticklabels = get_ticklabels(yticks, idxs=[np.where(yticks == x)[0] for x in [-.5, 0, .5, 1]])

        # Name for saved file
        file_name_save = f"glm_prediction_violin_{similarity_metric}_sigma{sigma}_" \
            f"{format_nwb_file_name(self.nwb_file_name)}_{get_epochs_id(self.epochs)}"

        # Get df to pass to seaborn plot function
        # ...Filter for brain regions, similarity metric, and sigma
        similarity_df_subset = df_filter_columns_isin(
            self.similarity_df, {
                "brain_region": brain_regions, "similarity_metric": [similarity_metric], "sigma": [sigma]})
        # ...Filter for train / test conditions in nnls analysis (potentially rewarded paths / wells)
        valid_bool = np.logical_or.reduce(
            [np.logical_and(similarity_df_subset.train_condition.isin(x), similarity_df_subset.test_condition.isin(x))
             for x in self.conditions_df.condition_vals])
        similarity_df_subset = similarity_df_subset[valid_bool]
        # ...Filter for cases where train and test conditions are same
        valid_bool = similarity_df_subset["test_condition"] == similarity_df_subset["train_condition"]
        similarity_df_subset = similarity_df_subset[valid_bool]
        # ...Filter using min spikes threshold
        valid_bool = [unpack_single_element(df_filter_columns(self.condition_spike_count_df, {
            "unit_name": unit_name, "test_condition_val": test_condition}).num_spikes.values > min_num_spikes)
                      for unit_name, test_condition in
                      zip_df_columns(similarity_df_subset, ["unit_name", "test_condition"])]
        similarity_df_subset = similarity_df_subset[valid_bool]
        if any(np.isnan(similarity_df_subset.similarity_val)):
            raise Exception(f"at least one nan in similarity vals; must consider how to account for this")

        # Impose order of brain region in brain_regions by converting brain region column in df
        # to categorical datatype
        brain_regions = [x for x in brain_regions if
                               x in set(similarity_df_subset["brain_region"])]  # narrow to available
        similarity_df_subset["brain_region"] = pd.Categorical(
            similarity_df_subset["brain_region"], categories=brain_regions)

        # Get name of glm restrictions for use in legend
        order = np.unique(similarity_df_subset.glm_restriction_idx)
        order_names = [self.glm_restriction_df.loc[idx]["name"] for idx in order]

        # Define colors: use brain region colors, convert to RGB, and make increasingly light shades to go along with
        # GLM restriction periods
        violin_colors = BrainRegionColor().tint_colors(brain_regions, num_tints=len(order), concatenate=True)

        # Rename columns for seaborn plot axis labels
        rename_map = {
            "brain_region": "brain region", "similarity_val": similarity_metric, "glm_restriction_idx": "task period"}
        df = similarity_df_subset.rename(rename_map, axis="columns")

        x = "task period"
        hue = "brain region"
        y = similarity_metric

        # Define conditions we want to compare statistically
        # ...First define pairs of brain regions
        def sort_order(x):
            if "CA1" in x:
                return 1
            elif "mPFC" in x:
                return 2
            elif "OFC" in x:
                return 3
            else:
                raise Exception(f"{x} not accounted for in code")
        brain_region_pairs = list(itertools.combinations(sorted(brain_regions, key=sort_order), r=2))
        brain_region_pairs = [sorted(list(x), key=sort_order) for x in brain_region_pairs]  # order
        # Assemble comparisons in tuples
        comparison_pairs = None
        comparison_pairs_ = [((idx, t1), (idx, t2)) for t1, t2 in brain_region_pairs for idx in
                             set(df["task period"])]  # pairs consist of pairs of brain region and restriction
        if include_stats and len(comparison_pairs_) > 0:
            comparison_pairs = comparison_pairs_

        plot_violin_or_box(df, hue, y, x, test=test, figsize=figsize, font_scale=font_scale, legend_fontsize=legend_fontsize,
                           order=order, order_names=order_names, show_legend=show_legend, violin_colors=violin_colors,
                           comparison_pairs=comparison_pairs, fig_ax_list=fig_ax_list, ylim=ylim, yticks=yticks,
                           yticklabels=yticklabels, file_name_save=file_name_save, save_fig=save_fig)

        # For now, return stats from hierarchical bootstrap. TODO: write code to add results to plot
        try:
            return hierarchical_bootstrap(
                df, ["unit_name", "train_condition"], similarity_metric, ["brain region", "task period"], ["task period"],
                "brain region")
        except:
            return None

    def plot_prediction_similarity(self, fig, ax, unit_name, glm_restriction_idx=None, similarity_metric=None,
                                   sigma=None, plot_color_bar=True, xlabel=None, ylabel=None, title=None, fontsize=18,
                                   ticklabels_fontsize=18, title_color="black"):
        # Plot prediction similarity array for one unit

        # Get inputs if not passed
        if similarity_metric is None:
            similarity_metric = self._get_default_param("similarity_metric")
        if sigma is None:
            sigma = self._get_default_param("sigma")
        if glm_restriction_idx is None:
            glm_restriction_idx = self.glm_restriction_df.index[0]
        if xlabel is None:
            xlabel = "Train"
        if ylabel is None:
            ylabel = "Test"
        if title is None:
            title = f"Unit: {unit_name}"

        # Get glm restriction
        glm_restriction = self.glm_restriction_df.loc[glm_restriction_idx]

        # Get similarity array
        similarity_arr = df_pop(self.similarity_arr_df, {
            "glm_restriction_idx": glm_restriction_idx, "sigma": sigma, "similarity_metric": similarity_metric,
            "unit_name": unit_name}, "similarity_arr", tolerate_no_entry=True)

        # If no similarity array, remove axis and exit function
        if len(similarity_arr) == 0:
            ax.axis("off")
            return

        # Plot similarity array as heatmap
        plot_heatmap(similarity_arr, fig_ax_list=[fig, ax], clim=[0, 1], plot_color_bar=plot_color_bar)
        # x ticks and labels
        conditions_df_subset = self.conditions_df.loc[glm_restriction.condition]
        xticks = np.arange(.5, len(conditions_df_subset.train_order) + .5)
        xticklabels = [abbreviate_condition_name(x) for x in similarity_arr.index]
        # y ticks and labels
        yticks = np.arange(.5, len(conditions_df_subset.test_order) + .5)
        yticklabels = [abbreviate_condition_name(x) for x in similarity_arr.columns]

        format_ax(ax=ax, xticks=xticks, xticklabels=xticklabels, yticks=yticks, yticklabels=yticklabels,
                  ticklabels_fontsize=ticklabels_fontsize,
                  xlabel=xlabel, ylabel=ylabel, title=title, title_color=title_color, fontsize=fontsize)

    def plot_prediction_similarities(
            self, similarity_metric=None, sigma=None, glm_restriction_idxs=None,
            max_units=None,  # maximum units to plot
            num_mega_columns=6,  # number of columns in plot
            subplot_width=2.5, subplot_height=1.7, mega_row_gap_factor=.2,
            mega_column_gap_factor=.1, wspace=None, hspace=None, brain_regions=None, suppress_labels=False,
            save_fig=False):

        # Get inputs if not passed
        if similarity_metric is None:
            similarity_metric = self._get_default_param("similarity_metric")
        if sigma is None:
            similarity_metric = self._get_default_param("sigma")
        if glm_restriction_idxs is None:
            glm_restriction_idxs = self.glm_restriction_df.index
        if brain_regions is None:
            brain_regions = self.brain_regions

        # Plot prediction similarity
        for brain_region in brain_regions:

            # Get unit names for this brain region
            # unit_names = brain_region_unit_name_map[brain_region]
            unit_names = df_filter_columns(self.unit_name_df, {"brain_region": brain_region}).index

            # Initialize plot
            # ...Define total number of plots
            num_plots = len(unit_names)
            if max_units is not None:
                num_plots = np.min([num_plots, max_units])
            # ...Define number of mega rows
            num_mega_rows = int(np.ceil(num_plots / num_mega_columns))
            # ...Initialize
            gs_map, ax_map, fig = get_gridspec_ax_maps(
                mega_row_iterables=np.arange(0, num_mega_rows), mega_column_iterables=np.arange(0, num_mega_columns),
                row_iterables=[0], column_iterables=glm_restriction_idxs, subplot_width=subplot_width,
                subplot_height=subplot_height, mega_row_gap_factor=mega_row_gap_factor,
                mega_column_gap_factor=mega_column_gap_factor, wspace=wspace, hspace=hspace)

            for glm_restriction_idx_idx, glm_restriction_idx in enumerate(glm_restriction_idxs):

                for unit_name_idx, unit_name in enumerate(unit_names):

                    # Continue if max units reached
                    if max_units is not None:
                        if unit_name_idx > max_units:
                            continue

                    # Get axis
                    row_idx, mega_column_idx = divmod(unit_name_idx, num_mega_columns)
                    ax = ax_map[(row_idx, mega_column_idx, 0, glm_restriction_idx)]

                    # Indicate whether to plot colorbar
                    plot_color_bar = False  # default
                    if unit_name_idx + 1 % num_mega_columns == 0:
                        plot_color_bar = True

                    # Labels
                    xlabel = ""  # default
                    ylabel = ""  # default
                    if unit_name_idx == 0 and glm_restriction_idx_idx == 0 and not suppress_labels:
                        xlabel = "Train"
                        ylabel = "Test"
                    # Title
                    title = ""  # default
                    if glm_restriction_idx_idx == 0:
                        title = unit_name
                    title_color = self.brain_region_color_map[brain_region]

                    # Call plotting code
                    self.plot_prediction_similarity(fig=fig, ax=ax, unit_name=unit_name,
                                                    glm_restriction_idx=glm_restriction_idx,
                                                    similarity_metric=similarity_metric, sigma=sigma,
                                                    plot_color_bar=plot_color_bar, xlabel=xlabel, ylabel=ylabel,
                                                    title=title, title_color=title_color)

            file_name_base = self._get_file_name_base(glm_restriction_idx)
            figure_name = f"glm_pred_sim_{brain_region}_{file_name_base}_{similarity_metric}_sigma{sigma}"
            save_figure(fig, figure_name, save_fig=save_fig)

    def _get_k_pool_df(self, glm_restriction_idx, similarity_metric, sigma):

        # Put all k into one df so can plot together
        condition = self._get_glm_restriction_val(glm_restriction_idx, "condition")
        k_set_size = self.nnls._get_k_set_size(condition)
        k_set = self.nnls.basis_obj._get_k_set(k_set_size)
        # ...filter df with metrics for desired similarity metric and sigma
        metrics_df = df_filter_columns(self.nnls.basis_estimation_df.loc[glm_restriction_idx].metrics_df,
                                       {"similarity_metric": similarity_metric, "sigma": sigma})
        # ...check units not represented more than once
        check_all_unique(metrics_df.index)
        # ...create a separate row for each k (currently different k in columns)
        metrics_df = df_from_data_list(
            [(unit_name, df_row.brain_region, k, df_row[f"k{k}coeff_sum_ecs"]) for unit_name, df_row in
             metrics_df.iterrows() for k in k_set],
            ["unit_name", "brain_region", "k", "k_coeff_sum_ecs"])
        # ...convert brain region to categorical so brain regions have desired order

        return convert_categorical(metrics_df, column_name="brain_region", categories=self.brain_regions)

    def plot_k_coeff_box(
            self, similarity_metric=None, sigma=None, brain_regions=None, subplot_width=3, subplot_height=3,
            ticklabels_fontsize=14, tint_glm_restriction=False, fig_ax_list=None, save_fig=False):
        # Plot box plot of sum of coefficients for each k across cells

        # Get inputs if not passed
        # ...Unpack figure / axis if passed, otherwise initialize
        if fig_ax_list is None:
            fig_ax_list = get_fig_axes(1, len(self.glm_restriction_df), subplot_width=subplot_width,
                                     subplot_height=subplot_height, sharey=True)
        fig, axes = fig_ax_list
        # ...Get similarity params
        if similarity_metric is None:
            similarity_metric = self._get_default_param("similarity_metric")
        if sigma is None:
            sigma = self._get_default_param("sigma")
        if brain_regions is None:
            brain_regions = self._get_default_param("brain_regions")

        # Define statistical test type
        test = "Mann-Whitney"

        # Define plot variables
        x = "k"
        y = f"k_coeff_sum_ecs"
        hue = "brain_region"

        # Define colors (color represents brain region. If tint_glm_restriction is True, shade represents
        # glm restriction). Store as map: glm restriction idx --> brain region --> colors
        colors_map = {idx: self._get_brain_region_color_map() for idx in self.glm_restriction_df.index}  # default
        if tint_glm_restriction:
            colors_map = BrainRegionColor().tint_colors(brain_regions, num_tints=len(self.glm_restriction_df))
            colors_map = {brain_region: {colors[idx, :] for idx in self.glm_restriction_df.index}
                          for brain_region, colors in colors_map.keys()}

        for glm_restriction_idx in self.glm_restriction_df.index:

            # Get df with different k columns placed into separate rows
            metrics_df = self._get_k_pool_df(glm_restriction_idx, similarity_metric, sigma)

            # Define statistical comparisons
            condition = self._get_glm_restriction_val(glm_restriction_idx, "condition")
            k_set_size = self.nnls._get_k_set_size(condition)
            k_set = self.nnls.basis_obj._get_k_set(k_set_size)
            comparison_pairs = [((k, b1), (k, b2)) for b1, b2 in list(itertools.combinations(brain_regions, r=2))
                                for k in k_set]

            # collect plot params
            plot_params = {"data": metrics_df, "y": y, "x": x, "hue": hue}
            # get axis
            ax = axes[glm_restriction_idx]
            # get colors for this glm restriction
            h = sns.boxplot(ax=ax, showfliers=False, palette=colors_map[glm_restriction_idx],
                            **plot_params)  # get handle so can remove legend
            # y limit
            ylim = [0, 1.8]
            # x tick labels
            xticklabels = [x.get_text().replace("_targeted", "") for x in ax.get_xticklabels()]
            # y ticks/labels
            yticks = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
            yticklabels = [x if x in [0, .2, .4, .6, .8, 1] else "" for x in yticks]
            # Set left spine to same range as yticks
            ax.spines['left'].set_bounds(np.min(yticks), np.max(yticks))
            format_ax(ax, ylim=ylim, ylabel="", xlabel="", xticklabels=xticklabels, yticks=yticks,
                      yticklabels=yticklabels, ticklabels_fontsize=ticklabels_fontsize)
            # remove legend
            h.legend_.remove()
            # Add statistical comparison of passed condition pairs to plot
            if comparison_pairs is not None:  # can only do if more than one violin
                annotator = Annotator(ax, comparison_pairs, **plot_params)
                annotator.configure(test=test).apply_and_annotate()

        # Save figure
        file_name = f"glm_bar_k_unit_ave_{format_nwb_file_name(self.nwb_file_name)}_eps{self.epochs_id}_" \
            f"{self.units_params['curation_set_name']}_{similarity_metric}_sigma{sigma}"
        save_figure(fig, file_name, save_fig=save_fig)


# TODO: add this to class
# def plot_predicted_smoothed_rate(self, unit_name=None, sigma=None, subplot_width=5,
#                                  subplot_height=3):
#     # Plot actual and predicted spikes (train condition: rows, test condition: columns)
#
#     # Get inputs if not passed
#     if unit_name is None:
#         unit_name = np.unique(self.results_df["unit_name"])[0]
#
#     # Get elnet container
#     elnet_container = df_pop(self.results_df, {"unit_name": unit_name}, "elnet_container")
#
#     # Get time bin size
#     time_bin_width = elnet_container.time_bin_width
#
#     # Get train and test conditions
#     train_conditions = np.unique(self.elnet_container.results_folds_merged_df["train_condition"])
#     test_conditions = np.unique(self.elnet_container.results_folds_merged_df["test_condition"])
#
#     # Initialize figure
#     num_rows = len(test_conditions)
#     num_columns = len(train_conditions)
#     fig, axes = plt.subplots(num_rows,
#                              num_columns,
#                              figsize=(subplot_width * num_columns, subplot_height * num_rows),
#                              sharey=True)
#
#     # Plot
#     for test_condition_idx, test_condition in enumerate(test_conditions):
#         for train_condition_idx, train_condition in enumerate(train_conditions):
#             # Unpack variables
#             df_subset = df_filter1_columns(elnet_container.results_folds_merged_df,
#                                            {"train_condition": train_condition,
#                                             "test_condition": test_condition})
#             y_test, y_test_predicted = df_subset[["y_test", "y_test_predicted"]].to_numpy()[0]
#             # Get plot axis
#             ax = get_ax_for_layout(axes, test_condition_idx * len(train_conditions) + train_condition_idx)
#             # In order to resolve in time dimension, plot trials concatenated rather than on true time axis (where they would be spread farther apart)
#             # To enable this, make time vector for the concatenated data that has same time bin width as data
#             pseudo_time_axis = np.linspace(0, len(y_test) * time_bin_width, len(y_test))
#             # Smoothed actual spike count within folds
#             trial_intervals = [[x[0], x[-1]] for x in df_subset["y_test_index"].values[0]]
#             y_test_smoothed = smooth_intervals(y_test / time_bin_width, trial_intervals,
#                                                sigma)  # smooth within each trial
#             ax.plot(pseudo_time_axis, y_test_smoothed.values, color="black", label="actual")
#             # Smoothed predicted spike rate (lambda)
#             y_test_predicted_smoothed = smooth_intervals(y_test_predicted / time_bin_width, trial_intervals,
#                                                          sigma)  # smooth within each trial
#             ax.plot(pseudo_time_axis, y_test_predicted_smoothed.values, color="orange", alpha=.8, label="predicted")
#             # Calculate correlation coefficient and overlap of smoothed actual and predicted spike rates
#             vals = [y_test_smoothed.values, y_test_predicted_smoothed.values]
#             corr = np.corrcoef(*vals)[0, 1]
#             overlap_metric = overlap(*vals)
#             # Title
#             title = f"train: {train_condition}\n test: {test_condition} \n corr: {corr: .2f}, overlap: {overlap_metric: .2f}"
#             title_color = "black"
#             if train_condition == test_condition:
#                 title_color = "red"
#             format_ax(ax=ax, title=title, title_color=title_color, fontsize=12)
#             if train_condition == 0 and test_condition_idxs == 0:
#                 ax.legend()
#     # Vertical lines to mark fold boundaries
#     ylim = ax.get_ylim()  # use common ylim
#     for test_condition_idx, test_condition in enumerate(test_conditions):
#         for train_condition_idx, train_condition in enumerate(train_conditions):
#             # Get lengths of folds
#             fold_lens = df_pop(elnet_container.results_folds_merged_df,
#                                        {"train_condition": train_condition,
#                                         "test_condition": test_condition}, "fold_lens")
#             # Get fold boundaries
#             fold_boundaries = np.cumsum(fold_lens)
#             # Get plot axis
#             ax = get_ax_for_layout(axes, test_condition_idx * len(train_conditions) + train_condition_idx)
#             for fold_boundary in fold_boundaries:
#                 plot_spanning_line(span_data=ylim, constant_val=fold_boundary * time_bin_width, ax=ax, span_axis="y")
#     fig.tight_layout()

