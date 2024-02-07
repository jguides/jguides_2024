import copy
from collections import namedtuple

import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_subject_id, \
    plot_junction_fractions, get_val_pairs, plot_horizontal_lines, \
    get_subject_id_shorthand
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SelBase, SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    unique_table_column_sets, \
    insert1_print, get_relationship_texts, format_nwb_file_name, get_default_param, get_table_name, \
    fetch1_tolerate_no_entry, delete_, get_boot_params
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_nwb_file_name_epochs_description
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionColor, BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, RunEpoch, EpochsDescriptions, \
    RecordingSet
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsParams, BrainRegionUnitsCohortType, \
    EpsUnitsParams, BrainRegionUnitsFail
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDDTrials
from src.jguides_2024.task_event.jguidera_task_performance import strip_performance_outcomes
from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWADigSingleAxisParams
from src.jguides_2024.utils.array_helpers import array_to_tuple_list, on_off_diagonal_ratio
from src.jguides_2024.utils.df_helpers import df_filter_columns, dfs_same_values, zip_df_columns, \
    df_from_data_list, df_pop, \
    df_filter1_columns_symmetric, unique_df_column_sets
from src.jguides_2024.utils.dict_helpers import add_defaults, dict_comprehension, check_return_single_dict, \
    return_shared_key_value
from src.jguides_2024.utils.for_loop_helpers import print_iteration_progress
from src.jguides_2024.utils.hierarchical_bootstrap import hierarchical_bootstrap
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements, check_return_single_element
from src.jguides_2024.utils.parse_matrix import parse_matrix
from src.jguides_2024.utils.plot_helpers import plot_heatmap, \
    get_ticklabels, format_ax, save_figure, plot_ave_conf, plot_spanning_line, get_gridspec_ax_maps, get_plot_idx_map
from src.jguides_2024.utils.save_load_helpers import save_json
from src.jguides_2024.utils.set_helpers import check_membership, check_set_equality
from src.jguides_2024.utils.state_evolution_estimation import AverageVectorDuringLabeledProgression
from src.jguides_2024.utils.string_helpers import format_number, format_bool, replace_chars
from src.jguides_2024.utils.vector_helpers import vectors_cosine_similarity, vectors_euclidean_distance, \
    check_all_unique, \
    unpack_single_element, unpack_single_vector, min_max, find_spans_increasing_list


def get_metadata(metric_name, vector_type):
    # Get metadata for firing rate vector analysis

    Metadata = namedtuple(
        "Metadata", "vector_df_name ave_vector_df_name vector_col_name ave_vector_col_name x_name metric fn")

    # Cosine similarity of difference vectors
    if metric_name == "cosine_similarity" and vector_type == "diff_vec":
        vector_df_name = "diff_vector_df"
        ave_vector_df_name = "ave_diff_vector_df"
        vector_df_metadata_obj = AverageVectorDuringLabeledProgression.get_df_metadata(vector_df_name)
        ave_vector_df_metadata_obj = AverageVectorDuringLabeledProgression.get_df_metadata(ave_vector_df_name)
        x_name = check_return_single_element([
            obj.x_name for obj in [vector_df_metadata_obj, ave_vector_df_metadata_obj]]).single_element
        metric = "cosine"
        metric_fn = vectors_cosine_similarity
        return Metadata(
            vector_df_name, ave_vector_df_name, vector_df_metadata_obj.vector_col_name,
            ave_vector_df_metadata_obj.vector_col_name, x_name, metric, metric_fn)

    # Cosine similarity of vectors
    elif metric_name == "cosine_similarity" and vector_type == "vec":
        vector_df_name = "vector_df"
        ave_vector_df_name = "ave_vector_df"
        vector_df_metadata_obj = AverageVectorDuringLabeledProgression.get_df_metadata(vector_df_name)
        ave_vector_df_metadata_obj = AverageVectorDuringLabeledProgression.get_df_metadata(ave_vector_df_name)
        x_name = check_return_single_element([
            obj.x_name for obj in [vector_df_metadata_obj, ave_vector_df_metadata_obj]]).single_element
        metric = "cosine"
        metric_fn = vectors_cosine_similarity
        return Metadata(
            vector_df_name, ave_vector_df_name, vector_df_metadata_obj.vector_col_name,
            ave_vector_df_metadata_obj.vector_col_name, x_name, metric, metric_fn)

    # Euclidean distance of vectors
    elif metric_name == "euclidean_distance" and vector_type == "vec":
        vector_df_name = "vector_df"
        ave_vector_df_name = "ave_vector_df"
        vector_df_metadata_obj = AverageVectorDuringLabeledProgression.get_df_metadata(vector_df_name)
        ave_vector_df_metadata_obj = AverageVectorDuringLabeledProgression.get_df_metadata(ave_vector_df_name)
        x_name = check_return_single_element([
            obj.x_name for obj in [vector_df_metadata_obj, ave_vector_df_metadata_obj]]).single_element
        metric = "euclidean"
        metric_fn = vectors_euclidean_distance
        return Metadata(
            vector_df_name, ave_vector_df_name, vector_df_metadata_obj.vector_col_name,
            ave_vector_df_metadata_obj.vector_col_name, x_name, metric, metric_fn)

    else:
        raise Exception(f"No case coded for metric name {metric_name} and vector type {vector_type}")


class CovariateFRVecSelBase(SelBase):
    """Base class for firing rate vector as a function of a covariate selection table"""

    @staticmethod
    def _key_filter():
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_fr_vec_default_params_map
        default_params = get_fr_vec_default_params_map()
        return {k: default_params[k] for k in [
            "res_time_bins_pool_param_name"]}


class CovariateFRVecBase(ComputedBase):
    """Base class for firing rate vector as a function of a covariate table"""

    def get_inputs(self, key, verbose=False, ax=None):

        # Separate method so can access inputs from outside make function (alternative would be to store these,
        # but that would be wasteful of space since computation isn't happening to get these inputs)
        raise Exception(f"This method should be implemented in child class")

    def _plot_labels(self, labels_pre_post_alteration, labels, verbose, ax):

        # Optionally plot labels while getting inputs

        if verbose:

            # Initialize axis if not passed
            if ax is None:
                _, ax = plt.subplots()

            # Define marker based on whether labels are pre or post alteration
            check_membership([labels_pre_post_alteration], ["pre", "post"])
            marker = "."
            if labels_pre_post_alteration == "post":
                marker = "x"

            # Plot labels
            ax.plot(labels, marker, alpha=.3, label=f"{self.table_name} labels {labels_pre_post_alteration} alteration")

        # Return axis
        return ax

    def make(self, key):

        # Reset tracker for upstream tiles (important so does not carry over from one populate command to next)
        self._initialize_upstream_entries_tracker()

        # Get inputs
        inputs = self.get_inputs(key)

        # Get average vector during path traversals
        obj = AverageVectorDuringLabeledProgression(inputs.x, inputs.labels, inputs.df)

        # Insert into table
        # ...Add names of unit that compose firing rate vectors
        main_table_key = {**key, **{"unit_names": inputs.unit_names}}
        insert_analysis_table_entry(
            self, [obj.vector_df, obj.ave_vector_df, obj.diff_vector_df, obj.ave_diff_vector_df], key=main_table_key)

        # Insert into part tables
        self._insert_part_tables(key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):

        # Convert arrays to tuples before returning df
        df = super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)
        if "x_pair" in df:
            df["x_pair"] = array_to_tuple_list(df["x_pair"])

        return df

    def concat_across_epochs(self, key, object_id_name):

        check_membership([object_id_name], self.get_object_id_name(leave_out_object_id=True), "passed object id name",
                         "available object id names")

        dfs = []
        for epoch in (EpochsDescription & key).fetch1("epochs"):
            df = (self & {**key, **{"epoch": epoch}}).fetch1_dataframe(object_id_name)
            df["epoch"] = [epoch] * len(df)
            dfs.append(df)

        return pd.concat(dfs, axis=0)

    def get_bin_centers(self):

        raise Exception(f"This method must be overwritten in child class")

    def metric_across_epochs(self, key, metric_name, vector_type):

        metadata = get_metadata(metric_name, vector_type)

        # Concatenate vectors across epochs
        concat_df = self.concat_across_epochs(key, metadata.vector_df_name)
        vectors = np.vstack(concat_df[metadata.vector_col_name])

        # Return metric between pairs of vectors
        return metadata.fn(vectors)

    def _visualize_vectors(self, df_name, overlay_x_type=True, **plot_params):

        metadata_obj = AverageVectorDuringLabeledProgression.get_df_metadata(df_name)
        vector_col_name = metadata_obj.vector_col_name
        x_name = metadata_obj.x_name
        vector_df = self.fetch1_dataframe(df_name)
        plot_params = add_defaults(plot_params, {"figsize": (12, 5), "scale_clim": .5}, add_nonexistent_keys=True)

        # Initialize figure if not passed
        fig_ax_list = plot_params.pop("fig_ax_list", None)
        if fig_ax_list is None:
            fig_ax_list = plt.subplots(figsize=plot_params["figsize"])
        plot_params["fig_ax_list"] = fig_ax_list
        fig, ax = fig_ax_list

        # Plot vectors as heatmap
        plot_heatmap(np.vstack(vector_df[vector_col_name]).T, **plot_params)

        if overlay_x_type:
            ax.plot(vector_df[x_name].values, label=x_name, color="red", linewidth=2)
            ax.legend()

    def visualize_vectors(self, overlay_x=True, **plot_params):

        self._visualize_vectors("vector_df", overlay_x, **plot_params)

    def visualize_diff_vectors(self, overlay_x_pair_int=True, **plot_params):

        self._visualize_vectors("diff_vector_df", overlay_x_pair_int, **plot_params)

    @staticmethod
    def alter_input_labels_even_odd(labels):
        # Alter labels to indicate whether trials even or odd

        # Get map from label to a list with trials with the label
        unique_labels = set(labels.values)
        trials_map = {label: find_spans_increasing_list(np.where(labels == label)[0])[0] for label in
                      unique_labels}
        # Add even or odd trials status to labels
        # ...initial vector with None so can check whether all samples assigned to even/odd
        # first check that no None in labels
        if any([x is None for x in labels]):
            raise Exception(f"Code expects no None in labels to be able to check whether "
                            f"even/odd assigned to all labels. Must make changes to code if "
                            f"want to allow None in labels")

        new_labels = np.asarray([None]*len(labels))
        for context_name, trial_spans in trials_map.items():
            for path_trial_num, (i1, i2) in enumerate(trial_spans):
                # must add one in index because trial span ends reflect true endpoint, but indexing
                # leaves out endpoint
                new_labels[i1:i2 + 1] = MazePathWell().get_even_odd_trial_name(context_name, path_trial_num)
        if any([x is None for x in new_labels]):
            raise Exception(f"could not assign even/odd to all labels")

        # Return labels
        return pd.Series(new_labels, index=labels.index)

    def alter_input_labels_correct_incorrect(self, labels, key):
        # Alter labels to indicate whether trials correct or incorrect

        if not isinstance(labels, pd.Series):
            raise Exception(f"labels must be a series but is {type(labels)}")

        # Get trial end epoch trial numbers over time as series
        epoch_trial_numbers = (DioWellDDTrials & key).label_time_vector(
            labels.index, ["trial_end_epoch_trial_numbers"])["trial_end_epoch_trial_numbers"]

        # Get map from departure to departure trial end epoch trial numbers (epoch trial number corresponding to
        # destination well) to trial end performance outcomes
        performance_outcomes_map = dict_comprehension(*(DioWellDDTrials & key).fetch1(
            "trial_end_epoch_trial_numbers", "trial_end_performance_outcomes"))
        # ...update upstream entries tracker
        self._update_upstream_entries_tracker(DioWellDDTrials, key)

        # Get performance outcomes corresponding to labels, via epoch trial numbers
        performance_outcomes_ = []
        for x in epoch_trial_numbers:
            performance_outcome = "none"
            if np.isfinite(x):
                if int(x) in performance_outcomes_map:
                    performance_outcome = performance_outcomes_map[int(x)]
            performance_outcomes_.append(performance_outcome)

        # Drop inbound/outbound from names
        performance_outcomes_ = [strip_performance_outcomes(x) for x in performance_outcomes_]

        # Get path names with correct/incorrect
        valid_outcomes = ["correct", "incorrect"]
        correct_incorrect_trial_path_names = [
            MazePathWell().get_correct_incorrect_trial_text(x, MazePathWell().correct_incorrect_trial_text(y))
            if y in valid_outcomes else "none"
            for x, y in zip(labels, performance_outcomes_)]

        # Get boolean indicating samples assigned to correct or incorrect, so can apply to
        # other data structures
        valid_bool = [x in valid_outcomes for x in performance_outcomes_]

        # Return valid labels and boolean used to get valid labels
        return pd.Series(
            np.asarray(correct_incorrect_trial_path_names)[valid_bool], index=labels.index[valid_bool]), valid_bool

    def plot_labels(self, ax=None):

        # Get results
        dfs = self.fetch1_dataframes()

        # Initialize figure if not passed
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))

        # Plot results
        vector_df = dfs.vector_df
        for _, df_row in vector_df.iterrows():
            ax.plot(df_row.bout_idxs, [df_row.label] * 2, linewidth=3, alpha=.5)

        # Plot horizontal lines to help visualize y positions
        bout_idxs = np.concatenate(vector_df.bout_idxs)
        xlims = [np.min(bout_idxs), np.max(bout_idxs)]
        path_names = np.unique(vector_df.label)
        plot_horizontal_lines(xlims, path_names, ax)


def _default_metric_vector_type():

    return [["cosine_similarity", "diff_vec"],
            ["euclidean_distance", "vec"]]


class CovariateFRVecSTAveParamsBase(SecKeyParamsBase):

    @staticmethod
    def _default_mask_duration():
        return 10

    def _default_params(self):
        return [x + [self._default_mask_duration()] for x in _default_metric_vector_type()]


class CovariateAveFRVecParamsBase(SecKeyParamsBase):

    def _default_params(self):
        return _default_metric_vector_type()


class CovariateFRVecAveSelBase(SelBase):

    @staticmethod
    def _fr_vec_table():

        raise Exception(f"Must override this method in child class")

    def _get_cov_fr_vec_param_names(self):

        # By default, return all covariate firing rate vector params. Can override in child class to restrict.

        return set(self._fr_vec_table().fetch(self._get_main_table()().get_cov_fr_vec_meta_param_name()))

    def _get_potential_keys(self, key_filter=None, populate_tables=False):

        # Get key filter if not passed
        if key_filter is None:
            key_filter = dict()

        # Define covariate firing rate vector params
        cov_fr_vec_param_names = self._get_cov_fr_vec_param_names()
        # If param passed, restrict to this
        cov_fr_vec_meta_param_name = self._get_main_table()().get_cov_fr_vec_meta_param_name()
        if cov_fr_vec_meta_param_name in key_filter:
            cov_fr_vec_param_name = key_filter[cov_fr_vec_meta_param_name]
            check_membership([cov_fr_vec_param_name], cov_fr_vec_param_names)
            cov_fr_vec_param_names = [cov_fr_vec_param_name]

        fr_vec_table = self._fr_vec_table()
        main_table = self._get_main_table()()
        params_table = main_table._get_params_table()
        if params_table is None:
            raise Exception(f"params table not found for {main_table.table_name}")

        # Define keys to loop through
        keys = np.concatenate(
            [unique_table_column_sets((fr_vec_table & {
                **key_filter, **{cov_fr_vec_meta_param_name: x}}), fr_vec_table.primary_key, as_dict=True)
            for x in cov_fr_vec_param_names])

        # Define number of print statements to indicate progression through for loop
        target_num_print_statements = 50

        print(f"getting potential keys for {self.table_name}...")

        potential_keys = []
        for idx, key in enumerate(keys):
            print_iteration_progress(idx, len(keys), target_num_print_statements)
            epoch = int(key.pop("epoch"))
            run_num = RunEpoch().get_run_num(key["nwb_file_name"], epoch)
            epochs_description = EpochsDescription().format_single_run_description(run_num)
            key.update({"epochs_description": epochs_description})
            epochs = (EpochsDescription & key).fetch1("epochs")
            if all([len(fr_vec_table & {**key, **{"epoch": epoch}}) == 1 for epoch in epochs]):
                for param_key in params_table.fetch("KEY"):
                    key.update(param_key)
                    potential_keys.append(copy.deepcopy(key))
            else:
                if populate_tables:
                    fr_vec_table().populate(key)

        return potential_keys


# Base class for trial averaged and single trial average tables
class CovariateFRVecAveBase(ComputedBase):

    @staticmethod
    def _fr_vec_table():

        raise Exception(f"This method should be overwritten in child class")

    def _get_label_relationship_inputs(self, nwb_file_name=None, valid_epochs=None, include_reversed_pairs=False):

        if nwb_file_name is None:
            nwb_file_name = self.fetch1("nwb_file_name")

        if valid_epochs is None:
            valid_epochs = (EpochsDescription & self.fetch1("KEY")).fetch1("epochs")

        # Get map from epoch to potentially rewarded paths
        rewarded = True
        pairs_map = MazePathWell().get_path_name_pair_types_map(
            nwb_file_name, valid_epochs, rewarded, include_reversed_pairs)

        return valid_epochs, pairs_map

    @staticmethod
    def _metric_column_name(metric_name):

        raise Exception(f"This method should be overwritten in child class and provide a map from metric name"
                        f" to name of column with metric")

    def _get_param_quantity(self, param_quantity_name):
        # Return param quantity name (e.g. metric name) across table entries (requires single
        # param name across table entries)

        params_table = self._get_params_table()
        meta_param_name = params_table().meta_param_name()
        param_name = check_return_single_element(self.fetch(meta_param_name)).single_element

        return (params_table & {meta_param_name: param_name}).fetch1(param_quantity_name)

    def _get_metric_name(self):

        return self._get_param_quantity("metric_name")

    def _get_vector_type(self):

        return self._get_param_quantity("vector_type")

    def _get_vals_index_name(self):

        vector_type = self._get_vector_type()
        vals_index_name = "x_1"
        if vector_type == "diff_vec":
            vals_index_name = "x_pair_int_1"

        return vals_index_name

    def _get_same_x_metric_vals(self, label_1, label_2, epoch_1, epoch_2, tolerate_no_vals=True):
        """
        Get subset of metric df for which metric on samples with same value at x, and the passed values at
        labels and epochs
        Note that x_relationship either relationship for x or x_pair (so "x" in variable name not to be taken
        literally here)
        :param metric_name: name of metric (e.g. cosine_similarity)
        :param label_1: str
        :param label_2: str
        :param epoch_1: int
        :param epoch_2: int
        :param tolerate_no_vals: True to raise error if no df entries found (useful for debugging)
        :return: subset of df
        """

        metric_name = self._get_metric_name()
        vector_type = self._get_vector_type()
        metadata = get_metadata(metric_name, vector_type)
        metric_df = self.fetch1_dataframe()

        # Filter for cases where "x" relationship the same
        metric_df = df_filter_columns(metric_df, {f"{metadata.x_name}_relationship": "same"})

        # For each "x" value, find the single entry for column settings (allowing the sets that compose
        # the pair to be in either order), then recombine these across "x" values into a single df
        column_set_1 = {"label": label_1, "epoch": epoch_1}
        column_set_2 = {"label": label_2, "epoch": epoch_2}
        x_vals = np.unique(np.concatenate([metric_df[f"{metadata.x_name}_{idx}"] for idx in [1, 2]])
                           )  # get unique "x" values across either item in pair

        # Loop through unique "x" values and get df entry for each
        df_subset = pd.concat([df_filter1_columns_symmetric(
            metric_df, {**column_set_1, **{metadata.x_name: x_val}},
            {**column_set_2, **{metadata.x_name: x_val}}, tolerate_no_vals) for x_val in x_vals])

        # Set index to x/x pair, get metric, and sort index (bear in mind that "x"_1 and "x"_2 are same by
        # construction in this function, so we can just sort by "x"_1)
        df_subset = df_subset.set_index(f"{metadata.x_name}_1")[self._metric_column_name(metric_name)].sort_index()

        # Ensure x/x pair represented no more than once
        check_all_unique(df_subset.index)

        # Name index based on epoch pair and label pair
        df_subset.name = self._get_eps_labels(epoch_1, epoch_2, label_1, label_2)

        return df_subset  # TODO (feature): return another df with num_samples

    @staticmethod
    def _eps_labels_split_char():

        return "_"

    @classmethod
    def _get_eps_labels(cls, epoch_1, epoch_2, label_1, label_2):

        # Check inputs
        if not all([float(x).is_integer() for x in [epoch_1, epoch_2]]):
            raise Exception(f"epoch_1 and epoch_2 must both be integers")

        return cls._eps_labels_split_char().join([str(epoch_1), str(epoch_2), label_1, label_2])

    def _split_eps_labels(self, eps_labels, nwb_file_name=None, include_reversed_pairs=False):

        # Split eps_labels by separating character
        split_char = self._eps_labels_split_char()
        split_eps_labels = eps_labels.split(split_char)

        # Get epochs
        epoch_1 = int(split_eps_labels[0])
        epoch_2 = int(split_eps_labels[1])

        # Get the joined labels portion of eps_labels
        joined_labels = split_char.join(split_eps_labels[2:])

        # Make map from joined labels to the two comprising labels
        _, pairs_map = self._get_label_relationship_inputs(
            nwb_file_name, [epoch_1, epoch_2], include_reversed_pairs=include_reversed_pairs)
        pairs = np.concatenate(list([np.concatenate(list(v.values())) for v in pairs_map.values()]))
        joined_labels_map = {split_char.join(pair): pair for pair in pairs}
        label_1, label_2 = joined_labels_map[joined_labels]

        # Use map to get comprising labels for joined labels
        return epoch_1, epoch_2, label_1, label_2

    def get_label_relationship_metric_map(self, relationships=None):
        # Get map from label (e.g. path) pair relationship to df with instances of metric on average
        # difference vectors with same element pair, same epoch, same or different label

        valid_epochs, pairs_map = self._get_label_relationship_inputs()

        # Get relationships if not passed
        if relationships is None:
            # Since pairs map is like {epoch: {relationship: [], ...}, ...} we must unpack relationships. These
            # should be the same across epochs. Unpack single set of relationships and raise error if not a single set.
            relationships = unpack_single_vector([list(v.keys()) for v in pairs_map.values()])

        # Make map
        metric_map = {relationship: [
            self._get_same_x_metric_vals(label_1, label_2, epoch, epoch)
            for epoch in valid_epochs for label_1, label_2 in pairs_map[epoch][relationship]
        ] for relationship in relationships}

        # Raise error if metric map empty
        if len(metric_map) == 0:
            raise Exception(f"metric_map empty. Check if the variable relationships is as expected. If it is not, "
                            f"check if label_name is as expected, as this determines relationships.")
        # Convert to dfs
        return {k: pd.concat(v, axis=1) for k, v in metric_map.items()}

    def get_x_relationship_metric_df(self):
        # Note that x_relationship either relationship for x or x_pair (so "x" in variable name not to be taken
        # literally here)

        metric_name = self._get_metric_name()
        vector_type = self._get_vector_type()
        metadata = get_metadata(metric_name, vector_type)
        metric_df = self.fetch1_dataframe()

        # Check that metric matrix entries represented only once if row and column idxs available (this currently
        # exists for single trial (ST) tables)
        if all([x in metric_df for x in ["row_idxs", "column_idxs"]]):
            row_column_idxs = [
                (x, y) for z in list(zip_df_columns(metric_df, ["row_idxs", "column_idxs"])) for x, y in zip(*z)]
            if len(row_column_idxs) != len(set(row_column_idxs)):
                raise Exception(f"Duplicate matrix entries. This is unexpected")

        # We want to put entries for (label_1 = a, label_2 = b, epoch_1 = c, epoch_2 = d) in the same array,
        # regardless of whether label_1, epoch_1 comes "first" or "second" (i.e. whether
        # we have (a, b, c, d) or (b, a, d, c)).
        # Note that whether label_1, epoch_1 comes "first" or "second" depends on which is present first (earlier
        # indexed rows) in metric_df. Since metric_df constructed using multiprocessing, the order of rows can vary
        # in a random manner, and so therefore for two metric_dfs with the same set of rows but in different orders,
        # one could end up with label_1, epoch_1 "first" and the other with label_1, epoch_1 "second". Practically,
        # this has been seen to occur for metric_dfs that differ only in the random subset of brain region units
        # used.
        # Note that the following shorter line did not work for getting these sets:
        # np.unique([{(x.label_1, x.epoch_1), (x.label_2, x.epoch_2)} for _, x in metric_df.iterrows()])
        # The snippet below takes about 10s to run
        unique_sets = []
        for _, x in metric_df.iterrows():
            set_1 = ((x.label_1, x.epoch_1), (x.label_2, x.epoch_2))
            set_2 = (set_1[1], set_1[0])
            if np.logical_and(set_1 not in unique_sets, set_2 not in unique_sets):
                unique_sets.append(set_1)

        data_list = []
        for unique_set in unique_sets:
            arrs = []
            # Define orders: single order if first and second items in unique_set are the same (e.g. same path and epoch
            # across items), or each ordering if items not the same (e.g. different path and/or epoch across items)
            orders = [(0, 1), (1, 0)]  # default
            if unique_set[0] == unique_set[1]:
                orders = [(0, 1)]
            for idx1, idx2 in orders:
                label_1, epoch_1 = unique_set[idx1]
                label_2, epoch_2 = unique_set[idx2]
                filter_key = {"label_1": label_1, "label_2": label_2, "epoch_1": epoch_1, "epoch_2": epoch_2}
                df_subset = df_filter_columns(metric_df, filter_key)

                # Get min and max x values to help initialize array for neural state below
                x_names = [f"{metadata.x_name}_{idx}" for idx in [1, 2]]
                max_x_1, max_x_2 = [int(max(metric_df[x])) for x in x_names]
                min_x_1, min_x_2 = [int(min(metric_df[x])) for x in x_names]

                # Initialize array with metric for x/x pair pairs
                arr = np.zeros((max_x_1 - min_x_1 + 1, max_x_2 - min_x_2 + 1
                                ))  # add one to account for zero, e.g. from -9 to 20 there are 30 values (20 + 9 + 1)
                arr[:] = np.nan
                # Populate array with metric
                for i, j, z in list(zip_df_columns(df_subset, x_names + [self._metric_column_name(metric_name)])):
                    # Order x1 and x2. Rationale:
                    # x1 and x2 not consistently ordered in metric_df (though note we do not expect duplicates of
                    # data, and checked there were none above if current code allowed), but we want all the values
                    # for a single x1 / x2 combination in one place
                    i1, i2 = min_max([i, j])
                    # ...shift x values (one indexed) to get indices
                    i1 -= min_x_1
                    i2 -= min_x_2
                    arr[int(i1), int(i2)] = z
                arrs.append(arr)  # store array

            # Take first array if only one
            if len(arrs) == 1:
                arr = arrs[0]

            # Otherwise, check that arrays dont have overlapping elements, then put finite elements from
            # one array into the other.
            else:
                # Take transpose of first array so that rows / columns match that of final array. For example
                # right to handle path in epoch 2 always in rows, and left to handle path epoch 4 always in columns
                arrs[0] = arrs[0].T  # transpose one of the arrays

                # Check that arrays dont have entries in same slots
                if np.sum(np.isfinite(arrs[0]).astype(int) + np.isfinite(arrs[1]).astype(int) > 1) != 0:
                    raise Exception(f"overlapping elements across arrays; this is unexpected")

                # Put finite elements from first array into final array (use final since this corresponds with
                # label_1, label_2, epoch_1, epoch_2 from last iteration through above for loop)
                arr = copy.deepcopy(arrs[1])
                mask = np.isfinite(arrs[0])
                arr[mask] = arrs[0][mask]

            # Convert to df so can keep x information
            df = pd.DataFrame(arr, index=np.arange(min_x_1, max_x_1 + 1), columns=np.arange(min_x_2, max_x_2 + 1))
            df.index.name, df.columns.name = x_names

            # Store (using final order)
            data_list.append((label_1, label_2, epoch_1, epoch_2, df))

        return df_from_data_list(data_list, ["label_1", "label_2", "epoch_1", "epoch_2", "metric_arr"])

    def get_cov_fr_vec_params_table(self):
        # Get upstream covariate firing rate vector table params table
        return self._fr_vec_table()()._get_params_table()()

    def get_cov_fr_vec_meta_param_name(self):
        # Get upstream covariate firing rate vector table param name (e.g. PathFRVecParams param name)
        return self.get_cov_fr_vec_params_table().meta_param_name()

    def get_labels_description(self, kwargs):

        # Get upstream covariate firing rate vector table (e.g. PathFRVec)
        params_table = self.get_cov_fr_vec_params_table()

        # Get param name for the table
        meta_param_name = self.get_cov_fr_vec_meta_param_name()
        cov_fr_vec_param_name = copy.deepcopy(kwargs).pop(meta_param_name, None)

        return fetch1_tolerate_no_entry(
            params_table & {meta_param_name: cov_fr_vec_param_name}, "labels_description")

    def _get_df_outer_loop(self, inner_fn, nwb_file_names, epochs_descriptions_names, verbose,
                           res_epoch_spikes_sm_param_name, zscore_fr,
                           brain_region_cohort_name, brain_regions_, curation_set_name,
                            min_epoch_mean_firing_rate, unit_subset_type, unit_subset_size,
                           unit_subset_iterations, tolerate_missing, extra_column_names=None, **kwargs):

        # Get inputs if not passed
        if res_epoch_spikes_sm_param_name is None:
            res_epoch_spikes_sm_param_name = get_default_param("res_epoch_spikes_sm_param_name")
        if zscore_fr is None:
            zscore_fr = get_default_param("zscore_fr")
        # ...Get upstream covariate firing rate vector table param name
        meta_param_name = self.get_cov_fr_vec_meta_param_name()
        cov_fr_vec_param_name = copy.deepcopy(kwargs).pop(meta_param_name, None)
        if cov_fr_vec_param_name is None:
            cov_fr_vec_param_name = get_default_param(meta_param_name)
        if curation_set_name is None:
            curation_set_name = get_default_param("curation_set_name")
        if brain_regions_ is None:
            brain_regions_ = np.unique(np.concatenate(BrainRegionCohort().fetch("brain_regions")))
        # ...extra column names for df
        if extra_column_names is None:
            extra_column_names = []

        # Define key for querying tables
        key = {"brain_region_cohort_name": brain_region_cohort_name,
               "res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name, "zscore_fr": zscore_fr,
               meta_param_name: cov_fr_vec_param_name}

        # Add extra items to key if passed
        # TODO (feature): define programmatically based on upstream table
        for k in ["ppt_dig_param_name", "path_ave_fr_vec_param_name", "path_fr_vec_st_ave_param_name",
                  "time_rel_wa_dig_single_axis_param_name", "time_rel_wa_dig_param_name",
                  "time_rel_wa_ave_fr_vec_param_name", "time_rel_wa_fr_vec_st_ave_param_name"]:
            if k in kwargs:
                key.update({k: kwargs[k]})

        data_list = []
        vals_index_name_list = []
        for nwb_file_name, epochs_descriptions_name in zip(nwb_file_names, epochs_descriptions_names):
            if verbose:
                print(f"On nwb_file_name: {nwb_file_name}...")

            # Update key
            key.update({"nwb_file_name": nwb_file_name})

            # Get brain regions for this nwb file that are specified by cohort and are in passed brain_regions
            brain_regions = [x for x in (BrainRegionCohort & key).fetch1("brain_regions") if x in brain_regions_]

            # Get map from brain region to curation name
            curation_names_df = (CurationSet & {"brain_region_cohort_name": brain_region_cohort_name,
                                                 "curation_set_name": curation_set_name,
                                                 "nwb_file_name": nwb_file_name}).fetch1_dataframe()

            for brain_region in brain_regions:
                if verbose:
                    print(f"\nOn brain region: {brain_region}...")

                # Update key
                key.update({"brain_region": brain_region})

                # Get rat name for storing later
                subject_id = get_subject_id(nwb_file_name)

                # Get epochs descriptions
                epochs_descriptions = (EpochsDescriptions & {
                    "nwb_file_name": nwb_file_name,
                    "epochs_descriptions_name": epochs_descriptions_name}).fetch1("epochs_descriptions")

                for epochs_description in epochs_descriptions:
                    if verbose:
                        print(f"On epochs_description: {epochs_description}...")

                    curation_name = df_pop(curation_names_df, {
                        "nwb_file_name": nwb_file_name, "epochs_description": epochs_description,
                        "brain_region": brain_region}, "curation_name")
                    key.update({"curation_name": curation_name})

                    for unit_subset_iteration in unit_subset_iterations:
                        if verbose:
                            print(f"On unit_subset_iteration {unit_subset_iteration}...")

                        # Get units param name
                        key.update({"epochs_description": epochs_description})
                        epoch = unpack_single_element((EpochsDescription & key).fetch1("epochs"))
                        brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(
                            nwb_file_name, epoch, min_epoch_mean_firing_rate=min_epoch_mean_firing_rate,
                            unit_subset_type=unit_subset_type, unit_subset_size=unit_subset_size,
                            unit_subset_iteration=unit_subset_iteration)
                        key.update({"brain_region_units_param_name": brain_region_units_param_name})

                        # Get subset of table with firing rate vector analysis results corresponding to current settings
                        table_subset = (self & key)

                        # If no entry in brain region units table, continue
                        if len(BrainRegionUnitsFail & key) == 1:
                            continue

                        # If no entry in table, either skip or throw error as indicated
                        if len(table_subset) == 0:
                            error_message = f"no entry in {get_table_name(self)} for key {key}. Continuing..."

                            if tolerate_missing:
                                print(error_message)
                                continue
                            raise Exception(error_message)

                        inner_fn(table_subset, vals_index_name_list, data_list, subject_id, nwb_file_name,
                                 epochs_description, brain_region_units_param_name, brain_region, **kwargs)

        # Get name of vals index
        vals_index_name = check_return_single_element(vals_index_name_list).single_element

        # Convert to df
        return df_from_data_list(
            data_list, ["brain_region", "subject_id", "nwb_file_name", "epochs_description", vals_index_name,
                        "x_val", "val", "brain_region_units_param_name"] + extra_column_names)

    def get_default_relationships(self, **kwargs):
        # Return relationships between entities (e.g. paths if a path-based table, wells if a well-based table)
        raise Exception(f"Must be implemented in child class")

    def _add_default_relationships(self, kwargs):

        # Add default relationships to kwargs if not in kwargs
        relationships = kwargs.pop("relationships", None)
        if relationships is None:
            relationships = self.get_default_relationships(**kwargs)
        kwargs.update({"relationships": relationships})

        return kwargs

    def get_concat_metric_df(
            self, nwb_file_names, epochs_descriptions_names, brain_region_cohort_name,
            brain_regions=None, curation_set_name=None, min_epoch_mean_firing_rate=.1,
            res_epoch_spikes_sm_param_name=None, unit_subset_type=None, unit_subset_size=None,
            unit_subset_iterations=None, zscore_fr=None, verbose=True, tolerate_missing=True,
            **kwargs):

        # Helper function for getting metric values across table entries. Return in a df

        # Add default relationships to kwargs if not present
        kwargs = self._add_default_relationships(kwargs)

        def unpack_df(table_subset, vals_index_name_list, data_list, subject_id, nwb_file_name, epochs_description,
                      brain_region_units_param_name, brain_region, **kwargs):

            # Get map from label pair relationship to df with results
            relationships = kwargs["relationships"]
            metric_map = table_subset.get_label_relationship_metric_map(relationships)

            # Loop through relationships, conditions, and value index and add tuples that will correspond to
            # df entries
            for relationship, df in metric_map.items():

                # note that eps_labels below is a combination of epoch pair and label (e.g. path)
                for eps_labels, vals in df.items():

                    # Store name of index. will then check that same across instances and take the
                    # single instance to define df column
                    vals_index_name_list.append(vals.index.name)

                    # Get x vals reflecting covariate (rough example: 10th of 10 total path fraction bins
                    # corresponds to .95 path fraction)
                    x_vals = table_subset._metric_index_as_bin_centers(vals.index)

                    for (val_index, val), x_val in zip(vals.items(), x_vals):

                        # Append to list that will be used to construct df
                        # Note: store brain_region_units_param_name just for convenience. e.g. helpful for
                        # finding unit information corresponding to entries
                        data_list.append(
                            (brain_region, subject_id, nwb_file_name, epochs_description,
                             val_index, x_val, val, brain_region_units_param_name,
                             relationship, eps_labels, get_nwb_file_name_epochs_description(
                                nwb_file_name, epochs_description)))

        extra_column_names = ["relationship", "eps_labels", "nwb_file_name_epochs_description"]

        return self._get_df_outer_loop(
            unpack_df, nwb_file_names, epochs_descriptions_names, verbose, res_epoch_spikes_sm_param_name, zscore_fr,
            brain_region_cohort_name, brain_regions, curation_set_name,
                           min_epoch_mean_firing_rate, unit_subset_type, unit_subset_size,
                           unit_subset_iterations, tolerate_missing, extra_column_names, **kwargs)

    def get_concat_same_diff_x_ratio_df(
            self, nwb_file_names, epochs_descriptions_names, brain_region_cohort_name,
            brain_regions=None, curation_set_name=None, min_epoch_mean_firing_rate=.1,
            res_epoch_spikes_sm_param_name=None, unit_subset_type=None, unit_subset_size=None,
            unit_subset_iterations=None, zscore_fr=None, verbose=True, tolerate_missing=True, **kwargs):

        # Helper function for getting a ratio of metric values across table entries. Return in a df

        # Add ratio name to kwargs if not there
        if "ratio_name" not in kwargs:
            kwargs["ratio_name"] = "one_minus_on_off_diagonal_ratio"

        # Add default relationships to kwargs if not there
        kwargs = self._add_default_relationships(kwargs)

        def unpack_df(table_subset, vals_index_name_list, data_list, subject_id, nwb_file_name, epochs_description,
                      brain_region_units_param_name, brain_region, **kwargs):

            ratio_name = kwargs["ratio_name"]

            df = table_subset.get_same_diff_x_ratio_df(**kwargs)

            # Loop through conditions and value index and add tuples that will correspond to df entries
            # note that eps_labels below is a combination of epoch pair and label (e.g. path)
            for _, df_row in df.iterrows():
                vals = getattr(df_row, ratio_name)
                # store name of index. will then check that same across instances and take the
                # single instance to define df column
                vals_index_name_list.append(vals.index.name)
                # get x vals reflecting covariate (rough example: 10th of 10 total path fraction bins
                # corresponds to .95 path fraction)
                x_vals = table_subset._metric_index_as_bin_centers(vals.index)
                for (val_index, val), x_val in zip(vals.items(), x_vals):
                    # note: store brain_region_units_param_name just for convenience. e.g. helpful for
                    # finding unit information corresponding to entries
                    data_list.append(
                        (brain_region, subject_id, nwb_file_name, epochs_description, val_index, x_val, val,
                         brain_region_units_param_name, df_row.label_1, df_row.label_2, df_row.relationship,
                         df_row.eps_labels,
                         get_nwb_file_name_epochs_description(nwb_file_name, epochs_description)))

        extra_column_names = ["label_1", "label_2", "relationship", "eps_labels", "nwb_file_name_epochs_description"]

        return self._get_df_outer_loop(
            unpack_df, nwb_file_names, epochs_descriptions_names, verbose, res_epoch_spikes_sm_param_name, zscore_fr,
            brain_region_cohort_name, brain_regions, curation_set_name,
                        min_epoch_mean_firing_rate, unit_subset_type, unit_subset_size,
                           unit_subset_iterations, tolerate_missing, extra_column_names, **kwargs)

    def get_same_diff_x_ratio_df(self, include_nwb_file_name=True, include_brain_region=True, **kwargs):
        # Get ratio of metric in same x bin vs different x bin, for same epoch

        def _symmetric_matrix(arr):
            # Make matrix symmetric (expected form for getting on/off diagonal ratios)
            lower_idxs = np.tril_indices(check_return_single_element(np.shape(arr)).single_element, -1)
            arr[lower_idxs] = arr.T[lower_idxs]
            return arr

        # Get ratio of metric on vs. off diagonal
        metric_arr_df = self.get_x_relationship_metric_df()
        diag_ratio_df = copy.deepcopy(metric_arr_df)
        diag_ratio_df.drop(columns=["metric_arr"], inplace=True)
        diag_ratio_df["on_off_diagonal_ratio"] = [
            pd.Series(on_off_diagonal_ratio(_symmetric_matrix(x.to_numpy())), index=x.index) for x in
            metric_arr_df.metric_arr]

        # Also get one minus the above ratio (in some contexts this is easier to interpret)
        diag_ratio_df["one_minus_on_off_diagonal_ratio"] = [
            1 - x for x in diag_ratio_df["on_off_diagonal_ratio"]]

        # Apply three restrictions: same epoch, valid labels, and
        # additional labels restriction based on labels description

        # 1) same epoch
        epoch_bool = diag_ratio_df["epoch_1"] == diag_ratio_df["epoch_2"]

        # 2) valid labels

        # ...To help with getting valid labels, first get potentially rewarded paths or wells
        key = self.fetch1("KEY")
        valid_epochs = (EpochsDescription & key).fetch1("epochs")
        valid_labels = MazePathWell().get_rewarded_path_names_across_epochs(key["nwb_file_name"], valid_epochs, True)

        # ...Update valid_labels to have even/odd, stay/leave, even/odd stay, or correct/incorrect text if indicated
        labels_description = self.get_labels_description(kwargs)
        if labels_description == "none":
            pass
        elif labels_description == "even_odd_trials":
            valid_labels = np.concatenate(
                [[MazePathWell.get_even_odd_trial_name(x, even_odd_text=even_odd_text) for x in valid_labels]
                    for even_odd_text in MazePathWell.even_odd_trial_text()])
        elif labels_description in ["stay_leave_trials", "stay_leave_trials_pre_departure"]:
            valid_labels = np.concatenate(
                [[MazePathWell.get_stay_leave_trial_path_name(x, trial_text=trial_text) for x in valid_labels]
                    for trial_text in MazePathWell.stay_leave_trial_text()])
        elif labels_description == "even_odd_stay_trials":
            valid_labels = [MazePathWell.get_stay_leave_trial_path_name(
                x, trial_text=MazePathWell.stay_leave_trial_text("stay")) for x in valid_labels]
            valid_labels = np.concatenate(
                [[MazePathWell.get_even_odd_trial_name(x, even_odd_text=even_odd_text) for x in valid_labels]
                    for even_odd_text in MazePathWell.even_odd_trial_text()])
        elif labels_description == "correct_incorrect_trials":
            valid_labels = np.concatenate([[MazePathWell.get_correct_incorrect_trial_text(
                x, correct_incorrect_text=correct_incorrect_trial_text) for x in valid_labels]
                for correct_incorrect_trial_text in MazePathWell.correct_incorrect_trial_text()])
        else:
            raise Exception(f"{labels_description} not valid labels_description")

        valid_labels_bool = np.logical_and(diag_ratio_df["label_1"].isin(valid_labels),
                                           diag_ratio_df["label_2"].isin(valid_labels))

        # 3) additional label restriction based on labels description

        # ...Restrict labels based on cases
        labels_bool = [True]*len(diag_ratio_df)  # initialize

        # Case 1: No further restriction
        if labels_description in [
            "none", "stay_leave_trials", "stay_leave_trials_pre_departure", "correct_incorrect_trials"]:
            pass
        # Case 2: Restrict to across even and odd trials
        elif labels_description in ["even_odd_trials", "even_odd_stay_trials"]:
            labels_bool = [
                MazePathWell.even_odd(df_row.label_1, df_row.label_2) for idx, df_row in diag_ratio_df.iterrows()]
        else:
            raise Exception(f"{labels_description} not valid labels_description")

        # Apply restrictions
        valid_bool = np.logical_and.reduce((epoch_bool, valid_labels_bool, labels_bool))
        df = diag_ratio_df[valid_bool]

        # ...Add label relationships
        valid_epochs, pairs_map = self._get_label_relationship_inputs()
        valid_relationships = kwargs["relationships"]  # potential relationships
        # check that epochs same; current code assumes this
        if not all(df.epoch_1 == df.epoch_2):
            raise Exception(f"Code currently only set up for case where epoch_1 equals epoch_2")
        # find relationships for each row of df. There is one case where a matching relationship is expected
        # to not be found: if the two labels are not in a relationship that is coded up (e.g. path pairs that
        # include non-rewarded paths)
        data_list = []  # initialize
        for idx, df_row in df.iterrows():
            # exclude relationships that are not in ones that were passed to function
            pairs_map_subset = {k: v for k, v in pairs_map[df_row.epoch_1].items() if k in valid_relationships}
            # find matching relationships. OR operation since labels could be in either order. Above, we already
            # excluded duplicate samples
            relationship_matches = [relationship for relationship, v in pairs_map_subset.items()
                                    if np.logical_or((df_row.label_1, df_row.label_2) in v,
                                                     (df_row.label_2, df_row.label_1) in v)]
            # loop through matching relationships and add a df row for each
            for relationship in relationship_matches:
                data_list.append(list(df_row.values) + [relationship])
        # raise error if no valid relationships found; this is unexpected
        if len(data_list) == 0:
            raise Exception(f"No identified relationships matched those in passed relationships; this is unexpected")

        # make df
        df = df_from_data_list(data_list, list(df.columns) + ["relationship"])

        # ...Add characteristics to df if indicated (so can distinguish entries in concatenated df across these)
        if include_nwb_file_name:
            df["nwb_file_name"] = key["nwb_file_name"]
        if include_brain_region:
            df["brain_region"] = key["brain_region"]

        # ...Add column with combination of epoch and label, to be consistent with concat_metric_df
        df["eps_labels"] = [
            self._get_eps_labels(x.epoch_1, x.epoch_2, x.label_1, x.label_2) for idx, x in df.iterrows()]

        # Raise error if empty df
        if len(df) == 0:
            raise Exception(f"within get_same_diff_x_ratio_df. No entries left in df; this is unexpected")

        return df

    def _get_plot_color(self):
        # Get color based on brain region

        return (BrainRegionColor & self.fetch1("KEY")).fetch1("color")

    def _metric_index_as_bin_centers(self, metric_index, key=None):
        # Convert metric index to bin centers units

        if key is None:
            key = dict()

        # x vals varies depending on whether we are dealing with metric as a function of x, or as a function
        # of x pair
        bin_centers_name_map = {"x_1": "x", "x_pair_int_1": "x_pair"}
        bin_centers_name = bin_centers_name_map[metric_index.name]
        bin_centers_map = (self & key).get_bin_centers_map()[bin_centers_name]

        return bin_centers_map.loc[metric_index]

    def plot_x_relationship_metric_arr(
            self, label_1, label_2, epoch_1, epoch_2, fig_ax_list=None, save_fig=False,
            plot_color_bar=True, **format_kwargs):

        # Get metric array for given characteristics
        metric_arr_df = self.get_x_relationship_metric_df()
        metric_arr = df_pop(metric_arr_df, {
            "label_1": label_1, "label_2": label_2, "epoch_1": epoch_1, "epoch_2": epoch_2}, "metric_arr")

        # Convert to bin centers units
        x_vals = self._metric_index_as_bin_centers(metric_arr.index).values
        metric_arr = pd.DataFrame(metric_arr.to_numpy(), columns=x_vals, index=x_vals)

        # Unpack figure / axis if passed, otherwise initialize
        if fig_ax_list is None:
            fig_ax_list = plt.subplots(figsize=(3, 2.5))
        fig, ax = fig_ax_list

        # Plot heatmap with metric
        # Here, transpose so data in left upper half of plot
        kwargs = {"spines_off_list": ["right", "bottom"]}
        metric_arr = metric_arr.T
        plot_heatmap(metric_arr, fig_ax_list=fig_ax_list, zorder=0, plot_color_bar=plot_color_bar, **kwargs)
        ax.xaxis.tick_top()  # x ticks on top

        # Format axis
        label_every_n = format_kwargs.pop("label_every_n", 2)
        ticklabels = get_ticklabels([np.round(x, 3) for x in metric_arr.index], idxs=np.arange(0, len(metric_arr), label_every_n))
        ticks = np.arange(0, len(ticklabels)) + .5  # place ticks in center of each bin
        # ...update passed format params with those defined above
        format_kwargs.update(kwargs)
        format_ax(ax=ax, xticks=ticks, yticks=ticks, xticklabels=ticklabels, yticklabels=ticklabels, **format_kwargs)
        ax.xaxis.set_label_position("top")  # put x label (if passed) on top

        # Add staircase frame around diagonal portion of matrix
        kwargs = {"color": "black", "linewidth": 1, "zorder": 100}
        # Remaining horizontal steps
        for x in np.arange(0, len(metric_arr)):
            # cant get zorder to work so that first horizontal step and last vertical line not hidden behind
            # heatmap, so for now increase linewidth for these lines
            # horizontal step
            use_kwargs = copy.deepcopy(kwargs)
            if x == 0:
                use_kwargs.update({"linewidth": 2})
            ax.plot([x, x + 1], [x]*2, **use_kwargs)
            # vertical line
            use_kwargs = copy.deepcopy(kwargs)
            if x == len(metric_arr) - 1:
                use_kwargs.update({"linewidth": 2})
            ax.plot([x + 1]*2, [x, x + 1], **use_kwargs)

        # Save figure if indicated
        if save_fig:
            key = self.fetch1("KEY")
            metric_name = self.get_metric_name()
            file_name_save = f"fr_vec_{metric_name}_heatmap_{label_1}_{label_2}_ep1{epoch_1}_ep2{epoch_2}_" \
                f"{format_nwb_file_name(key['nwb_file_name'])}_" \
                f"{key['brain_region']}_{key['brain_region_units_param_name']}_" \
                f"sm{key['res_epoch_spikes_sm_param_name']}"
            save_figure(fig, file_name_save, save_fig=save_fig)

    @staticmethod
    def _flatten_metric_map_vals(metric_map):
        return np.hstack([np.ndarray.flatten(x.values) for x in metric_map.values()])

    @staticmethod
    def get_valid_covariate_bin_nums(key):
        raise Exception(f"This method should be implemented in child class")

    def get_bin_centers_map(self, **kwargs):
        raise Exception(f"This method should be implemented in child class")

    def exclude_invalid_bins(self, df):

        valid_bin_nums = self.get_valid_covariate_bin_nums(self.fetch1("KEY"))
        valid_bool = [True] * len(df)
        for column_name in ["x_pair", "x_pair_partner_1", "x_pair_partner_2", "x_1", "x_2"]:
            if column_name in df:
                arr = np.vstack(df[column_name])
                valid_bool *= np.prod(
                    np.logical_and(arr >= np.min(valid_bin_nums), arr <= np.max(valid_bin_nums)), axis=1)

        return df.iloc[np.where(valid_bool)[0]]

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None,
                         exclude_invalid_bins=True):

        # Convert arrays back to tuples
        df = super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)
        for column_name in ["x_pair_partner_1", "x_pair_partner_2"]:
            if column_name in df:
                df[column_name] = array_to_tuple_list(df[column_name])

        # Exclude bins outside covariate range if indicated
        if exclude_invalid_bins:
            df = self.exclude_invalid_bins(df)

        return df


class CovariateFRVecSTAveBase(CovariateFRVecAveBase):

    @staticmethod
    def _metric_column_name(metric_name):

        return {"cosine_similarity": "cosine_similarity_mean",
                "euclidean_distance": "euclidean_distance_mean"}[metric_name]

    def make(self, key):

        def _get_mask_across_epochs(df_name):
            # Make mask that: 1) excludes sample pairs closer in index than a specified amount, 2) prevents
            # double counting sample pairs

            # ...Get threshold for excluding samples too close in index
            params_table = self._get_params_table()
            mask_duration = (params_table & key).fetch1("mask_duration")

            # ...Get closeness mask from each epoch from class. Requires first checking that data in table matches
            # that in reinitialized class
            fr_vec_table = self._fr_vec_table()
            masks = []
            for epoch in (EpochsDescription & key).fetch1("epochs"):
                key.update({"epoch": epoch})
                df = (fr_vec_table & key).fetch1_dataframe(df_name)

                # Initialize class again here
                inputs = fr_vec_table().get_inputs(key)
                obj = AverageVectorDuringLabeledProgression(inputs.x, inputs.labels, inputs.df)

                # Check that same data across dfs previously stored in datajoint table and newly initialized class,
                # then proceed to use newly initialized class. Note we cannot use built in equals method for
                # dfs since saving df out to nwb file converts tuples to arrays, so dfs before and after getting
                # saved out may have all the same values but not be strictly equal
                dfs_same_values([getattr(obj, df_name), df])

                # Store closeness mask for this epoch
                idxs_col_name = unpack_single_element([x for x in df.columns if "idxs" in x])
                mask = obj._get_closeness_mask(getattr(obj, df_name), idxs_col_name, mask_duration, verbose=False)
                masks.append(mask)

            # ...Make mask across epochs (populate a large mask with epoch masks)
            mask_lens = [len(x) for x in masks]
            valid_bool = np.ones([np.sum(mask_lens)] * 2)
            for (start_idx, end_idx), mask in zip(zip_adjacent_elements(np.concatenate(([0], np.cumsum(mask_lens)))),
                                                  masks):
                valid_bool[start_idx:end_idx, start_idx:end_idx] = mask

            # ...Mask out one half of symmetric matrix to avoid duplicating measurements
            tril_mask = np.zeros_like(valid_bool, dtype=bool)
            tril_mask[np.tril_indices_from(tril_mask)] = True
            valid_bool *= tril_mask
            return valid_bool

        # Get matrix entries corresponding to settings of trial characteristics
        # Note we use x_pair_int and not x_pair for x information for cosine similarity df because
        # easier to perform df operations with integer (form of x_pair_int) than tuple (form of x_pair)
        fr_vec_table = self._fr_vec_table()
        metric_name, vector_type = (self._get_params_table() & key).fetch1("metric_name", "vector_type")
        metadata = get_metadata(metric_name, vector_type)

        characteristics = [metadata.x_name, "epoch", "label"]

        df = parse_matrix(
            fr_vec_table().concat_across_epochs(key, metadata.vector_df_name), characteristics,
            fr_vec_table().metric_across_epochs(key, metric_name, vector_type),
            _get_mask_across_epochs(metadata.vector_df_name), metric_name=metric_name, debug_mode=False,
            verbose=True)

        # Add x_pair_partner as column if working with difference vectors so can query df using this if desired. This
        # contains equivalent information to x_pair_int
        if vector_type == "diff_vec":
            for column_name in ["x_pair_int_1", "x_pair_int_2"]:
                if column_name in df:
                    df[column_name.replace("int", "partner")] = [
                        AverageVectorDuringLabeledProgression.get_x_pair(x) for x in df[column_name]]

        # Add relationships between x or x_pair_int, between epochs, and between labels to df. These
        # can be helpful when querying the df
        for column_name in characteristics:
            df[f"{column_name}_relationship"] = get_relationship_texts(
                zip(*[df[f"{column_name}_{x}"].values for x in [1, 2]]))

        # Insert into main table
        key.pop("epoch")
        insert_analysis_table_entry(self, [df], key)

        # Insert into part table
        for epoch in (EpochsDescription & key).fetch1("epochs"):
            key.update({"epoch": epoch})
            insert1_print(self.Upstream, key)

    def get_default_relationships(self, **kwargs):
        # Get relationships depending on labels description

        # Get labels description
        labels_description = self.get_labels_description(kwargs)

        # Stay/leave trials by path
        if labels_description in ["stay_leave_trials", "stay_leave_trials_pre_departure"]:
            return ["same_path_stay_leave_trials", "same_path_stay_stay_trials", "same_path_leave_leave_trials"]

        # Correct/incorrect trials by path
        elif labels_description == "correct_incorrect_trials":
            return [
                "same_path_correct_correct_trials", "same_path_correct_incorrect_trials",
                "same_path_incorrect_incorrect_trials", "outbound_correct_correct_trials",
                "outbound_correct_incorrect_trials", "outbound_correct_incorrect_trials",
                "same_path_outbound_correct_correct_trials",
            ]

        # Same path comparison
        elif labels_description == "none":
            return ["same_path"]

        # Raise error if labels description not accounted for
        else:
            raise Exception(f"default relationships not defined for labels_description {labels_description}")


class CovariateFRVecTrialAveBase(CovariateFRVecAveBase):

    @staticmethod
    def _metric_column_name(metric_name):
        # Have this information here rather than available from get_metadata function because metric column names
        # vary by table, depending on whether table holds trial average or single trial average of metric
        return {"cosine_similarity": "cosine_similarity", "euclidean_distance": "euclidean_distance"}[metric_name]

    def make(self, key):
        # Get cosine similarity of average difference vectors and euclidean distance between average vectors

        # Get metadata
        metric_name, vector_type = (self._get_params_table() & key).fetch1("metric_name", "vector_type")
        metadata = get_metadata(metric_name, vector_type)

        # Concatenate average difference vectors across epochs
        concat_df = self._fr_vec_table()().concat_across_epochs(key, metadata.ave_vector_df_name)
        vectors = np.vstack(concat_df[metadata.ave_vector_col_name])

        # Get metric for pairs of average difference vectors
        metric_arr = metadata.fn(vectors)

        # Match up with metadata for vectors
        characteristics = [metadata.x_name, "label", "epoch"]
        column_tuples = list(zip_df_columns(concat_df, characteristics))

        # Note that x_1 and x_2 below are either x or x_pair (so "x" in variable name not to be taken literally
        # here)
        data_list = [tuple([x_1, x_2, label_1, label_2, epoch_1, epoch_2] + [metric_arr[idx_1, idx_2]])
                     for idx_1, (x_1, label_1, epoch_1) in enumerate(column_tuples)
                     for idx_2, (x_2, label_2, epoch_2) in enumerate(column_tuples)
                     if idx_1 < idx_2]
        df = df_from_data_list(
            data_list, [f"{metadata.x_name}_1", f"{metadata.x_name}_2", "label_1",
                        "label_2", "epoch_1", "epoch_2", metric_name])

        # Add x_pair_partner as column if working with difference vectors so can query df using this if desired. This
        # contains equivalent information to x_pair_int
        if vector_type == "diff_vec":
            for column_name in ["x_pair_int_1", "x_pair_int_2"]:
                if column_name in df:
                    df[column_name.replace("int", "partner")] = [
                        AverageVectorDuringLabeledProgression.get_x_pair(x) for x in df[column_name]]

        # Add relationships between x or x_pair_int, between epochs, and between labels to df. These
        # can be helpful when querying the df
        for column_name in characteristics:
            df[f"{column_name}_relationship"] = get_relationship_texts(
                zip(*[df[f"{column_name}_{x}"].values for x in [1, 2]]))

        # Insert into main table
        insert_analysis_table_entry(self, [df], key)

        # Insert into part table
        for epoch in (EpochsDescription & key).fetch1("epochs"):
            key.update({"epoch": epoch})
            insert1_print(self.Upstream, key)

    def get_default_relationships(self, **kwargs):
        # Return even/odd trials paths if indicated by key (covariate fr vec param name in key)

        labels_description = self.get_labels_description(kwargs)
        label_name = kwargs["label_name"]

        # Even odd trials
        if labels_description == "even_odd_trials":
            # By path
            if label_name == "path":
                return ["same_path_even_odd_trials", "same_turn_even_odd_trials", "different_turn_well_even_odd_trials"]
            # By end well
            elif label_name == "end_well":
                return ["same_end_well_even_odd_trials", "different_end_well_even_odd_trials"]

        elif labels_description == "even_odd_stay_trials":
            # By path
            if label_name == "path":
                return [
                    "same_path_even_odd_stay_trials", "same_turn_even_odd_stay_trials",
                    "different_turn_well_even_odd_stay_trials"]
            # By end well
            elif label_name == "end_well":
                return ["same_end_well_even_odd_stay_trials", "different_end_well_even_odd_stay_trials"]

        # Otherwise return same / different turn comparison
        elif labels_description == "none":
            return ["same_turn", "different_turn_well"]
        else:
            raise Exception(f"default relationships not defined for labels_description {labels_description}")


class PathWellPopSummBase(ComputedBase):
    """
    Base class for population analysis during path and well periods. Used for:
    1) Firing rate vector as a function of covariate (spatial bins during path traversals or temporal bins during
    delay period).
    2) Firing rate vector nearest neighbor similarity analysis.
    3) Decoding task progress during path traversals or the delay period.
    """

    @staticmethod
    def _upstream_table():
        # Return upstream covariate firing rate vector average table
        raise Exception(f"Must be implemented in child class")

    def _get_vals_index_name(self, **kwargs):

        if "key" not in kwargs:
            key = self.fetch1("KEY")
        else:
            key = kwargs["key"]
        upstream_table_subset = self._upstream_table() & key

        return upstream_table_subset._get_vals_index_name()

    @staticmethod
    def _alter_boot_params_rat_cohort(boot_set_name, resample_levels, ave_group_column_names):
        # Alter bootstrap parameters based on whether or not rat cohort

        if "rat_cohort" in boot_set_name:
            resample_levels = ["subject_id"] + resample_levels
        else:
            ave_group_column_names = ["subject_id"] + ave_group_column_names

        return resample_levels, ave_group_column_names

    def _get_relationship_div_column_params(self, **kwargs):

        raise Exception(f"Must be defined in child class")

    @staticmethod
    def _get_pair_column_names(target_column_name):
        # Define rule for getting name of columns for each member of pair
        return [f"{target_column_name}_{x}" for x in [1, 2]]

    @staticmethod
    def _get_joint_column_name(target_column_name):
        # Define rule for getting name of column for pair
        return f"joint_{target_column_name}"

    @staticmethod
    def _get_joint_column_val(pair):
        # Define rule for getting value for column for pair
        return "_".join(pair)

    def _get_brain_region_order_for_pairs(self):
        return ("OFC_targeted", "mPFC_targeted", "CA1_targeted")

    @staticmethod
    def _get_stay_leave_order_for_pairs():
        return ("same_path_stay_leave_trials", "same_path_leave_leave_trials", "same_path_stay_stay_trials")

    def _get_same_different_outbound_path_correct_order_for_pairs(self):
        return ("outbound_correct_correct_trials", "same_path_outbound_correct_correct_trials")

    @classmethod
    def get_paired_metric_df(
            cls, metric_df, target_column_name, target_column_pairs, metric_pair_fn, resample_quantity="val",
            exclude_columns=None):
        # Convert metric in df to paired metric

        # Get inputs if not passed
        if exclude_columns is None:
            exclude_columns = []

        # For each pair of values at an indicated column in metric_df, find settings of all other
        # columns (excluding quantity being resampled and any other user defined columns) that exist
        # for both values in the pair
        column_names = [
            x for x in metric_df.columns if x not in [target_column_name, resample_quantity] + exclude_columns]

        common_column_sets_map = dict()  # store settings of columns (as dictionary) present for both values in pair
        for x1, x2 in target_column_pairs:
            concat_df_subset = df_filter_columns(metric_df, {target_column_name: x1})
            column_sets_1 = unique_df_column_sets(concat_df_subset, column_names, as_dict=True)
            concat_df_subset = df_filter_columns(metric_df, {target_column_name: x2})
            column_sets_2 = unique_df_column_sets(concat_df_subset, column_names, as_dict=True)
            if len([x for x in column_sets_1 if x in column_sets_2]) == 0:
                raise Exception(f"No overlapping column sets; this is not expected")
            common_column_sets_map[(x1, x2)] = [x for x in column_sets_1 if x in column_sets_2]

        # Make new df with difference in values for brain region pairs

        # Define column names in new df
        # ...Define new df columns, one for each member of pair
        pair_column_names = cls._get_pair_column_names(target_column_name)
        # ...Define new df column for pair
        joint_column_name = cls._get_joint_column_name(target_column_name)
        # ...Define all columns
        df_column_names = column_names + [joint_column_name] + pair_column_names + [resample_quantity]

        # Define new df
        data_list = []
        for pair, common_column_sets in common_column_sets_map.items():
            for column_set in common_column_sets:

                # Get metric on pairs of value at target column
                val = metric_pair_fn(metric_df, pair, column_set, target_column_name, resample_quantity)

                # Append to list
                data_list.append(tuple(
                    list(column_set.values()) + [cls._get_joint_column_val(pair)] + list(pair) + [val]))

        # Return new df
        # if empty df, return with column names
        empty_df_has_cols = True
        return df_from_data_list(data_list, df_column_names, empty_df_has_cols)

    @staticmethod
    def metric_pair_div(metric_df, pair, column_set, target_column_name, resample_quantity):
        return np.divide(*[np.mean(df_filter_columns(metric_df, {
                    **column_set, **{target_column_name: x}})[resample_quantity]) for x in pair])

    @staticmethod
    def metric_pair_diff(metric_df, pair, column_set, target_column_name, resample_quantity):
        return unpack_single_element(np.diff([df_pop(metric_df, {
            **column_set, **{target_column_name: x}}, resample_quantity) for x in pair]))

    # Override in children classes where relationship exists
    def _get_relationship_meta_name(self):
        return None

    def get_ordered_relationships(self):
        raise Exception(f"Must be implemented in child class")

    def _get_brain_region_meta_name(self):

        boot_set_name = self.get_upstream_param("boot_set_name")
        params_table = self._get_params_table()
        if boot_set_name in params_table._valid_brain_region_diff_boot_set_names() + \
                            params_table._valid_stay_leave_diff_brain_region_diff_boot_set_names() + \
                params_table._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():
            return self._get_joint_column_name("brain_region")

        return "brain_region"

    def _get_colors_df(self, brain_region_meta_name, relationship_meta_name=None):

        data_list = []

        # Define map from brain region value and relationship value to color

        brain_regions = ["mPFC_targeted", "OFC_targeted", "CA1_targeted"]

        # ...Path and well relationships: colors correspond to brain regions
        relationship_vals_list = [[
            # Darkest tint
            "same_path", "same_path_stay_stay_trials", "same_path_even_odd_trials", "same_well",
            "same_end_well_even_odd_stay_trials",
            "same_path_correct_correct_trials",
            "same_path_outbound_correct_correct_trials",
            ], [
            # Intermediate tint
            "same_turn", "same_turn_even_odd_trials", "same_path_stay_leave_trials",
            "same_path_correct_incorrect_trials"], [
            # Lightest tint
            "different_turn_well", "different_turn_well_even_odd_trials", "different_turn_well_even_odd_stay_trials",
            "different_well", "different_end_well_even_odd_stay_trials",
            "same_path_leave_leave_trials",
            "same_path_incorrect_incorrect_trials",
            "outbound_correct_correct_trials"
            ]]

        colors_map = BrainRegionColor().tint_colors(brain_regions, num_tints=3)
        for brain_region_val, colors in colors_map.items():
            for color, relationship_vals in zip(colors, relationship_vals_list):
                for relationship_val in relationship_vals:
                    data_list.append((brain_region_val, relationship_val, color))

        # ...Same path correct and outbound correct difference: colors correspond to brain regions
        relationship_vals = [
            "outbound_correct_correct_trials_same_path_outbound_correct_correct_trials",
        ]
        for brain_region in brain_regions:
            for relationship_val in relationship_vals:
                data_list.append(
                    (brain_region, relationship_val, BrainRegionColor().get_brain_region_color(brain_region)))

        # ...Stay/leave difference: colors correspond to brain regions
        relationship_vals = [
            "same_path_stay_leave_trials_same_path_stay_stay_trials",
        ]
        for brain_region in brain_regions:
            for relationship_val in relationship_vals:
                data_list.append(
                    (brain_region, relationship_val, BrainRegionColor().get_brain_region_color(brain_region)))

        # ...Brain region difference: gray scale
        brain_region_vals = [
            "mPFC_targeted_OFC_targeted", "OFC_targeted_mPFC_targeted",
            "CA1_targeted_OFC_targeted", "OFC_targeted_CA1_targeted",
            "CA1_targeted_mPFC_targeted", "mPFC_targeted_CA1_targeted"]
        relationship_vals_list = [[
            # Darkest tint
            "same_path", "same_path_stay_stay_trials", "same_path_correct_correct_trials",
            "same_path_stay_stay_trials_same_path_stay_stay_trials",
            "same_path_stay_leave_trials_same_path_stay_stay_trials",
            "same_path_leave_leave_trials_same_path_stay_stay_trials"
        ], [
            # Medium tint
            "same_path_stay_leave_trials", "same_path_correct_incorrect_trials"], [
            # Lightest tint
            "same_path_leave_leave_trials", "same_path_incorrect_incorrect_trials",
        ]]
        color_vals = [.1, .4, .7]
        for relationship_vals, color_val in zip(relationship_vals_list, color_vals):
            for relationship_val in relationship_vals:
                for brain_region_val in brain_region_vals:
                    data_list.append((brain_region_val, relationship_val, [color_val]*3))

        # ...Same path correct and outbound correct difference, brain region difference: gray scale
        brain_region_vals = [
            "mPFC_targeted_OFC_targeted", "OFC_targeted_mPFC_targeted",
            "CA1_targeted_OFC_targeted", "OFC_targeted_CA1_targeted",
            "CA1_targeted_mPFC_targeted", "mPFC_targeted_CA1_targeted"]
        relationship_vals_list = [[
            # Darkest tint
            "outbound_correct_correct_trials_same_path_outbound_correct_correct_trials"]]
        color_vals = [.1, .4, .7]
        for relationship_vals, color_val in zip(relationship_vals_list, color_vals):
            for relationship_val in relationship_vals:
                for brain_region_val in brain_region_vals:
                    data_list.append((brain_region_val, relationship_val, [color_val]*3))

        # ...Relationship ratio: gray scale, colors correspond to path or well pairs
        relationship_vals_list = [[
            # Darkest tint
            "same_turn_even_odd_trials_same_path_even_odd_trials"], [
            # Lightest tint
            "different_turn_well_even_odd_trials_same_path_even_odd_trials",
            "different_end_well_even_odd_stay_trials_same_end_well_even_odd_stay_trials"]]
        color_vals = [.1, .7]
        for brain_region_val in brain_regions:
            for relationship_vals, color_val in zip(relationship_vals_list, color_vals):
                for relationship_val in relationship_vals:
                    data_list.append((brain_region_val, relationship_val, [color_val]*3))

        # Assemble df. If relationship_meta_name not defined, omit this column and use same path shade for each brain
        # region value
        colors_df = df_from_data_list(data_list, [brain_region_meta_name, relationship_meta_name, "color"])
        if relationship_meta_name is None:
            colors_df = df_filter_columns(colors_df, {relationship_meta_name: "same_path"}).drop(
                columns=[relationship_meta_name])

        # Return df with map to colors
        return colors_df

    def _get_val_text(self):
        raise Exception(f"Must be implemented in child class")

    def _get_val_lims(self, **kwargs):
        # Get a set range for value, e.g. for use in plotting value on same range across plots
        raise Exception(f"Must be implemented in child class")

    def _get_val_ticks(self):
        # Get axis ticks for value, e.g. for use across plots
        return None  # default is none

    def _get_xticks(self):
        raise Exception(f"Must be implemented in child class")

    def _get_xticklabels(self, ticks=None):

        if ticks is None:
            ticks = self._get_x_lims()
            if ticks is None:
                return None

        return [format_number(x) for x in ticks]

    def _get_yticklabels(self, ticks=None):

        if ticks is None:
            return ticks

        return [str(x) for x in ticks]

    def _get_x_text(self):
        raise Exception(f"Must be implemented in child class")

    def _get_x_lims(self):
        raise Exception(f"Must be implemented in child class")

    def _plot_params(self):
        return {"reverse_brain_region_panel_order": False}

    def _get_replace_char_map(self):
        return {"potentially_rewarded_trial": "prt",
         "mask_duration": "md", "rat_cohort": "coh", "brain_region_diff": "brdiff", "rand_target_region": "rtr",
         "iterations": "iter"}

    def extend_plot_results(self, **kwargs):
        # Extend in child class if desire further extension. Separate method rather than extent plot_results method
        # directly so can pass arguments created within that function easily

        # Horizontal line at zero
        if not kwargs["empty_plot"]:
            ax = kwargs["ax"]
            plot_spanning_line(ax.get_xlim(), 0, ax, span_axis="x", color="gray", zorder=0)

    def plot_results(self, **kwargs):
        # Get inputs if not passed

        # make copy of kwargs (except axis; copying causes creation of new axis) to avoid changing outside function
        plot_params = copy.deepcopy({k: v for k, v in kwargs.items() if k != "ax"})

        # Get df with average and confidence information if not passed
        ave_conf_df = plot_params.pop("ave_conf_df", None)
        if ave_conf_df is None:
            ave_conf_df = self.fetch1_dataframe("ave_conf_df")

        # Define alpha as maximum available if not passed
        alpha = plot_params.pop("alpha", None)
        if alpha is None:
            alpha = np.max(np.unique(ave_conf_df.alpha))

        # Define relationship values if not passed
        # Get relationship meta name
        relationship_meta_name = self._get_relationship_meta_name()
        relationship_vals = plot_params.pop("relationship_vals", None)
        if relationship_vals is None and relationship_meta_name is not None:
            relationship_vals = np.unique(ave_conf_df[relationship_meta_name])
            # Order relationship values to plot in desired order
            relationship_vals = [x for x in self.get_ordered_relationships() if x in relationship_vals]
            # Raise error if no relationship vals
            if len(relationship_vals) == 0:
                raise Exception(f"relationship_vals empty; this is unexpected")

        # Get brain region values if not passed
        # Get brain region meta name
        brain_region_meta_name = self._get_brain_region_meta_name()
        brain_region_vals = plot_params.pop("brain_region_vals", None)
        if brain_region_vals is None:
            brain_region_vals = np.unique(ave_conf_df[brain_region_meta_name])

        # Get colors df if not passed
        colors_df = plot_params.pop("colors_df", None)
        if colors_df is None:
            colors_df = self._get_colors_df(brain_region_meta_name, relationship_meta_name)

        # Get axis from original kwargs if passed (so that dont create new set of axes in copying), otherwise create
        if "ax" not in kwargs:
            ax = None
        else:
            ax = kwargs["ax"]
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 2))

        # Get x label if not passed
        xlabel = plot_params.pop("xlabel", None)
        if xlabel is None:
            xlabel = self._get_x_text()

        # Get y label if not passed
        ylabel = plot_params.pop("ylabel", None)
        if ylabel is None:
            ylabel = self._get_val_text()

        # Get title if not passed
        title = plot_params.pop("title", None)

        # Get x limits if not passed
        xlim = plot_params.pop("xlim", None)
        if xlim is None:
            xlim = self._get_x_lims()

        # Get y limits if not passed
        ylim = plot_params.pop("ylim", None)
        if ylim is None:
            ylim = self._get_val_lims()

        # Get legend if not passed
        legend = plot_params.pop("legend", None)
        if legend is None:
            legend = True

        # Get x ticks if not passed
        xticks = plot_params.pop("xticks", None)
        if xticks is None:
            xticks = self._get_xticks()

        # Get y ticks if not passed
        yticks = plot_params.pop("yticks", None)
        if yticks is None:
            yticks = self._get_val_ticks()

        # Get x tick labels if not passed
        xticklabels = plot_params.pop("xticklabels", None)
        if xticklabels is None:
            xticklabels = self._get_xticklabels(xticks)

        # Get y tick labels if not passed
        yticklabels = plot_params.pop("yticklabels", None)
        if yticklabels is None:
            yticklabels = self._get_yticklabels(yticks)

        # Get flag for whether to plot average and confidence intervals
        show_ave_conf = plot_params.pop("show_ave_conf", None)
        if show_ave_conf is None:
            show_ave_conf = True

        # Get flag for whether to plot single epoch average
        show_single_epoch = plot_params.pop("show_single_epoch", None)
        if show_single_epoch is None:
            show_single_epoch = False

        # Get font size
        fontsize = plot_params.pop("fontsize", None)
        if fontsize is None:
            fontsize = 20

        # Get tick labels fontsize
        ticklabels_fontsize = plot_params.pop("ticklabels_fontsize", None)
        if ticklabels_fontsize is None:
            ticklabels_fontsize = 18

        # Get flag for whether to remove axis on empty plot
        remove_axis_empty_plot = plot_params.pop("remove_axis_empty_plot", None)
        if remove_axis_empty_plot is None:
            remove_axis_empty_plot = True

        # Get additional quantities based on key

        # Get name of index from upstream table
        vals_index_name = self._get_vals_index_name()

        # Get metric_df if plotting single epoch data
        if show_single_epoch:
            metric_df = self.fetch1_dataframe("metric_df")

        # Track whether empty plot so can remove axis if empty
        empty_plot = True  # initialize

        # Get keys for different conditions (brain region +/- relationship)
        condition_keys = [{brain_region_meta_name: brain_region_val} for brain_region_val in brain_region_vals]
        if relationship_meta_name is not None:
            condition_keys = [
                {**x, **{relationship_meta_name: relationship_val}} for x in condition_keys
                for relationship_val in relationship_vals]

        for condition_key in condition_keys:

            # Get table entry with data for this brain region and relationship. Sort by x value
            df_subset = df_filter_columns(
                ave_conf_df, {**condition_key, **{"alpha": alpha}}).set_index(
                vals_index_name).sort_index().reset_index()

            # Continue if no entry
            if len(df_subset) == 0:
                continue

            # Otherwise indicate that plot not empty
            empty_plot = False

            # Check x values all unique
            check_all_unique(df_subset[vals_index_name].values)

            # Get color based on brain region value
            color = df_pop(colors_df, condition_key, "color")

            # Plot average and confidence bounds if indicated
            if show_ave_conf:
                # Get confidence bounds
                conf_bounds = list(zip(df_subset.lower_conf.to_numpy(), df_subset.upper_conf.to_numpy()))

                # Plot average and confidence bounds
                label = " ".join(list(condition_key.values()))
                plot_ave_conf(ax, df_subset.x_val, df_subset.ave_val, conf_bounds, color=color, label=label)

            # Plot epoch average if indicated
            if show_single_epoch:

                epoch_group_by_cols = [
                    "joint_brain_region", "brain_region", "subject_id", "nwb_file_name", "epochs_description",
                    relationship_meta_name, vals_index_name]

                epoch_group_by_cols = [x for x in epoch_group_by_cols if x in metric_df.columns]

                metric_epoch_df = metric_df.groupby(epoch_group_by_cols).mean().reset_index()

                for subject_id in np.unique(metric_epoch_df["subject_id"]):

                    metric_epoch_df_subset = df_filter_columns(metric_epoch_df, {**{
                        "subject_id": subject_id}, **condition_key})

                    for nwb_file_name in np.unique(metric_epoch_df_subset["nwb_file_name"]):

                        metric_epoch_df_subset_ = df_filter_columns(
                            metric_epoch_df_subset, {"nwb_file_name": nwb_file_name})

                        for epochs_description in np.unique(metric_epoch_df_subset_["epochs_description"]):

                            # Define line params based on contingency and environment
                            epoch = (EpochsDescription & {"epochs_description": epochs_description,
                                                          "nwb_file_name": nwb_file_name}).get_epoch()
                            ti_key = {"epoch": epoch, "nwb_file_name": nwb_file_name}
                            linestyle = (TaskIdentification & ti_key).get_line_param("linestyle")
                            line_alpha = (TaskIdentification & ti_key).get_line_param("alpha")

                            # Plot
                            ax.plot(df_filter_columns(metric_epoch_df_subset_, {
                                    "epochs_description": epochs_description}).set_index(
                                "x_val").val, linewidth=.5, linestyle=linestyle, color=color, zorder=10,
                                    alpha=line_alpha)

        # If brain region difference for one pair of brain regions, add rectangles to denote brain region colors
        if len(brain_region_vals) == 1 and brain_region_meta_name == "joint_brain_region" and not empty_plot:
            brain_region_val = unpack_single_element(brain_region_vals)
            df_subset = df_filter_columns(ave_conf_df, {brain_region_meta_name: brain_region_val})
            if len(df_subset) > 0:
                brain_regions = [check_return_single_element(df_subset[x].values).single_element for x in [
                    "brain_region_1", "brain_region_2"]]
                if ylim is None:
                    ylim = ax.get_ylim()
                if xlim is None:
                    xlim = ax.get_xlim()
                y_starts = [ylim[0], 0]
                y_ends = [0, ylim[1]]
                if self._plot_params()["reverse_brain_region_panel_order"]:
                    y_starts = y_starts[::-1]
                    y_ends = y_ends[::-1]
                for brain_region, y_start, y_end in zip(brain_regions, y_starts, y_ends):
                    color = BrainRegionColor().get_brain_region_color(brain_region)
                    ax.add_patch(patches.Rectangle(
                        [xlim[0], y_start], np.diff(xlim), y_end - y_start, facecolor=color, alpha=.3, zorder=10))

        # If plot empty, remove axis if indicated
        if empty_plot and remove_axis_empty_plot:
            ax.axis("off")

        # Format axis
        plot_params = {
            "xlabel": xlabel, "ylabel": ylabel, "title": title, "xlim": xlim, "ylim": ylim, "xticks": xticks,
            "xticklabels": xticklabels, "yticks": yticks, "yticklabels": yticklabels, "fontsize": fontsize,
            "ticklabels_fontsize": ticklabels_fontsize}
        format_ax(ax, **plot_params)

        # Legend if indicated
        if legend:
            ax.legend()

        # Extend method in optionally table specific manner
        extension_params = {"ax": ax, "empty_plot": empty_plot, "xlim": xlim}
        self.extend_plot_results(**extension_params)

    def get_default_table_entry_params(self):
        """
        Get default params for defining a table entry
        :return: dict with default table entry params
        """

        # Brain region cohort name
        brain_region_cohort_name = "all_targeted"

        # Brain region units cohort type
        min_epoch_mean_fr = .1
        unit_subset = True
        unit_subset_type = "rand_target_region"
        unit_subset_size = 50
        unit_subset_iterations = np.arange(0, 10)
        # ...Get eps_units_param_name
        eps_units_param_name = EpsUnitsParams().lookup_param_name([min_epoch_mean_fr])
        brain_region_units_cohort_type = BrainRegionUnitsCohortType().lookup_param_name([
            eps_units_param_name, int(unit_subset), unit_subset_type, unit_subset_size, unit_subset_iterations
        ])

        # Return default params
        return {"brain_region_cohort_name": brain_region_cohort_name,
                "brain_region_units_cohort_type": brain_region_units_cohort_type}

    @staticmethod
    def get_default_plot_params(brain_regions_separate=None):
        # Define default params

        if brain_regions_separate is None:
            brain_regions_separate = False

        # Subplot width
        subplot_width = 2.2
        if brain_regions_separate:
            subplot_width = .875

        # Subplot height
        subplot_height = 1.8
        # ...Smaller plot if brain regions separate
        if brain_regions_separate:
            subplot_height = 1.25

        # Gaps between plots
        # ...rows
        mega_row_gap_factor = .15
        if brain_regions_separate:
            mega_row_gap_factor = .2
        # ...columns
        mega_column_gap_factor = .04
        if brain_regions_separate:
            mega_column_gap_factor = .05
        wspace = .15
        hspace = .05

        # Fontsize
        fontsize = 15
        if brain_regions_separate:
            fontsize = 11.2

        ticklabels_fontsize = 13
        if brain_regions_separate:
            ticklabels_fontsize = 10.4

        # Figure type
        figure_type = ".svg"

        # dpi
        dpi = 300

        # Define whether to remove axis on empty plot
        remove_axis_empty_plot = False
        if brain_regions_separate:
            remove_axis_empty_plot = True

        # Return default params
        return {
            "brain_regions_separate": brain_regions_separate, "subplot_width": subplot_width,
            "subplot_height": subplot_height,
            "mega_row_gap_factor": mega_row_gap_factor, "mega_column_gap_factor": mega_column_gap_factor,
        "wspace": wspace, "hspace": hspace, "fontsize": fontsize, "ticklabels_fontsize": ticklabels_fontsize,
            "figure_type": figure_type, "dpi": dpi,
            "remove_axis_empty_plot": remove_axis_empty_plot}

    def initialize_multiplot(self, mega_row_iterables, mega_column_iterables, brain_regions_separate, **params):
        # Initialize plot. Have rats in rows and tables in columns.

        # Add default plot params
        default_params = self.get_default_plot_params(brain_regions_separate)
        # Add default params to passed params if not present
        params = add_defaults(params, default_params, add_nonexistent_keys=True)

        # Define what variables are in rows and which are in columns
        row_iterables = [0]
        # ...put brain regions in columns if indicated
        column_iterables = [0]  # default
        if brain_regions_separate:
            column_iterables = params["brain_region_vals"]

        gs_map, ax_map, fig = get_gridspec_ax_maps(
            mega_row_iterables, mega_column_iterables, row_iterables, column_iterables,
            subplot_width=params["subplot_width"], subplot_height=params["subplot_height"],
            mega_row_gap_factor=params["mega_row_gap_factor"],
            mega_column_gap_factor=params["mega_column_gap_factor"], wspace=params["wspace"],
            hspace=params["hspace"], sharey=True)
        plot_idx_map = get_plot_idx_map(mega_row_iterables, mega_column_iterables, row_iterables, column_iterables)

        return namedtuple("PlotObjs", "gs_map ax_map fig plot_idx_map")(gs_map, ax_map, fig, plot_idx_map)

    @staticmethod
    def _get_multiplot_ax_key(**kwargs):

        check_membership(
            ["brain_regions_separate", "brain_region_val", "table_name", "recording_set_name"], kwargs.keys())

        column_iterable = 0
        if kwargs["brain_regions_separate"]:
            column_iterable = kwargs["brain_region_val"]

        return (kwargs["recording_set_name"], kwargs["table_name"], 0, column_iterable)

    def _get_multiplot_fig_name(self, brain_region_vals, keys, plot_name):
        raise Exception(f"Must overwrite in child class")

    def _save_multiplot_fig(self, brain_region_vals, keys, fig, plot_name=""):
        # Save figure and parameters

        # Make file name
        fig_name = self._get_multiplot_fig_name(brain_region_vals, keys, plot_name)

        # Save figure
        figure_type = ".svg"
        dpi = 300
        save_figure(fig, f"{fig_name}", figure_type=figure_type, dpi=dpi)

        # Save params
        params_name = fig_name + f"_fig_keys"
        save_json(params_name, keys)

    def multiplot(
            self, mega_row_iterables, mega_column_iterables, brain_regions_separate, brain_region_vals, keys,
            rat_cohort, **kwargs):

        # Plot
        for key_idx, key in enumerate(keys):

            # Get flag indicating whether to populate tables
            populate_tables = key.pop("populate_tables", False)

            # Get table
            table = get_table(key["table_name"])

            # Initialize plot if first iteration
            if key_idx == 0:
                plot_objs = self.initialize_multiplot(
                    mega_row_iterables, mega_column_iterables, brain_regions_separate,
                    **{**kwargs, **{"brain_region_vals": brain_region_vals}})

            # Populate table if indicated
            if populate_tables:
                table().populate(key)

            # Get table subset
            table_key = {k: key[k] for k in table.primary_key if k in key}
            table_subset = (table & key)

            # Continue if no entry in table
            if len(table_subset) == 0:
                print(f"no entry found in {key['table_name']} for {table_key}")
                continue

            # Plot sample mean and confidence bounds
            for brain_region_val in brain_region_vals:

                # Get subplot
                ax_key = {**key, **{
                    "brain_region_val": brain_region_val, "brain_regions_separate": brain_regions_separate}}
                ax_key = table_subset._get_multiplot_ax_key(**ax_key)
                ax = plot_objs.ax_map[ax_key]
                plot_idx_obj = plot_objs.plot_idx_map[ax_key]

                # x label
                xlabel = ""
                xlabel_column_bool = plot_idx_obj.column_idx == 0
                if brain_regions_separate:
                    xlabel_column_bool = plot_idx_obj.column_idx == int(np.floor(len(brain_region_vals) / 2))
                if xlabel_column_bool and plot_idx_obj.mega_row_idx == 0:
                    xlabel = None

                # y label
                ylabel = ""
                if all([getattr(plot_idx_obj, x) == 0 for x in [
                    "mega_row_idx", "mega_column_idx", "row_idx", "column_idx"]]):
                    ylabel = None

                # y tick labels
                if "yticklabels" in kwargs:
                    yticklabels = kwargs["yticklabels"]
                else:
                    yticklabels = []
                    if all([getattr(plot_idx_obj, x) == 0 for x in [
                        "mega_row_idx", "mega_column_idx", "row_idx", "column_idx"]]):
                        yticklabels = None

                # title
                title = ""
                if plot_idx_obj.column_idx == 1 and plot_idx_obj.mega_column_idx == 1 and not rat_cohort:
                    subject_id = (RecordingSet & key).get_subject_id()  # TODO: use TrainTestSet also
                    title = get_subject_id_shorthand(subject_id)

                # Define plot params
                plot_params = {k: v for k, v in kwargs.items() if k in [
                    "show_ave_conf", "show_single_epoch", "remove_axis_empty_plot", "fontsize",
                    "ticklabels_fontsize", "subplot_width", "subplot_height", "hspace", "wspace",
                    "mega_row_gap_factor", "mega_column_gap_factor", "xticklabels"]}

                plot_params.update({
                    "ax": ax, "brain_region_vals": [brain_region_val], "legend": False,
                    "xlabel": xlabel, "ylabel": ylabel, "title": title,
                    "yticklabels": yticklabels, "ylim": None})

                for param_name in ["relationship_vals", "xticks", "yticks", "xlim", "ylim", ]:
                    if param_name in key:
                        plot_params[param_name] = key[param_name]

                default_plot_params = table_subset.get_default_plot_params(brain_regions_separate)
                plot_params = add_defaults(plot_params, default_plot_params, add_nonexistent_keys=True)

                # Plot data
                table_subset.plot_results(**plot_params)

        # Save figure if indicated
        save_fig = kwargs.pop("save_fig", False)
        plot_name = kwargs.pop("plot_type", "")
        if save_fig:
            table_subset._save_multiplot_fig(brain_region_vals, keys, plot_objs.fig, plot_name)


class PathWellFRVecSummBase(PathWellPopSummBase):

    # Override parent class method so can add params specific to fr vec tables
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        # Smoothed spikes
        res_epoch_spikes_sm_param_name = "0.1"

        # Z score
        zscore_fr = 0

        params.update(
            {"res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name, "zscore_fr": zscore_fr})

        # Return default params
        return params

    def _get_replace_char_map(self):
        replace_char_map = super()._get_replace_char_map()
        replace_char_map.update({"one_minus_on_off_diagonal_ratio": "ratio", "euclidean_distance": "euc", })
        return replace_char_map

    def _get_multiplot_fig_name(self, brain_region_vals, keys, plot_name=""):

        # Smoothing params text
        res_epoch_spikes_sm_param_name = return_shared_key_value(keys, "res_epoch_spikes_sm_param_name")
        smooth_text = f"_sm{res_epoch_spikes_sm_param_name}"

        # Z score text
        zscore_fr = return_shared_key_value(keys, "zscore_fr")
        zscore_text = format_bool(zscore_fr, "zscore_fr", prepend_underscore=True)

        # Text to denote whether showed single epochs
        show_single_epoch = return_shared_key_value(keys, "show_single_epoch")
        show_single_epoch_text = format_bool(
            show_single_epoch, "epochs", prepend_underscore=True)

        # Text to denote which brain regions are plotted
        brain_regions_text = f"_{len(brain_region_vals)}areas"

        # Define map to abbreviate parts of param names so that can meet file name character limit
        replace_char_map = self._get_replace_char_map()

        # Text to denote upstream table param names
        param_names_map = {table_name: [] for table_name in np.unique([key["table_name"] for key in keys])}
        for key in keys:
            table_name = key["table_name"]
            meta_param_name = get_table(table_name)()._upstream_table()()._get_params_table()().meta_param_name()
            param_names_map[table_name].append(key[meta_param_name])
        param_names = [
            check_return_single_element(param_names).single_element
            for table_name, param_names in param_names_map.items()]
        upstream_param_text = "_".join(param_names)
        # ...Abbreviate
        upstream_param_text = replace_chars(upstream_param_text, replace_char_map)
        # ...Append underscore
        upstream_param_text = "_" + upstream_param_text

        # Text to denote summary table param name
        param_names_map = {table_name: [] for table_name in np.unique([key["table_name"] for key in keys])}
        for key in keys:
            table_name = key["table_name"]
            meta_param_name = get_table(table_name)()._get_params_table()().meta_param_name()
            param_names_map[table_name].append(key[meta_param_name])
        param_names = [
            check_return_single_element(param_names).single_element
            for table_name, param_names in param_names_map.items()]
        summ_param_text = "_".join(param_names)
        # ...Abbreviate
        summ_param_text = replace_chars(summ_param_text, replace_char_map)
        # ...Append underscore
        summ_param_text = "_" + summ_param_text

        # Text for plot name
        if len(plot_name) > 0:
            plot_name = "_" + plot_name

        # Return name of saved figure
        return f"fr_vec{plot_name}{smooth_text}{zscore_text}{upstream_param_text}{summ_param_text}" + \
         f"{show_single_epoch_text}{brain_regions_text}"


class CovariateFRVecAveSummBase(PathWellFRVecSummBase):

    def make(self, key):

        verbose = True
        tolerate_missing = False

        # Get params
        metric_processing_name, label_name, brain_region_units_cohort_type, boot_set_name = (
                self._get_params_table() & key).fetch1(
            "metric_processing_name", "label_name", "brain_region_units_cohort_type", "boot_set_name")
        brain_region_cohort_name = key["brain_region_cohort_name"]
        zscore_fr = key["zscore_fr"]
        res_epoch_spikes_sm_param_name = key["res_epoch_spikes_sm_param_name"]
        unit_params = (BrainRegionUnitsCohortType & {
            "brain_region_units_cohort_type": brain_region_units_cohort_type}).fetch1()
        unit_subset_type = unit_params["unit_subset_type"]
        unit_subset_size = unit_params["unit_subset_size"]
        unit_subset_iterations = unit_params["unit_subset_iterations"]
        min_epoch_mean_firing_rate = (EpsUnitsParams & unit_params).fetch1("min_epoch_mean_firing_rate")

        # Get table
        table = self._upstream_table()

        # Get nwb file names and epochs_descriptions_names
        nwb_file_names, epochs_descriptions_names = (RecordingSet & key).fetch1(
            "nwb_file_names", "epochs_descriptions_names")

        # Define kwargs
        # TODO (feature): define programmatically based on upstream table
        kwargs = {k: key[k] for k in [
            "time_rel_wa_fr_vec_param_name", "time_rel_wa_dig_param_name",
            "time_rel_wa_dig_single_axis_param_name",
            "time_rel_wa_ave_fr_vec_param_name", "time_rel_wa_fr_vec_st_ave_param_name",
            "path_fr_vec_param_name", "ppt_dig_param_name", "path_ave_fr_vec_param_name",
            "path_fr_vec_st_ave_param_name"] if k in key}
        kwargs.update({"label_name": label_name})

        # Get metric values
        nwb_file_names = nwb_file_names
        if metric_processing_name == "one_minus_on_off_diagonal_ratio":
            kwargs.update({"ratio_name": metric_processing_name})
            metric_df = table().get_concat_same_diff_x_ratio_df(
                nwb_file_names, epochs_descriptions_names, brain_region_cohort_name,
                min_epoch_mean_firing_rate=min_epoch_mean_firing_rate, zscore_fr=zscore_fr,
                res_epoch_spikes_sm_param_name=res_epoch_spikes_sm_param_name,
                unit_subset_type=unit_subset_type, unit_subset_size=unit_subset_size,
                unit_subset_iterations=unit_subset_iterations, verbose=verbose,
                tolerate_missing=tolerate_missing, **kwargs)
        else:
            metric_df = table().get_concat_metric_df(
                nwb_file_names, epochs_descriptions_names, brain_region_cohort_name,
                min_epoch_mean_firing_rate=min_epoch_mean_firing_rate, zscore_fr=zscore_fr,
                res_epoch_spikes_sm_param_name=res_epoch_spikes_sm_param_name,
                unit_subset_type=unit_subset_type, unit_subset_size=unit_subset_size,
                unit_subset_iterations=unit_subset_iterations, verbose=verbose,
                tolerate_missing=tolerate_missing, **kwargs)

        # Restrict to non-nan samples
        valid_bool = np.invert(np.isnan(metric_df.val))
        metric_df = metric_df[valid_bool]

        # Hierarchical bootstrap and sample mean
        # Define bootstrap params as indicated
        boot_params = (self._get_params_table()() & key).get_boot_params()

        # Define name of value index
        vals_index_name = self._get_vals_index_name(key=key)

        # Define bootstrap parameters
        # NOTE: even though x_val is redundant with vals_index_name, we include it in ave_group_column_names
        # and ave_diff_group_column_names_ for convenience: so we can access in df when plotting

        # Params for all
        resample_quantity = "val"

        # Params specific to boot_set_name
        # get parameters table so can access types of boot set names
        params_table = self._get_params_table()

        # 1) Average values
        if boot_set_name in params_table._valid_default_boot_set_names():
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = ["nwb_file_name_epochs_description", "eps_labels", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = [vals_index_name, "x_val", "brain_region", "relationship"]
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        # 2) Average difference of values across brain regions
        elif boot_set_name in params_table._valid_brain_region_diff_boot_set_names():
            target_column_name = "brain_region"
            pairs_order = self._get_brain_region_order_for_pairs()
            metric_df, resample_levels, ave_group_column_names = self._get_boot_diff_params(
                target_column_name, metric_df, pairs_order, vals_index_name, boot_set_name)

        # 3) Average difference of stay/stay values (metric computed for pairs of stay trials) and
        # stay/leave values (metric computed for pairs of stay and leave trials)
        # 4) Average difference of stay/stay values and stay/leave values across brain regions
        elif boot_set_name in params_table._valid_stay_leave_diff_boot_set_names() + \
                params_table._valid_stay_leave_diff_brain_region_diff_boot_set_names():

            # We want to take difference of values for the same path or well, across stay/stay and stay/leave cases.
            # The function _get_boot_diff_params takes a df with values to be subtracted. Entries with pairs of
            # values to be subtracted differ at only a target column (here, the column 'relationship' which indicates
            # whether the metric is on stay/stay or stay/leave trials). All other columns should have the same values
            # across the pair.
            # To achieve this, we must remove from information about trial types (stay or leave) from eps_labels.
            # eps_labels contains information about the pair of epochs, trial types (stay or leave) and path names
            # on which the metric was computed.

            target_column_name = "relationship"
            pairs_order = self._get_stay_leave_order_for_pairs()

            # Get map from eps_labels to a version where label_1 and label_2 have stay/leave information removed
            # ...Include pairs of labels in either order ((label_1, label_2) or (label_2, label_1)) since either order
            # can be encountered.
            include_reversed_pairs = True
            # ...Get upstream table
            upstream_table = self._upstream_table()()
            # ...Initialize map
            split_eps_labels_map = dict()
            for upstream_key in (self._get_selection_table() & key).fetch1("upstream_keys"):
                filter_key = {k: v for k, v in upstream_key.items() if k in metric_df.columns}
                metric_df_subset = df_filter_columns(metric_df, filter_key)

                for eps_labels in np.unique(metric_df_subset["eps_labels"]):
                    epoch_1, epoch_2, label_1, label_2 = upstream_table._split_eps_labels(
                        eps_labels, upstream_key["nwb_file_name"], include_reversed_pairs)
                    label_1 = MazePathWell().split_stay_leave_trial_path_name(label_1)
                    label_2 = MazePathWell().split_stay_leave_trial_path_name(label_2)
                    split_eps_labels_map[eps_labels] = upstream_table._get_eps_labels(
                        epoch_1, epoch_2, label_1, label_2)

            # Remove stay/leave information from labels
            eps_labels_resample_col_name = "eps_labels_no_stay_leave_info"
            metric_df[eps_labels_resample_col_name] = [split_eps_labels_map[x] for x in metric_df["eps_labels"]]
            exclude_columns = ["eps_labels", "label_1", "label_2"]

            metric_df, resample_levels, ave_group_column_names = self._get_boot_diff_params(
                target_column_name, metric_df, pairs_order, vals_index_name, boot_set_name,
                eps_labels_resample_col_name=eps_labels_resample_col_name, exclude_columns=exclude_columns)

            # Now take brain region difference if indicated
            if boot_set_name in params_table._valid_stay_leave_diff_brain_region_diff_boot_set_names():

                target_column_name = "brain_region"
                pairs_order = self._get_brain_region_order_for_pairs()
                metric_df, resample_levels, ave_group_column_names = self._get_boot_diff_params(
                    target_column_name, metric_df, pairs_order, vals_index_name, boot_set_name,
                    eps_labels_resample_col_name, debug_mode=False)

        # 5) Average ratio of values across same path and same turn or different turn path relationships for
        # path traversals, or across same well and different well relationships for delay period
        elif boot_set_name in params_table._valid_relationship_div_boot_set_names():

            # First redefine metric_df to reflect ratio of val for same path and same turn or different turn
            # relationship
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
            exclude_columns = ["eps_labels", "label_1", "label_2"]
            # ...Get df with paired metric
            metric_df = self.get_paired_metric_df(
                metric_df, target_column_name, target_column_pairs, metric_pair_fn, exclude_columns=exclude_columns)

            # Define parameters for bootstrap
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = ["nwb_file_name_epochs_description", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = [
                vals_index_name, "x_val", "brain_region", self._get_joint_column_name(target_column_name)] + \
                                     self._get_pair_column_names(target_column_name)
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        # x) same outbound path correct, different outbound path correct

        # 6) Average difference of (same outbound path correct - different outbound path correct) across brain regions
        elif boot_set_name in \
                params_table._valid_same_different_outbound_path_correct_diff_boot_set_names() + \
                params_table._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():

            # We want to take difference of values for the same outbound path across correct/correct trials and
            # and different outbound paths across correct/correct trials.
            # The function _get_boot_diff_params takes a df with values to be subtracted. Entries with pairs of
            # values to be subtracted differ at only a target column (here, the column 'relationship' which indicates
            # whether the metric is on same outbound path correct/correct or different outbound path
            # correct/correct trials). All other columns should have the same values across the pair. Since
            # there is no direct correspondence between instances of same outbound path across correct/correct trials
            # (e.g. center_well_to_right_well on correct trials) and instances of different outbound paths across
            # correct/correct trials (e.g. center_well_to_right_well on correct trials and center_well_to_left_well
            # on correct trials), take average across instances for each case, within recording sessions.

            # First take difference across same outbound path correct/correct and different outbound path
            # correct/correct
            target_column_name = "relationship"
            pairs_order = self._get_same_different_outbound_path_correct_order_for_pairs()
            eps_labels_ave_obj = self._get_eps_labels_ave_metric_df(metric_df, resample_quantity)
            metric_df, resample_levels, ave_group_column_names = self._get_boot_diff_params(
                target_column_name, eps_labels_ave_obj.eps_labels_ave_metric_df, pairs_order, vals_index_name,
                boot_set_name, eps_labels_resample_col_name=eps_labels_ave_obj.eps_labels_resample_col_name,
                exclude_columns=eps_labels_ave_obj.exclude_columns, debug_mode=False)

            # Now take difference across brain regions if indicated
            if boot_set_name in \
                params_table._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():
                target_column_name = "brain_region"
                pairs_order = self._get_brain_region_order_for_pairs()
                eps_labels_resample_col_name = None
                metric_df, resample_levels, ave_group_column_names = self._get_boot_diff_params(
                    target_column_name, metric_df, pairs_order, vals_index_name, boot_set_name,
                    eps_labels_resample_col_name, debug_mode=False)

        # Raise error if boot set name not accounted for
        else:
            raise Exception(f"Have not written code for boot_set_name {boot_set_name}")

        # Perform bootstrap
        boot_results = hierarchical_bootstrap(
            metric_df, resample_levels, resample_quantity, ave_group_column_names,
            num_bootstrap_samples_=boot_params.num_bootstrap_samples, average_fn_=boot_params.average_fn,
            alphas=boot_params.alphas)

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
            insert1_print(self.Upstream, {**key, **upstream_key})

    def _get_eps_labels_ave_metric_df(self, metric_df, resample_quantity):
        # Average across eps_labels in metric_df. Meant to be used in cases where comparing relationships
        # whose eps_labels dont have a direct correspondence (e.g. same path vs. outbound paths)

        # Get unique settings of columns within which we want to average
        column_sets = unique_df_column_sets(metric_df, [
            x for x in metric_df.columns if x not in [
                "eps_labels", "label_1", "label_2", resample_quantity]], as_dict=True)

        # Assemble df
        data_list = []
        for column_set in column_sets:
            # Find average across eps_labels
            df_subset = df_filter_columns(metric_df, column_set)
            ave_val = np.mean(df_subset.val)
            df_entry = list(column_set.values()) + ["^".join(df_subset.eps_labels), ave_val]
            data_list.append(df_entry)
        eps_labels_resample_col_name = "eps_labels_joined"
        eps_labels_ave_metric_df = df_from_data_list(data_list, list(column_set.keys()) + [
            eps_labels_resample_col_name, resample_quantity])
        exclude_columns = [eps_labels_resample_col_name]
        return namedtuple("EpsLabelsAve", "eps_labels_ave_metric_df exclude_columns eps_labels_resample_col_name")(
            eps_labels_ave_metric_df, exclude_columns, eps_labels_resample_col_name)

    def _get_boot_diff_params(
            self, target_column_name, metric_df, pairs_order, vals_index_name, boot_set_name,
            eps_labels_resample_col_name="eps_labels", exclude_columns=None, debug_mode=False):
        """
        Get bootstrap parameters in cases where taking difference between values
        :param target_column_name: str, column in metric_df that takes on different values for the entries with which
        we find difference in values
        :param metric_df: pandas df. Subtract values for pairs of entries for which all columns in metric_df take on
        the same value, except for the column given by target_column_name, which takes on different values across
        pairs of entries
        :param pairs_order: desired order of values in target_column_name for subtraction
        (e.g. mPFC_targeted - OFC_targeted for the column brain_region)
        :param vals_index_name: str
        :param boot_set_name: str, name of bootstrap parameter set
        :param eps_labels_resample_col_name: str, name of eps_labels column for resampling
        :param exclude_columns: list, names of columns to exclude when getting paired metric df (values at these
        columns are allowed to differ across members of pair)
        :return: df with metric
        :return resample_levels: resample_levels_ in function hierarchical_bootstrap in hierarchical_bootstrap.py
        :return ave_group_column_names: ave_group_column_names_ in function hierarchical_bootstrap in
        hierarchical_bootstrap.py
        """

        # Define inputs if not passed
        if exclude_columns is None:
            exclude_columns = []

        # Define parameters for bootstrap
        # ...Define columns at which to resample during bootstrap, in order
        eps_labels_list = []
        if eps_labels_resample_col_name not in exclude_columns and eps_labels_resample_col_name is not None:
            eps_labels_list = [eps_labels_resample_col_name]
        resample_levels = [
            "nwb_file_name_epochs_description"] + eps_labels_list + ["brain_region_units_param_name"]
        # ...Define columns whose values to keep constant (no resampling across these)
        col_name = unpack_single_element(
            [x for x in ["relationship", "joint_relationship", "brain_region"] if x != target_column_name and
             x in metric_df.columns])
        ave_group_column_names = [
            vals_index_name, "x_val", self._get_joint_column_name(target_column_name), f"{target_column_name}_1",
            f"{target_column_name}_2"] + [col_name]
        # ...Alter params based on whether rat cohort
        resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
            boot_set_name, resample_levels, ave_group_column_names)

        # Redefine metric_df to reflect difference between val for pair of conditions
        # ...Define pairs of variable
        target_column_pairs = get_val_pairs(np.unique(metric_df[target_column_name]), pairs_order)
        # ...Define function for computing metric on brain region pairs
        metric_pair_fn = self.metric_pair_diff
        # ...Get df with paired metric
        metric_df = self.get_paired_metric_df(metric_df, target_column_name, target_column_pairs, metric_pair_fn,
                                              exclude_columns=exclude_columns)

        if debug_mode and len(metric_df) == 0:
            raise Exception

        # Return parameters
        return metric_df, resample_levels, ave_group_column_names

    def cleanup(self):
        # Delete orphaned part table entries (not sure why orphaned entries are able to be made)

        bad_keys = [key for key in self._get_selection_table().fetch("KEY")
            if len(self & key) == 0 and len(self.Upstream & key) > 1]
        print(f"Found {len(bad_keys)} bad keys. Total number of keys in main table: {len(self)}")
        for key in bad_keys:
            (self.Upstream & key).delete(force=True)  # note: part table delete method seems to not take safemode arg

    def get_meta_param_name(self):
        return self._get_params_table()().meta_param_name()

    def _get_relationship_meta_name(self):

        boot_set_name = self.get_upstream_param("boot_set_name")
        params_table = self._get_params_table()
        if boot_set_name in params_table._valid_relationship_div_boot_set_names() + \
            params_table._valid_stay_leave_diff_boot_set_names() + \
            params_table._valid_stay_leave_diff_brain_region_diff_boot_set_names() + \
            params_table._valid_same_different_outbound_path_correct_diff_boot_set_names() + \
            params_table._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():
            return self._get_joint_column_name("relationship")

        return "relationship"

    def _get_val_text(self):
        # Get name of value as friendly text
        # TODO: code up labels for cosine similarity, and ideally for single trial vs. trial ave cases

        # Filter for entries in upstream table using key for current table, so can use to pull quantities for text
        upstream_table_subset = self._upstream_table()() & self.fetch1("KEY")

        # Set y label to name of metric
        metric_name = upstream_table_subset._get_metric_name()
        val_text = f"{format_metric_name(metric_name)}"

        # If not difference vector, add text to indicate
        vector_type = upstream_table_subset._get_vector_type()
        if vector_type == "vec":
            val_text += f"\n{vector_type.replace('_', ' ')}"

        # If one minus on / off diagonal metric, add text to indicate
        metric_processing_name = self.get_upstream_param("metric_processing_name")
        if metric_processing_name == "one_minus_on_off_diagonal_ratio":
            val_text = "Proximity"

        # If relationship ratio or brain region difference, add text to indicate
        boot_set_name = self.get_upstream_param("boot_set_name")
        params_table = self._get_params_table()
        if boot_set_name in params_table._valid_relationship_div_boot_set_names():
            val_text += f"\nratio"
        elif boot_set_name in params_table._valid_brain_region_diff_boot_set_names():
            val_text += f"\nbrain region diff"
        elif boot_set_name in params_table._valid_brain_region_diff_boot_set_names():
            val_text += f"\nbrain region diff"
        elif boot_set_name in params_table._valid_stay_leave_diff_boot_set_names():
            val_text += f"\nstay/stay - stay/leave"
        elif boot_set_name in params_table._valid_stay_leave_diff_brain_region_diff_boot_set_names():
            val_text += f"\nstay/stay - stay/leave\nbrain region diff"
        elif boot_set_name in params_table._valid_same_different_outbound_path_correct_diff_boot_set_names():
            val_text += f"\nsame - diff outbound"
        elif boot_set_name in params_table._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():
            val_text += f"\nsame - diff outbound\nbrain region diff"

        # Return text describing value
        return val_text

    def _get_default_plot_cov_fr_vec_param_name(self):
        raise Exception(f"Must be overwritten in child class")

    def _get_param_names_obj(self, **kwargs):

        # summary table
        params_table = self._get_params_table()()
        param_name = params_table.lookup_param_name(
            kwargs, args_as_dict=True, tolerate_irrelevant_args=True)
        meta_param_name = params_table.meta_param_name()

        # upstream table
        upstream_table = self._upstream_table()()
        upstream_params_table = upstream_table._get_params_table()()
        upstream_param_name = upstream_params_table.lookup_param_name(
            kwargs, args_as_dict=True, tolerate_irrelevant_args=True)
        upstream_meta_param_name = upstream_params_table.meta_param_name()

        # covariate fr vec table (upstream of upstream table)
        cov_fr_vec_table = upstream_table._fr_vec_table()()
        cov_fr_vec_meta_param_name = cov_fr_vec_table._get_params_table()().meta_param_name()
        Params = namedtuple(
            "Params", "meta_param_name param_name upstream_meta_param_name upstream_param_name "
                      "cov_fr_vec_meta_param_name cov_fr_vec_param_name")
        cov_fr_vec_param_name = self._get_default_plot_cov_fr_vec_param_name()

        return Params(meta_param_name, param_name, upstream_meta_param_name, upstream_param_name,
                      cov_fr_vec_meta_param_name, cov_fr_vec_param_name)

    @staticmethod
    def _get_multiplot_params(**kwargs):
        # Define params for plotting multiple table entries. One param set per table entry.

        # Check that loop inputs passed
        check_membership(["table_names", "label_names", "relationship_vals_list", "recording_set_names"], kwargs)

        # Make copy of kwargs to serve as base of each key
        kwargs = copy.deepcopy(kwargs)

        # Remove iterables so that not in individual keys
        table_names = kwargs.pop("table_names")
        label_names = kwargs.pop("label_names")
        relationship_vals_list = kwargs.pop("relationship_vals_list")
        recording_set_names = kwargs.pop("recording_set_names")

        param_sets = []
        # Loop through table names (and corresponding label names and relationship vals)
        for table_name, label_name, relationship_vals in zip(
                table_names, label_names, relationship_vals_list):

            # Make copy of kwargs so that updates dont carry over from one for loop iteration to the next
            key = copy.deepcopy(kwargs)

            # Add table_name, label name and relationship vals to key
            key.update(
                {"table_name": table_name, "label_name": label_name, "relationship_vals": relationship_vals})

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
                obj.cov_fr_vec_meta_param_name: obj.cov_fr_vec_param_name, obj.meta_param_name: obj.param_name,
                obj.upstream_meta_param_name: obj.upstream_param_name}
            key = add_defaults(key, default_cov_params, add_nonexistent_keys=True)

            # Loop through recording set names
            for recording_set_name in recording_set_names:
                key.update({"recording_set_name": recording_set_name})

                # Append param set to list
                param_sets.append(copy.deepcopy(key))

        # Return param sets
        return param_sets


class CovariateAveFRVecSummBase(CovariateFRVecAveSummBase):

    def _get_val_lims(self):

        # Get values of parameters that influence metric values

        upstream_table_subset = self._upstream_table()() & self.fetch1("KEY")
        metric_name = upstream_table_subset._get_metric_name()
        vector_type = upstream_table_subset._get_vector_type()
        metric_processing_name = self.get_upstream_param("metric_processing_name")
        boot_set_name = self.get_upstream_param("boot_set_name")

        # Get metric value limits using these parameters

        # Cosine similarity
        if metric_name == "cosine_similarity":
            # Relationship ratio
            if boot_set_name in self._get_params_table()._valid_relationship_div_boot_set_names():
                return [-.3, 1.2]
            # Brain region difference
            elif boot_set_name in self._get_params_table()._valid_brain_region_diff_boot_set_names():
                return [-.5, .5]
            else:
                if vector_type == "diff_vec":
                    return [-.2, 1]

        # Euclidean distance
        if metric_name == "euclidean_distance":
            if metric_processing_name == "one_minus_on_off_diagonal_ratio":
                # Relationship ratio
                if boot_set_name in self._get_params_table()._valid_relationship_div_boot_set_names():
                    if vector_type == "vec":
                        return [-.3, 1.2]
                    elif vector_type == "diff_vec":
                        return [-.6, .6]
                # Brain region difference # TODO: code

                else:
                    return [-.2, 1]
            else:
                return None


class CovariateFRVecSTAveSummBase(CovariateFRVecAveSummBase):

    def _get_val_lims(self):
        # Get values of parameters that influence metric values

        upstream_table_subset = self._upstream_table()() & self.fetch1("KEY")
        metric_name = upstream_table_subset._get_metric_name()
        vector_type = upstream_table_subset._get_vector_type()
        metric_processing_name = self.get_upstream_param("metric_processing_name")
        boot_set_name = self.get_upstream_param("boot_set_name")

        # Get metric value limits using these parameters

        if metric_name == "cosine_similarity":
            params_table = self._get_params_table()
            if boot_set_name in params_table._valid_brain_region_diff_boot_set_names():
                if vector_type == "vec":
                    return [-.5, .5]
                elif vector_type == "diff_vec":
                    return [-.5, .5]
            elif boot_set_name in params_table._valid_stay_leave_diff_boot_set_names():
                if vector_type == "diff_vec":
                    return [-.1, .75]
            else:
                if vector_type == "vec":
                    return [.5, 1]
                elif vector_type == "diff_vec":
                    return [-.2, 1]
        if metric_name == "euclidean_distance":
            if metric_processing_name == "one_minus_on_off_diagonal_ratio":
                if "brain_region_diff" in boot_set_name:
                    return [-.4, .4]
                else:
                    return [-.1, 1]
            else:
                return None

    def _get_val_ticks(self):

        upstream_table_subset = self._upstream_table()() & self.fetch1("KEY")
        metric_name = upstream_table_subset._get_metric_name()
        metric_processing_name = self.get_upstream_param("metric_processing_name")

        if metric_name == "cosine_similarity":
            return [-.5, 0, .5]
        if metric_name == "euclidean_distance" and metric_processing_name == "one_minus_on_off_diagonal_ratio":
            return [-.5, 0, .5]


class PathFRVecSummBase(CovariateFRVecAveSummBase):

    def get_ordered_relationships(self):

        return [
            "same_path", "same_turn", "different_turn_well",
            "same_path_even_odd_trials", "same_turn_even_odd_trials", "different_turn_well_even_odd_trials",
            "same_turn_even_odd_trials_same_path_even_odd_trials",
            "different_turn_well_even_odd_trials_same_path_even_odd_trials",
            "same_path_correct_correct_trials", "same_path_correct_incorrect_trials",
            "same_path_incorrect_incorrect_trials", "outbound_correct_correct_trials",
            "outbound_correct_incorrect_trials", "outbound_incorrect_incorrect_trials",
            "outbound_correct_correct_trials_same_path_outbound_correct_correct_trials",
        ]

    def _get_x_lims(self):
        return [0, 1]

    def _get_xticks(self):
        return self._get_x_lims()

    def _get_x_text(self):
        # Get name of x value as friendly text
        return "Path fraction"

    def extend_plot_results(self, **kwargs):

        super().extend_plot_results(**kwargs)

        # Vertical lines to denote track segments
        if not kwargs["empty_plot"]:
            ax = kwargs["ax"]
            plot_junction_fractions(ax)

    # Override parent class method so can add params specific to path fr vec tables
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_fr_vec_default_param
        params.update({"ppt_dig_param_name": get_fr_vec_default_param("ppt_dig_param_name")})

        # Return default params
        return params

class TimeRelWAFRVecSummBase(CovariateFRVecAveSummBase):

    def get_ordered_relationships(self):
        return PathFRVecSummBase().get_ordered_relationships() + [
            "same_end_well_even_odd_trials", "different_end_well_even_odd_trials",
            "same_end_well_even_odd_stay_trials", "different_end_well_even_odd_stay_trials",
            "different_end_well_even_odd_stay_trials_same_end_well_even_odd_stay_trials",
            "same_path_stay_stay_trials", "same_path_stay_leave_trials", "same_path_leave_leave_trials",
            "same_path_stay_leave_trials_same_path_stay_stay_trials",
            "outbound_correct_correct_trials_same_path_outbound_correct_correct_trials"
        ]

    def _get_x_lims(self):
        return (TimeRelWADigSingleAxisParams & self.fetch1("KEY")).fetch1("rel_time_start", "rel_time_end")

    def _get_x_text(self):
        # Get name of x value as friendly text

        return "Time in delay (s)"

    def extend_plot_results(self, **kwargs):

        super().extend_plot_results(**kwargs)

        # Vertical lines to denote zero, if x lower limit before zero
        if not kwargs["empty_plot"] and kwargs["xlim"][0] < 0:
            ax = kwargs["ax"]
            plot_spanning_line(ax.get_ylim(), 0, ax, "y", color="brown")

    # Override parent class method so can add params specific to fr vec tables
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_fr_vec_default_param
        params.update({x: get_fr_vec_default_param(x) for x in [
            "time_rel_wa_dig_param_name", "time_rel_wa_dig_single_axis_param_name"
        ]})

        # Return default params
        return params


def format_metric_name(metric_name):
    metric_name_map = {"cosine_similarity": "Cosine similarity", "euclidean_distance": "Euclidean distance"}
    return metric_name_map[metric_name]


class PopulationAnalysisParamsBase(SecKeyParamsBase):

    @staticmethod
    def _default_brain_region_units_cohort_types():

        eps_units_param_name = EpsUnitsParams().lookup_param_name([.1])
        param_sets = [[eps_units_param_name, 0, "target_region", None, [None]],
            [eps_units_param_name, 1, "rand_target_region", 50, [0]],
            [eps_units_param_name, 1, "rand_target_region", 50, np.arange(0, 10)]]

        return [
            BrainRegionUnitsCohortType().lookup_param_name(x) for x in param_sets]

    @staticmethod
    def _valid_default_boot_set_names():
        # All valid default boot set names
        return ["default", "default_rat_cohort"]

    @staticmethod
    def _valid_relationship_div_boot_set_names():
        # All valid relationship ratio boot set names
        return ["relationship_div_rat_cohort", "relationship_div", "relationship_div_rat_cohort_median",
                "relationship_div_median"]

    @staticmethod
    def _valid_brain_region_diff_boot_set_names():
        # All valid brain region difference boot set names
        return ["brain_region_diff", "brain_region_diff_rat_cohort"]

    @staticmethod
    def _valid_stay_leave_diff_boot_set_names():
        # All valid stay/leave trial difference boot set names
        return ["stay_leave_diff", "stay_leave_diff_rat_cohort"]

    @staticmethod
    def _valid_stay_leave_diff_brain_region_diff_boot_set_names():
        # All valid stay/leave trial difference boot set names
        return ["stay_leave_diff_brain_region_diff", "stay_leave_diff_brain_region_diff_rat_cohort"]

    @staticmethod
    def _valid_same_different_outbound_path_correct_diff_boot_set_names():
        # All valid same path correct correct vs. outbound correct correct trial boot set names
        param_name = "same_different_outbound_path_correct_diff"
        return [param_name, f"{param_name}_rat_cohort"]

    @staticmethod
    def _valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():
        # All valid same path correct correct vs. outbound correct correct trial brain region difference boot set names
        param_name = "same_different_outbound_path_correct_diff_brain_region_diff"
        return [param_name, f"{param_name}_rat_cohort"]

    def _boot_set_names(self):
        # Boot set names to populate table with. Use different set depending on child class.
        return self._valid_default_boot_set_names()

    def get_boot_params(self):
        boot_set_name = self.fetch1("boot_set_name")
        return get_boot_params(boot_set_name)


class CovariateFRVecAveSummParamsBase(PopulationAnalysisParamsBase):

    @staticmethod
    def _default_label_names():
        return ["path"]

    def _default_processing_params(self):
        default_processing_params = []
        for bootstrap_param_name in self._boot_set_names():
            for label_name in self._default_label_names():
                default_processing_params += [
                    ["none", label_name, bootstrap_param_name],
                    ["one_minus_on_off_diagonal_ratio", label_name, bootstrap_param_name]]
        return default_processing_params

    def _default_params(self):
        # Combine each default processing param with each default BrainRegionUnitsCohortType param
        return [x + [y] for x in self._default_processing_params() for y in
                self._default_brain_region_units_cohort_types()]


class PopulationAnalysisSelBase(SelBase):

    def _recording_set_name_types(self):
        return ["Haight_rotation", "Haight_rotation_rat_cohort"]

    def get_recording_set_names(self, key_filter):
        # Return list of recording set names for a given key_filter, for use in populating tables
        return RecordingSet().get_recording_set_names(key_filter, self._recording_set_name_types())

    def _default_noncohort_boot_set_names(self):
        return ["default"]

    def _default_cohort_boot_set_names(self):
        return ["default_rat_cohort"]

    def _upstream_epochs_column_meta_name(self):
        return unpack_single_element([x for x in self._get_main_table()()._upstream_table()().primary_key if x in [
            "epochs_description", "epochs_id"]])

    def _get_param_name_map(self, key_filter, brain_region_units_cohort_types):
        raise Exception(f"Must be defined in child class")

    def _get_param_name_map_key(self, key, brain_region_units_cohort_type):
        # Make key to param name map given a set of parameters
        raise Exception(f"Must be defined in child class")

    # TODO (feature): limit search more fully with key_filter
    def _get_potential_keys(self, key_filter=None, verbose=True, debug_text=False):

        # Define key filter if not passed
        if key_filter is None:
            key_filter = dict()

        # Define curation set (specifies curation names across brain regions) and brain region
        # cohort name (specifies brain regions to combine across)
        curation_set_name = "runs_analysis_v1"
        brain_region_cohort_name = "all_targeted"

        # Get parameters table, filtering using passed key
        params_table = (self._get_params_table() & key_filter)

        # Define brain_region_units_cohort_types
        brain_region_units_cohort_types = params_table._default_brain_region_units_cohort_types()
        # ...apply passed filter
        if "brain_region_units_cohort_type" in key_filter:
            brain_region_units_cohort_types = [
                x for x in brain_region_units_cohort_types if x == key_filter["brain_region_units_cohort_type"]]

        # Define recording sets to loop through
        recording_set_names = self.get_recording_set_names(key_filter)

        # Get unique settings of params that we are not combining across
        upstream_table = self._get_main_table()()._upstream_table()()
        column_names = [x for x in upstream_table.primary_key if x not in [
            "nwb_file_name", self._upstream_epochs_column_meta_name(), "brain_region", "brain_region_units_param_name",
            "curation_name"]]
        upstream_table_subset = upstream_table & key_filter  # filter using passed key
        param_sets = unique_table_column_sets(upstream_table_subset, column_names, as_dict=True)

        # Get name of primary key that describes parameters
        meta_param_name = params_table.meta_param_name()

        # Get map from specific parameters to compatible summary table param names
        param_name_map = self._get_param_name_map(key_filter, brain_region_units_cohort_types)

        keys = []
        for brain_region_units_cohort_type in brain_region_units_cohort_types:

            units_params = (BrainRegionUnitsCohortType & {
                "brain_region_units_cohort_type": brain_region_units_cohort_type}).fetch1()
            min_epoch_mean_firing_rate = (EpsUnitsParams & units_params).fetch1("min_epoch_mean_firing_rate")

            if verbose:
                print(f"\nOn brain_region_units_cohort_type {brain_region_units_cohort_type}...")

            # Loop through recording set names
            for recording_set_name in recording_set_names:

                if verbose:
                    print(f"\nOn recording_set_name {recording_set_name}...")

                # Get nwb file names and epochs descriptions for this recording set
                nwb_file_names, epochs_descriptions_names = (
                        RecordingSet & {"recording_set_name": recording_set_name}).fetch1(
                    "nwb_file_names", "epochs_descriptions_names")

                # Loop through unique param sets and add key if entry exists in upstream table for each
                # epochs_description and brain_region / brain_region_units_param_name / curation_name
                for param_set in param_sets:

                    if verbose:
                        print(f"\nOn param set {param_set}...")

                    # Check if all necessary files from upstream exist to be able to populate summary table
                    # for current condition
                    populated = True
                    upstream_keys = []
                    for nwb_file_name, epochs_descriptions_name in zip(nwb_file_names, epochs_descriptions_names):

                        if debug_text:
                            print(nwb_file_name, epochs_descriptions_name)

                        # Get epochs descriptions
                        epochs_descriptions = (EpochsDescriptions & {
                            "nwb_file_name": nwb_file_name, "epochs_descriptions_name": epochs_descriptions_name}
                                               ).fetch1("epochs_descriptions")

                        # Get curation names
                        curation_names_df = (CurationSet & {
                            "nwb_file_name": nwb_file_name,
                            "brain_region_cohort_name": brain_region_cohort_name,
                            "curation_set_name": curation_set_name}).fetch1_dataframe()

                        # Get brain region units param names
                        brain_region_units_param_names_map = dict()
                        for epochs_description in epochs_descriptions:
                            brain_region_units_param_names_map[epochs_description] = [
                                BrainRegionUnitsParams().lookup_single_epoch_param_name(
                                    nwb_file_name, (EpochsDescription & {
                                    "nwb_file_name": nwb_file_name,
                                    "epochs_description": epochs_description}).get_epoch(),
                            min_epoch_mean_firing_rate, units_params["unit_subset_type"],
                            units_params["unit_subset_size"], unit_subset_iteration)
                            for unit_subset_iteration in units_params["unit_subset_iterations"]]

                        for epochs_description in epochs_descriptions:

                            if debug_text:
                                print(epochs_description)

                            curation_names_df_subset = df_filter_columns(
                                curation_names_df, {"epochs_description": epochs_description})

                            for brain_region, curation_name in zip_df_columns(
                                    curation_names_df_subset, ["brain_region", "curation_name"]):

                                if debug_text:
                                    print(brain_region, curation_name)

                                curation_name = df_pop(curation_names_df_subset, {
                                        "nwb_file_name": nwb_file_name, "brain_region": brain_region}, "curation_name")

                                brain_region_units_param_names = brain_region_units_param_names_map[epochs_description]

                                for brain_region_units_param_name in brain_region_units_param_names:

                                    # Define key to upstream table
                                    upstream_key = {**{
                                     "nwb_file_name": nwb_file_name, "brain_region": brain_region,
                                        "brain_region_units_param_name": brain_region_units_param_name,
                                        "curation_name": curation_name}, **param_set}

                                    # Add epochs information to key, depending on type of epochs information
                                    # Add epochs_id if indicated
                                    if self._upstream_epochs_column_meta_name() == "epochs_id":
                                        upstream_key["epochs_id"] = (EpochsDescription & {
                                            "nwb_file_name": nwb_file_name,
                                            "epochs_description": epochs_description}).fetch1("epochs_id")
                                    else:
                                        upstream_key["epochs_description"] = epochs_description

                                    # Check keys are same as primary key of table
                                    check_set_equality(upstream_table.primary_key, upstream_key.keys())

                                    # If could not populate brain region units table, continue
                                    if len(BrainRegionUnitsFail & upstream_key) > 0:
                                        continue

                                    # Indicate whether table populated for current entry
                                    populated_ = len(upstream_table & upstream_key) > 0

                                    # Print whether entry in table if indicated
                                    if debug_text:
                                        print(f"{populated_} {upstream_table.table_name} {upstream_key}")
                                        # Raise error so can investigate no entry
                                        if populated_ is False:
                                            raise Exception

                                    # Update flag for whether all entries in table
                                    populated *= populated_

                                    # Add upstream key to list
                                    upstream_keys.append(upstream_key)

                    # Raise error if no upstream keys
                    if len(upstream_keys) == 0:
                        raise Exception(f"upstream_keys is empty; this is not expected")

                    # Get part of key corresponding to main table (across epochs, brain regions, etc.)
                    if populated:
                        key = check_return_single_dict([
                            {k: x[k] for k in self.primary_key if k in x} for x in upstream_keys])
                        key.update({
                            "recording_set_name": recording_set_name, "brain_region_cohort_name":
                                brain_region_cohort_name, "curation_set_name": curation_set_name,
                            "upstream_keys": upstream_keys})

                        if verbose:
                            print(f"Upstream table populated for current key...")

                        # Add in summary table param name

                        # Get key to param name map
                        param_name_map_key = self._get_param_name_map_key(key, brain_region_units_cohort_type)

                        # If key in param name map, add to table keys
                        if param_name_map_key in param_name_map:
                            for summ_param_name in param_name_map[param_name_map_key]:
                                if verbose:
                                    print(f"Adding keys...")
                                # Add summary table param name to key
                                key.update({meta_param_name: summ_param_name})
                                # Add key to keys
                                keys.append(copy.deepcopy(key))
        return keys


class CovariateFRVecAveSummSelBase(PopulationAnalysisSelBase):

    def _get_param_name_map(self, key_filter, brain_region_units_cohort_types):

        # Define summary table param name based on cov_fr_vec_param_name, distance metric, and recording set name.
        # To do this, define map from cov_fr_vec_param_name, distance metric, recording set name,
        # and brain_region_units_cohort_type to summary table param names
        cov_fr_vec_param_names = self._default_cov_fr_vec_param_names()

        params_table = self._get_params_table()()

        recording_set_names_boot_set_names = [

         # Rat cohort
         (RecordingSet().lookup_rat_cohort_set_name(), boot_set_name)
         for boot_set_name in self._default_cohort_boot_set_names()] + [

         # Non rat cohort
         (recording_set_name, boot_set_name) for recording_set_name in
         RecordingSet().get_recording_set_names(
             key_filter, ["Haight_rotation", "first_day_learning_single_epoch"])
         for boot_set_name in self._default_noncohort_boot_set_names()]

        metric_names_metric_processing_names = [
            ("cosine_similarity", "none"), ("euclidean_distance", "one_minus_on_off_diagonal_ratio")]

        param_name_map = dict()
        for recording_set_name, boot_set_name in recording_set_names_boot_set_names:
            for metric_name, metric_processing_name in metric_names_metric_processing_names:
                # TODO: avoid pairing the following:
                # 1) 'path_fr_vec_param_name': 'none',
                # 2) 'path_fr_vec_st_ave_summ_param_name': 'one_minus_on_off_diagonal_ratio^path^same_different_outbound_path_correct_diff_brain_region_diff^0.1_1_rand_target_region_50_iterations_0to9'
                for cov_fr_vec_param_name in cov_fr_vec_param_names:
                    for brain_region_units_cohort_type in brain_region_units_cohort_types:
                        param_name_map_key = (
                            cov_fr_vec_param_name, metric_name, recording_set_name, brain_region_units_cohort_type)
                        if param_name_map_key not in param_name_map:
                            param_name_map[param_name_map_key] = []
                        param_name_map[param_name_map_key] += [params_table.lookup_param_name(
                            [metric_processing_name, label_name, boot_set_name, brain_region_units_cohort_type])
                            for label_name in params_table._default_label_names()]

        return param_name_map

    @staticmethod
    def _format_param_name_map_key(
            cov_fr_vec_param_name, metric_name, recording_set_name, brain_region_units_cohort_type):
        return (cov_fr_vec_param_name, metric_name, recording_set_name, brain_region_units_cohort_type)

    def _get_param_name_map_key(self, key, brain_region_units_cohort_type):
        # Make key to param name map given a set of parameters

        upstream_table = self._get_main_table()()._upstream_table()()
        upstream_params_table = upstream_table._get_params_table()()
        metric_name = (upstream_params_table & key).fetch1("metric_name")
        cov_fr_vec_meta_param_name = \
            upstream_table._fr_vec_table()()._get_params_table()().meta_param_name()

        return self._format_param_name_map_key(
            key[cov_fr_vec_meta_param_name], metric_name, key["recording_set_name"],
            brain_region_units_cohort_type)

    def _default_cov_fr_vec_param_names(self):
        raise Exception(f"Must overwrite in child class")

    @staticmethod
    def _get_matching_potential_key(key, potential_keys):
        # Find the dictionary in passed 'potential_keys' that matches passed 'key' at all shared keys (dictionaries
        # in 'potential_keys' have more keys than 'key')

        return unpack_single_element([k for k in potential_keys if all(
            [key[k_] == v for k_, v in k.items() if k_ in key])])

    @staticmethod
    def _check_dict_lists_same_elements(dict_list_1, dict_list_2, debug_mode=False):
        # Check two lists of dictionaries have the same dictionaries

        passed_check = np.logical_and(all([x in dict_list_1 for x in dict_list_2]),
                              all([x in dict_list_2 for x in dict_list_1]))

        # Print out if in debug mode and check not passed
        if not passed_check and debug_mode:
            print([x for x in dict_list_2 if x not in dict_list_1])
            print([x for x in dict_list_1 if x not in dict_list_2])
            print(f"\n")

        return passed_check

    # Override parent class method so can check both primary keys, as well as upstream keys
    def _get_bad_keys(self):

        # Get key names in potential_keys (should be same for every entry of potential_keys and we check this
        # is the case, then extract one set)
        potential_keys = self._get_potential_keys()  # gets primary and secondary columns in table
        key_names = unpack_single_vector(np.asarray([np.asarray(list(x.keys())) for x in potential_keys]))
        key_names = [k for k in key_names if k in self.primary_key]

        # Get corresponding values to key names as array so can more quickly check for matches across
        # potential and actual keys
        potential_keys_arr = np.asarray([[potential_key[k] for k in key_names] for potential_key in potential_keys])

        # Get all table entries, restricting to subset of columns present in potential_keys entries
        table_keys = self.fetch("KEY")
        table_keys_arr = np.asarray([[key[k] for k in key_names] for key in table_keys])

        # Find keys in table that are not in potential_keys
        bad_keys = [key for key_list, key in zip(table_keys_arr, table_keys) if not any(
            np.prod(key_list == potential_keys_arr, axis=1))]

        # Add on keys that are in table and potential_keys, but have upstream_keys that dont match those in
        # potential keys
        candidate_good_keys = [key for key_list, key in zip(table_keys_arr, table_keys) if any(
            np.prod(key_list == potential_keys_arr, axis=1))]
        bad_keys += [key for key in candidate_good_keys if not self._check_dict_lists_same_elements(
            self._get_matching_potential_key(
                 key, potential_keys)["upstream_keys"], (self & key).fetch1("upstream_keys"))]

        # Return bad keys
        return bad_keys

    def delete_(self, key, safemode=True):
        # If recording set name not in key but components that determine it are, then
        # find matching recording set names given the components, to avoid deleting irrelevant
        # entries

        key = copy.deepcopy(key)  # make copy of key to avoid changing outside function
        recording_set_names = RecordingSet().get_matching_recording_set_names(key)
        for recording_set_name in recording_set_names:
            key.update({"recording_set_name": recording_set_name})
            delete_(self, [self._get_main_table()], key, safemode)
