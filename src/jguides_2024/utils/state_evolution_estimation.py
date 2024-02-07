import copy
from collections import namedtuple

import numpy as np
import pandas as pd

from src.jguides_2024.utils.array_helpers import array_to_tuple_list, array_to_list
from src.jguides_2024.utils.check_well_defined import check_shape
from src.jguides_2024.utils.df_helpers import zip_df_columns, df_filter_columns, df_from_data_list, \
    check_same_index
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements, check_return_single_element, \
    return_n_empty_lists
from src.jguides_2024.utils.plot_helpers import plot_heatmap
from src.jguides_2024.utils.vector_helpers import repeat_elements_idxs, check_all_unique, \
    find_spans_increasing_list, vector_midpoints


class AverageVectorDuringLabeledProgression:

    def __init__(self, x, labels, df, new_bout_thresh=None):
        self.x = x
        self.labels = labels
        self.df = df
        self.new_bout_thresh = new_bout_thresh
        self._get_inputs()
        self._check_inputs()
        self.vector_df = self._get_vector_df()
        self.diff_vector_df = self._get_diff_vector_df()
        self.ave_vector_df = self._get_ave_vector_df()
        self.ave_diff_vector_df = self._get_ave_diff_vec_df()

    # x: series with discrete variable as values
    # labels: series with labels for x
    # df: df with vectors in columns (example shape: (time, cells))
    # new_bout_thresh: corresponds to index of x. If difference between consecutive x indices exceeds this threshold,
    # end current bout and begin new bout

    def _get_inputs(self):
        # If new_bout_thresh not passed, define as maximum difference between consecutive samples. This is equivalent
        # to having no threshold, as it will never be exceeded by the data
        if self.new_bout_thresh is None:
            self.new_bout_thresh = np.max(np.diff(self.x))

    def _check_inputs(self):
        # Check x and labels one dimensional
        for v, v_name in zip([self.x, self.labels], ["x", "labels"]):
            check_shape(v, 1, v_name)
        # Check variables share index
        check_same_index([self.x, self.labels, self.df])

    # The next two methods are for returning metadata for external use
    @staticmethod
    def _df_metadata_map():
        DfMetadata = namedtuple("DfMetadata", "vector_col_name x_name")
        return {"vector_df": DfMetadata("vector", "x"),
                "diff_vector_df": DfMetadata("diff_vector", "x_pair_int"),
                "ave_vector_df": DfMetadata("ave_vector", "x"),
                "ave_diff_vector_df": DfMetadata("ave_diff_vector", "x_pair_int")}

    @classmethod
    def get_df_metadata(cls, df_name):
        return cls._df_metadata_map()[df_name]

    def _get_vector_df(self):
        # Average vectors in contiguous stretches of constant x (i.e. bouts) (e.g. average neural firing rate in
        # contiguous stretches of animal at position_and_maze bin z), as well as corresponding label (options: one of the
        # elements in label, or "mixed" to denote a mixture of these)

        def _bout_label(labels):
            # For a bout, if all labels the same, return. Otherwise return "mixed".
            unique_labels = set(labels)
            if len(unique_labels) == 1:
                return list(unique_labels)[0]
            elif len(unique_labels) > 1:
                return "mixed"
            else:
                raise Exception(f"At least one label should have been passed")

        spans_slice_idxs = find_spans_increasing_list(self.x.index, self.new_bout_thresh, slice_idxs=True)[1]
        average_vector_bouts, bout_labels, bout_x, all_bout_idxs = return_n_empty_lists(4)
        for span_slice_idxs in spans_slice_idxs:  # spans of x for which index difference does not exceed threshold
            x_subset = self.x.iloc[slice(*span_slice_idxs)]  # x in this span
            bouts_idxs = repeat_elements_idxs(x_subset)  # indices of bouts of x being a constant value
            # Convert back to index on original vector. This operation also leaves us with a non-slice idx, which we
            # will use with df.loc, which assumes non-slice index
            bouts_idxs = [x_subset.index[[i1, i2 - 1]] for i1, i2 in bouts_idxs]  # minus one because was slice index
            all_bout_idxs += list(map(np.asarray, bouts_idxs))  # store
            average_vector_bouts += [np.mean(self.df.loc[i1:i2, :].to_numpy(), axis=0) for i1, i2 in
                                     bouts_idxs]  # array values in contiguous stretches of constant x
            bout_labels += [_bout_label(self.labels.loc[i1:i2].values) for i1, i2 in bouts_idxs]
            bout_x += [check_return_single_element(self.x.loc[i1:i2].values).single_element for i1, i2 in bouts_idxs]
        return pd.DataFrame.from_dict({
            "x": bout_x, "label": bout_labels, "vector": average_vector_bouts, "bout_idxs": all_bout_idxs})

    @staticmethod
    def get_x_pair_int(x_pair):
        if len(x_pair) != 2:
            raise Exception(f"x_pair_int must have length two")
        if x_pair[1] - x_pair[0] != 1:
            raise Exception(f"x_pair_int only applies to difference vectors. For these, x_pair should have components "
                            f"one apart, e.g. (3, 4), or be (-1, 1)")
        return x_pair[0]

    @staticmethod
    def get_x_pair(x_pair_int):
        if not int(x_pair_int) == x_pair_int:
            raise Exception(f"x_pair_int must be an integer but is {x_pair_int}")
        return x_pair_int, x_pair_int + 1

    def _get_diff_vector_df(self):
        # Difference of average vectors in *neighboring* contiguous stretches of constant x
        # First, get all difference vectors
        diff_vectors = np.diff(np.vstack(self.vector_df["vector"]), axis=0)
        # Now restrict difference vectors to ones corresponding to pairs of adjacent bouts for
        # which x(n+1) - x(n) is one (e.g. only forward movement along a task phase axis) AND for which the label
        # is the same for bouts (i.e. not "mixed") (e.g. a single maze path)
        non_mixed_bouts_bool = np.prod(zip_adjacent_elements(self.vector_df["label"] != "mixed"),
                                       axis=1)  # bout pairs for which neither one in pair has label "mixed"
        single_label_bouts_bool = self.vector_df["label"][:-1].values == \
                                  self.vector_df["label"][1:].values  # bout pairs with same label
        iterate_one_bouts_bool = np.diff(self.vector_df["x"]) == 1  # x increments by one across bouts
        valid_bool = np.logical_and.reduce((
            iterate_one_bouts_bool, non_mixed_bouts_bool, single_label_bouts_bool))  # require all conditions above

        # Get elements in x_pairs
        x_pairs = np.asarray(zip_adjacent_elements(self.vector_df["x"]))[
            valid_bool]  # pairs of elements of form (n, n + 1)
        x_pairs = array_to_tuple_list(x_pairs)  # convert to tuple form

        # Also represent x_pairs with integer for convenience
        unique_sorted_x_pairs = self._get_sorted_x_pairs_set(x_pairs)
        # ...Ensure first entries in pairs uniquely identify pairs
        check_all_unique([x[0] for x in unique_sorted_x_pairs])
        self.x_pairs_int_map = {x: self.get_x_pair_int(x) for x in unique_sorted_x_pairs}  # store for later use
        x_pairs_int = [self.x_pairs_int_map[x_pair] for x_pair in x_pairs]  # integer representation of x pairs

        # Get start/stop idxs for x_pairs (for each, one tuple with two elements: first element
        # is start idx of first bout in the pair, and second element is end idx of second bout in the pair)
        bout_start_idxs, bout_end_idxs = zip(*self.vector_df["bout_idxs"])
        bout_pair_idxs = array_to_tuple_list(
            np.asarray(list(zip(bout_start_idxs[:-1], bout_end_idxs[1:])))[valid_bool, :])

        return pd.DataFrame.from_dict({"x_pair": x_pairs,
                                       "x_pair_int": x_pairs_int,
                                       "diff_vector": array_to_list(diff_vectors[valid_bool]),
                                       "label": self.vector_df["label"][:-1][valid_bool],
                                       "bout_pair_idxs": bout_pair_idxs})

    def _get_ave_vector_df(self):
        data_list = []
        for x, label in set(list(zip_df_columns(self.vector_df, ["x", "label"]))):
            df_filter_columns(self.vector_df, {"x": x, "label": label})
            vectors = np.vstack(df_filter_columns(self.vector_df, {"x": x, "label": label})["vector"])
            ave_vectors = np.mean(vectors, axis=0)
            data_list.append((x, label, ave_vectors, len(vectors)))
        return df_from_data_list(data_list, ["x", "label", "ave_vector", "num_vectors"])

    def _get_ave_diff_vec_df(self):
        # Find the average difference vectors for each possible labeled bout pair (entity of the
        # form ((x_1, x_2), label))
        # First, get possible labeled bout pairs
        unique_pairs = list(set(list(zip_df_columns(self.diff_vector_df[["x_pair", "label"]]))))
        # Now loop through these
        data_list = []
        for x_pair, label in unique_pairs:
            # Find instances of current labeled bout pair
            df_subset = df_filter_columns(self.diff_vector_df, {"x_pair": x_pair, "label": label})
            vectors_diff = np.asarray([])  # initialize
            if len(df_subset) > 0:  # entries
                vectors_diff = np.vstack(df_subset["diff_vector"])
            # Get average across instances of the difference vector
            ave_vectors_diff = np.mean(vectors_diff, axis=0)
            # Get bout pair idxs for each partner (allows storage in analysis nwb files, since
            # cannot store nested array in these)
            partner_1_idxs, partner_2_idxs = zip(*df_subset["bout_pair_idxs"].values)
            data_list.append(
                (self.get_x_pair_int(x_pair), x_pair, label, ave_vectors_diff, len(vectors_diff),
                 partner_1_idxs, partner_2_idxs))
        return df_from_data_list(data_list, [
            "x_pair_int", "x_pair", "label", "ave_diff_vector", "num_bouts", "partner_1_idxs", "partner_2_idxs"])

    @staticmethod
    def _get_sorted_x_pairs(x_pairs):
        x_pairs_ = list(copy.deepcopy(x_pairs))
        x_pairs_.sort(key=lambda y: y[0])
        return x_pairs_

    @classmethod
    def _get_sorted_x_pairs_set(cls, x_pairs):
        # Set of element pairs sorted in ascending order by first element
        x_pairs = list(set(cls._get_sorted_x_pairs(x_pairs)))
        return x_pairs

    def get_sorted_x_pairs_set(self, filter_key=None):
        # Filter dataframe columns if indicated
        diff_vector_df = self.diff_vector_df  # default
        if filter_key is not None:
            diff_vector_df = df_filter_columns(diff_vector_df, filter_key)
        return self._get_sorted_x_pairs_set(diff_vector_df["x_pairs"])

    def _get_closeness_mask(self, vec_df, idxs_name, mask_duration, verbose=True):
        if verbose:
            print(f"applying closeness mask...")
        valid_bool = np.zeros([len(vec_df)] * 2)  # entries correspond to bout pairs
        valid_bool[:] = np.nan
        bout_idxs = vec_df[idxs_name]
        for idx_1, (start_1, end_1) in enumerate(bout_idxs):
            for idx_2, (start_2, end_2) in enumerate(bout_idxs):
                valid_bool[idx_1, idx_2] = np.min([abs(start_1 - end_2), abs(start_2 - end_1)]) > mask_duration
        # ...Check that all array components got a mask value
        if np.sum(np.isnan(valid_bool)) > 0:
            raise Exception(f"Not all values got masked")
        if verbose:
            plot_heatmap(valid_bool, **{"title": "closeness_mask"})
        return valid_bool

    @classmethod
    def get_bin_centers_map(cls, x, bin_centers):
        # Check inputs
        # ...Check x consists of consecutive integers
        if not all([int(x_i) == x_i for x_i in x]):
            raise Exception(f"all elements in x must be integers")
        # Require x consists of consecutive integers
        if not all(np.diff(x) == 1):
            raise Exception(f"x must consist of consecutive integers")
        # ...Check x and bin_centers same length
        if len(x) != len(bin_centers):
            raise Exception(f"x and bin_centers must be same length since these are meant to correspond to each other")

        # Make the following maps:
        # 1) From x to covariate bin centers
        x_covariate_map = pd.Series(bin_centers, index=x)

        # 2) From x pair to covariate value (center of covariate bin centers)
        # get x pairs
        x_pairs = zip_adjacent_elements(x)
        # get x pair ints
        x_pair_ints = [cls.get_x_pair_int(x_pair) for x_pair in x_pairs]
        # get covariate corresponding to x pair ints
        bin_centers_mean = vector_midpoints(bin_centers)
        # make map
        x_pair_covariate_map = pd.Series(bin_centers_mean, index=x_pair_ints)

        # Return maps
        return {"x": x_covariate_map, "x_pair": x_pair_covariate_map}


"""
# FOR TESTING:
index = np.asarray([1, 2, 3, 4, 6, 7, 8, 9, 11, 12])
x = pd.Series(np.asarray([1, 2, 3, 3, 3, 1, 1, 2, 2, 2]), index=index)
arr = np.tile(np.arange(0, 20, 2), (2, 1)).T
df = pd.DataFrame(arr, index=index)
labels = pd.Series(np.asarray(["a", "b", "b"]*len(x))[:len(x)], index=index)
new_bout_thresh = 1
# For testing individual sections of class:
self = namedtuple("test", "x df labels new_bout_thresh")(x, df, labels, new_bout_thresh)
"""
