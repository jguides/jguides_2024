import copy
from collections import namedtuple

import datajoint as dj
import numpy as np
import pandas as pd
import scipy as sp

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, PartBase, \
    AcrossFRVecTypeTableSelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    insert1_print, \
    delete_
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec, \
    populate_jguidera_firing_rate_vector
from src.jguides_2024.metadata.jguidera_epoch import EpochCohort
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams
from src.jguides_2024.utils.plot_helpers import plot_heatmap
from src.jguides_2024.utils.set_helpers import check_membership

# Needed for table definitions:
ResTimeBinsPoolCohortParams
BrainRegionUnits
FRVec

schema = dj.schema("jguidera_firing_rate_vector_euclidean_distance")


"""
Notes on design of FRVecEucDist tables:

1) We want FRVecEucDist to contain all primary keys of FRVec except for epoch and res_time_bins_pool_param_name
(nwb_file_name, brain_region, brain_region_units_param_name, curation name, res_epoch_spikes_sm_param_name,
zscore_fr), and to contain epochs_id and res_time_bins_pool_cohort_param_name (time bins across epochs)
as a primary key.
Rationale: we want to look at distances between firing vectors within and optionally across epochs (and each epoch
has an associated res_time_bins_pool_param_name, which we want to allow to be different across epochs for maximal
flexibility)
"""


@schema
class FRVecEucDistSel(AcrossFRVecTypeTableSelBase):
    definition = """
    # Selection from upstream tables for FRVecEucDist
    -> ResTimeBinsPoolCohortParams  # nwb_file_name, epochs_id, res_time_bins_cohort_param_name
    -> BrainRegionUnits  # nwb_file_name, brain_region, brain_region_units_param_name, curation_name
    res_epoch_spikes_sm_param_name : varchar(40)
    zscore_fr : bool
    """

    @staticmethod
    def _fr_vec_table():
        return FRVec

    def delete_(self, key, safemode=True):
        delete_(self, [FRVecEucDist], key, safemode)


@schema
class FRVecEucDist(ComputedBase):
    definition = """
    # Euclidean distance between firing rate vectors
    -> FRVecEucDistSel
    ---
    -> nd.common.AnalysisNwbfile
    fr_euc_dist_object_id : varchar(40)
    time_vector_object_id : varchar(40)
    epoch_vector_object_id : varchar(40)
    """

    class Upstream(PartBase):
        definition = """
        # Achieves dependence on upstream tables
        -> FRVecEucDist
        -> FRVec
        """

    def make(self, key):

        # Find euclidean distance between firing rate vectors
        dfs = FRVec().firing_rate_across_sort_groups_epochs(key, populate_tables=False)
        distance_vec = sp.spatial.distance.pdist(dfs.fr_concat_df)

        # Store distance in vector to save space (since symmetric matrix)
        distance_df = pd.DataFrame.from_dict({"fr_distance": distance_vec})

        # Store vector with concatenated time samples (corresponding to distance array columns and rows) in df
        time_df = pd.DataFrame({dfs.fr_concat_df.index.name: dfs.fr_concat_df.index})

        # Insert into main table
        insert_analysis_table_entry(self, [distance_df, time_df, dfs.epoch_vector], key)

        # Insert into part table
        for part_key in ResTimeBinsPoolCohortParams().get_keys_with_cohort_params(key):
            insert1_print(self.Upstream, part_key)

    def fetch1_fr_euc_dist(self):
        # Return firing rate euclidean distance in matrix
        dist_arr = sp.spatial.distance.squareform(self.fetch1_dataframe("fr_euc_dist").fr_distance.values)
        time_vector = np.ndarray.flatten(self.fetch1_dataframe("time_vector").values)
        return pd.DataFrame(dist_arr, index=time_vector, columns=time_vector)

    def fetch1_masked_fr_euc_dist(self, mask_duration=0, mask_value=np.inf):
        # Mask points within some time period of each other as indicated (set nearby points to have distance to
        # current point given by mask_value). Get time per sample so we can convert the mask duration from seconds
        # to sample indices. mask_duration specifies amount of time to mask on either side (so we mask a total of
        # twice the passed mask_duration). In masking, invalid values are converted to inf. Shape of distance
        # matrix remains unchanged.

        # Important TODO: should change masking procedure to account for fact that data can span epochs.
        #  Adjacent data from different epochs
        #  should not be masked. Approach: separate the data by epoch, perform the masking within each epoch, then
        #  rejoin the data across epochs

        # Check inputs
        if mask_duration < 0:
            raise Exception(f"mask_duration must be nonnegative")

        # Get time per index in array
        time_bin_width = (ResTimeBinsPoolCohortParams & self.fetch1("KEY")).get_time_bin_width()
        # Get number of idxs to mask before and after current sample to mask
        mask_idxs_pre_post = int(np.ceil(mask_duration / time_bin_width))

        distance_df = self.fetch1_fr_euc_dist()
        distance_arr = distance_df.to_numpy()
        n_row, n_col = np.shape(distance_arr)
        for col_num in np.arange(0, n_col):
            mask_start_idx = np.max([0, col_num - mask_idxs_pre_post])
            mask_end_idx = np.min([n_row, col_num + mask_idxs_pre_post])
            distance_arr[mask_start_idx:mask_end_idx, col_num] = mask_value

        return distance_arr, distance_df.index

    def get_nn_idxs(self, n_neighbors, nn_restrictions=None, state_restrictions=None):
        """
        Find indices that sort distance matrix within each column. The indices in the first n rows of a given column
        of the resulting sorted matrix are the n nearest neighbors of the state corresponding
        to that column (where column index corresponds to state number).

        :param n_neighbors: integer, number of nearest neighbors
        :param nn_restrictions: dict with restrictions on which states can be considered as nearest neighbors
            valid key/value pairs include:
            - 'mask_duration': time in seconds, used to exclude states within a certain proximity
            - 'exclude_final_n_samples': integer, number of final states to exclude from nearest neighbor determination.
              Useful when nearest neighbors will be applied to a corresponding matrix of difference vectors, as
              there are no difference vectors for final diff_n states where diff_n is the number of samples
              spanned by each difference vector)
        :param state_restrictions: list with restrictions on states for which nearest neighbors are found
            valid entries include:
            - 'exclude_final_n_samples': as above
            - 'correct_trial': bool, True to include only correct trials
        :return: named tuple with sort indices (in terms of original time index) and time vector corresponding to states
        """

        # Get inputs if not passed
        if nn_restrictions is None:
            nn_restrictions = dict()
        if state_restrictions is None:
            state_restrictions = dict()
        if "mask_duration" not in nn_restrictions:
            nn_restrictions["mask_duration"] = 0

        # Check inputs
        # ...Check that specified final number of samples to exclude is non-negative
        for restrictions in [state_restrictions, nn_restrictions]:
            if restrictions["exclude_final_n_samples"] < 0:
                raise Exception(f"exclude_final_n_samples must be nonnegative")
        # ...Check that nn restrictions are valid
        check_membership(nn_restrictions, ["mask_duration", "exclude_final_n_samples"])
        # ...Check that state restrictions are valid
        check_membership(list(state_restrictions.keys()), [
            "exclude_final_n_samples", "potentially_rewarded_trial", "correct_trial", "stay_trial", "leave_trial"])

        # Get firing rate vector euclidean distance matrix. Mask samples from nearest neighbor determination
        # based on proximity as indicated
        distance_arr, col_time_vector = self.fetch1_masked_fr_euc_dist(nn_restrictions["mask_duration"])

        # Make copy of time vector for rows
        row_time_vector = copy.deepcopy(col_time_vector)

        # 1) Restrictions on which states can be considered as nearest neighbors of other states:

        # Exclude a specified number of final samples
        if "exclude_final_n_samples" in nn_restrictions:
            # Exclude from rows since finding nearest neighbors within each column
            end_idx = -nn_restrictions["exclude_final_n_samples"]
            distance_arr = distance_arr[:end_idx, :]
            # Update corresponding elements in time vector for rows
            row_time_vector = row_time_vector[:end_idx]

        # 2) Restrictions on states for which nearest neighbors are found:

        # Exclude a specified number of final samples
        if "exclude_final_n_samples" in state_restrictions:
            # Exclude from columns since these correspond to states for which nearest neighbors found
            end_idx = -state_restrictions["exclude_final_n_samples"]
            distance_arr = distance_arr[:, :end_idx]
            # Exclude corresponding samples in time vector
            col_time_vector = col_time_vector[:end_idx]

        # Potentially rewarded trials, correct trials, "stay" or "leave" delay period at wells
        for restriction in [
            "potentially_rewarded_trial", "correct_trial", "incorrect_trial", "stay_trial", "leave_trial"]:

            if restriction in state_restrictions:

                # Get boolean indicating which states occurred during desired trials
                key = self.fetch1("KEY")
                epoch = (EpochCohort & key).get_epoch()
                key.update({"epoch": epoch})
                restriction_table = get_table(state_restrictions[restriction]["table_name"])
                valid_bool = (restriction_table & key).in_trial(col_time_vector, [restriction])
                # update upstream entries tracker
                self._update_upstream_entries_tracker(restriction_table, key)

                # Restrict states
                distance_arr = distance_arr[:, valid_bool]

                # Restrict corresponding samples in time vector
                col_time_vector = col_time_vector[valid_bool]

        # Sort distance matrix within each column
        sort_idxs = np.argsort(distance_arr, axis=0)

        # Restrict sorted distance matrix to just top n_neighbors (first n_neighbors rows within each column)
        sort_idxs = sort_idxs[:n_neighbors, :]

        return namedtuple(
            "Nn", "row_time_vector col_time_vector sort_idxs")(row_time_vector, col_time_vector, sort_idxs)

    def plot_results(self):
        fr_euc_dist_df = self.fetch1_fr_euc_dist()
        plot_heatmap(fr_euc_dist_df, scale_clim=.8, figsize=(10, 10))


def populate_jguidera_firing_rate_vector_euclidean_distance(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_firing_rate_vector_euclidean_distance"
    upstream_schema_populate_fn_list = [populate_jguidera_firing_rate_vector]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_firing_rate_vector_euclidean_distance():
    from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity_ave import \
        drop_jguidera_firing_rate_difference_vector_similarity_ave
    drop_jguidera_firing_rate_difference_vector_similarity_ave()
    schema.drop()
