import datajoint as dj
import numpy as np
import pandas as pd
import scipy as sp
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, PartBase, \
    AcrossFRVecTypeTableSelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    insert1_print, delete_
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector import FRDiffVec, \
    populate_jguidera_firing_rate_difference_vector, \
    FRDiffVecParams
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector_euclidean_distance import FRVecEucDist
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams

# Needed for table definitions:
ResTimeBinsPoolCohortParams
BrainRegionUnits
FRDiffVec
nd

schema = dj.schema("jguidera_firing_rate_difference_vector_similarity")

"""
Notes on design of FRDiffVecCosSim tables:
- We want the selection table to contain all primary keys of FRVec except for epoch and res_time_bins_pool_param_name
(nwb_file_name, brain_region, brain_region_units_param_name, curation name, res_epoch_spikes_sm_param_name,
zscore_fr), and to contain epochs_id and res_time_bins_pool_cohort_param_name (time bins across epochs)
as a primary key.
Rationale: we want to look at cosine similarity of firing rate difference vectors within and optionally
across epochs (and each epoch has an associated res_time_bins_pool_param_name, which we want to allow
to be different across epochs for maximal flexibility)
"""


@schema
class FRDiffVecCosSimSel(AcrossFRVecTypeTableSelBase):
    definition = """
    # Selection from upstream tables for FRDiffVecCosSim
    -> ResTimeBinsPoolCohortParams  # nwb_file_name, epochs_id, res_time_bins_cohort_param_name
    -> BrainRegionUnits  # nwb_file_name, brain_region, brain_region_units_param_name, curation_name
    res_epoch_spikes_sm_param_name : varchar(40)
    fr_diff_vec_param_name : varchar(40)
    zscore_fr : bool
    """

    @staticmethod
    def _fr_vec_table():
        return FRDiffVec

    def delete_(self, key, safemode=True):
        delete_(self, [FRDiffVecCosSim], key, safemode)


@schema
class FRDiffVecCosSim(ComputedBase):
    definition = """
    # Cosine similarity between firing rate difference vectors
    -> FRDiffVecCosSimSel
    ---
    -> nd.common.AnalysisNwbfile
    fr_diff_vec_cos_dist_object_id : varchar(40)
    vector_tail_time_object_id : varchar(40)
    vector_tip_time_object_id : varchar(40)
    epoch_vector_object_id : varchar(40)
    """

    class Upstream(PartBase):
        definition = """
        # Achieves dependence on upstream tables
        -> FRDiffVecCosSim
        -> FRDiffVec
        """

    def make(self, key):
        # Get firing rate difference vector dfs across epochs
        dfs = FRDiffVec().fetch1_dataframes_across_epochs(key)
        fr_diff_vec_arr = dfs.fr_diff_vec.to_numpy()

        # Find cosine distance between firing rate difference vectors
        fr_diff_vec_cos_dist = sp.spatial.distance.pdist(fr_diff_vec_arr, metric='cosine')

        # Store in vector form to save space (symmetric matrix). Note that we do not store cosine
        # similarity since the function to retrieve the matrix from vector expects distance measure
        fr_diff_vec_cos_dist_df = pd.DataFrame.from_dict({"fr_diff_vec_cos_dist": fr_diff_vec_cos_dist})

        # Insert into main table
        insert_analysis_table_entry(self, [
            fr_diff_vec_cos_dist_df, dfs.vector_tail_time, dfs.vector_tip_time, dfs.epoch_vector], key)

        # Insert into part table
        for part_key in ResTimeBinsPoolCohortParams().get_keys_with_cohort_params(key):
            insert1_print(self.Upstream, part_key)

    def fetch1_fr_diff_vec_cos_sim(self, df_index_name="vector_tail_time"):
        # Convert cosine distance to cosine similarity. Important to do this after data stored since
        # function to convert vector to matrix expects distance measure
        sim_arr = 1 - sp.spatial.distance.squareform(
            self.fetch1_dataframe("fr_diff_vec_cos_dist").fr_diff_vec_cos_dist.values)
        time_vector = np.ndarray.flatten(self.fetch1_dataframe(df_index_name).values)
        return pd.DataFrame(sim_arr, index=time_vector, columns=time_vector)

    def get_nn_cosine_similarity(
            self, n_neighbors, nn_restrictions=None, state_restrictions=None, populate_tables=True):

        key = self.fetch1("KEY")

        # Exclude diff_n rows and columns since no corresponding difference vectors
        # ...Get number of indices in firing rate vector array spanned by a difference vector
        diff_n = (FRDiffVecParams & key).fetch1("diff_n")
        # ...Update restrictions
        diff_n_restriction = {"exclude_final_n_samples": diff_n}
        nn_restrictions.update(diff_n_restriction)
        state_restrictions.update(diff_n_restriction)

        # Get idxs of nearest neighbors (per euclidean distance) for each firing rate vector
        # ...Populate upstream table if indicated
        if populate_tables:
            FRVecEucDist().populate_(key=key)
        # ...Get nearest neighbor indices for each state and time vector corresponding to states
        table_subset = (FRVecEucDist & key)
        nn_output = table_subset.get_nn_idxs(n_neighbors, nn_restrictions, state_restrictions)
        # update upstream entries tracker with that from FRVecEucDist
        self._merge_tracker(table_subset)

        # Get cosine similarity of difference vectors
        cos_df = (self & key).fetch1_fr_diff_vec_cos_sim()

        # Restrict to states as indicated
        cos_df = cos_df.loc[nn_output.row_time_vector][nn_output.col_time_vector]

        # Restrict to samples' cosine similarity with nearest neighbors
        # Currently, unsure how to achieve desired indexing efficiently when distance matrix is as
        # pandas df, so use array
        cos_arr = cos_df.to_numpy()  # convert to array
        # important to index columns with array of integers and not colon
        col_idxs = np.arange(0, np.shape(cos_arr)[1])
        nn_cos_arr = np.vstack([cos_arr[nth_nn_idxs, col_idxs] for nth_nn_idxs in nn_output.sort_idxs])

        # Return as df
        index = [f"nn_{x}" for x in np.arange(0, len(nn_output.sort_idxs))]
        return pd.DataFrame(nn_cos_arr, columns=cos_df.columns, index=index)

    def get_average_nn_cosine_similarity(self, n_neighbors, nn_restrictions, state_restrictions, populate_tables):
        return np.mean(self.get_nn_cosine_similarity(
            n_neighbors, nn_restrictions, state_restrictions, populate_tables), axis=0)

    def delete_(self, key, safemode=True):
        from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity_ave import \
            FRDiffVecCosSimPptNnAveSel
        delete_(self, [FRDiffVecCosSimPptNnAveSel], key, safemode)


def populate_jguidera_firing_rate_difference_vector_similarity(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_firing_rate_difference_vector_similarity"
    upstream_schema_populate_fn_list = [populate_jguidera_firing_rate_difference_vector]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_firing_rate_difference_vector_similarity():
    schema.drop()

