import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, fetch1_dataframes, \
    fetch1_dataframes_across_epochs
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec, populate_jguidera_firing_rate_vector
from src.jguides_2024.utils.plot_helpers import plot_heatmap

# Needed for table definitions:
nd

schema = dj.schema("jguidera_firing_rate_difference_vector")


@schema
class FRDiffVecParams(SecKeyParamsBase):
    definition = """
    # Parameters for FRDiffVec
    fr_diff_vec_param_name : varchar(40)
    ---
    diff_n : int
    """

    def _default_params(self):
        return [[1]]

    def insert1(self, key, **kwargs):

        # Check diff n valid
        if key["diff_n"] < 1:
            raise Exception(f"diff_n must be greater than zero")

        super().insert1(key, **kwargs)


@schema
class FRDiffVecSel(SelBase):
    definition = """
    # Selection from upstream tables for FRDiffVec
    -> FRVec
    -> FRDiffVecParams
    """


@schema
class FRDiffVec(ComputedBase):
    definition = """
    # Firing rate difference vectors in an epoch
    -> FRDiffVecSel
    ---
    -> nd.common.AnalysisNwbfile
    fr_diff_vec_object_id : varchar(40)
    vector_tip_time_object_id : varchar(40)
    vector_tail_time_object_id : varchar(40)
    """

    def make(self, key):

        # Get params
        diff_n = (FRDiffVecParams & key).fetch1("diff_n")

        # Get firing rate vectors and corresponding time samples
        fr_df, time_vector = FRVec().firing_rate_across_sort_groups(key, populate_tables=False)

        # Find difference between current sample and that diff_n indices away
        fr_arr = np.vstack(fr_df["firing_rate"])
        fr_diff_arr = fr_arr[:, diff_n:] - fr_arr[:, :-diff_n]

        # Store difference array in df so can save to nwb analysis file
        fr_diff_df = pd.DataFrame.from_dict({unit_name: x for unit_name, x in zip(
            fr_df.index, fr_diff_arr)})

        # Store vector with concatenated time samples of vector tip
        vector_tip_time_df = pd.DataFrame.from_dict({"vector_tip_time": time_vector[diff_n:]})

        # Store vector with concatenated time samples of vector tail
        vector_tail_time_df = pd.DataFrame.from_dict({"vector_tail_time": time_vector[:-diff_n]})

        # Insert into table
        insert_analysis_table_entry(self, [fr_diff_df, vector_tip_time_df, vector_tail_time_df], key)

    def fetch1_fr_diff_vec(self, df_index_name="vector_tail_time"):
        return pd.concat((self.fetch1_dataframe("fr_diff_vec"), self.fetch1_dataframe(df_index_name)),
                         axis=1).set_index(df_index_name)

    def fetch1_dataframes(self):
        return fetch1_dataframes(self)

    def fetch1_dataframes_across_epochs(self, key, axis=0):
        return fetch1_dataframes_across_epochs(self, key, axis)

    def plot_results(self):
        fr_diff_vec_arr = self.fetch1_fr_diff_vec().to_numpy().T
        plot_heatmap(fr_diff_vec_arr, scale_clim=.1, xlabel="time", ylabel="unit")


def populate_jguidera_firing_rate_difference_vector(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_firing_rate_difference_vector"
    upstream_schema_populate_fn_list = [populate_jguidera_firing_rate_vector]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_firing_rate_difference_vector():
    from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity import \
        drop_jguidera_firing_rate_difference_vector_similarity

    drop_jguidera_firing_rate_difference_vector_similarity()

    schema.drop()
