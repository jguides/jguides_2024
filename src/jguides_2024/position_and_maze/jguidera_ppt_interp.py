import datajoint as dj
import numpy as np
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SelBase, CovariateRCB, \
    CovariateDigParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    get_key_filter, delete_
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.glm.jguidera_basis_function import RaisedCosineBasis, RaisedCosineBasisParams
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt
from src.jguides_2024.time_and_trials.jguidera_ppt_trials import populate_jguidera_ppt_trials
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel, \
    ResTimeBinsPool
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import populate_jguidera_res_time_bins_pool
from src.jguides_2024.utils.basis_function_helpers import sample_basis_functions
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.dict_helpers import add_defaults
from src.jguides_2024.utils.digitize_helpers import digitize_indexed_variable

# These imports are called with eval or used in table definitions (do not remove):
ResTimeBinsPool
Ppt
nd

schema = dj.schema("jguidera_ppt_interp")  # define custom schema


@schema
class PptInterpSel(SelBase):
    definition = """
    # Selection from upstream tables for PptInterp
    -> ResTimeBinsPool
    -> Ppt
    """

    def insert_defaults(self, **kwargs):
        # There are two types of data coming together here: 1) proportion path traversed trials (in the form of time
        # intervals) and 2) time bins in which we want to find proportion path traversed in trials
        # NOTE: we want to use time bins that "make sense": that are within the path traversal period. This happens
        # through our choice of res_time_bins_pool_param_name.

        # Get inputs
        key_filter = get_key_filter(kwargs)

        # Get res_time_bins_pool_param_name
        for shorthand_param_name in ["epoch_100ms",  # for embedding
                                      "path_100ms",  # for ppt colored embedding and GLM analysis
                                              ]:
            res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand(shorthand_param_name)
            key_filter.update({"res_time_bins_pool_param_name": res_time_bins_pool_param_name})

            # Insert through potential keys with the above res_time_bins_pool_param_name
            for key in self._get_potential_keys(key_filter):
                self.insert1(key)


@schema
class PptInterp(ComputedBase):
    definition = """
    # Proportion path traversed interpolated using time bins from ResTimeBinsPool
    -> PptInterpSel
    ---
    -> nd.common.AnalysisNwbfile
    ppt_interp_object_id : varchar(40)
    ppt_range : blob  # copied from Ppt for convenience
    """

    def make(self, key):
        # Reindex proportion path traversed using time bin center from restricted time bins pool table
        time_bins_df = (ResTimeBinsPool & key).fetch1_dataframe()
        time_bin_centers = time_bins_df["time_bin_centers"].values
        ppt_interp_df = (Ppt & key).ppt_df_all_time(new_index=time_bin_centers)
        # Add time bin edges to df
        ppt_interp_df["time_bin_edges"] = time_bins_df["time_bin_edges"].values
        # Get proportion path traversed range to add to table
        key.update({"ppt_range": (Ppt & key).fetch1("ppt_range")})
        # Insert into table
        insert_analysis_table_entry(
            self, nwb_objects=[ppt_interp_df], key=key, reset_index=True, replace_none_col_names=["path_name"])

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def fetch1_ppt(self):
        return self.fetch1_dataframe().ppt

    def digitized_ppt(self, bin_width=None, bin_edges=None, verbose=False):
        # Check inputs
        check_one_none([bin_width, bin_edges], ["bin_width", "bin_edges"])

        # Make bin edges if not passed
        if bin_edges is None:
            bin_edges = PptDigParams().make_bin_edges(bin_width=bin_width)

        # Get interpolated ppt
        ppt_interp_df = self.fetch1_dataframe()

        # Digitize ppt
        ppt_interp_df["digitized_ppt"] = digitize_indexed_variable(
            indexed_variable=ppt_interp_df["ppt"], bin_edges=bin_edges, verbose=verbose)

        return ppt_interp_df

    def delete_(self, key=None, safemode=True):
        # Delete downstream entries first
        from src.jguides_2024.jguidera_firing_rate_difference_vector_similarity_ave import FRDiffVecCosSimPptNnAve
        delete_(self, [FRDiffVecCosSimPptNnAve], key, safemode)


@schema
class PptDigParams(CovariateDigParamsBase):
    definition = """
    # Parameters for PptFRDiffVecAnalysis
    ppt_dig_param_name : varchar(40)
    ---
    ppt_bin_width : float
    """

    def _default_params(self):
        return [
            # for GLM analysis and ppt aligned firing rate vector analysis
            [.05],
            # for ppt aligned firing rate vector analysis
            [1/16], [.1],
                ]

    def make_bin_edges(self, **kwargs):
        # Path fraction bin edges to be used to digitize path fraction
        if "bin_width" not in kwargs:
            raise Exception(f"bin_width must be passed")
        return Ppt.make_bin_edges(kwargs["bin_width"])

    def get_valid_bin_nums(self, **kwargs):
        return np.arange(1, self.get_num_bin_edges(**kwargs))


@schema
class PptDigSel(SelBase):
    definition = """
    # Selection from upstream tables for PptDig
    -> PptInterp
    -> PptDigParams
    """


@schema
class PptDig(ComputedBase):
    definition = """
    # Digitized interpolated fraction path traversed
    -> PptDigSel
    ---
    -> nd.common.AnalysisNwbfile
    ppt_dig_object_id : varchar(40)
    """

    def make(self, key):
        ppt_bin_width = (PptDigParams & key).fetch("ppt_bin_width")
        ppt_dig_df = (PptInterp & key).digitized_ppt(bin_width=ppt_bin_width)
        # Reset index since index not properly stored in analysis nwb file
        ppt_dig_df = ppt_dig_df.reset_index()
        insert_analysis_table_entry(self, [ppt_dig_df], key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)


@schema
class PptRCBSel(SelBase):
    definition = """
    # Selection from upstream tables for PptRCB
    -> PptDig
    -> RaisedCosineBasis
    """

    def insert1(self, key, **kwargs):
        # Ensure bin width match across raised cosine basis and fraction path traversed
        x1 = (RaisedCosineBasisParams & key).fetch1("bin_width")
        x2 = (PptDigParams & key).fetch1("ppt_bin_width")
        if x1 != x2:
            raise Exception(
                f"bin_width in RaisedCosineBasisParams must match ppt_bin_width in PptDigParams. "
                f"These are respectively {x1} and {x2}")
        super().insert1(key, **kwargs)

    # Override parent class method to ensure bin width matches across raised cosine basis and fraction path traversed,
    # and impose default params
    def _get_potential_keys(self, key_filter=None):
        if key_filter is None:
            key_filter = dict()
        # GLM analysis (path)
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_params
        default_params = get_glm_default_params(
            ["ppt_dig_param_name", "path_res_time_bins_pool_param_name", "path_raised_cosine_basis_param_name"])
        key_filter = add_defaults(key_filter, default_params, add_nonexistent_keys=True, require_match=True)
        keys = []
        for k1 in (RaisedCosineBasis & key_filter).fetch("KEY"):
            bin_width = (RaisedCosineBasisParams & k1).fetch1("bin_width")
            key_filter.update({"ppt_dig_param_name": bin_width})
            for k2 in (PptDig & key_filter).fetch("KEY"):
                keys.append({**k1, **k2})
        return keys

    def delete_(self, key, safemode=False):
        delete_(self, [PptRCB], key, safemode)


@schema
class PptRCB(CovariateRCB):
    definition = """
    # Sampled raised cosine basis, proportion path traversed
    -> PptRCBSel
    ---
    -> nd.common.AnalysisNwbfile
    ppt_rcb_df_object_id : varchar(40)
    """

    def make(self, key):
        ppt_dig = (PptDig & key).fetch1_dataframe().digitized_ppt
        basis_functions = (RaisedCosineBasis & key).fetch1_basis_functions()
        ppt_rcb_df = sample_basis_functions(ppt_dig, basis_functions, tolerate_outside_basis_domain=True)
        insert_analysis_table_entry(self, [ppt_rcb_df], key, reset_index=True)

    def delete_(self, key, safemode=False):
        from src.jguides_2024.glm.jguidera_measurements_interp_pool import XInterpPool
        delete_(self, [XInterpPool], key, safemode)


def populate_jguidera_ppt_interp(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_ppt_interp"
    upstream_schema_populate_fn_list = [populate_jguidera_res_time_bins_pool, populate_jguidera_ppt_trials]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_ppt_interp():
    from src.jguides_2024.glm.jguidera_measurements_interp_pool import drop_jguidera_measurements_interp_pool
    from src.jguides_2024.firing_rate_vector.jguidera_path_firing_rate_vector import drop_jguidera_path_firing_rate_vector
    from src.jguides_2024.jguidera_firing_rate_difference_vector_similarity_ave import drop_jguidera_firing_rate_difference_vector_similarity_ave
    drop_jguidera_measurements_interp_pool()
    drop_jguidera_path_firing_rate_vector()
    drop_jguidera_firing_rate_difference_vector_similarity_ave()
    schema.drop()
