from collections import namedtuple

import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_pool_table_base import PoolSelBase, PoolBase, \
    PoolCohortParamsBase, PoolCohortBase, \
    PoolCohortParamNameBase, EpsCohortParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, CohortBase, SelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, delete_, \
    insert_analysis_table_entry, unique_table_column_sets
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_delay_interval
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.glm.jguidera_basis_function import RaisedCosineBasisParams
from src.jguides_2024.position_and_maze.jguidera_ppt import PptParams
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptRCB, PptDigParams, \
    populate_jguidera_ppt_interp
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPool, \
    ResTimeBinsPoolCohortParams, ResTimeBinsPoolCohort, \
    ResTimeBinsPoolSel
from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWARCB, \
    populate_jguidera_time_relative_to_well_event, \
    TimeRelWADigSingleAxisParams, TimeRelWADigParams
from src.jguides_2024.utils.dict_helpers import add_defaults, check_return_single_dict
from src.jguides_2024.utils.vector_helpers import check_all_unique, unpack_single_element

# Needed for table definitions:
ResTimeBinsPool
ResTimeBinsPoolCohort
PptRCB
TimeRelWARCB
nd

schema_name = "jguidera_measurements_interp_pool"
schema = dj.schema(schema_name)


@schema
class InterceptSel(SelBase):
    definition = """
    # Selection from upstream table for Intercept
    -> ResTimeBinsPool
    """

    def _get_potential_keys(self, key_filter=None):
        keys = super()._get_potential_keys(key_filter)
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_params_map
        glm_default_params_map = get_glm_default_params_map()
        valid_res_time_bins_pool_param_names = [
            glm_default_params_map["delay_res_time_bins_pool_param_name"],
            glm_default_params_map["path_res_time_bins_pool_param_name"]]  # for GLM analysis
        return [k for k in keys if
                k["res_time_bins_pool_param_name"] in valid_res_time_bins_pool_param_names]


@schema
class Intercept(ComputedBase):
    definition = """
    # Intercept term for model (vector of ones)
    -> InterceptSel
    ---
    -> nd.common.AnalysisNwbfile
    intercept_df_object_id : varchar(40)
    """

    def make(self, key):
        time_bins_df = (ResTimeBinsPool & key).fetch1_dataframe()
        time_bin_centers = time_bins_df.time_bin_centers
        intercept_df = pd.DataFrame.from_dict({"intercept": np.ones(len(time_bin_centers)), "time": time_bin_centers})
        insert_analysis_table_entry(self, [intercept_df], key, ["intercept_df_object_id"])

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        if df_index_name is None:
            df_index_name = "time"
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)


@schema
class XInterpPoolSel(PoolSelBase):
    definition = """
    # Specifies entries from upstream tables for XInterpPool
    -> ResTimeBinsPool
    x_interp_pool_param_name : varchar(200)
    ---
    source_table_name : varchar(80)
    source_params_dict : blob
    param_name_dict : blob
    int_id : int
    """

    @staticmethod
    def _get_valid_source_table_names():
        return ["PptRCB", "Intercept", "TimeRelWARCB"]

    def get_shorthand_params_map(self):
        # Return map from shorthand for x_interp_pool_param_name to underlying params

        def _update_shorthand_params_map(
                shorthand_params_map, shorthand_params_name, source_table_name, source_table_key):
            x_interp_pool_param_name = self.lookup_param_name(source_table_name, source_table_key)
            shorthand_params_map.update({
                shorthand_params_name: ShorthandParams(source_table_name, source_table_key, x_interp_pool_param_name)})
            return shorthand_params_map

        # Return params in objects
        shorthand_params_map = dict()
        ShorthandParams = namedtuple(
            "ShorthandParams", "source_table_name source_table_key x_interp_pool_param_name")

        # DEFAULT PATH TRAVERSAL
        shorthand_params_name = "ppt"
        source_table_name = "PptRCB"
        ppt_param_name = PptParams().get_default_param_name()
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_param
        bin_width = get_glm_default_param("ppt_bin_width")
        source_table_key = {
            "raised_cosine_basis_param_name": RaisedCosineBasisParams().lookup_param_name_from_shorthand("ppt"),
                            "ppt_param_name": ppt_param_name,
                            "ppt_dig_param_name": PptDigParams().lookup_param_name([bin_width])}
        shorthand_params_map = _update_shorthand_params_map(
            shorthand_params_map, shorthand_params_name, source_table_name, source_table_key)

        # DEFAULT TIME IN DELAY
        shorthand_params_name = "delay"
        source_table_name = "TimeRelWARCB"
        source_table_key = {
            "time_rel_wa_dig_param_name": TimeRelWADigParams().lookup_param_name([.1]),
            "time_rel_wa_dig_single_axis_param_name": TimeRelWADigSingleAxisParams().lookup_param_name(
                get_delay_interval()),
            "raised_cosine_basis_param_name": RaisedCosineBasisParams().lookup_param_name_from_shorthand("delay"),
        }
        shorthand_params_map = _update_shorthand_params_map(
            shorthand_params_map, shorthand_params_name, source_table_name, source_table_key)

        # INTERCEPT
        shorthand_params_name = "intercept"
        source_table_name = "Intercept"
        source_table_key = dict()
        x_interp_pool_param_name = self.lookup_param_name(source_table_name, source_table_key)
        shorthand_params_map.update(
            {shorthand_params_name: ShorthandParams(source_table_name, source_table_key, x_interp_pool_param_name)})

        return shorthand_params_map

    def delete_(self, key, safemode=True):
        delete_(self, [XInterpPool], key, safemode)


@schema
class XInterpPool(PoolBase):
    definition = """
    # Placeholder for interpolated measurements within an epoch across sources
    -> XInterpPoolSel
    ---
    part_table_name : varchar(80)
    """

    class PptRCB(dj.Part):
        definition = """
        # Placeholder for entries from PptRCB
        -> XInterpPool
        -> PptRCB
        """

    class TimeRelWARCB(dj.Part):
        definition = """
        # Placeholder for entries from TimeRelWARCB
        -> XInterpPool
        -> TimeRelWARCB
        """

    class Intercept(dj.Part):
        definition = """
        # Placeholder for entries from Intercept
        -> XInterpPool
        -> Intercept
        """

    def delete_(self, key, safemode=True):
        delete_(self, [XInterpPoolCohortParams], key, safemode)


@schema
class XInterpPoolCohortParamName(PoolCohortParamNameBase):
    definition = """
    # Map between full param name and integer used as param name
    x_interp_pool_cohort_param_name : varchar(500)  # reflects x_interp_pool_param_names composing cohort
    ---
    int_id = NULL : int
    full_param_name : varchar(2000)
    """


@schema
class XInterpPoolCohortParams(PoolCohortParamsBase):
    definition = """
    # Specifies groups of entries from XInterpPool within an epoch across measurement types
    -> ResTimeBinsPool
    -> XInterpPoolCohortParamName
    ---
    x_interp_pool_param_names : blob
    """

    def insert1(self, key, **kwargs):
        # Check that entries in x_interp_pool_param_names all unique since no effect of including the same
        # entry more than once on downstream processing (i.e. would lead to multiple cohort entries with same effect)
        check_all_unique(key["x_interp_pool_param_names"])
        super().insert1(key, **kwargs)

    def get_shorthand_params_map(self):

        def _update_shorthand_params_map(shorthand_params_map, x_interp_pool_shorthand_param_names):
            data_list = [
                (params_map[x].source_table_name, params_map[x].source_table_key,
                 params_map[x].x_interp_pool_param_name) for x in x_interp_pool_shorthand_param_names]
            source_table_names, source_table_keys, x_interp_pool_param_names = list(zip(*data_list))
            shorthand_param_name = "_".join(x_interp_pool_shorthand_param_names)
            # Get cohort param name if available (only possible if table populated, which will not be case
            # when access this map to populate table)
            x_interp_pool_cohort_param_name = None
            try:
                x_interp_pool_cohort_param_name = self.lookup_param_name(source_table_names, source_table_keys)
            except Exception as e:
                print(f"could not get x_interp_pool_cohort_param_name: {e}. Setting to None")
            ShorthandParams = namedtuple(
                "ShorthandParams", "source_table_names source_table_keys x_interp_pool_param_names "
                                   "x_interp_pool_shorthand_param_names x_interp_pool_cohort_param_name")
            shorthand_params_map.update(
                {shorthand_param_name: ShorthandParams(
                    source_table_names, source_table_keys, x_interp_pool_param_names,
                    x_interp_pool_shorthand_param_names,
                    x_interp_pool_cohort_param_name)})
            return shorthand_params_map

        # Return params in objects
        shorthand_params_map = dict()

        # get shorthand params map for upstream pooled interpolated measurements
        params_map = XInterpPoolSel().get_shorthand_params_map()

        # "ppt_intercept": ppt and intercept
        shorthand_params_map = _update_shorthand_params_map(shorthand_params_map, ["ppt", "intercept"])

        # "delay_intercept": time in delay and intercept
        shorthand_params_map = _update_shorthand_params_map(shorthand_params_map, ["delay", "intercept"])

        return shorthand_params_map

    def insert_glm_defaults(self, **kwargs):

        # Populate cohort table with intercept and single entries for ppt (PptRCB) and time in delay period
        # (TimeRelWARCB) for GLM analysis
        shorthand_params_map = self.get_shorthand_params_map()

        key_filter = dict()
        if "key_filter" in kwargs:
            key_filter = kwargs["key_filter"]

        # Loop through potential keys and insert. We hold everything from XInterpPoolSel except
        # x_interp_pool_param_name constant. We add a list of x_interp_pool_param_names that make up the cohort.
        for params in shorthand_params_map.values():
            for key in unique_table_column_sets(
                    XInterpPoolSel & key_filter, [
                        x for x in XInterpPoolSel().primary_key if x != "x_interp_pool_param_name"], as_dict=True):
                # Only proceed if upstream table fully populated for the current case
                if all([len(XInterpPool & {**key, **{"x_interp_pool_param_name": x_interp_pool_param_name}}) > 0
                        for x_interp_pool_param_name in params.x_interp_pool_param_names]):
                    secondary_key_subset_map = {"x_interp_pool_param_names": params.x_interp_pool_param_names}
                    XInterpPoolCohortParams()._insert_from_upstream_param_names(
                        secondary_key_subset_map, key, use_full_param_name=True)
                else:
                    print(f"XInterpPool not fully populated for key: {key} and each of "
                          f"params.x_interp_pool_param_names: {params.x_interp_pool_param_names}")

    def insert_defaults(self, **kwargs):
        self.insert_glm_defaults(**kwargs)

    def lookup_param_name_from_shorthand(self, shorthand_param_name):
        return self.get_shorthand_params_map()[shorthand_param_name].x_interp_pool_cohort_param_name

    def lookup_shorthand_from_param_name(self, x_interp_pool_cohort_param_name):
        return unpack_single_element(
            [k for k, v in self.get_shorthand_params_map().items()
             if v.x_interp_pool_cohort_param_name == x_interp_pool_cohort_param_name])

    def delete_(self, key, safemode=True):
        delete_(self, [XInterpPoolCohort], key, safemode)


# Pool within epochs before pooling across epochs. Rationale: easily combine interpolated measurements with data
# in other tables restricted to single epochs. The two will share "epoch" as a primary key.
@schema
class XInterpPoolCohort(PoolCohortBase):
    definition = """
    # Groups of entries from XInterpPool across measurement types and within an epoch
    -> XInterpPoolCohortParams
    """

    class CohortEntries(dj.Part):
        definition = """
        # Entries from XInterpPool
        -> XInterpPoolCohort
        -> XInterpPool
        """

    def fetch_dataframes(self, **kwargs):
        # Do not include iterable in df
        kwargs = add_defaults(kwargs, {"add_iterable": False, "axis": 1}, add_nonexistent_keys=True, require_match=True)
        return super().fetch_dataframes(**kwargs)

    def get_covariate_group_map(self, shorthand_map=None):
        # Make map from x_interp_pool_param_name in cohort to corresponding covariate names
        covariate_group_map = {k: v.columns.values for k, v in self.fetch_dataframes(concatenate=False).items()}
        # Replace x_interp_pool_param_name with shorthand if indicated
        if shorthand_map is not None:
            check_all_unique(shorthand_map.values())  # ensure will not lose values due to redundant new key names
            for old_k, new_k in shorthand_map.items():
                v = covariate_group_map.pop(old_k)
                covariate_group_map[new_k] = v
        return covariate_group_map

    def delete_(self, key, safemode=True):
        delete_(self, [XInterpPoolCohortEpsCohortParams], key, safemode)


# Combine groups of measurements across epochs
@schema
class XInterpPoolCohortEpsCohortParams(EpsCohortParamsBase):
    definition = """
    # Specifies groups of entries from XInterpPoolCohort across epochs
    -> ResTimeBinsPoolCohort
    x_interp_pool_cohort_param_name : varchar(500)  # based on cohort param names in interpolated measurements pool cohort table
    ---
    epochs : blob
    res_time_bins_pool_param_names : blob
    """

    @staticmethod
    def _upstream_table():
        return XInterpPoolCohort

    # Extend parent class method so can handle time bins params
    @classmethod
    def _update_key(cls, key):
        key = super()._update_key(key)
        pool_param_names = [key.pop("res_time_bins_pool_param_name")]
        key.update({"res_time_bins_pool_param_names": pool_param_names,
                    "res_time_bins_pool_cohort_param_name": ResTimeBinsPoolCohortParams().lookup_param_name(
                        pool_param_names=pool_param_names)})
        ResTimeBinsPoolCohort.populate(key)  # ensure upstream table populated
        return key

    def delete_(self, key, safemode=True):
        delete_(self, [XInterpPoolCohortEpsCohort], key, safemode)


@schema
class XInterpPoolCohortEpsCohort(CohortBase):
    definition = """
    # Placeholder for groups of entries from XInterpPoolCohort across epochs
    -> XInterpPoolCohortEpsCohortParams
    """

    class CohortEntries(dj.Part):
        definition = """
        # Placeholder for entries from XInterpPoolCohort
        -> XInterpPoolCohortEpsCohort
        -> XInterpPoolCohort
        """

    def make(self, key):
        # Note that structure of this pool table is such that it does not make sense to use insert_pool_cohort here

        # Insert into main table
        insert1_print(self, key)

        # Insert into parts table
        for k in (ResTimeBinsPoolCohortParams & key).get_cohort_params():  # add epoch to key in loop
            key.update(k)
            insert1_print(self.CohortEntries, {**key, **k})

    def fetch_dataframes(self, **kwargs):

        kwargs = add_defaults(kwargs, {"iterable_name": "epoch", "fetch_function_name": "fetch_dataframes",
                                       "axis": 0, "add_iterable": False},
                              add_nonexistent_keys=True, require_match=True)

        return super().fetch_dataframes(**kwargs)

    def get_covariate_group_map(self, shorthand_map=None):
        # Get covariate group map for each epoch / time bins pair in cohort, check the same across all, then
        # return one
        key = self.fetch1("KEY")
        covariate_group_maps = [
            (XInterpPoolCohort & {**key, **k}).get_covariate_group_map(shorthand_map)
            for k in ResTimeBinsPoolCohortParams().get_cohort_params(key)]
        return check_return_single_dict(covariate_group_maps)

    def delete_(self, key, safemode=True):
        from src.jguides_2024.glm.jguidera_el_net import ElNetSel
        delete_(self, [ElNetSel], key, safemode)


def populate_jguidera_measurements_interp_pool(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_measurements_interp_pool"
    upstream_schema_populate_fn_list = [populate_jguidera_ppt_interp, populate_jguidera_time_relative_to_well_event]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_measurements_interp_pool():
    from src.jguides_2024.glm.jguidera_el_net import drop_jguidera_el_net
    drop_jguidera_el_net()
    schema.drop()
