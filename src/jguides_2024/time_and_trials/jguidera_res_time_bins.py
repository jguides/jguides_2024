import datajoint as dj
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, delete_
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellArrivalTrialsParams, DioWellADTrialsParams
from src.jguides_2024.time_and_trials.jguidera_res_set import ResSet, populate_jguidera_res_set, ResSetParams
from src.jguides_2024.time_and_trials.jguidera_time_bins import (EpochTimeBins, DioWATrialsTimeBins,
                                                                 populate_jguidera_time_bins,
                                                                 EpochTimeBinsParams, DioWATrialsTimeBinsParams,
                                                                 DioWellADTrialsTimeBins,
                                                                 DioWellADTrialsTimeBinsParams)
from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPoolCohortParams

# Needed for table definitions:
nd
DioWATrialsTimeBinsParams


schema_name = "jguidera_res_time_bins"
schema = dj.schema(schema_name)


def insert_res_time_bins_table(table, parent_table, key):
    # Insert into a restricted time bins table
    # Get centers and edges of time bins
    time_bins_df = (parent_table & key).fetch1_dataframe()
    time_bin_centers = time_bins_df["time_bin_centers"]
    time_bin_edges = time_bins_df["time_bin_edges"]
    # Apply restriction
    valid_time_bin_centers, valid_bool = ResSet().apply_restriction(
        key, time_bin_centers, time_bin_edges)
    res_time_bins_df = pd.DataFrame.from_dict({"time_bin_centers": valid_time_bin_centers,
                                               "time_bin_edges": time_bin_edges[valid_bool]})
    # Insert into table
    insert_analysis_table_entry(table, [res_time_bins_df], key)


@schema
class ResEpochTimeBinsSel(SelBase):
    definition = """
    # Selection from upstream tables for ResEpochTimeBins
    -> EpochTimeBins
    -> ResSet
    """

    # TODO (feature): restrict entries


@schema
class ResEpochTimeBins(ComputedBase):
    definition = """
    # Time bins within epoch with restrictions applied
    -> ResEpochTimeBinsSel
    ---
    -> nd.common.AnalysisNwbfile
    res_epoch_time_bins_object_id : varchar(40)
    """

    # Override parent class method since non-canonical naming leads to params table not found
    # with parent class code to find params table
    def _get_params_table(self):
        return EpochTimeBinsParams

    def make(self, key):
        insert_res_time_bins_table(self, EpochTimeBins, key)

    def delete_(self, key, safemode=True):
        from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPool
        delete_(self, [ResTimeBinsPool], key, safemode)


# Notes about ResDioWATrialsTimeBins tables setup:
# - Want restriction captured by dio_well_arrival_trials_param_name to be in res_set_param_name. Currently done in
# selection table insert_defaults method
# - Since dio_well_arrival_trials_param_name captured by res_set_param_name, do not include in primary key
# to reduce number of primary keys and improve readability. This requires dependence on DioWATrialsTimeBins
# in part table instead of selection table
# - Have dio_well_arrival_trials_param_name as secondary key in selection table so can easily access in main
# table, where helps with querying DioWellArrivalTrials
# - Order of dependent tables in ResDioWATrialsTimeBinsSel consistent with other restricted time bins tables so that
# trials pool cohort param names similar across time bin sources
# TODO (feature): alter table since fixed typo in table description (replaced epoch time bins table with "ResDioWATrialsTimeBins")
@schema
class ResDioWATrialsTimeBinsSel(SelBase):
    definition = """
    # Selection from upstream tables for ResDioWATrialsTimeBins
    -> DioWATrialsTimeBinsParams
    -> ResSet
    ---
    dio_well_arrival_trials_param_name : varchar(40)  # for convenience
    """

    # Override parent class method so can ensure restriction time period consistent across ResSet and
    # DioWATrialsTimeBins
    def insert_defaults(self, **kwargs):
        # Get entries in trials pool table with matching trials param name

        source_table_name = "DioWellArrivalTrials"
        dio_well_arrival_trials_param_names = [
            # for time relative to well arrival aligned firing rate vector analysis:
            DioWellArrivalTrialsParams().lookup_delay_param_name(),  # delay (0 to 2s after well arrival)
        ]

        for dio_well_arrival_trials_param_name in dio_well_arrival_trials_param_names:
            key = {"dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name}
            source_table_keys = [key]
            trials_pool_cohort_param_name = TrialsPoolCohortParams().lookup_param_name(
                [source_table_name], source_table_keys)
            key.update(
                {"res_set_param_name": ResSetParams().lookup_no_combination_param_name(trials_pool_cohort_param_name)})
            for insert_key in (ResSet * DioWATrialsTimeBins & key).fetch("KEY"):
                self.insert1(insert_key)


@schema
class ResDioWATrialsTimeBins(ComputedBase):
    definition = """
    # Time bins during trials based on single well arrival detected with dios, with restrictions applied
    -> ResDioWATrialsTimeBinsSel
    ---
    -> nd.common.AnalysisNwbfile
    res_dio_well_arrival_trials_time_bins_object_id : varchar(40)
    dio_well_arrival_trials_param_name : varchar(40)  # for convenience when inserting into table
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> ResDioWATrialsTimeBins
        -> DioWATrialsTimeBins
        """

    # Override parent class method since noncanonical params table name
    def _get_params_table(self):
        return DioWATrialsTimeBinsParams

    def make(self, key):
        key.update({"dio_well_arrival_trials_param_name": (ResDioWATrialsTimeBinsSel & key).fetch1(
            "dio_well_arrival_trials_param_name")})
        insert_res_time_bins_table(self, DioWATrialsTimeBins, key)


# Notes about ResDioWellADTrialsTimeBins tables setup:
# - Want restriction captured by dio_well_ad_trials_param_name to be in res_set_param_name. Currently done in
# selection table insert_defaults method
# - Since dio_well_ad_trials_param_name captured by res_set_param_name, do not include in primary key
# to reduce number of primary keys and improve readability. This requires dependence on DioWellADTrialsTimeBins
# in part table instead of selection table
# - Have dio_well_ad_trials_param_name as secondary key in selection table so can easily access in main
# table, where helps with querying DioWellADTrials
# - Order of dependent tables in ResDioWellADTrialsTimeBinsSel consistent with other restricted time bins tables so that
# trials pool cohort param names similar across time bin sources
@schema
class ResDioWellADTrialsTimeBinsSel(SelBase):
    definition = """
    # Selection from upstream tables for ResDioWellADTrialsTimeBins
    -> DioWellADTrialsTimeBinsParams
    -> ResSet
    ---
    dio_well_ad_trials_param_name : varchar(40)  # for convenience
    """

    # Override parent class method so can ensure restriction time period consistent across ResSet and
    # DioWellADTrialsTimeBins
    def insert_defaults(self, **kwargs):
        # Get entries in trials pool table with matching trials param name

        source_table_name = "DioWellADTrials"
        dio_well_ad_trials_param_names = [
            DioWellADTrialsParams().lookup_post_delay_param_name()
        ]

        for dio_well_ad_trials_param_name in dio_well_ad_trials_param_names:
            key = {"dio_well_ad_trials_param_name": dio_well_ad_trials_param_name}
            source_table_keys = [key]
            trials_pool_cohort_param_name = TrialsPoolCohortParams().lookup_param_name(
                [source_table_name], source_table_keys)
            key.update(
                {"res_set_param_name": ResSetParams().lookup_no_combination_param_name(trials_pool_cohort_param_name)})
            for insert_key in (ResSet * DioWellADTrialsTimeBins & key).fetch("KEY"):
                self.insert1(insert_key)


@schema
class ResDioWellADTrialsTimeBins(ComputedBase):
    definition = """
    # Time bins during trials that begin at well arrivals and end at well departure detected with dios, with restrictions applied
    -> ResDioWellADTrialsTimeBinsSel
    ---
    -> nd.common.AnalysisNwbfile
    res_dio_well_ad_trials_time_bins_object_id : varchar(40)
    dio_well_ad_trials_param_name : varchar(40)  # for convenience when inserting into table
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> ResDioWellADTrialsTimeBins
        -> DioWellADTrialsTimeBins
        """

    # Override parent class method since noncanonical params table name
    def _get_params_table(self):
        return DioWellADTrialsTimeBinsParams

    def make(self, key):
        key.update({"dio_well_ad_trials_param_name": (ResDioWellADTrialsTimeBinsSel & key).fetch1(
            "dio_well_ad_trials_param_name")})
        insert_res_time_bins_table(self, DioWellADTrialsTimeBins, key)


"""
# Script to check that restricted time bins and time bins are equivalent for well arrival case (for a test case)
key = ResDioWATrialsTimeBins().fetch("KEY")[0]
dio_well_arrival_trials_param_name = (ResDioWATrialsTimeBins & key).fetch1("dio_well_arrival_trials_param_name")
key.update({"dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name})
df = (DioWATrialsTimeBins & key).fetch1_dataframe()
res_df = (ResDioWATrialsTimeBins & key).fetch1_dataframe()
for attribute in ["time_bin_centers"]:
    if not (getattr(df, attribute) == getattr(res_df, attribute)).all():  
        raise Exception(f"{attribute} not same across DioWATrialsTimeBins and ResDioWATrialsTimeBins")
if not (np.vstack(df.time_bin_edges) == np.vstack(res_df.time_bin_edges)).all():  # error results with all function
    raise Exception(f"time bin edges not same across DioWATrialsTimeBins and ResDioWATrialsTimeBins")
"""


def populate_jguidera_res_time_bins(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_res_time_bins"
    upstream_schema_populate_fn_list = [populate_jguidera_time_bins, populate_jguidera_res_set]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_res_time_bins():
    from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import drop_jguidera_res_time_bins_pool
    drop_jguidera_res_time_bins_pool()
    schema.drop()
