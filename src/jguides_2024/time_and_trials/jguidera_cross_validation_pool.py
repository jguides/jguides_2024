import datajoint as dj

from src.jguides_2024.datajoint_nwb_utils.datajoint_pool_table_base import PoolSelBase, PoolBase
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.time_and_trials.jguidera_kfold_cross_validation import KFoldTrainTestSplit
from src.jguides_2024.time_and_trials.jguidera_leave_one_out_condition_trials_cross_validation import LOOCTTrainTestSplit, \
    populate_jguidera_leave_one_out_condition_trials_cross_validation
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohort
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema

# Needed for table definitions:
ResTimeBinsPoolCohort
KFoldTrainTestSplit
LOOCTTrainTestSplit

schema = dj.schema("jguidera_cross_validation_pool")


@schema
class TrainTestSplitPoolSel(PoolSelBase):
    definition = """
    # Specifies entries from upstream tables for TrainTestSplitPool
    -> ResTimeBinsPoolCohort
    train_test_split_pool_param_name : varchar(200)
    ---
    source_table_name : varchar(80)
    source_params_dict : blob
    param_name_dict : blob
    int_id : int
    """

    @staticmethod
    def _get_valid_source_table_names():
        return ["KFoldTrainTestSplit", "LOOCTTrainTestSplit"]

    def cleanup(self, safemode=True):
        # Get keys that have no corresponding entry in upstream table
        bad_keys = []
        for key in self.fetch("KEY"):
            source_table_name, source_params_dict = (self & key).fetch1("source_table_name", "source_params_dict")
            if len(get_table(source_table_name) & {**key, **source_params_dict}) == 0:
                bad_keys.append(key)

        for key in bad_keys:
            self.delete_(key)


@schema
class TrainTestSplitPool(PoolBase):
    definition = """
    # Placeholder for cross validation train/test splits (optionally across epochs) across sources
    -> TrainTestSplitPoolSel
    ---
    part_table_name : varchar(80)
    """

    class KFoldTrainTestSplit(dj.Part):
        definition = """
        # Placeholder for entries from KFoldTrainTestSplit
        -> TrainTestSplitPool
        -> KFoldTrainTestSplit
        """

    class LOOCTTrainTestSplit(dj.Part):
        definition = """
        # Placeholder for entries from LOOCTTrainTestSplit
        -> TrainTestSplitPool
        -> LOOCTTrainTestSplit
        """


def populate_jguidera_cross_validation_pool(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_cross_validation_pool"
    upstream_schema_populate_fn_list = [
        populate_jguidera_leave_one_out_condition_trials_cross_validation]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)



def drop_jguidera_cross_validation_pool():
    schema.drop()
