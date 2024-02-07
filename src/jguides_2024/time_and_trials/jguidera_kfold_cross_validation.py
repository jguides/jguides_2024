import datajoint as dj

from src.jguides_2024.datajoint_nwb_utils.datajoint_cross_validation_table_helpers import \
    insert_cross_validation_table
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert, get_table_secondary_key_names
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohort
from src.jguides_2024.utils.cross_validation_helpers import CrossValidate

schema_name = "jguidera_kfold_cross_validation"
schema = dj.schema(schema_name)


@schema
class KFoldTrainTestSplitParams(SecKeyParamsBase):
    definition = """
    # Params for k fold cross validation
    k_fold_train_test_split_param_name : varchar(40)
    ---
    n_splits : int
    use_random_state = 0 : int
    random_state = -1 : int
    """

    # Override parent class method so can check params
    def insert1(self, key, **kwargs):
        # Require random_state to be -1 if use_random_state is 0 (False)
        if not key["use_random_state"] and key["random_state"] != -1:
            raise Exception(f"random_state must be -1 if use_random_state is 0 (False)")
        super().insert1(key, **kwargs)

    def _default_params(self):
        params = CrossValidate.get_default_cross_validation_params("kfold")
        return [[params[k] for k in get_table_secondary_key_names(self)]]

    def fetch1_params(self):
        params = {k: self.fetch1(k) for k in ["n_splits", "use_random_state", "random_state"]}
        # Before returning params, set random state to None if use_random_state is zero (False)
        # This is helpful for use in downstream analyses that expect random state to be indicated by None when
        # random state not used
        if params["use_random_state"] == 0:
            params["random_state"] = None
        return params


@schema
class KFoldTrainTestSplitSel(SelBase):
    definition = """
    # Selection from upstream tables for KFoldTrainTestSplit
    -> KFoldTrainTestSplitParams
    -> ResTimeBinsPoolCohort
    """

    def insert_defaults(self, **kwargs):
        self.insert_test()


@schema
class KFoldTrainTestSplit(ComputedBase):
    definition = """
    # Train and test indices for k fold cross validation
    -> KFoldTrainTestSplitSel
    ---
    -> nd.common.AnalysisNwbfile
    train_set_df_object_id : varchar(40)
    test_set_df_object_id : varchar(40)
    train_test_set_df_object_id : varchar(40)
    time_bins_df_object_id : varchar(40)
    """

    def make(self, key):
        # Get cross validation params
        params = (KFoldTrainTestSplitParams & key).fetch1_params()

        # Get time bin information across cohort
        time_bins_df = (ResTimeBinsPoolCohort & key).fetch_dataframes()

        # Add time bins across cohort to params
        params["data_vector"] = time_bins_df["time_bin_centers"]

        # Insert into table
        insert_cross_validation_table(self, key, "kfold", params, time_bins_df)


def populate_jguidera_kfold_cross_validation(key=None, tolerate_error=False):
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_kfold_cross_validation():
    schema.drop()
