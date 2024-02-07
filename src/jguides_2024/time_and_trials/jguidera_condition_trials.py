import copy

import datajoint as dj

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, \
    get_table_column_names, delete_
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_epoch_keys
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohort, \
    ResTimeBinsPoolCohortParams, ResTimeBinsPoolSel
from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPoolEpsCohort, TrialsPool
from src.jguides_2024.utils.df_helpers import zip_df_columns, df_filter_columns
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals as in_intervals

schema = dj.schema("jguidera_condition_trials")


@schema
class ConditionTrialsParams(SecKeyParamsBase):
    definition = """
    # Params for ConditionTrials
    condition_trials_param_name : varchar(80)
    ---
    condition_name : varchar(40)  # name of condition used to group trials
    """

    def _default_params(self):
        params = ["path_names", "well_names"]
        return [[x] for x in params]


@schema
class ConditionTrialsSel(SelBase):
    definition = """
    # Selection from upstream tables for ConditionTrials
    -> ResTimeBinsPoolCohort  # supplies time bins
    -> TrialsPoolEpsCohort  # supplies trials and associated conditions
    -> ConditionTrialsParams  # defines condition
    """

    def insert1(self, key, **kwargs):
        # Ensure that pairing of condition_trials_param_name and trials_pool_param_name valid: the condition name
        # for the given condition_trials_param_name must exist in the trials table for the given trials_pool_param_name
        ConditionTrialsParams().insert_defaults()
        condition_name = (ConditionTrialsParams & key).fetch1("condition_name")
        df_columns = (TrialsPoolEpsCohort & key).fetch_dataframes().columns
        if condition_name not in df_columns:
            raise Exception(
                f"The condition name {condition_name} is not valid for the current trial type. Valid conditions are:"
                f" {df_columns}")
        super().insert1(key, **kwargs)

    @staticmethod
    def get_valid_condition_names(key):
        # Get valid condition names for a given key. These are columns in the trials table specified
        # by trials_pool_param_name in the key
        part_table = eval(check_return_single_element((TrialsPool & key).fetch("part_table_name")).single_element)
        source_table = eval(check_return_single_element((part_table & key).fetch("source_table_name")).single_element)
        return get_table_column_names(source_table)

    def _get_potential_keys(self, key_filter=None):
        # Currently, limit population to path traversals and delay period for single epochs, and GLM analysis params

        # Get nwb files / epochs (limit to highest priority)
        highest_priority = True
        nwbf_epoch_keys = get_jguidera_nwbf_epoch_keys(highest_priority=highest_priority)
        nwbf_epoch_keys = [{k.replace("epoch", "epochs_id"): str(v) for k, v in key.items()}
                           for key in nwbf_epoch_keys]  # convert epoch to epochs id and render as string

        # Loop through path / delay period using GLM params
        keys = []
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_params
        for shorthand_param_name, condition_name in zip(
                get_glm_default_params(["path_time_bins_shorthand", "delay_time_bins_shorthand"], as_dict=False),
                                        ["path_names", "well_names"]):
            trials_pool_param_name = ResTimeBinsPoolSel().lookup_trials_pool_param_name_from_shorthand(
                shorthand_param_name)
            res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand(
                shorthand_param_name)
            # Loop through nwb file / epoch keys and populate
            for key in nwbf_epoch_keys:
                key.update({"condition_trials_param_name":
                                ConditionTrialsParams()._param_name_from_secondary_key_values(
                                    [condition_name], param_name_subset=False),
                            "trials_pool_param_name": trials_pool_param_name,
                            "res_time_bins_pool_cohort_param_name": res_time_bins_pool_cohort_param_name})
                # Insert if entry exists in upstream tables to allow population with this entry
                if len((ResTimeBinsPoolCohort * TrialsPoolEpsCohort) & key) > 0:
                    keys.append(copy.deepcopy(key))
        return keys

    def delete_(self, key, safemode=True):
        from src.jguides_2024.time_and_trials.jguidera_cross_validation_pool import TrainTestSplitPool
        delete_(self, [TrainTestSplitPool], key, safemode)


@schema
class ConditionTrials(ComputedBase):
    definition = """
    # Map from start/stop indices of trials to value of condition associated with those trials
    -> ConditionTrialsSel
    ---
    condition_trials_map : blob  # map from condition values to trial start/stop indices of trials
    epoch_map : blob  # map from trial start/stop indices to epochs (for convenience)
    """

    def make(self, key):
        # Get trials information across epochs
        trials_df = (TrialsPoolEpsCohort & key).fetch_dataframes()

        # Get time bins across epochs
        time_bins_df = (ResTimeBinsPoolCohort & key).fetch_dataframes()

        # Get name of condition
        condition_name = (ConditionTrialsParams & key).fetch1("condition_name")

        # Make map from "trial indices" (indices of first and last time bin centers within trial)
        # to "condition values" (values that a given condition can take on)
        # Approach: For each condition value, find trials with that condition value.
        # IMPORTANT NOTE: each trial is associated with ONE condition value; e.g. each trial in an entry in
        # DioWellDepartureTrials is associated with one well name. This allows looking at periods of time
        # around events associated with a condition value (e.g. departures from the left well). Note that
        # this condition value is used for ALL times in the trial.
        unique_condition_values = set(trials_df[condition_name])  # get set of condition values
        condition_trials_map = {condition_value: [] for condition_value in unique_condition_values}
        epoch_map = copy.deepcopy(condition_trials_map)
        for condition_value in unique_condition_values:
            # Get trial df entries for trials that had this condition value
            df_subset = df_filter_columns(trials_df, {condition_name: condition_value})
            # Loop through trials and find first and last times in time bin centers within the trial. Store
            # corresponding indices
            for trial_start_time, trial_end_time, epoch in zip_df_columns(
                    df_subset, ["trial_start_times", "trial_end_times", "epoch"]):
                valid_idxs = in_intervals(time_bins_df["time_bin_centers"], [[trial_start_time, trial_end_time]])[0]
                # Store first and last index of time bin centers within trial
                condition_trials_map[condition_value].append((valid_idxs[0], valid_idxs[-1]))
                # Store epoch
                epoch_map[condition_value].append(epoch)

        # Insert into table
        key.update({"epoch_map": epoch_map, "condition_trials_map": condition_trials_map})
        insert1_print(self, key)


def populate_jguidera_condition_trials(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_condition_trials"
    upstream_schema_populate_fn_list = []
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_condition_trials():
    from src.jguides_2024.time_and_trials.jguidera_leave_one_out_condition_trials_cross_validation import \
        drop_jguidera_leave_one_out_condition_trials_cross_validation
    drop_jguidera_leave_one_out_condition_trials_cross_validation()
    schema.drop()

