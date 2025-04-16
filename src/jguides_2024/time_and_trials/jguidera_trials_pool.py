import datajoint as dj

from src.jguides_2024.datajoint_nwb_utils.datajoint_pool_table_base import PoolSelBase, PoolBase, \
    PoolCohortParamsBase, PoolCohortBase, \
    PoolCohortParamNameBase, EpsCohortParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import (CohortBase)
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, check_single_table_entry, \
    get_cohort_test_entry, get_table_name, \
    delete_, fetch_entries_as_dict, get_epochs_id
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.task_event.jguidera_dio_trials import (DioWellArrivalTrialsParams,
                                                             DioWellDATrialsParams, DioWellArrivalTrials,
                                                             DioWellDATrials,
                                                             populate_jguidera_dio_trials, DioWellADTrials,
                                                             DioWellADTrialsParams,
                                                             DioWellArrivalTrialsSubParams,
                                                             DioWellArrivalTrialsSub)
from src.jguides_2024.time_and_trials.jguidera_epoch_interval import EpochInterval, \
    populate_jguidera_epoch_interval
from src.jguides_2024.time_and_trials.jguidera_ppt_trials import populate_jguidera_ppt_trials
from src.jguides_2024.utils.dict_helpers import add_defaults
from src.jguides_2024.utils.vector_helpers import unpack_single_element

# Need for table definitions
TaskIdentification
DioWellArrivalTrials
DioWellArrivalTrialsSub
DioWellDATrials
DioWellADTrials
EpochInterval

schema = dj.schema("jguidera_trials_pool")


@schema
class TrialsPoolSel(PoolSelBase):
    definition = """
    # Specifies entries from upstream tables for TrialsPool
    -> TaskIdentification
    trials_pool_param_name : varchar(500)
    ---
    source_table_name : varchar(80)
    source_params_dict : blob
    param_name_dict : blob
    int_id : int
    """

    @staticmethod
    def _get_valid_source_table_names():
        # Valid source trials table names
        return [
            "DioWellArrivalTrials", "DioWellArrivalTrialsSub", "DioWellDATrials", "EpochInterval", "DioWellADTrials"]

    def delete_(self, key, safemode=True):
        # Delete from upstream tables, selection table, and current table
        delete_(self, [TrialsPool], key, safemode)


@schema
class TrialsPool(PoolBase):
    definition = """
    # Placeholder for trials across sources
    -> TrialsPoolSel
    ---
    part_table_name : varchar(80)
    """

    class DioWellArrivalTrials(dj.Part):
        definition = """
        # Placeholder for entries from DioWellArrivalTrials
        -> TrialsPool
        -> DioWellArrivalTrials
        """

    class DioWellArrivalTrialsSub(dj.Part):
        definition = """
        # Placeholder for entries from DioWellArrivalTrialsSub
        -> TrialsPool
        -> DioWellArrivalTrialsSub
        """

    class DioWellDATrials(dj.Part):
        definition = """
        # Placeholder for entries from DioWellDATrials
        -> TrialsPool
        -> DioWellDATrials
        """

    class DioWellADTrials(dj.Part):
        definition = """
        # Placeholder for entries from DioWellADTrials
        -> TrialsPool
        -> DioWellADTrials
        """

    class EpochInterval(dj.Part):
        definition = """
        # Placeholder for entries from EpochInterval
        -> TrialsPool
        -> EpochInterval
        """

    def trial_intervals(self):
        entry = self.fetch1_part_entry()
        return list(zip(entry["trial_start_times"], entry["trial_end_times"]))

    def delete_(self, key, safemode=True):
        # Delete from upstream tables and current table
        delete_(self, [TrialsPoolCohortParams, TrialsPoolEpsCohortParams], key, safemode)


# The following tables group trials WITHIN epochs
@schema
class TrialsPoolCohortParamName(PoolCohortParamNameBase):
    definition = """
    # Map between long and optionally short param name
    trials_pool_cohort_param_name : varchar(100)
    ---
    int_id = NULL : int
    full_param_name : varchar(1000)
    """


@schema
class TrialsPoolCohortParams(PoolCohortParamsBase):
    definition = """
    # Specifies groups of entries from TrialsPoolCohort in an epoch
    -> TrialsPoolCohortParamName
    -> TaskIdentification
    ---
    trials_pool_param_names : blob
    num_trials_pool_param_names : int  # for convenience
    """

    def insert_single_member_cohort_defaults(self, **kwargs):
        # Populate cohort table with single member cohort entries

        # Restrict population with these params
        dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_delay_param_name()
        dio_well_arrival_trials_sub_param_name = DioWellArrivalTrialsSubParams().lookup_param_name(["stay"])
        dio_well_da_trials_param_name = DioWellDATrialsParams().lookup_no_shift_param_name()
        param_name_dicts = [
                # 2s DELAY PERIOD
                {"source_table_name": "DioWellArrivalTrials",
                 'dio_well_arrival_trials_param_name': dio_well_arrival_trials_param_name},
                # 2s DELAY PERIOD, SUBSET WHERE RAT STAYED AT WELL FOR FULL 2s
                {"source_table_name": "DioWellArrivalTrialsSub",
                 'dio_well_arrival_trials_sub_param_name': dio_well_arrival_trials_sub_param_name},
                # TIME AROUND WELL ARRIVAL
                {"source_table_name": "DioWellArrivalTrials",
                 'dio_well_arrival_trials_param_name': DioWellArrivalTrialsParams().lookup_param_name([-1, 3])},
                # DELAY TO WELL DEPARTURE
                {"source_table_name": "DioWellADTrials",
                 "dio_well_ad_trials_param_name": DioWellADTrialsParams().lookup_post_delay_param_name()},
                # PATH TRAVERSAL
                {"source_table_name": "DioWellDATrials",
                 "dio_well_da_trials_param_name": dio_well_da_trials_param_name},
                # ENTIRE EPOCH
                {"source_table_name": "EpochInterval"}]
        for param_name_dict in param_name_dicts:
            self.insert_single_member_cohort(param_name_dict, **kwargs)

    def insert_test(self):
        # Insert an entry with two upstream trial entries
        num_epochs = 2
        test_entry_obj = get_cohort_test_entry(
            TrialsPool, col_vary=unpack_single_element(
                self._get_param_name_iterables(singular=True)), num_entries=num_epochs)
        # Insert if found valid set
        if test_entry_obj is not None:
            self._insert_from_upstream_param_names(
                secondary_key_subset_map={
                    unpack_single_element(self._get_param_name_iterables()): test_entry_obj.target_vals},
                                           key=test_entry_obj.same_col_vals_map, use_full_param_name=True)
        else:
            print(f"Could not insert test entry into {get_table_name(self)}")

    def insert_defaults(self, **kwargs):
        self.insert_single_member_cohort_defaults(**kwargs)

    def delete_(self, key, safemode=True):
        # If trials_pool_param_name in key but trials_pool_param_names not in key, find all trials_pool_param_names
        # corresponding to trials_pool_param_name, and loop through these and clear corresponding entries
        keys = [key]  # default
        if "trials_pool_param_names" not in key and "trials_pool_param_name" in key:
            keys = [
                k for k in fetch_entries_as_dict(self & key) if key["trials_pool_param_name"]
                                                                in k["trials_pool_param_names"]]
        for key in keys:
            delete_(self, [TrialsPoolCohort], key, safemode)


# Purpose is to use to group trials that we then intersect to make restrictions
# within epochs. We use these restrictions to get spikes at a particular period in an epoch, etc.
@schema
class TrialsPoolCohort(PoolCohortBase):
    definition = """
    # Groups of entries from TrialsPool in an epoch
    -> TrialsPoolCohortParams
    """

    class CohortEntries(dj.Part):
        definition = """
        # Entries from TrialsPool
        -> TrialsPoolCohort
        -> TrialsPool
        """

    def get_cohort_trial_intervals(self, key):
        # Return trial intervals for each set of trials in a cohort
        # Data for cohort entries lives in TrialsPool. To access, we find the trials pool param name
        # for each cohort member, then use to query TrialsPool
        pool_param_names = self.get_upstream_pool_table_param_names(key)
        # Get the trial start and end times for each trial set and convert to list of
        # trial start/end time tuples
        return {pool_param_name:
                (TrialsPool & {**key, **{"trials_pool_param_name": pool_param_name}}).trial_intervals()
                for pool_param_name in pool_param_names}

    def delete_(self, key, safemode=True):
        # Delete from upstream tables and selection table
        from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPool
        delete_(self, [ResTimeBinsPool], key, safemode)


# The following tables group trials ACROSS epochs
@schema
class TrialsPoolEpsCohortParams(EpsCohortParamsBase):
    definition = """
    # Specifies groups of entries from TrialsPool across epochs
    nwb_file_name : varchar(40)
    trials_pool_param_name : varchar(500)
    epochs_id : varchar(40)
    ---
    epochs : blob
    """

    @staticmethod
    def _upstream_table():
        return TrialsPool

    def cleanup(self):
        # Delete entries in params table that have no corresponding entry in TrialsPool
        for key in self.fetch("KEY"):
            epochs = (self & key).fetch1("epochs")
            continue_flag = False  # initialize flag for continuing once an entry has been deleted
            for epoch in epochs:
                if len(TrialsPool & {**key, **{"epoch": epoch}}) == 0:
                    print(f"deleting {self & key}")
                    (self & key).delete()
                    continue_flag = True  # dont need to check other epochs now
                if continue_flag:
                    continue

    def delete_(self, key, safemode=True):
        # Add epochs_id if epoch in key but epochs_id not
        if "epoch" in key and "epochs_id" not in key:
            key["epochs_id"] = get_epochs_id([key["epoch"]])
        delete_(self, [TrialsPoolEpsCohort], key, safemode)


# Purpose is to use to group same type trials across epochs
@schema
class TrialsPoolEpsCohort(CohortBase):
    definition = """
    # Groups of entries from TrialsPool across epochs
    -> TrialsPoolEpsCohortParams
    """

    class CohortEntries(dj.Part):
        definition = """
        # Entries from TrialsPool
        -> TrialsPoolEpsCohort
        -> TrialsPool
        """

    def make(self, key):
        # Note that structure of this pool table is such that it does not make sense to use insert_pool_cohort here

        # Insert into main table
        insert1_print(self, key)

        # Insert into parts table
        epochs = (TrialsPoolEpsCohortParams & key).fetch1("epochs")
        for epoch in epochs:  # add epoch to key in loop
            key.update({"epoch": epoch})
            insert1_print(self.CohortEntries, key)

    def get_cohort_epochs(self, key):
        # Check that passed key specifies a single cohort
        check_single_table_entry(self, key)
        # Return epochs for this cohort
        return (self.CohortEntries & key).fetch("epoch")

    def fetch_entries(self):
        # Fetch entries for a cohort
        return self._fetch("epoch", "fetch1_part_entry")

    def fetch_dataframes(self, **kwargs):
        kwargs = add_defaults(kwargs, {"iterable_name": "epoch"}, add_nonexistent_keys=True, require_match=True)
        return super().fetch_dataframes(**kwargs)

    def get_cohort_trial_intervals(self, key):
        # Return trial intervals for each set of trials in a cohort
        # Data for cohort entries lives in TrialsPool. To access, we find the epoch
        # for each cohort member, then use to query TrialsPool
        epochs = self.get_cohort_epochs(key)
        # Get the trial start and end times for each trial set and convert to list of
        # trial start/end time tuples. Return in dictionary with epoch as key
        return {epoch: (TrialsPool & {**key, **{"epoch": epoch}}).trial_intervals() for epoch in epochs}

    def delete_(self, key, safemode=True):
        from src.jguides_2024.time_and_trials.jguidera_condition_trials import ConditionTrialsSel
        delete_(self, [ConditionTrialsSel], key, safemode)


# Appears can take a long time to run (but eventually completes)
def populate_jguidera_trials_pool(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_trials_pool"
    upstream_schema_populate_fn_list = [
        populate_jguidera_dio_trials, populate_jguidera_ppt_trials, populate_jguidera_epoch_interval]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_trials_pool():
    from src.jguides_2024.time_and_trials.jguidera_res_set import drop_jguidera_res_set
    drop_jguidera_res_set()
    schema.drop()
