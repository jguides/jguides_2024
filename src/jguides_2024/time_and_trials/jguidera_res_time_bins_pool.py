from collections import namedtuple

import datajoint as dj
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_pool_table_base import lookup_pool_table_param_name, \
    PoolSelBase, PoolBase, \
    PoolCohortParamsBase, PoolCohortBase, PoolCohortParamNameBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import PartBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_cohort_test_entry, get_table_name, \
    get_key, get_key_filter, \
    fetch_iterable_array, delete_, insert1_print
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_epochs_id, check_epochs_id
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_delay_interval
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDATrialsParams, DioWellArrivalTrialsParams, \
    DioWellADTrialsParams, \
    DioWellArrivalTrialsSubParams
from src.jguides_2024.time_and_trials.jguidera_res_set import ResSetParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins import ResEpochTimeBins, \
    populate_jguidera_res_time_bins, \
    ResDioWATrialsTimeBins, ResDioWellADTrialsTimeBins
from src.jguides_2024.time_and_trials.jguidera_time_bins import EpochTimeBinsParams
from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPoolCohortParams
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.dict_helpers import dict_list, add_defaults
from src.jguides_2024.utils.interval_helpers import check_intervals_list
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.string_helpers import strip_string
from src.jguides_2024.utils.vector_helpers import check_all_unique

# Needed for table definitions:
TaskIdentification
ResEpochTimeBins
ResDioWATrialsTimeBins
ResDioWellADTrialsTimeBins

schema_name = "jguidera_res_time_bins_pool"
schema = dj.schema(schema_name)

"""
The primary purpose of the following tables is to be able to pool time bins across multiple epochs,
and the code is flexible to allow for combination of different types of time bins.
We first compile time bins of different types in one place using a pool table.
We then specify groupings of these time bins using a cohort table.
"""


@schema
class ResTimeBinsPoolSel(PoolSelBase):
    definition = """
    # Specifies entries from upstream tables for ResTimeBinsPool
    -> TaskIdentification
    res_time_bins_pool_param_name : varchar(1000)
    ---
    source_table_name : varchar(80)
    source_params_dict : blob
    param_name_dict : blob
    int_id : int
    """

    @staticmethod
    def _get_valid_source_table_names():
        return [
            "ResEpochTimeBins", "ResDioWATrialsTimeBins", "ResDioWellADTrialsTimeBins"]

    def lookup_param_name(self, source_time_bins_table_name, key_search_params=None,
                          source_time_bins_table_key=None, tolerate_no_entry=False):
        # We need name of table with time bins and key to that table to look up param name
        # Two methods of lookup:
        # 1) use source_time_bins_table_key if passed
        # 2) Otherwise, get source_time_bins_table_key using key_search_params

        # Check that either key (source_time_bins_table_key) OR params for getting key (key_search_params) passed
        check_one_none([source_time_bins_table_key, key_search_params])

        # Get key to table with time bins if not passed
        if source_time_bins_table_key is None:
            # Params needed in key_search_params:
            # 1) source_trials_table_names: names of tables supplying trials to form restriction
            # 2) source_trials_table_key: key to the above tables to get trials to form restriction
            # 3) combination_params: specifies how to "combine" (e.g. intersect? union?) above intervals to get
            #    restriction
            # 4) time_bins_table_beyond_res_params: parts of time bins table key beyond specifying restriction
            # The existence of these params is checked when get_pool_table_param_name_quantities is eventually called

            # Get trials_pool_cohort_param_name: refers to group of trials combined to form restriction
            trials_pool_cohort_param_name = TrialsPoolCohortParams().lookup_param_name(
                key_search_params["source_trials_table_names"], key_search_params["source_trials_table_keys"],
                tolerate_no_entry)

            # Get res_set_param_name: adds to trials_pool_cohort_param_name information about how to
            # combine the above intervals
            res_set_param_name = ResSetParams().lookup_param_name(trials_pool_cohort_param_name,
                key_search_params["combination_params"], tolerate_no_entry)

            # If either param name was not found, return None (note that only can occur if tolerating no entry)
            if trials_pool_cohort_param_name is None or res_set_param_name is None:
                return None

            # Otherwise, make key to source table with time bins with restriction applied (e.g. ResEpochTimeBins)
            source_time_bins_table_key = {**key_search_params["time_bins_table_beyond_res_params"],
                **{"trials_pool_cohort_param_name": trials_pool_cohort_param_name,
                   "res_set_param_name": res_set_param_name}}

        # Look up param name
        return lookup_pool_table_param_name(
            self, source_time_bins_table_name, source_time_bins_table_key, tolerate_no_entry)

    def get_time_bin_width(self, key=None):
        if key is None:
            key = self.fetch1("KEY")
        source_params_table = (self & key).get_source_table()()._get_params_table()
        source_table_entry = (self & key).get_source_table_entry().fetch1("KEY")
        return (source_params_table & source_table_entry).get_time_bin_width()

    @staticmethod
    def valid_shorthands():
        return ["epoch_100ms", "path_100ms", "path_20ms", "delay_100ms", "delay_20ms", "wa1_100ms",
                "delay_stay_100ms", "delay_stay_20ms", "post_delay_100ms"]

    @staticmethod
    def get_shorthand_params_map():
        # Return map from shorthand for res_time_bins_pool_param_name to underlying params

        # Return params in objects
        shorthand_params_map = dict()
        ShorthandParams = namedtuple("ShorthandParams", "time_bin_width domain")
        delay_interval = get_delay_interval()
        shorthand_params_map.update({
            # DEFAULT PERIODS RELATIVE TO WELL ARRIVAL
            "delay_20ms": ShorthandParams(.02, delay_interval),
            "delay_100ms": ShorthandParams(.1, delay_interval),
            "delay_stay_20ms": ShorthandParams(.02, delay_interval),
            "delay_stay_100ms": ShorthandParams(.1, delay_interval),
             "wa1_100ms": ShorthandParams(.1, [-1, 3]),
            # POST DELAY: END OF DELAY TO WELL DEPARTURE
            "post_delay_100ms": ShorthandParams(.1, DioWellADTrialsParams()._post_delay_params())
        })

        return shorthand_params_map

    def lookup_param_name_from_shorthand(self, shorthand_param_name):

        def _default_epoch(time_bin_width):
            source_time_bins_table_name = "ResEpochTimeBins"
            epoch_time_bins_param_name = EpochTimeBinsParams().lookup_param_name([time_bin_width])
            key_search_params = {
                "time_bins_table_beyond_res_params": {"epoch_time_bins_param_name": epoch_time_bins_param_name},
                "source_trials_table_names": ["EpochInterval"],  # entire epoch
                "source_trials_table_keys": [{}],  # no params in EpochInterval
                "combination_params": ResSetParams.get_no_combination_params(),  # no combination
            }
            return source_time_bins_table_name, key_search_params

        def _default_path(time_bin_width):
            source_time_bins_table_name = "ResEpochTimeBins"
            epoch_time_bins_param_name = EpochTimeBinsParams().lookup_param_name([time_bin_width])
            key_search_params = {
                "time_bins_table_beyond_res_params": {"epoch_time_bins_param_name": epoch_time_bins_param_name},
                "source_trials_table_names": ["DioWellDATrials"],  # path traversals
                "source_trials_table_keys": [{"dio_well_da_trials_param_name":
                                                 DioWellDATrialsParams().lookup_no_shift_param_name()}],
                "combination_params": ResSetParams.get_no_combination_params(),  # no combination
            }
            return source_time_bins_table_name, key_search_params

        # DEFAULT ENTIRE EPOCH, 100ms TIME BINS
        if shorthand_param_name == "epoch_100ms":
            return self.lookup_param_name(*_default_epoch(.1))

        # DEFAULT PATH TRAVERSAL, 100ms TIME BINS
        elif shorthand_param_name == "path_100ms":
            return self.lookup_param_name(*_default_path(.1))

        # DEFAULT PATH TRAVERSAL, 20ms TIME BINS
        elif shorthand_param_name == "path_20ms":
            return self.lookup_param_name(*_default_path(.02))

        # DEFAULT PERIODS RELATIVE TO WELL ARRIVAL
        elif shorthand_param_name in ["delay_20ms", "delay_100ms", "wa1_100ms"]:
            shorthand_params = self.get_shorthand_params_map()[shorthand_param_name]
            source_time_bins_table_name = "ResEpochTimeBins"
            epoch_time_bins_param_name = EpochTimeBinsParams().lookup_param_name([shorthand_params.time_bin_width])
            key_search_params = {
                "time_bins_table_beyond_res_params": {
                    "epoch_time_bins_param_name": epoch_time_bins_param_name},  # describes time bin width
                "source_trials_table_names": ["DioWellArrivalTrials"],
                "source_trials_table_keys": [
                    {"dio_well_arrival_trials_param_name":
                         DioWellArrivalTrialsParams().lookup_param_name(shorthand_params.domain)}],
                "combination_params": ResSetParams.get_no_combination_params(),  # no combination
            }
            return self.lookup_param_name(source_time_bins_table_name, key_search_params)

        # DEFAULT STAY TRIALS DURING DELAY PERIOD
        elif shorthand_param_name in ["delay_stay_20ms", "delay_stay_100ms"]:
            shorthand_params = self.get_shorthand_params_map()[shorthand_param_name]
            source_time_bins_table_name = "ResEpochTimeBins"
            epoch_time_bins_param_name = EpochTimeBinsParams().lookup_param_name([shorthand_params.time_bin_width])
            key_search_params = {
                "time_bins_table_beyond_res_params": {
                    "epoch_time_bins_param_name": epoch_time_bins_param_name},  # describes time bin width
                "source_trials_table_names": ["DioWellArrivalTrialsSub"],
                "source_trials_table_keys": [
                    {"dio_well_arrival_trials_param_name":
                         DioWellArrivalTrialsParams().lookup_param_name(shorthand_params.domain),
                     "dio_well_arrival_trials_sub_param_name":
                         DioWellArrivalTrialsSubParams().lookup_param_name(["stay"])}],
                "combination_params": ResSetParams.get_no_combination_params(),  # no combination
            }
            return self.lookup_param_name(source_time_bins_table_name, key_search_params)

        # DEFAULT PERIOD POST DELAY TO WELL DEPARTURE
        elif shorthand_param_name in ["post_delay_100ms"]:
            shorthand_params = self.get_shorthand_params_map()[shorthand_param_name]
            source_time_bins_table_name = "ResEpochTimeBins"
            epoch_time_bins_param_name = EpochTimeBinsParams().lookup_param_name([shorthand_params.time_bin_width])
            key_search_params = {
                "time_bins_table_beyond_res_params": {
                    "epoch_time_bins_param_name": epoch_time_bins_param_name},  # describes time bin width
                "source_trials_table_names": ["DioWellADTrials"],
                "source_trials_table_keys": [
                    {"dio_well_ad_trials_param_name":
                         DioWellADTrialsParams().lookup_param_name(shorthand_params.domain)}],
                "combination_params": ResSetParams.get_no_combination_params(),  # no combination
            }
            return self.lookup_param_name(source_time_bins_table_name, key_search_params)

        else:
            raise Exception(f"{shorthand_param_name} not accounted for in code to look up param name in "
                            f"{get_table_name(self)} from shorthand")

    def get_res_set_param_name(self):
        return check_return_single_element(
            [k["res_set_param_name"] for k in self.fetch("param_name_dict")]).single_element

    def get_trials_pool_param_name(self):
        res_set_param_name = self.get_res_set_param_name()
        trials_pool_cohort_param_name = check_return_single_element(
            (ResSetParams & {"res_set_param_name": res_set_param_name}).fetch(
                "trials_pool_cohort_param_name")).single_element
        return check_return_single_element(np.concatenate(
            (TrialsPoolCohortParams & {"trials_pool_cohort_param_name": trials_pool_cohort_param_name}).fetch(
                "trials_pool_param_names"))).single_element

    def lookup_trials_pool_param_name_from_shorthand(self, shorthand_param_name):
        res_time_bins_pool_param_name = self.lookup_param_name_from_shorthand(shorthand_param_name)
        return (self & {"res_time_bins_pool_param_name": res_time_bins_pool_param_name}).get_trials_pool_param_name()


@schema
class ResTimeBinsPool(PoolBase):
    definition = """
    # Placeholder for restricted time bins within an epoch across sources
    -> ResTimeBinsPoolSel
    ---
    part_table_name : varchar(80)
    """

    class ResEpochTimeBins(dj.Part):
        definition = """
        # Placeholder for entries from ResEpochTimeBins
        -> ResTimeBinsPool
        -> ResEpochTimeBins
        """

    class ResDioWATrialsTimeBins(dj.Part):
        definition = """
        # Placeholder for entries from ResDioWATrialsTimeBins
        -> ResTimeBinsPool
        -> ResDioWATrialsTimeBins
        """

    class ResDioWellADTrialsTimeBins(dj.Part):
        definition = """
        # Placeholder for entries from ResDioWellADTrialsTimeBins
        -> ResTimeBinsPool
        -> ResDioWellADTrialsTimeBins
        """

    def delete_(self, key, safemode=True):
        # Delete from upstream tables
        from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVecSel
        delete_(self, [ResTimeBinsPoolCohortParams, FRVecSel], key, safemode)


@schema
class ResTimeBinsPoolCohortParamName(PoolCohortParamNameBase):
    definition = """
    # Map between full param name and integer used as param name
    res_time_bins_pool_cohort_param_name : varchar(100)  # reflects res_time_bins_pool_param_names composing cohort
    ---
    int_id = NULL : int
    full_param_name : varchar(1000)
    """

    # Override parent method so can make param name more interpretable. Drawback is uniqueness of param name
    # is not guaranteed, so user should be aware of meaning of param names
    def get_full_param_name(self, secondary_key_subset_map):
        # Return single pool param name if one unique one
        res_time_bins_pool_param_names = secondary_key_subset_map["res_time_bins_pool_param_names"]
        if len(np.unique(res_time_bins_pool_param_names)) == 1:
            return res_time_bins_pool_param_names[0]
        # Otherwise string together param names and replace some characters
        return super().get_full_param_name(secondary_key_subset_map)


# Combine time bins across epochs, with arbitrary time bin restriction params in epochs. To achieve,
# this table has two "free" parameters: epochs and res_time_bins_pool_param_names. We let epochs vary
# because we want to be able to combine time bins across epochs. We let res_time_bins_pool_param_names vary
# because we want the flexibility to combine epoch time bins that have different origins (e.g. in a sleep
# epoch, we might want to use a different restriction set than during an awake epoch)
# TODO (feature): if delete this table, change primary key to have dependence on EpochCohort
# TODO (feature): speed up insert defaults
@schema
class ResTimeBinsPoolCohortParams(PoolCohortParamsBase):
    definition = """
    # Specifies groups of entries from ResTimeBinsPool
    nwb_file_name : varchar(80)  # to require single nwb file per cohort 
    epochs_id : varchar(40)  # primary key so can require same epochs setting across cohort tables
    -> ResTimeBinsPoolCohortParamName
    ---
    epochs : blob
    res_time_bins_pool_param_names : blob
    """

    # Overrides method in parent class since want only restricted time bins pool param names to contribute to param name
    def _get_param_name_iterables(self, singular=False):
        return [x for x in super()._get_param_name_iterables(singular) if "param_name" in x]

    # Override method in parent class so can check key
    def insert1(self, key, **kwargs):
        # If only one epoch, check that entries in res_time_bins_pool_param_names all unique,
        # to prevent having duplicate time bins in a cohort
        if len(np.unique(key["epochs"])) == 1:
            check_all_unique(key["res_time_bins_pool_param_names"])
        # Check that epochs_id consistent with epochs in increasing order
        check_epochs_id(key["epochs_id"], key["epochs"])
        # Check same number of epochs and res_time_bins_pool_param_names
        if len(key["epochs"]) != len(key["res_time_bins_pool_param_names"]):
            raise Exception(f"Must have same number of epochs and res_time_bins_pool_param_names")
        # Run base class insert1 and forgo checking that iterables unique (they need not be here; we checked that
        # iterables valid above)
        super().insert1(key, check_unique_iterables=False)

    def insert_entry(self, nwb_file_name, epochs, res_time_bins_pool_param_names, use_full_param_name):
        # Sort epochs in increasing order. Use same order on res_time_bins_pool_param_names to maintain correspondence
        # to epochs
        sort_idxs = np.argsort(epochs)
        epochs = np.asarray(epochs)[sort_idxs]
        res_time_bins_pool_param_names = np.asarray(res_time_bins_pool_param_names)[sort_idxs]
        # Get epochs id: strung together epochs
        epochs_id = get_epochs_id(epochs)
        # Make param name for cohort
        secondary_key_subset_map = {"res_time_bins_pool_param_names": res_time_bins_pool_param_names}
        cohort_param_name = self._get_insert_cohort_param_name(secondary_key_subset_map, use_full_param_name)
        # Check corresponding entries in upstream tables (to avoid error when populating main table)
        key = {"res_time_bins_pool_cohort_param_name": cohort_param_name, "nwb_file_name": nwb_file_name,
               "epochs_id": epochs_id, "epochs": epochs, "res_time_bins_pool_param_names":
                   res_time_bins_pool_param_names}
        for res_time_bins_pool_param_name, epoch in zip(res_time_bins_pool_param_names, epochs):
            upstream_key = {**key, **{"res_time_bins_pool_param_name": res_time_bins_pool_param_name, "epoch": epoch}}
            upstream_key = {k: upstream_key[k] for k in ResTimeBinsPoolSel.primary_key}
            if len(ResTimeBinsPoolSel & upstream_key) == 0:
                raise Exception(f"cannot insert entry because no corresponding entries in upstream table")
        # Insert into table
        self.insert1(key)

    def insert_single_epochs(self, key_filter=None):
        # Insert single epoch entries from restricted time bins pool table
        # Note: this is slightly different from parent class method for inserting single member cohorts, since in this
        # particular table we have two iterables, epochs and res_time_bins_pool_param_names
        if key_filter is None:
            key_filter = dict()
        for key in (ResTimeBinsPool & key_filter).fetch("KEY"):
            self.insert_entry(
                key["nwb_file_name"], [key["epoch"]], [key["res_time_bins_pool_param_name"]], use_full_param_name=True)

    def insert_runs(self, key_filter=None):
        # Insert all run epoch entries from restricted time bins pool table for an nwb file
        # Also insert pairs of run epochs
        key_filter = get_key_filter(key_filter)

        # Define default restricted time bins pool param name, to use in event that was not passed in key_filter
        default_res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand("path_100ms")
        res_time_bins_pool_param_name = key_filter.pop("res_time_bins_pool_param_name",
                                                       default_res_time_bins_pool_param_name)

        for nwb_file_name, epochs in fetch_iterable_array(
                EpochsDescription & key_filter, ["nwb_file_name", "epochs"]):
            res_time_bins_pool_param_names = [res_time_bins_pool_param_name]*len(epochs)
            self.insert_entry(nwb_file_name, epochs, res_time_bins_pool_param_names, use_full_param_name=True)

    def insert_test(self):
        # Insert an "across epochs" entry for testing
        # Use for test, first two entries where there is same nwb_file_name and res_time_bins_pool_param_names
        num_epochs = 2
        test_entry_obj = get_cohort_test_entry(ResTimeBinsPool, col_vary="epoch", num_entries=num_epochs)
        # Insert if found valid set
        if test_entry_obj is not None:
            self.insert_entry(
                test_entry_obj.same_col_vals_map["nwb_file_name"], epochs=test_entry_obj.target_vals,
                res_time_bins_pool_param_names=[test_entry_obj.same_col_vals_map[
                                                    "res_time_bins_pool_param_name"]]*num_epochs,
                use_full_param_name=True)
        else:
            print(f"Could not insert test entry into {get_table_name(self)}")

    def insert_defaults(self, **kwargs):
        key_filter = kwargs.pop("key_filter", None)
        self.insert_single_epochs(key_filter)
        self.insert_test()

    def get_cohort_params(self, key=None):
        key = get_key(self, key)
        check_membership(self.primary_key, key, "primary keys of ResTimeBinsPoolCohortParams", "entries in passed key")
        return [{"epoch": epoch, "res_time_bins_pool_param_name": res_time_bins_pool_param_name} for
                epoch, res_time_bins_pool_param_name in zip(*(self & key).fetch1(
                "epochs", "res_time_bins_pool_param_names"))]

    def get_keys_with_cohort_params(self, key):
        return [{**key, **params} for params in self.get_cohort_params(key)]

    def get_time_bin_width(self, key=None):
        if key is None:
            key = self.fetch1("KEY")
        return check_return_single_element(
            [(ResTimeBinsPoolSel & key).get_time_bin_width() for key in
             self.get_keys_with_cohort_params(key)]).single_element

    @staticmethod
    def get_cohort_entry_lookup_args(source_time_bins_table_name, time_bins_table_beyond_res_params, source_trials_table_names,
                                     source_trials_table_keys, combination_params=None):
        """
        # Get arguments for looking up param name (via lookup_param_name) for a SINGLE entry in cohort.
        These can be passed directly to lookup_param_name to look up the param name for cohorts with one entry, or
        combined across calls to the current function to look up the param name for a cohort with multiple entries.
        # Params specifying time bins source and width:
        :param source_time_bins_table_name: string, name of table with time bins that restriction gets applied to, e.g.
                                            "ResEpochTimeBins"
        :param time_bins_table_beyond_res_params: dictionary, contains keys in res time bins table that capture params
                                                  beyond restriction (e.g. {"epoch_time_bins_param_name": "0.1"})
        # Params specifying restriction that gets applied to time bins:
        # 1) Trials combined to form restriction:
        :param source_trials_table_names: list with names of tables supplying trials used to form restriction,
                                          e.g. ["DioWellArrivalTrials"]
        :param source_trials_table_keys: list with keys specifying a unique entry in each of the above tables, e.g.
                                         [{"dio_well_arrival_trials_param_name": param_name}]
         # 2) Nature of the combination of intervals when forming the restriction
        :param combination_params: dictionary that specifies how trial intervals should be combined to form restriction
        # Return inputs to lookup_param_name:
        :return: list with names of source time bins tables
        :return: list with dictionaries for identifying desired key to above source time bins tables
        """

        # Get inputs if not passed
        # If combination params not passed, use params specifying no combination of intervals
        if combination_params is None:
            combination_params = ResSetParams.get_no_combination_params()

        # Make key to search for appropriate key to source time bins table
        key_search_params = {"time_bins_table_beyond_res_params": time_bins_table_beyond_res_params,
                             "source_trials_table_names": source_trials_table_names,
                             "source_trials_table_keys": source_trials_table_keys,
                             "combination_params": combination_params}

        # For each cohort entry, we need the name of the source time bins table, and the params to search for the key
        # to this table. Return these in lists
        return [source_time_bins_table_name], [key_search_params]

    def lookup_single_member_cohort_param_name(
            self, source_time_bins_table_name, time_bins_table_beyond_res_params, source_trials_table_names,
            source_trials_table_keys, combination_params=None):
        # Wrapper to look up param name for cohort with single member
        source_time_bins_table_names, key_search_params_list = \
            self.get_cohort_entry_lookup_args(
                source_time_bins_table_name, time_bins_table_beyond_res_params, source_trials_table_names,
                source_trials_table_keys, combination_params)
        return self.lookup_param_name(source_time_bins_table_names=source_time_bins_table_names,
                                      key_search_params_list=key_search_params_list)

    # Overrides method in parent class to make it easier to look up param name
    def lookup_param_name(
            self, pool_param_names=None, source_time_bins_table_names=None, key_search_params_list=None,
            tolerate_no_entry=False):
        # Allows lookup of existing param name
        # Need restricted_time_bins_pool_param_name for each cohort entry to look up param name. Use these if
        # passed, otherwise use list with params (key_search_params_list) to find restricted_time_bins_pool_param_name
        # for each member of cohort

        # Check that params passed for only one lookup approach
        valid_approach_taken = (pool_param_names is None and key_search_params_list
                                is not None and key_search_params_list is not None
                                ) or (pool_param_names is not None and
                                      (key_search_params_list is None and key_search_params_list is None))
        if not valid_approach_taken:
            raise Exception(f"Must pass EITHER pool_param_names, OR key_search_params_list "
                            f"and key_search_params_list")

        # Get param names for each cohort entry if these were not passed
        if pool_param_names is None:
            pool_param_names = [self._get_pool_selection_table()().lookup_param_name(
                source_time_bins_table_name=source_time_bins_table_name, key_search_params=key_search_params,
                tolerate_no_entry=tolerate_no_entry) for source_time_bins_table_name, key_search_params in zip(
                    source_time_bins_table_names, key_search_params_list)]

        # Return param name
        secondary_key_subset_map = {"res_time_bins_pool_param_names": pool_param_names}
        return self._get_param_name_table()().lookup_param_name(secondary_key_subset_map=secondary_key_subset_map,
                                                                           tolerate_no_entry=tolerate_no_entry)

    """
    Example script for finding param name for a single entry cohort: 
    # Params specifying time bins source and width: 
    source_time_bins_table_name = "ResEpochTimeBins"
    time_bins_table_beyond_res_params = {epoch_time_bins_param_name: "0.1"}
    dio_well_da_trials_param_name = DioWellDATrialsParams().lookup_param_name([0, 0])
    # Params specifying restriction that gets applied to time bins: 
    # 1) Trials combined to form restriction 
    source_trials_table_names = ["DioWellDATrials"]
    source_trials_table_keys = [{"dio_well_da_trials_param_name": dio_well_da_trials_param_name}]
    # 2) Nature of the combination of intervals when forming the restriction 
    combination_params = ResSetParams.get_no_combination_params()
    
    # Get param name
    res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_single_member_cohort_param_name( 
                    source_time_bins_table_name, time_bins_table_beyond_res_params,
                    source_trials_table_names, source_trials_table_keys,
                    combination_params)
    """

    def lookup_param_name_from_shorthand(self, shorthand_param_name):
        # Return param name for shorthand
        # Note that advantage to this strategy rather than using shorthand for param name in table is flexibility:
        # if later on have more conditions that seem to fit a shorthand, can change map from shorthand to
        # param names without having to dump entries in downstream tables

        # Default single res_time_bins_pool_param_name per cohort cases
        if shorthand_param_name in ResTimeBinsPoolSel.valid_shorthands():
            return self.lookup_param_name(pool_param_names=[ResTimeBinsPoolSel().lookup_param_name_from_shorthand(
                shorthand_param_name)])
        else:
            raise Exception(f"{shorthand_param_name} not accounted for in code to look up param name in "
                            f"{get_table_name(self)} from shorthand")

    def fetch_single_member_cohort_keys(self, key_filter=None):
        if key_filter is None:
            key_filter = dict()
        keys = (ResTimeBinsPoolCohortParams & key_filter).fetch("KEY")
        return [k for k in keys if len((ResTimeBinsPoolCohortParams & k).fetch1("epochs")) == 1]

    def cleanup(self, safemode=True):
        print(
            f"Before clearing entries in ResTimeBinsPoolCohortParams with no upstream entry in ResTimeBinsPool, "
            f"populating ResTimeBinsPool...")
        ResTimeBinsPool.populate()
        for key in self.fetch("KEY"):
            if not all([len(ResTimeBinsPool & {**key, **k}) for k in self.get_cohort_params(key)]) > 0:
                print(f"deleting entry in ResTimeBinsPoolCohortParams for key {key}...")
                (self & key).delete(safemode=safemode)

    def delete_(self, key, safemode=True):
        # Add epochs_id if epoch in key but epochs_id not
        if "epoch" in key and "epochs_id" not in key:
            key["epochs_id"] = get_epochs_id([key["epoch"]])
        from src.jguides_2024.time_and_trials.jguidera_cross_validation_pool import TrainTestSplitPoolSel
        delete_(self, [TrainTestSplitPoolSel], key, safemode)


# Allows combination of time bins across epochs
@schema
class ResTimeBinsPoolCohort(PoolCohortBase):
    definition = """
    # Groups of entries from ResTimeBinsPool
    -> ResTimeBinsPoolCohortParams
    """

    class CohortEntries(PartBase):
        definition = """
        # Entries from ResTimeBinsPool
        -> ResTimeBinsPoolCohort
        -> ResTimeBinsPool        
        """

    def make(self, key):
        # Insert into main table
        self.insert1(key)

        # Insert into parts table
        # ...Get params table entry
        params_entry = (ResTimeBinsPoolCohortParams & key).fetch1()
        # ...Check that start and end of time bin edges concatenated across cohort are ordered.
        # Note that we do not require time bin edges within each epoch in cohort to be ordered, so this is not
        # a requirement upstream. Also note that we do not store the concatenated time bin edges,
        # since we can easily concatenate on the fly and this saves space
        key_names = ["res_time_bins_pool_param_names", "epochs"]
        new_key_names = [strip_string(x, "s", strip_start=False, strip_end=True) for x in key_names]
        new_keys = dict_list(value_lists=[params_entry[k] for k in key_names], key_names=new_key_names)
        cohort_bin_centers_start_stop = []
        for new_key in new_keys:
            df = (ResTimeBinsPool & {**key, **new_key}).fetch1_dataframe()
            cohort_bin_centers_start_stop.append([df.time_bin_centers.iloc[0], df.time_bin_centers.iloc[-1]])
        check_intervals_list(cohort_bin_centers_start_stop)
        # ...Check that number of entries to insert into parts table matches number of epochs
        num_epochs = len(params_entry["epochs"])
        if len(new_keys) != num_epochs:
            raise Exception(f"Should have as many keys to insert into parts table as epochs: {num_epochs}, but "
                            f"instead have {len(new_keys)}")
        # ...Insert each entry from ResTimeBinsPool for given key into cohort entries table
        for new_key in new_keys:
            key.update(new_key)
            insert1_print(self.CohortEntries, key)

    # Override parent class method since iterables used in param name creation are not the same as ones we
    # want to consider when merging dfs across a cohort
    def fetch_dataframes(self, **kwargs):
        kwargs = add_defaults(kwargs, {"iterable_name": "epoch"}, add_nonexistent_keys=True, require_match=True)
        return super().fetch_dataframes(**kwargs)


def populate_jguidera_res_time_bins_pool(key=None, tolerate_error=False, populate_upstream_limit=None,
                                         populate_upstream_num=None):
    schema_name = "jguidera_res_time_bins_pool"
    upstream_schema_populate_fn_list = [populate_jguidera_res_time_bins]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_res_time_bins_pool():
    from src.jguides_2024.time_and_trials.jguidera_condition_trials import drop_jguidera_condition_trials
    from src.jguides_2024.time_and_trials.jguidera_cross_validation_pool import drop_jguidera_cross_validation_pool
    from src.jguides_2024.time_and_trials.jguidera_kfold_cross_validation import drop_jguidera_kfold_cross_validation
    from src.jguides_2024.position_and_maze.jguidera_ppt_interp import drop_jguidera_ppt_interp
    drop_jguidera_condition_trials()
    drop_jguidera_cross_validation_pool()
    drop_jguidera_kfold_cross_validation()
    drop_jguidera_ppt_interp()
    schema.drop()

