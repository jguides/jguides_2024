import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, ComputedBase, \
    ParamNameBase, PartBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import convert_array_none, make_param_name, \
    insert1_print, get_key_filter
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPoolCohort, TrialsPoolCohortParams, \
    populate_jguidera_trials_pool
from src.jguides_2024.utils.dict_helpers import sort_dict_by_keys
from src.jguides_2024.utils.interval_helpers import CombineIntervalLists
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements
from src.jguides_2024.utils.plot_helpers import plot_intervals, format_ax
from src.jguides_2024.utils.point_process_helpers import bins_in_intervals_bool

schema = dj.schema("jguidera_res_set")

# Needed for table definitions
TaskIdentification


# Note that having ParamName table speeds up searching for an existing param name with desired characteristics.
# Otherwise, one must compare iterables, for example dictionaries. This is meant to only be populated when populating
# # corresponding params table.
@schema
class ResSetParamName(ParamNameBase):
    definition = """
    # Map between full param name and integer used as param name
    res_set_param_name : varchar(100)  # stands for how to combine intervals
    ---
    int_id = NULL : int
    full_param_name : varchar(1000)
    """

    # Overrides parent class method to sort appropriately and make param name more interpretable. Note that
    # drawback of interpretability is uniqueness of param name not guaranteed, so user should be aware of meaning
    # of param names
    def get_full_param_name(self, trials_pool_cohort_param_name, combination_params):
        # Before making param name, sort combination params so that order in which keys passed does not matter.
        # Here, we want to ONLY sort keys, and not the values within each key, since the order of values matters
        # to the combination operations that will be performed
        # Also, if combination params dict corresponds to interpretable case, abbreviate to reflect this
        def get_combination_params_text(combination_params):
            return make_param_name(list(sort_dict_by_keys(combination_params).values()), separating_character="_",
                                   tolerate_non_unique=True)
        combination_params_text = get_combination_params_text(combination_params)  # default
        # Make map with available abbreviations for combinations params text
        no_combination_params_text = get_combination_params_text(self._get_params_table().get_no_combination_params())
        abbreviation_map = {no_combination_params_text: "no_comb"}
        # Use map to abbreviate combination params text if indicated
        if combination_params_text in abbreviation_map:
            combination_params_text = abbreviation_map[combination_params_text]
        # Return full param name
        return make_param_name([trials_pool_cohort_param_name, combination_params_text], separating_character="_",
                               tolerate_non_unique=True)


@schema
class ResSetParams(SecKeyParamsBase):
    definition = """
    # Specifies parameters for ResSet
    -> ResSetParamName
    -> TaskIdentification
    ---
    -> TrialsPoolCohort
    combination_interval_list_names : blob
    combination_sources : blob
    combination_operations : blob
    """

    # Note that conceptually it makes more sense to have a single params table instead of a params table and
    # selection table here, since the information from the upstream TaskIdentification has a direct correspondence
    # with parameter settings. Note also we chose not to have a single primary key that is the param name here,
    # since we want to intersect this table with others, using attributes nwb_file_name and epoch (we achieve
    # this through primary dependence on TaskIdentification). We dont expect to want to intersect with tables based
    # only on trials pool cohort param name, so we fold this information in with the combination information
    # when creating the param name for ResSet

    # Override parent class method so can check combination params valid
    def insert1(self, key, **kwargs):
        # Before inserting, check that combination params make sense
        # ...Get interval lists: these are sets of trial times from specified sources
        interval_lists = TrialsPoolCohort().get_cohort_trial_intervals(key)
        # ...Package combination params and convert to form expected by class that will do the checking
        combination_params = {k: key[k] for k in ["combination_interval_list_names", "combination_sources",
                                                  "combination_operations"]}
        combination_params = self.convert_combination_params(combination_params)  # convert ["none"] to None
        CombineIntervalLists(interval_lists).convert_check_combination_params(**combination_params)
        # ...Get interval lists: these are sets of trial times from specified sources
        interval_lists = TrialsPoolCohort().get_cohort_trial_intervals(key)
        # ...Package combination params and convert to form expected by class that will do the checking
        combination_params = {k: key[k] for k in ["combination_interval_list_names", "combination_sources",
                                                  "combination_operations"]}
        combination_params = self.convert_combination_params(combination_params)  # convert ["none"] to None
        CombineIntervalLists(interval_lists).convert_check_combination_params(**combination_params)
        super().insert1(key, **kwargs)

    @staticmethod
    def convert_combination_params(combination_params):
        # Convert datajoint form of combination params to ones for class that combines interval lists. We do this
        # instead of using a common form because datajoint cannot store None, but it makes more sense to wider use
        # cases to take in None in functions/classes
        return {k: convert_array_none(v) for k, v in combination_params.items()}

    def _get_insert_param_name(self, trials_pool_cohort_param_name, combination_params, use_full_param_name):
        # Get param name and insert entry into param name table
        return self._get_param_name_table()().get_insert_param_name(
            use_full_param_name=use_full_param_name, trials_pool_cohort_param_name=trials_pool_cohort_param_name,
            combination_params=combination_params)

    @staticmethod
    def get_no_combination_params():
        # Get params that signal performing no combination of intervals
        return {"combination_interval_list_names": ["none"],
                  "combination_sources": ["none"],
                  "combination_operations": ["none"]}

    # Overrides parent class method
    def lookup_param_name(self, trials_pool_cohort_param_name, combination_params, tolerate_no_entry=False):
        return self._get_param_name_table()().lookup_param_name(
            trials_pool_cohort_param_name=trials_pool_cohort_param_name, combination_params=combination_params,
            tolerate_no_entry=tolerate_no_entry)

    def lookup_no_combination_param_name(self, trials_pool_cohort_param_name):
        return self.lookup_param_name(trials_pool_cohort_param_name, self.get_no_combination_params())

    def _insert_from_combination_params(self, combination_params, key, use_full_param_name):
        # Get param name and insert into param name table
        res_set_param_name = self._get_insert_param_name(
            key["trials_pool_cohort_param_name"], combination_params, use_full_param_name)
        # Update key
        key.update({**combination_params, **{"res_set_param_name": res_set_param_name}})
        # Insert into table
        self.insert1(key)

    def insert_no_combination(self, key):
        """
        Insert params for a direct use of a single set of trial intervals from a cohort with a
        single entry in TrialsPoolCohort
        :param key: dictionary, specifies single entry (cohort) in TrialsPoolCohort. This cohort
                    must consist of exactly one entry in TrialsPool
        """
        # Get TrialsPool param names for single set of trials
        trials_pool_param_names = TrialsPoolCohort().get_upstream_pool_table_param_names(key)

        # Check that exactly one param name
        if len(trials_pool_param_names) != 1:
            raise Exception(f"Must have exactly one entry in trials_pool_param_names to be able to "
                            f"insert no combination case")

        # Define combination params
        combination_params = self.get_no_combination_params()

        # Insert into table, using trials pool param name as the param name for res set (since no additional
        # information added in this table for this case due to no combination of intervals)
        self._insert_from_combination_params(combination_params, key, use_full_param_name=True)

    def insert_simple_intersection(self, key):
        """
        Insert params for a simple intersection of trial intervals in a cohort in TrialsPoolCohort
        :param key: dictionary, specifies single entry (cohort) in TrialsPoolCohort
        """
        # Get TrialsPool param names for each set of trials
        trials_pool_param_names = TrialsPoolCohort().get_upstream_pool_table_param_names(key)

        # Check that at least two param names, otherwise cannot do intersection
        if len(trials_pool_param_names) < 2:
            raise Exception(f"Must have at least two entries in trials_pool_param_names to be able to "
                            f"intersect trial sets")

        # Each combination is between two elements. Therefore, we iteratively intersect trial sets
        combination_interval_list_names = zip_adjacent_elements(trials_pool_param_names)
        num_combinations = len(combination_interval_list_names)
        combination_params = {"combination_interval_list_names": combination_interval_list_names,
                              "combination_sources": [["original", "original"]]*num_combinations,
                              "combination_operations": ["intersection"]*num_combinations}

        self._insert_from_combination_params(combination_params, key, use_full_param_name=False)

    def insert_test(self):
        # Insert simple intersection of trial times from a cohort in TrialsPoolCohort, for testing

        # Find cohorts in TrialsPoolCohort with more than one member
        # ..First get all entries in table
        potential_keys = np.asarray(TrialsPoolCohort.fetch("KEY"))
        # ..Narrow to cohorts with more than one member
        table_intersection = (TrialsPoolCohort * TrialsPoolCohortParams)  # helpful if cohort table not fully populated
        valid_bool = np.asarray(list(map(len, table_intersection.fetch("trials_pool_param_names")))) > 1
        potential_keys = potential_keys[valid_bool]

        # Exit if no entries in TrialsPoolCohort
        if len(potential_keys) == 0:
            print(f"No entries in TrialsPoolCohort that could be used for simple intersection, so cannot insert "
                  f"test entry into ResSetParams; exiting")
            return

        # Otherwise use first cohort available. We get the trials_pool_param_name corresponding
        # to each member of the cohort, and use these to name the corresponding
        # lists of trial times
        key = potential_keys[0]
        self.insert_simple_intersection(key)

    def insert_no_combination_defaults(self, **kwargs):
        # Populate with all single member cohorts in TrialsPoolCohort
        key_filter = get_key_filter(kwargs)
        key_filter.update({"num_trials_pool_param_names": 1})
        keys = (TrialsPoolCohortParams & key_filter).fetch("KEY")
        # Exclude PptTrials since currently no use case
        keys = [k for k in keys if "PptTrials" not in k["trials_pool_cohort_param_name"]]
        for key in keys:
            TrialsPoolCohort.populate(key)  # ensure upstream table populated
            self.insert_no_combination(key)

    def insert_defaults(self, **kwargs):
        # Populate with single member cohorts and test case
        self.insert_no_combination_defaults(**kwargs)

    def fetch1_combination_params(self):
        table_entry = self.fetch1()
        combination_params = {k: table_entry[k] for k in [
            "combination_interval_list_names", "combination_sources", "combination_operations"]}
        return self.convert_combination_params(combination_params)  # convert ["none"] to None

    def delete_(self, key):
        raise Exception(f"need to write")
        # TODO (feature): figure out if a good way to track entries up through pipeline from here. currently
        #  challenging because many different param names. leads to proposal to delete entries that are not
        #  dependent on what we want to delete.


# Note that decided to have this NOT span epochs, since wanted to use for restricted spikes tables that have
# data from just one epoch
@schema
class ResSet(ComputedBase):
    definition = """
    # Valid intervals obtained through intersection and union of trial intervals across sources within an epoch
    -> ResSetParams
    ---
    valid_time_intervals : blob
    """

    class Upstream(PartBase):
        definition = """
        # Achieves upstream dependence on TrialsPoolCohort
        -> ResSet
        -> TrialsPoolCohort
        """

    @staticmethod
    def _get_upstream_key(key):
        key.pop("valid_time_intervals", None)
        return {**key, **{"trials_pool_cohort_param_name": (ResSetParams & key).fetch1("trials_pool_cohort_param_name")}}

    def get_trial_intervals(self, key):
        return TrialsPoolCohort().get_cohort_trial_intervals(self._get_upstream_key(key))

    def make(self, key):
        # Get trial intervals
        trial_intervals = self.get_trial_intervals(key)
        # Get combination params
        combination_params = (ResSetParams & key).fetch1_combination_params()
        # Get trial intervals from each specified source
        key["valid_time_intervals"] = CombineIntervalLists(trial_intervals).get_combined_intervals(**combination_params)

        # Insert into main table
        insert1_print(self, key)
        # Insert into part table
        insert1_print(self.Upstream, self._get_upstream_key(key))

    def apply_restriction(self, key, time_bin_centers, time_bin_edges=None, verbose=False):
        # Return times within valid intervals. Use just time bin centers if only these passed. If accompanying
        # time bin edges passed, return time bin centers whose corresponding bin edges both fall within valid time
        # intervals
        valid_time_intervals = (self & key).fetch1("valid_time_intervals")  # valid time intervals
        valid_bool = bins_in_intervals_bool(valid_time_intervals, time_bin_centers, time_bin_edges)
        valid_time_bin_centers = time_bin_centers[valid_bool]

        # Plot restriction if indicated
        if verbose:
            # TODO: plot valid intervals with shaded rectangles
            fig, ax = plt.subplots(figsize=(12, 3))
            # Plot time bin centers
            ax.plot(time_bin_centers, [1.1] * len(time_bin_centers), 'o', label="time bin centers", color="gray")
            # Plot time bin edges if passed
            if time_bin_edges is not None:
                for label, x in zip(["bin_starts", "bin_stops"], zip(*time_bin_edges)):  # for bin edge starts and stops
                    ax.plot(x, 'o', label=label, alpha=.7, markersize=5, color='orange')
            # Plot valid time bin centers
            ax.plot(valid_time_bin_centers, [1.1] * len(valid_time_bin_centers), 'o', label="valid time bin centers",
                    color="orange")
            # Plot valid time bin centers bool
            ax.plot(time_bin_centers, valid_bool, 'o', label="valid time bin centers bool", color="blue")
            # Legend
            ax.legend()
            # Title
            ax.set_title(f"valid_intervals: {valid_time_intervals}")

        return valid_time_bin_centers, valid_bool

    def plot_result(self, key):
        # Plot original trial intervals and valid intervals resulting from combination
        # Initialize plot
        fig, ax = plt.subplots(figsize=(12, 2))
        # Plot original intervals
        trial_intervals = self.get_trial_intervals(key)
        for idx, (intervals_name, intervals) in enumerate(trial_intervals.items()):
            plot_intervals(intervals, val_list=[idx] * len(intervals), ax=ax, label=intervals_name)
        # Plot valid intervals from combination
        valid_time_intervals = (self & key).fetch1("valid_time_intervals")
        plot_intervals(valid_time_intervals, val_list=[idx + 1] * len(valid_time_intervals), ax=ax, color="green",
                       label="valid_time_intervals")
        format_ax(ax=ax, ylim=[-1, idx + 2])
        ax.legend()


def populate_jguidera_res_set(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_res_set"
    upstream_schema_populate_fn_list = [populate_jguidera_trials_pool]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_res_set():
    from src.jguides_2024.spikes.jguidera_res_spikes import drop_jguidera_res_spikes
    from src.jguides_2024.time_and_trials.jguidera_res_time_bins import drop_jguidera_res_time_bins
    drop_jguidera_res_spikes()
    drop_jguidera_res_time_bins()
    schema.drop()

