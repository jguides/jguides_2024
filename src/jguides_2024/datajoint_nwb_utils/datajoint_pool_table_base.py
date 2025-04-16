import copy
from collections import namedtuple

import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase, SelBase, \
    ParamNameBase, CohortBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import make_param_name, get_table_name, \
    insert1_print, get_meta_param_name, \
    get_non_param_name_primary_key_names, valid_candidate_keys_bool, get_next_int_id, check_int_id, \
    get_table_column_names, get_key_filter, \
    table_name_from_table_type, check_single_table_entry, \
    get_table_secondary_key_names, get_param_name_separating_character, get_num_param_name, fetch_entries_as_dict, \
    check_epochs_id, get_cohort_test_entry, replace_param_name_chars, get_epochs_id
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.utils.dict_helpers import add_defaults
from src.jguides_2024.utils.set_helpers import check_membership, check_set_equality
from src.jguides_2024.utils.string_helpers import strip_trailing_s
from src.jguides_2024.utils.vector_helpers import unpack_single_element, check_all_unique

"""
Documentation
Canonical pool table components:
source_params_dict: dictionary with key-value pairs for accessing relevant entry in source table when combined 
                    with primary key to pool selection table
param_name_dict: dictionary whose values were strung together to create  pool table param name
"""


def check_source_table_name_valid(pool_selection_table, source_table_name):
    check_membership([source_table_name], pool_selection_table._get_valid_source_table_names(),
                     "source table name", "valid source table names")


def make_pool_param_name_dict(source_params_dict, meta_source_table_name, source_table_name):
    return {**{meta_source_table_name: source_table_name}, **source_params_dict}


def make_pool_param_name(param_name_dict, table_name):
    return make_param_name(param_name_dict.values(),
                           separating_character=get_param_name_separating_character(table_name))


def get_pool_table_param_name_quantities(pool_selection_table, source_table, source_table_key):
    # Get pool table param name and related quantities

    # To define pool table param name, we string together "target keys": primary keys in source table that are NOT
    # primary keys in pool_selection_table. First define these target keys and check that all are present in
    # source_table_key
    source_table_name = get_table_name(source_table)
    target_keys = [k for k in source_table.primary_key if k not in pool_selection_table.primary_key]
    missing_keys = [x for x in target_keys if x not in source_table_key]
    if len(missing_keys) > 0:
        raise Exception(f"All primary keys of source_table ({source_table_name}) that are not primary keys of "
                        f"selection table must be present in source_table_key. The following were not: {missing_keys}")

    # To define the param name for the selection table, we string together values in the above target keys,
    # as well as the source table name. These, together with the primary keys of the selection table, define
    # a unique entry in the selection table. Also hold onto a dictionary with these same quantities (param_name_dict),
    # to help us interpret the selection table param name
    # ...Narrow source table key to target keys and ORDER SOURCE TABLE KEY entries in same manner as these. Ensures
    # consistent ordering in param name
    source_params_dict = {k: source_table_key[k] for k in target_keys if k in source_table_key}
    param_name_dict = make_pool_param_name_dict(
        source_params_dict, pool_selection_table._get_meta_source_table_name(), source_table_name)
    # ...String together param_name_dict values to make param name
    param_name = make_pool_param_name(param_name_dict, get_table_name(pool_selection_table))
    ParamNameQuantities = namedtuple("ParamNameQuantities", "param_name param_name_dict source_params_dict")
    return ParamNameQuantities(param_name, param_name_dict, source_params_dict)


def get_pool_table_param_name(pool_selection_table, source_table_name, source_table_key):
    # Intended to allow lookup of param name in pool table
    return get_pool_table_param_name_quantities(
        pool_selection_table, get_table(source_table_name), source_table_key).param_name


def lookup_pool_table_param_name(pool_table, source_table_name, source_table_key, tolerate_no_entry=False):
    # Wrapper function for looking up param name in pool table

    # Get param name for passed params
    param_name = get_pool_table_param_name(pool_table, source_table_name, source_table_key)

    # If entry does not exist in table, optionally raise error
    meta_param_name = get_meta_param_name(pool_table)
    if len(pool_table & {**source_table_key, **{meta_param_name: param_name}}) == 0:
        no_entry_message = f"Entry does not exist in {get_table_name(pool_table)} with param name {param_name}"
        if tolerate_no_entry:
            print(no_entry_message)
            return None
        else:
            raise Exception(no_entry_message)

    # Return param name
    return param_name


def _get_pool_table_insertion_arguments(pool_selection_table, source_table, source_table_key):
    # Get key-value pairs to insert into a pool selection table (with exception of that with int_id; we want to get this
    # just prior to insertion since it is defined based on existing entries in the table)

    # Check inputs
    # ...Check source table name valid
    check_source_table_name_valid(pool_selection_table, get_table_name(source_table))
    # ...Check source table key
    # Check that source_table_key specifies single entry in source table
    source_table_name = get_table_name(source_table)
    num_source_table_entries = len(source_table & source_table_key)
    if num_source_table_entries != 1:
        raise Exception(f"source_table_key should specify exactly one entry in {source_table_name}, but specifies "
                        f"{num_source_table_entries}. source_table_key: {source_table_key}")
    # Check that all keys in source_table_key are primary keys in source table, and that all primary keys in source
    # table are in source_table_key
    check_set_equality(source_table.primary_key, source_table_key.keys(), f"{source_table_name} primary key",
                       "source table key")

    # For source_table_key (dictionary that specifies a single entry in source table), we want to find the
    # key-value pairs corresponding to primary keys in the selection table (except for the selection table param name),
    # to insert into the selection table
    # ...Get primary keys in selection table heading except for the param name primary key
    non_param_name_primary_keys = get_non_param_name_primary_key_names(pool_selection_table)
    # ...Keep corresponding key-value pairs in source_table_key
    primary_key = {k: v for k, v in source_table_key.items() if k in non_param_name_primary_keys}

    # Get param name for pool selection table and related quantities, including: primary keys in source table that
    # are not primary keys in selection table (allows us to access relevant entry in source table)
    param_name_obj = get_pool_table_param_name_quantities(pool_selection_table, source_table, source_table_key)

    # Get what param name is called in selection table (meta param name)
    meta_param_name = get_meta_param_name(pool_selection_table)

    InsertArguments = namedtuple("InsertArguments", "primary_key meta_param_name param_name meta_source_table_name "
                                                    "source_table_name source_params_dict param_name_dict")
    return InsertArguments(primary_key, meta_param_name, param_name_obj.param_name,
                           pool_selection_table._get_meta_source_table_name(), source_table_name,
                           param_name_obj.source_params_dict, param_name_obj.param_name_dict)


def _get_pool_table_insertion_key(pool_selection_table, source_table, source_table_key):
    # Return pool table insert arguments as dictionary

    x = _get_pool_table_insertion_arguments(pool_selection_table, source_table, source_table_key)

    return {**x.primary_key,
            **{x.meta_param_name: x.param_name,
               x.meta_source_table_name: x.source_table_name,
               "source_params_dict": x.source_params_dict,
               "param_name_dict": x.param_name_dict}}


def get_pool_part_table_name(main_table, source_table_name):
    return f"{get_table_name(main_table)}.{source_table_name}"


def make_int_id_cohort_param_name(int_ids):
    # Make cohort param name by stringing together integer IDs for each member of cohort

    return f"_{make_param_name(np.sort(int_ids))}_"


# Base class for pool tables
# Here, we opted to have only selection table and no params table, because we are just collecting entries
# from upstream tables in one place, without any alteration or formation of groupings (cohorts) of them
# Still, we need a "param name" to stand for the unique identifiers of each entry
# TODO (feature): would be good to have cleanup method for when upstream entries are deleted
class PoolSelBase(SelBase):

    @staticmethod
    def _get_valid_source_table_names():
        raise Exception(f"Must overwrite in child class")

    # Extend parent class method so can add to list of bad keys those that share the same int_id
    def _get_bad_keys(self):

        bad_keys = []
        int_ids = self.fetch("int_id")
        bad_int_ids = np.unique([x for x in int_ids if np.sum(int_ids == x) > 1])
        if len(bad_int_ids) > 0:
            bad_keys += list(np.concatenate([
                (self & {"int_id": bad_int_id}).fetch("KEY") for bad_int_id in bad_int_ids]))
        bad_keys += super()._get_bad_keys()

        return bad_keys

    def insert1(self, source_table, source_table_key):
        # Insert into pool selection table

        # Get insertion arguments
        insert_dict = _get_pool_table_insertion_key(self, source_table, source_table_key)

        # Get integer ID for entry
        insert_dict.update({"int_id": get_next_int_id(self)})

        # Check int_id valid right before insertion (in case helpful to avoid inserting same int_id for different
        # entries in case populating table from multiple notebooks)
        check_int_id(self, insert_dict["int_id"])

        # Insert into table
        super().insert1(insert_dict, skip_duplicates=True)

    def insert_defaults(self, populate_pool_table=True, **kwargs):

        # Get key filter if passed
        key_filter = get_key_filter(kwargs)

        # Populate pool selection table: loop through source tables and copy entries to selection table
        source_table_names = set(self._get_valid_source_table_names()) - {"none"}

        for source_table_name in source_table_names:

            # Get source table and restrict based on key filter
            source_table = (get_table(source_table_name) & key_filter)

            # Get primary key key-value pairs from source table
            source_table_keys = source_table.fetch("KEY")

            # For each source table key, get insertion arguments to pool selection table
            insert_dicts = [
                _get_pool_table_insertion_key(self, source_table, source_table_key)
                for source_table_key in source_table_keys]

            # Find indices of insertion arguments that do not yet exist in pool selection table. This can speed
            # things up because using insert1 with existing entry seems to take a while
            valid_bool = valid_candidate_keys_bool(self, insert_dicts)

            # Insert into pool selection table
            for source_table_key in np.asarray(source_table_keys)[valid_bool]:
                self.insert1(source_table, source_table_key)

        # Populate pool table
        if populate_pool_table:
            self._get_main_table().populate()

    @staticmethod
    def _get_meta_source_table_name():
        return "source_table_name"

    def get_source_table(self, key=None):

        if key is None:
            key = self.fetch1("KEY")

        return get_table((self & key).fetch1("source_table_name"))

    def get_source_table_entry(self, key=None, tolerate_no_entry=False):
        """
        Return source table entry stored in pool table specified by passed key
        :param selection_table: selection table for pool table
        :param key: key to entry in selection table
        :param tolerate_no_entry: boolean, if no entry: if True return None, if False raise error
        :return: subset of source table
        """

        # Get key if not passed
        if key is None:
            key = self.fetch1("KEY")

        # Get source params dict: the extra params we need to add to key to specify a unique entry in the source table
        source_params_dict = (self & key).fetch1("source_params_dict")

        # Make key to source table
        source_table_key = {**key, **source_params_dict}

        # Return source table entry
        source_table_entry = self.get_source_table(key) & source_table_key

        # If indicated, raise error if no entry found
        if source_table_entry is None and not tolerate_no_entry:
            source_table_name = (self & key).fetch1("source_table_name")
            raise Exception(f"No entry found in {source_table_name} for key {source_table_key}")

        return source_table_entry

    def get_entries_with_param_name_dict(self, param_name_dict, key_filter=None, restrict_primary_key=False):
        # Find entries with param_dict matching the one passed by user

        # Narrow to subset of table matching key_filter if indicated
        table_subset = copy.deepcopy(self)  # default is entire table
        if key_filter is not None:
            table_subset = (table_subset & key_filter)

        # Filter for table entries for which passed param_name_dict encompassed in entry param_name_dict
        column_names = get_table_column_names(self)
        table_entries = [{k: v for k, v in zip(column_names, table_entry)} for table_entry in table_subset.fetch()
                if all([table_entry["param_name_dict"].get(k) == v for k, v in param_name_dict.items()])]

        # Restrict to primary keys if indicated. We do this after getting full table entry since we require
        # param name dict, a secondary key, in filtering step above
        if restrict_primary_key:
            table_entries = [{k: table_entry[k] for k in self.primary_key} for table_entry in table_entries]

        return table_entries

    def lookup_param_name(self, source_table_name, source_table_key, tolerate_no_entry=False):
        return lookup_pool_table_param_name(self, source_table_name, source_table_key, tolerate_no_entry)

    def alter(self, prompt=True, context=None):
        super().alter()


class PoolBase(ComputedBase):

    def make(self, key):
        # Insert into pool table

        selection_table = self._get_selection_table()

        # Insert into main table:
        # ...Set part table name to source table name
        selection_entry = (selection_table & key)
        source_table_name = selection_entry.fetch1('source_table_name')
        part_table_name = get_pool_part_table_name(self, source_table_name)
        # ...Insert into table
        main_key = {**key, **{"part_table_name": part_table_name}}
        insert1_print(self, main_key)

        # Insert into part table:
        # ...Get name of source table, and source params dict: the extra params we need
        # to add to key to specify a unique entry in the source table
        source_params_dict, source_table_name = (selection_table & key).fetch1(
            "source_params_dict", "source_table_name")
        # ...Get part table
        part_table = get_table(part_table_name)
        # ...Insert into parts table
        insert1_print(part_table, {**key, **source_params_dict})

    def _get_source_table_entry(self):
        return self._get_selection_table()().get_source_table_entry(key=self.fetch1())

    def _get_source_params_table(self):
        return

    def fetch1_part_entry(self):
        return self._get_source_table_entry().fetch1()

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        return self._get_source_table_entry().fetch1_dataframe(
            object_id_name=object_id_name, restore_empty_nwb_object=restore_empty_nwb_object,
            df_index_name=df_index_name)

    def fetch1_dataframes(self):
        return self._get_source_table_entry().fetch1_dataframes()


class PoolCohortParamNameBase(ParamNameBase):

    # TODO (feature): could constrain this more by checking if param name values that were passed exist in selection table
    # Override method in parent class. This is currently not a method of ParamNameBase because there are uses of that
    # for which we dont want to do all the ordering done in this method
    def get_full_param_name(self, secondary_key_subset_map, replace_chars=True):
        """
        Make full param name for params table by stringing together param names of upstream table that correspond to
        entries in cohort
        :param secondary_key_subset_map: {secondary key name: instances of secondary key to include in param name, ...}
        :param replace_chars: if True, replace certain characters. If this occurs, full_param_name may not be unique
         for a given secondary_key_subset_map. Therefore, if user thinks there is risk of this in a given table,
         replace_chars should be set to False there. Default is True
        :return: string, param name for current table (abbreviated version of "full" parameter name)
        """

        param_name_iterables = self._get_params_table()()._get_param_name_iterables()

        # Check that secondary key map contains exactly iterables we want to include in param name
        check_set_equality(secondary_key_subset_map.keys(), param_name_iterables, "keys in secondary key subset map",
                           "name of secondary keys we want to include in cohort param name")

        # Order the secondary keys by their order in param name iterables, so that order of passed keys does not matter
        secondary_key_subset_map = {k: secondary_key_subset_map[k] for k in param_name_iterables}

        # Order each set of values in secondary key, so that order of passed values does not matter
        secondary_key_subset_map = {k: np.sort(np.asarray(v)) for k, v in secondary_key_subset_map.items()}

        # Make param name by joining values of iterables
        param_name = make_param_name(list(secondary_key_subset_map.values()),
                               separating_character=get_param_name_separating_character(
                                   get_table_name(self._get_params_table())))

        # Replace characters. Note that this means param name not necessarily unique
        if replace_chars:
            param_name = replace_param_name_chars(param_name)

        return param_name


class PoolCohortParamsBase(SecKeyParamsBase):

    def _get_meta_cohort_param_name(self):
        return unpack_single_element([x for x in self.primary_key if "cohort_param_name" in x])

    def _get_pool_selection_table(self):
        # Get pool selection table

        # Get name of current table
        table_name = get_table_name(self)
        # Use to get main table name
        main_table_name = table_name_from_table_type(table_name, "CohortParams")
        # Use to get pool selection table name

        return get_table(main_table_name + "Sel")

    # Overrides method in parent class so can check key
    def insert1(self, key, check_unique_iterables=True):

        # If indicated, check that entries unique for at least one set of the iterables that compose param name.
        # This is to prevent inserting different cohort entries that contain identical information
        if check_unique_iterables:
            num_passed_check = 0
            for meta_param_name in self._get_param_name_iterables():
                num_passed_check += check_all_unique(key[meta_param_name], tolerate_error=True)
            if num_passed_check == 0:
                raise Exception(f"iterables that make up cohort param name not unique; may result in different "
                                f"cohort entries containing identical information")

        super().insert1(key, skip_duplicates=True)

    def _get_insert_cohort_param_name(self, secondary_key_subset_map, use_full_param_name=False):
        return self._get_param_name_table()().get_insert_param_name(use_full_param_name=use_full_param_name,
                                                                    secondary_key_subset_map=secondary_key_subset_map)

    def _insert_from_upstream_param_names(self, secondary_key_subset_map, key, use_full_param_name):

        # Update key with param names of upstream table
        key.update(secondary_key_subset_map)

        # Update key with cohort param name (param name for current table) and insert into param name table
        # TODO: think should SORT each set of values in secondary key prior to making param name. May need
         # to be done separately for each table because in some cases want to order by something specific (e.g. epoch)
        cohort_param_name = self._get_insert_cohort_param_name(secondary_key_subset_map, use_full_param_name)
        key.update({self._get_meta_cohort_param_name(): cohort_param_name})

        # For each upstream param name, add secondary key for number of param names if present in current table
        for meta_param_name, param_values in secondary_key_subset_map.items():
            meta_num_name = f"num_{meta_param_name}"
            if meta_num_name in get_table_secondary_key_names(self):
                key.update({meta_num_name: len(param_values)})

        # Insert into current table
        self.insert1(key)

    def _get_param_name_iterables(self, singular=False):

        # Get all secondary keys from corresponding params table
        param_name_iterables = get_table_secondary_key_names(self)

        # Exclude secondary keys that describe cohort size
        param_name_iterables = [
            x for x in param_name_iterables if x not in [get_num_param_name(y) for y in param_name_iterables]]

        # Return singular if indicated, otherwise return plural
        if singular:
            return [strip_trailing_s(x) for x in param_name_iterables]
        return param_name_iterables

    def insert_single_member_cohort(self, param_name_dict, **kwargs):
        """
        Insert cohorts with single member
        :param param_name_dict: dictionary, matching secondary key in pool table. Find all entries in pool table that
               have param_name_dict matching this
        """

        key_filter = dict()
        if "key_filter" in kwargs:
            key_filter = kwargs["key_filter"]

        # Find entries in pool selection table with param_dict matching the one passed by user
        pool_table_entries = (self._get_pool_selection_table() & key_filter).get_entries_with_param_name_dict(param_name_dict)

        # Insert each entry as its own cohort
        non_param_name_primary_key_names = get_non_param_name_primary_key_names(self)
        param_name_iterables_singular = self._get_param_name_iterables(singular=True)
        param_name_iterables_plural = self._get_param_name_iterables()
        print(f"Looping through {len(pool_table_entries)} pool_table_entries...")
        for idx, entry in enumerate(pool_table_entries):
            # Add primary keys that are not cohort param name
            key = {k: entry[k] for k in non_param_name_primary_key_names}
            # Make map from secondary keys used to generate cohort param name to single value for each
            secondary_key_subset_map = {
                k1: [entry[k2]] for k1, k2 in zip(param_name_iterables_plural, param_name_iterables_singular)}
            # For interpretability and since should respect character limit, use full param name for single
            # member cohort
            self._insert_from_upstream_param_names(secondary_key_subset_map, key, use_full_param_name=True)

    def lookup_param_name(self, source_table_names, source_table_keys, tolerate_no_entry=False, **kwargs):
        # Return cohort param name

        # Get param name of upstream pool table for each member of cohort, since these are used to construct
        # cohort param name
        # TODO (feature): allow different source table names / keys to be used for different param name iterables, to
        #  allow multiple param name iterables. For now, require this method only be used when a single
        #  param name iterable.
        if len(self._get_param_name_iterables()) > 1:
            raise Exception(f"Code currently not equipped to work with more than one param name iterable")

        secondary_key_subset_map = dict()

        for param_name_iterable in self._get_param_name_iterables():  # keep loop so easier to make above change
            pool_param_names = [
                self._get_pool_selection_table()().lookup_param_name(
                    source_table_name, source_table_key, tolerate_no_entry)
                for source_table_name, source_table_key in zip(source_table_names, source_table_keys)]
            secondary_key_subset_map.update({param_name_iterable: pool_param_names})

        # Return cohort param name
        return self._get_param_name_table()().lookup_param_name(
            secondary_key_subset_map=secondary_key_subset_map, tolerate_no_entry=tolerate_no_entry)

    def lookup_param_name_from_shorthand(self, shorthand_param_name):
        raise Exception(f"This method must be overwritten in child class")


class PoolCohortBase(CohortBase):

    def make(self, key):

        # Insert into main table
        insert1_print(self, key)

        # Insert into parts table
        params_table = self._get_params_table()
        param_name_iterables_plural = params_table()._get_param_name_iterables()
        param_name_iterables_singular = params_table()._get_param_name_iterables(singular=True)
        for x in fetch_entries_as_dict(params_table & key, param_name_iterables_plural):
            for params in np.asarray((list(x.values()))).T:
                key.update({k: v for k, v in zip(param_name_iterables_singular, params)})
                insert1_print(self.CohortEntries, key)

    def _get_param_name_iterables(self):
        # Get what params are called in upstream pool table
        return self._get_params_table()()._get_param_name_iterables(singular=True)

    def get_upstream_pool_table_param_names(self, key):

        # Check that passed key specifies a single cohort
        check_single_table_entry(self, key)  # one entry in main table per cohort

        # Return upstream param names for cohort members
        param_name_iterables = self._get_param_name_iterables()

        return (self.CohortEntries & key).fetch(*param_name_iterables)

    def fetch_dataframes(self, **kwargs):

        # Add iterable name and indicate that add to dfs, if these not in kwargs
        iterable_name = unpack_single_element(self._get_param_name_iterables())
        kwargs = add_defaults(kwargs, {"iterable_name": iterable_name, "add_iterable": True}, add_nonexistent_keys=True)

        return super().fetch_dataframes(**kwargs)


class EpsCohortParamsBase(SecKeyParamsBase):

    @staticmethod
    def _upstream_table():
        raise Exception(f"This method must be overwritten in child class")

    def insert1(self, key, **kwargs):

        # Check that epochs all unique since no effect of including the same epoch more than once on downstream
        # processing (i.e. would lead to multiple cohort entries with same effect)
        check_all_unique(key["epochs"])

        # Check that epochs_id consistent with epochs
        check_epochs_id(key["epochs_id"], key["epochs"])

        super().insert1(key, **kwargs)

    @staticmethod
    def _update_key(key):

        # If single epoch cohort, change epoch to epochs
        if "epoch" in key:
            epoch = key.pop("epoch")
            key.update({"epochs": [epoch]})

        # Add epochs_id to key
        # ...Ensure upstream table populated
        from src.jguides_2024.metadata.jguidera_epoch import EpochCohortParams
        EpochCohortParams().insert_from_epochs(key["nwb_file_name"], key["epochs"])
        key.update({"epochs_id":  get_epochs_id(key["epochs"])})

        # Return key
        return key

    def insert_single_member_cohort_defaults(self):
        # Populate cohort table with single epoch entries
        upstream_keys = self._upstream_table().fetch("KEY")
        print(f"Found {len(upstream_keys)} upstream_keys...")
        candidate_keys = [self._update_key(key) for key in upstream_keys]
        for key in candidate_keys:
            self.insert1(key, skip_duplicates=True)

    def insert_test(self):

        # Insert an across epochs entry for testing
        # Use for test, first two entries where there is same nwb_file_name and trials_pool_param_name
        num_epochs = 2
        test_entry_obj = get_cohort_test_entry(self._upstream_table(), col_vary="epoch", num_entries=num_epochs)

        # Insert if found valid set
        if test_entry_obj is not None:
            key = {**test_entry_obj.same_col_vals_map, **{"epochs": test_entry_obj.target_vals}}
            key = self._update_key(key)
            self.insert1(key, skip_duplicates=True)

    def insert_defaults(self, **kwargs):
        self.insert_single_member_cohort_defaults()
