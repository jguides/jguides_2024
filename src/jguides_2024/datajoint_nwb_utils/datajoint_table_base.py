import copy
from collections import namedtuple

import datajoint as dj
import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_table_name, make_param_name, \
    populate_flexible_key, get_upstream_table_names, \
    insert_analysis_table_entry, fetch1_dataframe, fetch_nwb, plot_datajoint_table_rate_map, intersect_tables, \
    insert_manual_table_test_entry, get_meta_param_name, fetch1_dataframes, \
    trial_duration_from_params_table, insert1_print, fetch1_dataframe_from_table_entry, get_table_secondary_key_names, \
    package_secondary_key, get_schema_table_names_from_file, get_schema_names, \
    get_downstream_table_names, special_fetch1, get_table_object_id_name, \
    populate_insert, fetch1_tolerate_no_entry, get_next_int_id, check_single_table_entry, fetch_entries_as_dict, \
    get_params_table_name, get_epochs_id, delete_, get_key_filter, UpstreamEntries
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.df_helpers import df_filter_columns_isin, add_column_to_df
from src.jguides_2024.utils.dict_helpers import check_same_values_at_shared_keys
from src.jguides_2024.utils.digitize_helpers import digitize_indexed_variable
from src.jguides_2024.utils.interval_helpers import fill_trial_values
from src.jguides_2024.utils.make_bins import make_bin_edges
from src.jguides_2024.utils.plot_helpers import plot_heatmap
from src.jguides_2024.utils.point_process_helpers import get_full_event_times_relative_to_trial_start, \
    event_times_in_intervals_bool
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.string_helpers import camel_to_snake_case
from src.jguides_2024.utils.vector_helpers import unpack_single_element, check_length, check_all_unique, \
    vector_midpoints, unpack_single_vector

"""
Notes
1) Had apparent issues using class methods (@classmethod) to override datajoint table methods, e.g.
populate, insert1, fetch1. So avoiding overriding datajoint table methods with these.
"""


class ParamsBase(dj.Manual):

    def meta_param_name(self):
        # As method rather than attribute because if datajoint table not initialized, get error when defining attribute

        return get_meta_param_name(self, verbose=False)  # set verbose to True to print warnings

    def lookup_param_name(self, **kwargs):
        raise Exception(f"Must be implemented in child class")

    def insert1(self, key, **kwargs):

        if "skip_duplicates" not in kwargs:
            kwargs["skip_duplicates"] = True

        super().insert1(key, **kwargs)

class SecKeyParamsBase(ParamsBase):

    def _make_param_name(self, secondary_key_subset_map, separating_character=None, tolerate_non_unique=False):
        # Make param name by stringing together values of a subset of secondary keys in table

        # Check that passed secondary key is subset of actual secondary key
        check_membership(
            secondary_key_subset_map.keys(), get_table_secondary_key_names(self),
            "passed secondary keys", f"{get_table_name(self)} secondary keys")

        # Enforce secondary key order in table definition
        secondary_key_subset_map = {k: secondary_key_subset_map[k] for k in get_table_secondary_key_names(self)
                                    if k in secondary_key_subset_map}

        # String together values
        return make_param_name(secondary_key_subset_map.values(), separating_character, tolerate_non_unique)

    def _param_name_secondary_key_columns(self):
        # Default is to use all available secondary keys in param name
        return get_table_secondary_key_names(self)

    def _get_secondary_key(self, secondary_key_values, param_name_subset):
        # Define subset of secondary key names that correspond to secondary key values

        secondary_key_names_subset = None  # default, corresponds to all secondary key values passed
        if param_name_subset:
            secondary_key_names_subset = self._param_name_secondary_key_columns()

        return package_secondary_key(
            self, secondary_key_values, get_table_secondary_key_names(self, secondary_key_names_subset))

    def _param_name_from_secondary_key_values(self, secondary_key_values, param_name_subset):

        # Construct param name from table secondary keys (optionally a subset of these)
        secondary_key_subset_map = self._get_secondary_key(secondary_key_values, param_name_subset)

        # Restrict to secondary keys we want in param name
        secondary_key_subset_map = {k: secondary_key_subset_map[k] for k in self._param_name_secondary_key_columns()}
        return self._make_param_name(secondary_key_subset_map)

    def get_insert_argument(self, secondary_key_values):
        """
        Make dictionary with all secondary keys for current table, as well as table param name
        Requires that table have single primary key that is param name
        :param secondary_key_values: all secondary key values, in same order as in table
        :return: dictionary for table insertion
        """

        # Return param name and all secondary key values together
        param_name_subset = False
        param_name = self._param_name_from_secondary_key_values(secondary_key_values, param_name_subset)
        secondary_key = self._get_secondary_key(secondary_key_values, param_name_subset)

        return {**{self.meta_param_name(): param_name}, **secondary_key}

    def lookup_param_name(
            self, secondary_key_values, as_dict=False, args_as_dict=False, tolerate_irrelevant_args=False):
        """
        Look up param name
        :param secondary_key_values: If args_as_dict is True: list of values of secondary keys in order used to
        construct param name. If args_as_dict is False: dictionary.
        :param as_dict: bool. If True, return {meta_param_name: param_name}, otherwise return param_name. Default
        is False.
        :param args_as_dict: If True, expect secondary_key_values to be dictionary.
        :param tolerate_irrelevant_args: bool, default False. Applies when args_as_dict is True. If True,
        tolerate passage of secondary keys that are not part of params table. If False, require that
        all passed secondary keys are in params table.
        :return: param name, optionally in dictionary (if as_dict is True).
        """

        # Get secondary key values from dictionary if passed
        if args_as_dict:
            secondary_key_columns = self._param_name_secondary_key_columns()
            # Check that all secondary key columns are present in passed dictionary if indicated
            if not tolerate_irrelevant_args:
                check_membership(secondary_key_columns, list(secondary_key_values.keys()))
            secondary_key_values = [secondary_key_values[x] for x in secondary_key_columns]
        param_name_subset = True
        param_name = self._param_name_from_secondary_key_values(secondary_key_values, param_name_subset)
        param_name_dict = {self.meta_param_name(): param_name}
        if len(self & param_name_dict) == 0:
            raise Exception(f"param name not found in {get_table_name(self)} for {secondary_key_values}")

        # Return as dictionary if indicated, otherwise return just param name
        if as_dict:
            return param_name_dict
        return param_name

    def get_default_param_name(self):
        return self.lookup_param_name(unpack_single_element(self._default_params()))

    def insert_entry(self, secondary_key_values, key_filter=None):

        # Insert from secondary key values
        insert_key = self.get_insert_argument(secondary_key_values)

        # Exit if key_filter passed and insert_key doesnt match
        if key_filter is not None:
            if not all({insert_key[k] == v for k, v in key_filter.items() if k in insert_key}):
                return

        # Otherwise insert into table
        self.insert1(insert_key)

    def _default_params(self):
        # Lists of secondary keys to insert into table
        return []

    def insert_defaults(self, **kwargs):
        key_filter = kwargs.pop("key_filter", None)
        for secondary_key_values in self._default_params():
            self.insert_entry(secondary_key_values, key_filter)

    def _get_param_name_table(self):
        # Get param name table. Error thrown if not present
        return get_table(unpack_single_element([x for x in get_upstream_table_names(self) if "ParamName" in x]))

    def get_params(self):
        return {k: self.fetch1(k) for k in get_table_secondary_key_names(self)}


class SelBase(dj.Manual):

    def _get_main_table(self):
        """
        Get "main" datajoint table for current selection table
        :return: datajoint table
        """

        # Get current table name
        table_name = get_table_name(self)

        # Raise error if current table doesnt following naming convention (require override of this method in that case)
        if table_name[-3:] != "Sel":
            raise Exception(
                f"Non canonical selection table name; must override this method for current table given this")

        # Return main table name. By convention, this has the same name as the current able without the final "Sel"
        return get_table(table_name[:-3])

    def _get_params_table(self):
        return self._get_main_table()()._get_params_table()

    def _get_potential_keys(self, key_filter=None):
        # Get potential keys from intersection of upstream tables
        return intersect_tables([get_table(x) for x in get_upstream_table_names(self)], key_filter).fetch("KEY")

    def insert_test(self):
        insert_manual_table_test_entry(self)
    
    def insert_defaults(self, **kwargs):

        # Copy key filter to avoid changing (note we perform a similar kind of copy prior to accessing this method
        # in the function populate_insert, but important to do here as well since dont always access this method
        # from that function)
        key_filter = copy.deepcopy(get_key_filter(kwargs))

        # Get potential keys from intersection of upstream tables
        keys = self._get_potential_keys(key_filter)

        # Insert each entry
        for key in keys:
            self.insert1(key, skip_duplicates=True)

    def insert1(self, key, **kwargs):

        if "skip_duplicates" not in kwargs:
            kwargs["skip_duplicates"] = True

        super().insert1(key, **kwargs)

    def alter(self, prompt=True, context=None):
        super().alter()

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)

    def _get_bad_keys(self):
        # Return "bad keys": keys in table that are not in potential_keys (works with primary keys)

        # Get key names in potential_keys (should be same for every entry of potential_keys and we check this
        # is the case, then extract one set)
        potential_keys = self._get_potential_keys()  # gets primary and secondary columns in table
        key_names = unpack_single_vector(np.asarray([np.asarray(list(x.keys())) for x in potential_keys]))
        key_names = [k for k in key_names if k in self.primary_key]

        # Get corresponding values to key names as array so can more quickly check for matches across
        # potential and actual keys
        potential_keys_arr = np.asarray([[potential_key[k] for k in key_names] for potential_key in potential_keys])

        # Get all table entries, restricting to subset of columns present in potential_keys entries
        table_keys = self.fetch("KEY")
        table_keys_arr = np.asarray([[key[k] for k in key_names] for key in table_keys])

        # Return keys in table that are not in potential_keys
        return [key for key_list, key in zip(table_keys_arr, table_keys) if not any(
            np.prod(key_list == potential_keys_arr, axis=1))]

    def cleanup(self, safemode=True):

        # Delete entries for keys not in list of potential keys
        bad_entries = self._get_bad_keys()
        keys = self.fetch("KEY")

        print(f"found {len(bad_entries)} bad entries of {len(keys)}")

        for key in bad_entries:
            self.delete_(key, safemode)

    def get_object_id_name(self, leave_out_object_id=False, unpack_single_object_id=True):
        return get_table_object_id_name(self, leave_out_object_id, unpack_single_object_id)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        return fetch1_dataframe(self, object_id_name, restore_empty_nwb_object, df_index_name)

    def fetch_nwb(self):
        return fetch_nwb(self)


class PartBase(dj.Part):

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        return fetch1_dataframe(self, object_id_name, restore_empty_nwb_object, df_index_name)
    
    def fetch_nwb(self):
        return fetch_nwb(self)

    def get_key(self, key):
        return intersect_tables([get_table(x) for x in get_upstream_table_names(self)], key).fetch1("KEY")


class ComputedBase(dj.Computed):

    def _initialize_upstream_entries_tracker(self):
        self.upstream_obj = UpstreamEntries()

    def _update_upstream_entries_tracker(self, table=None, key=None):

        # Initialize upstream object if non-existent or None
        if "upstream_obj" not in dir(self):
            self._initialize_upstream_entries_tracker()
        if self.upstream_obj is None:
            self._initialize_upstream_entries_tracker()

        self.upstream_obj.update_upstream_entries(table, key)

    def _merge_tracker(self, table):

        # Initialize upstream object if non-existent
        if "upstream_obj" not in dir(self):
            self._initialize_upstream_entries_tracker()

        # Merge upstream tracker from another table with that from the current table
        self.upstream_obj.merge_tracker(table)

    def _insert_part_tables(self, key, upstream_obj=None):
        # Insert into part tables

        # Exit if no upstream tracker
        if upstream_obj is None and not hasattr(self, "upstream_obj"):
            return

        # Otherwise if upstream tracker not passed, get as attribute
        elif upstream_obj is None:
            upstream_obj = self.upstream_obj

        # Insert into part tables
        upstream_entries = upstream_obj.remove_repeat_entries()
        for part_table_name, part_table_entries in upstream_entries.items():
            part_table = getattr(self, part_table_name)
            for part_key in part_table_entries:
                check_same_values_at_shared_keys([key, part_key])
                part_table.insert1({**key, **part_key})

    def _get_params_table(self):

        # Search for params table immediately upstream
        params_table = get_upstream_table(self, "Params")

        # If none, search for one upstream of selection table with expected name
        if params_table is None and self._get_selection_table() is not None:
            params_table_name = get_params_table_name(self)  # expected name of params table for current table
            if any(np.asarray(get_upstream_table_names(self._get_selection_table())) == params_table_name):
                params_table = get_table(params_table_name)

        return params_table

    def get_upstream_param(self, param_name):
        params_table = self._get_params_table()()
        return (params_table & self.fetch1("KEY")).get_params()[param_name]

    # More interpretable as method than as attribute since not all computed tables have params table
    def meta_param_name(self):

        if self._get_params_table() is None:
            raise Exception(f"table {self.table_name} has no params table, so cant get meta param name")

        return self._get_params_table()().meta_param_name()
    
    def _get_selection_table(self):
        # Get selection table
        return get_upstream_table(self, "Sel")

    def populate_(self, key=None, tolerate_error=False, populated_tables=None, recursive=False, **kwargs):

        verbose = True

        key = copy.deepcopy(key)  # copy key to avoid altering outside function

        # Recursively populate all upstream if indicated
        # Note that using datajoint free table did not work here because tables gotten with this seem to not have
        # populate method
        if populated_tables is None:
            populated_tables = []  # tables already tried populating
        if len(set(get_upstream_table_names(self)) - set(populated_tables)) < len(set(get_upstream_table_names(self))):
            print(f"recursive populate working (outer)")
        if recursive:
            for upstream_table_name in set(get_upstream_table_names(self)) - set(populated_tables):
                if upstream_table_name in populated_tables:
                    print('recursive populate working (inner)')
                    continue
                try:
                    upstream_table = get_table(upstream_table_name)
                    populated_tables.append(upstream_table)
                except Exception as error:
                    print(f"could not get table {upstream_table_name} during recursive populate for "
                          f"{get_table_name(self)}. Error: {error}")
                    upstream_table = None
                if upstream_table is not None:
                    populated_tables += populate_insert(upstream_table, key=key, tolerate_error=tolerate_error,
                                                        populated_tables=populated_tables, populate_all_upstream=recursive)

        # Otherwise just insert default arguments into params and selection tables if they exist
        else:
            kwargs = {"key_filter": key}
            params_table = self._get_params_table()
            if params_table is not None:
                if verbose:
                    print(f"inserting defaults into {get_table_name(params_table)}...")
                params_table().insert_defaults(**kwargs)
            selection_table = self._get_selection_table()
            if selection_table is not None:
                if verbose:
                    print(f"inserting defaults into {get_table_name(selection_table)}...")
                selection_table().insert_defaults(**kwargs)

        # Populate main table
        populate_flexible_key(self, key, tolerate_error, verbose=verbose)

        # Return list with populated tables so we can avoid running populate on these again
        return populated_tables

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        return fetch1_dataframe(self, object_id_name, restore_empty_nwb_object, df_index_name)

    def fetch1_dataframes(self):
        return fetch1_dataframes(self)

    def get_object_id_name(self, leave_out_object_id=False, unpack_single_object_id=True):
        return get_table_object_id_name(self, leave_out_object_id, unpack_single_object_id)

    def fetch_nwb(self, **kwargs):
        return fetch_nwb(self, **kwargs)

    @staticmethod
    def get_default_df_index_name(df_index_name, object_id_name, df_index_name_map):

        if object_id_name in df_index_name_map and df_index_name is None:
            df_index_name = df_index_name_map[object_id_name]

        return df_index_name

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)


class TemporalFrmapParamsBase(SecKeyParamsBase):

    def _default_params(self):
        return [[.1]]


class TemporalFrmapSmParamsBase(SecKeyParamsBase):

    def _default_params(self):
        return [[.1]]


class FrmapBase(ComputedBase):

    def make(self, key):
        raise Exception(f"Must implement this method in subclass")

    def fetch1_dataframe(self):
        return super().fetch1_dataframe().set_index('unit_id')

    def plot_rate_map(self, key, ax=None, color="black"):
        plot_datajoint_table_rate_map(self, key, ax, color)


class FrmapSmBase(ComputedBase):

    def make(self, key):
        # Local import to avoid circular import error
        from src.jguides_2024.datajoint_nwb_utils.datajoint_fr_table_helpers import smooth_datajoint_table_fr

        smoothed_firing_rate_map_df = smooth_datajoint_table_fr(self._get_firing_rate_map_table(),
                                                                self._get_params_table()(), key, self._data_type())
        insert_analysis_table_entry(self, [smoothed_firing_rate_map_df], key, [get_table_object_id_name(self)])

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="unit_id"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def plot_rate_map(self, key, ax=None, color="black"):
        plot_datajoint_table_rate_map(self, key, ax, color)

    def _get_firing_rate_map_table(self):
        split_table_name = get_table_name(self).split("Sm")
        check_length(split_table_name, 2)

        return get_table(split_table_name[0])

    @staticmethod
    def _data_type():
        raise Exception(f"Must define data type in subclass")


class EventTrialsParamsBase(SecKeyParamsBase):

    def trial_duration(self, param_name):
        return trial_duration_from_params_table(self, self.meta_param_name(), param_name)

    def trial_shifts(self, param_name=None):
        if param_name is None:
            param_name = self.fetch1(self.meta_param_name())
        key = {self.meta_param_name(): param_name}

        return map(int, (self & key).fetch1("trial_start_time_shift", "trial_end_time_shift"))


class WellEventTrialsBase(ComputedBase):

    def fetch1_dataframe(self, strip_s=False, index_column=None, **kwargs):

        # kwargs input makes easy to use this method with fetch1_dataframe methods that take in different
        # inputs (e.g. to fetch dataframe from nwb file)

        return fetch1_dataframe_from_table_entry(self, strip_s, index_column)

    def trial_intervals(self, trial_feature_map=None, restrictions=None, as_dict=False):

        # Get trials info
        trials_df = self.fetch1_dataframe()

        # Filter for certain trial features if indicated
        if trial_feature_map is not None:
            trials_df = df_filter_columns_isin(trials_df, trial_feature_map)

        # Define trial intervals
        trial_intervals = list(zip(trials_df["trial_start_times"], trials_df["trial_end_times"]))

        # Return as dictionary if indicated
        if as_dict:
            return {epoch_trial_number: trial_interval for epoch_trial_number, trial_interval in
                    zip(trials_df.index, trial_intervals)}

        # Otherwise return as list
        return trial_intervals


class WellEventTrialsBaseExt(WellEventTrialsBase):

    # Extension of WellEventTrialsBase

    def _event_info(self, start_event_name=None, end_event_name=None, start_event_idx_shift=None,
                    end_event_idx_shift=None, handle_trial_start_after_end=False):

        EventInfo = namedtuple(
            "EventInfo",
            "start_event_name end_event_name start_event_idx_shift end_event_idx_shift handle_trial_start_after_end")

        return EventInfo(start_event_name, end_event_name, start_event_idx_shift, end_event_idx_shift,
                         handle_trial_start_after_end)

    def _well_trial_characteristics(self, remove_trial_start_end_text=False):
        # Get characteristics of trials in dio well trials table that are used in current table

        # Some trials tables have trial start and end that are in two different "epoch trials". Here, trial
        # characteristics are defined separately for trial start and end by adding a prefix: "trial_start" or
        # "trial_end". We need to remove this to identify characteristics in current table that are in
        # dio well trials table
        def _remove_trial_start_end_text(x):
            return x.replace("trial_start_", "").replace("trial_end_", "")

        # Find characteristics
        characteristics = np.unique(
            [x for x in get_table_secondary_key_names(self) if _remove_trial_start_end_text(x) in
             get_table_secondary_key_names(get_table("DioWellTrials"))])

        # Return characteristics as they appear in dio well trials table if indicated
        if remove_trial_start_end_text:
            characteristics = [_remove_trial_start_end_text(x) for x in characteristics]

        # Otherwise return characteristics as they appear in current table
        return characteristics

    def make(self, key):

        # Update key with trial event info from well trials table
        # ...Define trial start and end time shifts
        start_time_shift, end_time_shift = self._get_params_table()().trial_shifts(key[self.meta_param_name()])
        # ...Get table specific information about definition of trials relative to events
        event_info = self.event_info()
        # ...Get names of trial characteristics we want to get values for
        trial_characteristics = self._well_trial_characteristics(remove_trial_start_end_text=True)
        key.update((get_table("DioWellTrials") & key).get_trial_event_info(
            event_info.start_event_name, event_info.end_event_name, start_time_shift, end_time_shift,
            event_info.start_event_idx_shift, event_info.end_event_idx_shift,
            trial_characteristics, event_info.handle_trial_start_after_end))

        # Update key with path names if indicated
        if "path_names" in get_table_secondary_key_names(self):
            # local import to avoid circular import error
            from src.jguides_2024.position_and_maze.jguidera_maze import RewardWellPath
            key.update({"path_names": RewardWellPath().get_path_names(
                key["trial_start_well_names"], key["trial_end_well_names"])})

        # Insert into table
        insert1_print(self, key)

    def rt_df_all_time(self, nwb_file_name, epoch, param_name, new_index=None):
        # TODO (feature): verify that this method can be moved to parent class
        """
        Get dataframe with time relative to event trials, when within event trials (nan otherwise),
        and other information
        :param nwb_file_name: string
        :param epoch: int
        :param param_name: string, value for meta param name for table
        :param new_index: optional new index along which to find the above
        :return: dataframe
        """

        # Define key for querying tables
        key = {"nwb_file_name": nwb_file_name, "epoch": epoch, self.meta_param_name(): param_name}

        # Use position_and_maze data as index if none passed
        time_vector = new_index  # default
        if new_index is None:
            # Local import to avoid circular import error
            from src.jguides_2024.position_and_maze.datajoint_position_table_helpers import fetch1_IntervalPositionInfo
            time_vector = fetch1_IntervalPositionInfo(nwb_file_name, epoch).index

        # Get time relative to trial start time, within trials
        trial_start_time_shift, trial_end_time_shift = (self._get_params_table() & key).trial_shifts()

        # Get trial intervals
        trial_intervals = (self._get_params_table() & key).trial_intervals()

        # Get times in trials relative to trial start
        relative_time_in_trial = get_full_event_times_relative_to_trial_start(
            time_vector, trial_intervals, trial_start_time_shift)

        # For each trial, get boolean indicating which times in time_vector are in trial
        trials_valid_bools = [event_times_in_intervals_bool(time_vector, [trial_interval]) for trial_interval in
                             trial_intervals]

        # For each trial characteristic, fill vector across time with value of that characteristic in each trial
        trials_df = (self & key).fetch1_dataframe().reset_index()
        extra_info_dict = {column_name: fill_trial_values(trials_df[column_name], trials_valid_bools) for column_name in
                           self._well_trial_characteristics()}

        return pd.DataFrame.from_dict({**extra_info_dict,
                                       **{"time": relative_time_in_trial.index,
                                        "relative_time_in_trial": relative_time_in_trial.values}}).set_index("time")


class TrialsTimeBinsParamsBase(SecKeyParamsBase):

    def _default_params(self):
        return [[x] for x in [.02, .1]]

    def get_time_bin_width(self):
        return float(self.fetch1("time_bin_width"))


class ParamNameBase(dj.Manual):

    def cleanup(self):
        # Delete entries that have no corresponding entry in params table
        params_table = self._get_params_table()
        for key in self.fetch("KEY"):
            if len(params_table & key) == 0:
                (self & key).delete()

    def _get_insertion_key(self, full_param_name, use_full_param_name=False, filter_key=None):
        # Get key for insertion into a "param name" table (a table used to restrict insertion into a params table)
        # Multiple options for defining abbreviated param name:
        # 1) use full param name
        # 2) use an integer. Here, have the option of considering a restricted set of table via filter_key

        # Option 1:
        if use_full_param_name:
            # Define integer id to be none, since not using
            int_id = None
            # Define param name as full param name
            param_name = full_param_name

        # Optional 2:
        else:
            # Get inputs if not passed
            if filter_key is None:
                filter_key = dict()

            # Get int_id
            # If entry exists in table for passed full param name, use int_id from this
            int_id = fetch1_tolerate_no_entry(self & {"full_param_name": full_param_name},
                                              "int_id")  # return None if no entry

            # Otherwise, define int_id as next available integer after existing integer ids. We will insert this in
            # table so that standardized code can identify next int_id from existing entries in table
            # We also use this as the param name for the param name table
            if int_id is None:
                int_id = get_next_int_id(self & filter_key)

            # Return key to insert into param name table
            # First, make text from filter key
            filter_text = "_".join(list(filter_key.values()))  # join params with underscore
            filter_text += "_" * (len(filter_text) > 0)  # add trailing underscore if joined filter key not empty
            param_name = filter_text + str(int_id)

        return {get_meta_param_name(self): param_name, "int_id": int_id, "full_param_name": full_param_name}

    def insert1(self, key, **kwargs):

        # First delete existing entries that have no corresponding entry in params table, to avoid skipping insertion
        # here because of orphan entry
        self.cleanup()

        # Add skip_duplicates = True if not in keyword arguments
        if "skip_duplicates" not in kwargs:
            kwargs["skip_duplicates"] = True

        super().insert1(key, **kwargs)
    
    def get_full_param_name(self, **kwargs):
        raise Exception(f"Must implement in subclass")

    def lookup_param_name(self, full_param_name=None, as_dict=False, **kwargs):
        return self.get_insert_param_name(full_param_name, as_dict, insert=False, **kwargs)

    def get_insert_param_name(self, full_param_name=None, use_full_param_name=False, filter_key=None, as_dict=False,
                              insert=True, tolerate_no_entry=False, **kwargs):
        # Get param name, optionally insert into table, and return

        # Get full param name if not passed
        if full_param_name is None:
            full_param_name = self.get_full_param_name(**kwargs)

        # Insert into table if indicated
        if insert:
            key = self._get_insertion_key(full_param_name, use_full_param_name, filter_key)
            self.insert1(key)

        # Get param name. Tolerate param name not existing if indicated
        meta_param_name = get_meta_param_name(self)
        param_name = special_fetch1(self, meta_param_name, {"full_param_name": full_param_name}, tolerate_no_entry)

        # Return param name
        # ...in dictionary with meta param name if indicated
        if as_dict:
            return {meta_param_name: param_name}
        # ...otherwise just param name
        return param_name

    def _get_params_table(self):

        # Get params table that corresponds to the current param name table
        table_name = get_table_name(self)
        params_table_name = table_name.replace("ParamName", "Params")
        if params_table_name not in get_downstream_table_names(self):
            raise Exception(f"params table not found for {table_name}")

        return get_table(params_table_name)


def get_upstream_table(table, target_string):

    table_names = [x for x in get_upstream_table_names(table) if target_string in x]

    if len(table_names) == 0:
        return None
    return get_table(unpack_single_element(table_names))


def _table_function_loop(fn, schema_names=None, key=None):

    # Get all schema names if none passed
    if schema_names is None:
        schema_names = get_schema_names()

    # If string passed, treat as a schema name and package into list
    if isinstance(schema_names, str):
        schema_names = [schema_names]

    # Define key if not passed
    if key is None:
        key = {}
    # Loop through tables in schema and print characteristics for key
    for schema_name in schema_names:
        print(f"\nIn schema {schema_name}")
        for table_name in get_schema_table_names_from_file(schema_name):
            table_subset = (get_table(table_name) & key)
            fn(table_subset)


# Print table characteristics
def _print_table_entries(table_subset):
    print(f"entries in {get_table_name(table_subset)}: {table_subset}")


def _print_table_primary_key_len(table_subset):
    print(f"{len(table_subset.primary_key)} {get_table_name(table_subset)}")


def _print_table_object_id(table):
    object_id_name = get_table_object_id_name(table, tolerate_none=True, unpack_single_object_id=True)
    if object_id_name is not None:
        print(object_id_name)


def print_table_entries(schema_names=None, key=None):
    _table_function_loop(_print_table_entries, schema_names, key)


def print_table_primary_key_len(schema_names=None):
    _table_function_loop(_print_table_primary_key_len, schema_names)


def print_table_object_id(schema_names=None):
    _table_function_loop(_print_table_object_id, schema_names)


# Check table characteristics
def _check_table_object_ids(table):

    object_id_name = get_table_object_id_name(
        table, leave_out_object_id=True, tolerate_none=True, unpack_single_object_id=True)
    if object_id_name is not None:
        table_name = camel_to_snake_case(get_table_name(table), group_uppercase=True)
        if table_name != object_id_name:
            print(table_name, object_id_name)


def check_table_object_ids(schema_names=None):
    _table_function_loop(_check_table_object_ids, schema_names)


def _check_table_param_names(table):
    table_name = get_table_name(table)
    if table_name[-6:] == "Params":
        expected_param_name = f"{camel_to_snake_case(table_name[:-6], group_uppercase=True)}_param_name"
        if expected_param_name not in table.primary_key:
            print(f"{expected_param_name} not in {table_name}")


def check_table_param_names(schema_names=None):
    _table_function_loop(_check_table_param_names, schema_names)


class CohortBase(ComputedBase):

    def _get_upstream_table_name(self):
        # Gets single upstream table from which cohort formed. Assumes exactly one upstream table
        # and CohortEntries part table dependent on it
        return unpack_single_element(
            [x for x in get_upstream_table_names(self.CohortEntries) if x != get_table_name(self)])

    def _get_upstream_table(self):
        return get_table(self._get_upstream_table_name())

    def get_cohort_keys(self):
        # Get keys to cohort entries parts table for a cohort

        # Check that passed key specifies a single cohort
        check_single_table_entry(self)

        return fetch_entries_as_dict(self.CohortEntries & self.fetch1("KEY"))

    def _fetch(self, iterable_name, fetch_function_name, **kwargs):
        # Helper function to get data for entries in a cohort

        keys = self.get_cohort_keys()

        # Return data in dictionary with iterable as key
        # ...First check no duplicate keys. If there are, entries overwrite each other in dictionary creation
        check_all_unique([key[iterable_name] for key in keys])

        return {key[iterable_name]:
                getattr(self._get_upstream_table() & key, fetch_function_name)(**kwargs) for key in keys}

    def fetch_dataframes(self, **kwargs):
        # Get params. We get from kwargs rather than specifying in method definition so that overriding methods
        # can accept different inputs from each other

        iterable_name = kwargs.pop("iterable_name")
        fetch_function_name = kwargs.pop("fetch_function_name", "fetch1_dataframe")
        object_id_name = kwargs.pop("object_id_name", None)
        concatenate = kwargs.pop("concatenate", True)
        df_index_name = kwargs.pop("df_index_name", None)
        add_iterable = kwargs.pop("add_iterable", True)
        axis = kwargs.pop("axis", 0)

        # Check that single entry in table
        check_length(self, 1)

        # Fetch dfs for a cohort
        dfs_dict = self._fetch(
            iterable_name, fetch_function_name, object_id_name=object_id_name, df_index_name=df_index_name)

        # If indicated, concatenate across iterables (e.g. epochs), optionally adding iterable as a column,
        # and optionally setting index
        if concatenate:
            # Add iterable as column to dfs if indicated
            if add_iterable:
                df_list = [add_column_to_df(df, k, iterable_name) for k, df in dfs_dict.items()]
            else:
                df_list = [df for df in dfs_dict.values()]
            # Concatenate dfs
            df_concat = pd.concat(df_list, axis=axis)
            # Set df index if indicated
            if df_index_name is not None:
                df_concat = df_concat.set_index(df_index_name)
            return df_concat

        # Otherwise return as dictionary
        return dfs_dict


class AcrossFRVecTypeTableSelBase(SelBase):

    @staticmethod
    def _fr_vec_table():
        raise Exception(f"This method must be overwritten in child class")

    # Takes a long time to run
    def _get_potential_keys(self, key_filter=None):

        from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel, \
            ResTimeBinsPoolCohortParams

        if key_filter is None:
            key_filter = dict()
        # Limit to single epochs with desired time bins params
        keys = []
        for shorthand_param_name in ["epoch_100ms",  # embeddings, firing rate vector analysis
                                     "path_100ms",  # path embeddings
                                    ]:
            res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand(
                    shorthand_param_name)
            res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand(
                    shorthand_param_name)
            # note that the below lookup is slow, so best to not iterate through multiple times
            key_filter.update({"res_time_bins_pool_param_name": res_time_bins_pool_param_name})
            for idx, key in enumerate((self._fr_vec_table() & key_filter).fetch("KEY")):
                epoch = key.pop("epoch")
                key["epochs_id"] = get_epochs_id([epoch])
                key.pop("res_time_bins_pool_param_name")
                key.update({"epochs_id": get_epochs_id([epoch]),
                            "res_time_bins_pool_cohort_param_name": res_time_bins_pool_cohort_param_name})
                keys.append(key)

        return keys


class CovariateRCB(ComputedBase):

    def make(self, key):
        raise Exception(f"This method must be overitten in child class")

    def visualize_result(self):
        plot_params = {"xlabel": "sample num", "ylabel": "basis function num"}
        plot_heatmap(self.fetch1_dataframe().to_numpy().T, **plot_params)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        if df_index_name is None:
            df_index_name = "time"
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def delete_(self, key, safemode=True):
        from src.jguides_2024.glm.jguidera_measurements_interp_pool import XInterpPool
        delete_(self, [XInterpPool], key, safemode)


class CovariateDigParamsBase(SecKeyParamsBase):

    def bin_width_name(self):
        return unpack_single_element([x for x in get_table_secondary_key_names(self) if "bin_width" in x])

    def get_bin_centers(self, **kwargs):
        if "bin_width" in kwargs:
            raise Exception(f"bin_width should not be passed as it is defined here based on table entry")
        table_subset = self
        if "key" in kwargs:
            table_subset = (self & kwargs["key"])
        kwargs["bin_width"] = table_subset.fetch1(self.bin_width_name())
        return vector_midpoints(self.make_bin_edges(**kwargs))

    def get_num_bin_edges(self, **kwargs):
        if "bin_width" in kwargs:
            raise Exception(f"kwargs should not contain bin_width as this is defined based on table entry")
        table_subset = self
        if "key" in kwargs:
            table_subset = (self & kwargs["key"])
        kwargs["bin_width"] = table_subset.fetch1(self.bin_width_name())
        return len(self.make_bin_edges(**kwargs))

    def get_num_bins(self, **kwargs):
        return self.get_num_bin_edges(**kwargs) - 1

    def get_valid_bin_nums(self, **kwargs):
        # Bin nums depends on how table handles negative values, so require custom function in each child class
        raise Exception(f"This method should be overridden in child class")

    def make_bin_edges(self, **kwargs):
        raise Exception(f"This method should be overridden in child class")


class CovDigmethBase(ComputedBase):
    """
    Table with covariate values and a method for digitizing those values
    """

    @staticmethod
    def _covariate_name():
        raise Exception(f"This method must be overwritten in child class")

    @staticmethod
    def get_range():
        raise Exception(f"This method must be overwritten in child class")

    @classmethod
    def make_bin_edges(cls, **kwargs):
        if "bin_width" not in kwargs:
            raise Exception(f"bin_width must be passed")
        return make_bin_edges(cls.get_range(), kwargs["bin_width"])

    def fetch1_dataframe_exclude(self, exclusion_params=None, object_id_name=None, restore_empty_nwb_object=True,
                                 df_index_name=None):
        # Return df with exclusion criteria applied

        raise Exception(f"This method must be overwritten in child class")

    def digitized(self, bin_width=None, bin_edges=None, exclusion_params=None, verbose=False):

        # Check inputs
        check_one_none([bin_width, bin_edges], ["bin_width", "bin_edges"])

        # Make bin edges if not passed
        if bin_edges is None:
            bin_edges = self.make_bin_edges(bin_width=bin_width)

        # Get covariate
        cov_df = self.fetch1_dataframe_exclude(exclusion_params=exclusion_params)

        # Digitize relative time
        covariate_name = self._covariate_name()
        cov_df[f"digitized_{covariate_name}"] = digitize_indexed_variable(
            indexed_variable=cov_df[covariate_name], bin_edges=bin_edges, right=True, verbose=verbose)

        return cov_df

