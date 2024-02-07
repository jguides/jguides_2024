import datajoint as dj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spyglass.common import Nwbfile, Session

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    insert1_print, delete_
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import get_nwb_file, events_in_epoch_bool
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.utils.df_helpers import df_filter_columns_contains, df_pop
from src.jguides_2024.utils.list_helpers import return_n_empty_lists, check_alternating_elements, \
    check_return_single_element
from src.jguides_2024.utils.plot_helpers import format_ax
from src.jguides_2024.utils.point_process_helpers import not_small_diff_bool, event_times_in_intervals_bool
from src.jguides_2024.utils.vector_helpers import remove_repeat_elements

schema = dj.schema("jguidera_dio_event")


@schema
class DioEvents(ComputedBase):
    definition = """
    # DIO events recorded at full sampling rate
    -> TaskIdentification
    ---
    -> nd.common.AnalysisNwbfile
    dio_events_object_id : varchar(40)
    """

    def make(self, key):
        # Get DIO events for this epoch from nwb file
        nwbf = get_nwb_file(Nwbfile().get_abs_path(key['nwb_file_name']))
        nwbf_dios = nwbf.fields["processing"]["behavior"]["behavioral_events"].fields["time_series"]
        dio_event_values_list, dio_event_times_list, dio_descriptions = return_n_empty_lists(3)
        for dio_name, dios in nwbf_dios.items():  # for each DIO type
            # Filter for events in epoch
            dio_event_times = np.asarray(dios.fields["timestamps"])
            valid_bool = events_in_epoch_bool(nwb_file_name=key["nwb_file_name"],
                                              epoch=key["epoch"],
                                              event_times=dio_event_times)
            dio_event_values_list.append(np.asarray(dios.fields["data"])[valid_bool])  # DIO event values
            dio_event_times_list.append(dio_event_times[valid_bool])  # DIO event times
            dio_descriptions.append(dios.fields["description"])
        dio_event_df = pd.DataFrame.from_dict({"dio_name": list(nwbf_dios.keys()),
                                               "dio_description": dio_descriptions,
                                               "dio_int": [convert_dio_description(x) for x in dio_descriptions],
                                               "dio_event_times": dio_event_times_list,
                                               "dio_event_values": dio_event_values_list})
        # Store
        insert_analysis_table_entry(self, [dio_event_df], key, ["dio_events_object_id"])

    def fetch1_dataframe(self):
        return super().fetch1_dataframe().set_index("dio_int")

    def plot_dios(self, nwb_file_name, epoch):
        df = (self & {"nwb_file_name": nwb_file_name,
                      "epoch": epoch}).fetch1_dataframe()
        fig, axes = plt.subplots(len(df), 1, sharex=True, figsize=(15, 2*len(df)))
        fig.tight_layout()
        for (_, df_row), ax in zip(df.iterrows(), axes):
            ax.plot(df_row["dio_event_times"], df_row["dio_event_values"], ".")
            format_ax(ax, title=f"{df_row['dio_name']} {df_row['dio_description']}")

    def process_dio_events(self, nwb_file_name, epoch, diff_threshold, verbose=True):
        """
        "Process" dio events:
        1) Exclude dio ticks very close in time
        2) Exclude ticks of "dio n - 1" that are within dio up periods of "dio n" (to get rid of fast ticks)
        :param nwb_file_name:
        :param epoch:
        :return:
        """

        dio_df = (self & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1_dataframe()
        num_excluded_dio_events_dict = {
            dio_name: 0 for dio_name in dio_df["dio_name"]}  # track number of excluded dio events
        dio_poke_ints = df_filter_columns_contains(dio_df, "dio_name", "_poke").index  # dio pokes

        # Stage 1: exclude dio ticks very close in time
        # Refactored to avoid pandas slicing warning (chained indexing)
        for dio_int in dio_poke_ints:
            df_subset = dio_df.loc[dio_int]
            valid_bool = not_small_diff_bool(df_subset.dio_event_times, diff_threshold=diff_threshold)
            new_dio_event_times, new_dio_event_values = return_n_empty_lists(2)  # initialize
            if len(df_subset.dio_event_times) > 0:  # if dio events
                new_dio_event_times = df_subset.dio_event_times[valid_bool]
                new_dio_event_values = df_subset.dio_event_values[valid_bool]
                check_alternating_elements(new_dio_event_values, 0, 1)
                num_excluded_dio_events_dict[df_subset["dio_name"]] += np.sum(np.invert(valid_bool))
            # Replace event times and values
            df_subset["dio_event_times"] = np.asarray(new_dio_event_times)
            df_subset["dio_event_values"] = np.asarray(new_dio_event_values)

        # Stage 2: exclude ticks of "dio n - 1" that are within dio up periods of "dio n"
        df_pokes = df_filter_columns_contains(dio_df, "dio_name", "_poke")  # dio pokes
        new_dio_event_times_dict = {
            df_pokes.index[-1]: df_pokes.iloc[-1].dio_event_times}  # initialize with "dio N"
        new_dio_event_values_dict = {
            df_pokes.index[-1]: df_pokes.iloc[-1].dio_event_values}  # initialize with "dio N"
        # Traverse df in reverse order (from "dio N" to "dio 1"), and on the iteration for "dio x"
        # exclude ticks from "dio x - 1" (this is why we dont include "dio 0" in our loop)
        for dio_int in df_pokes.index[1::][::-1]:
            # Continue if no "dio x - 1"
            if dio_int - 1 not in df_pokes.index:
                continue
            # Continue if "dio x - 1" has no events
            if len(df_pokes.loc[dio_int - 1].dio_event_times) == 0:
                continue
            df_row = df_pokes.loc[dio_int]  # df row for current dio
            dio_event_values = df_row.dio_event_values  # event values for current dio
            dio_event_times = df_row.dio_event_times  # event times for current dio
            # Get dio up periods: from one dio up to the next dio down
            dio_up_idxs = np.where(dio_event_values == 1)[0]
            if len(dio_up_idxs) == 0:  # continue if no dio ups
                continue
            dio_down_idxs = np.where(dio_event_values == 0)[0]
            dio_up_end_idxs = dio_down_idxs[dio_down_idxs > dio_up_idxs[0]]
            if check_return_single_element(dio_up_end_idxs - dio_up_idxs[:len(dio_up_end_idxs)]).single_element != 1:
                raise Exception(f"dio up end idxs should be one more than dio up idxs")
            dio_up_periods = list(zip(dio_event_times[dio_up_idxs], dio_event_times[dio_up_end_idxs]))
            # Retain dio events on dio one less on breakout board which do not fall within up periods of current dio
            invalid_bool = event_times_in_intervals_bool(df_pokes.loc[dio_int - 1].dio_event_times,
                                                         dio_up_periods)
            valid_bool = np.invert(invalid_bool)
            new_dio_event_times = df_pokes.loc[dio_int - 1].dio_event_times[valid_bool]
            new_dio_event_values = df_pokes.loc[dio_int - 1].dio_event_values[valid_bool]
            check_alternating_elements(new_dio_event_values, 0, 1)
            new_dio_event_times_dict[dio_int - 1] = new_dio_event_times
            new_dio_event_values_dict[dio_int - 1] = new_dio_event_values
            num_excluded_dio_events_dict[df_pokes.loc[dio_int - 1]["dio_name"]] += np.sum(
                invalid_bool)  # update excluded dio counter
            # Plot
            if verbose:
                fig, ax = plt.subplots(figsize=(15, 2))
                ax.plot(dio_event_times, dio_event_values, label=f"dio {dio_int}")
                label = None
                for period_idx, dio_up_period in enumerate(dio_up_periods):
                    if period_idx == len(dio_up_periods) - 1:
                        label = f"dio {dio_int} UP periods"
                    ax.plot(dio_up_period, [1] * 2, color="red", alpha=1, label=label)
                ax.plot(dio_df.loc[dio_int - 1].dio_event_times,
                        dio_df.loc[dio_int - 1].dio_event_values + 1, label=f"ORIGINAL dio {dio_int - 1}")
                ax.plot(new_dio_event_times, new_dio_event_values + 1, label=f"NEW dio {dio_int - 1}")
                format_ax(ax, title=f"{nwb_file_name} ep{epoch}")
                ax.legend()

        # Replace event times and values in df with all dios
        for dio_int, dio_event_times in new_dio_event_times_dict.items():  # must use loop to replace using [row_indexer, col_indexer]
            dio_df.loc[dio_int, "dio_event_times"] = dio_event_times
        for dio_int, dio_event_values in new_dio_event_values_dict.items():
            dio_df.loc[dio_int, "dio_event_values"] = dio_event_values

        # Add number of excluded dio events to df
        dio_df["num_excluded_dio_events"] = [num_excluded_dio_events_dict[dio_name] for dio_name in dio_df["dio_name"]]

        return dio_df


@schema
class ProcessedDioEvents(ComputedBase):
    definition = """
    # Processed DIO events
    -> DioEvents
    ---
    diff_threshold : float
    num_excluded_dio_events : blob
    -> nd.common.AnalysisNwbfile
    processed_dio_events_object_id : varchar(40)
    """

    class Pokes(dj.Part):
        definition = """
        # DIO well poke events
        -> ProcessedDioEvents
        ---
        dio_poke_names : blob
        dio_poke_times : blob
        dio_poke_values : blob
        """

    class FirstUpPokes(dj.Part):
        definition = """
        # DIO well poke events with consecutive up pokes (after first) at same well removed
        -> ProcessedDioEvents.Pokes
        ---
        dio_first_poke_names : blob
        dio_first_poke_times : blob   
        dio_first_poke_values : blob        
        """

    class LastDownPokes(dj.Part):
        definition = """
        # DIO well poke events with consecutive down pokes (until last) at same well removed
        -> ProcessedDioEvents.Pokes
        ---
        dio_last_poke_names : blob
        dio_last_poke_times : blob 
        dio_last_poke_values : blob        
        """

    class Pumps(dj.Part):
        definition = """
        # DIO pump events
        -> ProcessedDioEvents
        ---
        dio_pump_names : blob
        dio_pump_times : blob
        dio_pump_values : blob
        """

    def make(self, key):
        # Hard code diff_threshold
        diff_threshold = .001

        # Unpack key
        nwb_file_name = key["nwb_file_name"]
        epoch = key["epoch"]

        # Put all processed dio events into a single df
        dio_df = DioEvents().process_dio_events(nwb_file_name, epoch, diff_threshold, verbose=True)
        dio_names = dio_df["dio_name"]  # DIO event names
        dict_temp = {"dio_event_names": [],
                     "dio_event_times": [],
                     "dio_event_values": []}  # initialize dictionary to store all dio events across epochs
        for dio_name in dio_names:  # for each dio poke event
            dio_event_times = df_pop(dio_df, {"dio_name": dio_name}, "dio_event_times")
            dio_event_values = df_pop(dio_df, {"dio_name": dio_name}, "dio_event_values")
            dict_temp["dio_event_names"] += [dio_name] * len(dio_event_times)
            dict_temp["dio_event_times"] += list(dio_event_times)
            dict_temp["dio_event_values"] += list(dio_event_values)
        all_dio_df = pd.DataFrame.from_dict(dict_temp)
        all_dio_df.set_index("dio_event_times", inplace=True)
        all_dio_df.sort_index(inplace=True)

        # Insert into table
        full_key = {**key, **{"num_excluded_dio_events":
                                  dio_df.set_index("dio_name")["num_excluded_dio_events"].to_dict(),
                              "diff_threshold": diff_threshold}}
        all_dio_df_reset_index = all_dio_df.reset_index()  # index does not save out to analysis nwbf so must reset
        insert_analysis_table_entry(self, [all_dio_df_reset_index], full_key, ["processed_dio_events_object_id"])

        # Now prepare to populate subtables for pokes
        # Filter for dio poke events
        poke_dio_names = [dio_name for dio_name in dio_names if "poke" in dio_name]  # get names of poke dio events
        dio_pokes_df = all_dio_df[all_dio_df["dio_event_names"].isin(poke_dio_names)]

        # Get first dio up event in series of consecutive up events at same well
        dio_pokes_ups_df = dio_pokes_df[dio_pokes_df["dio_event_values"] == 1]  # filter for up events
        _, idxs = remove_repeat_elements(dio_pokes_ups_df["dio_event_names"],
                                         keep_first=True)  # find consecutive pokes at same well (after first)
        dio_pokes_first_ups_df = dio_pokes_ups_df.iloc[idxs]  # remove consecutive pokes at same well

        # Get last DIO down events in series of consecutive down events at same well
        # Only consider DIO down events that happen after first DIO up event
        dio_pokes_downs_df = dio_pokes_df[dio_pokes_df["dio_event_values"] == 0]  # filter for down events
        # Initialize variable for first up time
        if len(dio_pokes_downs_df) > 0:  # if down events
            first_up_time = dio_pokes_downs_df.index[-1]  # initialize variable for first up time to last down event
        else:
            first_up_time = -1  # value will not matter since no down pokes to filter
        if len(dio_pokes_first_ups_df) > 0:  # if up events
            first_up_time = dio_pokes_first_ups_df.index[0]  # get time of first up event
        dio_pokes_downs_df = dio_pokes_downs_df[
            dio_pokes_downs_df.index > first_up_time]  # filter for down events after first up event
        _, idxs = remove_repeat_elements(dio_pokes_downs_df["dio_event_names"],
                                         keep_first=False)  # find consecutive pokes at same well (before last)
        dio_pokes_last_downs_df = dio_pokes_downs_df.iloc[idxs]  # remove consecutive pokes at same well

        # Check that same well visits found for first dio ups and last dio downs, tolerating having one less down than
        # up event (since recording can be stopped during a dio up)
        if len(dio_pokes_first_ups_df) - len(dio_pokes_last_downs_df) not in [0, 1]:
            raise Exception(
                f"Should have found either zero or one more dio up events than dio down events, but found "
                f"{len(dio_pokes_first_ups_df) - len(dio_pokes_last_downs_df)}")
        if not all(dio_pokes_last_downs_df["dio_event_names"].values ==
                   dio_pokes_first_ups_df["dio_event_names"].iloc[:len(dio_pokes_last_downs_df)].values):
            raise Exception(f"Not all well identities the same for first dio ups and last dio downs")
        # Check that each dio down after same index dio up and before next index dio up
        if not np.logical_and(all((dio_pokes_last_downs_df["dio_event_names"].index[:-1] -
                                   dio_pokes_first_ups_df["dio_event_names"].iloc[:len(dio_pokes_last_downs_df)].index[
                                   1:]) < 0),
                              all((dio_pokes_last_downs_df["dio_event_names"].index -
                                   dio_pokes_first_ups_df["dio_event_names"].iloc[
                                   :len(dio_pokes_last_downs_df)].index) > 0)):
            raise Exception(f"At least one dio down is not after same index dio up and next index dio up")

        # Populate subtable for well pokes
        insert1_print(self.Pokes,
                      {**key, **{"dio_poke_names": dio_pokes_df["dio_event_names"].to_numpy(),
                                 "dio_poke_times": dio_pokes_df.index.to_numpy(),
                                 "dio_poke_values": dio_pokes_df["dio_event_values"].to_numpy()}})

        # Populate subtable for first well up pokes
        insert1_print(self.FirstUpPokes,
                      {**key, **{"dio_first_poke_names": dio_pokes_first_ups_df["dio_event_names"].to_numpy(),
                                 "dio_first_poke_times": dio_pokes_first_ups_df.index.to_numpy(),
                                 "dio_first_poke_values": dio_pokes_first_ups_df["dio_event_values"].to_numpy()}})

        # Populate subtable for last well down pokes
        insert1_print(self.LastDownPokes,
                      {**key, **{"dio_last_poke_names": dio_pokes_last_downs_df["dio_event_names"].to_numpy(),
                                 "dio_last_poke_times": dio_pokes_last_downs_df.index.to_numpy(),
                                 "dio_last_poke_values": dio_pokes_last_downs_df["dio_event_values"].to_numpy()}})

        # Populate subtable for pump events
        pump_dio_names = [dio_name for dio_name in dio_names if "pump" in dio_name]  # filter for pump dio events
        dio_pumps_df = all_dio_df[all_dio_df["dio_event_names"].isin(pump_dio_names)]
        # Only consider dio down events that happen after first dio up event
        valid_idxs = []  # initialize valid idxs to empty list
        if np.sum(dio_pumps_df["dio_event_values"] == 1) > 0:  # if dio up events
            idx_first_up = np.where(dio_pumps_df["dio_event_values"] == 1)[0][0]
            valid_idxs = np.arange(idx_first_up, len(dio_pumps_df))
        dio_pumps_df = dio_pumps_df.iloc[valid_idxs]
        insert1_print(self.Pumps, {**key, **{"dio_pump_names": dio_pumps_df["dio_event_names"].to_numpy(),
                                             "dio_pump_times": dio_pumps_df.index.to_numpy(),
                                             "dio_pump_values": dio_pumps_df["dio_event_values"].to_numpy()}})

    def populate_(self, **kwargs):
        # Exit if nwb file name in key and does not correspond to a run epoch
        if "key" in kwargs:
            key = kwargs["key"]
            if key is not None:
                if "nwb_file_name" in key:
                    if key["nwb_file_name"] not in RunEpoch.fetch("nwb_file_name"):
                        print(f"Only populating ProcessedDioEvents for run epochs. Continuing...")
                        return
        super().populate_(**kwargs)

    def fetch1_dataframe(self):
        return super().fetch1_dataframe().set_index("dio_event_times")

    def delete_(self, key=None, safemode=True):
        from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPool
        from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt
        delete_(self, [TrialsPool, Ppt], key, safemode)


@schema
class PumpDiosComplete(ComputedBase):
    definition = """
    # Indicate whether dio pumps complete (denoted by 1) or incomplete (denoted by 0) for epochs
    -> TaskIdentification
    ---
    dio_pumps_complete : int
    """

    def make(self, key):
        dio_pumps_complete = 0  # initialize to incomplete
        subject_id = (Session() & {"nwb_file_name": key["nwb_file_name"]}).fetch1("subject_id")
        task_environment, contingency = (TaskIdentification & key).fetch1("task_environment", "contingency")
        if any([subject_id not in ["peanut", "J16"],
                np.logical_and(subject_id == "fig",
                               int(key["nwb_file_name"].split(subject_id)[1].split("_")[0]) > 20211106),
                task_environment in ["HaightRight", "SleepBox", "HomeBox"],
                contingency == "handleAlternation"]):  # conditions for which dio pumps complete
            dio_pumps_complete = 1  # indicate that dio pumps complete

        # Insert into table
        insert1_print(self, {**key, **{"dio_pumps_complete": dio_pumps_complete}})


def convert_dio_description(dio_description, convert_to_type=None):
    """
    Convert dio description to/from "Dio{x}" (string) and x (int)
    :param dio_description:
    :param convert_to_type:
    :return:
    """

    # Check inputs
    valid_types = ["int", "string", None]
    if convert_to_type not in valid_types:
        raise Exception(f"convert_to_type must be in {valid_types}")
    # Get digital in as integer (helpful for all cases below)
    if isinstance(dio_description, str):
        dio_description_int = int(dio_description.split("Din")[-1].split("Dout")[-1])
    else:
        dio_description_int = int(dio_description)
    # Return in desired form
    if convert_to_type == "int" or (convert_to_type is None and isinstance(dio_description, str)):
        return dio_description_int
    elif convert_to_type == "string" or (convert_to_type is None and isinstance(dio_description, int)):
        return f"Din{dio_description_int}"
    else:
        raise Exception(f"No valid conditions met to convert digital input")


def populate_jguidera_dio_event(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_dio_event"
    upstream_schema_populate_fn_list = None
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_dio_event():
    schema.drop()
