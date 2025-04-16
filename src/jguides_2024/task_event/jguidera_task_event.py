# This module defines tables related to task performance that depend on both DIO and statescript information
import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.task_event.jguidera_dio_event import (PumpDiosComplete, ProcessedDioEvents,
                                                            populate_jguidera_dio_event)
from src.jguides_2024.task_event.jguidera_statescript_event import ProcessedStatescriptEvents

# Needed for table definitions
ProcessedStatescriptEvents
ProcessedDioEvents

schema = dj.schema("jguidera_task_event")  # define custom schema


@schema
class EventNamesMapDioStatescript(dj.Manual):

    definition = """
    # Map between statescript event names and DIO event names
    dio_event_name : varchar(40)
    ---
    statescript_event_name : varchar(40)
    """

    def insert_defaults(self, **kwargs):
        for dio_event_name, statescript_event_name in {'HaightLeft_poke_center_SA_poke_center': 'center_poke',
                                                       'HaightLeft_poke_handle_SA_poke_handle': 'handle_poke',
                                                       'HaightLeft_poke_left_SA_poke_left': 'left_poke',
                                                       'HaightLeft_poke_right_SA_poke_right': 'right_poke',
                                                       'HaightRight_poke_center': 'center_poke',
                                                       'HaightRight_poke_handle': 'left_poke',
                                                       'HaigthRight_poke_right': 'right_poke',
                                                       'HaightRight_poke_left': 'left_poke'}.items():
            self.insert1({"dio_event_name": dio_event_name,
                          "statescript_event_name": statescript_event_name},
                          skip_duplicates=True)


@schema
class PumpTimes(ComputedBase):

    definition = """
    # Times of reward delivery (from DIOs when possible, otherwise from statescript)
    -> PumpDiosComplete
    -> ProcessedDioEvents
    -> ProcessedStatescriptEvents
    ---
    pump_names : blob
    pump_times : blob
    reward_times_source : varchar(40)
    """

    def make(self, key):
        if (PumpDiosComplete & key).fetch1("dio_pumps_complete"):  # if complete set of pump events available from DIO
            reward_times_source = "ProcessedDioEvents"
            pump_names, pump_times, pump_values = (eval(reward_times_source).Pumps() & key).fetch1("dio_pump_names",
                                                                                                   "dio_pump_times",
                                                                                                   "dio_pump_values")
            # Consider only dio up events
            pump_names = pump_names[pump_values == 1]
            pump_times = pump_times[pump_values == 1]
            # Convert pump names to well names
            environment = (TaskIdentification & key).fetch1("task_environment")  # get environment for this epoch

            # Only include pumps in the environment
            is_pump_env = np.asarray([f"{environment}" in pump_name for pump_name in pump_names])
            pump_names = pump_names[is_pump_env]
            pump_times = pump_times[is_pump_env]

            pump_names = np.asarray([pump_name.split(f"{environment}_pump_")[1].split("_")[0] + "_well" for
                                     pump_name in pump_names])  # get names of wells at which dio pump up event detected
        else:  # use statescript pump events
            reward_times_source = "ProcessedStatescriptEvents"
            pump_names, pump_times = (eval(reward_times_source).Pumps & key).fetch1("processed_statescript_pump_names",
                                                                                    "processed_statescript_pump_times_ptp")
            # Convert pump names to well names
            pump_names = np.asarray([f"{pump_name.split('rewarding_')[1]}_well" for pump_name in pump_names])
        # Insert into table
        key.update({"pump_names": pump_names, "pump_times": pump_times, "reward_times_source": reward_times_source})
        insert1_print(self, key)


@schema
class ContingencyEnvironmentColor(dj.Manual):

    definition = """
    # Mapping from contingency/environment pair to color
    contingency : varchar(40)
    task_environment : varchar(40)
    ---
    color : blob
    """

    def insert_defaults(self, **kwargs):
        task_environments = ["HaightLeft"]*2 + ["HaightRight"]*2 + ["SA"]*2
        contingencies = ["centerAlternation", "handleAlternation"]*int(len(task_environments)/2)
        colors = np.asarray(plt.cm.tab20.colors)[[0, 1, 6, 7, 10, 11]]
        for task_environment, contingency, color in zip(task_environments, contingencies, colors):
            self.insert1({"task_environment": task_environment,
                            "contingency": contingency,
                            "color": color}, skip_duplicates=True)


def get_contingency_task_environment_color(nwb_file_name, epoch):
    contingency, task_environment = (TaskIdentification & {"nwb_file_name": nwb_file_name,
                                         "epoch": epoch}).fetch1("contingency", "task_environment")
    return (ContingencyEnvironmentColor & {"contingency": contingency,
                              "task_environment": task_environment}).fetch1("color")


def get_contingency_task_environment_colors(nwb_file_name, epoch_list):
    return [get_contingency_task_environment_color(nwb_file_name, epoch) for epoch in epoch_list]


def populate_jguidera_task_event(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_task_event"
    upstream_schema_populate_fn_list = [populate_jguidera_dio_event]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_task_event():
    schema.drop()

