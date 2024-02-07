import datajoint as dj
import numpy as np
import pandas as pd
from spyglass.common import TaskEpoch

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.time_and_trials.define_interval_list import NewIntervalList
from src.jguides_2024.utils.vector_helpers import unpack_single_element

# Needed for table definitions:
TaskIdentification

schema = dj.schema("jguidera_epoch_interval")


# This table is meant to be used with TrialsPool. This motivates the choice of secondary keys
# We take start to be after arrival at first
@schema
class EpochInterval(ComputedBase):

    definition = """
    # Start and end time of epoch
    -> TaskIdentification
    ---
    trial_start_times : blob
    trial_end_times : blob
    """

    def make(self, key):
        interval_list_name = (TaskEpoch & key).fetch1("interval_list_name")  # get interval list name for epoch
        interval_list = NewIntervalList(
            [interval_list_name], key["nwb_file_name"], NO_PREMAZE=True).new_interval_list
        epoch_start = interval_list[0][0]
        epoch_end = interval_list[-1][-1]
        key.update({"trial_start_times": [epoch_start], "trial_end_times": [epoch_end]})
        insert1_print(self, key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        trial_start_times, trial_end_times = self.fetch1("trial_start_times", "trial_end_times")
        return pd.DataFrame.from_dict({"trial_start_times": trial_start_times, "trial_end_times": trial_end_times})

    def get_epoch_end_time(self):
        return unpack_single_element(self.fetch1("trial_end_times"))

    def get_epoch_duration(self):
        return np.diff([unpack_single_element(x) for x in self.fetch1("trial_start_times", "trial_end_times")])[0]


def populate_jguidera_epoch_interval(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_epoch_interval"
    upstream_schema_populate_fn_list = None
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_epoch_interval():
    schema.drop()
