import copy
import itertools

import datajoint as dj
import numpy as np
from spyglass.common import TaskEpoch, IntervalList

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import special_fetch1
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import get_epoch_time_interval
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema

schema = dj.schema("jguidera_interval")  # define custom schema


@schema
class EpochIntervalListName(ComputedBase):

    definition = """
    # Map between epochs and interval list names
    -> TaskIdentification
    -> IntervalList
    """

    # Note that tried having IntervalList as upstream table but for it to not influence primary key;
    # This led to mismatched epochs and interval list names.
    # Takes a long time to populate because consider all pairs of epochs and interval list names
    # for an nwb file.

    # Static method so can access outside make function for investigating cases where no matching interval list
    # name found
    @staticmethod
    def get_epoch_matched_pos_interval_list_names(
            nwb_file_name, epoch_interval_list_name, epsilon):
        # Match pos valid time interval to epoch
        epoch_matched_pos_interval_list_names = []
        close_matches = []
        epoch_valid_times = (IntervalList & {
            "nwb_file_name": nwb_file_name, "interval_list_name": epoch_interval_list_name}).fetch1(
            "valid_times")  # get valid times in epoch
        epoch_time_interval = [epoch_valid_times[0][0], epoch_valid_times[-1][-1]]  # [epoch start, epoch end]
        # widen to tolerate small differences in epoch boundaries across epoch/pos intervals
        epoch_time_interval_widened = np.asarray(
            [epoch_time_interval[0] - epsilon, epoch_time_interval[1] + epsilon])
        for pos_interval_list_name in get_pos_interval_list_names(
                nwb_file_name):  # for each pos valid time interval list
            pos_valid_times = (IntervalList & {
                "nwb_file_name": nwb_file_name, "interval_list_name": pos_interval_list_name}).fetch1(
                "valid_times")  # get interval valid times
            pos_time_interval = np.asarray([pos_valid_times[0][0], pos_valid_times[-1][
                -1]])  # [pos valid time interval start, pos valid time interval end]
            # Store matching pos interval list
            # deal with cases where disconnect more than epsilon, so only start times will be close across intervals
            if (nwb_file_name, epoch_interval_list_name, pos_interval_list_name) in [
                ("J1620210604_.nwb", "17_s9", "pos 16 valid times"),
                ("june20220425_.nwb", "18_h1", "pos 17 valid times"),
                ("june20220416_.nwb", "19_h2", "pos 18 valid times"),
                ("peanut20201106_.nwb", "19_h2", "pos 18 valid times"),
                ("peanut20201106_.nwb", "20_h3", "pos 19 valid times"),
                ("peanut20201112_.nwb", "17_s9", "pos 16 valid times"),
                ("peanut20201116_.nwb", "20_h2", "pos 19 valid times"),
                ("fig20211117_.nwb", "19_h2", "pos 18 valid times"),
            ]:
                if epoch_time_interval_widened[0] < pos_time_interval[0]:
                    epoch_matched_pos_interval_list_names.append(pos_interval_list_name)
            # all other cases
            else:
                if np.logical_and(epoch_time_interval_widened[0] < pos_time_interval[0],
                                  epoch_time_interval_widened[1] > pos_time_interval[
                                      1]):  # if pos valid time interval within epoch interval
                    epoch_matched_pos_interval_list_names.append(pos_interval_list_name)
            # Store close matches for investigating cases where no matching pos interval list name found
            expand_window = 60  # seconds
            start_diff = abs(pos_time_interval[0] - epoch_time_interval_widened[0])
            end_diff = abs(pos_time_interval[1] - epoch_time_interval_widened[1])
            if np.logical_or(start_diff < expand_window, end_diff < expand_window):
                close_matches.append((pos_interval_list_name, start_diff, end_diff))
        # Print useful information if no matching pos interval list name found
        if len(epoch_matched_pos_interval_list_names) == 0:
            print(f"No matching pos interval found for {epoch_interval_list_name}. The following were close matches: "
                  f"(pos_interval_list_name, start diff, end diff): {close_matches}")
        return epoch_matched_pos_interval_list_names

    def make(self, key):
        # Find correspondence between pos valid times names and epochs
        # Use epsilon to tolerate differences in epoch boundaries across epoch/pos intervals. Large differences
        # (many seconds) can be present in recordings with disconnects

        # *** HARD CODED VALUES ***
        epsilon = 60  # seconds. Tolerated time difference in epoch boundaries across epoch/pos intervals
        # *************************

        # Unpack key
        nwb_file_name = key["nwb_file_name"]

        # Get pos interval list names
        pos_interval_list_names = get_pos_interval_list_names(nwb_file_name)

        # Skip populating if no pos interval list names
        if len(pos_interval_list_names) == 0:
            print(f"NO POS INTERVALS FOR {key}; CANNOT POPULATE EpochIntervalListName")
            return

        # Get epoch number and corresponding interval list name
        x = (TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch", "interval_list_name")
        epochs, epoch_interval_list_names = x[0], x[1]
        # Initialize dictionary to store correspondence between epoch number and pos x valid time interval names
        epoch_pos_valid_time_dict = {epoch: [] for epoch in epochs}
        for epoch, epoch_interval_list_name in zip(epochs, epoch_interval_list_names):  # for each epoch
            # Find matching pos interval list names for this epoch
            epoch_matched_pos_interval_list_names = self.get_epoch_matched_pos_interval_list_names(
                nwb_file_name, epoch_interval_list_name, epsilon)
            # Store
            epoch_pos_valid_time_dict[epoch] += epoch_matched_pos_interval_list_names

        # Check that each pos interval was matched to only one epoch
        matched_pos_interval_list_names = list(itertools.chain.from_iterable(epoch_pos_valid_time_dict.values()))
        if len(np.unique(matched_pos_interval_list_names)) != len(matched_pos_interval_list_names):
            raise Exception(f"At least one pos interval list name was matched with more than one epoch")
        # Check that exactly one pos interval was matched to each epoch
        num_matches = [len(pos_interval_list_names) for pos_interval_list_names in
                       epoch_pos_valid_time_dict.values()]
        if not all(num_matches == np.asarray([1])):
            raise Exception(f"Should have found exactly one pos interval per epoch, but found {num_matches} matches"
                            f" (each entry corresponds to number of matches for one epoch")
        # Unpack matching pos interval lists from array
        epoch_pos_valid_time_dict = {k: v[0] for k, v in epoch_pos_valid_time_dict.items()}

        # Insert into table if epoch matches interval list name
        if epoch_pos_valid_time_dict[key["epoch"]] == key["interval_list_name"]:
            self.insert1(key)
            print('Populated EpochIntervalListName for {nwb_file_name}, {epoch}'.format(**key))

    def get_interval_list_name(self, nwb_file_name, epoch, tolerate_no_entry=False):
        key = {"nwb_file_name": nwb_file_name, "epoch": epoch}
        return special_fetch1(self, "interval_list_name", key, tolerate_no_entry)

    def get_epoch(self, nwb_file_name, interval_list_name, tolerate_no_entry=False):
        key = {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}
        return special_fetch1(self, "epoch", key, tolerate_no_entry)

    def populate_(self, **kwargs):
        # Check if table populated since takes a long time to populate. Only exit if one entry, since if key not
        # constraining enough we want to populate with entries that do not exist
        # TODO (feature): figure out how to detect case where table fully populated; here we also dont want to populate
        if "key" in kwargs:
            if kwargs["key"] is not None:
                if len(self & kwargs["key"]) == 1:
                    return [self.table_name]
        return super().populate_(**kwargs)


def get_pos_interval_list_names(nwb_file_name):
    return [interval_list_name for interval_list_name in
     (IntervalList & {"nwb_file_name": nwb_file_name}).fetch("interval_list_name")
     if np.logical_and(interval_list_name.split(" ")[0] == "pos",
                       " ".join(interval_list_name.split(" ")[2:]) == "valid times")]


def get_epoch_interval_list_names(nwb_file_name):
    return (TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch("interval_list_name")


def intervals_within_epoch_bool(key, start_times, end_times):
    # Exclude trials with start or end time outside epoch
    epoch_start_time, epoch_end_time = get_epoch_time_interval(key["nwb_file_name"], key["epoch"])
    return np.logical_and(start_times > epoch_start_time, end_times < epoch_end_time)


def populate_jguidera_interval(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_interval"
    upstream_schema_populate_fn_list = None
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_interval():
    schema.drop()
