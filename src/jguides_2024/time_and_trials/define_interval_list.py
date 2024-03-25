import itertools

import numpy as np
from spyglass.common import IntervalList, TaskEpoch

from src.jguides_2024.metadata.jguidera_premaze_durations import PremazeDurations
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.utils.exclude_intervals import exclude_intervals
# Note than on 12/3/22 changed widen_exclusion_factor from .001 to .002 , since
# for J1620210604, small snippet of home session was getting included after exclusion of home sessions
# with .001 widen_exclusion_factor
from src.jguides_2024.utils.set_helpers import check_membership


class NewIntervalList:

    def __init__(self, starting_interval_list_names, nwb_file_name, NO_PREMAZE=False, NO_HOME=False, NO_SLEEP=False,
                 widen_exclusion_factor=.002):

        # Check inputs
        if not isinstance(starting_interval_list_names, list):
            raise Exception("starting_interval_list_names must be a list")
        check_membership(starting_interval_list_names, self._valid_starting_interval_list_names())

        self.starting_interval_list_names = starting_interval_list_names
        self.nwb_file_name = nwb_file_name
        self.NO_PREMAZE = NO_PREMAZE
        self.NO_HOME = NO_HOME
        self.NO_SLEEP = NO_SLEEP
        self.widen_exclusion_factor = widen_exclusion_factor

        self.new_interval_list_name = self.define_new_interval_list_name()
        self.new_interval_list = self._get_new_interval_list()

    @staticmethod
    def _valid_starting_interval_list_names():
        valid_starting_interval_list_names = ["raw data valid times"] + [
            f"pos {x} valid times" for x in np.arange(0, 20)]
        # Check that no valid starting interval list names are contained within another (important for getting
        # starting interval list names from interval_list_names)
        if any([x in y for idx_x, x in enumerate(valid_starting_interval_list_names) for idx_y, y in enumerate(
                valid_starting_interval_list_names)
                if idx_x != idx_y]):
            raise Exception(f"no entry in valid_starting_interval_list_names is allowed to be contained within "
                            f"another")
        return valid_starting_interval_list_names

    @staticmethod
    def _no_premaze_text():
        return " no premaze"

    @staticmethod
    def _no_home_text():
        return " no home"

    @staticmethod
    def _no_sleep_text():
        return " no sleep"

    def define_new_interval_list_name(self):
        # Define new interval list name
        # Join starting inteval list names
        new_interval_list_name = " ".join(self.starting_interval_list_names)
        # Add exclusion text
        if self.NO_PREMAZE:
            new_interval_list_name += self._no_premaze_text()
        if self.NO_HOME:
            new_interval_list_name += self._no_home_text()
        if self.NO_SLEEP:
            new_interval_list_name += self._no_sleep_text()
        return new_interval_list_name

    @classmethod
    def get_starting_interval_list_names_from_new_interval_list_name(cls, new_interval_list_name):
        # Inverse of define_new_interval_list_name
        # Remove exclusion text
        for fn in ["_no_sleep_text", "_no_home_text", "_no_premaze_text"]:
            text = getattr(cls, fn)()
            text_len = len(text)
            if new_interval_list_name[-text_len:] == text:
                new_interval_list_name = new_interval_list_name[:-text_len]

        # Split interval list name into starting interval list names
        starting_interval_list_names = [
            x for x in cls._valid_starting_interval_list_names() if x in new_interval_list_name]
        if len(" ".join(starting_interval_list_names)) != len(new_interval_list_name):
            raise Exception(f"starting_interval_list_names not as expected; one way this can happen is if passed"
                            f"new_interval_list_name is not constructed solely from _valid_starting_interval_list_names"
                            f"and exclusion text")
        return starting_interval_list_names

    @classmethod
    def get_epochs_for_new_interval_list_name(cls, nwb_file_name, new_interval_list_name):
        starting_interval_list_names = NewIntervalList.get_starting_interval_list_names_from_new_interval_list_name(
            new_interval_list_name)

        def _get_epoch(nwb_file_name, interval_list_name):
            # If raw data valid times is interval list name, return all epochs for nwb file
            if interval_list_name == "raw data valid times":
                return (TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch")
            # Other case currently covered is pos interval list name (corresponds to one epoch)
            return [EpochIntervalListName().get_epoch(nwb_file_name, interval_list_name)]

        return np.concatenate([_get_epoch(nwb_file_name, interval_list_name)
                               for interval_list_name in starting_interval_list_names])

    def _define_exclusion_periods(self):
        # Define exclusion periods as indicated by flags

        exclude_interval_list = []  # initialize list for intervals to exclude

        # Get names of epochs in IntervalList to be able to search for relevant epochs based on flags
        # (e.g. run epochs if excluding premaze times)
        epoch_names = (TaskEpoch & {"nwb_file_name": self.nwb_file_name}).fetch("interval_list_name")
        epoch_names.sort()  # sort epoch names numerically
        print(f"Identified epochs: {epoch_names}")

        def _get_epoch_names_nums(epoch_names, identifier):
            # Identify epoch names with identifier, and extract epoch number.
            # Assumes epoch names have form: "epochNum_identifier" where identifier is one of "_s", "_r", "_h".
            epoch_names = [epoch_name for epoch_name in epoch_names if identifier in epoch_name]
            epoch_nums = [int(epoch_name.split("_")[0]) for epoch_name in epoch_names]
            return epoch_names, epoch_nums

        if self.NO_PREMAZE:  # if want to exclude period when rat being carried to track during run sessions
            # Get names and epoch integers (use to get premaze durations) of run epochs
            run_epoch_names, run_epoch_nums = _get_epoch_names_nums(epoch_names, "_r")
            print(f"\nIdentified run epochs: {run_epoch_names}")
            # Add intervals corresponding to run epochs to list of intervals to exclude
            for run_epoch_name, run_epoch_num in zip(run_epoch_names, run_epoch_nums):  # for each run epoch
                # Get valid times intervals
                interval_list = (IntervalList & {"nwb_file_name": self.nwb_file_name,
                                                "interval_list_name": run_epoch_name}).fetch1("valid_times")
                # Get premaze duration
                premaze_duration = (PremazeDurations & {"nwb_file_name": self.nwb_file_name,
                                               "epoch": run_epoch_num}).fetch1("premaze_duration")
                # Append premaze period to list of intervals to exclude
                exclude_interval_list.append([interval_list[0][0], interval_list[0][0] + premaze_duration])

        if self.NO_HOME:  # if want to exclude home epochs
            # Get names of home epochs
            home_epoch_names, _ = _get_epoch_names_nums(epoch_names, "_h")
            print(f"\nIdentified home epochs: {home_epoch_names}")
            # Add intervals corresponding to home sessions to list of intervals to exclude
            for home_epoch_name in home_epoch_names:  # for each home epoch
                # Get valid times interval
                interval_list = (IntervalList & {"nwb_file_name": self.nwb_file_name,
                                                "interval_list_name": home_epoch_name}).fetch1("valid_times")
                # Append home epoch period to list of intervals to exclude
                exclude_interval_list.append([interval_list[0][0], interval_list[-1][-1]])

        if self.NO_SLEEP:  # if want to exclude sleep epochs
            # Get names of sleep epochs
            sleep_epoch_names, _ = _get_epoch_names_nums(epoch_names, "_s")
            print(f"\nIdentified sleep epochs: {sleep_epoch_names}")
            # Add intervals corresponding to sleep sessions to list of intervals to exclude
            for sleep_epoch_name in sleep_epoch_names:  # for each sleep epoch
                # Get valid times interval
                interval_list = (IntervalList & {"nwb_file_name": self.nwb_file_name,
                                                 "interval_list_name": sleep_epoch_name}).fetch1("valid_times")
                # Append sleep epoch period to list of intervals to exclude
                exclude_interval_list.append([interval_list[0][0], interval_list[-1][-1]])

        return exclude_interval_list

    def _widen_exclusion_periods(self, exclude_interval_list):
        # Widen exclusion periods
        print(f"\nNOTE: Widening exclusion periods by {self.widen_exclusion_factor}s to account for small differences "
              f"in start/stop of what should be same interval in different IntervalList entries")
        return [[exclude_interval[0] - self.widen_exclusion_factor, exclude_interval[1] + self.widen_exclusion_factor]
                for exclude_interval in exclude_interval_list]

    def _get_new_interval_list(self):

        # Define exclusion periods as indicated by flags
        exclude_interval_list = self._define_exclusion_periods()

        # Widen exclusion periods
        exclude_interval_list = self._widen_exclusion_periods(exclude_interval_list)

        # Remove exclusion periods from concatenation of starting interval lists
        starting_interval_list = np.asarray(
                list((itertools.chain.from_iterable([(IntervalList & {"nwb_file_name": self.nwb_file_name,
                  "interval_list_name": interval_list_name}).fetch1("valid_times")
                 for interval_list_name in self.starting_interval_list_names]))))  # get concatenation of interval lists that we will change

        new_interval_list = exclude_intervals(starting_interval_list=starting_interval_list,
                                              exclude_interval_list=exclude_interval_list)  # exclude periods from starting interval list

        return np.asarray(new_interval_list)

