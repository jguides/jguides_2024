from collections import namedtuple

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spyglass.common import IntervalList

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_reliability_paper_nwb_file_names, \
    plot_horizontal_lines
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, EventTrialsParamsBase, \
    WellEventTrialsBase, SecKeyParamsBase, SelBase, \
    WellEventTrialsBaseExt
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import fetch1_dataframe_from_table_entry, \
    insert1_print, get_table_column_names, delete_, \
    get_entry_secondary_key
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_delay_interval, get_delay_duration
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import get_epoch_time_interval
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch
from src.jguides_2024.position_and_maze.jguidera_maze import RewardWellPath, MazePathWell, RewardWellPathColor
from src.jguides_2024.task_event.jguidera_dio_event import ProcessedDioEvents, populate_jguidera_dio_event
from src.jguides_2024.task_event.jguidera_task_event import PumpTimes, populate_jguidera_task_event
from src.jguides_2024.task_event.jguidera_task_performance import AlternationTaskPerformance, \
    populate_jguidera_task_performance
from src.jguides_2024.time_and_trials.jguidera_epoch_interval import EpochInterval
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName, intervals_within_epoch_bool
from src.jguides_2024.utils.array_helpers import array_to_tuple_list
from src.jguides_2024.utils.df_helpers import df_filter_columns_isin
from src.jguides_2024.utils.dict_helpers import dict_comprehension
from src.jguides_2024.utils.dtype_helpers import get_null_value
from src.jguides_2024.utils.interval_helpers import check_interval_start_before_end
from src.jguides_2024.utils.list_helpers import check_lists_same_length, zip_adjacent_elements
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import index_vectors, unpack_single_element

schema = dj.schema("jguidera_dio_trials")  # define custom schema


def _dio_poke_names_to_well_names(dio_poke_names):
    return np.asarray([f"{poke_name.split('_')[-1]}_well" for poke_name in dio_poke_names])


@schema
class DioWellTrials(ComputedBase):
    definition = """
    # Trials based on well arrivals detected with dios, meant to be referenced by other trials tables
    -> ProcessedDioEvents
    -> PumpTimes
    -> AlternationTaskPerformance
    ---
    -> EpochIntervalListName
    -> IntervalList
    epoch_trial_numbers : blob
    well_arrival_times : blob
    well_departure_times : blob
    well_durations : blob
    well_names : blob
    performance_outcomes : blob
    reward_outcomes : blob
    reward_times : blob
    reward_times_source : varchar(40)
    """

    def make(self, key):
        """
        Each trial spans the time from one well arrival to the next.
        """

        # Define well arrival times associated wells
        arrival_times, arrival_names = (ProcessedDioEvents.FirstUpPokes & key).fetch1(
            "dio_first_poke_times", "dio_first_poke_names")
        arrival_well_names = _dio_poke_names_to_well_names(arrival_names)

        # Define departures associated with each well arrival
        departure_times = np.asarray([np.nan] * len(arrival_times))  # initialize to have same length as arrival times
        departure_times_, departure_names = (ProcessedDioEvents.LastDownPokes & key).fetch1(
            "dio_last_poke_times", "dio_last_poke_names")
        departure_times[:len(departure_times_)] = departure_times_  # update departure times
        departure_well_names = _dio_poke_names_to_well_names(departure_names)

        # Check that departures and arrivals at same wells
        # Here, short index into arrival well names since final arrival may not be followed by a departure
        if not all(departure_well_names == arrival_well_names[:len(departure_well_names)]):
            raise Exception(f"Departure and arrival well names not aligned")
        # Check that there is exactly one departure time between each arrival time
        if unpack_single_element(np.unique(np.histogram(departure_times, bins=arrival_times)[0])) != 1:
            raise Exception(
                "There should be exactly one departure time between each well arrival time, but this is not the case")

        # Define well duration (time from arrival to departure)
        well_durations = departure_times - arrival_times
        if not all(well_durations[np.isfinite(well_durations)] > 0):  # check that departures after arrivals
            raise Exception(f"All departure times should come after arrival times but this is not the case")

        # Get trial performance and reward outcomes and restrict to entries corresponding to trials
        performance_entry = (AlternationTaskPerformance & key).fetch1()
        # First check that well visit sequence from DIOs matched in alternation task performance table
        if not all(arrival_well_names == performance_entry["current_wells"][:len(arrival_well_names)]):
            raise Exception(
                f"Sequence of well visits from dios not matched in alternation task performance (statescript)")
        performance_outcomes, reward_outcomes = (performance_entry["performance_outcomes"][:len(arrival_well_names)],
                                                 performance_entry["reward_outcomes"][:len(arrival_well_names)])

        # Get reward times
        # Approach: match dio pump times to trials defined as one well arrival to the next
        # Motivation for approach: make sure all dio pump times accounted for
        interval_list_name = (EpochIntervalListName & key).fetch1("interval_list_name")
        epoch_valid_times = (IntervalList & {"nwb_file_name": key["nwb_file_name"],
                                             "interval_list_name": interval_list_name}).fetch1("valid_times")
        epoch_end_time = epoch_valid_times[-1][-1]
        # Get delay period duration and add to epoch end to account for cases where pump goes off after recording stops
        delay_duration = get_delay_duration()
        trial_time_intervals = np.asarray(list(zip(arrival_times,
                                                   np.concatenate(
                                                       (arrival_times[1:], [epoch_end_time + delay_duration])))))
        pump_times, pump_names, reward_times_source = (PumpTimes & key).fetch1(
            "pump_times", "pump_names", "reward_times_source")
        reward_times = np.asarray([None] * len(arrival_times))  # initialize list for dio pump times
        for pump_name, pump_time in zip(pump_names, pump_times):
            trial_idx = np.where(np.product(trial_time_intervals - pump_time, axis=1) <= 0)[
                0]  # indices of trials during which pump event occurred
            if len(trial_idx) != 1:  # check that pump time matched to exactly one trial
                raise Exception(
                    f"pump_time should have fallen within exactly one trial interval but fell within {len(trial_idx)}")
            trial_idx = trial_idx[0]  # get trial idx out of array
            if pump_name == arrival_well_names[trial_idx]:  # if well for dio pump event same as well at end of trial
                reward_times[trial_idx] = pump_time  # store pump time
            elif all(
                    [pump_name != arrival_well_names[trial_idx],  # dio pump event well not same as well at end of trial
                     pump_name == arrival_well_names[trial_idx - 1],
                     # dio pump event well same as at previous trial end
                     reward_outcomes[trial_idx - 1] == "reward",  # previous trial rewarded
                     reward_times[trial_idx - 1] is None]):  # no dio pump event in previous trial
                reward_times[trial_idx - 1] = pump_time  # store pump time
            else:
                raise Exception(
                    "Pump should correspond to well that rat leaves from on current trial, or meet characteristics for "
                    "case where rat left early on previous trial before pump went off. Neither of these cases met.")

        # Check that dio pump events match rewarded trials according to AlternationTaskPerformance
        dio_reward_idxs = np.where(np.isfinite(np.array(reward_times, dtype=float)))[0]  # trial indices with pump dio
        performance_table_reward_idxs = np.where(reward_outcomes == "reward")[
            0]  # indices of trials in AlternationTaskPerformance table labeled as rewarded
        # Note that index below accounts for fact that reward can be delivered after epoch end
        if not all(dio_reward_idxs == performance_table_reward_idxs[:len(dio_reward_idxs)]):
            raise Exception(f"Dio pump event trials do not match rewarded trials in AlternationTaskPerformance table")

        # Check that correct trials have a corresponding dio pump time, with exception of last correct trial,
        # which can lack dio pump time if epoch could ends before pump goes off
        correct_trial_idxs = np.asarray([idx for idx, performance_outcome in enumerate(performance_outcomes)
                                         if performance_outcome in ["correct_inbound", "correct_outbound"]])
        if not all([j in dio_reward_idxs for j in correct_trial_idxs[:-1]]):
            raise Exception(f"A dio pump event was not detected during at least one trial labeled as correct in "
                            f"AlternationTaskPerformance table")

        # Define epoch trial numbers
        epoch_trial_numbers = np.arange(0, len(arrival_times))

        # Check that trial variables have same number of elements
        check_lists_same_length([epoch_trial_numbers, arrival_times, departure_times,
                                 well_durations, arrival_well_names, performance_outcomes, reward_times],
                                lists_description="Trial variables")

        # Insert into table
        insert1_print(self, {**key, **{"interval_list_name": interval_list_name,
                                       "epoch_trial_numbers": epoch_trial_numbers,
                                       "well_arrival_times": arrival_times,
                                       "well_departure_times": departure_times,
                                       "well_durations": well_durations,
                                       "well_names": arrival_well_names,
                                       "performance_outcomes": performance_outcomes,
                                       "reward_outcomes": reward_outcomes,
                                       "reward_times": reward_times,
                                       "reward_times_source": reward_times_source}})

    def fetch1_dataframe(self, strip_s=False, **kwargs):
        return fetch1_dataframe_from_table_entry(self, index_column="epoch_trial_numbers")

    def well_a_d_times(self):
        well_arrival_times, well_departure_times = self.fetch1("well_arrival_times", "well_departure_times")
        return list(zip(well_arrival_times, well_departure_times))

    def well_times(self):
        # Replace final well departure time with epoch end if nan (happens if animal doesnt leave well before
        # epoch ends)
        start_times, end_times = self.fetch1("well_arrival_times", "well_departure_times")
        if np.isnan(end_times[-1]):
            end_times = list(end_times)  # lists are mutable
            end_times[-1] = unpack_single_element((EpochInterval & self.fetch1("KEY")).fetch1("trial_end_times"))
        return list(zip(start_times, end_times))

    def get_post_delay_or_departure_trials(self):
        # Events are whatever comes first, delay end or well departure, at each well. Trials are defined
        # as the time between two events.
        well_arrival_times, well_departure_times = self.fetch1("well_arrival_times", "well_departure_times")
        delay_end_times = well_arrival_times + get_delay_duration()
        # Take whatever comes first, delay end or well departure
        trial_start_times = np.min(np.vstack([delay_end_times, well_departure_times]).T, axis=1)
        return list(zip(trial_start_times[:-1], trial_start_times[1:]))

    def get_stay_leave_trial_bool(self, stay_leave_trial_type, as_dict=False):
        """
        Return boolean indicating whether rat stayed for entire delay period at well
        :param stay_leave_trial_type: str. If "stay_trial", True on stay trials. If "leave_trial", True on leave trials.
        :param as_dict: bool. If True, return as dictionary where keys are epoch trial numbers. Otherwise,
        return vector of True and False.
        :return: vector of True and False that indicates whether rat stayed for full delay period or left early
        (as indicated) on each trial.
        """

        # Find well durations that are shorter (leave) or longer (stay) than delay duration
        # Note that final value in well_durations is nan if epoch ended before rat left well, and in this
        # case the leave and stay bools will be False in the last entry. We address this below.
        well_durations = self.fetch1("well_durations")
        delay_duration = get_delay_duration()
        leave_bool = well_durations < delay_duration
        stay_bool = well_durations >= delay_duration

        # If rat did not depart from well on final trial (final departure time is nan and therefore well duration is
        # nan), then we must figure out if rat left early or not using a different strategy.
        # Case 1: epoch ended before delay period ended. In this case, we cannot say whether rat stayed or left
        # during the delay, so we do nothing further since stay / leave bool will both
        # already be False in this case.
        # Case 2: epoch did not end before delay period ended. In this case, rat stayed at well.
        # ...Get time of epoch end
        epoch_end_time = (EpochInterval & self.fetch1("KEY")).get_epoch_end_time()
        # ...Set final value of stay_bool to True if rat stayed at well on final trial
        # get start of delay: time when rat arrives to well
        delay_start_times = self.fetch1("well_arrival_times")
        if np.logical_and(
                np.isnan(well_durations[-1]), epoch_end_time - delay_start_times[-1] >= delay_duration):
            stay_bool[-1] = True

        # Ensure no overlap between stay and leave trials
        if np.sum(stay_bool * leave_bool) > 0:
            raise Exception("stay and leave trials are overlapping; this should not be the case")

        # Define valid bool based on whether want only stay trials or only leave trials
        if stay_leave_trial_type == "stay_trial":
            valid_bool = stay_bool
        elif stay_leave_trial_type == "leave_trial":
            valid_bool = leave_bool
        else:
            raise Exception(f"stay_leave_trial_type must be stay_trial or leave_trial")

        # Return as dictionary where keys are epoch trial numbers if indicated
        if as_dict:
            return dict_comprehension(self.fetch1("epoch_trial_numbers"), valid_bool)
        # Otherwise return boolean
        return valid_bool

    def delay_times(self, stay_leave_trial_type=None):

        # Get start and end of each delay period

        # Check inputs
        check_membership([stay_leave_trial_type], [None, "stay_trial", "leave_trial"])

        # Get start of delay: time when rat arrives to well
        delay_start_times = self.fetch1("well_arrival_times")

        # Get end of delay
        delay_end_times = delay_start_times + get_delay_duration()

        # Make tuples with start/end of delays
        delay_times = np.asarray(list(zip(delay_start_times, delay_end_times)))

        # Restrict to stay or leave trials as indicated
        valid_bool = [True]*len(delay_times)  # default
        if stay_leave_trial_type in ["stay_trial", "leave_trial"]:
            valid_bool = self.get_stay_leave_trial_bool(stay_leave_trial_type)

        return array_to_tuple_list(delay_times[valid_bool])

    def well_post_delay_times(self):
        delay_end_times = self.fetch1("well_arrival_times") + get_delay_duration()
        well_departure_times = self.fetch1("well_departure_times")
        valid_trials = well_departure_times > delay_end_times
        return list(zip(delay_end_times[valid_trials], well_departure_times[valid_trials]))

    def get_trial_event_info(
            self, start_event_name, end_event_name, start_time_shift, end_time_shift, start_event_idx_shift=0,
            end_event_idx_shift=0, trial_characteristics=None, handle_trial_end_after_start=False):

        # Get inputs if not passed
        if trial_characteristics is None:
            trial_characteristics = []

        # Check inputs
        # ...Check event names and trial characteristics valid
        check_membership([start_event_name, end_event_name] + trial_characteristics, get_table_column_names(self),
                         "passed event names", "valid event names")
        # ...Check event shift indices valid
        invalid_num_shift_names = [num_shift_name for num_shift_name, num_shift in
                                   zip(["start_event_idx_shift", "end_event_idx_shift"],
                                       [start_event_idx_shift, end_event_idx_shift]) if num_shift < 0]
        if len(invalid_num_shift_names) > 0:
            raise Exception(f"start and end event idx shifts must be greater than or equal to zero. This was not the "
                            f"case for: {invalid_num_shift_names}")

        # Define trial start and end times
        # ...Get indices of trial starts and ends in well table
        table_entry = self.fetch1()
        num_epoch_trials = len(table_entry["epoch_trial_numbers"])  # number of epoch trials
        trial_start_idxs = np.arange(start_event_idx_shift, num_epoch_trials)
        trial_end_idxs = np.arange(end_event_idx_shift, num_epoch_trials)
        # ...Drop unpaired trial start or end idxs
        trial_start_idxs, trial_end_idxs = list(
            map(np.asarray, list(zip(*list(zip(trial_start_idxs, trial_end_idxs))))))
        # ...Get shifted times of start and end events
        start_times = table_entry[start_event_name][trial_start_idxs] + start_time_shift
        end_times = table_entry[end_event_name][trial_end_idxs] + end_time_shift
        # ...Check trial starts before ends if indicated. Here, account for edge case: the final epoch trial can have
        # no well departure. In this case, trial end is nan. Note that in making well trial info above, we required
        # that only final well departure allowed to be missing, so we dont need to check this again here
        if not handle_trial_end_after_start:  # throw error if trial end after start
            finite_bool = np.isfinite(end_times)
            check_interval_start_before_end(start_times[finite_bool], end_times[finite_bool])

        # Identify valid trials: start and end time within epoch, start time after end time
        # ...Exclude trials with start or end time outside epoch
        valid_bool = intervals_within_epoch_bool(table_entry, start_times, end_times)
        # ...Exclude trials where start time is after end time
        valid_bool *= (np.asarray(end_times) - np.asarray(start_times) >= 0)

        # Apply valid_trial_bool directly to start/end time
        start_times = start_times[valid_bool]
        end_times = end_times[valid_bool]

        # Return dictionary with trial start and event times, and corresponding desired trial characteristics
        # ...Trial start/end times
        trial_start_end_map = {"trial_start_times": start_times, "trial_end_times": end_times}
        # ...Trial characteristics: if different trial index for start and end events, return trial information
        # separately for trial start and trial end. Otherwise, return a single set of trial information
        if start_event_idx_shift != end_event_idx_shift:
            trial_start_idxs, trial_end_idxs = index_vectors([trial_start_idxs, trial_end_idxs], valid_bool)  # like map
            trial_characteristics_map = {
                f"{start_end_text}{trial_characteristic}": table_entry[trial_characteristic][idxs]
                for trial_characteristic in trial_characteristics
                for (start_end_text, idxs) in zip(["trial_start_", "trial_end_"],
                                                  [trial_start_idxs, trial_end_idxs])}  # trial info
        else:
            trial_characteristics_map = {k: table_entry[k][valid_bool] for k in trial_characteristics}  # trial info
        return {**trial_start_end_map, **trial_characteristics_map}

    def get_epoch_trial_nums(self, restrictions=None):
        """
        Get epoch trial numbers, optionally for specific kinds of trials
        :param restrictions: list of strings describing trial restrictions
        :return: epoch trial numbers
        """

        # Check inputs
        check_membership(restrictions, [
            "correct_trial", "correct_next_trial", "potentially_rewarded_trial", "potentially_rewarded_next_trial",
            "stay_trial", "leave_trial"])

        # Get epoch trial numbers
        epoch_trial_nums = self.fetch1("epoch_trial_numbers")

        # Restrict to correct trials. In table, correct/incorrect defined based on initial well arrival that
        # marks the start of the trial. A "correct_trial" is a well arrival to well arrival trial
        # for which the first well arrival was correct. A "correct_next_trial" is one for which the next
        # trial is correct.
        for restriction in ["correct_trial", "correct_next_trial"]:

            if restriction in restrictions:

                performance_outcomes = self.fetch1("performance_outcomes")

                if restriction == "correct_trial":
                    performance_outcomes_map = dict_comprehension(epoch_trial_nums, performance_outcomes)

                elif restriction == "correct_next_trial":
                    performance_outcomes_map = dict_comprehension(epoch_trial_nums[:-1], performance_outcomes[1:])

                restriction_epoch_trial_nums = [
                    epoch_trial_num for epoch_trial_num, performance_outcome in performance_outcomes_map.items() if
                    performance_outcome.startswith("correct")]

                epoch_trial_nums = [x for x in epoch_trial_nums if x in restriction_epoch_trial_nums]

        # Restrict to potentially rewarded trials. A "potentially_rewarded_trial" is a well arrival
        # to well arrival trial for which at the initial well arrival, the rat traversed a path that
        # has the possibility to yield reward under the task rules at any point during the recording.
        # A "potentially_rewarded_next_trial" is one for which this is the case on the next trial.
        for restriction in ["potentially_rewarded_trial", "potentially_rewarded_next_trial"]:

            if restriction in restrictions:

                # Get names of paths that rat traversed in each well arrival to well arrival trial
                path_names = self.get_path_names()

                # Get names of potentially rewarded paths for this recording
                nwb_file_name, epoch = self.fetch1("nwb_file_name", "epoch")
                potentially_rewarded_paths = MazePathWell().get_rewarded_path_names(nwb_file_name, epoch)

                epoch_trial_numbers = self.fetch1("epoch_trial_numbers")

                if restriction == "potentially_rewarded_next_trial":
                    paths_map = dict_comprehension(epoch_trial_numbers, path_names)

                elif restriction == "potentially_rewarded_trial":
                    paths_map = dict_comprehension(epoch_trial_numbers[1:], path_names[:-1])

                restriction_epoch_trial_nums = [
                    epoch_trial_num for epoch_trial_num, path_name in paths_map.items()
                    if path_name in potentially_rewarded_paths]

                epoch_trial_nums = [x for x in epoch_trial_nums if x in restriction_epoch_trial_nums]

        # Restrict to "stay" or "leave" trials at well
        if "stay_trial" in restrictions and "leave_trial" in restrictions:
            raise Exception(f"Only one of stay_trial and leave_trial can be in restrictions")

        elif "stay_trial" in restrictions or "leave_trial" in restrictions:

            trial_type = unpack_single_element([x for x in restrictions if x in ["stay_trial", "leave_trial"]])

            # Get stay/leave trial bool
            epoch_trial_nums = [x for x in epoch_trial_nums if x in np.where(
                self.get_stay_leave_trial_bool(trial_type))[0]]

        return epoch_trial_nums

    def get_path_names(self):
        well_names = self.fetch1("well_names")
        return RewardWellPath().get_path_names(well_names[:-1], well_names[1:])

    def epoch_trial_times(self, nwb_file_name=None, epoch=None, as_dict=False, restrictions=None):
        # Get times of "epoch trials". Each epoch trial is the period between one well arrival and the next or
        # the epoch end, whichever comes first. Optionally restrict to particular kinds of trials (e.g. correct trials)

        # Get inputs if not passed
        if nwb_file_name is None:
            nwb_file_name = self.fetch1("nwb_file_name")
        if epoch is None:
            epoch = self.fetch1("epoch")

        # Get times of epoch trials
        # ...Get table entry for this nwb file and epoch
        table_entry = (self & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1()
        # ...Find start/end times of last trial: starts at last well arrival, ends at epoch end
        well_arrival_times = table_entry["well_arrival_times"]  # well arrival times
        final_well_arrival_time = well_arrival_times[-1]  # last well arrival time
        epoch_end_time = get_epoch_time_interval(nwb_file_name, epoch)[-1]  # end of epoch
        final_epoch_trial_times = tuple([final_well_arrival_time, epoch_end_time])
        # ...Combine trials from one well arrival to the next with times of final trial from last well arrival
        # to epoch end
        well_arrival_trial_times = zip_adjacent_elements(well_arrival_times) + [final_epoch_trial_times]

        # Make map from epoch trial numbers to trial times
        # ...Get epoch trial numbers
        epoch_trial_nums = table_entry["epoch_trial_numbers"]
        # ...Make map
        epoch_trial_times_map = {k: v for k, v in zip(epoch_trial_nums, well_arrival_trial_times)}

        # Restrict to particular kinds of trials as indicated
        if restrictions is not None:

            # Get numbers of desired trials
            epoch_trial_nums = self.get_epoch_trial_nums(restrictions)

            # Filter for desired trials
            epoch_trial_times_map = {k: epoch_trial_times_map[k] for k in epoch_trial_nums}

        # Return as dictionary if indicated
        if as_dict:
            return epoch_trial_times_map

        # Otherwise return as list
        return list(epoch_trial_times_map.values())

    def get_time_epoch_trial_nums(self, time_vector, tolerate_outside=False, verbose=False):
        """
        Get epoch trial numbers at times in passed vector
        :param time_vector: vector with times
        :param tolerate_outside: tolerate times outside of epoch trials. If True, nan these values. If False,
        raise error if these values are present
        :return: epoch trial numbers at passed times
        """

        # Match passed times to numbers of epoch trials in which they fall

        # Get trial numbers and trial times for current table entry
        trial_nums = self.fetch1("epoch_trial_numbers")
        trial_times = self.epoch_trial_times()
        if len(trial_nums) != len(trial_times):
            raise Exception(f"different number of trial numbers and trial times")

        # Find the index of the trial interval in which each time falls
        idxs = [unpack_single_element(
            np.where(np.prod(np.asarray(trial_times) - x, axis=1) <= 0)[0], tolerate_no_entry=tolerate_outside,
            return_no_entry=np.nan) for x in time_vector]

        # Get the epoch trial number that each time falls within
        time_vec_epoch_trial_nums = [np.nan if np.isnan(idx) else trial_nums[idx] for idx in idxs]

        # Plot if indicated (sanity check) -- expect to see trial numbers as a function of passed time vector
        # fall within correspondingly numbered trial intervals from table
        if verbose:

            # Initialize plot
            fig, ax = plt.subplots(figsize=(10, 3))

            # Plot trial numbers for each time sample
            ax.plot(time_vector, time_vec_epoch_trial_nums, '.')

            # Plot trial intervals
            for trial_num, trial_start_end_time in enumerate(trial_times):
                ax.plot(trial_start_end_time, [trial_num] * 2)

        # Return trial numbers corresponding to each time
        return time_vec_epoch_trial_nums

    def in_trial(self, time_vector, restrictions=None):
        # Return boolean indicating whether passed times are within (optionally certain kinds of) epoch trials

        # Get epoch trial intervals (start and stop times), restricting to particular kinds of trials as indicated
        epoch_trial_times = self.epoch_trial_times(restrictions=restrictions)

        # Return boolean indicating whether times within trials (inclusive of endpoints)
        return np.asarray([any(np.prod(np.asarray(epoch_trial_times) - x, axis=1) <= 0) for x in time_vector])

    def get_leave_trial_info(self, verbose=False):
        """
        Get information about leave trials for a given table entry
        :return: object with number of leave trials, indices of leave trials, and names of paths preceeding leave trials
        """

        leave_trial_bool = self.get_stay_leave_trial_bool("leave_trial")
        num_leave_trials = np.sum(leave_trial_bool)
        leave_trial_idxs = np.where(leave_trial_bool)[0]
        preceeding_path_names = np.asarray(self.get_path_names())[leave_trial_bool[1:]]

        if verbose:
            nwb_file_name, epoch = self.fetch1("nwb_file_name", "epoch")
            print(f"{nwb_file_name}, {epoch}:")
            print(f"num leave trials: {num_leave_trials}")
            print(f"leave trial idxs: {leave_trial_idxs}")
            print(f"preceeding path names: {preceeding_path_names}")
            print("\n")

        return namedtuple(
            "LeaveTrialInfo", "num_leave_trials leave_trial_idxs preceeding_path_names")(
                num_leave_trials, leave_trial_idxs, preceeding_path_names)


@schema
class DioWellDDTrialsParams(EventTrialsParamsBase):
    definition = """
    # Parameters for defining time shifts for trials defined based on dio well departures 
    dio_well_dd_trials_param_name : varchar(40)
    ---
    trial_start_time_shift : decimal(10,5)
    trial_end_time_shift : decimal(10,5)
    """

    def _default_params(self):
        return [(0, 0)]  # start time shift, end time shift

    def lookup_no_shift_param_name(self, as_dict=False):
        dio_well_dd_trials_param_name = self.lookup_param_name([0, 0])
        if as_dict:
            return {"dio_well_dd_trials_param_name": dio_well_dd_trials_param_name}
        return dio_well_dd_trials_param_name


"""
Notes on DioWellDDTrials
1) Trials begin at a well departure and end at the next well departure.
2) "epoch_trial_number" refers to the same variable in DioWellTrials (these are numbers of trials defined in that 
table, which begin at well arrivals and end at the next well arrival). In DioWellDDTrials, the convention we have
chosen is to "map" the initial departure of a trial to the immediately preceeding well arrival. So 
"trial_start_epoch_trial_number" for a given trial in DioWellDDTrials refers to the trial in DioWellTrials 
that has an initial well arrival immediately preceeding the initial well departure of the given trial in 
DioWellDDTrials. Similarly, "trial_end_epoch_trial_number" for a given trial in DioWellDDTrials refers to the 
trial in DioWellTrials that has an initial well arrival immediately preceeding the final well departure of 
the given trial in DioWellDDTrials.
"""

@schema
class DioWellDDTrials(WellEventTrialsBaseExt):
    definition = """
    # Characteristics of trials that begin and end at well departures detected with dios
    -> DioWellTrials
    -> DioWellDDTrialsParams
    ---
    trial_start_times : blob
    trial_end_times : blob
    trial_start_epoch_trial_numbers : blob
    trial_end_epoch_trial_numbers : blob
    trial_start_well_names : blob
    trial_end_well_names : blob
    path_names : blob
    trial_start_performance_outcomes : blob
    trial_end_performance_outcomes : blob
    trial_start_reward_outcomes : blob
    trial_end_reward_outcomes : blob
    trial_start_reward_times : blob
    trial_end_reward_times : blob
    trial_end_well_arrival_times : blob
    trial_start_well_arrival_times : blob  # note that this is outside trial bounds unless trial time shifted back
    """

    def event_info(self):
        return self._event_info("well_departure_times", "well_departure_times", 0, 1, False)

    def label_time_vector(self, time_vector, column_names, add_dd_text=False):
        # NOTE: if want to make this more general and work with other trials tables, would want to change how we get
        # bin edges

        # Get departure to departure trials information
        dd_trials_df = self.fetch1_dataframe()

        # Check inputs
        check_membership(column_names, dd_trials_df, "passed column names",
                         "available column names in df from DioWellDDTrials")

        # First check that trial n start time same as trial n - 1 end time, so can use just trial
        # start times to assess which trial each external time falls within
        if not all(dd_trials_df.trial_start_times.values[1:] == dd_trials_df.trial_end_times.values[:-1]):
            raise Exception(
                f"current code assumes trial n start time same as trial n - 1 end time, and this was not the case")

        # Get dd trial idxs corresponding to each external time sample
        # ...Get bin edges
        trial_bin_edges = list(dd_trials_df.trial_start_times.values) + [dd_trials_df.trial_end_times.values[-1]]
        # ...Digitize and subtract one to account for one indexing by np.digitize
        dd_trial_idxs = np.digitize(time_vector, trial_bin_edges) - 1

        # Use to get corresponding trial information for each external time sample
        time_trials_info_map = dict()
        for column_name in column_names:
            dtype = dd_trials_df.dtypes[column_name]
            # Convert datatype from int to float so can define a null value (no null value for int datatype)
            if dtype == "int":
                dtype = "float"
            time_trials_info_map[column_name] = np.asarray([get_null_value(dtype)] * len(time_vector), dtype=dtype)
        valid_bool = np.logical_and(dd_trial_idxs <= np.max(dd_trials_df.index), dd_trial_idxs > -1)
        for column_name, vec in time_trials_info_map.items():
            vec[valid_bool] = dd_trials_df[column_name].iloc[dd_trial_idxs[valid_bool]]

        # Add "dd" to column names if indicated
        if add_dd_text:
            time_trials_info_map = {f"dd_{k}": v for k, v in time_trials_info_map.items()}

        # Add time
        time_trials_info_map["time"] = time_vector

        return pd.DataFrame.from_dict(time_trials_info_map).set_index("time")

    def times_by_paths(self, time_vector, path_names, task_period=None):

        trial_path_names = self.fetch1("path_names")

        if task_period is None:
            trial_intervals = np.asarray(self.trial_intervals())

        elif task_period == "delay":
            well_arrival_times = self.fetch1("trial_end_well_arrival_times")
            delay_end_times = well_arrival_times + get_delay_duration()
            trial_intervals = np.asarray(list(zip(well_arrival_times, delay_end_times)))

        valid_bool = [x in path_names for x in trial_path_names]
        path_trial_intervals = trial_intervals[valid_bool, :]

        return event_times_in_intervals(time_vector, path_trial_intervals)[1]

    # Override parent class method so can apply restrictions specific to this table, and use epoch trial end numbers
    # as epoch trial numbers
    def trial_intervals(self, trial_feature_map=None, restrictions=None, as_dict=False):

        # Get trials info
        trials_df = self.fetch1_dataframe()

        # Filter for certain trial features if indicated
        if trial_feature_map is not None:
            trials_df = df_filter_columns_isin(trials_df, trial_feature_map)

        # Define trial intervals
        trial_intervals = np.asarray(list(zip(trials_df["trial_start_times"], trials_df["trial_end_times"])))

        # Get the epoch trial numbers of the well arrival that is encompassed by the trial
        trial_end_epoch_trial_numbers = trials_df.trial_end_epoch_trial_numbers

        # Get subset of trial intervals based on restrictions if indicated
        if restrictions is not None:

            # Get valid epoch trial numbers per restrictions
            valid_epoch_trial_nums = (DioWellTrials & self.fetch1("KEY")).get_epoch_trial_nums(restrictions)

            # Determine which trial intervals are valid using the above trials
            valid_bool = [x in valid_epoch_trial_nums for x in trial_end_epoch_trial_numbers]

            # Get trial intervals and trial numbers subset
            trial_intervals = trial_intervals[valid_bool]
            trial_end_epoch_trial_numbers = trial_end_epoch_trial_numbers[valid_bool]

        # Return as dictionary if indicated
        if as_dict:
            return {epoch_trial_number: trial_interval for epoch_trial_number, trial_interval in
                    zip(trial_end_epoch_trial_numbers, trial_intervals)}

        # Otherwise return as list
        return trial_intervals

    def in_trial(self, time_vector, restrictions=None):

        # Return boolean indicating whether passed times are within (optionally certain kinds of) trials

        # Get trial intervals (start and stop times), restricting to particular kinds of trials as indicated
        trial_times = self.trial_intervals(restrictions=restrictions)

        # Return boolean indicating whether times within trials (inclusive of endpoints)
        return np.asarray([any(np.prod(np.asarray(trial_times) - x, axis=1) <= 0) for x in time_vector])

    def plot_results(self, ax=None, plot_legend=False):
        """
        Plot trial events
        """

        # Get quantities
        path_names, trial_end_well_arrival_times, trial_end_reward_outcomes = self.fetch1(
            "path_names", "trial_end_well_arrival_times", "trial_end_reward_outcomes")
        trial_intervals = self.trial_intervals()

        # Initialize figure if not passed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))

        # Plot horizontal lines to help visualize y positions
        xlims = [np.min(trial_intervals), np.max(trial_intervals)]
        plot_horizontal_lines(xlims, path_names, ax)

        # Plot departure to departure intervals by path
        linewidth = 3
        path_color_map = RewardWellPathColor().get_color_map()
        for x, path_name in zip(trial_intervals, path_names):
            color = path_color_map[path_name]
            ax.plot(x, [path_name] * 2, linewidth=linewidth, color=color)

        # Plot arrivals
        for idx, (arrival_time, path_name) in enumerate(zip(trial_end_well_arrival_times, path_names)):
            # Define color based on path
            color = path_color_map[path_name]
            ax.plot(arrival_time, path_name, ".", color=color)

        # Plot outcome at delay
        labels = []  # initialize list for tracking which labels have been defined for legend
        for idx, (arrival_time, path_name, outcome) in enumerate(zip(
                trial_end_well_arrival_times, path_names, trial_end_reward_outcomes)):
            # Define color based on outcome
            color = "green"
            if outcome == "no_reward":
                color = "red"
            # Define label if not already defined
            label = None
            if outcome not in labels:
                label = outcome
                labels.append(label)
            ax.plot(arrival_time + get_delay_duration(), path_name, ".", color=color, label=label)

        if plot_legend:
            ax.legend()


@schema
class DioWellDATrialsParams(EventTrialsParamsBase):
    definition = """
    # Parameters for defining time shifts for trials defined based on dio well departures/arrivals
    dio_well_da_trials_param_name : varchar(40)
    ---
    trial_start_time_shift : decimal(10,5)
    trial_end_time_shift : decimal(10,5)
    """

    def _default_params(self):
        return [(0, 0)]  # start time shift, end time shift

    def lookup_no_shift_param_name(self, as_dict=False):
        dio_well_da_trials_param_name = self.lookup_param_name([0, 0])
        if as_dict:
            return {"dio_well_da_trials_param_name": dio_well_da_trials_param_name}
        return dio_well_da_trials_param_name


@schema
class DioWellDATrials(WellEventTrialsBaseExt):
    definition = """
    # Characteristics of trials that begin at well departure and end at well arrival detected with dios
    -> DioWellTrials
    -> DioWellDATrialsParams
    ---
    trial_start_times : blob
    trial_end_times : blob
    trial_start_epoch_trial_numbers : blob
    trial_end_epoch_trial_numbers : blob
    trial_start_well_names : blob
    trial_end_well_names : blob
    path_names : blob
    trial_start_performance_outcomes : blob
    trial_end_performance_outcomes : blob
    trial_start_reward_outcomes : blob
    trial_end_reward_outcomes : blob
    """

    def event_info(self):
        return self._event_info("well_departure_times", "well_arrival_times", 0, 1, False)

    # Override parent class method so can define df index
    def fetch1_dataframe(self, strip_s=False, **kwargs):
        # kwargs input makes easy to use this method with fetch1_dataframe methods that take in different
        # inputs (e.g. to fetch dataframe from nwb file)
        return super().fetch1_dataframe(strip_s, "trial_start_epoch_trial_numbers")


@schema
class DioWellADTrialsParams(EventTrialsParamsBase):
    definition = """
    # Parameters for DioWellADTrials
    dio_well_ad_trials_param_name : varchar(40)
    ---
    trial_start_time_shift : decimal(10,5)
    trial_end_time_shift : decimal(10,5)
    """

    @staticmethod
    def _post_delay_params():
        return [get_delay_duration(), 0]  # start time shift, end time shift

    def _default_params(self):
        return [self._post_delay_params(), [0, 0]]

    def lookup_post_delay_param_name(self):
        return self.lookup_param_name(self._post_delay_params())


@schema
class DioWellADTrials(WellEventTrialsBaseExt):
    definition = """
    # Characteristics of trials that begin at well arrivals and end at well departure detected with dios
    -> DioWellTrials
    -> DioWellADTrialsParams
    ---
    epoch_trial_numbers : blob
    trial_start_times : blob
    trial_end_times : blob
    well_names : blob
    performance_outcomes : blob
    reward_outcomes : blob
    """

    def event_info(self):
        return self._event_info("well_arrival_times", "well_departure_times", 0, 0, True)


@schema
class DioWellArrivalTrialsParams(EventTrialsParamsBase):
    definition = """
    # Parameters for defining time shifts for trials based on single well arrival detected with dios
    dio_well_arrival_trials_param_name : varchar(40)
    ---
    trial_start_time_shift : decimal(10,5)
    trial_end_time_shift : decimal(10,5)
    """

    def _default_params(self):
        return [get_delay_interval(), [-1, 3]]  # start time shift, end time shift

    def lookup_delay_param_name(self):
        return self.lookup_param_name(get_delay_interval())

    def delete_(self, key, safemode=True):
        from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPoolSel
        # Delete all associated entries in TrialsPoolSel to be able to then delete in current table
        keys = TrialsPoolSel().get_entries_with_param_name_dict(key, restrict_primary_key=True)
        for k in keys:
            TrialsPoolSel().delete_(key=k, safemode=safemode)
        delete_(self, [], key, safemode)


@schema
class DioWellArrivalTrials(WellEventTrialsBaseExt):
    definition = """
    # Characteristics of trials based on single well arrival detected with dios
    -> DioWellTrials
    -> DioWellArrivalTrialsParams
    ---
    epoch_trial_numbers : blob
    trial_start_times : blob
    trial_end_times : blob
    well_names : blob
    performance_outcomes : blob
    reward_outcomes : blob
    """

    def event_info(self):
        return self._event_info("well_arrival_times", "well_arrival_times", 0, 0)


@schema
class DioWellArrivalTrialsSubParams(SecKeyParamsBase):
    definition = """
    # Parameters for DioWellArrivalTrialsSub
    dio_well_arrival_trials_sub_param_name : varchar(40)
    ---
    subset_type : varchar(40)
    """

    def _default_params(self):
        return [["stay"]]


@schema
class DioWellArrivalTrialsSubSel(SelBase):
    definition = """
    # Selection from upstream tables for DioWellArrivalTrialsSub
    -> DioWellArrivalTrials
    -> DioWellArrivalTrialsSubParams
    ---
    upstream_keys : blob  # part table keys
    """

    class DioWellADTrials(dj.Part):
        definition = """
        # Achieves upstream dependence on DioWellADTrials
        -> DioWellArrivalTrialsSubSel
        -> DioWellADTrials
        """

    def _get_potential_keys(self, key_filter=None):

        # Stay trials
        dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_delay_param_name()
        dio_well_arrival_trials_sub_param_name = DioWellArrivalTrialsSubParams().lookup_param_name(["stay"])
        dio_well_ad_trials_param_name = DioWellADTrialsParams().lookup_param_name([0, 0])

        keys = []
        for nwb_file_name in get_reliability_paper_nwb_file_names():
            for epoch in (RunEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch"):

                key = {
                    "nwb_file_name": nwb_file_name, "epoch": epoch,
                    "dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name,
                    "dio_well_arrival_trials_sub_param_name": dio_well_arrival_trials_sub_param_name}

                ad_key = {"nwb_file_name": nwb_file_name, "epoch": epoch,
                          "dio_well_ad_trials_param_name": dio_well_ad_trials_param_name}

                # Add key if upstream tables populated
                if len(DioWellArrivalTrials & key) > 0 and len(DioWellADTrials & ad_key) > 0:
                    keys.append({**key, **{"upstream_keys": {"DioWellADTrials": [ad_key]}}})

        return keys

    # Override parent class method so can insert into part table
    def insert1(self, key, **kwargs):

        # Insert into main table
        super().insert1(key, **kwargs)

        # Insert into part table
        upstream_keys = key.pop("upstream_keys")
        for part_table_name, upstream_ks in upstream_keys.items():
            for k in upstream_ks:
                get_table(f"DioWellArrivalTrialsSubSel.{part_table_name}").insert1({**key, **k}, skip_duplicates=True)


@schema
class DioWellArrivalTrialsSub(WellEventTrialsBase):
    definition = """
    # Subset of trials for entries in DioWellArrivalTrials
    -> DioWellArrivalTrialsSubSel
    ---
    epoch_trial_numbers : blob
    trial_start_times : blob
    trial_end_times : blob
    well_names : blob
    performance_outcomes : blob
    reward_outcomes : blob
    """

    def make(self, key):

        # Take subset of trials

        # Get subset type for this key
        subset_type = (DioWellArrivalTrialsSubParams & key).fetch1("subset_type")

        # Stay trials
        if subset_type == "stay":

            upstream_keys = (DioWellArrivalTrialsSubSel & key).fetch1("upstream_keys")
            upstream_key = unpack_single_element(upstream_keys["DioWellADTrials"])
            valid_bool = np.concatenate(
                np.diff((DioWellADTrials & upstream_key).trial_intervals()) >= get_delay_duration())

            table_vals = get_entry_secondary_key((DioWellArrivalTrials & key))

            # If rat at well when epoch ended, there will be one more trial in DioWellArrivalTrials than in
            # DioWellADTrials. If rat left well before 2s delay ended and epoch ended before 2s delay ended,
            # there will be one more trial in DioWellADTrials than in DioWellArrivalTrials. Given these can occur,
            # take overlapping trials across the tables.
            ad_epoch_trial_numbers = (DioWellADTrials & upstream_key).fetch1("epoch_trial_numbers")
            a_epoch_trial_numbers = table_vals["epoch_trial_numbers"]
            # Check that if epoch trial numbers differ from tables, it's only at last trial
            shared_bool_1 = [x in ad_epoch_trial_numbers for x in a_epoch_trial_numbers]
            shared_bool_2 = [x in a_epoch_trial_numbers for x in ad_epoch_trial_numbers]
            if not all(np.concatenate((shared_bool_1[:-1], shared_bool_2[:-1]))):
                raise Exception(
                    f"Expect only the last trial can differ between arrival/departure trials table and arrival trials "
                    f"table, but this wasnt the case")
            # Account for different last trial across tables
            # 1) rat at well when epoch ended --> last trial is stay trial --> add a True to valid_bool to reflect
            if len(a_epoch_trial_numbers) > len(ad_epoch_trial_numbers):
                valid_bool = np.append(valid_bool, True)
            # 2) rat leaves well before delay period ends, and epoch ends before delay period ends --> last trial is
            # leave trial and isn't included in arrival epoch trial numbers --> drop last element of valid_bool
            # to reflect
            elif len(ad_epoch_trial_numbers) > len(a_epoch_trial_numbers):
                valid_bool = valid_bool[:-1]

            # Take stay trials
            table_vals = {k: v[valid_bool] for k, v in table_vals.items()}

            # Add table values to key
            key.update(table_vals)

        # Raise error if havent accounted for current type of subset in code
        else:
            raise Exception(f"subset type {subset_type} not accounted for in code")

        # Insert into table
        insert1_print(self, key)


@schema
class DioWellDepartureTrialsParams(EventTrialsParamsBase):
    definition = """
    # Parameters for defining time shifts for trials based on single well departure detected with dios
    dio_well_departure_trials_param_name : varchar(40)
    ---
    trial_start_time_shift : decimal(10,5)
    trial_end_time_shift : decimal(10,5)
    """

    def _default_params(self):
        return [[-2, 0], [-1, 1]]  # start time shift, end time shift

    def delete_(self, key, safemode=True):
        from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPoolSel
        # Delete all associated entries in TrialsPoolSel to be able to then delete in current table
        keys = TrialsPoolSel().get_entries_with_param_name_dict(key, restrict_primary_key=True)
        for k in keys:
            TrialsPoolSel().delete_(key=k, safemode=safemode)
        delete_(self, [], key, safemode)


@schema
class DioWellDepartureTrials(WellEventTrialsBaseExt):
    definition = """
    # Characteristics of trials based on single well departure detected with dios
    -> DioWellTrials
    -> DioWellDepartureTrialsParams
    ---
    epoch_trial_numbers : blob
    trial_start_times : blob
    trial_end_times : blob
    well_names : blob
    performance_outcomes : blob
    reward_outcomes : blob
    """

    def event_info(self):
        return self._event_info("well_departure_times", "well_departure_times", 0, 0, False)


def populate_jguidera_dio_trials(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_dio_trials"
    upstream_schema_populate_fn_list = [populate_jguidera_dio_event, populate_jguidera_task_event, populate_jguidera_task_performance]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_dio_trials():
    schema.drop()
