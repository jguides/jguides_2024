import copy

import datajoint as dj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spyglass.common import StateScriptFile

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    insert1_print
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_environments
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import events_in_epoch_bool
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.task_event.jguidera_dio_event import ProcessedDioEvents, PumpDiosComplete
from src.jguides_2024.time_and_trials.jguidera_timestamps import EpochTimestamps
from src.jguides_2024.utils.check_well_defined import failed_check
from src.jguides_2024.utils.df_helpers import unpack_df_columns, df_filter_columns
from src.jguides_2024.utils.vector_helpers import match_increasing_elements, unpack_single_element, \
    remove_repeat_elements

schema = dj.schema("jguidera_statescript_event")


@schema
class StatescriptEvents(ComputedBase):
    definition = """
    # DIO events in statescript log
    -> TaskIdentification  # use this instead of TaskEpoch to limit table to JAG recordings
    -> StateScriptFile
    ---
    statescript_event_names : blob
    statescript_event_times_trodes : blob
    """

    def make(self, key):
        # Get statescript file for this epoch and split into lines
        ss_file_entry = (StateScriptFile & key).fetch_nwb()
        # Find statescript printouts (lines that are not comments and have content)
        state_script_printouts = [x for x in [z for z in ss_file_entry[0]["file"].fields["content"].split("\n")
                                              if len(z) > 0]
            if x[0] != "#"]  # note that must first find lines with content to enbale search for hash indicating comment
        # Parse printouts into event times and event (printouts have form "time event")
        event_times, event_names = zip(
            *[(int(line.split(" ")[0]), " ".join(line.split(" ")[1:])) for line in state_script_printouts])
        key.update({"statescript_event_names": np.asarray(event_names),
                    "statescript_event_times_trodes": np.asarray(event_times)})
        insert1_print(self, key)


def get_ss_trigger_type(ss_dio_name):
    """
    # Return trigger type (currently limited to well poke or pump event) for a given dio event
    :param ss_dio_name:
    :return: trigger type
    """

    if "rewarding" in ss_dio_name:
        return "pump"
    elif "poke" in ss_dio_name:
        return "poke"
    else:
        raise Exception(f"no trigger type coded up for {ss_dio_name}")


def get_ss_nwbf_dio_name_map():
    """
    Map from statescript dio name and environment to nwbf dio name
    :param ss_dio_name: statescript dio name
    :param environment: environment
    :return: map from statescript dio name and environment to nwbf dio name
    """

    nwbf_ss_dio_name_map = get_nwbf_ss_dio_name_map()
    nwbf_dio_names = list(nwbf_ss_dio_name_map.keys())
    ss_dio_names = list(nwbf_ss_dio_name_map.values())
    ss_nwbf_dio_name_map = dict()
    for environment in get_environments():  # for environments
        for ss_dio_name in ss_dio_names:  # for statescript dio names
            ss_nwbf_dio_name_map[(ss_dio_name, environment)] = unpack_single_element(
                [x for x in nwbf_dio_names if environment in x and
                 ss_dio_name.split("_poke")[0].split("rewarding_")[-1] in x and
                 get_ss_trigger_type(ss_dio_name) in x])  # matching nwbf dio name
    return ss_nwbf_dio_name_map


@schema
class ProcessedStatescriptEventsDioMismatch(dj.Manual):
    definition = """
    # Track files for which default max time difference between statescript and DIO events is exceeded
    -> TaskIdentification
    dio_name_ss : varchar(40)
    ---
    times_dist : blob  
    """


@schema
class ProcessedStatescriptEvents(ComputedBase):
    definition = """
    # Processed DIO events in statescript log
    -> StatescriptEvents
    -> EpochTimestamps
    ---
    -> nd.common.AnalysisNwbfile
    processed_statescript_events_object_id : varchar(40)
    """

    class Pokes(dj.Part):
        definition = """
            # Processed DIO well poke events in statescript log
            -> ProcessedStatescriptEvents
            ---
            processed_statescript_poke_names : blob
            processed_statescript_poke_times_ptp : blob
            """

    class FirstPokes(dj.Part):

        definition = """
            # Processed DIO well poke events in statescript log with consecutive pokes (after first) at same well removed
            -> ProcessedStatescriptEvents.Pokes
            ---
            processed_statescript_first_poke_names : blob
            processed_statescript_first_poke_times_ptp : blob        
            """

    class LastPokes(dj.Part):
        definition = """
            # Processed DIO well poke events in statescript log with consecutive pokes (until last) at same well removed
            -> ProcessedStatescriptEvents.Pokes
            ---
            processed_statescript_last_poke_names : blob
            processed_statescript_last_poke_times_ptp : blob        
        """

    class Pumps(dj.Part):
        definition = """
            # Processed DIO pump events in statescript log
            -> ProcessedStatescriptEvents
            ---
            processed_statescript_pump_names : blob
            processed_statescript_pump_times_ptp : blob
            """

    def make(self, key):

        # Functions to get PTP times (in seconds) for statescript DIO printouts
        # (in Trodes time: rounded ms from MCU being turned on).
        # Approach: rely on correspondence between PTP and Trodes times in nwb file.

        def _get_ss_dio_times_as_ptp(epoch_all_dio_times_ss_trodes,
                                     epoch_sample_counts_trodes,
                                     epoch_sample_times_ptp,
                                     fs):

            # Match DIO event times from statescript (Trodes time) and nwbf (PTP).
            # First try to match times using correspondence defined in nwbf.
            # If not possible because no match in nwbf for a given statescript Trodes time,
            # use statescript DIO event closest in time that has match in nwbf.

            epoch_all_dio_times_ss_ptp = []  # initialize list for a matched PTP times
            print("Progress in matching Trodes times for DIO events to PTP times:")
            idx_has_nwbf_match = None  # use to keep track of which DIOs in statescript have nwbf match
            for idx, dio_time_trodes in enumerate(
                    epoch_all_dio_times_ss_trodes):  # for each dio time from statescript file
                # Print progress of calculation
                if idx in np.arange(0, len(epoch_all_dio_times_ss_trodes),
                                    np.round(len(epoch_all_dio_times_ss_trodes) / 10)):
                    print(f"{np.round(idx / len(epoch_all_dio_times_ss_trodes) * 100)}%")
                # Find PTP sample times corresponding to statescript DIO times
                dio_sample_count_trodes = dio_time_trodes * fs / 1000  # convert Trodes time (in ms) to sample count
                matched_times_ptp = epoch_sample_times_ptp[(epoch_sample_counts_trodes == dio_sample_count_trodes)]
                # Use match in nwbf if found
                if len(matched_times_ptp) > 0:
                    # Check that identified PTP times do not span more than an expected time range
                    matched_times_ptp_range = np.max(matched_times_ptp) - np.min(matched_times_ptp)
                    if matched_times_ptp_range > max_expected_time_span:
                        raise Exception(
                            f"range of matched ptp times for dio is {matched_times_ptp_range}, "
                            f"which exceeds {max_expected_time_span}")
                    matched_time_ptp = np.median(np.median(
                        matched_times_ptp))  # define matched time as median of identified ptp times for this DIO event
                    epoch_all_dio_times_ss_ptp.append(matched_time_ptp)  # store
                    idx_has_nwbf_match, time_ptp_nwbf_match = copy.deepcopy(idx), copy.deepcopy(
                        matched_time_ptp)  # hold onto index and time
                # If no matching times in nwbf, extrapolate from closest DIO with nwbf match
                else:
                    print(
                        f"No matching PTP times in nwbf found for statescript DIO event number {idx}. "
                        f"To define PTP time for this DIO, comparing to closest DIO with match in nwbf.")
                    time_to_closest_match = dio_time_trodes - epoch_all_dio_times_ss_trodes[idx_has_nwbf_match]
                    time_to_closest_match /= 1000  # convert to s
                    epoch_all_dio_times_ss_ptp.append(time_to_closest_match + time_ptp_nwbf_match)
            return np.asarray(epoch_all_dio_times_ss_ptp)

        def _process_statescript_events(epoch_all_dio_events_ss, epoch_all_dio_times_ss_ptp, nwb_file_name, epoch):
            # "Process" statescript event times by:
            # 1) Replace "rewarding_side" (first poke at a side well) in statescript output with rewarding_rewardedWell
            # where rewardedWell is name of well that was rewarded (find as well at which most recent poke happened).
            idxs_rewarding_side = np.where(epoch_all_dio_events_ss == "rewarding_side")[0]
            for idx_rewarding_side in idxs_rewarding_side:  # for each instance of rewarding side
                side_well_name = [event for event in list(epoch_all_dio_events_ss[:idx_rewarding_side]) if
                                              "_poke" in event][-1].split("_poke")[0]
                # Check side well name valid
                valid_side_well_names = ["left", "right", "center", "handle"]
                if side_well_name not in valid_side_well_names:
                    raise Exception(f"Side well name not in {valid_side_well_names}")
                rewarding_side_replacement = "rewarding_" + side_well_name
                print(f"WARNING: replacing {epoch_all_dio_events_ss[idx_rewarding_side]} in statescript output with \
                      {rewarding_side_replacement}")
                epoch_all_dio_events_ss[idx_rewarding_side] = rewarding_side_replacement
            # 2) Add delay to reward to reward times in statescript.
            # These occur after correct poke, but want them to reflect when pump triggered.
            # Find indices of statescript events where well poked that triggers pump after delay
            idx_pump = np.asarray(
                [idx for idx, event_name in enumerate(epoch_all_dio_events_ss) if "rewarding_" in event_name])
            print(
                f"To account for reward delay, adding {reward_delay}s to statescript time for these events: \
                    {np.unique(epoch_all_dio_events_ss[idx_pump])}")
            # Add reward_delay to these statescript event times
            epoch_all_dio_times_ss_ptp[idx_pump] += reward_delay
            # 3) Exclude statescript events that occurred outside epoch (as defined by epoch entry in IntervalList)
            valid_bool = events_in_epoch_bool(nwb_file_name=nwb_file_name,
                                              epoch=epoch,
                                              event_times=epoch_all_dio_times_ss_ptp)
            epoch_all_dio_times_ss_ptp = epoch_all_dio_times_ss_ptp[valid_bool]
            epoch_all_dio_events_ss = epoch_all_dio_events_ss[valid_bool]
            return epoch_all_dio_times_ss_ptp, epoch_all_dio_events_ss

        # Functions to compare statescript DIO event PTP times (ms resolution -- lower)
        # to those in nwbf (sampling rate resolution -- higher).
        # Approach: check that same number of DIO events, and DIO events have similar times, in statescript and nwbf.

        def _check_dio_times_ss_breakoutboard(
                epoch_dio_times_ss_ptp, epoch_dio_up_times_ptp, max_expected_time_diff_dios,
                max_mean_expected_time_diff_dios, tolerate_error):
            # GOAL: Make sure distance between each ss event and closest dio on breakout board not "too much".
            times_dist = match_increasing_elements(epoch_dio_times_ss_ptp, epoch_dio_up_times_ptp)
            # Check distances between each ss and closest breakout board DIO event
            tolerate_error_ = copy.deepcopy(tolerate_error)
            if any(times_dist > max_expected_time_diff_dios):
                plt.figure()
                plt.hist(times_dist)
                plt.title("epoch_dio_times_ss_ptp - epoch_dio_up_times_ptp")

                # For some files, max_expected_time_diff_dios is exceeded, leading to error. Track these cases
                # and allow larger max_expected_time_diff_dios
                insert_key = {**key, **{"dio_name_ss": dio_name_ss, "times_dist": times_dist}}
                ProcessedStatescriptEventsDioMismatch().insert1(insert_key, skip_duplicates=True)
                # if insertion into tracking table worked, tolerate exception for this and next check (since a single
                # large time difference is likely to cause average to exceed max value in the next check)
                if len(ProcessedStatescriptEventsDioMismatch & {
                    k: insert_key[k] for k in ProcessedStatescriptEventsDioMismatch.primary_key}) == 1:
                    tolerate_error_ = True
                failed_check(tolerate_error=tolerate_error_,
                             print_statement=f"At least one {dio_name_ss} DIO has statescript/nwbf time differing "
                              f"by more than {max_expected_time_diff_dios}")
            if np.mean(times_dist) > max_mean_expected_time_diff_dios:
                failed_check(tolerate_error=tolerate_error_,
                             print_statement=f"Average time distance for ss/nwbf {dio_name_ss} DIO times"
                              f"greater than {max_mean_expected_time_diff_dios}")

        def _plot_epoch_dios(
                epoch_dio_times_ptp, epoch_dio_events_ptp, epoch_dio_up_times_ptp, epoch_dio_times_ss_ptp,
                title_text=None):
            # Plot DIO event times from statescript and nwbf (subplot 1) and their differences (subplot 2)
            fig, axes = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [3, 1]})
            ax_temp = axes[0]  # subplot 1: dio event times
            ax_temp.plot(epoch_dio_times_ptp, epoch_dio_events_ptp, 'o', color="black",
                         alpha=.5)  # plot events from nwbf
            ax_temp.plot(epoch_dio_times_ss_ptp, [1] * len(epoch_dio_times_ss_ptp), 'x',
                         color="red")  # plot events from statescript
            ax_temp.set_title(title_text)
            ax_temp.set_xlabel("PTP time (s)")
            ax_temp.set_ylabel("DIO value")
            ax_temp = axes[1]  # subplot 2: dio event time differences
            times_dist = match_increasing_elements(epoch_dio_times_ss_ptp, epoch_dio_up_times_ptp)
            counts, _, _ = ax_temp.hist(times_dist)  # plot difference in DIO times
            ax_temp.plot([0, 0], [0, np.max(counts)], color="red")  # plot line at zero
            ax_temp.set_title("event time difference \n(statescript - nwbf)")

        def _get_analogous_nwbf_events(epoch_all_dio_times_ss_ptp, epoch_all_dio_events_ss, nwb_file_name, epoch):
            # Get map from ss dio names to nwbf dio names, and environment which is needed to use map
            ss_nwbf_dio_name_map = get_ss_nwbf_dio_name_map()
            environment = (TaskIdentification & {"nwb_file_name": nwb_file_name,
                                                 "epoch": epoch}).fetch1("task_environment")
            # Initialize array for analogous dio names in ProcessedDioEvents for statescript event names
            epoch_all_dio_events_bb_analogue = np.asarray(
                [""]*len(epoch_all_dio_events_ss))  # use quotes b/c None cannot be stored as hdf5
            epoch_all_dio_events_bb_analogue_value = np.asarray([np.nan]*len(epoch_all_dio_events_ss))
            for idx, (ss_dio_time, ss_dio_name) in enumerate(zip(epoch_all_dio_times_ss_ptp,
                                                                 epoch_all_dio_events_ss)):
                # Current understanding is "0 0" lines are dio downs for poke events only (not pump events)
                # POKES
                # UP events
                if "poke" in ss_dio_name:
                    nwbf_dio_name = ss_nwbf_dio_name_map[(ss_dio_name, environment)]
                    epoch_all_dio_events_bb_analogue[idx] = nwbf_dio_name
                    epoch_all_dio_events_bb_analogue_value[idx] = 1  # dio up
                    # Store poke event name so can use for dio down estimation
                    previous_poke_name = nwbf_dio_name
                # DOWN events
                if ss_dio_name == "0 0":
                    epoch_all_dio_events_bb_analogue[idx] = previous_poke_name
                    epoch_all_dio_events_bb_analogue_value[idx] = 0  # dio down

                # PUMPS (only UP events)
                if "rewarding" in ss_dio_name:
                    epoch_all_dio_events_bb_analogue[idx] = ss_nwbf_dio_name_map[(ss_dio_name, environment)]
                    epoch_all_dio_events_bb_analogue_value[idx] = 1  # dio up
            return epoch_all_dio_events_bb_analogue, epoch_all_dio_events_bb_analogue_value

        # Get PTP times (in seconds) for statescript DIO printouts (in Trodes time: rounded ms from when MCU turned on).
        # To validate, compare to DIO event times from ProcessedDioEvents.

        # *** Hard code ***
        max_expected_time_span = .0011  # max tolerated range of PTP times matched to Trodes DIO times from statescript
        max_expected_time_diff_dios = .018  # max tolerated distance between nwbf and statescript DIO times
        max_mean_expected_time_diff_dios = .005  # max tolerated mean distance between nwbf and statescript DIO times
        reward_delay = 2  # seconds. Delay until reward. # TODO (feature): ideally would pull from statescript file
        fs = 30000  # TODO (feature): ideally would pull from data
        tolerate_error = False  # False for failed checks to raise exception
        # ******************

        nwb_file_name = key["nwb_file_name"]
        epoch = key["epoch"]

        print(f"Matching statescript DIO event times to PTP times for {nwb_file_name} epoch {epoch}")

        # Get statescript events (DIO event times in Trodes time and names in statescript file)
        ss_events = (StatescriptEvents & key).fetch1()
        epoch_all_dio_times_ss_trodes = ss_events["statescript_event_times_trodes"]
        epoch_all_dio_events_ss = ss_events["statescript_event_names"]

        # Convert statescript event times from Trodes time to PTP
        # To do this, first get epoch sample times (PTP time, Trodes time, and Trodes sample count)
        epoch_timestamps_df = (EpochTimestamps & key).fetch1_dataframe()
        epoch_all_dio_times_ss_ptp = _get_ss_dio_times_as_ptp(
            epoch_all_dio_times_ss_trodes=epoch_all_dio_times_ss_trodes,
            epoch_sample_counts_trodes=epoch_timestamps_df ["trodes_sample_count"].values,
            epoch_sample_times_ptp=epoch_timestamps_df ["ptp"].values,
            fs=fs)

        # Process statescript events
        epoch_all_dio_times_ss_ptp, epoch_all_dio_events_ss = _process_statescript_events(
            epoch_all_dio_events_ss, epoch_all_dio_times_ss_ptp, key["nwb_file_name"], key["epoch"])

        # Sort statescript dio events/times by time
        ss_dio_df = pd.DataFrame.from_dict({"epoch_all_dio_times_ss_ptp": epoch_all_dio_times_ss_ptp,
                                            "epoch_all_dio_events_ss": epoch_all_dio_events_ss}).set_index(
            "epoch_all_dio_times_ss_ptp")
        ss_dio_df.sort_index(inplace=True)  # sort events by time
        ss_dio_df.reset_index(inplace=True)  # get back time column so can unpack below
        epoch_all_dio_times_ss_ptp, epoch_all_dio_events_ss = unpack_df_columns(
            ss_dio_df, ["epoch_all_dio_times_ss_ptp", "epoch_all_dio_events_ss"])

        # Validation: Compare well poke and pump DIO event times in current environment from statescript
        # and ProcessedDioEvents.
        # Hardcode correspondence between DIO names in nwbf and statescript, to enable comparison of the two.
        # ...Get epoch DIOs from ProcessedDioEvents
        environment = (TaskIdentification & key).fetch1("task_environment")
        dio_names_map = {k: v for k, v in get_nwbf_ss_dio_name_map().items()
                         if environment in k}  # restrict to DIOs in current environment
        # ...For each DIO type, check that each statescript event "matched" by a ProcessedDioEvent event thats close
        # in time. Then plot statescript and ProcessedDioEvents dios.
        for dio_name_nwbf, dio_name_ss in dio_names_map.items():  # for each DIO
            # Get DIO event times in statescript
            epoch_dio_times_ss_ptp = epoch_all_dio_times_ss_ptp[epoch_all_dio_events_ss == dio_name_ss]
            # Get DIO event times and values in ProcessedDioEvents
            # Note that cannot populate tables within make function, so cannot first populate ProcessedDioEvents
            dio_df = (ProcessedDioEvents & key).fetch1_dataframe()
            dio_df_subset = df_filter_columns(dio_df,
                                              {"dio_event_names": dio_name_nwbf})  # restrict to current dio
            # Legacy: detect Haigth typo (expect that no longer present but check since would affect downstream
            # code if present)
            if len(dio_df_subset) == 0:
                if "Haigth" in dio_name_nwbf:
                    raise Exception(f"Haigth typo in file")
            epoch_dio_events_ptp = dio_df_subset["dio_event_values"]
            epoch_dio_times_ptp = dio_df_subset.index
            epoch_dio_up_times_ptp = df_filter_columns(dio_df_subset, {"dio_event_values": 1}).index  # up events
            if len(epoch_dio_times_ss_ptp) > 0:  # if epoch DIOs in statescript
                # Skip check and plotting for known cases where HaightLeft_pump_center_SA_pump_center DIO not
                # recorded. These recordings having False in PumpDiosComplete
                if np.logical_and(not (PumpDiosComplete & {"nwb_file_name": nwb_file_name,
                                                           "epoch": epoch}).fetch1("dio_pumps_complete"),
                                  dio_name_nwbf == 'HaightLeft_pump_center_SA_pump_center'):
                    continue
                # Note: used to check whether same number of DIO events in statescsript and ProcessedDioEvents.
                # Reason that removed this: there can be very fast DIO events not registered by statescript even
                # in cases where not associated with dio up state on the dio one lower on the breakout board.
                # Example: HaightLeft_poke_right_SA_poke_right in june20220412_.nwb, epoch 2.
                # Given this, does not make sense to check whether same number of DIO events in statescript and
                # ProcessedDioEvents.
                # Check that DIO events in statescript "matched" by an event from breakout board.
                _check_dio_times_ss_breakoutboard(epoch_dio_times_ss_ptp=epoch_dio_times_ss_ptp,
                                         epoch_dio_up_times_ptp=epoch_dio_up_times_ptp,
                                         max_expected_time_diff_dios=max_expected_time_diff_dios,
                                         max_mean_expected_time_diff_dios=max_mean_expected_time_diff_dios,
                                         tolerate_error=tolerate_error)
                # Plot DIO event times from statescript and nwbf (subplot 1) and difference of matched times (subplot 2)
                _plot_epoch_dios(
                    epoch_dio_times_ptp=epoch_dio_times_ptp,
                    epoch_dio_events_ptp=epoch_dio_events_ptp,
                    epoch_dio_up_times_ptp=epoch_dio_up_times_ptp,
                    epoch_dio_times_ss_ptp=epoch_dio_times_ss_ptp,
                    title_text=
                    f"{nwb_file_name} ep{epoch}\n{dio_name_nwbf} (nwbf) \n{dio_name_ss} (statescript) \nepoch {epoch}")
                # TODO: Check task events against position_and_maze and reward rules

        # Get nwbf events that are analogous to each statescript event to store
        (processed_statescript_event_names_bb_analogue,
         processed_statescript_event_values_bb_analogue) = _get_analogous_nwbf_events(epoch_all_dio_times_ss_ptp,
                                                                                       epoch_all_dio_events_ss,
                                                                                       nwb_file_name,
                                                                                       epoch)

        # Insert into table
        ss_event_df = pd.DataFrame.from_dict(
            {"processed_statescript_event_names": epoch_all_dio_events_ss,
              "processed_statescript_event_times_ptp": epoch_all_dio_times_ss_ptp,
              "processed_statescript_event_names_bb_analogue": processed_statescript_event_names_bb_analogue,
              "processed_statescript_event_values_bb_analogue": processed_statescript_event_values_bb_analogue})
        insert_analysis_table_entry(self, [ss_event_df], key, ["processed_statescript_events_object_id"])

        # Populate subtable for well poke events
        subtables_key = {k:key[k] for k in ["nwb_file_name", "epoch"]}
        poke_names, poke_times = zip(*[(event_name, event_time) for event_name, event_time in
                                       zip(epoch_all_dio_events_ss, epoch_all_dio_times_ss_ptp)
                                       if "_poke" in event_name])
        insert1_print(
            self.Pokes, {**subtables_key, **{"processed_statescript_poke_names" : np.asarray(poke_names),
                                             "processed_statescript_poke_times_ptp" : np.asarray(poke_times)}})

        # Populate subtable for first pokes at same well
        poke_names_consecutive_elements_removed, idxs = remove_repeat_elements(poke_names, keep_first=True)
        insert1_print(
            self.FirstPokes, {
                **subtables_key, **{"processed_statescript_first_poke_names": poke_names_consecutive_elements_removed,
                                    "processed_statescript_first_poke_times_ptp": np.asarray(poke_times)[idxs]}})

        # Populate subtable for last pokes at same well
        poke_names_consecutive_elements_removed, idxs = remove_repeat_elements(poke_names, keep_first=False)
        insert1_print(
            self.LastPokes, {
                **subtables_key, **{"processed_statescript_last_poke_names" : poke_names_consecutive_elements_removed,
                                    "processed_statescript_last_poke_times_ptp" : np.asarray(poke_times)[idxs]}})

        # Populate subtable for pump events
        pump_names, pump_times = zip(*[(event_name, event_time) for event_name, event_time in
                                       zip(epoch_all_dio_events_ss, epoch_all_dio_times_ss_ptp)
                                       if "rewarding_" in event_name])
        insert1_print(
            self.Pumps, {**subtables_key, **{"processed_statescript_pump_names" : np.asarray(pump_names),
                                             "processed_statescript_pump_times_ptp" : np.asarray(pump_times)}})

    # Override base class method so can populate table with whether pump DIOs complete and table with dio events,
    # which are not used explicitly in make function for this table, but are used for validation.
    def populate_(self, **kwargs):
        PumpDiosComplete().populate_(**kwargs)
        ProcessedDioEvents().populate_(**kwargs)
        return super().populate_(**kwargs)


@schema
class StatescriptEventInt(dj.Manual):

    definition = """
    # Map from statescript well poke and pump events to integers
    event_name : varchar(40)
    ---
    event_int : int
    """

    def insert_defaults(self, **kwargs):
        event_name_list = ["handle_poke", "center_poke", "right_poke", "left_poke",
                           "rewarding_handle", "rewarding_center", "rewarding_right", "rewarding_left"]
        self.insert([(event_name, idx) for idx, event_name in enumerate(event_name_list)], skip_duplicates=True)


def get_nwbf_ss_dio_name_map():
    """
    Map from dio names from nwbf, to PROCESSED event names in processed statescript (i.e. statescript dio events that
    have been passed through _process_dio_events)
    :return: map from dio names from nwbf to event names in statescript
    """
    dio_ss_names_map = dict()
    well_names = ["left", "right", "center", "handle"]
    # Pokes
    dio_ss_names_map.update({f"HaightLeft_poke_{x}_SA_poke_{x}": f"{x}_poke" for x in well_names})
    dio_ss_names_map.update({f"HaightRight_poke_{x}": f"{x}_poke" for x in well_names})
    # Pumps
    dio_ss_names_map.update({f"HaightLeft_pump_{x}_SA_pump_{x}": f"rewarding_{x}" for x in well_names})
    dio_ss_names_map.update({f"HaightRight_pump_{x}": f"rewarding_{x}" for x in well_names})
    return dio_ss_names_map


def populate_jguidera_statescript_event(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_statescript_event"
    upstream_schema_populate_fn_list = None
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_statescript_event():
    schema.drop()
