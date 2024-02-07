import datajoint as dj
import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, convert_path_names
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.task_event.jguidera_statescript_event import ProcessedStatescriptEvents, \
    populate_jguidera_statescript_event
from src.jguides_2024.utils.list_helpers import return_n_empty_lists, check_lists_same_length
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import unpack_single_element, remove_repeat_elements

schema = dj.schema("jguidera_task_performance")


@schema
class AlternationTaskWellIdentities(dj.Manual):
    definition = """
    # Mapping from physical wells to task-relevant wells for alternation task on four-arm maze
    active_contingency : varchar(40)
    ---
    center_well : varchar(40)
    handle_well : varchar(40)
    right_well : varchar(40)
    left_well : varchar(40)
    """

    def insert_defaults(self, **kwargs):
        """
        Populate AlternationTaskWellIdentities table in jguidera_task schema
        """
        self.insert(
            [("centerAlternation", "home_well", "extra_well", "right_well", "left_well"),
             ("handleAlternation", "extra_well", "home_well", "right_well", "left_well")],
            skip_duplicates=True)

    def get_well_name(self, abstract_well_name, active_contingency):
            return unpack_single_element(
                [k for k, v in (self & {"active_contingency": active_contingency}).fetch1().items() if
                 v == abstract_well_name])


@schema
class AlternationTaskRule(dj.Manual):
    definition = """
    # Mapping from well visits to reward and performance outcomes for alternation task on four-arm maze
    previous_side_well : varchar(40) 
    previous_well : varchar(40)
    current_well : varchar(40)
    ---
    reward_outcome : varchar(40)
    performance_outcome : varchar(40)
    """

    def insert_defaults(self, **kwargs):
        """
        Populate AlternationTaskRule table in jguidera_task schema
        NOTE: for the purposes of processing downstream, all performance outcomes should start with "correct",
        "incorrect", or "neutral".
        """

        condition_list = []  # initialize list for tuples with alternation task rule conditions

        def _append_condition_list(condition_list, past_side_well, past_well, current_well, reward_outcome,
                                   performance_outcome):
            """Function for appending to list what will be a row in the table"""
            # Check that performance_outcome starts with "correct", "incorrect", or "neutral". Important for downstream
            # processing.
            if performance_outcome.split('_')[0] not in ["correct", "incorrect", "neutral"]:
                raise Exception(f"Performance outcome must start with correct, incorrect, or neutral,"
                                f" but starts with: {performance_outcome}")
            condition_list.append((past_side_well, past_well, current_well, reward_outcome, performance_outcome))
            return condition_list

        side_wells = ["right_well", "left_well"]

        # Poke at home
        current_well = "home_well"
        # First poke ever
        condition_list = _append_condition_list(condition_list, "none", "none", current_well, "reward", "neutral")
        # Correct inbound
        for past_well in side_wells:
            condition_list = _append_condition_list(condition_list, past_well, past_well, current_well, "reward",
                                                    "correct_inbound")
        # After extra well poke
        for past_side_well in ["none"] + side_wells:
            condition_list = _append_condition_list(condition_list, past_side_well, "extra_well", current_well,
                                                    "no_reward",
                                                    "neutral")

        # Poke at extra well
        current_well = "extra_well"
        # First poke ever
        condition_list = _append_condition_list(condition_list, "none", "none", current_well, "no_reward", "neutral")
        # After poke at side well
        for past_well in side_wells:
            condition_list = _append_condition_list(condition_list, past_well, past_well, current_well, "no_reward",
                                                    "incorrect_inbound")
        # After poke at home well
        for past_side_well in ["none"] + side_wells:
            condition_list = _append_condition_list(condition_list, past_side_well, "home_well", current_well,
                                                    "no_reward",
                                                    "incorrect_outbound")

        # Poke at side well
        for current_well in side_wells:
            # First poke at side well after poke at home or no pokes
            for past_well in ["none", "home_well"]:
                condition_list = _append_condition_list(condition_list, "none", past_well, current_well, "reward",
                                                        "neutral")
            # After poke at extra well
            for past_side_well in ["none"] + side_wells:
                condition_list = _append_condition_list(condition_list, past_side_well, "extra_well", current_well,
                                                        "no_reward",
                                                        "neutral")
            # After poke at home
            past_well = "home_well"
            # Correct outbound
            condition_list = _append_condition_list(condition_list,
                                                    [side_well for side_well in side_wells if
                                                     side_well != current_well][0],
                                                    past_well, current_well, "reward", "correct_outbound")
            # Incorrect outbound
            condition_list = _append_condition_list(condition_list, current_well, past_well, current_well, "no_reward",
                                                    "incorrect_outbound")
            # After poke at other side well
            past_well = [side_well for side_well in side_wells if side_well != current_well][0]
            condition_list = _append_condition_list(condition_list, past_well, past_well, current_well, "no_reward",
                                                    "incorrect_inbound")

        # Insert into table
        self.insert(condition_list, skip_duplicates=True)

    def outcomes_by_keyword(self, keywords):
        performance_outcomes = set(self.fetch("performance_outcome"))
        return {keyword: [x for x in performance_outcomes if keyword in x.split("_")] for keyword in keywords}


@schema
class AlternationTaskPerformance(ComputedBase):
    definition = """
    # Mapping of well visits to reward and performance outcome
    -> ProcessedStatescriptEvents
    ---
    previous_side_wells : blob
    previous_wells : blob
    current_wells : blob
    abstract_previous_side_wells : blob
    abstract_previous_wells : blob
    abstract_current_wells : blob
    reward_outcomes : blob
    performance_outcomes : blob
    contingencies : blob
    """

    def make(self, key):

        def _get_current_wells(well_pokes):
            return np.asarray(well_pokes)

        def _get_previous_wells(well_pokes, previous_well="none"):
            return np.asarray([previous_well] + well_pokes[:-1])

        def _get_previous_side_wells(well_pokes,
                                     side_wells,
                                     previous_side_well):
            previous_side_wells = [previous_side_well]  # initialize list for previous side wells
            for well in well_pokes[:-1]:  # for each well visit
                if well in side_wells:  # if side well
                    previous_side_well = well  # update previous side well
                previous_side_wells.append(previous_side_well)  # append previous side well
            return np.asarray(previous_side_wells)

        def _wells_wrapper(active_contingency_, well_pokes_, side_wells=np.asarray(["right_well", "left_well"]),
                           previous_well="none", previous_side_well="none"):
            # To determine reward and performance outcomes, find previous side well, previous well,
            # and current well for each well visit in terms of abstract wells
            abstract_well_map = (AlternationTaskWellIdentities &
                                 {"active_contingency": active_contingency_}).fetch1()  # map from wells to abstract wells
            abstract_well_map["none"] = "none"  # define abstract well in case of no defined well
            abstract_well_pokes_ = [abstract_well_map[well] for well in well_pokes_]
            abstract_side_wells = [abstract_well_map[well] for well in side_wells]
            abstract_previous_side_well = abstract_well_map[previous_side_well]
            abstract_previous_side_wells_ = _get_previous_side_wells(abstract_well_pokes_,
                                                                     abstract_side_wells,
                                                                     abstract_previous_side_well)
            abstract_previous_well = abstract_well_map[previous_well]
            abstract_previous_wells_ = _get_previous_wells(abstract_well_pokes_,
                                                           abstract_previous_well)
            abstract_current_wells_ = _get_current_wells(abstract_well_pokes_)
            reward_outcomes_, performance_outcomes_ = zip(
                *[(AlternationTaskRule & {"previous_side_well": abstract_previous_side_well,
                                          "previous_well": abstract_previous_well,
                                          "current_well": abstract_current_well}).fetch1("reward_outcome", "performance_outcome")
                  for abstract_previous_side_well, abstract_previous_well, abstract_current_well
                  in zip(abstract_previous_side_wells_, abstract_previous_wells_, abstract_current_wells_)])
            # For storage in table only, find previous side well, previous well and current well for each well visit
            previous_side_wells_ = _get_previous_side_wells(well_pokes_,
                                                            side_wells,
                                                            previous_side_well)  # find out which actual wells correspond to abstract side wells
            previous_wells_ = _get_previous_wells(well_pokes_)
            current_wells_ = _get_current_wells(well_pokes_)
            return (previous_side_wells_, previous_wells_, current_wells_,
                    abstract_previous_side_wells_, abstract_previous_wells_,
                    abstract_current_wells_, np.asarray(reward_outcomes_),
                    np.asarray(performance_outcomes_))

        # Parse well pokes to determine reward and performance outcome (differs by contingency)
        contingency = (TaskIdentification & key).fetch1("contingency")  # get contingency

        # Case where contingency doesnt change during epoch
        if contingency in ["centerAlternation", "handleAlternation"]:
            well_pokes = [x.replace("poke", "well") for x in (ProcessedStatescriptEvents.FirstPokes & key).fetch1(
                "processed_statescript_first_poke_names")]  # get well pokes
            (previous_side_wells, previous_wells, current_wells,
             abstract_previous_side_wells, abstract_previous_wells,
             abstract_current_wells, reward_outcomes,
             performance_outcomes) = _wells_wrapper(contingency, well_pokes)
            contingencies = [contingency]*len(performance_outcomes)

        # Case where contingency changes during epoch
        elif contingency in ["centerThenHandleAlternation", "handleThenCenterAlternation"]:
            # Find well visits within contingency blocks
            well_pokes_contingencies = []
            well_pokes = []
            contingency_blocks = [f"{x.split('Alternation')[0]}Alternation" for x in contingency.split('Then')]
            def _format_well_pokes(well_pokes_):
                return [well_poke.replace("poke", "well") for well_poke in remove_repeat_elements(well_pokes_)[0]]
            for event_name in (ProcessedStatescriptEvents & key).fetch1_dataframe()["processed_statescript_event_names"]:
                if "_poke" in event_name:
                    well_pokes.append(event_name)
                if "switching contingency" in event_name:
                    well_pokes_contingencies.append(_format_well_pokes(well_pokes))
                    well_pokes = []
            well_pokes_contingencies.append(_format_well_pokes(well_pokes))
            # Remove repeat pokes that span contingencies (e.g. rat at right well and pokes before and after switch)
            well_pokes_contingencies = [well_pokes_contingencies[0]] + \
                                       [[well_poke for poke_num, well_poke in enumerate(well_pokes)
                                         if np.logical_or(poke_num > 0, well_poke != well_pokes_contingencies[prev_contingency_num][-1])]
                                        for prev_contingency_num, well_pokes in enumerate(well_pokes_contingencies[1:])]
            contingency_blocks = np.asarray(
                contingency_blocks * int(np.ceil(len(well_pokes_contingencies) / len(contingency_blocks))))[
                                      :len(well_pokes_contingencies)]  # contingency for stretches of same contingency
            # Define well lists for each contingency block
            (previous_side_wells, previous_wells, current_wells,
             abstract_previous_side_wells, abstract_previous_wells,
             abstract_current_wells, reward_outcomes,
             performance_outcomes) = return_n_empty_lists(8)
            previous_well = "none"
            previous_side_well = "none"
            side_wells = ["left_well", "right_well"]
            for contingency_idx, (well_pokes, active_contingency) in enumerate(zip(well_pokes_contingencies,
                                                                                   contingency_blocks)):
                # Continue if last contingency and no well pokes (otherwise get error below)
                if contingency_idx == len(contingency_blocks) - 1 and len(well_pokes) == 0:
                    continue
                # Define well lists for contingency block
                (psw, pw, cw, apsw, apw, acw, ro, po) = _wells_wrapper(active_contingency, well_pokes,
                                                                       previous_well=previous_well,
                                                                       previous_side_well=previous_side_well)
                check_lists_same_length([psw, pw, cw, apsw, apw, acw, ro, po])
                # Update previous well and previous side well, if not on last iteration (if on last iteration,
                # we dont need to update. Avoids indexing error if no side well in current wells on last contingency
                # block)
                if contingency_idx < len(contingency_blocks) - 1:
                    previous_well = cw[-1]
                    previous_side_well = np.asarray([well for well in cw if well in side_wells])[-1]
                # Append to main lists
                for main_list, part_list in zip((previous_side_wells, previous_wells, current_wells,
                                                 abstract_previous_side_wells, abstract_previous_wells,
                                                 abstract_current_wells, reward_outcomes,
                                                 performance_outcomes),
                                                (psw, pw, cw, apsw, apw, acw, ro, po)):
                    main_list += list(part_list)
            contingencies = np.concatenate([[active_contingency] * len(well_pokes)
                                            for well_pokes, active_contingency in
                                            zip(well_pokes_contingencies, contingency_blocks)])

        else:
            raise Exception(f"Contingency {contingency} not accounted for in AlternationTaskPerformance")

        # Populate parent table
        check_lists_same_length([previous_side_wells, previous_wells, current_wells,
                                 abstract_previous_side_wells, abstract_previous_wells,
                                 abstract_current_wells, reward_outcomes,
                                 performance_outcomes, contingencies])
        insert1_print(self,
                      {**key, **{"previous_side_wells": previous_side_wells,
                                 "previous_wells": previous_wells,
                                 "current_wells": current_wells,
                                 "abstract_previous_side_wells": abstract_previous_side_wells,
                                 "abstract_previous_wells": abstract_previous_wells,
                                 "abstract_current_wells": abstract_current_wells,
                                 "reward_outcomes": np.asarray(reward_outcomes),
                                 "performance_outcomes": np.asarray(performance_outcomes),
                                 "contingencies": np.asarray(contingencies)}})

    def fetch1_dataframe(self, column_names=None):
        if column_names is None:
            return pd.DataFrame.from_dict(self.fetch1())
        return pd.DataFrame.from_dict({k: v for k, v in (self & column_names).fetch1().items()
                                       if k not in ["nwb_file_name", "epoch"]})

    def get_outcomes_by_path(self, nwb_file_name, epochs):
        data_list = []
        for epoch in epochs:
            (previous_wells, current_wells, performance_outcomes, reward_outcomes) = (
                    AlternationTaskPerformance & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1(
                "previous_wells", "current_wells", "performance_outcomes", "reward_outcomes")
            path_names = np.asarray(convert_path_names(list(zip(previous_wells, current_wells))))
            performance_outcomes = np.asarray(
                performance_outcomes_to_int(performance_outcomes))  # convert performance outcomes to binary
            reward_outcomes = np.asarray(reward_outcomes_to_int(reward_outcomes))  # convert reward outcomes to binary
            unique_path_names = set(path_names)  # paths traversed during epoch
            for path_name in unique_path_names:
                path_percent_correct = np.nanmean(performance_outcomes[path_names == path_name])
                path_percent_reward = np.nanmean(reward_outcomes[path_names == path_name])
                path_num_trials = np.sum(path_names == path_name)
                data_list.append((epoch, path_name, path_percent_correct, path_percent_reward, path_num_trials))

        epoch_list, path_name_list, path_percent_correct_list, path_percent_reward_list, path_num_trials_list = zip(
            *data_list)
        return pd.DataFrame.from_dict({"epoch": epoch_list,
                                       "path_name": path_name_list,
                                       "path_percent_correct": path_percent_correct_list,
                                       "path_percent_reward": path_percent_reward_list,
                                       "path_num_trials": path_num_trials_list})


@schema
class PerformanceOutcomeColors(dj.Manual):
    definition = """
    # Map performance outcomes in AlternationTaskPerformance to colors
    outcome_name : varchar(40)
    ---
    color : varchar(40)
    """

    def make(self, key):
        self.insert(key)

    def get_performance_outcome_color_map(self):
        return {k: v for k, v in self.fetch()}

    def get_performance_outcome_color(self, outcome_name):
        return self.get_performance_outcome_color_map()[outcome_name]

    def insert_defaults(self, **kwargs):
        performance_outcome_colors = [("correct_inbound", "green"),
                                      ("correct_outbound", "limegreen"),
                                      ("incorrect_inbound", "salmon"),
                                      ("incorrect_outbound", "crimson"),
                                      ("neutral", "tan")]
        self.insert(performance_outcome_colors, skip_duplicates=True)


@schema
class AlternationTaskPerformanceStatistics(ComputedBase):
    definition = """
    # Summary statistics for epochs of alternation task
    -> AlternationTaskPerformance
    ---
    correct_count : int
    incorrect_count : int 
    neutral_count : int
    trial_count : int
    percent_correct : float
    reward_count : int
    no_reward_count : int
    """

    def make(self, key):
        def _check_list_membership(x, valid_entries):
            invalid_members = [i for i in x if i not in valid_entries]
            if len(invalid_members) > 0:
                raise Exception(f"List should only have {valid_entries} but has {invalid_members}")

        # Performance outcomes
        performance_outcomes = np.asarray([outcome.split("_")[0] for outcome in
                                           (AlternationTaskPerformance() & key).fetch1()[
                                               "performance_outcomes"]])  # get correct/incorret/neutral status of outcomes
        _check_list_membership(performance_outcomes,
                               valid_entries=["correct", "incorrect",
                                             "neutral"])  # check that performance outcomes correct/incorrect/neutral
        # Compute metrics
        key["neutral_count"] = np.sum(performance_outcomes == "neutral")
        key["correct_count"] = np.sum(performance_outcomes == "correct")
        key["incorrect_count"] = np.sum(performance_outcomes == "incorrect")
        key["trial_count"] = len(performance_outcomes)
        key["percent_correct"] = key["correct_count"] / (key["correct_count"] + key["incorrect_count"])

        # Reward outcomes
        reward_outcomes = np.asarray(
            (AlternationTaskPerformance() & key).fetch1()["reward_outcomes"])  # get reward outcomes
        _check_list_membership(reward_outcomes,
                               valid_entries=["reward", "no_reward"])  # check that outcomes either reward or no reward
        # Compute metrics
        key["reward_count"] = np.sum(reward_outcomes == "reward")
        key["no_reward_count"] = np.sum(reward_outcomes == "no_reward")

        self.insert1(key, skip_duplicates=True)
        print('Populated AlternationTaskPerformanceStatistics for file {nwb_file_name}, epoch {epoch}'.format(**key))

    def get_at_criterion_epochs(self, nwb_file_name, criterion=.8):
        """
        Get epochs at or above a percent correct value
        :param nwb_file_name: str, name of nwb file
        :param criterion: float, percent correct
        :return: list of epochs at or above percent correct criterion
        """
        epochs, percent_correct_list = (self &
                                        {"nwb_file_name": nwb_file_name}).fetch("epoch", "percent_correct")
        return epochs[percent_correct_list > criterion]


@schema
class ContingencyActiveContingenciesMap(dj.Manual):
    definition = """
    # Map from contingency to possible active contingencies
    contingency : varchar(40)
    ---
    active_contingencies : blob
    """

    def insert_defaults(self, **kwargs):
        for contingency, active_contingencies in [("centerAlternation", ["centerAlternation"]),
                                                  ("handleAlternation", ["handleAlternation"]),
                                                  ("centerThenHandleAlternation",
                                                   ["centerAlternation", "handleAlternation"]),
                                                  ("handleThenCenterAlternation",
                                                   ["handleAlternation", "centerAlternation"])]:
            self.insert1({"contingency": contingency, "active_contingencies": active_contingencies},
                         skip_duplicates=True)


def performance_outcomes_to_int(performance_outcomes):
    performance_outcomes_int_map = {"neutral": np.nan,
                                    "correct_inbound": 1,
                                    "correct_outbound": 1,
                                    "incorrect_inbound": 0,
                                    "incorrect_outbound": 0}
    return [performance_outcomes_int_map[x] for x in performance_outcomes]


def ints_to_performance_outcomes(ints):
    check_membership(ints, [0, 1])
    int_performance_outcomes_map = {0: "incorrect", 1: "correct"}
    return [int_performance_outcomes_map[x] for x in ints]


def reward_outcomes_to_int(performance_outcomes):
    reward_outcomes_int_map = {"reward": 1, "no_reward": 0}
    return [reward_outcomes_int_map[x] for x in performance_outcomes]


def strip_performance_outcomes(performance_outcome):
    """
    Strip inbound and outbound from performance outcomes
    :param performance_outcome: list of performance outcomes including inbound and outbound
    :return: list of performance outcomes without inbound / outbound
    """
    stripped_performance_outcomes_map = {
        "correct_inbound": "correct", "correct_outbound": "correct", "incorrect_inbound": "incorrect",
        "incorrect_outbound": "incorrect", "none": "none", "neutral": "neutral"}
    return stripped_performance_outcomes_map[performance_outcome]


def populate_jguidera_task_performance(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_task_performance"
    upstream_schema_populate_fn_list = [populate_jguidera_statescript_event]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_task_performance():
    schema.drop()
