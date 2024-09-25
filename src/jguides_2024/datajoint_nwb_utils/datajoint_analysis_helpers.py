import copy
import itertools

import numpy as np
import pandas as pd

from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name
from src.jguides_2024.utils.df_helpers import df_filter_columns_isin, zip_df_columns
from src.jguides_2024.utils.dict_helpers import dict_comprehension
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_subject_ids
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import subject_id_date_from_nwbf_name
from src.jguides_2024.utils.plot_helpers import plot_spanning_line
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.string_helpers import replace_chars
from src.jguides_2024.utils.vector_helpers import find_spans_increasing_list


def _intervals_from_bool(bool_, index):
    valid_idxs = np.where(bool_)[0]
    valid_idx_spans = find_spans_increasing_list(valid_idxs)[0]
    return index[valid_idx_spans]


def get_unit_subset_text(unit_subset, num_units):
    if unit_subset:
        return f"_randunits{num_units}"
    return ""


def get_sort_group_unit_id(unit_name):
    sort_group_id, unit_id = unit_name.split("_")
    return int(sort_group_id), int(unit_id)


def get_nwb_file_name_epoch_text(nwb_file_name, epoch):
    return f"{format_nwb_file_name(nwb_file_name)}_ep{epoch}"


def get_trial_info_by_condition(trial_type, group_trials_by, key, restrict_conditions_map):
    # Group trial time and outcome information by a specified variable

    # Define map from trial type to trial table and name of performance outcomes in trial table
    from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDATrials, DioWellArrivalTrials
    trial_type_map = {"well_arrival": {"trial_table": DioWellArrivalTrials,
                                       "performance_outcomes_name": "performance_outcomes"},
                      "path_traversals": {"trial_table": DioWellDATrials,
                                          "performance_outcomes_name": "trial_end_performance_outcomes"}}

    # Check inputs
    # ...Check valid trial type
    if trial_type not in trial_type_map:
        raise Exception(f"trial_type must be in {list(trial_type_map.keys())} but is {trial_type}")
    # ...Check valid variable to group trials by
    valid_group_trials_by = ["well_names", "path_names"]
    if group_trials_by not in valid_group_trials_by:
        raise Exception(f"group_trials_by must be in {valid_group_trials_by}")
    # ...Check necessary params in key
    if group_trials_by == "well_arrival" and "dio_well_arrival_trials_param_name" not in key:
        raise Exception(f"If group_trials_by is well_arrival, dio_well_arrival_trials_param_name must be in key")

    # Get trials information from table
    trial_table = trial_type_map[trial_type]["trial_table"]
    trial_table.populate(key)  # populate table
    trial_df = (trial_table & key).fetch1_dataframe()

    # Restrict trials as indicated
    if restrict_conditions_map:
        # First, check that if restrict_conditions_map passed, trial df has all conditions in the map
        if any([restrict_name not in trial_df for restrict_name in restrict_conditions_map.keys()]):
            raise Exception(f"restrict_conditions_map was passed, but not all keys in the map are in trial_df")
        # Now restrict
        trial_df = df_filter_columns_isin(trial_df, restrict_conditions_map)

    # Store trial times and performance outcomes by condition
    # ...Get performance outcomes
    performance_outcomes_name = trial_type_map[trial_type]["performance_outcomes_name"]
    performance_outcomes = np.asarray(trial_df[performance_outcomes_name])
    # ...Get trial times
    trial_times = np.asarray(list(zip_df_columns(trial_df, ["trial_start_times", "trial_end_times"])))
    # ...Put all trials together if no group by variable
    if group_trials_by is None:
        return pd.DataFrame.from_dict(
            {"condition": "",  "trial_times": trial_times, "performance_outcomes": performance_outcomes})
    # ...Otherwise group by condition restriction
    else:
        # Get entity we want to group trials by
        group_trials_by_values = trial_df[group_trials_by]
        group_names = np.unique(group_trials_by_values)
        group_bools = [group_trials_by_values == x for x in group_names]
        trial_times = [trial_times[group_bool] for group_bool in group_bools]
        performance_outcomes = [performance_outcomes[group_bool] for group_bool in group_bools]
        return pd.DataFrame.from_dict({
            "condition": group_names, "trial_times": trial_times, "performance_outcomes": performance_outcomes})


def format_epochs(epochs):
    epochs_text = "".join(list(map(str, epochs)))
    return f"ep{epochs_text}"


def get_ordered_subject_ids():
    """
    # Return ordered list of subject IDs
    :return: list of subject IDs
    """

    # Define ordered list of subject IDs
    ordered_subject_ids = ["J16", "mango", "june", "peanut", "fig"]

    # Check that all subject IDs above are valid
    subject_ids = get_subject_ids()
    check_membership(ordered_subject_ids, subject_ids)

    # Return ordered list of subject IDs
    return ordered_subject_ids


def get_plot_marker(subject_id=None, nwb_file_name=None):
    check_one_none([nwb_file_name, subject_id])
    if subject_id is None:
        subject_id = get_subject_id(nwb_file_name)
    marker_map = dict_comprehension(get_ordered_subject_ids(), ["P", "s", "o", "p", 'D'])
    return marker_map[subject_id]


def get_subject_id_shorthand(subject_id):
    subject_id_int_map = {subject_id: idx + 1 for idx, subject_id in enumerate(get_ordered_subject_ids())}
    return f"Rat {subject_id_int_map[subject_id]}"


def get_subject_id(nwb_file_name):
    return subject_id_date_from_nwbf_name(nwb_file_name)[0]


def plot_junction_fractions(
        ax, span_data=None, linewidth=1, span_axis="y", color="gray", linestyle="solid", alpha=1, zorder=1,
        x_scale_factor=1):

    # Plot lines denoting junction fractions

    # Span axis lim if span_data not passed
    if span_data is None:
        if span_axis == "y":
            span_data = ax.get_ylim()
        elif span_axis == "x":
            span_data = ax.get_xlim()

    # Get path fraction value at maze turns
    from src.jguides_2024.position_and_maze.jguidera_maze import get_n_junction_path_junction_fractions
    junction_fractions = get_n_junction_path_junction_fractions(2)

    # Scale junction path fractions to x axis in plot
    junction_fractions *= x_scale_factor
    for junction_fraction in junction_fractions:
        plot_spanning_line(span_data, junction_fraction, ax, span_axis, linewidth, color, linestyle, alpha, zorder)


def plot_well_events(
        ax, span_data=None, linewidth=1, color="brown", linestyle="solid", alpha=1, zorder=1, x_scale_factor=1,
        shift_x=0):
    # Plot vertical lines denoting well arrival and reward delivery

    # Span y lim if span_data not passed
    if span_data is None:
        span_data = ax.get_ylim()
    for well_event_time in np.asarray([0, 2])*x_scale_factor + shift_x:
        plot_spanning_line(span_data, well_event_time, ax, "y", linewidth, color, linestyle, alpha, zorder)


def key_text(key, separating_character="_"):
    # Get text from key values

    # Abbreviate certain key values
    key = copy.deepcopy(key)  # copy to avoid changing key outside function
    if "nwb_file_name" in key:
        key.update({"nwb_file_name": format_nwb_file_name(key["nwb_file_name"])})

    # Define order and prefix for keys
    key_prefix_map = {"nwb_file_name": "",
                      "epoch": "ep",
                      "epochs_id": "eps",
                      "brain_region": "",
                      "brain_region_units_param_name": "",
                      "res_set_param_name": "",
                      "res_time_bins_pool_param_name": "",
                      "res_time_bins_pool_cohort_param_name": "",
                      "res_epoch_spikes_sm_param_name": "sd",
                      "fr_diff_vec_cos_sim_ppt_nn_ave_param_name": "",
                      }
    x = [f"{prefix}{key[k]}" for k, prefix in key_prefix_map.items() if k in key]

    return separating_character.join(x)


def format_brain_region(brain_region, replace_mpfc=True):
    # Format brain region text for figure

    replace_chars_map = {"_targeted": "", "CA1": "HPc"}

    if replace_mpfc:
        replace_chars_map.update({"mPFC": "dmPFC"})

    return replace_chars(brain_region, replace_chars_map)


def get_relationship_alpha(relationship):
    alpha1 = 1
    alpha2 = .6
    alpha_map = {
        # PATH RELATIONSHIPS
        "same_path": alpha1, "different_path": alpha2, "same_turn": alpha1, "different_turn_well": alpha2,
        "outbound": alpha1, "inbound": alpha2,
        # WELL RELATIONSHIPS
        "same_well": alpha1, "different_well": alpha2}
    check_membership([relationship], alpha_map.keys(), "passed relationship", "available relationships")
    return alpha_map[relationship]


def get_thesis_nwb_file_names():
    return np.asarray([
        "J1620210606_.nwb", "mango20211207_.nwb", "june20220419_.nwb", "peanut20201108_.nwb", "fig20211109_.nwb"])


def get_reliability_paper_nwb_file_names(as_dict=False):
    paper_nwb_file_names_map = {
        "J16": ["J1620210605_.nwb", "J1620210606_.nwb", "J1620210607_.nwb"],
        "mango": ["mango20211205_.nwb", "mango20211206_.nwb", "mango20211207_.nwb"],
        "june": ["june20220419_.nwb", "june20220420_.nwb", "june20220421_.nwb"],
        "peanut": ["peanut20201107_.nwb", "peanut20201108_.nwb", "peanut20201109_.nwb"],
        "fig": ["fig20211108_.nwb", "fig20211109_.nwb", "fig20211110_.nwb"]}
    if as_dict:
        return paper_nwb_file_names_map
    return np.concatenate(list(paper_nwb_file_names_map.values()))


def get_nwb_file_names_text(nwb_file_names):
    return "_".join([format_nwb_file_name(x) for x in nwb_file_names])


def get_val_pairs(vals, val_order):
    # Get pairs of values

    # Order vals
    vals = [x for x in val_order if x in vals]

    # Return val pairs
    return list(itertools.combinations(vals, r=2))


def plot_horizontal_lines(xlims, y_vals, ax):
    # Plot horizontal lines to help visualize y positions
    for y_val in np.unique(y_vals):
        ax.plot(xlims, [y_val] * 2, color="gray", alpha=.3)


