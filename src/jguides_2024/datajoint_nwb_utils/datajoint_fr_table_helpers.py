# Helpers functions for firing rate map tables

import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry
from src.jguides_2024.spikes.jguidera_spikes import EpochSpikeTimesRelabel
from src.jguides_2024.utils.df_helpers import get_empty_df, df_filter_columns, copy_df_columns
from src.jguides_2024.utils.list_helpers import duplicate_elements
from src.jguides_2024.utils.make_rate_map import make_rate_map_constant_fs, smooth_mask_rate_map
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals, \
    get_event_times_relative_to_trial_start
from src.jguides_2024.utils.string_helpers import strip_string
from src.jguides_2024.utils.vector_helpers import vector_midpoints, convert_inf_to_nan


# Functions for making firing rate maps aligned to well events

def get_well_trials_bin_edges(trial_start_time_shift, trial_end_time_shift, bin_width, trial_intervals):

    # Get bin edges for each trial: start is shift from start event time, end is shift from end event time plus
    # the time between the start and end event times
    # ...Get time between start and end event times on each trial
    total_shift = trial_end_time_shift - trial_start_time_shift
    unshifted_trial_durations = np.diff(trial_intervals) - total_shift
    # ...Get bin edges for each trial
    trials_bin_edges = [np.arange(trial_start_time_shift,
                                  unshifted_trial_duration + trial_end_time_shift + bin_width,
                                  bin_width) for unshifted_trial_duration in unshifted_trial_durations]
    trials_bin_centers = list(map(vector_midpoints, trials_bin_edges))

    # Get a common set of bin edges spanning those from individual trials: take longest set of bin edges
    bin_edges = trials_bin_edges[np.argmax(list(map(len, trials_bin_edges)))]  # if "ties", argmax takes first
    bin_centers = vector_midpoints(bin_edges)

    return trials_bin_edges, trials_bin_centers, bin_edges, bin_centers


def _get_trial_time_info(trials_table, trials_params_table, key):

    # Get trial start and end time shifts
    trials_entry = (trials_table & key).fetch1()
    trial_start_time_shift, trial_end_time_shift = map(
        float, (trials_params_table & trials_entry).fetch1("trial_start_time_shift", "trial_end_time_shift"))

    # Get trial times
    trial_intervals = np.asarray(list(
        zip(trials_entry["trial_start_times"], trials_entry["trial_end_times"])))

    return trial_start_time_shift, trial_end_time_shift, trial_intervals


def _get_info_for_well_trial_table_fr_df(key, trials_table, trials_params_table, firing_rate_map_params_table):

    # Get trial time info
    (trial_start_time_shift, trial_end_time_shift, trial_intervals) = _get_trial_time_info(
        trials_table, trials_params_table, key)

    # Get bin params and make bins relative to event start time
    bin_width = float((firing_rate_map_params_table & key).fetch1(f"time_bin_width"))
    trials_bin_edges, trials_bin_centers, bin_edges, bin_centers = get_well_trials_bin_edges(
        trial_start_time_shift, trial_end_time_shift, bin_width, trial_intervals)

    # Get spike times in epoch
    epoch_spike_times = (EpochSpikeTimesRelabel.RelabelEntries & key).fetch1_dataframe()["epoch_spike_times"]

    return trial_start_time_shift, trial_end_time_shift, trial_intervals, bin_width, trials_bin_edges, \
           trials_bin_centers, bin_edges, bin_centers, epoch_spike_times


def _get_well_trial_rate_map(unit_spike_times, trial_intervals, trial_start_time_shift, trials_bin_centers,
                             bin_width, bin_edges):
    """
    Get spike times relative to closest "trial start event" (event used to define trial start time) (note this
    is distinct from trial start time, which is shifted trial start event)

    IMPORTANT NOTE TO USER: spikes are not "double counted" if trials overlap

    :param unit_spike_times: spike times from units
    :param trial_intervals: n by 2 array with trial start/stop times
    :param trial_start_time_shift: how much start event time was shifted to get trial start
    :param trials_bin_centers: bin centers for each set of trials. Use to get occupancy for rate mape
    :param bin_width: width of time bins
    :param bin_edges: vector. A common set of bin edges in which to find rate
    :return: rate map
    """

    spike_times_rel_event, _ = get_event_times_relative_to_trial_start(
        event_times=unit_spike_times, trial_intervals=trial_intervals, trial_start_time_shift=trial_start_time_shift)

    return make_rate_map_constant_fs(
        measurements=np.asarray([np.concatenate(trials_bin_centers)]),
        sampling_rate=1 / bin_width,
        measurements_at_events=np.asarray([spike_times_rel_event]),
        bin_edges=np.asarray([bin_edges]))[0]


def _get_fr_df_column_names(trial_feature_name=None):

    meta_data_column_names = ["unit_id"] + [trial_feature_name]*(trial_feature_name is not None)
    data_column_names = ["rate_map", "time_bin_centers", "time_bin_edges", "num_trials"]

    return meta_data_column_names + data_column_names


def make_well_trial_table_fr_df(key, trials_table, trials_params_table, firing_rate_map_params_table):

    # Get quantities for making rate maps
    trial_start_time_shift, trial_end_time_shift, trial_intervals, bin_width, _, trials_bin_centers, \
    bin_edges, bin_centers, epoch_spike_times = _get_info_for_well_trial_table_fr_df(
        key, trials_table, trials_params_table, firing_rate_map_params_table)

    # Return empty dataframe if no trial times
    column_names = _get_fr_df_column_names()
    if len(trial_intervals) == 0:
        return get_empty_df(column_names)

    # Otherwise, make rate maps
    rate_maps = [_get_well_trial_rate_map(
        unit_spike_times, trial_intervals, trial_start_time_shift, trials_bin_centers, bin_width, bin_edges)
        for unit_spike_times in epoch_spike_times]
    unit_ids = epoch_spike_times.index
    return pd.DataFrame.from_dict({"unit_id": unit_ids,
                                   "rate_map": rate_maps,
                                   f"time_bin_centers": [bin_centers] * len(unit_ids),
                                   f"time_bin_edges": [bin_edges] * len(unit_ids),
                                   "num_trials": [len(trial_intervals)] * len(unit_ids)})


def make_well_trial_table_fr_df_trial_subsets(
        key, trials_table, trials_params_table, firing_rate_map_params_table, trial_feature_name):

    # Get quantities for making rate maps
    trial_start_time_shift, trial_end_time_shift, trial_intervals, bin_width, _, trials_bin_centers, \
    bin_edges, bin_centers, epoch_spike_times = _get_info_for_well_trial_table_fr_df(
        key, trials_table, trials_params_table, firing_rate_map_params_table)

    # Return empty dataframe if no trial times or spike times
    column_names = _get_fr_df_column_names(trial_feature_name)
    if len(trial_intervals) == 0 or len(epoch_spike_times) == 0:
        return get_empty_df(column_names)

    # Separate trials into subsets according to value of trial_feature_name
    trial_feature_values = (trials_table & key).fetch1_dataframe(strip_s=True)[trial_feature_name].values
    trial_intervals_map = {trial_feature_value: trial_intervals[trial_feature_values == trial_feature_value]
                           for trial_feature_value in set(trial_feature_values)}  # map from trial feature value to trial intervals

    # Make rate maps
    data_list = []
    for trial_feature_value, trial_intervals_subset in trial_intervals_map.items():  # trial feature values
        for unit_id, unit_spike_times in epoch_spike_times.items():  # units
            rate_map = _get_well_trial_rate_map(unit_spike_times, trial_intervals_subset, trial_start_time_shift,
                             trials_bin_centers, bin_width, bin_edges)
            data_list.append((unit_id, trial_feature_value, rate_map, bin_centers, bin_edges, len(trial_intervals_subset)))
    return pd.DataFrame.from_dict({k: v for k, v in zip(column_names, zip(*data_list))})


def get_well_single_trial_table_fr_df_extra_info_names(strip_s=False):

    extra_info_names = ["epoch_trial_numbers", "trial_start_times",
                        "trial_end_times", "well_names",
                        "performance_outcomes", "reward_outcomes"]
    if strip_s:
        return [strip_string(x, "s") for x in extra_info_names]
    return extra_info_names


def make_well_single_trial_table_fr_df(key,
                                       trials_table,
                                       trials_params_table,
                                       firing_rate_map_params_table):

    # Get quantities for making rate maps
    trial_start_time_shift, trial_end_time_shift, trial_intervals, bin_width, trials_bin_edges, \
    trials_bin_centers, bin_edges, bin_centers, epoch_spike_times = _get_info_for_well_trial_table_fr_df(
        key, trials_table, trials_params_table, firing_rate_map_params_table)

    # Get column names without trailing s
    extra_info_names_no_s = get_well_single_trial_table_fr_df_extra_info_names(strip_s=True)

    # Return empty dataframe if no trial times
    empty_df = pd.DataFrame.from_dict(
            {k: [] for k in ["unit_id", "rate_map", "time_bin_centers", "time_bin_edges"] + extra_info_names_no_s})
    if len(trial_intervals) == 0:
        return empty_df

    # Get rate maps
    data_list = [(unit_id, make_rate_map_constant_fs(
            measurements=time_bin_edges,
            sampling_rate=1 / bin_width,
            # measurements_at_events: spike times in trial expressed relative to unshifted trials (do this so can use
            # bin edges, which span the start time and end time shifts. For example, for well arrival trials
            # with minus/plus one second shift, express spike times on domain: -1 to 1 second around well arrival)
            measurements_at_events=np.asarray([event_times_in_intervals(unit_spike_times, [trial_interval])[1] - \
                                               trial_interval[0] + trial_start_time_shift]),
            bin_edges=np.asarray([time_bin_edges]))[0],
                         time_bin_centers, time_bin_edges)
                        for trial_interval, time_bin_centers, time_bin_edges in zip(trial_intervals,
                                                                                      trials_bin_centers,
                                                                                      trials_bin_edges)
                        for unit_id, unit_spike_times in epoch_spike_times.items()]

    # Return empty dataframe if no units
    if len(data_list) == 0:
        return empty_df

    # Otherwise, return in dataframe with additional information
    unit_ids = list(epoch_spike_times.index)
    rate_map_dict = {k: v for k, v in zip(["unit_id", "rate_map", "time_bin_centers", "time_bin_edges"],
                                          zip(*data_list))}
    # Get trials entry so can store extra information from this table
    extra_info_names = get_well_single_trial_table_fr_df_extra_info_names()
    trials_entry = (trials_table & key).fetch1()
    extra_info_dict = {k_no_s: duplicate_elements(list(trials_entry[k]), len(unit_ids))
                       for k_no_s, k in zip(extra_info_names_no_s, extra_info_names)}

    return pd.DataFrame.from_dict({**rate_map_dict, **extra_info_dict})


# Functions for smoothing firing rate maps
def smooth_datajoint_table_fr(fr_table, params_table, key, data_type):

    rate_map_df = (fr_table & key).fetch1_dataframe()
    kernel_sd = (params_table & key).fetch1("kernel_sd")

    # Convert inf to nan in rate map, interpolate over nans, smooth rate map, then remask nans
    edges_rate_map_smoothed_list = [smooth_mask_rate_map(
        event_rate=convert_inf_to_nan(event_rate), bin_edges=bin_edges, sigma=(
                float(kernel_sd) / np.mean( np.diff(bin_edges))),  # express sigma in samples
        nan_treatment="interpolate") for bin_edges, event_rate in zip(
        rate_map_df[f"{data_type}_bin_edges"], rate_map_df["rate_map"])]

    # Get unit_id (this is index for across trial fr maps, and a column for
    # single trial fr maps (since in this case many entries for same unit id)
    if rate_map_df.index.name == "unit_id":
        unit_ids = rate_map_df.index
    else:
        unit_ids = rate_map_df["unit_id"]

    return pd.DataFrame.from_dict({
        "unit_id": unit_ids, "smoothed_rate_map": edges_rate_map_smoothed_list,
        get_bin_edges_name(data_type): rate_map_df[get_bin_edges_name(data_type)],
        get_bin_centers_name(data_type): rate_map_df[get_bin_centers_name(data_type)]})


def get_bin_edges_name(data_type):
    return f"{data_type}_bin_edges"


def get_bin_centers_name(data_type):
    return f"{data_type}_bin_centers"


def get_smoothed_fr_temporal_event_table_map():
    """
    :return: df with relevant table names and other information for tables with smoothed firing rate
             as a function of time (e.g. as a function of time from well arrival)
    """

    trial_align_events = ["well_arrival",
                          "unique_well_arrival",
                          "well_departure",
                          "unique_well_departure",
                          "stop_like_well_arrival"]

    # If a feature of trials was used to form trial subsets with which rate maps were formed, define name of that
    # trial feature here. "none" indicates no trial feature was used. Use string instead of None because hard
    # to work with None in searches.
    trial_feature_names = ["none",
                           "well_name",
                           "none",
                           "well_name",
                           "none"]
    trial_table_names = ["DioWellArrivalTrials",
                         "DioWellArrivalTrials",
                         "DioWellDepartureTrials",
                         "DioWellDepartureTrials",
                         "StopLikeWellArrivalTrials"]
    trial_params_table_names = ["DioWellArrivalTrialsParams",
                                "DioWellArrivalTrialsParams",
                                "DioWellDepartureTrialsParams",
                                "DioWellDepartureTrialsParams",
                                "StopLikeWellArrivalTrialsParams"]
    trial_param_names = ["dio_well_arrival_trials_param_name",
                         "dio_well_arrival_trials_param_name",
                         "dio_well_departure_trials_param_name",
                         "dio_well_departure_trials_param_name",
                         "stop_like_well_arrival_trials_param_name"]
    fr_map_smoothed_table_names = ["FrmapWellArrivalSm",
                                   "FrmapUniqueWellArrivalSm",
                                   "FrmapWellDepartureSm",
                                   "FrmapUniqueWellDepartureSm",
                                   "FrmapStopLikeWellArrivalSm"]

    return pd.DataFrame.from_dict({"trial_align_event": trial_align_events,
                                   "trial_feature_name": trial_feature_names,
                                   "fr_map_smoothed_table_name": fr_map_smoothed_table_names,
                                   "trial_table_name": trial_table_names,
                                   "trial_params_table_name": trial_params_table_names,
                                   "trial_param_name": trial_param_names}).set_index("trial_align_event")


def get_smoothed_fr_spatial_table_map():
    """
    :return: df with relevant table names and other information for tables with smoothed firing rate
             as a function of space (e.g. as a function of proportion path traversed)
    """

    trial_align_events = ["pupt"]
    trial_feature_names = ["path_name"]
    trial_table_names = [
        "DioWellDATrials"]  # pupt table depends on ppt which depends on DioWellDATrials
    trial_params_table_names = ["DioWellDATrialsParams"]
    trial_param_names = ["dio_well_da_trials_param_name"]
    fr_map_smoothed_table_names = ["FrmapPuptSm"]

    return pd.DataFrame.from_dict({"trial_align_event": trial_align_events,
                                   "trial_feature_name": trial_feature_names,
                                   "fr_map_smoothed_table_name": fr_map_smoothed_table_names,
                                   "trial_table_name": trial_table_names,
                                   "trial_params_table_name": trial_params_table_names,
                                   "trial_param_name": trial_param_names}).set_index("trial_align_event")


def get_smoothed_fr_table_map():

    # Get maps for space and time based firing rate maps
    smoothed_fr_spatial_table_map = get_smoothed_fr_spatial_table_map()
    smoothed_fr_temporal_event_table_map = get_smoothed_fr_temporal_event_table_map()

    # Add column to distinguish these two types of firing rate maps
    smoothed_fr_spatial_table_map["map_type"] = "spatial"
    smoothed_fr_temporal_event_table_map["map_type"] = "temporal"

    return pd.concat((smoothed_fr_spatial_table_map, smoothed_fr_temporal_event_table_map))


# Functions for inserting into firing rate map table

def insert_firing_rate_map_unique_well_table(
        table, trials_table, trials_params_table, firing_rate_map_params_table, trial_feature_name, key):

    # Get df with firing rate maps
    firing_rate_map_df = make_well_trial_table_fr_df_trial_subsets(
        key, trials_table=trials_table, trials_params_table=trials_params_table,
        firing_rate_map_params_table=firing_rate_map_params_table, trial_feature_name=trial_feature_name)

    # Insert separate entry for each trial feature value
    for trial_feature_value in set(firing_rate_map_df[trial_feature_name]):
        firing_rate_map_df_subset = df_filter_columns(firing_rate_map_df,
                                                      {trial_feature_name: trial_feature_value})
        # Update key for table entry with trial feature value
        key.update({trial_feature_name: trial_feature_value})
        insert_analysis_table_entry(table, nwb_objects=[firing_rate_map_df_subset], key=key)


def insert_single_trial_firing_rate_map_smoothed_well_table(fr_smoothed_table, fr_table, params_table, key, data_type):

    # Smooth rate maps
    smoothed_firing_rate_map_df = smooth_datajoint_table_fr(fr_table=fr_table,
                                                            params_table=params_table,
                                                            key=key,
                                                            data_type=data_type)

    # Copy additional columns from non-smoothed firing rate map table
    rate_map_df = (fr_table & key).fetch1_dataframe()
    extra_info_names = get_well_single_trial_table_fr_df_extra_info_names(strip_s=True)
    copy_df_columns(rate_map_df, smoothed_firing_rate_map_df, extra_info_names)

    # Store in table
    object_id_name = fr_smoothed_table.get_object_id_name()
    insert_analysis_table_entry(fr_smoothed_table, [smoothed_firing_rate_map_df], key, [object_id_name])