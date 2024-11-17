import os

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import custom datajoint tables
os.chdir("/home/jguidera/Src/jguides_2024/")

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    get_schema_table_names_from_file, populate_insert
from src.jguides_2024.position_and_maze.jguidera_position import IntervalPositionInfoRelabel
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDATrials, DioWellDATrialsParams
from src.jguides_2024.utils.plot_helpers import plot_spanning_line, format_ax
from src.jguides_2024.utils.similarity_measure_helpers import SimilarOverlapPeriods


schema_name = "jguidera_position_stop"
schema = dj.schema(schema_name)


def _get_da_trials_info(key):
    trials_entry = (DioWellDATrials & key)
    trial_intervals = trials_entry.trial_intervals()
    trial_path_names = (trials_entry).fetch1("path_names")
    return trial_intervals, trial_path_names


@schema
class StopLikeWellArrivalParams(SecKeyParamsBase):

    definition = """
    # Parameters for determining periods when rat speed resembles that during well arrival
    stop_like_well_arrival_param_name : varchar(500)
    ---
    well_arrival_forward_shift : decimal(10,5)
    well_arrival_backward_shift : decimal(10,5)
    speed_profile_overlap_threshold : decimal(10,5) unsigned
    ppt_lower_threshold : decimal(10,5) unsigned
    ppt_upper_threshold : decimal(10,5) unsigned
    combine_spans_within : decimal(10,5) unsigned
    """

    def _default_params(self):
        return [(1, -1, .8, .2, .95, .1)]


@schema
class StopLikeWellArrivalSel(SelBase):

    definition = """
    # Selection from upstream tables for StopLikeWellArrival
    -> DioWellDATrialsParams
    position_info_param_name : varchar(40)
    """

    def insert1(self, key):
        trial_start_time_shift, trial_end_time_shift = (DioWellDATrialsParams & key).trial_shifts()
        if trial_start_time_shift != 0 or trial_end_time_shift != 0:
            raise Exception(f"Trial start and end time for dio well departure to arrival trials must both be zero")
        super().insert1(key, skip_duplicates=True)

    def insert_defaults(self, **kwargs):
        for key in DioWellDATrialsParams.fetch("KEY"):
            key.update({"position_info_param_name": "default"})
            self.insert1(key)


@schema
class StopLikeWellArrival(ComputedBase):

    definition = """
    # Periods when rat speed resembles that during well arrival
    -> IntervalPositionInfoRelabel
    -> Ppt
    -> DioWellDATrials
    -> StopLikeWellArrivalParams
    -> StopLikeWellArrivalSel
    ---
    position_stop_times : blob
    well_arrival_times : blob
    wa_average_speed_profile : blob
    above_overlap_thresh_idxs : blob
    above_overlap_thresh_spans : blob
    valid_above_overlap_thresh_spans : blob
    -> nd.common.AnalysisNwbfile
    stop_like_well_arrival_object_id : varchar(40)
    """

    def make(self, key):

        well_event_name = "well_arrival"
        params_table_name = StopLikeWellArrivalParams
        params_entry = (params_table_name & key).fetch1()
        well_event_times = (DioWellDATrials & key).fetch1("trial_end_times")
        speed = (IntervalPositionInfoRelabel.RelabelEntries & key).fetch1_dataframe()["head_speed"]
        da_trial_intervals, _ = _get_da_trials_info(key)  # departure to arrival time_and_trials

        similar_overlap_periods_obj = SimilarOverlapPeriods(time_series=speed,
                                                            event_times=well_event_times,
                                                            forward_shift=float(params_entry[f"{well_event_name}_forward_shift"]),
                                                            backward_shift=float(params_entry[f"{well_event_name}_backward_shift"]),
                                                            overlap_threshold=float(params_entry['speed_profile_overlap_threshold']),
                                                            valid_intervals=da_trial_intervals,
                                                            external_time_series=(Ppt &
                                                                                  key).ppt_all_time(),
                                                            external_time_series_lower_threshold=float(params_entry['ppt_lower_threshold']),
                                                            external_time_series_upper_threshold=float(params_entry['ppt_upper_threshold']),
                                                            combine_close_spans_threshold=float(params_entry['combine_spans_within']),
                                                            ignore_external_vector_nan=True)  # ignore nans in ppt

        key.update({"position_stop_times": np.asarray(similar_overlap_periods_obj.max_overlap_times),
                    "well_arrival_times": np.asarray(well_event_times),
                    "wa_average_speed_profile": similar_overlap_periods_obj.average_profile,
                    "above_overlap_thresh_idxs": similar_overlap_periods_obj.above_overlap_thresh_idxs,
                    "above_overlap_thresh_spans": similar_overlap_periods_obj.above_overlap_thresh_spans,
                    "valid_above_overlap_thresh_spans": similar_overlap_periods_obj.valid_above_overlap_thresh_spans})
        speed_profile_overlap_df = pd.DataFrame.from_dict(
            {"time": similar_overlap_periods_obj.ts_profile_overlap.index,
             "speed_profile_overlap": similar_overlap_periods_obj.ts_profile_overlap.values})
        insert_analysis_table_entry(self, [speed_profile_overlap_df], key)

    def plot_overlap_at_well_arrivals(self, well_event_name):
        # Plot overlap measure at well events
        nwb_file_name, epoch, well_arrival_times = self.fetch1("nwb_file_name",
                                                               "epoch",
                                                               "well_arrival_times")
        speed_profile_overlap = self.fetch1_dataframe()
        fig, ax = plt.subplots(figsize=(3, 3))
        well_arrival_idxs = np.searchsorted(speed_profile_overlap.index,
                                            well_arrival_times)  # indices in speed profile close to well events
        counts = ax.hist(speed_profile_overlap.iloc[well_arrival_idxs], color="gray")[
            0]  # overlap between speed and average speed profile near well events
        speed_profile_overlap_threshold = float(
            (StopLikeWellArrivalParams & self.fetch1()).fetch1('speed_profile_overlap_threshold'))
        plot_spanning_line(counts, speed_profile_overlap_threshold, span_axis="y", ax=ax)
        format_ax(ax, title=f"{nwb_file_name}\nep{epoch}", ylabel="count",
                  xlabel=f"overlap between speed \n & average speed profile  \n at {well_event_name}")

    def plot_average_speed_profile(self):
        # Plot average speed profile at well event
        (nwb_file_name, epoch, average_speed_profile,
         stop_like_well_arrival_param_name) = self.fetch1("nwb_file_name",
                                                          "epoch",
                                                          "wa_average_speed_profile",
                                                          "stop_like_well_arrival_param_name")
        (well_arrival_backward_shift,
         well_arrival_forward_shift) = (StopLikeWellArrivalParams & {
                                    "stop_like_well_arrival_param_name": stop_like_well_arrival_param_name}).fetch1(
                                    "well_arrival_backward_shift",
                                    "well_arrival_forward_shift")
        well_arrival_backward_shift = float(well_arrival_backward_shift)
        well_arrival_forward_shift = float(well_arrival_forward_shift)
        fig, ax = plt.subplots(figsize=(3, 3))
        rel_time = np.linspace(float(well_arrival_backward_shift),
                               float(well_arrival_forward_shift),
                               len(average_speed_profile))  # time relative to position_and_maze stop
        ax.plot(rel_time, average_speed_profile, color='black')
        format_ax(ax, title=f"{nwb_file_name}\nep{epoch}", ylabel="speed", xlabel="time from 'stop' (s)")

    def plot_stop_periods(self, well_event_name, ax=None, plot_legend=True):
        # Plot position_and_maze information, stop periods, and intermediates along way to finding stop periods.
        # Plot time from epoch start so can more easily compare to Trodes plaback.
        fontsize1 = 10
        # Position information
        key = {k: v for k, v in self.fetch1().items() if k in IntervalPositionInfoRelabel.RelabelEntries.primary_key}
        pos_df = (IntervalPositionInfoRelabel.RelabelEntries & key).fetch1_dataframe()  # position_and_maze data
        epoch_start_time = pos_df["head_speed"].index[0]  # epoch start time
        t = pos_df["head_speed"].index - epoch_start_time  # time from epoch start
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(t, pos_df["head_speed"], label="head speed")
        ax.plot(t, pos_df["head_position_x"], label="head x position_and_maze")
        ax.plot(t, pos_df["head_position_y"], label="head y position_and_maze")
        # Horizontal lines to visualize zero speed and 4cm/s speed
        for plot_y in [0, 4]:
            plot_spanning_line(t, plot_y, ax, span_axis="x")
        # Departure to arrival trials and text for corresponding paths
        trial_intervals, trial_path_names = _get_da_trials_info(key)
        for idx, (interval, path_name) in enumerate(zip(trial_intervals,
                                                        trial_path_names)):
            label = None
            if idx == 0:
                label = "departure to arrival time_and_trials"
            plot_x = interval - epoch_start_time
            ax.plot(plot_x, [50] * 2, 'o-', color='red', label=label)
            ax.text(plot_x[0], 60,
                    path_name.replace("_well_to_", "\n").split("_well")[0], fontsize=fontsize1)  # path name
        # Speed overlap with average speed profile at well event
        speed_profile_overlap = self.fetch1_dataframe()
        ax.plot(speed_profile_overlap.index - epoch_start_time, speed_profile_overlap['speed_profile_overlap'] * 100,
                alpha=.5,
                label=f"overlap between speed and average \n speed profile at {well_event_name}")
        # Above threshold samples
        table_entry = self.fetch1()
        above_overlap_thresh_idxs = table_entry['above_overlap_thresh_idxs']
        ax.plot(t[above_overlap_thresh_idxs],
                [60] * len(above_overlap_thresh_idxs), 'o', label="above overlap thresh samples")
        # Spans of above threshold samples
        for idx, (span_start, span_end) in enumerate(table_entry['above_overlap_thresh_spans']):
            label = None
            if idx == 0:
                label = "above overlap thresh spans"
            ax.plot([t[span_start], t[span_end]], [70] * 2, '-o', color='gray', label=label)
        # Plot valid spans of points that exceed overlap threshold
        for idx, (span_start, span_end) in enumerate(table_entry['valid_above_overlap_thresh_spans']):
            label = None
            if idx == 0:
                label = "valid above overlap thresh spans"
            ax.plot([t[span_start], t[span_end]],
                    [70] * 2, '-o', color='black', label=label)
        # Plot position_and_maze stop points
        position_stop_times = table_entry['position_stop_times']
        ax.scatter(position_stop_times - epoch_start_time,
                   [80] * len(position_stop_times), color="red", s=50, label="position_and_maze stops")
        # Legend
        if plot_legend:
            ax.legend()


def populate_jguidera_position_stop(key=None, tolerate_error=False):
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_position_stop():
    from development.jguidera_position_stop_trials import drop_jguidera_position_trials
    from development.jguidera_position_stop_firing_rate_map import drop_jguidera_position_stop_firing_rate_map
    drop_jguidera_position_stop_firing_rate_map()
    drop_jguidera_position_trials()
    schema.drop()
