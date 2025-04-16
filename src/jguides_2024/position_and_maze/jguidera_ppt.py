# This module defines tables related to proportion path traversed

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import (SelBase, SecKeyParamsBase, ComputedBase)
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, \
    delete_multiple_flexible_key
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    intersect_tables
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.datajoint_nwb_utils.trials_container_helpers import trials_container_default_time_vector
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.position_and_maze.jguidera_maze import (TrackGraphUniversalTrackGraphMap,
                                                              UniversalForkMazePathEdgePathFractionMap,
                                                              AnnotatedUniversalTrackGraph)
from src.jguides_2024.position_and_maze.jguidera_maze import get_default_environment_track_graph_name_map
from src.jguides_2024.position_and_maze.jguidera_position import IntervalLinearizedPositionRescaled
from src.jguides_2024.task_event.jguidera_dio_trials import (DioWellDATrials, populate_jguidera_dio_trials)
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDATrialsParams
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.df_helpers import df_filter_index, zip_df_columns, df_filter_columns, \
    df_filter1_columns, df_from_data_list, df_filter_columns_isin
from src.jguides_2024.utils.digitize_helpers import digitize_indexed_variable
from src.jguides_2024.utils.make_bins import make_bin_edges
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool as in_intervals_bool
from src.jguides_2024.utils.vector_helpers import (find_scale_factors, check_in_range,
                                                   find_spans_increasing_list)
from src.jguides_2024.utils.vector_helpers import linspace

# Needed for table definitions
RunEpoch
nd

schema_name = "jguidera_ppt"
schema = dj.schema(schema_name)  # define custom schema


@schema
class PptParams(SecKeyParamsBase):
    definition = """
    # Consolidate primary keys for Ppt
    ppt_param_name : varchar(40)  # reflects set of track graphs (one per environment) and linearization param
    ---
    -> DioWellDATrialsParams  # describes da trials shift, will not enter into primary key
    position_info_param_name : varchar(40)
    interval_linearized_position_relabel_param_name : varchar(40)
    environment_track_graph_name_map : blob
    """

    @staticmethod
    def _valid_params():
        return {"dio_well_da_trials_param_name": DioWellDATrialsParams().lookup_no_shift_param_name(),
                "position_info_param_name": "default"}

    # Override parent class method since non-canonical param name
    def get_default_param_name(self, as_dict=False):
        default_param_name = "ppt_1"
        if as_dict:
            return {"ppt_param_name": default_param_name}
        return "ppt_1"

    def insert1(self, key, **kwargs):
        # Enforce no shift of trial start/end times and single position_and_maze info param name
        for meta_param_name, param_name_value in self._valid_params().items():
            if key[meta_param_name] != param_name_value:
                raise Exception(f"meta_param_name must be {param_name_value}")

        # Insert into table
        super().insert1(key, **kwargs)

    def insert_defaults(self, **kwargs):
        # Insert into upstream table
        DioWellDATrialsParams().insert_defaults()

        # Define base params
        key = self._valid_params()
        interval_linearized_position_relabel_param_name = "default"

        # Defaults for default fork maze track graphs
        ppt_param_name = self.get_default_param_name()
        key.update({"interval_linearized_position_relabel_param_name": interval_linearized_position_relabel_param_name,
                    "environment_track_graph_name_map": get_default_environment_track_graph_name_map(),
                    "ppt_param_name": ppt_param_name})  # must update to make param name
        self.insert1(key)


@schema
class PptSel(SelBase):
    definition = """
    # Selection from upstream tables for Ppt
    -> RunEpoch
    -> PptParams
    """

    def insert_defaults(self, **kwargs):
        # Populate upstream table
        PptParams().insert_defaults()
        # Restrict insertion to those for which the upstream tables ultimately used are populated
        for key in (DioWellDATrials * IntervalLinearizedPositionRescaled * PptParams).fetch("KEY"):
            super().insert1(intersect_tables([RunEpoch, PptParams], key).fetch1("KEY"))


@schema
class Ppt(ComputedBase):
    definition = """
    # Proportion path traversed
    -> PptSel
    ---
    -> nd.common.AnalysisNwbfile
    ppt_object_id : varchar(40)
    ppt_range : blob
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> Ppt
        -> DioWellDATrials
        -> IntervalLinearizedPositionRescaled
        -> TrackGraphUniversalTrackGraphMap
        """

    @staticmethod
    def get_range():
        # Hard coded ppt range
        return [0, 1]

    @classmethod
    def make_bin_edges(cls, bin_width):
        return make_bin_edges(cls.get_range(), bin_width)

    def make(self, key):
        # Get range of values ppt can take on
        ppt_range = self.get_range()

        # Get key to upstream tables
        upstream_key = {**key, **(PptParams & key).fetch1()}
        # Replace map to track graph with track graph name so can query upstream tables
        environment_track_graph_name_map = upstream_key.pop("environment_track_graph_name_map")
        upstream_key["track_graph_name"] = environment_track_graph_name_map[
            TaskIdentification.get_environment(key["nwb_file_name"], key["epoch"])]

        # Get trials info
        trials_df = (DioWellDATrials & upstream_key).fetch1_dataframe()

        # Get path fraction map
        universal_track_graph_name = (TrackGraphUniversalTrackGraphMap & upstream_key).fetch1("universal_track_graph_name")
        path_fraction_map = (UniversalForkMazePathEdgePathFractionMap &
                             {**upstream_key, **{"universal_track_graph_name": universal_track_graph_name}}).fetch1_dataframe()
        # Get pos info
        pos_df = (IntervalLinearizedPositionRescaled & upstream_key).fetch1_dataframe()
        directional_edge_names_pos_map = {k: v for k, v in list(
            zip(*(AnnotatedUniversalTrackGraph & {"universal_track_graph_name":
                                                  universal_track_graph_name}).fetch1("directional_edge_names",
                                                             "directional_edge_linear_position")))}

        ppt_trials_list = []
        trials_time_list = []
        for trial_start_time, trial_end_time, path_name in zip_df_columns(
                trials_df, ["trial_start_times", "trial_end_times", "path_names"]):  # path names for da trials

            pos_trial_df = df_filter_index(pos_df, [[trial_start_time, trial_end_time]])  # pos during trial

            ppt_list = []
            for pos, edge_name in zip_df_columns(
                    pos_trial_df, ["linear_position_rescaled", "edge_name"]):
                path_fraction_map_entry = df_filter_columns(
                    path_fraction_map, {"path_names": path_name, "path_edge_names": edge_name})

                if len(path_fraction_map_entry) > 1:
                    raise Exception(f"More than one entry found in path_fraction_map for given position/edge")

                elif len(path_fraction_map_entry) == 0:  # occurs if edge not on path
                    ppt_list.append(np.nan)  # in this case set trajectory phase to nan

                elif len(path_fraction_map_entry) == 1:
                    _, path_edge_name, path_directional_edge_name, path_fraction = \
                    df_filter1_columns(path_fraction_map, {"path_names": path_name,
                                                          "path_edge_names": edge_name}).to_numpy()[0]
                    directional_edge_pos = directional_edge_names_pos_map[path_directional_edge_name]
                    check_in_range(pos, directional_edge_pos)  # check that position_and_maze on directional edge

                    # Find percent directional edge traversed
                    percent_directional_edge_traversed = abs(
                        (pos - directional_edge_pos[0]) / np.diff(directional_edge_pos)[0])

                    # Convert to fraction of path traversed
                    mult_factor, shift_factor = find_scale_factors(x=path_fraction, y=ppt_range)
                    ppt_list.append(
                        percent_directional_edge_traversed * mult_factor + shift_factor)
            trials_time_list.append(pos_trial_df.index.to_numpy())
            ppt_trials_list.append(ppt_list)

        # Insert into main table
        key.update({"ppt_range": ppt_range})
        ppt_df = pd.DataFrame.from_dict({"trials_time": trials_time_list,
                                         "trials_ppt": ppt_trials_list,
                                         "trials_path_name": trials_df["path_names"].values,
                                         "trial_start_epoch_trial_numbers": trials_df.reset_index()[
                                         "trial_start_epoch_trial_numbers"].values})
        insert_analysis_table_entry(self, [ppt_df], key, ["ppt_object_id"])

        # Insert into part table
        insert1_print(self.Upstream, upstream_key)

    def ppt_all_time(self, new_index=None):
        return self.ppt_df_all_time(new_index)["ppt"]

    def ppt_df_all_time(self, new_index=None):
        # Get inputs if not passed
        nwb_file_name, epoch = self.fetch1("nwb_file_name", "epoch")
        time_vector = trials_container_default_time_vector(nwb_file_name, [epoch], new_index)

        # Proportion path traversed in main table is by trial. Combine across trials
        ppt_df = self.fetch1_dataframe()

        # Initialize vectors for ppt and corresponding path names/epoch numbers across all epoch
        ppt = np.asarray([np.nan] * len(time_vector))
        path_names = np.asarray([None] * len(time_vector), dtype=object)
        epoch_trial_numbers = np.asarray([np.nan] * len(time_vector))

        # Interpolate within trials
        for epoch_trial_number, df_row in ppt_df.iterrows():  # trials
            # Unpack variables
            trial_t = df_row["trials_time"]
            trial_ppt = df_row["trials_ppt"]
            trial_path_name = df_row["trials_path_name"]
            # Find times in index within current ppt interval
            idxs, ppt_trial_t = event_times_in_intervals(time_vector, [[trial_t[0], trial_t[-1]]])
            ppt[idxs] = np.interp(ppt_trial_t, trial_t, trial_ppt)
            path_names[idxs] = [trial_path_name]*len(idxs)
            epoch_trial_numbers[idxs] = [epoch_trial_number]*len(idxs)

        # Return df with ppt across all time
        return pd.DataFrame.from_dict({"time": time_vector,
                                       "ppt": ppt,
                                       "path_name": path_names,
                                       "epoch_trial_number": epoch_trial_numbers}).set_index("time")

    def digitized_ppt_df_all_time(self, new_index=None, bin_width=None, bin_edges=None, verbose=False):
        # Check inputs
        check_one_none([bin_width, bin_edges], ["bin_width", "bin_edges"])

        # Make bin edges if not passed
        if bin_edges is None:
            bin_edges = self.make_bin_edges(bin_width)

        # Get ppt and reindex in this process
        ppt_df = self.ppt_df_all_time(new_index)

        # Digitize ppt
        ppt_df["digitized_ppt"] = digitize_indexed_variable(
                                    indexed_variable=ppt_df["ppt"],
                                    bin_edges=bin_edges,
                                    verbose=verbose)

        return ppt_df

    def ppt_df_all_trials(self):
        ppt_df = self.fetch1_dataframe()
        trials_ppt = np.concatenate(ppt_df["trials_ppt"])
        trials_paths = np.concatenate([[path_name]*len(trial_ppt) for trial_ppt, path_name in
                                       zip_df_columns(ppt_df, ["trials_ppt",
                                                               "trials_path_name"])])
        return pd.DataFrame.from_dict({"all_trials_ppt": trials_ppt,
                                       "all_trials_paths": trials_paths,
                                       "all_trials_time": np.concatenate(ppt_df["trials_time"])}).set_index("all_trials_time")

    def trial_intervals(self):
        ppt_df = self.fetch1_dataframe()
        return [(trial_time[0], trial_time[-1]) for trial_time in ppt_df["trials_time"]]

    def upsample_trials(self, upsample_fs):
        # Upsample ppt within trials
        from src.jguides_2024.utils.list_helpers import return_n_empty_lists
        upsample_t_bin_width = 1/upsample_fs
        ppt_df = self.fetch1_dataframe()
        upsample_trials_time, upsample_trials_ppt, upsample_path_names = return_n_empty_lists(3)
        for trial_time, trial_ppt, trial_path_name in zip_df_columns(ppt_df, ["trials_time",
                                                             "trials_ppt",
                                                             "trials_path_name"]):
            new_trial_time = np.arange(trial_time[0], trial_time[-1], upsample_t_bin_width)
            new_trial_ppt = np.interp(new_trial_time, trial_time, trial_ppt)
            upsample_trials_time += list(new_trial_time)
            upsample_trials_ppt += list(new_trial_ppt)
            upsample_path_names += [trial_path_name]*len(new_trial_ppt)
        return pd.DataFrame.from_dict({"time": upsample_trials_time,
                                                  "ppt": upsample_trials_ppt,
                                                  "path_name": upsample_path_names}).set_index("time")

    def get_ppt_trials(self, valid_ppt_intervals, verbose=False):
        # Find spans of time when ppt within valid intervals
        # Approach: If we just use times corresponding to ppt samples to define valid intervals, we would typically
        # cut intervals short, since ppt likely to have entered valid range between two discrete samples (only case
        # where this doesnt happen is when ppt samples fall on valid interval bounds). To account for this, we
        # look backwards and forwards one sample from start / end of time spans identified above
        # and use linear interpolation to find time when ppt would have been at boundary of valid ppt interval
        # that was entered

        # Get df with proportion path traversed by trial
        ppt_df = self.fetch1_dataframe()  # each row corresponds to a trial
        # Loop through trials and find spans of time when ppt within valid ppt intervals
        data_list = []
        for trial_start_epoch_trial_number, df_row in ppt_df.iterrows():
            trial_ppt = pd.Series(df_row.trials_ppt, index=df_row.trials_time)
            for valid_ppt_interval in valid_ppt_intervals:
                # Find spans of ppt within valid interval
                valid_ppt_idxs = np.where(in_intervals_bool(trial_ppt.values, [valid_ppt_interval]))[0]
                valid_ppt_idx_spans = find_spans_increasing_list(valid_ppt_idxs)[0]
                # For each span, find start and end times corresponding to ppt bounds
                for span_start_idx, span_end_idx in valid_ppt_idx_spans:
                    # Define default start and end times as current times
                    start_time, end_time = trial_ppt.index[span_start_idx], trial_ppt.index[span_end_idx]
                    # Unpack valid ppt interval bounds
                    valid_ppt_interval_start, valid_ppt_interval_end = valid_ppt_interval
                    # 1) Find time corresponding to ppt lower bound
                    if span_start_idx != 0:
                        previous_ppt = trial_ppt.iloc[span_start_idx - 1]
                        # if previous ppt is not nan and is less than valid ppt lower bound
                        if not np.isnan(
                                previous_ppt) and previous_ppt < valid_ppt_interval_start:
                            current_ppt = trial_ppt.iloc[span_start_idx]
                            current_time = trial_ppt.iloc[span_start_idx]
                            previous_time = trial_ppt.iloc[span_start_idx - 1]
                            m = (current_ppt - previous_ppt) / (current_time - previous_time)
                            y = (valid_ppt_interval_start - current_ppt)
                            start_time += y / m
                    # 2) Find time corresponding to ppt upper bound
                    if span_end_idx != len(trial_ppt) - 1:
                        next_ppt = trial_ppt.iloc[span_end_idx + 1]
                        # if next ppt is not nan and is greater than valid ppt upper bound
                        if not np.isnan(
                                next_ppt) and next_ppt > valid_ppt_interval_end:
                            current_ppt = trial_ppt.iloc[span_end_idx]
                            current_time = trial_ppt.iloc[span_end_idx]
                            next_time = trial_ppt.iloc[span_end_idx + 1]
                            m = (current_ppt - next_ppt) / (current_time - next_time)
                            y = (valid_ppt_interval_end - current_ppt)
                            end_time += y / m
                    data_list.append((start_time, end_time, trial_start_epoch_trial_number, df_row.trials_path_name))
        # Store trials information in dictionary
        trials_map = {k: v for k, v in zip(["trial_start_times", "trial_end_times",
                                            "trial_start_epoch_trial_numbers", "trial_path_names"],
                                     zip(*data_list))}

        # Plot valid time intervals if indicated
        if verbose:
            # Initialize plot
            fig, ax = plt.subplots(figsize=(15, 3))
            # Plot ppt
            ppt_all_time = pd.Series(np.concatenate(ppt_df.trials_ppt),
                                     index=np.concatenate(ppt_df.trials_time))
            ax.plot(ppt_all_time, 'o', color="gray")
            valid_time_intervals = list(zip(trials_map["trial_start_times"],
                                            trials_map["trial_end_times"]))
            valid_bool = in_intervals_bool(ppt_all_time.index, valid_time_intervals)
            ax.plot(ppt_all_time[valid_bool], 'x', color="green")
            # Plot valid ppt intervals
            xlims = [np.min(ppt_all_time.index),
                     np.max(ppt_all_time.index)]
            for valid_ppt_interval in valid_ppt_intervals:
                for x in valid_ppt_interval:
                    ax.plot(xlims, [x] * 2, color="green")
            # Plot identified valid time intervals
            for valid_time_interval in valid_time_intervals:
                ax.plot(valid_time_interval, [1.1] * 2, color="green")

        return trials_map

    def delete_(self, key):
        delete_multiple_flexible_key([self], key)

    @classmethod
    def get_ppt_bin_edges(cls, ppt_bin_width=.1):
        if ppt_bin_width < 0 or ppt_bin_width > 1:
            raise Exception(f"ppt_bin_width must be on [0, 1] but is {ppt_bin_width}")
        return linspace(*cls.get_range(), ppt_bin_width)

    def get_first_crossed_junction_df(self, verbose=False):

        # Get when rat first crossed maze junctions on each trial
        ppt_df = self.fetch1_dataframe()

        from src.jguides_2024.position_and_maze.jguidera_maze import get_n_junction_path_junction_fractions, return_n_junction_path_names
        junction_fractions = get_n_junction_path_junction_fractions(2)

        # Restrict to valid paths (two maze junctions)
        valid_path_names = return_n_junction_path_names(2)
        ppt_df = df_filter_columns_isin(ppt_df, {"trials_path_name": valid_path_names})

        # Estimate when rat first crossed each track junction, for paths with two maze junctions
        data_list = []
        for epoch_trial_number, trials_ppt, trials_time in zip(
                ppt_df.trial_start_epoch_trial_numbers, ppt_df.trials_ppt, ppt_df.trials_time):
            for junction_num, junction_fraction in enumerate(junction_fractions):
                crossed_idx = np.where(trials_ppt >= junction_fraction)[0][0]

                if crossed_idx == 0:
                    raise Exception(f"Expecting idx for when rat first crossed junction to be greater than zero")

                time_estimate = np.interp(
                    junction_fraction, trials_ppt[crossed_idx - 1:crossed_idx],
                    trials_time[crossed_idx - 1:crossed_idx])

                data_list.append((epoch_trial_number, junction_num, time_estimate, junction_fraction))

        first_crossed_junction_df = df_from_data_list(
            data_list, ["epoch_trial_number", "junction_number", "time_estimate", "ppt"]).set_index(
            "epoch_trial_number")

        # Plot estimates along with ppt for sanity check if indicated
        if verbose:
            ppt_df = self.fetch1_dataframe()
            fig, ax = plt.subplots(figsize=(20, 5))
            for _, df_row in ppt_df.iterrows():
                ax.plot(df_row.trials_time, df_row.trials_ppt, ".")
            for _, df_row in first_crossed_junction_df.iterrows():
                ax.plot(df_row.time_estimate, df_row.ppt, "x")

        return first_crossed_junction_df


@schema
class PptBinEdgesParams(SecKeyParamsBase):
    definition = """
    # Parameters for proportion path traversed bins
    ppt_bin_edges_param_name : varchar(40)
    ---
    ppt_bin_width : decimal(10,5) unsigned
    """

    def _default_params(self):
        return [[.05]]


@schema
class PptBinEdges(ComputedBase):
    definition = """
    # Bin edges for proportion path traversed 
    -> PptBinEdgesParams
    ---
    ppt_bin_edges : blob  # proportion path traversed bin edges for maze edges
    """

    def make(self, key):
        ppt_bin_width = float((PptBinEdgesParams & key).fetch1("ppt_bin_width"))
        ppt_start, ppt_end = Ppt.get_range()
        ppt_bin_edges = np.arange(ppt_start, ppt_end + ppt_bin_width, ppt_bin_width)  # make bins
        key.update({"ppt_bin_edges": ppt_bin_edges})  # update key
        insert1_print(self, key)


def populate_jguidera_ppt(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_ppt"
    upstream_schema_populate_fn_list = [populate_jguidera_dio_trials]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list, populate_upstream_limit,
                    populate_upstream_num)


def drop_jguidera_ppt():
    from src.jguides_2024.position_and_maze.jguidera_ppt_interp import drop_jguidera_ppt_interp
    from src.jguides_2024.time_and_trials.jguidera_ppt_trials import drop_jguidera_ppt_trials
    from src.jguides_2024.firing_rate_map.jguidera_ppt_firing_rate_map import drop_jguidera_ppt_firing_rate_map
    from development.jguidera_position_stop import drop_jguidera_position_stop
    drop_jguidera_ppt_interp()
    drop_jguidera_ppt_trials()
    drop_jguidera_ppt_firing_rate_map()
    drop_jguidera_position_stop()
    schema.drop()



