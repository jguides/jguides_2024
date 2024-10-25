# This module defines tables related to position_and_maze

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd
from spyglass.common.common_position import (IntervalLinearizedPosition, IntervalPositionInfo)
from spyglass.utils.dj_helper_fn import fetch_nwb

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import fetch1_dataframe, \
    insert_analysis_table_entry, get_schema_table_names_from_file, \
    add_param_defaults, get_valid_position_info_param_names, populate_insert, insert1_print, get_key_filter
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_names, \
    get_jguidera_nwbf_epoch_keys
from src.jguides_2024.position_and_maze.jguidera_maze import (TrackGraphUniversalTrackGraphMap,
                                                              AnnotatedTrackGraph, AnnotatedUniversalTrackGraph,
                                                              flip_nodes_edge_name,
                                                              get_universal_track_graph_name)
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.utils.digitize_helpers import digitize_indexed_variable
from src.jguides_2024.utils.make_bins import make_bin_edges
from src.jguides_2024.utils.vector_helpers import (remove_repeat_elements, vector_midpoints, none_to_string_none,
                                                   return_constant_vector)
from src.jguides_2024.utils.vector_helpers import unpack_single_element

schema_name = "jguidera_position"
schema = dj.schema(schema_name)  # define custom schema


def linear_positions_to_edge_idxs(linear_positions, edge_linear_position):
    edge_linear_position = np.asarray(edge_linear_position)
    edge_idxs = []  # initialize list for edges identified using edge_linear_position
    for pos in linear_positions:
        if np.isnan(pos):  # if pos is nan, set edge number to nan
            edge_idxs.append(np.nan)
        else:
            z = np.where(np.product(edge_linear_position - pos, axis=1) <= 0)[0]  # edge number matches
            if len(z) != 1:  # check that one edge found for this position_and_maze
                raise Exception(f"Should have found exactly one edge for position_and_maze but found {len(z)}")
            edge_idxs.append(z[0])
    edge_idxs = np.asarray(edge_idxs)
    return np.reshape(edge_idxs, (len(edge_idxs)))  # reshape to vector


def linear_positions_to_edge_names(linear_positions, edge_linear_position, edge_names):
    linear_position_edge_idxs = linear_positions_to_edge_idxs(linear_positions, edge_linear_position)
    return np.asarray(edge_names)[linear_position_edge_idxs]


def plot_edge_linear_position(edge_linear_position, x_extent, ax):
    for edge in edge_linear_position:  # plot edge boundaries
        for node in edge:
            ax.plot(x_extent, [node] * 2, color="gray")


def get_fractional_edge_position(linear_position, linear_position_edge_idxs, edge_linear_position):
    pos_edge_lengths = np.ndarray.flatten(np.diff(edge_linear_position)[
                                          linear_position_edge_idxs])  # array with edge lengths for linear position_and_maze sample edges
    edge_startpoints = edge_linear_position[:, 0][
        linear_position_edge_idxs]  # linear position_and_maze of end node for edges of linear position_and_maze samples
    fractional_edge_position = (np.asarray(linear_position) - edge_startpoints) / pos_edge_lengths
    if not all(fractional_edge_position[np.invert(np.isnan(fractional_edge_position))] >= 0):  # check that non-nan fractional edge positions are nonnegative
        raise Exception(f"Non-nan fractional edge position_and_maze values should all be nonnegative")
    return fractional_edge_position


def fetch1_dataframe_position_info(table_entry):
    nwb_data = unpack_single_element(table_entry.fetch_nwb(close_file=False))
    index = pd.Index(np.asarray(
        nwb_data['head_position'].get_spatial_series().timestamps), name='time')
    columns = ['head_position_x', 'head_position_y', 'head_orientation',
               'head_velocity_x', 'head_velocity_y', 'head_speed']
    return pd.DataFrame(np.concatenate(
        (np.asarray(nwb_data['head_position'].get_spatial_series().data),
         np.asarray(nwb_data['head_orientation'].get_spatial_series().data)[
         :, np.newaxis],
         np.asarray(nwb_data['head_velocity'].time_series['head_velocity'].data)),
        axis=1), columns=columns, index=index)


@schema
class IntervalPositionInfoRelabelParams(SecKeyParamsBase):
    definition = """
    # Exchange interval_list_name for epoch for entries in IntervalPositionInfo
    position_info_param_name : varchar(50)
    -> TaskIdentification
    ---
    -> EpochIntervalListName
    """

    def insert_single_epoch(self, nwb_file_name, epoch, position_info_param_name=None, tolerate_no_entry=False):
        interval_list_name = EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch, tolerate_no_entry)
        # If no epoch found and tolerating this, print message and exit
        if interval_list_name is None:
            print(f"Could not populate IntervalLinearizedPositionRelabelParams for {nwb_file_name}, epoch {epoch} "
                  f"because no relevant entry in EpochIntervalListName with which to get interval list name. Exiting")
            return
        # Otherwise, populate table
        key = {"nwb_file_name": nwb_file_name,
               "epoch": epoch,
               "interval_list_name": interval_list_name,
               "position_info_param_name": position_info_param_name}
        # Get inputs if not passed
        key = add_param_defaults(key)
        self.insert1(key, skip_duplicates=True)

    def insert_defaults(self, **kwargs):
        # Define keys. If key_filter passed, define a single key that has nwb_file_name and epoch (if present
        # in key_filter_. Otherwise define nwbf/epoch keys with possible settings of position_info_param_name
        key_filter = get_key_filter(kwargs)
        key_ = {k: v for k, v in key_filter.items() if k in ["nwb_file_name", "epoch"]}
        if len(key_) > 0:
            keys = [key_]
        else:
            keys = [{**k, **{"position_info_param_name": position_info_param_name}} for k in
                    get_jguidera_nwbf_epoch_keys() for position_info_param_name in
                    get_valid_position_info_param_names()]

        # Loop through keys
        for key in keys:
            self.insert_single_epoch(**key, tolerate_no_entry=True)


@schema
class IntervalPositionInfoRelabel(ComputedBase):
    definition = """
    # Relabeled entries in IntervalPositionInfo
    -> IntervalPositionInfoRelabelParams
    """

    class RelabelEntries(dj.Part):
        definition = """
        -> IntervalPositionInfoRelabel
        -> IntervalPositionInfo
        ---
        analysis_file_name : varchar(100)
        head_position_object_id : varchar(40)
        head_orientation_object_id : varchar(40)
        head_velocity_object_id : varchar(40)
        """

        def fetch1_dataframe(self):
            return fetch1_dataframe_position_info(self)

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(self, (nd.common.AnalysisNwbfile, 'analysis_file_abs_path'),
                             *attrs, **kwargs)

    def make(self, key):
        relabel_params = (IntervalPositionInfoRelabelParams & key).fetch1()
        if len(IntervalPositionInfo & relabel_params) == 0:
            print(
                f"No entry in IntervalPositionInfo for {relabel_params}; cannot populate IntervalPositionInfoRelabel")
            return
        # Insert into main table
        insert1_print(self, key)
        print(f'Populated IntervalPositionInfoRelabel with key {key}')
        # Insert into parts table
        table_entry = (IntervalPositionInfo & relabel_params).fetch1()
        insert1_print(self.RelabelEntries, {**table_entry, **key})


# Rationale for choices of which primary keys to consolidate vs. preserve:
# Keep track_graph_name because intersect interval linearized position_and_maze information with track graph annotations using
# name of track graph
# Rationale for choices of which secondary keys to include in param name:
# Drop interval list name since redundant with epoch
@schema
class IntervalLinearizedPositionRelabelParams(SecKeyParamsBase):
    definition = """
    # Consolidate primary keys for entries in IntervalLinearizedPosition
    -> TaskIdentification
    -> TrackGraph
    position_info_param_name : varchar(50)
    interval_linearized_position_relabel_param_name : varchar(100)
    ---
    interval_list_name : varchar(50)
    linearization_param_name : varchar(100)
    """

    def insert_single_epoch(self, nwb_file_name, epoch, position_info_param_name=None, linearization_param_name=None,
                            tolerate_no_entry=True):
        interval_list_name = EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch, tolerate_no_entry)
        # If no epoch found and tolerating this, print message and exit
        if interval_list_name is None:
            print(f"Could not populate IntervalLinearizedPositionRelabelParams for {nwb_file_name}, epoch {epoch} "
                  f"because no relevant entry in EpochIntervalListName with which to get interval list name. Exiting")
            return
        # Otherwise, populate table
        key = {"nwb_file_name": nwb_file_name,
               "epoch": epoch,
               "interval_list_name": interval_list_name,
               "position_info_param_name": position_info_param_name,
               "linearization_param_name": linearization_param_name}
        # Get inputs if not passed
        key = add_param_defaults(key)
        # Get track graph name
        key.update({"track_graph_name": (IntervalLinearizedPosition & key).fetch1("track_graph_name")})
        # Add relabel param name using just linearization param name (since interval list name redundant with epoch)
        key.update({self.meta_param_name(): self._make_param_name({"linearization_param_name":
                                                                       key["linearization_param_name"]})})
        self.insert1(key, skip_duplicates=True)

    def insert_defaults(self, **kwargs):
        # Define inputs if not passed
        tolerate_no_entry = True  # default
        if "tolerate_no_entry" in kwargs:
            tolerate_no_entry = kwargs["tolerate_no_entry"]

        # Define keys. If key_filter passed, define a single key that has nwb_file_name and epoch (if present
        # in key_filter_. Otherwise define nwbf/epoch keys with possible settings of position_info_param_name
        key_filter = get_key_filter(kwargs)
        key_ = {k: v for k, v in key_filter.items() if k in ["nwb_file_name", "epoch"]}
        if len(key_) > 0:
            keys = [key_]
        else:
            keys = [{**k, **{"position_info_param_name": position_info_param_name}} for k in
                    get_jguidera_nwbf_epoch_keys() for position_info_param_name in
                    get_valid_position_info_param_names()]

        # Loop through keys and insert entries
        for key in keys:
            self.insert_single_epoch(**key, tolerate_no_entry=tolerate_no_entry)

    def cleanup(self, nwb_file_name):
        # Delete entries for which epoch and interval_list_name do not match (according to EpochIntervalListName).
        # Can occur if nwb file regenerated after code changed that assigns pos valid time interval list names.
        key = {"nwb_file_name": nwb_file_name}
        epochs, param_names = (self & key).fetch("epoch", "interval_linearized_position_relabel_param_name")
        for position_info_param_name in ["default", "default_decoding"]:
            for epoch in epochs:
                key = {"nwb_file_name": nwb_file_name,
                       "epoch": epoch,
                       "position_info_param_name": position_info_param_name}
                correct_interval_list_name = (EpochIntervalListName & key).fetch1("interval_list_name")
                entries = (self & key)
                interval_linearized_position_relabel_param_names = entries.fetch(
                    "interval_linearized_position_relabel_param_name")
                for param_name in interval_linearized_position_relabel_param_names:
                    if correct_interval_list_name not in param_name:
                        print(
                            f"entry interval list name: {param_name}. correct interval list name: "
                            f"{correct_interval_list_name}")
                        (self & {**key, **{"interval_linearized_position_relabel_param_name": param_name}}).delete()

    def cleanup_all(self):
        for nwb_file_name in get_jguidera_nwbf_names():
            self.cleanup(nwb_file_name)


@schema
class IntervalLinearizedPositionRelabel(ComputedBase):
    definition = """
    # Relabeled entries in IntervalLinearizedPosition
    -> IntervalLinearizedPositionRelabelParams
    """

    class RelabelEntries(dj.Part):
        definition = """
        -> IntervalLinearizedPositionRelabel
        -> IntervalLinearizedPosition
        ---
        analysis_file_name : varchar(100)
        linearized_position_object_id : varchar(100)
        """

        def fetch1_dataframe(self):
            return fetch1_dataframe(self, 'linearized_position').set_index('time')

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(self, (nd.common.AnalysisNwbfile, 'analysis_file_abs_path'),
                             *attrs, **kwargs)

    def make(self, key):
        relabel_params = (IntervalLinearizedPositionRelabelParams & key).fetch1()
        if len(IntervalLinearizedPosition & relabel_params) == 0:
            print(
                f"No entry in IntervalLinearizedPosition for {relabel_params}; "
                f"cannot populate IntervalLinearizedPositionRelabel")
            return
        # Insert into main table
        insert1_print(self, key)
        # Insert into parts table
        key.update((IntervalLinearizedPosition & relabel_params).fetch1())
        insert1_print(self.RelabelEntries, key)


@schema
class IntervalLinearizedPositionRescaled(ComputedBase):
    definition = """
    # Table with linearized position_and_maze rescaled to be in terms of universal track graph linearized position_and_maze
    -> IntervalLinearizedPositionRelabel
    -> AnnotatedTrackGraph
    -> TrackGraphUniversalTrackGraphMap
    ---
    position_info_param_name : varchar(50)
    interval_list_name : varchar(50)
    linearization_param_name : varchar(100)
    -> nd.common.AnalysisNwbfile
    interval_linearized_position_rescaled_object_id : varchar(40)
    """

    def make(self, key, verbose=False):

        # Get track graph parameters for specific and universal track
        track_graph_parameters = (AnnotatedTrackGraph & key).fetch1()
        universal_track_graph_name = (TrackGraphUniversalTrackGraphMap & key).fetch1("universal_track_graph_name")
        universal_track_graph_parameters = (AnnotatedUniversalTrackGraph & {"universal_track_graph_name":
                                                                            universal_track_graph_name}).fetch1()

        # First, check that edges based on IntervalLinearizedPositionRelabel and AnnotatedTrackGraph match at
        # valid positions. Get edge numbers according to AnnotatedTrackGraph by assigning linear positions in
        # IntervalLinearizedPositionRelabel to edges in AnnotatedTrackGraph.
        linear_position_df = (IntervalLinearizedPositionRelabel.RelabelEntries() & key).fetch1_dataframe()
        edge_linear_position = np.asarray(track_graph_parameters["edge_linear_position"])
        edge_idxs = linear_positions_to_edge_idxs(linear_position_df["linear_position"],
            edge_linear_position)  # get indices of edges to which positions belong
        valid_pos_bool = np.isfinite(linear_position_df["linear_position"])
        track_segment_ids = linear_position_df["track_segment_id"].to_numpy()  # unpack track segment IDs
        if not all(track_segment_ids[valid_pos_bool] == edge_idxs[valid_pos_bool]):
            raise Exception(f"Edges corresponding to linear position_and_maze identified using edge_linear_position "
                            f"AnnotatedTrackGraph do not match those using track_segment_id")

        # Also check that edges in specific track graph and universal track graph match
        if not all(np.ndarray.flatten(np.asarray(universal_track_graph_parameters["edge_names"]) == np.asarray(
                track_graph_parameters["edge_names"]))):
            raise Exception(f"Edge names in track graph do not match those in universal track graph")

        # Now find fractional distance along edge for each linear position_and_maze sample
        fractional_edge_position = get_fractional_edge_position(linear_position_df["linear_position"],
                                                                track_segment_ids,
                                                                edge_linear_position)

        # Convert fractional distance to actual distance on universal track graph ("linear position_and_maze rescaled")
        universal_edge_linear_position = np.asarray(universal_track_graph_parameters["edge_linear_position"])
        universal_pos_edge_lengths = np.ndarray.flatten(
            np.diff(universal_edge_linear_position)[track_segment_ids])
        universal_edge_startpoints = universal_edge_linear_position[:, 0][track_segment_ids]
        linear_position_rescaled = pd.Series(
            universal_edge_startpoints + universal_pos_edge_lengths * fractional_edge_position,
            index=linear_position_df.index)

        # Now identify edge and extent of edge traversal for each rescaled linear position_and_maze
        edge_names = np.asarray([None]*len(linear_position_rescaled))
        finite_bool = np.isfinite(linear_position_rescaled)
        edge_names[finite_bool] = np.asarray(universal_track_graph_parameters["edge_names"])[
            [np.where(np.prod(np.asarray(universal_track_graph_parameters["edge_linear_position"]) -
                              pos, axis=1) <= 0)[0][0]
             for pos in linear_position_rescaled[finite_bool]]]  # match positions to nondirectional edges

        # Now identify the direction and extent of directional edge traversal for each rescaled linear position_and_maze.
        # Approach: assess direction using interpolated difference between position_and_maze samples. At samples where
        # edge identity before or after is different (i.e. edge identity has just or will next switch), set
        # direction using nearest neighbor sample with same edge identity
        # Get linear position_and_maze rescaled, its difference, and its difference interpolated at original samples
        linear_position_rescaled_diff = pd.Series(
            np.diff(linear_position_rescaled.values), index=vector_midpoints(
                                     linear_position_rescaled.index))  # linear position_and_maze difference between samples
        # Find periods where same edge consecutively ("bouts") and in each find direction of travel along edge
        _, same_edge_bout_start_idxs = remove_repeat_elements(edge_names)  # bouts of same edge
        same_edge_bout_end_idxs = np.append(same_edge_bout_start_idxs[1:] - 1,
                                            len(edge_names) - 1)  # include last pos sample idx
        d_linear_position_rescaled = np.concatenate(
            [np.interp(x=linear_position_rescaled.index[idx_1:idx_2 + 1],
                       xp=linear_position_rescaled_diff.index[idx_1:idx_2],
                       fp=linear_position_rescaled_diff.values[idx_1:idx_2])
             if idx_1 != idx_2 else [np.nan] for idx_1, idx_2 in list(zip(
                same_edge_bout_start_idxs, same_edge_bout_end_idxs))])  # define direction
        # Define directional edge
        directional_edge_names = np.asarray([None] * len(linear_position_rescaled)).astype(
            "object")  # array for directional edges at positions
        directional_edge_names[d_linear_position_rescaled > 0] = edge_names[d_linear_position_rescaled > 0]
        directional_edge_names[d_linear_position_rescaled < 0] = [flip_nodes_edge_name(edge_name)
                                                                  for edge_name in
                                                                  edge_names[d_linear_position_rescaled < 0]]
        # Define percent directional edge traversed
        edge_linear_position_dict = {k: v for k, v in zip(
            universal_track_graph_parameters["directional_edge_names"],
            universal_track_graph_parameters["directional_edge_linear_position"])}
        percent_directional_edge_traversed = return_constant_vector(np.nan, len(linear_position_rescaled))
        valid_directional_edge_names = np.unique([edge_name for edge_name in directional_edge_names if
                                                  edge_name is not None])
        for directional_edge_name in valid_directional_edge_names:  # assess percent edge traversed for each edge type
            edge_name_bool = directional_edge_names == directional_edge_name  # where this edge type occurs
            linear_position_subset = linear_position_rescaled[edge_name_bool]  # positions on edge
            directional_edge_start_linear_position = edge_linear_position_dict[directional_edge_name][
                0]  # position_and_maze of edge start
            edge_length = abs(np.diff(edge_linear_position_dict[directional_edge_name]))  # edge length
            percent_directional_edge_traversed[edge_name_bool] = abs(
                linear_position_subset - directional_edge_start_linear_position) / edge_length

        if verbose:
            # For checking whether direction correctly assigned, plot rescaled linear position_and_maze,
            # its difference, and direction
            fig, ax = plt.subplots(figsize=(15, 2))
            ax.plot(linear_position_rescaled, '.-', color="orange", label="rescaled linear position_and_maze")
            for bool_temp, color_temp in zip([(linear_position_rescaled_diff <= 0),
                                              (linear_position_rescaled_diff > 0)], ["red", "blue"]):
                ax.scatter(linear_position_rescaled_diff.index[bool_temp], linear_position_rescaled_diff[bool_temp],
                           s=20, color=color_temp, alpha=.5, label="difference, linear position_and_maze rescaled")
            for bool_temp, color_temp in zip([(d_linear_position_rescaled <= 0),
                                              (d_linear_position_rescaled > 0)], ["pink", "lightblue"]):
                ax.scatter(d_linear_position_rescaled.index[bool_temp], d_linear_position_rescaled[bool_temp],
                           s=20, color=color_temp, alpha=.9, label="difference, INTERPOLATED linear position_and_maze rescaled")

                plot_edge_linear_position(edge_linear_position=universal_track_graph_parameters["edge_linear_position"],
                                          x_extent=[linear_position_rescaled.index[0],
                                                    linear_position_rescaled.index[-1]],
                                          ax=ax)
            ax.legend()

        # Assemble dataframe
        # First, must convert None in arrays with strings to "none", since hd5f does not handle mixed data types
        linear_position_rescaled_df = pd.DataFrame.from_dict(
            {"time": linear_position_rescaled.index,
             "linear_position_rescaled": linear_position_rescaled.values,
             "edge_name": none_to_string_none(edge_names),
             "d_linear_position_rescaled": d_linear_position_rescaled,
             "directional_edge_name": none_to_string_none(
                 directional_edge_names),
             "percent_directional_edge_traversed": percent_directional_edge_traversed})

        # Insert into table
        key.update({k: (IntervalLinearizedPositionRelabel.RelabelEntries & key).fetch1(k)
                    for k in ["interval_list_name", "linearization_param_name"]})  # additional attributes
        insert_analysis_table_entry(self, nwb_objects=[linear_position_rescaled_df], key=key,
                                    nwb_object_names=["interval_linearized_position_rescaled_object_id"])

    def fetch1_dataframe(self):
        return super().fetch1_dataframe().set_index("time")


def digitize_linear_position_rescaled_wrapper(nwb_file_name, epoch, new_index, bin_width_pos=4):
    edge_linear_position = (AnnotatedUniversalTrackGraph() &
                   {"universal_track_graph_name": get_universal_track_graph_name(nwb_file_name, epoch)}).fetch(
        "edge_linear_position")
    bin_edges = make_bin_edges(x=[np.min(np.min(edge_linear_position)),
                                  np.max(np.max(edge_linear_position))],
                               bin_width=bin_width_pos)
    lin_pos_original = (IntervalLinearizedPositionRescaled &
                        {"nwb_file_name": nwb_file_name,
                         "epoch": epoch,
                         "position_info_param_name": "default_decoding"}).fetch1_dataframe()["linear_position_rescaled"]
    return digitize_indexed_variable(indexed_variable=lin_pos_original,
                                     bin_edges=bin_edges,
                                     new_index=new_index)


def populate_jguidera_position(key=None, tolerate_error=False):
    from src.jguides_2024.position_and_maze.populate_position_tables import populate_position_tables_wrapper  # local import to avoid circular import error
    populate_position_tables_wrapper([key], tolerate_error=tolerate_error)  # populate lab position_and_maze tables
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_position():
    schema.drop()
