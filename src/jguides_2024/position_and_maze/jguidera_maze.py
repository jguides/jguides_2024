import copy
import itertools

import datajoint as dj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spyglass.common.common_position import TrackGraph

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SelBase, SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    insert1_print, populate_insert, \
    convert_path_names, fetch1_dataframe_from_table_entry, get_table_secondary_key_names, convert_path_name, \
    fetch_entries_as_dict
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_environments
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.position_and_maze.make_fork_maze_track_graph import make_fork_maze_track_graph, \
    simulate_fork_maze
from src.jguides_2024.task_event.jguidera_task_performance import ContingencyActiveContingenciesMap, \
    AlternationTaskWellIdentities, \
    AlternationTaskRule
from src.jguides_2024.utils.array_helpers import array_to_tuple_list
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.df_helpers import (df_filter_columns, df_filter1_columns, df_pop)
from src.jguides_2024.utils.dict_helpers import pairs_keys_same_value, pairs_keys_different_value
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements
from src.jguides_2024.utils.make_bins import make_int_bin_edges
from src.jguides_2024.utils.plot_helpers import plot_text_color
from src.jguides_2024.utils.set_helpers import check_set_equality, check_membership
from src.jguides_2024.utils.string_helpers import abbreviate_join_strings, get_even_odd_text
from src.jguides_2024.utils.tuple_helpers import add_reversed_pairs
from src.jguides_2024.utils.vector_helpers import check_range_within, unpack_single_element, unique_in_order

schema = dj.schema("jguidera_maze")  # define custom schema


@schema
class RewardWell(dj.Manual):
    definition = """
    # Table with names of physical milk reward wells
    universal_track_graph_name : varchar(40)
    ---
    well_names : blob
    """

    def insert_defaults(self, **kwargs):
        well_names = ["center_well", "handle_well", "right_well", "left_well"]
        self.insert1({"universal_track_graph_name": "fork_maze_universal",
                      "well_names": well_names}, skip_duplicates=True)


@schema
class RewardWellPath(ComputedBase):
    definition = """
    # Names of paths between physical milk reward wells
    -> RewardWell
    ---
    path_names : blob
    """

    def make(self, key):
        well_names = (RewardWell & key).fetch1("well_names")
        key.update(
            {"path_names": np.asarray(["_to_".join(x) for x in list(itertools.permutations(well_names, r=2))])})
        insert1_print(self, key)

    @staticmethod
    def _join_well_char():
        return "_to_"

    @classmethod
    def make_path_name(cls, start_well, end_well):
        return f"{start_well}{cls._join_well_char()}{end_well}"

    @classmethod
    def split_path_name(cls, path_name):
        return path_name.split(cls._join_well_char())

    @classmethod
    def get_end_well_name(cls, path_name):
        return cls.split_path_name(path_name)[1]

    def get_path_names(self, start_well_names, end_well_names, universal_track_graph_name=None):

        if universal_track_graph_name is None:
            universal_track_graph_name = "fork_maze_universal"

        # Get path names from a list of start wells and a corresponding list of end wells, and check that resulting
        # path names valid

        # Get path names
        path_names = [self.make_path_name(x, y) for x, y in list(zip(start_well_names, end_well_names))]

        # Check path names valid
        valid_path_names = (self & {"universal_track_graph_name": universal_track_graph_name}).fetch1("path_names")
        check_membership(path_names, valid_path_names)

        # Return path names
        return path_names

    def populate_(self, **kwargs):
        RewardWell().insert_defaults()
        return super().populate_(**kwargs)


@schema
class ForkMazeRewardWellPathPairSel(SelBase):
    definition = """
    # Selection from upstream tables for ForkMazeRewardWellPathPair
    -> RewardWellPath
    """

    def insert1(self, key, **kwargs):
        if key["universal_track_graph_name"] != "fork_maze_universal":
            raise Exception(f"universal_track_graph_name must be fork_maze_universal")
        super().insert1(key, **kwargs)

    def insert_defaults(self, **kwargs):
        RewardWellPath().populate_()
        self.insert1({"universal_track_graph_name": "fork_maze_universal"}, skip_duplicates=True)


@schema
class ForkMazeRewardWellPathPair(ComputedBase):
    definition = """
    # Pairs of names of paths for fork maze
    -> ForkMazeRewardWellPathPairSel
    -> RewardWellPath
    """

    class DistinctPathPair(dj.Part):
        definition = """
        # Pairs of names of paths for fork maze
        path_name_1 : varchar(40)
        path_name_2 : varchar(40)
        ---
        -> ForkMazeRewardWellPathPair
        """

    class PathPair(dj.Part):
        definition = """
        # Pairs of names of paths for fork maze
        path_name_1 : varchar(40)
        path_name_2 : varchar(40)
        ---
        -> ForkMazeRewardWellPathPair
        """

    def make(self, key):
        # Insert into main table
        insert1_print(self, key)

        # Insert into part tables
        path_names = (RewardWellPath & key).fetch1("path_names")
        path_name_pairs = list(itertools.combinations(path_names, r=2))
        for path_name_1, path_name_2 in path_name_pairs:
            insert1_print(self.DistinctPathPair, {**key, **{"path_name_1": path_name_1, "path_name_2": path_name_2}})
        for path_name_1, path_name_2 in path_name_pairs + [(path_name, path_name) for path_name in path_names]:
            insert1_print(self.PathPair, {**key, **{"path_name_1": path_name_1,  "path_name_2": path_name_2}})

    def populate_(self, **kwargs):
        RewardWellPath().populate_()
        return super().populate_(**kwargs)


class MazeElementColorBase(dj.Manual):

    @staticmethod
    def _maze_element_name():
        raise Exception(f"This method must be overwritten in child class")

    def get_color_map(self, universal_track_graph_name="fork_maze_universal"):
        names, colors = (
                self & {"universal_track_graph_name": universal_track_graph_name}).fetch1(
            f"{self._maze_element_name()}_names", f"{self._maze_element_name()}_colors")
        return {k: v for k, v in zip(names, colors)}

    def visualize_colors(self, universal_track_graph_name="fork_maze_universal", fig_ax_list=None, save_fig=False):
        table_entry = (self & {"universal_track_graph_name": universal_track_graph_name}).fetch1()
        plot_text_color(
            table_entry[f"{self._maze_element_name()}_names"], table_entry[f"{self._maze_element_name()}_colors"],
            fig_ax_list=fig_ax_list, file_name_save=f"{self._maze_element_name()}_colors", save_fig=save_fig)


@schema
class RewardWellColor(MazeElementColorBase):
    definition = """
    # Mapping between reward wells and colors
    -> RewardWell
    contingency : varchar(40)
    ---
    well_names : blob
    well_colors : blob
    """

    @staticmethod
    def _maze_element_name():
        return "well"

    def insert_defaults(self, **kwargs):
        color_order_map = {"centerAlternation": [10, 11, 0, 6], "handleAlternation": [10, 11, 8, 4]}
        for key in fetch_entries_as_dict(RewardWell):
            for contingency, color_order in color_order_map.items():
                well_colors = np.asarray(plt.cm.tab20.colors)[color_order]
                key.update({"contingency": contingency,
                            "well_colors": well_colors, })
                self.insert1(key, skip_duplicates=True)


@schema
class RewardWellPathColor(MazeElementColorBase):
    definition = """
    # Mapping between reward well paths and colors 
    -> RewardWellPath
    ---
    path_names : blob 
    path_colors : blob
    """

    @staticmethod
    def _maze_element_name():
        return "path"

    def insert_defaults(self, **kwargs):
        color_order = [10, 0, 6, 11, 8, 4,
                       7, 9, 17, 1, 5, 16]
        path_colors = np.asarray(plt.cm.tab20.colors)[color_order]
        for key in fetch_entries_as_dict(RewardWellPath):
            key.update({"path_colors": path_colors})
            self.insert1(key, skip_duplicates=True)

    def plot_colors(self, file_name_save="path_colors", save_fig=False):
        x = self.get_color_map()
        plot_text_color(list(x.keys()), list(x.values()), file_name_save=file_name_save, save_fig=save_fig)


@schema
class RewardWellPathTurnDirectionParams(SecKeyParamsBase):
    definition = """
    # Parameters for RewardWellPathTurnDirection
    universal_track_graph_name : varchar(40)
    ---
    reward_well_path_turn_direction_map : blob
    """

    def insert_defaults(self, **kwargs):
        universal_track_graph_name = "fork_maze_universal"
        reward_well_path_turn_direction_map = {'center_well_to_handle_well': "straight",
                                               'center_well_to_right_well': "left_left",
                                               'center_well_to_left_well': "right_right",
                                               'handle_well_to_center_well': "straight",
                                               'handle_well_to_right_well': "right_left",
                                               'handle_well_to_left_well': "left_right",
                                               'right_well_to_center_well': "right_right",
                                               'right_well_to_handle_well': "right_left",
                                               'right_well_to_left_well': "right_straight_right",
                                               'left_well_to_center_well': "left_left",
                                               'left_well_to_handle_well': "left_right",
                                               'left_well_to_right_well': "left_straight_left"}
        self.insert1({"universal_track_graph_name": universal_track_graph_name,
                      "reward_well_path_turn_direction_map": reward_well_path_turn_direction_map},
                     skip_duplicates=True)


@schema
class RewardWellPathTurnDirection(ComputedBase):
    definition = """
    # Left/right identity of turns along paths connecting reward well 
    -> RewardWellPath
    -> RewardWellPathTurnDirectionParams
    ---
    reward_well_path_turn_direction_map : blob
    """

    def make(self, key):
        # Check that path names in RewardWellPathTurnDirectionParams same as in RewardWellPath
        path_names_1 = (RewardWellPath & key).fetch1("path_names")
        reward_well_path_turn_direction_map = (RewardWellPathTurnDirectionParams & key).fetch1("reward_well_path_turn_direction_map")
        path_names_2 = reward_well_path_turn_direction_map.keys()
        check_set_equality(path_names_1, path_names_2, "RewardWellPath path names",
                           "RewardWellPathTurnDirectionParams path names")
        insert1_print(self, {**key, **{"reward_well_path_turn_direction_map": reward_well_path_turn_direction_map}})

    def populate_(self, **kwargs):
        RewardWellPath().populate_(**kwargs)
        return super().populate_(**kwargs)


@schema
class MazeEdgeType(dj.Manual):
    definition = """
    # Edge type for maze edges 
    universal_track_graph_name : varchar(40)
    ---
    edge_names : blob
    edge_types : blob
    """

    def insert_defaults(self, **kwargs):
        universal_track_graph_name = "fork_maze_universal"
        edge_names = (AnnotatedUniversalTrackGraph() & {"universal_track_graph_name":
                                    universal_track_graph_name}).fetch1("edge_names")
        edge_types = ["prong" if "well" in edge_name else "connector" for edge_name in edge_names]
        self.insert1({"universal_track_graph_name": universal_track_graph_name,
                             "edge_names": edge_names,
                             "edge_types": edge_types}, skip_duplicates=True)

    def fetch1_dict(self, universal_track_graph_name):
        return {edge_name: edge_type for edge_name, edge_type in
                zip(*(self & {"universal_track_graph_name": universal_track_graph_name}).fetch1("edge_names", "edge_types"))}

    def return_names_edge_type(self, edge_type, universal_track_graph_name="fork_maze_universal"):
        """
        Return names of edges of a certain type
        :param edge_type: type of edge
        :param universal_track_graph_name: name of universal track graph
        :return: list with names of edges of edge_type
        """
        maze_edge_type_map = (self & {"universal_track_graph_name": universal_track_graph_name}).fetch1()
        return np.asarray(maze_edge_type_map["edge_names"])[np.asarray(maze_edge_type_map["edge_types"]) == edge_type]  # names of edges of type edge_type


@schema
class TrackGraphAnnotations(dj.Manual):
    definition = """
    track_graph_name : varchar(40)
    ---
    node_names : blob
    """


@schema
class UniversalTrackGraphAnnotations(dj.Manual):
    definition = """
    universal_track_graph_name : varchar(40)
    ---
    node_names : blob
    """


def _plot_track_graph(track_graph_parameters, ax):
    from src.jguides_2024.position_and_maze.make_fork_maze_track_graph import make_track_graph
    track_graph = make_track_graph(node_names=track_graph_parameters["node_names"],
                                   node_positions=track_graph_parameters["node_positions"],
                                   edge_list=track_graph_parameters["edges"])
    if ax is None:  # initialize figure if not passed
        fig, ax = plt.subplots()
    from development.plot_maze import plot_nx_track_graph
    plot_nx_track_graph(track_graph=track_graph,
                        ax=ax)


def _get_track_graph_annotations(key, universal_track_graph):

    if universal_track_graph:
        annotations_table = UniversalTrackGraphAnnotations
        track_graph_key = {"track_graph_name": key["universal_track_graph_name"]}
    else:
        annotations_table = TrackGraphAnnotations
        track_graph_key = {"track_graph_name": key["track_graph_name"]}

    node_names = (annotations_table & key).fetch1("node_names")  # get node names

    track_graph_parameters = (TrackGraph & track_graph_key).fetch1()  # get track graph information from TrackGraph

    # Get edge names and linear position_and_maze
    edge_names = [f"{node_names[idx1]}_to_{node_names[idx2]}"
                  for idx1, idx2 in track_graph_parameters["edges"]]  # name edges
    from track_linearization import make_track_graph
    track_graph = make_track_graph(node_positions=track_graph_parameters['node_positions'],
                                   edges=track_graph_parameters['edges'])
    start_node_linear_position = 0
    edge_linear_position = []  # initialize list for linear position_and_maze of segments
    for edge in np.asarray(track_graph.edges):  # for each edge
        end_node_linear_position = start_node_linear_position + track_graph.edges[edge]["distance"]
        edge_linear_position.append((start_node_linear_position, end_node_linear_position))
        start_node_linear_position += (
                track_graph.edges[edge]["distance"] +
                track_graph_parameters["linear_edge_spacing"])  # udpate position_and_maze of start node

    # Get directional edge names and linear position_and_maze
    directional_edge_names = edge_names + [flip_nodes_edge_name(edge_name) for edge_name in edge_names]
    directional_edge_linear_position = edge_linear_position + [(x2, x1) for x1, x2 in edge_linear_position]

    return {**{k: track_graph_parameters[k] for k in ["environment",
                                                      "node_positions",
                                                      "edges",
                                                      "linear_edge_order",
                                                      "linear_edge_spacing"]},
            **{"node_names": node_names,
               "edge_names": edge_names,
               "edge_linear_position": edge_linear_position,
               "directional_edge_names": directional_edge_names,
               "directional_edge_linear_position": directional_edge_linear_position}}


@schema
class AnnotatedTrackGraph(ComputedBase):
    definition = """
    # Information in TrackGraph plus additional information about nodes and edges
    -> TrackGraph
    -> TrackGraphAnnotations
    ---
    environment : varchar(40)
    node_positions : blob
    edges : blob
    linear_edge_order : blob
    linear_edge_spacing : blob
    node_names : blob
    edge_names : blob
    edge_linear_position : blob
    directional_edge_names : blob 
    directional_edge_linear_position : blob
    """

    def make(self, key):
        key.update(_get_track_graph_annotations(universal_track_graph=False, key=key))
        insert1_print(self, key)

    def plot_track_graph(self, ax=None):
        _plot_track_graph(track_graph_parameters=self.fetch1(), ax=ax)


@schema
class AnnotatedUniversalTrackGraph(ComputedBase):
    definition = """
    # Information in TrackGraph plus additional information about nodes and edges
    -> TrackGraph.proj(universal_track_graph_name="track_graph_name")
    -> UniversalTrackGraphAnnotations
    ---
    environment : varchar(40)
    node_positions : blob
    edges : blob
    linear_edge_order : blob
    linear_edge_spacing : blob
    node_names : blob
    edge_names : blob
    edge_linear_position : blob
    directional_edge_names : blob 
    directional_edge_linear_position : blob
    """

    def make(self, key):
        key.update(_get_track_graph_annotations(universal_track_graph=True, key=key))
        insert1_print(self, key)

    def plot_track_graph(self, ax=None):
        _plot_track_graph(track_graph_parameters=self.fetch1(), ax=ax)

    def return_edge_names_linear_position_map(self, universal_track_graph_name="fork_maze_universal"):
        edge_names, edge_linear_position = (self &
                                            {"universal_track_graph_name": universal_track_graph_name}).fetch1(
            "edge_names", "edge_linear_position")
        return {k: v for k, v in zip(edge_names, edge_linear_position)}


@schema
class TrackGraphUniversalTrackGraphMapParams(SecKeyParamsBase):
    definition = """
    # Table with theoretical universal track graph that a specific track graph corresponds to
    track_graph_name : varchar(40)
    ---
    universal_track_graph_name : varchar(40)
    """

    def insert_defaults(self, **kwargs):
        for track_graph_name in ["fork_maze_HaightLeft", "fork_maze_HaightRight", "fork_maze_SA"]:
            self.insert1({"track_graph_name": track_graph_name, "universal_track_graph_name": "fork_maze_universal"},
                         skip_duplicates=True)


@schema
class UniversalTrackGraphPosBinEdgesParams(SecKeyParamsBase):
    definition = """
    # Parameters for bin edges for track graph
    universal_track_graph_pos_bin_edges_param_name : varchar(40)
    ---
    position_bin_width : decimal(10,5) unsigned
    """

    def _default_params(self):
        return [[4]]


@schema
class UniversalTrackGraphPosBinEdges(ComputedBase):
    definition = """
    # Bin edges for track graph 
    -> UniversalTrackGraphPosBinEdgesParams
    -> AnnotatedUniversalTrackGraph
    ---
    edges_position_bin_edges : blob
    edge_names : blob
    """

    def make(self, key):
        edge_linear_position_list = (AnnotatedUniversalTrackGraph & key).fetch1(
            "edge_linear_position")  # get edge linear position_and_maze
        bin_width = float((UniversalTrackGraphPosBinEdgesParams & key).fetch1("position_bin_width"))
        edges_position_bin_edges = [make_int_bin_edges(edge_linear_position, bin_width=bin_width)
                                    for edge_linear_position in edge_linear_position_list]  # make bins
        key = {**key, **{"edges_position_bin_edges": edges_position_bin_edges,
                         "edge_names": (AnnotatedUniversalTrackGraph & key).fetch1("edge_names")}}  # update key
        insert1_print(self, key)


@schema
class ForkMazePathEdgesSel(SelBase):
    definition = """
    # Universal track graph name for fork maze (use to restrict ForkMazePathEdges to fork maze)
    -> RewardWellPath
    """

    def insert_defaults(self, **kwargs):
        RewardWellPath().populate_()
        self.insert1({"universal_track_graph_name": "fork_maze_universal"}, skip_duplicates=True)


@schema
class ForkMazePathEdges(ComputedBase):
    definition = """
    # Sequence of nondirectional edges for a fork maze path
    -> ForkMazePathEdgesSel
    -> AnnotatedUniversalTrackGraph
    ---
    path_names: blob
    path_edge_names: blob  # using set of path names where only single name per path
    path_directional_edge_names: blob  # nodes oriented correctly along path
    """

    def make(self, key):
        path_edge_names_list = []
        path_names = convert_path_names(
            (RewardWellPath & key).fetch1("path_names"))  # as tuple so can extract endpoints
        # Get names of edges along paths (with regard to direction of edge)
        for path_name in path_names:
            # Get nodes along path. Add center_maze manually because always must be in path, but will not be detected
            # for left to right or right to left paths.
            edge_names = [convert_path_name(edge_name) for edge_name in
                          (AnnotatedUniversalTrackGraph & key).fetch1("edge_names")]  # all edge names as tuples
            path_nodes = np.unique(list(path_name)
                                   + list(itertools.chain(*[list(edge_name) for edge_name in edge_names
                                                            if any([w in edge_name for w in path_name])]))
                                   + ["center_maze"])
            # Get edges along path
            path_edges = [segment for segment in edge_names if
                          len([w for w in segment if w in path_nodes]) == 2]
            # Order the path edges
            ordered_path_edges = []  # initialize list for path edges in order of traversal
            starting_node = path_name[0]  # initialize starting node
            for _ in path_edges:  # for each path segment
                ordered_path_edges += [path_edge for path_edge in path_edges if
                                       starting_node in path_edge and path_edge not in ordered_path_edges]
                starting_node = np.asarray(ordered_path_edges)[-1][
                    np.asarray(ordered_path_edges[-1]) != starting_node]  # update starting node
            path_edge_names_list.append(ordered_path_edges)  # convert to string
        # Get names of edges along paths, where edge is oriented correctly
        path_directional_edge_names_list = return_directional_edges(path_names, path_edge_names_list)

        # Insert into table
        self.insert1({**key, **{"path_names": convert_path_names(path_names),  # convert to string
                                "path_edge_names": [convert_path_names(edge_names)
                                                    for edge_names in path_edge_names_list],
                                "path_directional_edge_names": [convert_path_names(edge_names)
                                                                for edge_names in path_directional_edge_names_list]}})

    def fetch1_dataframe(self):
        return fetch1_dataframe_from_table_entry(self, column_subset=get_table_secondary_key_names(self))

    def return_path_edge_names_map(self, universal_track_graph_name):
        path_names, path_edge_names = (self & {"universal_track_graph_name":
                                                   universal_track_graph_name}).fetch1("path_names", "path_edge_names")
        return {path_name: edge_name for path_name, edge_name in zip(path_names, path_edge_names)}

    def return_path_directional_edge_names_map(self, universal_track_graph_name="fork_maze_universal"):
        path_names, path_directional_edge_names = (self & {"universal_track_graph_name":
                                                   universal_track_graph_name}).fetch1("path_names",
                                                                                       "path_directional_edge_names")
        return {path_name: edge_name for path_name, edge_name in zip(path_names, path_directional_edge_names)}


@schema
class UniversalForkMazePathEdgePathFractionMap(ComputedBase):
    definition = """
    # Map from fork maze path and edge to fraction of entire path
    -> AnnotatedUniversalTrackGraph
    -> MazeEdgeType
    -> ForkMazePathEdges
    ---
    path_names : blob
    path_edge_names : blob
    path_directional_edge_names : blob
    path_fractions : blob
    """

    def make(self, key):
        # Hard code maximum allowed difference between edge lengths for edges of same type
        max_tolerated_edge_length_difference = .0001

        # Get edge information for universal fork maze
        edge_names, edge_linear_position = (AnnotatedUniversalTrackGraph & key).fetch1("edge_names",
                                                                                       "edge_linear_position")
        edge_lengths = np.ndarray.flatten(np.diff(edge_linear_position))  # lengths of edges
        edge_name_edge_type_map = MazeEdgeType().fetch1_dict(
            key["universal_track_graph_name"])  # map between edges and their "type" (e.g. prong, connector)
        edge_types = np.asarray([edge_name_edge_type_map[edge_name] for edge_name in
                                 edge_names])  # types corresponding to universal fork maze edges

        # To align paths that may have slightly different length edges, define a common average length for edge types,
        # and use to map path & edge combination to fraction of path spanned by edge
        edge_type_length_dict = dict()  # map between edge type and length
        for edge_type in np.unique(edge_types):  # for each type of edge
            edge_type_lengths = edge_lengths[edge_types == edge_type]  # lengths of edges of this type
            check_range_within(edge_type_lengths,
                               max_tolerated_edge_length_difference)  # make sure edges of this type similar in length
            edge_type_length_dict[edge_type] = np.mean(edge_type_lengths)

        # Create map from path & edge name to fraction of path spanned by edge
        path_names_edge_names_map = (ForkMazePathEdges & key).fetch1_dataframe()  # maps paths to composing edges
        path_fraction_list = []
        path_name_edge_names_tuple = list(zip(path_names_edge_names_map["path_names"],
                                         path_names_edge_names_map["path_edge_names"]))
        for path_name, edge_names in path_name_edge_names_tuple:
            path_edges_total_length = np.sum([edge_type_length_dict[edge_name_edge_type_map[edge_name]] for
                                              edge_name in edge_names])  # get total length of edges in path
            cumulative_length = 0  # count cumulative edge length along path
            for edge_name in edge_names:  # for edge in path
                edge_length = edge_type_length_dict[edge_name_edge_type_map[edge_name]]
                path_fraction_list.append(tuple(np.asarray(
                    [cumulative_length, cumulative_length + edge_length])/path_edges_total_length))
                cumulative_length += edge_length  # update cumulative length along path

        path_names_list = np.concatenate([[path_name] * len(edge_names)
                                          for path_name, edge_names in path_name_edge_names_tuple])  # convert path name back to string
        path_edge_names_list = np.concatenate([edge_names for edge_names in path_names_edge_names_map["path_edge_names"]])
        path_directional_edge_names_list = np.concatenate([edge_names for edge_names
                                                           in path_names_edge_names_map["path_directional_edge_names"]])

        insert1_print(self, {**key, **{"path_names": path_names_list,
                                       "path_edge_names": path_edge_names_list,
                                       "path_directional_edge_names": path_directional_edge_names_list,
                                       "path_fractions": path_fraction_list}})

    def fetch1_dataframe(self):
        return fetch1_dataframe_from_table_entry(self, column_subset=get_table_secondary_key_names(self))

    def return_directional_path_fractions(self, path_name):
        directional_edge_names = ForkMazePathEdges().return_path_directional_edge_names_map()[path_name]
        path_fraction_map = self.fetch1_dataframe()  # map from path and edge to fraction of track traversed
        return np.asarray([df_filter1_columns(path_fraction_map, {"path_names": path_name,
                                                                            "path_directional_edge_names": edge_name})[
                                         "path_fractions"].values[0] for edge_name in directional_edge_names])


@schema
class TrackGraphUniversalTrackGraphMap(ComputedBase):
    definition = """
    # Table with theoretical universal track graph that a specific track graph corresponds to
    -> TrackGraphUniversalTrackGraphMapParams
    ---
    universal_track_graph_name : varchar(40)
    """

    class Map(dj.Part):
        definition = """
        # Achieves dependency on AnnotatedUniversalTrackGraph and derivatives
        -> TrackGraphUniversalTrackGraphMap
        -> AnnotatedUniversalTrackGraph
        -> ForkMazePathEdges
        -> UniversalForkMazePathEdgePathFractionMap
        """

    def make(self, key):
        universal_track_graph_name = (TrackGraphUniversalTrackGraphMapParams & key).fetch1("universal_track_graph_name")
        key.update({"universal_track_graph_name": universal_track_graph_name})

        # Insert into main table
        self.insert1(key)

        # Insert into part table
        insert1_print(self.Map, key)

    def populate_(self, **kwargs):
        # Populate upstream tables
        AnnotatedUniversalTrackGraph().populate_(**kwargs)
        return super().populate_(**kwargs)


def flip_nodes_edge_name(edge_name, separating_character="_to_"):
    node_1, node_2 = edge_name.split(separating_character)
    return f"{node_2}{separating_character}{node_1}"


def return_directional_edges(path_names_,
                             path_edge_names_list_):

    # Check arguments well defined
    if not all(np.asarray([len(k) for j in path_edge_names_list_ for k in j]) == 2):
        raise Exception(f"Path edge names list must be a list of path edges, each of which is a length two array-like")
    if not np.shape(np.asarray(path_names_))[1] == 2:
        raise Exception(f"Path names must be a 2D array-like, where second dimension (nodes) are in columns")

    # Convert to array
    path_names_ = np.asarray(path_names_)
    path_edge_names_list_ = [np.asarray(path_edge_names) for path_edge_names in path_edge_names_list_]

    # Find directional edges for paths
    path_directional_edge_names_list_ = []
    for path_name, path_edge_names in zip(path_names_, path_edge_names_list_):
        start_node = path_name[0]
        path_directional_edge_names = []
        for edge_name in path_edge_names:
            end_node = unpack_single_element(edge_name[edge_name != start_node])  # check that one end node then unpack
            path_directional_edge_names.append((start_node, end_node))
            start_node = copy.deepcopy(end_node)  # next edge starts at end of current edge
        path_directional_edge_names_list_.append(path_directional_edge_names)

    # Check that directional edge list has expected structure
    for edge_names in path_directional_edge_names_list_:
        arr = np.asarray(edge_names)
        if not all(arr[:-1, 1] == arr[1:, 0]):
            raise Exception(f"Edge names non-contiguous: {edge_names}")

    return path_directional_edge_names_list_


def get_universal_track_graph_name(nwb_file_name, epoch):
    environment = (TaskIdentification & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1("task_environment")
    track_graph_name = (TrackGraph() & {"environment": environment}).fetch1("track_graph_name")
    return (TrackGraphUniversalTrackGraphMap &
            {"track_graph_name": track_graph_name}).fetch1("universal_track_graph_name")


def get_average_length_edge_type(edge_type,
                                 universal_track_graph_name="fork_maze_universal",
                                 average_fn=np.mean):
    edge_names_linear_position_map = AnnotatedUniversalTrackGraph().return_edge_names_linear_position_map(universal_track_graph_name)
    edge_names = MazeEdgeType().return_names_edge_type(edge_type,
                                                       universal_track_graph_name)
    return average_fn([np.diff(edge_names_linear_position_map[edge_name])[0] for edge_name in edge_names])


def get_path_turn_fraction_df(universal_track_graph_name="fork_maze_universal"):
    key = {"universal_track_graph_name": universal_track_graph_name}
    path_turns_map = (RewardWellPathTurnDirection & key).fetch1("reward_well_path_turn_direction_map")
    path_names = list(path_turns_map.keys())
    path_turns = [path_turns_map[path_name].split("_")
                                 for path_name in path_names]  # turns in each path
    num_junctions = list(map(len, path_turns))  # number of junctions encountered in each path
    turn_1 = [x[0] for x in path_turns]
    turn_2 = [x[1] if len(x) > 1 else "none" for x in path_turns]
    turn_3 = [x[2] if len(x) > 2 else "none" for x in path_turns]
    _path_names, _path_directional_edge_names, _path_fractions = (
            UniversalForkMazePathEdgePathFractionMap & key).fetch1("path_names", "path_directional_edge_names",
                                                                   "path_fractions")
    path_names_directional_edges_map = ForkMazePathEdges().return_path_directional_edge_names_map(
        universal_track_graph_name)
    path_fractions = [[_path_fractions[unpack_single_element(
        np.where(np.logical_and(_path_names == path_name,
                                _path_directional_edge_names == edge_name))[0])]
                       for edge_name in path_names_directional_edges_map[path_name]]
                      for path_name in path_names]
    return pd.DataFrame.from_dict({"path_name": path_names,
                                   "turn_1": turn_1,
                                   "turn_2": turn_2,
                                   "turn_3": turn_3,
                                   "path_fractions": path_fractions,
                                   "number_junctions": num_junctions})


def get_path_junction_abbreviation(path_name):
    df_subset = df_filter1_columns(get_path_turn_fraction_df(), {"path_name": path_name}).iloc[0]
    junction_turn_directions = df_subset[["turn_1", "turn_2", "turn_3"]].to_numpy()
    junction_turn_directions = junction_turn_directions[junction_turn_directions != "none"]
    return abbreviate_join_strings(junction_turn_directions)


def get_path_junction_fractions(path_name):
    junction_fractions = df_pop(
        get_path_turn_fraction_df(), {"path_name": path_name}, "path_fractions")[1:-1]
    if len(junction_fractions) == 0:  # path with no junctions
        return []
    return np.unique(np.concatenate(junction_fractions))  # middle entries have middle segments coordinates


def get_n_junction_path_junction_fractions(n_junctions):
    return unpack_single_element(np.unique([get_path_junction_fractions(x)
                                            for x in return_n_junction_path_names(n_junctions)], axis=0))


def get_path_segment_fractions(n_junctions):
    return zip_adjacent_elements([0] + list(get_n_junction_path_junction_fractions(n_junctions)) + [1])


def return_universal_fork_maze_turn_zones(turn_direction, pre_turn_margin, post_turn_margin):
    # TODO: write in check that pre and post turn margins are small enough that no overlapping pre/post turn segments
    universal_track_graph_name = "fork_maze_universal"
    valid_turn_columns = ["turn_1", "turn_2", "turn_3"]
    # Get map between paths, turns, and fraction path traversed
    path_turn_fraction_df = get_path_turn_fraction_df(universal_track_graph_name)
    turn_columns = [turn_x for turn_x in valid_turn_columns if turn_x in path_turn_fraction_df.columns]
    valid_path_names = df_filter_columns(path_turn_fraction_df, {turn_column: turn_direction
                                    for turn_column in turn_columns}, column_and=False)["path_name"]
    turn_zones_dict = {path_name: [] for path_name in valid_path_names}
    turn_centers_dict = {path_name: [] for path_name in valid_path_names}
    for turn_column in turn_columns:
        df_subset = df_filter_columns(path_turn_fraction_df, {turn_column: turn_direction})  # df subset with only paths for which nth turn (given by turn_column) has turn same as turn_direction
        turn_num = int(turn_column.split("turn_")[1])
        turn_zone_centers = np.asarray([path_edges_ppt[turn_num - 1][1]  # path_edges_ppt[edge_ind][node_ind]
                                       for path_edges_ppt in df_subset["path_fractions"]])  # path fraction at which nth turn which is same as turn_direction occurs
        turn_zones = list(zip(turn_zone_centers - pre_turn_margin,
                              turn_zone_centers + post_turn_margin))  # expand region around turn centers to define turn zone
        for path_name, turn_zone, turn_zone_center in zip(df_subset["path_name"],
                                                          turn_zones,
                                                          turn_zone_centers):
            turn_zones_dict[path_name].append(turn_zone)
            turn_centers_dict[path_name].append(turn_zone_center)
    return pd.DataFrame.from_dict({"path_name": valid_path_names,
                                   "turn_zones": [turn_zones_dict[path_name] for path_name in valid_path_names],
                                   "turn_zone_centers": [turn_centers_dict[path_name]
                                                         for path_name in valid_path_names]})


def return_n_junction_path_names(n_junctions, universal_track_graph_name="fork_maze_universal"):
    return np.asarray(df_filter_columns(get_path_turn_fraction_df(universal_track_graph_name),
                                        {"number_junctions": n_junctions})["path_name"])


def get_nonoverlapping_ppt(path_name_1, path_name_2):
    # ...Get edges along train and test paths
    path_1_edges = np.asarray(
        ForkMazePathEdges().return_path_directional_edge_names_map()[path_name_1])
    path_2_edges = np.asarray(
        ForkMazePathEdges().return_path_directional_edge_names_map()[path_name_2])
    # ...Find valid ppt values: where path 1 has different edges from path 2
    # (use zip to account for fact that can have different number of edges across paths)
    valid_edge_bool = [path_1_edge != path_2_edge
                      for path_1_edge, path_2_edge
                      in zip(path_1_edges, path_2_edges)]

    # Now identify ppt interval for each valid edge, for one of the paths
    # We initialize a valid bool with "True" that is length of number of segments in one of the paths,
    # to account for fact that compared paths may have unequal length. Any "extra" edges in one of the paths
    # is by definition not shared with the other, and so valid
    path_fractions = np.asarray(
        get_path_turn_fraction_df().set_index("path_name").loc[path_name_1].path_fractions)
    valid_bool = np.asarray([True] * len(path_fractions))
    valid_bool[np.arange(0, len(valid_bool))] = valid_edge_bool
    return path_fractions[valid_bool, :]


@schema
class EnvironmentColor(dj.Manual):
    definition = """
    # Map from environment to color 
    environment : varchar(100)
    ---
    color : blob
    """

    def insert_defaults(self, **kwargs):
        environment_color_map = {"HaightLeft": "lightblue", "HaightRight": "purple", "SA": "orange"}
        for environment, color in environment_color_map.items():
            self.insert1({"environment": environment, "color": color}, skip_duplicates=True)

    def get_environment_color_map(self):
        return {k:v for k,v in zip(self.fetch("environment"),
                                   self.fetch("color"))}


def get_fork_maze_track_graph_name(environment):
    return f"fork_maze_{environment}"


def get_valid_fork_maze_track_graph_names():
    return [get_fork_maze_track_graph_name(x) for x in get_environments()]


def get_default_environment_track_graph_name_map():
    return {x: get_fork_maze_track_graph_name(x) for x in get_environments()}


def populate_track_graph_tables_fork_maze():
    """
    Populate TrackGraph, TrackGraphAnnotations, UniversalTrackGraphAnnotations, AnnotatedTrackGraph,
    AnnotatedUniversalTrackGraph for fork mazes track graphs
    """

    def _get_track_graph_node_names(track_graph):
        return [track_graph.nodes[node]["node_name"] for node in track_graph.nodes]

    def _get_track_graph_node_positions(track_graph):
        return [track_graph.nodes[node]["pos"] for node in track_graph.nodes]

    def _switch_left_right(x):
        new_list = []
        for s in x:
            if "right" in s and "left" in s:
                raise Exception(f"Both left and right found in string; code not built to account for this")
            if "right" in s:
                new_list.append(s.replace("right", "left"))
            elif "left" in s:
                new_list.append(s.replace("left", "right"))
            else:
                new_list.append(s)
        return new_list

    # *** Parameters to apply across mazes ***
    linear_edge_spacing = 15
    # ****************************************

    def _populate_tables(environment, node_positions_dict, linear_edge_spacing):
        track_graph_name = get_fork_maze_track_graph_name(environment)
        track_graph = make_fork_maze_track_graph(node_positions_dict)
        if environment == "universal":
            UniversalTrackGraphAnnotations.insert1({"universal_track_graph_name": track_graph_name,
                                                    "node_names": _get_track_graph_node_names(track_graph)},
                                                   skip_duplicates=True)
        else:
            TrackGraphAnnotations.insert1({"track_graph_name": track_graph_name,
                                           "node_names": _get_track_graph_node_names(track_graph)},
                                          skip_duplicates=True)
        TrackGraph.insert1({'track_graph_name': track_graph_name,
                            'environment': environment,
                            'node_positions': _get_track_graph_node_positions(track_graph),
                            'edges': list(track_graph.edges),
                            'linear_edge_order': list(track_graph.edges),
                            'linear_edge_spacing': linear_edge_spacing}, skip_duplicates=True)

    # IMPORTANT NOTE ABOUT FORK MAZES: POSITION DATA IS REFLECTED, I.E. REFLECT A BOTTOM UP
    # VIEW OF TRACK. SO, MUST CODE MAZES ALSO USING A BOTTOM UP VIEW.

    # "IDEAL" MAZE, TO BE USED TO ALIGN REAL MAZES ACROSS ENVIRONMENTS
    environment = "universal"
    track_graph = simulate_fork_maze(arm_length=76.84,
                                     connector_length=44)  # values are from schematic of real fork maze
    node_names = _get_track_graph_node_names(track_graph)
    node_names = _switch_left_right(node_names)  # change left and right so that have bottom up view of track
    node_positions = _get_track_graph_node_positions(track_graph)
    node_positions_dict = {node_name: {} for node_name in node_names}
    for node_name, (x, y) in zip(node_names, node_positions):
        node_positions_dict[node_name]["x"] = x
        node_positions_dict[node_name]["y"] = y
    _populate_tables(environment,
                     node_positions_dict,
                     linear_edge_spacing)

    # LEFT HAIGHT
    environment = "HaightLeft"
    node_positions_dict = {"center_well": {"x": 69,
                                           "y": 36},
                           "center_maze": {"x": 66,
                                           "y": 110},
                           "left_corner": {"x": 31,
                                            "y": 110},
                           "left_well": {"x": 34,
                                          "y": 36},
                           "right_corner": {"x": 103,
                                           "y": 110},
                           "right_well": {"x": 104,
                                         "y": 34},
                           "handle_well": {"x": 65,
                                           "y": 192}}
    _populate_tables(environment, node_positions_dict, linear_edge_spacing)

    # RIGHT HAIGHT
    environment = "HaightRight"
    node_positions_dict = {"center_well": {"x": 228,
                                           "y": 195},
                           "center_maze": {"x": 224,
                                           "y": 112},
                           "left_corner": {"x": 267,
                                            "y": 110},
                           "left_well": {"x": 270,
                                          "y": 196},
                           "right_corner": {"x": 184,
                                           "y": 112},
                           "right_well": {"x": 188,
                                         "y": 194},
                           "handle_well": {"x": 222,
                                           "y": 35}}
    _populate_tables(environment, node_positions_dict, linear_edge_spacing)

    # RIGHT HAIGHT
    environment = "SA"
    node_positions_dict = {"center_well": {"x": 204,
                                           "y": 78},
                           "center_maze": {"x": 210,
                                           "y": 158},
                           "left_corner": {"x": 173,
                                           "y": 160},
                           "left_well": {"x": 166,
                                         "y": 78},
                           "right_corner": {"x": 243,
                                            "y": 156},
                           "right_well": {"x": 242,
                                          "y": 78},
                           "handle_well": {"x": 212,
                                           "y": 240}}
    _populate_tables(environment, node_positions_dict, linear_edge_spacing)

    # Bring annotations together with track graph information into one table
    AnnotatedTrackGraph().populate()
    AnnotatedUniversalTrackGraph().populate()


class MazePathWell:
    """
    Group of functions that depend on overlapping sets of tables and return information about maze path and well
    """

    @staticmethod
    def get_rewarded_path_name_tuples(nwb_file_name=None, epoch=None, contingency=None):

        # Check inputs (must pass either nwb_file_name and epoch, OR contingency
        if (nwb_file_name is not None and epoch is not None) + (contingency is not None) != 1:
            raise Exception(f"must pass either nwb_file_name and epoch, OR contingency")

        # Get contingency if not passed
        if contingency is None:
            contingency = TaskIdentification().get_contingency(nwb_file_name, epoch)

        previous_well, current_well, performance_outcomes = AlternationTaskRule().fetch(
            "previous_well", "current_well", "performance_outcome")
        # Check outcomes begin with "correct", "incorrect" or "neutral", followed by "_", as parsing depends on this
        if not all([outcome for outcome in performance_outcomes if outcome.split("_")[0] in
                                                                   ["correct", "incorrect",
                                                                    "neutral"]]):
            raise Exception(
                f"Performance outcomes do not have expected form (begin with correct, "
                f"incorrect or neutral followed by underscore)")
        correct_outcome_idxs = [idx for idx, outcome in enumerate(performance_outcomes) if
                                outcome.split("_")[0] == "correct"]
        abstract_path_name_tuples = (list(zip(previous_well[correct_outcome_idxs],
                                              current_well[correct_outcome_idxs])))

        potential_active_contingencies = (ContingencyActiveContingenciesMap &
                                          {"contingency": contingency}).fetch1("active_contingencies")

        return [tuple([AlternationTaskWellIdentities().get_well_name(w, active_contingency) for w in t])
                for active_contingency in potential_active_contingencies for t in abstract_path_name_tuples]

    @classmethod
    def get_well_names(cls, nwb_file_name=None, epoch=None, contingency=None, rewarded_wells=False, well_order=None):
        # Get well names for fork maze

        # Check inputs
        if rewarded_wells and ((nwb_file_name is not None and epoch is not None) + (contingency is not None) != 1):
            raise Exception(f"If rewarded_wells is True, must pass either nwb_file_name and epoch, OR contingency")
        # check valid well order
        check_membership([well_order], ["left_right", None])

        # Get potentially rewarded wells if indicated
        if rewarded_wells:
            _, destination_wells = zip(*cls.get_rewarded_path_name_tuples(nwb_file_name, epoch, contingency))
            well_names = np.unique(destination_wells)
        # Otherwise get all wells
        else:
            well_names = RewardWell().fetch1("well_names")

        # Order wells if indicated
        ordered_well_names = well_names
        if well_order == "left_right":
             ordered_well_names = np.concatenate(
                [["left_well", f"{home}_well", f"right_well"] for home in ["center", "handle"]])
        check_membership(well_names, ordered_well_names, "well names", "ordered well names")

        return [x for x in ordered_well_names if x in well_names]

    @classmethod
    def get_well_names_across_epochs(cls, nwb_file_name, epochs, rewarded_wells=False, well_order=None):
        all_well_names = np.concatenate([cls.get_well_names(
            nwb_file_name, epoch, rewarded_wells=rewarded_wells, well_order=well_order)
                                         for epoch in epochs], dtype=object)
        return unique_in_order(all_well_names)

    @classmethod
    def get_rewarded_path_names(
            cls, nwb_file_name=None, epoch=None, contingency=None, organize_by_same_turn=False, start_well=None):

        # Check inputs (must pass either nwb_file_name and epoch, OR contingency
        if (nwb_file_name is not None and epoch is not None) + (contingency is not None) != 1:
            raise Exception(f"must pass either nwb_file_name and epoch, OR contingency")

        # Get names of potentially rewarded path in a contingency
        if organize_by_same_turn:
            return np.concatenate(cls.get_same_turn_path_names(nwb_file_name, epoch, contingency, rewarded_paths=True))
        path_name_tuples = cls.get_rewarded_path_name_tuples(nwb_file_name, epoch, contingency)
        rewarded_path_names = convert_path_names(path_name_tuples)
        if start_well is not None:
            return [x for x in rewarded_path_names if x.split("_")[0] == start_well]
        return rewarded_path_names

    @classmethod
    def get_rewarded_path_names_across_epochs(cls, nwb_file_name, epochs, epoch_intersection=False,
                                              organize_by_same_turn=False):
        # Get rewarded paths in each epoch
        rewarded_paths_list = [cls.get_rewarded_path_names(nwb_file_name, epoch,
                                                           organize_by_same_turn=organize_by_same_turn)
                               for epoch in epochs]
        # Take intersection of rewarded paths in each epoch if indicated
        if epoch_intersection:
            return np.asarray(list(set.intersection(*map(set, rewarded_paths_list))))
        # Otherwise, union of rewarded paths in each epoch
        return unique_in_order(np.concatenate(rewarded_paths_list, dtype=object))

    @classmethod
    def get_rewarded_paths_turn_map(cls, nwb_file_name=None, epoch=None, contingency=None):
        path_turn_direction_map = RewardWellPathTurnDirection.fetch1("reward_well_path_turn_direction_map")
        return {path_name: path_turn_direction_map[path_name]
                for path_name in cls.get_rewarded_path_names(nwb_file_name, epoch, contingency)}

    @classmethod
    def get_same_turn_path_names(cls, nwb_file_name=None, epoch=None, contingency=None, rewarded_paths=False):
        if rewarded_paths:
            paths_turn_map = cls.get_rewarded_paths_turn_map(nwb_file_name, epoch, contingency)
        else:
            paths_turn_map = RewardWellPathTurnDirection.fetch1("reward_well_path_turn_direction_map")
        return array_to_tuple_list(pairs_keys_same_value(paths_turn_map))

    @classmethod
    def get_same_turn_path_names_across_epochs(
            cls, nwb_file_name, epochs, rewarded_paths=False, collapse=False, collapsed_path_order=None):

        # Check inputs
        # check collapsed_path_order valid
        check_membership([collapsed_path_order], ["left_right", None])

        # Get same turn path name pairs for each epoch
        path_names = np.concatenate([cls.get_same_turn_path_names(
            nwb_file_name, epoch, rewarded_paths=rewarded_paths) for epoch in epochs], dtype=object)
        # Get unique path name pairs across epochs, preserving order
        path_names = unique_in_order(path_names)
        # Collapse into one list if indicated
        if collapse:
            path_names = np.concatenate(path_names)
            # Order if indicated
            ordered_path_names = copy.deepcopy(path_names)
            if collapsed_path_order == "left_right":
                ordered_path_names = [
                    "left_well_to_center_well", "center_well_to_right_well", "right_well_to_center_well",
                    "center_well_to_left_well", "left_well_to_handle_well", "handle_well_to_left_well",
                    "right_well_to_handle_well", "handle_well_to_right_well", "center_well_to_handle_well",
                    "handle_well_to_center_well"]
            check_membership(path_names, ordered_path_names)
            path_names = [x for x in ordered_path_names if x in path_names]
        return path_names

    @classmethod
    def get_different_turn_path_names(cls, nwb_file_name=None, epoch=None, contingency=None, rewarded_paths=False):
        if rewarded_paths:
            paths_turn_map = cls.get_rewarded_paths_turn_map(nwb_file_name, epoch, contingency)
        else:
            paths_turn_map = RewardWellPathTurnDirection.fetch1("reward_well_path_turn_direction_map")
        return pairs_keys_different_value(paths_turn_map)

    @classmethod
    def get_different_turn_well_path_names(cls, nwb_file_name=None, epoch=None, contingency=None, rewarded_paths=False):
        """
        Get paths with different set of turns and different start/end wells
        :param nwb_file_name: nwb file name
        :param epoch: integer, epoch
        :param rewarded_paths: boolean, True to restrict to rewarded paths
        :return:
        """

        different_turn_path_names = np.asarray(cls.get_different_turn_path_names(
            nwb_file_name, epoch, contingency, rewarded_paths=rewarded_paths))
        different_well_bool = [all([path_1_well != path_2_well
                                    for path_1_well, path_2_well in np.asarray(convert_path_names(path_names)).T])
                               for path_names in different_turn_path_names]  # paths that do not share start/end well
        different_turn_well_path_names = different_turn_path_names[different_well_bool]

        return [(x1, x2) for x1, x2 in different_turn_well_path_names]  # convert back to tuple

    @classmethod
    def get_path_with_home_well_at_position(
            cls, position, nwb_file_name=None, epoch=None, contingency=None, as_dict=False):

        # Check inputs (must pass either nwb_file_name and epoch, OR contingency
        if (nwb_file_name is not None and epoch is not None) + (contingency is not None) != 1:
            raise Exception(f"must pass either nwb_file_name and epoch, OR contingency")

        # Get contingency if not passed
        if contingency is None:
            contingency = TaskIdentification().get_contingency(nwb_file_name, epoch)

        potential_active_contingencies = (ContingencyActiveContingenciesMap &
                                            {"contingency": contingency}).fetch1("active_contingencies")
        home_well_names = [AlternationTaskWellIdentities().get_well_name("home_well", active_contingency)
                           for active_contingency in potential_active_contingencies]
        path_names = {home_well_name: [path_name for path_name in cls.get_rewarded_path_names(contingency=contingency)
                if path_name.split("_to_")[position] == home_well_name] for home_well_name in home_well_names}

        # Return in dictionary with home well name as key if indicated, otherwise concatenate all path names
        if as_dict:
            return path_names
        return np.concatenate(list(path_names.values()))

    @classmethod
    def get_outbound_path_names(cls, nwb_file_name=None, epoch=None, contingency=None, as_dict=False):
        position = 0
        return cls.get_path_with_home_well_at_position(position, nwb_file_name, epoch, contingency, as_dict=as_dict)

    @classmethod
    def get_inbound_path_names(cls, nwb_file_name=None, epoch=None, contingency=None, as_dict=False):
        position = 1
        return cls.get_path_with_home_well_at_position(position, nwb_file_name, epoch, contingency, as_dict=as_dict)

    @classmethod
    def get_path_fns_map(cls):
        # Map from type of path pair relationship to function for obtaining instances of that type
        return {"same_turn": cls.get_same_turn_path_names,
                "same_turn_even_odd_trials": cls.get_same_turn_path_names,
                "same_turn_even_odd_stay_trials": cls.get_same_turn_path_names,
                "same_turn_even_odd_correct_stay_trials": cls.get_same_turn_path_names,

                "different_turn_well": cls.get_different_turn_well_path_names,
                "different_turn_well_even_odd_trials": cls.get_different_turn_well_path_names,
                "different_turn_well_even_odd_stay_trials": cls.get_different_turn_well_path_names,
                "different_turn_well_even_odd_correct_stay_trials": cls.get_different_turn_well_path_names,

                "inbound": cls.get_inbound_path_names,

                "inbound_even_odd_correct_stay_trials": cls.get_inbound_path_names,

                "outbound": cls.get_outbound_path_names,

                "outbound_correct_correct_trials": cls.get_outbound_path_names,
                "outbound_correct_incorrect_trials": cls.get_outbound_path_names,
                "outbound_incorrect_incorrect_trials": cls.get_outbound_path_names,

                "outbound_correct_correct_stay_trials": cls.get_outbound_path_names,
                "outbound_correct_incorrect_stay_trials": cls.get_outbound_path_names,
                "outbound_incorrect_incorrect_stay_trials": cls.get_outbound_path_names,

                "outbound_even_odd_correct_stay_trials": cls.get_outbound_path_names,

                "same_path_outbound_correct_correct_trials": cls.get_outbound_path_names,
                "same_path_outbound_correct_incorrect_trials": cls.get_outbound_path_names,
                "same_path_outbound_prev_correct_incorrect_trials": cls.get_outbound_path_names,

                "same_path_outbound_correct_correct_stay_trials": cls.get_outbound_path_names,

                "same_path": cls.get_rewarded_path_names,

                "same_path_even_odd_trials": cls.get_rewarded_path_names,
                "same_path_even_odd_stay_trials": cls.get_rewarded_path_names,
                "same_path_even_odd_correct_stay_trials": cls.get_rewarded_path_names,

                "same_path_stay_leave_trials": cls.get_rewarded_path_names,
                "same_path_stay_stay_trials": cls.get_rewarded_path_names,
                "same_path_leave_leave_trials": cls.get_rewarded_path_names,

                "same_path_correct_correct_trials": cls.get_rewarded_path_names,
                "same_path_correct_incorrect_trials": cls.get_rewarded_path_names,
                "same_path_incorrect_incorrect_trials": cls.get_rewarded_path_names,

                "same_path_correct_correct_stay_trials": cls.get_rewarded_path_names,
                "same_path_correct_incorrect_stay_trials": cls.get_rewarded_path_names,
                "same_path_incorrect_incorrect_stay_trials": cls.get_rewarded_path_names,

                "same_path_prev_correct_incorrect_trials": cls.get_rewarded_path_names,

                "different_path": cls.get_rewarded_path_names,
                "same_end_well_even_odd_trials": cls.get_rewarded_path_names,
                "same_end_well_even_odd_stay_trials": cls.get_rewarded_path_names,
                "different_end_well_even_odd_trials": cls.get_rewarded_path_names,
                "different_end_well_even_odd_stay_trials": cls.get_rewarded_path_names,
                }

    @staticmethod
    def even_odd_trial_text():
        return ["even", "odd"]

    @classmethod
    def get_even_odd_trial_name(cls, context_name, trial_num=None, even_odd_text=None):
        # Make text like "{context_name}_{even or odd}", e.g. "left_well_to_handle_well_even"

        # Require that only one of path trial number or even/odd text passed
        check_one_none([trial_num, even_odd_text])

        # If even/odd text not passed, get from path trial number
        if even_odd_text is None:
            even_odd_text = get_even_odd_text(trial_num)

        # Check even/odd text valid
        check_membership([even_odd_text], cls.even_odd_trial_text())

        # Return path name and even/odd text together
        return f"{context_name}_{even_odd_text}"

    @classmethod
    def split_even_odd_trial_name(cls, even_odd_trial_path_name):
        # Split path_name from even_odd_text (conceptually, does inverse of above function)
        split_text = even_odd_trial_path_name.split("_")
        path_name = "_".join(split_text[:-1])
        even_odd_text = split_text[-1]
        check_membership([even_odd_text], cls.even_odd_trial_text())
        return path_name, even_odd_text

    @classmethod
    def same_context_even_odd(cls, label_1, label_2):
        path_name_1, even_odd_1 = cls.split_even_odd_trial_name(label_1)
        path_name_2, even_odd_2 = cls.split_even_odd_trial_name(label_2)
        return np.logical_and(path_name_1 == path_name_2, even_odd_1 != even_odd_2)

    @classmethod
    def even_odd(cls, label_1, label_2):
        _, even_odd_1 = cls.split_even_odd_trial_name(label_1)
        _, even_odd_2 = cls.split_even_odd_trial_name(label_2)
        return even_odd_1 != even_odd_2

    @classmethod
    def get_stay_leave_trial_path_name(cls, path_name, trial_text):
        # Check inputs
        check_membership([trial_text], cls.stay_leave_trial_text())
        return f"{path_name}_{trial_text}"

    @classmethod
    def split_stay_leave_trial_path_name(cls, stay_leave_trial_path_name, trial_text=None):

        # Get trial text if not passed
        if trial_text is None:
            trial_text = unpack_single_element(
                [x for x in cls.stay_leave_trial_text() if stay_leave_trial_path_name.endswith(x)])

        # Check inputs
        check_membership([trial_text], cls.stay_leave_trial_text())
        if not stay_leave_trial_path_name.endswith(trial_text):
            raise Exception(f"{stay_leave_trial_path_name} must end with {trial_text}")

        num_exclude = len(trial_text) + 1  # add one to exclude underscore prior to trial text
        return stay_leave_trial_path_name[:-num_exclude]

    @staticmethod
    def stay_leave_trial_text(text_type=None):
        text_map = {"stay": "stay_trial", "leave": "leave_trial"}
        if text_type is not None:
            return text_map[text_type]
        return list(text_map.values())

    @staticmethod
    def correct_incorrect_trial_text(text_type=None, previous_trial=False):
        previous_trial_text = ""
        if previous_trial:
            previous_trial_text = "previous_"
        text_map = {
            "correct": f"{previous_trial_text}correct_trial", "incorrect": f"{previous_trial_text}incorrect_trial"}
        if text_type is not None:
            return text_map[text_type]
        return list(text_map.values())

    @classmethod
    def get_correct_incorrect_trial_text(cls, path_name, correct_incorrect_text=None, previous_trial=False):
        # Make text like "{path_name}_{correct or incorrect}", e.g. "left_well_to_handle_well_correct" if
        # current trial, or "{path_name}_previous_{correct or incorrect}" if previous trial

        # Check correct/incorrect text valid
        check_membership([correct_incorrect_text], cls.correct_incorrect_trial_text(previous_trial=previous_trial))

        # Return path name and correct/incorrect text together
        return f"{path_name}_{correct_incorrect_text}"

    @classmethod
    def _access_get_path_fn(cls, name, params):
        # Helper function for getting instances of a path pair type. Allows calling subfunctions with
        # different inputs programmatically
        fn = cls.get_path_fns_map()[name]

        def _get_param_names(params, param_names):
            return {k: params[k] for k in param_names}

        # Get path name pairs
        # Here, if (a, b) included, (b, a) not included.
        # Note that option to include these suffixes symmetrically (e.g. (a, b) and (b, a)) is in
        # get_path_name_pair_types_map method (with include_reversed_pairs=True), which calls the current method.
        if name in [
            "same_turn", "same_turn_even_odd_trials", "same_turn_even_odd_stay_trials",
            "same_turn_even_odd_correct_stay_trials",
            "different_turn_well",
            "different_turn_well_even_odd_trials", "different_turn_well_even_odd_stay_trials",
            "different_turn_well_even_odd_correct_stay_trials"
        ]:
            param_names = ["nwb_file_name", "epoch", "contingency", "rewarded_paths"]
            path_name_pairs = fn(**_get_param_names(params, param_names))

        elif name in [
            "inbound", "outbound",

            "outbound_correct_correct_trials", "outbound_correct_incorrect_trials",
            "outbound_incorrect_incorrect_trials",

            "outbound_correct_correct_stay_trials", "outbound_correct_incorrect_stay_trials",
            "outbound_incorrect_incorrect_stay_trials",

            "inbound_even_odd_correct_stay_trials", "outbound_even_odd_correct_stay_trials",

        ]:
            param_names = ["nwb_file_name", "epoch", "contingency"]
            fn_params = _get_param_names(params, param_names)
            fn_params.update({"as_dict": True})
            path_name_pairs_map = fn(**fn_params)  # inbound or outbound path names as dictionary
            path_name_pairs = array_to_tuple_list(list(path_name_pairs_map.values()))  # convert to list of tuples

        elif name in [
            "same_path", "same_path_even_odd_trials", "same_path_even_odd_stay_trials",
            "same_path_even_odd_correct_stay_trials", "same_path_stay_leave_trials",
            "same_path_stay_stay_trials", "same_path_leave_leave_trials", "same_path_correct_incorrect_trials",
            "same_path_correct_correct_trials", "same_path_correct_correct_stay_trials",
            "same_path_incorrect_incorrect_trials", "same_path_incorrect_incorrect_stay_trials",
            "same_path_prev_correct_incorrect_trials",
            "same_path_outbound_correct_incorrect_trials", "same_path_outbound_correct_correct_trials",
            "same_path_outbound_correct_correct_stay_trials",
            "same_path_outbound_prev_correct_incorrect_trials",
            "same_path_correct_incorrect_stay_trials"]:
            param_names = ["nwb_file_name", "epoch", "contingency"]
            path_name_pairs = [(x, x) for x in fn(**_get_param_names(params, param_names))]  # pairs of same path

        elif name == "different_path":
            param_names = ["nwb_file_name", "epoch", "contingency"]
            path_name_pairs = list(itertools.combinations(fn(**_get_param_names(params, param_names)), r=2))

        elif name in ["same_end_well_even_odd_trials", "same_end_well_even_odd_stay_trials"]:
            path_name_pairs = np.concatenate(
                [cls._access_get_path_fn(x, params) for x in ["same_path", "different_path"]])
            path_name_pairs = [x for x in path_name_pairs if
                RewardWellPath().get_end_well_name(x[0]) == RewardWellPath().get_end_well_name(x[1])]

        elif name in ["different_end_well_even_odd_trials", "different_end_well_even_odd_stay_trials"]:
            path_name_pairs = np.concatenate(
                [cls._access_get_path_fn(x, params) for x in ["same_path", "different_path"]])
            path_name_pairs = [x for x in path_name_pairs if
                RewardWellPath().get_end_well_name(x[0]) != RewardWellPath().get_end_well_name(x[1])]

        # Raise error if case not accounted for in code
        else:
            raise Exception(f"{name} not accounted for in code")

        # Add suffix to path name pairs if indicated: even/odd trials, stay/leave trials, correct/incorrect trials

        # Add symmetrically if indicated.
        # Below, the default is to include (a_suffix_1, b_suffix_2) but not (a_suffix_2, a_suffix_1).
        # If indicated, also include the latter.
        symmetric_suffix = False
        if "symmetric_suffix" in params:
            symmetric_suffix = params["symmetric_suffix"]

        def _get_suffix_sets(t1, t2, symmetric_suffix):
            suffix_sets = [[t1, t2]]
            if symmetric_suffix and t1 != t2:
                suffix_sets.append([t2, t1])
            return suffix_sets

        # If want even/odd trials, add even and odd to path names
        if "even_odd_trials" in name:
            t1, t2 = cls.even_odd_trial_text()
            suffix_sets = _get_suffix_sets(t1, t2, symmetric_suffix)
            return [
                (cls.get_even_odd_trial_name(x1, even_odd_text=t1),
                 cls.get_even_odd_trial_name(x2, even_odd_text=t2))
                for x1, x2 in path_name_pairs for t1, t2 in suffix_sets]

        # If want stay/leave trials, add stay/leave to path names
        elif any([x in name for x in ["stay_leave_trials", "stay_stay_trials", "leave_leave_trials"]]):
            if "stay_leave_trials" in name:
                t1, t2 = cls.stay_leave_trial_text()
            elif "stay_stay_trials" in name:
                t1 = t2 = cls.stay_leave_trial_text("stay")
            elif "leave_leave_trials" in name:
                t1 = t2 = cls.stay_leave_trial_text("leave")
            suffix_sets = _get_suffix_sets(t1, t2, symmetric_suffix)
            return [
                (cls.get_stay_leave_trial_path_name(x1, t1),
                 cls.get_stay_leave_trial_path_name(x2, t2)) for x1, x2 in path_name_pairs for t1, t2 in suffix_sets]

        # If want even vs. odd stay trials, add stay to path names, then even and odd
        elif "even_odd_stay_trials" in name:
            stay_text = cls.stay_leave_trial_text("stay")
            t1, t2 = cls.even_odd_trial_text()
            suffix_sets = _get_suffix_sets(t1, t2, symmetric_suffix)
            return [
                (cls.get_even_odd_trial_name(cls.get_stay_leave_trial_path_name(x1, stay_text), even_odd_text=t1),
                 cls.get_even_odd_trial_name(cls.get_stay_leave_trial_path_name(x2, stay_text), even_odd_text=t2))
                 for x1, x2 in path_name_pairs for t1, t2 in suffix_sets]

        # If want even vs. odd stay correct trials, add stay to path names, then correct, then even and odd
        elif "even_odd_correct_stay_trials" in name:
            stay_text = cls.stay_leave_trial_text("stay")
            correct_text = cls.correct_incorrect_trial_text("correct")
            t1, t2 = cls.even_odd_trial_text()
            suffix_sets = _get_suffix_sets(t1, t2, symmetric_suffix)
            return [
                (cls.get_even_odd_trial_name(cls.get_correct_incorrect_trial_text(
                    cls.get_stay_leave_trial_path_name(x1, stay_text), correct_text), even_odd_text=t1),
                 cls.get_even_odd_trial_name(cls.get_correct_incorrect_trial_text(
                     cls.get_stay_leave_trial_path_name(x2, stay_text), correct_text), even_odd_text=t2))
                for x1, x2 in path_name_pairs for t1, t2 in suffix_sets]

        # If want correct/incorrect stay trials, add stay to path names, then correct and incorrect
        # If want correct/incorrect trials, add correct and incorrect to path name
        elif any([x in name for x in [
            "correct_incorrect_trials", "correct_correct_trials", "incorrect_incorrect_trials",
            "correct_incorrect_stay_trials", "correct_correct_stay_trials", "incorrect_incorrect_stay_trials"
        ]]):

            previous_trial = False
            if "prev_" in name:
                previous_trial = True

            if "correct_correct" in name:
                t1 = t2 = cls.correct_incorrect_trial_text("correct", previous_trial)
            elif "incorrect_incorrect" in name:
                t1 = t2 = cls.correct_incorrect_trial_text("incorrect", previous_trial)
            elif "correct_incorrect" in name:
                t1, t2 = cls.correct_incorrect_trial_text(previous_trial=previous_trial)
            else:
                raise Exception(f"case not accounted for")

            suffix_sets = _get_suffix_sets(t1, t2, symmetric_suffix)

            stay_text = cls.stay_leave_trial_text("stay")

            stay_trials_fn = return_input
            if "stay_trials" in name:
                stay_trials_fn = cls.get_stay_leave_trial_path_name

            return [
                (cls.get_correct_incorrect_trial_text(stay_trials_fn(x1, trial_text=stay_text), t1, previous_trial),
                 cls.get_correct_incorrect_trial_text(stay_trials_fn(x2, trial_text=stay_text), t2, previous_trial))
                for x1, x2 in path_name_pairs for t1, t2 in suffix_sets]

        # Otherwise return path name pairs unaltered
        return path_name_pairs

    @classmethod
    def get_path_name_pair_types_map(
            cls, nwb_file_name=None, epoch=None, contingency=None, rewarded_paths=True, include_reversed_pairs=False,
            symmetric_suffix=False, valid_names=None):

        # If rewarded_paths True, restrict to potentially rewarded paths

        # Initialize map
        path_name_pair_types_map = dict()

        # Get names for all possible pair types
        names = cls.get_path_fns_map().keys()

        # Restrict names if indicated
        if valid_names is not None:
            names = [x for x in names if x in valid_names]

        # Loop through names
        for name in names:

            pair_types = cls._access_get_path_fn(name, {
                "nwb_file_name": nwb_file_name, "epoch": epoch, "contingency": contingency,
                "rewarded_paths": rewarded_paths, "symmetric_suffix": symmetric_suffix})

            # If include reversed pairs, then for each (a, b), add (b, a)
            if include_reversed_pairs:
                pair_types = add_reversed_pairs(pair_types)

            # Add to map pair types for this epoch and name
            path_name_pair_types_map[name] = pair_types

        # Return map
        return path_name_pair_types_map

    @classmethod
    def get_well_name_pair_types_map(cls, nwb_file_name=None, epoch=None, contingency=None, rewarded_wells=True):

        # Check inputs (must pass either nwb_file_name and epoch, OR contingency
        if (nwb_file_name is not None and epoch is not None) + (contingency is not None) != 1:
            raise Exception(f"must pass either nwb_file_name and epoch, OR contingency")

        # If rewarded_wells True, restrict to potentially rewarded wells
        well_names = cls.get_well_names(nwb_file_name, epoch, contingency, rewarded_wells=rewarded_wells)
        return {
            "same_well": [(x, x) for x in well_names], "different_well": list(
                itertools.combinations(well_names, r=2))}

    @classmethod
    def get_end_well_name_pair_types_map(cls, nwb_file_name, epochs, rewarded_wells=True):
        # If rewarded_wells True, restrict to potentially rewarded wells
        well_name_pair_types_map = dict()
        for epoch in epochs:
            well_names = cls.get_well_names(nwb_file_name, epoch, rewarded_wells=rewarded_wells)
            well_name_pair_types_map[epoch] = {
                "same_well": [(x, x) for x in well_names], "different_well": list(
                    itertools.combinations(well_names, r=2))}
        return well_name_pair_types_map


def return_input(x, **kwargs):
    return x


def populate_jguidera_maze(key=None, tolerate_error=False):
    populate_track_graph_tables_fork_maze()
    schema_name = "jguidera_maze"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_maze():
    schema.drop()
