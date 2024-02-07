"""This module has functions to check whether data is well defined"""

from src.jguides_2024.position_and_maze.jguidera_maze import AnnotatedUniversalTrackGraph


def check_edge_names_valid(edge_names, universal_track_graph_name="fork_maze_universal"):
    """
    Check whether names of edges of track graph are valid
    :param edge_names: list of edge names as strings
    :param universal_track_graph_name: str, name of univresal track graph
    """

    valid_edge_names = (AnnotatedUniversalTrackGraph & {"universal_track_graph_name":
                                                         universal_track_graph_name}).fetch1("edge_names")
    if not all([edge_name in valid_edge_names for edge_name in edge_names]):
        raise Exception(f"All edge names must be in {valid_edge_names}")


def check_num_units(num_units, max_num_units=None):
    """
    CHeck number of units well defined
    :param num_units: number of units (should be integer)
    :param max_num_units: int, maximum tolerated number of units
    """

    # Check num_units well defined
    if not isinstance(num_units, int):
        raise Exception(f"num_units must be an integer")
    if num_units < 1:
        raise Exception(f"num_units must be greater than zero")
    if max_num_units is not None:
        if num_units > max_num_units:
            raise Exception(f"num_units must be less than {max_num_units}")


def check_unit_subset_vars(unit_subset, num_units, max_num_units=None):
    """
    Check variables for unit subset
    :param unit_subset: should be boolean indicating whether using unit subset
    :param num_units: number of units (should be integer)
    :param max_num_units: int, maximum tolerated number of units
    """

    if not isinstance(unit_subset, bool):
        raise Exception(f"unit_subset must be a boolean")
    if unit_subset:
        check_num_units(num_units, max_num_units)
