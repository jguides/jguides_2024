import copy

import datajoint as dj
import numpy as np
import pandas as pd
from replay_trajectory_classification.environments import Environment
from spyglass.common.common_position import TrackGraph
from spyglass.decoding.sorted_spikes import SortedSpikesClassifierParameters

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_reliability_paper_nwb_file_names
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.position_and_maze.jguidera_maze import AnnotatedTrackGraph, get_fork_maze_track_graph_name, \
    MazePathWell, \
    return_n_junction_path_names
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellArrivalTrialsParams
from src.jguides_2024.utils.dict_helpers import check_dict_equality
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements
from src.jguides_2024.utils.vector_helpers import unpack_single_element

schema = dj.schema("jguidera_edeno_decoder_helpers")


# TODO (p2): delete from TrackGraph and SortedSpikesClassifierParameters when delete from tables used to insert into it.


@schema
class StackedEdgeTrackGraph(dj.Manual):
    definition = """
    # Track graph with single edge for edeno Bayesian decoder
    -> TrackGraph
    ---
    decode_variable : varchar(40)
    track_graph_idx : int
    stacked_edge_track_graph_params : blob
    """

    @staticmethod
    def get_decode_variable_track_graph_num_segments(decode_variable):
        num_segments_map = {
            "pstpt": [2, 4], "popt": [2], "pipt": [2], "plpt": [2], "prpt": [2], "purpt": [4, 8],
            "ptjpt": [8], "ppt": [1], "wa": [1]}
        return num_segments_map[decode_variable]

    @classmethod
    def get_track_graph_name_for_decode_variable(
            cls, decode_variable, num_segments=None, idx=1, nwb_file_name=None, epoch=None):

        # Check inputs
        valid_decode_var_names = get_valid_decode_variable_names()
        if decode_variable not in valid_decode_var_names:
            raise Exception(f"decode variable must be in {valid_decode_var_names}")
        if decode_variable == "pos" and (nwb_file_name is None or epoch is None):
            raise Exception(f"Must pass nwb_file_name and epoch if decode_variable is pos")
        if decode_variable == "purpt" and (nwb_file_name is None or epoch is None) and num_segments is None:
            raise Exception(f"Must pass nwb_file_name and epoch, OR num_segments, if decode_variable is purpt")

        # Use environment as track graph name for linear position_and_maze
        if decode_variable == "pos":
            environment = (TaskIdentification & {
                "nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1("task_environment")
            return get_fork_maze_track_graph_name(environment)
        # Otherwise, use name of decode variable, number of segments in track graph, and an index
        # ...Get number of segments if not passed
        if num_segments is None:
            num_segments_list = np.asarray(cls.get_decode_variable_track_graph_num_segments(decode_variable))
            # Case where number of segments depends on contingency:
            if decode_variable in ["pstpt", "purpt"]:
                target_quantity = len(MazePathWell.get_rewarded_path_names(nwb_file_name, epoch))  # default
                # Divide number of segments by 2 for pstpt since combine paths with same turn direction
                if decode_variable == "pstpt":
                    target_quantity /= 2
                num_segments = unpack_single_element(num_segments_list[num_segments_list == target_quantity])
            # All other cases:
            else:
                num_segments = unpack_single_element(num_segments_list)
        return f"{decode_variable}_seg{num_segments}_{idx}"

    @staticmethod
    def _get_node_positions(scale_factor, end_node_value, num_segments):

        edge_distance = end_node_value * scale_factor

        return [(0, edge_distance*(segment_num + 1)) for segment_num in np.arange(0, num_segments + 1)]

    @staticmethod
    def _get_edges(num_segments):
        return np.asarray(zip_adjacent_elements(np.arange(0, num_segments + 1)))

    @classmethod
    def _insert_upstream(cls, track_graph_name, params):

        # Define node positions
        node_positions = cls._get_node_positions(
            params["scale_factor"], params["end_node_value"], params["num_segments"])

        # Define edges
        edges = cls._get_edges(params["num_segments"])

        # Define linear edge spacing
        path_groups_spacing = np.nan
        if "path_groups_spacing" in params:
            path_groups_spacing = params["path_groups_spacing"]
        linear_edge_spacing = path_groups_spacing*params["scale_factor"]

        # Insert into upstream table
        environment = copy.deepcopy(track_graph_name)  # set environment to track graph name
        # if entry already exists, check that matches entry here (limiting to table columns present in
        # entry here, since there may be table columns not specified here where defaults are then inserted)
        upstream_key = {
            "track_graph_name": track_graph_name, "environment": environment,
            "node_positions": node_positions, "edges": edges, "linear_edge_order": edges,
            "linear_edge_spacing": linear_edge_spacing}
        table_subset = TrackGraph & {"track_graph_name": track_graph_name}
        if len(table_subset) > 0:
            check_dict_equality(
                [upstream_key, {k: table_subset.fetch1(k) for k in upstream_key}], tolerate_nonequal_nan=True)
        TrackGraph.insert1(upstream_key, skip_duplicates=True)

    def _insert_entry(self, track_graph_name, decode_variable, track_graph_idx, params):

        # Insert into TrackGraph
        self._insert_upstream(track_graph_name, params)

        # Insert into main table
        self.insert1({
            "track_graph_name": track_graph_name, "decode_variable": decode_variable,
            "track_graph_idx": track_graph_idx, "stacked_edge_track_graph_params": params}, skip_duplicates=True)

    def insert_defaults(self, **kwargs):

        # First parameter set
        track_graph_idx = 1

        # Define common parameters

        # ...params for path variables
        scale_factor = 200
        end_node_value = 1
        well_period_value = np.nan

        # Define parameter sets for each variable

        # ...ppt (fraction path traversed)
        decode_variable = "ppt"
        num_segments = 1
        track_graph_name = self.get_track_graph_name_for_decode_variable(decode_variable, num_segments, track_graph_idx)
        params = {
            "scale_factor": scale_factor, "end_node_value": end_node_value, "well_period_value":
                well_period_value, "num_segments": num_segments}
        self._insert_entry(track_graph_name, decode_variable, track_graph_idx, params)

        # ....turn zone decode variables
        insert_tz = False  # temporarily remove
        if insert_tz:
            default_path_groups_spacing = .15
            turn_zone_decode_variable_names = get_valid_turn_zone_decode_variable_names()
            for decode_variable in turn_zone_decode_variable_names:
                num_segments_list = self.get_decode_variable_track_graph_num_segments(decode_variable)
                for num_segments in num_segments_list:
                    track_graph_name = self.get_track_graph_name_for_decode_variable(
                        decode_variable, num_segments, track_graph_idx)
                    params = {
                        "scale_factor": scale_factor, "end_node_value": end_node_value,
                        "well_period_value": well_period_value,
                        "num_segments": num_segments, "path_groups_spacing": default_path_groups_spacing}
                    self._insert_entry(track_graph_name, decode_variable, track_graph_idx, params)

        # ...wa (time elapsed in 2s delay)
        scale_factor = 25
        decode_variable = "wa"
        dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_delay_param_name()
        end_node_value = DioWellArrivalTrialsParams().trial_duration(dio_well_arrival_trials_param_name)
        num_segments = 1
        track_graph_name = self.get_track_graph_name_for_decode_variable(decode_variable, num_segments, track_graph_idx)
        params = {
            "scale_factor": scale_factor, "end_node_value": end_node_value, "num_segments": num_segments,
            "dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name}
        self._insert_entry(track_graph_name, decode_variable, track_graph_idx, params)


@schema
class EDPathGroups(dj.Manual):
    definition = """
    # Map from decode variable to path group
    -> RunEpoch
    decode_variable : varchar(40)
    ---
    path_groups : blob
    """

    @staticmethod
    def _package_path_names(path_names):
        return [[x] for x in path_names]

    @classmethod
    def _get_decode_variable_path_groups_map(cls, nwb_file_name, epoch):
        # Define map from decode variable to path groups
        # TODO: consider renaming pstpt to pstrpt (since only using rewarded paths)

        return {"pstpt": cls._package_path_names(
            MazePathWell.get_same_turn_path_names(nwb_file_name, epoch, rewarded_paths=True)),
            "pipt": cls._package_path_names(MazePathWell.get_inbound_path_names(nwb_file_name, epoch)),
            "popt": cls._package_path_names(MazePathWell.get_outbound_path_names(nwb_file_name, epoch)),
            "plpt": cls._package_path_names(
                MazePathWell.get_rewarded_path_names(nwb_file_name, epoch, start_well="left")),
            "prpt": cls._package_path_names(
                MazePathWell.get_rewarded_path_names(nwb_file_name, epoch, start_well="right")),
            "purpt": cls._package_path_names(MazePathWell.get_rewarded_path_names(nwb_file_name, epoch)),
            "ptjpt": cls._package_path_names(return_n_junction_path_names(
                n_junctions=2, universal_track_graph_name="fork_maze_universal"))}

    def insert_defaults(self):

        for nwb_file_name in get_reliability_paper_nwb_file_names():

            key = {"nwb_file_name": nwb_file_name}

            for epoch in (RunEpoch & key).fetch("epoch"):

                key.update({"epoch": epoch})
                decode_variable_path_groups_map = self._get_decode_variable_path_groups_map(nwb_file_name, epoch)

                for decode_variable, path_groups in decode_variable_path_groups_map.items():
                    key.update({"decode_variable": decode_variable, "path_groups": path_groups})

                self.insert1(key, skip_duplicates=True)


@schema
class TrackGraphSourceSortedSpikesClassifierParams(dj.Manual):
    definition = """
    # Map from single edge track graph to source sorted spikes classifier parameters used to insert into TrackGraph
    -> TrackGraph
    ---
    -> SortedSpikesClassifierParameters.proj(source_classifier_param_name = 'classifier_param_name')
    """

    # Perform insertion into SortedSpikesClassifierParameters while inserting into this table
    def insert1(self, key, **kwargs):
        """
        # Start with a "source" entry in SortedSpikesClassifierParameters, replace environment information using a
        given track graph, and store as new entry
        """
        # Unpack key
        track_graph_name = key["track_graph_name"]
        # Get starting parameters to edit
        parameters = (SortedSpikesClassifierParameters & {
            "classifier_param_name": key["source_classifier_param_name"]}).fetch1()

        # Replace sections of classifier parameters
        place_bin_size = 2  # 1/100  # 2/18/23: for some reason decoding extremely slow using unscaled path fraction
        position_std = place_bin_size * 3
        track_graph = (TrackGraph & {'track_graph_name': track_graph_name}).get_networkx_track_graph()
        track_graph_params = (TrackGraph & {'track_graph_name': track_graph_name}).fetch1()
        parameters['classifier_params'].update({'environments': [
            Environment(track_graph=track_graph,
                        place_bin_size=place_bin_size,
                        edge_order=track_graph_params['linear_edge_order'],
                        edge_spacing=track_graph_params['linear_edge_spacing'])],
            'sorted_spikes_algorithm': 'spiking_likelihood_kde_gpu',
            "sorted_spikes_algorithm_params": {"position_std": position_std}})
        parameters['classifier_params']["continuous_transition_types"][0][0].movement_var = position_std

        # Insert into table
        SortedSpikesClassifierParameters.insert1(
            {'classifier_param_name': track_graph_name,
             'classifier_params': parameters['classifier_params'],
             'fit_params': parameters['fit_params'],
             'predict_params': parameters['predict_params']},
            skip_duplicates=True)
        super().insert1(key, **kwargs)

    def insert_defaults(self, **kwargs):
        source_classifier_param_name = "default_decoding_gpu2"  # "default_decoding_gpu_source"
        for track_graph_name in get_jguidera_track_graph_names():
            self.insert1(
                {"track_graph_name": track_graph_name, "source_classifier_param_name": source_classifier_param_name},
                skip_duplicates=True)


def get_jguidera_track_graph_names():

    decode_variable_track_graph_names = StackedEdgeTrackGraph.fetch("track_graph_name")
    fork_maze_track_graph_names = AnnotatedTrackGraph.fetch("track_graph_name")

    return set(np.concatenate((decode_variable_track_graph_names, fork_maze_track_graph_names)))


def get_valid_turn_zone_decode_variable_names():
    # TODO: consider renaming turn zone to path group
    # TODO: would be good to store all details of these in one table
    return ["pstpt", "popt", "pipt", "plpt", "prpt", "purpt", "ptjpt"]


def get_valid_one_start_well_decode_variable_names():
    return ["plpt", "prpt"]


def get_valid_decode_variable_names():
    """Return names of decode variables"""
    return ["pos", "ppt", "wa"] + list(get_valid_turn_zone_decode_variable_names()) + \
           list(get_valid_one_start_well_decode_variable_names())



def max_posterior_position(posterior, environment_obj):
    """
    Get position_and_maze at maximum posterior value at each time step
    :param posterior: object from edeno classifier
    :return: position_and_maze at maximum posterior value
    """

    posterior = posterior.sum("state").where(environment_obj.is_track_interior_)
    max_idxs = np.nanargmax(posterior.data, axis=1)

    return pd.DataFrame.from_dict({
        "map_posterior": posterior.position[max_idxs], posterior.time.name: posterior.time}).set_index("time")


def populate_jguidera_edeno_decoder_helpers(key=None, tolerate_error=False):
    schema_name = "jguidera_edeno_decoder_helpers"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_edeno_decoder_helpers():
    schema.drop()

