import networkx as nx


def simulate_fork_maze(arm_length=50, connector_length=20):
    """
    Make networkx track graph for simulated fork maze
    :param arm_length: length of long arms of fork maze
    :param connector_length: length of short connecting segments of fork maze
    :return: networkx graph object
    """

    node_positions_dict = {"center_well": {"x": connector_length,
                                           "y": arm_length*2},
                           "center_maze": {"x": connector_length,
                                           "y": arm_length},
                           "right_corner": {"x": connector_length*2,
                                            "y": arm_length},
                           "right_well": {"x": connector_length*2,
                                          "y": arm_length*2},
                           "left_corner": {"x": 0,
                                           "y": arm_length},
                           "left_well": {"x": 0,
                                         "y": arm_length*2},
                           "handle_well": {"x": connector_length,
                                           "y": 0}}

    return make_fork_maze_track_graph(node_positions_dict)


def make_fork_maze_track_graph(node_positions_dict):
    """
    Make fork maze track graph
    :param node_positions_dict: contains x,y positions of nodes in fork maze.
    Has form {"node_name": {"x": value, "y": value},...} for each node in hardcoded
    node_names.
    :return: networkX track graph
    """

    # Define node names and positions
    node_names = ["center_well",
                 "center_maze",
                 "right_corner",
                 "right_well",
                 "left_corner",
                 "left_well",
                 "handle_well"]

    node_positions = tuple([(node_positions_dict[node_name]["x"], node_positions_dict[node_name]["y"])
                           for node_name in node_names])

    # Define edges
    edge_list = [(0, 1),
                 (6, 1),
                 (1, 2),
                 (2, 3),
                 (1, 4),
                 (4, 5)]

    return make_track_graph(node_names, node_positions, edge_list)


def make_track_graph(node_names, node_positions, edge_list):

    track_graph = nx.Graph()  # initialize track graph
    track_graph.add_nodes_from([(node_num, {"node_name": node_name,
                                            "pos": node_pos})
                                for node_num, (node_pos, node_name) in
                                enumerate(zip(node_positions, node_names))])  # add nodes to graph
    from src.jguides_2024.utils.vector_helpers import euclidean_distance
    track_graph.add_edges_from([(edge[0], edge[1],
                                 {"length": euclidean_distance(track_graph.nodes[edge[0]]["pos"],
                                                                 track_graph.nodes[edge[1]]["pos"]),
                                  "edge_name": tuple([track_graph.nodes[edge[i]]["node_name"] for i in [0, 1]])})
                                  for edge in edge_list])  # add edges to graph
    return track_graph
