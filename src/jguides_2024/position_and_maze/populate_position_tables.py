import numpy as np
# Populate lab position_and_maze tables
from spyglass.common.common_position import (IntervalPositionInfoSelection, IntervalPositionInfo,
                                             TrackGraph, IntervalLinearizationSelection, IntervalLinearizedPosition)

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import populate_flexible_key, \
    add_param_defaults
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_epoch_keys
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.position_and_maze.jguidera_maze import get_fork_maze_track_graph_name
from src.jguides_2024.position_and_maze.jguidera_position import IntervalLinearizedPositionRelabel, \
    IntervalLinearizedPositionRescaled, \
    IntervalPositionInfoRelabel
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import check_length


def populate_IntervalLinearizedPosition(nwb_file_name, epoch, position_info_param_name="default",
                                        linearization_param_name="default", tolerate_error=False):

    # Get interval list name for epoch
    EpochIntervalListName().populate_(
        key={"nwb_file_name": nwb_file_name, "epoch": epoch}, tolerate_error=tolerate_error)
    interval_list_name = EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch, tolerate_error)

    # Exit if cannot populate and tolerating error
    if interval_list_name is None:
        print(f"Could not populate IntervalLinearizedPosition for {nwb_file_name}, epoch {epoch} because no "
              f"corresponding entry in EpochIntervalListName. Exiting")
        return

    # Make key
    key = {"nwb_file_name": nwb_file_name,
           "interval_list_name": interval_list_name,
           "position_info_param_name": position_info_param_name,
           "linearization_param_name": linearization_param_name}

    # Get track graph name based on environment
    environment = TaskIdentification.get_environment(nwb_file_name, epoch)
    track_graph_name = get_fork_maze_track_graph_name(environment)
    # Check track graph name exists in track graph table
    check_length((TrackGraph & {"track_graph_name": track_graph_name}), 1)
    # Add to key
    key.update({"track_graph_name": track_graph_name})

    # Insert into selection and position_and_maze tables
    try:
        IntervalLinearizationSelection.insert1(key, skip_duplicates=True)
        populate_flexible_key(IntervalLinearizedPosition, key=key, tolerate_error=tolerate_error)
    except Exception as e:
        if tolerate_error:
            print(e)
        else:
            raise e


def populate_IntervalPositionInfo(nwb_file_name, epoch, position_info_param_name="default", tolerate_error=False):
    # Initialize key for populating table
    key = {"nwb_file_name": nwb_file_name, "position_info_param_name": position_info_param_name}
    # Add interval list name for epoch to key
    EpochIntervalListName().populate_(key={"nwb_file_name": nwb_file_name, "epoch": epoch})
    interval_list_name = EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch, tolerate_error)
    if interval_list_name is None:
        print(f"Could not insert {nwb_file_name} epoch {epoch} into IntervalPositionInfoSelection because "
              f"no corresponding entry in EpochIntervalListName")
        return
    key.update({"interval_list_name": interval_list_name})
    # Populate selection and position_and_maze tables
    IntervalPositionInfoSelection.insert1(key, skip_duplicates=True)
    populate_flexible_key(IntervalPositionInfo, key, tolerate_error)


def populate_position_tables_wrapper(keys=None, tolerate_error=False):
    # Check keys not dict (to signal to user if a key was accidentally passed)
    if isinstance(keys, dict):
        raise Exception(f"keys should be a list of dictionaries, not a dict")

    # Check whether to get default keys (if keys is None or [None])
    get_default_keys = False  # default
    if keys is None:
        get_default_keys = True
    else:  # keys not None
        if np.logical_and(len(keys) == 1, keys[0] is None):
            get_default_keys = True

    # Get keys if not passed
    if get_default_keys:
        keys = get_jguidera_nwbf_epoch_keys()

    # Otherwise if keys passed, check whether each key has epoch. If not, add keys with run epochs (need to pass next
    # function a key with both nwb file and epoch)
    else:
        new_keys = []
        for key in keys:
            if "epoch" not in key and "nwb_file_name" in key:
                RunEpoch().populate_(key)
                for epoch in RunEpoch().get_epochs(key["nwb_file_name"]):
                    new_keys.append({**key, **{"epoch": epoch}})
            else:
                new_keys.append(key)
        keys = new_keys

    # Loop through keys and populate position_and_maze table
    for key in keys:
        for position_info_param_name in ["default", "default_decoding"]:
            key.update({"position_info_param_name": position_info_param_name})
            populate_position_tables(key, tolerate_error)


def populate_position_tables(key, tolerate_error=False):
    # Check inputs
    check_membership(["nwb_file_name", "epoch"], key, "required params", "passed key")

    # Get inputs if not passed
    key = add_param_defaults(key, add_nonexistent_keys=True, replace_none=True)

    # Unpack key
    position_info_param_name = key["position_info_param_name"]
    linearization_param_name = key["linearization_param_name"]
    nwb_file_name = key["nwb_file_name"]
    epoch = key["epoch"]

    # Populate lab position_and_maze tables
    populate_IntervalPositionInfo(nwb_file_name, epoch, position_info_param_name, tolerate_error)
    populate_IntervalLinearizedPosition(nwb_file_name, epoch, position_info_param_name, linearization_param_name,
                                        tolerate_error)

    # Populate custom position_and_maze tables that build on lab tables
    for table in [IntervalPositionInfoRelabel, IntervalLinearizedPositionRelabel, IntervalLinearizedPositionRescaled]:
        table().populate_(key=key, tolerate_error=tolerate_error)
