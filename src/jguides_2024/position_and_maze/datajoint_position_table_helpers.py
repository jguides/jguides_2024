from spyglass.common.common_position import IntervalPositionInfo

from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName


def fetch1_IntervalPositionInfo(nwb_file_name, epoch, position_info_param_name="default"):
    interval_list_name = EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch)
    return (IntervalPositionInfo & {"position_info_param_name": position_info_param_name,
                                    "nwb_file_name": nwb_file_name,
                                    "interval_list_name": interval_list_name}).fetch1_dataframe()