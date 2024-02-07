import os

import numpy as np
import pandas as pd

"""
Note that some imports in this module are local b/c sometimes import functions from this module in rec to nwb 
environment
"""


def get_environments():
    return ["HaightLeft", "HaightRight", "SA"]


def get_all_subject_id_recording_dates():

    return {"peanut": [str(x) for x in np.arange(20201101, 20201120)],
           "J16": [str(x) for x in np.concatenate((np.arange(20210529, 20210532),
                                                   np.arange(20210601, 20210621)))],
           "fig": [str(x) for x in np.arange(20211101, 20211121)],
           "mango": ["20211129", "20211130"] +
                    [str(x) for x in np.arange(20211201, 20211220)],
           "june": [str(x) for x in np.arange(20220412, 20220428)]}


def get_subject_id_recording_dates(subject_id):
    return get_all_subject_id_recording_dates()[subject_id]


def get_subject_ids():
    return list(get_all_subject_id_recording_dates().keys())


def get_jguidera_subject_ids():
    return list(get_all_subject_id_recording_dates().keys())


def get_high_priority_nwb_file_names():
    return ["J1620210529_.nwb", "J1620210605_.nwb", "J1620210606_.nwb",
            "J1620210607_.nwb",
            "mango20211205_.nwb", "mango20211206_.nwb", "mango20211207_.nwb",
            "june20220419_.nwb", "june20220420_.nwb", "june20220421_.nwb",
            "peanut20201107_.nwb", "peanut20201108_.nwb", "peanut20201109_.nwb",
            "fig20211108_.nwb", "fig20211109_.nwb", "fig20211110_.nwb"]


def get_highest_priority_nwb_file_names():
    return ["J1620210605_.nwb", "J1620210606_.nwb", "J1620210607_.nwb",
            "mango20211205_.nwb", "mango20211206_.nwb", "mango20211207_.nwb",
            "june20220419_.nwb", "june20220420_.nwb", "june20220421_.nwb",
            "peanut20201107_.nwb", "peanut20201108_.nwb", "peanut20201109_.nwb",
            "fig20211108_.nwb", "fig20211109_.nwb", "fig20211110_.nwb", ]


def get_jguidera_nwbf_names(high_priority=True, highest_priority=False, as_dict=False):

    # Import spyglass file here to avoid error when running other functions in this script
    # outside spyglass environment
    from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import nwbf_name_from_subject_id_date

    nwbf_names = [nwbf_name_from_subject_id_date(subject_id, date)
            for subject_id, dates in get_all_subject_id_recording_dates().items()
            for date in dates]

    # Restrict to subset deemed high priority if indicated
    if highest_priority:
        nwbf_names = [x for x in get_highest_priority_nwb_file_names() if x in nwbf_names]
    if high_priority:
        nwbf_names = [x for x in get_high_priority_nwb_file_names() if x in nwbf_names]

    # Return as dictionary if indicated, otherwise as list
    if as_dict:
        return [{"nwb_file_name": x} for x in nwbf_names]
    return nwbf_names


def get_jguidera_nwbf_epoch_keys(high_priority=False, highest_priority=True):
    """
    Get keys of form [{"nwb_file_name": nwb_file_name, "epoch": epoch}, ...] for run epochs
    :param high_priority: bool. If True and highest_priority False, return high priority subset. Default is False
    :param highest_priority: bool. If True, return highest priority subset. Default is False.
    :return: key_list: list with keys in above form
    """

    # If go back to using spyglass table: import locally to avoid error when importing module to run other
    # functions in this script outside spyglass environment
    from src.jguides_2024.metadata.jguidera_epoch import RunEpoch  # local import to avoid circular import error

    # Note that cannot use RunEpoch here because would require circular import
    return [{"nwb_file_name": nwb_file_name, "epoch": epoch} for nwb_file_name in
            get_jguidera_nwbf_names(high_priority, highest_priority)
            for epoch in RunEpoch().get_epochs(nwb_file_name)]


def get_recording_spreadsheet_path(subject_id):
    """
    Return path to google spreadsheet with notes on neural recordings from rats
    :param subject_id: str, name of rat
    :return: path to google spreadsheet
    """

    return f'/media/jguidera/5fa57296-2b8d-4c4f-90e7-7edf05167424/Frank Lab Dropbox/Jennifer Guidera/UCSF_PhD/' \
        f'Frank_lab/data/{subject_id}_data'


def _load_notes(subject_id, spreadsheet_name, recording_spreadsheet_path=None, header=0, tolerate_no_notes=False):
    """
    Load spreadsheet for a given subject
    """

    # Get inputs if not passed
    if recording_spreadsheet_path is None:
        recording_spreadsheet_path = get_recording_spreadsheet_path(subject_id)

    # Get file path
    file_path = os.path.join(recording_spreadsheet_path, spreadsheet_name)

    # If tolerating no notes and no notes, return empty df
    if tolerate_no_notes and not os.path.exists(file_path):
        return pd.DataFrame()

    # Return recording google spreadsheet as pandas dataframe
    return pd.read_csv(file_path, header=header)


def load_recording_notes(subject_id, recording_spreadsheet_path=None, tolerate_no_notes=False):
    """
    Load recording google spreadsheet for a given subject
    """

    return _load_notes(
        subject_id, spreadsheet_name=f"{subject_id}_experimentalPlan - Recordings.csv",
        recording_spreadsheet_path=recording_spreadsheet_path, header=1, tolerate_no_notes=tolerate_no_notes)


def load_curation_merge_notes(subject_id, date, recording_spreadsheet_path=None, tolerate_no_notes=False):
    """
    Load recording google spreadsheet from saved file for a given subject
    """

    return _load_notes(
        subject_id, spreadsheet_name=f"curation_merge - {subject_id}{date}_summary.csv",
        recording_spreadsheet_path=recording_spreadsheet_path, header=[0, 1], tolerate_no_notes=tolerate_no_notes)


def is_fl_date(x):
    """
    Finds date directories in the 21st century formated according to Frank lab convention:
    YYYYMMDD where YYYY is the year, MM is the month, and DD is the date
    :param x: string
    :return: boolean indicating whether string is a date directory
    """

    return len(x) == 8 and x[:2] == "20"


def get_date_subcontents(path=None):
    return np.sort([x for x in os.listdir(path) if is_fl_date(x)])


def get_video_files(path=None):
    video_file_endings = ["pos_offline", "trackgeometry", "videoPositionTracking"]
    return np.sort([x for x in os.listdir(path) if any([y in x for y in video_file_endings])])


def get_dropbox_path(backup_folder=False):

    path = "/media/jguidera/5fa57296-2b8d-4c4f-90e7-7edf05167424/Frank\ Lab\ Dropbox/Jennifer\ Guidera/"

    if backup_folder:
        path += "backup_from_waterspout/"

    return path


def get_brain_regions(nwb_file_name, targeted=False):

    from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import subject_id_date_from_nwbf_name

    subject_id, _ = subject_id_date_from_nwbf_name(nwb_file_name)
    brain_regions = ["CA1", "mPFC", "OFC"]  # default
    if subject_id in ["peanut"]:
        brain_regions = ["CA1", "OFC"]
    elif subject_id in ["fig"]:
        brain_regions = ["OFC"]

    # Add "_targeted" if indicated, otherwise return brain regions as they are
    if targeted:
        return [f"{x}_targeted" for x in brain_regions]
    return brain_regions


class Contingency:

    def __init__(self):
        self.valid_contingencies = self.get_contingency_map()

    @staticmethod
    def get_contingency_map():

        return {"run": ["centerAlternation", "handleAlternation", "leftAlternation", "rightAlternation",
                          "centerThenHandleAlternation", "handleThenCenterAlternation"],
                "sleep": ["sleep"],
                "home": ["home"]}

    def check_valid_contingency(self, contingency):

        if contingency not in np.concatenate(list(self.valid_contingencies.values())):
            raise Exception(f"Contingency {contingency} not accounted for in code.")

    def get_contingency_type(self, contingency):

        from src.jguides_2024.utils.vector_helpers import unpack_single_element

        return unpack_single_element([contingency_type for contingency_type, contingencies
                                      in self.valid_contingencies.items() if contingency in contingencies])


def get_delay_duration():
    """
    Return duration of delay period in seconds
    """
    return 2


def get_delay_interval():
    """
    Return delay interval (0s to 2s)
    """
    return [0, get_delay_duration()]


def get_nwb_file_name_epochs_description(nwb_file_name, epochs_description):
    """
    Concatenate nwb file name and epochs description
    :param nwb_file_name: str, name of nwb file
    :param epochs_description: str, description of epoch
    :return: str
    """
    return f"{nwb_file_name}_{epochs_description}"

