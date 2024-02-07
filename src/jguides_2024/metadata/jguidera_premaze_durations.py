# This module defines a table with durations in seconds of the period from starting a recording to placing the rat
# on the track

import datajoint as dj
import numpy as np
from spyglass.common import TaskEpoch

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert

schema = dj.schema("jguidera_premaze_durations")


@schema
class PremazeDurations(dj.Manual):
    definition = """
    # Table with durations in seconds of period after recording started and before rat placed on track
    -> TaskEpoch
    ---
    premaze_duration : int
    """

    def insert_defaults(self, **kwargs):
        # Hardcode map between nwb file name and run epoch, and premaze duration (period when rat being
        # carried to track).
        # Premaze durations determined by watching playback of Trodes rec file
        premaze_duration_dict = {
                                # Peanut
                                "peanut20201101_.nwb": {"premaze_durations": [16, 12, 13, 16, 15, 18, 17, 24],
                                                        "epochs": [2, 5, 7, 9, 11, 13, 15, 17]},
                                "peanut20201103_.nwb": {"premaze_durations": [17, 21, 27, 22, 23, 17, 14, 12],
                                                        "epochs": np.arange(2, 17, 2)},
                                "peanut20201104_.nwb": {"premaze_durations": [17, 15, 16, 17, 15, 13, 16, 18, 13],
                                                        "epochs": [2, 4, 6, 8, 10, 11, 13, 15, 17]},
                                "peanut20201107_.nwb": {"premaze_durations": [16, 12, 15, 16, 16, 14, 11, 17],
                                                        "epochs": np.arange(2, 17, 2)},
                                "peanut20201108_.nwb": {"premaze_durations": [25, 12, 14, 13, 14, 20, 12, 16],
                                                        "epochs": np.arange(2, 17, 2)},
                                "peanut20201109_.nwb": {"premaze_durations": [15, 14, 17, 15, 14, 14, 15, 16, 13],
                                                        "epochs": [2, 3, 5, 7, 9, 11, 13, 15, 17]},
                                "peanut20201110_.nwb": {"premaze_durations": [15, 13, 15, 12, 17, 14, 18, 20],
                                                        "epochs": np.arange(2, 17, 2)},
                                "peanut20201114_.nwb": {"premaze_durations": [15, 16, 25, 15, 22, 24, 13, 13],
                                                        "epochs": np.arange(2, 17, 2)},
                                "peanut20201117_.nwb": {"premaze_durations": [13, 16, 15, 12, 15, 15, 12, 13],
                                                        "epochs": [2, 4, 6, 9, 11, 13, 15, 17]},
                                "peanut20201119_.nwb": {"premaze_durations": [12, 14, 14, 12, 15, 11, 13, 14],
                                                        "epochs": [2, 5, 7, 9, 11, 13, 15, 17]},

                                # J16
                                "J1620210529_.nwb": {"premaze_durations": [18, 16, 17, 17, 15, 15, 14, 20],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210531_.nwb": {"premaze_durations": [15, 19, 13, 14, 15, 22, 15, 21],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210601_.nwb": {"premaze_durations": [12, 17],
                                                     "epochs": np.arange(2, 5, 2)},
                                "J1620210602_.nwb": {"premaze_durations": [15, 15, 15, 18, 18, 13, 13, 16],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210603_.nwb": {"premaze_durations": [13, 13, 14, 14],
                                                     "epochs": np.arange(2, 9, 2)},
                                "J1620210604_.nwb": {"premaze_durations": [12, 17, 14, 12, 13, 13, 19, 14],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210605_.nwb": {"premaze_durations": [15, 11, 19, 16, 16, 13, 13, 19],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210606_.nwb": {"premaze_durations": [16, 14, 13, 12, 11, 12, 20, 13],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210607_.nwb": {"premaze_durations": [18, 17, 19, 16, 18, 13, 14, 15],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210608_.nwb": {"premaze_durations": [14, 17, 13, 20, 14, 19, 14, 17],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210609_.nwb": {"premaze_durations": [17, 19, 16, 17, 15, 19, 18, 17],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210611_.nwb": {"premaze_durations": [20, 16, 15, 13, 17, 11, 16, 14],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210612_.nwb": {"premaze_durations": [12, 17, 14, 12, 17, 17, 12, 14],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210613_.nwb": {"premaze_durations": [15, 12, 19, 17, 10, 15, 16, 11],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210614_.nwb": {"premaze_durations": [14, 12, 12, 18, 18, 15, 10, 19],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210615_.nwb": {"premaze_durations": [14, 13, 16, 12, 15, 14, 16, 17],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210616_.nwb": {"premaze_durations": [19, 15, 14, 15, 18, 16, 14, 11],
                                                     "epochs": np.arange(2, 17, 2)},
                                "J1620210617_.nwb": {"premaze_durations": [18, 20, 11, 20, 18, 15, 13, 16],
                                                     "epochs": np.arange(2, 17, 2)},

                                # Fig
                                "fig20211101_.nwb": {"premaze_durations": [11, 12, 18, 11, 11, 11, 10, 10],
                                                     "epochs": np.arange(2, 17, 2)},
                                "fig20211103_.nwb": {"premaze_durations": [11, 21, 15, 18, 12, 15, 19],
                                                     "epochs": np.arange(2, 17, 2)},
                                "fig20211104_.nwb": {"premaze_durations": [13, 12, 13, 11, 11, 11, 10, 16, 11],
                                                     "epochs": [2, 5, 7, 9, 10, 12, 14, 16, 18]},
                                "fig20211108_.nwb": {"premaze_durations": [11, 13, 11, 12, 16, 12, 12, 10],
                                                     "epochs": np.arange(2, 17, 2)},
                                "fig20211109_.nwb": {"premaze_durations": [10, 19, 11, 9, 10, 15, 10, 15],
                                                     "epochs": np.arange(2, 17, 2)},
                                "fig20211110_.nwb": {"premaze_durations": [10, 11, 10, 17, 10, 11, 10, 9],
                                                     "epochs": np.arange(2, 17, 2)},
                                "fig20211111_.nwb": {"premaze_durations": [9, 20, 11, 9, 16, 9, 14, 10],
                                                     "epochs": np.arange(2, 17, 2)},
                                "fig20211115_.nwb": {"premaze_durations": [10, 14, 12, 9, 15, 11, 10, 14],
                                                     "epochs": np.arange(2, 17, 2)},

                                # Mango
                                "mango20211129_.nwb": {"premaze_durations": [12, 11, 12, 12, 11, 11, 11],
                                                       "epochs": np.arange(2, 15, 2)},
                                "mango20211130_.nwb": {"premaze_durations": [14, 16, 17, 12, 12, 11, 12, 11],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211201_.nwb": {"premaze_durations": [14, 15, 13],
                                                       "epochs": np.arange(2, 7, 2)},
                                "mango20211202_.nwb": {"premaze_durations": [14, 14, 11, 12, 12, 10, 15, 14],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211203_.nwb": {"premaze_durations": [11, 11, 11, 11, 10],
                                                       "epochs": np.arange(2, 11, 2)},
                                "mango20211204_.nwb": {"premaze_durations": [15, 15, 15, 13, 13, 16, 11, 12],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211205_.nwb": {"premaze_durations": [17, 13, 16, 14, 18, 14, 12, 17],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211206_.nwb": {"premaze_durations": [11, 0, 18, 12, 12, 10, 13, 16, 13],
                                                       "epochs": [2, 3, 5, 7, 9, 11, 13, 15, 17]},
                                "mango20211207_.nwb": {"premaze_durations": [12, 11, 11, 17, 10, 11, 12, 11],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211208_.nwb": {"premaze_durations": [16, 11, 12, 10, 16, 15, 14, 11],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211209_.nwb": {"premaze_durations": [12, 11, 15, 12, 10, 11, 12, 11],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211213_.nwb": {"premaze_durations": [8, 9, 12, 9, 10, 10, 10, 11],
                                                       "epochs": np.arange(2, 17, 2)},
                                "mango20211214_.nwb": {"premaze_durations": [10, 12, 11, 11, 9, 9, 9, 16],
                                                       "epochs": np.arange(2, 17, 2)},

                                # June
                                "june20220412_.nwb": {"premaze_durations": [15, 14, 12, 15, 15, 13, 17, 13],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220413_.nwb": {"premaze_durations": [15, 12, 13, 14, 11, 10, 11, 13],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220414_.nwb": {"premaze_durations": [12, 10],
                                                      "epochs": np.arange(2, 5, 2)},
                                "june20220415_.nwb": {"premaze_durations": [11, 15, 18, 10, 10, 11, 14, 13],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220416_.nwb": {"premaze_durations": [17, 12, 12, 12, 11, 9, 12, 13],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220417_.nwb": {"premaze_durations": [11, 11, 12, 12, 11, 12, 12],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220418_.nwb": {"premaze_durations": [10, 11, 9, 10, 9, 10, 11, 12],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220419_.nwb": {"premaze_durations": [10, 11, 15, 11, 12, 9, 10, 14],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220420_.nwb": {"premaze_durations": [12, 9, 17, 19, 11, 12, 15, 10],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220421_.nwb": {"premaze_durations": [11, 12, 12, 18, 15, 9, 9, 16],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220422_.nwb": {"premaze_durations": [9, 13, 11, 14, 16, 14, 11, 14],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220423_.nwb": {"premaze_durations": [11, 11, 14, 18, 17, 14, 12, 13],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220424_.nwb": {"premaze_durations": [16, 12, 14, 12, 11, 13, 11, 15],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220425_.nwb": {"premaze_durations": [10, 15, 15, 14, 11, 19, 12, 12],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220426_.nwb": {"premaze_durations": [11, 12, 15, 10, 9, 12, 10, 10],
                                                      "epochs": np.arange(2, 17, 2)},
                                "june20220427_.nwb": {"premaze_durations": [14, 12, 10, 12, 14, 11, 20, 12],
                                                      "epochs": np.arange(2, 17, 2)},
                                 }

        # Insert into table
        for nwb_file_name, dict_temp in premaze_duration_dict.items():  # for each nwb file
            for premaze_duration, epoch in zip(dict_temp["premaze_durations"], dict_temp["epochs"]):  # for each epoch
                # Check that task epoch entry exists, since we require this
                task_epoch_entries = (TaskEpoch & {"nwb_file_name": nwb_file_name,
                                                   "epoch": epoch}).fetch()
                if len(task_epoch_entries) != 1:
                    print(f"Should have found one entry in TaskEpoch for {nwb_file_name} epoch {epoch}, but found "
                          f"{len(task_epoch_entries)}. Cannot add premaze durations for this nwbf and epoch.")
                else:
                    self.insert1({"nwb_file_name": nwb_file_name, "epoch": epoch, "premaze_duration": premaze_duration},
                                 skip_duplicates=True)


def populate_jguidera_premaze_durations(key=None, tolerate_error=False):
    schema_name = "jguidera_premaze_durations"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_premaze_durations():
    schema.drop()
