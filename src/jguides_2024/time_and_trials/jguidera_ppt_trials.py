import datajoint as dj

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import (SecKeyParamsBase, SelBase, ComputedBase)
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import fetch1_dataframe_from_table_entry, \
    insert1_print
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.position_and_maze.jguidera_maze import get_path_segment_fractions
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt, populate_jguidera_ppt
from src.jguides_2024.utils.interval_helpers import check_intervals_list

schema = dj.schema("jguidera_ppt_trials")


@schema
class PptTrialsParams(SecKeyParamsBase):

    definition = """
    # Parameters for PptTrials
    ppt_trials_param_name : varchar(500)
    ---
    valid_ppt_intervals : blob  # list of valid ppt intervals
    """

    def insert_maze_segments_defaults(self):
        # Restrict to times within segments between maze junctions
        n_junctions = 2
        valid_ppt_intervals_map = {f"junc{n_junctions}seg{seg_num}": [x]
                                   for seg_num, x in enumerate(get_path_segment_fractions(n_junctions=n_junctions))}
        # Insert params
        for param_name, valid_ppt_intervals in valid_ppt_intervals_map.items():
            self.insert1({"ppt_trials_param_name": param_name,
                          "valid_ppt_intervals": valid_ppt_intervals})

    # Override parent class method to avoid long decimals in param name
    def insert_defaults(self, **kwargs):
        key_filter = kwargs.pop("key_filter", None)
        # Include all ppt between reward wells
        self.insert1({"ppt_trials_param_name": "all_path",
                      "valid_ppt_intervals": [[0, 1]]})

    # Require valid ppt intervals be well defined
    def insert1(self, key, **kwargs):
        check_intervals_list(key["valid_ppt_intervals"], require_monotonic_increasing=True)
        super().insert1(key, **kwargs)


@schema
class PptTrialsSel(SelBase):

    definition = """
    # Selection from upstream tables for PptTrials
    -> Ppt 
    -> PptTrialsParams
    """


@schema
class PptTrials(ComputedBase):

    definition = """
    # Trials corresponding to spans of time within valid proportion path traversed ranges
    -> PptTrialsSel
    ---
    trial_start_times : blob
    trial_end_times : blob
    trial_start_epoch_trial_numbers : blob
    trial_path_names : blob
    """

    def make(self, key):
        # Get valid time intervals: spans of time when ppt within valid intervals
        valid_ppt_intervals = (PptTrialsParams & key).fetch1("valid_ppt_intervals")
        key.update((Ppt & key).get_ppt_trials(valid_ppt_intervals=valid_ppt_intervals))
        insert1_print(self, key)

    def fetch1_dataframe(self, strip_s=False, **kwargs):
        # kwargs input makes easy to use this method with fetch1_dataframe methods that take in different
        # inputs (e.g. to fetch dataframe from nwb file)
        return fetch1_dataframe_from_table_entry(self, strip_s=strip_s)

    def trial_intervals(self):
        trials_df = self.fetch1()
        return list(zip(trials_df["trial_start_times"], trials_df["trial_end_times"]))


def populate_jguidera_ppt_trials(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_ppt_trials"
    upstream_schema_populate_fn_list = [populate_jguidera_ppt]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_ppt_trials():
    schema.drop()
