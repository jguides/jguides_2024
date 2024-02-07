import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, CovDigmethBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_delay_duration
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellTrials
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPool, ResTimeBinsPoolSel
from src.jguides_2024.utils.make_bins import make_bin_edges
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals, event_times_in_intervals_bool
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import unpack_single_element

# Needed for table definitions:
DioWellTrials
ResTimeBinsPool
nd

schema = dj.schema("jguidera_relative_time_at_well")


class RelTimeSelBase(SelBase):
    def _get_potential_keys(self, key_filter=None):
        res_time_bins_pool_param_names = [
            ResTimeBinsPoolSel().lookup_param_name_from_shorthand(shorthand) for shorthand in [
                "epoch_100ms",  # for embedding
            ]]
        return [k for k in super()._get_potential_keys() if k["res_time_bins_pool_param_name"] in
                res_time_bins_pool_param_names]


@schema
class RelTimeWellSel(RelTimeSelBase):
    definition = """
    # Selection from upstream tables for RelTimeWell
    -> DioWellTrials
    -> ResTimeBinsPool
    """

    def insert1(self, key, **kwargs):
        if key["res_time_bins_pool_param_name"] != ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms"):
            raise Exception
        super().insert1(key, **kwargs)


class RelTimeBase(CovDigmethBase):

    @staticmethod
    def _valid_interval_fn_name():
        raise Exception(f"This method must be overwritten in child class")

    @staticmethod
    def get_range():
        return np.asarray([0, 1])

    @classmethod
    def make_bin_edges(cls, bin_width):
        return make_bin_edges(cls.get_range(), bin_width)

    def make(self, key):
        valid_intervals = getattr((DioWellTrials & key), self._valid_interval_fn_name())()
        time_bins_df = (ResTimeBinsPool & key).fetch1_dataframe()
        rel_time_vec = np.asarray([np.nan] * len(time_bins_df))  # initialize
        for valid_interval in valid_intervals:
            idxs, times_in_intervals = event_times_in_intervals(time_bins_df.time_bin_centers, [valid_interval])
            rel_time_vec[idxs] = (times_in_intervals - valid_interval[0]) / (
                unpack_single_element(np.diff(valid_interval)))
        rel_time_df = pd.DataFrame.from_dict(
            {"relative_time": time_bins_df.time_bin_centers, self._covariate_name(): rel_time_vec})

        # Insert into table
        insert_analysis_table_entry(self, [rel_time_df], key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="relative_time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)


@schema
class RelTimeWell(RelTimeBase):
    definition = """
    # Relative time at well
    -> RelTimeWellSel
    ---
    -> nd.common.AnalysisNwbfile
    rel_time_well_object_id : varchar(40)
    """

    @staticmethod
    def _covariate_name():
        return "relative_time_at_well"

    @staticmethod
    def _valid_interval_fn_name():
        return "well_a_d_times"


@schema
class RelTimeDelaySel(RelTimeSelBase):
    definition = """
    # Selection from upstream tables for RelTimeDelay
    -> DioWellTrials
    -> ResTimeBinsPool
    """


@schema
class RelTimeDelay(RelTimeBase):
    definition = """
    # Relative time in 2s delay
    -> RelTimeDelaySel
    ---
    -> nd.common.AnalysisNwbfile
    rel_time_delay_object_id : varchar(40)
    """

    @staticmethod
    def _covariate_name():
        return "relative_time_in_delay"

    @staticmethod
    def _valid_interval_fn_name():
        return "delay_times"

    def fetch1_dataframe_exclude(self, exclusion_params=None, object_id_name=None, restore_empty_nwb_object=True,
                                 df_index_name="relative_time"):

        # Get df
        df = self.fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

        # Exclude as indicated
        if exclusion_params is not None:
            # Check exclusion type passed if want to exclude
            if "exclusion_type" not in exclusion_params:
                raise Exception(f"exclusion_type must be in exclusion_params")
            # Check exclusion type valid
            exclusion_type = exclusion_params["exclusion_type"]
            check_membership([exclusion_type], [
                "at_well", "not_at_well", "long_well_duration", "short_well_duration", "stay_trial", "leave_trial",
                None])

            dio_well_trials_subset = DioWellTrials & self.fetch1("KEY")
            well_intervals = np.asarray(dio_well_trials_subset.well_times())  # times at well
            invalid_bool = [False]*len(df)  # default is to not exclude

            # Case 1: exclude times based on whether rat at well or not
            if exclusion_type in ["at_well", "not_at_well"]:
                at_well_bool = event_times_in_intervals_bool(np.asarray(df.index), well_intervals)  # rat at well
                # exclude times when rat at well
                if exclusion_type == "at_well":
                    invalid_bool = at_well_bool
                # exclude times when rat not at well
                elif exclusion_type == "not_at_well":
                    invalid_bool = np.invert(at_well_bool)  # rat not at well

            # Case 2: exclude delay periods corresponding to well periods based on duration
            elif exclusion_type in ["long_well_duration", "short_well_duration"]:
                well_durations = np.concatenate(np.diff(well_intervals))
                if exclusion_type == "long_well_duration":
                    if "max_duration" not in exclusion_params:
                        raise Exception(f"max_duration must be passed if exclusion_type is long_well_duration")
                    valid_bool = well_durations <= exclusion_params["max_duration"]
                elif exclusion_type == "short_well_duration":
                    if "min_duration" not in exclusion_params:
                        raise Exception(f"min_duration must be passed if exclusion type is short_well_duration")
                    valid_bool = well_durations >= exclusion_params["min_duration"]
                valid_intervals = np.asarray(
                    [well_intervals[valid_bool, 0], well_intervals[valid_bool, 0] + get_delay_duration()]).T
                invalid_bool = np.invert(event_times_in_intervals_bool(np.asarray(df.index), valid_intervals))

            # Case 3: exclude stay or leave trials
            elif exclusion_type in ["stay_trial", "leave_trial"]:
                exclude_intervals_bool = dio_well_trials_subset.get_stay_leave_trial_bool(exclusion_type)
                exclude_intervals = np.asarray(
                    [well_intervals[exclude_intervals_bool, 0],
                     well_intervals[exclude_intervals_bool, 0] + get_delay_duration()]).T
                invalid_bool = event_times_in_intervals_bool(np.asarray(df.index), exclude_intervals)

            # apply the exclusion
            df[self._covariate_name()][invalid_bool] = np.nan

        # Return df with exclusion applied
        return df


@schema
class RelTimeWellPostDelaySel(RelTimeSelBase):
    definition = """
    # Selection from upstream tables for RelTimeWellPostDelay
    -> DioWellTrials
    -> ResTimeBinsPool
    """


@schema
class RelTimeWellPostDelay(RelTimeBase):
    definition = """
    # Relative time at well following 2s delay
    -> RelTimeWellPostDelaySel
    ---
    -> nd.common.AnalysisNwbfile
    rel_time_well_post_delay_object_id : varchar(40)
    """

    @staticmethod
    def _covariate_name():
        return "relative_time_at_well_post_delay"

    @staticmethod
    def _valid_interval_fn_name():
        return "well_post_delay_times"

    def fetch1_dataframe_exclude(self, exclusion_params=None, object_id_name=None, restore_empty_nwb_object=True,
                                 df_index_name="relative_time"):

        # Get df
        df = super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

        raise_error = False
        if exclusion_params is not None:
            if len(exclusion_params) == 1:
                if list(exclusion_params.keys())[0] != "highlight_trial_num":
                    raise_error = True
            elif len(exclusion_params) > 1:
                raise_error = True
        if raise_error:
            raise Exception(f"exclusion_params not accounted for in RelTimeWellPostDelay currently")

        return df


def populate_jguidera_relative_time_at_well(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_relative_time_at_well"
    upstream_schema_populate_fn_list = []
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_relative_time_at_well():
    schema.drop()
