import copy

import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd
import matplotlib.pyplot as plt

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, CovDigmethBase, ComputedBase, \
    CovariateDigParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_delay_duration
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellTrials, DioWellDDTrialsParams, DioWellDDTrials
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPool, ResTimeBinsPoolSel
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.digitize_helpers import digitize_indexed_variable
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
    def make_bin_edges(cls, **kwargs):
        if "bin_width" not in kwargs:
            raise Exception(f"bin_width must be passed")
        return make_bin_edges(cls.get_range(), kwargs["bin_width"])

    def make(self, key):

        # Get trial times
        valid_intervals = getattr((DioWellTrials & key), self._valid_interval_fn_name())()

        # Get time bin centers
        time_vector = (ResTimeBinsPool & key).fetch1_dataframe().time_bin_centers.values

        # Initialize vector for relative time
        rel_time_vec = np.asarray([np.nan] * len(time_vector))

        for valid_interval in valid_intervals:
            idxs, times_in_intervals = event_times_in_intervals(time_vector, [valid_interval])
            rel_time_vec[idxs] = (times_in_intervals - valid_interval[0]) / (
                unpack_single_element(np.diff(valid_interval)))

        df = pd.DataFrame.from_dict(
            {"time": time_vector, self._covariate_name(): rel_time_vec}).set_index("time")

        # Add information about what well animal is traveling to/from, and path animal is on or came from
        # (collectively "path_name")
        # ...Add param indicating no start/end shift to dd trials to key
        dd_trials_meta_param_name = DioWellDDTrialsParams().meta_param_name()
        key.update({dd_trials_meta_param_name: DioWellDDTrialsParams().lookup_no_shift_param_name()})
        column_names = ["trial_start_well_names", "trial_end_well_names", "path_names"]
        trials_info_df = (DioWellDDTrials & key).label_time_vector(time_vector, column_names, add_dd_text=True)
        # ...Flag above column names for replacing None with "none"
        replace_none_col_names = [x for x in trials_info_df.columns if any([y in x for y in column_names])]
        df = pd.concat((df, trials_info_df), axis=1)

        # Store in table
        main_key = copy.deepcopy(key)
        main_key.pop(dd_trials_meta_param_name)
        insert_analysis_table_entry(
            self, [df], main_key, reset_index=True, replace_none_col_names=replace_none_col_names)

        # Insert into part table
        self.DioWellDDTrials.insert1(key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
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

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves upstream dependence on DioWellDDTrials
        -> RelTimeWell
        -> DioWellDDTrials
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

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves upstream dependence on DioWellDDTrials
        -> RelTimeDelay
        -> DioWellDDTrials
        """

    @staticmethod
    def _covariate_name():
        return "relative_time_in_delay"

    @staticmethod
    def _valid_interval_fn_name():
        return "delay_times"

    def fetch1_dataframe_exclude(self, exclusion_params=None, object_id_name=None, restore_empty_nwb_object=True,
                                 df_index_name="time"):

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

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves upstream dependence on DioWellDDTrials
        -> RelTimeWellPostDelay
        -> DioWellDDTrials
        """

    @staticmethod
    def _covariate_name():
        return "relative_time_at_well_post_delay"

    @staticmethod
    def _valid_interval_fn_name():
        return "well_post_delay_times"

    def fetch1_dataframe_exclude(self, exclusion_params=None, object_id_name=None, restore_empty_nwb_object=True,
                                 df_index_name="time"):

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

    def digitized_rel_time_well_post_delay(self, bin_width=None, bin_edges=None, verbose=False):
        # Check inputs
        check_one_none([bin_width, bin_edges], ["bin_width", "bin_edges"])

        # Make bin edges if not passed
        if bin_edges is None:
            bin_edges = RelTimeWellPostDelayDigParams().make_bin_edges(key={"bin_width": bin_width})

        # Get relative time at well in post delay period
        rel_time_well_post_delay_df = self.fetch1_dataframe()

        # Digitize relative time
        rel_time_well_post_delay_df["digitized_relative_time_at_well_post_delay"] = digitize_indexed_variable(
            indexed_variable=rel_time_well_post_delay_df["relative_time_at_well_post_delay"], bin_edges=bin_edges,
            verbose=verbose)

        return rel_time_well_post_delay_df

@schema
class RelTimeWellPostDelayDigParams(CovariateDigParamsBase):
    definition = """
    # Parameters for RelTimeWellPostDelayDig
    rel_time_well_post_delay_dig_param_name : varchar(40)
    ---
    rel_time_well_post_delay_bin_width : float
    """

    def _default_params(self):
        return [
            # for population firing rate vectors distance analysis
            [.1]]

    def make_bin_edges(self, **kwargs):
        if "bin_width" not in kwargs:
            raise Exception(f"bin_width must be passed")
        return RelTimeWellPostDelay.make_bin_edges(**kwargs)

    def get_valid_bin_nums(self, **kwargs):
        return np.arange(1, self.get_num_bin_edges(**kwargs))


@schema
class RelTimeWellPostDelayDigSel(SelBase):
    definition = """
    # Selection from upstream tables for RelTimeWellPostDelayDig
    -> RelTimeWellPostDelay
    -> RelTimeWellPostDelayDigParams
    """


@schema
class RelTimeWellPostDelayDig(ComputedBase):
    definition = """
    # Digitized relative time during post delay period
    -> RelTimeWellPostDelayDigSel
    ---
    -> nd.common.AnalysisNwbfile
    rel_time_well_post_delay_dig_object_id : varchar(100)
    """

    def make(self, key):

        # Get relative time bin width
        rel_time_well_post_delay_bin_width = (RelTimeWellPostDelayDigParams & key).fetch1(
            "rel_time_well_post_delay_bin_width")

        # Replace relative time column values with digitized version
        dig_df = (RelTimeWellPostDelay & key).digitized_rel_time_well_post_delay(
            key={"bin_width": rel_time_well_post_delay_bin_width})
        dig_df.drop(columns="relative_time_at_well_post_delay", inplace=True)

        # Insert table entry
        insert_analysis_table_entry(self, [dig_df], key, reset_index=True)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def plot_results(self):

        # Get data
        df = self.fetch1_dataframe()

        # Initialize plot
        fig, ax = plt.subplots()

        # Plot data from current table in cyan, using very thick line
        current_table_color = "cyan"
        current_table_linewidth = 10
        ax.plot(df.digitized_relative_time_at_well_post_delay, current_table_color)
        ax.plot(df.dd_path_names, current_table_color, linewidth=current_table_linewidth)

        # Plot data from DD trials table
        (DioWellDDTrials & self.fetch1("KEY")).plot_results(ax=ax)


def populate_jguidera_relative_time_at_well(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_relative_time_at_well"
    upstream_schema_populate_fn_list = []
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_relative_time_at_well():
    schema.drop()
