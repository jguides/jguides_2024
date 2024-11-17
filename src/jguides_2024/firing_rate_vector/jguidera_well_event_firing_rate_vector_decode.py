import datajoint as dj
import spyglass as nd
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import plot_task_phases
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_decode_table_base import \
    DecodeCovFRVecParamsBase, DecodeCovFRVecBase, DecodeCovFRVecSelBase, DecodeCovFRVecSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecAveSelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import delete_, get_table_secondary_key_names
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_delay_duration
from src.jguides_2024.firing_rate_vector.jguidera_path_firing_rate_vector_decode import DecodeCovFRVecSummSecKeyParamsBase
from src.jguides_2024.firing_rate_vector.jguidera_well_event_firing_rate_vector import TimeRelWAFRVec, \
    TimeRelWAFRVecParams
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, BrainRegionUnitsCohortType
from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWADigParams, \
    TimeRelWADigSingleAxisParams
from src.jguides_2024.utils.dict_helpers import check_same_values_at_shared_keys
from src.jguides_2024.utils.plot_helpers import plot_spanning_line
from src.jguides_2024.utils.vector_helpers import unpack_single_element, unpack_single_vector

schema = dj.schema("jguidera_well_event_firing_rate_vector_decode")


# These imports are called with eval or used in table definitions (do not remove):
EpochsDescription
TimeRelWADigParams
TimeRelWADigSingleAxisParams
BrainRegionUnits
ResEpochSpikesSmParams
TimeRelWAFRVec
nd
BrainRegionUnitsCohortType
TimeRelWAFRVecParams
BrainRegionCohort
CurationSet


@schema
class DecodeTimeRelWAFRVecParams(DecodeCovFRVecParamsBase):
# class DecodeTimeRelWAFRVecParams(dj.Manual):  # use when initially generating table; if not cannot update table later

    definition = """
    # Parameters for DecodeTimeRelWAFRVec
    decode_time_rel_wa_fr_vec_param_name : varchar(100)
    ---
    decode_time_rel_wa_fr_vec_params : blob
    """

    def get_valid_bin_nums(self, **kwargs):
        key = kwargs["key"]
        params = (self & key).get_params()
        check_same_values_at_shared_keys((params, key))
        return TimeRelWADigSingleAxisParams().get_valid_bin_nums({**key, **params})

    def insert_defaults(self, **kwargs):

        # Convention for naming decode_time_rel_wa_fr_vec_param_name: string together:
        # classifier_name, decode_var, cross_validation_method, time_rel_wa_fr_vec_param_name

        time_rel_wa_dig_param_name = TimeRelWADigParams().lookup_param_name([.25])

        time_period_params = [("_expand", [-1, 3])]  # ("", [0, 2]),
        for time_period_name, time_period_range in time_period_params:

            pdaw_text = ""
            if time_period_range[1] > get_delay_duration():
                pdaw_text = "_pdaw"

            time_rel_wa_dig_single_axis_param_name = TimeRelWADigSingleAxisParams().lookup_param_name(time_period_range)
            time_bin_params = {"time_rel_wa_dig_param_name": time_rel_wa_dig_param_name,
                "time_rel_wa_dig_single_axis_param_name": time_rel_wa_dig_single_axis_param_name}

            # Decode time in delay on correct trials where rat stayed for full delay period
            decode_time_rel_wa_fr_vec_param_name = f"LDA_time_in_delay_loocv_correct_stay_trials{pdaw_text}{time_period_name}"
            decode_time_rel_wa_fr_vec_params = {**time_bin_params, **{
                "classifier_name": "linear_discriminant_analysis", "decode_var": "time_in_delay",
                "time_rel_wa_fr_vec_param_name": f"correct_incorrect_stay_trials{pdaw_text}",
                "cross_validation_method": "loocv"}}
            self.insert1(
                {"decode_time_rel_wa_fr_vec_param_name": decode_time_rel_wa_fr_vec_param_name,
                 "decode_time_rel_wa_fr_vec_params": decode_time_rel_wa_fr_vec_params}, skip_duplicates=True)

            # Decode previous path identity on trials where rat stayed for full delay period at destination well
            decode_time_rel_wa_fr_vec_param_name = f"LDA_path_loocv_correct_stay_trials{pdaw_text}{time_period_name}"
            decode_time_rel_wa_fr_vec_params = {**time_bin_params, **{
                "classifier_name": "linear_discriminant_analysis", "decode_var": "path",
                "time_rel_wa_fr_vec_param_name": f"correct_incorrect_stay_trials{pdaw_text}",
                "cross_validation_method": "loocv",
                "cross_validation_always": True,
            }}
            self.insert1(
                {"decode_time_rel_wa_fr_vec_param_name": decode_time_rel_wa_fr_vec_param_name,
                 "decode_time_rel_wa_fr_vec_params": decode_time_rel_wa_fr_vec_params}, skip_duplicates=True)

            # Decode current well identity on trials where rat stayed for full delay period at destination well
            decode_time_rel_wa_fr_vec_param_name = f"LDA_well_loocv_correct_stay_trials{pdaw_text}{time_period_name}"
            decode_time_rel_wa_fr_vec_params = {**time_bin_params, **{
                "classifier_name": "linear_discriminant_analysis", "decode_var": "well",
                "time_rel_wa_fr_vec_param_name": f"correct_incorrect_stay_trials{pdaw_text}",
                "cross_validation_method": "loocv",
                "cross_validation_always": True,
            }}
            self.insert1(
                {"decode_time_rel_wa_fr_vec_param_name": decode_time_rel_wa_fr_vec_param_name,
                 "decode_time_rel_wa_fr_vec_params": decode_time_rel_wa_fr_vec_params}, skip_duplicates=True)

            # Decode correct vs. incorrect on trials where rat stayed for full delay period at destination well
            decode_time_rel_wa_fr_vec_param_name = f"SVC_correct_incorrect_loocv_stay_trials{pdaw_text}{time_period_name}"
            decode_time_rel_wa_fr_vec_params = {**time_bin_params, **{
                "classifier_name": "SVC", "decode_var": "correct_incorrect",
                "time_rel_wa_fr_vec_param_name": f"correct_incorrect_stay_trials{pdaw_text}",
                "cross_validation_method": "loocv",
                "cross_validation_always": True,
            }}
            self.insert1(
                {"decode_time_rel_wa_fr_vec_param_name": decode_time_rel_wa_fr_vec_param_name,
                 "decode_time_rel_wa_fr_vec_params": decode_time_rel_wa_fr_vec_params}, skip_duplicates=True)

            # Decode outbound path on correct trials where rat stayed for full delay period at destination well
            decode_time_rel_wa_fr_vec_param_name = f"LDA_outbound_path_loocv_correct_stay_trials{pdaw_text}{time_period_name}"
            decode_time_rel_wa_fr_vec_params = {**time_bin_params, **{
                "classifier_name": "linear_discriminant_analysis", "decode_var": "outbound_path",
                "time_rel_wa_fr_vec_param_name": f"correct_incorrect_stay_trials{pdaw_text}",
                "cross_validation_method": "loocv",
                "cross_validation_always": True,
            }}
            self.insert1(
                {"decode_time_rel_wa_fr_vec_param_name": decode_time_rel_wa_fr_vec_param_name,
                 "decode_time_rel_wa_fr_vec_params": decode_time_rel_wa_fr_vec_params}, skip_duplicates=True)

        # Decode stay vs. leave when rat at well
        decode_time_rel_wa_fr_vec_param_name = "SVC_stay_leave_loocv_stay_leave_trials_pre_departure"
        time_period_range = [0, 2]
        time_rel_wa_dig_single_axis_param_name = TimeRelWADigSingleAxisParams().lookup_param_name(time_period_range)
        time_bin_params = {"time_rel_wa_dig_param_name": time_rel_wa_dig_param_name,
                           "time_rel_wa_dig_single_axis_param_name": time_rel_wa_dig_single_axis_param_name}
        decode_time_rel_wa_fr_vec_params = {**time_bin_params, **{
            "classifier_name": "SVC", "decode_var": "stay_leave",
            "time_rel_wa_fr_vec_param_name": "stay_leave_trials_pre_departure",
            "cross_validation_method": "loocv",
            "cross_validation_always": True,
        }}
        self.insert1(
            {"decode_time_rel_wa_fr_vec_param_name": decode_time_rel_wa_fr_vec_param_name,
             "decode_time_rel_wa_fr_vec_params": decode_time_rel_wa_fr_vec_params}, skip_duplicates=True)


@schema
class DecodeTimeRelWAFRVecSel(CovariateFRVecAveSelBase):
# class DecodeTimeRelWAFRVecSel(dj.Manual):  # use when initially generating table; if not cannot update table later

    definition = """
    # Selection from upstream tables for DecodeTimeRelWAFRVec 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> TimeRelWADigParams
    -> TimeRelWADigSingleAxisParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> DecodeTimeRelWAFRVecParams
    zscore_fr : bool
    time_rel_wa_fr_vec_param_name : varchar(40)
    """

    # Override parent class method to further restrict potentials keys and limit time_rel_wa_fr_vec_param_name
    # to those defined in params for a given decode_time_rel_wa_fr_vec_param_name
    def _get_potential_keys(self, key_filter=None, populate_tables=False):

        if key_filter is None:
            key_filter = dict()

        key_filter.update({
            "time_rel_wa_dig_param_name": TimeRelWADigParams().lookup_param_name([.25]),
            "res_epoch_spikes_sm_param_name": "0.1", "zscore_fr": 0})

        potential_keys = []
        for param_name, params in (DecodeTimeRelWAFRVecParams & key_filter).fetch():
            key_filter.update({
                "decode_time_rel_wa_fr_vec_param_name": param_name,
                "time_rel_wa_dig_param_name": params["time_rel_wa_dig_param_name"],
                "time_rel_wa_dig_single_axis_param_name": params["time_rel_wa_dig_single_axis_param_name"],
                "time_rel_wa_fr_vec_param_name": params["time_rel_wa_fr_vec_param_name"]})
            potential_keys += super()._get_potential_keys(key_filter, populate_tables)

        return potential_keys

    @staticmethod
    def _fr_vec_table():
        return TimeRelWAFRVec

    def delete_(self, key, safemode=True):
        delete_(self, [DecodeTimeRelWAFRVec], key, safemode)

    def _get_cov_fr_vec_param_names(self):
        return [
            "correct_incorrect_stay_trials", "correct_incorrect_stay_trials_pdaw", "stay_leave_trials_pre_departure"]


@schema
class DecodeTimeRelWAFRVec(DecodeCovFRVecBase):
# class DecodeTimeRelWAFRVec(ComputedBase):  # use when initially generating table; if not cannot update table later
    definition = """
    # Decode covariate in delay period bins
    -> DecodeTimeRelWAFRVecSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> DecodeTimeRelWAFRVec
        -> TimeRelWAFRVec
        """

    @staticmethod
    def _fr_vec_table():
        return TimeRelWAFRVec


"""
Notes on DecodeTimeRelWAFRVecSumm table setup:
- We want to combine entries across DecodeTimeRelWAFRVec, across nwb_file_names, epochs_description, 
and brain_region. For this reason, we want DecodeTimeRelWAFRVecSummSel to have all primary keys of DecodeTimeRelWAFRVec
except for nwb_file_name, epochs_description, brain_region, brain_region_units_param_name, and 
curation_name. 
  To specify the nwb_file_names and corresponding epochs_descriptions we want to combine across, we use recording_set.
  To specify the brain regions we want to combine across, we use brain_region_cohort. 
  To specify curation_name, we use curation_set_name.
  To specify brain region unit information, we use BrainRegionUnitsCohortType
- We include BrainRegionUnitsCohortType in DecodeTimeRelWAFRVecSummParams so that we can stay within the
limit on number of primary keys
"""


@schema
class DecodeTimeRelWAFRVecSummParams(DecodeCovFRVecSummSecKeyParamsBase):
# class DecodeTimeRelWAFRVecSummParams(dj.Manual):  # use when initially generating table; if not cannot update table later

    definition = """
    # Parameters for DecodeTimeRelWAFRVecSumm
    decode_time_rel_wa_fr_vec_summ_param_name : varchar(200)
    ---
    decode_time_rel_wa_fr_vec_summ_params : blob
    """

    # Had to shorten name to comply with mysql requirements
    class BRUCT(dj.Part):
        definition = """
        # Achieves dependence on BrainRegionUnitsCohortType
        -> BrainRegionUnitsCohortType
        -> DecodeTimeRelWAFRVecSummParams
        """

    def insert_defaults(self, **kwargs):

        # Get names of brain_region_units_cohort_type
        brain_region_units_cohort_types = self._default_brain_region_units_cohort_types()

        # Get names for bootstrap param sets
        boot_set_names = self._boot_set_names()

        for boot_set_name in boot_set_names:

            for brain_region_units_cohort_type in brain_region_units_cohort_types:

                param_name = f"{boot_set_name}^{brain_region_units_cohort_type}"

                decode_time_rel_wa_fr_vec_summ_params = {
                    "boot_set_name": boot_set_name, "brain_region_units_cohort_type": brain_region_units_cohort_type}

                # Insert into table
                self.insert1({"decode_time_rel_wa_fr_vec_summ_param_name": param_name,
                              "decode_time_rel_wa_fr_vec_summ_params": decode_time_rel_wa_fr_vec_summ_params},
                             skip_duplicates=True)

    def _boot_set_names(self):
        return super()._boot_set_names() + self._valid_brain_region_diff_boot_set_names()

    def get_params(self):
        return self.fetch1("decode_time_rel_wa_fr_vec_summ_params")


@schema
class DecodeTimeRelWAFRVecSummSel(DecodeCovFRVecSelBase):
    definition = """
    # Selection from upstream tables for DecodeTimeRelWAFRVecSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    -> TimeRelWADigParams
    -> TimeRelWADigSingleAxisParams
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> DecodeTimeRelWAFRVecParams
    zscore_fr : bool
    -> TimeRelWAFRVecParams
    -> DecodeTimeRelWAFRVecSummParams
    ---
    upstream_keys : mediumblob
    -> nd.common.AnalysisNwbfile
    df_concat_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> DecodeTimeRelWAFRVecSummSel
        -> BrainRegionCohort
        -> CurationSet
        -> DecodeTimeRelWAFRVec
        """


@schema
class DecodeTimeRelWAFRVecSumm(DecodeCovFRVecSummBase):
# class DecodeTimeRelWAFRVecSumm(dj.Computed):  # use to initialize table
    definition = """
    # Summary of decodes of covariate in delay period bins
    -> DecodeTimeRelWAFRVecSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    @staticmethod
    def _upstream_table():
        return DecodeTimeRelWAFRVec

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "correct_incorrect_stay_trials"

    # Override in children classes where relationship exists
    def _get_relationship_meta_name(self):

        params_table = self._get_params_table()()
        boot_set_name = (params_table & self.fetch1("KEY")).get_params()["boot_set_name"]
        if boot_set_name in params_table._valid_relationship_div_boot_set_names() + \
            params_table._valid_stay_leave_diff_boot_set_names() + \
            params_table._valid_stay_leave_diff_brain_region_diff_boot_set_names() + \
            params_table._valid_same_different_outbound_path_correct_diff_boot_set_names() + \
            params_table._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names():
            return self._get_joint_column_name("relationship")

        return "relationship"

    def _get_x_lims(self):
        return (TimeRelWADigSingleAxisParams & self.fetch1("KEY")).fetch1("rel_time_start", "rel_time_end")

    def _get_xticks(self):
        x1, x2 = self._get_x_lims()
        return np.arange(x1, x2 + 1, 1)

    def _get_x_text(self):
        return "Time from well arrival (s)"

    def _get_vals_index_name(self):
        return "x_val"

    def extend_plot_results(self, **kwargs):

        super().extend_plot_results(**kwargs)

        if not kwargs["empty_plot"]:

            ax = kwargs["ax"]

            # Colored patches to denote task phase
            plot_task_phases(ax, "time_in_delay")

            # Line at chance
            # ...get decode variable
            key = self.fetch1("KEY")
            upstream_params = (self._upstream_table()()._get_params_table() & key).get_params()
            decode_var = upstream_params["decode_var"]
            boot_set_name = (self._get_params_table() & key).get_params()["boot_set_name"]
            if decode_var in ["time_in_delay"] and boot_set_name in ["default", "default_rat_cohort"]:
                y_val = 1/len(np.unique(self.fetch1_dataframes().metric_df.x_val))
                plot_spanning_line(ax.get_xlim(), y_val, ax, "x", color="black")
