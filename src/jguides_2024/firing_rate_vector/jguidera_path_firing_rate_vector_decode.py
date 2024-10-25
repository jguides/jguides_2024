import datajoint as dj
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import plot_junction_fractions
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_decode_table_base import \
    DecodeCovFRVecBase, DecodeCovFRVecParamsBase, DecodeCovFRVecSelBase, DecodeCovFRVecSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecAveSelBase, PopulationAnalysisParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import delete_, get_table_secondary_key_names
from src.jguides_2024.firing_rate_vector.jguidera_path_firing_rate_vector import PathFRVec, PathFRVecParams
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.position_and_maze.jguidera_ppt import PptParams
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptDigParams
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, BrainRegionUnitsCohortType
from src.jguides_2024.utils.plot_helpers import plot_spanning_line
from src.jguides_2024.utils.vector_helpers import unpack_single_element

schema = dj.schema("jguidera_path_firing_rate_vector_decode")


# These imports are called with eval or used in table definitions (do not remove):
EpochsDescription
PptParams
PptDigParams
BrainRegionUnits
ResEpochSpikesSmParams
PathFRVec
nd
BrainRegionUnitsCohortType
PathFRVecParams
BrainRegionCohort
CurationSet


@schema
class DecodePathFRVecParams(DecodeCovFRVecParamsBase):
# class DecodePathFRVecParams(dj.Manual):  # use when initially generating table; if not cannot update table later
    definition = """
    # Parameters for DecodePathFRVec
    decode_path_fr_vec_param_name : varchar(100)
    ---
    decode_path_fr_vec_params : blob
    """

    def get_valid_bin_nums(self, **kwargs):
        key = kwargs.pop("key")
        return (PptDigParams & key).get_valid_bin_nums()

    def insert_defaults(self, **kwargs):

        # Decode path progression on correct trials where rat stayed for full delay period at destination well
        # Decode training on one path, and testing on another path
        decode_path_fr_vec_param_name = "LDA_path_progression_loocv_correct_stay_trials"
        decode_path_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "path_progression",
            "path_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv"}
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)

        # Decode path progression on correct trials where rat stayed for full delay period at destination well
        # AND IGNORE PATH NAME
        decode_path_fr_vec_param_name = "LDA_path_progression_collapse_path_loocv_correct_stay_trial"
        decode_path_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "path_progression_collapse_path",
            "path_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv"}
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)

        # Decode outbound path on correct trials where rat stayed for full delay period at destination well
        decode_path_fr_vec_param_name = "LDA_outbound_path_loocv_correct_stay_trials"
        decode_path_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "outbound_path",
            "path_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv",
            "cross_validation_always": True,
        }
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)

        # Decode path identity on correct trials where rat stayed for full delay period at destination well
        decode_path_fr_vec_param_name = "LDA_path_loocv_correct_stay_trials"
        decode_path_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "path",
            "path_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv",
            "cross_validation_always": True,
        }
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)

        # Decode upcoming well identity on correct trials where rat stayed for full delay period at destination well
        decode_path_fr_vec_param_name = "LDA_destination_well_loocv_correct_stay_trials"
        decode_path_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "destination_well",
            "path_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv",
            "cross_validation_always": True,
        }
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)

        # Decode correct vs. incorrect on trials where rat stayed for full delay period at destination well
        decode_path_fr_vec_param_name = "SVC_correct_incorrect_loocv_stay_trials"
        decode_path_fr_vec_params = {
            "classifier_name": "SVC", "decode_var": "correct_incorrect",
            "path_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv",
            "cross_validation_always": True,
        }
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)

        # Decode correct vs. incorrect on all trials
        decode_path_fr_vec_param_name = "SVC_correct_incorrect_loocv"
        decode_path_fr_vec_params = {
            "classifier_name": "SVC", "decode_var": "correct_incorrect",
            "path_fr_vec_param_name": "correct_incorrect_trials", "cross_validation_method": "loocv",
            "cross_validation_always": True,
        }
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)

        # Decode correct vs. incorrect on all previous trials
        decode_path_fr_vec_param_name = "SVC_previous_correct_incorrect_loocv"
        decode_path_fr_vec_params = {
            "classifier_name": "SVC", "decode_var": "previous_correct_incorrect",
            "path_fr_vec_param_name": "prev_correct_incorrect_trials", "cross_validation_method": "loocv",
            "cross_validation_always": True,
        }
        self.insert1(
            {"decode_path_fr_vec_param_name": decode_path_fr_vec_param_name,
             "decode_path_fr_vec_params": decode_path_fr_vec_params}, skip_duplicates=True)


@schema
class DecodePathFRVecSel(CovariateFRVecAveSelBase):
# class DecodePathFRVecSel(dj.Manual):  # use when initially generating table; if not cannot update table later

    definition = """
    # Selection from upstream tables for DecodePathFRVec 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> DecodePathFRVecParams
    zscore_fr : bool
    path_fr_vec_param_name : varchar(40)
    """

    # Override parent class method to further restrict potentials keys and limit path_fr_vec_param_name
    # to those defined in params for a given decode_path_fr_vec_param_name

    def _get_potential_keys(self, key_filter=None, populate_tables=False):

        if key_filter is None:
            key_filter = dict()

        key_filter.update({"ppt_dig_param_name": "0.0625", "res_epoch_spikes_sm_param_name": "0.1", "zscore_fr": 0})

        potential_keys = []
        for param_name, params in (DecodePathFRVecParams & key_filter).fetch():
            key_filter.update({"decode_path_fr_vec_param_name": param_name,
                          "path_fr_vec_param_name": params["path_fr_vec_param_name"]})
            potential_keys += super()._get_potential_keys(key_filter, populate_tables)

        return potential_keys

    def delete_(self, key, safemode=True):
        delete_(self, [DecodePathFRVec], key, safemode)

    def _get_cov_fr_vec_param_names(self):
        return ["correct_incorrect_trials", "correct_incorrect_stay_trials", "prev_correct_incorrect_trials"]

    @staticmethod
    def _fr_vec_table():
        return PathFRVec


@schema
class DecodePathFRVec(DecodeCovFRVecBase):
    definition = """
    # Decode covariate along paths
    -> DecodePathFRVecSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> DecodePathFRVec
        -> PathFRVec
        """

    @staticmethod
    def _fr_vec_table():
        return PathFRVec


"""
Notes on DecodePathFRVecSumm table setup:
- We want to combine entries across DecodePathFRVec, across nwb_file_names, epochs_description, 
and brain_region. For this reason, we want DecodePathFRVecSummSel to have all primary keys of DecodePathFRVec
except for nwb_file_name, epochs_description, brain_region, brain_region_units_param_name, and 
curation_name. 
  To specify the nwb_file_names and corresponding epochs_descriptions we want to combine across, we use recording_set.
  To specify the brain regions we want to combine across, we use brain_region_cohort. 
  To specify curation_name, we use curation_set_name.
  To specify brain region unit information, we use BrainRegionUnitsCohortType
- We include BrainRegionUnitsCohortType in DecodePathFRVecSummParams so that we can stay within the
limit on number of primary keys
"""


class DecodeCovFRVecSummSecKeyParamsBase(PopulationAnalysisParamsBase):

    def get_param_name(self, **kwargs):
        return f"{kwargs['boot_set_name']}^{kwargs['brain_region_units_cohort_type']}"

    def insert_defaults(self, **kwargs):

        # Get names of brain_region_units_cohort_type
        brain_region_units_cohort_types = self._default_brain_region_units_cohort_types()

        # Get names for bootstrap param sets
        boot_set_names = self._boot_set_names()

        for boot_set_name in boot_set_names:

            for brain_region_units_cohort_type in brain_region_units_cohort_types:

                param_name = self.get_param_name(**{
                    "boot_set_name": boot_set_name, "brain_region_units_cohort_type": brain_region_units_cohort_type})

                decode_path_fr_vec_summ_params = {
                    "boot_set_name": boot_set_name, "brain_region_units_cohort_type": brain_region_units_cohort_type}

                # Insert into table
                meta_param_name = self.meta_param_name()
                meta_params_name = unpack_single_element(get_table_secondary_key_names(self))

                key = {meta_param_name: param_name,
                              meta_params_name: decode_path_fr_vec_summ_params}

                # Only do insertion if no entries. Reason for including this condition here, even though datajoint
                # will automatically skip if main table populated: if there are cases where new entry conflicts
                # with existing entry in BRUCT even though no corresponding entry in main table, want to be able to
                # detect this. If just set skip_duplicates to True, would not detect.
                if len(self & {k: v for k, v in key.items() if k in self.primary_key}) == 0:

                    # Insert into main table
                    self.insert1(key, skip_duplicates=True)

                    # Insert into part table
                    self.BRUCT.insert1({
                        meta_param_name: param_name, "brain_region_units_cohort_type": brain_region_units_cohort_type})

    def _boot_set_names(self):
        return super()._boot_set_names() + self._valid_brain_region_diff_boot_set_names() \
               + ["relationship_div_median", "relationship_div_rat_cohort_median"]

    def get_params(self):
        return self.fetch1(unpack_single_element(get_table_secondary_key_names(self)))


@schema
class DecodePathFRVecSummParams(DecodeCovFRVecSummSecKeyParamsBase):
# class DecodePathFRVecSummParams(dj.Manual):  # use when initially generating table; if not cannot update table later
    definition = """
    # Parameters for DecodePathFRVecSumm
    decode_path_fr_vec_summ_param_name : varchar(200)
    ---
    decode_path_fr_vec_summ_params : blob
    """

    # Had to shorten name to comply with mysql requirements
    class BRUCT(dj.Part):
        definition = """
        # Achieves dependence on BrainRegionUnitsCohortType
        -> BrainRegionUnitsCohortType
        -> DecodePathFRVecSummParams
        """


@schema
class DecodePathFRVecSummSel(DecodeCovFRVecSelBase):
    definition = """
    # Selection from upstream tables for DecodePathFRVecSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> DecodePathFRVecParams
    zscore_fr : bool
    -> PathFRVecParams
    -> DecodePathFRVecSummParams
    ---
    upstream_keys : mediumblob
    -> nd.common.AnalysisNwbfile
    df_concat_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> DecodePathFRVecSummSel
        -> BrainRegionCohort
        -> CurationSet
        -> DecodePathFRVec
        """


@schema
class DecodePathFRVecSumm(DecodeCovFRVecSummBase):
# class DecodePathFRVecSumm(dj.Computed):  # use to initialize table
    definition = """
    # Summary of decodes of covariate along paths
    -> DecodePathFRVecSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    @staticmethod
    def _upstream_table():
        return DecodePathFRVec

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        # Return default params
        return params

    def _get_x_lims(self):
        return [0, 1]

    def _get_xticks(self):
        return self._get_x_lims()

    def _get_x_text(self):
        return "Path fraction"

    def extend_plot_results(self, **kwargs):

        super().extend_plot_results(**kwargs)

        if not kwargs["empty_plot"]:

            ax = kwargs["ax"]

            # Vertical lines to denote track segments
            plot_junction_fractions(ax)

            # Colored patches to denote task phase
            from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_task_period_color_map
            task_period_color_map = get_task_period_color_map()

            xlims = ax.get_xlim()
            x_start = xlims[0]
            x_extent = xlims[1] - xlims[0]
            ylims = ax.get_ylim()
            y_start = ylims[1]
            y_extent = (ylims[1] - ylims[0])*.1
            color = task_period_color_map["path traversal"]
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((x_start, y_start), x_extent, y_extent, color=color))

            # ...Update y lims
            ax.set_ylim([ylims[0], ylims[1] + y_extent])

            # Line at chance
            # ...get decode variable
            key = self.fetch1("KEY")
            upstream_params = (self._upstream_table()()._get_params_table() & key).get_params()
            decode_var = upstream_params["decode_var"]
            boot_set_name = (self._get_params_table() & key).get_params()["boot_set_name"]
            if decode_var in ["path_progression"] and boot_set_name in ["default", "default_rat_cohort"]:
                valid_bin_nums = (PptDigParams & self.fetch1()).get_valid_bin_nums()

                y_val = 1/len(valid_bin_nums)
                plot_spanning_line(ax.get_xlim(), y_val, ax, "x", color="black")

