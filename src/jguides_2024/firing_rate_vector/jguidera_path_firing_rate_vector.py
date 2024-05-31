import copy
from collections import namedtuple

import datajoint as dj
import numpy as np
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecSTAveParamsBase, \
    CovariateFRVecAveSelBase, CovariateFRVecTrialAveBase, CovariateFRVecSTAveBase, CovariateFRVecBase, \
    CovariateFRVecSelBase, CovariateFRVecAveSummSelBase, CovariateFRVecAveSummParamsBase, \
    CovariateAveFRVecParamsBase, CovariateFRVecSTAveSummBase, CovariateAveFRVecSummBase, PathFRVecSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import delete_, drop_
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_names
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec, \
    populate_jguidera_firing_rate_vector
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, RunEpoch, RecordingSet
from src.jguides_2024.position_and_maze.jguidera_ppt import PptParams
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptDig, populate_jguidera_ppt_interp, \
    PptDigParams
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, BrainRegionUnitsParams, \
    BrainRegionUnitsCohortType
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellTrials, DioWellDDTrials
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel
from src.jguides_2024.utils.dict_helpers import make_keys
from src.jguides_2024.utils.state_evolution_estimation import AverageVectorDuringLabeledProgression

# Needed for table definitions:
nd
PptParams
PptDigParams
BrainRegionUnits
ResEpochSpikesSmParams
EpochsDescription
RecordingSet
BrainRegionUnitsCohortType
DioWellTrials
DioWellDDTrials

schema = dj.schema("jguidera_path_firing_rate_vector")


@schema
class PathFRVecParams(SecKeyParamsBase):
    definition = """
    # Parameters for PathFRVec
    path_fr_vec_param_name : varchar(40)
    ---
    labels_description = "none" : varchar(40)  # indicate how to alter labels from their original form (e.g. split path label in even / odd trials on the path)
    """

    def _default_params(self):
        return [[x] for x in ["none", "even_odd_trials", "correct_incorrect_trials"]]

    # Drop dependent tables and avoid error related to attempt to
    # drop part table before main table
    def drop_(self):
        drop_([PathAveFRVecSel(), PathFRVecSTAveSel(), PathFRVecSel(), self])

    def delete_(self, key, safemode=True):
        # Delete from upstream tables
        delete_(self, [PathFRVecSTAveSummSel, PathFRVecSTAveSel, PathAveFRVecSummSel, PathAveFRVecSel], key, safemode)


@schema
class PathFRVecSel(CovariateFRVecSelBase):
    definition = """
    # Selection from upstream tables for PathFRVec
    -> PptDig
    -> FRVec
    -> PathFRVecParams
    """

    # Override parent class method so can restrict entries to downstream tables
    def _get_potential_keys(self, key_filter=None, verbose=True):
        # Approach: avoid a combinatorial explosion of entries

        if verbose:
            print(f"getting potential keys for PathFRVecSel...")

        # Define a first set of parameters, to be used with all units
        min_epoch_mean_firing_rate = .1
        primary_kernel_sds = [.1]
        primary_path_fr_vec_param_names = ["none"]  # pair all possible combinations with these
        primary_ppt_dig_param_name = PptDigParams().lookup_param_name([.05])
        curation_set_name = "runs_analysis_v1"
        brain_region_cohort_name = "all_targeted"
        primary_features = {
            "zscore_fr": 0, "ppt_param_name": "ppt_1", "ppt_dig_param_name": primary_ppt_dig_param_name,
            "res_time_bins_pool_param_name": ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms"),
        }
        all_features = {
            "res_epoch_spikes_sm_param_name": set(FRVec().fetch("res_epoch_spikes_sm_param_name")),
            "ppt_dig_param_name": [PptDigParams().lookup_param_name([x]) for x in [.05]],
            "zscore_fr": set(FRVec().fetch("zscore_fr")),
            "path_fr_vec_param_name": set(PathFRVecParams().fetch("path_fr_vec_param_name")),
        }

        # Define a second set of parameters, to be used with unit subset
        unit_params_2 = {
            "unit_subset_type": "rand_target_region", "unit_subset_size": 50}
        unit_subset_iterations = np.arange(0, 10)
        # ...we want to populate using the above across multiple path_fr_vec_param_names
        primary_features_2 = copy.deepcopy(primary_features)
        primary_path_fr_vec_param_names_2 = ["none", "even_odd_trials", "correct_incorrect_trials"]
        primary_kernel_sds_2 = [.1, .2]

        # Define nwb file names
        nwb_file_names = get_jguidera_nwbf_names()
        # Restrict to single nwb file name if passed in key_filter
        if "nwb_file_name" in key_filter:
            nwb_file_names = [key_filter["nwb_file_name"]]

        keys = []
        for nwb_file_name in nwb_file_names:
            if verbose:
                print(f"\non {nwb_file_name}...")
            key = {"nwb_file_name": nwb_file_name}
            # Get brain regions for this nwb file name
            brain_regions = (BrainRegionCohort & {
                "nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name}).fetch1(
                "brain_regions")
            for brain_region in brain_regions:
                if verbose:
                    print(f"\non {brain_region}...")
                key.update({"brain_region": brain_region})

                for epoch in (RunEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch"):
                    if verbose:
                        print(f"on epoch {epoch}...")

                    # Add curation name to key
                    epochs_description = EpochsDescription().get_single_run_description(nwb_file_name, epoch)
                    curation_name = (CurationSet & {
                        "nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name,
                        "curation_set_name": curation_set_name}).get_curation_name(brain_region, epochs_description)
                    key.update({"curation_name": curation_name})

                    # No unit subset (use make_keys function):
                    if verbose:
                        print(f"on no unit subset cases...")
                    brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(
                        nwb_file_name, epoch, min_epoch_mean_firing_rate)
                    key.update({
                        "epoch": epoch, "brain_region_units_param_name": brain_region_units_param_name})
                    for path_fr_vec_param_name in primary_path_fr_vec_param_names:
                        for kernel_sd in primary_kernel_sds:
                            k = {"res_epoch_spikes_sm_param_name": ResEpochSpikesSmParams().lookup_param_name(
                                [kernel_sd]), "path_fr_vec_param_name": path_fr_vec_param_name}
                            keys += make_keys({**primary_features, **key, **k}, all_features)

                    # Unit subset (do not use make_keys function):
                    if verbose:
                        print(f"on unit subset cases...")
                    for unit_subset_iteration in unit_subset_iterations:
                        unit_params_2.update({"unit_subset_iteration": unit_subset_iteration})
                        brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(
                            nwb_file_name, epoch, min_epoch_mean_firing_rate, **unit_params_2)
                        key.update({
                            "epoch": epoch, "brain_region_units_param_name": brain_region_units_param_name})
                        for path_fr_vec_param_name in primary_path_fr_vec_param_names_2:
                            for kernel_sd in primary_kernel_sds_2:
                                k = {"res_epoch_spikes_sm_param_name": ResEpochSpikesSmParams().lookup_param_name(
                                    [kernel_sd]), "path_fr_vec_param_name": path_fr_vec_param_name}
                                keys.append({**primary_features_2, **key, **k})

        table_intersection_keys = super()._get_potential_keys()
        return [x for x in keys if x in table_intersection_keys]

    # Drop dependent tables and avoid error related to attempt to
    # drop part table before main table
    def drop_(self):
        drop_([PathAveFRVecSel(), PathFRVecSTAveSel(), self])

    def delete_(self, key, safemode=True):
        delete_(self, [PathFRVec], key, safemode)


@schema
class PathFRVec(CovariateFRVecBase):
    definition = """
    # Firing rate vectors averaged in contiguous stretches in path bins
    -> PathFRVecSel
    ---
    unit_names : blob
    -> nd.common.AnalysisNwbfile
    vector_df_object_id : varchar(40)
    ave_vector_df_object_id : varchar(40)
    diff_vector_df_object_id : varchar(40)
    ave_diff_vector_df_object_id : varchar(40)
    """

    class DioWellTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellTrials
        -> PathFRVec
        -> DioWellTrials
        """

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellDDTrials
        -> PathFRVec
        -> DioWellDDTrials
        """

    def get_inputs(self, key, verbose=False, ax=None):

        # Get firing rate vectors
        fr_vec_df = (FRVec & key).firing_rate_vector_across_sort_groups(populate_tables=False)

        # Get digitized ppt
        ppt_dig_df = (PptDig & key).fetch1_dataframe()

        # Check index matches across data sources
        if not all(ppt_dig_df.index == fr_vec_df.index):
            raise Exception(f"Index from PptDig should match that from FRVec")

        # Restrict to times when ppt finite and label not "none"
        valid_bool = np.logical_and(
            np.isfinite(ppt_dig_df.ppt), ppt_dig_df.path_name != "none")
        ppt_dig_df = ppt_dig_df[valid_bool]
        fr_vec_df = fr_vec_df[valid_bool]

        # Get labels
        labels = ppt_dig_df.path_name

        # Plot labels prior to altering (if indicated)
        ax = self._plot_labels("pre", labels, verbose, ax)

        # Alter labels as indicated
        labels_description = (PathFRVecParams & key).fetch1("labels_description")

        if labels_description == "even_odd_trials":
            labels = self.alter_input_labels_even_odd(labels)

        elif labels_description == "correct_incorrect_trials":
            labels, valid_bool = self.alter_input_labels_correct_incorrect(labels, key)

            # Restrict other quantities to times of valid labels
            fr_vec_df = fr_vec_df[valid_bool]
            ppt_dig_df = ppt_dig_df[valid_bool]
        elif labels_description == "none":
            pass
        else:
            raise Exception

        # Plot labels again, now that have altered (if indicated)
        self._plot_labels("post", labels, verbose, ax)

        return namedtuple("Inputs", "x labels df unit_names")(
            ppt_dig_df.digitized_ppt, labels, fr_vec_df, np.asarray(fr_vec_df.columns))

    def get_bin_centers(self):
        return (PptDigParams & self.fetch1("KEY")).get_bin_centers()

    def delete_(self, key, safemode=True):
        delete_(self, [PathAveFRVecSel, PathFRVecSTAveSel], key, safemode)


# Note on setup of PathFRVecSTAve tables:
# - We combine entries from PathFRVec across epochs holding everything else constant, so we want all primary keys of
# PathFRVec except epoch to be primary keys in selection table, and we want epoch_description to be
# primary key serving to describe epochs we combine across
# - We pair entries above with each available path_fr_vec_st_ave_param_name from PathFRVecSTAveParams
@schema
class PathFRVecSTAveParams(CovariateFRVecSTAveParamsBase):
    definition = """
    # Parameters for PathFRVecSTAve
    path_fr_vec_st_ave_param_name : varchar(40)
    ---
    metric_name : varchar(40)
    vector_type : varchar(40)
    mask_duration : float  # seconds
    """


class PathFRVecAveSelBase(CovariateFRVecAveSelBase):
    @staticmethod
    def _fr_vec_table():
        return PathFRVec


@schema
class PathFRVecSTAveSel(PathFRVecAveSelBase):
    definition = """
    # Selection from upstream tables for PathFRVecSTAve 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> PathFRVecSTAveParams
    zscore_fr : bool
    path_fr_vec_param_name : varchar(40)
    """

    def delete_(self, key, safemode=True):
        delete_(self, [PathFRVecSTAve], key, safemode)

    def _get_cov_fr_vec_param_names(self):
        return ["none", "correct_incorrect_trials"]


# Overrides methods in CovariateFRVecAveBase in a manner specific to path covariate and invariant to trial averaged
# vs. single trial table
class PathFRVecAveBase:

    @staticmethod
    def get_valid_covariate_bin_nums(key):
        return (PptDigParams & key).get_valid_bin_nums()

    def get_bin_centers_map(self):
        key = self.fetch1("KEY")
        x = (PptDigParams & key).get_valid_bin_nums()
        bin_centers = (self._fr_vec_table() & key).get_bin_centers()
        return AverageVectorDuringLabeledProgression.get_bin_centers_map(x, bin_centers)

    @staticmethod
    def _fr_vec_table():
        return PathFRVec


@schema
class PathFRVecSTAve(PathFRVecAveBase, CovariateFRVecSTAveBase):
    definition = """
    # Single 'trial' comparison of firing rate vectors across combinations of path bin and path identity 
    -> PathFRVecSTAveSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> PathFRVecSTAve
        -> PathFRVec
        """

    def delete_(self, key, safemode=True):
        delete_(self, [PathFRVecSTAveSummSel], key, safemode)


"""
Notes on setup of PathAveFRVec tables:
- We combine entries from PathFRVec across epochs holding everything else constant, so we want all primary keys of
PathFRVec except epoch to be primary keys in selection table, and we want epoch_description to be primary key serving
to describe epochs we combine across
"""


@schema
class PathAveFRVecParams(CovariateAveFRVecParamsBase):
    definition = """
    path_ave_fr_vec_param_name : varchar(40)
    ---
    metric_name : varchar(40)
    vector_type : varchar(40)
    """


@schema
class PathAveFRVecSel(PathFRVecAveSelBase):
    definition = """
    # Selection from upstream tables for PathAveFRVec
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> PathAveFRVecParams
    zscore_fr : bool
    path_fr_vec_param_name : varchar(40)
    """

    # Override parent class method so can restrict res_time_bins_pool_param_name
    def insert_defaults(self, key_filter=None):
        # Define key filter if not passed
        if key_filter is None:
            key_filter = dict()

        for res_time_bins_pool_param_name in [ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms")]:
            key_filter.update({"res_time_bins_pool_param_name": res_time_bins_pool_param_name})
            keys = self._get_potential_keys(key_filter)
            for key in keys:
                self.insert1(key)

    def delete_(self, key, safemode=True):
        delete_(self, [PathAveFRVec], key, safemode)

    def _get_cov_fr_vec_param_names(self):
        return ["even_odd_trials"]


@schema
class PathAveFRVec(PathFRVecAveBase, CovariateFRVecTrialAveBase):
    definition = """
    # Comparison of average firing rate difference vectors across combinations of path bin, path identity, and epoch
    -> PathAveFRVecSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> PathAveFRVec
        -> PathFRVec
         """


"""
Notes on PathFRVecSTAveSumm table setup:
- We want to combine entries across PathFRVecSTAve, across nwb_file_names, epochs_description, 
and brain_region. For this reason, we want PathFRVecSTAveSel to have all primary keys of PathFRVecSTAve
except for nwb_file_name, epochs_description, brain_region, brain_region_units_param_name, and 
curation_name. 
  To specify the nwb_file_names and corresponding epochs_descriptions we want to combine across, we use recording_set.
  To specify the brain regions we want to combine across, we use brain_region_cohort. 
  To specify curation_name, we use curation_set_name.
  To specify brain region unit information, we use BrainRegionUnitsCohortType
- We include BrainRegionUnitsCohortType in PathFRVecSTAveSummParams so that we can stay within the
limit on number of primary keys
"""


@schema
class PathFRVecSTAveSummParams(CovariateFRVecAveSummParamsBase):
    definition = """
    # Parameters for PathFRVecSTAveSumm
    path_fr_vec_st_ave_summ_param_name : varchar(200)
    ---
    metric_processing_name : varchar(40)  # describes additional processing on metric
    label_name : varchar(40)
    boot_set_name : varchar(120)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    def _boot_set_names(self):
        return super()._boot_set_names() + self._valid_brain_region_diff_boot_set_names() + \
               self._valid_same_different_outbound_path_correct_diff_boot_set_names() + \
               self._valid_same_different_outbound_path_correct_diff_brain_region_diff_boot_set_names()


@schema
class PathFRVecSTAveSummSel(CovariateFRVecAveSummSelBase):
    definition = """
    # Selection from upstream tables for PathFRVecSTAveSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> PathFRVecSTAveParams
    zscore_fr : bool
    -> PathFRVecParams
    -> PathFRVecSTAveSummParams
    ---
    upstream_keys : mediumblob
    """

    def _default_cov_fr_vec_param_names(self):
        return ["none", "correct_incorrect_trials"]

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + [
            "brain_region_diff", "same_different_outbound_path_correct_diff",
            "same_different_outbound_path_correct_diff_brain_region_diff"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + [
            "brain_region_diff_rat_cohort",
            "same_different_outbound_path_correct_diff_rat_cohort",
            "same_different_outbound_path_correct_diff_brain_region_diff_rat_cohort"]

    def delete_(self, key, safemode=True):
        # If recording set name not in key but components that determine it are, then
        # find matching recording set names given the components, to avoid deleting irrelevant
        # entries
        key = copy.deepcopy(key)  # make copy of key to avoid changing outside function
        recording_set_names = RecordingSet().get_matching_recording_set_names(key)
        for recording_set_name in recording_set_names:
            key.update({"recording_set_name": recording_set_name})
            delete_(self, [PathFRVecSTAveSumm], key, safemode)


@schema
class PathFRVecSTAveSumm(CovariateFRVecSTAveSummBase, PathFRVecSummBase):
    definition = """
    # Summary of single 'trial' comparison of firing rate vectors
    -> PathFRVecSTAveSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves upstream dependence on upstream tables
        -> PathFRVecSTAveSumm
        -> BrainRegionCohort
        -> CurationSet
        -> PathFRVecSTAve
        """

    @staticmethod
    def _upstream_table():
        return PathFRVecSTAve

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "none"

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        params.update({"mask_duration": self._upstream_table()()._get_params_table()._default_mask_duration()})

        # Return default params
        return params


"""
Notes on PathAveFRVecSumm table setup: same reasoning as for PathFRVecSTAveSumm table setup
"""


@schema
class PathAveFRVecSummParams(CovariateFRVecAveSummParamsBase):
    definition = """
    # Parameters for PathAveFRVecSumm
    path_ave_fr_vec_summ_param_name : varchar(160)
    ---
    metric_processing_name : varchar(40)  # describes additional processing on metric
    label_name : varchar(40)
    boot_set_name : varchar(40)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    # Override parent class method so can add bootstrap param name that specifies ratio of same path and same turn
    # or different turn path relationship values
    def _boot_set_names(self):
        return super()._boot_set_names() + [
            "relationship_div_median", "relationship_div_rat_cohort_median"]


@schema
class PathAveFRVecSummSel(CovariateFRVecAveSummSelBase):
    definition = """
    # Selection from upstream tables for PathAveFRVecSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    -> PptParams
    -> PptDigParams
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> PathAveFRVecParams
    zscore_fr : bool
    -> PathFRVecParams
    -> PathAveFRVecSummParams
    ---
    upstream_keys : mediumblob
    """

    # Override parent class method so can include across rat cohort
    def _recording_set_name_types(self):
        return super()._recording_set_name_types() + ["Haight_rotation_rat_cohort"]

    def _default_cov_fr_vec_param_names(self):
        return ["even_odd_trials"]

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + ["relationship_div", "relationship_div_median"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + [
            "relationship_div_rat_cohort", "relationship_div_rat_cohort_median"]


@schema
class PathAveFRVecSumm(CovariateAveFRVecSummBase, PathFRVecSummBase):
    definition = """
    # Summary of single 'trial' comparison of firing rate vectors
    -> PathAveFRVecSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves upstream dependence on upstream tables
        -> PathAveFRVecSumm
        -> BrainRegionCohort
        -> CurationSet
        -> PathAveFRVec
        """

    @staticmethod
    def _upstream_table():
        return PathAveFRVec

    def _get_relationship_div_column_params(self, **kwargs):

        cov_fr_vec_param_name = kwargs[self._upstream_table()().get_cov_fr_vec_meta_param_name()]

        if cov_fr_vec_param_name == "even_odd_trials":
            return {
                "denominator_column_name": "same_path_even_odd_trials", "numerator_column_names": [
                    "same_turn_even_odd_trials", "different_turn_well_even_odd_trials"]}
        else:
            raise Exception(f"{cov_fr_vec_param_name} not accounted for")

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "even_odd_trials"

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)


def populate_jguidera_path_firing_rate_vector(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_path_firing_rate_vector"
    upstream_schema_populate_fn_list = [populate_jguidera_ppt_interp, populate_jguidera_firing_rate_vector]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_path_firing_rate_vector():
    schema.drop()
