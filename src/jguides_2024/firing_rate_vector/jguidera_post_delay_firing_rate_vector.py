from collections import namedtuple

import datajoint as dj
import numpy as np
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecSelBase, CovariateFRVecBase, CovariateAveFRVecParamsBase, CovariateFRVecTrialAveBase, \
    CovariateFRVecAveSelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import drop_
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_names
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, RunEpoch
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsParams, BrainRegionUnits
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDDTrials, DioWellTrials
from src.jguides_2024.time_and_trials.jguidera_relative_time_at_well import RelTimeWellPostDelayDig, \
    RelTimeWellPostDelayDigParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel

# Needed for table definitions:
from src.jguides_2024.utils.df_helpers import check_same_index
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
from src.jguides_2024.utils.state_evolution_estimation import AverageVectorDuringLabeledProgression

nd
RelTimeWellPostDelayDig
FRVec
DioWellDDTrials
BrainRegionUnits
ResEpochSpikesSmParams

schema = dj.schema("jguidera_post_delay_firing_rate_vector")


@schema
class RelPostDelFRVecParams(SecKeyParamsBase):
    definition = """
    # Parameters for RelPostDelFRVec
    rel_post_del_fr_vec_param_name : varchar(40)
    ---
    labels_description = "none" : varchar(40)  # indicate how to alter labels from their original form
    """

    def _default_params(self):
        return [[x] for x in [
            "even_odd_correct_incorrect_stay_trials"]]

    def drop_(self):
        drop_([RelPostDelFRVecSel(), self])  # RelPostDelFRVec(),


@schema
class RelPostDelFRVecSel(CovariateFRVecSelBase):
    definition = """
    # Selection from upstream tables for RelPostDelFRVec
    -> RelTimeWellPostDelayDig
    -> FRVec
    -> RelPostDelFRVecParams
    """

    # Override parent class method so further restrict potential keys
    def _get_potential_keys(self, key_filter=None, verbose=True):

        if verbose:
            print(f"getting potential keys for RelPostDelFRVecSel...")

        min_epoch_mean_firing_rate = .1
        brain_region_cohort_name = "all_targeted"
        curation_set_name = "runs_analysis_v1"
        primary_features = {
            "zscore_fr": 0,
            "res_time_bins_pool_param_name": ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms"),
            "rel_time_well_post_delay_dig_param_name": "0.1",
            "rel_post_del_fr_vec_param_name": "even_odd_correct_incorrect_stay_trials"}
        unit_params = {
            "unit_subset_type": "rand_target_region", "unit_subset_size": 50}
        unit_subset_iterations = np.arange(0, 10)
        kernel_sds = [.1]

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

                    if verbose:
                        print(f"on unit subset cases...")
                    for unit_subset_iteration in unit_subset_iterations:
                        unit_params.update({"unit_subset_iteration": unit_subset_iteration})
                        brain_region_units_param_name = BrainRegionUnitsParams().lookup_single_epoch_param_name(
                            nwb_file_name, epoch, min_epoch_mean_firing_rate, **unit_params)
                        key.update({
                            "epoch": epoch, "brain_region_units_param_name": brain_region_units_param_name})
                        for kernel_sd in kernel_sds:
                            k = {"res_epoch_spikes_sm_param_name": ResEpochSpikesSmParams().lookup_param_name(
                                [kernel_sd])}
                            keys.append({**primary_features, **key, **k})

        print(f"\nDefined {len(keys)} potential keys, now restricting to those with matches in upstream tables...\n")

        # Calls parent class method
        table_intersection_keys = super()._get_potential_keys()
        print(f"...found {len(table_intersection_keys)} upstream table keys, now checking which potential keys "
              f"are in these...")
        potential_keys = [x for x in keys if x in table_intersection_keys]

        if len(potential_keys) == 0:
            raise Exception

        print(f"\nReturning {len(potential_keys)} potential keys...")

        return potential_keys


@schema
class RelPostDelFRVec(CovariateFRVecBase):
    definition = """
    # Firing rate vectors averaged in contiguous stretches in relative time in post delay period bins
    -> RelPostDelFRVecSel
    ---
    unit_names : blob
    -> nd.common.AnalysisNwbfile
    vector_df_object_id : varchar(40)
    ave_vector_df_object_id : varchar(40)
    diff_vector_df_object_id : varchar(40)
    ave_diff_vector_df_object_id : varchar(40)
    """

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellDDTrials
        -> RelPostDelFRVec
        -> DioWellDDTrials
        """

    @staticmethod
    def alter_input_labels_stay_leave(labels, key):
        # Add to labels whether time is "associated" with "stay trial" (rat stayed at well for full delay period)
        # or "leave trial" (rat did not stay at well for full delay period).

        # IMPORTANT NOTE: all trials for which digitized values are within valid range
        # should be "stay" trials, since to get to the post delay period, rat has to
        # wait full delay period. However, this function still serves a purpose: to add "stay" to label
        # so that can match up with labels from path and delay task phases where trials can be "stay" or "leave".

        # Also track which samples correspond to previous stay or leave trial, and exclude those
        # corresponding to neither.

        in_intervals_bool_map = dict()  # store booleans indicating whether samples in stay or leave trials
        table_subset = (DioWellTrials & key)

        # DioWellTrials trials are defined from one well arrival to the next. We want whether or rat stayed
        # at well for full delay duration on current trial.

        # Restrict to post delay intervals (these all follow full delay period)
        trial_intervals = np.asarray(table_subset.well_post_delay_times())

        # Add text to labels with times within these intervals
        # ...Get boolean indicating whether samples in intervals above
        in_intervals_bool = event_times_in_intervals_bool(labels.index, trial_intervals)
        # ...Add text to labels in interval
        label_name = MazePathWell().stay_leave_trial_text("stay")
        labels.loc[in_intervals_bool] = [
            MazePathWell().get_stay_leave_trial_path_name(x, label_name)
            for x in labels[in_intervals_bool].values]
        # Store boolean indicating whether samples in intervals above, so can ultimately exclude samples
        # not associated with stay or leave trials
        in_intervals_bool_map[label_name] = in_intervals_bool

        # Return updated labels and boolean indicating labels that are associated with stay or leave trials
        return labels, in_intervals_bool_map

    def get_inputs(self, key, verbose=False, ax=None):

        # Get firing rate vectors
        fr_vec_df = (FRVec & key).firing_rate_vector_across_sort_groups(populate_tables=False)

        # Get digitized relative time in post delay period
        df = (RelTimeWellPostDelayDig & key).fetch1_dataframe()
        dig_rel_time = df.digitized_relative_time_at_well_post_delay

        # Get labels (departure to departure trials path names)
        labels = df["dd_path_names"]

        # Check same index across data sources
        check_same_index([fr_vec_df, dig_rel_time, labels])

        # Restrict to times when covariate finite and labels not "none"
        valid_bool = np.logical_and(np.isfinite(dig_rel_time), labels != "none")

        # Update quantities to reflect
        dig_rel_time = dig_rel_time[valid_bool]
        labels = labels[valid_bool]
        fr_vec_df = fr_vec_df[valid_bool]

        # Plot labels prior to altering (if indicated)
        ax = self._plot_labels("pre", labels, verbose, ax)

        # Alter labels as indicated
        labels_description = (RelPostDelFRVecParams & key).fetch1("labels_description")

        if labels_description == "even_odd_correct_incorrect_stay_trials":

            # Add text to denote whether "stay" or "leave" trial
            labels, in_intervals_bool_map = self.alter_input_labels_stay_leave(labels, key)

            # Add text to denote whether correct/incorrect trial in a given context (e.g. left to center path)
            # note that the line below returns only labels with correct/incorrect
            labels, label_bool = self.alter_input_labels_correct_incorrect(labels, key)

            # Add text to denote whether even/odd trial in a given context (e.g. left to center path) if indicated
            if labels_description == "even_odd_correct_incorrect_stay_trials":
                labels = self.alter_input_labels_even_odd(labels)

            # Keep only stay trials
            stay_bool = in_intervals_bool_map["stay_trial"]

            # Update quantities
            labels = labels[stay_bool[label_bool]]
            valid_bool = np.logical_and(label_bool, stay_bool)
            dig_rel_time = dig_rel_time[valid_bool]
            fr_vec_df = fr_vec_df[valid_bool]

        elif labels_description == "none":
            pass

        else:
            raise Exception

        # Plot labels again, now that have altered (if indicated)
        self._plot_labels("post", labels, verbose, ax)

        return namedtuple("Inputs", "x labels df unit_names")(
            dig_rel_time, labels, fr_vec_df, np.asarray(fr_vec_df.columns))

    def get_valid_covariate_bin_nums(self, key):
        return (RelTimeWellPostDelayDigParams & key).get_valid_bin_nums()

    def get_bin_centers_map(self):
        key = self.fetch1("KEY")
        x = (RelTimeWellPostDelayDigParams & key).get_valid_bin_nums()
        bin_centers = (self._fr_vec_table() & key).get_bin_centers()

        return AverageVectorDuringLabeledProgression.get_bin_centers_map(x, bin_centers)

    def get_bin_centers(self, key=None):

        # Get key if not passed
        if key is None:
            key = self.fetch1("KEY")

        return (RelTimeWellPostDelayDigParams & key).get_bin_centers()


# Overrides methods in CovariateFRVecAveBase in a manner specific to time relative to well arrival covariate
class RelPostDelFRVecAveBase:

    @staticmethod
    def _fr_vec_table():
        return RelPostDelFRVec


class RelPostDelFRVecAveSelBase(CovariateFRVecAveSelBase):

    @staticmethod
    def _fr_vec_table():
        return RelPostDelFRVec


"""
Notes on setup of RelPostDelAveFRVec tables:
 - We combine entries from RelPostDelFRVec across epochs holding everything else constant, so we want all primary
   keys of RelPostDelFRVec except epoch to be primary keys in selection table, and we want epoch_description to be
   primary key serving to describe epochs we combine across
"""


@schema
class RelPostDelAveFRVecParams(CovariateAveFRVecParamsBase):
    definition = """
    rel_post_del_ave_fr_vec_param_name : varchar(40)
    ---
    metric_name : varchar(40)
    vector_type : varchar(40)
    """


@schema
class RelPostDelAveFRVecSel(RelPostDelFRVecAveSelBase):
    definition = """
    # Selection from upstream tables for RelPostDelAveFRVec 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> RelTimeWellPostDelayDigParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> RelPostDelAveFRVecParams
    zscore_fr : bool
    rel_post_del_fr_vec_param_name : varchar(40)
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

    def _get_cov_fr_vec_param_names(self):
        return ["even_odd_correct_incorrect_stay_trials"]


@schema
class RelPostDelAveFRVec(RelPostDelFRVecAveBase, CovariateFRVecTrialAveBase):
    definition = """
    # Comparison of average firing rate vectors across combinations of relative time bin, path identity, and epoch
    -> RelPostDelAveFRVecSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> RelPostDelAveFRVec
        -> RelPostDelFRVec
        """


def populate_jguidera_post_delay_firing_rate_vector(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_post_delay_firing_rate_vector"
    upstream_schema_populate_fn_list = []
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_post_delay_firing_rate_vector():
    schema.drop()

