import copy
from collections import namedtuple

import datajoint as dj
import numpy as np
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecBase, CovariateFRVecSTAveParamsBase, \
    CovariateFRVecAveSelBase, CovariateFRVecTrialAveBase, CovariateFRVecSTAveBase, CovariateFRVecSelBase, \
    CovariateFRVecAveSummSelBase, CovariateFRVecAveSummParamsBase, \
    CovariateAveFRVecParamsBase, CovariateFRVecSTAveSummBase, CovariateAveFRVecSummBase, TimeRelWAFRVecSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import delete_, drop_
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_names
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec, \
    populate_jguidera_firing_rate_vector
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, RunEpoch, RecordingSet
from src.jguides_2024.position_and_maze.jguidera_maze import MazePathWell
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, BrainRegionUnitsParams, \
    BrainRegionUnitsCohortType
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellTrials, DioWellDDTrials
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel
from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWADig, \
    populate_jguidera_time_relative_to_well_event, TimeRelWADigSingleAxis, TimeRelWADigSingleAxisParams, \
    TimeRelWADigParams
from src.jguides_2024.utils.df_helpers import check_same_index
from src.jguides_2024.utils.dict_helpers import make_keys
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
from src.jguides_2024.utils.state_evolution_estimation import AverageVectorDuringLabeledProgression

# Needed for table definitions:
nd
EpochsDescription
BrainRegionUnits
ResEpochSpikesSmParams
TimeRelWADigParams
RecordingSet
BrainRegionUnitsCohortType
DioWellDDTrials

schema = dj.schema("jguidera_well_event_firing_rate_vector")


"""
Notes on table setup for TimeRelWAFRVec:
1) Currently, x_pair_int in diff_vector_df in main table goes from -10 to -2 for the pre well arrival
period (because x_pair_int gets defined as the first element in an x_pair) and from 1 to 9 in the
post well arrival period.
"""


@schema
class TimeRelWAFRVecParams(SecKeyParamsBase):
    definition = """
    # Parameters for TimeRelWAFRVec
    time_rel_wa_fr_vec_param_name : varchar(40)
    ---
    labels_description = "none" : varchar(40)  # indicate how to alter labels from their original form
    """

    def _default_params(self):
        return [[x] for x in [
            "even_odd_stay_trials", "stay_leave_trials_pre_departure", "correct_incorrect_stay_trials"]]

    def drop_(self):
        drop_([TimeRelWAAveFRVecSel(), TimeRelWAFRVecSTAveSel(), TimeRelWAFRVecSel(), self])


@schema
class TimeRelWAFRVecSel(CovariateFRVecSelBase):
    definition = """
    # Selection from upstream tables for TimeRelWAFRVec
    -> FRVec
    -> TimeRelWADigSingleAxis
    -> TimeRelWAFRVecParams
    """

    # Override parent class method so further restrict potential keys
    def _get_potential_keys(self, key_filter=None, verbose=True):

        # Approach: avoid a combinatorial explosion of entries
        # With about 9000 entries in keys, takes a minute or so to run

        if verbose:
            print(f"getting potential keys for TimeRelWAFRVecSel...")

        # Define key filter if not passed
        if key_filter is None:
            key_filter = dict()

        # Define a first set of parameters, to be used with all units
        # get param names outside loops to save compute
        min_epoch_mean_firing_rate = .1
        primary_kernel_sds = [.1, .2]
        primary_res_epoch_spikes_sm_param_names = [
            ResEpochSpikesSmParams().lookup_param_name([kernel_sd]) for kernel_sd in primary_kernel_sds]
        primary_time_rel_wa_dig_single_axis_param_names = [
                TimeRelWADigSingleAxisParams().lookup_param_name(x)
                for x in [[0, 2]]]  # make all combinations with each of these
        brain_region_cohort_name = "all_targeted"
        curation_set_name = "runs_analysis_v1"
        primary_features = {
            "time_rel_wa_fr_vec_param_name": "none", "zscore_fr": 0,
            "time_rel_wa_dig_param_name": TimeRelWADigParams().lookup_param_name([.25]),
            "res_time_bins_pool_param_name": ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms")}
        all_features = {
            "res_epoch_spikes_sm_param_name": set(FRVec().fetch("res_epoch_spikes_sm_param_name")),
            "time_rel_wa_fr_vec_param_name": set(TimeRelWAFRVecParams().fetch(
                "time_rel_wa_fr_vec_param_name")),
            "zscore_fr": set(FRVec().fetch("zscore_fr"))}

        # Define a second set of parameters, to be used with unit subset
        unit_params_2 = {
            "unit_subset_type": "rand_target_region", "unit_subset_size": 50}
        unit_subset_iterations = np.arange(0, 10)
        # ...we want to populate using the above across multiple time_rel_wa_fr_vec_param_names
        primary_features_2 = copy.deepcopy(primary_features)
        primary_features_2.pop("time_rel_wa_fr_vec_param_name")
        primary_time_rel_wa_fr_vec_param_names_2 = set(TimeRelWAFRVecParams().fetch(
                "time_rel_wa_fr_vec_param_name"))
        primary_kernel_sds_2 = [.1, .2]
        primary_res_epoch_spikes_sm_param_names_2 = [
            ResEpochSpikesSmParams().lookup_param_name([kernel_sd]) for kernel_sd in primary_kernel_sds_2]
        primary_time_rel_wa_dig_single_axis_param_names_2 = [
                TimeRelWADigSingleAxisParams().lookup_param_name(x)
                for x in [[0, 2]]]  # make all combinations with each of these

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
                    for time_rel_wa_dig_single_axis_param_name in primary_time_rel_wa_dig_single_axis_param_names:
                        for res_epoch_spikes_sm_param_name in primary_res_epoch_spikes_sm_param_names:
                            k = {"res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name, "time_rel_wa_dig_single_axis_param_name":
                                time_rel_wa_dig_single_axis_param_name}
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
                        for time_rel_wa_dig_single_axis_param_name in primary_time_rel_wa_dig_single_axis_param_names_2:
                            for res_epoch_spikes_sm_param_name in primary_res_epoch_spikes_sm_param_names_2:
                                for time_rel_wa_fr_vec_param_name in primary_time_rel_wa_fr_vec_param_names_2:
                                    k = {"res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name,
                                         "time_rel_wa_dig_single_axis_param_name":
                                             time_rel_wa_dig_single_axis_param_name,
                                         "time_rel_wa_fr_vec_param_name": time_rel_wa_fr_vec_param_name}
                                    keys.append({**primary_features_2, **key, **k})

        table_intersection_keys = super()._get_potential_keys()

        return [x for x in keys if x in table_intersection_keys]

    def delete_(self, key, safemode=True):
        delete_(self, [TimeRelWAFRVec], key, safemode)

    # Drop dependent tables and avoid error related to attempt to drop p
    # art table before main table
    def drop_(self):
        drop_([TimeRelWAAveFRVecSel(), TimeRelWAFRVecSTAveSel(), TimeRelWAFRVec(), self])


@schema
class TimeRelWAFRVec(CovariateFRVecBase):
    definition = """
    # Firing rate vectors averaged in contiguous stretches in time bins aligned to well arrival 
    -> TimeRelWAFRVecSel
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
        -> TimeRelWAFRVec
        -> DioWellTrials
        """

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellDDTrials
        -> TimeRelWAFRVec
        -> DioWellDDTrials
        """

    @staticmethod
    def alter_input_labels_stay_leave(labels, key):
        # Add to labels whether time is "associated" with "stay trial" (rat stays at well for full delay period)
        # or "leave trial" (rat does not stay at well for full delay period). What is means to be "associated":
        # with a trial: time falls within the interval around well arrival defined by rel_time_start and
        # rel_time_end in corresponding entry in TimeRelWADigSingleAxisParams.

        # Get relative time around well arrival (our covariate)
        rel_time_start, rel_time_end = (TimeRelWADigSingleAxisParams & key).fetch1("rel_time_start", "rel_time_end")

        # Find well arrivals after which rat stayed full delay period (stay trials), and add to the time of
        # each the relative start and end times found above. Do the same for those well arrivals after which
        # rat did NOT stay full delay period (leave trials). In each of these cases, label times that
        # fall within these intervals as "stay" or "leave", respectively.
        # Also track which samples fall within a stay or leave interval, and exclude those falling within neither.
        in_intervals_bool_map = dict()  # store booleans indicating whether samples in stay or leave trials
        table_subset = (DioWellTrials & key)
        well_arrival_times = table_subset.fetch1("well_arrival_times")
        peri_wa_intervals = np.asarray([well_arrival_times + rel_time_start, well_arrival_times + rel_time_end]).T
        for label_name in MazePathWell().stay_leave_trial_text():  # for trial types
            # Get peri well arrival intervals corresponding to trial type (stay or leave)
            # ...Get boolean indicating which well arrivals correspond to stay and which to leave
            trial_bool = table_subset.get_stay_leave_trial_bool(label_name)
            trial_peri_wa_intervals = peri_wa_intervals[trial_bool, :]
            # Add text to labels with times within these intervals
            # ...Get boolean indicating whether samples in intervals above
            in_intervals_bool = event_times_in_intervals_bool(labels.index, trial_peri_wa_intervals)
            # ...Add text to labels in interval
            labels[in_intervals_bool] = [
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

        # Get time relative to well arrival (on a single axis where negative means before well arrival and
        # positive means after)
        time_rel_wa = (TimeRelWADigSingleAxis & key).fetch1_dataframe().time_rel_wa

        # Get average vector during time relative to well arrival bins
        labels = (TimeRelWADig & key).fetch1_dataframe()["dd_path_names"]  # departure to departure trials path names

        # Check same index across data sources
        check_same_index([fr_vec_df, time_rel_wa, labels])

        # Restrict to times when covariate finite and labels not "none"
        valid_bool = np.logical_and(np.isfinite(time_rel_wa), labels != "none")

        # Update quantities to reflect
        time_rel_wa = time_rel_wa[valid_bool]
        labels = labels[valid_bool]
        fr_vec_df = fr_vec_df[valid_bool]

        # Get labels
        labels_description = (TimeRelWAFRVecParams & key).fetch1("labels_description")

        # Plot labels prior to altering (if indicated)
        ax = self._plot_labels("pre", labels, verbose, ax)

        # Alter labels as indicated by labels_description
        # ...add even / odd trials on top of current labels
        if labels_description == "even_odd_trials":
            labels = self.alter_input_labels_even_odd(labels)

        # ...add whether stay or leave trial on top of current labels and optionally restrict to samples
        # where rat at well if also indicated by labels_description
        elif labels_description in ["stay_leave_trials", "stay_leave_trials_pre_departure"]:
            labels, in_intervals_bool_map = self.alter_input_labels_stay_leave(labels, key)

            # Apply additional exclusions: 1) samples not labeled as stay or leave, 2) samples where rat left well if
            # indicated

            # 1) Restrict to samples labeled as stay or leave
            in_intervals_bool_list = list(in_intervals_bool_map.values())
            valid_bool = np.logical_or(*in_intervals_bool_list)

            # 2) If indicated, further restrict to times before rat left well
            if labels_description == "stay_leave_trials_pre_departure":
                rel_time_start = (TimeRelWADigSingleAxisParams & key).fetch1("rel_time_start")
                well_arrival_times, well_departure_times = np.asarray((DioWellTrials & key).well_times()).T
                pre_departure_intervals = list(zip(well_arrival_times + rel_time_start, well_departure_times))
                valid_bool = np.logical_and(valid_bool, event_times_in_intervals_bool(
                    labels.index, pre_departure_intervals))

            # Update quantities
            time_rel_wa = time_rel_wa[valid_bool]
            labels = labels[valid_bool]
            fr_vec_df = fr_vec_df[valid_bool]

        elif labels_description == "even_odd_stay_trials":

            # Add text to denote whether "stay" or "leave" trial
            labels, in_intervals_bool_map = self.alter_input_labels_stay_leave(labels, key)

            # Add text to denote whether even/odd trial in a given context (e.g. left to center path)
            labels = self.alter_input_labels_even_odd(labels)

            # Keep only stay trials
            valid_bool = in_intervals_bool_map["stay_trial"]

            # Update quantities
            time_rel_wa = time_rel_wa[valid_bool]
            labels = labels[valid_bool]
            fr_vec_df = fr_vec_df[valid_bool]

        elif labels_description == "correct_incorrect_stay_trials":

            # Add text to denote whether "stay" or "leave" trial
            labels, in_intervals_bool_map = self.alter_input_labels_stay_leave(labels, key)

            # Add text to denote whether correct/incorrect trial in a given context (e.g. left to center path)
            # note that the line below returns only labels with correct/incorrect
            labels, label_bool = self.alter_input_labels_correct_incorrect(labels, key)

            # Keep only stay trials
            stay_bool = in_intervals_bool_map["stay_trial"]

            # Update quantities
            labels = labels[stay_bool[label_bool]]
            valid_bool = np.logical_and(label_bool, stay_bool)
            time_rel_wa = time_rel_wa[valid_bool]
            fr_vec_df = fr_vec_df[valid_bool]

        elif labels_description == "none":
            pass

        else:
            raise Exception(f"labels_description {labels_description} not accounted for in code")

        # Plot labels again, now that have altered (if indicated)
        self._plot_labels("post", labels, verbose, ax)

        return namedtuple("Inputs", "x labels df unit_names")(
            time_rel_wa, labels, fr_vec_df, np.asarray(fr_vec_df.columns))

    def get_bin_centers(self):
        key = self.fetch1("KEY")
        return TimeRelWADigSingleAxisParams().get_bin_centers(key)

    def delete_(self, key=None, safemode=True):
        # Delete downstream entries first
        delete_(self, [TimeRelWAFRVecSTAveSel, TimeRelWAAveFRVecSel], key, safemode)


"""
Notes on setup of TimeRelWAFRVecSTAve tables:
- We combine entries from TimeRelWAFRVec across epochs holding everything else constant, so we want all primary
keys of TimeRelWAFRVec except epoch to be primary keys in selection table, and we want epoch_description to be
primary key serving to describe epochs we combine across
- We pair entries above with each available time_rel_wa_fr_vec_st_ave_param_name from TimeRelWAFRVecSTAveParams
"""


@schema
class TimeRelWAFRVecSTAveParams(CovariateFRVecSTAveParamsBase):
    definition = """
    # Parameters for TimeRelWAFRVecSTAve
    time_rel_wa_fr_vec_st_ave_param_name : varchar(40)
    ---
    metric_name : varchar(40)
    vector_type : varchar(40)
    mask_duration : float  # seconds
    """

    def delete_(self, key, safemode=True):
        # Delete from upstream tables

        delete_(self, [
            TimeRelWAFRVecSTAveSummSel, TimeRelWAFRVecSTAveSel, TimeRelWAAveFRVecSummSel, TimeRelWAAveFRVecSel],
                key, safemode)


class TimeRelWAFRVecAveSelBase(CovariateFRVecAveSelBase):

    @staticmethod
    def _fr_vec_table():
        return TimeRelWAFRVec


@schema
class TimeRelWAFRVecSTAveSel(TimeRelWAFRVecAveSelBase):
    definition = """
    # Selection from upstream tables for TimeRelWAFRVecSTAve 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> TimeRelWADigParams
    -> TimeRelWADigSingleAxisParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> TimeRelWAFRVecSTAveParams
    zscore_fr : bool
    time_rel_wa_fr_vec_param_name : varchar(40)
    """

    def _get_cov_fr_vec_param_names(self):
        return ["stay_leave_trials_pre_departure", "correct_incorrect_stay_trials"]

    def delete_(self, key, safemode=True):
        delete_(self, [TimeRelWAFRVecSTAve], key, safemode)


# Overrides methods in CovariateFRVecAveBase in a manner specific to time relative to well arrival covariate
class TimeRelWAFRVecAveBase:

    @staticmethod
    def get_valid_covariate_bin_nums(key):
        # Important to pass full key because bins depend on file in this case
        return TimeRelWADigSingleAxisParams().get_valid_bin_nums(key)

    def get_bin_centers_map(self):
        key = self.fetch1("KEY")
        x = TimeRelWADigSingleAxisParams().get_valid_bin_nums(key)
        bin_centers = (self._fr_vec_table() & key).get_bin_centers()

        return AverageVectorDuringLabeledProgression.get_bin_centers_map(x, bin_centers)

    @staticmethod
    def _fr_vec_table():
        return TimeRelWAFRVec


@schema
class TimeRelWAFRVecSTAve(TimeRelWAFRVecAveBase, CovariateFRVecSTAveBase):
    definition = """
    # Single 'trial' comparison of firing rate vectors across combinations of time bin and path identity 
    -> TimeRelWAFRVecSTAveSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> TimeRelWAFRVecSTAve
        -> TimeRelWAFRVec
        """

    def delete_(self, key, safemode=True):
        delete_(self, [TimeRelWAFRVecSTAveSummSel], key, safemode)


"""
Notes on setup of TimeRelWAAveFRVec tables:
 - We combine entries from TimeRelWAFRVec across epochs holding everything else constant, so we want all primary
   keys of TimeRelWAFRVec except epoch to be primary keys in selection table, and we want epoch_description to be
   primary key serving to describe epochs we combine across
"""


@schema
class TimeRelWAAveFRVecParams(CovariateAveFRVecParamsBase):
    definition = """
    time_rel_wa_ave_fr_vec_param_name : varchar(40)
    ---
    metric_name : varchar(40)
    vector_type : varchar(40)
    """


@schema
class TimeRelWAAveFRVecSel(TimeRelWAFRVecAveSelBase):
    definition = """
    # Selection from upstream tables for TimeRelWAAveFRVec 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> TimeRelWADigParams
    -> TimeRelWADigSingleAxisParams
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> TimeRelWAAveFRVecParams
    zscore_fr : bool
    time_rel_wa_fr_vec_param_name : varchar(40)
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
        return ["even_odd_stay_trials"]


@schema
class TimeRelWAAveFRVec(TimeRelWAFRVecAveBase, CovariateFRVecTrialAveBase):
    definition = """
    # Comparison of average firing rate difference vectors across combinations of path bin, path identity, and epoch
    -> TimeRelWAAveFRVecSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> TimeRelWAAveFRVec
        -> TimeRelWAFRVec
        """


"""
Notes on TimeRelWAFRVecSTAveSumm table setup:
- We want to combine entries across TimeRelWAFRVecSTAve, across nwb_file_names, epochs_description, 
and brain_region. For this reason, we want TimeRelWAFRVecSTAveSummSel to have all primary keys of TimeRelWAFRVecSTAve
except for nwb_file_name, epochs_description, brain_region, brain_region_units_param_name, and 
curation_name. 
  To specify the nwb_file_names and corresponding epochs_descriptions we want to combine across, we use recording_set.
  To specify the brain regions we want to combine across, we use brain_region_cohort. 
  To specify curation_name, we use curation_set_name.
  To specify brain region unit information, we use BrainRegionUnitsCohortType
- We include BrainRegionUnitsCohortType in TimeRelWAFRVecSTAveSummParams so that we can stay within the
limit on number of primary keys
"""


@schema
class TimeRelWAFRVecSTAveSummParams(CovariateFRVecAveSummParamsBase):
    definition = """
    # Parameters for TimeRelWAFRVecSTAveSumm
    time_rel_wa_fr_vec_st_ave_summ_param_name : varchar(200)
    ---
    metric_processing_name : varchar(40)  # describes additional processing on metric
    label_name : varchar(40)
    boot_set_name : varchar(120)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    def _boot_set_names(self):
        return super()._boot_set_names() + self._valid_brain_region_diff_boot_set_names() + \
               self._valid_stay_leave_diff_boot_set_names() + \
               self._valid_stay_leave_diff_brain_region_diff_boot_set_names()


@schema
class TimeRelWAFRVecSTAveSummSel(CovariateFRVecAveSummSelBase):
    definition = """
    # Selection from upstream tables for TimeRelWAFRVecSTAveSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    -> TimeRelWADigParams
    -> TimeRelWADigSingleAxisParams
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> TimeRelWAFRVecSTAveParams
    zscore_fr : bool
    -> TimeRelWAFRVecParams
    -> TimeRelWAFRVecSTAveSummParams
    ---
    upstream_keys : mediumblob
    """

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + [
            "brain_region_diff", "stay_leave_diff", "stay_leave_diff_brain_region_diff"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + [
            "brain_region_diff_rat_cohort", "stay_leave_diff_rat_cohort",
            "stay_leave_diff_brain_region_diff_rat_cohort"]

    def _default_cov_fr_vec_param_names(self):
        return ["none", "stay_leave_trials_pre_departure"]

    # Override parent class method so can look at reliability of firing rate vector geometry and dynamics
    # during delay period on first day of learning
    def _recording_set_name_types(self):
        return super()._recording_set_name_types() + ["first_day_learning_single_epoch"]


@schema
class TimeRelWAFRVecSTAveSumm(CovariateFRVecSTAveSummBase, TimeRelWAFRVecSummBase):
    definition = """
    # Summary of single 'trial' comparison of firing rate vectors
    -> TimeRelWAFRVecSTAveSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves upstream dependence on upstream tables
        -> TimeRelWAFRVecSTAveSumm
        -> BrainRegionCohort
        -> CurationSet
        -> TimeRelWAFRVecSTAve
        """

    @staticmethod
    def _upstream_table():
        return TimeRelWAFRVecSTAve

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "stay_leave_trials_pre_departure"

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)

    def _get_xticks(self):
        return [.5, 1, 1.5]

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):
        params = super().get_default_table_entry_params()

        params.update({"mask_duration": self._upstream_table()()._get_params_table()._default_mask_duration()})

        # Return default params
        return params


"""
Notes on TimeRelWAAveFRVecSumm table setup: same reasoning as for TimeRelWAFRVecSTAveSumm table setup
"""


@schema
class TimeRelWAAveFRVecSummParams(CovariateFRVecAveSummParamsBase):
    definition = """
    # Parameters for TimeRelWAAveFRVecSumm
    time_rel_wa_ave_fr_vec_summ_param_name : varchar(160)
    ---
    metric_processing_name : varchar(40)  # describes additional processing on metric
    label_name : varchar(40)
    boot_set_name : varchar(40)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    # Override parent class method so can compare quantities along paths with same or different end well
    def _default_label_names(self):
        return super()._default_label_names() + ["end_well"]

    # Override parent class method so can add bootstrap param name that specifies ratio of same path and same turn
    # or different turn path relationship values
    def _boot_set_names(self):
        return super()._boot_set_names() + ["relationship_div_median", "relationship_div_rat_cohort_median"]


@schema
class TimeRelWAAveFRVecSummSel(CovariateFRVecAveSummSelBase):
    definition = """
    # Selection from upstream tables for TimeRelWAAveFRVecSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    -> TimeRelWADigParams
    -> TimeRelWADigSingleAxisParams
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> TimeRelWAAveFRVecParams
    zscore_fr : bool
    -> TimeRelWAFRVecParams
    -> TimeRelWAAveFRVecSummParams
    ---
    upstream_keys : mediumblob
    """

    def _default_cov_fr_vec_param_names(self):
        return ["even_odd_stay_trials"]

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + ["relationship_div_median"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + ["relationship_div_rat_cohort_median"]

    # Override parent class method so can look at reliability of firing rate vector geometry and dynamics
    # during delay period on first day of learning
    def _recording_set_name_types(self):
        return super()._recording_set_name_types() + ["first_day_learning_single_epoch"]


@schema
class TimeRelWAAveFRVecSumm(CovariateAveFRVecSummBase, TimeRelWAFRVecSummBase):
    definition = """
    # Summary of single 'trial' comparison of firing rate vectors
    -> TimeRelWAAveFRVecSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves upstream dependence on upstream tables
        -> TimeRelWAAveFRVecSumm
        -> BrainRegionCohort
        -> CurationSet
        -> TimeRelWAAveFRVec
        """

    @staticmethod
    def _upstream_table():
        return TimeRelWAAveFRVec

    def _get_relationship_div_column_params(self, **kwargs):

        cov_fr_vec_param_name = kwargs[self._upstream_table()().get_cov_fr_vec_meta_param_name()]
        label_name = kwargs["label_name"]

        if cov_fr_vec_param_name == "even_odd_trials":

            if label_name == "path":
                return {
                    "denominator_column_name": "same_path_even_odd_trials", "numerator_column_names": [
                    "same_turn_even_odd_trials", "different_turn_well_even_odd_trials"]}

            elif label_name == "end_well":
                return {
                    "denominator_column_name": "same_end_well_even_odd_trials", "numerator_column_names": [
                        "different_end_well_even_odd_trials"]}

        elif cov_fr_vec_param_name == "even_odd_stay_trials":

            if label_name == "path":
                return {
                    "denominator_column_name": "same_path_even_odd_stay_trials", "numerator_column_names": [
                    "same_turn_even_odd_stay_trials", "different_turn_well_even_odd_stay_trials"]}

            elif label_name == "end_well":
                return {
                    "denominator_column_name": "same_end_well_even_odd_stay_trials", "numerator_column_names": [
                        "different_end_well_even_odd_stay_trials"]}

        else:
            raise Exception(f"{cov_fr_vec_param_name} not accounted for")

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "even_odd_stay_trials"

    def _get_xticks(self):
        return self._get_x_lims()

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)


def populate_jguidera_well_event_firing_rate_vector(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_well_event_firing_rate_vector"
    upstream_schema_populate_fn_list = [
        populate_jguidera_time_relative_to_well_event, populate_jguidera_firing_rate_vector]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_well_event_firing_rate_vector():
    schema.drop()
