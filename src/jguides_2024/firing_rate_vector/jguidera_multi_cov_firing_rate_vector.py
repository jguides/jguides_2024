import copy
from collections import namedtuple

import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    CovariateFRVecSelBase, CovariateFRVecBase, CovariateFRVecTrialAveBase, CovariateAveFRVecParamsBase, \
    CovariateFRVecAveSelBase, CovariateFRVecSTAveBase, CovariateFRVecSTAveParamsBase, CovariateFRVecSTAveSummBase, \
    CovariateFRVecAveSummSelBase, CovariateFRVecAveSummParamsBase, MultiCovFRVecSummBase, CovariateAveFRVecSummBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import drop_, delete_
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_names
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_post_delay_firing_rate_vector import RelPostDelFRVec
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch, EpochsDescription, RecordingSet
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptDigParams
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsParams, BrainRegionUnits, BrainRegionUnitsCohortType
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellTrials, DioWellDDTrials
from src.jguides_2024.time_and_trials.jguidera_relative_time_at_well import RelTimeWellPostDelayDigParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel
from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWADigSingleAxisParams
from src.jguides_2024.utils.df_helpers import check_df_indices_close
from src.jguides_2024.utils.state_evolution_estimation import AverageVectorDuringLabeledProgression

# Needed for table definitions:
nd
DioWellTrials
DioWellDDTrials
BrainRegionUnits
BrainRegionUnitsCohortType

schema = dj.schema("jguidera_multi_cov_firing_rate_vector")


@schema
class MultiCovFRVecParams(dj.Manual):
    definition = """
    # Parameters for MultiCovFRVec
    multi_cov_fr_vec_param_name : varchar(100)
    ---
    multi_cov_fr_vec_params : blob
    """

    def insert_defaults(self, **kwargs):

        # Initialize list for storing params for each future table entry
        params_list = []

        # Path and delay, EVEN AND ODD or SINGLE TRIALS
        cov_fr_vec_param_names = [
            "even_odd_correct_incorrect_stay_trials", "correct_incorrect_stay_trials",
            "stay_leave_trials_pre_departure"]
        for cov_fr_vec_param_name in cov_fr_vec_param_names:

            multi_cov_fr_vec_param_name = "path_delay" + "^" + cov_fr_vec_param_name
            multi_cov_fr_vec_params = {
                "table_names": ["PathFRVec", "TimeRelWAFRVec"], "keys": [
                    {'ppt_param_name': "ppt_1",
                     'ppt_dig_param_name': "0.0625",
                     "path_fr_vec_param_name": cov_fr_vec_param_name
                     },
                    {'time_rel_wa_dig_param_name': "0.25",
                     "time_rel_wa_dig_single_axis_param_name": "0^2",
                     'time_rel_wa_fr_vec_param_name': cov_fr_vec_param_name},
                ]}
            params_list.append({
                "multi_cov_fr_vec_param_name": multi_cov_fr_vec_param_name, "multi_cov_fr_vec_params":
                    multi_cov_fr_vec_params})

            # Path, delay, and post delay, EVEN AND ODD or SINGLE TRIALS
            cov_fr_vec_param_name = "even_odd_correct_incorrect_stay_trials"
            multi_cov_fr_vec_param_name = "path_delay_postdelay" + "^" + cov_fr_vec_param_name
            multi_cov_fr_vec_params = {
                "table_names": ["PathFRVec", "TimeRelWAFRVec", "RelPostDelFRVec"], "keys": [
                    {'ppt_param_name': "ppt_1",
                     'ppt_dig_param_name': "0.0625",
                     "path_fr_vec_param_name": cov_fr_vec_param_name
                     },
                    {'time_rel_wa_dig_param_name': "0.25",
                     "time_rel_wa_dig_single_axis_param_name": "0^2",
                     'time_rel_wa_fr_vec_param_name': cov_fr_vec_param_name},
                    {'rel_time_well_post_delay_dig_param_name': "0.1",
                     'rel_post_del_fr_vec_param_name': cov_fr_vec_param_name},
                ]}
            params_list.append({
                "multi_cov_fr_vec_param_name": multi_cov_fr_vec_param_name, "multi_cov_fr_vec_params":
                    multi_cov_fr_vec_params})

        # Previous correct / incorrect trial
        cov_fr_vec_param_name = "prev_correct_incorrect_trials"
        # TODO: finish coding this

        # Insert into table
        for params in params_list:
            self.insert1(params, skip_duplicates=True)

    def meta_param_name(self):
        return "multi_cov_fr_vec_param_name"

    # Drop dependent tables and avoid error related to attempt to
    # drop part table before main table
    def drop_(self):
        drop_([MultiCovFRVecSel(), MultiCovFRVec(), self])

    def delete_(self, key, safemode=True):
        # Delete from upstream tables
        delete_(self, [MultiCovFRVecSel], key, safemode)


@schema
class MultiCovFRVecSel(CovariateFRVecSelBase):
    definition = """
    # Selection from upstream tables for MultiCovFRVec
    -> FRVec
    -> MultiCovFRVecParams
    """

    # Override parent class method so can restrict entries to downstream tables
    def _get_potential_keys(self, key_filter=None, verbose=True):
        # Approach: avoid a combinatorial explosion of entries

        if verbose:
            print(f"getting potential keys for MultiCovFRVecSel...")

        min_epoch_mean_firing_rate = .1
        brain_region_cohort_name = "all_targeted"
        curation_set_name = "runs_analysis_v1"
        primary_features = {
            "zscore_fr": 0,
            "res_time_bins_pool_param_name": ResTimeBinsPoolSel().lookup_param_name_from_shorthand("epoch_100ms")}
        unit_params = {
            "unit_subset_type": "rand_target_region", "unit_subset_size": 50}
        unit_subset_iterations = np.arange(0, 10)
        multi_cov_fr_vec_param_names = MultiCovFRVecParams().fetch("multi_cov_fr_vec_param_name")
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
                        for multi_cov_fr_vec_param_name in multi_cov_fr_vec_param_names:
                            key.update({"multi_cov_fr_vec_param_name": multi_cov_fr_vec_param_name})
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

        print(f"\nReturning {len(potential_keys)} potential keys...")

        return potential_keys

    # Drop dependent tables and avoid error related to attempt to
    # drop part table before main table
    def drop_(self):
        drop_([MultiCovFRVec(), self])

    def delete_(self, key, safemode=True):
        delete_(self, [MultiCovFRVec], key, safemode)


@schema
class MultiCovFRVec(CovariateFRVecBase):
    definition = """
    # Firing rate vectors averaged in contiguous stretches in multiple covariate bins
    -> MultiCovFRVecSel
    ---
    unit_names : blob
    table_old_new_bin_nums_map : blob
    -> nd.common.AnalysisNwbfile
    vector_df_object_id : varchar(40)
    ave_vector_df_object_id : varchar(40)
    diff_vector_df_object_id : varchar(40)
    ave_diff_vector_df_object_id : varchar(40)
    """

    class DioWellTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellTrials
        -> MultiCovFRVec
        -> DioWellTrials
        """

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellDDTrials
        -> MultiCovFRVec
        -> DioWellDDTrials
        """

    class PptDigParams(dj.Part):
        definition = """
        # Achieves dependence on PptDigParams
        -> MultiCovFRVec
        -> PptDigParams
        """

    class TimeRelWADigSingleAxisParams(dj.Part):
        definition = """
        # Achieves dependence on TimeRelWADigSingleAxisParams
        -> MultiCovFRVec
        -> TimeRelWADigSingleAxisParams
        """

    class RelTimeWellPostDelayDigParams(dj.Part):
        definition = """
        # Achieves dependence on RelTimeWellPostDelayDigParams
        -> MultiCovFRVec
        -> RelTimeWellPostDelayDigParams
        """

    def _store_table_specific_quantities(self, inputs, main_table_key):
        main_table_key.update({"table_old_new_bin_nums_map": inputs.table_old_new_bin_nums_map})
        return main_table_key

    def get_inputs(self, key, verbose=False, ax=None):

        # Get parameters
        multi_cov_fr_vec_params = (MultiCovFRVecParams & key).fetch1("multi_cov_fr_vec_params")

        # Get inputs for each table, and corresponding bin numbers
        inputs_list = []  # initialize list for inputs across tables
        bin_nums_list = []  # initialize list for bin numbers across tables
        table_names_list = []  # initialize list for table names

        for table_name, params in zip(multi_cov_fr_vec_params["table_names"], multi_cov_fr_vec_params["keys"]):

            # Append inputs
            table_key = {**key, **params}
            table = get_table(table_name)()
            inputs = table.get_inputs(table_key)
            inputs_list.append(inputs)
            # ...update upstream table entry tracker
            self._merge_tracker(table)

            # Append bin nums
            if table_name == "PathFRVec":
                dig_params_table = PptDigParams()
                bin_nums = dig_params_table.get_valid_bin_nums(key=table_key)
            elif table_name == "TimeRelWAFRVec":
                dig_params_table = TimeRelWADigSingleAxisParams()
                bin_nums = dig_params_table.get_valid_bin_nums(key=table_key)
            elif table_name == "RelPostDelFRVec":
                dig_params_table = RelTimeWellPostDelayDigParams()
                bin_nums = dig_params_table.get_valid_bin_nums(key=table_key)
            else:
                raise Exception(f"Currently no code to get bin nums for {table_name}")
            bin_nums_list.append(bin_nums)
            # ...update upstream table entry tracker
            self._update_upstream_entries_tracker(dig_params_table, table_key)

            # Append table name
            table_names_list.append(table_name)

        # "Stack" x (i.e. bin numbers) in each set of inputs from a table
        # Approach: maintain same x for first table. For each subsequent table, add the maximum x from previous tables

        # First define new bin numbers across tables
        new_bin_nums = []  # initialize list for new bin numbers across tables
        table_old_new_bin_nums_map = dict()  # initialize map from table names to new bin numbers
        for idx, (table_name, bin_nums) in enumerate(zip(table_names_list, bin_nums_list)):
            # For first table, use bin numbers unchanged
            if idx == 0:
                new_table_bin_numbers = list(bin_nums)
            # Otherwise shift bin numbers by maximum bin number from previous tables
            else:
                max_previous_bin = np.max(new_bin_nums)
                # Define new bin numbers from this table
                new_table_bin_numbers = list(bin_nums + max_previous_bin)
            # Check that no overlap in new bin numbers and running list
            if any(x in new_bin_nums for x in new_table_bin_numbers):
                raise Exception(f"bin numbers from current table overlap with previous ones")
            # Update running list with all bin numbers
            new_bin_nums += new_table_bin_numbers
            # Update map from table names to new bin numbers
            table_old_new_bin_nums_map[table_name] = {x: y for x, y in zip(bin_nums, new_table_bin_numbers)}

        # Join x across tables, iterating x using map from old to new bin numbers. Also
        # concatenate labels and df across tables
        for idx, (table_name, inputs) in enumerate(zip(table_names_list, inputs_list)):
            # For first table, use x values unchanged, and initialize labels and df as those for current table
            if idx == 0:
                new_x = inputs.x
                new_labels = inputs.labels
                new_df = inputs.df

            # If after first table, shift bin numbers by maximum bin number from previous tables
            else:
                # Update x values from table
                new_table_x = pd.Series([table_old_new_bin_nums_map[table_name][x] for x in inputs.x],
                                        index=inputs.x.index)
                # Add to running list of x values across tables
                new_x = pd.concat((new_x, new_table_x), axis=0)

                # Concatenate labels and df
                new_labels = pd.concat((new_labels, inputs.labels), axis=0)
                new_df = pd.concat((new_df, inputs.df), axis=0)

        # Check same indices across new x, labels, and df
        check_df_indices_close((new_x, new_labels, new_df), 0)

        # Sort all of these
        new_x.sort_index(inplace=True)
        new_labels.sort_index(inplace=True)
        new_df.sort_index(inplace=True)

        # Plot labels again, now that have concatenated (if indicated)
        self._plot_labels("concatenated", new_labels, verbose, ax)

        return namedtuple("Inputs", "x labels df unit_names table_old_new_bin_nums_map")(
            new_x, new_labels, new_df, np.asarray(new_df.columns), table_old_new_bin_nums_map)

    def get_bin_centers(self, key=None, as_dict=False):

        # Get key if not passed
        if key is None:
            key = self.fetch1("KEY")

        # Get parameters
        multi_cov_fr_vec_params = (MultiCovFRVecParams & key).fetch1("multi_cov_fr_vec_params")

        bin_centers_across_tables = []  # initialize list for bin centers across tables
        for table_name, params in zip(multi_cov_fr_vec_params["table_names"], multi_cov_fr_vec_params["keys"]):

            table_key = {**key, **params}

            # Get bin centers
            if table_name == "PathFRVec":
                bin_centers = (PptDigParams & table_key).get_bin_centers()
            elif table_name == "TimeRelWAFRVec":
                bin_centers = TimeRelWADigSingleAxisParams().get_bin_centers(table_key)
            elif table_name == "RelPostDelFRVec":
                bin_centers = RelPostDelFRVec().get_bin_centers(key)
            else:
                raise Exception(f"Missing code to get bin centers for table {table_name}")

            # Store as dictionary if indicated. Otherwise concatenate to running list
            if as_dict:
                bin_centers_across_tables[table_name] = bin_centers
            else:
                bin_centers_across_tables += list(bin_centers)

        return bin_centers_across_tables

    def drop_(self):
        drop_([MultiCovAveFRVecSel(), MultiCovFRVecSTAveSel, self])

    def delete_(self, key, safemode=True):
        delete_(self, [MultiCovAveFRVecSel, MultiCovFRVecSTAveSel], key, safemode)


class MultiCovFRVecAveSelBase(CovariateFRVecAveSelBase):

    @staticmethod
    def _fr_vec_table():
        return MultiCovFRVec


# Overrides methods in CovariateFRVecAveBase in a manner specific to multiple covariate case
class MultiCovFRVecAveBase:

    @staticmethod
    def get_valid_covariate_bin_nums(key):
        # Important to pass full key because bins depend on file in this case
        table_old_new_bin_nums_map = (MultiCovFRVec & key).fetch1("table_old_new_bin_nums_map")
        return np.concatenate([list(x.values()) for x in table_old_new_bin_nums_map.values()])

    def get_bin_centers_map(self):
        key = self.fetch1("KEY")
        x = self.get_valid_covariate_bin_nums(key)
        bin_centers = (self._fr_vec_table() & key).get_bin_centers()
        return AverageVectorDuringLabeledProgression.get_bin_centers_map(x, bin_centers)

    @staticmethod
    def _fr_vec_table():
        return MultiCovFRVec


"""
Notes on setup of MultiCovAveFRVec tables:
 - We combine entries from MultiCovFRVec across epochs holding everything else constant, so we want as many primary
   keys of MultiCovFRVec except epoch to be primary keys in selection table, and we want epoch_description to be
   primary key serving to describe epochs we combine across
"""


@schema
class MultiCovAveFRVecParams(CovariateAveFRVecParamsBase):
    definition = """
    multi_cov_ave_fr_vec_param_name : varchar(40)
    ---
    metric_name : varchar(40)
    vector_type : varchar(40)
    """


@schema
class MultiCovAveFRVecSel(MultiCovFRVecAveSelBase):
    definition = """
    # Selection from upstream tables for MultiCovAveFRVec 
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> MultiCovAveFRVecParams
    zscore_fr : bool
    multi_cov_fr_vec_param_name : varchar(100)
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
        return MultiCovFRVecParams().fetch("multi_cov_fr_vec_param_name")

    def delete_(self, key, safemode=True):
        delete_(self, [MultiCovAveFRVec], key, safemode)


# TODO: regenerate table if possible; changed table heading
@schema
class MultiCovAveFRVec(MultiCovFRVecAveBase, CovariateFRVecTrialAveBase):
    definition = """
    # Comparison of average firing rate vectors across combinations of bin, label, and epoch
    -> MultiCovAveFRVecSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables 
        -> MultiCovAveFRVec
        -> MultiCovFRVec
        """


"""
Notes on setup of MultiCovFRVecSTAve tables:
- We combine entries from MultiCovFRVec across epochs holding everything else constant, so we want all primary
keys of MultiCovFRVec except epoch to be primary keys in selection table, and we want epoch_description to be
primary key serving to describe epochs we combine across
- We pair entries above with each available multi_cov_fr_vec_st_ave_param_name from MultiCovFRVecSTAveParams
"""


@schema
class MultiCovFRVecSTAveParams(CovariateFRVecSTAveParamsBase):
    definition = """
    # Parameters for MultiCovFRVecSTAve
    multi_cov_fr_vec_st_ave_param_name : varchar(40)
    ---
    metric_name : varchar(40)
    vector_type : varchar(40)
    mask_duration : float  # seconds
    """

    def delete_(self, key, safemode=True):
        # Delete from upstream tables

        delete_(self, [
            MultiCovFRVecSTAveSel, MultiCovAveFRVecSel],
                key, safemode)


class MultiCovFRVecAveSelBase(CovariateFRVecAveSelBase):

    @staticmethod
    def _fr_vec_table():
        return MultiCovFRVec


@schema
class MultiCovFRVecSTAveSel(MultiCovFRVecAveSelBase):
    definition = """
    # Selection from upstream tables for MultiCovFRVecSTAve
    -> EpochsDescription
    res_time_bins_pool_param_name : varchar(1000)
    -> BrainRegionUnits
    -> ResEpochSpikesSmParams
    -> MultiCovFRVecSTAveParams
    zscore_fr : bool
    multi_cov_fr_vec_param_name : varchar(100)
    """

    def _get_cov_fr_vec_param_names(self):
        return ["path_delay^correct_incorrect_stay_trials", "path_delay_postdelay^correct_incorrect_stay_trials"]

    def delete_(self, key, safemode=True):
        delete_(self, [MultiCovFRVecSTAve], key, safemode)


@schema
class MultiCovFRVecSTAve(MultiCovFRVecAveBase, CovariateFRVecSTAveBase):
    definition = """
    # Single 'trial' comparison of firing rate vectors across combinations of time bin and path identity
    -> MultiCovFRVecSTAveSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> MultiCovFRVecSTAve
        -> MultiCovFRVec
        """

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)


"""
Notes on MultiCovFRVecSTAveSumm table setup:
- We want to combine entries across MultiCovFRVecSTAve, across nwb_file_names, epochs_description, 
and brain_region. For this reason, we want MultiCovFRVecSTAveSel to have all primary keys of MultiCovFRVecSTAve
except for nwb_file_name, epochs_description, brain_region, brain_region_units_param_name, and 
curation_name. 
  To specify the nwb_file_names and corresponding epochs_descriptions we want to combine across, we use recording_set.
  To specify the brain regions we want to combine across, we use brain_region_cohort. 
  To specify curation_name, we use curation_set_name.
  To specify brain region unit information, we use BrainRegionUnitsCohortType
- We include BrainRegionUnitsCohortType in MultiCovFRVecSTAveSummParams so that we can stay within the
limit on number of primary keys
"""


@schema
class MultiCovFRVecSTAveSummParams(CovariateFRVecAveSummParamsBase):
    definition = """
    # Parameters for MultiCovFRVecSTAveSumm
    multi_cov_fr_vec_st_ave_summ_param_name : varchar(200)
    ---
    metric_processing_name : varchar(40)  # describes additional processing on metric
    label_name : varchar(40)
    boot_set_name : varchar(120)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    def _boot_set_names(self):
        return super()._boot_set_names() + self._valid_brain_region_diff_boot_set_names()


@schema
class MultiCovFRVecSTAveSummSel(CovariateFRVecAveSummSelBase):
    definition = """
    # Selection from upstream tables for MultiCovFRVecSTAveSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> MultiCovFRVecSTAveParams
    zscore_fr : bool
    -> MultiCovFRVecParams
    -> MultiCovFRVecSTAveSummParams
    ---
    upstream_keys : mediumblob
    """

    # TODO: remove this once done with testing. This is just to pass fewer recording set names for testing.
    def get_recording_set_names(self, key_filter):
        # Return list of recording set names for a given key_filter, for use in populating tables
        return RecordingSet().get_recording_set_names(key_filter, ["single_epoch_testing"])

    def _default_cov_fr_vec_param_names(self):
        return [
            "path_delay^correct_incorrect_stay_trials",
            "path_delay_postdelay^correct_incorrect_stay_trials"]

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + [
            "brain_region_diff"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + [
            "brain_region_diff_rat_cohort"]

    def delete_(self, key, safemode=True):
        # If recording set name not in key but components that determine it are, then
        # find matching recording set names given the components, to avoid deleting irrelevant
        # entries
        key = copy.deepcopy(key)  # make copy of key to avoid changing outside function
        recording_set_names = RecordingSet().get_matching_recording_set_names(key)
        for recording_set_name in recording_set_names:
            key.update({"recording_set_name": recording_set_name})
            delete_(self, [MultiCovFRVecSTAveSumm], key, safemode)


@schema
class MultiCovFRVecSTAveSumm(CovariateFRVecSTAveSummBase, MultiCovFRVecSummBase):
    definition = """
    # Summary of single 'trial' comparison of firing rate vectors
    -> MultiCovFRVecSTAveSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves upstream dependence on upstream tables
        -> MultiCovFRVecSTAveSumm
        -> BrainRegionCohort
        -> CurationSet
        -> MultiCovFRVecSTAve
        """

    @staticmethod
    def _upstream_table():
        return MultiCovFRVecSTAve

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "path_delay^correct_incorrect_stay_trials"

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        params.update({"mask_duration": self._upstream_table()()._get_params_table()._default_mask_duration()})

        # Return default params
        return params


"""
Notes on MultiCovAveFRVecSumm table setup: same reasoning as for MultiCovFRVecSTAveSumm table setup
"""


@schema
class MultiCovAveFRVecSummParams(CovariateFRVecAveSummParamsBase):
    definition = """
    # Parameters for MultiCovAveFRVecSumm
    multi_cov_ave_fr_vec_summ_param_name : varchar(160)
    ---
    metric_processing_name : varchar(40)  # describes additional processing on metric
    label_name : varchar(40)
    boot_set_name : varchar(40)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    # Override parent class method so can add bootstrap param name that specifies ratio of same path and same turn
    # or different turn path relationship values
    def _boot_set_names(self):
        return super()._boot_set_names()


@schema
class MultiCovAveFRVecSummSel(CovariateFRVecAveSummSelBase):
    definition = """
    # Selection from upstream tables for MultiCovAveFRVecSumm
    -> RecordingSet
    res_time_bins_pool_param_name : varchar(1000)
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> MultiCovAveFRVecParams
    zscore_fr : bool
    -> MultiCovFRVecParams
    -> MultiCovAveFRVecSummParams
    ---
    upstream_keys : mediumblob
    """

    # Override parent class method so can include across rat cohort
    def _recording_set_name_types(self):
        return super()._recording_set_name_types() + ["Haight_rotation_rat_cohort"]

    def _default_cov_fr_vec_param_names(self):
        return ["path_delay^even_odd_correct_incorrect_stay_trials"]

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names()

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names()


@schema
class MultiCovAveFRVecSumm(CovariateAveFRVecSummBase, MultiCovFRVecSummBase):
    definition = """
    # Summary of trial average comparison of firing rate vectors
    -> MultiCovAveFRVecSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves upstream dependence on upstream tables
        -> MultiCovAveFRVecSumm
        -> BrainRegionCohort
        -> CurationSet
        -> MultiCovAveFRVec
        """

    @staticmethod
    def _upstream_table():
        return MultiCovAveFRVec

    def _get_relationship_div_column_params(self, **kwargs):

        cov_fr_vec_param_name = kwargs[self._upstream_table()().get_cov_fr_vec_meta_param_name()]

        if cov_fr_vec_param_name == "even_odd_trials":
            return {
                "denominator_column_name": "same_path_even_odd_trials", "numerator_column_names": [
                    "same_turn_even_odd_trials", "different_turn_well_even_odd_trials"]}
        else:
            raise Exception(f"{cov_fr_vec_param_name} not accounted for")

    def _get_default_plot_cov_fr_vec_param_name(self):
        return "path_delay^even_odd_correct_incorrect_stay_trials"

    def delete_(self, key, safemode=True):
        delete_(self, [], key, safemode)


def populate_jguidera_multi_cov_firing_rate_vector(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_multi_cov_firing_rate_vector"
    upstream_schema_populate_fn_list = []
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_multi_cov_firing_rate_vector():
    schema.drop()
