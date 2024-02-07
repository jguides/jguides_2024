import copy

import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd
import statsmodels.api as sm

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_reliability_paper_nwb_file_names
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import unique_table_column_sets, \
    insert_analysis_table_entry, insert1_print, \
    _get_idx_column_name, preserve_df_row_idx, restore_df_row_idx, get_unit_name, delete_, fetch1_tolerate_no_entry
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.glm.glm_helpers import ElasticNetContainer
from src.jguides_2024.glm.jguidera_measurements_interp_pool import XInterpPoolCohortEpsCohort, \
    XInterpPoolCohortParams, \
    populate_jguidera_measurements_interp_pool
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import EpochCohort, EpochsDescription
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikeCounts, populate_jguidera_res_spikes
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsParams, BrainRegionUnits, EpsUnits
from src.jguides_2024.time_and_trials.jguidera_cross_validation_pool import TrainTestSplitPool, \
    populate_jguidera_cross_validation_pool
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams
from src.jguides_2024.utils.df_helpers import check_same_index, zip_df_columns, df_pop
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import unpack_single_element, check_all_unique

# Needed for table definitions:
nd

schema_name = "jguidera_el_net"
schema = dj.schema(schema_name)


@schema
class ElNetParams(SecKeyParamsBase):
    definition = """
    # Parameters for generalized linear model with L1 and L2 regularization
    el_net_param_name : varchar(100)
    ---
    family : varchar(40)
    alpha : decimal(11,8) unsigned
    l1_wt : decimal(9,8) unsigned  # note that datajoint would not allow L1_wt
    """

    def _default_params(self):
        return [("poisson", .00005, 0)]

    def insert1(self, key, **kwargs):
        valid_families = ["poisson"]
        if key["family"] not in valid_families:
            raise Exception(f"family must be in {valid_families} but is {key['family']}")
        super().insert1(key, **kwargs)

    def fetch1_params(self):
        # Replace family text with stats models family, convert decimal to float, convert "L1" to "l1",
        # and drop param name
        family_map = {"poisson": sm.families.Poisson()}
        family, alpha, l1_wt = self.fetch1("family", "alpha", "l1_wt")
        return {"family": family_map[family], "alpha": float(alpha), "L1_wt": float(l1_wt)}


# We want entries from ResEpochSpikeCounts.Unit across epochs
@schema
class ElNetSel(SelBase):
    definition = """
    # Selection from upstream tables for ElNet
    -> XInterpPoolCohortEpsCohort
    -> TrainTestSplitPool
    -> ElNetParams
    sort_group_id : int
    curation_name : varchar(80)
    unit_id : int
    """

    def _get_potential_keys(self, key_filter=None, verbose=True):
        if key_filter is None:
            key_filter = dict()
        keys = []
        k1s = ((XInterpPoolCohortEpsCohort * TrainTestSplitPool * ElNetParams) & key_filter).fetch("KEY")
        if verbose:
            print(f"Getting potential keys for ElNetSel...")
        for k1 in k1s:
            if verbose:
                print(f"\nOn k1 {k1}...")
            k3_list = ResTimeBinsPoolCohortParams().get_cohort_params(k1)
            for k2 in unique_table_column_sets(
                    (ResEpochSpikeCounts & k1), [
                        x for x in ResEpochSpikeCounts.primary_key if x not in [
                            "epoch", "res_time_bins_pool_param_name"]], as_dict=True):
                if verbose:
                    print(f"On k2 {k2}...")
                if all([ResEpochSpikeCounts & {**k2, **k3} for k3 in
                        k3_list]):  # proceed if entry for all epochs in cohort
                    insert_key = {**k1, **k2}
                    if verbose:
                        print(f"Adding keys...")
                    for k3 in (ResEpochSpikeCounts.Unit & insert_key).fetch("KEY"):
                        insert_key.update({"unit_id": k3["unit_id"]})
                        keys.append(copy.deepcopy(insert_key))
        if verbose:
            print(f"Returning keys...")
        return keys

    def delete_(self, key, safemode=True):
        delete_(self, [ElNet], key, safemode)


@schema
class ElNet(ComputedBase):
    definition = """
    # Elastic net, statsmodels
    -> ElNetSel
    ---
    -> nd.common.AnalysisNwbfile
    fit_params_df_object_id : varchar(40)
    log_likelihood_object_id : varchar(40)
    results_folds_merged_df_object_id : varchar(40)
    folds_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream tables
        -> ElNet
        -> ResEpochSpikeCounts.Unit
        """

    @staticmethod
    def get_spike_counts(key):
        # Get spike counts across epochs
        return pd.concat([(ResEpochSpikeCounts.Unit() & {**key, **k}).fetch1_dataframe() for k in
                                  ResTimeBinsPoolCohortParams().get_cohort_params(key)], axis=0)

    @staticmethod
    def get_cross_validation_dfs(key):
        return (TrainTestSplitPool & key).fetch1_dataframes()

    @staticmethod
    def get_design_df(key):
        return (XInterpPoolCohortEpsCohort & key).fetch_dataframes()

    def make(self, key):

        # Get spike counts across epochs
        spike_counts = self.get_spike_counts(key)

        # Get design matrix
        design_df = self.get_design_df(key)

        # Check same index across spike counts and design matrix
        check_same_index((spike_counts, design_df))

        # Get train / test sets
        cv_dfs = self.get_cross_validation_dfs(key)

        # Get params
        params = (ElNetParams & key).fetch1_params()

        # Get maps spanning the chain: model name, covariate "types", covariate "groups", covariate names
        model_name = XInterpPoolCohortParams().lookup_shorthand_from_param_name(key["x_interp_pool_cohort_param_name"])
        obj = XInterpPoolCohortParams().get_shorthand_params_map()[model_name]
        covariate_types = obj.x_interp_pool_shorthand_param_names
        covariate_model_name_map = {model_name: covariate_types}  # map from model name to covariate types
        x_interp_pool_param_names = obj.x_interp_pool_param_names
        covariate_type_map = {k: [v] for k, v in zip(covariate_types, x_interp_pool_param_names)}
        covariate_group_map = (
                    XInterpPoolCohortEpsCohort & key).get_covariate_group_map()  # map from covariate groups to covariate names

        # Get glm results
        elnet_container = ElasticNetContainer(
            covariate_model_name_map, covariate_type_map, covariate_group_map, design_df, spike_counts, cv_dfs,
            cross_validation_params=None, **params)

        # Prepare to store results
        # Convert likelihood to df so can store in analysis nwb file
        log_likelihood = pd.DataFrame.from_dict({"train_set_id": elnet_container.log_likelihood.index,
                                                 "log_likelihood": elnet_container.log_likelihood.values})
        # Reset df index to be able to store in analysis nwb file
        fit_params_df = elnet_container.fit_params_df.reset_index()
        folds_df = elnet_container.folds_df.reset_index()

        # For columns with a pandas series in each row, we will lose index upon saving out to analysis nwbf. To
        # preserve the index, break out into separate column. Note that the name of the column here cannot be
        # the name of the column plus '_index', as this leads to an error on saving. For this reason we use '_idx'
        # suffix
        # ...folds_df
        folds_df = preserve_df_row_idx(folds_df, "y_test")
        # ...results_folds_merged_df. y_test_predicted and y_test_predicted_count should share same index, so only
        # save out one. First verify same index
        results_folds_merged_df = elnet_container.results_folds_merged_df
        for x1, x2 in zip_df_columns(results_folds_merged_df, ["y_test_predicted", "y_test_predicted_count"]):
            if not all(x1.index == x2.index):
                raise Exception(f"index should match across y_test_predicted and y_test_predicted_count")
        results_folds_merged_df = preserve_df_row_idx(results_folds_merged_df, "y_test_predicted")

        # Insert into main table
        insert_analysis_table_entry(
            self, nwb_objects=[fit_params_df, log_likelihood, results_folds_merged_df, folds_df], key=key)

        # Insert into part table
        for k in ResTimeBinsPoolCohortParams().get_cohort_params(key):
            insert1_print(self.Upstream, {**key, **k})

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):

        df_index_name_map = {"fit_params_df": "train_set_id", "log_likelihood": "train_set_id",
                             "folds_df": "test_condition"}
        df_index_name = self.get_default_df_index_name(df_index_name, object_id_name, df_index_name_map)
        df = super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)
        # Combine y_test and y_test_index into series in folds_df (note this could not be done before saving
        # analysis nwbf because cannot store series in df here)
        if object_id_name == "folds_df":
            df = restore_df_row_idx(df, "y_test")
        elif object_id_name == "results_folds_merged_df":
            df = restore_df_row_idx(df, "y_test_predicted", drop_idx_column=False)
            df = restore_df_row_idx(df, "y_test_predicted_count", _get_idx_column_name("y_test_predicted"))

        return df

    def print_populated(
            self, el_net_param_name=None, curation_set_name="runs_analysis_v1",
            res_time_bins_pool_cohort_param_name=None, x_interp_pool_cohort_param_name=None, populate_tables=True,
            verbose=True, return_keys=False, nwb_file_names=None):

        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_param
        if el_net_param_name is None:
            el_net_param_name = ElNetParams().get_default_param_name()
        if res_time_bins_pool_cohort_param_name is None:
            res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand(
                get_glm_default_param("delay_time_bins_shorthand"))
        if x_interp_pool_cohort_param_name is None:
            x_interp_pool_cohort_param_name = XInterpPoolCohortParams().lookup_param_name_from_shorthand(
                "delay_intercept")
        if nwb_file_names is None:
            nwb_file_names = get_reliability_paper_nwb_file_names()
        k1 = {
            'el_net_param_name': el_net_param_name,
            'res_time_bins_pool_cohort_param_name': res_time_bins_pool_cohort_param_name,
            'x_interp_pool_cohort_param_name': x_interp_pool_cohort_param_name}

        # Get unique sets of nwb file name / epochs_id that have entries in ElNet for k1
        col_sets = unique_table_column_sets(self & k1, ["nwb_file_name", "epochs_id"])
        # Restrict to certain nwb file names if indicated
        if nwb_file_names is not None:
            col_sets = [x for x in col_sets if x[0] in nwb_file_names]

        # To detect all units, set minimum firing rate threshold to zero
        min_epoch_mean_firing_rate = 0

        if verbose:
            print(f"The following nwb file names / epochs have all sort group / unit ids in ElNet for key {k1}:")

        keys_list = []
        for nwb_file_name, epochs_id in col_sets:

            # Get units param name
            # Note that currently only works for single epoch
            k2 = {**copy.deepcopy(k1), **{"nwb_file_name": nwb_file_name, "epochs_id": epochs_id}}
            epochs = (EpochCohort & k2).fetch1("epochs")
            if len(epochs) > 1:
                raise Exception(f"code currently only equipped to work with single epoch")
            k2.update({"brain_region_units_param_name": BrainRegionUnitsParams().lookup_single_epoch_param_name(
                nwb_file_name, unpack_single_element(epochs), min_epoch_mean_firing_rate)})

            # Get all possible unit names for this nwb file and epoch ("expected" unit names)
            brain_region_cohort_name = "all_targeted"
            brain_regions = (BrainRegionCohort & {"nwb_file_name": nwb_file_name, "brain_region_cohort_name":
                brain_region_cohort_name}).fetch1("brain_regions")

            # Populate tables if indicated
            if populate_tables:
                EpsUnits().populate_(key=k2)
                BrainRegionUnits().populate_(key=k2)

            # Get map to curation name
            curation_set_key = {"nwb_file_name": nwb_file_name, "brain_region_cohort_name":
                brain_region_cohort_name, "curation_set_name": curation_set_name}
            table_subset = (CurationSet & curation_set_key)
            # Exit if does not exist
            if len(table_subset) == 0:
                print(f"No curation_names_map for {curation_set_key}. Exiting...")
                continue

            # Get map from brain region to unit IDs
            curation_names_map = table_subset.fetch1_dataframe()
            # get epochs description
            epochs_description = (EpochsDescription & k2).fetch1("epochs_description")
            maps = {brain_region: fetch1_tolerate_no_entry((BrainRegionUnits & {**k2, **{
                "brain_region": brain_region, "curation_name":
                    df_pop(curation_names_map, {"brain_region": brain_region, "epochs_description": epochs_description},
                           "curation_name")
            }}), "sort_group_unit_ids_map")
                    for brain_region in brain_regions}  # maps from sort group to unit ids, per brain region
            if any([x is None for x in list(maps.values())]):
                if verbose:
                    print(f"not all entries available in BrainRegionUnits for {nwb_file_name}, eps {epochs_id}. "
                          f"At least one entry in maps is None: {maps}. Continuing...")
                continue
            expected_unit_names = [get_unit_name(k, v_i) for m in maps.values() for k, v in m.items() for v_i in v]

            # Get unit names in ElNet ("available" unit names)
            sort_group_ids, unit_ids = list(map(np.concatenate, list(zip(*[(self & {**k2, **{
                "curation_name":
                    df_pop(curation_names_map, {"brain_region": brain_region, "epochs_description": epochs_description},
                           "curation_name"),
                "sort_group_id": sort_group_id,
            }}).fetch(
                "sort_group_id", "unit_id") for brain_region in brain_regions
                for sort_group_id in maps[brain_region].keys()]))))
            available_unit_names = [get_unit_name(x, y) for x, y in zip(sort_group_ids, unit_ids)]
            check_all_unique(available_unit_names)  # check each unit represented only once

            # Check if all available units are in ElNet for this nwb file and epochs id
            all_units_available = check_membership(
                expected_unit_names, available_unit_names, "expected unit names", "available unit names",
                tolerate_error=True)

            # Print outcome if indicated
            if verbose:
                if all_units_available:
                    print(nwb_file_name, epochs_id)
                else:
                    print(f"missing units for {nwb_file_name}, eps {epochs_id}:", set(expected_unit_names) -
                          set(available_unit_names))

            # Store key with nwb file name and epochs id if all units available
            if all_units_available:
                keys_list.append({"nwb_file_name": nwb_file_name, "epochs_id": epochs_id})

        # Return keys if indicated
        if return_keys:
            return keys_list


def populate_jguidera_el_net(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_el_net"
    upstream_schema_populate_fn_list = [
        populate_jguidera_measurements_interp_pool, populate_jguidera_cross_validation_pool,
        populate_jguidera_res_spikes]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_el_net():
    schema.drop()
