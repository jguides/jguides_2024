import copy
from collections import namedtuple

import datajoint as dj
import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, ComputedBase, PartBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import unique_table_column_sets, insert1_print, \
    delete_, get_epochs_id
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionSortGroup
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, EpochCohort, RunEpoch, EpochsDescriptions
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmDs, populate_jguidera_res_spikes
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, populate_jguidera_unit, \
    BrainRegionUnitsParams, EpsUnitsParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel, \
    ResTimeBinsPoolCohortParams
from src.jguides_2024.utils.df_helpers import unpack_single_df, df_filter_columns
from src.jguides_2024.utils.dict_helpers import check_equality, dict_comprehension
from src.jguides_2024.utils.list_helpers import check_single_element
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import check_all_unique, unpack_single_vector, unpack_single_element

# Needed for table definitions:
TaskIdentification
BrainRegionUnits
ResTimeBinsPoolSel
ResEpochSpikesSmDs

schema = dj.schema("jguidera_firing_rate_vector")

"""
Notes on design of FRVec tables:
- We want this table to depend on BrainRegionUnits, as well as all primary keys in ResEpochSpikesSmDs except
for sort group id (so nwb file name, epoch, curation name (covered by BrainRegionUnits and TaskIdentification),
res_time_bins_pool_param_name (covered by ResTimeBinsPoolCohortParamName) and res_epoch_spikes_sm_param_name)
- We also want a primary key that indicates whether to return a z scored sample or not
"""


@schema
class FRVecSel(SelBase):
    definition = """
    # Selection from upstream tables for FRVec
    -> TaskIdentification  # nwb file name and epoch
    -> BrainRegionUnits  # nwb file name, brain region, brain region units param name, and curation_name
    -> ResTimeBinsPoolSel  # nwb file name, epoch, and param that describes time bins
    res_epoch_spikes_sm_param_name : varchar(40)  # describes smoothing kernel
    zscore_fr = 0 : bool  # if True, return z score firing rate, otherwise return firing rate
    """

    @staticmethod
    def check_consistent_key(key, verbose=False):
        key = copy.deepcopy(key)  # copy key to avoid changing outside function
        # Check that epochs_id consistent across sources
        epochs_id = BrainRegionUnitsParams().get_epochs_id(key["nwb_file_name"], key["brain_region_units_param_name"])
        # Tolerate two cases: 1) brain region units param for all runs, used for any number of run epochs, and 2)
        # brain region units param for single run, used for corresponding run epoch
        # ...add epochs_id if not in key
        if "epochs_id" not in key:
            if "epochs" in key:
                epochs = key["epochs"]
            elif "epoch" in key:
                epochs = [key["epoch"]]
            else:
                raise Exception(f"epochs_id not in key, and no way to define")
            key["epochs_id"] = get_epochs_id(epochs)
        # ...check for case 1: all runs
        key.pop("epoch")  # drop epoch and use just epochs_id
        if np.logical_and((BrainRegionUnitsParams & key).fetch1("epochs_description") ==
                          EpochsDescription().get_all_runs_description(), all([
            epoch in (RunEpoch & key).fetch("epoch") for epoch in (EpochCohort & key).fetch1("epochs")])):
            return True
        elif epochs_id == key["epochs_id"]:
            return True
        if verbose:
            print(f"In FRVecSel. Inconsistent key: {key}")
        return False

    def _get_potential_keys(self, key_filter=None):

        print(f"Getting potential keys for FRVecSel...")

        # Get inputs if not passed
        if key_filter is None:
            key_filter = dict()

        # Restrict to epoch mean firing rate of 0.1 Hz
        min_epoch_mean_fr = 0.1
        valid_eps_units_param_name = EpsUnitsParams().lookup_param_name([min_epoch_mean_fr])

        # If brain_region_units_param_name was passed, check that corresponds to epoch mean firing
        # rate of 0.1 Hz
        if "brain_region_units_param_name" in key_filter:
            brain_region_units_param_name = key_filter["brain_region_units_param_name"]
            eps_units_param_name = (BrainRegionUnitsParams & {
                "brain_region_units_param_name": brain_region_units_param_name}).fetch1("eps_units_param_name")
            if eps_units_param_name != valid_eps_units_param_name:
                raise Exception(f"passed brain_region_units_param_name must correspond to eps_units_param_name"
                                f" {valid_eps_units_param_name} but instead corresponds to {eps_units_param_name}")

        # Get map from brain region units param name to epochs description
        brup_epd_map = dict_comprehension(
            *BrainRegionUnitsParams().fetch("brain_region_units_param_name", "epochs_description"))

        # Get map from (nwb_file_name, epochs_description) to epoch, in case of valid single epochs

        # Get map that indicates which epochs are valid for which nwb files
        # ...define valid epochs as valid single contingency runs
        epochs_descriptions_name = "valid_single_contingency_runs"
        key = {"epochs_descriptions_name": epochs_descriptions_name}
        valid_nwb_file_name_epochs_map = (EpochsDescriptions & key).get_nwb_file_name_epochs_map()
        # ...Get starting nwb files / epochs
        nwb_file_names, epochs_descriptions, epochs_list = EpochsDescription().fetch(
            "nwb_file_name", "epochs_description", "epochs")
        # ...Add nwb file names with empty epochs list to avoid error below
        for nwb_file_name in nwb_file_names:
            if nwb_file_name not in valid_nwb_file_name_epochs_map:
                valid_nwb_file_name_epochs_map[nwb_file_name] = []  # indicate no valid epochs
        # ...add select other epochs
        # add within session contingency switch sessions from first day where this is introduced
        nwb_file_name = "J1620210607_.nwb"
        for epochs_description in ["run7", "run8"]:
            valid_nwb_file_name_epochs_map[nwb_file_name] += (EpochsDescription & {
                    "nwb_file_name": nwb_file_name, "epochs_description": epochs_description}).fetch1("epochs")
        nwb_file_name = "mango20211207_.nwb"
        for epochs_description in ["run7", "run8"]:
            valid_nwb_file_name_epochs_map[nwb_file_name] += (EpochsDescription & {
                "nwb_file_name": nwb_file_name, "epochs_description": epochs_description}).fetch1("epochs")
        nwb_file_name = "june20220421_.nwb"
        for epochs_description in ["run7", "run8"]:
            valid_nwb_file_name_epochs_map[nwb_file_name] += (EpochsDescription & {
                "nwb_file_name": nwb_file_name, "epochs_description": epochs_description}).fetch1("epochs")
        # ...Print valid epochs so user aware
        print("valid_nwb_file_name_epochs_map: ", valid_nwb_file_name_epochs_map)

        valid_bool = [
            np.logical_and(
                len(epochs) == 1, epochs[0] in valid_nwb_file_name_epochs_map[nwb_file_name])
            for nwb_file_name, epochs in zip(nwb_file_names, epochs_list)]
        nwbf_epd_ep_map = {k: v for k, v in zip(list(zip(nwb_file_names[valid_bool], epochs_descriptions[valid_bool])),
                                                [x[0] for x in epochs_list[valid_bool]])}

        # Loop through keys of BrainRegionUnits which has implicit epoch information in 'brain_region_units_param_name',
        # and find consistent entries in the intersection of ResTimeBinsPoolSel and TaskIdentification,
        # which have explicit epoch information in 'epoch'
        table_intersection = (ResTimeBinsPoolSel * TaskIdentification) & key_filter
        keys = []
        for k1 in unique_table_column_sets(
                BrainRegionUnits & key_filter, BrainRegionUnits.primary_key, as_dict=True):

            # Continue on if eps_units_param_name not valid
            if (BrainRegionUnitsParams & k1).fetch1("eps_units_param_name") != valid_eps_units_param_name:
                continue

            map_k = (k1["nwb_file_name"], brup_epd_map[k1["brain_region_units_param_name"]])
            epochs_description = (BrainRegionUnitsParams & k1).fetch1("epochs_description")

            # Define epochs. Currently, we only allow a single or all run epochs. Note that this requires
            # more code below; otherwise could pull epochs directly from EpochsDescription
            # 1) if key corresponds to single epoch run description, define epochs with the single epoch
            if map_k in nwbf_epd_ep_map:
                epochs = [nwbf_epd_ep_map[map_k]]
            # 2) if all runs epochs description in brain region units param name, define all relevant epochs
            elif epochs_description == EpochsDescription.get_all_runs_description():
                epochs = (EpochsDescription & {**k1, "epochs_description": epochs_description}).fetch1("epochs")
            # 3) otherwise continue  # TODO: cover case with a subset of epochs
            else:
                continue

            # Loop through epochs identified above and append keys
            for epoch in epochs:
                # If epochs_id in key_filter, check that consistent with epoch in k1, and continue if not
                if "epochs_id" in k1:
                    if k1["epoch"] not in (EpochCohort & k1).fetch1("epochs"):
                        continue
                k1.update({"epoch": epoch})
                for res_time_bins_pool_param_name in (table_intersection & k1).fetch("res_time_bins_pool_param_name"):
                    k1.update({"res_time_bins_pool_param_name": res_time_bins_pool_param_name})
                    for zscore_fr in [0, 1]:
                        k1.update({"zscore_fr": zscore_fr})
                        keys.append(copy.deepcopy(k1))

        # Add sort groups IDs. Loop through above keys and, if all sort group ids present in spikes table,
        # include keys with sort group ids
        # ...get names of primary keys in ResEpochSpikesSmDs that are not sort gropu ID
        column_names = [x for x in ResEpochSpikesSmDs.primary_key if x != "sort_group_id"]
        # ...initialize variable to keep track of cases of sort group IDs missing from spikes table to print if desired
        missing_summary = []
        # ...initialize variable for potential keys
        potential_keys = []
        # ...loop through keys above
        for k1 in keys:

            # Add key_filter here in addition to using above since intersected tables above dont have all keys
            # that could be used to restrict entries
            k1.update(key_filter)

            # Get expected sort group ids
            expected_sort_group_ids = (BrainRegionSortGroup & k1).fetch1("sort_group_ids")

            # Loop through unique keys in ResEpochSpikesSmDs, excluding sort group ID
            for k2 in unique_table_column_sets(ResEpochSpikesSmDs, column_names, k1, as_dict=True):

                # Get available sort group IDs
                available_sort_group_ids = (ResEpochSpikesSmDs & k2).fetch("sort_group_id")

                # If all sort groups present, insert entry
                if check_membership(expected_sort_group_ids, available_sort_group_ids, tolerate_error=True):
                    insert_key = {k: v for k, v in {**k1, **k2}.items() if k in self.primary_key}
                    self.check_consistent_key(insert_key)
                    potential_keys.append(insert_key)
                else:
                    missing_sort_group_ids = [x for x in expected_sort_group_ids if
                                              x not in available_sort_group_ids]
                    missing_summary.append(
                        (k1["nwb_file_name"], k1["epoch"], tuple(missing_sort_group_ids)))

        # Print cases of missing sort group IDs if indicated
        verbose = key_filter.pop("verbose", True)
        if verbose:
            print(
                f"Cases of missing sort group IDs in ResEpochSpikesSmDs preventing population of FRVecSel: "
                f"(Note that the following only includes entries in immediate upstream tables (so files of "
                f"interest that are not in those upstream tables will not be printed here))")
            missing_summary = list(set(missing_summary))
            missing_summary.sort(key=lambda x: x[0])  # restore an order after set operation
            for nwb_file_name, epoch, sort_group_ids in missing_summary:
                print(
                    f"{nwb_file_name}, ep{epoch} is missing sort group IDs from ResEpochSpikesSmDs: "
                    f"{sort_group_ids}")

        return potential_keys

    def _get_bad_keys(self):
        return [key for key in self.fetch("KEY") if not self.check_consistent_key(key)] + [
               # entries in main table that have no entry in part table
               key for key in FRVec().fetch("KEY") if len(FRVec.Upstream & key) == 0]

    def delete_(self, key, safemode=True):
        delete_(self, [FRVec], key, safemode)


@schema
class FRVec(ComputedBase):
    definition = """
    # Firing rate vectors in an epoch
    -> FRVecSel
    """

    class Upstream(PartBase):
        definition = """
        # Achieves dependence on upstream tables
        -> FRVec
        -> ResEpochSpikesSmDs  # supplies firing rates
        -> BrainRegionUnits  # supplies information about brain region and units
        """

    @staticmethod
    def get_sort_group_unit_ids_map(key):
        # Get map from sort group to units
        return (BrainRegionUnits & key).fetch1("sort_group_unit_ids_map")

    @classmethod
    def check_sort_group_unit_ids_map(cls, key):

        # Check that sort group ids appear in spikes table no more than once
        sort_group_ids = (ResEpochSpikesSmDs & key).fetch("sort_group_id")
        check_all_unique(sort_group_ids)

        # Check that all sort groups present in spikes table
        sort_group_unit_ids_map = cls.get_sort_group_unit_ids_map(key)
        check_membership(sort_group_unit_ids_map.keys(), sort_group_ids, "sort group ids in BrainRegionUnits entry",
                         "sort group ids in cohort from ResEpochSpikesSmDs")

    @classmethod
    def _get_firing_rate_across_x_inputs(cls, key, label_brain_region):
        # Get inputs to methods named like "get_firing_rate_across..."

        # Get sort group unit ids map
        # First check sort group unit ids map interaction with spikes table
        cls.check_sort_group_unit_ids_map(key)
        sort_group_unit_ids_map = cls.get_sort_group_unit_ids_map(key)

        # Get params to add brain region label to df if indicated. Both params below must be defined for brain region
        # label to be added
        label_name = None
        sort_group_id_label_map = None
        if label_brain_region:
            label_name = "brain_region"
            # Make map from sort group to brain region
            sort_group_id_label_map = {
                sort_group_id: key[label_name] for sort_group_id in sort_group_unit_ids_map.keys()}

        return sort_group_unit_ids_map, sort_group_id_label_map, label_name

    def make(self, key):

        # Check key valid before inserting into table
        self.check_sort_group_unit_ids_map(key)

        # Insert into main table
        # Note that we dont store firing rate vectors because they are the same as in upstream source, just pooled
        # in a particular way. Storing these vectors again would be duplicating data, and this could get
        # costly from a storage perspective. Along these lines, we also dont score the z scored firing rates, since
        # the z score is such a simple transformation that we can quickly apply before returning data
        insert1_print(self, key)

        # Insert into part table
        for sort_group_id in self.get_sort_group_unit_ids_map(key).keys():
            key.update({"sort_group_id": sort_group_id})
            insert1_print(self.Upstream, key)

    def firing_rate_across_sort_groups(self, key=None, label_brain_region=True, populate_tables=True):
        # Get firing rates for units across sort groups
        # Return dataframe with firing rate and brain region, and dataframe with time vector

        # Get key if not passed
        if key is None:
            key = self.fetch1("KEY")

        # Check inputs
        if "sort_group_id" in key:
            raise Exception(f"sort_group_id must not be in key")

        # Get inputs to firing rate across function
        sort_group_unit_ids_map, sort_group_id_label_map, label_name = self._get_firing_rate_across_x_inputs(
            key, label_brain_region)

        return ResEpochSpikesSmDs().firing_rate_across_sort_groups(
            key, sort_group_unit_ids_map, sort_group_id_label_map, label_name, populate_tables)

    def firing_rate_vector_across_sort_groups(self, key=None, populate_tables=True):
        firing_rate_df, time_vector = self.firing_rate_across_sort_groups(key, populate_tables=populate_tables)
        return pd.DataFrame(
            np.vstack(firing_rate_df["firing_rate"]).T, index=time_vector, columns=firing_rate_df.index)

    def firing_rate_across_sort_groups_epochs(
            self, key, label_brain_region=True, populate_tables=True, concatenate=True):
        # Get firing rates for units across sort groups and epochs, optionally concatenated across epochs

        # Assemble keys, one for each epoch/res_epoch_time_bins_pool_param_name pair
        keys = ResTimeBinsPoolCohortParams().get_keys_with_cohort_params(key)

        # Get inputs to function. Check that same across keys, then take one set
        data_list = [self._get_firing_rate_across_x_inputs(key, label_brain_region) for key in keys]
        for x in list(zip(*data_list)):
            check_equality(x)
        sort_group_unit_ids_map, sort_group_id_label_map, label_name = data_list[0]

        # Get epochs
        epochs = (ResTimeBinsPoolCohortParams & key).fetch1("epochs")

        # Get firing rate across sort groups and epochs. Note that use keyword arguments here since
        # firing rate function set up to take key or keys
        fr_df, time_vector = ResEpochSpikesSmDs().firing_rate_across_sort_groups_epochs(
            epochs, sort_group_unit_ids_map, keys=keys, sort_group_id_label_map=sort_group_id_label_map,
            label_name=label_name, populate_tables=populate_tables)

        # If indicated, return dataframe with firing rates concatenated across epochs, and dataframe with epoch
        # corresponding to each time sample
        if concatenate:
            # Sort dfs by epoch to ensure that when concatenate data this occurs across epochs in chronological order
            fr_df = fr_df.set_index("epoch").sort_index()
            time_vector = time_vector.reset_index().set_index("epoch").sort_index()

            # Concatenate time vector across epochs
            time_vector_concat = np.hstack(time_vector.time_vector)

            # Concatenate firing rates across epochs
            unit_names = np.unique(fr_df["unit_name"])
            data_list = []
            for unit_name in unit_names:
                df_subset = df_filter_columns(fr_df, {"unit_name": unit_name})
                check_single_element(
                    df_subset.brain_region)  # check brain region same across instances of this unit across epochs
                data_list.append(
                    list(map(np.hstack, list(zip(*[
                        ([epoch] * len(fr), fr) for epoch, fr in df_subset.firing_rate.items()])))))
            epoch_concat, fr_concat = list(zip(*data_list))

            # Put concatenated firing rates and time vector in one dataframe
            fr_concat_df = pd.DataFrame(np.asarray(fr_concat).T, index=time_vector_concat, columns=unit_names)
            fr_concat_df.index.name = "time"

            # Make dataframe with epoch corresponding to samples
            epoch_df = pd.DataFrame.from_dict(
                {"epoch": unpack_single_vector(epoch_concat), "time": time_vector_concat}).set_index("time")

            # Collect dfs to return: df with concatenated firing rates, and df with epoch corresponding to those samples
            dfs_concat = {"fr_concat_df": fr_concat_df, "epoch_vector": epoch_df}

        # Otherwise, return firing rate df with separate frs for each epoch, and df with time vector for each epoch
        else:
            dfs_concat = {"fr_df": fr_df, "time_vector": time_vector}

        return namedtuple("dfs_concat", dfs_concat)(**dfs_concat)

    def firing_rate_across_sort_groups_epochs_brain_regions(self, key, brain_regions, populate_tables=True):
        # Get firing rates for units across sort groups, epochs, and brain regions

        # Check passed brain regions all unique
        check_all_unique(brain_regions)

        # Indicate we want a brain region column in df
        label_brain_region = True

        # Get firing rate dfs and time vectors
        firing_rate_dfs, time_vector_dfs = list(zip(*[
            self.firing_rate_across_sort_groups_epochs(
                {**key, **{"brain_region": brain_region}}, label_brain_region, populate_tables, concatenate=False)
            for brain_region in brain_regions]))
        return pd.concat(firing_rate_dfs), unpack_single_df(time_vector_dfs)

    def get_mua(self, key):
        fr_df, time_vector = self.firing_rate_across_sort_groups(key)
        return pd.Series(np.sum(np.vstack(fr_df["firing_rate"]), axis=0) / len(fr_df), index=time_vector)

    def delete_(self, key, safemode=True):
        from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity import FRDiffVecCosSimSel
        from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector_euclidean_distance import FRVecEucDistSel
        from src.jguides_2024.firing_rate_vector.jguidera_path_firing_rate_vector import PathFRVecSel
        from src.jguides_2024.firing_rate_vector.jguidera_well_event_firing_rate_vector import TimeRelWAFRVecSel
        from src.jguides_2024.firing_rate_vector.jguidera_post_delay_firing_rate_vector import RelPostDelFRVecSel
        delete_(self, [
            FRDiffVecCosSimSel, FRVecEucDistSel, PathFRVecSel, TimeRelWAFRVecSel, RelPostDelFRVecSel], key, safemode)


def populate_jguidera_firing_rate_vector(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_firing_rate_vector"
    upstream_schema_populate_fn_list = [populate_jguidera_res_spikes, populate_jguidera_unit]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_firing_rate_vector():
    from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector import \
        drop_jguidera_firing_rate_difference_vector
    from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector_euclidean_distance import \
        drop_jguidera_firing_rate_vector_euclidean_distance
    from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector_embedding import \
        drop_jguidera_firing_rate_vector_embedding
    from src.jguides_2024.firing_rate_vector.jguidera_path_firing_rate_vector import \
        drop_jguidera_path_firing_rate_vector
    from src.jguides_2024.firing_rate_vector.jguidera_well_event_firing_rate_vector import \
        drop_jguidera_well_event_firing_rate_vector

    drop_jguidera_firing_rate_difference_vector()
    drop_jguidera_firing_rate_vector_euclidean_distance()
    drop_jguidera_firing_rate_vector_embedding()
    drop_jguidera_path_firing_rate_vector()
    drop_jguidera_well_event_firing_rate_vector()
    schema.drop()
