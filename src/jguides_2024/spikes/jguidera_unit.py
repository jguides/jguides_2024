import copy

import datajoint as dj
import numpy as np
from spyglass.common import Electrode
from spyglass.spikesorting import CuratedSpikeSorting

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_subject_id
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, \
    get_table_secondary_key_names, get_unit_name, split_unit_names, \
    get_key_filter, split_curation_name, delete_, get_default_param, split_unit_name, make_param_name, get_table_name
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_brain_region import SortGroupTargetedLocation, BrainRegionSortGroup, \
    BrainRegionCohort, CurationSet, \
    ElectrodeGroupTargetedLocation, get_targeted_location_from_brain_region
from src.jguides_2024.metadata.jguidera_epoch import EpochCohort, RunEpoch, EpochsDescription
from src.jguides_2024.metadata.jguidera_histology import ValidShank, LivermoreD2
from src.jguides_2024.metadata.jguidera_metadata import JguideraNwbfile
from src.jguides_2024.spike_sorting_curation.jguidera_spikesorting import DefineSortInterval
from src.jguides_2024.spike_sorting_curation.spikesorting_helpers import get_peak_ch_map
from src.jguides_2024.spikes.jguidera_spikes import EpochMeanFiringRate, EpochSpikeTimesRelabel
from src.jguides_2024.utils.df_helpers import df_from_data_list, df_filter_columns, df_pop
from src.jguides_2024.utils.dict_helpers import dict_comprehension, dict_comprehension_repeated_keys
from src.jguides_2024.utils.for_loop_helpers import print_iteration_progress
from src.jguides_2024.utils.list_helpers import check_single_element, check_return_single_element
from src.jguides_2024.utils.set_helpers import check_set_equality, check_membership
from src.jguides_2024.utils.stats_helpers import random_sample
from src.jguides_2024.utils.vector_helpers import check_all_unique, unpack_single_element

schema = dj.schema("jguidera_unit")


@schema
class EpsUnitsParams(SecKeyParamsBase):
    definition = """
    # Parameters for EpsUnits
    eps_units_param_name : varchar(80)
    ---
    min_epoch_mean_firing_rate : float
    """

    def _default_params(self):
        return [[0], [.1]]


# TODO (feature): restrict insertion into this table to entries that will successfully populate downstream (have enough units
#  in cases of unit subset)
@schema
class EpsUnitsSel(SelBase):
    definition = """
    # Selection from upstream tables for EpsUnits
    -> EpochCohort
    -> SortGroupTargetedLocation
    -> EpsUnitsParams
    curation_name : varchar(80)
    """

    def _get_potential_keys(self, key_filter=None):
        # Insert defaults
        if key_filter is None:
            key_filter = dict()
        SortGroupTargetedLocation().populate(key_filter)

        # Continue if entries in mean firing rate table for all epochs in cohort, and use only
        # curation names that correspond to epochs that encompass epochs in cohort. Achieve
        # by defining specific cases we want using curation set
        brain_region_cohort_name = "all_targeted"
        curation_set_name = "runs_analysis_v1"
        brain_region_sort_group_id_map = BrainRegionSortGroup().get_brain_region_sort_group_id_map()

        # Define keys to loop through
        keys = super()._get_potential_keys(key_filter)
        num_keys = len(keys)

        # Print out progress if large number of keys
        large_num_keys_thresh = 1000
        verbose = False
        if num_keys > large_num_keys_thresh:
            verbose = True

        if verbose:
            print(f"Looping through {num_keys} keys...")

        potential_keys = []
        for idx, key in enumerate(keys):

            if verbose:
                print_iteration_progress(idx, num_keys, 20)

            nwb_file_name = key["nwb_file_name"]
            brain_region_sort_group_id_map_subset = df_filter_columns(brain_region_sort_group_id_map, {
                "nwb_file_name": nwb_file_name, "sort_group_id": key["sort_group_id"]})
            if len(brain_region_sort_group_id_map_subset) == 0:
                continue
            brain_region = unpack_single_element(brain_region_sort_group_id_map_subset["brain_region"].values)

            # Continue if brain region not in brain region cohort we want
            if brain_region not in (BrainRegionCohort() & {
                "nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name}
            ).fetch1("brain_regions"):
                continue
            epochs = (EpochCohort & key).fetch1("epochs")

            # Case 1: individual run epochs
            if len(epochs) == 1:
                table_subset = (CurationSet() & {
                    "nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name,
                    "curation_set_name": curation_set_name})
                if len(table_subset) > 0:

                    curation_set_df = table_subset.fetch1_dataframe()

                    epoch = (EpochCohort & key).get_epoch()
                    if epoch in (RunEpoch & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch("epoch"):
                        epochs_description = EpochsDescription().get_single_run_description(
                            nwb_file_name, epoch)
                        curation_name = df_pop(
                            curation_set_df, {"brain_region": brain_region, "epochs_description": epochs_description},
                                               "curation_name")
                        curation_epochs = DefineSortInterval.get_epochs_for_curation_name(nwb_file_name, curation_name)

                        epoch_mean_fr_populated = all([len(EpochMeanFiringRate & {**key, **{
                            "epoch": epoch, "curation_name": curation_name}}) > 0 for epoch in epochs])
                        valid_epochs = all([x in curation_epochs for x in epochs])

                        if epoch_mean_fr_populated and valid_epochs:
                            key.update({"curation_name": curation_name})
                            potential_keys.append(copy.deepcopy(key))
        return potential_keys


# Note that EpsUnits contains both single units and multiunits
@schema
class EpsUnits(ComputedBase):
    definition = """
    # Units active across epochs
    -> EpsUnitsSel
    ---
    unit_ids : blob
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream table with single epoch information 
        -> EpsUnits
        -> EpochMeanFiringRate
        """

    def make(self, key):
        min_epoch_mean_firing_rate = (EpsUnitsParams & key).fetch1("min_epoch_mean_firing_rate")
        epochs = (EpochCohort & key).fetch1("epochs")
        unit_ids = []
        for epoch in epochs:
            unit_ids_epoch = set()  # default (if no units in eopch)
            # Exit if no entry in EpochMeanFiringRate
            epoch_key = {**key, **{"epoch": epoch}}
            if len(EpochMeanFiringRate & epoch_key) == 0:
                print(f"Could not populate EpsUnits for key {key} because no corresponding entry in "
                      f"EpochMeanFiringRate. Exiting")
                return
            # Apply minimum mean firing rate threshold
            df = (EpochMeanFiringRate & epoch_key).fetch1_dataframe()
            if len(df) > 0:  # if units for this epoch, update unit ids
                unit_ids_epoch = set(df[df["mean_firing_rate"] >= min_epoch_mean_firing_rate].index)
            unit_ids.append(unit_ids_epoch)

        # Take intersection of units across epochs if any were found
        if len(unit_ids) > 0:  # units
            unit_ids = list(set.intersection(*unit_ids))  # units active across epochs

        # Insert into main table
        insert1_print(self, {**key, **{"unit_ids": unit_ids}})

        # Insert into part table
        for epoch in epochs:
            key.update({"epoch": epoch})
            insert1_print(self.Upstream, {**key, **(EpochMeanFiringRate & key).fetch1("KEY")})

    # Override parent class so can populate table upstream of parts table
    def populate_(self, **kwargs):
        EpochMeanFiringRate().populate_(**kwargs)
        super().populate_(**kwargs)

    def get_sort_group_unit_ids_map(self, sort_group_ids=None, single_unit=True):
        # Return map from sort group id to unit ids
        # First, check that same value at primary keys other than sort group id
        [check_single_element(x) for x in self.fetch(*[x for x in self.primary_key if x != "sort_group_id"])]
        # Make map
        sort_group_unit_ids_map = dict_comprehension(*self.fetch("sort_group_id", "unit_ids"))
        # Restrict to certain sort group ids if passed
        if sort_group_ids is not None:
            # Check that all passed sort groups are available ones
            check_membership(sort_group_ids, sort_group_unit_ids_map.keys(), "passed sort group ids",
                             "available sort group ids")
            sort_group_unit_ids_map = {k: sort_group_unit_ids_map[k] for k in sort_group_ids}
        # Restrict to units with "accept" label, if indicated
        if single_unit:
            nwb_file_name, curation_name = [x.single_element for x in list(
                map(check_return_single_element, self.fetch("nwb_file_name", "curation_name")))]
            sort_group_unit_ids_map = {
                sort_group_id: [unit_id for unit_id in unit_ids if get_unit_label(
                    nwb_file_name, sort_group_id, unit_id, curation_name) == "accept"]
                for sort_group_id, unit_ids in sort_group_unit_ids_map.items()}
        return sort_group_unit_ids_map

    @staticmethod
    def get_unit_names(sort_group_unit_ids_map):
        return [get_unit_name(sort_group_id, unit_id) for sort_group_id, unit_ids in
                sort_group_unit_ids_map.items() for unit_id in unit_ids]

    def _get_unit_names(self, sort_group_ids=None, single_unit=True):
        return self.get_unit_names(
            self.get_sort_group_unit_ids_map(sort_group_ids, single_unit))

    @staticmethod
    def unit_names_to_sort_group_unit_ids_map(unit_names, sort_unit_ids=True):
        sort_group_ids, unit_ids = split_unit_names(unit_names)
        return dict_comprehension_repeated_keys(sort_group_ids, unit_ids, sort_unit_ids)

    def rand_subset_units(self, unit_subset_size, single_unit=True, sort_group_ids=None, replace=False,
                          sort_unit_ids=True, as_dict=False, tolerate_error=False):
        # Get unit names
        unit_names = self._get_unit_names(sort_group_ids, single_unit)
        # Take random subset
        # TODO: could take in random seed (and could pass unit_subset_iteration from BrainRegionUnitsParams for this)
        rand_unit_names = random_sample(unit_names, unit_subset_size, replace, tolerate_error=tolerate_error)
        # Exit function if no random sample
        if rand_unit_names is None:
            return
        # Return as map from sort group id to unit ids if indicated
        if as_dict:
            return self.unit_names_to_sort_group_unit_ids_map(rand_unit_names, sort_unit_ids)
        # Otherwise return unit names
        return rand_unit_names

    def delete_(self, key, safemode=True):
        delete_(self, [BrainRegionUnitsSel], key, safemode)


@schema
class BrainRegionUnitsParams(SecKeyParamsBase):
    definition = """
    # Parameters for BrainRegionUnits
    brain_region_units_param_name : varchar(80)
    ---
    -> EpsUnitsParams  # describes min firing rate threshold
    epochs_description : varchar(40)
    unit_subset = 0 : bool
    unit_subset_type = "all" : varchar(40)
    unit_subset_size = NULL : int
    unit_subset_iteration = NULL : int
    """

    def insert1(self, key, **kwargs):
        self._check_params(key)
        super().insert1(key, **kwargs)

    def insert_epochs(
            self, nwb_file_name, epochs, min_epoch_mean_firing_rate=0, unit_subset_type="all_targeted",
            unit_subset_size=None, unit_subset_iteration=None):

        # Define boolean indicating whether taking subset of units or not
        unit_subset = True
        if unit_subset_type == "all":
            unit_subset = False

        # Insert into upstream table
        EpochsDescription().insert_runs(nwb_file_name, epochs)

        # Insert into current table
        epochs_description = EpochsDescription().lookup_epochs_description(nwb_file_name, epochs)
        eps_units_param_name = EpsUnitsParams().lookup_param_name([min_epoch_mean_firing_rate])
        key = {"eps_units_param_name": eps_units_param_name, "epochs_description": epochs_description,
               "unit_subset": unit_subset, "unit_subset_type": unit_subset_type,
               "unit_subset_size": unit_subset_size, "unit_subset_iteration": unit_subset_iteration}
        secondary_key_subset_map = {k: key[k] for k in self._param_name_secondary_key_columns()}
        key.update({"brain_region_units_param_name": self._make_param_name(secondary_key_subset_map)})
        self.insert1(key)

    @classmethod
    def _check_params(cls, key):
        # Check that unit subset size is None if unit subset is False
        if "unit_subset" in key:
            if not key["unit_subset"] and key["unit_subset_size"] is not None:
                raise Exception(f"unit_subset_size must be None if no unit subset")
        # Check unit_subset_type valid
        check_membership(
            [key["unit_subset_type"]], cls._valid_unit_subset_type(), "passed unit subset type",
            "valid unit subset types")
        if key["unit_subset_type"] == "all":
            if not all([x is None for x in [key["unit_subset_size"], key["unit_subset_iteration"]]]):
                raise Exception(f"unit_subset_size and unit_subset_iteration must be None if unit_subset_type is all")

    @staticmethod
    def _valid_epochs_descriptions():
        return EpochsDescription.valid_epochs_descriptions()

    @staticmethod
    def _combination_param_sets():
        return [[True, "target_region", None, None]] + [
            [True, "rand_target_region", 50, unit_subset_num] for unit_subset_num in np.arange(0, 10)]

    def _default_params(self):
        # Define a primary set of parameters
        combination_param_sets = self._combination_param_sets()
        epochs_descriptions = self._valid_epochs_descriptions()
        primary_min_epoch_mean_frs = [.1]
        primary_eps_units_param_names = [EpsUnitsParams().lookup_param_name([x]) for x in primary_min_epoch_mean_frs]

        # Define a separate set of parameters just for GLM analysis, where want all units
        glm_combination_param_set = [False, "target_region", None, None]
        glm_params = [[EpsUnitsParams().lookup_param_name([0]), epochs_description] + glm_combination_param_set
                      for epochs_description in epochs_descriptions]

        # Return parameters
        return [[eps_units_param_name, epochs_description] + combination_param_set
                for eps_units_param_name in primary_eps_units_param_names
                for epochs_description in epochs_descriptions
                for combination_param_set in combination_param_sets] + glm_params

    def _param_name_secondary_key_columns(self):
        return [x for x in get_table_secondary_key_names(self) if x != "unit_subset"]

    def _get_eps_description_key(self, nwb_file_name, brain_region_units_param_name=None):
        if brain_region_units_param_name is None:
            brain_region_units_param_name = self.fetch1("brain_region_units_param_name")
        key = {"nwb_file_name": nwb_file_name, "brain_region_units_param_name": brain_region_units_param_name}
        # Add epochs description
        key.update({"epochs_description": (self & key).fetch1("epochs_description")})
        return key

    def get_epochs(self, nwb_file_name, brain_region_units_param_name=None):
        return (EpochsDescription & self._get_eps_description_key(
            nwb_file_name, brain_region_units_param_name)).fetch1("epochs")

    def get_epochs_id(self, nwb_file_name, brain_region_units_param_name=None):
        return (EpochsDescription & self._get_eps_description_key(
            nwb_file_name, brain_region_units_param_name)).fetch1("epochs_id")

    @staticmethod
    def _valid_unit_subset_type():
        return ["all", "rand", "target_region", "rand_target_region"]

    def _make_param_name(self, secondary_key_subset_map, separating_character="_", tolerate_non_unique=True):
        # Check inputs
        # ...Check that exactly the table secondary keys used to make param name were passed
        check_set_equality(secondary_key_subset_map.keys(), self._param_name_secondary_key_columns(),
                           "passed secondary key subset keys", "table secondary key names")
        # ...Check unit subset type valid
        check_membership([secondary_key_subset_map["unit_subset_type"]], self._valid_unit_subset_type(),
                         "list with passed unit subset type", "available unit subset types")

        # Make param name using secondary key values
        # Approach: only include unit_subset_size and unit_subset_iteration if random subset
        # This will make it more convenient to query this and downstream tables.
        # ...if not random subset
        if secondary_key_subset_map["unit_subset_type"] not in ["rand", "rand_target_region"]:  # not random subset
            non_unit_subset_names = [x for x in self._param_name_secondary_key_columns() if x not in [
                "unit_subset_size", "unit_subset_iteration"]]
            return "_".join([secondary_key_subset_map[k] for k in non_unit_subset_names])
        # ...if random subset, string together secondary key values except for unit subset which provides no additional
        # information
        secondary_key_subset_map = {k: secondary_key_subset_map[k] for k in self._param_name_secondary_key_columns()}
        return super()._make_param_name(secondary_key_subset_map, separating_character, tolerate_non_unique)

    def lookup_epochs_param_name(
            self, nwb_file_name, epochs, min_epoch_mean_firing_rate=0, unit_subset_type=None, unit_subset_size=None,
            unit_subset_iteration=None, insert_upstream=False):

        # Get inputs if not passed
        if unit_subset_type is None:
            unit_subset_type = "target_region"  # default

        # Check inputs
        self._check_params(
            {"unit_subset_type": unit_subset_type, "unit_subset_size": unit_subset_size,
             "unit_subset_iteration": unit_subset_iteration})

        # Look up param name in table for an arbitrary set of epochs
        # ...First insert into table if indicated
        if insert_upstream:
            if all([epoch in (RunEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch") for epoch in epochs]):
                EpochsDescription().insert_runs(nwb_file_name, epochs)  # insert runs
            else:
                raise Exception(f"Need to write function for inserting epochs into EpochsDescription when "
                                f"epochs not all runs")
        epochs_description = EpochsDescription().lookup_epochs_description(nwb_file_name, epochs)
        eps_units_param_name = EpsUnitsParams().lookup_param_name([min_epoch_mean_firing_rate])

        return BrainRegionUnitsParams().lookup_param_name(
            [eps_units_param_name, epochs_description, unit_subset_type, unit_subset_size, unit_subset_iteration])

    def lookup_single_epoch_param_name(
            self, nwb_file_name, epoch, min_epoch_mean_firing_rate=0, unit_subset_type=None, unit_subset_size=None,
            unit_subset_iteration=None):
        # Return param name
        return self.lookup_epochs_param_name(nwb_file_name, [epoch], min_epoch_mean_firing_rate, unit_subset_type,
                                             unit_subset_size, unit_subset_iteration)

    def lookup_runs_param_name(self, nwb_file_name, min_epoch_mean_firing_rate=0, unit_subset_type=None,
                               unit_subset_size=None, unit_subset_iteration=None):
        # Look up param name in table for all run epochs
        epochs = (RunEpoch & {"nwb_file_name": nwb_file_name}).fetch("epoch")
        return self.lookup_epochs_param_name(nwb_file_name, epochs, min_epoch_mean_firing_rate, unit_subset_type,
                                             unit_subset_size, unit_subset_iteration)


@schema
class BrainRegionUnitsSel(SelBase):
    definition = """
    # Selection from upstream tables for BrainRegionUnits
    -> JguideraNwbfile  # nwb file name
    -> BrainRegionSortGroup  # supplies map from brain region to sort groups
    -> BrainRegionUnitsParams  # units params (min firing rate threshold, across epochs params, unit subset params) 
    curation_name : varchar(80)
    """

    @staticmethod
    def get_eps_units_key(key):
        # Add information in brain_region_units_param_name for querying EpsUnits: eps units param name and epochs id
        eps_units_param_name = (BrainRegionUnitsParams & key).fetch1("eps_units_param_name")
        epochs_id = BrainRegionUnitsParams().get_epochs_id(key["nwb_file_name"], key["brain_region_units_param_name"])
        return {**key, **{"eps_units_param_name": eps_units_param_name, "epochs_id": epochs_id}}

    def check_sort_group_ids_present(self, key, tolerate_error=True):
        # Check if all sort group ids are in EpsUnits for epochs, to ensure units from all available
        # sort groups make it into entries in BrainRegionUnits

        # Get expected sort group IDs
        sort_group_ids = (BrainRegionSortGroup & key).fetch1("sort_group_ids")
        # Get sort group ids in EpsUnits
        eps_units_sort_group_ids = (EpsUnits & self.get_eps_units_key(key)).fetch("sort_group_id")
        # Check that one set of sort group ids for current key (not necessary with current table setup since only free
        # parameter is sort group id, but useful in case things change)
        check_all_unique(eps_units_sort_group_ids)
        # Check that all expected sort group ids are in EpsUnits
        return check_membership(sort_group_ids, eps_units_sort_group_ids, "sort group ids in BrainRegionSortGroup",
                                "sort group ids in EpsUnits", tolerate_error)

    def _get_potential_keys(self, key_filter=None):

        key_filter = get_key_filter(key_filter)

        valid_brain_region_cohort_name = "all_targeted"
        valid_curation_set_name = "runs_analysis_v1"

        # Populate upstream tables
        EpochsDescription().insert_defaults(key_filter=key_filter)
        keys = []
        for k1 in ((JguideraNwbfile * BrainRegionSortGroup) & key_filter).fetch("KEY"):

            k1.update(key_filter)

            for k2 in (EpochsDescription & k1).fetch("KEY"):

                k2.update(k1)

                brup_keys = (BrainRegionUnitsParams & k2).fetch("KEY")

                # Limit to target_region
                brup_keys = [k for k in brup_keys if (BrainRegionUnitsParams & k).fetch1("unit_subset_type")
                             in ["target_region", "rand_target_region"]]

                for k3 in brup_keys:

                    k3.update(k2)

                    curation_names = set((EpsUnits & self.get_eps_units_key(k3)).fetch("curation_name"))

                    if len(curation_names) == 0:
                        continue

                    # Limit to desired curation name
                    curation_set_key = {
                        "nwb_file_name": k3["nwb_file_name"],
                        "brain_region_cohort_name": valid_brain_region_cohort_name,
                        "curation_set_name": valid_curation_set_name}
                    valid_curation_name = (CurationSet & curation_set_key).get_curation_name(
                        k3["brain_region"], k3["epochs_description"], tolerate_no_entry=True)
                    if valid_curation_name is None:
                        continue

                    curation_names = [x for x in curation_names if x == valid_curation_name]

                    for curation_name in curation_names:

                        k3.update({"curation_name": curation_name})

                        # Insert into table if all sort group ids present
                        if self.check_sort_group_ids_present(k3):
                            keys.append({k: v for k, v in k3.items() if k in self.primary_key})

        # Return keys
        return keys

    def delete_(self, key, safemode=True):
        delete_(self, [BrainRegionUnits], key, safemode)


@schema
class BrainRegionUnitsFail(dj.Manual):
    definition = """
    # Failed population of BrainRegionUnits because not enough units to subsample
    -> BrainRegionUnitsSel
    """

    class ValidShank(dj.Part):
        definition = """
        # Achieves dependence on ValidShank
        -> BrainRegionUnitsFail
        -> ValidShank
        """


@schema
class BrainRegionUnits(ComputedBase):
    definition = """
    # Group of units in brain region across epochs, optionally subsampled
    -> BrainRegionUnitsSel
    ---
    sort_group_unit_ids_map = NULL : blob  # map from sort group id to unit ids
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves dependence on upstream table where single sort groups are primary key
        -> BrainRegionUnits
        -> EpsUnits
        """

    class ValidShank(dj.Part):
        definition = """
        # Achieves dependence on ValidShank
        -> BrainRegionUnits
        -> ValidShank
        """

    def make(self, key):

        # Immediately exit if already determined that cannot populate for this key (because not enough
        # units to subsample)
        if len(BrainRegionUnitsFail & key) > 0:
            return

        # Get sort group ids for this brain region
        sort_group_ids = (BrainRegionSortGroup & key).fetch1("sort_group_ids")

        # Check that all expected sort group ids are represented in EpsUnits
        selection_table = self._get_selection_table()
        selection_table().check_sort_group_ids_present(key, tolerate_error=False)

        # Define quantities for taking unit subset as indicated
        unit_subset_type = (BrainRegionUnitsParams & key).fetch1("unit_subset_type")
        eps_units_key = selection_table().get_eps_units_key(key)
        single_unit = True  # restrict to single units (do not include units marked as mua)
        unit_subset_size = (BrainRegionUnitsParams & key).fetch1("unit_subset_size")

        # Get electrode groups in this brain region
        targeted_location = get_targeted_location_from_brain_region(key["brain_region"])
        nwb_file_name = key["nwb_file_name"]
        electrode_group_names = (ElectrodeGroupTargetedLocation & {
            "nwb_file_name": nwb_file_name, "targeted_location": targeted_location}).fetch(
            "electrode_group_name")

        # Initialize key for querying table with valid shank length information
        subject_id = get_subject_id(key["nwb_file_name"])
        hist_key = {"subject_id": subject_id}

        # ...All units
        sort_group_unit_ids_map = (EpsUnits & eps_units_key).get_sort_group_unit_ids_map(
            sort_group_ids, single_unit)  # default

        # ...Unit subset
        # 1) random subset
        if unit_subset_type == "rand":
            sort_group_unit_ids_map = (EpsUnits & eps_units_key).rand_subset_units(
                unit_subset_size, single_unit, sort_group_ids=sort_group_ids, replace=False, as_dict=True,
                tolerate_error=True)
            # If could not subsample, store key in alternate table so can track these cases,
            # then exit function
            if sort_group_unit_ids_map is None:
                BrainRegionUnitsFail().insert1(key, skip_duplicates=True)
                # Insert into part tables relevant to only some entries
                if unit_subset_type in ["target_region", "rand_target_region"] and key[
                    "brain_region"] != "CA1_targeted":
                    for electrode_group_name in electrode_group_names:
                        hist_key.update({"electrode_group_name": electrode_group_name})
                        BrainRegionUnits.ValidShank.insert1({**key, **hist_key})
                return

        # 2) within target region based on histology
        elif unit_subset_type in ["target_region", "rand_target_region"]:

            # Check that brain region is one that is accounted for in code currently
            check_membership([key["brain_region"]], ["mPFC_targeted", "OFC_targeted", "CA1_targeted"])

            # If CA1 targeted, include based on tetrodes (individual electrodes of a tetrode are in same area)
            # Histology suggests CA1 tetrodes in target region the following, so no narrowing required:
            if subject_id in ["J16", "mango", "june", "fig", "peanut"] and key["brain_region"] == "CA1_targeted":
                pass

            # If mPFC or OFC targeted, include based on individual electrodes (individual electrode of a probe shank
            # could be in different areas)
            else:

                # Loop through electrode groups and get valid channels in each
                valid_chs = []
                for electrode_group_name in electrode_group_names:
                    # Get electrode IDs for this brain region
                    electrode_ids = (Electrode & {
                        "nwb_file_name": nwb_file_name, "electrode_group_name": electrode_group_name}).fetch(
                        "electrode_id")

                    # Get valid lens
                    hist_key.update({"electrode_group_name": electrode_group_name})
                    below_dorsal_limit_lens, below_ventral_limit_lens = (
                            ValidShank & hist_key).fetch1("below_dorsal_limit_lens", "below_ventral_limit_lens")

                    # Get valid channels (those within target region)
                    valid_chs += list(electrode_ids[LivermoreD2().get_valid_idxs(
                        below_dorsal_limit_lens, below_ventral_limit_lens)])

                # Update units to be those within valid range from probe tip to edge of target region
                for sort_group_id, unit_ids in sort_group_unit_ids_map.items():
                    peak_ch_map = get_peak_ch_map(key["nwb_file_name"], sort_group_id)  # map from unit id to peak ch
                    valid_unit_ids = [unit_id for unit_id, peak_ch in peak_ch_map.items() if peak_ch in valid_chs]
                    sort_group_unit_ids_map[sort_group_id] = [x for x in unit_ids if x in valid_unit_ids]

            # Subsample if indicated
            if unit_subset_type == "rand_target_region":
                # Get unit names across sort groups
                unit_names = EpsUnits.get_unit_names(sort_group_unit_ids_map)
                # Get random sample
                unit_names_sample = random_sample(unit_names, unit_subset_size, replace=False, tolerate_error=True)
                # If could not subsample, store key in alternate table so can track these cases, then exit function
                if unit_names_sample is None:
                    BrainRegionUnitsFail().insert1(key, skip_duplicates=True)
                    return
                # Otherwise use random sample
                sort_group_unit_ids_map = EpsUnits.unit_names_to_sort_group_unit_ids_map(unit_names_sample)

        # Insert into main table
        insert1_print(self, {**key, **{"sort_group_unit_ids_map": sort_group_unit_ids_map}})

        # Insert into the part table relevant for all entries
        parts_key = copy.deepcopy(eps_units_key)
        for sort_group_id in sort_group_ids:
            parts_key.update({"sort_group_id": sort_group_id})
            insert1_print(self.Upstream, {**key, **parts_key})

        # Insert into part tables relevant to only some entries
        if unit_subset_type in ["target_region", "rand_target_region"] and key["brain_region"] != "CA1_targeted":
            for electrode_group_name in electrode_group_names:
                hist_key.update({"electrode_group_name": electrode_group_name})
                BrainRegionUnits.ValidShank.insert1({**key, **hist_key})

    def get_unit_name_df(self, nwb_file_name, brain_region_units_param_name, brain_region_cohort_name,
                         curation_set_name, epochs_description=None):

        key = {"nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name,
               "curation_set_name": curation_set_name}
        brain_regions = (BrainRegionCohort & key).fetch1("brain_regions")

        data_list = []
        for brain_region in brain_regions:

            # Get curation name
            curation_names_df = (CurationSet & key).fetch1_dataframe()
            df_key = {"brain_region": brain_region}
            if epochs_description is not None:
                df_key.update({"epochs_description": epochs_description})

            curation_name = check_return_single_element(df_filter_columns(
                curation_names_df, df_key).curation_name.values).single_element

            sort_group_unit_ids_map = (self & {
                "nwb_file_name": nwb_file_name, "brain_region_units_param_name": brain_region_units_param_name,
                "brain_region": brain_region, "curation_name": curation_name}).fetch1(
                "sort_group_unit_ids_map")

            for sort_group_id, unit_ids in sort_group_unit_ids_map.items():
                for unit_id in unit_ids:
                    data_list.append(
                        (get_unit_name(sort_group_id, unit_id), brain_region, sort_group_id, unit_id,curation_name))

        return df_from_data_list(data_list, [
            "unit_name", "brain_region", "sort_group_id", "unit_id", "curation_name"]).set_index("unit_name")

    def fetch1_sort_group_unit_ids_map(self, key, populate_tables=False):
        # Returns None for cases where could not populate main table (due to too few units to subsample specified
        # number)

        # Populate table if indicated
        if populate_tables:
            self.populate(key)

        # If entry in BrainRegionUnitsFail, return None
        if len(BrainRegionUnitsFail & key) == 1:
            return None

        # Otherwise return sort_group_unit_ids_map for single entry in main table
        return (BrainRegionUnits & key).fetch1("sort_group_unit_ids_map")

    def get_num_units(self, key):

        # Get map from sort group ID to unit IDS, where map is None is doesnt exist
        sort_group_unit_ids_map = self.fetch1_sort_group_unit_ids_map(key)

        # Return nan if too few units
        if sort_group_unit_ids_map is None:
            return np.nan

        # Otherwise return number of units
        return len(np.concatenate(list(sort_group_unit_ids_map.values())))

    def delete_(self, key, safemode=True):
        from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVec
        delete_(self, [FRVec], key, safemode)

    def populate_(self,**kwargs):
        EpochSpikeTimesRelabel().populate_(**kwargs)  # use populate_ to populate epoch spike times relabel params table
        super().populate_(**kwargs)

    def drop(self):
        from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import drop_jguidera_firing_rate_vector
        drop_jguidera_firing_rate_vector()
        super().drop()


# Not fully a params table but similar
@schema
class BrainRegionUnitsCohortType(SecKeyParamsBase):
    definition = """
    # Holds parameters for group of brain_region_units_param_names
    brain_region_units_cohort_type : varchar(100)
    ---
    -> EpsUnitsParams
    unit_subset : bool
    unit_subset_type : varchar(40)
    unit_subset_size = NULL : int
    unit_subset_iterations : blob
    """

    def insert_defaults(self, **kwargs):
        keys = []

        min_epoch_mean_firing_rate = .1
        eps_units_param_name = EpsUnitsParams().lookup_param_name([min_epoch_mean_firing_rate])

        # no unit subset
        keys.append({
            "eps_units_param_name": eps_units_param_name,
            "unit_subset": 0,
            "unit_subset_type": "target_region",
            "unit_subset_size": None,
            "unit_subset_iterations": [None]})

        # unit subset (just one)
        keys.append({
            "eps_units_param_name": eps_units_param_name,
            "unit_subset": 1,
            "unit_subset_type": "rand_target_region",
            "unit_subset_size": 50,
            "unit_subset_iterations": [0]})

        # unit subsets (two)
        keys.append({
            "eps_units_param_name": eps_units_param_name,
            "unit_subset": 1,
            "unit_subset_type": "rand_target_region",
            "unit_subset_size": 50,
            "unit_subset_iterations": np.arange(0, 10)})

        # Add cohort type names for each key
        for key in keys:
            key.update({"brain_region_units_cohort_type": self._make_param_name(key)})

        # Insert into table
        for key in keys:
            self.insert1(key, skip_duplicates=True)

    @classmethod
    def _unit_subset_iterations_map(cls):
        # make map from unit_subset_iterations to param name

        # a single subset of units: return "single_iteration" plus unit subset iteration number
        unit_subset_iterations_map = dict()
        for unit_subset_iteration in np.arange(0, 10):
            unit_subset_iterations_map[cls._make_unit_subset_iterations_map_key([unit_subset_iteration])] = \
                f"single_iteration_{unit_subset_iteration}"

        # two unit subsets: return unit subset iteration numbers
        unit_subset_iterations = [0, 1]
        unit_subset_iterations_map[cls._make_unit_subset_iterations_map_key(unit_subset_iterations)] = \
            "iterations_" + "_".join([str(x) for x in unit_subset_iterations])

        # consecutive ten unit subsets: return first and last iteration numbers
        x1 = 0
        x2 = 9
        unit_subset_iterations = np.arange(x1, x2 + 1)
        unit_subset_iterations_map[cls._make_unit_subset_iterations_map_key(unit_subset_iterations)] = \
            f"iterations_{x1}to{x2}"

        # no unit subset
        unit_subset_iterations = [None]
        unit_subset_iterations_map[cls._make_unit_subset_iterations_map_key(unit_subset_iterations)] = ""

        return unit_subset_iterations_map

    @staticmethod
    def _make_unit_subset_iterations_map_key(unit_subset_iterations):
        if unit_subset_iterations is not None:
            return tuple(unit_subset_iterations)
        return unit_subset_iterations

    @classmethod
    def _get_unit_subset_iterations_shorthand(cls, unit_subset_iterations):
        return cls._unit_subset_iterations_map()[cls._make_unit_subset_iterations_map_key(unit_subset_iterations)]

    def _make_param_name(self, secondary_key_subset_map):
        separating_character = "_"
        tolerate_non_unique = True

        # Make param name by stringing together values of a subset of secondary keys in table
        # Check that passed secondary key is subset of actual secondary key
        check_membership(
            secondary_key_subset_map.keys(), get_table_secondary_key_names(self),
            "passed secondary keys", f"{get_table_name(self)} secondary keys")
        # Enforce secondary key order in table definition
        secondary_key_subset_map = {k: secondary_key_subset_map[k] for k in get_table_secondary_key_names(self)
                                    if k in secondary_key_subset_map}

        # Use shorthand for unit_subset_iterations in param name (to avoid very long param name
        # if have several unit subset iterations)
        secondary_key_subset_map["unit_subset_iterations"] = self._get_unit_subset_iterations_shorthand(
            secondary_key_subset_map["unit_subset_iterations"])

        # If unit unit_subset_size is None, use empty string
        if secondary_key_subset_map["unit_subset_size"] is None:
            secondary_key_subset_map["unit_subset_size"] = ""

        # Drop empty string from param name (otherwise get multiple separating characters strung together)
        param_name_vals = [x for x in secondary_key_subset_map.values() if x != ""]

        # String together values
        return make_param_name(param_name_vals, separating_character, tolerate_non_unique)


def get_unit_label(nwb_file_name, sort_group_id, unit_id, curation_name=None):
    # Get inputs if not passed
    if curation_name is None:
        curation_name = get_default_param("curation_name")
    # Get curation_id and sort_interval_name from curation_name
    sort_interval_name, curation_id = split_curation_name(curation_name)
    return (CuratedSpikeSorting.Unit & {"curation_id": curation_id, "sort_interval_name": sort_interval_name,
                "sort_group_id": sort_group_id, "nwb_file_name": nwb_file_name, "unit_id": unit_id}).fetch1("label")


def get_unit_pair_name(unit_1_name, unit_2_name):
    # Return a single unit pair name for a pair of units passed in any order
    # Strategy for ordering: unit with smaller sort group ID comes first. If
    # units have the same sort group ID, unit with smaller unit ID comes first

    def _get_name(x1, x2):
        return f"{x1}.{x2}"

    # Get sort group ID and unit ID for each unit
    sort_group_id_1, unit_id_1 = split_unit_name(unit_1_name)
    sort_group_id_2, unit_id_2 = split_unit_name(unit_2_name)

    # Order unit names
    # If sort group IDs not the same
    if sort_group_id_1 != sort_group_id_2:
        x1, x2 = unit_1_name, unit_2_name  # default
        if sort_group_id_2 < sort_group_id_1:
            x1, x2 = unit_2_name, unit_1_name
    # If sort group IDs the same
    else:
        if unit_id_1 != unit_id_2:
            x1, x2 = unit_1_name, unit_2_name  # default
            if unit_id_2 < unit_id_1:
                x1, x2 = unit_2_name, unit_1_name

    return _get_name(x1, x2)


def populate_jguidera_unit(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_unit"
    upstream_schema_populate_fn_list = None
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_unit():
    schema.drop()



