import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import spyglass as nd
from spyglass.common import ElectrodeGroup
from spyglass.spikesorting.v0.spikesorting_recording import SortGroup

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase, SelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, check_nwb_file_name, \
    split_unit_name, unique_table_column_sets, \
    split_curation_name, get_curation_name, insert_analysis_table_entry, fetch_entries_as_dict
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_epoch_keys, \
    get_brain_regions, get_jguidera_nwbf_names
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import get_nwb_file
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification, JguideraNwbfile
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.df_helpers import df_from_data_list, df_filter_columns
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.plot_helpers import tint_color
from src.jguides_2024.utils.set_helpers import check_set_equality
from src.jguides_2024.utils.string_helpers import format_bool
from src.jguides_2024.utils.vector_helpers import unpack_single_element

# These imports are called with eval or used in table definitions (do not remove):
ElectrodeGroup
nd

schema = dj.schema("jguidera_brain_region")


@schema
class BrainRegionColor(dj.Manual):
    definition = """
    # Table that matches brain regions to colors
    brain_region : varchar(40)
    ---
    color : blob
    """

    def return_brain_region_color_map(self, brain_regions=None):
        # Define brain regions if not passed
        if brain_regions is None:
            brain_regions = self.fetch("brain_region")
        # Return map from brain region to color
        return {brain_region: (self & {"brain_region": brain_region}).fetch1("color")
                for brain_region in brain_regions}

    def get_brain_region_color(self, brain_region):
        brain_region_color_map = self.return_brain_region_color_map(brain_regions=[brain_region])
        return brain_region_color_map[brain_region]

    def tint_colors(self, brain_regions, num_tints, concatenate=False):
        brain_region_colors = {x: self.return_brain_region_color_map()[x] for x in brain_regions}
        brain_region_colors_tint = {x: tint_color(color, num_tints) for x, color in brain_region_colors.items()}
        if concatenate:  # concatenate across brain regions if indicated
            return np.vstack(
                list(zip(*list(brain_region_colors_tint.values()))))
        return brain_region_colors_tint

    def insert_defaults(self, **kwargs):
        # Insert for brain regions
        for brain_region, color in zip(["OFC_2", "OFC", "mPFC_2", "mPFC", "CA1_2", "CA1"], plt.cm.Paired.colors):
            self.insert1({"brain_region": brain_region, "color": color}, skip_duplicates=True)
        # Insert for targeted regions -- same as corresponding brain regions
        for targeted_location, color in zip(["OFC", "mPFC", "CA1"], plt.cm.Paired.colors[1::2]):
            brain_region = get_brain_region_from_targeted_location(targeted_location)
            self.insert1({"brain_region": brain_region, "color": color}, skip_duplicates=True)


def get_brain_region_from_targeted_location(targeted_location):
    return f"{targeted_location}_targeted"


def get_targeted_location_from_brain_region(brain_region):
    # Check input has expected form
    if brain_region[-9:] != "_targeted":
        raise Exception(f"brain_region must end with _targeted")
    return brain_region.split("_targeted")[0]


def _get_jguidera_keys(key=None):
    # Define keys
    if key is not None:
        if len(TaskIdentification & key) == 0:
            raise Exception(f"There must be entry in TaskIdentification to populate ElectrodeGroupTargetedLocation. "
                            f"There was none for passed key: {key}")
        return [key]
    else:
        return get_jguidera_nwbf_epoch_keys()


@schema
class ElectrodeGroupTargetedLocation(ComputedBase):
    definition = """
    # Table with targeted location from nwb file for electrode groups
    -> ElectrodeGroup
    ---
    targeted_location : varchar(40)
    """

    def make(self, key):
        check_nwb_file_name(key["nwb_file_name"])
        nwbf = get_nwb_file(key["nwb_file_name"])
        key = {**key, **{
            "targeted_location": nwbf.electrode_groups[key["electrode_group_name"]].targeted_location}}  # update key
        insert1_print(self, key)

    def populate_(self, **kwargs):
        for key in _get_jguidera_keys(kwargs["key"]):
            super().populate_(key=key)


@schema
class SortGroupTargetedLocation(ComputedBase):
    definition = """
    # Table with targeted location from nwb file for sort groups
    -> SortGroup
    -> JguideraNwbfile
    ---
    targeted_location : varchar(40)
    """

    def make(self, key):

        electrode_group_names = np.unique(
            (SortGroup.SortGroupElectrode & {"nwb_file_name": key["nwb_file_name"],
                                             "sort_group_id": key["sort_group_id"]}).fetch("electrode_group_name"))
        if len(electrode_group_names) != 1:  # check that only found one electrode group for sort group
            raise Exception(f"Should have found exactly one electrode group for "
                            f"sort group {key['sort_group_id']} but found {len(electrode_group_names)}")
        electrode_group_name = electrode_group_names[0]
        nwbf = get_nwb_file(key["nwb_file_name"])
        key.update({"targeted_location": nwbf.electrode_groups[electrode_group_name].targeted_location})  # update key
        insert1_print(self, key)

    def _get_sort_group_id_targeted_location(self, nwb_file_name, exclude_no_unit_sort_group_ids=False):

        sort_group_ids, targeted_locations = (self & {"nwb_file_name": nwb_file_name}).fetch("sort_group_id",
                                                                                             "targeted_location")

        # Exclude sort group IDs with no units if indicated
        valid_bool = [True]*len(sort_group_ids)  # default

        if exclude_no_unit_sort_group_ids:
            # Local import to avoid circular import
            from src.jguides_2024.spikes.jguidera_spikes import EpochSpikeTimes
            valid_bool = [len(EpochSpikeTimes & {"nwb_file_name": nwb_file_name,
                                                  "sort_group_id": sort_group_id}) > 0
                          for sort_group_id in sort_group_ids]

        return sort_group_ids[valid_bool], targeted_locations[valid_bool]

    def return_targeted_location_sort_group_map(self, nwb_file_name, exclude_no_unit_sort_group_ids=False):
        sort_group_ids, targeted_locations = self._get_sort_group_id_targeted_location(nwb_file_name,
                                                                                       exclude_no_unit_sort_group_ids)
        return {targeted_location: sort_group_ids[targeted_locations == targeted_location]
                                             for targeted_location in set(targeted_locations)}

    def return_sort_group_targeted_location_map(self, nwb_file_name, exclude_no_unit_sort_group_ids=False):
        sort_group_ids, targeted_locations = self._get_sort_group_id_targeted_location(nwb_file_name,
                                                                                       exclude_no_unit_sort_group_ids)
        return {k: v for k, v in zip(sort_group_ids, targeted_locations)}

    def get_sort_group_ids(self, nwb_file_name, targeted_locations=None):
        sort_group_map = self.return_targeted_location_sort_group_map(nwb_file_name)
        if targeted_locations is not None:  # take only those in list of brain regions if passed
            sort_group_ids = [sort_group_map[x] for x in targeted_locations if x in sort_group_map]
        else:
            sort_group_ids = list(sort_group_map.values())
        return np.sort(np.concatenate(sort_group_ids))

    def lookup_targeted_location(self, nwb_file_name, sort_group_id=None, unit_name=None):
        check_one_none([sort_group_id, unit_name])
        if sort_group_id is None:
            sort_group_id, _ = split_unit_name(unit_name)
        return self.return_sort_group_targeted_location_map(nwb_file_name)[sort_group_id]

    def populate_(self, **kwargs):
        JguideraNwbfile().populate_(**kwargs)
        super().populate_(**kwargs)


@schema
class BrainRegionSortGroupParams(SecKeyParamsBase):
    definition = """
    # Parameters for relabeling entries from upstream tables in BrainRegionSortGroup
    -> JguideraNwbfile
    brain_region : varchar(40)  # name of brain region in downstream pool table
    ---
    source_brain_region : varchar(40)  # name of brain region in upstream source table
    source_table_name : varchar(80)
    """

    @staticmethod
    def _get_source_table_name_suffix_map():
        # Map from source table with brain region information to suffix to add to brain region primary key text
        return {"SortGroupTargetedLocation": "targeted", }

    def get_brain_region_text(self, brain_region, source_table_name):
        return f"{brain_region}_{self._get_source_table_name_suffix_map()[source_table_name]}"

    def insert_defaults(self, **kwargs):
        key_filter = kwargs.pop("key_filter", None)
        for source_table_name, suffix in self._get_source_table_name_suffix_map().items():
            table = get_table(source_table_name)
            for nwb_file_name, targeted_location in unique_table_column_sets(
                    table, ["nwb_file_name", "targeted_location"], key_filter):
                brain_region = self.get_brain_region_text(targeted_location, source_table_name)
                key = {"nwb_file_name": nwb_file_name, "brain_region": brain_region,
                       "source_brain_region": targeted_location, "source_table_name": source_table_name}
                super().insert1(key)


@schema
class BrainRegionSortGroup(ComputedBase):
    definition = """
    # Map from brain region to sort groups
    -> BrainRegionSortGroupParams
    ---
    sort_group_ids : blob
    """

    class SortGroupTargetedLocation(dj.Part):
        definition = """
        # Achieves dependence on SortGroupTargetedLocation
        -> BrainRegionSortGroup
        -> SortGroupTargetedLocation
        """

    def make(self, key):

        source_table_name, source_brain_region = (BrainRegionSortGroupParams & key).fetch1("source_table_name",
                                                                                           "source_brain_region")
        if source_table_name == "SortGroupTargetedLocation":
            sort_group_ids = SortGroupTargetedLocation().get_sort_group_ids(key["nwb_file_name"], [source_brain_region])

            # Insert into main table
            insert1_print(self, {**key, **{"sort_group_ids": sort_group_ids}})

            # Insert into part table
            for sort_group_id in sort_group_ids:
                insert1_print(self.SortGroupTargetedLocation, {**key, **{"sort_group_id": sort_group_id}})

    def get_brain_region_sort_group_id_map(self):
        # Return df with correspondence between nwb_file_name, brain_region and sort_group_id across all
        # table entries
        return df_from_data_list([(entry["nwb_file_name"], entry["brain_region"], sort_group_id)
                           for entry in fetch_entries_as_dict(BrainRegionSortGroup)
                           for sort_group_id in entry["sort_group_ids"]],
                          ["nwb_file_name", "brain_region", "sort_group_id"])


@schema
class BrainRegionCohort(dj.Manual):
    definition = """
    # Group of brain regions for a nwb file
    -> JguideraNwbfile
    brain_region_cohort_name : varchar(80)
    ---
    brain_regions : blob
    """

    def insert_defaults(self, **kwargs):

        for nwb_file_name in JguideraNwbfile.fetch("nwb_file_name"):
            for cohort_name_suffix, targeted in zip(["targeted", ""], [True, False]):
                brain_region_cohort_name = f"all{format_bool(targeted, cohort_name_suffix, prepend_underscore=True)}"
                brain_regions = get_brain_regions(nwb_file_name, targeted=targeted)
                self.insert1({"nwb_file_name": nwb_file_name,
                              "brain_region_cohort_name": brain_region_cohort_name,
                              "brain_regions": brain_regions}, skip_duplicates=True)

        # temporary until process CA1 data
        for nwb_file_name in ["J1620210529_.nwb", "J1620210531_.nwb",
                              "mango20211129_.nwb", "mango20211130_.nwb",
                              "june20220412_.nwb", "june20220415_.nwb",
                              ]:
            self.insert1({"nwb_file_name": nwb_file_name,
                          "brain_region_cohort_name": "mPFC_targeted_OFC_targeted",
                          "brain_regions": ["mPFC_targeted", "OFC_targeted"]}, skip_duplicates=True)

    def lookup_cohort_name(self, nwb_file_name, brain_regions):
        return unpack_single_element([k["brain_region_cohort_name"] for k in (
                self & {"nwb_file_name": nwb_file_name}).fetch("KEY") if check_set_equality(
            (self & k).fetch1("brain_regions"), brain_regions, tolerate_error=True)])


@schema
class CurationSetSel(SelBase):
    definition = """
    # Selection from upstream tables for CurationSet
    -> BrainRegionCohort
    curation_set_name : varchar(40)
    """

    def _get_potential_keys(self, key_filter=None):

        brain_region_cohort_name = "all_targeted"

        # "runs analysis": single run sessions. v1: probe geometry issue. v2: after probe geometry fixed.
        curation_set_names = ["runs_analysis_v1", "runs_analysis_v2"]

        potential_keys = [
            {"nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name,
             "curation_set_name": curation_set_name} for nwb_file_name in JguideraNwbfile().fetch("nwb_file_name")
        for curation_set_name in curation_set_names
        ]

        # "across runs analysis": units that are present across run sessions. v1: probe geometry issue. v2: after probe
        # geometry fixed.
        curation_set_names = ["across_runs_analysis_v2"]
        potential_keys += [
            {"nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name,
             "curation_set_name": curation_set_name} for nwb_file_name in ["J1620210529_.nwb", "J1620210531_.nwb"]
        for curation_set_name in curation_set_names
        ]

        # !!! temporary until processed CA1 data
        nwb_file_names = ["J1620210529_.nwb", "J1620210531_.nwb",
                          "mango20211129_.nwb", "mango20211130_.nwb",
                          "june20220412_.nwb", "june20220415_.nwb", ]
        brain_region_cohort_name = "mPFC_targeted_OFC_targeted"
        curation_set_names = ["runs_analysis_v2", "across_runs_analysis_v2"]
        for curation_set_name in curation_set_names:
            for nwb_file_name in nwb_file_names:
                potential_keys += [{"nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name,
                                "curation_set_name": curation_set_name,
                                }]

        return potential_keys


@schema
class CurationSet(ComputedBase):
    definition = """
    # Curation names corresponding to brain regions in a brain region cohort
    -> CurationSetSel
    ---
    -> nd.common.AnalysisNwbfile
    curation_names_df_object_id : varchar(40)
    """

    def make(self, key):

        # Local import to avoid circular import error
        from src.jguides_2024.spike_sorting_curation.jguidera_spikesorting import define_sort_intervals

        brain_region_cohort_name = key["brain_region_cohort_name"]
        curation_set_name = key["curation_set_name"]
        curation_id = 3

        nwb_file_name = key["nwb_file_name"]

        data_list = []

        # Get brain regions
        brain_regions = (BrainRegionCohort & key).fetch1("brain_regions")
        for brain_region in brain_regions:

            # Get sort intervals based on brain region
            targeted_location = get_targeted_location_from_brain_region(brain_region)
            objs = define_sort_intervals(targeted_location, nwb_file_name, curation_set_name)
            for obj in objs:

                # Get curation ID
                curation_name = get_curation_name(obj.sort_interval_name, curation_id)

                # Single run sessions
                if curation_set_name in ["runs_analysis_v1", "runs_analysis_v2"]:

                    # single epoch sort
                    if "pos" in obj.interval_list_name:
                        # match up current interval list name with appropriate epoch
                        interval_list_name = obj.interval_list_name.split(" no premaze")[0].split(" no home")[0]
                        epoch = EpochIntervalListName().get_epoch(nwb_file_name, interval_list_name)
                        epochs_description = EpochsDescription().get_single_run_description(nwb_file_name, epoch)
                        epochs_descriptions = [epochs_description]

                    # across runs (and potentially sleeps) sort
                    else:
                        epochs_descriptions = EpochsDescription().get_single_run_descriptions(nwb_file_name)

                    for epochs_description in epochs_descriptions:
                        data_list.append((curation_name, nwb_file_name, brain_region, epochs_description))

                # All run sessions
                elif curation_set_name in ["across_runs_analysis_v1", "across_runs_analysis_v2"]:

                    data_list.append((curation_name, nwb_file_name, brain_region, "runs"))

                else:
                    raise Exception(f"curation_set_name {curation_set_name} not accounted for")

        curation_names_df = df_from_data_list(data_list, [
            "curation_name", "nwb_file_name", "brain_region", "epochs_description"])

        # Insert into table
        key = {"nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name,
                          "curation_set_name": curation_set_name}
        insert_analysis_table_entry(self, [curation_names_df], key)

    def get_split_curation_names_map(self):
        # TODO: update
        # Return map from brain region to dictionary with components of curation_name (sort_interval_name
        # and curation_id)
        curation_names_map = self.fetch1("curation_names_map")
        split_curation_names_map = dict()
        for brain_region, curation_name in curation_names_map.items():
            sort_interval_name, curation_id = split_curation_name(curation_name)
            split_curation_names_map[brain_region] = {"sort_interval_name": sort_interval_name, "curation_id":
                curation_id}

        return split_curation_names_map

    def get_curation_name(self, brain_region, epochs_description=None, tolerate_no_entry=False):

        filter_key = {"brain_region": brain_region}
        if epochs_description is not None:
            filter_key.update({"epochs_description": epochs_description})

        df_subset = df_filter_columns(self.fetch1_dataframe(), filter_key)

        if len(df_subset) == 0 and tolerate_no_entry:
            return None

        return check_return_single_element(df_subset.curation_name.values).single_element


def populate_jguidera_brain_region(key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_brain_region"
    upstream_schema_populate_fn_list = None
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_brain_region():
    schema.drop()


