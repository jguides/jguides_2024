# This module defines a table with electrode reference, as well as related functions
import copy

import datajoint as dj
from spyglass.common import Nwbfile, ElectrodeGroup, BrainRegion

# Define schema
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert

schema = dj.schema("jguidera_reference_electrode")


@schema
class ReferenceElectrode(dj.Manual):
    definition = """
    # Table with electrode references
    -> Nwbfile
    ---
    hpc_reference_electrode_left_hemisphere : varchar(40)
    hpc_reference_electrode_right_hemisphere : varchar(40)
    ctx_reference : varchar(40)
    """

    def make(self, key):
        self.insert(key)
        print('Populated ElectrodeReferences for {nwb_file_name}'.format(**key))

    def insert_defaults(self, **kwargs):
        # Rats for which references did not change during experiment
        hpc_ref_dict = {"peanut": {"hpc_left_nt": 13,  # reference ntrode, left hemisphere HPc
                                   "hpc_left_ch": 1,  # reference wire, left hemisphere HPc
                                   "hpc_right_nt": 13,  # reference ntrode, right hemisphere HPc
                                   "hpc_right_ch": 1},  # reference wire, right hemisphere HPc
                        "J16": {"hpc_left_nt": 13,
                                "hpc_left_ch": 1,
                                "hpc_right_nt": 3,
                                "hpc_right_ch": 1},
                        "fig": {"hpc_left_nt": 16,
                                "hpc_left_ch": 1,
                                "hpc_right_nt": 1,
                                "hpc_right_ch": 4},
                        "mango": {"hpc_left_nt": 18,
                                  "hpc_left_ch": 1,
                                  "hpc_right_nt": 9,
                                  "hpc_right_ch": 1}}  # HPc references
        for subject_id, ref_dict in hpc_ref_dict.items():
            # Get nwb file names for rat
            nwb_file_names = [nwb_file_name for nwb_file_name in Nwbfile.fetch("nwb_file_name") if
                              subject_id in nwb_file_name]
            for nwb_file_name in nwb_file_names:  # for each nwb file
                self._insert_references({"nwb_file_name": nwb_file_name}, ref_dict)

        # Rats for which references changed during experiment
        june_group_1 = {"hpc_left_nt": 23,
                        "hpc_left_ch": 3,
                        "hpc_right_nt": 10,
                        "hpc_right_ch": 1}
        june_group_2 = copy.deepcopy(june_group_1)
        june_group_2["hpc_right_ch"] = 2
        june_group_3 = {"hpc_left_nt": 10,
                        "hpc_left_ch": 1,
                        "hpc_right_nt": 10,
                        "hpc_right_ch": 1}  # use for day when nt23 went out and had not yet moved up nt22: june20220418
        june_group_4 = {"hpc_left_nt": 22,
                        "hpc_left_ch": 1,
                        "hpc_right_nt": 22,
                        "hpc_right_ch": 1}  # use for days when nt10 has axons
        june_group_5 = {"hpc_left_nt": 22,
                        "hpc_left_ch": 1,
                        "hpc_right_nt": 10,
                        "hpc_right_ch": 1}
        hpc_ref_dict = {**{file_name: june_group_1
                           for file_name in ["june20220412_.nwb",
                                             "june20220413_.nwb",
                                             "june20220414_.nwb",
                                             "june20220416_.nwb",
                                             "june20220417_.nwb"]},
                        **{file_name: june_group_2
                           for file_name in ["june20220415_.nwb"]},
                        **{file_name: june_group_3
                           for file_name in ["june20220418_.nwb"]},
                        **{file_name: june_group_4
                           for file_name in ["june20220419_.nwb"]},
                        **{file_name: june_group_5
                           for file_name in ["june20220420_.nwb",
                                             "june20220421_.nwb",
                                             "june20220422_.nwb",
                                             "june20220423_.nwb",
                                             "june20220424_.nwb",
                                             "june20220425_.nwb",
                                             "june20220426_.nwb",
                                             "june20220427_.nwb"]}}
        for nwb_file_name, ref_dict in hpc_ref_dict.items():  # for each nwb file
            if nwb_file_name not in Nwbfile.fetch("nwb_file_name"):
                continue
            self._insert_references({"nwb_file_name": nwb_file_name}, ref_dict)

    @staticmethod
    def make_hpc_refs(hpc_ref_electrode, hpc_ref_ch):
        return [(e - 1) * 4 - 1 + c for e, c in zip(hpc_ref_electrode, hpc_ref_ch)]

    def _insert_references(self, key, ref_dict):
        # Define HPc references
        for hemisphere in ["right", "left"]:
            key[f"hpc_reference_electrode_{hemisphere}_hemisphere"] = \
                self.make_hpc_refs(hpc_ref_electrode=[ref_dict[f"hpc_{hemisphere}_nt"]],
                                   hpc_ref_ch=[ref_dict[f"hpc_{hemisphere}_ch"]])[0]
        # Define CTX reference
        key["ctx_reference"] = -2
        # Populate table
        self.insert1(key, skip_duplicates=True)


def make_refs_dict(nwb_file_name):
    """
    Make dictionary that maps electrode groups to references
    :param nwb_file_name:
    :return: refs_dict
    """

    refs_dict = dict()  # map from electrode groups to references
    electrode_group_names, region_ids, target_hemispheres = (
            ElectrodeGroup & {"nwb_file_name": nwb_file_name}).fetch("electrode_group_name", "region_id",
                                                                     "target_hemisphere")
    for electrode_group_name, region_id, target_hemisphere in zip(electrode_group_names, region_ids, target_hemispheres):
        region_name = (BrainRegion & {"region_id": region_id}).fetch1("region_name")
        if region_name in ["Hippocampus", "hippocampus", "corpus callosum"]:
            refs_dict[electrode_group_name] = int((ReferenceElectrode() & {"nwb_file_name": nwb_file_name}).fetch1(
                f"hpc_reference_electrode_{target_hemisphere.lower()}_hemisphere"))
        elif region_name in ["Cortex", "cortex"]:
            refs_dict[electrode_group_name] = int((ReferenceElectrode() & {"nwb_file_name": nwb_file_name}).fetch1(f"ctx_reference"))
        else:
            raise Exception(f"{region_name} not accounted for in code. Must specify map between this region and ReferenceElectrode columns")

    return refs_dict


def populate_jguidera_reference_electrode(key=None, tolerate_error=False):
    schema_name = "jguidera_reference_electrode"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


