# This module defines tables related to epochs

import copy
import itertools

import datajoint as dj
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_subject_id, \
    get_reliability_paper_nwb_file_names
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert1_print, \
    get_schema_table_names_from_file, \
    populate_insert, get_epochs_id, check_epochs_id, get_key_filter, fetch_iterable_array, fetch_entries_as_dict
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import Contingency
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification, JguideraNwbfile
from src.jguides_2024.time_and_trials.jguidera_epoch_interval import EpochInterval
from src.jguides_2024.utils.dict_helpers import add_defaults
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import check_all_unique, unpack_single_element

schema_name = "jguidera_epoch"
schema = dj.schema(schema_name)  # define custom schema


def insert_epoch_table(table, key, target_contingency_type):
    # Get contingency
    entry = (TaskIdentification & key).fetch1()
    contingency = entry["contingency"]  # contingency
    # Check valid
    Contingency().check_valid_contingency(contingency)
    # Get contingency type
    contingency_type = Contingency().get_contingency_type(contingency)
    # If contingency type matches desired, insert into table
    if contingency_type == target_contingency_type:
        key.update(entry)
        insert1_print(table, key)


@schema
class EpochCohortParams(SecKeyParamsBase):
    definition = """
    # Parameters for specifying groups of run epochs
    -> JguideraNwbfile
    epochs_id : varchar(40)
    ---
    epochs : blob
    """

    def insert1(self, key, **kwargs):
        check_epochs_id(key["epochs_id"], key["epochs"])
        super().insert1(key, **kwargs)

    def insert_from_epochs(self, nwb_file_name, epochs, populate_tables=True):
        key = {"nwb_file_name": nwb_file_name, "epochs_id": get_epochs_id(epochs), "epochs": epochs}
        self.insert1(key)
        if populate_tables:
            EpochCohort().populate(key)

    def lookup_epochs_id(self, nwb_file_name, epochs):
        key = {"nwb_file_name": nwb_file_name, "epochs_id": get_epochs_id(epochs)}
        return (self & key).fetch1("epochs_id")

    def insert_epoch_combinations(self, epoch_source_table, num_in_comb, key_filter):
        # (note that jguidera nwb file and epoch source table intersection below useful since these tables may not
        # contain the same entries and we only want entries in both tables)
        for nwb_file_name in ((JguideraNwbfile * epoch_source_table) & key_filter).fetch("nwb_file_name"):
            epochs = epoch_source_table().get_epochs(nwb_file_name)
            r = copy.deepcopy(num_in_comb)
            if num_in_comb == "max":
                r = len(epochs)
            epoch_combinations = list(itertools.combinations(epochs, r))
            for epoch_combination in epoch_combinations:
                self.insert_from_epochs(nwb_file_name, epoch_combination)

    def insert_defaults(self, **kwargs):
        key_filter = get_key_filter(kwargs)
        # Run epochs
        for num_in_comb in [1, "max"]:
            RunEpoch().populate()
            self.insert_epoch_combinations(RunEpoch, num_in_comb, key_filter)


@schema
class EpochCohort(ComputedBase):
    definition = """
    # Group of epochs within an nwb file
    -> EpochCohortParams
    ---
    epochs : blob  # for convenience
    num_epochs : int  # for convenience
    """

    class CohortEntries(dj.Part):
        definition = """
        # Achieves dependence on single epoch information in TaskIdentification
        -> EpochCohort
        -> TaskIdentification
        """

    def make(self, key):
        # Insert into main table
        epochs = (EpochCohortParams & key).fetch1("epochs")
        self.insert1({**key, **{"epochs": epochs, "num_epochs": len(epochs)}})
        # Insert into part table
        for epoch in epochs:
            key.update({"epoch": epoch})
            insert1_print(self.CohortEntries, key)

    def get_epoch(self):
        return unpack_single_element(self.fetch1("epochs"))

    def add_epoch_to_key(self, key):
        key.update({"epoch": (self & key).get_epoch()})
        return key

    def drop(self):
        from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits
        BrainRegionUnits().drop()
        super.drop()


@schema
class RunEpoch(ComputedBase):
    definition = """
    # Table with run epochs
    -> TaskIdentification
    ---
    contingency : varchar(40)
    task_environment : varchar(40)
    """

    def make(self, key):
        insert_epoch_table(self, key, "run")

    def get_epochs(self, nwb_file_name):
        return (self & {"nwb_file_name": nwb_file_name}).fetch("epoch")

    def get_run_num(self, nwb_file_name, epoch):
        return unpack_single_element(np.where(self.get_epochs(nwb_file_name) == epoch)[0]) + 1

    def get_single_contingency_epochs(self, nwb_file_name):
        # Return epochs with single contingency, i.e. without epoch contingency switch
        return [epoch for epoch in (self & {"nwb_file_name": nwb_file_name}).fetch("epoch")
                if TaskIdentification().is_single_contingency_epoch(nwb_file_name, epoch)]


@schema
class DistinctRunEpochPair(ComputedBase):
    definition = """
    # Pairs of distinct run epochs 
    -> RunEpoch.proj(epoch_1='epoch')
    -> RunEpoch.proj(epoch_2='epoch')
    """

    def make(self, key):
        # To avoid effective duplicates, insert epoch combinations where epoch 1 is smaller than epoch 2
        if key["epoch_1"] < key["epoch_2"]:
            # Limit key to those in table
            key = {k: key[k] for k in ["nwb_file_name", "epoch_1", "epoch_2"]}
            # Insert into table
            insert1_print(self, key)


@schema
class RunEpochPair(ComputedBase):
    definition = """
    # Pairs of run epochs 
    -> RunEpoch.proj(epoch_1='epoch')
    -> RunEpoch.proj(epoch_2='epoch')
    """

    def make(self, key):
        # To avoid effective duplicates, insert epoch combinations where epoch 1 is smaller than epoch 2
        if key["epoch_1"] <= key["epoch_2"]:
            # Limit key to those in table
            key = {k: key[k] for k in ["nwb_file_name", "epoch_1", "epoch_2"]}
            # Insert into table
            insert1_print(self, key)


@schema
class SleepEpoch(ComputedBase):
    definition = """
    # Table with sleep epochs
    -> TaskIdentification
    ---
    contingency : varchar(40)
    task_environment : varchar(40)
    """

    def make(self, key):
        insert_epoch_table(self, key, "sleep")

    def get_epochs(self, nwb_file_name):
        return (self & {"nwb_file_name": nwb_file_name}).fetch("epoch")


@schema
class HomeEpoch(ComputedBase):
    definition = """
    # Table with sleep epochs
    -> TaskIdentification
    ---
    contingency : varchar(40)
    task_environment : varchar(40)
    """

    def make(self, key):
        insert_epoch_table(self, key, "home")

    def get_epochs(self, nwb_file_name):
        return (self & {"nwb_file_name": nwb_file_name}).fetch("epoch")


@schema
class EpochsDescription(dj.Manual):
    definition = """
    # A certain kind of grouping of epochs, e.g. all runs
    -> JguideraNwbfile
    epochs_description : varchar(40)
    ---
    -> EpochCohort  # epochs_id
    epochs : blob  # for convenience
    """

    @staticmethod
    def format_single_run_description(run_num):
        return f"run{run_num}"

    @staticmethod
    def format_runs_description(run_nums):
        return "runs" + "_".join([str(x) for x in run_nums])

    @staticmethod
    def get_all_runs_description():
        return "runs"

    @classmethod
    def valid_epochs_descriptions(cls):
        return [cls.get_all_runs_description()] + [cls.format_single_run_description(x) for x in np.arange(1, 10)]

    def get_single_contingency_descriptions(self, nwb_file_name, exclude_non_full_len=False, exclude_artifact=False):

        # Return descriptions of single epochs with single contingency, i.e. without epoch contingency switch,
        # also optionally excluding non-full length epochs and/or epochs with artifacts

        # Get single contigency epoch descriptions
        epochs_descriptions = [epochs_description for epochs_description, epochs in fetch_iterable_array((
            self & {"nwb_file_name": nwb_file_name}), ["epochs_description", "epochs"])
                if len(epochs) == 1  # single epochs
                and TaskIdentification().is_single_contingency_epoch(nwb_file_name, epochs[0])
               ]

        # Exclude epochs that are not full length if indicated
        if exclude_non_full_len:
            epochs_descriptions = EpochFullLen().exclude_non_full_len_epochs_descriptions(
                nwb_file_name, epochs_descriptions)

        # Exclude epochs with artifacts if indicated
        if exclude_artifact:
            epochs_descriptions = EpochArtifactFree().exclude_artifact_epoch_descriptions(
                nwb_file_name, epochs_descriptions)

        # Return epochs descriptions
        return epochs_descriptions

    def get_single_run_descriptions(self, nwb_file_name):
        epochs_descriptions = (self & {"nwb_file_name": nwb_file_name}).fetch("epochs_description")
        return [x for x in epochs_descriptions if x[:3] == "run" and x[3:].isdigit()]

    def get_single_run_description(self, nwb_file_name, epoch):
        run_num = RunEpoch().get_run_num(nwb_file_name, epoch)
        return self.format_single_run_description(run_num)

    def get_runs_description(self, nwb_file_name, epochs):
        run_nums = [RunEpoch().get_run_num(nwb_file_name, epoch) for epoch in epochs]
        return self.format_runs_description(run_nums)

    def lookup_epochs_description(self, nwb_file_name, epochs):
        # Look up epochs description for nwb file / epoch set. Assumes one epochs description per
        # nwb file / epoch set (must change code if want to not require this in future)
        return (self & {"nwb_file_name": nwb_file_name, "epochs_id": get_epochs_id(epochs)}).fetch1(
            "epochs_description")

    def get_epoch(self):
        # Use if expect single epoch
        return unpack_single_element(self.fetch1("epochs"))

    def insert1(self, key, **kwargs):
        kwargs = add_defaults(kwargs, {"skip_duplicates": True}, add_nonexistent_keys=True)
        super().insert1(key, **kwargs)

    def insert_defaults(self, **kwargs):
        key_filter = get_key_filter(kwargs)
        # Populate upstream table
        EpochCohort().populate_(key=key_filter)
        # (note that jguidera nwb file and run epoch table intersection below useful since these tables may not contain
        # the same entries and we only want entries in both tables)
        for nwb_file_name in set(((JguideraNwbfile * RunEpoch) & key_filter).fetch("nwb_file_name")):
            key = {"nwb_file_name": nwb_file_name}
            epochs = np.sort(RunEpoch().get_epochs(nwb_file_name))
            check_all_unique(epochs)
            # All runs
            key.update({
                "epochs": epochs, "epochs_id": get_epochs_id(epochs),
                "epochs_description": self.get_all_runs_description()})
            self.insert1(key)
            # Single runs
            for epoch in epochs:
                epochs_ = [epoch]
                run_num = RunEpoch().get_run_num(nwb_file_name, epoch)
                key.update({
                    "epochs": epochs_, "epochs_id": get_epochs_id(epochs_),
                    "epochs_description": self.format_single_run_description(run_num)})
                self.insert1(key)

    def insert_runs(self, nwb_file_name, epochs):
        # Insert an arbitrary number of run epochs

        # Check all epochs are run epochs
        run_epochs = RunEpoch().get_epochs(nwb_file_name)
        check_membership(epochs, run_epochs, "passed epochs", "run epochs")
        # Define description based on number of runs
        if len(epochs) == 1:  # single run gets "run" prefix
            epochs_description = self.get_single_run_description(nwb_file_name, unpack_single_element(epochs))
        elif len(epochs) == len(run_epochs):  # all runs gets just "runs", nothing else
            epochs_description = self.get_all_runs_description()
        else:  # multiple but not all runs get "runs" prefix
            epochs_description = self.get_runs_description(nwb_file_name, epochs)
        key = {"nwb_file_name": nwb_file_name, "epochs_description": epochs_description}
        epochs_id = EpochCohortParams().lookup_epochs_id(nwb_file_name, epochs)
        self.insert1({**key, **{"epochs_id": epochs_id, "epochs": epochs}})


@schema
class EpochFullLen(dj.Computed):
    definition = """
    # Indicates whether or not epoch full length
    -> EpochInterval
    ---
    epoch_full_len : bool
    """

    def make(self, key):
        key.update({"epoch_full_len": int((EpochInterval & key).get_epoch_duration() > 19 * 60)})
        insert1_print(self, key)

    def exclude_non_full_len_epochs_descriptions(self, nwb_file_name, epochs_descriptions):
        return [x for x in epochs_descriptions if (self & {
            "nwb_file_name": nwb_file_name, "epoch": (EpochsDescription & {
                "nwb_file_name": nwb_file_name, "epochs_description": x}).get_epoch()}).fetch1("epoch_full_len")]


@schema
class EpochArtifactFree(dj.Computed):
    definition = """
    # Indicates whether epoch free of artifacts
    -> TaskIdentification
    ---
    epoch_artifact_free : bool
    """

    def make(self, key):
        key.update({"epoch_artifact_free": bool((key["nwb_file_name"], key["epoch"]) not in self._artifact_epochs())})
        insert1_print(self, key)

    @staticmethod
    def _artifact_epochs():
        return [("june20220420_.nwb", 8)]

    def exclude_artifact_epoch_descriptions(self, nwb_file_name, epochs_descriptions):
        return [x for x in epochs_descriptions if (self & {
            "nwb_file_name": nwb_file_name, "epoch": (EpochsDescription & {
                "nwb_file_name": nwb_file_name, "epochs_description": x}).get_epoch()}).fetch1(
            "epoch_artifact_free")]


@schema
class EpochsDescriptions(dj.Manual):
    definition = """
    # A group of epochs descriptions within a subject
    -> JguideraNwbfile
    epochs_descriptions_name : varchar(40)
    ---
    epochs_descriptions : blob
    """

    class EpochArtifactFree(dj.Part):
        definition = """
        # Achieves upstream dependence on EpochArtifactFree
        -> EpochsDescriptions
        -> EpochArtifactFree
        """

    class EpochFullLen(dj.Part):
        definition = """
        # Achieves upstream dependence on EpochFullLen
        -> EpochsDescriptions
        -> EpochFullLen
        """

    def _insert_parts(self, key, nwb_file_name, epochs_description):
        epoch = unpack_single_element((EpochsDescription & {
            "nwb_file_name": nwb_file_name, "epochs_description": epochs_description}).fetch1("epochs"))
        parts_key = {"nwb_file_name": nwb_file_name, "epoch": epoch}
        for parts_table in [self.EpochArtifactFree, self.EpochFullLen]:
            parts_table.insert1({**key, **parts_key}, skip_duplicates=True)

    # TODO (feature): use key_filter if passed
    def insert_defaults(self, **kwargs):

        # CASE 1: For single nwb files, insert single contingency runs that are: 1) full length 2) free of artifacts
        epochs_descriptions_name = "valid_single_contingency_runs"
        for nwb_file_name in np.unique(EpochInterval().fetch("nwb_file_name")):

            # Get single contingency epochs descriptions that are full length and without artifacts
            epochs_descriptions = EpochsDescription().get_single_contingency_descriptions(
                nwb_file_name, exclude_non_full_len=True, exclude_artifact=True)

            # Insert into main table
            key = {
                "nwb_file_name": nwb_file_name, "epochs_descriptions_name": epochs_descriptions_name}
            self.insert1({**key, **{"epochs_descriptions": epochs_descriptions}}, skip_duplicates=True)

            # Insert into parts tables
            for epochs_description in epochs_descriptions:
                self._insert_parts(key, nwb_file_name, epochs_description)

        # CASE 2: For single nwb files on first day of learning, insert single contingency run, one at a time
        for nwb_file_name in ["J1620210529_.nwb"]:

            # Get single contingency epochs descriptions that are full length and without artifacts
            epochs_descriptions = EpochsDescription().get_single_contingency_descriptions(
                nwb_file_name, exclude_non_full_len=True, exclude_artifact=True)

            for epochs_description in epochs_descriptions:

                # Insert into main table
                key = {
                    "nwb_file_name": nwb_file_name, "epochs_descriptions_name": epochs_description}
                self.insert1({**key, **{"epochs_descriptions": [epochs_description]}}, skip_duplicates=True)

                # Insert into parts tables
                self._insert_parts(key, nwb_file_name, epochs_description)

    def get_nwb_file_name_epochs_map(self, nwb_file_names=None):

        if nwb_file_names is None:
            nwb_file_names = get_reliability_paper_nwb_file_names()

        nwb_file_name_epochs_map = dict()
        for nwb_file_name in nwb_file_names:
            epochs_descriptions = (self & {"nwb_file_name": nwb_file_name}).fetch1(
                "epochs_descriptions")
            epochs = []
            for epochs_description in epochs_descriptions:
                epochs.append((EpochsDescription & {
                    "nwb_file_name": nwb_file_name, "epochs_description": epochs_description}).get_epoch())
            nwb_file_name_epochs_map[nwb_file_name] = epochs
        return nwb_file_name_epochs_map


class NwbfSetBase(dj.Manual):

    def _meta_set_name(self):
        return unpack_single_element(self.primary_key)

    def lookup_rat_cohort_set_name(self):
        subject_ids = list(get_reliability_paper_nwb_file_names(as_dict=True).keys())
        # checks that in table
        return (self & {
            self._meta_set_name(): self.get_Haight_single_contingency_rotation_set_name(subject_ids)}).fetch1(
            self._meta_set_name())

    @staticmethod
    def get_single_nwbf_set_name(nwb_file_name, epochs_name):
        return f"{format_nwb_file_name(nwb_file_name)}_{epochs_name}"

    @staticmethod
    def get_Haight_single_contingency_rotation_set_name(subject_ids):
        subject_id_text = "_".join(subject_ids)
        return f"{subject_id_text}_Haight_single_contingency_rotation"

    @staticmethod
    def get_single_epochs_description_set_name(nwb_file_name, epochs_description):
        return f"{format_nwb_file_name(nwb_file_name)}_{epochs_description}"

    def get_subject_id(self):
        # Return single subject ID across table entries
        nwb_file_names = np.concatenate(self.fetch("nwb_file_names"))
        return check_return_single_element([get_subject_id(x) for x in nwb_file_names]).single_element

    def get_nwb_file_name(self):
        # Return single nwb file name for single table entry; raise error if more than one nwbf name for the entry
        nwb_file_names = self.fetch1("nwb_file_names")
        if len(nwb_file_names) > 1:
            raise Exception(f"More than one nwbf name for {self.fetch1('KEY')}")
        return unpack_single_element(nwb_file_names)


@schema
class RecordingSet(NwbfSetBase):
    definition = """
    # Group of nwb files and corresponding epochs descriptions
    recording_set_name : varchar(80)
    ---
    nwb_file_names : blob
    epochs_descriptions_names : blob
    """

    class JguideraNwbfile(dj.Part):
        definition = """
        # Achieves upstream dependence on JguideraNwbfile
        -> RecordingSet
        -> JguideraNwbfile
        """

    class EpochsDescriptions(dj.Part):
        definition = """
        # Achieves upstream dependence on EpochsDescriptions
        -> RecordingSet
        -> EpochsDescriptions
        """

    def _insert_tables(self, recording_set_name, nwb_file_names, epochs_descriptions_names):

        # Insert into main table
        key = {"recording_set_name": recording_set_name}
        self.insert1({**key, **{
            "nwb_file_names": nwb_file_names, "epochs_descriptions_names": epochs_descriptions_names}},
                     skip_duplicates=True)

        # Insert into parts tables
        for nwb_file_name, epochs_descriptions_name in zip(nwb_file_names, epochs_descriptions_names):
            self.JguideraNwbfile.insert1({**key, **{"nwb_file_name": nwb_file_name}}, skip_duplicates=True)
            self.EpochsDescriptions.insert1(
                {**key,
                 **{"nwb_file_name": nwb_file_name, "epochs_descriptions_name": epochs_descriptions_name}},
                skip_duplicates=True)

    def insert_defaults(self, **kwargs):

        # 1) For 2024 reliability paper

        # Run sessions during HR/HL post learning days for SINGLE RATS
        epochs_descriptions_name = "valid_single_contingency_runs"
        for subject_id, nwb_file_names in get_reliability_paper_nwb_file_names(as_dict=True).items():
            recording_set_name = self.get_Haight_single_contingency_rotation_set_name([subject_id])
            epochs_descriptions_names = [epochs_descriptions_name]*len(nwb_file_names)
            self._insert_tables(recording_set_name, nwb_file_names, epochs_descriptions_names)

        # Run sessions during HR/HL post learning days for SINGLE NWB FILES
        epochs_descriptions_name = "valid_single_contingency_runs"
        for entry in fetch_entries_as_dict((EpochsDescriptions & {
                "epochs_descriptions_name": epochs_descriptions_name})):
            nwb_file_name = entry["nwb_file_name"]
            recording_set_name = self.get_single_nwbf_set_name(nwb_file_name, epochs_descriptions_name)
            nwb_file_names = [nwb_file_name]
            epochs_descriptions_names = [entry["epochs_descriptions_name"]]
            self._insert_tables(recording_set_name, nwb_file_names, epochs_descriptions_names)

        # Run sessions during HR/HL post learning days for ALL RATS
        epochs_descriptions_name = "valid_single_contingency_runs"
        nwb_file_names_list = []
        subject_ids = []
        for subject_id, nwb_file_names in get_reliability_paper_nwb_file_names(as_dict=True).items():
            nwb_file_names_list += list(nwb_file_names)
            subject_ids.append(subject_id)
        epochs_descriptions_names = [epochs_descriptions_name] * len(nwb_file_names_list)
        recording_set_name = self.get_Haight_single_contingency_rotation_set_name(subject_ids)
        self._insert_tables(recording_set_name, nwb_file_names_list, epochs_descriptions_names)

        # 2) First day of learning

        for nwb_file_name in ["J1620210529_.nwb"]:
            epochs_descriptions = EpochsDescription().get_single_contingency_descriptions(
                nwb_file_name, exclude_non_full_len=True, exclude_artifact=True)
            for epochs_description in epochs_descriptions:
                recording_set_name = self.get_single_epochs_description_set_name(nwb_file_name, epochs_description)
                self._insert_tables(recording_set_name, [nwb_file_name], [epochs_description])

    def get_matching_nwb_file_names(self, key):
        # Get matching nwb file names based on key

        # There are currently two sources of nwb_file_names: recording_set_name, and nwb_file_name
        # Get nwb_file_names according to each. If these match, return the common set. If not,
        # raise an error

        # Get nwb file names based on nwb_file_name, if passed in key
        nwb_file_names = None
        if "nwb_file_name" in key:
            nwb_file_names = [key["nwb_file_name"]]

        # Get nwb file names based on recording_set_name, if passed in key
        recording_set_nwb_file_names = None
        if "recording_set_name" in key:
            recording_set_nwb_file_names = (self & key).fetch("nwb_file_names")

        # Raise error if nwb file names from the above dont agree
        if all([x is not None for x in [nwb_file_names, recording_set_nwb_file_names]]):
            if set(recording_set_nwb_file_names) != set(nwb_file_names):
                raise Exception(
                    "recording_set_name and nwb_file_name in key, and they disagree on which nwb_file_names to return")

        return nwb_file_names

    def get_matching_recording_set_names(self, key):
        # Return recording set names that correspond to a query dictionary that can contain keys that are directly or
        # indirectly used to construct recording set names.
        # Most directly, we have nwb_file_name and epochs_description_name
        # For now, just handle nwb_file_name case. TODO: handle epochs_description_name and less direct keys like epoch

        # Make copy of key to avoid changing outside function
        key = copy.deepcopy(key)

        # Return recording set name if in key
        if "recording_set_name" in key:
            return (self & key).fetch1("recording_set_name")  # this ensures the passed recording set name is in table

        # Get nwb file name if passed
        nwb_file_name = None
        if "nwb_file_name" in key:
            nwb_file_name = key["nwb_file_name"]

        return [recording_set_name for recording_set_name, nwb_file_names in zip(*self.fetch(
            "recording_set_name", "nwb_file_names")) if np.logical_or(
            nwb_file_name is None, nwb_file_name in nwb_file_names)]

    def get_recording_set_names(self, key_filter=None, recording_set_names_types=None):
        # Return list of recording set names for a given key_filter, for use in populating tables

        # Define key_filter if not passed
        if key_filter is None:
            key_filter = dict()

        # Get default recording set names if not passed
        if recording_set_names_types is None:
            recording_set_names_types = ["Haight_rotation", "Haight_rotation_single_nwb_files"]

        # Check that default recording set names are valid. Default recording set names are used if
        # no information to define recording set names in key_filter
        valid_recording_set_names_types = [
            "Haight_rotation", "Haight_rotation_single_nwb_files", "Haight_rotation_rat_cohort",
            "first_day_learning_single_epoch"]
        check_membership(recording_set_names_types, valid_recording_set_names_types)

        # Include types based on recording_set_names_types:
        # 1) runs across nwb files for a subject during single contingency rotation in Haight phase of
        # behavior (3 nwb files per subject)
        # 2) runs for single nwb files

        recording_set_names = []

        if "Haight_rotation" in recording_set_names_types:
            # 1) Haight rotation, across days, within rats
            recording_set_names += [
                self.get_Haight_single_contingency_rotation_set_name([subject_id]) for subject_id in
                get_reliability_paper_nwb_file_names(as_dict=True).keys()]

        if "Haight_rotation_single_nwb_files" in recording_set_names_types:
            # 2) Haight rotation, within days, within rats
            epochs_descriptions_name = "valid_single_contingency_runs"
            recording_set_names += [
            self.get_single_nwbf_set_name(
                nwb_file_name, epochs_descriptions_name) for nwb_file_name in get_reliability_paper_nwb_file_names()]

        if "Haight_rotation_rat_cohort" in recording_set_names_types:
            # 3) Haight rotation, across days, across rats
            recording_set_names += [self.lookup_rat_cohort_set_name()]

        if "first_day_learning_single_epoch" in recording_set_names_types:
            nwb_file_names = [
                "J1620210529_.nwb", "mango20211101_.nwb", "june20220412_.nwb", "peanut20201101_.nwb",
                "fig20211101_.nwb"]
            for nwb_file_name in nwb_file_names:
                epochs_descriptions = EpochsDescription().get_single_contingency_descriptions(
                    nwb_file_name, exclude_non_full_len=True, exclude_artifact=True)
                for epochs_description in epochs_descriptions:
                    recording_set_names.append(self.get_single_epochs_description_set_name(
                            nwb_file_name, epochs_description))

        # Limit to recording set name if passed in key filter
        if "recording_set_name" in key_filter:
            recording_set_names = [x for x in recording_set_names if x == key_filter["recording_set_name"]]

        # Limit recording names to a subject if nwb_file_name passed in key filter
        if "nwb_file_name" in key_filter:
            recording_set_names = [
                x for x in recording_set_names if x in self.get_matching_recording_set_names(
                {"nwb_file_name": key_filter["nwb_file_name"]})]

        # Return recording set names
        return recording_set_names


@schema
class TrainTestEpoch(dj.Manual):
    definition = """
    # Train and test epochs for a given nwb file
    -> JguideraNwbfile
    train_test_epoch_name : varchar(40)
    ---
    train_epochs : blob
    test_epochs : blob
    """

    @staticmethod
    def make_param_name(train_epochs, test_epochs):
        train_epochs_text = "_".join([str(x) for x in train_epochs])
        test_epochs_text = "_".join([str(x) for x in test_epochs])
        return f"train{train_epochs_text}_test{test_epochs_text}"

    def insert_defaults(self, **kwargs):
        # Insert defaults for decoding analysis for paper 1: same epoch for train and test
        epochs_description_name = "valid_single_contingency_runs"
        for nwb_file_name in get_reliability_paper_nwb_file_names():
            epochs_descriptions = (EpochsDescriptions & {
                "nwb_file_name": nwb_file_name, "epochs_description_name": epochs_description_name}).fetch1(
                "epochs_descriptions")
            for epochs_description in epochs_descriptions:
                epoch = (EpochsDescription & {
                    "nwb_file_name": nwb_file_name, "epochs_description": epochs_description}).get_epoch()
                train_epochs = test_epochs = [epoch]
                train_test_epoch_name = self.make_param_name(train_epochs, test_epochs)
                key = {
                    "nwb_file_name": nwb_file_name, "train_test_epoch_name": train_test_epoch_name,
                    "train_epochs": train_epochs, "test_epochs": test_epochs}
                self.insert1(key, skip_duplicates=True)

    def lookup_train_test_epoch_name(self, train_epochs, test_epochs, nwb_file_name=None):
        if nwb_file_name is None:
            nwb_file_name = check_return_single_element(self.fetch("nwb_file_name")).single_element
        key = {"nwb_file_name": nwb_file_name, "train_test_epoch_name": self.make_param_name(train_epochs, test_epochs)}
        return (self & key).fetch1("train_test_epoch_name")

    def get_train_test_epoch(self):
        # Return single train and test epoch
        train_epochs, test_epochs = self.fetch1("train_epochs", "test_epochs")
        return unpack_single_element(train_epochs), unpack_single_element(test_epochs)

    def get_epochs_description(self):
        train_epochs, test_epochs = self.fetch1("train_epochs", "test_epochs")
        epochs = np.unique(train_epochs + test_epochs)
        return EpochsDescription().lookup_epochs_description(self.fetch1("nwb_file_name"), epochs)


@schema
class TrainTestEpochSet(NwbfSetBase):
    definition = """
    # Group of nwb files and corresponding train/test epochs
    train_test_epoch_set_name : varchar(80)
    ---
    nwb_file_names : blob
    train_test_epoch_names : blob
    """

    class TrainTestEpoch(dj.Part):
        definition = """
        # Achieves upstream dependence on TrainTestEpoch
        -> TrainTestEpochSet
        -> TrainTestEpoch
        """

    def get_single_nwbf_same_train_test_set_name(self, nwb_file_name, train_test_epoch_set_name):
        return self.get_single_nwbf_set_name(nwb_file_name, train_test_epoch_set_name)

    def _insert_tables(self, train_test_epoch_set_name, nwb_file_names, train_test_epoch_names):
        # Insert into main table
        key = {"train_test_epoch_set_name": train_test_epoch_set_name}
        self.insert1({**key, **{
            "nwb_file_names": nwb_file_names, "train_test_epoch_names": train_test_epoch_names}},
                     skip_duplicates=True)

        # Insert into parts table
        for nwb_file_name, train_test_epoch_name in zip(nwb_file_names, train_test_epoch_names):
            self.TrainTestEpoch.insert1(
                {**key, **{"nwb_file_name": nwb_file_name, "train_test_epoch_name": train_test_epoch_name}},
                skip_duplicates=True)

    def insert_defaults(self, **kwargs):
        # For first paper
        # For state space analysis
        # HR/HL post learning days for each rat (3 days per rat), train / test on same epoch
        epochs_descriptions_name = "valid_single_contingency_runs"
        for subject_id, nwb_file_names in get_reliability_paper_nwb_file_names(as_dict=True).items():
            train_test_epoch_names = []
            nwb_file_names_ = []
            for nwb_file_name in nwb_file_names:
                epochs_descriptions = (EpochsDescriptions & {
                    "nwb_file_name": nwb_file_name, "epochs_descriptions_name": epochs_descriptions_name}).fetch1(
                    "epochs_descriptions")
                for epochs_description in epochs_descriptions:
                    epoch = (EpochsDescription & {"nwb_file_name": nwb_file_name,
                                                  "epochs_description": epochs_description}).get_epoch()
                    train_test_epoch_names.append(
                        (TrainTestEpoch & {"nwb_file_name": nwb_file_name}).lookup_train_test_epoch_name(
                        [epoch], [epoch]))
                    nwb_file_names_.append(nwb_file_name)
            train_test_epoch_set_name = self.get_Haight_single_contingency_rotation_set_name([subject_id])
            self._insert_tables(train_test_epoch_set_name, nwb_file_names_, train_test_epoch_names)

        # Valid runs for single nwb files
        epochs_descriptions_name = "valid_single_contingency_runs"
        for subject_id, nwb_file_names in get_reliability_paper_nwb_file_names(as_dict=True).items():
            for nwb_file_name in nwb_file_names:
                epochs_descriptions = (EpochsDescriptions & {
                    "nwb_file_name": nwb_file_name, "epochs_descriptions_name": epochs_descriptions_name}).fetch1(
                    "epochs_descriptions")
                train_test_epoch_names = []
                nwb_file_names_ = []
                for epochs_description in epochs_descriptions:
                    epoch = (EpochsDescription & {"nwb_file_name": nwb_file_name,
                                                  "epochs_description": epochs_description}).get_epoch()
                    train_test_epoch_names.append(
                        (TrainTestEpoch & {"nwb_file_name": nwb_file_name}).lookup_train_test_epoch_name(
                        [epoch], [epoch]))
                    nwb_file_names_.append(nwb_file_name)
                train_test_epoch_set_name = self.get_single_nwbf_same_train_test_set_name(
                    nwb_file_name, epochs_descriptions_name)
                self._insert_tables(train_test_epoch_set_name, nwb_file_names_, train_test_epoch_names)

        # HR/HL post learning days across rats
        epochs_descriptions_name = "valid_single_contingency_runs"
        train_test_epoch_names = []
        nwb_file_names_ = []
        subject_ids = []
        for subject_id, nwb_file_names in get_reliability_paper_nwb_file_names(as_dict=True).items():
            for nwb_file_name in nwb_file_names:
                epochs_descriptions = (EpochsDescriptions & {
                    "nwb_file_name": nwb_file_name, "epochs_descriptions_name": epochs_descriptions_name}).fetch1(
                    "epochs_descriptions")
                for epochs_description in epochs_descriptions:
                    epoch = (EpochsDescription & {"nwb_file_name": nwb_file_name,
                                                  "epochs_description": epochs_description}).get_epoch()
                    train_test_epoch_names.append(
                        (TrainTestEpoch & {"nwb_file_name": nwb_file_name}).lookup_train_test_epoch_name(
                            [epoch], [epoch]))
                    nwb_file_names_.append(nwb_file_name)
            subject_ids.append(subject_id)
        # Get name of train test epoch set
        train_test_epoch_set_name = self.get_Haight_single_contingency_rotation_set_name(subject_ids)
        # Insert into tables
        self._insert_tables(train_test_epoch_set_name, nwb_file_names_, train_test_epoch_names)

    def get_matching_train_test_epoch_set_names(self, key):
        # Return train test epoch set names that correspond to a query dictionary that can contain keys that
        # are directly or indirectly used to construct train test epoch set names.
        # Most directly, we have nwb_file_name and epochs_description_name
        # For now, just handle nwb_file_name case; TODO: handle epochs_description_name and less direct keys like epoch

        # Make copy of key to avoid changing outside function
        key = copy.deepcopy(key)

        # Return train test epoch set name if in key
        if "train_test_epoch_set_name" in key:
            # this ensures the passed train test epoch set name is in table
            return (self & key).fetch1("train_test_epoch_set_name")

        # Get nwb file name if passed
        nwb_file_name = None
        if "nwb_file_name" in key:
            nwb_file_name = key["nwb_file_name"]

        return [train_test_epoch_set_name for train_test_epoch_set_name, nwb_file_names in zip(*self.fetch(
            "train_test_epoch_set_name", "nwb_file_names")) if np.logical_or(
            nwb_file_name is None, nwb_file_name in nwb_file_names)]

    def get_train_test_epoch_set_names(self, key_filter=None, train_test_epoch_set_names_types=None):
        # Return list of train test epoch set names for a given key_filter, for use in populating tables

        # Define key_filter if not passed
        if key_filter is None:
            key_filter = dict()

        # Get train test epoch set names if not passed
        if train_test_epoch_set_names_types is None:
            train_test_epoch_set_names_types = ["Haight_rotation", "Haight_rotation_single_nwb_files"]

        # Check that default train test epoch set names are valid. Default train test epoch set names are used if
        # no information to define train test epoch set names in key_filter
        valid_train_test_epoch_set_names_types = [
            "Haight_rotation", "Haight_rotation_single_nwb_files", "Haight_rotation_rat_cohort"]
        check_membership(train_test_epoch_set_names_types, valid_train_test_epoch_set_names_types)

        # Include types based upon train_test_epoch_set_names_types:
        # 1) runs across nwb files for a subject during single contingency rotation in Haight phase of
        # behavior (3 nwb files per subject)
        # 2) runs for single nwb files
        train_test_epoch_set_names = []
        if "Haight_rotation" in train_test_epoch_set_names_types:
            # 1) Haight rotation, across days, within rats
            train_test_epoch_set_names += [
                self.get_Haight_single_contingency_rotation_set_name([subject_id]) for subject_id in
                get_reliability_paper_nwb_file_names(as_dict=True).keys()]
        if "Haight_rotation_single_nwb_files" in train_test_epoch_set_names_types:
            # 2) Haight rotation, within days, within rats, sam epoch
            epochs_descriptions_name = "valid_single_contingency_runs"
            train_test_epoch_set_names += [
            self.get_single_nwbf_same_train_test_set_name(
                nwb_file_name, epochs_descriptions_name) for nwb_file_name in get_reliability_paper_nwb_file_names()]
        if "Haight_rotation_rat_cohort" in train_test_epoch_set_names_types:
            # 3) Haight rotation, across days, across rats
            train_test_epoch_set_names += [self.lookup_rat_cohort_set_name()]

        # Limit to train test epoch set name if passed in key filter
        if "train_test_epoch_set_name" in key_filter:
            train_test_epoch_set_names = [
                x for x in train_test_epoch_set_names if x == key_filter["train_test_epoch_set_name"]]

        # Limit train test epoch names to a subject if nwb_file_name passed in key filter
        if "nwb_file_name" in key_filter:
            train_test_epoch_set_names = [
                x for x in train_test_epoch_set_names if x in self.get_matching_train_test_epoch_set_names(
                {"nwb_file_name": key_filter["nwb_file_name"]})]

        return train_test_epoch_set_names


def populate_jguidera_epoch(key=None, tolerate_error=False):
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_epoch():
    schema.drop()
