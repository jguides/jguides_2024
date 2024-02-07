import itertools

import datajoint as dj
import numpy as np
from spyglass.common import Nwbfile, TaskEpoch

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert, insert1_print, \
    get_relationship_text
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_jguidera_nwbf_names
from src.jguides_2024.utils.df_helpers import df_from_data_list, df_pop
from src.jguides_2024.utils.dict_helpers import invert_dict
from src.jguides_2024.utils.set_helpers import check_membership

schema = dj.schema("jguidera_metadata")


# TODO (table drop): if drop table, alter table title in heading to be "Selection from upstream tables for JguideraNwbfile"
@schema
class JguideraNwbfileSel(SelBase):
    definition = """
    # Available sort group IDs for nwb files
    -> Nwbfile
    """

    # Overrides parent class method so can restrict nwb file names
    def insert_defaults(self, **kwargs):
        valid_nwb_file_names = [nwb_file_name for nwb_file_name in Nwbfile.fetch("nwb_file_name") if nwb_file_name
                                in get_jguidera_nwbf_names(high_priority=False, highest_priority=False)]
        for nwb_file_name in valid_nwb_file_names:
            self.insert1({"nwb_file_name": nwb_file_name}, skip_duplicates=True)


@schema
class JguideraNwbfile(ComputedBase):
    definition = """
    # jguidera nwb files
    -> JguideraNwbfileSel
    """

    def make(self, key):
        self.insert1(key)


@schema
class TaskIdentification(ComputedBase):
    definition = """
    # Epoch environment and reward contingency
    -> TaskEpoch
    ---
    contingency : varchar(40)
    task_environment : varchar(40)
    """

    def make(self, key):

        # Check that key corresponds to desired recording files
        if key['nwb_file_name'] not in get_jguidera_nwbf_names(highest_priority=False, high_priority=False):
            raise Exception("key with nwb_file_name: {nwb_file_name} not allowed".format(**key))

        # Parse entry in TaskEpoch
        task_epoch_entry = (TaskEpoch & key).fetch1()
        # Assumes task_name has either form: "trackName_contingencyName_environmentName_delay.sc" or "sleep" or "home"
        if len(task_epoch_entry["task_name"].split("_")) == 1:
            key["contingency"] = task_epoch_entry["task_name"]
        else:
            key["contingency"] = task_epoch_entry["task_name"].split("_")[1]
        key["task_environment"] = task_epoch_entry["task_environment"]

        # Insert into table
        insert1_print(self, key)

    def get_epochs(self, nwb_file_name):
        return (self & {"nwb_file_name": nwb_file_name}).fetch("epoch")

    @classmethod
    def get_contingency(cls, nwb_file_name, epoch):
        return (cls & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1("contingency")

    @classmethod
    def get_environment(cls, nwb_file_name, epoch):
        return (cls & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1("task_environment")

    @classmethod
    def get_contingency_color(cls, nwb_file_name, epoch):
        return get_contingency_color(cls.get_contingency(nwb_file_name, epoch))

    @classmethod
    def get_environment_color(cls, nwb_file_name, epoch):
        return get_environment_color(cls.get_environment(nwb_file_name, epoch))

    @staticmethod
    def single_contingencies():
        return ["centerAlternation", "handleAlternation"]

    def is_single_contingency_epoch(self, nwb_file_name, epoch):
        contingency = (self & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1("contingency")
        return contingency in self.single_contingencies()

    def get_single_contingency_epoch_map(self, nwb_file_name, epochs=None):
        single_contingencies = self.single_contingencies()
        # Restrict to passed epochs
        epochs_, contingencies = (self & {"nwb_file_name": nwb_file_name}).fetch("epoch", "contingency")
        if epochs is not None:
            valid_bool = [x in epochs for x in epochs_]
            epochs_ = epochs_[valid_bool]
            contingencies = contingencies[valid_bool]
        return {contingency: epochs_[contingencies == contingency] for contingency in single_contingencies}

    def get_single_contingency_epoch_pair_map(self, nwb_file_name, epochs=None):
        single_contingency_epoch_map = self.get_single_contingency_epoch_map(nwb_file_name, epochs)
        return {k: list(itertools.combinations(v, r=2)) for k, v in single_contingency_epoch_map.items()}

    def get_same_environment_epoch_map(self, nwb_file_name):
        epochs, environments = (self & {"nwb_file_name": nwb_file_name}).fetch("epoch", "task_environment")
        return {environment: epochs[environments == environment] for environment in environments}

    def get_single_contingency_epoch_pairs_by_envs_relationship(self, nwb_file_name, epochs=None):
        single_contingency_epoch_pair_map = self.get_single_contingency_epoch_pair_map(nwb_file_name, epochs)
        same_contingency_pairs = np.concatenate(list(single_contingency_epoch_pair_map.values()))
        return invert_dict({tuple(epochs): get_relationship_text(
            *[self.get_environment(nwb_file_name, epoch) for epoch in epochs])
                     for epochs in same_contingency_pairs})

    def get_line_param(self, param_name):
        # Make sure param name valid
        check_membership([param_name], ["linestyle", "alpha"])
        # Return param value for given contingency and task environment
        line_df = df_from_data_list([
            ("centerAlternation", "HaightRight", "solid", .8), ("centerAlternation", "HaightLeft", "solid", .3),
            ("handleAlternation", "HaightRight", "dashed", .8), ("handleAlternation", "HaightLeft", "dashed", .3)],
            ["contingency", "task_environment", "linestyle", "alpha"])
        return df_pop(line_df, {k: self.fetch1(k) for k in ["contingency", "task_environment"]}, param_name)

    def get_line_params(self):
        return self.get_line_df

    def populate_(self, high_priority=False, highest_priority=False, **kwargs):
        for nwb_file_name in get_jguidera_nwbf_names(high_priority, highest_priority):
            self.populate({"nwb_file_name": nwb_file_name})
        return [self.table_name]


def get_contingency_color_map():
    return {"centerAlternation": "darkgray",
            "handleAlternation": "lightgray",
            "handleThenCenterAlternation": "black",
            "centerThenHandleAlternation": "black"}


def get_contingency_color(contingency):
    return get_contingency_color_map()[contingency]


def get_environment_color_map():
    return {"HaightRight": "brown",
            "HaightLeft": "peru",
            "SA": "orange"}


def get_environment_color(environment):
    return get_environment_color_map()[environment]


def populate_jguidera_task_identification(key=None, tolerate_error=False):
    schema_name = "jguidera_metadata"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_task_identification():
    schema.drop()

