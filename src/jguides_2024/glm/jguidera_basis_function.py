from collections import namedtuple

import datajoint as dj
import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert, replace_param_name_chars, delete_
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt
from src.jguides_2024.utils.basis_function_helpers import RaisedCosineBasis as RCB
from src.jguides_2024.utils.basis_function_helpers import plot_basis_functions

schema = dj.schema("jguidera_basis_function")


# If want to shorten param name, could use combination of covariate_group_name and an integer id. In this case,
# would want a param name table
@schema
class RaisedCosineBasisParams(SecKeyParamsBase):
    definition = """
    # Params for raised cosine basis
    raised_cosine_basis_param_name : varchar(100)
    ---
    covariate_group_name : varchar(40)  # this is useful for naming basis functions
    domain : blob
    bin_width : float
    num_basis_functions : int
    """

    @staticmethod
    def shorthand_params_map():
        # Return map from shorthand for raised_cosine_basis_param_name to underlying params
        # Return params in objects
        ShorthandParams = namedtuple("ShorthandParams", "covariate_group_name domain bin_width num_basis_functions")
        return {"ppt": ShorthandParams("ppt", Ppt.get_range(), .05, 14),
                "delay": ShorthandParams("delay", [0, 2], .1, 14)}

    def _default_params(self):
        shorthand_params_map = self.shorthand_params_map()
        return [list(x._asdict().values()) for x in shorthand_params_map.values()]

    # Override parent class method so can drop brackets in ppt range text, for better readability and so
    # param name here consistent with when appears in pool tables later, where brackets are dropped
    def _make_param_name(self, secondary_key_subset_map, separating_character=None):
        param_name = super()._make_param_name(secondary_key_subset_map, separating_character)
        return replace_param_name_chars(param_name)

    def lookup_param_name_from_shorthand(self, shorthand_param_name):
        return self.lookup_param_name(list(self.shorthand_params_map()[shorthand_param_name]._asdict().values()))

    def delete_(self, key, safemode=True):
        from src.jguides_2024.glm.jguidera_measurements_interp_pool import XInterpPool
        delete_(self, [XInterpPool], key, safemode)


# Note that table name must be on a single line, so cannot split line with table name in definition below
@schema
class RaisedCosineBasis(ComputedBase):
    definition = """
    # Raised cosine basis as in Park et al. 2014: Encoding and decoding in parietal cortex during sensorimotor decision-making
    -> RaisedCosineBasisParams
    ---
    basis_bin_edges : blob
    basis_bin_centers : blob
    basis_functions : mediumblob
    """

    def make(self, key):

        params_entry = (RaisedCosineBasisParams & key).fetch1()
        raised_cosine_basis_obj = RCB(**{
            k: params_entry[k] for k in ["domain", "bin_width", "num_basis_functions", "covariate_group_name"]})
        self.insert1({**key,
                      **{"basis_bin_edges": raised_cosine_basis_obj.basis_bin_edges,
                         "basis_bin_centers": raised_cosine_basis_obj.basis_bin_centers,
                         "basis_functions": raised_cosine_basis_obj.basis_functions.to_dict(orient='list')}})

    def plot_basis_functions(self, raised_cosine_basis_param_name):

        entry = (self & {"raised_cosine_basis_param_name": raised_cosine_basis_param_name}).fetch1()
        basis_function_names, basis_functions = zip(*entry['basis_functions'].items())
        plot_basis_functions(np.asarray(basis_functions).T,
                             basis_functions_x=entry['basis_bin_centers'],
                             basis_function_names=np.asarray(basis_function_names))

    def fetch1_basis_functions(self):
        # Convert basis functions to df
        return pd.DataFrame.from_dict(self.fetch1("basis_functions"))


def populate_jguidera_basis_function(key=None, tolerate_error=False):
    schema_name = "jguidera_basis_function"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_basis_function():
    schema.drop()
