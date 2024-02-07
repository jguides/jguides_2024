import datajoint as dj
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert
from src.jguides_2024.utils.vector_helpers import vector_midpoints

schema = dj.schema("jguidera_warped_axis_bins")


@schema
class WarpedAxisBinsParams(SecKeyParamsBase):
    definition = """
    # Parameters for WarpedAxisBins
    warped_axis_bins_param_name : varchar(40)
    ---
    warped_axis_start : decimal(10,5) unsigned
    warped_axis_end : decimal(10,5) unsigned
    warped_axis_bin_width : decimal(10,5) unsigned
    """

    def _default_params(self):
        return [(0, 1, .05)]

    def fetch1_params(self):
        # Return params as float
        return {k: float(v) for k, v in self.fetch1().items() if k not in self.primary_key}


@schema
class WarpedAxisBins(ComputedBase):
    definition = """
    # Bins along an arbitrary axis
    -> WarpedAxisBinsParams
    ---
    bin_edges : blob
    bin_centers : blob
    """

    def make(self, key):
        params = (WarpedAxisBinsParams & key).fetch1_params()
        bin_width = params["warped_axis_bin_width"]
        bin_edges = np.arange(params["warped_axis_start"], params["warped_axis_end"] + bin_width, bin_width)
        key.update({"bin_edges": bin_edges,
                    "bin_centers": vector_midpoints(bin_edges)})
        self.insert1(key)


def populate_jguidera_warped_axis_bins(key=None, tolerate_error=False):
    schema_name = "jguidera_warped_axis_bins"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_warped_axis_bins():
    schema.drop()

