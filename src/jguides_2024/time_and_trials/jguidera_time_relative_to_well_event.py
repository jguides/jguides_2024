import copy

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, ComputedBase, SecKeyParamsBase, \
    CovariateRCB, CovariateDigParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, delete_
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.glm.jguidera_basis_function import RaisedCosineBasisParams, RaisedCosineBasis
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellTrials, DioWellDDTrials, DioWellDDTrialsParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPool, ResTimeBinsPoolSel, \
    populate_jguidera_res_time_bins_pool
from src.jguides_2024.utils.array_helpers import min_positive_val_arr
from src.jguides_2024.utils.basis_function_helpers import sample_basis_functions
from src.jguides_2024.utils.dict_helpers import add_defaults
from src.jguides_2024.utils.digitize_helpers import digitize_indexed_variable
from src.jguides_2024.utils.make_bins import get_peri_event_bin_edges
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.vector_helpers import unpack_single_element, vector_midpoints

# These imports are called with eval or used in table definitions (do not remove):
nd

schema = dj.schema("jguidera_time_relative_to_well_event")


@schema
class TimeRelWASel(SelBase):
    definition = """
    # Selection from upstream tables for TimeRelWA
    -> DioWellTrials
    -> ResTimeBinsPool
    """

    def _get_potential_keys(self, key_filter=None):
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_param, get_fr_vec_default_param
        res_time_bins_pool_param_names = [
                get_fr_vec_default_param("res_time_bins_pool_param_name"),
            get_glm_default_param("delay_res_time_bins_pool_param_name")]
        return [k for k in super()._get_potential_keys(key_filter) if k["res_time_bins_pool_param_name"] in
                res_time_bins_pool_param_names]


@schema
class TimeRelWA(ComputedBase):
    definition = """
    # Time relative to well arrival
    -> TimeRelWASel
    ---
    -> nd.common.AnalysisNwbfile
    time_rel_wa_object_id : varchar(100)
    """

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves upstream dependence on DioWellDDTrials
        -> TimeRelWASel
        -> DioWellDDTrials
        """

    def make(self, key):

        # Get time bin centers
        time_vector = (ResTimeBinsPool & key).fetch1_dataframe().time_bin_centers.values

        # Get well arrival times
        well_arrival_times = (DioWellTrials & key).fetch1("well_arrival_times")

        # Get time to/from next well arrival
        time_tile = np.tile(time_vector, (len(well_arrival_times), 1))  # columns correspond to time
        wa_tile = np.tile(well_arrival_times, (len(time_vector), 1)).T  # each row corresponds to a well arrival
        time_to_wa = min_positive_val_arr(wa_tile - time_tile, axis=0)
        time_from_wa = min_positive_val_arr(time_tile - wa_tile, axis=0)

        # Replace inf with nan
        # setting df index useful for adding trials information below
        df = pd.DataFrame.from_dict(
            {"time": time_vector, "time_to_wa": time_to_wa, "time_from_wa": time_from_wa}).set_index("time")
        df.replace(np.inf, np.nan, inplace=True)

        # Add information about what well animal is traveling to/from, and path animal is on or came from
        # (collectively "path_name")
        # ...Add param indicating no start/end shift to dd trials to key
        dd_trials_meta_param_name = DioWellDDTrialsParams().meta_param_name()
        key.update({dd_trials_meta_param_name: DioWellDDTrialsParams().lookup_no_shift_param_name()})
        column_names = ["trial_start_well_names", "trial_end_well_names", "path_names"]
        trials_info_df = (DioWellDDTrials & key).label_time_vector(time_vector, column_names, add_dd_text=True)
        # ...Flag above column names for replacing None with "none"
        replace_none_col_names = [x for x in trials_info_df.columns if any([y in x for y in column_names])]
        df = pd.concat((df, trials_info_df), axis=1)

        # Store in table
        main_key = copy.deepcopy(key)
        main_key.pop(dd_trials_meta_param_name)
        insert_analysis_table_entry(self, [df], main_key, reset_index=True, replace_none_col_names=replace_none_col_names)

        # Insert into part table
        self.DioWellDDTrials.insert1(key)

    def plot_results(self):
        df = self.fetch1_dataframe()
        fig, ax = plt.subplots()
        for x in ["time_to_wa", "time_from_wa"]:
            ax.plot(df[x], '.', label=x)
        ax.legend()

    def get_bin_edges_map(self, bin_width):
        # Define time relative to well arrival bins
        df = self.fetch1_dataframe()
        return {column_name: get_peri_event_bin_edges([0, np.nanmax(df[column_name])], bin_width=bin_width)
                for column_name in ["time_to_wa", "time_from_wa"]}

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def delete_(self, key=None, safemode=True):
        # Delete downstream entries first
        from src.jguides_2024.jguidera_firing_rate_difference_vector_similarity_ave import FRDiffVecCosSimWANnAve
        delete_(self, [FRDiffVecCosSimWANnAve], key, safemode)


@schema
class TimeRelWADigParams(CovariateDigParamsBase):
    definition = """
    # Parameters for TimeRelWADig
    time_rel_wa_dig_param_name : varchar(40)
    ---
    time_bin_width : float  # seconds
    """

    def _default_params(self):
        return [[.1],  # GLM analysis
                [.25],  # firing rate vector analysis
                ]

    def get_bin_edges_map(self, **kwargs):
        key = kwargs.pop("key", dict())
        # If key passed, and epoch not in key but epochs_description is, get epoch from epochs_description
        if "epoch" not in key and "epochs_description" in key:
            key["epoch"] = unpack_single_element((EpochsDescription & key).fetch1("epochs"))
        # Get bin width
        bin_width = (self & key).fetch1("time_bin_width")
        return (TimeRelWA & key).get_bin_edges_map(bin_width)

    def make_bin_edges(self, **kwargs):
        # Check bin edges type passed
        if "bin_edges_type" not in kwargs:
            raise Exception(f"must pass bin_edges_type to specify whether want bin edges for time relative to, "
                            f"or from, well arrival")
        bin_edges_type = kwargs.pop("bin_edges_type")

        # Check valid bin edges type
        check_membership([bin_edges_type], ["time_to_wa", "time_from_wa"])

        # Get map from time to or from wa to bin edges
        bin_edges_map = self.get_bin_edges_map(**kwargs)

        return bin_edges_map[bin_edges_type]

    def get_valid_bin_nums(self, **kwargs):
        bin_centers = self.get_bin_centers(**kwargs)
        if any(bin_centers == 0):
            raise Exception(f"bin_centers not expected to contain zero")
        return np.arange(1, len(bin_centers) + 1)


@schema
class TimeRelWADigSel(SelBase):
    definition = """
    # Selection from upstream tables for TimeRelWADig
    -> TimeRelWA
    -> TimeRelWADigParams
    """

    # Restrict combination of time bin width (TimeRelWADigParams) and time bins param name
    def _get_potential_keys(self, key_filter=None):
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_params_map, get_fr_vec_default_params_map
        if key_filter is None:
            key_filter = dict()
        glm_default_params_map = get_glm_default_params_map()
        wa_fr_vec_default_params_map = get_fr_vec_default_params_map()
        keys = []
        for key in (TimeRelWA & key_filter).fetch("KEY"):
            time_rel_wa_dig_param_name = None
            if key["res_time_bins_pool_param_name"] == glm_default_params_map["delay_res_time_bins_pool_param_name"]:
                time_rel_wa_dig_param_name = glm_default_params_map["time_rel_wa_dig_param_name"]
            elif key["res_time_bins_pool_param_name"] == wa_fr_vec_default_params_map["res_time_bins_pool_param_name"]:
                time_rel_wa_dig_param_name = wa_fr_vec_default_params_map["time_rel_wa_dig_param_name"]
            if time_rel_wa_dig_param_name is not None:
                key.update({"time_rel_wa_dig_param_name": time_rel_wa_dig_param_name})
                keys.append(key)
        return keys


@schema
class TimeRelWADig(ComputedBase):
    definition = """
    # Digitized time relative to well arrival
    -> TimeRelWADigSel
    ---
    -> nd.common.AnalysisNwbfile
    time_rel_wa_dig_object_id : varchar(100)
    """

    def make(self, key):
        bin_edges_map = TimeRelWADigParams().get_bin_edges_map(key=key)
        df = (TimeRelWA & key).fetch1_dataframe()
        # Replace relative time column values with digitized version
        rel_time_column_names = ["time_to_wa", "time_from_wa"]
        for column_name in rel_time_column_names:
            df[f"digitized_{column_name}"] = digitize_indexed_variable(
                df[column_name], bin_edges=bin_edges_map[column_name])
        df.drop(columns=rel_time_column_names, inplace=True)
        insert_analysis_table_entry(self, [df], key, reset_index=True)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)


# Notes on TimeRelWADigSingleAxis tables:
# - These tables put digitized time relative to well arrival on a single axis. Rules for conversion:
# 1) we make no change to integers, which are output from np.digitize and begin at one, that correspond to bins
# after well arrival
# 2) integers corresponding to bins before well arrival are "flipped" around the origin to be negative,
# then one is added to them so that -1 becomes 0, -2 becomes -1, and so on. The reason we do this, rather
# than leave a gap at zero, is that downstream analyses expect adjacent bins to be separated by one. So
# in order for the bins before and after the origin to be seen as consecutive by these analyses, they must
# be separated by one
@schema
class TimeRelWADigSingleAxisParams(SecKeyParamsBase):
    definition = """
    # Digitized time relative to well arrival on a single axis
    time_rel_wa_dig_single_axis_param_name : varchar(40)
    ---
    rel_time_start : float
    rel_time_end : float
    """

    def _default_params(self):
        return [[-2, 2],  # for firing rate vector analysis
                [0, 2]]  # for GLM analysis

    def get_bin_edges(self, key):
        rel_time_start, rel_time_end = (self & key).fetch1("rel_time_start", "rel_time_end")
        bin_edges_map = (TimeRelWADigParams & key).get_bin_edges_map(key=key)
        single_axis_bin_edges = np.sort(np.concatenate(
            (-bin_edges_map["time_to_wa"], bin_edges_map["time_from_wa"])))
        # ...Remove second zero (occurs if time to / from bin edges meet at origin)
        zero_idxs = np.where(single_axis_bin_edges == 0)[0]
        if len(zero_idxs) == 2:
            single_axis_bin_edges = np.delete(single_axis_bin_edges, zero_idxs[0])
        # ...Restrict to bins that fall within relative time bounds
        valid_bool = np.logical_and(single_axis_bin_edges >= rel_time_start, single_axis_bin_edges <= rel_time_end)
        return single_axis_bin_edges[valid_bool]

    def get_bin_centers(self, key):
        return vector_midpoints(self.get_bin_edges(key))

    def get_valid_bin_nums(self, key):
        # Bin nums before well arrival start at zero and decrease by one, moving away from well arrival.
        # Bin nums after wella rrival start at one and increase by one, moving away from well arrival.
        bin_centers = self.get_bin_centers(key)
        # Check no bin centers are zero (this is not expected, and if it occurred, we would need to consder how
        # to account for it in code)
        if any(bin_centers == 0):
            raise Exception(f"Did not expect a bin center to be zero, as bin edges should be completely on one or the"
                            f"other side of well arrival")
        num_pos = np.sum(bin_centers > 0)
        num_neg = np.sum(bin_centers < 0)
        return np.sort(np.concatenate((-np.arange(0, num_neg), np.arange(1, num_pos + 1))))

    def delete_(self, key, safemode=True):
        delete_(self, [TimeRelWADigSingleAxisSel], key, safemode)


@schema
class TimeRelWADigSingleAxisSel(SelBase):
    definition = """
    # Digitized time relative to well arrival on a single axis
    -> TimeRelWADig
    -> TimeRelWADigSingleAxisParams
    """

    # Override parent class method to ensure valid time interval in res_time_bins_pool_param_name
    # consistent with domain in time_rel_wa_dig_single_axis_param_name
    def _get_potential_keys(self, key_filter=None):
        if key_filter is None:
            key_filter = dict()
        # Loop through default param sets
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_params_map, get_fr_vec_default_params_map
        keys = []
        for params_map, meta_shorthand_name in zip([
            get_glm_default_params_map(), get_fr_vec_default_params_map()],
                ["delay_time_bins_shorthand", "time_bins_shorthand"]):
            # Make new copy of key filter each time through loop to avoid using values from previous loop iteration
            key_filter_ = copy.deepcopy(key_filter)
            time_bins_shorthand = params_map[meta_shorthand_name]
            if time_bins_shorthand in ResTimeBinsPoolSel().get_shorthand_params_map():
                domain = ResTimeBinsPoolSel().get_shorthand_params_map()[time_bins_shorthand].domain
                key_filter_.update({
                    "time_rel_wa_dig_single_axis_param_name": TimeRelWADigSingleAxisParams().lookup_param_name(domain),
                })
            key_filter_.update({
                "res_time_bins_pool_param_name": ResTimeBinsPoolSel().lookup_param_name_from_shorthand(
                    time_bins_shorthand)})
            keys += super()._get_potential_keys(key_filter=key_filter_)
        return keys

    def delete_(self, key, safemode=True):
        delete_(self, [TimeRelWADigSingleAxis], key, safemode)


@schema
class TimeRelWADigSingleAxis(ComputedBase):
    definition = """
    # Digitized time relative to well arrival on a single axis
    -> TimeRelWADigSingleAxisSel
    ---
    -> nd.common.AnalysisNwbfile
    time_rel_wa_dig_single_axis_df_object_id : varchar(40)
    """

    @staticmethod
    def _convert_pre_wa_int(x):
        return -x + 1

    def make(self, key):
        # Get params and time relative to well arrival
        bins_per_s = 1 / (TimeRelWADigParams & key).fetch1("time_bin_width")
        rel_time_start, rel_time_end = (TimeRelWADigSingleAxisParams & key).fetch1("rel_time_start", "rel_time_end")
        time_rel_wa_dig_df = (TimeRelWADig & key).fetch1_dataframe()

        # For bins that fall within BOTH pre or post well arrival period, assign bin to just pre well
        # arrival period (choice here has been to not "double count" times)
        num_bins_pos = np.floor(bins_per_s * rel_time_end)
        num_bins_neg = np.floor(-bins_per_s * rel_time_start)
        neg_bool = time_rel_wa_dig_df.digitized_time_to_wa.between(0, num_bins_neg)
        pos_bool = time_rel_wa_dig_df.digitized_time_from_wa.between(0, num_bins_pos)
        pos_bool[neg_bool * pos_bool] = False  # if in pre well period, remove from post well period

        # Convert digitized time relative to well arrival to single axis
        # ...Initialize vector with nans. Note that datatype will be float instead of int to enable use of nan
        time_rel_wa_comb = pd.Series([np.nan] * len(time_rel_wa_dig_df), index=time_rel_wa_dig_df.index)
        time_rel_wa_comb[neg_bool] = [
            self._convert_pre_wa_int(x) for x in time_rel_wa_dig_df.digitized_time_to_wa[neg_bool]]
        time_rel_wa_comb[pos_bool] = time_rel_wa_dig_df.digitized_time_from_wa[pos_bool]
        time_rel_wa_dig_single_axis_df = pd.DataFrame.from_dict(
            {time_rel_wa_comb.index.name: time_rel_wa_comb.index, "time_rel_wa": time_rel_wa_comb})
        insert_analysis_table_entry(self, [time_rel_wa_dig_single_axis_df], key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def delete_(self, key, safemode=True):
        from src.jguides_2024.firing_rate_vector.jguidera_well_event_firing_rate_vector import TimeRelWAFRVecSel
        delete_(self, [TimeRelWARCBSel, TimeRelWAFRVecSel], key, safemode)

    def drop_(self):
        raise Exception(f"must finish writing")




@schema
class TimeRelWARCBSel(SelBase):
    definition = """
    # Selection from upstream tables for TimeRelWARCB
    -> TimeRelWADigSingleAxis
    -> RaisedCosineBasis
    """

    def insert1(self, key, **kwargs):
        # Ensure bin width match across raised cosine basis and time relative to well arrival
        x1 = (RaisedCosineBasisParams & key).fetch1("bin_width")
        x2 = (TimeRelWADigParams & key).fetch1("time_bin_width")
        if x1 != x2:
            raise Exception(
                f"bin_width in RaisedCosineBasisParams must match time_bin_width in TimeRelWADigParams. "
                f"These are respectively {x1} and {x2}")
        # Ensure range matches across raised cosine basis and time relative to well arrival

        super().insert1(key, **kwargs)

    # Override parent class method to ensure bin width matches across raised cosine basis and time relative to
    # well arrival, and impose default params
    def _get_potential_keys(self, key_filter=None):
        if key_filter is None:
            key_filter = dict()
        # Define defaults for GLM analysis
        # Note we do not need bin_width to match time_bin_width in shorthand_params_map; bin_width is resolution
        # of covariate digitization, whereas time_bin_width corresponds to space between time samples. It just happens
        # that in this case, covariate has unit time
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_params_map
        glm_params = get_glm_default_params_map()
        default_params = {k: glm_params[k] for k in
                          ["time_rel_wa_dig_param_name", "time_rel_wa_dig_single_axis_param_name"]}
        default_params.update({
            "res_time_bins_pool_param_name": glm_params["delay_res_time_bins_pool_param_name"]})
        key_filter = add_defaults(key_filter, default_params, add_nonexistent_keys=True, require_match=False)
        keys = []
        # Loop through raised cosine basis params and proceed for those that have bin width matching
        # that in time_rel_wa_dig_param_name and domain matching that in time_rel_wa_dig_single_axis_param_name )
        for k1 in (RaisedCosineBasis & key_filter).fetch("KEY"):
            # Get domain and bin width for raised cosine param name
            bin_width_rcb, domain_rcb = (RaisedCosineBasisParams & k1).fetch1("bin_width", "domain")
            # Get domain and bin width for upstream tables param names
            bin_width = (TimeRelWADigParams & key_filter).fetch1("time_bin_width")
            domain = (TimeRelWADigSingleAxisParams & key_filter).fetch1("rel_time_start", "rel_time_end")
            if not np.logical_and(bin_width_rcb == bin_width, tuple(domain_rcb) == domain):
                continue
            for k2 in (TimeRelWADigSingleAxis & key_filter).fetch("KEY"):
                keys.append({**k1, **k2})
        return keys

    def delete_(self, key, safemode=True):
        delete_(self, [TimeRelWARCB], key, safemode)


# TODO: inspect for case with pre and post well arrival
@schema
class TimeRelWARCB(CovariateRCB):
    definition = """
    # Sampled raised cosine basis, time relative to well arrival
    -> TimeRelWARCBSel
    ---
    -> nd.common.AnalysisNwbfile
    time_rel_wa_rcb_df_object_id : varchar(40)
    """

    def make(self, key):
        # Get digitized time relative to well arrival (on single axis) as integer. Note that couldnt
        # save out as int upstream because some cases have nans, which seem to require float datatype to be
        # saved in analysis nwb file
        time_rel_wa_dig = (TimeRelWADigSingleAxis & key).fetch1_dataframe().time_rel_wa.astype(int)
        basis_functions = (RaisedCosineBasis & key).fetch1_basis_functions()
        time_rel_wa_rcb_df = sample_basis_functions(
            time_rel_wa_dig, basis_functions, tolerate_outside_basis_domain=True)
        insert_analysis_table_entry(self, [time_rel_wa_rcb_df], key, reset_index=True)

    def delete_(self, key, safemode=True):
        from src.jguides_2024.glm.jguidera_measurements_interp_pool import XInterpPoolSel
        delete_(self, [XInterpPoolSel], key, safemode)


def populate_jguidera_time_relative_to_well_event(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_time_relative_to_well_event"
    upstream_schema_populate_fn_list = [populate_jguidera_res_time_bins_pool]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_time_relative_to_well_event():
    from src.jguides_2024.firing_rate_vector.jguidera_well_event_firing_rate_vector import drop_jguidera_well_event_firing_rate_vector
    from src.jguides_2024.jguidera_firing_rate_difference_vector_similarity_ave import \
        drop_jguidera_firing_rate_difference_vector_similarity_ave
    drop_jguidera_well_event_firing_rate_vector()
    drop_jguidera_firing_rate_difference_vector_similarity_ave()
    schema.drop()
