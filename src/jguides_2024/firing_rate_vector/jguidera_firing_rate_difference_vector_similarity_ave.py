import copy

import datajoint as dj
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_subject_id, get_val_pairs, \
    plot_junction_fractions
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    PathWellFRVecSummBase, PopulationAnalysisParamsBase, \
    PopulationAnalysisSelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, SelBase, ParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_key_filter, make_param_name
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_nwb_file_name_epochs_description
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector import FRDiffVec, FRDiffVecParams
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity import \
    populate_jguidera_firing_rate_difference_vector_similarity, FRDiffVecCosSim
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector_euclidean_distance import FRVecEucDist, \
    populate_jguidera_firing_rate_vector_euclidean_distance
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionColor, BrainRegionCohort, CurationSet
from src.jguides_2024.metadata.jguidera_epoch import RecordingSet, EpochsDescription
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt, PptParams
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptInterp, populate_jguidera_ppt_interp
from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsCohortType
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDDTrialsParams, DioWellDDTrials
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohortParams, \
    ResTimeBinsPoolCohortParamName
from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWA
from src.jguides_2024.utils.df_helpers import df_from_data_list
from src.jguides_2024.utils.dict_helpers import add_defaults, return_shared_key_value
from src.jguides_2024.utils.hierarchical_bootstrap import hierarchical_bootstrap
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements
from src.jguides_2024.utils.make_bins import get_peri_event_bin_edges
from src.jguides_2024.utils.plot_helpers import plot_ave_conf
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.stats_helpers import average_confidence_interval
from src.jguides_2024.utils.vector_helpers import vector_midpoints, unpack_single_element

# Needed for table definitions:
FRDiffVecCosSim
PptParams
DioWellDDTrialsParams
PptInterp
DioWellDDTrials
FRDiffVec
FRVecEucDist
RecordingSet
ResTimeBinsPoolCohortParamName
BrainRegionCohort
CurationSet
ResEpochSpikesSmParams
BrainRegionUnitsCohortType
FRDiffVecParams
nd

schema = dj.schema("jguidera_firing_rate_difference_vector_similarity_ave")


"""
Notes on FRDiffVecCosSimPptNnAve and FRDiffVecCosSimWANnAve tables setup
1) the parameter 'mask_duration' is the amount of time to mask on either side of a sample. This was chosen to be 
consistent with 'mask_duration' in the single trial firing rate vector tables (PathFRVecSTAve and TimeRelWAFRVecSTAve)

2) dio_well_dd_trials_param_name is a secondary key of this table, rather than a primary key here or in params table, 
because we will require it to always have the same value (representing no trial start/end shift). So we dont want 
it taking up space unecessarily as a primary key. It is better a secondary key in the selection table than in the 
params table, because we want the secondary keys in the params table to reflect params that can change in the analysis.

3) multiple epochs (epochs_id) allowed in FRDiffVecCosSim, and only single epochs allowed in PptInterp and 
DioWellDDTrials. Unlikely that will want to do analysis across epochs, but to allow for this possibility, 
use epochs_id. Requires having dependencies on Ppt and DioWellDDTrials in a part table.
"""


class FRDiffVecCosSimCovNnAveParamsBase(ParamsBase):

    def lookup_param_name(self, params):
        """
        Return parameter name for a given set of parameters
        :param params: dictionary with parameters for an existing entry in the current table
        :return: str, parameter name
        """

        # Check that passed parameters is a dictionary
        if not isinstance(params, dict):
            raise Exception(f"params must be a dictionary")

        # Get parameter name based on params
        param_name = self._make_param_name(params)

        # Get name of parameter name in table
        meta_param_name = self.meta_param_name()

        # Return parameter name if exists in table (this step ensures parameter name exists in table)
        return (self & {meta_param_name: param_name}).fetch1(meta_param_name)

    def _make_param_name(self, params):

        nn_restrictions_text = "_".join(["_".join([k, str(v)]) for k, v in params["nn_restrictions"].items()])

        state_restrictions_text = ""
        for idx, (restriction, restriction_params) in enumerate(params["state_restrictions"].items()):
            state_restrictions_text += "_"*(idx > 0) + restriction
            restriction_params_text = "_".join([str(v) for k, v in restriction_params.items() if not np.logical_and(
                k == "table_name", v == self._default_restrictions_table_name())])
            if len(restriction_params_text) > 0:
                state_restrictions_text += restriction_params_text

        return make_param_name(
            [params[k] for k in ["n_neighbors", "bin_width"]] + [nn_restrictions_text, state_restrictions_text])

    def _default_params(self):
        # This method should be extended in child class to add bin_width
        return {"n_neighbors": 1, "mask_duration": 10}

    @staticmethod
    def _default_restrictions_table_name():
        return "DioWellDDTrials"

    def _default_nn_restrictions(self):
        return [{"mask_duration": self._default_params()["mask_duration"]}]

    def _default_state_restrictions(self):
        table_name_map = {"table_name": self._default_restrictions_table_name()}
        return [
            {"potentially_rewarded_trial": table_name_map, "stay_trial": table_name_map},
            {"correct_trial": table_name_map, "stay_trial": table_name_map}]

    def _make_params_entry(self, params):
        param_name = self._make_param_name(params)
        return {unpack_single_element(self.primary_key): param_name, "params": params}

    def insert_defaults(self, **kwargs):

        # Make params base
        params_base = {k: self._default_params()[k] for k in ["n_neighbors", "bin_width"]}

        # Add restrictions and populate table
        for nn_restrictions in self._default_nn_restrictions():
            nn_restrictions = {"nn_restrictions": nn_restrictions}
            for state_restrictions in self._default_state_restrictions():
                state_restrictions = {"state_restrictions": state_restrictions}
                params_entry = self._make_params_entry({**params_base, **nn_restrictions, **state_restrictions})
                self.insert1(params_entry, skip_duplicates=True)


class FRDiffVecCosSimVarNnAveSelBase(SelBase):

    @staticmethod
    def _get_covariate_table():
        raise Exception(f"must override in parent class")

    @staticmethod
    def _valid_dio_well_dd_trials_param_name():
        return DioWellDDTrialsParams().lookup_no_shift_param_name()

    def insert1(self, key, **kwargs):
        # Require dio_well_dd_trials_param_name to reflect no shift of trial/start/stop times
        if key["dio_well_dd_trials_param_name"] != self._valid_dio_well_dd_trials_param_name():
            raise Exception(
                f"dio_well_dd_trials_param_name must be {self._valid_dio_well_dd_trials_param_name} "
                f"but is {key['dio_well_dd_trials_param_name']}")
        super().insert1(key, **kwargs)

    def insert_defaults(self, **kwargs):

        # Populate upstream table
        key_filter = get_key_filter(kwargs)
        self._get_covariate_table()().populate_(key=key_filter)  # run this here instead of below for time (avoid loop)

        # Insert into table
        for key in self._get_potential_keys(key_filter):
            self.insert1(key)

    def _get_potential_keys(self, key_filter=None):

        # Define valid params
        # ...Restrict to 100ms epoch time bins
        res_time_bins_pool_cohort_param_name = ResTimeBinsPoolCohortParams().lookup_param_name_from_shorthand(
            "epoch_100ms")
        # ...Collect valid params
        valid_params = {
            "res_epoch_spikes_sm_param_name": ["0.1"], "zscore_fr": [0], "res_time_bins_pool_cohort_param_name":
                [res_time_bins_pool_cohort_param_name]}

        # Get keys from intersection of upstream tables
        keys = super()._get_potential_keys()

        # Restrict to valid params
        return [key for key in keys if np.logical_and.reduce(
            [key[param_name] in valid_vals for param_name, valid_vals in valid_params.items()])]


class FRDiffVecCosSimVarNnAveBase(ComputedBase):

    @staticmethod
    def _covariate_name():
        raise Exception(f"must overwrite in child class")

    @classmethod
    def _get_covariate_entry_fns(cls):
        raise Exception(f"must overwrite in child class")

    def _make_bin_edges(self, bin_width, **kwargs):
        raise Exception(f"must overwrite in child class")

    # The following methods are used in the make function. They are standalone to allow testing/verification.
    def _get_trial_intervals(self, key):
        # Return well departure to departure trials across epochs

        # Get keys to dd trials table for each epoch
        epoch_keys = [{**key, **params} for params in (ResTimeBinsPoolCohortParams & key).get_cohort_params()]

        # Update upstream entries tracker
        for epoch_key in epoch_keys:
            self._update_upstream_entries_tracker(DioWellDDTrials, epoch_key)

        # Return trial intervals
        return np.concatenate(
            [(DioWellDDTrials & epoch_key).trial_intervals() for epoch_key in epoch_keys])

    def _get_ave_nn_cos(self, key, params):
        # Return average difference vector cosine similarity across nearest neighbors
        # Note that setting populate_tables to False since cannot populate another table within make function
        table_subset = (FRDiffVecCosSim & key)
        ave_nn_cos = table_subset.get_average_nn_cosine_similarity(
            params["n_neighbors"], params["nn_restrictions"], params["state_restrictions"], populate_tables=False)
        # update upstream entries tracker
        self._merge_tracker(table_subset)
        return ave_nn_cos

    def _get_covariates(self, key):
        # Return covariates across epochs
        return [pd.concat([
                 get_covariate_entry_fn({**key, **epoch_key}) for epoch_key in (
                        ResTimeBinsPoolCohortParams & key).get_cohort_params()])
            for get_covariate_entry_fn in self._get_covariate_entry_fns()]

    def make(self, key):
        # Get average difference vector cosine similarity in covariate bins across well departure to departure trials

        # Add dio well departure to departure trials param name to key so can query trials table
        key.update({"dio_well_dd_trials_param_name": (self._get_selection_table() & key).fetch1(
            "dio_well_dd_trials_param_name")})

        # Get params
        params = (self._get_params_table() & key).fetch1("params")

        # Get average difference vector cosine similarity across nearest neighbors
        ave_nn_cos = self._get_ave_nn_cos(key, params)

        # Get well departure to departure trials across epochs
        trial_intervals = self._get_trial_intervals(key)

        dfs = []
        # Get covariates across epochs
        covariates = self._get_covariates(key)

        for covariate in covariates:

            # Limit to times from nearest neighbors analysis
            covariate = covariate.loc[ave_nn_cos.index]

            # Get average difference vector cosine similarity in covariate bins across well departure to departure
            # trials. Tolerate nan so can estimate averages in bins with representation from only a subset of trials
            # ...Define covariate bins
            bin_edges = self._make_bin_edges(params["bin_width"], covariate=covariate)
            # ...Get average cosine similarity in covariate bins and append to dfs
            dfs.append(average_value_in_trials_in_bins(
                covariate, ave_nn_cos, bin_edges, trial_intervals, "cos_sim", tolerate_nan=True))

        # Insert into main table
        insert_key = {k: key[k] for k in self.primary_key}
        insert_analysis_table_entry(self, dfs, insert_key)

        # Insert into part tables
        self._insert_part_tables(insert_key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="bin_center"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def _get_plot_color(self):
        # Get color based on brain region
        return (BrainRegionColor & self.fetch1("KEY")).fetch1("color")

    def _plot_results(self, ax, x_vals, mean, conf_bounds, **plot_params):
        # Get color based on brain region
        plot_params.update({"color": self._get_plot_color()})
        # Add default plot params
        default_params = {"alpha": 1,
            "title": "{nwb_file_name}, eps {epochs_id}, {brain_region}".format(**self.fetch1("KEY")),
            "xlabel": f"{self._covariate_name()}", "ylabel": "cos sim", "fontsize": 12}
        plot_params = add_defaults(plot_params, default_params, add_nonexistent_keys=True)
        plot_ave_conf(ax, x_vals, mean, conf_bounds, **plot_params)

    def plot_results(self, ax=None, object_id_name=None, **plot_params):
        # Get name of results df if not passed
        if object_id_name is None:
            object_id_name = self.get_object_id_name(unpack_single_object_id=True, leave_out_object_id=True)
        df = self.fetch1_dataframe(object_id_name=object_id_name)

        # Plot mean and confidence bounds for mean
        self._plot_results(ax, df.index, df.mean_cos_sim.values, df.mean_cos_sim_conf.values, **plot_params)


@schema
class FRDiffVecCosSimPptNnAveParams(FRDiffVecCosSimCovNnAveParamsBase):
    definition = """
    # Parameters for FRDiffVecCosSimPptNnAve
    fr_diff_vec_cos_sim_ppt_nn_ave_param_name : varchar(400)
    ---
    params : blob
    """

    def _default_params(self):
        default_params = super()._default_params()
        default_params["bin_width"] = .05
        return default_params


@schema
class FRDiffVecCosSimPptNnAveSel(FRDiffVecCosSimVarNnAveSelBase):
    definition = """
    # Selection from upstream tables for FRDiffVecCosSimPptNnAve
    -> FRDiffVecCosSim
    -> FRVecEucDist
    -> PptParams
    -> FRDiffVecCosSimPptNnAveParams
    ---
    -> DioWellDDTrialsParams
    """

    @staticmethod
    def _get_covariate_table():
        return PptInterp


@schema
class FRDiffVecCosSimPptNnAve(FRDiffVecCosSimVarNnAveBase):
    definition = """
    # Cosine similarity of firing rate difference vectors to nearest neighbors, averaged in proportion path traversed bins
    -> FRDiffVecCosSimPptNnAveSel
    ---
    -> nd.common.AnalysisNwbfile
    fr_diff_vec_cos_sim_ppt_nn_ave_object_id : varchar(40)
    """

    class PptInterp(dj.Part):
        definition = """
            # Achieves dependence on PptInterp
            -> FRDiffVecCosSimPptNnAve
            -> PptInterp
            """

    class DioWellDDTrials(dj.Part):
        definition = """
            # Achieves dependence on DioWellDDTrials
            -> FRDiffVecCosSimPptNnAve
            -> DioWellDDTrials
            """

    @staticmethod
    def _covariate_name():
        return "ppt"

    def _get_covariate_entry_fns(self):
        return [self._get_ppt_interp]

    def _get_ppt_interp(self, key):
        self._update_upstream_entries_tracker(PptInterp, key)
        return (PptInterp & key).fetch1_ppt()

    @staticmethod
    def _make_bin_edges(bin_width, **kwargs):
        # Define ppt bins
        return Ppt.get_ppt_bin_edges(bin_width)


"""
Notes on FRDiffVecCosSimWANnAve tables setup:
- multiple epochs (epochs_id) allowed in FRDiffVecCosSim, and only single epochs allowed in TimeRelWA. Unlikely that will want
    to do analysis across epochs, but to allow for this possibility, use epochs_id. Requires having dependencies on TimeRelWA
    in a part table.
"""


@schema
class FRDiffVecCosSimWANnAveParams(FRDiffVecCosSimCovNnAveParamsBase):
    definition = """
    # Parameters for FRDiffVecCosSimWANnAve
    fr_diff_vec_cos_sim_wa_nn_ave_param_name : varchar(400)
    ---
    params : blob
    """

    def _default_params(self):
        default_params = super()._default_params()
        default_params["bin_width"] = .25
        return default_params


@schema
class FRDiffVecCosSimWANnAveSel(FRDiffVecCosSimVarNnAveSelBase):
    definition = """
    # Selection from upstream tables for FRDiffVecCosSimWANnAve
    -> FRDiffVecCosSim
    -> FRVecEucDist
    -> FRDiffVecCosSimWANnAveParams
    ---
    -> DioWellDDTrialsParams
    """

    @staticmethod
    def _get_covariate_table():
        return TimeRelWA


@schema
class FRDiffVecCosSimWANnAve(FRDiffVecCosSimVarNnAveBase):
    definition = """
    # Cosine similarity of firing rate difference vectors to nearest neighbors, averaged in proportion path traversed bins
    -> FRDiffVecCosSimWANnAveSel
    ---
    -> nd.common.AnalysisNwbfile
    fr_diff_vec_cos_sim_to_wa_nn_ave_object_id : varchar(40)
    fr_diff_vec_cos_sim_from_wa_nn_ave_object_id : varchar(40)
    """

    class TimeRelWA(dj.Part):
        definition = """
            # Achieves dependence on TimeRelWA
            -> FRDiffVecCosSimWANnAve
            -> TimeRelWA
            """

    class DioWellDDTrials(dj.Part):
        definition = """
            # Achieves dependence on DioWellDDTrials
            -> FRDiffVecCosSimWANnAve
            -> DioWellDDTrials
            """

    @staticmethod
    def _covariate_name():
        return "time"

    def _get_covariate_entry_fns(self):
        return [self._get_time_to_wa, self._get_time_from_wa]

    def _get_time_to_wa(self, key):
        self._update_upstream_entries_tracker(TimeRelWA, key)
        return (TimeRelWA & key).fetch1_dataframe().time_to_wa

    def _get_time_from_wa(self, key):
        self._update_upstream_entries_tracker(TimeRelWA, key)
        return (TimeRelWA & key).fetch1_dataframe().time_from_wa

    @staticmethod
    def _make_bin_edges(bin_width, **kwargs):
        # Define time relative to well arrival bins
        covariate = kwargs.pop("covariate")
        return get_peri_event_bin_edges([0, np.nanmax(covariate)], bin_width=bin_width)

    def get_pre_post_quantites_single_axis(self, column_name, object_id_name=None):
        # Get quantities across pre and post well arrival periods on a single axis

        # Function to reverse index order for reverse (negative) direction
        def _order_index(x, direction):
            if direction == -1:
                return x[::-1]
            return x

        # Get map from result type (before or after well arrival) to direction on plot: reverse direction for
        # "to well arrival" result
        direction_map = {f"fr_diff_vec_cos_sim_{rel_text}_wa_nn_ave": direction for rel_text, direction in
                         zip(["to", "from"], [-1, 1])}
        # restrict to passed object_id_name if passed
        if object_id_name is not None:
            direction_map = {k: direction_map[k] for k in [object_id_name]}

        # Check that passed column name valid
        valid_column_names = ["index", "mean_cos_sim", "mean_cos_sim_conf"]
        check_membership([column_name], valid_column_names)

        # If index (x values), multiply by direction so can have "to well arrival" extending backwards
        # in time relative to well arrival
        def _get_mult_factor(column_name, direction):
            if column_name == "index":
                return direction
            return 1

        return np.concatenate([
            _order_index(getattr(
                self.fetch1_dataframe(object_id_name), column_name).values, direction) *
            _get_mult_factor(column_name, direction)
            for object_id_name, direction in direction_map.items()])

    # Override parent class method so can plot results from pre and post well arrival together
    def plot_results(self, ax=None, object_id_name=None, **plot_params):
        # Plot mean and confidence bounds for mean
        x_vals = self.get_pre_post_quantites_single_axis("index", object_id_name)
        plot_y = self.get_pre_post_quantites_single_axis("mean_cos_sim", object_id_name)
        conf_bounds = self.get_pre_post_quantites_single_axis("mean_cos_sim_conf", object_id_name)
        self._plot_results(ax, x_vals, plot_y, conf_bounds, **plot_params)


def average_value_in_trials_in_bins(x1, x2, x1_bin_edges, trial_intervals, x2_name="x2", tolerate_nan=True):
    """
    Get average value of a variable x2 (e.g. cosine similarity) in bins of a different variable x1 (e.g. fraction
    path traversed).
    :param x1: pandas series, values of first variable indexed in time
    :param x2: numpy array, values of second variable which correspond element-wise to values of first variable
    :param x1_bin_edges: edges of bins of first variable
    :param trial_intervals: list with start/stop times of "trials". Find average value in bins within each trial
    separately, and treat these as independent samples: find average and confidence intervals across trials
    :param x2_name: str, name of second variable
    :param tolerate_nan: bool, True to tolerate nans in x2 when finding average
    :return: df with average value and confidence intervals of x2 in trials in x1 bins
    """

    # Check inputs
    if not isinstance(x1, pd.Series):
        raise Exception(f"indexed_data must be a pandas Series but is {type(x1)}")
    if len(x1) != len(x2):
        raise Exception(f"x1 and x2 must be same length, but have lengths {len(x1)} and {len(x2)}")

    # Get functions to average and find confidence intervals depending on whether tolerating nan
    average_function = np.mean  # default
    if tolerate_nan:
        average_function = np.nanmean

    # Get bin centers and bin edges as tuples corresponding to bin centers, so can store along with results
    x1_bin_centers = vector_midpoints(x1_bin_edges)
    x1_bin_edge_tuples = zip_adjacent_elements(x1_bin_edges)

    # Get bin indices
    bin_nums = np.arange(1, len(x1_bin_edges))

    # Identify x1 samples in each bin
    x1_digitzed = pd.Series(np.digitize(x1.values, x1_bin_edges), index=x1.index)

    # Loop through x1 bins
    summary_list = []
    for bin_num in bin_nums:

        valid_bool = x1_digitzed.values == bin_num
        valid_times = x1_digitzed.index[valid_bool]

        # Loop through trials and find average of data values in this trial, in this data bin
        bin_summary_list = []
        for trial_interval in trial_intervals:

            trial_bool = event_times_in_intervals_bool(valid_times, [trial_interval])

            # Get number of samples in x1 bin for this trial
            trial_len = np.sum(trial_bool)

            # Get average x2 values for this trial in this x1 bin
            trial_x2 = average_function(x2[valid_bool][trial_bool])
            bin_summary_list.append((trial_len, trial_x2))

        trial_lens, x2_trials = zip(*bin_summary_list)

        # Get mean x2 across average values in trials, and confidence bounds for this average
        x2_trial_ave = average_function(x2_trials)
        x2_trial_ave_conf = average_confidence_interval(x2_trials, exclude_nan=tolerate_nan)

        # Store
        summary_list.append(
            (bin_num, x1_bin_centers[bin_num - 1], x1_bin_edge_tuples[bin_num - 1], x2_trial_ave, x2_trial_ave_conf,
             x2_trials, len(trial_lens), trial_lens))  # -1 idx because bin number of 1 corresponds to first bin center

    return df_from_data_list(
        summary_list, ["bin_num", "bin_center", "bin_edges", f"mean_{x2_name}", f"mean_{x2_name}_conf",
                       f"trials_{x2_name}", "num_trials", "trial_lens"])


"""
# Code for testing average_value_in_trials_in_bins:

# Make ppt
ppt = list(np.arange(0.05, 1, .2))*2
x1 = pd.Series(ppt, index=np.arange(0, len(ppt)))

# Make ppt bin edges
x1_bin_edges = np.arange(0, 1.1, .5)

# Make cosine sim
c1 = [.1]*3 + [.7]*2
c2 = [x + .1 for x in c1]
x2 = np.asarray(c1 + c2)

# Make trial intervals
trial_intervals = [[-.5, 4.5], [4.5, 10.5]]

# Get average value in trials in bins
df = average_value_in_trials_in_bins(x1, x2, x1_bin_edges, trial_intervals)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 3))
# Plot x1
ax.plot(x1, "o", label="x1", color="gray")
# Plot x2
ax.plot(x1.index, x2, "x", color="red", label="x2")
# Plot trial intervals
y_val_list = [ax.get_ylim()[1]]*len(trial_intervals)
plot_intervals(trial_intervals, ax, y_val_list, label="trial intervals")

# Plot results
# Plot bin edges
intervals = df.bin_edges
x_val = np.min(trial_intervals)
val_list = [x_val]*len(intervals)
plot_intervals(intervals, ax, val_list, label="bin edges", interval_axis="y", color="gray")
    
for idx, df_row in df.iterrows():
    label = None
    if idx == 0:
        label = "conf ave x2"
    ax.plot([x_val]*2, df_row.mean_x2_conf, color="red", label=label)
    
    label = None
    if idx == 0:
        label = "ave x2"
    ax.plot(x_val, df_row.mean_x2, "o", color="red", label=label)

# Format axis
ax.set_xlabel("time")
# Legend
ax.legend()
"""


"""
Notes on FRDiffVecCosSimWANnAveSumm table setup:
- We want to combine entries across FRDiffVecCosSimWANnAveSumm, across nwb_file_names, epochs_description,
brain_region, and brain_region_units_param_names (in case want to do hierarchical bootstrap across unit subsets). 
For this reason, we want FRDiffVecCosSimWANnAveSummSel to have all primary keys of FRDiffVecCosSimWANnAve
except for nwb_file_name, epochs_id, brain_region, brain_region_units_param_name, and 
curation_name. 
  To specify the nwb_file_names and corresponding epochs_id we want to combine across, we use recording_set.
  To specify the brain regions we want to combine across, we use brain_region_cohort_name. 
  To specify curation_name, we use curation_set_name.
  To specify brain region unit information, we use brain_region_units_params_cohort_name in 
  FRDiffVecCosSimWANnAveSummParams
- We include BrainRegionUnitsCohortType in PathAveFRVecSummParams so that we can stay within the
limit on number of primary keys
"""


class FRDiffVecCosSimCovNnAveSummParamsBase(PopulationAnalysisParamsBase):

    def get_default_param_name(self):
        return self.lookup_param_name(self._default_params()[0])

    def _boot_set_names(self):
        # Boot set names to populate table with. Use different set depending on child class.
        return self._valid_default_boot_set_names() + self._valid_brain_region_diff_boot_set_names()

    def _default_params(self):

        params = []
        for boot_set_name in self._boot_set_names():
            for brain_region_units_cohort_type in self._default_brain_region_units_cohort_types():
                params.append([boot_set_name, brain_region_units_cohort_type])

        # Return parameters
        return params


class FRDiffVecCosSimCovNnAveSummSelBase(PopulationAnalysisSelBase):

    def _get_param_name_map(self, key_filter, brain_region_units_cohort_types):
        # Define summary table param name based on recording set name.
        # To do this, define map from recording set name and brain_region_units_cohort_type to summary table param names

        params_table = self._get_params_table()()

        recording_set_names_boot_set_names = [

             # Rat cohort
             (RecordingSet().lookup_rat_cohort_set_name(), boot_set_name)
             for boot_set_name in self._default_cohort_boot_set_names()] + [

             # Non rat cohort
             (recording_set_name, boot_set_name) for recording_set_name in
             RecordingSet().get_recording_set_names(
                 key_filter, ["Haight_rotation", "first_day_learning_single_epoch"])
             for boot_set_name in self._default_noncohort_boot_set_names()]

        param_name_map = dict()
        for recording_set_name, boot_set_name in recording_set_names_boot_set_names:
            for brain_region_units_cohort_type in brain_region_units_cohort_types:
                param_name_map_key = self._format_param_name_map_key(recording_set_name, brain_region_units_cohort_type)
                if param_name_map_key not in param_name_map:
                    param_name_map[param_name_map_key] = []
                param_names = (params_table & {
                    "boot_set_name": boot_set_name,
                    "brain_region_units_cohort_type": brain_region_units_cohort_type}).fetch(
                    params_table.meta_param_name())
                param_name_map[param_name_map_key] += list(param_names)

        return param_name_map

    @staticmethod
    def _format_param_name_map_key(recording_set_name, brain_region_units_cohort_type):
        return (recording_set_name, brain_region_units_cohort_type)

    def _get_param_name_map_key(self, key, brain_region_units_cohort_type):
        # Make key to param name map given a set of parameters
        return self._format_param_name_map_key(key["recording_set_name"], brain_region_units_cohort_type)

    def _default_noncohort_boot_set_names(self):
        return super()._default_noncohort_boot_set_names() + ["brain_region_diff"]

    def _default_cohort_boot_set_names(self):
        return super()._default_cohort_boot_set_names() + ["brain_region_diff_rat_cohort"]


class FRDiffVecCosSimCovNnAveSummBase(PathWellFRVecSummBase):

    def _get_upstream_data(self, upstream_key):
        raise Exception(f"This method must be overwritten in child class")

    def _additional_restrict_metric_df(self, metric_df, key):
        raise Exception(f"This method must be overwritten in child class")

    def _get_vals_index_name(self, **kwargs):
        return "x_val"

    def make(self, key):

        # Get data across conditions into one place so can do bootstrap resample
        data_list = []
        for upstream_key in (self._get_selection_table() & key).fetch1("upstream_keys"):
            # Get data for this table entry
            x_vals, mean_cos_sim_vals = self._get_upstream_data(upstream_key)

            # Store
            nwb_file_name = upstream_key["nwb_file_name"]
            subject_id = get_subject_id(nwb_file_name)
            for x_val, val in zip(x_vals, mean_cos_sim_vals):
                epochs_description = (EpochsDescription & upstream_key).fetch1("epochs_description")
                data_list.append(
                    (upstream_key["brain_region"], subject_id, nwb_file_name, epochs_description,
                     x_val, val, upstream_key["brain_region_units_param_name"],
                     get_nwb_file_name_epochs_description(nwb_file_name, epochs_description)))

        metric_df = df_from_data_list(
            data_list, [
                "brain_region", "subject_id", "nwb_file_name", "epochs_description", self._get_vals_index_name(key=key),
                "val", "brain_region_units_param_name", "nwb_file_name_epochs_description"])

        # Restrict to non-nan samples
        valid_bool = np.invert(np.isnan(metric_df.val))
        metric_df = metric_df[valid_bool]
        # Apply additional restrictions as indicated
        metric_df = self._additional_restrict_metric_df(metric_df, key)

        # Hierarchical bootstrap and sample mean

        # ...Define bootstrap params as indicated
        params_table_subset = (self._get_params_table()() & key)
        bootstrap_params = params_table_subset.get_boot_params()
        boot_set_name = params_table_subset.fetch1("boot_set_name")

        # average values
        if boot_set_name in ["default", "default_rat_cohort"]:
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = ["nwb_file_name_epochs_description", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = ["x_val", "brain_region"]
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        # average difference values across brain regions
        elif boot_set_name in ["brain_region_diff", "brain_region_diff_rat_cohort"]:

            # First redefine metric_df to reflect difference between val for different brain regions
            target_column_name = "brain_region"
            # ...Define pairs of brain regions
            target_column_pairs = get_val_pairs(
                np.unique(metric_df[target_column_name]), self._get_brain_region_order_for_pairs())
            # ...Define function for computing metric on brain region pairs
            metric_pair_fn = self.metric_pair_diff
            # ...Get df with paired metric
            metric_df = self.get_paired_metric_df(metric_df, target_column_name, target_column_pairs, metric_pair_fn)
            # Define parameters for bootstrap
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = ["nwb_file_name_epochs_description", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = [
                "x_val", self._get_joint_column_name(target_column_name), "brain_region_1",
                "brain_region_2"]
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        # Raise exception if boot set name not accounted for in code
        else:
            raise Exception(f"Have not written code for boot_set_name {boot_set_name}")

        # Perform bootstrap
        boot_results = hierarchical_bootstrap(
            metric_df, resample_levels, "val", ave_group_column_names,
            num_bootstrap_samples_=bootstrap_params.num_bootstrap_samples, average_fn_=bootstrap_params.average_fn,
            alphas=bootstrap_params.alphas)

        # Store dfs with results together to save out below
        # ...df with metric values
        results_dict = {"metric_df": metric_df}
        # ...bootstrap results. Convert results that are None to dfs
        for x in ["ave_conf_df", "boot_ave_df"]:
            results_dict[x] = pd.DataFrame()
            if getattr(boot_results, x) is not None:
                results_dict[x] = getattr(boot_results, x)

        # Insert into main table
        insert_analysis_table_entry(self, list(results_dict.values()), key)

        # Insert into parts tables
        for upstream_key in (self._get_selection_table() & key).fetch1("upstream_keys"):
            self.Upstream.insert1({**key, **upstream_key})

    def _get_val_text(self):
        return "Nearest neighbor\ncosine similarity"

    def _get_val_lims(self, **kwargs):
        # Get a set range for value, e.g. for use in plotting value on same range across plots
        boot_set_name = self.get_upstream_param("boot_set_name")
        if boot_set_name in self._get_params_table()._valid_brain_region_diff_boot_set_names():
            return [-.5, .3]
        return [0, .8]

    # Override parent class method so can add params specific to nn fr vec tables
    def get_default_table_entry_params(self):
        params = super().get_default_table_entry_params()

        # Firing rate difference vector param name
        fr_diff_vec_param_name = "1"

        params.update(
            {"fr_diff_vec_param_name": fr_diff_vec_param_name})

        # Return default params
        return params

    @staticmethod
    def _get_multiplot_params(**kwargs):
        # Define params for plotting multiple table entries. One param set per table entry.

        # Check that loop inputs passed
        check_membership(["table_names", "recording_set_names"], kwargs)

        # Make copy of kwargs to serve as base of each key
        kwargs = copy.deepcopy(kwargs)

        # Remove iterables so that not in individual keys
        table_names = kwargs.pop("table_names")
        recording_set_names = kwargs.pop("recording_set_names")

        param_sets = []
        # Loop through table names
        for table_name in table_names:

            # Make copy of kwargs so that updates dont carry over from one for loop iteration to the next
            key = copy.deepcopy(kwargs)

            # Add table_name to key
            key.update({"table_name": table_name})

            # Add to key param name for summary table
            # ...Get table
            table = get_table(table_name)()
            # ...Add default params
            default_params = table.get_default_table_entry_params()
            key = add_defaults(key, default_params, add_nonexistent_keys=True)
            # ...Add summary table param name
            params_table = table._get_params_table()()
            param_name = params_table.lookup_param_name(key, args_as_dict=True, tolerate_irrelevant_args=True)
            key.update({params_table.meta_param_name(): param_name})

            # Loop through recording set names
            for recording_set_name in recording_set_names:

                # Add recording set name to key
                key.update({"recording_set_name": recording_set_name})

                # Append param set to list
                param_sets.append(copy.deepcopy(key))

        # Return param sets
        return param_sets

    def _get_multiplot_fig_name(self, brain_region_vals, keys, plot_name=""):

        fig_name = super()._get_multiplot_fig_name(brain_region_vals, keys, plot_name)

        # Add text to denote firing rate difference vector param name
        fr_diff_vec_param_name = return_shared_key_value(keys, "fr_diff_vec_param_name")
        fig_name += f"_diff{fr_diff_vec_param_name}"

        return fig_name


@schema
class FRDiffVecCosSimWANnAveSummParams(FRDiffVecCosSimCovNnAveSummParamsBase):
    definition = """
    # Parameters for FRDiffVecCosSimWANnAveSumm
    fr_diff_vec_cos_sim_wa_nn_ave_summ_param_name : varchar(200)
    ---
    x_range : blob  # restrict x range for tractability in bootstrap
    boot_set_name : varchar(40)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """

    @staticmethod
    def _default_x_range():
        return [-3, 4]

    # Override parent class method so can add x range params
    def _default_params(self):
        params = super()._default_params()
        return [[self._default_x_range()] + x for x in params]


@schema
class FRDiffVecCosSimWANnAveSummSel(FRDiffVecCosSimCovNnAveSummSelBase):
    definition = """
    # Selection from upstream tables for FRDiffVecCosSimWANnAveSumm
    -> RecordingSet
    -> ResTimeBinsPoolCohortParamName
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> FRDiffVecParams
    zscore_fr : bool
    -> FRDiffVecCosSimWANnAveParams
    -> FRDiffVecCosSimWANnAveSummParams
    ---
    upstream_keys : blob
    """

    @staticmethod
    def _upstream_table():
        return FRDiffVecCosSimWANnAve


@schema
class FRDiffVecCosSimWANnAveSumm(FRDiffVecCosSimCovNnAveSummBase):
    definition = """
    # Summary of cosine similarity of firing rate difference vectors to nearest neighbors, averaged in proportion path traversed bins
    -> FRDiffVecCosSimWANnAveSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
            # Achieves upstream dependence on upstream tables
            -> FRDiffVecCosSimWANnAveSumm
            -> BrainRegionCohort
            -> CurationSet
            -> FRDiffVecCosSimWANnAve
            """

    def _upstream_table(self):
        return FRDiffVecCosSimWANnAve

    def _get_upstream_data(self, upstream_key):
        table_subset = self._upstream_table() & upstream_key
        x_vals = table_subset.get_pre_post_quantites_single_axis("index")
        mean_cos_sim_vals = table_subset.get_pre_post_quantites_single_axis("mean_cos_sim")
        return x_vals, mean_cos_sim_vals

    def _additional_restrict_metric_df(self, metric_df, key):
        # Restrict x vals for tractability in bootstrap
        params_table_subset = (self._get_params_table()() & key)
        x_range = params_table_subset.fetch1("x_range")
        valid_bool = np.logical_and(metric_df["x_val"] > x_range[0], metric_df["x_val"] < x_range[1])
        # Apply restrictions
        return metric_df[valid_bool]

    def _get_x_text(self):
        return "Time in delay (s)"

    def _get_x_lims(self):
        return [0, 2]

    def _get_xticks(self):
        return [.5, 1, 1.5]

    # Override parent class method so can add params specific to this table
    def get_default_table_entry_params(self):

        params = super().get_default_table_entry_params()

        params.update({"x_range": self._get_params_table()._default_x_range()})

        # Return default params
        return params


@schema
class FRDiffVecCosSimPptNnAveSummParams(FRDiffVecCosSimCovNnAveSummParamsBase):
    definition = """
    # Parameters for FRDiffVecCosSimPptNnAveSumm
    fr_diff_vec_cos_sim_ppt_nn_ave_summ_param_name : varchar(200)
    ---
    boot_set_name : varchar(40)  # describes bootstrap parameters
    -> BrainRegionUnitsCohortType
    """


@schema
class FRDiffVecCosSimPptNnAveSummSel(FRDiffVecCosSimCovNnAveSummSelBase):
    definition = """
    # Selection from upstream tables for FRDiffVecCosSimPptNnAveSumm
    -> RecordingSet
    -> ResTimeBinsPoolCohortParamName
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> ResEpochSpikesSmParams
    -> FRDiffVecParams
    zscore_fr : bool
    -> FRDiffVecCosSimPptNnAveParams
    -> FRDiffVecCosSimPptNnAveSummParams
    ---
    upstream_keys : blob
    """

    @staticmethod
    def _upstream_table():
        return FRDiffVecCosSimPptNnAve


@schema
class FRDiffVecCosSimPptNnAveSumm(FRDiffVecCosSimCovNnAveSummBase):
    definition = """
    # Summary of cosine similarity of firing rate difference vectors to nearest neighbors, averaged in proportion path traversed bins
    -> FRDiffVecCosSimPptNnAveSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    class Upstream(dj.Part):
        definition = """
        # Achieves upstream dependence on upstream tables
        -> FRDiffVecCosSimPptNnAveSumm
        -> BrainRegionCohort
        -> CurationSet
        -> FRDiffVecCosSimPptNnAve
        """

    def _upstream_table(self):
        return FRDiffVecCosSimPptNnAve

    def _get_upstream_data(self, upstream_key):
        df = (self._upstream_table() & upstream_key).fetch1_dataframe()
        x_vals = df.index
        mean_cos_sim_vals = df.mean_cos_sim
        return x_vals, mean_cos_sim_vals

    def _additional_restrict_metric_df(self, metric_df, key):
        return metric_df

    def _get_x_text(self):
        return "Path fraction"

    def _get_x_lims(self):
        return [0, 1]

    def _get_xticks(self):
        return [0, .5, 1]

    def extend_plot_results(self, **kwargs):

        # Vertical lines to denote track segments
        if not kwargs["empty_plot"]:
            ax = kwargs["ax"]
            plot_junction_fractions(ax)


def populate_jguidera_firing_rate_difference_vector_similarity_ave(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_firing_rate_difference_vector_similarity_ave"
    upstream_schema_populate_fn_list = [
        populate_jguidera_firing_rate_difference_vector_similarity, populate_jguidera_ppt_interp,
        populate_jguidera_firing_rate_vector_euclidean_distance]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_firing_rate_difference_vector_similarity_ave():
    schema.drop()
