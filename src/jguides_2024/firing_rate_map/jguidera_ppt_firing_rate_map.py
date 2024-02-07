# This module defines tables with proportion path traversed firing rate maps

import datajoint as dj
import numpy as np
import pandas as pd
import scipy as sp
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_fr_table_helpers import smooth_datajoint_table_fr, get_bin_centers_name
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import FrmapSmBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import (SecKeyParamsBase, SelBase, ComputedBase, FrmapBase)
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import (plot_datajoint_table_rate_map, get_table_object_id_name,
                                                                          insert_analysis_table_entry, get_schema_table_names_from_file, populate_insert)
from src.jguides_2024.utils.df_helpers import df_filter_columns, unpack_df_columns, get_empty_df
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellDATrials
from src.jguides_2024.metadata.jguidera_epoch import DistinctRunEpochPair, RunEpochPair
from src.jguides_2024.position_and_maze.jguidera_maze import ForkMazeRewardWellPathPair
from src.jguides_2024.position_and_maze.jguidera_ppt import PptParams, Ppt, PptBinEdges, populate_jguidera_ppt
from src.jguides_2024.spikes.jguidera_spikes import EpochSpikeTimesRelabel
from src.jguides_2024.utils.list_helpers import duplicate_elements
from src.jguides_2024.utils.make_rate_map import make_1D_rate_map_measurement_bouts
from src.jguides_2024.utils.tuple_helpers import unzip_as_list
from src.jguides_2024.utils.vector_helpers import vector_midpoints, vectors_finite_idxs, overlap

# Needed for table definitions:
PptParams
Ppt
PptBinEdges
DistinctRunEpochPair, RunEpochPair
ForkMazeRewardWellPathPair
nd

# Define custom task schema
schema = dj.schema("jguidera_ppt_firing_rate_map")


@schema
class FrmapPptSel(SelBase):
    definition = """
    # Selection from upstream tables for FrmapPpt
    -> EpochSpikeTimesRelabel
    -> Ppt
    -> PptBinEdges
    """


@schema
class FrmapPpt(FrmapBase):
    definition = """ 
    # Firing rate as a function of proportion path traversed
    -> FrmapPptSel
    ---
    -> nd.common.AnalysisNwbfile
    frmap_ppt_object_id : varchar(40)
    """

    def make(self, key):

        ppt_df = (Ppt & key).fetch1_dataframe()
        ppt_bouts = list(ppt_df["trials_ppt"])
        ppt_bouts_t = list(ppt_df["trials_time"])
        ppt_bin_edges = (PptBinEdges & key).fetch1("ppt_bin_edges")
        epoch_spike_times_df = (EpochSpikeTimesRelabel.RelabelEntries & key).fetch1_dataframe()

        # If no spike times, insert empty df
        if len(epoch_spike_times_df) == 0:
            frmap_ppt_df = get_empty_df(["unit_id", "rate_map", "ppt_bin_centers", "ppt_bin_edges", "num_trials"])

        # Otherwise get df with rate map
        else:
            unit_ids, event_rates = unzip_as_list(
                [(unit_id, make_1D_rate_map_measurement_bouts(
                    event_times, ppt_bouts, ppt_bouts_t, ppt_bin_edges)[0])
                 for unit_id, event_times in epoch_spike_times_df["epoch_spike_times"].items()])
            frmap_ppt_df = pd.DataFrame.from_dict({
                "unit_id": unit_ids, "rate_map": event_rates,
                "ppt_bin_centers": [vector_midpoints(ppt_bin_edges)] * len(unit_ids),
                "ppt_bin_edges": [ppt_bin_edges] * len(unit_ids), "num_trials": [len(ppt_bouts)] * len(unit_ids)})

        # Insert into table
        insert_analysis_table_entry(self, [frmap_ppt_df], key)  # insert into table


@schema
class FrmapPptSmParams(SecKeyParamsBase):
    definition = """
    # Parameters for smoothing proportion path traversed firing rate maps
    frmap_ppt_sm_param_name : varchar(40)
    ---
    kernel_sd : decimal(10,5) unsigned
    """

    def _default_params(self):
        return [[.05]]

    def _get_xlims(self):
        return Ppt().get_range()


@schema
class FrmapPptSm(FrmapSmBase):
    definition = """
    # Smoothed proportion path traversed firing rate
    -> FrmapPpt
    -> FrmapPptSmParams
    ---
    -> nd.common.AnalysisNwbfile
    frmap_ppt_sm_object_id : varchar(40)
    """

    @staticmethod
    def _data_type():
        return "ppt"


# Helper functions for making similarity tables
def _unit_data(fr_df, unit_id, data_name):
    # Unpack unit data

    return np.asarray(fr_df.loc[unit_id][data_name])


def _rate_maps_both_finite(fr_df_1, fr_df_2, unit_id):
    fr_1 = _unit_data(fr_df_1, unit_id, "smoothed_rate_map")  # first epoch rate map
    fr_2 = _unit_data(fr_df_2, unit_id, "smoothed_rate_map")  # second epoch rate map
    valid_bool = vectors_finite_idxs([fr_1, fr_2])  # indices where rate maps both finite
    return fr_1[valid_bool], fr_2[valid_bool]


def _calculate_corr(fr_df_1, fr_df_2, unit_id):
    fr_1_valid, fr_2_valid = _rate_maps_both_finite(fr_df_1, fr_df_2, unit_id)
    return tuple([len(fr_1_valid)] + list(sp.stats.pearsonr(fr_1_valid, fr_2_valid)))


def _calculate_overlap(fr_df_1, fr_df_2, unit_id):
    fr_1_valid, fr_2_valid = _rate_maps_both_finite(fr_df_1, fr_df_2, unit_id)
    return len(fr_1_valid), overlap(fr_1_valid, fr_2_valid)


def _get_fr_df_unit_ids_two_entries(key, fr_df, original_key_names):
    fr_df_1 = (fr_df & {**key, **{original_key_name: key[f"{original_key_name}_1"] for original_key_name in original_key_names}}).fetch1_dataframe()
    fr_df_2 = (fr_df & {**key, **{original_key_name: key[f"{original_key_name}_2"] for original_key_name in original_key_names}}).fetch1_dataframe()
    unit_ids = list(set(fr_df_2.index).intersection(fr_df_1.index))  # units present in both epochs
    for unit_id in unit_ids:  # check same bin edges across epochs
        if not all(_unit_data(fr_df_1, unit_id, "ppt_bin_edges") ==  # check same bin edges across epochs
                   _unit_data(fr_df_2, unit_id, "ppt_bin_edges")):
            raise Exception(f"Bin edges across epochs dont match")
    return fr_df_1, fr_df_2, unit_ids


def _similarity_df(unit_ids, similarity_measures, similarity_measure_name, num_samples,
                   key, original_attribute_name, p_vals=None):

    similarity_dict = {"unit_id": unit_ids, similarity_measure_name: np.asarray(similarity_measures),
                                  "number_samples": np.asarray(num_samples),
                                  f"{original_attribute_name}_1": [key[f"{original_attribute_name}_1"]] * len(unit_ids),
                                  f"{original_attribute_name}_2": [key[f"{original_attribute_name}_2"]] * len(unit_ids)}

    if p_vals is not None:
        similarity_dict.update({"p_value": np.asarray(p_vals)})

    return pd.DataFrame.from_dict(similarity_dict)


def _insert_similarity_table_entry(table, key, similarity_df, nwb_object_name):
    insert_analysis_table_entry(table, nwb_objects=[similarity_df], key=key, nwb_object_names=[nwb_object_name])


@schema
class CorrFrmapPptSm(ComputedBase):
    definition = """
    # Pearson correlation coefficient between smoothed proportion path traversed firing rate maps
    -> FrmapPptSm.proj(epoch_1='epoch', interval_list_name_epoch_1='interval_list_name', track_graph_name_epoch_1='track_graph_name')
    -> FrmapPptSm.proj(epoch_2='epoch', interval_list_name_epoch_2='interval_list_name', track_graph_name_epoch_2='track_graph_name')
    -> DistinctRunEpochPair
    ---
    -> nd.common.AnalysisNwbfile
    corr_frmap_ppt_sm_object_id : varchar(40)
    """

    def make(self, key):
        fr_df_1, fr_df_2, unit_ids = _get_fr_df_unit_ids_two_entries(key, FrmapPptSm, ["epoch"])
        num_samples, corr_coeffs, p_vals = zip(*[_calculate_corr(fr_df_1, fr_df_2, unit_id)
                                                 for unit_id in unit_ids])  # correlation coefficient, p value
        similarity_df = _similarity_df(
            unit_ids, corr_coeffs, "correlation_coefficient", num_samples, key, "epoch", p_vals)
        _insert_similarity_table_entry(self, key, similarity_df,
                                       "corr_frmap_ppt_sm_object_id")


@schema
class OverlapFrmapPptSm(ComputedBase):
    definition = """
    # Overlap between smoothed proportion path traversed firing rate maps
    -> FrmapPptSm.proj(epoch_1='epoch', interval_list_name_epoch_1='interval_list_name', track_graph_name_epoch_1='track_graph_name')
    -> FrmapPptSm.proj(epoch_2='epoch', interval_list_name_epoch_2='interval_list_name', track_graph_name_epoch_2='track_graph_name')
    -> DistinctRunEpochPair
    ---
    -> nd.common.AnalysisNwbfile
    overlap_frmap_ppt_sm_object_id : varchar(40)
    """

    def make(self, key):
        fr_df_1, fr_df_2, unit_ids = _get_fr_df_unit_ids_two_entries(key, FrmapPptSm, ["epoch"])
        num_samples, overlaps = zip(*[_calculate_overlap(fr_df_1, fr_df_2, unit_id) for unit_id in unit_ids])
        similarity_df = _similarity_df(unit_ids, overlaps, "overlap", num_samples, key, "epoch")
        _insert_similarity_table_entry(self, key, similarity_df,
                                       "overlap_frmap_ppt_sm_object_id")


@schema
class FrmapPupt(FrmapBase):
    definition = """ 
    # Firing rate as a function of percent unique path traversed
    -> FrmapPptSel
    path_name : varchar(40)
    ---
    -> nd.common.AnalysisNwbfile
    frmap_pupt_object_id : varchar(40)
    """

    def make(self, key):
        epoch_spike_times_df = (EpochSpikeTimesRelabel.RelabelEntries & key).fetch1_dataframe()
        ppt_bin_edges = (PptBinEdges & key).fetch1("ppt_bin_edges")
        ppt_df = (Ppt & key).fetch1_dataframe()
        path_names = np.unique(ppt_df["trials_path_name"])

        for path_name in path_names:
            ppt_df_subset = df_filter_columns(ppt_df, {"trials_path_name": path_name})
            ppt_bouts = list(ppt_df_subset["trials_ppt"])
            ppt_bouts_t = list(ppt_df_subset["trials_time"])
            unit_ids, event_rates = unzip_as_list(
                [(unit_id, make_1D_rate_map_measurement_bouts(event_times, ppt_bouts, ppt_bouts_t, ppt_bin_edges)[0])
                 for unit_id, event_times in epoch_spike_times_df["epoch_spike_times"].items()])
            frmap_pupt_df = pd.DataFrame.from_dict(
                {"unit_id": unit_ids, "path_name": path_name, "rate_map": event_rates,
                 "ppt_bin_centers": [vector_midpoints(ppt_bin_edges)] * len(unit_ids),
                 "ppt_bin_edges": [ppt_bin_edges] * len(unit_ids), "num_trials": [len(ppt_bouts)] * len(unit_ids)})
            # Insert into table
            key.update({"path_name": path_name})
            insert_analysis_table_entry(self, nwb_objects=[frmap_pupt_df], key=key,
                                        nwb_object_names=["frmap_pupt_object_id"])


@schema
class FrmapPuptSm(FrmapSmBase):
    definition = """
    # Smoothed proportion path traversed firing rate
    -> FrmapPupt
    -> FrmapPptSmParams
    ---
    -> nd.common.AnalysisNwbfile
    frmap_pupt_sm_object_id : varchar(40)
    """

    @staticmethod
    def _data_type():
        return "ppt"


@schema
class STFrmapPupt(ComputedBase):
    definition = """ 
    # Firing rate as a function of percent unique path traversed on single trials
    -> FrmapPptSel
    path_name : varchar(40)
    ---
    -> nd.common.AnalysisNwbfile
    st_frmap_pupt_object_id : varchar(40)
    """

    def make(self, key):

        epoch_spike_times_df = (EpochSpikeTimesRelabel.RelabelEntries & key).fetch1_dataframe()
        ppt_bin_edges = (PptBinEdges & key).fetch1("ppt_bin_edges")
        ppt_df = (Ppt & key).fetch1_dataframe()
        path_names = np.unique(ppt_df["trials_path_name"])

        for path_name in path_names:
            ppt_df_subset = df_filter_columns(ppt_df, {"trials_path_name": path_name})
            ppt_bouts = list(ppt_df_subset["trials_ppt"])
            ppt_bouts_t = list(ppt_df_subset["trials_time"])
            event_rates_list = []
            for ppt_bout, ppt_bout_t in zip(ppt_bouts, ppt_bouts_t):
                event_rates_list += [make_1D_rate_map_measurement_bouts(
                    event_times, [ppt_bout], [ppt_bout_t], ppt_bin_edges)[0]
                                     for event_times in epoch_spike_times_df["epoch_spike_times"].values]
            unit_ids = list(epoch_spike_times_df["epoch_spike_times"].index)
            trial_start_epoch_trial_numbers = duplicate_elements(ppt_df_subset.index, len(unit_ids))

            # Store in dictionary:
            fr_dict = {
                "trial_start_epoch_trial_number": trial_start_epoch_trial_numbers,
                "unit_id": unit_ids * len(ppt_bouts),
                "path_name": path_name,
                "rate_map": event_rates_list,
                "ppt_bin_centers": [vector_midpoints(ppt_bin_edges)] * len(ppt_bouts) * len(unit_ids),
                "ppt_bin_edges": [ppt_bin_edges] * len(ppt_bouts) * len(unit_ids)}

            # Add additional information about trials from DioWellDATrials (for convenience)
            # Add param name for da trials, using ppt params
            dio_well_da_trials_param_name = (PptParams & key).fetch1("dio_well_da_trials_param_name")
            trials_df = (DioWellDATrials & {**key, **{"dio_well_da_trials_param_name": dio_well_da_trials_param_name}}
                         ).fetch1_dataframe()
            copy_column_names = ["trial_start_times",
                                 "trial_end_times",
                                 "trial_end_epoch_trial_numbers",
                                 "trial_start_well_names",
                                 "trial_end_well_names",
                                 "trial_start_performance_outcomes",
                                 "trial_end_performance_outcomes",
                                 "trial_start_reward_outcomes",
                                 "trial_end_reward_outcomes"]
            info_dict = {k: v for k, v in zip(copy_column_names,
                                  unpack_df_columns(
                                      trials_df.reset_index().set_index("trial_start_epoch_trial_numbers").loc[
                                          trial_start_epoch_trial_numbers],
                                      copy_column_names))}  # ensure correct index

            # Combine fr and trials info into one dataframe
            st_frmap_pupt_df = pd.DataFrame.from_dict({**fr_dict, **info_dict})

            # Insert into table
            key.update({"path_name": path_name})
            insert_analysis_table_entry(
                self, nwb_objects=[st_frmap_pupt_df], key=key, nwb_object_names=[self.get_object_id_name()])


@schema
class STFrmapPuptSm(ComputedBase):
    definition = """
    # Smoothed proportion path traversed firing rate
    -> STFrmapPupt
    -> FrmapPptSmParams
    ---
    -> nd.common.AnalysisNwbfile
    st_frmap_pupt_sm_object_id : varchar(40)
    """

    @staticmethod
    def _data_type():
        return "ppt"

    def make(self, key):

        # Get parent table
        rate_map_df = (STFrmapPupt & key).fetch1_dataframe()

        # Smooth firing rate maps
        smoothed_firing_rate_map_df = smooth_datajoint_table_fr(
            fr_table=STFrmapPupt, params_table=FrmapPptSmParams, key=key, data_type=self._data_type())

        # Copy columns from parent table with trial info
        copy_column_names = [column_name for column_name in rate_map_df.columns if column_name
                             not in list(smoothed_firing_rate_map_df) + ["rate_map"]]
        for column_name in copy_column_names:
            smoothed_firing_rate_map_df[column_name] = rate_map_df[column_name].values

        # Store in table
        insert_analysis_table_entry(self, [smoothed_firing_rate_map_df], key, [get_table_object_id_name(self)])

    def plot_rate_map(self, key, ax=None, color="black"):
        plot_datajoint_table_rate_map(self, key, ax, color)

    def _get_xlims(self):
        return FrmapPptSmParams()._get_xlims()

    def get_bin_centers_name(self):
        return get_bin_centers_name(self._data_type())


def populate_jguidera_ppt_firing_rate_map(key=None, tolerate_error=False):
    populate_jguidera_ppt(key, tolerate_error)
    schema_name = "jguidera_ppt_firing_rate_map"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_ppt_firing_rate_map():
    schema.drop()
