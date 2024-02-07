import datajoint as dj
import numpy as np

from src.jguides_2024.datajoint_nwb_utils.datajoint_fr_table_helpers import make_well_single_trial_table_fr_df, \
    insert_single_trial_firing_rate_map_smoothed_well_table
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, ComputedBase, SelBase, FrmapBase, \
    TemporalFrmapParamsBase, TemporalFrmapSmParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import (plot_datajoint_table_rate_map,
                                                                          insert_analysis_table_entry, get_schema_table_names_from_file, populate_insert)
from src.jguides_2024.utils.df_helpers import unique_df_column_sets, df_filter_columns, zip_df_columns, df_from_data_list
from src.jguides_2024.task_event.jguidera_dio_trials import (DioWellADTrials,
                                                             DioWellADTrialsParams)
from src.jguides_2024.firing_rate_map.jguidera_ppt_firing_rate_map import populate_jguidera_ppt_firing_rate_map
from src.jguides_2024.time_and_trials.jguidera_warped_axis_bins import WarpedAxisBins, WarpedAxisBinsParams
from src.jguides_2024.utils.vector_helpers import rescale_1d, vector_midpoints, check_all_unique

# Imports used in table definition below (do not remove):

schema = dj.schema("jguidera_well_arrival_departure_firing_rate_map")


@schema
class FrmapWADParams(TemporalFrmapParamsBase):
    definition = """
    # Parameters for well arrival to departure firing rate maps
    frmap_wad_param_name : varchar(50)
    ---
    time_bin_width : decimal(10,5) unsigned
    """


@schema
class STFrmapWAD(ComputedBase):
    definition = """ 
    # Firing rate as a function of relative time within trials relative to well arrival/departure
    -> DioWellADTrials
    -> EpochSpikeTimesRelabel
    -> FrmapWADParams
    ---
    -> nd.common.AnalysisNwbfile
    st_frmap_wad_object_id : varchar(40)
    """

    def make(self, key):
        # Get rate maps
        single_trial_firing_rate_map_df = make_well_single_trial_table_fr_df(
            key, trials_table=DioWellADTrials, trials_params_table=DioWellADTrialsParams,
            firing_rate_map_params_table=FrmapWADParams)

        # Insert into table
        insert_analysis_table_entry(self, [single_trial_firing_rate_map_df], key, [self.get_object_id_name()])


@schema
class FrmapWADSmParams(TemporalFrmapSmParamsBase):
    definition = """
    # Parameters for smoothing well arrival to departure firing rate maps
    frmap_wad_sm_param_name : varchar(40)
    ---
    kernel_sd : decimal(10,5) unsigned
    """


@schema
class STFrmapWADSm(ComputedBase):
    definition = """
    # Smoothed well arrival to departure single trial firing rate
    -> STFrmapWAD
    -> FrmapWADSmParams
    ---
    -> nd.common.AnalysisNwbfile
    st_frmap_wad_sm_object_id : varchar(40)
    """

    def make(self, key):
        insert_single_trial_firing_rate_map_smoothed_well_table(
            fr_smoothed_table=self, fr_table=STFrmapWAD, params_table=FrmapWADSmParams, key=key,
            data_type="time")

    def plot_rate_map(self, key, ax=None, color="black"):
        # TODO: update this function for single trial FR maps; seems it only plots one trial currently
        plot_datajoint_table_rate_map(self, key, ax, color)


@schema
class STFrmapWADSmWTSel(SelBase):
    definition = """
    # Selection from upstream tables for STFrmapWADSmWT
    -> STFrmapWADSm
    -> WarpedAxisBins
    """

    # Override parent method so can populate upstream tables with params
    def insert_defaults(self, **kwargs):

        for key in STFrmapWADSm().fetch("KEY"):
            for params in [(0, 1, .05)]:
                WarpedAxisBinsParams().insert_entry(params)
                WarpedAxisBins.populate()
                warped_axis_bins_param_name = WarpedAxisBinsParams().lookup_param_name(params)
                key.update({"warped_axis_bins_param_name": warped_axis_bins_param_name})
                self.insert1(key, skip_duplicates=True)


@schema
class STFrmapWADSmWT(ComputedBase):
    definition = """
    # Smoothed well arrival to departure single trial firing rate on warped time axis
    -> STFrmapWADSmWTSel
    ---
    -> nd.common.AnalysisNwbfile
    st_frmap_wad_sm_wt_object_id : varchar(40)
    """

    def make(self, key):
        # Get firing rate map and warped axis bin edges
        fr_df = (STFrmapWADSm & key).fetch1_dataframe()
        bin_centers, bin_edges = (WarpedAxisBins & key).fetch1("bin_centers", "bin_edges")

        # 1) Rescale time bin edges to specified arbitrary axis
        rescaled_time_bin_edges_list = [rescale_1d(bin_edges, x) for x in fr_df["time_bin_edges"]]
        rescaled_time_bin_centers_list = list(map(vector_midpoints, rescaled_time_bin_edges_list))

        # 2) Sample rescaled rate map in warped axis bins
        rate_map_interp = [np.interp(bin_centers, rescaled_time_bin_centers, rate_map)
                           for rescaled_time_bin_centers, rate_map in zip(rescaled_time_bin_centers_list,
                                                                          fr_df["smoothed_rate_map"])]

        # Add new bin edges and centers and new rate map to dataframe
        fr_df["bin_edges"] = [bin_edges] * len(fr_df)
        fr_df["bin_centers"] = [bin_centers] * len(fr_df)
        fr_df["smoothed_rate_map"] = rate_map_interp

        # Drop time bin column names from dataframe
        fr_df.drop(columns=["time_bin_edges", "time_bin_centers"], inplace=True)

        # Store in table
        insert_analysis_table_entry(self, [fr_df], key, [self.get_object_id_name()])

    def plot_rate_map(self, key, ax=None, color="black"):
        # TODO: update this function for single trial FR maps; seems it only plots one trial currently
        plot_datajoint_table_rate_map(self, key, ax, color)


@schema
class FrmapWADSmWTParams(SecKeyParamsBase):
    definition = """
    # Parameters for FrmapWADSmWT
    frmap_wad_sm_wt_param_name : varchar(40)
    ---
    trial_feature_names : blob
    """

    def _default_params(self):
        return [[["unit_id", "well_name"]], ]

    # Overrides method in parent class
    def _make_param_name(self, secondary_key_subset_map):
        # Return unique trial features names in order to avoid inserting multiple combinations with same effect
        def order_trial_feature_names(k, v):
            if k == "trial_feature_names":
                return np.unique(v)

        return super()._make_param_name({k: order_trial_feature_names(k, v) for k, v in
                                         secondary_key_subset_map.items()})


@schema
class FrmapWADSmWT(FrmapBase):
    definition = """
    # Smoothed well arrival to departure firing rate on warped time axis
    -> STFrmapWADSmWT
    -> FrmapWADSmWTParams
    ---
    -> nd.common.AnalysisNwbfile
    frmap_wad_sm_wt_object_id : varchar(40)
    """

    def make(self, key):
        trial_feature_names = (FrmapWADSmWTParams & key).fetch1("trial_feature_names")
        fr_df = (STFrmapWADSmWT & key).fetch1_dataframe()
        new_bin_centers, new_bin_edges = (WarpedAxisBins & key).fetch1("bin_centers", "bin_edges")

        data_list = []
        for df_key in unique_df_column_sets(fr_df, trial_feature_names, as_dict=True):
            df_subset = df_filter_columns(fr_df, df_key)
            # Check that each epoch trial represented no more than once
            check_all_unique(df_subset.epoch_trial_number)
            # Take mean firing rate in common time bins across trials
            ave_fr = np.nanmean(np.asarray([np.interp(new_bin_centers, bin_centers, rate_map)
                                            for bin_centers, rate_map in
                                            zip_df_columns(df_subset, ["bin_centers", "smoothed_rate_map"])]), axis=0)
            # Store
            data_list.append(tuple(list(df_key.values()) + [ave_fr, new_bin_centers, new_bin_edges]))

        # Store in dataframe
        entry_names = trial_feature_names + ["smoothed_rate_map", "bin_centers", "bin_edges"]
        ave_fr_df = df_from_data_list(data_list, entry_names)

        # Store in table
        insert_analysis_table_entry(self, [ave_fr_df], key, [self.get_object_id_name()])


def populate_jguidera_well_arrival_departure_firing_rate_map(key=None, tolerate_error=False):
    populate_jguidera_ppt_firing_rate_map(key, tolerate_error)
    schema_name = "jguidera_well_arrival_departure_firing_rate_map"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_well_arrival_departure_firing_rate_map():
    schema.drop()
