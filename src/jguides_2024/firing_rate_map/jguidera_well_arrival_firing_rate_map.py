# This module defines tables with well arrival firing rate maps

import datajoint as dj
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_fr_table_helpers import make_well_trial_table_fr_df, \
    make_well_single_trial_table_fr_df, \
    insert_firing_rate_map_unique_well_table, \
    insert_single_trial_firing_rate_map_smoothed_well_table, get_bin_centers_name
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import ComputedBase, FrmapBase, FrmapSmBase, \
    TemporalFrmapParamsBase, TemporalFrmapSmParamsBase, SelBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import (insert_analysis_table_entry,
                                                                          get_schema_table_names_from_file,
                                                                          populate_insert)
from src.jguides_2024.spikes.jguidera_spikes import EpochSpikeTimesRelabel
from src.jguides_2024.task_event.jguidera_dio_trials import (DioWellArrivalTrials, DioWellArrivalTrialsParams)

# Needed for table definitions:
nd
EpochSpikeTimesRelabel
DioWellArrivalTrials
DioWellArrivalTrialsParams


schema = dj.schema("jguidera_well_arrival_firing_rate_map")


@schema
class FrmapWellArrivalParams(TemporalFrmapParamsBase):
    definition = """
    # Parameters for calculating firing rate as a function of time relative to well arrival
    frmap_well_arrival_param_name : varchar(50)
    ---
    time_bin_width : decimal(10,5) unsigned
    """


@schema
class FrmapWellArrivalSel(SelBase):
    definition = """
    # Selection from upstream tables for FrmapWellArrival
    -> DioWellArrivalTrials
    -> EpochSpikeTimesRelabel
    -> FrmapWellArrivalParams
    """


@schema
class FrmapWellArrival(FrmapBase):
    definition = """
    # Firing rate as a function of time relative to well arrival
    -> FrmapWellArrivalSel
    ---
    -> nd.common.AnalysisNwbfile
    frmap_well_arrival_object_id : varchar(40)
    """

    def make(self, key):
        firing_rate_map_wa_df = make_well_trial_table_fr_df(
            key, trials_table=DioWellArrivalTrials, trials_params_table=DioWellArrivalTrialsParams,
            firing_rate_map_params_table=FrmapWellArrivalParams)
        insert_analysis_table_entry(self, nwb_objects=[firing_rate_map_wa_df], key=key)


@schema
class FrmapUniqueWellArrival(ComputedBase):
    definition = """
    # Firing rate as a function of time relative to arrival at a specific well
    -> DioWellArrivalTrials
    -> EpochSpikeTimesRelabel
    -> FrmapWellArrivalParams
    well_name : varchar(40)
    ---
    -> nd.common.AnalysisNwbfile
    frmap_unique_well_arrival_object_id : varchar(40)
    """

    def make(self, key):
        trial_feature_name = "well_name"

        insert_firing_rate_map_unique_well_table(
            self, trials_table=DioWellArrivalTrials, trials_params_table=DioWellArrivalTrialsParams,
            firing_rate_map_params_table=FrmapWellArrivalParams, trial_feature_name=trial_feature_name, key=key)


@schema
class FrmapWellArrivalSmParams(TemporalFrmapSmParamsBase):
    definition = """
    # Parameters for smoothing well arrival firing rate maps
    frmap_well_arrival_sm_param_name : varchar(40)
    ---
    kernel_sd : decimal(10,5) unsigned
    """


@schema
class FrmapWellArrivalSm(FrmapSmBase):
    definition = """
    # Smoothed well arrival firing rate
    -> FrmapWellArrival
    -> FrmapWellArrivalSmParams
    ---
    -> nd.common.AnalysisNwbfile
    frmap_well_arrival_sm_object_id : varchar(80)
    """

    @staticmethod
    def _data_type():
        return "time"


@schema
class FrmapUniqueWellArrivalSm(FrmapSmBase):
    definition = """
    # Smoothed unique well arrival firing rate
    -> FrmapUniqueWellArrival
    -> FrmapWellArrivalSmParams
    ---
    -> nd.common.AnalysisNwbfile
    frmap_unique_well_arrival_sm_object_id : varchar(80)
    """

    @staticmethod
    def _data_type():
        return "time"


@schema
class STFrmapWellArrival(ComputedBase):
    definition = """ 
    # Firing rate as a function of time relative to well arrival on single trials
    -> DioWellArrivalTrials
    -> EpochSpikeTimesRelabel
    -> FrmapWellArrivalParams
    ---
    -> nd.common.AnalysisNwbfile
    st_frmap_well_arrival_object_id : varchar(40)
    """

    def make(self, key):
        # Get rate maps
        st_frmap_well_arrival_df = make_well_single_trial_table_fr_df(
            key, trials_table=DioWellArrivalTrials, trials_params_table=DioWellArrivalTrialsParams,
            firing_rate_map_params_table=FrmapWellArrivalParams)

        # Insert into table
        insert_analysis_table_entry(self, [st_frmap_well_arrival_df], key, [self.get_object_id_name()])


@schema
class STFrmapWellArrivalSm(ComputedBase):
    definition = """
    # Smoothed proportion path traversed firing rate
    -> STFrmapWellArrival
    -> FrmapWellArrivalSmParams
    ---
    -> nd.common.AnalysisNwbfile
    st_frmap_well_arrival_sm_object_id : varchar(40)
    """

    @staticmethod
    def _data_type():
        return "time"

    def make(self, key):
        # Smooth rate maps

        insert_single_trial_firing_rate_map_smoothed_well_table(
            fr_smoothed_table=self, fr_table=STFrmapWellArrival, params_table=FrmapWellArrivalSmParams, key=key,
            data_type=self._data_type())

    def get_bin_centers_name(self):
        return get_bin_centers_name(self._data_type())

    def _get_xlims(self):
        trial_start_time_shift, trial_end_time_shift = (DioWellArrivalTrialsParams & self.fetch1("KEY")).trial_shifts()
        return [trial_start_time_shift, trial_end_time_shift]


def populate_jguidera_well_arrival_firing_rate_map(key=None, tolerate_error=False):
    schema_name = "jguidera_well_arrival_firing_rate_map"

    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_well_arrival_firing_rate_map():
    schema.drop()
