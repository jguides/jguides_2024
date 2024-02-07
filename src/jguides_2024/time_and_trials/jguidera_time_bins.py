import datajoint as dj
import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase, \
    TrialsTimeBinsParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    fetch1_dataframe, get_key_filter
from src.jguides_2024.datajoint_nwb_utils.metadata_helpers import get_high_priority_nwb_file_names
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import get_epoch_time_interval
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellArrivalTrials, DioWellDATrials, \
    DioWellADTrials, DioWellArrivalTrialsSub
from src.jguides_2024.utils.df_helpers import zip_df_columns
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements, unzip_adjacent_elements
from src.jguides_2024.utils.make_bins import make_bin_edges
from src.jguides_2024.utils.vector_helpers import vector_midpoints

schema = dj.schema("jguidera_time_bins")


@schema
class EpochTimeBinsParams(SecKeyParamsBase):
    definition = """
    # Table with width parameter in seconds for binning time
    epoch_time_bins_param_name : varchar(40)
    ---
    time_bin_width : decimal(10,5) unsigned
    """

    def _default_params(self):
        params = np.asarray([.1, .02, .001])
        return [[x] for x in params]

    def get_time_bin_width(self):
        return float(self.fetch1("time_bin_width"))


@schema
class EpochTimeBins(ComputedBase):
    definition = """
    # Time bins within epoch 
    -> TaskIdentification
    -> EpochTimeBinsParams
    ---
    -> nd.common.AnalysisNwbfile
    epoch_time_bins_object_id : varchar(40)
    """

    def make(self, key):
        time_bin_width = float((EpochTimeBinsParams & key).fetch1("time_bin_width"))  # decimal to float
        # Get epoch start and end
        epoch_start, epoch_end = get_epoch_time_interval(key["nwb_file_name"], key["epoch"])
        # To get start of first bin and end of last bin, round epoch start up and epoch end down to nearest multiple
        # of time bin size. We use linspace because it is more precise than np.arange, and in this context respects
        # a constant bin width because of our choice of start and end times for the binning
        bins_start = np.ceil(epoch_start / time_bin_width) * time_bin_width
        bins_end = np.floor(epoch_end / time_bin_width) * time_bin_width
        num_bins = int((bins_end - bins_start) / time_bin_width)
        time_bin_edges = np.linspace(bins_start, bins_end, num=num_bins)
        time_bin_edges_zip = zip_adjacent_elements(time_bin_edges)  # tuples with start/end of time bins
        epoch_time_bins = pd.DataFrame({"time_bin_edges": time_bin_edges_zip,
                                        "time_bin_centers": vector_midpoints(time_bin_edges)})
        insert_analysis_table_entry(self, [epoch_time_bins], key, ["epoch_time_bins_object_id"])

    def fetch1_epoch_time_bin_edges(self):
        # Unroll time bin edges
        return unzip_adjacent_elements(fetch1_dataframe(self, "epoch_time_bins")["time_bin_edges"].values)


@schema
class DioWATrialsTimeBinsParams(TrialsTimeBinsParamsBase):
    definition = """
    # Parameters for DioWATrialsTimeBins
    dio_wa_trials_time_bins_param_name : varchar(40)
    ---
    time_bin_width : decimal(10,5) unsigned
    """


@schema
class DioWATrialsTimeBinsSel(SelBase):
    definition = """
    # Selection from upstream tables for DioWATrialsTimeBins
    -> DioWellArrivalTrials
    -> DioWATrialsTimeBinsParams
    """

    def insert_defaults(self, **kwargs):
        key_filter = kwargs.pop("key_filter", None)
        keys = self._get_potential_keys(key_filter)
        # Restrict to high priority nwb files
        high_priority_nwb_file_names = get_high_priority_nwb_file_names()
        keys = [key for key in keys if key["nwb_file_name"] in high_priority_nwb_file_names]
        for key in keys:
            self.insert1(key, skip_duplicates=True)


@schema
class DioWATrialsTimeBins(ComputedBase):
    definition = """
    # Time bins during trials based on single well arrival detected with dios
    -> DioWATrialsTimeBinsSel
    ---
    -> nd.common.AnalysisNwbfile
    dio_well_arrival_trials_time_bins_object_id : varchar(40)
    """

    def make(self, key):
        insert_trials_time_bins_table(
            self, params_table=DioWATrialsTimeBinsParams, trials_table=DioWellArrivalTrials, key=key,
            nwb_object_name="dio_well_arrival_trials_time_bins_object_id")


@schema
class DioWATrialsSubTimeBinsParams(TrialsTimeBinsParamsBase):
    definition = """
    # Parameters for DioWATrialsSubTimeBins
    dio_wa_trials_sub_time_bins_param_name : varchar(40)
    ---
    time_bin_width : decimal(10,5) unsigned
    """

    # Override parent class method
    def _default_params(self):
        return [[x] for x in [.02]]


@schema
class DioWATrialsSubTimeBinsSel(SelBase):
    definition = """
    # Selection from upstream tables for DioWATrialsSubTimeBins
    -> DioWellArrivalTrialsSub
    -> DioWATrialsSubTimeBinsParams
    """

    def insert_defaults(self, **kwargs):
        key_filter = kwargs.pop("key_filter", None)
        keys = self._get_potential_keys(key_filter)
        # Restrict to high priority nwb files
        high_priority_nwb_file_names = get_high_priority_nwb_file_names()
        keys = [key for key in keys if key["nwb_file_name"] in high_priority_nwb_file_names]
        for key in keys:
            self.insert1(key, skip_duplicates=True)


@schema
class DioWATrialsSubTimeBins(ComputedBase):
    definition = """
    # Time bins during subset of trials based on single well arrival detected with dios
    -> DioWATrialsSubTimeBinsSel
    ---
    -> nd.common.AnalysisNwbfile
    dio_wa_trials_sub_time_bins_object_id : varchar(40)
    """

    def make(self, key):
        insert_trials_time_bins_table(
            self, params_table=DioWATrialsSubTimeBinsParams, trials_table=DioWellArrivalTrialsSub, key=key)


@schema
class DioWellDATrialsTimeBinsParams(TrialsTimeBinsParamsBase):
    definition = """
    # Parameters for DioWellDATrialsTimeBins
    dio_well_da_trials_time_bins_param_name : varchar(40)
    ---
    time_bin_width : decimal(10,5) unsigned
    """


@schema
class DioWellDATrialsTimeBinsSel(SelBase):
    definition = """
    # Selection from upstream tables for DioWellDATrialsTimeBins
    -> DioWellDATrials
    -> DioWellDATrialsTimeBinsParams
    """

    def insert_defaults(self, **kwargs):
        key_filter = get_key_filter(kwargs)
        keys = self._get_potential_keys(key_filter)
        # Restrict to high priority nwb files
        high_priority_nwb_file_names = get_high_priority_nwb_file_names()
        keys = [key for key in keys if key["nwb_file_name"] in high_priority_nwb_file_names]
        for key in keys:
            self.insert1(key, skip_duplicates=True)


@schema
class DioWellDATrialsTimeBins(ComputedBase):
    definition = """
    # Time bins during trials based on single well arrival detected with dios
    -> DioWellDATrialsTimeBinsSel
    ---
    -> nd.common.AnalysisNwbfile
    dio_well_da_trials_time_bins_object_id : varchar(40)
    """

    def make(self, key):
        insert_trials_time_bins_table(
            self, params_table=DioWellDATrialsTimeBinsParams, trials_table=DioWellDATrials, key=key)


@schema
class DioWellADTrialsTimeBinsParams(TrialsTimeBinsParamsBase):
    definition = """
    # Parameters for DioWellADTrialsTimeBins
    dio_well_ad_trials_time_bins_param_name : varchar(40)
    ---
    time_bin_width : decimal(10,5) unsigned
    """


@schema
class DioWellADTrialsTimeBinsSel(SelBase):
    definition = """
    # Selection from upstream tables for DioWellADTrialsTimeBins
    -> DioWellADTrials
    -> DioWellADTrialsTimeBinsParams
    """

    def insert_defaults(self, **kwargs):
        key_filter = get_key_filter(kwargs)
        keys = self._get_potential_keys(key_filter)
        # Restrict to high priority nwb files
        high_priority_nwb_file_names = get_high_priority_nwb_file_names()
        keys = [key for key in keys if key["nwb_file_name"] in high_priority_nwb_file_names]
        for key in keys:
            self.insert1(key, skip_duplicates=True)


@schema
class DioWellADTrialsTimeBins(ComputedBase):
    definition = """
    # Time bins during trials that begin at well arrivals and end at well departure detected with dios
    -> DioWellADTrialsTimeBinsSel
    ---
    -> nd.common.AnalysisNwbfile
    dio_well_ad_trials_time_bins_object_id : varchar(40)
    """

    def make(self, key):
        insert_trials_time_bins_table(
            self, params_table=DioWellADTrialsTimeBinsParams, trials_table=DioWellADTrials, key=key)


def insert_trials_time_bins_table(table, params_table, trials_table, key, nwb_object_name=None):
    # Unpack source values
    bin_width = float((params_table & key).fetch1("time_bin_width"))
    trials_df = (trials_table & key).fetch1_dataframe()

    # Get time bin edges and centers
    # ...Get time bin edges for each trial
    time_bin_edges_list = [make_bin_edges(
        [trial_start_time, trial_end_time], bin_width, match_min_max="min", bins_encompass_x=False)
        for trial_start_time, trial_end_time in zip_df_columns(trials_df, ["trial_start_times", "trial_end_times"])]
    # ...Get time bin centers in trials
    time_bin_centers_list = [vector_midpoints(x) for x in time_bin_edges_list]
    # ...Hold onto number of time bin centers per trial for later use
    num_time_bin_centers_list = list(map(len, time_bin_centers_list))
    # ...Zip time bin edges so that have edges corresponding to time bin centers, and concatenate across trials
    # convert each list of edges to list of tuples with start/end of bins
    time_bin_edges_zip = [zip_adjacent_elements(x) for x in time_bin_edges_list]
    # remove empty lists before concatenation across trials to avoid error
    time_bin_edges_zip = [x for x in time_bin_edges_zip if len(x) > 0]
    # concatenate lists of bin edges across trials
    time_bin_edges = np.concatenate(time_bin_edges_zip)
    # restore tuple
    time_bin_edges = list(map(tuple, time_bin_edges))
    # ...Concatenate time bin centers across trials
    time_bin_centers = np.concatenate(time_bin_centers_list)

    # Make vectors with values in other columns of trials_df that are same length as bin centers
    keep_column_names = set(trials_df.columns) - set(["trial_start_times", "trial_end_times"])
    other_info_dict = {column_name: np.concatenate(
        [[trials_df.iloc[idx][column_name]] * num_time_bin_edges for idx, num_time_bin_edges in enumerate(
            num_time_bin_centers_list)]) for column_name in keep_column_names}

    # Store time bin edges and other quantities in trials_df
    time_bins_df = pd.DataFrame.from_dict(
        {**{"time_bin_centers": time_bin_centers, "time_bin_edges": time_bin_edges}, **other_info_dict})

    # Insert into table
    if nwb_object_name is None:
        nwb_object_names = None
    else:
        nwb_object_names = [nwb_object_name]
    insert_analysis_table_entry(table, [time_bins_df], key, nwb_object_names)


def populate_jguidera_time_bins(key=None, tolerate_error=False, populate_upstream_limit=None,
                                         populate_upstream_num=None):
    schema_name = "jguidera_time_bins"
    upstream_schema_populate_fn_list = None
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_time_bins():
    schema.drop()
