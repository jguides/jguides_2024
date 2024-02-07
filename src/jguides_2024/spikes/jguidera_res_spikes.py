import copy

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, ComputedBase, SecKeyParamsBase, \
    PartBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry, \
    format_nwb_file_name, \
    get_table_object_id_name, insert1_print, \
    add_upstream_res_set_params, delete_, get_unit_name, get_key_filter, get_table_curation_names_for_key
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionSortGroup
from src.jguides_2024.spikes.datajoint_spikes_table_helpers import plot_smoothed_spikes_table_result, \
    firing_rate_across_sort_groups, \
    firing_rate_across_sort_groups_epochs
from src.jguides_2024.spikes.jguidera_spikes import EpochSpikeTimesRelabel, _get_kernel_standard_deviations
from src.jguides_2024.time_and_trials.jguidera_res_set import ResSet, ResSetParams
from src.jguides_2024.time_and_trials.jguidera_res_time_bins import ResEpochTimeBins
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel, ResTimeBinsPool, \
    populate_jguidera_res_time_bins_pool
from src.jguides_2024.time_and_trials.jguidera_time_bins import EpochTimeBinsParams
from src.jguides_2024.time_and_trials.jguidera_trials_pool import TrialsPoolCohortParams
from src.jguides_2024.utils.convolve_point_process import convolve_point_process_efficient
from src.jguides_2024.utils.df_helpers import get_empty_df, df_from_data_list, zip_df_columns
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.plot_helpers import plot_intervals, format_ax
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool, event_times_in_intervals, \
    bins_in_intervals
from src.jguides_2024.utils.vector_helpers import check_all_unique

# Needed for table definitions:
nd
ResTimeBinsPool


schema_name = "jguidera_res_spikes"
schema = dj.schema(schema_name)


@schema
class ResEpochSpikeTimesSel(SelBase):
    definition = """
    # Selection from upstream tables for ResEpochSpikeTimes
    -> EpochSpikeTimesRelabel
    -> ResSet
    """


@schema
class ResEpochSpikeTimes(ComputedBase):
    definition = """
    # Spike times from EpochSpikeTimesRelabel within valid time intervals
    -> ResEpochSpikeTimesSel
    ---
    valid_time_intervals : blob  # carry over from ResSet for convenience
    -> nd.common.AnalysisNwbfile
    res_epoch_spike_times_object_id : varchar(40)
    """

    def make(self, key):
        # Get epoch spike times
        spike_times_df = (
                    EpochSpikeTimesRelabel.RelabelEntries & key).fetch1_dataframe()  # each row has spike times for a unit
        # Get spike times from each unit within valid intervals
        spike_times = [ResSet().apply_restriction(key, time_bin_centers=x)[0]
                                    for x in spike_times_df.epoch_spike_times.values]
        # Store in dataframe
        df = pd.DataFrame.from_dict({"unit_id": spike_times_df.index, "spike_times": spike_times})
        # Store valid time intervals from upstream ResSet for convenience
        key["valid_time_intervals"] = (ResSet & key).fetch1("valid_time_intervals")
        # Insert into table
        insert_analysis_table_entry(self, [df], key, [self.get_object_id_name()])

    def plot_result(self, key):
        table_entry = self & key
        spikes_df = table_entry.fetch1_dataframe()
        valid_time_intervals = table_entry.fetch1("valid_time_intervals")
        fig, ax = plt.subplots(figsize=(15, 3))
        plot_intervals(valid_time_intervals, ax=ax, label="valid_time_intervals")
        for unit_id, plot_x in spikes_df.spike_times.items():
            ax.plot(plot_x, [1] * len(plot_x), '.', label=f"unit {unit_id}")
        ax.legend()
        format_ax(ax=ax, title=f"{format_nwb_file_name(key['nwb_file_name'])} ep{key['epoch']}")

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="unit_id"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def populate_(self, **kwargs):
        kwargs = add_upstream_res_set_params(kwargs)
        super().populate_(**kwargs)


@schema
class ResEpochSpikeCountsSel(SelBase):
    definition = """
    # Selection from upstream tables for ResEpochSpikeCounts
    -> EpochSpikeTimesRelabel
    -> ResTimeBinsPool
    """

    def _get_potential_keys(self, key_filter=None):
        keys = super()._get_potential_keys(key_filter)

        # Restrict time bins param to those used in certain analyses
        # First get param names
        res_time_bins_pool_param_names = []
        # 1) GLM analysis
        from src.jguides_2024.datajoint_nwb_utils.analysis_default_params import get_glm_default_param  # necessary to avoid circular import
        res_time_bins_pool_param_names += get_glm_default_param("res_time_bins_pool_param_names")
        # 2) Correlation analysis
        res_time_bins_pool_param_names += [
            ResTimeBinsPoolSel().lookup_param_name_from_shorthand(x) for x in ["path_100ms", "delay_100ms"]]
        # Now restrict
        # Note that narrowing operation below is fairly quick
        keys = [k for k in keys if k["res_time_bins_pool_param_name"] in res_time_bins_pool_param_names]

        # Return keys
        return keys

    def delete_(self, key, safemode=True):
        # Add curation_name if not present but params to define it are. Helps ensure only relevant entries for key
        # are deleted
        key = copy.deepcopy(key)
        for curation_name in get_table_curation_names_for_key(self, key):
            key.update({"curation_name": curation_name})
            delete_(self, [ResEpochSpikeCounts], key, safemode)



@schema
class ResEpochSpikeCounts(ComputedBase):
    definition = """
    # Selection from upstream tables for ResEpochSpikeCounts
    -> ResEpochSpikeCountsSel
    ---
    -> nd.common.AnalysisNwbfile
    res_epoch_spike_counts_df_object_id : varchar(40)
    """

    class Unit(dj.Part):
        definition = """
        # Placeholder for results by unit
        -> ResEpochSpikeCounts
        unit_id : int
        ---
        unit_name : varchar(10)
        """

        def fetch1_dataframe(self):
            key = self.fetch1()
            return (ResEpochSpikeCounts & key).fetch1_spike_counts_map()[key["unit_name"]]

    def make(self, key):
        # Get spike counts
        epoch_spike_times_df = (EpochSpikeTimesRelabel.RelabelEntries & key).fetch1_dataframe()  # epoch spike times
        time_bin_edges = (ResTimeBinsPool & key).fetch1_dataframe().time_bin_edges  # time bins
        spike_counts_df = df_from_data_list([
            (get_unit_name(key["sort_group_id"], unit_id), unit_id, np.histogram(
                epoch_spike_times, bins=np.hstack(time_bin_edges))[0][::2]) for unit_id, epoch_spike_times in
            epoch_spike_times_df.epoch_spike_times.items()], ["unit_name", "unit_id", "spike_counts"])

        # Insert into main table
        # (we copy df to avoid altering the one in this function; df gets altered to contain nan if empty,
        # to allow storage in analysis nwbf. This would then need to be accounted for below)
        insert_analysis_table_entry(self, [copy.deepcopy(spike_counts_df)], key)

        # Insert into part table
        if len(spike_counts_df) > 0:
            for unit_id in spike_counts_df["unit_id"]:
                key.update({"unit_id": unit_id, "unit_name": get_unit_name(key["sort_group_id"], unit_id)})
                insert1_print(self.Unit, key)

    def get_time_bin_centers(self):
        return (ResTimeBinsPool & self.fetch1("KEY")).fetch1_dataframe().time_bin_centers

    def fetch1_spike_counts_map(self):
        # Return dictionary where keys are unit names and values are series with spike counts
        return {unit_name: pd.Series(spike_counts, index=self.get_time_bin_centers()) for unit_name, spike_counts in
         list(zip_df_columns(self.fetch1_dataframe(), ["unit_name", "spike_counts"]))}

    def fetch1_spike_counts_across_brain_regions(self, brain_regions, min_spike_count=0):
        def _add_brain_region_to_df(df_, brain_region):
            df_["brain_region"] = brain_region
            return df_

        # Check sort groups represented at most once across entries
        check_all_unique(self.fetch("sort_group_id"))
        # Check and get single nwb file name across entries
        nwb_file_name = check_return_single_element(self.fetch("nwb_file_name")).single_element
        # Get sort groups across passed brain regions
        sort_group_ids_map = {brain_region: (BrainRegionSortGroup & {
                "nwb_file_name": nwb_file_name, "brain_region": brain_region}).fetch1("sort_group_ids")
             for brain_region in brain_regions}
        # Get spike counts from sort groups across brain regions
        df = pd.concat(
            [_add_brain_region_to_df((self & {"sort_group_id": sort_group_id}).fetch1_dataframe(), brain_region)
             for brain_region, sort_group_ids in sort_group_ids_map.items() for sort_group_id in sort_group_ids])
        # Restrict to units meeting minimum spike count
        valid_bool = np.sum(np.vstack(df.spike_counts), axis=1) >= min_spike_count
        return df[valid_bool]

    def delete_(self, key, safemode=True):
        from src.jguides_2024.glm.jguidera_el_net import ElNetSel
        # Add curation_name if not present but params to define it are. Helps ensure only relevant entries for key
        # are deleted
        key = copy.deepcopy(key)
        for curation_name in get_table_curation_names_for_key(self, key):
            key.update({"curation_name": curation_name})
            delete_(self, [ElNetSel], key, safemode)


""" 
Notes on restricted epoch spikes smoothed / downsampled tables:

1) The chosen workflow for smoothing spikes via these tables is as follows: we convolve spikes with a Gaussian
kernel. Here, we use an approximate method where we place Gaussians at the centers of 1ms time bins, which
is computationally more tractable than an alternative approach of sampling a Gaussian around the precise location
of each spike. This more computationally tractable approach was recommended by Christoph Kirst. This procedure
takes place in the ResEpochSpikesSm table and even though it is relatively more computationally tractable,
it is still very computationally intensive and a rate-limiting step in most analyses that use firing rate.
Since we want firing rate at different time resolutions, we subsequently downsample the firing rates above
at desired sampling rates in the ResEpochSpikesSmDs table. This avoids having to perform the computationally
expensive smoothing procedure for every desired sampling rate.

2) having multiple primary keys for params in smoothed spikes table useful when downsampling in downstream table.

3) For the smoothed table, we could have used pooled restricted time bins, but we chose instead to limit entries to
time bins from ResEpochSpikeTimes, which helps avoid multiple similar entries. One set of time bins per
restriction (epoch time bins with restriction applied) should be sufficient since we expect
to use a high sampling rate in the smoothed table.
"""

# TODO (feature): if remake table, add in parameter for kernel type
@schema
class ResEpochSpikesSmParams(SecKeyParamsBase):
    definition = """
    res_epoch_spikes_sm_param_name : varchar(10)
    ---
    kernel_standard_deviation : decimal(10,5) unsigned
    """

    def _default_params(self):
        return [[x] for x in _get_kernel_standard_deviations()]

    def delete_(self, key, safemode=True):
        delete_(self, [ResEpochSpikesSmSel], key, safemode)


@schema
class ResEpochSpikesSmSel(SelBase):
    definition = """
    # Selection from upstream tables for ResEpochSpikesSm
    -> ResEpochTimeBins
    -> ResEpochSpikeTimes
    -> ResEpochSpikesSmParams
    """

    @staticmethod
    def _valid_epoch_time_bins_param_name():
        return EpochTimeBinsParams().lookup_param_name([.001])

    # Override parent class method to check epoch time bins param name
    def insert1(self, key, **kwargs):
        if key["epoch_time_bins_param_name"] != self._valid_epoch_time_bins_param_name():
            raise Exception(f"epoch_time_bins_param_name must be {self._valid_epoch_time_bins_param_name()}")
        super().insert1(key, **kwargs)

    # Override parent class method to enforce single, small time bin width
    def _get_potential_keys(self, key_filter=None):
        if key_filter is None:
            key_filter = dict()
        key_filter.update({"epoch_time_bins_param_name": self._valid_epoch_time_bins_param_name()})
        return super()._get_potential_keys(key_filter)

    # Extend parent class method so can restrict time period to full epoch, if time period restriction not passed.
    # Note we do this here instead of in _get_potential_keys, since cleanup method will delete any entries in table
    # with key not in keys returned from _get_potential_keys. We want to be able to enter entries that
    # are not defaults, e.g. smoothed spikes on non-epoch interval for just an epoch
    def insert_defaults(self, **kwargs):
        # Restrict time period to full epoch if time period restriction not passed
        # ...Make copy of kwargs to avoid altering outside this method
        kwargs_ = copy.deepcopy(kwargs)
        # ...Get key_filter from kwargs
        key_filter = get_key_filter(kwargs_)
        # ...Add res_set_param_name corresponding to full epoch interval to key_filter if not present
        trials_pool_cohort_param_name = TrialsPoolCohortParams().lookup_param_name(["EpochInterval"], [{}])
        if "res_set_param_name" not in key_filter:
            key_filter.update(
                {"res_set_param_name": ResSetParams().lookup_no_combination_param_name(trials_pool_cohort_param_name)})
        kwargs_["key_filter"] = key_filter  # put key_filter back into kwargs

        # Run parent class method to insert defaults
        super().insert_defaults(**kwargs_)

    def delete_(self, key, safemode=True):
        # Add curation name if not present
        curation_names = get_table_curation_names_for_key(self, key)
        keys = [key]
        if curation_names is not None:
            keys = [{**key, **{"curation_name": curation_name}} for curation_name in curation_names]
        for key in keys:
            delete_(self, [ResEpochSpikesSmDsParams], key, safemode)


@schema
class ResEpochSpikesSm(ComputedBase):
    definition = """
    # Spikes convolved with a Gaussian kernel within valid intervals
    -> ResEpochSpikesSmSel
    ---
    valid_time_intervals : blob  # carry over from ResSet for convenience
    -> nd.common.AnalysisNwbfile
    res_epoch_spikes_sm_object_id : varchar(40)
    """

    def make(self, key):

        # Get spike times
        table_entry = (ResEpochSpikeTimes & key)
        spikes_df = table_entry.fetch1_dataframe()
        # Get valid time intervals
        valid_time_intervals = table_entry.fetch1("valid_time_intervals")

        # Make empty dataframe if no units
        if len(spikes_df) == 0:
            res_epoch_spikes_sm_df = get_empty_df(["unit_id", "firing_rate", "sample_times"])

        # Otherwise, get dataframe with smoothed spikes
        else:
            # Get width of Gaussian smoothing kernel
            # use param name to search to avoid conflict when using full key (from epoch time bins param name
            # being used in more than one place)
            kernel_sd = float(
                (ResEpochSpikesSmParams & {"res_epoch_spikes_sm_param_name": key["res_epoch_spikes_sm_param_name"]}
                 ).fetch1("kernel_standard_deviation"))

            # Get time bin edges for convolving spikes
            time_bin_df = (ResEpochTimeBins & key).fetch1_dataframe()

            # Check that all spike times fall within valid intervals
            all_spike_times = np.concatenate(spikes_df.spike_times.values)
            if not all(event_times_in_intervals_bool(all_spike_times, valid_time_intervals)):
                raise Exception(f"Not all spike times fell within valid intervals")

            # For each unit, loop through valid intervals and smooth spikes falling within the interval
            firing_rate_list, sample_times_list = zip(*[list(map(np.concatenate, list(zip(*[
                convolve_point_process_efficient(
                    event_times=event_times_in_intervals(spike_times, [valid_time_interval])[1],
                    kernel_sd=kernel_sd,
                    time_bins=bins_in_intervals(
                        [valid_time_interval], time_bin_df.time_bin_centers.values,
                        bin_edges=time_bin_df.time_bin_edges.values),
                    verbose=False)  # lots of printouts with restricted spikes since multiple trials
                for valid_time_interval in valid_time_intervals]))))
                for spike_times in spikes_df.spike_times.values])
            # Get corresponding unit IDs
            unit_ids = spikes_df.spike_times.index
            # Store in df
            res_epoch_spikes_sm_df = pd.DataFrame.from_dict(
                {"unit_id": unit_ids, "firing_rate": firing_rate_list, "sample_times": sample_times_list})

        # Store valid time intervals for convenience
        key["valid_time_intervals"] = valid_time_intervals

        # Insert into table
        insert_analysis_table_entry(self, [res_epoch_spikes_sm_df], key)

    def populate_(self, **kwargs):
        kwargs = add_upstream_res_set_params(kwargs)
        super().populate_(**kwargs)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="unit_id"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def get_firing_rate_in_intervals(self):
        # Return map from unit ID to array with firing rates in each valid time interval

        # Get valid time intervals and firing rates
        valid_time_intervals = self.fetch1("valid_time_intervals")
        fr_df = self.fetch1_dataframe()

        # Make map from unit id to array where rows correspond to valid time intervals and contain boolean indicating
        # sample times that fell within that valid time interval
        valid_bool_map = {unit_id: [event_times_in_intervals_bool(df_row.sample_times, [valid_time_interval])
                                    for valid_time_interval in valid_time_intervals]
                          for unit_id, df_row in fr_df.iterrows()}

        # For each unit, make array where rows correspond to valid time intervals and contain a series with firing
        # rate samples within the valid time interval
        return {unit_id: [pd.Series(fr_df.loc[unit_id].firing_rate[valid_bool],
                                    index=fr_df.loc[unit_id].sample_times[valid_bool])
                          for valid_bool in valid_bool_list] for unit_id, valid_bool_list in valid_bool_map.items()}

    def plot_result(self, key, unit_id=None, ax=None):
        plot_smoothed_spikes_table_result(self, key, unit_id=unit_id, ax=ax)


# Notes on ResEpochSpikesSmDsParams setup:
# Keep ResTimeBinsPool as primary key so can intersect tables on res_time_bins_pool_param_name
# In param name (res_epoch_spikes_sm_param_name), keep only information about smoothing kernel, and leave out
# information about time bin size and restriction used to originally smooth spikes, for user convenience.
# This is a non-canonical params table because the param name is not constructed from secondary key values; it is
# taken directly from ResEpochSpikesSm. So, base class methods for constructing param name will not work here.
# Notes on checks performed:
# 1) in params table only allow one setting of epoch time bins param name that specifies time
# bin width used for defining epoch time bins
# 2) check while downsampling spikes that restriction information matches across time bins used to downsample,
# and epoch time bins used to initially smooth
@schema
class ResEpochSpikesSmDsParams(SecKeyParamsBase):
    definition = """
    # Parameters for downsampling entries in ResEpochSpikesSm. Consolidates param names in the process
    -> EpochSpikeTimesRelabel  # supplies information about subject, epoch, sort group, and curation
    -> ResTimeBinsPool  # supplies information about time bins used for downsampling
    res_epoch_spikes_sm_param_name : varchar(40)  # information about smoothing kernel
    ---
    -> ResEpochSpikesSm  # supplies information about upstream smoothing process beyond smoothing kernel (but is not primary key since dont want all primary dependencies)
    time_bin_width : float  # for convenience, implicitly captured in ResTimeBinsPool dependence
    source_table_name : varchar(40)  # for convenience, implicitly captured in ResTimeBinsPool dependence
    """

    # Makes sense to have separate method here from smooth table since want to allow more time bins
    # params in smooth table (e.g. in case ever wanted even higher temporal resolution)
    @staticmethod
    def _valid_epoch_time_bins_param_name():
        return EpochTimeBinsParams().lookup_param_name([.001])

    def insert1(self, key, **kwargs):
        # Only allow one epoch time bin width (used for smoothing spikes originally) since subsequently
        # lose this information
        valid_param_val = self._valid_epoch_time_bins_param_name()
        if key["epoch_time_bins_param_name"] != valid_param_val:
            raise Exception(f"Only epoch_time_bin_width = {valid_param_val} is allowed")
        super().insert1(key, **kwargs)

    # Override parent class method since non-canonical params table
    def insert_defaults(self, **kwargs):

        # Get key filter if passed
        key_filter = get_key_filter(kwargs)

        # Define sets of time bins we want to downsample on
        # ...Use single res_time_bins_pool_param_name if passed
        if "res_time_bins_pool_param_name" in key_filter:
            res_time_bin_pool_param_names = [key_filter["res_time_bins_pool_param_name"]]
        # .... Otherwise define here
        else:
            ResTimeBinsPool().populate_(key=key_filter)  # ensure table populated to avoid dependency error
            res_time_bin_pool_param_names = [
                ResTimeBinsPoolSel().lookup_param_name_from_shorthand(x) for x in
                ["epoch_100ms",  # embedding
                  "wa1_100ms",  # embedding
                  "delay_100ms",  # embedding
                  "post_delay_100ms",  # embedding
                ]]

        # Constrain epoch time bin width for initial computation of smoothed spikes to be a single value. The idea is we
        # use one set of highly sampled smoothed spikes to downsample here
        key = {"epoch_time_bins_param_name": self._valid_epoch_time_bins_param_name()}

        # Loop through possible entries and insert
        for res_time_bin_pool_param_name in res_time_bin_pool_param_names:

            # Update key with source_table_name and res_set_param_name for the current res_time_bin_pool_param_name
            key.update({**{
                param_name: check_return_single_element(
                    [x[param_name] for x in (ResTimeBinsPoolSel & {
                        "res_time_bins_pool_param_name": res_time_bin_pool_param_name}).fetch(
                                    "param_name_dict")]).single_element
                        for param_name in ["source_table_name", "res_set_param_name"]},
                        **{"res_time_bins_pool_param_name": res_time_bin_pool_param_name}})

            # Add time bin width (will be secondary key)
            key.update({"time_bin_width": check_return_single_element(
                [(ResTimeBinsPoolSel & k).get_time_bin_width() for k in
                 (ResTimeBinsPoolSel & key).fetch("KEY")]).single_element})

            for k in (ResEpochSpikesSm & {**key_filter, **key}).fetch("KEY"):
                super().insert1({**key, **k})

    # Override parent class method since non-canonical params table
    def lookup_param_name(self, secondary_key_values, as_dict=False):
        return ResEpochSpikesSmParams().lookup_param_name(secondary_key_values, as_dict)

    def delete_(self, key, safemode=True):
        # Add res_time_bins_pool_param_name if not in key and quantities to define it are
        # if "res_time_bins_pool_param_name" not in key and

        # Add curation name if not present
        from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import FRVecSel
        curation_names = get_table_curation_names_for_key(self, key)
        keys = [key]
        if curation_names is not None:
            keys = [{**key, **{"curation_name": curation_name}} for curation_name in curation_names]
        for key in keys:
            delete_(self, [FRVecSel], key, safemode)


@schema
class ResEpochSpikesSmDs(ComputedBase):
    definition = """
    # Downsampled smoothed epoch spikes within valid intervals
    -> ResEpochSpikesSmDsParams
    ---
    valid_time_intervals : blob  # carry over from ResEpochSpikesSm for convenience
    -> nd.common.AnalysisNwbfile
    res_epoch_spikes_sm_ds_object_id : varchar(40)
    """

    class Upstream(PartBase):
        definition = """
        # Achieves dependence on ResEpochSpikesSm
        -> ResEpochSpikesSmDs
        -> ResEpochSpikesSm
        """

    def make(self, key):
        # Get firing rate
        upstream_key = {**key, **{k: (ResEpochSpikesSmDsParams & key).fetch1(k) for k in
                                  ["res_epoch_spikes_sm_param_name", "epoch_time_bins_param_name", "res_set_param_name"]}}
        table_entry = (ResEpochSpikesSm & upstream_key)
        fr_map = table_entry.get_firing_rate_in_intervals()

        # If no units, make empty df
        if len(fr_map) == 0:
            downsampled_rate_df = get_empty_df(["unit_id", "firing_rate", "sample_times"])

        # Otherwise, downsample firing rate using restricted time bins from pool table
        else:
            # Get time bin centers as new time index
            time_bin_centers = (ResTimeBinsPool & key).fetch1_dataframe().time_bin_centers

            # Check that all time bins fall within valid intervals
            valid_time_intervals = (ResEpochSpikesSm & upstream_key).fetch1("valid_time_intervals")
            if not all(event_times_in_intervals_bool(time_bin_centers, valid_time_intervals)):
                raise Exception(f"All time bins must be in valid intervals")

            # Interpolate firing rate on new index
            data_list = []
            for unit_id, unit_fr in fr_map.items():
                # concatenate series with unit firing rate across all valid time intervals
                unit_fr = pd.concat(unit_fr)
                # Loop through valid intervals and interpolate within each
                fr_ds_list = []
                new_times_list = []
                for valid_time_interval in valid_time_intervals:
                    time_bin_centers_subset = event_times_in_intervals(time_bin_centers, [valid_time_interval])[1]
                    unit_fr_subset = unit_fr[event_times_in_intervals_bool(unit_fr.index, [valid_time_interval])]
                    fr_ds_list.append(np.interp(time_bin_centers_subset, unit_fr_subset.index, unit_fr_subset.values))
                    new_times_list.append(time_bin_centers_subset)
                # Concatenate results of interpolation across valid intervals
                data_list.append((unit_id, np.concatenate(fr_ds_list), np.concatenate(new_times_list)))

            # Store in dataframe
            downsampled_rate_df = df_from_data_list(data_list, ["unit_id", "firing_rate", "sample_times"])

        # Store valid time intervals for convenience
        key["valid_time_intervals"] = table_entry.fetch1("valid_time_intervals")

        # Insert into main table
        insert_analysis_table_entry(self, [downsampled_rate_df], key, [get_table_object_id_name(self)])

        # Insert into part table
        insert1_print(self.Upstream, upstream_key)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="unit_id"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def plot_result(self, key=None, unit_id=None, ax=None):
        plot_smoothed_spikes_table_result(self, key, unit_id=unit_id, ax=ax)

    def firing_rate_across_sort_groups(self, key, sort_group_unit_ids_map, sort_group_id_label_map=None,
                                       label_name=None, populate_tables=True):
        return firing_rate_across_sort_groups(
            self, key, sort_group_unit_ids_map, sort_group_id_label_map, label_name, populate_tables)

    def firing_rate_across_sort_groups_epochs(
            self, epochs, sort_group_unit_ids_map, key=None, keys=None, sort_group_id_label_map=None,
            label_name=None, populate_tables=True, verbose=False):
        # Return firing rate across sort groups and epochs, and corresponding time vectors
        return firing_rate_across_sort_groups_epochs(self, epochs, sort_group_unit_ids_map, key, keys,
                                                     sort_group_id_label_map, label_name, populate_tables, verbose)


def populate_jguidera_res_spikes(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_res_spikes"
    upstream_schema_populate_fn_list = [populate_jguidera_res_time_bins_pool]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_res_spikes():
    from src.jguides_2024.glm.jguidera_el_net import drop_jguidera_el_net
    drop_jguidera_el_net()
    schema.drop()
