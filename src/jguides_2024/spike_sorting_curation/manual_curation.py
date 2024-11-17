import itertools
import multiprocessing as mp
import os
import pickle
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import spikeinterface as si
from spyglass.common import IntervalList
from spyglass.spikesorting.v0.spikesorting_recording import SpikeSortingRecording
from spyglass.spikesorting.v0.spikesorting_sorting import SpikeSorting
from spyglass.spikesorting.v0.sortingview import SortingviewWorkspace
from spyglass.spikesorting.v0.spikesorting_curation import (CuratedSpikeSorting, Waveforms, WaveformSelection)

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name, make_params_string
from src.jguides_2024.datajoint_nwb_utils.nwbf_helpers import nwbf_name_from_subject_id_date
from src.jguides_2024.metadata.jguidera_brain_region import SortGroupTargetedLocation
from src.jguides_2024.metadata.jguidera_epoch import RunEpoch, SleepEpoch
from src.jguides_2024.spike_sorting_curation.jguidera_spikesorting import return_spikesorting_params, \
    get_default_waveform_params_name
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.utils.array_helpers import mask_upper_diagonal
from src.jguides_2024.utils.cd_make_if_nonexistent import cd_make_if_nonexistent
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.client_helpers import get_google_spreadsheet
from src.jguides_2024.utils.df_helpers import zip_df_columns
from src.jguides_2024.utils.parallelization_helpers import show_error
from src.jguides_2024.utils.plot_helpers import add_colorbar, format_ax, get_fig_axes, get_ax_for_layout, \
    plot_spanning_line
from src.jguides_2024.utils.vector_helpers import check_all_unique, unpack_single_element, overlap


def _convert_curation_spreadsheet_dtype(row, column_name):
    """
    Helper function for converting spreadsheet data to particular datatypes
    :param row: pandas df row, row of spreadsheet with manual curation
    :param column_name: str, name of column in spreadsheet with manual curation
    :return: pandas df row, passed row with indicated data as integer datatype
    """

    # Define names of columns in spreadsheet for which to convert values to integer datatype
    int_column_names = ["sort_group_id", "unit_id_1", "unit_id_2", "unit_id"]

    # Define names of columns in spreadsheet for which to convert values to boolean datatype
    bool_column_names = ["label"]

    # Define map from values in spreadsheet to new boolean values
    bool_map = {"yes": True, "no": False, "unsure": None}

    # Convert datatypes
    # ...Convert to integer datatype
    if column_name in int_column_names:
        return row.astype(int)
    # ...Convert to boolean datatype
    elif column_name in bool_column_names:
        return [bool_map[x.strip()] for x in row]  # strip_string whitespace and convert to bool variable

    # Return dataframe row with new datatypes
    return row


def get_curation_spreadsheet_notes(subject_id, date, sort_description, curation_version, tolerate_no_notes=True):
    """
    Get google spreadsheet with manual curation notes
    :param subject_id: str, name of subject
    :param date: str, date in the form YYYYMMDD where YYYY corresponds to the year, MM corresponds to the month, and
    DD corresponds to the day
    :param sort_description: str, description of the spike sorting
    :param curation_version: str, v2 or v3 currently supported
    :param tolerate_no_notes: bool, True to tolerate notes not existing, False to raise error if no notes
    :return: pandas df with merge labels
    :return: pandas df with merge labels
    """

    # Define directory with service account file
    service_account_dir = f"/home/jguidera/Src/jguides_2024_archive/nwb_custom_analysis_untracked/spikesorting_notebooks/curation_merge_notes"

    # Define name of service account file
    service_account_json = "frank-lab-jupyter-01e1afefcf28.json"

    # Define key to spreadsheet
    if curation_version == "v2":
        spreadsheet_key = "1GKyg4Apwwk6kd48_tuyBC5fI_reE8Gn9pRQ9J0X7hfo"
    elif curation_version == "v3":
        spreadsheet_key = "1WEwkIGHfzcpvT30ySAHAk_xWc3f9rT-SbInx5Qsa_Ck"
    else:
        raise Exception(f"spreadsheet_key {spreadsheet_key} not recognized")

    # Define names of columns in spreadsheet to retrieve
    column_names = np.asarray([
        "sort_group_id", "unit_id_1", "unit_id_2", "merge_type", "notes", "label_1", "label_2",
        "least restrictive label", "most restrictive label", "action items"])
    spreadsheet_tab_name = f"{subject_id}{date}_{sort_description}"

    # Load spreadsheet into df if possible
    try:
        table = np.asarray(get_google_spreadsheet(
            service_account_dir, service_account_json, spreadsheet_key, spreadsheet_tab_name))

    # Otherwise either return empty df, or raise error as indicated
    except:

        # Define message to display if could not get spreadsheet
        failure_message = f"Could not get google spreadsheet with curation notes with name {spreadsheet_tab_name}"

        # If tolerating no spreadsheet, return empty dataframe with the desired spreadsheet column names
        if tolerate_no_notes:
            print(failure_message)
            return pd.DataFrame(columns=column_names)

        # Otherwise raise error
        else:
            raise Exception(failure_message)

    # Get merge labels and additional labels in separate dfs

    # First indices of spreadsheet rows where merge labels and additional labels start and end. Note
    # that column names come one before section start
    # ...Get index of row where merge labels start: two after "merge group labels"
    merge_idx_start = unpack_single_element(np.where(table[:, 0] == "merge group labels")[0]) + 2
    # ...Get index of row where merge group labels end: at "additional labels"
    merge_idx_end = unpack_single_element(np.where(table[:, 0] == "additional labels")[0])
    # ...Get index of row where additional labels start: two after "additional labels"
    label_idx_start = merge_idx_end + 2
    # ...Get index of row where additional labels end: end of array
    label_idx_end = len(table)

    # Return merge labels and additional labels in separate dataframes
    return [pd.DataFrame.from_dict(
        {column_name: _convert_curation_spreadsheet_dtype(row, column_name)
         for column_name, row in zip(table[idx_start - 1, :], table[idx_start:idx_end, :].T)})
         for (idx_start, idx_end) in [(merge_idx_start, merge_idx_end), (label_idx_start, label_idx_end)]]


def get_cluster_data_file_name(
        nwb_file_name, sort_interval_name, sorter, preproc_params_name, curation_id, sort_group_id=None,
        target_region=None):
    """
    Get name of file with cluster data
    :param nwb_file_name: str, name of nwb file
    :param sort_interval_name: str, name of sort interval
    :param sorter: str, name of sorter used for spikesorting
    :param preproc_params_name: str, name of task_event parameters used for spikesorting
    :param curation_id: int, curation ID
    :param sort_group_id: int, corresponds to sort group
    :param target_region: str, targeted brain region
    :return: str, name of file with cluster data
    """

    # Check that only one of sort group ID and target region passed
    check_one_none([sort_group_id, target_region])

    # Define text to describe electrode as target region if passed, or sort group ID if passed
    electrode_text = target_region
    if sort_group_id is not None:
        electrode_text = sort_group_id

    # Define file name as joined parameter names
    return make_params_string(
        [format_nwb_file_name(nwb_file_name), sort_interval_name, sorter, preproc_params_name, curation_id,
         electrode_text])


def load_curation_data(
        save_dir, nwb_file_name, sort_interval_name, sorter="mountainsort4",
        preproc_params_name="franklab_tetrode_hippocampus", curation_id=1, sort_group_ids=None, verbose=True):
    """
    Load curation data
    :param save_dir: str, name of directory with curation data
    :param nwb_file_name: str, name of nwb file
    :param sort_interval_name: str, name of sort interval
    :param sorter: str, name of sorter used for spikesorting
    :param preproc_params_name: str, name of task_event parameters used for spikesorting
    :param sort_group_ids: int, corresponds to sort group
    :param curation_id: int, curation ID
    :param verbose: bool, True to print progress in function
    :return: dictionary with metadata and curation data organized by sort group
    """

    if verbose:
        print(f"Loading curation data for {nwb_file_name}...")

    # Make directory with curation data if does not exist, then change to directory with curation data
    cd_make_if_nonexistent(save_dir)

    # Get curation data for each sort group and store in dictionary where key is sort group ID
    sort_groups_data = dict()
    for sort_group_id in sort_group_ids:  # loop through sort group IDs

        # Get name of cluster data for this sort group
        file_name_save = get_cluster_data_file_name(
            nwb_file_name, sort_interval_name, sorter, preproc_params_name, curation_id, sort_group_id=sort_group_id)

        # Continue if data doesnt exist
        if not os.path.exists(file_name_save):
            print(f"No data for {file_name_save}, skipping...")
            continue
        else:
            print(f"Loading {file_name_save}...")

        # Otherwise store sort group data
        sort_groups_data[sort_group_id] = pickle.load(open(file_name_save, "rb"))

    # Return dictionary with metadata and curation data organized by sort group
    return {"nwb_file_name": nwb_file_name, "sort_interval_name": sort_interval_name, "sorter": sorter,
            "preproc_params_name": preproc_params_name, "n_sort_groups": len(sort_group_ids),
            "sort_groups": sort_groups_data}


def make_curation_data(
        save_dir, nwb_file_name, sort_interval_name, sorter="mountainsort4",
        preproc_params_name="franklab_tetrode_hippocampus", sort_group_ids=None, curation_id=1, get_workspace_url=True,
        ignore_invalid_sort_group_ids=False, overwrite_existing=False, include_amplitude_decrement_quantities=True,
        tolerate_nonexisting=False, verbose=True):
    """
    Compute quantities from spikesorted data for use in manual curation
    :param save_dir: str, name of directory with curation data
    :param nwb_file_name: str, name of nwb file
    :param sort_interval_name: str, name of sort interval
    :param sorter: str, name of sorter used for spikesorting
    :param preproc_params_name: str, name of task_event parameters used for spikesorting
    :param sort_group_ids: list of sort group IDs
    :param curation_id: int, curation ID
    :param get_workspace_url: bool, True to get figurl workspace URL
    :param ignore_invalid_sort_group_ids: bool, True to skip invalid sort group ID, False to raise error
    :param overwrite_existing: bool, True to overwrite existing file with curation data, False to skip
    :param include_amplitude_decrement_quantities:
    :param tolerate_nonexisting:
    :param verbose: bool, True to print progress
    :return:
    """

    # Check that key specific enough (each sort group represented no more than once)
    key = {"nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
            "sorter": sorter,
            "preproc_params_name": preproc_params_name,
            "curation_id": curation_id}
    valid_sort_group_ids = [x for x in (SpikeSorting & key).fetch("sort_group_id")]
    check_all_unique(valid_sort_group_ids)

    # Define sort group ids if not passed
    if sort_group_ids is None:
        sort_group_ids = valid_sort_group_ids

    # Check that passed sort group ids are valid
    if not ignore_invalid_sort_group_ids and not set(sort_group_ids).issubset(set(valid_sort_group_ids)):
        raise ValueError(f"List of sort groups includes invalid sort group IDs")

    # Get correlogram default params
    default_params = get_correlogram_default_params()
    correlogram_max_dt, correlogram_min_dt = default_params["max_dt"], default_params["min_dt"]

    # Loop through sort groups and make cluster data if does not exist or want to overwrite
    for sort_group_id in sort_group_ids:

        # Continue if sort group invalid and want to tolerate this
        if ignore_invalid_sort_group_ids and sort_group_id not in valid_sort_group_ids:
            continue

        # Continue if file already exists and dont want to overwrite
        file_name_save = get_cluster_data_file_name(
            nwb_file_name, sort_interval_name, sorter, preproc_params_name, curation_id, sort_group_id)
        if os.path.exists(os.path.join(save_dir, file_name_save)) and not overwrite_existing:
            print(f"Cluster data exists for {nwb_file_name}, sort group {sort_group_id}; continuing")
            continue

        # Otherwise, make cluster data
        if verbose:
            print(f"Making cluster data for {nwb_file_name}, sort group {sort_group_id}")

        # Get key from CuratedSpikeSorting since will need all fields (some of which were not defined
        # by user, e.g. team_name) to populate other tables
        k = {**key, **{"sort_group_id": sort_group_id}}
        table_entry = (CuratedSpikeSorting & k)
        if len(table_entry) == 0 and tolerate_nonexisting:
            print(f"No entry in CuratedSpikeSorting for key {k} and tolerate_nonexisting is True. Continuing...")
            continue
        sort_group_key = table_entry.fetch1("KEY")
        data = dict()  # for cluster data

        # Make keys for getting whitened and unwhitened waveforms, and populate waveforms table as needed. Note
        # that here we use a curation id of ZERO, regardless of what curation_id was passed
        additional_params = {k: v for k, v in sort_group_key.items()
                             if k not in ["curation_id", "waveform_params_name"]}

        waveform_params_name_map = {"whitened": get_default_waveform_params_name(),
                                    "unwhitened": "default_not_whitened"}
        waveforms_key_map = {waveform_type: {**additional_params,
                                             **{"curation_id": 0,
                                                "waveform_params_name": waveform_params_name}}
                             for waveform_type, waveform_params_name in waveform_params_name_map.items()}

        # Populate waveforms tables if no entry
        for waveforms_key in waveforms_key_map.values():
            if not (Waveforms & waveforms_key):
                if verbose:
                    print(f"Populating Waveforms table with key {waveforms_key}...")
                WaveformSelection.insert1(waveforms_key, skip_duplicates=True)
                Waveforms.populate([(WaveformSelection & waveforms_key).proj()])

        # Store waveform param names
        data["waveform_param_names"] = waveform_params_name_map

        # Get workspace URL if indicated
        if get_workspace_url:
            data["workspace_url"] = SortingviewWorkspace().url(sort_group_key)

        # Get timestamps
        if verbose:
            print(f"Getting timestamps...")
        recording_path = (SpikeSortingRecording & sort_group_key).fetch1("recording_path")
        recording = si.load_extractor(recording_path)
        timestamps_raw = SpikeSortingRecording._get_recording_timestamps(recording)

        # Get total recording duration in seconds
        data["recording_duration"] = recording.get_total_duration()

        # Get spikes data

        if verbose:
            print(f"Getting spikes data...")

        # ...First get valid unit IDs, for the passed curation_id and metric restrictions.
        # If not unit IDs for given sort group, continue
        css_entry = unpack_single_element((CuratedSpikeSorting &
                                           {**sort_group_key, **{"curation_id": curation_id}}).fetch_nwb())
        if "units" in css_entry:
            units_df = css_entry["units"]
            valid_unit_ids = units_df.index

            # ...Get unit metrics
            metric_names = ["snr", "isi_violation", "nn_isolation", "nn_noise_overlap"]  # desired metrics
            for metric_name in metric_names:
                # Continue if metric name not in curated spike sorting entry
                if metric_name not in units_df:
                    continue
                data[metric_name] = units_df[metric_name].to_dict()

            # ...Get waveform extractor, which will be used to get other quantities
            for waveforms_type, waveforms_key in waveforms_key_map.items():

                we = (Waveforms & waveforms_key).load_waveforms(waveforms_key)  # waveform extractor
                data_subset = data[waveforms_type] = dict()
                data_subset["sampling_frequency"] = we.sorting.get_sampling_frequency()
                data_subset["unit_ids"] = valid_unit_ids
                data_subset["n_clusters"] = len(valid_unit_ids)
                data_subset["n_channels"] = len(we.recording.get_channel_ids())
                data_subset["waveform_window"] = np.arange(-we.nbefore, we.nafter)
                # IMPORTANT NOTE: WAVEFORMS AND SPIKE TIMES ARE SUBSAMPLED (SEEMS MAX IS AT 20000). This
                # happens in line below.
                waveform_data = {unit_id: we.get_waveforms(
                    unit_id, with_index=True, force_dense=True) for unit_id in valid_unit_ids}
                spike_samples = {unit_id: we.sorting.get_unit_spike_train(unit_id=unit_id) for unit_id in valid_unit_ids}
                # TODO: check line below
                data_subset["waveforms"] = {unit_id: np.swapaxes(wv[0], 0, 2) for unit_id, wv in waveform_data.items()}
                data_subset["waveform_indices"] = {unit_id: np.array(list(zip(*wv[1]))[0]).astype(int)
                                            for unit_id, wv in waveform_data.items()}

                # Get spike times
                data_subset["spike_times"] = {unit_id: timestamps_raw[samples[data_subset["waveform_indices"][unit_id]]]
                                       for unit_id, samples in spike_samples.items()}

                # Get average waveforms
                data_subset["average_waveforms"] = get_average_waveforms(data_subset["waveforms"])

                # Get peak channels
                data_subset["peak_channels"] = get_peak_channels(data_subset["average_waveforms"])

                # Get waveform amplitudes
                data_subset["amplitudes"] = get_waveform_amplitudes(data_subset["waveforms"])

                # Get amplitude size comparison
                data_subset["amplitude_size_comparisons"] = get_amplitude_size_comparisons(data_subset)

                # Get max cosine similarity across waveforms shifts
                if verbose:
                    print(f"Getting max cosine similarity across waveform shifts...")
                data_subset["max_cosine_similarities"] = get_max_shift_cosine_similarities(data_subset["average_waveforms"])

                # Get cosine similarity
                if verbose:
                    print(f"Getting cosine similarities...")
                data_subset["cosine_similarities"] = get_cosine_similarities(data_subset["average_waveforms"])

                # Get correlogram quantities
                if verbose:
                    print(f"Getting correlograms...")
                data_subset["correlograms"] = get_correlograms(
                    data_subset["spike_times"], max_dt=correlogram_max_dt, min_dt=correlogram_min_dt)
                data_subset["correlogram_isi_violation_ratios"] = get_correlogram_isi_violation_ratios(data_subset,
                                                              max_dt=correlogram_max_dt,
                                                              min_dt=correlogram_min_dt)
                data_subset["correlogram_asymmetries"] = get_correlogram_asymmetries(data_subset["correlograms"])
                data_subset["correlogram_asymmetry_directions"] = get_correlogram_asymmetry_directions(
                    data_subset["correlograms"])
                data_subset["correlogram_counts"] = get_correlogram_counts(data_subset["correlograms"])
                data_subset["correlogram_min_dt"] = correlogram_max_dt
                data_subset["correlogram_max_dt"] = correlogram_max_dt

                # Get amplitude overlap
                if verbose:
                    print(f"Getting amplitude overlaps...")
                data_subset["amplitude_overlaps"] = get_amplitude_overlaps(data_subset)

                # Get burst pair amplitude correlogram asymmetry metric
                data_subset["burst_pair_amplitude_timing_bools"] = get_burst_pair_amplitude_timing_bools(data_subset)

                # Get ISI violation percent for merged unit pairs
                data_subset["unit_pair_percent_isi_violations"] = get_unit_pair_percent_isi_violations(data_subset)

                # Get amplitude decrement quantities if indicated. Optional since can take a while for large datasets
                if include_amplitude_decrement_quantities:
                    # Get amplitude decrement metrics
                    if verbose:
                        print(f"Getting amplitude decrement quantities...")
                    for max_dt in [.015, .4]:
                        data_subset[f"amplitude_decrements_{max_dt}"] = get_unit_amplitude_decrements(
                            data_subset, max_dt)
                        data_subset[f"unit_merge_amplitude_decrements_{max_dt}"] = get_unit_merge_amplitude_decrements(
                            data_subset, max_dt)
                        data_subset[f"amplitude_decrement_changes_{max_dt}"] = get_amplitude_decrement_changes(
                            data_subset, max_dt)

                    # Valid lower amplitude fractions
                    if verbose:
                        print(f"Getting valid lower amplitude fractions...")
                    data_subset["valid_lower_amplitude_fractions"] = get_valid_lower_amplitude_fractions(data_subset)
                    data_subset["unit_merge_valid_lower_amplitude_fractions"] = \
                        get_unit_merge_valid_lower_amplitude_fractions(data_subset)

        # Save data
        if verbose:
            print(f"Saving {file_name_save} in {save_dir}...")
        cd_make_if_nonexistent(save_dir)
        pickle.dump(data, open(file_name_save, "wb"))  # save data


def get_curation_data_save_dir(subject_id):
    return f"/cumulus/jguidera/curation_data/{subject_id}"


def make_curation_data_wrapper(subject_ids,
                               dates,
                               sort_interval_name="raw data valid times no premaze no home",
                               preproc_params_name="franklab_tetrode_hippocampus",
                               sorter="mountainsort4",
                               sort_group_ids=None,
                               get_workspace_url=False,
                               curation_id=1,
                               ignore_invalid_sort_group_ids=False,
                               overwrite_existing=False,
                               include_amplitude_decrement_quantities=True,
                               tolerate_nonexisting=False,
                               verbose=True):
    # Make curation data

    for subject_id, date in zip(subject_ids, dates):
        # Get nwb file name
        nwb_file_name = nwbf_name_from_subject_id_date(subject_id, date)
        # Define directory to save data in
        save_dir = get_curation_data_save_dir(subject_id)
        # Make curation data
        make_curation_data(save_dir=save_dir,
                           nwb_file_name=nwb_file_name,
                           sort_interval_name=sort_interval_name,
                           sorter=sorter,
                           preproc_params_name=preproc_params_name,
                           sort_group_ids=sort_group_ids,
                           curation_id=curation_id,
                           get_workspace_url=get_workspace_url,
                           ignore_invalid_sort_group_ids=ignore_invalid_sort_group_ids,
                           overwrite_existing=overwrite_existing,
                           include_amplitude_decrement_quantities=include_amplitude_decrement_quantities,
                           tolerate_nonexisting=tolerate_nonexisting,
                           verbose=verbose)


def label_units(subject_id, date, targeted_location, sort_description, sort_interval_name, curation_id=1,
                label_col_name="label_1", curation_version="v3", verbose=True):

    # Define directory to save data in
    save_dir = f"/cumulus/jguidera/curation_data/{subject_id}/"

    # Get nwb file name
    nwb_file_name = nwbf_name_from_subject_id_date(subject_id, date)

    # Get sort group IDs
    sort_group_ids = (SortGroupTargetedLocation & {"nwb_file_name": nwb_file_name,
                                                   "targeted_location": targeted_location}).fetch("sort_group_id")

    # Get task_event params name
    preproc_params_name = return_spikesorting_params()["preproc_params_name"][targeted_location]

    # Get cluster data
    cluster_data = load_curation_data(
        save_dir=save_dir, nwb_file_name=nwb_file_name, sort_interval_name=sort_interval_name,
        preproc_params_name=preproc_params_name, curation_id=curation_id, sort_group_ids=sort_group_ids,
        verbose=verbose)

    # Get curation merge notes
    curation_merge_notes, unit_notes = get_curation_spreadsheet_notes(subject_id, date, sort_description, curation_version)

    # Label units
    # Initialize all units to accept / no merge, then fill in changes
    # note that "whitened" not in cluster data if no units found
    sort_group_unit_ids = np.concatenate([[(sort_group_id, unit_id) for unit_id in vals["whitened"]["waveforms"].keys()]
                                          for sort_group_id, vals in cluster_data["sort_groups"].items() if
                                          "whitened" in vals])
    sort_group_ids, unit_ids = list(map(np.asarray, list(zip(*sort_group_unit_ids))))

    # Initialize dictionary with labels
    unique_sort_group_ids = list(np.unique(sort_group_ids))
    # ...add in sort groups for which no units. These were not identified above
    unique_sort_group_ids += [sort_group_id for sort_group_id, vals in cluster_data["sort_groups"].items() if
                              "whitened" not in vals]
    # ...set labels to accept
    labels = {sort_group_id: {unit: ["accept"] for unit in unit_ids[sort_group_ids == sort_group_id]} for sort_group_id
              in unique_sort_group_ids}

    # Initialize dictionary with merge groups
    merge_groups = {sort_group_id: [] for sort_group_id in unique_sort_group_ids}

    # Fill in changes
    if len(curation_merge_notes) > 0:
        merge_bool = curation_merge_notes[label_col_name] == "merge"
        merge_unit_id_pairs = list(zip_df_columns(curation_merge_notes[merge_bool], ["unit_id_1", "unit_id_2"]))
        merge_sort_group_ids = curation_merge_notes[merge_bool]["sort_group_id"]
        # ...merge groups
        for merge_sort_group_id, merge_unit_id_pair in zip(merge_sort_group_ids, merge_unit_id_pairs):
            if merge_sort_group_id not in unique_sort_group_ids:  # restrict to current brain region
                continue
            merge_groups[merge_sort_group_id].append(merge_unit_id_pair)
    # ...labels
    # based on merge groups:
    if len(curation_merge_notes) > 0:
        mua_bool = np.asarray([x.strip() for x in curation_merge_notes[label_col_name]]) == "both mua"
        for sort_group_id, unit_id_1, unit_id_2 in zip_df_columns(curation_merge_notes[mua_bool],
                                                                  ["sort_group_id", "unit_id_1", "unit_id_2"]):
            if sort_group_id not in unique_sort_group_ids:  # restrict to current brain region
                continue
            for unit_id in [unit_id_1, unit_id_2]:
                labels[sort_group_id][unit_id] = ["mua"]

    # based on single unit notes:
    if len(unit_notes) > 0:
        for label in ["mua", "reject"]:
            invalid_bool = np.asarray([x.strip() for x in unit_notes[label_col_name]]) == label
            for sort_group_id, unit_id in zip_df_columns(unit_notes[invalid_bool], ["sort_group_id", "unit_id"]):
                if sort_group_id not in unique_sort_group_ids:  # restrict to current brain region
                    continue
                labels[sort_group_id][unit_id] = [label]

    if verbose:
        print("merge_groups:")
        pprint(merge_groups)
        print("labels:")
        pprint(labels)

    return merge_groups, labels


def _compute_cluster_data(func_name, data_in):
    data_out = {cluster: None for cluster in data_in.keys()}

    for cluster, data in data_in.items():
        data_out[cluster] = func_name(data)

    return data_out


def _compute_pairwise_cluster_data(func_name, data_in, nested_dict=False, kwargs=None):
    # Get inputs if not passed
    if kwargs is None:
        kwargs = {}

    # Initialize output dictionary
    data_out = {cluster_1: {cluster_2: None for cluster_2 in data_in.keys()}
                for cluster_1 in data_in.keys()}

    if nested_dict:
        for cluster_1 in data_in.keys():
            for cluster_2 in data_in[cluster_1].keys():
                data_out[cluster_1][cluster_2] = func_name(data_in[cluster_1][cluster_2], **kwargs)
    else:
        for cluster_1, data_1 in data_in.items():
            for cluster_2, data_2 in data_in.items():
                data_out[cluster_1][cluster_2] = func_name(data_1, data_2, **kwargs)

    return data_out


def _compute_average_waveform(wv):
    wv_avg = np.mean(wv, axis=2)
    return wv_avg


def _compute_peak_channel(wv_avg):
    idx = np.argmax(_compute_waveform_amplitude(wv_avg))
    return idx


def _compute_waveform_amplitude(wv):
    amp = np.max(wv, axis=1) - np.min(wv, axis=1)
    return amp


def _compute_amplitude_overlaps(data, unit_1, unit_2, bin_width=.1):
    # Find peak amplitude channel for each unit
    unit_ids = [unit_1, unit_2]
    peak_channels = np.unique([data["peak_channels"][unit_id] for unit_id in unit_ids])
    # For each unique peak amplitude channel, find overlap of normalized histograms of
    # amplitude distribution across units
    overlaps = []  # overlap across peak channels
    for peak_channel in peak_channels:  # peak channels
        # Get amplitudes for units
        unit_amplitudes = np.asarray(
            [unpack_single_element(_compute_waveform_amplitude(data["waveforms"][unit_id][[peak_channel], :, :]))
             for unit_id in unit_ids])
        # Use minimum and maximum amplitude seen across units to form histogram bins
        concatenated_unit_amplitudes = np.concatenate(unit_amplitudes)
        bin_edges = np.arange(np.min(concatenated_unit_amplitudes),
                              np.max(concatenated_unit_amplitudes) + bin_width,
                              bin_width)
        # Find overlap between normalized histograms
        overlaps.append(overlap(*[np.histogram(amplitudes, bin_edges, density=True)[0]
                                  for amplitudes in unit_amplitudes]))
    # Take average of overlaps across unit peak amplitude channels
    return np.mean(overlaps)


def _compare_amplitude_size(data, unit_1, unit_2):
    unit_1_mean = np.mean(data["amplitudes"][unit_1][data["peak_channels"][unit_1]])
    unit_2_mean = np.mean(data["amplitudes"][unit_2][data["peak_channels"][unit_1]])
    if unit_1_mean < unit_2_mean:
        return -1
    if unit_1_mean == unit_2_mean:
        return 0
    if unit_1_mean > unit_2_mean:
        return 1


def _compute_cosine_similarity(wv_avg_1, wv_avg_2):
    wv_avg_1, wv_avg_2 = (np.ravel(wv_avg) for wv_avg in (wv_avg_1, wv_avg_2))
    wv_avg_nrm_1, wv_avg_nrm_2 = (wv_avg / np.linalg.norm(wv_avg, axis=0) for wv_avg in (wv_avg_1, wv_avg_2))
    sim = np.dot(wv_avg_nrm_1, wv_avg_nrm_2)
    return sim


def _compute_max_shift_cosine_similarity(wv_avg_1, wv_avg_2):
    wv_avg_1, wv_avg_2 = (np.ravel(wv_avg) for wv_avg in (wv_avg_1, wv_avg_2))

    # Define number of shifts on either side
    shift = 10  # samples

    sims = []
    for i in np.arange(-shift, shift + 1):
        if i == 0:
            wv_avg_1_ = wv_avg_1
            wv_avg_2_ = wv_avg_2
        elif i < 0:
            i *= -1
            wv_avg_1_ = wv_avg_1[:-i]
            wv_avg_2_ = wv_avg_2[i:]
        elif i > 0:
            wv_avg_1_ = wv_avg_1[i:]
            wv_avg_2_ = wv_avg_2[:-i]
        wv_avg_nrm_1, wv_avg_nrm_2 = (wv_avg / np.linalg.norm(wv_avg, axis=0) for wv_avg in (wv_avg_1_, wv_avg_2_))
        sims.append(np.dot(wv_avg_nrm_1, wv_avg_nrm_2))

    max_sim = np.max(sims)

    return max_sim


def get_correlogram_default_params():
    return {"max_dt": .4,
            "min_dt": 0}


def _compute_correlogram(spk_times_1, spk_times_2, unit_1, unit_2, max_dt=None, min_dt=None):
    # Get inputs if not passed
    if max_dt is None:
        max_dt = get_correlogram_default_params()["max_dt"]
    if min_dt is None:
        min_dt = get_correlogram_default_params()["min_dt"]
    time_diff = np.tile(spk_times_1, (spk_times_2.size, 1)) - spk_times_2[:, np.newaxis]
    ind = np.logical_and(np.abs(time_diff) > min_dt, np.abs(time_diff) <= max_dt)
    time_diff = np.sort(time_diff[ind])
    return (unit_1, unit_2, time_diff)


def _compute_correlogram_count(time_diff, min_dt=-200 / 1000, max_dt=200 / 1000):
    return np.sum(np.logical_and(time_diff > min_dt, time_diff < max_dt))


def _compute_correlogram_asymmetry_direction(time_diff, min_dt=-200 / 1000, max_dt=200 / 1000):
    neg_count = np.sum(np.logical_and(time_diff > min_dt, time_diff < 0))
    pos_count = np.sum(np.logical_and(time_diff > 0, time_diff < max_dt))
    if neg_count > pos_count:
        return -1
    if neg_count == pos_count:
        return 0
    if pos_count > neg_count:
        return 1


def _compute_correlogram_asymmetry(time_diff, min_dt=-200 / 1000, max_dt=200 / 1000):
    zero_count = np.sum(time_diff == 0)
    neg_count = np.sum(np.logical_and(time_diff > min_dt, time_diff < 0))
    pos_count = np.sum(np.logical_and(time_diff > 0, time_diff < max_dt))
    asym = (np.max([neg_count, pos_count]) + zero_count / 2) / (zero_count + neg_count + pos_count)
    return asym


def percent_isi_violations(spike_train, isi_threshold):
    isis = np.diff(spike_train)
    num_isi_violations = np.sum(isis < isi_threshold)
    return 100*num_isi_violations/len(isis)


def _compute_correlogram_isi_violation_ratio(correlogram, correlogram_window_width, isi_threshold=None):
    # Get inputs if not passed
    if isi_threshold is None:
        isi_threshold = .0015
    # Find violations in correlogram
    invalid_bool = abs(correlogram) < isi_threshold
    # Compute fraction of correlogram that is violations
    correlogram_isi_violation = np.sum(invalid_bool) / len(invalid_bool)
    # Calculate expected violation ratio if correlogram uniform
    uniform_violation = (isi_threshold * 2) * correlogram_window_width
    # Return ratio of actual violation ratio to ratio expected if uniform correlogram
    return correlogram_isi_violation / uniform_violation


def _burst_pair_amplitude_timing_bool(data, unit_1, unit_2):
    return data["amplitude_size_comparisons"][unit_1][unit_2]*data["correlogram_asymmetry_directions"][unit_1][unit_2] < 0


# Amplitude decrement
def _channel_amplitudes(data, unit_id, channel):
    return unpack_single_element(_compute_waveform_amplitude(data["waveforms"][unit_id][[channel], :, :]))


def _time_diff(spike_times, min_dt, max_dt):
    time_diff = np.tile(spike_times, (spike_times.size, 1)) - spike_times[:, np.newaxis]
    ind = np.logical_and(np.abs(time_diff) > min_dt, np.abs(time_diff) <= max_dt)
    return time_diff, ind


def _compute_amplitude_decrement(spike_times, amplitudes, max_dt=None):
    # Get inputs if not passed
    if max_dt is None:
        max_dt = .015
    time_diff, ind = _time_diff(spike_times, 0, max_dt)
    amplitude_diff = np.tile(amplitudes, (amplitudes.size, 1)) - amplitudes[:, np.newaxis]
    valid_time_diff = time_diff[ind]
    valid_amplitude_diff = amplitude_diff[ind]
    # Return nan if fewer than two valid samples, since in this case cannot calculate correlation
    if len(valid_time_diff) < 2:
        return np.nan
    return sp.stats.pearsonr(valid_time_diff, valid_amplitude_diff)[0]


def _compute_unit_amplitude_decrement(data, unit_id, max_dt=None):
    return _compute_amplitude_decrement(
                        spike_times=data["spike_times"][unit_id],
                        amplitudes=_channel_amplitudes(data, unit_id, data["peak_channels"][unit_id]),
                        max_dt=max_dt)


def _compute_unit_merge_amplitude_decrement(data, unit_1, unit_2, max_dt=None):
    unit_ids = [unit_1, unit_2]
    return np.mean([_compute_amplitude_decrement(
        spike_times=np.concatenate([data["spike_times"][unit_id] for unit_id in unit_ids]),
        amplitudes=np.concatenate([_channel_amplitudes(data, unit_id, data["peak_channels"][peak_channel_unit_id])
                                   for unit_id in unit_ids]),
        max_dt=max_dt)
        for peak_channel_unit_id in unit_ids])


def _compute_amplitude_decrement_change(data, unit_1, unit_2, max_dt):
    # Get average amplitude decrement across the two units
    unit_amplitude_decrement = np.mean([data[f"amplitude_decrements_{max_dt}"][unit_id]
                                          for unit_id in [unit_1, unit_2]])
    # Get amplitude decrement metric for merged case
    unit_merge_amplitude_decrement = data[f"unit_merge_amplitude_decrements_{max_dt}"][unit_1][unit_2]
    return unit_merge_amplitude_decrement - unit_amplitude_decrement


# Valid lower amplitude fraction
def _compute_valid_lower_amplitude_fraction(spike_times, amplitudes, percentile=None, valid_window=None):
    # Get inputs if not passed
    if percentile is None:
        percentile = 5
    if valid_window is None:
        valid_window = .5
    # Get data value at passed percentile
    threshold = np.percentile(amplitudes, percentile)
    # Threshold data
    below_threshold_spike_times = spike_times[amplitudes < threshold]
    above_threshold_spike_times = spike_times[amplitudes >= threshold]
    # Find fraction of lower amplitude spikes that have an upper amplitude spike within some amount of time
    below_threshold_tile = np.tile(below_threshold_spike_times, (len(above_threshold_spike_times), 1))
    above_threshold_tile = np.tile(above_threshold_spike_times, (len(below_threshold_spike_times), 1)).T
    spike_time_differences = above_threshold_tile - below_threshold_tile
    valid_bool = np.sum(abs(spike_time_differences) < valid_window, axis=0) > 0
    # Return quantities
    fraction_lower_amplitude_valid = np.sum(valid_bool)/len(valid_bool)
    valid_lower_amplitude_spike_times = below_threshold_spike_times[valid_bool]
    valid_lower_amplitudes = amplitudes[amplitudes < threshold][valid_bool]
    return fraction_lower_amplitude_valid, valid_lower_amplitude_spike_times, valid_lower_amplitudes


def _compute_unit_merge_valid_lower_amplitude_fraction(data, unit_1, unit_2, percentile=None,
                                                    valid_window=None):
    unit_ids = [unit_1, unit_2]
    return np.mean([_compute_valid_lower_amplitude_fraction(
        spike_times=np.concatenate([data["spike_times"][unit_id] for unit_id in unit_ids]),
        amplitudes=np.concatenate([_channel_amplitudes(data, unit_id, data["peak_channels"][peak_channel_unit_id])
                                   for unit_id in unit_ids]), percentile=percentile, valid_window=valid_window)[0]
                    for peak_channel_unit_id in unit_ids])


# Get quantities
def get_average_waveforms(waveforms):
    return _compute_cluster_data(_compute_average_waveform, waveforms)


def get_peak_channels(average_waveforms):
    return _compute_cluster_data(_compute_peak_channel, average_waveforms)


def get_waveform_amplitudes(waveforms):
    return _compute_cluster_data(_compute_waveform_amplitude, waveforms)


def get_amplitude_overlaps(data):
    unit_ids = data["unit_ids"]
    return {unit_1: {unit_2: _compute_amplitude_overlaps(data, unit_1, unit_2) for unit_2 in unit_ids}
            for unit_1 in unit_ids}


def get_amplitude_size_comparisons(data):
    unit_ids = data["unit_ids"]
    return {unit_1: {unit_2: _compare_amplitude_size(data, unit_1, unit_2) for unit_2 in unit_ids}
            for unit_1 in unit_ids}


def get_cosine_similarities(average_waveforms):
    return _compute_pairwise_cluster_data(_compute_cosine_similarity, average_waveforms)


def get_max_shift_cosine_similarities(average_waveforms):
    return _compute_pairwise_cluster_data(_compute_max_shift_cosine_similarity, average_waveforms)


# Get correlograms using multiprocessing

def append_result(x):
    if x is not None:
        data_list.append(x)


def get_correlograms(spike_times, max_dt=None, min_dt=None):
    global data_list
    data_list = []
    pool = mp.Pool(mp.cpu_count())
    for cluster_1, data_1 in spike_times.items():
        for cluster_2, data_2 in spike_times.items():
            pool.apply_async(
                _compute_correlogram, args=(data_1, data_2, cluster_1, cluster_2, max_dt, min_dt),
                callback=append_result, error_callback=show_error)
    pool.close()
    pool.join()  # waits until all processes done before running next line

    # Store results in dictionary
    results_dict = dict()
    for (unit_1, unit_2, time_diff) in data_list:
        if unit_1 not in results_dict:
            results_dict[unit_1] = dict()
        if unit_2 not in results_dict[unit_1]:
            results_dict[unit_1][unit_2] = dict()
        results_dict[unit_1][unit_2] = time_diff

    return results_dict


def get_correlogram_counts(spike_time_differences, kwargs=None):
    return _compute_pairwise_cluster_data(
        _compute_correlogram_count, spike_time_differences, nested_dict=True, kwargs=kwargs)


def get_correlogram_asymmetries(spike_time_differences, kwargs=None):
    return _compute_pairwise_cluster_data(
        _compute_correlogram_asymmetry, spike_time_differences, nested_dict=True, kwargs=kwargs)


def get_correlogram_asymmetry_directions(spike_time_differences, kwargs=None):
    return _compute_pairwise_cluster_data(
        _compute_correlogram_asymmetry_direction, spike_time_differences, nested_dict=True, kwargs=kwargs)


def get_burst_pair_amplitude_timing_bools(data):
    unit_ids = data["unit_ids"]
    return {unit_1: {unit_2: _burst_pair_amplitude_timing_bool(data, unit_1, unit_2) for unit_2 in unit_ids}
            for unit_1 in unit_ids}


def merge_spike_times(data, unit_1, unit_2):
    return np.sort(np.concatenate((data["spike_times"][unit_1], data["spike_times"][unit_2])))


def get_unit_pair_percent_isi_violations(data, isi_threshold=.0015):
    unit_ids = data["unit_ids"]
    return {unit_1: {unit_2: percent_isi_violations(
        merge_spike_times(data, unit_1, unit_2), isi_threshold) for unit_2 in unit_ids} for unit_1 in unit_ids}


def get_correlogram_isi_violation_ratios(data, max_dt, min_dt, isi_threshold=None):
    unit_ids = data["unit_ids"]
    correlogram_window_width = max_dt*2 - min_dt*2
    return {unit_1: {unit_2: _compute_correlogram_isi_violation_ratio(
        correlogram=data["correlograms"][unit_1][unit_2], correlogram_window_width=correlogram_window_width,
        isi_threshold=isi_threshold) for unit_2 in unit_ids} for unit_1 in unit_ids}


def get_unit_amplitude_decrements(data, max_dt=None):
    return {unit_id: _compute_unit_amplitude_decrement(data, unit_id, max_dt)
            for unit_id in data["unit_ids"]}


def get_unit_merge_amplitude_decrements(data, max_dt=None):
    unit_ids = data["unit_ids"]
    return {unit_1: {unit_2: _compute_unit_merge_amplitude_decrement(data, unit_1, unit_2, max_dt)
                     for unit_2 in unit_ids} for unit_1 in unit_ids}


def get_amplitude_decrement_changes(data, max_dt):
    unit_ids = data["unit_ids"]
    return {unit_1: {unit_2: _compute_amplitude_decrement_change(data, unit_1, unit_2, max_dt)
                     for unit_2 in unit_ids} for unit_1 in unit_ids}


def get_valid_lower_amplitude_fractions(data, percentile=None, valid_window=None):
    return {unit_id: _compute_valid_lower_amplitude_fraction(
        spike_times=data["spike_times"][unit_id],
        amplitudes=data["amplitudes"][unit_id][data["peak_channels"][unit_id], :], percentile=percentile,
        valid_window=valid_window)[0] for unit_id in data["unit_ids"]}


def get_unit_merge_valid_lower_amplitude_fractions(data, percentile=None, valid_window=None):
    unit_ids = data["unit_ids"]
    return {unit_1: {unit_2: _compute_unit_merge_valid_lower_amplitude_fraction(
        data, unit_1, unit_2, percentile=percentile, valid_window=valid_window) for unit_2 in unit_ids}
        for unit_1 in unit_ids}


# Analysis
def get_merge_candidates(cluster_data, threshold_sets, sort_group_ids=None, waveform_type="whitened"):
    # Get inputs if not passed
    if sort_group_ids is None:
        sort_group_ids = list(cluster_data["sort_groups"].keys())
    # Loop through sort group IDs and apply thresholds to get merge candidates
    merge_candidates_map = {threshold_set_name: [] for threshold_set_name in threshold_sets.keys()}
    for sort_group_id in sort_group_ids:
        if waveform_type not in cluster_data["sort_groups"][sort_group_id]:
            continue
        data = cluster_data["sort_groups"][sort_group_id][waveform_type]
        # Apply threshold sets
        valid_bool_map = get_above_threshold_matrix_indices(
            cluster_data, sort_group_id, threshold_sets, waveform_type)

        for threshold_set_name, valid_bool in valid_bool_map.items():  # threshold sets
            # Find indices in array corresponding to merge candidates
            merge_candidate_idxs = list(zip(*np.where(valid_bool)))
            # Convert merge candidate indices in array to unit IDs
            merge_candidates_map[threshold_set_name] += [tuple([sort_group_id] +
                                                               list(np.asarray(data["unit_ids"])[np.asarray(idxs)]))
                                for idxs in merge_candidate_idxs]
    return merge_candidates_map


def merge_plots_wrapper(cluster_data,
                        threshold_sets,
                        waveform_type="whitened",
                        fig_scale=.8,
                        subplot_width=4,
                        subplot_height=5,
                        plot_merge_candidates=None,
                        show_merge_matrices=False):

    for sort_group_id, sg_data in cluster_data["sort_groups"].items():
        # Get data for waveform type
        if waveform_type not in sg_data:
            continue
        data = sg_data[waveform_type]

        # Apply threshold sets
        valid_bool_map = get_above_threshold_matrix_indices(
            cluster_data, sort_group_id, threshold_sets, waveform_type)

        # Continue of no passed merged candidates have current sort group
        if plot_merge_candidates is not None:
            if sort_group_id not in [x[0] for x in plot_merge_candidates]:
                continue

        # Plot matrices with pairwise metrics relevant for merging if indicated
        if show_merge_matrices:
            plot_merge_matrices(cluster_data, sort_group_id, valid_bool_map,  threshold_sets, waveform_type,
                                fig_scale=fig_scale)

        # For threshold sets, plot metrics for merge candidates
        for threshold_name, valid_bool in valid_bool_map.items():  # threshold sets
            # Find indices in array corresponding to merge candidates
            merge_candidate_idxs = list(zip(*np.where(valid_bool)))
            # Convert merge candidate indices in array to unit IDs
            merge_candidates = [tuple(np.asarray(data["unit_ids"])[np.asarray(idxs)])
                                for idxs in merge_candidate_idxs]
            # Loop through merge candidates and plot metrics
            for unit_1, unit_2 in merge_candidates:  # units

                if plot_merge_candidates is not None:
                    if (sort_group_id, unit_1, unit_2) not in plot_merge_candidates:
                        continue

                    merge_plot(cluster_data, sort_group_id, unit_1, unit_2, waveform_type, threshold_name,
                               subplot_width, subplot_height)


def merge_plot(cluster_data, sort_group_id, unit_1, unit_2, waveform_type="whitened", threshold_name=None, subplot_width=4,
               subplot_height=3):

    # Define plot parameters
    num_rows = 2
    num_columns = 4
    gridspec_kw = {"width_ratios": [1, 1, 4, 4]}
    unit_colors = ["crimson", "#2196F3"]

    # Initialize figure
    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(num_columns * subplot_width, num_rows * subplot_height),
        gridspec_kw=gridspec_kw)

    # Get data for waveform type and sort group ID
    data = cluster_data["sort_groups"][sort_group_id][waveform_type]

    # Use peak channel of first unit to display data from both units
    peak_ch = data["peak_channels"][unit_1]

    # Leftmost subplots: average waveforms
    for unit_id_idx, unit_id in enumerate([unit_1, unit_2]):
        title = f"{sort_group_id}_{unit_id}"
        if unit_id_idx == 1:
            cosine_similarity = data["cosine_similarities"][unit_1][unit_2]
            title = f"cosine similarity: {cosine_similarity: .2f}\n{title}"
        gs = axes[0, unit_id_idx].get_gridspec()
        # Remove underlying axis
        for row_num in np.arange(0, num_rows):
            axes[row_num, unit_id_idx].remove()
        ax = fig.add_subplot(gs[:, unit_id_idx])
        plot_average_waveforms(cluster_data, sort_group_id, unit_id, title=title, color=unit_colors[unit_id_idx], ax=ax)

    # Second subplot: amplitude distributions
    ax = axes[0, 2]
    for unit_id_idx, unit_id in enumerate([unit_1, unit_2]):
        title = f"amplitude overlap: {data['amplitude_overlaps'][unit_1][unit_2]: .3f}"
        plot_amplitude_distribution(cluster_data, sort_group_id, unit_id,
                                    ch=peak_ch,
                                    max_amplitude=None,
                                    amplitude_bin_size=2,
                                    histtype="step",
                                    density=True,
                                    label=f"{sort_group_id}_{unit_id}",
                                    color=unit_colors[unit_id_idx],
                                    title=title,
                                    ax=ax)

    # Third subplot: correlograms
    ax = axes[0, 3]
    plot_correlogram(cluster_data, sort_group_id,
                     unit_1, unit_2,
                     max_time_difference=200 / 1000,
                     color="gray",
                     ax=ax)

    # Fourth subplot: amplitudes over time
    # Use peak channel from first unit to plot amplitudes for both units
    gs = axes[1, 2].get_gridspec()
    # Remove underlying axes
    for ax in axes[1, 2:]:
        ax.remove()
    ax = fig.add_subplot(gs[1, 2:])
    # Plot amplitudes over time for each unit
    for unit_id_idx, unit_id in enumerate([unit_1, unit_2]):
        plot_amplitude(cluster_data, sort_group_id, unit_id, peak_ch,
                       color=unit_colors[unit_id_idx], ax=ax)

    # Global title
    threshold_name_text = ""
    if threshold_name is not None:
        threshold_name_text = f"\n{threshold_name}"
    fig.suptitle(f"{sort_group_id}_{unit_1} vs. {sort_group_id}_{unit_2}{threshold_name_text}",
                 fontsize=20)
    fig.tight_layout()


# VISUALIZATION

def plot_amplitude(cluster_data, sort_group_id, unit_id, ch=None, waveforms_type="whitened",  color="black", ax=None):
    # Get inputs if not passed
    if ax is None:
        _, ax = plt.subplots()
    if ch is None:
        ch = cluster_data["sort_groups"][sort_group_id][waveforms_type]["peak_channels"][unit_id]
    # Get whitened or unwhitened waveforms as indicated
    data_subset = cluster_data["sort_groups"][sort_group_id][waveforms_type]
    # Plot amplitudes over time
    ax.scatter(data_subset["spike_times"][unit_id], data_subset["amplitudes"][unit_id][ch, :], s=1, color=color)


def plot_amplitudes(nwb_file_name, cluster_data, target_sort_group_unit_ids, waveforms_type="unwhitened",
                    ylims=np.asarray([None])):

    # Define colors for units
    sort_group_colors = ["gray", "blue", "red", "orange", "green", "purple"] * 10

    # Key for looking up interval lists to mark epoch boundaries
    key = {"nwb_file_name": nwb_file_name}

    plot_counter = 0
    fig, axes = get_fig_axes(num_rows=len(target_sort_group_unit_ids),
                             num_columns=len(ylims), subplot_width=10, subplot_height=2)
    for sort_group_id, unit_id in target_sort_group_unit_ids:
        if waveforms_type not in cluster_data["sort_groups"][sort_group_id]:
            continue
        data_subset = cluster_data["sort_groups"][sort_group_id][waveforms_type]
        peak_ch = data_subset["peak_channels"][unit_id]
        for ylim_idx, ylim in enumerate(ylims):
            ax = get_ax_for_layout(axes, plot_counter)
            plot_counter += 1
            plot_amplitude(cluster_data, sort_group_id, unit_id, peak_ch, waveforms_type=waveforms_type,
                           color=sort_group_colors[sort_group_id], ax=ax)

            if ylim is not None:  # for some reason, seems if pass None, restriction still applied in some weird way
                ax.set_ylim(ylim)

            # Mark epoch boundaries
            epoch_type_table_names = [RunEpoch, SleepEpoch]
            epoch_type_colors = ["blue", "red"]
            span_data = ax.get_ylim()
            for table_name, color in zip(epoch_type_table_names, epoch_type_colors):
                for epoch in table_name().get_epochs(nwb_file_name):
                    key.update(
                        {"interval_list_name": EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch)})
                    valid_times = (IntervalList & key).fetch1("valid_times")
                    for epoch_bound in valid_times:
                        plot_spanning_line(ax=ax, span_axis="y", span_data=span_data, constant_val=epoch_bound,
                                           color=color, linewidth=2)

            ax.set_title(f"{sort_group_id}_{unit_id}")


def _matrix_grid(ax, n_clusters, fig_scale):
    for ndx in range(n_clusters - 1):
        ax.axvline(x=ndx + 1, color="#FFFFFF", linewidth=fig_scale * 0.5)
        ax.axhline(y=ndx + 1, color="#FFFFFF", linewidth=fig_scale * 0.5)


def _format_matrix_ax(ax, ticks, ticklabels, fig_scale, title):
    format_ax(ax=ax, xticks=ticks, yticks=ticks, xticklabels=ticklabels, yticklabels=ticklabels,
              fontsize=fig_scale * 10, title=title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=fig_scale * 12)


def get_above_threshold_matrix_indices(cluster_data,
                                       sort_group_id,
                                       threshold_sets,
                                       waveform_type):
    return {threshold_name: _apply_metric_matrix_thresholds(cluster_data, sort_group_id, threshold_set.thresholds,
                                                            waveform_type)
              for threshold_name, threshold_set in threshold_sets.items()}


def _highlight_matrix_indices(valid_bool_map, threshold_sets, ax):
    for threshold_name, valid_bool in valid_bool_map.items():
        ii, jj = np.where(valid_bool)
        for ndx in range(np.sum(valid_bool)):
            ax.add_patch(matplotlib.patches.Rectangle((jj[ndx], ii[ndx]), 1, 1,
                                                      edgecolor=threshold_sets[threshold_name].color,
                                                      fill=False,
                                                      lw=threshold_sets[threshold_name].lw,
                                                      zorder=2 * len(valid_bool) ** 2,
                                                      clip_on=False))


def _get_metric_matrix(cluster_data, sort_group_id, metric_name, waveform_type="whitened",
                       apply_upper_diagonal_mask=False,
                       mask_value=np.nan):
    data = cluster_data["sort_groups"][sort_group_id][waveform_type]
    metric_dict = data[metric_name]
    index = metric_dict.keys()
    matrix = np.array([[metric_dict[ii][jj] for jj in index]
                       for ii in index]).astype(float)  # float so can mask with nan
    # Mask upper diagonal if indicated
    if apply_upper_diagonal_mask:
        matrix = mask_upper_diagonal(matrix, mask_value=mask_value)
    return pd.DataFrame(matrix, index=index, columns=index)


def _apply_metric_matrix_thresholds(cluster_data, sort_group_id, threshold_objs, waveform_type):
    return np.prod(
        [threshold_obj.threshold_direction(
            _get_metric_matrix(cluster_data, sort_group_id, threshold_obj.metric_name, waveform_type,
                               apply_upper_diagonal_mask=True,
                               mask_value=np.nan),
            threshold_obj.threshold_value)
         for threshold_obj in threshold_objs], axis=0)


def plot_amplitude_overlap_matrix(cluster_data, sort_group_id, waveform_type="whitened", fig_scale=1, fig_ax_list=None,
                                  plot_color_bar=True):
    data = cluster_data["sort_groups"][sort_group_id][waveform_type]
    n_clusters = data["n_clusters"]

    # Get amplitude overlap matrix
    ao_matrix = _get_metric_matrix(cluster_data, sort_group_id, "amplitude_overlaps",
                       apply_upper_diagonal_mask=True,
                       mask_value=0)

    # Unpack figure and axis if passed
    if fig_ax_list is not None:
        fig, ax = fig_ax_list
    # Otherwise make these
    else:
        fig = plt.figure(figsize=(n_clusters / 2, n_clusters / 2) * fig_scale)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
    pcm = plt.pcolormesh(ao_matrix, cmap="inferno", vmin=0, vmax=1)
    _matrix_grid(ax, n_clusters, fig_scale)
    label = "".join((cluster_data["nwb_file_name"],
                     "\n",
                     "interval: ",
                     cluster_data["sort_interval_name"],
                     "\n",
                     f"sort group: {sort_group_id}",
                     "\n"
                     "amplitude overlap"))
    _format_matrix_ax(ax, ticks=np.arange(0.5, n_clusters + 0.5),
                      ticklabels=ao_matrix.index,
                      fig_scale=fig_scale,
                      title=label)
    # Color bar
    if plot_color_bar:
        add_colorbar(pcm, fig, ax)

    return fig, ax


def plot_merge_matrices(
        cluster_data, sort_group_id, valid_bool_map, threshold_sets, waveform_type="whitened", fig_scale=1,
        plot_color_bar=True):

    data = cluster_data["sort_groups"][sort_group_id][waveform_type]
    n_clusters = data["n_clusters"]

    # Get cosine similarity matrix
    cs_matrix = _get_metric_matrix(
        cluster_data, sort_group_id, "cosine_similarities", apply_upper_diagonal_mask=True, mask_value=0)

    # Get correlogram asymmetry matrix
    ca_matrix = _get_metric_matrix(
        cluster_data, sort_group_id, "correlogram_asymmetries", apply_upper_diagonal_mask=True, mask_value=0)

    # Initialize figure
    num_columns = 3
    fig = plt.figure(figsize=(fig_scale * (num_columns * n_clusters / 2 + 2),
                              fig_scale * (n_clusters / 2)))
    width_ratios = [n_clusters / 2]*3
    gs = fig.add_gridspec(1, num_columns, wspace=0.2, width_ratios=width_ratios)

    # Ticks across plots
    ticks = np.arange(0.5, n_clusters + 0.5)

    # First subplot: cosine similarity
    ax = fig.add_subplot(gs[0])
    pcm = plt.pcolormesh(cs_matrix, cmap="inferno", vmin=0, vmax=1)
    # Grid
    _matrix_grid(ax, n_clusters, fig_scale)
    # Highlight indices crossing metric thresholds
    _highlight_matrix_indices(valid_bool_map, threshold_sets, ax)
    # Axis
    _format_matrix_ax(ax=ax, ticks=ticks, ticklabels=cs_matrix.index, fig_scale=fig_scale, title="cosine similarity")
    # Color bar
    if plot_color_bar:
        add_colorbar(pcm, fig, ax)

    # Second subplot: correlogram asymmetry
    ax = fig.add_subplot(gs[1])
    pcm = plt.pcolormesh(ca_matrix, cmap="inferno", vmin=0.5, vmax=1)
    # Grid
    _matrix_grid(ax, n_clusters, fig_scale)
    # Highlight indices crossing metric thresholds
    _highlight_matrix_indices(valid_bool_map, threshold_sets, ax)
    # Axis
    _format_matrix_ax(ax=ax, ticks=ticks, ticklabels=ca_matrix.index,
                      fig_scale=fig_scale, title="correlogram asymmetry")
    # Color bar
    if plot_color_bar:
        add_colorbar(pcm, fig, ax)

    # Third subplot: amplitude overlap
    ax = fig.add_subplot(gs[2])
    fig, ax = plot_amplitude_overlap_matrix(
        cluster_data, sort_group_id, fig_scale=fig_scale, fig_ax_list=[fig, ax], plot_color_bar=plot_color_bar)
    # Highlight indices crossing metric thresholds
    _highlight_matrix_indices(valid_bool_map, threshold_sets, ax)
    plt.show()


def plot_average_waveforms(cluster_data, sort_group_id, unit_id, waveform_type="whitened", color=None,
                           title=None, ax=None):
    # Get inputs if not passed
    if ax is None:
        _, ax = plt.subplots()
    if title is None:
        title = f"{sort_group_id}_{unit_id}"
    if color is None:
        color = "#2196F3"

    # Define amplitude range and spacing between traces
    if waveform_type == "whitened":
        amplitude_range = 80
        trace_offset = 40
    elif waveform_type == "unwhitened":
        amplitude_range = 1000
        trace_offset = 200

    data = cluster_data["sort_groups"][sort_group_id][waveform_type]
    wv_avg = data["average_waveforms"][unit_id]

    n_channels = np.shape(wv_avg)[0]
    n_points = data["waveform_window"].size
    ax.axvline(x=n_points / 2, color="#9E9E9E", linewidth=1)

    offset = np.tile(-np.arange(n_channels) * trace_offset, (n_points, 1))
    trace = wv_avg.T + offset
    peak_ind = np.full(n_channels, False)
    peak_ind[data["peak_channels"][unit_id]] = True

    ax.plot(trace[:, ~peak_ind], color=color, linewidth=1, clip_on=False)
    ax.plot(trace[:, peak_ind], color=color, linewidth=2.5, clip_on=False)
    ax.set_xlim([0, n_points])
    ax.set_ylim([-2 * amplitude_range / 3 - (n_channels - 1) * trace_offset,
                 amplitude_range / 3])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title, fontsize=12)


def plot_average_waveforms_wrapper(cluster_data, sort_group_unit_ids=None, subplot_height=3):
    # Get inputs if not passed
    if sort_group_unit_ids is None:
        sort_group_unit_ids = np.concatenate(
            [[(sort_group_id, unit_id) for unit_id in cluster_data["sort_groups"][sort_group_id]["whitened"]["average_waveforms"].keys()]
             for sort_group_id in cluster_data["sort_groups"]])

    # Plot waveforms for multiple units
    num_columns = 10
    subplot_width = 1

    num_units = len(sort_group_unit_ids)
    # Initialize plot
    fig, axes = get_fig_axes(num_rows=None, num_columns=num_columns, num_subplots=num_units, sharex=False, sharey=False,
                     subplot_width=subplot_width, subplot_height=subplot_height)
    # Plot
    plot_counter = 0
    for sort_group_id, unit_id in sort_group_unit_ids:
        ax = get_ax_for_layout(axes, plot_counter)
        plot_counter += 1
        # Different color if merged unit
        color = None
        if "merged_units" in cluster_data["sort_groups"][sort_group_id]["whitened"]:
            if unit_id in cluster_data["sort_groups"][sort_group_id]["whitened"]["merged_units"]:
                color = "red"
        plot_average_waveforms(cluster_data, sort_group_id, unit_id, color=color, ax=ax)


def plot_amplitude_distribution(cluster_data, sort_group_id, unit_id,
                                waveform_type="whitened",
                                ch=None,
                                max_amplitude=None,
                                amplitude_bin_size=2,
                                density=False,
                                histtype=None,
                                color="#2196F3",
                                label=None,
                                title=None,
                                remove_axes=False,
                                ax=None):
    data = cluster_data["sort_groups"][sort_group_id][waveform_type]
    # Define channel if not passed
    if ch is None:
       ch = data["peak_channels"][unit_id]
    amp = data["amplitudes"][unit_id][ch, :]
    # Get inputs if not passed
    if ax is None:
        _, ax = plt.subplots()
    if max_amplitude is None:
        max_amplitude = np.max(amp)
    if title is None:
        title = f"{sort_group_id}_{unit_id}"
    bin_edges = np.arange(0, max_amplitude + amplitude_bin_size, amplitude_bin_size)
    ax.hist(amp, bin_edges, density=density, histtype=histtype, color=color, label=label, linewidth=4, alpha=.9)
    # ax.set_xlim([bin_edges[0], bin_edges[-1]])
    if remove_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.set_title(title, fontsize=12)


def plot_amplitude_distributions(cluster_data, max_amplitude=50, amplitude_bin_size=2):
    for sort_group_id, data in cluster_data["sort_groups"].items():
        n_clusters = data["n_clusters"]

        fig = plt.figure(figsize=(n_clusters + 2, 1))
        width_ratios = np.ones(n_clusters + 1)
        width_ratios[0] = 2
        gs = fig.add_gridspec(1, n_clusters + 1, wspace=0.1, width_ratios=width_ratios)
        ax = fig.add_subplot(gs[0])
        label = "".join((cluster_data["nwb_file_name"],
                         "\n",
                         "interval: ",
                         cluster_data["sort_interval_name"],
                         "\n",
                         f"sort group: {sort_group_id}"))
        ax.text(-0.3, 0.3, label, multialignment="center", fontsize=12)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for ndx, (unit_id, time_diff) in enumerate(data["amplitudes"].items()):
            ax = fig.add_subplot(gs[ndx + 1])
            plot_amplitude_distribution(cluster_data, sort_group_id, unit_id, max_amplitude=max_amplitude,
                                        amplitude_bin_size=amplitude_bin_size, ax=ax)
        plt.show()


def plot_correlogram(cluster_data, sort_group_id,
                     cluster_1, cluster_2, waveform_type="whitened",
                     max_time_difference=20 / 1000, time_bin_size=1 / 1000,
                     color="#2196F3", remove_axes=False, ax=None):
    # Get inputs if not passed
    if ax is None:
        _, ax = plt.subplots()
    data = cluster_data["sort_groups"][sort_group_id][waveform_type]
    time_diff = data["correlograms"][cluster_1][cluster_2]
    bin_edges = np.arange(-max_time_difference, max_time_difference + time_bin_size, time_bin_size)
    ax.hist(time_diff, bin_edges, color=color)
    ax.set_xlim([bin_edges[0], bin_edges[-1]])
    ax.set_ylim([0, np.max(np.histogram(time_diff, bin_edges)[0])])
    if remove_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    burst_pair_amplitude_timing_bool = data["burst_pair_amplitude_timing_bools"][cluster_1][cluster_2]
    correlogram_asymmetry = data["correlogram_asymmetries"][cluster_1][cluster_2]
    isi_violation = data["unit_pair_percent_isi_violations"][cluster_1][cluster_2]
    correlogram_count = int(data["correlogram_counts"][cluster_1][cluster_2])
    ax.set_title(f"{sort_group_id}_{cluster_1} vs {sort_group_id}_{cluster_2}"
                 f"\ncount: {correlogram_count: .2f}"
                 f"\nasymmetry: {correlogram_asymmetry: .2f}"
                 f"\nISI violation: {isi_violation:.5f}"
                 f"\nburst_pair_amplitude_timing_bool: {burst_pair_amplitude_timing_bool}",
                 fontsize=12)


def plot_autocorrelograms(cluster_data, max_time_difference=20 / 1000, time_bin_size=1 / 1000):
    for sort_group_id, data in cluster_data["sort_groups"].items():
        n_clusters = data["n_clusters"]

        fig = plt.figure(figsize=(n_clusters + 2, 1))
        width_ratios = np.ones(n_clusters + 1)
        width_ratios[0] = 2
        gs = fig.add_gridspec(1, n_clusters + 1, wspace=0.1, width_ratios=width_ratios)
        ax = fig.add_subplot(gs[0])
        label = "".join((cluster_data["nwb_file_name"],
                         "\n",
                         "interval: ",
                         cluster_data["sort_interval_name"],
                         "\n",
                         f"sort group: {sort_group_id}"))
        ax.text(-0.3, 0.3, label, multialignment="center", fontsize=12)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for ndx, cluster_num in enumerate(data["correlograms"].keys()):
            ax = fig.add_subplot(gs[ndx + 1])
            plot_correlogram(cluster_data, sort_group_id,
                             cluster_1=cluster_num, cluster_2=cluster_num, max_time_difference=max_time_difference,
                             time_bin_size=time_bin_size, ax=ax)
        plt.show()


def plot_cosine_similarity_distribution(cluster_data, fig_scale=1):

    n_clusters = np.array([data["n_clusters"] for data in cluster_data["sort_groups"].values()])
    ind = [np.triu(np.full((count, count), True), 1) for count in n_clusters]
    cs_list = [np.array([[data["cosine_similarities"][ii][jj]
                          for jj in data["cosine_similarities"].keys()]
                         for ii in data["cosine_similarities"].keys()])
               for data in cluster_data["sort_groups"].values()]
    cs = [list(np.ravel(cs[ind[ndx]])[:]) for ndx, cs in enumerate(cs_list)]
    cs = np.array(list(itertools.chain(*cs)))

    fig = plt.figure(figsize=(fig_scale * 12, fig_scale * 3))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    label = "".join((cluster_data["nwb_file_name"],
                     "\n",
                     "interval: ",
                     cluster_data["sort_interval_name"]))

    plt.hist(cs, 240, color="#2196F3")
    ax.axvline(x=0, color="#424242", linewidth=fig_scale * 1.)
    ax.set_xlim([-1, 1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([-1, -0.5, 0, 0.5, 1], fontsize=fig_scale * 10)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels([0, 20, 40, 60], fontsize=fig_scale * 10)
    ax.set_xlabel("Cosine Similarity", fontsize=fig_scale * 12)
    ax.set_ylabel("Count", fontsize=fig_scale * 12)
    ax.set_title(label, fontsize=fig_scale * 12)

    plt.show()


def plot_correlogram_asymmetry_distribution(cluster_data, fig_scale=1):
    n_clusters = np.array([data["n_clusters"] for data in cluster_data["sort_groups"].values()])
    ind = [np.triu(np.full((count, count), True), 1) for count in n_clusters]
    ca_list = [np.array([[data["correlogram_asymmetries"][ii][jj]
                          for jj in data["correlogram_asymmetries"].keys()]
                         for ii in data["correlogram_asymmetries"].keys()])
               for data in cluster_data["sort_groups"].values()]
    ca = [list(np.ravel(ca[ind[ndx]])[:]) for ndx, ca in enumerate(ca_list)]
    ca = np.array(list(itertools.chain(*ca)))

    fig = plt.figure(figsize=(fig_scale * 12, fig_scale * 3))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    label = "".join((cluster_data["nwb_file_name"],
                     "\n",
                     "interval: ",
                     cluster_data["sort_interval_name"]))

    plt.hist(ca, 240, color="#2196F3")
    ax.set_yscale("log")
    ax.set_xlim([0.5, 1])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels([0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=fig_scale * 10)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels([1, 10, 100, 1000], fontsize=fig_scale * 10)
    ax.set_xlabel("Correlogram Asymmetry", fontsize=fig_scale * 12)
    ax.set_ylabel("Count", fontsize=fig_scale * 12)
    ax.set_title(label, fontsize=fig_scale * 12)

    plt.show()
