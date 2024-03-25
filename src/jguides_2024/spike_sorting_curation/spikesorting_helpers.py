import os as os
from pathlib import Path

from spyglass.spikesorting.v0.spikesorting_sorting import SpikeSorterParameters


def set_spikesorting_directories(base_dir):
    base_dir = Path(base_dir)  # change this to your desired directory
    if (base_dir).exists() is False:
        os.mkdir(base_dir)
    raw_dir = base_dir / 'raw'
    if (raw_dir).exists() is False:
        os.mkdir(raw_dir)
    analysis_dir = base_dir / 'analysis'
    if (analysis_dir).exists() is False:
        os.mkdir(analysis_dir)
    kachery_storage_dir = base_dir / 'kachery-storage'
    if (kachery_storage_dir).exists() is False:
        os.mkdir(kachery_storage_dir)
    recording_dir = base_dir / 'recording'
    if (recording_dir).exists() is False:
        os.mkdir(recording_dir)
    sorting_dir = base_dir / 'sorting'
    if (sorting_dir).exists() is False:
        os.mkdir(sorting_dir)
    waveforms_dir = base_dir / 'waveforms'
    if (waveforms_dir).exists() is False:
        os.mkdir(waveforms_dir)
    tmp_dir = base_dir / 'tmp/spyglass'
    if (tmp_dir).exists() is False:
        os.mkdir(tmp_dir)

    return (base_dir, raw_dir, analysis_dir, kachery_storage_dir,
            recording_dir, sorting_dir, waveforms_dir, tmp_dir)


def add_franklab_sorter_params():
    sorter = "mountainsort4"
    # Hippocampus tetrode default
    sorter_params_name = "franklab_tetrode_hippocampus_30KHz"
    sorter_params = {'detect_sign': -1,
                      'adjacency_radius': -1,
                      'freq_min': 600,
                      'freq_max': 6000,
                      'filter': False,
                      'whiten': True,
                      'num_workers': 1,
                      'clip_size': 40,
                      'detect_threshold': 3,
                      'detect_interval': 10}
    SpikeSorterParameters.insert1({"sorter": sorter,
                                    "sorter_params_name": sorter_params_name,
                                    "sorter_params": sorter_params}, skip_duplicates=True)

    # Cortical probe default
    sorter_params_name = "franklab_probe_ctx_30KHz"
    sorter_params = {'detect_sign': -1,
                  'adjacency_radius': 100,
                  'freq_min': 300,
                  'freq_max': 6000,
                  'filter': False,
                  'whiten': True,
                  'num_workers': 1,
                  'clip_size': 40,
                  'detect_threshold': 3,
                  'detect_interval': 10}
    SpikeSorterParameters.insert1({"sorter": sorter,
                                    "sorter_params_name": sorter_params_name,
                                    "sorter_params": sorter_params}, skip_duplicates=True)

    sorter_params_name = "franklab_probe_ctx_30KHz_115rad"
    sorter_params = {'detect_sign': -1,
                  'adjacency_radius': 115,
                  'freq_min': 300,
                  'freq_max': 6000,
                  'filter': False,
                  'whiten': True,
                  'num_workers': 1,
                  'clip_size': 40,
                  'detect_threshold': 3,
                  'detect_interval': 10}
    SpikeSorterParameters.insert1({"sorter": sorter,
                                    "sorter_params_name": sorter_params_name,
                                    "sorter_params": sorter_params}, skip_duplicates=True)


# Get channels with peak waveform, with code used for CuratedSpikeSorting
def get_peak_ch_map(nwb_file_name, sort_group_id, sort_interval_name="raw data valid times no premaze no home",
                    curation_id=2, peak_sign="neg"):

    # Define key for querying table
    key = {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id, "sort_interval_name": sort_interval_name,
           "curation_id": curation_id}
    # Get waveform extractor
    from spyglass.spikesorting.v0.spikesorting_curation import _get_peak_channel, Waveforms
    waveform_extractor = Waveforms().load_waveforms(key)
    return _get_peak_channel(waveform_extractor, peak_sign)