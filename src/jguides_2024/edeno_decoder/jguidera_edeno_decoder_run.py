import copy
import multiprocessing as mp
import os
import pickle
import time
from collections import namedtuple

import cupy as cp
import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd
import xarray as xr
from replay_trajectory_classification import ClusterlessClassifier, SortedSpikesClassifier
from spyglass.common import IntervalList, IntervalLinearizedPosition
from spyglass.decoding.v0.sorted_spikes import (
    SortedSpikesIndicatorSelection, SortedSpikesIndicator, SortedSpikesClassifierParameters)
from spyglass.decoding.v0.clusterless import ClusterlessClassifierParameters
from spyglass.spikesorting.v0.spikesorting_curation import CuratedSpikeSorting

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_reliability_paper_nwb_file_names, \
    get_subject_id
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SecKeyParamsBase, SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert, get_default_param, \
    split_curation_name, make_param_name, insert_analysis_table_entry
from src.jguides_2024.edeno_decoder.jguidera_edeno_decoder_helpers import get_valid_decode_variable_names, \
    StackedEdgeTrackGraph, EDPathGroups, max_posterior_position, get_valid_turn_zone_decode_variable_names
from src.jguides_2024.metadata.jguidera_brain_region import CurationSet, BrainRegionCohort
from src.jguides_2024.metadata.jguidera_epoch import EpochsDescription, TrainTestEpoch, EpochsDescriptions
from src.jguides_2024.position_and_maze.jguidera_maze import get_path_junction_abbreviation, MazePathWell
from src.jguides_2024.position_and_maze.jguidera_ppt import Ppt
from src.jguides_2024.spikes.jguidera_unit import EpsUnitsParams, BrainRegionUnits, BrainRegionUnitsParams
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellArrivalTrials, DioWellDDTrials, \
    DioWellArrivalTrialsSub
from src.jguides_2024.time_and_trials.jguidera_interval import EpochIntervalListName
from src.jguides_2024.utils.cross_validation_helpers import CrossValidate
from src.jguides_2024.utils.df_helpers import zip_df_columns, match_two_dfs, df_pop, df_from_data_list
from src.jguides_2024.utils.dict_helpers import merge_dicts
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.plot_helpers import path_name_to_plot_string, format_ax
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool, event_times_in_intervals
from src.jguides_2024.utils.print_helpers import optional_print
from src.jguides_2024.utils.save_load_helpers import pickle_file
from src.jguides_2024.utils.set_helpers import check_membership, check_set_equality
from src.jguides_2024.utils.vector_helpers import unpack_single_element, check_all_unique, vector_midpoints

schema = dj.schema("jguidera_edeno_decoder_run")

# These imports are called with eval or used in table definitions (do not remove):
TrainTestEpoch
nd


# Unpickles file, for use with multiprocessing
def _unpickle(file_name):

    return pickle.load(open(file_name, "rb"))


@schema
class EDDecodeVariableParams(dj.Manual):
    definition = """
    # Decode variables parameters for EdenoDecodeParams
    decode_variable_param_name : varchar(40)
    ---
    decode_variable_params : blob
    """

    def insert_defaults(self, **kwargs):
        param_sets = []

        decode_variable_param_name = "ppt_default"
        param_sets.append(
            {"decode_variable_param_name": decode_variable_param_name, "decode_variable_params": {
             "decode_variable": "ppt", "only_two_junction_paths": False,
             "position_info_param_name": "default_decoding", "linearization_param_name": "default",
             "track_graph_idx": 1}})

        decode_variable_param_name = "wa_stay"
        param_sets.append(
            {"decode_variable_param_name": decode_variable_param_name, "decode_variable_params": {
             "decode_variable": "wa", "only_two_junction_paths": False,
             "position_info_param_name": "default_decoding", "linearization_param_name": "default",
             "track_graph_idx": 1, "trial_restriction": "stay_trial"}})

        for param_set in param_sets:
            self.insert1(param_set, skip_duplicates=True)


@schema
class EDCrossValidationParams(dj.Manual):
    definition = """
    # Cross validation parameters for EdenoDecodeParams
    cross_validation_param_name : varchar(40)
    ---
    cross_validation_params : blob
    """

    def insert_defaults(self, **kwargs):

        param_sets = [
            {"cross_validation_param_name": "looct_path_default", "cross_validation_params":
            {"cross_validation_method": "leave_one_out_condition_trials", "trials_description": "path",
             "limit_same_train_test": True}},
        ]

        for param_set in param_sets:
            self.insert1(param_set, skip_duplicates=True)


@schema
class EDAlgorithmParams(dj.Manual):
    definition = """
    # Algorithm parameters for EdenoDecodeParams
    algorithm_param_name : varchar(40)
    ---
    algorithm_params : blob
    """

    def insert_defaults(self, **kwargs):

        param_sets = [
            {"algorithm_param_name": "sorted_spikes_default", "algorithm_params": {
             "clusterless": False, "sampling_rate": 500}}]

        for param_set in param_sets:
            self.insert1(param_set, skip_duplicates=True)

    # Override method in parent class so can check key
    def insert1(self, key, **kwargs):

        required_params = ["clusterless", "sampling_rate"]
        check_membership(required_params, key["algorithm_params"].keys())
        super().insert1(key, skip_duplicates=True)


@schema
class EDComputeParams(dj.Manual):
    definition = """
    # Compute parameters for EdenoDecodeParams
    compute_param_name : varchar(40)
    ---
    compute_params : blob
    """

    def insert_defaults(self, **kwargs):

        param_sets = [
            {"compute_param_name": "gpu_default", "compute_params": {
             "use_gpu": 1, "max_tolerated_diff": .01}}]

        for param_set in param_sets:
            self.insert1(param_set, skip_duplicates=True)


@schema
class EDStorageParams(dj.Manual):
    definition = """
    # Storage parameters for EdenoDecodeParams
    storage_param_name : varchar(40)
    ---
    storage_params : blob
    """

    def insert_defaults(self, **kwargs):

        param_sets = [
            {"storage_param_name": "nimbus",
             "storage_params": {"path_results": f"/nimbus/jguidera/decoding/"}}]

        for param_set in param_sets:
            self.insert1(param_set, skip_duplicates=True)


@schema
class EdenoDecodeParams(SecKeyParamsBase):
    definition = """
    # Parameters for EdenoDecode
    edeno_decode_param_name : varchar(100)
    ---
    -> EDDecodeVariableParams
    -> EDCrossValidationParams
    -> EDAlgorithmParams
    -> EDComputeParams
    -> EDStorageParams
    """

    def _default_params(self):

        decode_variable_param_names = ["ppt_default", "wa_stay"]
        cross_validation_param_names = ["looct_path_default", "looct_path_default"]
        algorithm_param_name = "sorted_spikes_default"
        compute_param_name = "gpu_default"
        storage_param_name = "nimbus"

        return [[x, y, algorithm_param_name, compute_param_name, storage_param_name]
                for x, y in zip(decode_variable_param_names, cross_validation_param_names)]

    def get_params(self):

        ed_param_names = self.fetch1()

        # Get params from each sub table
        decode_variable_params = (EDDecodeVariableParams & ed_param_names).fetch1("decode_variable_params")
        cross_validation_params = (EDCrossValidationParams & ed_param_names).fetch1("cross_validation_params")
        algorithm_params = (EDAlgorithmParams & ed_param_names).fetch1("algorithm_params")
        computed_params = (EDComputeParams & ed_param_names).fetch1("compute_params")
        storage_params = (EDStorageParams & ed_param_names).fetch1("storage_params")

        # Return together
        # First check that no repeated param names
        all_param_names = np.concatenate(
            [list(x.keys()) for x in [
                decode_variable_params, cross_validation_params, algorithm_params, computed_params, storage_params]])
        check_all_unique(all_param_names)
        return {
            **decode_variable_params, **cross_validation_params, **algorithm_params, **computed_params,
            **storage_params}

    def get_param(self, param_name):
        return self.get_params()[param_name]

    def get_merge_key(self, key):
        params = (self & key).get_params()
        return merge_dicts([key, params])

    # Override parent class method so can insert into upstream tables
    def insert_defaults(self, **kwargs):
        for table in [
            EDDecodeVariableParams, EDCrossValidationParams, EDAlgorithmParams, EDComputeParams, EDStorageParams]:
            getattr(table(), "insert_defaults")()
        super().insert_defaults(**kwargs)


@schema
class EdenoDecodeSel(SelBase):
    definition = """
    # Selection from upstream tables for EdenoDecode
    -> BrainRegionUnits
    -> TrainTestEpoch
    -> EdenoDecodeParams
    """

    def _get_potential_keys(self, key_filter=None, verbose=True):

        curation_set_name = "runs_analysis_v1"
        brain_region_cohort_name = "all_targeted"
        epochs_description_name = "valid_single_contingency_runs"
        min_epoch_mean_firing_rate = .1
        eps_units_param_name = EpsUnitsParams().lookup_param_name([min_epoch_mean_firing_rate])
        unit_subset_type = "rand_target_region"
        unit_subset_size = 50
        unit_subset_iterations = np.arange(0, 10)

        # Define nwb file names
        nwb_file_names = get_reliability_paper_nwb_file_names()
        # ...if nwb file name passed, restrict to this
        if "nwb_file_name" in key_filter:
            nwb_file_names = [x for x in nwb_file_names if x == key_filter["nwb_file_name"]]

        # Train/test on same epoch
        keys = []
        for nwb_file_name in nwb_file_names:

            if verbose:
                print(f"On {nwb_file_name}...")

            # Get map from identifiers to curation name
            curation_set_key = {
                "curation_set_name": curation_set_name, "brain_region_cohort_name": brain_region_cohort_name,
                 "nwb_file_name": nwb_file_name}
            curation_set_df = (CurationSet & curation_set_key).fetch1_dataframe()

            # Get epochs descriptions for this nwb_file_name
            epochs_descriptions = (EpochsDescriptions & {
                "nwb_file_name": nwb_file_name, "epochs_description_name": epochs_description_name}).fetch1(
                "epochs_descriptions")

            # Get brain regions for this nwb_file_name
            brain_regions = (BrainRegionCohort & {
                "nwb_file_name": nwb_file_name, "brain_region_cohort_name": brain_region_cohort_name}).fetch1(
                "brain_regions")

            for epochs_description in epochs_descriptions:

                if verbose:
                    print(f"On {epochs_description}...")

                # Get epoch
                epoch = unpack_single_element((EpochsDescription & {
                    "nwb_file_name": nwb_file_name, "epochs_description": epochs_description}).fetch1(
                    "epochs"))

                # Get train / test name
                train_test_epoch_name = TrainTestEpoch().lookup_train_test_epoch_name([epoch], [epoch], nwb_file_name)

                for brain_region in brain_regions:

                    if verbose:
                        print(f"On {brain_region}...")

                    # Get curation name
                    curation_name = df_pop(
                        curation_set_df, {
                            "brain_region": brain_region, "epochs_description": epochs_description}, "curation_name")

                    for unit_subset_iteration in unit_subset_iterations:

                        if verbose:
                            print(f"On unit_subset_iteration {unit_subset_iteration}...")

                        # Get brain region units param name
                        brain_region_units_param_name = BrainRegionUnitsParams().lookup_param_name(
                            [eps_units_param_name, epochs_description, unit_subset_type,
                             unit_subset_size, unit_subset_iteration])

                        # Make insert key
                        insert_key = {"nwb_file_name": nwb_file_name, "brain_region": brain_region,
                                      "brain_region_units_param_name": brain_region_units_param_name,
                                      "curation_name": curation_name, "train_test_epoch_name": train_test_epoch_name}

                        # Continue if no entry in BrainRegionUnits
                        if len(BrainRegionUnits & insert_key) == 0:
                            continue

                        # Loop through decoding params
                        for edeno_decode_param_name in EdenoDecodeParams().fetch("edeno_decode_param_name"):
                            insert_key.update({"edeno_decode_param_name": edeno_decode_param_name})

                            # Populate spikes or marks table
                            merge_key = EdenoDecodeParams().get_merge_key(insert_key)
                            merge_key.update({"epoch": epoch})  # be sure to edit this if train != test
                            spikes_obj = EdenoDecode().get_spikes_obj(**merge_key)
                            spikes_obj.populate_upstream()

                            # Add keys
                            if verbose:
                                print(f"Adding key...")
                            keys.append(copy.deepcopy(insert_key))

        return keys


class SortedSpikesIndicatorData:

    def __init__(self, nwb_file_name, epoch, curation_name, brain_region, brain_region_units_param_name,
                 sampling_rate=500, sorter=None, team_name=None, upstream_obj=None):
        self.nwb_file_name = nwb_file_name
        self.epoch = epoch
        self.curation_name = curation_name
        self.brain_region = brain_region
        self.brain_region_units_param_name = brain_region_units_param_name
        self.sampling_rate = sampling_rate
        self.sorter = sorter
        self.team_name = team_name
        self.upstream_obj = upstream_obj
        # Get quantities derived from passed inputs
        self.interval_list_name = self._get_interval_list_name()
        self.sort_interval_name, self.curation_id = split_curation_name(self.curation_name)
        # Get default inputs if not passed
        self._get_default_inputs()

    def _get_interval_list_name(self):
        # Get interval list name
        return EpochIntervalListName().get_interval_list_name(self.nwb_file_name, self.epoch)

    def _get_default_inputs(self):
        # Get default inputs if not passed
        for param_name in ["team_name", "sorter"]:
            if getattr(self, param_name) is None:
                setattr(self, param_name, get_default_param(param_name))

    @staticmethod
    def _fetch_ssi_df(ssi_key):

        table_subset = (SortedSpikesIndicator & ssi_key)
        check_all_unique(table_subset.fetch("sort_group_id"))  # ensure no sort group repeated

        return table_subset.fetch_dataframe()

    @staticmethod
    def _get_sort_group_id_unit_id(column_name):
        return list(map(int, column_name.split("_")))

    def populate_upstream(self):
        key = {"nwb_file_name": self.nwb_file_name, "interval_list_name": self.interval_list_name,
               "sort_interval_name": self.sort_interval_name, "curation_id": self.curation_id}
        selection = (CuratedSpikeSorting * IntervalList) & key
        SortedSpikesIndicatorSelection.insert(selection, ignore_extra_fields=True, skip_duplicates=True)
        SortedSpikesIndicator.populate(key)

    def get_spikes(self):
        # Get spikes

        # First, get spikes across sort groups for this brain region
        # Get map from sort group to units
        bru_key = {
            "nwb_file_name": self.nwb_file_name, "brain_region": self.brain_region,
            "brain_region_units_param_name": self.brain_region_units_param_name, "curation_name": self.curation_name}
        sort_group_unit_ids_map = BrainRegionUnits().fetch1_sort_group_unit_ids_map(bru_key)
        # update upstream entries tracker
        self.upstream_obj.update_upstream_entries(BrainRegionUnits, bru_key)

        # If no map (happens if not enough units to subsample), return None
        if sort_group_unit_ids_map is None:
            return None

        # Otherwise, get spikes
        ssi_key = {k: getattr(self, k) for k in SortedSpikesIndicator.primary_key if k in dir(self)}
        ssi_keys = [{**ssi_key, **{"sort_group_id": sort_group_id}} for sort_group_id in sort_group_unit_ids_map]
        spikes = pd.concat([self._fetch_ssi_df(k) for k in ssi_keys], axis=1)
        # ...ensure no repeat units
        check_all_unique(spikes.columns)
        # ...update upstream entries tracker
        for k in ssi_keys:
            self.upstream_obj.update_upstream_entries(SortedSpikesIndicator, k)

        # Now take subset specified by brain_region_units_param_name
        keep_columns = []
        for column_name in spikes.columns:
            sort_group_id, unit_id = self._get_sort_group_id_unit_id(column_name)
            if sort_group_id in sort_group_unit_ids_map:
                if unit_id in sort_group_unit_ids_map[sort_group_id]:
                    keep_columns.append(column_name)
        # Check all units found
        expected_num_units = sum([len(x) for x in sort_group_unit_ids_map.values()])
        num_units = len(keep_columns)
        if num_units != expected_num_units:
            raise Exception(f"Expected to find {expected_num_units} units, found {num_units}")
        # Check columns unique
        check_all_unique(keep_columns)

        # Return spikes
        return spikes[keep_columns]


@schema
class EdenoDecode(ComputedBase):
    definition = """
    # Results from edeno Bayesian decoder
    -> EdenoDecodeSel
    ---
    file_names : blob
    """

    class BrainRegionUnits(dj.Part):
        definition = """
        # Achieves dependence on BrainRegionUnits
        -> EdenoDecode
        -> BrainRegionUnits
        """

    class EpochIntervalListName(dj.Part):
        definition = """
        # Achieves dependence on EpochIntervalListName
        -> EdenoDecode
        -> EpochIntervalListName
        """

    class DioWellDDTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellDDTrials
        -> EdenoDecode
        -> DioWellDDTrials
        """

    class DioWellArrivalTrials(dj.Part):
        definition = """
        # Achieves dependence on DioWellArrivalTrials
        -> EdenoDecode
        -> DioWellArrivalTrials
        """

    class DioWellArrivalTrialsSub(dj.Part):
        definition = """
        # Achieves dependence on DioWellArrivalTrialsSub
        -> EdenoDecode
        -> DioWellArrivalTrialsSub
        """

    class IntervalLinearizedPosition(dj.Part):
        definition = """
        # Achieves dependence on IntervalLinearizedPosition
        -> EdenoDecode
        -> IntervalLinearizedPosition
        """

    class Ppt(dj.Part):
        definition = """
        # Achieves dependence on Ppt
        -> EdenoDecode
        -> Ppt
        """

    class SortedSpikesClassifierParameters(dj.Part):
        definition = """
        # Achieves dependence on SortedSpikesClassifierParameters
        -> EdenoDecode
        -> SortedSpikesClassifierParameters
        """

    class SortedSpikesIndicator(dj.Part):
        definition = """
        # Achieves dependence on SortedSpikesIndicator
        -> EdenoDecode
        -> SortedSpikesIndicator
        """

    class StackedEdgeTrackGraph(dj.Part):
        definition = """
        # Achieves dependence on StackedEdgeTrackGraph
        -> EdenoDecode
        -> StackedEdgeTrackGraph
        """

    def _get_key_with_epoch_info(self, key, epoch):

        # Make copy of key to avoid changing outside function
        key = copy.deepcopy(key)

        # Add epoch and interval list name to key
        key["epoch"] = epoch
        key["interval_list_name"] = EpochIntervalListName().get_interval_list_name(key["nwb_file_name"], epoch)
        self._update_upstream_entries_tracker(EpochIntervalListName, key)

        # Return key
        return key

    def get_spikes_obj(self, **kwargs):
        # Return train and test spikes

        if "clusterless" not in kwargs:
            raise Exception(
                f"The boolean variable clusterless indicating whether or not to use clusterless "
                f"decoding must be passed in kwargs")

        # Define inputs to class based on whether using clusterless decoder or not
        inputs = ["nwb_file_name", "epoch", "curation_name", "brain_region", "brain_region_units_param_name",
                  "sampling_rate"]  # default
        if kwargs["clusterless"]:  # clusterless decoding
            inputs = ["nwb_file_name", "sort_interval_name", "epoch", "sampling_rate"]

        # Check that all inputs present in passed key
        check_membership(inputs, kwargs)
        # Get subset of passed key
        key_subset = {k: kwargs[k] for k in inputs}
        # Add in upstream_obj if exists
        if hasattr(self, "upstream_obj"):
            key_subset.update({"upstream_obj": self.upstream_obj})

        if kwargs["clusterless"]:
            raise Exception(f"Need to implement unit marks object")
        else:
            spikes_obj = SortedSpikesIndicatorData(**key_subset)  # clustered spikes
        self.upstream_obj = spikes_obj.upstream_obj

        return spikes_obj

    def _get_classifier_parameters(self, nwb_file_name, epoch, decode_variable, track_graph_idx, clusterless):

        track_graph_name = StackedEdgeTrackGraph().get_track_graph_name_for_decode_variable(
            decode_variable, idx=track_graph_idx, nwb_file_name=nwb_file_name, epoch=epoch)
        if clusterless:
            params_table = ClusterlessClassifierParameters
        else:
            params_table = SortedSpikesClassifierParameters
        params_key = {"classifier_param_name": track_graph_name}
        classifier_params = (params_table & params_key).fetch1()
        self._update_upstream_entries_tracker(params_table, params_key)

        return classifier_params

    def _get_combined_decoding_parameters(
            self, nwb_file_name, train_epoch, test_epoch, decode_variable, track_graph_idx, clusterless):

        # Get classifier and fit params from training data and predict params from test data
        train_parameters = self._get_classifier_parameters(
            nwb_file_name, train_epoch, decode_variable, track_graph_idx, clusterless)
        test_parameters = self._get_classifier_parameters(
            nwb_file_name, test_epoch, decode_variable, track_graph_idx, clusterless)

        return {"classifier_params": train_parameters["classifier_params"],
                "fit_params": train_parameters["fit_params"],
                "predict_params": test_parameters["predict_params"],
                "train_classifier_param_name": train_parameters["classifier_param_name"],
                "test_classifier_param_name": test_parameters["classifier_param_name"]}

    @staticmethod
    def _file_name_separating_character():
        return "|"

    @staticmethod
    def get_file_name_endings_map():
        return {**{"results": "results.nc"}, **{x: x for x in ["results_train_test_info", "params", "pos", "spikes"]}}

    @classmethod
    def get_file_name_base(cls, key):
        ordered_keys = [key[k] for k in cls.primary_key]
        return make_param_name(ordered_keys, cls._file_name_separating_character())

    @classmethod
    def key_from_file_name(cls, file_name):

        # Remove file name ending
        # ...detect file name ending (check there is only one)
        file_name_ending = check_return_single_element(
            [x for x in cls.get_file_name_endings_map().values() if file_name.endswith(x)]).single_element
        # ...remove file name ending and underscore
        idx = len(file_name_ending) + 1  # add one since in addition to file name ending, there is underscore
        file_name = file_name[:-idx]

        # Split file name into primary keys
        return {k: v for k, v in zip(cls.primary_key, file_name.split(cls._file_name_separating_character()))}

    @staticmethod
    def _get_classifier_class(clusterless):
        if clusterless:
            return ClusterlessClassifier
        return SortedSpikesClassifier

    @staticmethod
    def _run_classifier(classifier, position, train_spikes, test_spikes,  # marks or sorted spikes
                        fit_params, predict_params, use_gpu=False, gpu_id=None):

        # Check inputs
        if use_gpu and gpu_id is None:
            raise Exception(f"gpu_id must not be None if use_gpu is True")

        # Fit classifier using GPU
        if use_gpu:
            with cp.cuda.Device(gpu_id):
                print(f"fitting...")
                classifier.fit(position, train_spikes, **fit_params)
                print(f"predicting...")
                results = classifier.predict(spikes=test_spikes, time=test_spikes.index, **predict_params)

        # Fit classifier without GPU
        else:
            classifier.fit(position, train_spikes, **fit_params)
            results = classifier.predict(spikes=test_spikes, time=test_spikes.index, **predict_params)

        return classifier, results

    @classmethod
    def get_decode_variable_path_group_abbreviations(cls, decode_variable, nwb_file_name, epoch):

        path_groups = StackedEdgeTrackGraph().get_decode_variable_path_groups(
            decode_variable, nwb_file_name, epoch)

        if decode_variable == "pstpt":
            return [unpack_single_element(np.unique(
                [get_path_junction_abbreviation(y) for y in unpack_single_element(x)])) for x in path_groups]

        else:
            return [path_name_to_plot_string(unpack_single_element(x)) for x in path_groups]

    def get_decode_position(
            self, nwb_file_name, epoch, decode_variable,
            position_info_param_name="default_decoding", linearization_param_name="default", verbose=False, **kwargs):

        # Check inputs
        # ...Check decode variable name valid
        valid_decode_var_names = get_valid_decode_variable_names()
        if decode_variable not in valid_decode_var_names:
            raise Exception(f"valid decode variable names include: {valid_decode_var_names}")
        # ...Check that track_graph_idx passed if decode variable is not position_and_maze
        if decode_variable != "pos" and "track_graph_idx" not in kwargs:
            raise Exception(f"track_graph_idx must be passed if decode_variable is not pos")
        # ...Check that only_two_junction_paths passed if decoding ppt
        if decode_variable == "ppt" and "only_two_junction_paths" not in kwargs:
            raise Exception(f"only_two_junction_paths must be passed if decode_variable is ppt")

        # Get inputs if not passed
        trial_restriction = kwargs.pop("trial_restriction", None)

        # Get linear position_and_maze df
        # First, get pos track graph name
        pos_track_graph_name = StackedEdgeTrackGraph().get_track_graph_name_for_decode_variable(
            "pos", nwb_file_name=nwb_file_name, epoch=epoch)
        # update upstream entries tracker
        self._update_upstream_entries_tracker(StackedEdgeTrackGraph, {"track_graph_name": pos_track_graph_name})
        # Now get linear position_and_maze
        interval_list_name = EpochIntervalListName().get_interval_list_name(nwb_file_name, epoch)
        linpos_key = {
            "nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name,
            "position_info_param_name": position_info_param_name, "linearization_param_name": linearization_param_name,
            "track_graph_name": pos_track_graph_name}
        linear_position_df = (IntervalLinearizedPosition & linpos_key).fetch1_dataframe()
        # update upstream entries tracker
        self._update_upstream_entries_tracker(
                EpochIntervalListName, {"nwb_file_name": nwb_file_name, "epoch": epoch})
        self._update_upstream_entries_tracker(IntervalLinearizedPosition, linpos_key)

        # Get decode values

        # ...Initialize variables for decode values and corresponding path, and track graph name for non pos decode var
        decode_position, decode_position_path, decode_var_track_graph_name = None, None, None

        # ...Linear position_and_maze
        if decode_variable == "pos":
            decode_position = linear_position_df.linear_position

        # ...All others
        else:

            # Get track graph name
            decode_var_track_graph_name = StackedEdgeTrackGraph().get_track_graph_name_for_decode_variable(
                decode_variable, idx=kwargs["track_graph_idx"], nwb_file_name=nwb_file_name, epoch=epoch)

            # update upstream entries tracker
            self._update_upstream_entries_tracker(
                StackedEdgeTrackGraph, {"track_graph_name": decode_var_track_graph_name})

            setg_key = {"track_graph_name": decode_var_track_graph_name}
            setg_params = (StackedEdgeTrackGraph & setg_key).fetch1("stacked_edge_track_graph_params")
            scale_factor = setg_params["scale_factor"]

            # update upstream entries tracker
            self._update_upstream_entries_tracker(StackedEdgeTrackGraph, setg_key)

            key = {"nwb_file_name": nwb_file_name, "epoch": epoch, "position_info_param_name": position_info_param_name,
                   "track_graph_name": pos_track_graph_name}

            # Time relative to well arrival
            if decode_variable == "wa":  # well arrival
                decode_position = self.get_well_arrival_for_decode(
                    nwb_file_name, epoch, linear_position_df.linear_position, scale_factor=float(scale_factor),
                    dio_well_arrival_trials_param_name=setg_params["dio_well_arrival_trials_param_name"],
                    trial_restriction=trial_restriction)

            # All others
            else:

                # Get fraction path traversed
                ppt_df = (Ppt & key).fetch1_dataframe()

                # Get value for when rat at well
                well_period_value = setg_params["well_period_value"]

                # update upstream entries tracker if passed
                self._update_upstream_entries_tracker(Ppt, key)

                # Fraction path traversed
                if decode_variable == "ppt":  # proportion path traversed
                    decode_position, decode_position_path = self.get_ppt_for_decode(
                        linear_position_df.linear_position, ppt_df, scale_factor=float(scale_factor),
                        well_period_value=float(well_period_value),
                        only_two_junction_paths=kwargs["only_two_junction_paths"])

                # Turn zone decode variables
                elif decode_variable in self.get_valid_turn_zone_decode_variable_names():

                    # Get path groups for decode variable
                    pg_key = {**key, **{"decode_variable": decode_variable}}
                    decode_variable_path_groups = (EDPathGroups & pg_key).fetch1("path_groups")

                    # update upstream entries tracker
                    self._update_upstream_entries_tracker(EDPathGroups, pg_key)

                    # Get decode variable values over time
                    decode_position, decode_position_path = self._get_path_groups_decode_variable_position(
                        nwb_file_name, epoch, linear_position_df.linear_position, ppt_df, float(scale_factor),
                        float(well_period_value), float(setg_params["path_groups_spacing"]),
                        decode_variable_path_groups, verbose)

                else:
                    raise Exception(f"{decode_variable} not accounted for in code")

        # Store decode values and corresponding paths in df
        position_df = pd.DataFrame.from_dict(
            {"decode_position": decode_position, "decode_position_path": decode_position_path})

        # Return position_and_maze variables
        return {
            "decode_var_track_graph_name": decode_var_track_graph_name, "pos_track_graph_name": pos_track_graph_name,
            "position_df": position_df}

    def get_well_arrival_for_decode(
            self, nwb_file_name, epoch, linear_position, scale_factor, dio_well_arrival_trials_param_name,
            trial_restriction=None, verbose=False):

        # Get well arrival trial times. Restrict trials as indicated

        # ...define key for querying tables
        wa_key = {
            "nwb_file_name": nwb_file_name, "epoch": epoch,
            "dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name}

        def _get_wa_trial_times(table, wa_key):
            # Get entry from table
            table_entry = (table & wa_key).fetch1()
            # Update upstream entries tracker
            self._update_upstream_entries_tracker(table, wa_key)
            # Return start and end of each well arrival trial
            return np.asarray(list(
                zip(table_entry["trial_start_times"], table_entry["trial_end_times"])))

        # ...no restriction
        if trial_restriction is None:
            wa_trial_times = _get_wa_trial_times(DioWellArrivalTrials, wa_key)
            raise Exception

        # ...restrict to trials during which rat stayed at well for full 2s delay period
        elif trial_restriction == "stay_trial":
            # ...update table key
            wa_key.update({"dio_well_arrival_trials_sub_param_name": "stay"})
            wa_trial_times = _get_wa_trial_times(DioWellArrivalTrialsSub, wa_key)

        # ...raise exception if restriction not accounted for in code
        else:
            raise Exception(f"trial_restriction {trial_restriction} not accounted for in code")

        # Create vector with well arrival times
        pos_t = linear_position.index  # time of upsampled position_and_maze measurements
        wa_decode = pd.Series([np.nan] * len(pos_t), index=pos_t)  # well arrival for decode. Initialize to nan
        for t_start, t_end in wa_trial_times:  # for each trial
            pos_trial_t = pos_t[np.logical_and(pos_t > t_start, pos_t < t_end)]  # upsampled position_and_maze times within trial
            wa_decode[pos_trial_t] = pos_trial_t - t_start  # update

        # Plot if indicated
        if verbose:
            fig, ax = plt.subplots(figsize=(20, 2))
            ax.plot(wa_decode, '.', color="green")
            ax.plot(np.concatenate(wa_trial_times),
                    [0] * len(np.concatenate(wa_trial_times)),
                    '.', color="orange")
            format_ax(ax=ax, title="well arrival trial")

        # Return well arrival variable
        return wa_decode * scale_factor

    @classmethod
    def get_ppt_for_decode(
            cls, linear_position, ppt_df, scale_factor, well_period_value, only_two_junction_paths=False,
            verbose=False):
        """
        Define proportion path traversed for decoding.
        Approach: Use proportion path traversed from table, and further mark times between trials as
        having a proportion path traversed of 110%. Scale proportion path traversed to range from 0 to 400 so
        that on similar scale as linear position_and_maze.
        :param linear_position: series
        :param ppt_df: dataframe from Ppt entry
        :param scale_factor: float. multiply proportion path traversed ranging from 0 to 1 by this
        :param well_period_value: period between departure to arrival trials set to this value
        :param only_two_junction_paths: True to restrict paths to those with two corners
        :param verbose: True to plot ppt
        :return:
        """

        from src.jguides_2024.utils.df_helpers import zip_df_columns
        from src.jguides_2024.position_and_maze.jguidera_maze import return_n_junction_path_names

        # Get time of upsampled position_and_maze measurements
        pos_t = linear_position.index

        # Get proportion path traversed for decode. Initialize to "well period" value
        ppt_decode = pd.Series([well_period_value] * len(pos_t), index=pos_t)
        ppt_path_decode = pd.Series([None] * len(pos_t), index=pos_t)  # path for each ppt sample
        for ppt_trial_t, ppt_trial, path_name in zip_df_columns(
                ppt_df, ["trials_time", "trials_ppt", "trials_path_name"]):  # for each trial
            pos_trial_t = pos_t[np.logical_and(pos_t > ppt_trial_t[0],
                                               pos_t < ppt_trial_t[-1])]  # upsampled position_and_maze times within trial
            ppt = [np.nan] * len(pos_trial_t)  # intialize
            if not only_two_junction_paths or path_name in return_n_junction_path_names(2):
                ppt = np.interp(x=pos_trial_t, xp=ppt_trial_t, fp=ppt_trial)
            ppt_decode[pos_trial_t] = ppt  # update proportion path traversed
            ppt_path_decode[pos_trial_t] = path_name

        # Plot if indicated
        if verbose:
            fig, ax = plt.subplots(figsize=(20, 2))
            ax.plot(ppt_decode, '.', color="green")
            ax.plot(np.concatenate(ppt_df["trials_time"]), np.concatenate(ppt_df["trials_ppt"]), '.', color="orange")
            format_ax(ax=ax, title="proportion path traversed")

        return ppt_decode * scale_factor, ppt_path_decode

    def _get_path_groups_decode_variable_position(
            self, nwb_file_name, epoch, linear_position, ppt_df, scale_factor, well_period_value,
            path_groups_spacing, path_groups, verbose=False):
        """
        Get "position_and_maze" for decode variable that has "path groups" (paths within same path group get assigned
        to same set of positions). Example: percent same turn path traversed.
        :param nwb_file_name:
        :param epoch:
        :param linear_position:
        :param ppt_df:
        :param scale_factor:
        :param well_period_value:
        :param path_groups_spacing:
        :param path_groups:
        :param verbose:
        :return:
        """

        # Get proportion path traversed
        ppt_decode, ppt_decode_path = self.get_ppt_for_decode(
            linear_position, ppt_df, scale_factor=1, well_period_value=well_period_value, verbose=verbose)

        # Assign value for trials whose paths are in path_groups, otherwise nan
        dd_key = {"nwb_file_name": nwb_file_name, "epoch": epoch}
        trials_df = (DioWellDDTrials & dd_key).fetch1_dataframe()
        # update upstream entries tracker
        self._update_upstream_entries_tracker(DioWellDDTrials, dd_key)

        position = pd.Series([np.nan] * len(ppt_decode), index=ppt_decode.index)  # initialize
        position_path = pd.Series([np.nan] * len(ppt_decode), index=ppt_decode.index)  # path at each position_and_maze
        for idx, path_group in enumerate(path_groups):
            path_group_trials = list(zip_df_columns(trials_df[trials_df["path_names"].isin(path_group)],
                                                    ["trial_start_times", "trial_end_times"]))
            if len(path_group_trials) > 0:  # if trials for this path group
                time_mask = event_times_in_intervals_bool(
                    event_times=ppt_decode.index, valid_time_intervals=path_group_trials)
                position[time_mask] = ppt_decode[time_mask] + idx * (well_period_value + path_groups_spacing)
                position_path[time_mask] = ppt_decode_path[time_mask]

        # Plot if indicated
        if verbose:
            fig, ax = plt.subplots(figsize=(20, 3))
            ax.plot(position * scale_factor, ".", alpha=.4)
            format_ax(ax=ax, title="path group decode variable")

        return position * scale_factor, position_path

    def run_decoding(self, key, verbose=True):
        """
        Stand alone method to allow decoding without populating table if desired
        :param key: dictionary, key for current entry
        :param verbose: boolean, if True print progress
        :return: named tuple with several decoding results
        """

        # Put params into one dictionary, making sure keys dont overlap
        merge_key = EdenoDecodeParams().get_merge_key(key)
        # ...include train and test epochs
        train_epoch, test_epoch = (TrainTestEpoch & key).get_train_test_epoch()
        merge_key = merge_dicts([merge_key, {"train_epoch": train_epoch, "test_epoch": test_epoch, "verbose": verbose}])

        # Get spikes and position_and_maze for train and test epochs
        spikes_map, pos_map = dict(), dict()
        for x in ["train_epoch", "test_epoch"]:
            epoch = merge_key[x]

            # Add epoch and corresponding interval list name to key
            epoch_key = self._get_key_with_epoch_info(merge_key, epoch)

            # Get spikes
            spikes_map[x] = self.get_spikes_obj(**epoch_key).get_spikes()

            # Get position_and_maze
            inputs = ["nwb_file_name", "epoch", "decode_variable", "track_graph_idx", "position_info_param_name",
                      "linearization_param_name", "only_two_junction_paths", "trial_restriction", "verbose"]
            pos_map[x] = self.get_decode_position(**{k: epoch_key[k] for k in inputs if k in epoch_key})

        # Check units same across train and test epochs
        check_set_equality(
            spikes_map["train_epoch"].columns, spikes_map["test_epoch"].columns, "train spikes", "test spikes")

        # Get classifier and fit parameters using training data and predict parameters using test data
        parameters = self._get_combined_decoding_parameters(
            **{k: merge_key[k] for k in [
                "nwb_file_name", "train_epoch", "test_epoch", "decode_variable", "track_graph_idx", "clusterless"]})

        # Take subset of spikes and position_and_maze data where the two have matching indices
        if merge_key["clusterless"]:
            # TODO: write code to find matching indices in clusterless case
            raise Exception(f"Must write this code")
        else:
            for x in ["train_epoch", "test_epoch"]:
                pos_map[x]["position_df"], spikes_map[x] = match_two_dfs(
                    pos_map[x]["position_df"], spikes_map[x], merge_key["max_tolerated_diff"])

        # Find non-null training position_and_maze values
        non_null_train_position_idxs = np.where(pos_map["train_epoch"]["position_df"].decode_position.notnull())[0]

        # Fit and predict
        store_params = dict()  # initialize
        classifier_class = self._get_classifier_class(merge_key["clusterless"])  # get classifier
        cv_dfs = None  # initialize

        # Initialize variables
        results_train_test_info = None

        # Case 1: cross validation
        if merge_key["cross_validation_method"] is not None:

            # Define params for cross validation
            cross_validation_params = {
                **{"data_vector": pos_map["train_epoch"]["position_df"].index}, **{
                    k: merge_key[k] for k in ["n_splits", "limit_same_train_test"] if k in merge_key}}

            # If leave one out cross validation, add trials information to params (one trial goes from one well
            # departure to next)
            if merge_key["cross_validation_method"] == "leave_one_out_condition_trials":

                # Get departure to departure trials
                dd_key = {"nwb_file_name": merge_key["nwb_file_name"], "epoch": merge_key["train_epoch"]}
                df = (DioWellDDTrials & dd_key).fetch1_dataframe()
                trial_intervals = list(zip_df_columns(df, ["trial_start_times", "trial_end_times"]))
                # ...update upstream entries tracker
                self._update_upstream_entries_tracker(DioWellDDTrials, dd_key)

                # Find training position_and_maze indices in each trial
                trial_idxs = [event_times_in_intervals(
                    pos_map["train_epoch"]["position_df"].index, [trial_interval])[0]
                              for trial_interval in trial_intervals]
                # ...hold onto just first and last of training position_and_maze indices in each trial
                trial_idx_intervals = [[x[0], x[-1]] for x in trial_idxs]

                # Get map from condition to indices of trials of this condition

                # Case 1: all trials together
                if merge_key["trials_description"] == "all":
                    condition_trials_map = {"all_paths": trial_idx_intervals}

                # Case 2: trials grouped by path
                elif merge_key["trials_description"] == "path":

                    # Get names of potentially rewarded paths
                    path_names = MazePathWell().get_rewarded_path_names_across_epochs(
                        merge_key["nwb_file_name"], [merge_key[x] for x in ["train_epoch", "test_epoch"]])

                    condition_trials_map = {
                        path_name: [
                            x for idx, x in enumerate(trial_idx_intervals) if df.path_names.iloc[idx] == path_name]
                        for path_name in path_names}

                else:
                    raise Exception(f"{merge_key['trials_description']} not accounted for in code")

                # Store trial slice indices in a dictionary so can pass to cross validation class
                cross_validation_params.update({"condition_trials_map": condition_trials_map})

            # Get cross validation quantities
            cv_obj = CrossValidate(
                cross_validation_method=merge_key["cross_validation_method"],
                cross_validation_params=cross_validation_params)

            cv_dfs = cv_obj.get_train_test_sets()

            # Update train indices to be only on non-null indices
            for train_set_id in cv_dfs.train_set_df.index:  # for cv folds
                cv_dfs.train_set_df.loc[train_set_id].train_idxs = list(
                    set(cv_dfs.train_set_df.loc[train_set_id].train_idxs) & set(non_null_train_position_idxs))

            # Decode

            # Define GPU ID if using gpu
            gpu_id = None  # default
            if merge_key["use_gpu"]:
                gpu_id = 6

            # Initialize lists for results across cross validation folds
            results_list = []
            results_train_test_info = []

            # Loop through cross validation folds
            for (train_set_id, test_set_id, fold_num) in zip_df_columns(
                    cv_dfs.train_test_set_df, ["train_set_id", "test_set_id", "fold_num"]):
                optional_print(f"On {train_set_id}, {test_set_id}, fold {fold_num}...", verbose)

                # Get train and test indices
                train_idxs = cv_dfs.train_set_df.loc[train_set_id].train_idxs
                test_idxs = cv_dfs.test_set_df.loc[test_set_id].test_idxs

                # Train classifier and predict
                classifier = classifier_class(**parameters["classifier_params"])
                classifier, results = self._run_classifier(
                    classifier, pos_map["train_epoch"]["position_df"].decode_position.iloc[train_idxs],
                    spikes_map["train_epoch"].iloc[train_idxs], spikes_map["test_epoch"].iloc[test_idxs],
                    parameters["fit_params"], parameters["predict_params"], merge_key["use_gpu"], gpu_id)

                # Store information about train/test ID, fold number
                times = results.time.values
                results_train_test_info += list(
                    zip(times, [train_set_id]*len(times), [test_set_id]*len(times), [fold_num]*len(times)))

                # Store classifier
                classifier_name = make_param_name(["classifier", train_set_id, test_set_id, fold_num])
                store_params[classifier_name] = classifier  # store classifier

                # Append results
                results_list.append(results)

            # Concatenate results across folds
            results = xr.concat(results_list, dim="time")

            # Convert train test info to df
            results_train_test_info = df_from_data_list(
                results_train_test_info, ["time", "train_set_id", "test_set_id", "fold_num"])

        # Case 2: no cross validation
        elif merge_key["cross_validation_method"] is None:
            # Train classifier and predict
            classifier = classifier_class(**parameters["classifier_params"])
            classifier, results = self._run_classifier(
                classifier, pos_map["train_epoch"]["position_df"].decode_position.iloc[non_null_train_position_idxs],
                spikes_map["train_epoch"].iloc[non_null_train_position_idxs],
                spikes_map["test_epoch"], parameters["fit_params"], parameters["predict_params"], merge_key["use_gpu"])
            # Store classifier
            store_params[f"classifier"] = classifier

        else:
            raise Exception(f"Must write code for cross validation method {merge_key['cross_validation_method']}")

        # Store additional parameter quantities
        store_params.update({
            "key": key, "parameters": parameters, "non_null_train_position_idxs": non_null_train_position_idxs})
        track_graph_quantities = {f"{x}_{y}": pos_map[x][y] for y in [
            "decode_var_track_graph_name", "pos_track_graph_name"] for x in ["train_epoch", "test_epoch"]}
        store_params.update(track_graph_quantities)
        if cv_dfs is not None:
            store_params.update(cv_dfs._asdict())

        # Store position_and_maze
        store_pos = {
            "train_position": pos_map["train_epoch"]["position_df"],
            "test_position": pos_map["test_epoch"]["position_df"]}

        # Store spikes
        store_spikes = {"train_spikes": spikes_map["train_epoch"], "test_spikes": spikes_map["test_epoch"]}

        # Return results and upstream entries tracker (since tracker doesnt seem to persist as attribute in make fn)
        DecodeOutput = namedtuple("DecodeOutput", "results results_train_test_info params pos spikes upstream_obj")

        return DecodeOutput(results, results_train_test_info, store_params, store_pos, store_spikes, self.upstream_obj)

    def _get_path_results(self, key=None):

        # Get file path to results
        if key is None:
            key = self.fetch1()
        path_results = (EdenoDecodeParams & key).get_params()["path_results"]

        # Add subject ID to path
        return os.path.join(path_results, get_subject_id(key["nwb_file_name"]))

    def make(self, key):
        """
        Populate decoding table
        :param key: key to current entry
        """

        # Reset tracker for upstream tiles (important so does not carry over from one populate command to next)
        self._initialize_upstream_entries_tracker()

        # Get names of files with data
        file_name_base = self.get_file_name_base(key)
        file_names_map = {
            data_type: f"{file_name_base}_{file_name_ending}"
            for data_type, file_name_ending in self.get_file_name_endings_map().items()}

        # Raise exception if files exist
        path_results = self._get_path_results(key)
        if any([os.path.exists(os.path.join(path_results, file_name)) for file_name in file_names_map.values()]):
            raise Exception(f"One or more files already exists for {key}; this should not be the case")

        # Run decoding
        results_obj = self.run_decoding(key)

        # Save results, parameters, position_and_maze, and spikes
        print(f"Saving files for {file_name_base}...")

        # Change to directory where want to save results
        current_dir = os.getcwd()  # get current directory so can change back to this
        os.chdir(path_results)

        # Save results
        results_obj.results.to_netcdf(file_names_map["results"])  # save results

        # Save other outputs
        for data_type in ["results_train_test_info", "params", "pos", "spikes"]:
            pickle_file(
                getattr(results_obj, data_type), file_names_map[data_type], save_dir=path_results, overwrite=True)

        # Change back to previous directory
        os.chdir(current_dir)

        # Insert into table
        main_key = {**key, **{"file_names": file_names_map}}
        self.insert1(main_key)

        # Insert into part tables
        self._insert_part_tables(key, results_obj.upstream_obj)

    def fetch1_output(self, data_types=None):

        if data_types is None:
            data_types = list(self.get_file_name_endings_map().keys())

        elif isinstance(data_types, str):
            data_types = [data_types]

        decode_output = dict()

        file_paths_map = self.fetch1_file_paths()

        if "results" in data_types:
            decode_output.update({"results": xr.open_dataset(file_paths_map["results"])})

        for data_type in ["results_train_test_info", "params", "spikes", "pos"]:
            if data_type in data_types:
                decode_output.update({data_type: pickle.load(open(file_paths_map[data_type], "rb"))})

        # Return single output if only one passed, otherwise as named tuple
        if len(data_types) == 1:
            return unpack_single_element(list(decode_output.values()))
        return namedtuple("DecodeOutput", " ".join(data_types))(*[decode_output[x] for x in data_types])

    def fetch1_file_paths(self):
        # Return full path to files from table entry

        # Get path to results
        path_results = self._get_path_results()

        # Get file names for stored quantities
        file_names = self.fetch1("file_names")

        return {k: os.path.join(path_results, v) for k, v in file_names.items()}

    # Override parent class method so can delete files upon deletion of table entry
    def delete(self, transaction=True, safemode=None):

        for key in self.fetch("KEY"):

            table_subset = self & key

            # Delete files
            file_paths_map = table_subset.fetch1_file_paths()  # get file paths

            for file_path in file_paths_map.values():  # loop through file paths

                print(f"Deleting {file_path}...")

                if os.path.exists(file_path):
                   os.remove(file_path)  # delete file

                else:
                    print(f"Could not remove {file_path}; does not exist")

        # Delete table entry
        super().delete(transaction, safemode)

    def cleanup_lone_files(self):
        # Delete files that have no corresponding table entry

        for storage_params in EDStorageParams().fetch("storage_params"):

            os.chdir(storage_params["path_results"])

            for subject_id in get_reliability_paper_nwb_file_names(as_dict=True):

                os.chdir(subject_id)

                file_names = os.listdir()

                bad_file_names = []
                for file_name in file_names:
                    key = EdenoDecode().key_from_file_name(file_name)
                    if len(EdenoDecode & key) == 0:
                        bad_file_names.append(file_name)

                print(f"Found {len(bad_file_names)} bad files. Deleting...")
                for file_name in bad_file_names:
                    os.remove(file_name)

                os.chdir("..")

    def cleanup_lone_entries(self, safemode=None):
        # Delete table entries that have no corresponding file

        if safemode is None:
            safemode = True

        bad_keys = []
        for key in self.fetch("KEY"):
            storage_param_name = (EdenoDecodeParams & key).fetch1("storage_param_name")
            storage_params = (EDStorageParams & {"storage_param_name": storage_param_name}).fetch1("storage_params")
            results_path = os.path.join(storage_params["path_results"], get_subject_id(key["nwb_file_name"]))
            os.chdir(results_path)

            file_names = (self & key).fetch1("file_names")

            if not all([os.path.exists(x) for x in file_names.values()]):
                bad_keys.append(key)

        print(f"Found {len(bad_keys)} bad keys. Deleting...")
        for bad_key in bad_keys:
            (self & bad_key).delete(safemode=safemode)

    def cleanup_lone(self, safemode=None):
        # Delete files that have no corresponding table entry, and delete table entries that have no corresponpding
        # file
        self.cleanup_lone_files()
        self.cleanup_lone_entries(safemode)

    def get_stacked_edge_track_graph_param(self, param_name, train_test, params=None):

        # Get params if not passed
        if params is None:
            params = self.fetch1_output("params")

        # Return stacked edge track graph name
        track_graph_name = params["parameters"][f"{train_test}_classifier_param_name"]
        return (StackedEdgeTrackGraph & {"track_graph_name": track_graph_name}).fetch1(
            "stacked_edge_track_graph_params")[param_name]

    @staticmethod
    def get_classifier_names(params):
        keyword = "classifier"
        return [x for x in params if x.startswith(keyword) and x[-1].isdigit()]

    def get_environment_obj(self, params=None):
        # Return single environment object across model folds

        # Get params if not passed
        if params is None:
            params = self.fetch1_output("params")

        # Get classifier model fold names
        classifier_names = self.get_classifier_names(params)

        # Get single environment per classifier model fold
        environments = [unpack_single_element(params[x].environments) for x in classifier_names]

        # Get single environment across classifier model folds
        return unpack_single_element([environments[0]] + [
            x for x in environments[1:] if x != environments[0]])

    def get_pos_bins(self, num_pos_bins, train_test, keys=None, params_list=None):
        # Options for loading: load pos bin edges for one or multiple files where pos bins are the same
        # Use multiprocessing because takes a long time to load each file

        t1 = time.process_time()  # time process

        # Get position_and_maze bin edges for train or test case

        # Check inputs
        check_membership([train_test], ["train", "test"])

        # Get keys if not passed
        if keys is None:
            keys = self.fetch("KEY")

        print(f"Getting pos bin edges across {len(keys)} files...")

        # Get decode variable so can define bins based upon this
        decode_variable = check_return_single_element([
            (EdenoDecodeParams & key).get_param("decode_variable") for key in keys]).single_element

        # Get params list if not passed
        if params_list is None:

            file_names = [(self & key).fetch1_file_paths()["params"] for key in keys]

            pool = mp.Pool(mp.cpu_count())
            params_list = pool.map(_unpickle, file_names)
            pool.close()
            pool.join()  # waits until all processes done before running next line

        # Get bin edges, bin centers, and "x values": unscaled bin centers in cases where decode variable
        # reflects a scaled version of the original variable
        bin_edges_, bin_centers_, x_vals_ = None, None, None
        for idx, params in enumerate(params_list):

            # Get scale factor if stacked edge track graph
            if decode_variable not in ["pos"]:
                scale_factor = self.get_stacked_edge_track_graph_param("scale_factor", train_test, params)

            # First, get bin edges and centers
            # Case 1: well arrival. Simply return smallest and largest values in track graph
            if decode_variable in ["pos", "wa"]:
                environment_obj = self.get_environment_obj(params)
                bin_edges = np.linspace(
                    np.min(environment_obj.edges_), np.max(environment_obj.edges_), num_pos_bins + 1)
                bin_centers = vector_midpoints(bin_edges)

            # Case 2: Single edge track graph decode variables. Restrict to positions on track (exclude period at well)
            elif decode_variable == "ppt":
                bin_edges = np.linspace(0, scale_factor, num_pos_bins + 1)
                bin_centers = vector_midpoints(bin_edges)

            # Case 3: Turn zone decode variables. Get bins for each turn zone, and concatenate.
            # This means middle bin should have no data.
            # TODO: change names to path group decode variables. Also test this portion of code.
            elif decode_variable in get_valid_turn_zone_decode_variable_names():
                path_groups_spacing = self.get_stacked_edge_track_graph_param("path_groups_spacing", train_test, params)
                well_period_value = self.get_stacked_edge_track_graph_param("well_period_value", train_test, params)
                num_segments = self.get_stacked_edge_track_graph_param("num_segments", train_test, params)
                turn_zone_bin_edges = []
                for segment_num in np.arange(0, num_segments):
                    start_pos = (well_period_value + path_groups_spacing) * scale_factor * segment_num
                    end_pos = start_pos + scale_factor
                    turn_zone_bin_edges.append(np.linspace(start_pos, end_pos, num_pos_bins + 1))
                bin_edges = np.concatenate(turn_zone_bin_edges)
                bin_centers = vector_midpoints(bin_edges)

            else:
                raise Exception(f"no code written yet for {decode_variable}")

            # Now get scaled bin centers values ("x values")
            x_vals = bin_centers / scale_factor

            # Compare with previous iteration and raise error if not same values
            if idx > 0:
                if any(np.concatenate([x != y for x, y in
                                       zip([bin_edges, bin_centers, x_vals], [bin_edges_, bin_centers_, x_vals_])])):
                    raise Exception(f"at least one of bin_edges, bin_centers, x_vals not uniform across table entries")

            # Store quantities so can compare to those on next iteration
            bin_edges_ = copy.deepcopy(bin_edges)
            bin_centers_ = copy.deepcopy(bin_centers)
            x_vals_ = copy.deepcopy(x_vals)

        print(f"Finished getting pos bins. {time.process_time() - t1 : 2f}s elapsed")

        return namedtuple("Bins", "bin_edges bin_centers_df")(bin_edges, pd.DataFrame.from_dict(
                {"bin_centers": bin_centers, "x_vals": x_vals}))


@schema
class EdenoDecodeMAPSel(SelBase):
    definition = """
    # Selection from upstream tables for EdenoDecodeMAP
    -> EdenoDecode
    """


@schema
class EdenoDecodeMAPSel(SelBase):
    definition = """
    # Selection from upstream tables for EdenoDecode
    -> EdenoDecode
    """


@schema
class EdenoDecodeMAP(ComputedBase):
    definition = """
    # Maximum a posteriori probability estimate of decode variable
    -> EdenoDecodeMAPSel
    ---
    -> nd.common.AnalysisNwbfile
    edeno_decode_map_object_id : varchar(40)
    """

    def make(self, key):

        decode_output = (EdenoDecode & key).fetch1_output()
        # Get environment object (currently requires one environment object for key)

        environment_obj = EdenoDecode().get_environment_obj(decode_output.params)

        # Define position_and_maze estimate from posterior
        map_pos = max_posterior_position(decode_output.results.acausal_posterior, environment_obj)
        insert_analysis_table_entry(self, [map_pos], key, reset_index=True)

    # Override parent class method so can specify index name
    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):

        return super().fetch1_dataframe(object_id_name=None, restore_empty_nwb_object=True, df_index_name=df_index_name)

    def get_aligned_map_test_pos(self, max_tolerated_time_diff, key=None):

        # Get aligned MAP position_and_maze and test position_and_maze

        # Get key if not passed
        if key is None:
            key = self.fetch1("KEY")

        # Get actual and decoded position_and_maze
        pos = (EdenoDecode & key).fetch1_output("pos")
        map_pos = (self & key).fetch1_dataframe()

        # Match dfs
        test_position, map_pos = match_two_dfs(
            pos["test_position"], map_pos, max_tolerated_diff=max_tolerated_time_diff)

        return namedtuple("Pos", "test_position map_pos")(test_position, map_pos)


def populate_jguidera_edeno_decoder_run(key=None, tolerate_error=False):
    schema_name = "jguidera_edeno_decoder_run"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_edeno_decoder_run():
    schema.drop()
