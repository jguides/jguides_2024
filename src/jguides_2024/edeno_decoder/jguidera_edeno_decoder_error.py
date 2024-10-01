import copy

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd
from spyglass.common import close_nwb_files

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import get_subject_id, \
    plot_junction_fractions, get_val_pairs
from src.jguides_2024.datajoint_nwb_utils.datajoint_covariate_firing_rate_vector_table_base import \
    PathWellPopSummBase, PopulationAnalysisSecKeyParamsBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, SecKeyParamsBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import populate_insert, \
    get_schema_table_names_from_file, insert_analysis_table_entry, \
    unique_table_column_sets, get_table_secondary_key_names
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table
from src.jguides_2024.edeno_decoder.jguidera_edeno_decoder_run import EdenoDecodeParams, EdenoDecode, \
    EdenoDecodeMAP
from src.jguides_2024.metadata.jguidera_brain_region import CurationSet, BrainRegionCohort
from src.jguides_2024.metadata.jguidera_epoch import TrainTestEpoch, TrainTestEpochSet
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnitsFail, BrainRegionUnitsParams, \
    BrainRegionUnitsCohortType
from src.jguides_2024.utils.df_helpers import df_from_data_list
from src.jguides_2024.utils.dict_helpers import check_return_single_dict, add_defaults, return_shared_key_value
from src.jguides_2024.utils.for_loop_helpers import print_iteration_progress
from src.jguides_2024.utils.hierarchical_bootstrap import hierarchical_bootstrap
from src.jguides_2024.utils.list_helpers import check_return_single_element
from src.jguides_2024.utils.plot_helpers import format_ax
from src.jguides_2024.utils.set_helpers import check_set_equality, check_membership
from src.jguides_2024.utils.string_helpers import format_bool, replace_chars

schema = dj.schema("jguidera_edeno_decoder_error")

# These imports are called with eval or used in table definitions (do not remove):
EdenoDecodeParams
EdenoDecode
EdenoDecodeMAP
nd


@schema
class EdenoDecodeErrParams(SecKeyParamsBase):
    definition = """
    # Parameters for EdenoDecodeErr
    edeno_decode_err_param_name : varchar(40)
    ---
    error_type : varchar(40)
    max_tolerated_time_diff : float
    """

    def _default_params(self):
        return [["absolute_original_scale", .01]]


@schema
class EdenoDecodeErrSel(SelBase):
    definition = """
    # Selection from upstream tables for EdenoDecodeErr
    -> EdenoDecode
    -> EdenoDecodeMAP
    -> EdenoDecodeErrParams
    """


@schema
class EdenoDecodeErr(ComputedBase):
    definition = """
    # Decoding errors using edeno Bayesian decoder
    -> EdenoDecodeErrSel
    ---
    -> nd.common.AnalysisNwbfile
    edeno_decode_err_object_id : varchar(40)
    """

    def get_aligned_map_test_pos(self, key=None):

        # Get key if not passed
        if key is None:
            key = self.fetch1("KEY")

        # Get maximum tolerated offset in times across dfs
        max_tolerated_time_diff = (EdenoDecodeErrParams & key).fetch1("max_tolerated_time_diff")

        return (EdenoDecodeMAP & key).get_aligned_map_test_pos(max_tolerated_time_diff)

    def make(self, key):

        # Restrict test position_and_maze and map posterior to close indices across the two (important because
        # time index can be slightly different across the two due to Bayesian decoder implementation)
        pos_obj = self.get_aligned_map_test_pos(key)

        # Compute error

        # ...get type of error
        error_type = (EdenoDecodeErrParams & key).fetch1("error_type")

        # ...find absolute error if indicated
        if error_type == "absolute_original_scale":
            scale_factor = (EdenoDecode & key).get_stacked_edge_track_graph_param("scale_factor", "test")

            error_df = pd.DataFrame(
                abs(pos_obj.test_position.decode_position - pos_obj.map_pos.map_posterior)/scale_factor,
                columns=["error"])

        # ...raise exception if not accounted for in code
        else:
            raise Exception(f"error_type {error_type} not accounted for in code")

        # Insert into table
        insert_analysis_table_entry(self, [error_df], key, reset_index=True)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name="time"):
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def _get_vals_index_name(self):
        return "x_val"

    def plot_error(self):
        # Plot error along with MAP and test position_and_maze

        # Get error
        error_df = self.fetch1_dataframe()

        # Get aligned MAP and test position_and_maze
        pos_obj = self.get_aligned_map_test_pos()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(error_df.error, color="blue", label="decode error")
        ax.plot(pos_obj.map_pos.map_posterior, color="red", label="MAP position_and_maze")
        ax.plot(pos_obj.test_position.decode_position, color="gray", label="test position_and_maze")

        # Legend
        plt.legend()

        # Format axis
        format_ax(ax, xlabel="Time (s)", ylabel="Position")


@schema
class EdenoDecodeErrSummBinParams(dj.Manual):
    definition = """
    # Parameters for averaging and binning for EdenoDecodeErrSumm
    edeno_decode_err_summ_bin_param_name : varchar(80)
    ---
    edeno_decode_err_summ_bin_params : blob
    """

    @staticmethod
    def _get_default_params_map():

        params_map = dict()

        # Epoch average, some number of uniform bins
        num_pos_bins_list = [10]
        averaging_method = "epoch_average"
        binning_method = "uniform"

        for num_pos_bins in num_pos_bins_list:
            # make param name
            param_name = f"{averaging_method}_{binning_method}_{num_pos_bins}"
            # update params map
            params_map.update({param_name: {
                "averaging_method": averaging_method, "binning_method": binning_method, "num_pos_bins": num_pos_bins}})

        # Return map from parameter name to parameters
        return params_map

    def get_default_param_names(self):
        return list(self._get_default_params_map().keys())

    def insert_defaults(self, **kwargs):

        for param_name, params in self._get_default_params_map().items():

            self.insert1(
                {"edeno_decode_err_summ_bin_param_name": param_name, "edeno_decode_err_summ_bin_params": params},
                skip_duplicates=True)


@schema
class EdenoDecodeErrSummSecKeyParams(PopulationAnalysisSecKeyParamsBase):
    definition = """
    # Parameters for EdenoDecodeErrSumm
    edeno_decode_err_summ_param_name : varchar(200)
    ---
    boot_set_name : varchar(40)
    -> EdenoDecodeErrSummBinParams
    -> BrainRegionUnitsCohortType
    """

    def _default_params(self):
        boot_set_names = self._valid_default_boot_set_names() + self._valid_brain_region_diff_boot_set_names()
        brain_region_units_cohort_types = [
            "0.1_1_rand_target_region_50_iterations_0to9"]
        edeno_decode_err_summ_bin_param_names = EdenoDecodeErrSummBinParams().get_default_param_names()
        return [
            [x, y, z] for x in boot_set_names for y in edeno_decode_err_summ_bin_param_names
            for z in brain_region_units_cohort_types]

    def insert_defaults(self, **kwargs):
        EdenoDecodeErrSummBinParams().insert_defaults(**kwargs)
        super().insert_defaults(**kwargs)

    def get_params(self):
        return {k: self.fetch1(k) for k in get_table_secondary_key_names(self)}


@schema
class EdenoDecodeErrSummSel(SelBase):
    definition = """
    # Selection from upstream tables for EdenoDecodeErrSumm
    -> TrainTestEpochSet
    brain_region_cohort_name : varchar(40)
    curation_set_name : varchar(40)
    -> EdenoDecodeErrSummSecKeyParams
    -> EdenoDecodeErrParams
    -> EdenoDecodeParams
    ---
    upstream_keys : mediumblob
    """

    class EdenoDecodeErr(dj.Part):
        definition = """
        # Achieves upstream dependence on EdenoDecodeErr
        -> EdenoDecodeErrSummSel
        -> EdenoDecodeErr
        """

    # Override parent class method so can insert into part table
    def insert1(self, key, **kwargs):

        # Insert into main table
        super().insert1(key, **kwargs)

        # Insert into part table
        upstream_keys = key.pop("upstream_keys")
        for upstream_key in upstream_keys:
            self.EdenoDecodeErr.insert1({**key, **upstream_key}, skip_duplicates=True)

    def _get_potential_keys(self, key_filter=None, verbose=True, debug_text=False):

        # Define key filter if not passed
        if key_filter is None:
            key_filter = dict()

        # Define curation set (specifies curation names across brain regions) and brain region
        # cohort name (specifies brain regions to combine across)
        brain_region_cohort_name = "all_targeted"
        curation_set_name = "runs_analysis_v1"

        # Define brain_region_units_cohort_types
        brain_region_units_cohort_types = np.unique(EdenoDecodeErrSummSecKeyParams.fetch("brain_region_units_cohort_type"))
        # ...apply passed filter
        if "brain_region_units_cohort_type" in key_filter:
            brain_region_units_cohort_types = [
                x for x in brain_region_units_cohort_types if x == key_filter["brain_region_units_cohort_type"]]

        # Get unique settings of params that we are not combining across
        upstream_table = EdenoDecodeErr()
        column_names = [x for x in upstream_table.primary_key if x not in [
            "nwb_file_name", "brain_region", "brain_region_units_param_name", "curation_name", "train_test_epoch_name"]]
        upstream_table_subset = upstream_table & key_filter  # filter using passed key
        param_sets = unique_table_column_sets(upstream_table_subset, column_names, as_dict=True)

        # Define map from train test epoch set name to boot set name name, so that rat cohort train test epoch
        # sets are matched with summary table param names with rat cohort boot set names, and similarly
        # for non rat cohort case
        train_test_epoch_set_boot_set_name_map = dict()
        # non rat cohort case:
        for train_test_epoch_set_name in TrainTestEpochSet().get_train_test_epoch_set_names(
                    train_test_epoch_set_names_types=["Haight_rotation"]):
            train_test_epoch_set_boot_set_name_map[train_test_epoch_set_name] = [
                "default", "brain_region_diff"]
        # rat cohort case:
        for train_test_epoch_set_name in TrainTestEpochSet().get_train_test_epoch_set_names(
                train_test_epoch_set_names_types=["Haight_rotation_rat_cohort"]):
            train_test_epoch_set_boot_set_name_map[train_test_epoch_set_name] = [
                "default_rat_cohort", "brain_region_diff_rat_cohort"]

        # Limit to train_test_epoch_set_name if passed
        if "train_test_epoch_set_name" in key_filter:
            train_test_epoch_set_boot_set_name_map = {
                k: v for k, v in train_test_epoch_set_boot_set_name_map.items() if k == key_filter[
                    "train_test_epoch_set_name"]}

        keys = []

        for brain_region_units_cohort_type in brain_region_units_cohort_types:

            if verbose:
                print(f"\nOn {brain_region_units_cohort_type}...")

            eps_units_param_name, unit_subset_type, unit_subset_size, unit_subset_iterations = (
                    BrainRegionUnitsCohortType & {
                "brain_region_units_cohort_type": brain_region_units_cohort_type}).fetch1(
                "eps_units_param_name", "unit_subset_type", "unit_subset_size",
                "unit_subset_iterations")

            for train_test_epoch_set_name, boot_set_names in train_test_epoch_set_boot_set_name_map.items():

                if verbose:
                    print(f"\nOn {train_test_epoch_set_name}...")

                nwb_file_names, train_test_epoch_names = (
                            TrainTestEpochSet & {"train_test_epoch_set_name": train_test_epoch_set_name}).fetch1(
                    "nwb_file_names", "train_test_epoch_names")

                for param_set in param_sets:

                    if verbose:
                        print(f"\nOn {param_set}...")

                    # Check if entries exist in EdenoDecodeErr for cohort, and insert into table if so
                    populated = True
                    upstream_keys = []
                    for nwb_file_name, train_test_epoch_name in zip(nwb_file_names, train_test_epoch_names):

                        epochs_description = (TrainTestEpoch & {
                            "nwb_file_name": nwb_file_name, "train_test_epoch_name": train_test_epoch_name}
                                              ).get_epochs_description()

                        brain_regions = (BrainRegionCohort & {
                            "nwb_file_name": nwb_file_name,
                            "brain_region_cohort_name": brain_region_cohort_name}).fetch1("brain_regions")

                        for brain_region in brain_regions:

                            curation_name = (CurationSet & {"nwb_file_name": nwb_file_name,
                                                            "brain_region_cohort_name": brain_region_cohort_name,
                                                            "curation_set_name": curation_set_name}).get_curation_name(
                                brain_region, epochs_description)

                            for unit_subset_iteration in unit_subset_iterations:

                                brain_region_units_param_name = BrainRegionUnitsParams().lookup_param_name(
                                    [eps_units_param_name, epochs_description, unit_subset_type, unit_subset_size,
                                     unit_subset_iteration])
                                upstream_key = {**{
                                    "nwb_file_name": nwb_file_name, "brain_region": brain_region,
                                    "brain_region_units_param_name": brain_region_units_param_name,
                                    "curation_name": curation_name, "train_test_epoch_name": train_test_epoch_name},
                                                **param_set}

                                # Check keys are same as primary key of table
                                check_set_equality(upstream_table.primary_key, upstream_key.keys())

                                # If could not populate brain region units table, continue
                                if len(BrainRegionUnitsFail & upstream_key) > 0:
                                    continue

                                # Indicate whether table populated for current entry
                                populated_ = len(upstream_table & upstream_key) > 0

                                # Print whether entry in table if indicated
                                if debug_text:
                                    print(f"{populated_} {upstream_table.table_name} {upstream_key}")
                                    # Raise error so can investigate no entry
                                    if populated_ is False:
                                        raise Exception

                                # Updated flag for whether all entries in table
                                populated *= populated_

                                # Add upstream key to list
                                upstream_keys.append(upstream_key)

                    # Get part of key corresponding to main table (across epochs, brain regions, etc.)
                    if populated:

                        key = check_return_single_dict([
                            {k: x[k] for k in self.primary_key if k in x} for x in upstream_keys])
                        key.update({
                            'train_test_epoch_set_name': train_test_epoch_set_name,
                            'brain_region_cohort_name': brain_region_cohort_name,
                            'curation_set_name': curation_set_name,
                            "upstream_keys": upstream_keys})

                        if verbose:
                            print(f"Upstream table populated for current key...")

                        # Add in summary table param name
                        for boot_set_name in boot_set_names:
                            for edeno_decode_err_summ_param_name in (EdenoDecodeErrSummSecKeyParams & {
                                "boot_set_name": boot_set_name,
                                "brain_region_units_cohort_type": brain_region_units_cohort_type}).fetch(
                                "edeno_decode_err_summ_param_name"):
                                if verbose:
                                    print(f"Adding keys...")
                                # Add summary table param name to key
                                key.update({'edeno_decode_err_summ_param_name': edeno_decode_err_summ_param_name})
                                # Add key to keys
                                keys.append(copy.deepcopy(key))

        return keys


@schema
class EdenoDecodeErrSumm(PathWellPopSummBase):
    definition = """
    # Summary of decode errors using edeno Bayesian decoder
    -> EdenoDecodeErrSummSel
    ---
    -> nd.common.AnalysisNwbfile
    metric_df_object_id : varchar(40)
    ave_conf_df_object_id : varchar(40)
    boot_ave_df_object_id : varchar(40)
    """

    def _get_brain_region_order_for_pairs(self):

        # Reverse order of brain regions so that directionality consistent with other tables where greater
        # values indicates greater reliability

        return super()._get_brain_region_order_for_pairs()[::-1]

    def make(self, key):

        # Get keys for fetching decoding errors from individual files
        upstream_keys = (EdenoDecodeErrSummSel & key).fetch1("upstream_keys")

        # Get parameters
        boot_set_name, edeno_decode_err_summ_bin_param_name = (EdenoDecodeErrSummSecKeyParams & key).fetch1(
            "boot_set_name", "edeno_decode_err_summ_bin_param_name")

        # Get parameters for binning error
        bin_key = {"edeno_decode_err_summ_bin_param_name": edeno_decode_err_summ_bin_param_name}
        bin_params = (EdenoDecodeErrSummBinParams & bin_key).fetch1("edeno_decode_err_summ_bin_params")

        # Get position_and_maze bin edges
        bins_obj = EdenoDecode().get_pos_bins(bin_params["num_pos_bins"], "test", upstream_keys)

        # Get name of x vals
        vals_index_name = EdenoDecodeErr()._get_vals_index_name()

        # Find mean error in uniform position_and_maze bins during potentially rewarded trials across the entire epoch
        print(f"Getting metric_df...")
        if bin_params["averaging_method"] == "epoch_average" and bin_params["binning_method"] == "uniform":

            pos_bin_nums = np.arange(1, len(bins_obj.bin_edges))  # one indexed for use with np.digitize output

            keep_columns = ["brain_region_units_param_name", "brain_region", "nwb_file_name", "train_test_epoch_name"]

            data_list = []
            for idx, upstream_key in enumerate(upstream_keys):

                close_nwb_files()

                print_iteration_progress(idx, len(upstream_keys))

                # Get description of epochs involved in training/testing to store
                epochs_description = (TrainTestEpoch & upstream_key).get_epochs_description()

                # Get decode error
                error_df = (EdenoDecodeErr & upstream_key).fetch1_dataframe()

                # Digitize test position_and_maze
                pos_obj = (EdenoDecodeErr & upstream_key).get_aligned_map_test_pos()
                pos_digitized = np.digitize(pos_obj.test_position.decode_position, bins=bins_obj.bin_edges)

                # Get mean error in position_and_maze bins
                for bin_num, x_val in zip(pos_bin_nums, bins_obj.bin_centers_df.x_vals):

                    # Find samples in bin on potentially rewarded trials
                    valid_bool = pos_digitized == bin_num

                    # Get mean error in position_and_maze bin
                    error = np.mean(error_df.error[valid_bool].values)
                    nwb_file_name_train_test_epoch_name = \
                        f"{upstream_key['nwb_file_name']}_{upstream_key['train_test_epoch_name']}"
                    subject_id = get_subject_id(upstream_key["nwb_file_name"])

                    # Store
                    data_list.append(tuple(
                        [upstream_key[x] for x in keep_columns] +
                        [subject_id, nwb_file_name_train_test_epoch_name, epochs_description, x_val, error]))

            df_columns = keep_columns + [
                "subject_id", "nwb_file_name_train_test_epoch_name", "epochs_description", vals_index_name, "val"]

            metric_df = df_from_data_list(data_list, df_columns)

        else:
            raise Exception("bin_params settings not accounted for in code")

        # Define bootstrap parameters

        # Params for all
        resample_quantity = "val"

        # Params specific to boot_set_name
        # get parameters table so can access types of boot set names
        params_table = self._get_params_table()

        # 1) Average values
        if boot_set_name in params_table._valid_default_boot_set_names():
            # ...Define columns at which to resample during bootstrap, in order
            resample_levels = ["nwb_file_name_train_test_epoch_name", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = [vals_index_name, "brain_region"]
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        # 2) Average difference values across brain regions
        elif boot_set_name in params_table._valid_brain_region_diff_boot_set_names():

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
            resample_levels = ["nwb_file_name_train_test_epoch_name", "brain_region_units_param_name"]
            # ...Define columns whose values to keep constant (no resampling across these)
            ave_group_column_names = [
                vals_index_name, self._get_joint_column_name(target_column_name), "brain_region_1",
                "brain_region_2"]
            # ...Alter params based on whether rat cohort
            resample_levels, ave_group_column_names = self._alter_boot_params_rat_cohort(
                boot_set_name, resample_levels, ave_group_column_names)

        else:
            raise Exception(f"boot_set_name {boot_set_name} not accounted for in code")

        print(f"Getting bootstrap results...")

        boot_results = hierarchical_bootstrap(metric_df, resample_levels, resample_quantity, ave_group_column_names)

        # Store dfs with results together to save out below
        # ...df with metric values
        results_dict = {"metric_df": metric_df}
        # ...bootstrap results
        for x in ["ave_conf_df", "boot_ave_df"]:
            results_dict[x] = getattr(boot_results, x)

        # Insert into main table
        insert_analysis_table_entry(self, list(results_dict.values()), key)

    def _upstream_table(self):
        return EdenoDecodeErr

    def _get_val_text(self):
        # Define text based on whether brain region difference
        boot_set_name = (EdenoDecodeErrSummSecKeyParams & self.fetch1("KEY")).fetch1("boot_set_name")
        if "brain_region_diff" in boot_set_name:
            return "Decode error\ndifference"
        return "Decode error"

    def _get_val_lims(self, **kwargs):
        # Get a set range for value, e.g. for use in plotting value on same range across plots

        # Get key for querying tables
        key = self.fetch1("KEY")

        # Define limits based on decode variable and whether brain region difference
        decode_variable_param_name = (EdenoDecodeParams & key).fetch1("decode_variable_param_name")
        boot_set_name = (EdenoDecodeErrSummSecKeyParams & key).fetch1("boot_set_name")
        brain_region_diff = "brain_region_diff" in boot_set_name

        if decode_variable_param_name == "ppt_default":
            if brain_region_diff:
                return [-.2, .2]
            else:
                return [0, .2]

        elif decode_variable_param_name == "wa_stay":
            if brain_region_diff:
                return [-.2, .2]
            else:
                return [0, 1]

        else:
            raise Exception(f"decode_variable_param_name {decode_variable_param_name} not accounted for in code")

    def _get_val_ticks(self):
        return self._get_val_lims()

    def _get_xticks(self):
        decode_variable_param_name = (EdenoDecodeParams() & self.fetch1("KEY")).fetch1("decode_variable_param_name")
        if decode_variable_param_name == "ppt_default":
            return [0, .5, 1]
        elif decode_variable_param_name == "wa_default":
            return [0, 1, 2]

    def _get_x_text(self):
        decode_variable_param_name = (EdenoDecodeParams() & self.fetch1("KEY")).fetch1("decode_variable_param_name")
        if decode_variable_param_name == "ppt_default":
            return "Path fraction"
        elif decode_variable_param_name == "wa_default":
            return "Time in delay (s)"

    def _get_x_lims(self):
        decode_variable_param_name = (EdenoDecodeParams() & self.fetch1("KEY")).fetch1("decode_variable_param_name")
        if decode_variable_param_name == "ppt_default":
            return [0, 1]
        elif decode_variable_param_name == "wa_default":
            return [0, 2]

    def _plot_params(self):
        """
        Reverse order of panels denoting brain regions
        :return:
        """

        return {"reverse_brain_region_panel_order": True}

    def extend_plot_results(self, **kwargs):

        super().extend_plot_results(**kwargs)

        # Vertical lines to denote track segments if decoding ppt
        decode_variable_param_name = (EdenoDecodeParams() & self.fetch1("KEY")).fetch1("decode_variable_param_name")
        if decode_variable_param_name == "ppt_default" and not kwargs["empty_plot"]:
            ax = kwargs["ax"]
            plot_junction_fractions(ax)

    def _get_multiplot_fig_name(self, brain_region_vals, keys, plot_name=""):

        # Text to denote whether showed single epochs
        show_single_epoch = return_shared_key_value(keys, "show_single_epoch")
        show_single_epoch_text = format_bool(
            show_single_epoch, "epochs", prepend_underscore=True)

        # Text to denote which brain regions are plotted
        brain_regions_text = f"_{len(brain_region_vals)}areas"

        # Define map to abbreviate parts of param names so that can meet file name character limit
        replace_char_map = self._get_replace_char_map()

        # Text to denote upstream table param names
        param_names_map = {table_name: [] for table_name in np.unique([key["table_name"] for key in keys])}
        for key in keys:
            table_name = key["table_name"]
            meta_param_name = get_table(table_name)()._upstream_table()()._get_params_table()().meta_param_name()
            param_names_map[table_name].append(key[meta_param_name])
        param_names = [
            check_return_single_element(param_names).single_element
            for table_name, param_names in param_names_map.items()]
        upstream_param_text = "_".join(param_names)
        # ...Abbreviate
        upstream_param_text = replace_chars(upstream_param_text, replace_char_map)
        # ...Append underscore
        upstream_param_text = "_" + upstream_param_text

        # Text to denote summary table param name
        param_names_map = {table_name: [] for table_name in np.unique([key["table_name"] for key in keys])}
        for key in keys:
            table_name = key["table_name"]
            meta_param_name = get_table(table_name)()._get_params_table()().meta_param_name()
            param_names_map[table_name].append(key[meta_param_name])
        param_names = [
            check_return_single_element(param_names).single_element
            for table_name, param_names in param_names_map.items()]
        summ_param_text = "_".join(param_names)
        # ...Abbreviate
        summ_param_text = replace_chars(summ_param_text, replace_char_map)
        # ...Append underscore
        summ_param_text = "_" + summ_param_text

        # Text for plot name
        if len(plot_name) > 0:
            plot_name = "_" + plot_name

        # Return name of saved figure
        return f"decode{plot_name}{upstream_param_text}{summ_param_text}{show_single_epoch_text}{brain_regions_text}"

    @staticmethod
    def _get_multiplot_params(**kwargs):
        # Define params for plotting multiple table entries. One param set per table entry.

        # Check that loop inputs passed
        check_membership(
            ["train_test_epoch_set_names", "edeno_decode_param_names", "boot_set_name"], kwargs.keys())

        # Make copy of kwargs to avoid changing outside function
        kwargs = copy.deepcopy(kwargs)

        # Define table name
        table_name = "EdenoDecodeErrSumm"

        # Add table_name to key
        kwargs.update({"table_name": table_name})

        param_sets = []
        # Add to key summary table param name
        # ...Get table
        table = get_table(table_name)
        # ...Add default params
        default_params = table().get_default_table_entry_params()
        kwargs = add_defaults(kwargs, default_params, add_nonexistent_keys=True)
        # ...Add summary table param name
        kwargs.update({"edeno_decode_err_summ_param_name": EdenoDecodeErrSummSecKeyParams().lookup_param_name(
            {k: kwargs[k] for k in [
                "boot_set_name", "edeno_decode_err_summ_bin_param_name", "brain_region_units_cohort_type"]},
            args_as_dict=True)})

        # Loop through decode param names
        for edeno_decode_param_name in kwargs["edeno_decode_param_names"]:
            kwargs.update({"edeno_decode_param_name": edeno_decode_param_name})

            # Loop through train test epoch set names
            for train_test_epoch_set_name in kwargs["train_test_epoch_set_names"]:
                kwargs.update({"train_test_epoch_set_name": train_test_epoch_set_name})

                # Append param set to list
                param_sets.append(copy.deepcopy(kwargs))

        # Return param sets
        return param_sets

    @staticmethod
    def _get_multiplot_ax_key(**kwargs):

        check_membership(
            ["brain_regions_separate", "brain_region_val", "edeno_decode_param_name", "train_test_epoch_set_name"],
            kwargs.keys())

        column_iterable = 0
        if kwargs["brain_regions_separate"]:
            column_iterable = kwargs["brain_region_val"]

        return (kwargs["train_test_epoch_set_name"], kwargs["edeno_decode_param_name"], 0, column_iterable)


def populate_jguidera_edeno_decoder_error(key=None, tolerate_error=False):
    schema_name = "jguidera_edeno_decoder_error"
    for table_name in get_schema_table_names_from_file(schema_name):
        table = eval(table_name)
        populate_insert(table, key=key, tolerate_error=tolerate_error)


def drop_jguidera_edeno_decoder_error():
    schema.drop()
