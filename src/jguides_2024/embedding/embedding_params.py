from collections import namedtuple

import numpy as np

from src.jguides_2024.utils.df_helpers import df_from_data_list
from src.jguides_2024.utils.dict_helpers import check_same_values_at_shared_keys
from src.jguides_2024.utils.list_helpers import check_return_single_element


class EmbeddingParams:

    def __init__(self):
        self.embedding_params = self._get_embedding_params()

    def _get_embedding_params(self):
        data_list = []
        ZoomScaleFactor = namedtuple("ZoomScaleFactor", "x1 x2 y1 y2 z1 z2")

        # J1620210606, 0.1 kernel SD, 3D
        base_k = {
          'fr_vec_emb_param_name': '15^3',
          'nwb_file_name': 'J1620210606_.nwb',
          'res_epoch_spikes_sm_param_name': '0.1',
          'res_time_bins_pool_cohort_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb',
          'zscore_fr': False}
        k_ctx = {'curation_name': 'raw data valid times no premaze no home_3'}
        k_hpc = {'curation_name': 'raw data valid times no premaze no home no sleep_3'}

        # epoch 4
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run2_target_region',
          'epochs_id': '4'}}
        scale_factor = .15
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "mPFC_targeted"}}, [], 82, -128, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], -112, -128, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_hpc, **{"brain_region": "CA1_targeted"}}, [], 63, -162,  ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # epoch 6
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run3_target_region',
          'epochs_id': '6'}}
        scale_factor = .15
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "mPFC_targeted"}}, [], 82, 90, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], -114, -62, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_hpc, **{"brain_region": "CA1_targeted"}}, [], -130, -72, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # epoch 8
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run4_target_region',
          'epochs_id': '8'}}
        scale_factor = .85
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "mPFC_targeted"}}, [], -86, 78, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], -122, -46, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_hpc, **{"brain_region": "CA1_targeted"}}, [], -170, -32, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # epoch 10
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run5_target_region',
          'epochs_id': '10'}}
        scale_factor = .15
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "mPFC_targeted"}}, [], -175, -174, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], -110, -118, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
         ({**k, **k_hpc, **{"brain_region": "CA1_targeted"}}, [], -136, -78, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # mango20211207, 0.1 kernel SD, 3D
        base_k = {
          'fr_vec_emb_param_name': '15^3',
          'nwb_file_name': 'mango20211207_.nwb',
          'res_epoch_spikes_sm_param_name': '0.1',
          'res_time_bins_pool_cohort_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb',
          'zscore_fr': False}
        k_ctx = {'curation_name': 'raw data valid times no premaze no home_3'}

        # epoch 6
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run3_target_region',
          'epochs_id': '6'}}
        scale_factor = .12
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "mPFC_targeted"}}, [], -103, -70, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], -80, 106, ZoomScaleFactor(
            scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ({**k, **k_hpc, **{"brain_region": "CA1_targeted"}}, [], -116, 90, ZoomScaleFactor(
            scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # june20220420, 0.1 kernel SD, 3D
        base_k = {
          'fr_vec_emb_param_name': '15^3',
          'nwb_file_name': 'june20220420_.nwb',
          'res_epoch_spikes_sm_param_name': '0.1',
          'res_time_bins_pool_cohort_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb',
          'zscore_fr': False}
        k_ctx = {'curation_name': 'raw data valid times no premaze no home_3'}

        # epoch 6
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run3_target_region',
          'epochs_id': '6'}}
        scale_factor = 0
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "mPFC_targeted"}}, [], 8, 177, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], -65, 90, ZoomScaleFactor(
            scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # epoch 10
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run5_target_region',
          'epochs_id': '10'}}
        scale_factor = .15
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "mPFC_targeted"}}, [], 70, -10, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], -86, 88, ZoomScaleFactor(
            scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # peanut20201108, 0.1 kernel SD, 3D
        base_k = {
          'fr_vec_emb_param_name': '15^3',
          'nwb_file_name': 'peanut20201108_.nwb',
          'res_epoch_spikes_sm_param_name': '0.1',
          'res_time_bins_pool_cohort_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb',
          'zscore_fr': False}
        k_ctx = {'curation_name': 'raw data valid times no premaze no home_3'}

        # epoch 8
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run4_target_region',
          'epochs_id': '8'}}
        k_hpc = {'curation_name': 'pos 7 valid times no premaze no home no sleep_3'}
        scale_factor = .08
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], 50, -50, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ({**k, **k_hpc, **{"brain_region": "CA1_targeted"}}, [], 126, -129, ZoomScaleFactor(
            scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        # fig20211109, 0.1 kernel SD, 3D
        base_k = {
          'fr_vec_emb_param_name': '15^3',
          'nwb_file_name': 'fig20211109_.nwb',
          'res_epoch_spikes_sm_param_name': '0.1',
          'res_time_bins_pool_cohort_param_name': 'ResEpochTimeBins#0.1#EpochInterval_no_comb',
          'zscore_fr': False}
        k_ctx = {'curation_name': 'raw data valid times no premaze no home_3'}

        # epoch 8
        k = {**base_k, **{
          'brain_region_units_param_name': '0.1_run4_target_region',
          'epochs_id': '8'}}
        scale_factor = .15
        data_list += [
         ({**k, **k_ctx, **{"brain_region": "OFC_targeted"}}, [], 76, -42, ZoomScaleFactor(
             scale_factor, scale_factor, scale_factor, scale_factor, scale_factor, scale_factor)),
        ]

        return df_from_data_list(data_list, ["key", "reverse_axes", "azim", "elev", "zoom_scale_factor", "zoom_region"])

    def _get_matching_params_entry(self, key, tolerate_no_entry=False):

        target_embedding_params_keys = check_return_single_element(
            [x.keys() for x in self.embedding_params["key"]]).single_element

        df_key = {k: key[k] for k in target_embedding_params_keys if k in key}
        valid_bool = [check_same_values_at_shared_keys(
            [x, df_key], tolerate_error=True) for x in self.embedding_params["key"].values]

        num_matching_entries = np.sum(valid_bool)

        # Raise error if no corresponding entry if indicated
        if num_matching_entries == 0 and not tolerate_no_entry:
            raise Exception("No matching entries in embedding params for passed key")

        # Raise error if multiple corresponding embedding params for passed key
        if num_matching_entries > 1:
            raise Exception(f"Multiple matching entries in embedding params for passed key")

        return self.embedding_params[valid_bool]

    def apply_embedding_params(self, key, kwargs, ax=None, param_names=None, verbose=False):
        # Apply params if single corresponding entry for passed key

        # Define inputs if not passed
        if param_names is None:
            param_names = ["reverse_axes", "azim", "elev"]

        # Make sure all necessary inputs passed for params to be updated
        if "zoom" in param_names and ax is None:
            raise Exception(f"Must pass ax to apply zoom params")

        if verbose:
            print(f"Applying params for {key}...")

        # Get params
        df_subset = self._get_matching_params_entry(key).iloc[0]

        # Apply params

        # Flip axes so embeddings epochs in similar locations; facilitates qualitative comparison, and
        # valid since axes have arbitrary units and no inherent meaning to x / y position_and_maze
        kwargs.update({"reverse_axes": df_subset.reverse_axes})

        # Add azimuth and elevation
        for param_name in ["azim", "elev"]:
            if param_name in param_names:
                val = df_subset[param_name]
                if val is not None:
                    kwargs[param_name] = val

        # Set zoom
        if "zoom" in param_names:
            for axis_name in ["x", "y", "z"]:
                x1, x2 = getattr(ax, f"get_{axis_name}lim")()
                lim_width = x2 - x1
                delta1 = getattr(df_subset.zoom_scale_factor, f"{axis_name}1") * lim_width
                delta2 = getattr(df_subset.zoom_scale_factor, f"{axis_name}2") * lim_width
                new_lims = [x1 + delta1, x2 - delta2]
                getattr(ax, f"set_{axis_name}lim")(new_lims)

        return kwargs
