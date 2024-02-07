import numpy as np
from src.jguides_2024.time_and_trials.jguidera_time_bins import EpochTimeBinsParams
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptDigParams

from src.jguides_2024.utils.dict_helpers import dict_comprehension


def get_glm_default_params_map():
    """
    Return map from parameter name to default parameter value for GLM analysis
    :return: dictonary with parameter names (keys) and values (values)
    """

    from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel
    from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWADigParams, \
        TimeRelWADigSingleAxisParams
    from src.jguides_2024.glm.jguidera_basis_function import RaisedCosineBasisParams
    from src.jguides_2024.position_and_maze.jguidera_ppt_interp import PptDigParams

    time_bin_width = .1
    epoch_time_bins_param_name = EpochTimeBinsParams().lookup_param_name([time_bin_width])

    # DELAY
    delay_time_bins_shorthand = "delay_stay_20ms"
    delay_res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand(
        delay_time_bins_shorthand)
    time_covariate_bin_width = .1  # this is distinct from time bin related to sampling rate
    time_rel_wa_dig_param_name = TimeRelWADigParams().lookup_param_name([time_covariate_bin_width])
    time_rel_wa_dig_single_axis_param_name = TimeRelWADigSingleAxisParams().lookup_param_name([0, 2])

    # PATH
    path_time_bins_shorthand = "path_100ms"
    path_res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand(path_time_bins_shorthand)
    res_time_bins_pool_param_names = [delay_res_time_bins_pool_param_name, path_res_time_bins_pool_param_name]
    ppt_bin_width = .05
    path_raised_cosine_basis_param_name = RaisedCosineBasisParams().lookup_param_name_from_shorthand("ppt")
    ppt_dig_param_name = PptDigParams().lookup_param_name([ppt_bin_width])

    return {
        "epoch_time_bins_param_name": epoch_time_bins_param_name,
        # delay
        "time_bin_width": time_bin_width, "delay_time_bins_shorthand": delay_time_bins_shorthand,
        "delay_res_time_bins_pool_param_name": delay_res_time_bins_pool_param_name,
        "time_rel_wa_dig_single_axis_param_name": time_rel_wa_dig_single_axis_param_name,
        # path
        "path_time_bins_shorthand": path_time_bins_shorthand, "path_res_time_bins_pool_param_name":
            path_res_time_bins_pool_param_name, "res_time_bins_pool_param_names": res_time_bins_pool_param_names,
        "ppt_bin_width": ppt_bin_width, "ppt_dig_param_name": ppt_dig_param_name,
        "time_rel_wa_dig_param_name": time_rel_wa_dig_param_name,
        "path_raised_cosine_basis_param_name": path_raised_cosine_basis_param_name,
    }


def get_glm_default_param(param_name):
    """
    Get the value of a single default parameter for GLM analysis
    :param param_name: str. name of parameter
    :return: default value
    """

    return get_glm_default_params_map()[param_name]


def get_glm_default_params(param_names, as_dict=True):
    """
    Get the values of several default parameters for GLM analysis
    :param param_names: list of strings with names of parameters
    :param as_dict: True to return as dictionary where key is parameter name and value is value of parameter
    :return: param values (as list or as dictionary with parameter names as keys)
    """

    params_map = get_glm_default_params_map()
    default_params = [params_map[param_name] for param_name in param_names]

    # If returning as dictionary, convert param names for which there are multiple to version without prefix if only
    # one of multiple passed, so can use dictionary as key to query tables
    if as_dict:
        # Return res_time_bins_pool_param_name without prefix (delay or path) if only one passed
        res_time_bins_pool_param_names = [
            "path_res_time_bins_pool_param_name", "delay_res_time_bins_pool_param_name"]
        if np.sum([x in param_names for x in res_time_bins_pool_param_names]) == 1:
            param_names = ["res_time_bins_pool_param_name" if x in res_time_bins_pool_param_names
                           else x for x in param_names]

        # Return raised cosine basis param name without prefix if only one passed else
        raised_cosine_basis_param_names = [
            "path_raised_cosine_basis_param_name", "delay_raised_cosine_basis_param_name"]
        if np.sum([x in param_names for x in raised_cosine_basis_param_names]) == 1:
            param_names = ["raised_cosine_basis_param_name" if x in raised_cosine_basis_param_names
                           else x for x in param_names]
        return dict_comprehension(param_names, default_params)

    # Otherwise just return params unchanged
    return default_params


def get_fr_vec_default_params_map():
    """
    Get map from parameters names to default values for firing rate vector analysis
    :return: dictonary with parameter names (keys) and values (values)
    """

    from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolSel
    from src.jguides_2024.spikes.jguidera_res_spikes import ResEpochSpikesSmParams
    from src.jguides_2024.time_and_trials.jguidera_time_relative_to_well_event import TimeRelWADigParams, \
        TimeRelWADigSingleAxisParams

    time_bins_shorthand = "epoch_100ms"
    res_time_bins_pool_param_name = ResTimeBinsPoolSel().lookup_param_name_from_shorthand(time_bins_shorthand)
    time_rel_wa_dig_single_axis_param_name = TimeRelWADigSingleAxisParams().lookup_param_name([0, 2])
    res_epoch_spikes_sm_param_name = ResEpochSpikesSmParams().lookup_param_name([.1])
    ppt_dig_param_name = PptDigParams().lookup_param_name([.05])

    return {"time_bins_shorthand": time_bins_shorthand, "res_time_bins_pool_param_name": res_time_bins_pool_param_name,
            "res_epoch_spikes_sm_param_name": res_epoch_spikes_sm_param_name,
            "time_rel_wa_dig_param_name": TimeRelWADigParams().lookup_param_name([.25]),
            "time_rel_wa_dig_single_axis_param_name": time_rel_wa_dig_single_axis_param_name,
            "ppt_dig_param_name": ppt_dig_param_name}


def get_fr_vec_default_param(param_name):
    return get_fr_vec_default_params_map()[param_name]


# Define defaults for GLM analysis
# Note we do not need bin_width to match time_bin_width in shorthand_params_map; bin_width here is resolution
# of covariate digitization, whereas time_bin_width corresponds to space between time samples. It just happens
# that in this case, covariate has unit time
glm_params = get_glm_default_params_map()
default_params = {k: glm_params[k] for k in [
    "delay_res_time_bins_pool_param_name", "time_rel_wa_dig_param_name",
    "time_rel_wa_dig_single_axis_param_name"]}