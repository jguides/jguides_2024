import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.jguides_2024.utils.dict_helpers import add_pandas_index_to_dict
from src.jguides_2024.utils.list_helpers import zip_adjacent_elements
from src.jguides_2024.utils.plot_helpers import format_ax, get_ax_for_layout
from src.jguides_2024.utils.series_helpers import check_series
from src.jguides_2024.utils.vector_helpers import linspace, vector_midpoints, check_monotonic_increasing, \
    unpack_single_element, check_vectors_close


def raised_cosine(x, p):
    """
    Return raised cosines as defined in Park et al. 2014, "Encoding and decoding in parietal cortex during
    sensorimotor decision-making" and corresponding code: https://github.com/pillowlab/GLMspiketraintutorial.
    :param x: array with values at which to evaluate cosine
    :param p: period
    :return: raised cosines
    """
    return (abs(x/p) < .5)*(np.cos(x*2*np.pi/p)*.5 + .5)


def format_basis_name(covariate_group_name, basis_num):
    return f"{covariate_group_name}_basis_{basis_num}"


def return_indicator_basis_df(basis_bin_edges, covariate_group_name):
    check_monotonic_increasing(basis_bin_edges)
    return pd.DataFrame(np.eye(len(basis_bin_edges) - 1),
                 columns=[format_basis_name(covariate_group_name, basis_num) for basis_num in
                          np.arange(0, len(basis_bin_edges) - 1)],
                 index=zip_adjacent_elements(basis_bin_edges))


def raised_cosine_basis(num_bins, num_basis_functions):
    """
    Return raised cosines as defined in Park et al. 2014, "Encoding and decoding in parietal cortex during
    sensorimotor decision-making" and corresponding code: https://github.com/pillowlab/GLMspiketraintutorial.
    :param num_bins: number of bins
    :param num_basis_functions: number of basis functions
    :return: raised cosines
    """
    centers_dist = num_bins / (3 + num_basis_functions)  # distance between cosine centers
    cos_width = 4*centers_dist  # width of cosine is 4x distance between centers
    cos_centers = 2*centers_dist + centers_dist*np.arange(0, num_basis_functions)
    basis_indices = np.tile(np.arange(1, num_bins + 1), (num_basis_functions, 1)).T
    x = basis_indices - np.tile(cos_centers, (num_bins, 1))
    return raised_cosine(x, cos_width)


def return_raised_cosine_basis_df(num_bins, num_basis_functions, covariate_group_name, pad_basis=False):
    basis = raised_cosine_basis(num_bins, num_basis_functions)
    if pad_basis:
        basis = zero_pad_basis(basis)
    return pd.DataFrame(basis, columns=[format_basis_name(covariate_group_name, basis_num)
                                          for basis_num in np.arange(0, num_basis_functions)])


def zero_pad_basis(basis_fn, num_pre_zeros=1, num_post_zeros=1):
    """
    Add leading and/or lagging zeros to basis functions
    :param basis_fn: array with basis functions. Shape: (number of points in each basis function, number of basis functions)
    :param num_pre_zeros: number of zeros to add before each basis
    :param num_post_zeros: number of zeros to add after each basis
    :return: basis functions array with padding zeros
    """
    zeros_pre = np.zeros((num_pre_zeros, np.shape(basis_fn)[1]))
    zeros_post = np.zeros((num_post_zeros, np.shape(basis_fn)[1]))
    return np.vstack((zeros_pre, basis_fn, zeros_post))


def plot_basis_functions(basis_functions, basis_functions_x=None, basis_function_names=None):
    """
    Plot basis functions
    :param basis_functions: arr with shape (num samples in each basis function, num basis functions)
    :param basis_functions_x: x values corresponding to basis function values
    """
    if basis_functions_x is None:
        basis_functions_x = np.arange(0, np.shape(basis_functions)[0])
    if basis_function_names is None:
        basis_function_names = [""]*np.shape(basis_functions)[1]
    num_basis_functions = np.shape(basis_functions)[1]
    fig, axes = plt.subplots(num_basis_functions, 1,
                             figsize=(5, 2*num_basis_functions))
    if len(basis_function_names) != np.shape(basis_functions)[1]:
        raise Exception(f"basis_function_names must be same length as basis_functions")
    for idx, (x, bf_name) in enumerate(zip(basis_functions.T, basis_function_names)):
        if num_basis_functions == 1:
            ax = axes
        else:
            ax = axes[idx]
        ax.plot(basis_functions_x, x, '.-', color="black")
        ax.set_title(bf_name)
    fig.tight_layout()


def sample_basis_function(basis_function, sample_idxs, tolerate_outside_basis_domain):
    # If tolerating idxs outside the idxs of basis_function, return zero when idx above basis_function length
    if tolerate_outside_basis_domain:
        sampled_basis = np.zeros(len(sample_idxs))  # initialize
        valid_sample_idxs = np.where(sample_idxs < len(basis_function))[0]
        sampled_basis[valid_sample_idxs] = basis_function[sample_idxs[valid_sample_idxs]]
        return sampled_basis
    return basis_function[sample_idxs]


def sample_basis_functions(digitized_vector, basis_functions, tolerate_outside_basis_domain):
    check_series(digitized_vector, require_index_name=True)
    return pd.DataFrame.from_dict(add_pandas_index_to_dict(
        {basis_name: sample_basis_function(
            basis_function=basis_functions[basis_name].values,
            sample_idxs=digitized_vector.values - 1,  # digitized vector is one indexed
            tolerate_outside_basis_domain=tolerate_outside_basis_domain)
            for basis_name in basis_functions.columns},
        pandas_obj=digitized_vector)).set_index(digitized_vector.index.name)


class RaisedCosineBasis:
    def __init__(self, domain, bin_width, num_basis_functions, covariate_group_name):
        """
        Container for evenly spaced raised cosine basis for a covariate on a specified domain.
        :param domain: domain of basis functions. [start stop].
        :param bin_width: width of bins in which basis function defined.
        :param num_basis_functions: int. Number of functions in basis.
        :param covariate_group_name: str. Name of covariate.
        """
        self.domain = domain
        self.bin_width = bin_width
        self.num_basis_functions = num_basis_functions
        self.covariate_group_name = covariate_group_name
        self.basis_bin_edges = self._get_basis_bin_edges()
        self.basis_bin_centers = vector_midpoints(self.basis_bin_edges)
        self.basis_functions = self._get_basis_functions()

    def _check_inputs(self):
        if len(self.domain) != 2:
            raise Exception(f"domain for basis functions must have exactly two members")

    def _get_basis_bin_edges(self):
        return linspace(self.domain[0], self.domain[1], self.bin_width)

    def _get_basis_functions(self):
        return return_raised_cosine_basis_df(num_bins=len(self.basis_bin_edges) - 1,
                                             num_basis_functions=self.num_basis_functions,
                                             covariate_group_name=self.covariate_group_name)

    def plot_basis_functions(self):
        plot_basis_functions(self.basis_functions.to_numpy(),
                             basis_functions_x=self.basis_bin_centers,
                             basis_function_names=self.basis_functions.columns)

    def sample_basis_functions(self, digitized_vector, tolerate_outside_basis_domain=True):
        return sample_basis_functions(digitized_vector, self.basis_functions, tolerate_outside_basis_domain)


class SampledRaisedCosineBasis:
    def __init__(self, domain, bin_width, num_basis_functions, covariate_group_name, measurements,
                 tolerate_outside_basis_domain=True):
        """
        Container for sampled raised cosine basis for a covariate.
        :param domain: domain of basis functions. [start stop].
        :param bin_width: width of bins in which basis function defined.
        :param num_basis_functions: int. Number of functions in basis.
        :param covariate_group_name: str. Name of covariate.
        :param tolerate_outside_basis_domain: bool.
               If True, then basis function evaluates to zero when measurement outside domain
        """
        self.measurements = measurements
        self.tolerate_outside_basis_domain = tolerate_outside_basis_domain
        self._check_inputs(domain)
        self.basis = self._get_basis(domain, bin_width, num_basis_functions, covariate_group_name)
        self.digitized_measurements = self._get_digitized_measurements()
        self.sampled_basis = self.basis.sample_basis_functions(self.digitized_measurements,
                                       tolerate_outside_basis_domain=self.tolerate_outside_basis_domain)

    def _check_inputs(self, domain):
        # Check measurements is a series
        check_series(self.measurements, require_index_name=True)
        if not self.tolerate_outside_basis_domain and not all(
                self.measurements.between(domain[0], domain[1], inclusive="left")):  # inclusive left to match np.digitize behavior
            raise Exception(f"All measurements must be in the half-open set: [domain[0], domain[1])")

    def _get_basis(self, domain, bin_width, num_basis_functions, covariate_group_name):
        return RaisedCosineBasis(domain, bin_width, num_basis_functions, covariate_group_name)

    def _get_digitized_measurements(self):
        return pd.Series(np.digitize(self.measurements.values, bins=self.basis.basis_bin_edges),
                         index=self.measurements.index)

    def plot_measurements(self):
        fig, axes = plt.subplots(2, 1, figsize=(15, 3), sharex=True)
        axes[0].plot(self.digitized_measurements)
        axes[1].plot(self.measurements)


class RelativeMeasureBasis:
    def __init__(self, covariate_group_name, basis_bin_lags):
        """
        Container for relative measure basis: measurements in bins relative to reference bins.
        :param covariate_group_name: str. Name of covariate.
        :param basis_bin_lags: list with integers denoting index of covariate bin relative to predicted bin.
               One basis for each.
        """
        self.covariate_group_name = covariate_group_name
        self.basis_bin_lags = basis_bin_lags  # [edge_1, edge_2, edge_3, ...]
        self._get_defaults()
        self._check_inputs()
        self.num_basis_functions = len(self.basis_bin_lags)
        self.basis_functions = self._get_basis_functions()
        self.basis_function_bin_lag_map = self._get_basis_function_bin_lag_map()

    def _get_defaults(self):
        if self.basis_bin_lags is None:
            self.basis_bin_lags = [0]

    def _check_inputs(self):
        if not all([isinstance(x, int) for x in self.basis_bin_lags]):
            raise Exception(f"All elements of basis_bin_lags must be integers")

    def _get_basis_functions(self):
        return pd.DataFrame(np.eye(self.num_basis_functions),
                            columns=[format_basis_name(self.covariate_group_name, idx) for idx in self.basis_bin_lags],
                            index=self.basis_bin_lags)

    def _get_basis_function_bin_lag_map(self):
        """Make convenient mapping from basis function name to bin lag"""
        return {basis_function_name: unpack_single_element(self.basis_functions.index[
            unpack_single_element(np.where(self.basis_functions[basis_function_name] == 1))])
                for basis_function_name in self.basis_functions.columns}

    def plot_basis_functions(self):
        num_basis_functions = self.num_basis_functions
        fig, axes = plt.subplots(num_basis_functions, 1,
                                 figsize=(5, 2 * num_basis_functions))
        fig.tight_layout()
        for basis_function_name, ax in zip(self.basis_functions.columns, axes):
            ax.stem(self.basis_functions[basis_function_name])
            format_ax(ax=ax, title=basis_function_name)

    def sample_basis_functions(self, external_measure, reference_measure=None):
        # Check inputs are series
        check_series(external_measure, require_index_name=False)
        # If reference measure passed, just use to make sure external and reference measures on effectively same index
        if reference_measure is not None:
            check_series(reference_measure, require_index_name=False)
            # Check indices close (ideally they are same but could have imprecision from float datatype)
            check_vectors_close([reference_measure.index, external_measure.index], epsilon=.0001)

        # "Sample" basis
        time_bin_centers = external_measure.index
        sampled_basis_arr = np.zeros((len(time_bin_centers), len(self.basis_bin_lags)))
        sampled_basis_arr[:] = np.nan
        for basis_bin_lag_idx, (basis_function_name,
                                basis_bin_lag) in enumerate(self.basis_function_bin_lag_map.items()):
            if basis_bin_lag == 0:
                sampled_basis_arr[:, basis_bin_lag_idx] = external_measure.values
            if basis_bin_lag < 0:
                sampled_basis_arr[-basis_bin_lag:, basis_bin_lag_idx] = external_measure.values[:basis_bin_lag]
            elif basis_bin_lag > 0:
                sampled_basis_arr[:-basis_bin_lag, basis_bin_lag_idx] = external_measure.values[basis_bin_lag:]
        return pd.DataFrame(sampled_basis_arr, index=time_bin_centers, columns=list(self.basis_function_bin_lag_map.keys()))


class SampledRelativeMeasureBasis:
    def __init__(self, covariate_group_name, external_measure, basis_bin_lags=None, reference_measure=None):
        """
        Container for sampled relative measure basis: measurements in bins relative to reference bins.
        :param covariate_group_name: str. Name of covariate.
        :param basis_bin_lags: list with integers denoting index of covariate bin relative to predicted bin.
               One basis for each.
        :param external_measure: series with predictor measurements in bins.
        :param refernce_measure: series with predicted measurements in bins. Optional.
        """

        self.basis = self._get_basis(covariate_group_name, basis_bin_lags)
        self.external_measure = external_measure
        self.reference_measure = reference_measure
        self.sampled_basis = self._sample_basis()

    def _get_basis(self, covariate_group_name, basis_bin_lags):
        return RelativeMeasureBasis(covariate_group_name=covariate_group_name, basis_bin_lags=basis_bin_lags)

    def _sample_basis(self):
        return self.basis.sample_basis_functions(external_measure=self.external_measure,
                                                 reference_measure=self.reference_measure)

    def plot_events(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(15, 3), sharex=True)
        for idx, (measure_name, event_times, color) in enumerate(
                zip(["external measure", "reference measure"], [self.external_measure, self.reference_measure],
                    ["gray", "tan"])):
            if event_times is None:
                continue
            ax.plot(event_times, '.', label=measure_name, alpha=.7, color=color)

    def plot_sampled_basis(self):
        fig, axes = plt.subplots(len(self.basis.basis_bin_lags), 1, figsize=(15, 3), sharex=True)
        for basis_function_name_idx, basis_function_name in enumerate(self.sampled_basis.columns):
            ax = get_ax_for_layout(axes, basis_function_name_idx)
            self.plot_events(ax=ax)
            ax.plot(self.sampled_basis[basis_function_name], 'x', color="red", alpha=.7, label=basis_function_name)
            ax.legend()
