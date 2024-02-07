import copy
import itertools

import numpy as np
import scipy as sp

from src.jguides_2024.utils.df_helpers import df_from_data_list, df_filter_columns, df_pop
from src.jguides_2024.utils.vector_helpers import vectors_finite_idxs


def check_confidence_interval(confidence_interval, allow_small_values=False):
    if len(confidence_interval) != 2:
        raise Exception(f"Confidence interval must have two elements")
    if np.diff(confidence_interval) < 0:
        raise Exception(f"Second element of confidence interval must be greater than first")
    if np.logical_or(confidence_interval[0] < 0,
        confidence_interval[1] > 100):
        raise Exception(f"Confidence interval must be on [0 100]")
    if np.max(confidence_interval) < 1 and not allow_small_values:  # check that confidence interval reasonable
        raise Exception("Upper bound in confidence interval is less than one; this may be an error."
                        "Percentiles go from 0 to 100, not 0 to 1.")


def estimate_pmf(x_observed):
    # Estimate probability mass function (pmf) for a random variable giving rise to observations x
    x = np.unique(x_observed)  # estimated sample space for X
    p_x = np.asarray([np.sum(x_observed == x_i)/len(x_observed) for x_i in x])  # estimated pmf for X
    return x, p_x


def return_confidence_interval(x, alpha=.05, allow_small_values=False):
    confidence_interval = alpha_to_percentage_confidence_interval(alpha)
    check_confidence_interval(confidence_interval, allow_small_values)  # ensure valid confidence interval
    return np.percentile(x, confidence_interval, axis=0)


def return_confidence_interval_vectorized(shuffles, confidence_interval=[5, 95], allow_small_values=False):
    check_confidence_interval(confidence_interval, allow_small_values)  # ensure valid confidence interval
    return np.asarray(list(zip(*(np.percentile(shuffles, confidence_interval, axis=1)))))


def return_significance_vectorized(confidence_interval_shuffles, measurements):
    return np.prod(confidence_interval_shuffles - np.reshape(measurements, (-1, 1)), axis=1) > 0


def return_confidence_interval_significance_vectorized(shuffles, measurements, confidence_interval=[5, 95]):
    confidence_interval_shuffles = return_confidence_interval_vectorized(shuffles, confidence_interval)
    return confidence_interval_shuffles, return_significance_vectorized(confidence_interval_shuffles, measurements)


def alpha_to_percentage_confidence_interval(alpha):
    return np.asarray([alpha, 1 - alpha])*100


def circular_shuffle(x, num_shuffles):
    x = np.asarray(x)
    split_idxs = np.random.choice(np.arange(0, len(x)), num_shuffles)
    return [np.concatenate([x[split_idx:], x[:split_idx]]) for split_idx in split_idxs]


def partial_correlation_from_cov(cov_matrix):
    """
    Calculate partial correlation from covariance matrix.
    :param cov_matrix: numpy array. Covariance matrix.
    :return: matrix with partial correlations.
    """

    # Initialize array to store partial correlations.
    # i,jth entry is partial correlation between items i and j
    pcorr_matrix = np.empty(cov_matrix.shape)
    pcorr_matrix[:] = np.nan

    # Calculate inverse of covariance matrix
    cov_matrix_inverse = np.linalg.inv(cov_matrix)

    # Calculate partial correlation
    for i in np.arange(0, cov_matrix_inverse.shape[0]):  # for rows
        for j in np.arange(0, cov_matrix_inverse.shape[1]):  # for columns
            # Compute partial correlation
            partial_correlation_temp = -cov_matrix_inverse[i, j] / np.sqrt(
                cov_matrix_inverse[i, i] * cov_matrix_inverse[j, j])
            # Store
            pcorr_matrix[i, j] = partial_correlation_temp

    return pcorr_matrix


def partial_correlation(x):
    """
    Calculate partial correlation of vectors in array
    :param x: array.
    :return: partial correlation.
    """
    return partial_correlation_from_cov(np.cov(x))


def return_bootstrap_sample_idxs(x):
    return return_bootstrap_sample(np.arange(0, len(x)))


def return_bootstrap_sample(x, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if len(np.shape(x)) > 1:  # avoid error when nd array; treat each row as a sample
        return np.asarray(x)[return_bootstrap_sample_idxs(x)]
    return rng.choice(x, size=len(x), replace=True)


def average_difference_confidence_interval(x, y, num_bootstrap_samples=1000, alpha=.05, average_function=None):
    if average_function is None:
        average_function = np.mean
    rng = np.random.default_rng()
    x_boot = rng.choice(x, replace=True, size=(num_bootstrap_samples, len(x)))
    y_boot = rng.choice(y, replace=True, size=(num_bootstrap_samples, len(y)))
    diff_boot = average_function(x_boot, axis=1) - average_function(y_boot, axis=1)
    return return_confidence_interval(diff_boot, alpha)


def average_confidence_interval(x, num_bootstrap_samples=1000, alpha=.05, average_function=None,
                                exclude_nan=False):
    if average_function is None:
        average_function = np.mean
    if exclude_nan:
        x = np.asarray(x)
        x = x[np.invert(np.isnan(x))]
    rng = np.random.default_rng()
    x_boot = rng.choice(x, replace=True, size=(num_bootstrap_samples, len(x)))
    x_ave_boot = average_function(x_boot, axis=1)
    return return_confidence_interval(x_ave_boot, alpha)


def percent_match(x, y):
    if len(x) != len(y):
        raise Exception(f"x and y must same length")
    return np.sum((x == y)/len(x))


def mean_squared_error(x, y, tolerate_nan=True):
    mean_function = np.mean  # default
    if tolerate_nan:
        mean_function = np.nanmean
    if len(x) != len(y):
        raise Exception(f"x and y must be same length")
    ndims = len(np.shape(x))
    if ndims > 2:
        raise Exception(f"x and y must be either 1 or 2 dimensional")
    elif ndims == 1:
        return mean_function((np.asarray(x) - np.asarray(y))**2)
    elif ndims == 2:
        return mean_function((np.asarray(x) - np.asarray(y))**2, axis=1)


def aic(log_likelihood, num_params):
    return 2*num_params - 2*log_likelihood


def finite_corr(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    valid_bool = vectors_finite_idxs([v1, v2])
    return sp.stats.pearsonr(v1[valid_bool], v2[valid_bool])


def random_sample(x, size, replace=True, seed=None, tolerate_error=False):
    if len(x) < size and tolerate_error:
        print(f"Could not subsample x because fewer than size.")
        return
    rng = np.random.default_rng(seed)
    return rng.choice(x, size, replace)


# TODO: carefully check this and associated helper fn
# TODO: consider merging into one function
def recursive_resample(df, resample_levels, resample_quantity):
    return df_from_data_list(_recursive_resample(df, resample_levels, resample_quantity), resample_levels +
                             [resample_quantity])


def _recursive_resample(df, resample_levels, resample_quantity, upstream_level_vals=None):
    if upstream_level_vals is None:
        upstream_level_vals = []
    upstream_level_vals = copy.deepcopy(upstream_level_vals)
    if len(resample_levels) == 0:
        raise Exception(f"Must pass at least one level at which to resample")
    resample_level = resample_levels[0]
    boot_sample = return_bootstrap_sample(np.unique(df[resample_level]))
    # If at final level at which to resample, return tuple with level values and resampled value
    if len(resample_levels) == 1:
        return [tuple(upstream_level_vals + [x, df_pop(df, {resample_level: x}, resample_quantity)])
                for x in boot_sample]
    # Otherwise recursively call the same function
    return list(itertools.chain.from_iterable([_recursive_resample(df_filter_columns(
        df, {resample_level: x}), resample_levels[1:], resample_quantity, upstream_level_vals + [x])
                           for x in boot_sample]))
