import numpy as np
import pandas as pd
import scipy as sp

from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool as in_intervals_bool


def smooth_intervals(x, trial_intervals, sigma):
    """
    Smooth values of x that fall within trials. Smooth values in each trial separately
    :param x: pandas Series with data to smooth
    :param trial_intervals: list of intervals of form (start, stop)
    :param sigma: float, standard deviation of Gaussian kernel used to smooth
    :return: smoothed values of x that fall within trial_intervals
    """

    x_smoothed = pd.Series([np.nan]*len(x), index=x.index)
    for trial_interval in trial_intervals:
        trial_bool = in_intervals_bool(x.index, [trial_interval])
        x_smoothed[trial_bool] = sp.ndimage.gaussian_filter(x[trial_bool], sigma=sigma, order=0)

    return x_smoothed
