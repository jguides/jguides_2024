import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from src.jguides_2024.utils.interpolate_helpers import interpolate_at_average_sampling_rate
from src.jguides_2024.utils.make_bins import make_int_bin_edges
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals
from src.jguides_2024.utils.vector_helpers import check_uniform_spacing, vector_midpoints, \
    check_monotonic_increasing


# TODO: consider naming so that easy to understand purpose of each function.

def make_rate_map_from_event_times_trodes_epoch(event_times,
                                                measurement_times,
                                                measurements,
                                                bin_width=None,
                                                bin_edges=None,
                                                verbose=False,
                                                error_tolerance=None):
    event_times = event_times[np.logical_and(event_times >= measurement_times[0],
                                             event_times <= measurement_times[
                                                 -1])]  # filter for event times during period when measurements exist
    measurements_at_events = np.asarray([np.interp(x=event_times, xp=measurement_times, fp=r) for r in
                                         measurements])  # interpolate measurements at event times
    return make_rate_map_trodes_epoch(measurements=measurements,
                                      measurement_times=measurement_times,
                                      measurements_at_events=measurements_at_events,
                                      bin_width=bin_width,
                                      bin_edges=bin_edges,
                                      verbose=verbose,
                                      error_tolerance=error_tolerance)


def calculate_occupancy_trodes_epoch_constant_sampling_rate(measurements,
                                                            measurement_times,
                                                            bin_edges,
                                                            error_tolerance):
    """
    Calculate occupancy (time spent in each measurement bin) for Trodes epoch.
    In calculating sampling rate, account for half second pause in camera acquisition soon after recording start.
    :param measurements: array with measurements
    :param measurement_times: times of measurements
    :param sampling_rate: constant sampling rate
    :param bin_edges: list with bin edges: [[dim1 bin edges], ...]
    :param error_tolerance: fraction of average distance between samples that difference between average
                        distance between samples and maximum distance between samples can deviate from without
                        raising error
    :return: array with occupancy
    """

    # Find camera time differences consistent with Trodes pausing camera acquisition at
    # recording start. Exclude these from consideration below.
    from src.jguides_2024.datajoint_nwb_utils.edge_case_helpers import identify_trodes_camera_pause_at_epoch_start
    valid_diff_mask = np.invert(identify_trodes_camera_pause_at_epoch_start(camera_times=measurement_times,
                                                                 epoch_start_time=measurement_times[0]))[:-1]
    mean_times_diff = np.mean(np.diff(measurement_times)[valid_diff_mask])  # mean time between camera samples
    sampling_rate = 1 / mean_times_diff   # convert to sampling rate
    check_uniform_spacing(measurement_times,
                          valid_diff_mask=valid_diff_mask,
                          error_tolerance=error_tolerance)  # check camera samples uniformly spaced
    return calculate_occupancy_constant_sampling_rate(measurements,
                                                      sampling_rate,
                                                      bin_edges)


def calculate_occupancy_constant_sampling_rate(measurements,
                                               sampling_rate,
                                               bin_edges):
    """
    Calculate occupancy (time spent in each measurement bin) assuming a constant measurement sampling rate.
    :param measurements: array with measurements
    :param sampling_rate: constant sampling rate
    :param bin_edges: list with bin edges: [[dim1 bin edges], ...]
    :return: array with occupancy
    """

    measurements_bin_counts, _ = np.histogramdd(measurements.T, bins=bin_edges)  # histogram measurements

    return measurements_bin_counts / sampling_rate  # time spent in each measurement bin


def calculate_occupancy_uneven_sampling_rate(measurements,
                                             measurement_times,
                                             bin_edges):
    # Check inputs valid
    check_monotonic_increasing(measurement_times)
    check_monotonic_increasing(bin_edges)
    # Define measurement durations
    # Approach: taking time window around each measurement equal to the sum of 1) half the window from sample before to the
    # current sample and 2) half the window from sample after to the current sample
    measurement_times_midpoints = vector_midpoints(measurement_times)
    dt_1 = measurement_times_midpoints - measurement_times[:-1]  # duration from midpoint to previous time
    dt_2 = measurement_times[1:] - measurement_times_midpoints  # duration from time to previous midpoint
    measurement_durations = np.asarray([np.nan]*len(measurement_times))  # initialize array for measurement durations
    measurement_durations[0] = dt_1[0]
    measurement_durations[-1] = dt_2[-1]
    measurement_durations[1:-1] = dt_1[1:] + dt_2[:-1]
    if any(np.isnan(measurement_durations)):
        raise Exception(f"nans in measurement durations")
    # Bin measurements
    measurements_bin = np.digitize(measurements, bins=bin_edges) - 1  # substract 1 to zero index
    if any(measurements_bin < 0) or any(measurements_bin > len(bin_edges) - 1):  # check that measurements within bins
        raise Exception(f"At least some measurements fell outside bin_edges")
    # Sum durations for each bin
    bins = np.arange(0, len(bin_edges) - 1)
    occupancy = np.asarray([np.sum(measurement_durations[measurements_bin == b]) for b in bins])
    if np.sum(occupancy) != (measurement_times[-1] - measurement_times[0]):  # sanity check
        raise Exception("Occupancy should sum to total duration of measurements")
    return occupancy


def calculate_occupancy_interp(measurements,
                               measurement_times,
                               bin_edges,
                               error_tolerance=.001):
    """
    Intended use case is to compute occupancy (time spent in each measurement bin) when sampling rate of measurements
    is uneven. Approach is to interpolate measurements at fixed sampling rate.
    Note that interpolation can lead to regions that would have otherwise had zero occupancy having non-zero occupancy.
    :param measurements: array with measurements
    :param measurement_times: array with measurement times
    :param bin_edges: list with bin edges: [[dim1 bin edges], ...]
    :param error_tolerance: fraction of average time between samples that difference between average
                            time between samples and maximum time between samples can deviate from without raising
                            error. Use to ensure that not performing this interpolation operation on a vector with
                            very uneven time spacing between samples.
    :return: array with occupancy
    """

    # Sample measurements at regular interval
    measurements = np.asarray([interpolate_at_average_sampling_rate(
        t_original=measurement_times, y_original=r, error_tolerance=error_tolerance)[1] for r in measurements])
    return calculate_occupancy_constant_sampling_rate(measurements, measurement_times, bin_edges)


def make_rate_map_trodes_epoch(
        measurements, measurement_times, measurements_at_events, bin_width=None, bin_edges=None, verbose=False,
        error_tolerance=.001):
    """
    Find rate of an event (e.g. neuron spikes/second) as a function of a measurement (e.g. 2D position_and_maze).
    :param measurements: array with measurements. Shape: (measurement types, time).
                         E.g. (2, 1000) for 1000 samples of 2D position_and_maze.
    :param measurement_times: vector with times of measurements.
    :param measurements_at_events: array with measurements at events. Shape: (measurement types, time).
    :param bin_width: width of bin for forming rate histogram
    :param bin_edges: edges of bins for forming rate histogram
    :param verbose: True for printouts
    :return event_rate: rate
    :return bin_edges: bin edges for rate
    """

    # Check that array shapes are well defined
    for arr, arr_name in zip([measurements, measurements_at_events],
                             ["measurements", "measurements at events"]):
        if len(np.shape(arr)) < 2:
            raise Exception(f"Shape of {arr_name} array must have at least two dimensions, but has shape \
                             {np.shape(arr)}. Its possible 1D input was not put inside array.")
    # Check bin_edges or bin_width passed
    if bin_edges is None and bin_width is None:
            raise Exception("bin_width or bin_edges must be passed")
    if bin_edges is not None and bin_width is not None:
        raise Exception(f"Only bin_edges or bin_width can be passed, not both")
    # Define bin edges if not passed
    if bin_edges is None:
        bin_edges = np.asarray([make_int_bin_edges(x, bin_width) for x in measurements])
    # Calculate occupancy
    time_in_measurement_bins = calculate_occupancy_trodes_epoch_constant_sampling_rate(measurements=measurements,
                                                                measurement_times=measurement_times,
                                                                bin_edges=bin_edges,
                                                                error_tolerance=error_tolerance)
    # Calculate rate map
    event_rate, bin_edges = make_rate_map(measurements_at_events, time_in_measurement_bins, bin_edges)
    if verbose:
        check_rate_map(measurements, event_rate, bin_edges, measurements_at_events)
    return event_rate, bin_edges


def make_1D_rate_map_measurement_bouts(
        event_times, measurement_bouts, measurement_bouts_t, bin_edges, error_tolerance=.006, verbose=False):

    # Ensure same number of samples in measurements and time for each bout
    if not list(map(len, measurement_bouts)) == list(map(len, measurement_bouts_t)):
        raise Exception(
            f"Each bout of measurements must have same number of samples as corresponding time bout")

    # Find event times within bouts and interpolate measurement at these times
    measurement_bouts_joined = np.concatenate(measurement_bouts)
    measurement_bouts_t_joined = np.concatenate(measurement_bouts_t)
    measurement_bouts_t_interval = [(t[0], t[-1]) for t in measurement_bouts_t]
    _, trials_event_times = event_times_in_intervals(event_times, measurement_bouts_t_interval)
    measurements_interp = np.interp(trials_event_times,
                                    measurement_bouts_t_joined,
                                    measurement_bouts_joined)

    # Estimate sampling rate
    measurement_bouts_t_diff_joined = np.concatenate([np.diff(x) for x in measurement_bouts_t])
    check_uniform_spacing(np.cumsum(measurement_bouts_t_diff_joined), error_tolerance=error_tolerance)
    ave_t_diff = np.mean(measurement_bouts_t_diff_joined)

    # Make rate map
    return make_rate_map_constant_fs(measurements=np.asarray([measurement_bouts_joined]),
                                     sampling_rate=1 / ave_t_diff,
                                     measurements_at_events=np.asarray([measurements_interp]),
                                     bin_edges=[bin_edges],
                                     verbose=verbose)


def make_rate_map_uneven_fs(measurements, measurement_times, measurements_at_events, bin_edges, verbose=False):

    # Calculate occupancy
    time_in_measurement_bins = calculate_occupancy_uneven_sampling_rate(
        measurements=measurements, measurement_times=measurement_times, bin_edges=bin_edges)
    event_rate, bin_edges = make_rate_map(measurements_at_events, time_in_measurement_bins, bin_edges)
    if verbose:
        check_rate_map(measurements, event_rate, bin_edges, measurements_at_events)

    return event_rate, bin_edges


def make_rate_map_constant_fs(measurements, sampling_rate,
                              measurements_at_events, bin_edges, verbose=False):
    time_in_measurement_bins = calculate_occupancy_constant_sampling_rate(measurements,
                                               sampling_rate,
                                               bin_edges)
    event_rate, bin_edges = make_rate_map(measurements_at_events, time_in_measurement_bins, bin_edges)
    if verbose:
        check_rate_map(measurements, event_rate, bin_edges, measurements_at_events)
    return event_rate, bin_edges


# TODO (feature): remove bin_edges output
def make_rate_map(measurements_at_events, time_in_measurement_bins, bin_edges):
    # Histogram measurements at events
    measurements_at_events_bin_counts, _ = np.histogramdd(measurements_at_events.T, bins=bin_edges)
    # Create event rate map (events per second at each measurement bin)
    event_rate = measurements_at_events_bin_counts/time_in_measurement_bins  # convert event counts to rate
    return event_rate, bin_edges


def check_rate_map(measurements, event_rate, bin_edges, measurements_at_events):
    if len(np.shape(event_rate)) == 1:  # 1D
        fig, ax = plt.subplots()
        ax.plot(bin_edges[0][:-1] + np.diff(bin_edges[0]) / 2, event_rate, 'o-', color='black')
        ax.hist(measurements[0], bins=bin_edges[0], alpha=.5, color='gray', edgecolor="gray")
        ax.hist(measurements_at_events[0], bins=bin_edges[0], alpha=.5, color='red', edgecolor="red")
        ax.legend(labels=["event rate", "measurements", "measurements at events"])
    elif len(np.shape(event_rate)) == 2:  # 2D
        plot_rate_map(event_rate=event_rate, bin_edges=bin_edges)


def smooth_mask_rate_map(event_rate,
                         bin_edges=None,
                         sigma=None,
                         nan_treatment="interpolate"):
    """
    Smooth rate map containing nans. Prior to smoothing, replace nans with finite values
    (either zeros or interpolated values, as indicated), then mask back out after smoothing.
    :param event_rate: array, rate map
    :param sigma: tuple, standard deviation of Gaussian kernel for smoothing
    :param nan_treatment: string, default is "interpolate", options are "interpolate" and "zero"
    :return: masked array, rate map with nans masked out
    """

    if nan_treatment not in ["interpolate", "zero"]:
        raise Exception(f"nan_treatment must be either interpolate or zero")
    if nan_treatment == "interpolate" and bin_edges is None:  # check that bin edges passed if interpolating over nans
        raise Exception(f"bin_edges must be defined if interpolating over nans")
    if np.sum(np.isinf(event_rate)) > 0:  # check that no infinite values in rate map
        raise Exception(f"Infinite values in rate map")
    if sigma is None:  # define sigma is not passed
        sigma = tuple([1]*len(np.shape(event_rate)))

    # Alter nan values prior to smoothing rate map
    mask = np.isnan(event_rate)  # find nans in rate map
    valid_bool = np.ndarray.flatten(np.invert(mask))
    if nan_treatment == "interpolate":  # interpolate nan values
        if len(np.shape(event_rate)) == 2:  # if 2D rate map
            # Flatten x, y, z
            # For example convert x = [0,1,2],  y = [0,1], z = [[1,2,3], [4,5,6]]
            # to x = [0,1,2,0,1,2];  y = [0,0,0,1,1,1]; z = [1,2,3,4,5,6]
            x = vector_midpoints(bin_edges[0])
            y = vector_midpoints(bin_edges[1])
            y_flat = np.asarray(list(y) * len(x))
            x_flat = np.ndarray.flatten(np.asarray([[x_i] * len(y) for x_i in x]))
            z_flat = np.ndarray.flatten(event_rate)  # convert rate array to vector
            # Define values at which to interpolate
            interp_x = np.asarray([[x_i] * len(y) for x_i in x])
            interp_y = np.asarray([y] * len(x))
            # Interpolate
            interpolated_event_rate = sp.interpolate.griddata((x_flat[valid_bool], y_flat[valid_bool]),  # data coordinates
                                               z_flat[valid_bool],  # data values
                                               (interp_x, interp_y),  # points at which to interpolate data
                                               method="linear")
            # Set nans to zero in interpolated result, then smooth with Gaussian kernel,
            # then re-mask nans from interpolated result
            # If directly smooth without setting nans to zero, large portion of rate map set to nan
            smoothed_event_rate = smooth_mask_rate_map(interpolated_event_rate,
                                                     nan_treatment="zero",
                                                     sigma=sigma)
        elif len(np.shape(event_rate)) == 1:  # if 1D rate map
            interpolated_event_rate = np.interp(x=vector_midpoints(bin_edges),  # x values to interpolate over
                                            xp=vector_midpoints(bin_edges)[valid_bool],  # data x
                                            fp=event_rate[valid_bool])  # data y
            smoothed_event_rate = smooth_rate_map(interpolated_event_rate,
                                                  sigma=sigma)
    elif nan_treatment == "zero":  # set nans to zero
        event_rate[mask] = 0
        smoothed_event_rate = smooth_rate_map(event_rate,
                                              sigma=sigma)  # smooth rate map

    # Re-mask nans from original rate map
    smoothed_event_rate[mask] = np.nan

    return smoothed_event_rate


def smooth_rate_map(event_rate, sigma=None):
    """
    Smooth rate map with Gaussian kernel
    :param event_rate: array, rate of event in bins
    :param sigma: tuple, kernel standard deviation
    :return: array, smoothed rate map
    """

    if sigma is None:  # define sigma is not passed
        sigma = tuple([1]*len(np.shape(event_rate)))

    return sp.ndimage.gaussian_filter(event_rate, sigma=sigma, order=0)  # smooth rate map


def plot_rate_map(event_rate,
                  bin_edges,
                  interpolation=None,
                  fig_ax_list=None,
                  figsize=(5, 3),
                  cmap=plt.cm.rainbow,
                  plot_cbar=True,
                  clim=None,
                  color="black",
                  axis_off=False,
                  cbar_location="right"):

    if fig_ax_list is None:
        fig, ax = plt.subplots(figsize=figsize)  # initialize plot
    else:
        fig, ax = tuple(fig_ax_list)  # unpack figure and axis object
    if len(np.shape(event_rate)) == 1:  # 1D
        ax.plot(bin_edges[0][:-1] + np.diff(bin_edges[0]) / 2, event_rate, '.-', color=color)
    elif len(np.shape(event_rate)) == 2:  # 2D
        img = ax.imshow(event_rate.T, origin="lower", aspect="auto", interpolation=interpolation,
                                   extent=([np.min(bin_edges[0]), np.max(bin_edges[0]),
                                           np.min(bin_edges[1]), np.max(bin_edges[1])]),
                        cmap=cmap)
        if clim is not None:
            img.set_clim(clim)
        if plot_cbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbar_location, size="5%", pad=.1)
            cbar = fig.colorbar(img, ax=[ax], cax=cax)
            if cbar_location == "left":
                cax.yaxis.tick_left()
                cax.yaxis.set_label_position("left")
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label("rate", size=18)
        if axis_off:
            ax.axis("off")

    else:
        print(f"First dimension of event_rate of size {np.shape(event_rate)[0]} not recognized by code.")
