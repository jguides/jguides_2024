import numpy as np

from src.jguides_2024.utils.vector_helpers import check_monotonic_increasing


def make_bin_edges(x, bin_width, match_min_max="min", bins_encompass_x=True):
    """
    Make bins based on values in x, with the first bin starting at the minimum or
    the last bin starting at the maximum value in the list as indicated by match_min_max.
    Make enough bins to encompass all values in x, unless otherwise indicated.
    value of x
    :param x: array-like
    :param bin_width: bin width
    :param match_min_max: start bins at min ("min") or max ("max") value in list, default is "min"
    :param bins_encompass_x: bool, if True then make enough bins to encompass all values in x even if this
    means bin edges fall outside of x, if False do not include bins whose edges fall outside of min
    and max values in x. Default is True.
    :return: vector with bin edges
    """

    if match_min_max not in ["min", "max"]:
        raise Exception(f"match_min_max must be either min or max")
    # Make bin edges that encompass x
    if match_min_max == "min":
        bin_edges = np.arange(np.min(x), np.max(x) + bin_width, bin_width)
    elif match_min_max == "max":
        bin_edges = np.sort(make_bin_edges(np.asarray(x)*-1, bin_width, match_min_max="min")*-1)
    # Return bin edges if want them to encompass x
    if bins_encompass_x:
        return bin_edges
    # Otherwise, return subset of bin edges that fall within min and max of x
    valid_bool = np.logical_and(bin_edges >= np.min(x),
                                bin_edges <= np.max(x))

    return bin_edges[valid_bool]


def bins_from_edges(bin_edges):
    return list(zip(bin_edges[:-1], bin_edges[1:]))


def flatten_bin_edge_tuples(bin_edge_tuples):
    # Confirm that bins are contiguous
    bin_starts, bin_ends = list(map(np.asarray, list(zip(*bin_edge_tuples))))
    if not all(bin_starts[1:] == bin_ends[:-1]):
        raise Exception(f"bins not contiguous")
    return np.concatenate((bin_starts, bin_ends[-1:]))


def make_int_bin_edges(x, bin_width):
    return np.arange(np.floor(np.nanmin(x)),
                     np.ceil(np.nanmax(x)) + bin_width, bin_width)


def get_peri_event_bin_edges(peri_event_period, bin_width, event_time=0):
    # Check inputs well defined
    check_monotonic_increasing(peri_event_period)
    if len(peri_event_period) != 2:
        raise Exception(f"peri_event_period must have exactly two elements")
    if np.diff(peri_event_period)[0] == 0:
        raise Exception(f"peri_event_period cant have same start and endpoint")
    # Relative period AFTER event
    if peri_event_period[0] > 0:  # zero here and below since peri event period expressed relative to event
        relative_bin_edges = np.arange(peri_event_period[0], peri_event_period[1] + bin_width, bin_width)
    # Relative period BEFORE event
    elif peri_event_period[1] < 0:
        relative_bin_edges = np.arange(peri_event_period[0], peri_event_period[1] + bin_width, bin_width)
    # Relative period DURING event
    else:
        relative_bin_edges_at_before_zero = np.sort(-np.arange(0, -peri_event_period[0] + bin_width, bin_width))
        relative_bin_edges_after_zero = np.arange(bin_width, peri_event_period[1] + bin_width, bin_width)
        relative_bin_edges = np.concatenate((relative_bin_edges_at_before_zero, relative_bin_edges_after_zero))
    # Check relative bin edges make sense
    if np.max(abs(np.diff(relative_bin_edges) - bin_width)) > .001:
        raise Exception(f"bin_width not respected by relative_bin_edges")
    return event_time + relative_bin_edges
