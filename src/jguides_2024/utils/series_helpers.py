import numpy as np
import pandas as pd


def check_series(x, require_index_name=False):
    if not isinstance(x, pd.Series):
        raise Exception(f"vector must be a series")
    if require_index_name and x.index.name is None:
        raise Exception(f"series index must have a name")


def series_between_bool(series, valid_intervals):
    return np.sum([series.between(*valid_interval).values
                          for valid_interval in valid_intervals], axis=0) > 0
