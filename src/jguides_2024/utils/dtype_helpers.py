import numpy as np


def get_null_value(dtype):
    if dtype == "O":
        return None
    return np.nan