import warnings

import numpy as np


def check_one_none(x, list_element_names=""):
    num_none = len([x_i for x_i in x if x_i is None])
    if num_none != 1:
        raise Exception(f"Need exactly one None in passed arguments {list_element_names}"
                        f" but got {num_none} Nones")


def check_shape(x, expected_shape, x_name="x"):
    if len(np.shape(x)) != expected_shape:
        raise Exception(f"{x_name} must be one dimensional")


def failed_check(tolerate_error, print_statement="FAILED CHECK", issue_warning=True):
    # Meant to be called when a check failed, to optionally raise error, or not raise error and optionally
    # issue warning
    if tolerate_error:
        if issue_warning:
            warnings.warn(print_statement)
    else:
        raise Exception(print_statement)
