import warnings

import numpy as np


def check_membership(set_1, set_2, set_1_name="set_1", set_2_name="set_2", tolerate_error=False):

    """Check that all members in set_1 are in set_2 and optionally raise error if not"""

    # Get members in set 1 that are not in set 2. Note we convert set1_1 to list then array b/c converting directly
    # from set to array gives an array with a set
    invalid_set_1_members = np.asarray(list(set_1))[[x not in set_2 for x in set_1]]
    passed_check = len(invalid_set_1_members) == 0
    if not passed_check and not tolerate_error:  # raise error if indicated
        raise Exception(f"All elements in {set_1_name} should be contained within {set_2_name}. "
                        f"The following elements in {set_1_name} were not in {set_2_name}: {invalid_set_1_members}. "
                        f"Elements in {set_2_name} include: {set_2}")

    return passed_check


def check_set_equality(set_1, set_2, set_1_name="set 1", set_2_name="set 2", tolerate_error=False, issue_warning=True):
    # Useful implementation because if no set equality, user finds out which items are missing from which set
    # These raise error if tolerate_error is False
    passed_check = check_membership(set_1, set_2, set_1_name, set_2_name, tolerate_error)
    passed_check *= check_membership(set_2, set_1, set_2_name, set_1_name, tolerate_error)

    # If error but want to tolerate, issue warning if indicated
    if not passed_check and issue_warning:
        # note that error would have already been raised if tolerate_error = False, so no need to include in condition
        # above
        warnings.warn(f"Sets not equal")

    # If want to tolerate error, return result of check
    return passed_check
