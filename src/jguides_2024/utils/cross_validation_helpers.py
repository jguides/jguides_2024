import itertools
import random
from collections import namedtuple

import numpy as np
from sklearn.model_selection import KFold

from src.jguides_2024.utils.df_helpers import df_from_data_list
from src.jguides_2024.utils.dict_helpers import invert_dict
from src.jguides_2024.utils.list_helpers import check_in_list, check_lists_same_length
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.string_helpers import abbreviate_camel_case
from src.jguides_2024.utils.vector_helpers import repeat_elements_idxs, check_all_unique, unpack_single_element


def train_test_idxs(x, test_size, random_state=None):
    """
    Return a single train and test set for a given dataset (x)
    :param x: dataset (or vector of equivalent length)
    :param test_size: fraction of x to use for testing
    :param random_state: random state
    :return: train idxs and test idxs in x
    """
    # Check inputs
    if test_size < 0 or test_size > 1:
        raise Exception(f"test_size must be between 0 and 1 but is {test_size}")

    # Get indices spanning passed vector
    idxs = list(np.arange(0, len(x)))

    # Get approximate number of test trials to match user-defined proportion
    num_test_trials = int(np.round(len(idxs) * test_size))

    # Apply random state if passed
    if random_state is not None:
        random.seed(random_state)

    # Randomly select test idxs
    test_idxs = random.sample(idxs, num_test_trials)

    # Define remaining idxs as train idxs
    train_idxs = [x for x in idxs if x not in test_idxs]

    return train_idxs, test_idxs


def trials_train_test_split(trial_intervals,
                            time_vector,
                            test_size,
                            max_test_split_deviation=None,
                            random_state=None):

    # Return random subset of trials for testing and training (with proportion of trials approximately equal to
    # test_size and 1 - test_size). Note that this does not guarantee test_size and 1 - test_size amounts of data
    # for testing and training, since trials can be of variable duration. Given this, allow user to check that data
    # split close enough to test_size and 1 - test_size

    # Get inputs if not passed
    if max_test_split_deviation is None:
        max_test_split_deviation = 1

    # Check inputs
    # First, require that all times in time_vector fall within trial_intervals. Otherwise, unclear how to
    # assign those outside samples to train/test
    if not all(event_times_in_intervals_bool(time_vector, trial_intervals)):
        raise Exception(f"All samples in time_vector must be in trial_intervals")

    # Get indices of test and train trials
    train_trial_idxs, test_trial_idxs = train_test_idxs(trial_intervals, test_size, random_state=random_state)

    # Use above indices to get test and train time intervals
    trial_intervals = np.asarray(trial_intervals)
    test_intervals = trial_intervals[test_trial_idxs]
    train_intervals = trial_intervals[train_trial_idxs]

    # Use above time intervals to get test and train times
    train_idxs = np.where(event_times_in_intervals_bool(time_vector, train_intervals))[0]
    test_idxs = np.where(event_times_in_intervals_bool(time_vector, test_intervals))[0]

    # Check that each index in time_vector appears exactly once in test and train idxs
    if not all(np.unique(np.concatenate((test_idxs, train_idxs))) == np.arange(0, len(time_vector))):
        raise Exception(
            f"Each idx in time_vector should be appear once in either test_idxs and train_idxs but this was not the case")

    # Check that proportion of idxs in test set does not deviate too much from test_size (as indicated)
    fraction_test = len(test_idxs) / (len(train_idxs) + len(test_idxs))
    test_split_deviation = abs(fraction_test - test_size)
    if test_split_deviation > max_test_split_deviation:
        raise Exception(
            f"fraction_test {fraction_test} deviates from test_size {test_size} by more than tolerated amount ({max_test_split_deviation})")

    return train_idxs, test_idxs, train_trial_idxs, test_trial_idxs


class CrossValidate:

    def __init__(self, cross_validation_method=None, cross_validation_params=None):
        self.valid_cross_validation_methods = self._get_valid_cross_validation_methods()
        self.cross_validation_method = cross_validation_method
        self.cross_validation_params = cross_validation_params
        # Add default params if not passed
        self._get_inputs()
        # Check inputs
        self._check_inputs()

    @staticmethod
    def _get_valid_cross_validation_methods():
        return ["kfold", "leave_one_out_condition_trials", "same_condition_across_contexts", None]

    @classmethod
    def check_cross_validation_method_valid(cls, cross_validation_method):
        valid_cross_validation_methods = cls._get_valid_cross_validation_methods()
        if cross_validation_method not in valid_cross_validation_methods:
            raise Exception(f"cross_validation_method must be in {valid_cross_validation_methods} "
                            f"but is {cross_validation_method}")

    @staticmethod
    def get_default_cross_validation_params(cross_validation_method):
        # Map from cross validation method to default cross validation params
        default_params_map = {"kfold": {
            "n_splits": 5, "use_random_state": 0, "random_state": -1, "data_vector": None},
            # limit_same_train_test: True to limit to same train and test
            "leave_one_out_condition_trials": {
                "conditions_vector": None, "condition_trials_map": None, "limit_same_train_test": False},
            "same_condition_across_contexts": {"conditions": None, "contexts": None, "idxs": None}}

        # Return None if no entry in map
        if cross_validation_method not in default_params_map:
            return None

        # Otherwise, return default params
        return default_params_map[cross_validation_method]

    def _get_inputs(self):
        # Get inputs if not passed

        # Initialize dictionary for cross validation params if these not passed
        if self.cross_validation_params is None:
            self.cross_validation_params = dict()

        # Fill in defaults for missing params
        default_params = self.get_default_cross_validation_params(self.cross_validation_method)
        if default_params is not None:
            for k, v in default_params.items():
                if k not in self.cross_validation_params:
                    self.cross_validation_params[k] = v

    def _check_inputs(self):
        # Check cross validation method valid
        self.check_cross_validation_method_valid(self.cross_validation_method)

        # Check all necessary params passed
        # ...k fold
        valid_param_names = self.get_default_cross_validation_params(self.cross_validation_method).keys()
        if self.cross_validation_method == "kfold":
            check_membership(valid_param_names, self.cross_validation_params.keys(), "required cross validation params",
                             "passed cross validation params")
        # ...Leave one out condition trials: check that only one of conditions_vector and
        # condition_trials_map passed, since these contain redundant information
        elif self.cross_validation_method == "leave_one_out_condition_trials":
            param_names = ["conditions_vector", "condition_trials_map"]
            if np.sum([v is not None for k, v in self.cross_validation_params.items() if k in param_names]) != 1:
                raise Exception(f"Exactly one of {param_names} must be passed for {self.cross_validation_method}"
                                f" cross validation")

        # Check param values
        # ..k fold
        if self.cross_validation_method == "kfold":
            # Require random_state to be -1 if use_random_state is 0 (False)
            if not self.cross_validation_params["use_random_state"] and \
                    self.cross_validation_params["random_state"] != -1:
                raise Exception(f"random_state must be -1 if use_random_state is 0 (False)")
            # Check that number of splits is integer
            n_splits = self.cross_validation_params["n_splits"]
            if not isinstance(n_splits, int):
                raise Exception(f"n_splits must be an integer but is {type(n_splits)}")
        # ...same condition across contexts: check lists with condition values, contexts, and trial idxs are same length
        elif self.cross_validation_method == "same_condition_across_contexts":
            check_lists_same_length(list(self.cross_validation_params.values()))

    @classmethod
    def _get_cross_validation_method_abbreviation_map(cls):
        abbreviation_map = {k: abbreviate_camel_case(k) for k in cls._get_valid_cross_validation_methods()
                            if k is not None}  # default is to use uppercase first letter
        abbreviation_map[None] = ""  # return empty string if no cross validation method
        preserve_full = ["kfold"]  # use full name as abbreviation for these variables
        abbreviation_map.update({x: x for x in preserve_full})
        # Check that no redundant abbreviations in map
        check_all_unique(list(abbreviation_map.values()))
        return abbreviation_map

    @classmethod
    def abbreviate_cross_validation_method(cls, cross_validation_method):
        # Return empty string if cross validation method is None
        if cross_validation_method is None:
            return ""
        # Otherwise, check that valid cross validation method, then return abbreviation
        cls.check_cross_validation_method_valid(cross_validation_method)
        return cls._get_cross_validation_method_abbreviation_map()[cross_validation_method]

    @staticmethod
    def _get_leave_one_out_train_set_id(condition, left_out_trial_num=None):
        if left_out_trial_num is None:
            return f"train_{condition}_alltrials"
        return f"train_{condition}_no{left_out_trial_num}"

    @staticmethod
    def _get_leave_one_out_test_set_id(condition, trial_num):
        return f"test_{condition}_{trial_num}"

    @staticmethod
    def _get_same_condition_across_contexts_set_id(condition, context):
        return f"{condition}_{context}"

    @staticmethod
    def _get_train_test_set_df(train_test_set_tuples):
        train_test_set_column_names = ["train_set_id", "test_set_id", "train_condition", "test_condition", "fold_num"]
        return df_from_data_list(train_test_set_tuples, train_test_set_column_names)

    @staticmethod
    def _get_train_set_df(train_set_tuples):
        train_set_column_names = ["train_set_id", "train_idxs"]
        return df_from_data_list(train_set_tuples, train_set_column_names).set_index("train_set_id")

    @staticmethod
    def _get_test_set_df(test_set_tuples):
        test_set_column_names = ["test_set_id", "test_idxs"]
        return df_from_data_list(test_set_tuples, test_set_column_names).set_index("test_set_id")

    @staticmethod
    def _package_train_test_info(train_set_df, test_set_df, train_test_set_df):
        TrainTestSet = namedtuple("TrainTestSet", "train_set_df test_set_df train_test_set_df")
        return TrainTestSet(train_set_df, test_set_df, train_test_set_df)

    def _kfold_train_test_split(self):
        # Get train_set_df, test_set_df, and train_test_set_df for kfold case

        # Get cross vaslidation params
        params = self.cross_validation_params

        # Convert random_state of -1 to None
        if params["random_state"] == -1:
            params["random_state"] = None

        # Get KFold object
        # Note that random_state parameter has no effects on KFold object unless shuffle is True (currently no
        # implementation where this could be the case)
        kf = KFold(n_splits=params["n_splits"], random_state=params["random_state"])
        kf.get_n_splits(params["data_vector"])  # get splits

        # Get train/test set indices for k folds
        condition = "kfold"
        fold_nums, train_idxs_list, test_idxs_list = zip(*[(fold_num, train_idxs, test_idxs)
                                                           for fold_num, (train_idxs, test_idxs) in
                                                           enumerate(kf.split(params["data_vector"]))])

        # Get train set idxs in one place
        train_set_ids = [self._get_leave_one_out_train_set_id(condition, left_out_trial_num=fold_num)
                         for fold_num in fold_nums]
        train_set_tuples = [(train_set_id, train_idxs) for train_set_id, train_idxs in
                            zip(train_set_ids, train_idxs_list)]
        train_set_df = self._get_train_set_df(train_set_tuples)

        # Get test set idxs in another place
        test_set_ids = [self._get_leave_one_out_test_set_id(condition, trial_num=fold_num)
                        for fold_num in fold_nums]
        test_set_tuples = [(test_set_id, test_idxs) for test_set_id, test_idxs in zip(test_set_ids, test_idxs_list)]
        test_set_df = self._get_test_set_df(test_set_tuples)

        # Get train/test set pairs
        train_test_tuples = [(self._get_leave_one_out_train_set_id(condition, left_out_trial_num=fold_num),
                              self._get_leave_one_out_test_set_id(condition, trial_num=fold_num),
                              condition,
                              condition,
                              fold_num) for fold_num in fold_nums]
        train_test_set_df = self._get_train_test_set_df(train_test_tuples)

        return self._package_train_test_info(train_set_df, test_set_df, train_test_set_df)

    def _leave_one_out_condition_trials_train_test_split(self):
        # Get train_set_df, test_set_df, and train_test_set_df for leave one out condition trials case

        # Get cross vaslidation params
        params = self.cross_validation_params

        # If condition_trials_map not passed, use conditions_vector to define
        # Approach: "condition trials" are spans of a single condition. Make map from condition value to
        # "trial indices": indices in conditions_vector that are part of the trial (inclusive of endpoint, i.e.
        # NOT slice indices)
        if params["condition_trials_map"] is None:
            params["condition_trials_map"] = invert_dict(
                repeat_elements_idxs(params["conditions_vector"], slice_idxs=False, as_dict=True))

        # Drop conditions with only a single trial, since cannot "leave one out" here
        params["condition_trials_map"] = {
            condition: trial_idxs for condition, trial_idxs in params["condition_trials_map"].items()
            if len(trial_idxs) > 1}

        # ...Check that at least some trials left
        if len(params["condition_trials_map"]) == 0:
            raise Exception(f"condition_trials_map is empty")

        # Convert condition trial slice (start/stop range) into full vector of idxs ([start, ..., stop])
        # add one to x2 since trial indices include endpoint (i.e. NOT slice indices) and np.arange does not include
        # endpoint
        condition_trial_idxs_map = {condition: [np.arange(x1, x2 + 1) for x1, x2 in trial_idxs]
                                    for condition, trial_idxs in params["condition_trials_map"].items()}

        # For each trial in a condition, train using "valid" trials from each of the other conditions:
        # If train and test conditions are distinct, then all trials from train condition are valid
        # If train and test conditions are the same, then all trials except test trial are valid

        # Get test set idxs
        test_set_tuples = [(self._get_leave_one_out_test_set_id(condition, trial_num), trial_idxs)
                           for condition, trials_idxs in condition_trial_idxs_map.items()
                           for trial_num, trial_idxs in enumerate(trials_idxs)]
        test_set_df = self._get_test_set_df(test_set_tuples)

        # Get train set idxs
        train_set_left_out_trial_tuples = [
            (self._get_leave_one_out_train_set_id(condition, trial_num),
             list(set(np.concatenate(trials_idxs)) - set(trial_idxs)))
            for condition, trials_idxs in condition_trial_idxs_map.items()
            for trial_num, trial_idxs in enumerate(trials_idxs)]
        train_set_all_trials_tuples = [
            (self._get_leave_one_out_train_set_id(condition), np.concatenate(trials_idxs))
            for condition, trials_idxs in condition_trial_idxs_map.items()]
        train_set_tuples = train_set_left_out_trial_tuples + train_set_all_trials_tuples
        train_set_df = self._get_train_set_df(train_set_tuples)

        # Get train/test set pairs
        train_test_set_tuples = []
        for test_condition, test_trials_idxs in condition_trial_idxs_map.items():  # test conditions
            for test_trial_num, _ in enumerate(test_trials_idxs):  # test trials
                for train_condition, _ in condition_trial_idxs_map.items():  # train conditions

                    # Continue if limiting to same training and testing only on same condition
                    if params["limit_same_train_test"] and test_condition != train_condition:
                        continue

                    # Get train set ID. Here, in case that train and test conditions are the same, do not
                    # include train trial in test set
                    if test_condition == train_condition:
                        left_out_trial_num = test_trial_num
                    else:
                        left_out_trial_num = None
                    train_set_id = self._get_leave_one_out_train_set_id(train_condition, left_out_trial_num)

                    # Get test set ID
                    test_set_id = self._get_leave_one_out_test_set_id(test_condition, test_trial_num)

                    # Store train/test indices and identifiers
                    train_test_set_tuples.append(
                        (train_set_id, test_set_id, train_condition, test_condition, test_trial_num))

        train_test_set_df = self._get_train_test_set_df(train_test_set_tuples)

        return self._package_train_test_info(train_set_df, test_set_df, train_test_set_df)

    def _same_condition_across_contexts_train_test_split(self):
        # Get train_set_df, test_set_df, and train_test_set_df for same condition across contexts case

        # Get cross validation params
        params = self.cross_validation_params
        conditions = np.asarray(params["conditions"])
        contexts = np.asarray(params["contexts"])
        idxs = np.asarray(params["idxs"])

        # Define possible train and test sets: pairs of condition value and context
        x_set_tuples = []
        for condition, context in set(zip(conditions, contexts)):
            valid_bool = np.logical_and(conditions == condition, contexts == context)
            x_set_tuples.append((self._get_same_condition_across_contexts_set_id(condition, context),
                                 np.concatenate(idxs[valid_bool])))
        train_set_df = self._get_train_set_df(x_set_tuples)
        test_set_df = self._get_test_set_df(x_set_tuples)

        # Train and test pairs consist of same condition across different contexts. To get, we must identify
        # condition values that are present. For each condition value, we must identify pairs of contexts with
        # that condition value
        train_test_set_tuples = []
        for condition in set(conditions):
            valid_bool = conditions == condition  # find indices of list corresponding to condition value
            # Loop through context pairs that have this condition value represented
            for train_context, test_context in list(itertools.permutations(contexts[valid_bool], r=2)):
                train_set_id = self._get_same_condition_across_contexts_set_id(condition, train_context)
                test_set_id = self._get_same_condition_across_contexts_set_id(condition, test_context)
                train_test_set_tuples.append((train_set_id, test_set_id, condition, condition, 1))
        train_test_set_df = self._get_train_test_set_df(train_test_set_tuples)

        return self._package_train_test_info(train_set_df, test_set_df, train_test_set_df)

    def get_train_test_sets(self):
        # Return three dataframes:
        # 1) training sets: name, indices
        # 2) test sets: name, indices
        # 3) pairs of train and test sets to be used in some downstream analysis

        # Return None if no cross validation method
        if self.cross_validation_method is None:
            return None, None, None

        # Call appropriate cross validation function
        cross_validation_function_map = {
            "kfold": self._kfold_train_test_split, "leave_one_out_condition_trials":
                self._leave_one_out_condition_trials_train_test_split, "same_condition_across_contexts":
                self._same_condition_across_contexts_train_test_split}

        return cross_validation_function_map[self.cross_validation_method]()

    def x_set_as_dtype(self, set_name, dtype="dict"):
        # Return train or test set as dictionary or list
        # Check dtype valid
        check_in_list(dtype, ["dict", "list"], x_name="dtype")
        # Check set name valid
        check_in_list(set_name, ["train_set_df", "test_set_df"], x_name="set_name")

        # Get train or test set
        set_df = getattr(self, set_name)  # train set or test set
        # Get column name of train or test set (ensure only one)
        column_name = unpack_single_element(set_df.columns)

        # Return rows of train or test set dataframe as key-value pairs in a dictionary
        if dtype == "dict":
            return {k: v for k, v in set_df[column_name].items()}

        # Return rows of train or test st dataframe as entries in a list, and also return corresponding set ids
        elif dtype == "list":
            return [x for x in set_df[column_name]], set_df.index

