import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels import api as sm

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import format_nwb_file_name, make_param_name, \
    get_epochs_id
from src.jguides_2024.utils.array_helpers import check_arrays_equal
from src.jguides_2024.utils.check_well_defined import check_one_none
from src.jguides_2024.utils.cross_validation_helpers import CrossValidate
from src.jguides_2024.utils.df_helpers import zip_df_columns, df_filter_columns, df_from_data_list
from src.jguides_2024.utils.dict_helpers import find_key_for_list_value
from src.jguides_2024.utils.for_loop_helpers import print_iteration_progress
from src.jguides_2024.utils.list_helpers import check_single_element
from src.jguides_2024.utils.plot_helpers import plot_text_color, get_cmap_colors
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.string_helpers import format_optional_var, format_bool
from src.jguides_2024.utils.vector_helpers import repeat_elements_idxs, unpack_single_element, \
    unpack_single_vector, check_vectors_equal


# HELPER FUNCTIONS
# GLM FITTING
def glm_link(covariates, parameters):
    return covariates @ parameters


def poisson_glm_response(covariates, parameters):
    return np.exp(glm_link(covariates, parameters))


def poisson_glm_gradient(observations, covariates, parameters):
    return covariates.T @ (observations - poisson_glm_response(covariates, parameters))


def poisson_glm_hessian(covariates, parameters):
    return covariates.T @ np.diag(poisson_glm_response(covariates, parameters).ravel()) @ covariates


def poisson_glm_proportional_loglikelihood(observations, covariates, parameters):
    return observations.T @ covariates @ parameters - np.sum(poisson_glm_response(covariates, parameters))


# DESIGN MATRIX
def add_intercept_column(design_df):
    design_df["intercept"] = np.ones(len(design_df))
    return design_df


def return_intercept_design_df(covariates_df):
    return pd.DataFrame.from_dict({covariates_df.index.name: covariates_df.index,
                                         "intercept": np.ones(len(covariates_df))}).set_index(covariates_df.index)


# SIMULATE SPIKES
def simulate_spike_counts_poisson_glm(covariates, true_parameters):
    return np.random.poisson(np.exp(glm_link(covariates, true_parameters)))


# COVARIATE COEFFICIENTS
def initialize_covariate_coefficients(covariate_names, initialize_method):
    if initialize_method == "zero":
        return pd.Series(np.zeros(len(covariate_names)),
                                    index=covariate_names) # initialize to zero
    elif initialize_method == "random":
        return pd.Series(np.random.sample(len(covariate_names))*.1,
                                    index=covariate_names)  # initialize to random value
    else:
        raise Exception(f"initialize_params must be zero or random")


# PLOTTING
def format_model_name(model_name):
    # Replace acronyms
    model_names_map = {"ppt": "journey", "pstpt": "same turn path", "pupt": "unique path"}
    for k, v in model_names_map.items():
        model_name = model_name.replace(k, v)
    # Drop intercept
    model_name = model_name.replace("intercept", "")
    # Substitute new line for underscore
    model_name = model_name.replace("_", "\n")
    return model_name


# CLASSES
class DesignMatrix:

    def __init__(self, sampled_raised_cosine_basis_list=None, sampled_relative_measure_basis_list=None, intercept=True):
        """
        Container for design matrix for generalized linear model.
        :param sampled_raised_cosine_basis_list: list with instances of SampledRaisedCosineBasis.
        :param sampled_relative_measure_basis_list: list with instances of SampledRelativeMeasureCosineBasis.
        :param intercept: bool. True to include intercept in design matrix.
        """

        self.intercept = intercept
        self._check_inputs(sampled_raised_cosine_basis_list, sampled_relative_measure_basis_list)
        # Maps from covariate name to sampled basis object
        self.sampled_raised_cosine_basis_map = self._get_sampled_basis_map(sampled_raised_cosine_basis_list)
        self.sampled_relative_measure_basis_map = self._get_sampled_basis_map(sampled_relative_measure_basis_list)
        # Concatenate across basis functions
        self.sampled_bases, self.sampled_bases_covariate_group_name_map = self._get_sampled_bases()
        # Add intercept term to sampled bases to get design_df
        self.design_df, self.design_covariate_group_name_map = self._get_design_df()  # design matrix in df form
        self.design_covariate_group_names = self._get_design_covariate_group_names()
        self.design_matrix = self._get_design_matrix()  # design matrix in array form
        self.covariate_names = np.asarray(self.design_df.columns)  # covariate names (each basis function, intercept)
        # Define # covariate "groups": basically just those corresponding to basis function groups, plus intercept
        self.covariate_group_names = self._get_covariate_group_names()
        # Map from covariate "group" to color
        self.covariate_group_name_color_map = self._get_covariate_group_name_color_map()

    def _check_inputs(self, sampled_raised_cosine_basis_list, sampled_relative_measure_basis_list):

        # Check that at least one basis passed
        if sampled_raised_cosine_basis_list is None and sampled_relative_measure_basis_list is None:
            raise Exception(f"No basis lists passed")
        if all([len(x) == 0 for x in [sampled_raised_cosine_basis_list, sampled_relative_measure_basis_list]
                if x is not None]):
            raise Exception(f"Empty basis list(s)")

        if self.intercept not in [True, False]:
            raise Exception(f"intercept is a flag to indicate whether or not to use intercept and must be True or False")

    @staticmethod
    def _get_sampled_basis_map(basis_list):
        if basis_list is None:
            return {}
        return {x.basis.covariate_group_name: x for x in basis_list}

    def _get_sampled_bases(self):
        # Make sampled bases and map between the involved covariate group name and covariate names.
        # To get sampled bases, concatenate sampled basis functions across each sampled basis.
        sampled_bases_covariate_group_name_map = dict()  # map from covariate group name to covariate names
        sampled_bases_list = []  # for concatenating sampled basis functions across bases
        for basis_map in [self.sampled_raised_cosine_basis_map,
                          self.sampled_relative_measure_basis_map]:
            for covariate_group_name, basis_obj in basis_map.items():
                sampled_basis = basis_obj.sampled_basis
                sampled_bases_covariate_group_name_map[covariate_group_name] = np.asarray(sampled_basis.columns)
                sampled_bases_list.append(sampled_basis)
        return pd.concat(sampled_bases_list, axis=1), sampled_bases_covariate_group_name_map

    def _get_design_df(self):
        """
        Convert sampled bases to design df by adding intercept if indicated
        :return: design matrix in df form
        """

        design_df = copy.deepcopy(self.sampled_bases)  # important so that sampled_bases not overwritten
        design_covariate_group_name_map = copy.deepcopy(self.sampled_bases_covariate_group_name_map)

        # Add intercept term if indicated
        if self.intercept:
            design_df["intercept"] = [1] * len(design_df)
            design_covariate_group_name_map["intercept"] = ["intercept"]

        return design_df, design_covariate_group_name_map

    def _get_design_covariate_group_names(self):
        return np.asarray([find_key_for_list_value(self.design_covariate_group_name_map, covariate_name)
                for covariate_name in self.design_df.columns])

    def _get_design_matrix(self):
        return self.design_df.to_numpy()

    def _get_covariate_group_names(self):
        return list(self.sampled_raised_cosine_basis_map.keys()) + \
               list(self.sampled_relative_measure_basis_map.keys()) + ["intercept"] * self.intercept

    def _get_covariate_group_name_color_map(self):
        color_list = np.concatenate((get_cmap_colors("tab20"),  # intended to be for spatial covariates
                                    np.tile([1, 1, 1], (np.shape(self.design_matrix)[1], 1))))  # inteded to be for spike covariates
        return {covariate_group_name: color
                  for covariate_group_name, color in
                  zip(self.covariate_group_names,
                      color_list)}

    def plot_bases(self):
        for sampled_raised_cosine_basis in self.sampled_raised_cosine_basis_map.values():
            sampled_raised_cosine_basis.basis.plot_basis_functions()

    def plot_covariate_group_name_colors(self):
        plot_text_color(self.covariate_group_name_color_map.keys(),
                        self.covariate_group_name_color_map.values())

    def plot_design_matrix(self, figsize=(10, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(self.design_df, cmap="Greys")
        # Plot colored rectangles to indicate covariate group names
        anchor_point = [0, 0]  # initialize anchor point for rectangles
        for idx_span in repeat_elements_idxs(self.design_covariate_group_names):
            width = unpack_single_element(np.diff(idx_span))
            color = self.covariate_group_name_color_map[self.design_covariate_group_names[idx_span[0]]]
            ax.add_patch(matplotlib.patches.Rectangle(anchor_point, width=width, height=len(self.design_df),
                                                      color=color, alpha=.2))
            anchor_point[0] += width  # update anchor point


class ElasticNet:

    def __init__(self, design_df, family=sm.families.Poisson(), alpha=0, L1_wt=0):
        # Store attributes
        self.design_df = design_df
        self.family = family
        self.alpha = alpha
        self.L1_wt = L1_wt

    def train_glm(self, covariate_names, spike_counts, train_idxs):

        # Check inputs
        check_membership(covariate_names, self.design_df, "passed covariate names", "available covariate names")

        # Assemble design matrix with passed covariates
        design_matrix = self.design_df[covariate_names]

        # Get train/test data using train/test indices
        y_train, x_train = spike_counts.iloc[train_idxs], design_matrix.iloc[train_idxs]

        # Estimate parameters
        model = sm.GLM(y_train, x_train, family=self.family, missing='drop')  # initialize
        return model.fit_regularized(alpha=self.alpha, L1_wt=self.L1_wt)

    def equals(self, external_obj):

        # Check if attributes in class instance are same as in an external object. Useful because seems
        # train_glm method not same across two class instances

        return (self.design_df == external_obj.design_matrix_obj and
                self.family == external_obj.family and
                self.alpha == external_obj.alpha and
                self.L1_wt == external_obj.L1_wt)


class ElasticNetContainer:
    """
    Set up to allow training multiple models under the same conditions
    """

    def __init__(self, covariate_model_name_map, covariate_type_map, covariate_group_map, design_df, spike_counts,
                 cross_validation_dfs=None, cross_validation_params=None, family=sm.families.Poisson(), alpha=0,
                 L1_wt=0, verbose=False):
        self.covariate_model_name_map = covariate_model_name_map
        self.covariate_type_map = covariate_type_map
        self.covariate_group_map = covariate_group_map
        self.spike_counts = spike_counts
        self._check_inputs()
        self.elastic_net = ElasticNet(design_df, family, alpha, L1_wt)
        self.model_names = self._get_model_names()
        self.model_name_covariate_names_map = self._get_model_name_covariate_names_map()
        self.train_set_df, self.test_set_df, self.train_test_set_df = self._get_cross_validation_dfs(
            cross_validation_dfs, cross_validation_params)
        self.fit_params_df, self.log_likelihood, self.results_folds_separate_df = self._train_test_glm(verbose)
        self.results_folds_merged_df, self.folds_df = self._merge_across_folds()

    def _check_inputs(self):
        # Check that spike counts are as series
        if not isinstance(self.spike_counts, pd.Series):
            raise Exception(f"spike_counts must be pandas Series but are type {type(self.spike_counts)}")

    def _get_model_names(self):
        return list(self.covariate_model_name_map.keys())

    def _get_model_name_covariate_names_map(self):
        model_name_covariate_names_map = dict()
        for model_name, covariate_types in self.covariate_model_name_map.items():  # for each model
            # Get covariate group names for the covariate types for this model
            covariate_group_names = np.concatenate([self.covariate_type_map[covariate_type]
                                                    for covariate_type in covariate_types])
            # Pool covariate names across covariate groups
            covariate_names = np.concatenate(
                [self.elastic_net.design_df[self.covariate_group_map[covariate_group_name]].columns
                 for covariate_group_name in covariate_group_names])
            model_name_covariate_names_map[model_name] = covariate_names
        return model_name_covariate_names_map

    def _get_cross_validation_dfs(self, cv_dfs, cv_params):
        # Check that either cross validation dfs passed or cross validation
        # params passed to be able to get cross validation dfs
        check_one_none([cv_dfs, cv_params])
        # Note that if need to get cross validation dfs, cross validation params are checked during this process

        # Get cross validation dfs if not passed
        if cv_dfs is None:
            cv_method = cv_params.pop("cross_validation_method")
            cv_dfs = CrossValidate(cv_method, cv_params)

        # Unpack cross validation dfs
        return cv_dfs.train_set_df, cv_dfs.test_set_df, cv_dfs.train_test_set_df

    def _get_test_idxs(self, test_set_id):
        return self.test_set_df.loc[test_set_id]["test_idxs"]

    def _train_test_glm(self, verbose):

        # Get all unique train set IDs in the df holding the train/test set pairs we want
        train_set_ids = np.unique(self.train_test_set_df["train_set_id"])

        test_list = []
        train_list = []
        for model_name, covariate_names in self.model_name_covariate_names_map.items():  # for each model

            if verbose:
                print(f"On model {model_name}...")

            for train_set_id_idx, train_set_id in enumerate(train_set_ids):

                if verbose:
                    print_iteration_progress(
                        iteration_num=train_set_id_idx, num_iterations=len(train_set_ids),
                        target_num_print_statements=20)

                # Train GLM
                fit_obj = self.elastic_net.train_glm(
                    covariate_names, self.spike_counts, self.train_set_df.loc[train_set_id]["train_idxs"])

                # Store training information: fit params and log likelihood
                log_likelihood = fit_obj.model.loglike(fit_obj.params)
                train_list.append((train_set_id, fit_obj.params, log_likelihood))

                # Test for all train/test pairs that include current train_set_id
                df_subset = df_filter_columns(self.train_test_set_df, {"train_set_id": train_set_id})
                for test_set_id, train_condition, test_condition in zip_df_columns(
                        df_subset, ["test_set_id", "train_condition", "test_condition"]):

                    test_idxs = self._get_test_idxs(test_set_id)

                    # Define test set so can store with predictions for convenience
                    y_test = self.spike_counts.iloc[test_idxs]

                    # Predict
                    x_test = self.elastic_net.design_df[covariate_names].iloc[test_idxs]
                    y_test_predicted = pd.Series(fit_obj.predict(x_test), index=x_test.index)

                    # Predict counts if Poisson GLM
                    y_test_predicted_count = None

                    if isinstance(self.elastic_net.family, sm.families.Poisson):

                        # Using predicted lambda, generate Poisson spike counts (can only do at places where
                        # lambda finite)
                        y_test_predicted_count = pd.Series(np.asarray([np.nan] * len(y_test_predicted)),
                                                           index=y_test_predicted.index)  # initialize vector for counts
                        valid_idxs = np.where(np.isfinite(y_test_predicted))[0]  # where prediction y finite
                        y_test_predicted_count.iloc[valid_idxs] = np.random.poisson(y_test_predicted.iloc[valid_idxs])

                    # Store test results
                    test_list.append((model_name, train_set_id, test_set_id, train_condition, test_condition, y_test,
                                      y_test_predicted, y_test_predicted_count))

        # Return training and testing results in dataframes
        train_set_ids, params_list, log_likelihood_list = zip(*train_list)
        param_names = unpack_single_vector([x.index for x in params_list])
        params_df = pd.DataFrame(np.asarray([x.values for x in params_list]), columns=param_names, index=train_set_ids)
        params_df.index.name = "train_set_id"
        log_likelihoods = pd.Series(log_likelihood_list, index=train_set_ids)
        test_column_names = ["model_name", "train_set_id", "test_set_id", "train_condition", "test_condition",
                             "y_test", "y_test_predicted", "y_test_predicted_count"]
        results_folds_separated_df = df_from_data_list(test_list, test_column_names)

        return params_df, log_likelihoods, results_folds_separated_df

    def _merge_across_folds(self):
        # Merge results across folds

        def _check_vectors_equal(x):
            check_vectors_equal([np.asarray(z) for z in x])

        def _check_arrays_equal(x):
            check_arrays_equal([np.asarray(z) for z in x])

        def _check_return_single_val(vals, equality_fn):
            equality_fn(vals)
            return vals[0]

        # Define entries in results_df to merge across folds
        merge_results_names = ["y_test", "y_test_predicted"]  # concatenate these variables across folds
        if isinstance(self.elastic_net.family, sm.families.Poisson):
            merge_results_names += ["y_test_predicted_count"]

        # Get unique settings of train/test conditions so can merge folds within each
        train_test_condition_tuple_names = ["model_name", "train_condition", "test_condition"]
        train_test_condition_tuples = list(set(list(zip_df_columns(
            self.results_folds_separate_df, train_test_condition_tuple_names))))

        # Merge across folds
        data_list = []
        for model_name, train_condition, test_condition in train_test_condition_tuples:
            df_subset = df_filter_columns(
                self.results_folds_separate_df,
                {"model_name": model_name, "train_condition": train_condition, "test_condition": test_condition})
            # Order rows by test_set_id (use same order as in test_set_df; note that this order is not reproduced
            # by np.sort due to integers as text)
            df_subset = df_subset.set_index("test_set_id", inplace=False)
            new_index = [x for x in self.test_set_df.index if x in df_subset.index]
            if not np.logical_and(set(new_index) == set(df_subset.index), len(new_index) == len(df_subset.index)):
                raise Exception(f"new index has different elements from previous index")
            df_subset = df_subset.reindex(new_index)  # inplace not available

            data_list.append(tuple(
                [model_name, train_condition, test_condition, len(df_subset),
                 [len(x) for x in df_subset["y_test"]],  # store test fold lengths
                 *[pd.concat([x for x in df_subset[k].values]) for k in merge_results_names],
                 [np.asarray([x.index[0], x.index[-1]]) for x in df_subset["y_test"]]]))  # store fold start/stop
        column_names = train_test_condition_tuple_names + ["num_folds", "fold_lens"] + merge_results_names + \
                         ["fold_intervals"]
        results_folds_merged_df = df_from_data_list(data_list, column_names)

        # Put test folds information in separate df to avoid copying same information across df rows in
        # results_folds_merged_df that share same test condition
        # ...First create new df with folds information
        column_names = ["num_folds", "fold_lens", "fold_intervals", "y_test"]
        equality_fns = [check_single_element, _check_vectors_equal, _check_arrays_equal, _check_vectors_equal]
        data_list = []
        for test_condition in np.unique(results_folds_merged_df.test_condition):
            # Extract single value for test condition
            val_list = [_check_return_single_val(df_filter_columns(
                    results_folds_merged_df, {"test_condition": test_condition})[column_name].values, equality_fn)
                        for column_name, equality_fn in zip(column_names, equality_fns)]
            data_list.append(tuple([test_condition] + val_list))
        folds_df = df_from_data_list(data_list, ["test_condition"] + column_names).set_index("test_condition")
        # ...Now remove that information from results_folds_merged_df
        results_folds_merged_df.drop(columns=column_names, inplace=True)

        return results_folds_merged_df, folds_df


class SimulatedSpikes:

    def __init__(self, design_matrix_obj, replace_parameters_map, initialize_parameters):
        """
        Simulate spike counts
        :param design_matrix_obj: instance of DesignMatrix class.
        :param replace_parameters_map: maps covariate groups to param vals. {covariate_group_name: {basis_num: val}, ...}
        :param initialize_parameters: str. Must be "zero" (params 0) or random (params random).
        """
        self.design_matrix_obj = design_matrix_obj
        self.replace_parameters_map = replace_parameters_map
        self.initialize_parameters = initialize_parameters
        self._check_inputs()
        self.parameters = self._get_parameters()
        self.poisson_spike_counts = self._get_poisson_spike_counts()

    def _check_inputs(self):

        if not isinstance(self.design_matrix_obj, DesignMatrix):
            raise Exception(f"design_df must be instance of DesignMatrix")

        if not all([k in self.design_matrix_obj.covariate_group_names for k in self.replace_parameters_map.keys()]):
            raise Exception(f"replace_parameters_map keys must all be in design_df.covariate_group_names")

        # Ensure replace_parameters_map entries well-defined
        for covariate_group_name, covariate_group_map in self.replace_parameters_map.items():
            if covariate_group_name == "intercept":
                continue  # no need to check intercept
            if np.max(list(covariate_group_map.keys())) > self.design_matrix_obj.sampled_raised_cosine_basis_map[
                covariate_group_name].basis.num_basis_functions:
                raise Exception(
                    f"Keys in replace_parameters_map.values() greater than number of basis functions "
                    f"for {covariate_group_name}")
            if np.min(list(covariate_group_map.keys())) < 0:
                raise Exception(f"Keys in replace_parameters_map.values() must be greater than zero")

        if self.initialize_parameters not in ["zero", "random"]:
            raise Exception(f"initialize_params must be zero or random")

    def _get_parameters(self):

        # Initialize parameters
        if self.initialize_parameters == "zero":
            parameters = pd.Series(np.zeros(len(self.design_matrix_obj.covariate_names)),
                                   index=self.design_matrix_obj.covariate_names)  # initialize to zero
        elif self.initialize_parameters == "random":
            parameters = pd.Series(np.random.sample(len(self.design_matrix_obj.covariate_names)),
                                   index=self.design_matrix_obj.covariate_names)  # initialize to random value between 0 and 1

        # Update parameters using passed values
        for covariate_group_name, covariate_group_map in self.replace_parameters_map.items():
            for basis_num, val in covariate_group_map.items():
                if covariate_group_name in self.design_matrix_obj.sampled_raised_cosine_basis_map:
                    covariate_name = f"{covariate_group_name}_basis_{basis_num}"
                elif covariate_group_name == "intercept":
                    covariate_name = covariate_group_name
                else:
                    raise Exception(f"covariate_group_name should have either been in "
                                    f"design_df.sampled_raised_cosine_basis_map or been intercept, "
                                    f"but is {covariate_group_name}")
                parameters.loc[covariate_name] = val

        return parameters

    def _get_poisson_spike_counts(self):
        lambdas = poisson_glm_response(self.design_matrix_obj.design_matrix, self.parameters)
        return np.random.poisson(lambdas)

    def plot_spike_counts(self):
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.plot(self.poisson_spike_counts)


def get_glm_file_name_base(nwb_file_name, epochs, time_bin_size, alpha, L1_wt, glm_restriction_name=None,
                           neural_bin_lags=None, max_units=None, cross_validation_method=None, n_splits=None,
                           simulation=False):

    params_text = get_glm_params_text(time_bin_size, alpha, L1_wt, glm_restriction_name, neural_bin_lags, max_units,
                                      cross_validation_method, n_splits, simulation)

    return f"glm_{format_nwb_file_name(nwb_file_name)}_eps{get_epochs_id(epochs)}_{params_text}"


def get_glm_params_text(time_bin_width, alpha, L1_wt, glm_restriction_name=None, neural_bin_lags=None, max_units=None,
                        cross_validation_method=None, n_splits=None, simulation=False):

    # Abbreviate cross validation method if passed
    if cross_validation_method is not None:
        cross_validation_method = CrossValidate.abbreviate_cross_validation_method(cross_validation_method)
    cross_validation_method_text = format_optional_var(cross_validation_method, append_underscore=True)
    n_splits_text = format_optional_var(n_splits, leading_text="nsplits", append_underscore=True)
    max_units_text = format_optional_var(max_units, leading_text="units", append_underscore=True)
    glm_restriction_name_text = format_optional_var(glm_restriction_name, append_underscore=True)
    neural_bin_lag_text = ""
    if neural_bin_lags is not None:
        neural_bin_lag_text = "_lag" + "_".join([str(x) for x in neural_bin_lags])
    simulation_text = format_bool(simulation, "sim", prepend_underscore=True)

    return f"{cross_validation_method_text}{glm_restriction_name_text}{max_units_text}" \
        f"bin{time_bin_width}_{n_splits_text}alpha{alpha}_L1{L1_wt}{neural_bin_lag_text}{simulation_text}"


def make_glm_model_name(covariate_types):
    return make_param_name(covariate_types, tolerate_non_unique=True, separating_character="_")


