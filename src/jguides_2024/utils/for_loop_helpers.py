import numpy as np

from src.jguides_2024.utils.list_helpers import duplicate_elements


def return_for_loop_variable_lists(for_loop_variables_lists):
    """
    Return variables corresponding to each evaluated statement within a nested for loop
    :param variables_lists: list with [outermost_variables, ..., innermost_variables]
    :return: variables_lists: list with [outermost_variables_corresponding_to_each_evaluated_statement,...,
            innermost_variables_corresponding_to_each_evaluated_statement]
    """

    # TODO (feature): figure out how to code this up in a general way (same code no matter how many nests in for loop)
    # Ensure for loop variables in list
    for_loop_variables_lists = [list(x) for x in for_loop_variables_lists]
    if len(for_loop_variables_lists) == 3:  # 3 nested for loops
        level_1_variables, level_2_variables, level_3_variables = for_loop_variables_lists
        return [duplicate_elements(level_1_variables, len(level_2_variables) * len(level_3_variables)),
                duplicate_elements(level_2_variables, len(level_3_variables)) * len(level_1_variables),
                level_3_variables * len(level_2_variables) * len(level_1_variables)]


def print_iteration_progress(iteration_num, num_iterations, target_num_print_statements=10):
    min_step_size = 1/target_num_print_statements
    target_step_size = np.round(num_iterations / target_num_print_statements)
    step_size = np.max((target_step_size, min_step_size))

    if iteration_num in np.arange(0, num_iterations,
                                  step_size):
        print(f"{np.round(iteration_num / num_iterations * 100)}%", flush=True)


def stoppable_function_loop(function_map, stop_loop_after, function_args=None):
    for schema_name, function in function_map.items():
        # This setup avoids having to accept kwargs in all functions we want to use this loop with
        if function_args is None:
            function()
        else:
            function(**function_args)
        if schema_name == stop_loop_after:
            return True
    return False


def stoppable_function_outer_loop(outer_functions, stop_loop_after, function_args=None):
    """
    Wrapper for looping through functions that loop through functions
    :param outer_functions: List with functions that take function maps
    :param stop_loop_after: text, if matches key in function map, break out of loop
    :param function_args: dictionary with arguments to pass to function
    :return:
    """

    for outer_function in outer_functions:
        # This setup avoids having to accept kwargs in all functions we want to use this loop with
        if function_args is None:
            stop_loop = outer_function(stop_loop_after)
        else:
            stop_loop = outer_function(stop_loop_after, function_args)
        if stop_loop:
            return
