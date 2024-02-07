import numpy as np
import re


def camel_to_snake_case(string, group_uppercase=False):
    """Convert string in camel case (words capitalized, not separated)
    to string in snake case (words lowercase, separated by underline)"""
    if group_uppercase:  # keep consecutive uppercase together
        return _camel_to_snake_case_group_uppercase(string)
    return re.sub(r"(?<!^)r(?=[A-Z])", "_", string).lower()


def _camel_to_snake_case_group_uppercase(string):
    for idx, x in enumerate(string):
        # If first character, initialize string and continue
        if idx == 0:
            new_string = x.lower()
            continue
        # If previous character was lowercase and curent character uppercase, add underscore
        # before new character
        if x.isupper() and string[idx - 1].islower():
            new_string += "_"
        # If current character is upercase, previous character is uppercase, and next character
        # is lowercase, add underscore before new character
        if idx < len(string) - 1:  # if before last index
            if x.isupper() and string[idx - 1].isupper() and string[idx + 1].islower():
                new_string += "_"
        # Add new character
        new_string += x.lower()
    return new_string


def snake_to_camel_case(string, capitalize_first=True):
    split_string = string.split('_')
    if capitalize_first:
        return ''.join(x.title() for x in split_string)
    return split_string[0] + ''.join(x.title() for x in split_string[1:])


def plural_to_singular(string):
    if string[-1] == "s":
        return string[:-1]
    else:
        return string


def join_prepend_list_items(x, prefix=None):
    ret = "_".join([str(i) for i in x])
    if prefix is not None:
        ret = f"{str(prefix)}_{ret}"
    return ret


def underscore_to_space(string):
    return string.replace("_", " ")


def replace_multiple_substrings(string, dictionary):
    for old_substring, new_substring in dictionary.items():
        string = string.replace(old_substring, new_substring)
    return string


def leading_zero(x, add_zero_set=np.arange(1, 10)):
    if x in add_zero_set:
        return f"0{x}"
    else:
        return f"{x}"


def trailing_zero(x, add_zero_set=[0]):
    if x in add_zero_set:
        return f"{x}0"
    else:
        return f"{x}"


def strip_string(x, strip_character, strip_start=False, strip_end=True):
    if strip_start:
        if x[0] == strip_character:
            x = x[1:]
    if strip_end:
        if x[-1] == strip_character:
            x = x[:-1]
    return x


def strip_trailing_s(x):
    return strip_string(x, strip_character="s", strip_start=False, strip_end=True)


def add_underscore(x, prepend_underscore=False, append_underscore=True):
    return "_" * prepend_underscore + str(x) + "_" * append_underscore


def format_optional_var(x, leading_text="", prepend_underscore=False, append_underscore=False):
    if x is None:
        return ""
    return add_underscore(leading_text + str(x), prepend_underscore, append_underscore)


def format_bool(bool_val, bool_str, prepend_underscore=False, append_underscore=False):
    if bool_val:
        return add_underscore(bool_str, prepend_underscore, append_underscore)
    return ""


def abbreviate_camel_case(x):
    return "".join([z[0] for z in x.split("_")])


def abbreviate_join_strings(strings):
    # Abbreviate each string using first letter as uppercase. Join these across strings, separating
    # strings with hyphen
    return "-".join([x[0].upper() for x in strings])


def get_name_of_var(var):
    return f"{var=}".split('=')[0]  # var= in f string gives var=var value


def remove_leading_dunder(x):
    if x[:2] == "__":
        return x[2:]
    return x


def get_string_prior_to_dunder(x):
    return x.split("__")[0]


def replace_chars(x, replace_char_map):
    for old_char, new_char in replace_char_map.items():
        x = x.replace(old_char, new_char)
    return x

def get_even_odd_text(num):
    even_odd_bool = num % 2
    return {0: "even", 1: "odd"}[even_odd_bool]


def format_number(x):
    # Strip leading/trailing zeros from integer
    if x == int(x):
        x = int(x)

    return str(x)