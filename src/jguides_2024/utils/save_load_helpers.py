import json
import os
import pickle

import numpy as np

from src.jguides_2024.utils.cd_make_if_nonexistent import cd_make_if_nonexistent
from src.jguides_2024.utils.plot_helpers import get_default_plot_save_dir


def pickle_file(data, file_name, save_dir=None, overwrite=False):
    if save_dir is not None:
        os.chdir(save_dir)  # change to directory where want to save file
    print(f"saving {file_name}")
    if os.path.exists(file_name) and not overwrite:
        raise Exception(f"{file_name} already exists at {os.getcwd()}")
    pickle.dump(data, open(file_name, "wb"))  # save data


def unpickle_file(file_name, save_dir=None):
    if save_dir is not None:
        os.chdir(save_dir)
    return pickle.load(open(file_name, "rb"))


def append_iteration_num_to_file_name(file_name_base, save_dir):
    # Get files in directory where want to save file
    os.chdir(save_dir)
    dir_file_names = os.listdir()
    # Define current iteration as one more than largest iteration from past files. If no past files, define
    # current iteration as zero
    current_iteration = 0  # default
    previous_iterations = [int(x.split("_iteration")[-1]) for x in dir_file_names if file_name_base in x]
    if len(previous_iterations) > 0:
        current_iteration = np.max(previous_iterations) + 1
    return f"{file_name_base}_iteration{current_iteration}"


def get_file_contents(file_name, file_path=None):

    # Get current directory so can change back to it
    current_dir = os.getcwd()

    # Change to directory with file if passed
    if file_path is not None:
        os.chdir(file_path)

    file_obj = open(file_name, "r")
    file_contents = file_obj.read()
    file_obj.close()

    # Change back to current directory
    os.chdir(current_dir)

    return file_contents


def save_json(file_name, file_contents, save_dir=None):

    # Save contents to json file

    # Get inputs if not passed
    if save_dir is None:
        save_dir = get_default_plot_save_dir()
    current_dir = os.getcwd()  # change back to this directory after saving
    cd_make_if_nonexistent(save_dir)
    file_name += ".json"
    print(f"Saving {file_name} in {os.getcwd()}")
    with open(file_name, "w") as f:
        print(f"Saving {file_name}")
        f.write(json.dumps(file_contents))
    f.close()
    os.chdir(current_dir)  # change back to directory

