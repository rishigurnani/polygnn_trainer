from os.path import exists
from polygnn_trainer.os import path_join, makedirs
from pickle import dump
import json
from . import constants


def safe_save(object, path, save_method):
    """
    Safely save objects to a path if the path does not already exist
    """
    if not exists(path):
        if save_method == "pickle":
            with open(path, "wb") as f:
                dump(
                    obj=object,
                    file=f,
                    protocol=constants.PICKLE_PROTOCOL,
                )
        elif save_method == "text":
            with open(path, "w") as f:
                f.write(object)
        elif save_method == "json":
            with open(path, "w") as f:
                json.dump(object, f)
    else:
        raise ValueError(f"{path} already exists. Object was not saved.")


def prepare_root(path_to_root):
    model_dir, md_dir = get_root_subdirs(path_to_root)
    makedirs(path_to_root)
    makedirs(model_dir)
    makedirs(md_dir)

    return model_dir, md_dir


def get_root_subdirs(path_to_root):
    model_dir = path_join(path_to_root, constants.MODELS_DIR)
    md_dir = path_join(path_to_root, constants.METADATA_DIR)

    return model_dir, md_dir
