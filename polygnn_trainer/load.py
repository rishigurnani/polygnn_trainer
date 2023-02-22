from polygnn_trainer.os import path_join
from polygnn_trainer.hyperparameters import HpConfig
from os import listdir, path
import pickle
from torch import load as torch_load
from re import search, compile


from . import constants as ks
from . import models, load2


def get_selectors_path(root):
    return path_join(root, ks.METADATA_DIR, ks.SELECTORS_FILENAME)


def file_filter(root_dir, pattern):
    if isinstance(pattern, str):
        pattern = compile(pattern)
    return [path_join(root_dir, f) for f in listdir(root_dir) if search(pattern, f)]


def load_model(path, submodel_cls, **kwargs):
    model = submodel_cls(
        **kwargs,
    )
    model.load_state_dict(torch_load(path))
    return model


def load_selectors(root_dir):
    selectors_path = get_selectors_path(root_dir)
    with open(selectors_path, "rb") as f:
        return pickle.load(f)


def get_features_path(root_dir):
    return path_join(root_dir, ks.METADATA_DIR, ks.FEATURE_FILENAME_PKL)


def load_features(root_dir):
    path = get_features_path(root_dir)
    with open(path, "rb") as f:
        return pickle.load(f)


def get_scalers_path(root_dir):
    return path_join(root_dir, ks.METADATA_DIR, ks.SCALERS_FILENAME)


def load_scalers(root_dir):
    scalers_path = get_scalers_path(root_dir)
    with open(scalers_path, "rb") as f:
        return pickle.load(f)


def get_hps_path(root_dir):
    return path_join(root_dir, ks.METADATA_DIR, ks.HPS_FILENAME)


def safe_pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_hps(root_dir, base_hps=HpConfig()):
    """
    Replace the values of `base_hps` with the saved values in `root_dir`.
    """
    hps_path = get_hps_path(root_dir)
    loaded_hps = safe_pickle_load(hps_path)
    if loaded_hps:
        assert type(base_hps) == type(
            loaded_hps
        ), f"{type(base_hps)}.{type(loaded_hps)}"
        # If hyperparameters were saved, let's load them into base_hps.
        base_hps.set_values_from_string(str(loaded_hps))
    return base_hps


def load_ensemble(root_dir, submodel_cls, device, submodel_kwargs_dict):
    """
    Load the ensemble from root_dir. The ensemble type will
    be inferred from the contents of root_dir

    Keywords args
        root_dir: The path to the directory containing the model information
        submodel_cls: The class corresponding to the submodel to load
        device (torch.device)
        submodel_kwargs_dict: Other arguments needed to instantiate the submodel
    """
    # #########
    # Load hps
    # #########
    hps_path_pkl = path_join(root_dir, ks.METADATA_DIR, ks.HPS_FILENAME)
    # Use the txt file if it exists.
    if path.exists(load2.pkl_to_txt(hps_path_pkl)):
        hps = load2.load_hps(root_dir)
    else:
        with open(hps_path_pkl, "rb") as f:
            hps = pickle.load(f)
    # #############
    # Load scalers
    # #############
    scalers_path_pkl = path_join(root_dir, ks.METADATA_DIR, ks.SCALERS_FILENAME)
    # Use the json file if it exists.
    if path.exists(load2.pkl_to_json(scalers_path_pkl)):
        scalers = load2.load_scalers(root_dir)
    else:
        with open(scalers_path_pkl, "rb") as f:
            scalers = pickle.load(f)
    # get the path to all submodels
    model_dir = path_join(root_dir, ks.MODELS_DIR)
    submodel_paths = sorted(file_filter(model_dir, ks.submodel_re))
    # determine the ensemble class
    all_model_dir_paths = sorted(
        file_filter(model_dir, ".*")
    )  # ".*" should match anything
    if submodel_paths == all_model_dir_paths:
        ensemble_cls = models.LinearEnsemble
    # load the submodels
    submodel_dict = {
        ind: load_model(path, submodel_cls, hps=hps, **submodel_kwargs_dict)
        for ind, path in enumerate(submodel_paths)
    }
    # send each submodel to the appropriate device
    for model in submodel_dict.values():
        model.to(device)
    # instantiate the ensemble
    ensemble = ensemble_cls(
        submodel_dict,
        device,
        scalers,
    )

    return ensemble
