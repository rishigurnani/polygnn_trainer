from polygnn_trainer.os import path_join
from os import listdir
import pickle
from torch import load as torch_load
from re import search, compile


from . import constants as ks
from . import models


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
    selectors_path = path_join(root_dir, ks.METADATA_DIR, ks.SELECTORS_FILENAME)
    with open(selectors_path, "rb") as f:
        return pickle.load(f)


def load_features(root_dir):
    path = path_join(root_dir, ks.METADATA_DIR, ks.FEATURE_FILENAME_PKL)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scalers(root_dir):
    scalers_path = path_join(root_dir, ks.METADATA_DIR, ks.SCALERS_FILENAME)
    with open(scalers_path, "rb") as f:
        return pickle.load(f)


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
    # load hps
    hps_path = path_join(root_dir, ks.METADATA_DIR, ks.HPS_FILENAME)
    with open(hps_path, "rb") as f:
        hps = pickle.load(f)
    # load scalers
    scalers_path = path_join(root_dir, ks.METADATA_DIR, ks.SCALERS_FILENAME)
    with open(scalers_path, "rb") as f:
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
