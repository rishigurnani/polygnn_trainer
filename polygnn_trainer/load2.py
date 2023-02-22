# Loading that does not rely on pickle.

import json
import torch
import polygnn_trainer as pt


def pkl_to_json(path):
    return path.replace("pkl", "json")


def pkl_to_txt(path):
    return path.replace("pkl", "txt")


def safe_load_json(path):
    with open(path, "r") as f:
        _dict = json.load(f)
    return _dict


def load_selectors(root_dir):
    selectors_path = pkl_to_json(pt.load.get_selectors_path(root_dir))
    _dict = safe_load_json(selectors_path)
    return {k: torch.tensor(v) for k, v in _dict.items()}


def load_scalers(root_dir):
    scalers_path = pkl_to_json(pt.load.get_scalers_path(root_dir))
    _dict = safe_load_json(scalers_path)
    result = {}
    for k in _dict:
        scaler = pt.scale.SequentialScaler()
        scaler.from_string(_dict[k])
        result[k] = scaler
    return result


def load_hps(root_dir):
    hps_path = pkl_to_txt(pt.load.get_hps_path(root_dir))
    with open(hps_path, "r") as f:
        hp_str = f.read()
    hps = pt.hyperparameters.HpConfig()
    hps.set_values_from_string(hp_str)
    return hps


def load_features(root_dir):
    features_path = pkl_to_txt(pt.load.get_features_path(root_dir))
    with open(features_path, "r") as f:
        string = f.read()
    string = string.replace("The graph features used during training are: ", "")
    string = string.replace("The node features used during training are: ", "")
    string = string.split("\n")
    _dict = {
        "graph_srt_keys": string[0].split(", "),
        "node_srt_keys": string[-1].split(", "),
    }
    return _dict
