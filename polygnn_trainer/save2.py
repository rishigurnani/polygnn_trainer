# Saving that does not rely on pickle.

from .save import safe_save


def save_selectors(selector_dict, path):
    """
    Keyword arguments:
        selector_dict (Dict[str:np.array])
        path (str)
    """
    obj = {k: v.tolist() for k, v in selector_dict.items()}
    safe_save(obj, path, "json")


def save_scalers(scaler_dict, path):
    """
    Keyword arguments:
        scaler_dict (Dict[str:SequentialScaler])
        path (str)
    """
    obj = {k: str(v) for k, v in scaler_dict.items()}
    safe_save(obj, path, "json")


def save_hps(hps, path):
    """
    Keyword arguments:
        hps (HpConfig)
        path (str)
    """
    safe_save(str(hps), path, "text")
