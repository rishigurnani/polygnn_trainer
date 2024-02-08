import numpy as np
from torch import device as torch_device
from torch import cuda
from torch_geometric.loader import DataLoader

from polygnn_trainer import constants
from polygnn_trainer.load import load_ensemble, load_selectors
from polygnn_trainer.load2 import load_selectors as load_selectors2

from .prepare import prepare_infer


def modulate_dropout(model, mode):
    """
    Function to enable the dropout layers during test-time.
    Taken from https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            if mode == "train":
                m.train()
            elif mode == "test":
                m.eval()
            else:
                raise ValueError("Invalid option passed in for mode")


def _model_eval_mode(model, dropout_mode):
    """
    Function to control how the model behaves in
    eval mode
    """
    model.eval()
    if dropout_mode == "train":
        modulate_dropout(model, "train")


def init_evaluation(model):
    y_val = []  # true labels
    y_val_hat_mean = []  # prediction mean
    y_val_hat_std = []  # prediction uncertainty
    selectors = []
    model.eval()

    return y_val, y_val_hat_mean, y_val_hat_std, selectors


def eval_ensemble(
    model,
    root_dir,
    dataframe,
    smiles_featurizer,
    device=torch_device("cuda" if cuda.is_available() else "cpu"),
    ensemble_kwargs_dict={},
):
    """
    Evaluate ensemble on the data contained in dataframe.

    Keyword arguments:
        model (nn.Module): The ensemble to be evaluated
        root_dir (str): The path to the directory containing the ensemble
            information
        dataframe (pd.DataFrame): The data to evaluate, in melted form.
        smiles_featurizer
        device (torch.device)
        ensemble_kwargs_dict: Arguments to pass into the 'forward' method
            of the ensemble

    Outputs:
        y (np.ndarray): Data labels
        y_hat_mean (np.ndarray): Mean of data predictions
        y_hat_std (np.ndarray): Std. dev. of data predictions
        y_selectors (np.ndarray): Selector for each data point
    """
    model.to(device)
    try:
        selectors = load_selectors(root_dir)
    except FileNotFoundError:
        selectors = load_selectors2(root_dir)
    selector_dim = len(selectors)
    # prepare dataframe
    dataframe = prepare_infer(
        dataframe,
        smiles_featurizer,
        selectors,
        root_dir=root_dir,
    )
    data_ls = dataframe.data.values.tolist()
    loader = DataLoader(data_ls, batch_size=constants.BS_MAX, shuffle=False)

    return _evaluate_ensemble(
        model, loader, device, selector_dim, **ensemble_kwargs_dict
    )


def _evaluate_ensemble(model, val_loader, device, selector_dim, **kwargs):
    """
    Evaluate ensemble on the data contained in val_loader. This function is not
    to be called directly. It is a helper function for eval_ensemble.
    Keyword arguments:
        model (nn.Module): The ensemble to be evaluated
        val_loader (DataLoader): The data to evaluate
        device (torch.device)
        selector_dim (int): The number of selector dimensions
        **kwargs: Arguments to pass into the 'forward' method of model
    Outputs:
        y (np.ndarray): Data labels
        y_hat_mean (np.ndarray): Mean of data predictions
        y_hat_std (np.ndarray): Std. dev. of data predictions
        y_selectors (np.ndarray): Selector for each data point
    """
    return _evaluate(model, val_loader, device, False, selector_dim, **kwargs)


def eval_submodel(model, val_loader, device, selector_dim=None):
    """
    Evaluate model on the data contained in val_loader.

    Outputs:
        y (np.ndarray): Data labels
        y_hat (np.ndarray): Data predictions
        y_selectors (np.ndarray): Selector for each data point
    """
    return _evaluate(
        model, val_loader, device, is_submodel=True, selector_dim=selector_dim
    )


def _evaluate(model, val_loader, device, is_submodel, selector_dim, **kwargs):
    """
    Evaluate model on the data contained in val_loader. This function is not
    to be called directly. It is a helper function for eval_submodel and
    eval_ensemble.
    """

    y_val, y_val_hat_mean, y_val_hat_std, selectors = init_evaluation(model)
    for ind, data in enumerate(val_loader):  # loop through validation batches
        data = data.to(device)
        # sometimes the batch may have labels associated. Let's check
        if data.y is not None:
            y_val += data.y.detach().flatten().cpu().numpy().tolist()
        # sometimes the batch may have selectors associated. Let's check
        if selector_dim:
            selectors += data.selector.cpu().numpy().tolist()
        if is_submodel:
            output = model(data).view(
                data.num_graphs,
            )
            y_val_hat_mean += output.flatten().detach().cpu().numpy().tolist()
        # if we are not dealing with a submodel then we have an ensemble.
        # The ensemble will have two outputs: the mean and standard deviation.
        else:
            mean, std = model(data, **kwargs)
            y_val_hat_mean += mean.flatten().detach().cpu().numpy().tolist()
            y_val_hat_std += std.flatten().detach().cpu().numpy().tolist()
    del data  # free memory
    if is_submodel:
        return np.array(y_val), np.array(y_val_hat_mean), selectors
    else:
        return (
            np.array(y_val),
            np.array(y_val_hat_mean),
            np.array(y_val_hat_std),
            selectors,
        )
