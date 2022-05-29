import os

curr_file_path = os.path.dirname(os.path.abspath(__file__))

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except:
    print(f"Matplotlib not found in {curr_file_path}")
from sklearn.metrics import r2_score, mean_squared_error
from math import nan
import random
import numpy as np
import torch
import polygnn_trainer.constants as ks

# fix random seed
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


class DummyScaler:
    """
    This is a "Scaler" which just returns the object passed in.
    """

    def __init__(self):
        pass

    def transform(self, data):
        """
        Just return the data that is passed in
        """
        return data

    def inverse_transform(self, data):
        return data


def compute_batch_regression_metrics(y, y_hat, selectors, property_names, debug=False):
    """
    Compute regression metrics for several properties separately. Return
    the metrics as a dictionary of type {name: (rmse, r2)}

    Keyword Arguments and types:
    y - a *numpy array*
    y_hat - a *numpy array*
    selectors - list of *lists*
    property_names - list of *strings*. The order of strings
    should match the selector dimension
    """
    if debug:
        print(y[0:5])
        print(y_hat[0:5])
        print(selectors[0:5])
        print(property_names)

    return_dict = {}
    n_props = len(property_names)
    for ind in range(n_props):
        name = property_names[ind]
        data_subset = [j for j, x in enumerate(selectors) if x[ind] != 0.0]
        if len(data_subset) > 0:
            y_subset = y[data_subset]
            y_hat_subset = y_hat[data_subset]
            return_dict[name] = compute_regression_metrics(
                y_subset, y_hat_subset, mt=False
            )
        else:
            # if there are no samples correspond to name then the error metrics must be nan
            return_dict[name] = nan, nan

    return return_dict


def batch_scale_back(y, y_hat, scalers, selectors):
    return_y = np.zeros(y.shape)
    return_y_hat = np.zeros(y_hat.shape)
    n_props = len(scalers)
    names = list(scalers.keys())
    for ind in range(n_props):
        name = names[ind]
        scaler = scalers[name]
        data_subset = [j for j, x in enumerate(selectors) if x[ind] != 0.0]
        y_subset = np.expand_dims(y[data_subset], 0)  # scaler needs 2D array
        y_hat_subset = np.expand_dims(y_hat[data_subset], 0)  # scaler needs 2D array
        return_y[data_subset] = scaler.inverse_transform(y_subset).squeeze()
        return_y_hat[data_subset] = scaler.inverse_transform(y_hat_subset).squeeze()
    return return_y, return_y_hat


def mt_print_metrics(
    y_val, y_val_hat, selectors_val, scalers: dict, inverse_transform: bool
):
    """
    Compute and report error metrics of model predictions, separated out
    by each property in "scalers"

    Keyword arguments:
        y_val (iterable): Labels
        y_val_hat (iterable): Model predictions
        selectors_val (iterable): A collection of selector vectors
        scalers (dict)
        inverse_transform (bool): If true, scale (y_val, y_val_hat) back
            to the original scale of each property using the
            inverse_transform method of the correct scaler in scalers.
    """
    if inverse_transform:
        y_val, y_val_hat = batch_scale_back(y_val, y_val_hat, scalers, selectors_val)
    property_names = list(scalers.keys())
    error_dict = compute_batch_regression_metrics(
        y_val, y_val_hat, selectors_val, property_names, debug=False
    )
    for key in error_dict:
        print(
            f"[{key} orig. scale val rmse] {error_dict[key][0]} [{key} orig. scale val r2 {error_dict[key][1]}]",
            flush=True,
        )


def compute_regression_metrics(y, y_hat, mt, round_to=3):

    if mt:
        y = np.array(y)
        y_hat = np.array(y_hat)
        keep_inds = np.flatnonzero(y + 999)
        y_hat = y_hat[keep_inds]
        y = y[keep_inds]

    try:
        rmse = round(np.sqrt(mean_squared_error(y, y_hat)), round_to)
        r2 = round(r2_score(y, y_hat), round_to)
    except ValueError as e:
        print((y, y_hat))
        raise e
    return rmse, r2


def analyze_gradients(named_parameters, allow_errors=False):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            try:
                ave_grad = cpu_detach(p.grad.abs().mean().flatten())
                max_grad = cpu_detach(p.grad.abs().max().flatten())
                ave_grads.extend(ave_grad.numpy().tolist())
                max_grads.extend(max_grad.numpy().tolist())
                layers.append(n)
            except:
                print(n)
                print(p.grad)
                if not allow_errors:
                    raise BaseException
    ave_grads = np.array(ave_grads)
    max_grads = np.array(max_grads)
    print("\n..Ave_grads: ", list(zip(layers, ave_grads)))
    return layers, ave_grads, max_grads


def plot_grad_flow(named_parameters, filename, allow_errors=False):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    Source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    """

    layers, ave_grads, max_grads = analyze_gradients(
        named_parameters, allow_errors=False
    )

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.01)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def weight_reset(layer):
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()


def cpu_detach(tens):

    return tens.detach().cpu()


def get_unit_sequence(input_dim, output_dim, n_hidden):
    """
    Smoothly decay the number of hidden units in each layer.
    Start from 'input_dim' and end with 'output_dim'.

    Examples:
    get_unit_sequence(32, 8, 3) = [32, 16, 16, 16, 3]
    """

    decrement = lambda x: 2 ** (x // 2 - 1).bit_length()
    sequence = [input_dim]
    for _ in range(n_hidden):
        last_num_units = sequence[-1]
        power2 = decrement(last_num_units)
        if power2 > output_dim:
            sequence.append(power2)
        else:
            sequence.append(last_num_units)
    sequence.append(output_dim)

    return sequence


def sorted_attrs(obj):
    """
    Get back sorted attributes of obj. All methods are filtered out.
    """
    return sorted(
        [
            (a, v)
            for a, v in obj.__dict__.items()
            if not a.startswith("__") and not callable(getattr(obj, a))
        ],
        key=lambda x: x[0],
    )


def module_name(obj):
    klass = obj.__class__
    module = klass.__module__

    return module


def get_input_dim(data):
    """
    Get the input dimension of your PyTorch Data
    Keyword arguments:
        data (torch_geometric.data.Dara)
    """
    dim = 0
    for name in ks._F_SET:
        tens = getattr(data, name)
        if torch.numel(tens) == 0:
            dim += 0
        else:
            dim += tens.shape[1]
    return dim


def get_output_dim(data):
    """
    Get the output dimension of your PyTorch Data
    Keyword arguments:
        data (torch_geometric.data.Data)
    """
    return torch.numel(getattr(data, ks._Y))
