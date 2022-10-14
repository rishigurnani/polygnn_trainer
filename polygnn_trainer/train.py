import os

curr_file_path = os.path.dirname(os.path.abspath(__file__))

from torch import nn, manual_seed, optim, cuda
from torch.cuda import amp
from torch import save as torch_save
from torch import device as torch_device
import numpy as np
from torch_geometric.loader import DataLoader
from collections import deque
import random
from os.path import join as path_join
from sklearn.model_selection import KFold
from dataclasses import dataclass

try:
    from . import constants
except ModuleNotFoundError:
    import constants
from .utils import (
    weight_reset,
    cpu_detach,
    analyze_gradients,
    compute_regression_metrics,
    mt_print_metrics,
)
from .hyperparameters import HpConfig
from .infer import eval_submodel
from .scale import *
from . import save
from polygnn_trainer import prepare

# fix random seed
random.seed(2)
manual_seed(2)
np.random.seed(2)


def initialize_training(model, r_learn, device):
    """
    Initialize a model and optimizer for training using just the model's class
    """
    # deal with optimizer
    optimizer = optim.Adam(model.parameters(), lr=r_learn)  # Adam optimization
    # deal with model
    # implementation modified from https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
    model.apply(weight_reset)
    model = model.to(device)
    model.train()

    return model, optimizer


def amp_train(model, data, optimizer, tc, selector_dim):
    """
    This function handles the parts of the per-epoch loop that torch's
    autocast methods can speed up. See https://pytorch.org/docs/1.9.1/notes/amp_examples.html
    """
    if tc.amp:
        with amp.autocast(device_type=tc.device, enabled=True):
            if tc.multi_head:
                output = model(data).view(data.num_graphs, selector_dim)
            else:
                output = model(data)
            loss = tc.loss_obj(output, data)
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            amp.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            amp.scaler.step(optimizer)

            # Updates the scale for next iteration.
            amp.scaler.update()
    else:
        output, loss = minibatch(data, tc, model, selector_dim)
        loss.backward()
        optimizer.step()

    # the only thing we need to return is "output". Both "model" and
    # "optimizer" get updated. But these updates happen in place.
    return output


def minibatch(data, tc, model, selector_dim):
    if tc.multi_head:
        output = model(data).view(data.num_graphs, selector_dim)
    else:
        output = model(data)
    loss = tc.loss_obj(output, data)

    return output, loss


def train_submodel(
    model,
    train_pts,
    val_pts,
    scalers,
    tc,  # train_config
):
    """
    Train a model and save it to tc.model_save_path.
    
    Keyword arguments:
        model (polygnn_trainer.std_module.StandardModule): The model architecture.
        train_pts (List[pyg.data.Data]): The training data.
        val_pts (List[pyg.data.Data]): The validation data.
        scalers (Dict[str, polygnn_trainer.scale.SequentialScaler]): Scalers for
            each property/task being modeled.
    """
    # error handle inputs
    if tc.model_save_path:
        if not tc.model_save_path.endswith(".pt"):
            raise ValueError(f"The model_save_path you passed in does not end in .pt")
    # create the epoch suffix for this submodel
    epoch_suffix = f"{tc.epoch_suffix}, fold {tc.fold_index}"
    # to determine the presence of a "selector_dim" the first member
    # of val_X should have an attribute named selector and the value of
    # the attribute should not be None
    if hasattr(val_pts[0], "selector") and (cpu_detach(val_pts[0].selector) != None):
        selector_dim = val_pts[0].selector.size()[-1]
    else:
        selector_dim = None

    model.to(tc.device)
    optimizer = optim.Adam(
        model.parameters(), lr=tc.hps.r_learn.value
    )  # Adam optimization
    val_loader = DataLoader(
        val_pts, batch_size=tc.hps.batch_size.value * 2, shuffle=True
    )
    # intialize a few variables that get reset during the training loop
    min_val_rmse = np.inf  # epoch-wise loss
    max_val_r2 = -np.inf
    best_val_epoch = 0
    vanishing_grads = False
    exploding_grads = False
    exploding_errors = False
    grad_hist_per_epoch = deque(
        maxlen=constants.GRADIENT_HISTORY_LENGTH
    )  # gradients for last maxlen epochs

    # if we do not need to make a new dataloader inside each epoch,
    # let us make the dataloader now.
    if not tc.get_train_dataloader:
        train_loader = DataLoader(
            train_pts, batch_size=tc.hps.batch_size.value, shuffle=True
        )
    for epoch in range(tc.epochs):
        # Let's stop training and not waste time if we have vanishing
        # gradients early in training. We won't
        # be able to learn anything anyway.
        if vanishing_grads:
            print("Vanishing gradients detected")
        if exploding_errors:
            print("Exploding errors detected")
        if exploding_grads:
            print("Exploding gradients detected")
        if (vanishing_grads or exploding_grads) and (epoch < 50):
            break
        # if the errors or gradients are messed up later in training,
        # let us just re-initialize
        # the model. Perhaps this new initial point on the loss surface
        # will lead to a better local minima
        elif exploding_errors or vanishing_grads or exploding_grads:
            model, optimizer = initialize_training(
                model, tc.hps.r_learn.value, tc.device
            )
        # augment data, if necessary
        if tc.get_train_dataloader:
            train_pts = tc.get_train_dataloader()
            train_loader = DataLoader(
                train_pts, batch_size=tc.hps.batch_size.value, shuffle=True
            )
        # train loop
        y = []
        y_hat = []
        selectors = []
        model.train()
        for _, data in enumerate(train_loader):  # loop through training batches
            data = data.to(tc.device)
            optimizer.zero_grad()
            output = amp_train(model, data, optimizer, tc, selector_dim)
            y += data.y.flatten().cpu().numpy().tolist()
            y_hat += output.flatten().detach().cpu().numpy().tolist()
            if selector_dim:
                selectors += data.selector.cpu().numpy().tolist()

        _, ave_grads, _ = analyze_gradients(
            model.named_parameters(), allow_errors=False
        )
        grad_hist_per_epoch.append(ave_grads)

        # rmse on data in loss function scale
        tr_rmse, tr_r2 = compute_regression_metrics(y, y_hat, tc.multi_head)
        # check for exploding errors, vanishing grads, and exploding grads
        if tr_r2 < constants.DL_STOP_TRAIN_R2:
            exploding_errors = True
        else:
            exploding_errors = False
        if np.sum(grad_hist_per_epoch) == 0:
            vanishing_grads = True
        else:
            vanishing_grads = False
        if int(np.sum(np.isnan(grad_hist_per_epoch))) == len(grad_hist_per_epoch):
            exploding_grads = True
        else:
            exploding_grads = False
        # ################################################################
        # val loop
        y_val, y_val_hat, selectors_val = eval_submodel(
            model, val_loader, tc.device, selector_dim
        )
        print(f"\nEpoch {epoch}{epoch_suffix}", flush=True)
        val_rmse, val_r2 = compute_regression_metrics(y_val, y_val_hat, tc.multi_head)
        if selector_dim:
            # print overall error metrics
            print(
                "[loss scale val rmse] %s [loss scale val r2] %s [loss scale tr rmse] %s [loss scale tr r2] %s"
                % (val_rmse, val_r2, tr_rmse, tr_r2),
                flush=True,
            )
            # print error metrics per property
            mt_print_metrics(
                y_val, y_val_hat, selectors_val, scalers, inverse_transform=True
            )
        else:
            print(
                "[val rmse] %s [val r2] %s [tr rmse] %s [tr r2] %s"
                % (val_rmse, val_r2, tr_rmse, tr_r2),
                flush=True,
            )

        if val_rmse < min_val_rmse:
            min_val_rmse = val_rmse
            max_val_r2 = val_r2
            best_val_epoch = epoch
            if tc.model_save_path:
                torch_save(model.state_dict(), tc.model_save_path)
                print("Best model saved", flush=True)

        print(
            "[best val epoch] %s [best val loss scale rmse] %s [best val loss scale r2] %s"
            % (best_val_epoch, min_val_rmse, max_val_r2),
            flush=True,
        )

    return min_val_rmse


def train_kfold_ensemble(
    dataframe,
    model_constructor,
    train_config,
    submodel_trainer,
    augmented_featurizer,
    scaler_dict,
    root_dir,
    n_fold,
    random_seed,
):
    """
    Train an ensemble model on dataframe.

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of atleast two columns:
            value, data. smiles_string is an optional column. This dataframe
            should not contain any na.
        model_constructor: A lambda function that returns an nn.Module object
            when called.
        train_config (trainConfig)
        augmented_featurizer: A function that takes in a smiles string,
            augments it, and returns its features.
        scaler_dict (dict):
        root_dir (str): The path to directory where all data will be
            saved. This string should match the string used in
            prepare_train.
        n_fold (int): The number of folds to use during training
        random_seed (int)
    """
    # The root directories and subdirectories should have already been
    # created by prepare_train. Let us retrieve the path to the
    # subdirectories so that we can save some more stuff
    model_dir, md_dir = save.get_root_subdirs(root_dir)
    # save scaler_dict
    save.safe_save(
        scaler_dict, path_join(md_dir + constants.SCALERS_FILENAME), "pickle"
    )
    # make selector dict and save it
    prop_cols = sorted(list(scaler_dict.keys()))
    selector_dict = {}
    for prop in prop_cols:
        selector = dataframe[dataframe.prop == prop].selector.values.tolist()[0]
        if not isinstance(selector, Tensor):
            selector = tensor(selector)
        selector.requires_grad = False
        selector_dict[prop] = selector
    save.safe_save(
        selector_dict, path_join(md_dir + constants.SELECTORS_FILENAME), "pickle"
    )
    # save property name list
    save.safe_save(
        "\n".join(prop_cols), path_join(md_dir + constants.PROP_FILENAME), "text"
    )
    # save hyperparams
    save.safe_save(
        train_config.hps, path_join(md_dir + constants.HPS_FILENAME), "pickle"
    )
    # ######################################
    # helper functions for CPU-based data augmentation
    # ######################################
    def get_data_augmented(x):
        """
        Return a Data object with the augmented smiles. This function
        requires that dataframe contains a column named data
        """
        data = augmented_featurizer(x.smiles_string)
        data.y = x.data.y  # copy label directly from x.data as that label
        # should already be scaled.
        prepare.copy_attribute_safe(x.data, data, "selector")
        prepare.copy_attribute_safe(x.data, data, "graph_feats")
        prepare.copy_attribute_safe(x.data, data, "node_feats")

        return data

    def cv_get_train_dataloader(training_df):
        """
        Return a list of augmented Data objects
        """

        return training_df.apply(get_data_augmented, axis=1).values.tolist()

    # #########################################

    # do cross-validation
    kf_ = KFold(
        n_splits=n_fold,
        shuffle=True,
        random_state=random_seed,
    )
    kf = kf_.split(range(len(dataframe)))
    ind = 0
    for train, val in kf:
        # train submodel
        print(f"Fold {ind}: training inds are ... {train}")
        print(f"Fold {ind}: validation inds are ... {val}")
        train_config.model_save_path = path_join(model_dir, f"model_{ind}.pt")
        training_df = dataframe.iloc[train, :]
        val_df = dataframe.iloc[val, :]
        val_pts = val_df["data"].values.tolist()
        train_pts = training_df["data"].values.tolist()
        if augmented_featurizer:
            train_config.get_train_dataloader = lambda: cv_get_train_dataloader(
                training_df
            )  # get one of multiple equivalent graphs for training data
        model = model_constructor()
        train_config.fold_index = ind  # add the fold index to train_config
        submodel_trainer(model, train_pts, val_pts, scaler_dict, train_config)
        ind += 1


@dataclass
class trainConfig:
    """
    A class to pass into the submodel trainer
    """

    # need to be set manually
    loss_obj: nn.Module
    amp: bool  # when using T2 this should be set to False
    # hps: HpConfig = None
    # loss_obj: nn.Module = None
    # device: torch_device = None
    # amp: bool = False # this should automatically

    # initialized during instantiation
    device: torch_device = torch_device(
        "cuda" if cuda.is_available() else "cpu"
    )  # specify GPU
    epoch_suffix: str = ""
    multi_head: bool = None

    # set dynamically inside train_kfold_ensemble, so
    # we can set each attribute to None on instantiation
    hps: HpConfig = None
    model_save_path: str = None
    fold_index = None
    get_train_dataloader = None


def prepare_train(
    dataframe,
    smiles_featurizer,
):
    """
    An alias to prepare.prepare_train
    """
    return prepare.prepare_train(
        dataframe,
        smiles_featurizer,
    )
