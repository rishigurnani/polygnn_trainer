import pytest
from os import remove
from torch import manual_seed, nn
from shutil import rmtree
import pandas as pd
from skopt import gp_minimize
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from polygnn_trainer import __version__
from polygnn_trainer import save, loss, constants, models
from polygnn_trainer import utils as pt_utils
from polygnn_trainer.hyperparameters import HpConfig
from polygnn_trainer.infer import eval_ensemble
from polygnn_trainer.layers import my_hidden
from polygnn_trainer.load import load_ensemble, load_selectors
from polygnn_trainer.train import (
    train_kfold_ensemble,
    train_submodel,
    trainConfig,
)
from polygnn_trainer.prepare import prepare_train

from .utils_test import MathModel1, trainer_MathModel, morgan_featurizer

# set seeds for reproducibility
random.seed(12)
manual_seed(12)
np.random.seed(12)


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def example_data():
    properties = ["property1"] * 7 + ["property2"] * 7
    values = np.random.randn(
        14,
    )  # random data
    smiles = [
        "[*]CC[*]",
        "[*]CC(C)[*]",
        "[*]CCN[*]",
        "[*]CCO[*]",
        "[*]CCCN[*]",
        "[*]C(O)C[*]",
        "[*]C(CCC)C[*]",
    ] * 2
    data = {"prop": properties, "value": values, "smiles_string": smiles}
    dataframe = pd.DataFrame(data)
    return {
        "dataframe": dataframe,
        "properties": properties,
    }


@pytest.fixture
def example_mt_data(example_data):
    root = ("mt_ensemble/",)  # root directory for ensemble trained on MT data
    return {
        "root": root,
    }


@pytest.fixture
def example_st_data(example_data):
    root = ("st_ensemble/",)  # root directory for ensemble trained on MT data
    return {
        "root": root,
    }


def test_safe_save_pickle(example_data):
    # save something the first time. This should go without a hitch.
    obj = example_data["properties"]
    save.safe_save(obj, "properties.pkl", "pickle")
    with pytest.raises(ValueError):
        # try saving the same thing again
        save.safe_save(obj, "properties.pkl", "pickle")


# a mark so that tests are run on both single-task and multi-task data
@pytest.mark.parametrize("fixture", ["example_st_data", "example_mt_data"])
def test_ensemble_trainer(fixture, request, example_data):
    """
    Test if we can run train_kfold_ensemble after hyperparameter optimization without error
    """
    data_for_test = request.getfixturevalue(fixture)
    root = data_for_test["root"]
    dataframe, scaler_dict = prepare_train(
        example_data["dataframe"], morgan_featurizer, root_dir=root
    )
    training_df, val_df = train_test_split(
        dataframe,
        test_size=constants.VAL_FRAC,
        stratify=dataframe.prop,
        random_state=constants.RANDOM_SEED,
    )
    train_pts, val_pts = training_df.data.values.tolist(), val_df.data.values.tolist()
    epochs = 2
    # create hyperparameter space
    hp_space = [
        (np.log10(0.0003), np.log10(0.03)),  # learning rate
        (1, 10),  # batch size
        (0, 0.5),  # dropout
        (2, 4),  # capacity
    ]
    input_dim = pt_utils.get_input_dim(train_pts[0])
    output_dim = pt_utils.get_output_dim(train_pts[0])
    # create objective function
    def obj_func(x):
        print(f"Testing hps: {x}")
        hps = HpConfig()
        hps.r_learn.set_value(10 ** x[0])
        hps.batch_size.set_value(x[1])
        hps.dropout_pct.set_value(x[2])
        hps.capacity.set_value(x[3])
        hps.activation.set_value(nn.functional.leaky_relu)

        # trainConfig for the hp search
        tc_search = trainConfig(
            loss_obj=loss.sh_mse_loss(),
            amp=False,  # False since we are running a test on a CPU
        )
        # some attributes need to be defined AFTER instantiation since
        # __init__ does not know about them
        tc_search.hps = hps
        tc_search.device = torch.device("cpu")
        tc_search.epochs = epochs
        tc_search.do_augment = False
        tc_search.multi_head = False
        model = models.MlpOut(
            input_dim=input_dim,
            output_dim=output_dim,
            hps=hps,
        )
        val_rmse = train_submodel(  # val_rmse = RMSE on the validation data set, contained in 'val_pts'.
            model,
            train_pts,
            val_pts,
            scaler_dict,
            tc_search,
        )
        return val_rmse

    # obtain the optimal point in hp space
    opt_obj = gp_minimize(
        func=obj_func,
        dimensions=hp_space,
        n_calls=10,
        random_state=0,
    )
    # create an HpConfig from the optimal point in hp space
    hps = HpConfig()
    hps.r_learn.set_value(10 ** opt_obj.x[0])
    hps.batch_size.set_value(opt_obj.x[1])
    hps.dropout_pct.set_value(opt_obj.x[2])
    hps.capacity.set_value(opt_obj.x[3])
    hps.activation.set_value(nn.functional.leaky_relu)
    # create inputs for train_kfold_ensemble
    model_constructor = lambda: models.MlpOut(
        input_dim=input_dim,
        output_dim=output_dim,
        hps=hps,
    )
    dataframe = pd.concat(
        [val_df, training_df],
        ignore_index=True,
    )
    # trainConfig for the ensemble training
    tc_ensemble = trainConfig(
        loss_obj=loss.sh_mse_loss(),
        amp=False,  # False since we are running a test on a CPU
    )
    # some attributes need to be defined AFTER instantiation since
    # __init__ does not know about them
    tc_ensemble.hps = hps
    tc_ensemble.device = torch.device("cpu")
    tc_ensemble.epochs = epochs
    tc_ensemble.do_augment = False
    tc_ensemble.multi_head = False
    tc_ensemble.loss_obj = loss.sh_mse_loss()
    train_kfold_ensemble(
        dataframe,
        model_constructor,
        tc_ensemble,
        train_submodel,
        augmented_featurizer=None,  # since we do not want augmentation
        scaler_dict=scaler_dict,
        root_dir=root,
        n_fold=2,
        random_seed=234,
    )

    assert True


# a mark so that tests are run on both single-task and multi-task data
@pytest.mark.parametrize("fixture", ["example_st_data", "example_mt_data"])
def test_load_ensemble_noerror(fixture, request):
    """
    This test checks that load_ensemble can be performed without error
    """
    data_for_test = request.getfixturevalue(fixture)
    selectors = load_selectors(data_for_test["root"])
    selector_dim = torch.numel(list(selectors.values())[0])
    # calculate input dimension
    input_dim = 512 + selector_dim
    ensemble = load_ensemble(
        data_for_test["root"],
        models.MlpOut,
        device="cpu",
        submodel_kwargs_dict={
            "input_dim": input_dim,
            "output_dim": 1,
        },
    )

    assert len(ensemble.submodel_dict) > 0


@pytest.fixture
def example_linear_data():
    root_dir = "ensemble_linear/"
    graph_feats = [
        {"feat0": 0, "feat1": 0},
        {"feat0": 1, "feat1": 1},
        {"feat0": 2, "feat1": 2},
        {"feat0": 3, "feat1": 3},
    ] * 2
    props = ["prop1"] * 4 + ["prop2"] * 4
    values = [0, 99, 999, 9999] * 2  # trick the code into scaling
    data = {"prop": props, "value": values, "graph_feats": graph_feats}
    dataframe = pd.DataFrame(data)
    dataframe_processed, scaler_dict = prepare_train(
        dataframe,
        smiles_featurizer=None,
        root_dir=root_dir,
    )
    return {
        "dataframe": dataframe,
        "dataframe_processed": dataframe_processed,
        "scaler_dict": scaler_dict,
        "root_dir": root_dir,
    }


def test_prepTrainSaveLoad_output(example_linear_data):
    """
    This test checks that the output of a training run are what we
    expect after:
        - preparing the data
        - training the ensemble
            - Check for no errors
            - Check that tc.get_train_dataloader is None
        - saving the ensemble
        - loading the ensemble
    """
    tc = trainConfig(
        loss_obj=None,
        amp=False,
    )
    tc.device = "cpu"  # run this on a CPU since this is just a test

    # "train" the ensemble with trainer_MathModel
    train_kfold_ensemble(
        example_linear_data["dataframe_processed"],
        model_constructor=lambda: MathModel1(None, 1),
        train_config=tc,
        submodel_trainer=trainer_MathModel,
        augmented_featurizer=None,  # since we do not want augmentation
        scaler_dict=example_linear_data["scaler_dict"],
        root_dir=example_linear_data["root_dir"],
        n_fold=2,
        random_seed=234,
    )
    # check that tc.get_train_dataloader is set to None
    assert tc.get_train_dataloader == None
    # ####################################
    # load the ensemble and run it forward
    # ####################################
    ensemble = load_ensemble(
        example_linear_data["root_dir"],
        submodel_cls=MathModel1,
        device="cpu",
        submodel_kwargs_dict={"dummy_attr": 1},
    )
    _, mean, _, _ = eval_ensemble(
        model=ensemble,
        root_dir=example_linear_data["root_dir"],
        dataframe=example_linear_data["dataframe"],
        smiles_featurizer=None,
        device=None,
        ensemble_kwargs_dict={"n_passes": 1},
    )
    # ###########################
    inv_transform = lambda x: (10 ** (4 * x)) - 1  # define the inverse transformation
    # that should have been computed on example_linear_data

    # loop through results
    # TODO: Parallelize?
    for ind, x in enumerate(
        example_linear_data["dataframe_processed"].data.values.tolist()
    ):
        trans_val = (
            x.graph_feats[0, 0]
            + x.graph_feats[0, 1]
            + x.selector[0, 0]
            - x.selector[0, 1]
        )  # output of the model
        ens_output = mean[ind]  # output of the ensemble
        np.testing.assert_allclose(
            ens_output, inv_transform(trans_val), rtol=1e-5, atol=0
        )  # check that the output
        # of the ensemble matches the transformed output of the submodel


@pytest.fixture
def example_unit_sequence():
    hps = HpConfig()
    dictionary = {
        "capacity": 3,
        "dropout_pct": 0.0,
        "activation": nn.functional.leaky_relu,
    }
    hps.set_values(dictionary)
    return {
        "input_dim": 32,
        "output_dim": 8,
        "hps": hps,
        "capacity": dictionary["capacity"],
        "unit_sequence": [32, 16, 16, 16, 8],
    }


def test_unit_sequence(example_unit_sequence):
    assert (
        pt_utils.get_unit_sequence(
            example_unit_sequence["input_dim"],
            example_unit_sequence["output_dim"],
            example_unit_sequence["capacity"],
        )
        == example_unit_sequence["unit_sequence"]
    )


def test_hp_str(example_unit_sequence, capsys):
    # check that hps are printed correctly
    assert (
        str(example_unit_sequence["hps"])
        == "{activation: leaky_relu, batch_size: None, capacity: 3, dropout_pct: 0.0, r_learn: None}"
    )
    # ############################################################
    # check that hps are NOT printed when a layer is instantiated
    # ############################################################
    layer = my_hidden(size_in=1, size_out=1, hps=example_unit_sequence["hps"])
    assert (
        pt_utils.module_name(layer) == f"{constants.PACKAGE_NAME}.layers"
    )  # test that module_name works
    captured = capsys.readouterr()
    check_str = "Hyperparameters after model instantiation"
    assert check_str not in captured.out
    # ############################################################
    # check that hps are printed when a model is instantiated
    # ############################################################
    models.MlpOut(input_dim=1, output_dim=1, hps=example_unit_sequence["hps"])
    captured = capsys.readouterr()
    assert check_str in captured.out


def test_unit_sequence_MlpOut(example_unit_sequence):
    model = models.MlpOut(
        example_unit_sequence["input_dim"],
        example_unit_sequence["output_dim"],
        example_unit_sequence["hps"],
        False,
    )
    assert (
        model.mlp.unit_sequence + [model.output_dim]
        == example_unit_sequence["unit_sequence"]
    )


@pytest.fixture(scope="session", autouse=True)  # this will tell
# Pytest to run the function below after all tests are completed
def cleanup(request):
    # clean up any files that were created
    def remove_data():
        try:
            remove("properties.pkl")
        except FileNotFoundError:
            pass
        try:
            rmtree("st_ensemble/")
        except FileNotFoundError:
            pass
        try:
            rmtree("mt_ensemble/")
        except FileNotFoundError:
            pass
        try:
            rmtree("ensemble_linear/")
        except FileNotFoundError:
            pass

    # execute remove_dirs at the end of the session
    request.addfinalizer(remove_data)
