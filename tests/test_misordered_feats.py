import pytest
import pandas as pd
from shutil import rmtree
from tests.utils_test import MathModel2, trainer_MathModel

from polygnn_trainer.prepare import prepare_infer, prepare_train
from polygnn_trainer.train import train_kfold_ensemble, trainConfig
from polygnn_trainer.infer import eval_ensemble
from polygnn_trainer.load import load_ensemble, load_selectors


def test_misordered_feats():
    root_dir = "ensemble_misordered/"
    props = ["prop1"] * 4
    values = [0, 0, 0, 0]
    # check that error is returned when input data with missing values
    # is given
    train_graph_feats = [
        {"feat0": 0, "feat1": 1},
        {"feat1": 2},
        {"feat0": 4, "feat1": 5},
        {"feat0": 6, "feat1": 7},
    ]
    data = {"prop": props, "value": values, "graph_feats": train_graph_feats}
    dataframe = pd.DataFrame(data)
    with pytest.raises(ValueError) as e:
        dataframe, scaler_dict = prepare_train(
            dataframe,
            smiles_featurizer=None,
            root_dir=root_dir,
        )
    assert "Dataframe contains missing values in graph_feats: (1, 'feat0')" == str(
        e.value
    )
    # check that error is returned when input data with non-dictionary values
    # is given
    train_graph_feats = [
        {"feat0": 0, "feat1": 1},
        [2, 2],
        {"feat0": 4, "feat1": 5},
        {"feat0": 6, "feat1": 7},
    ]
    data = {"prop": props, "value": values, "graph_feats": train_graph_feats}
    dataframe = pd.DataFrame(data)
    with pytest.raises(ValueError) as e:
        dataframe, scaler_dict = prepare_train(
            dataframe,
            smiles_featurizer=None,
            root_dir=root_dir,
        )
    assert (
        "Dataframe does not contain a dictionary in following rows of graph_feats: 1"
        == str(e.value)
    )
    # train and save the ensemble with good data
    train_graph_feats = [
        {"feat0": 0, "feat1": 1},
        {"feat1": 3, "feat0": 2},
        {"feat0": 4, "feat1": 5},
        {"feat0": 6, "feat1": 7},
    ]
    data = {"prop": props, "value": values, "graph_feats": train_graph_feats}
    dataframe = pd.DataFrame(data)
    dataframe, scaler_dict = prepare_train(
        dataframe,
        smiles_featurizer=None,
        root_dir=root_dir,
    )
    tc = trainConfig(
        loss_obj=None,
        amp=False,
    )
    tc.device = "cpu"  # run this on a CPU since this is just a test

    # "train" the ensemble with trainer_MathModel
    train_kfold_ensemble(
        dataframe,
        model_constructor=lambda: MathModel2(None),
        train_config=tc,
        submodel_trainer=trainer_MathModel,
        augmented_featurizer=None,  # since we do not want augmentation
        scaler_dict=scaler_dict,
        root_dir=root_dir,
        n_fold=2,
        random_seed=234,
    )

    # load the model
    selectors = load_selectors(root_dir)

    # check that inference with missing values does not work
    train_graph_feats = [
        {"feat0": 0},
    ]
    data = {"prop": ["prop1"], "graph_feats": train_graph_feats}
    dataframe = pd.DataFrame(data)
    with pytest.raises(ValueError) as e:
        prepare_infer(
            dataframe,
            smiles_featurizer=None,
            selectors=selectors,
            root_dir=root_dir,
        )
    assert "Dataframe contains missing values in graph_feats: (0, 'feat1')" == str(
        e.value
    )
    # check that inference with added values does not work
    train_graph_feats = [
        {"feat0": 0, "feat2": 1, "feat3": 2},
    ]
    data = {"prop": ["prop1"], "graph_feats": train_graph_feats}
    dataframe = pd.DataFrame(data)
    with pytest.raises(ValueError) as e:
        prepare_infer(
            dataframe,
            smiles_featurizer=None,
            selectors=selectors,
            root_dir=root_dir,
        )
    assert (
        "Dataframe contains extra values in graph_feats: (0, 'feat2'), (0, 'feat3'). Dataframe contains missing values in graph_feats: (0, 'feat1')"
        == str(e.value)
    )
    # infer on valid data and make sure that the output is what
    # we expect
    infer_graph_feats = [
        {"feat0": 12, "feat1": 13},
    ]
    data = {"prop": ["prop1"], "graph_feats": infer_graph_feats}
    dataframe = pd.DataFrame(data)
    # load the ensemble
    ensemble = load_ensemble(
        root_dir, MathModel2, device="cpu", submodel_kwargs_dict={}
    )
    # below, let's set "n_passes" equal to 2 so that we get a standard
    # deviation. If we only use one pass, the standard deviation will be
    # nan.
    _, mean, std, _ = eval_ensemble(
        ensemble, root_dir, dataframe, None, ensemble_kwargs_dict={"n_passes": 2}
    )

    assert mean == -1
    assert std == 0


@pytest.fixture(scope="session", autouse=True)  # this will tell
# Pytest to run the function below after all tests are completed
def cleanup(request):
    # clean up any files that were created
    def remove_data():
        try:
            rmtree("ensemble_misordered/")
        except FileNotFoundError:
            pass

    # execute remove_dirs at the end of the session
    request.addfinalizer(remove_data)
