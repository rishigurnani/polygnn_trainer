# polygnn_trainer
This repository contains one of the custom packages used to train the machine learning models presented in a companion paper, [*polyGNN: Multitask graph neural networks for polymer informatics*](https://arxiv.org/abs/2209.13557). Currently, `polygnn_trainer` (pt) contains general code for the following tasks
- Data preparation
- Submodel training (and limited analysis of training metrics, provided by the [`parse`](https://github.com/rishigurnani/polygnn_trainer/tree/main/polygnn_trainer/parse) module)
- Submodel saving & loading
- Composing the submodels into an ensemble for inference
## Installation
This repository is currently set up to run on 1) Mac OSX and 2) Linux/Windows machines with CUDA 10.2. Please raise a GitHub issue if you want to use this repo with a different configuration. Otherwise, please follow these steps for installation:

1. Install [poetry](https://python-poetry.org/) on your machine.
2. If Python3.7 is installed on your machine skip to step 3, if not you will need to install it. There are many ways to do this, one option is detailed below:
    * Install [Homebrew](https://brew.sh/) on your machine.
    * Run `brew install python@3.7`. Take note of the path to the python executable.
3. Clone this repo on your machine.
4. Open a terminal at the root directory of this repository.
5. Run `poetry env use /path/to/python3.7/executable`. If you installed Python3.7 with Homebrew, the path may be something like
  `/usr/local/Cellar/python\@3.7/3.7.13_1/bin/python3.7`.
7. Run `poetry install`.
8. If your machine is a Mac, run `poetry run poe torch-osx`. If not, run `poetry run poe torch-linux_win-cuda102`.
9. If your machine is a Mac, run `poetry run poe pyg-osx`. If not, run `poetry run poe pyg-linux_win-cuda102`.
## Usage
### General usage
A usage pattern may look something like below:
- First, import `polygnn_trainer`
  ```python
  import polygnn_trainer as pt
  ```
- Then, create a pandas Dataframe out of the data set (see [here](https://github.com/rishigurnani/polygnn_trainer/blob/main/sample_data/sample.csv) for an example CSV data set, note that you can distinguish between tasks using the "prop" column). Split this dataset into training+val and test set. It is required that graph features are provided in several dictionaries; one dictionary per row of the dataframe.
- Pass the training+val DataFrame into `pt.train.prepare_train`. The outputs will be the dataframe, with SMILES converted to features, and dictionary of scalers. One scaler per task. Additionally, some metadata on training features will be saved to path input to `root_dir`. This will be useful to order features during inference.
  ```python    
  dataframe, scaler_dict = prepare_train(
      "dataframe", morgan_featurizer, root_dir=/path/to/model/root/
  )
  ```
- Split the training+val DataFrame into two DataFrames using your favorite splitting strategy (example shown below). One DataFrame for validation and one for training. See the example snippet below.
  ```python    
   training_df, val_df = sklearn.model_selection.train_test_split(
      dataframe,
      test_size=constants.VAL_FRAC,
      stratify=dataframe.prop,
      random_state=constants.RANDOM_SEED,        
   )
  ```
- Run hyperparameter optimization using your favorite algorithm (example shown below). Use the error on the validation set to find the optimal set. See the example snippet below. The unittest named `test_ensemble_trainer` contains a more detailed, working, example. Unittests are located in [this folder](https://github.com/rishigurnani/polygnn_trainer/tree/main/tests).
  ```python    
  from skopt import gp_minimize
  # obtain the optimal point in hp space
  opt_obj = gp_minimize(
      func=obj_func, # define this yourself
      dimensions=hp_space,
      n_calls=10,
      random_state=0,
  )
  # create an HpConfig from the optimal point in hp space
  hps = pt.hyperparameters.HpConfig()
  hps.set_values(
    {
      "r_learn": 10 ** opt_obj.x[0],
      "batch_size": opt_obj.x[1],
      "dropout_pct": opt_obj.x[2],
      "capacity": opt_obj.x[3],
      "activation": torch.nn.functional.leaky_relu,
    }
  )
  ```
- Train an ensemble of submodels using `pt.train.train_kfold_ensemble`. Again, `test_ensemble_trainer` contains a nice working example. If the ensemble is being trained for research, we can train the ensemble using just the train+val set (so that we have the test set to assess model accuracy). If the ensemble is being trained for production, we can train the ensemble on the ***entire*** data set. The details of the model and its metadata will be stored in a root directory. The contents of that directory are as follows:
   ```
  root
  │
  └───metadata
  │   │   features.pkl
  │   │   features.txt
  │   │   hyperparams.pkl
  │   │   properties.txt
  │   │   scalers.pkl
  │   │   selectors.pkl
  │   
  └───models
      │   model_0.pt
      │   model_1.pt
      │   ...
  ```
- Prepare a DataFrame of points to run inference on. Again, it is required that graph features are provided in several dictionaries; one dictionary per row of the melted dataframe.
- Load the ensemble using `pt.load.load_ensemble`. Fields needed to instantiate the submodels can be passed into `submodel_kwargs_dict`.
  ```python
  ensemble = pt.load.load_ensemble(
    "path/to/model/root/",
    MyModelName,
    device,
    submodel_kwargs_dict={},
  )
  ```
- Run inference on the ensemble using `pt.infer.eval_ensemble`. See below for example usage.
  ```python
  y_val, y_val_mean_hat, y_val_std_hat, selectors = pt.infer.eval_ensemble(
    ensemble,
    "path/to/model/root/",
    dataframe,
    smiles_featurizer,
    device,
    ensemble_kwargs_dict={"monte_carlo": False},
  )
  ```
### `example.py`
Much of the information in the "General usage" section is combined into one file, `example.py`. Here we use training data located in the directory `sample_data` to train an ensemble model (composed of several submodels). The submodels, by default, are saved in a directory named `example_models`. The data in `sample_data` is a small subset of the DFT data used to train the models in the companion paper. A complete set of the DFT data can be found at [Khazana](https://khazana.gatech.edu/).

To train models run: `poetry run python example.py`. This should not take longer than 3 minutes on a machine with at least 8GB of free GPU memory. To manually specify the device you want to use for training, set the device flag. For example `poetry run python example.py --device cpu`. Otherwise, the device will automatically be chosen.

Looking at `sample_data/sample.csv`, you will notice that this dataset contains multiple different properties (e.g., band gap, electron affinity, etc.). In `example.py`, we use this data to train a multitask model, capable of predicting each property. To train your own multitask model, you can replace `sample_data/sample.csv` with your own dataset containing multiple properties. Single task models are also supported.
## Citation
If you use this repository in your work please consider citing us.
```
@article{Gurnani2023,
   annote = {doi: 10.1021/acs.chemmater.2c02991},
   author = {Gurnani, Rishi and Kuenneth, Christopher and Toland, Aubrey and Ramprasad, Rampi},
   doi = {10.1021/acs.chemmater.2c02991},
   issn = {0897-4756},
   journal = {Chemistry of Materials},
   month = {feb},
   number = {4},
   pages = {1560--1567},
   publisher = {American Chemical Society},
   title = {{Polymer Informatics at Scale with Multitask Graph Neural Networks}},
   url = {https://doi.org/10.1021/acs.chemmater.2c02991},
   volume = {35},
   year = {2023}
}
```
## License
This repository is protected under a General Public Use License Agreement, the details of which can be found in `GT Open Source General Use License.pdf`.
