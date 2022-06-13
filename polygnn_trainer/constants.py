import re

PACKAGE_NAME = "polygnn_trainer"
RANDOM_SEED = 123
DL_STOP_TRAIN_R2 = -(10**8)
GRADIENT_HISTORY_LENGTH = 5  # the number of latest epochs for which
# the gradients will be stored in memory
LOG_THRESHOLD = (
    10**3
)  # if properties have a range above this value, we will use log-delta scaling
BS_MAX = 450
VAL_FRAC = 0.2  # fraction of data to use for validation
N_MODELS = 5

MODELS_DIR = "models/"
METADATA_DIR = "metadata/"
SCALERS_FILENAME = "scalers.pkl"
HPS_FILENAME = "hyperparams.pkl"
SELECTORS_FILENAME = "selectors.pkl"
PROP_FILENAME = "properties.txt"
FEATURE_PREFIX = "features"
FEATURE_FILENAME_PKL = f"{FEATURE_PREFIX}.pkl"
FEATURE_FILENAME_TXT = f"{FEATURE_PREFIX}.txt"
PICKLE_PROTOCOL = 2

# regex
integer = "[-+]?[0-9]+"
submodel_re = re.compile(f"model_{integer}.pt")

# feature and target names
_F_SMILES = "x"
_F_GRAPH = "graph_feats"
_F_NODE = "node_feats"
_F_SELECTORS = "selector"
_F_SET = [_F_SMILES, _F_GRAPH, _F_NODE, _F_SELECTORS]
_Y = "y"
