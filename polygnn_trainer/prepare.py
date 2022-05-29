from copy import deepcopy
import numpy as np
from torch import Tensor, FloatTensor
from pandas import get_dummies, options
from polygnn_trainer.os import path_join

options.mode.chained_assignment = None  # set to None to avoid erroneous warnings
from torch_geometric.data import Data

from .scale import *
from polygnn_trainer import save
from polygnn_trainer import constants as ks
from polygnn_trainer import load


def obj_to_tensor(obj):
    """
    Convert an input object to a Tensor. This
    will work unless obj is a str. In this case,
    obj will be left as a numpy str
    """
    if not isinstance(obj, Tensor):
        obj = np.array(obj)
        ndim = len(obj.shape)
        if ndim == 1:
            obj = np.expand_dims(obj, axis=0)
        if obj.dtype.type == np.str_:
            pass
        else:
            obj = FloatTensor(obj)
            obj.requires_grad = False

    return obj


def copy_attribute_safe(via, to, attr_name_via, attr_name_to=None, fillna=None):
    """
    Copy the attr_name attribute from row or data point to x to data point data,
    if the attribute exits. Each attribute created here will not require
    gradients.
    """
    # if attr_name_to is not set. Let's set it equal to  attr_name_via
    if not attr_name_to:
        attr_name_to = attr_name_via
    if hasattr(via, attr_name_via):
        obj = getattr(via, attr_name_via)
        # handle the typing of tens
        tens = obj_to_tensor(obj)
        # OK, now we are ready to transfer the attribute
        setattr(to, attr_name_to, tens)
    # if the "via" object does not have the attribute, then we can fill it
    # with whatever is specified in "fillna"
    elif fillna != None:
        # make sure what the user put in fillna is a Tensor
        assert isinstance(fillna, Tensor)
        # OK, now we are ready to transfer the attribute
        setattr(to, attr_name_to, fillna)


def prepare_train(
    dataframe,
    smiles_featurizer,
    root_dir,
):
    """
    Prepare necessary data for training.

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of two columns:
            prop, value. At least one of the following
            optional columns should also be present: smiles_string,
            node_feats, and graph_feats. This dataframe should not contain
            any na.
        smiles_featurizer: A function that takes in a smiles string
            and returns its features. For polymers, the smiles strings
            are of type '[*]CC[*]' or the equivalent for ladders.
    Outputs:
        dataframe (pd.DataFrame): The input dataframe with two columns
            added: data and selector
        scaler_dict
    """
    return prepare_data(
        dataframe,
        smiles_featurizer,
        True,
        None,
        root_dir,
    )


def prepare_infer(
    dataframe,
    smiles_featurizer,
    selectors,
    root_dir,
):
    """
    Prepare necessary data for inference.

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of one column:
            prop. At least one of the following optional columns should
            also be present: smiles_string, node_feats, and graph_feats.
            The column 'value' may also be included.
            This dataframe should not contain any na.
        smiles_featurizer: A function that takes in a smiles string
            and returns its features. For polymers, the smiles strings
            are of type '[*]CC[*]' or the equivalent for ladders.
    Outputs:
        dataframe (pd.DataFrame): The input dataframe with two columns
            added: data and selector
    """
    return prepare_data(
        dataframe,
        smiles_featurizer,
        False,
        selectors,
        root_dir,
    )


def append_selectors(dataframe, selectors, n_props, for_train):
    # if training, we need to make the selectors
    if for_train:
        # if we have more than one property, we need to make selectors
        # that are not empty
        if n_props > 1:
            selectors = get_dummies(dataframe.prop).values.tolist()
            selectors = [[x] for x in selectors]  # add a dimension to each selector
            dataframe[ks._F_SELECTORS] = selectors
        # if we have just one property, the selector can be empty
        elif n_props == 1:
            dataframe[ks._F_SELECTORS] = [empty_input_tensor()] * len(dataframe)
    # if inferring, we need to copy from the selectors passed in
    else:

        def copy_selectors(x):
            x[ks._F_SELECTORS] = selectors[x.prop]
            return x

        dataframe = dataframe.apply(copy_selectors, axis=1)

    return dataframe


def empty_input_tensor():
    tens = Tensor()
    tens.requires_grad = False
    return tens


def sort_pandas_column(dataframe, colname, srt_keys):
    """
    Takes in a pandas dataframe with dictionary values,
    sorts the values in each row, then overwrites each
    dict with a numpy array

    Keyword arguments:
        dataframe (pd.DataFrame)
        colname (str)
        srt_keys (list): The keys in each dictionary, sorted

    Outputs:
        dataframe (pd.DataFrame)
    """

    def helper(row):
        feat_dict = row[colname]
        arr = np.array([feat_dict[x] for x in srt_keys], dtype=float)
        row[colname] = arr
        return row

    dataframe = dataframe.apply(helper, axis=1)
    return dataframe


def check_series_types(dataframe, colname):
    """
    Check that the Series specified by (dataframe, colname) only contains
    Python dicts
    """
    series = dataframe[colname]
    dict_ls = [x for x in series]
    # check types
    diff_types = [ind for ind, d in enumerate(dict_ls) if not isinstance(d, dict)]
    if diff_types:
        diff_string = ", ".join(map(str, diff_types))
        raise ValueError(
            f"Dataframe does not contain a dictionary in following rows of {colname}: {diff_string}"
        )


def get_series_keys(dataframe, colname):
    """
    Return the union of all dictionary keys in Pandas series
    specified by (dataframe, colname)
    """
    series = dataframe[colname]
    dict_ls = [x for x in series]
    all_keys = set().union(*(d.keys() for d in dict_ls))
    return sorted(all_keys)


def check_series_keys(dataframe, colname, all_keys, for_train):
    """
    Check that the Series specified by (dataframe, colname) contains
    no rows with missing or extra features
    """
    all_keys = set(all_keys)
    series = dataframe[colname]
    dict_ls = [x for x in series]
    all_AdiffB = []
    all_BdiffA = []
    for ind, d in enumerate(dict_ls):
        keys = set(d.keys())
        AdiffB = sorted(list(keys.difference(all_keys)))  # A diff B
        BdiffA = sorted(list(all_keys.difference(keys)))  # B diff A
        all_AdiffB.extend([(ind, key) for key in AdiffB])
        all_BdiffA.extend([(ind, key) for key in BdiffA])

    string_helper = lambda diff: ", ".join(map(str, diff))
    if for_train:
        all_diffs = all_AdiffB + all_BdiffA
        if all_diffs:
            diff_string = string_helper(all_diffs)
            raise ValueError(
                f"Dataframe contains missing values in {colname}: {diff_string}"
            )
    else:
        exception_ls = []
        if all_AdiffB:
            diff_string = string_helper(all_AdiffB)
            exception_ls.append(
                f"Dataframe contains extra values in {colname}: {diff_string}"
            )
        if all_BdiffA:
            diff_string = string_helper(all_BdiffA)
            exception_ls.append(
                f"Dataframe contains missing values in {colname}: {diff_string}"
            )
        if exception_ls:
            raise ValueError(". ".join(exception_ls))


def prepare_data(
    dataframe,
    smiles_featurizer,
    for_train,
    selectors,
    root_dir,
):
    """
    Prepare necessary data. This function is not to be called directly,
    but is meant as a helper for prepare_train and prepare_infer.

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of one column:
            prop. At least one of the following optional columns should
            also be present: smiles_string, node_feats, and graph_feats.
            The column 'value' may also be included.
            This dataframe should not contain any na.
        smiles_featurizer: A function that takes in a smiles string
            and returns its features. For polymers, the smiles strings
            are of type '[*]CC[*]' or the equivalent for ladders.
        for_train (bool): True if dataframe contains training data
        selectors (dict)
    """
    prop_cols = prepare_init(dataframe, for_train)
    # convert dictionary-like columns to arrays and save the associated
    # metadata
    if for_train:
        has_graph_feats = hasattr(dataframe, ks._F_GRAPH)
        has_node_feats = hasattr(dataframe, ks._F_NODE)
        # check that graph_feats are formatted correctly
        if has_graph_feats:
            check_series_types(dataframe, ks._F_GRAPH)
            graph_srt_keys = get_series_keys(dataframe, ks._F_GRAPH)
            check_series_keys(dataframe, ks._F_GRAPH, graph_srt_keys, for_train)
        else:
            graph_srt_keys = ["<empty>"]
        # check that node_feats are formatted correctly
        if has_node_feats:
            check_series_types(dataframe, ks._F_NODE)
            node_srt_keys = get_series_keys(dataframe, ks._F_NODE)
            check_series_keys(dataframe, ks._F_NODE, node_srt_keys, for_train)
        else:
            node_srt_keys = ["<empty>"]

        # prepare root, subdirectories, and save *_srt_keys in a readable form
        _, md_dir = save.prepare_root(root_dir)
        pkl_path = path_join(md_dir, ks.FEATURE_FILENAME_PKL)
        srt_keys_dict = {
            "graph_srt_keys": graph_srt_keys,
            "node_srt_keys": node_srt_keys,
        }
        save.safe_save(srt_keys_dict, pkl_path, "pickle")
        graph_str = (
            f"The graph features used during training are: {', '.join(graph_srt_keys)}"
        )
        node_str = (
            f"The node features used during training are: {', '.join(node_srt_keys)}"
        )
        feature_string = "\n\n".join([graph_str, node_str])
        txt_path = path_join(md_dir, ks.FEATURE_FILENAME_TXT)
        save.safe_save(feature_string, txt_path, "text")
    else:
        # retrieve the names of features used during training
        srt_keys_dict = load.load_features(root_dir)
        graph_srt_keys = srt_keys_dict["graph_srt_keys"]
        node_srt_keys = srt_keys_dict["node_srt_keys"]
        helper = lambda x: x != ["<empty>"]
        has_graph_feats = helper(graph_srt_keys)
        has_node_feats = helper(node_srt_keys)
        # check that graph_feats are formatted correctly
        if has_graph_feats:
            check_series_types(dataframe, ks._F_GRAPH)
            check_series_keys(dataframe, ks._F_GRAPH, graph_srt_keys, for_train)
        # check that node_feats are formatted correctly
        if has_node_feats:
            check_series_types(dataframe, ks._F_NODE)
            check_series_keys(dataframe, ks._F_NODE, node_srt_keys, for_train)

    # sort graph_feats
    if has_graph_feats:
        dataframe = sort_pandas_column(dataframe, ks._F_GRAPH, graph_srt_keys)
    # sort node feats
    if has_node_feats:
        dataframe = sort_pandas_column(dataframe, ks._F_NODE, node_srt_keys)

    if not for_train:
        # if running evaluation, use prop_cols made during training
        prop_cols = sorted(selectors.keys())

    # append selectors to dataframe
    n_props = len(prop_cols)
    dataframe = append_selectors(dataframe, selectors, n_props, for_train)

    if for_train:
        # initialize scaler for each property
        scaler_dict = {prop: SequentialScaler() for prop in prop_cols}
        # add log-delta scaling for each property, if necessary
        for prop in prop_cols:
            vals = dataframe[dataframe.prop == prop]["value"]
            rng = vals.max() - vals.min()
            if rng > ks.LOG_THRESHOLD:
                print(f"Using log-delta scaling for property {prop}")
                scaler_dict[prop].append(
                    LogTenDeltaScaler()
                )  # log the property according to https://doi.org/10.1016/j.patter.2021.100238
            else:
                print(f"Using no scaling for property {prop}")
        # add MinMax scaler to each property, if necessary
        if n_props > 1:
            for prop in prop_cols:
                mm_scaler = MinMaxScaler()
                prop_vals = dataframe[dataframe.prop == prop]["value"].values
                print(f"Shape of training labels for {prop} is {prop_vals.shape}")
                # transform the property values according to all the scalers
                # added so far before we fit the next scaler
                trans_prop_vals = scaler_dict[prop].transform(prop_vals)
                mm_scaler.fit(trans_prop_vals)  # fit scaler on training data
                scaler_dict[prop].append(mm_scaler)

    def get_data(x):
        # prepare smiles-based features
        if smiles_featurizer:
            data = smiles_featurizer(x.smiles_string)
        else:
            data = Data(x=empty_input_tensor())
        if for_train:
            # if we are training then data.y must be set
            y = scaler_dict[x.prop].transform(x.value)  # scale the label
            data.y = obj_to_tensor(y)
            # during training if "selector", "graph_feats", or "node_feats"
            # are missing, we should fill it with an empty tensor
            fillna = empty_input_tensor()
        else:
            # for inference, we may or may not need data.y
            copy_attribute_safe(x, data, "value", ks._Y)
            # during inference, we do not support filling
            # missing attributes of rows. So we set fillna to None
            fillna = None

        # prepare selector
        copy_attribute_safe(x, data, ks._F_SELECTORS, fillna=deepcopy(fillna))
        # prepare node-features
        copy_attribute_safe(x, data, ks._F_NODE, fillna=deepcopy(fillna))
        # prepare graph-features
        copy_attribute_safe(x, data, ks._F_GRAPH, fillna=deepcopy(fillna))
        if not for_train:
            # if we are not training, then we need to copy "prop" over
            # to make sure we scale correctly inside the ensemble
            copy_attribute_safe(x, data, "prop")

        return data

    # prepare Data object for each row in dataframe
    dataframe["data"] = dataframe.apply(get_data, axis=1)
    if for_train:
        return dataframe, scaler_dict
    else:
        return dataframe


def prepare_init(
    dataframe,
    for_train,
):
    """
    Initial steps for data preparation

    Keyword arguments:
        dataframe (pd.DataFrame): A dataframe consisting of two columns:
            prop, value. Optional columns are: smiles_string, node_feats,
            and graph_feats. This dataframe should not contain any na.
        smiles_featurizer: A function that takes in a smiles string
            and returns its features. For polymers, the smiles strings
            are of type '[*]CC[*]' or the equivalent for ladders.
    """
    # error handling
    if dataframe.isnull().sum().sum() > 0:
        raise ValueError("Datafame contains null values")
    if not hasattr(dataframe, "prop"):
        raise KeyError(f"Key 'prop' not present in dataframe. Cannot process data.")
    if for_train and not hasattr(dataframe, "value"):
        raise KeyError(
            f"Key 'value' not present in dataframe. Cannot process training data."
        )
    if not (
        hasattr(dataframe, ks._F_GRAPH)
        or hasattr(dataframe, ks._F_NODE)
        or hasattr(dataframe, "smiles_string")
    ):
        raise KeyError(
            f"None of the following keys are present in dataframe: 'graph_feats', 'node_feats', 'smiles_string'. Cannot process data."
        )
    # ##############
    prop_cols = sorted(
        dataframe["prop"].unique().tolist()
    )  # sorted list of property names
    print(f"The following properties will be modeled: {prop_cols}", flush=True)
    # count the number of data points in each class
    # TODO: Speed this up.
    for prop in prop_cols:
        n_prop_data = len(dataframe[dataframe["prop"] == prop])
        print(f"Detected {n_prop_data} data points for {prop}", flush=True)

    return prop_cols
