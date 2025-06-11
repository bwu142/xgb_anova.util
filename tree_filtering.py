# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import plotly.express as px
import os
import plotly.graph_objects as go


###### TREE HELPER FUNCTIONS ######
def get_bias(model):
    bias = model.get_booster().attr("base_score")
    if bias is not None:
        return float(bias)
    return float(model.base_score)  # Fallback


def get_tree_leaf_indices(model_file, tree_index):
    """
    model_file: editable xgboost regressor file
        with open("model_file.json", "r") as f:
            model_file = json.load(f)
    tree_index: index of the tree we want to get the leaf indices of
        0-indexing

    returns a list of the leaf indices in the file
    """
    # lists of children
    tree = model_file["learner"]["gradient_booster"]["model"]["trees"][tree_index]
    left_children = tree["left_children"]
    right_children = tree["right_children"]
    # get leaf indices (index = -1 for left and right children)
    leaf_indices = [
        i
        for i, (left_child_val, right_child_val) in enumerate(
            zip(left_children, right_children)
        )
        if left_child_val == -1 and right_child_val == -1
    ]
    return leaf_indices


def get_all_tree_list(model):
    """
    model: xgb regressor (model = xgb.XGBRegressor())

    Returns list of trees (represented as dictionaries) resulting from json format
    """
    trees_json = model.get_booster().get_dump(dump_format="json")
    all_trees = [json.loads(tree_str) for tree_str in trees_json]
    return all_trees


def get_filtered_trees(model, features_needed=None):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    features_needed: set of features (represented as strings, e.g. 'x1')

    Returns tuple:
        tuple[0]: list of ranges of relevant trees with all features | e.g. [(0, 1), (2, 3)] means tree 1, tree 3 are the only decision trees that contain all features in features_needed

        tuple[1]: list of all trees (represented as dictionaries) from model that contain the intersection of all features_needed as splits
    """

    def get_features_used(node, features=None):
        """
        node: tree (originally root node); dictionary
        features: set of features (represented as strings, e.g. 'x1') used as splits in "node" decision tree

        """
        if features is None:
            features = set()
        if "split" in node:
            features.add(node["split"])
            # Don't recurse if leaf
            if "children" in node:
                for child in node["children"]:
                    get_features_used(child, features)
        return features

    # Get all trees in json dictionary format
    trees_json = model.get_booster().get_dump(dump_format="json")

    filtered_trees_ranges_list = []  # tuple[0]
    filtered_trees_json_list = []  # tuple[1]
    filtered_trees = (filtered_trees_ranges_list, filtered_trees_json_list)

    # Loop through each tree, and append to filtered_trees if tree contains features
    for i, tree_str in enumerate(trees_json):
        tree = json.loads(tree_str)
        features_used = get_features_used(tree)
        # print(f"Tree {i} uses features: {features_used}")

        # if features_needed.issubset(features_used):  # old
        if features_needed == features_used:  # set equality
            filtered_trees[0].append((i, i + 1))
            filtered_trees[1].append(tree)

    # print(
    #     f"number of original trees: {len(trees_json)}\nnumber of filtered trees: {len(filtered_trees[1])}"
    # )

    return filtered_trees


def get_filtered_tree_list_ranges_from_tuple(model, features_tuple=None):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    features_tuple: tuple of features --> e.g. (0, 1) would be only trees with x1x2
        Assuming "containing x1x2" means there's a split on x1 followed by split on x2 or vice versa

    Returns list of ranges of relevant trees with all features in feature_tuple| e.g. [(0, 1), (2, 3)] means tree 1, tree 3 are the only decision trees that contain features in features_tuple
    """
    features_needed = set()

    for feature_num in features_tuple:
        feature_num += 1  # 1-indexing
        features_needed.add("x" + str(feature_num))

    # print(f"features_needed: {features_needed}")

    return get_filtered_trees(model, features_needed)[0]


def get_split_tree_predictions(model, test_set, ranges):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    test_set: Pandas Dataframe of x-values (testing input set)
    ranges: list of ranges (representing the trees we care about)
        Output of get_filtered_tree_list_ranges_from_tuple

    Returns tuple:
        tuple[0]: 2D numpy array of predictions, each prediction from a tree in 'ranges'
            col: prediction vector from one tree
            row: test point
        tuple[1]: length of ranges (i.e., # trees)
    """

    num_trees = len(ranges)
    # arr is of dimension (num test points x num trees)
    arr = np.zeros((test_set.shape[0], num_trees))

    # put prediction values vectors into columns of arr
    for i, rng in enumerate(ranges):
        new_col = np.array(
            model.predict(test_set, iteration_range=rng, output_margin=False)
        )
        arr[:, i] = new_col

    output = (arr, num_trees)

    return output


def get_all_split_tree_predictions(model, test_set):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    test_set: Pandas Dataframe of x-values (testing input set)

    Returns tuple:
        tuple[0]: 2D numpy array of predictions, with each prediction coming from EACH tree in MODEL
        tuple[1]: length of ranges (i.e., # trees)
    """

    num_trees = model.n_estimators
    ranges = [(i, i + 1) for i in range(num_trees)]
    return get_split_tree_predictions(model, test_set, ranges)


def sum_split_tree_predictions(model, predictions):
    """
    predictions: tuple (2D numpy array of predictions, num_trees) that get_split_tree_predictions returns

    Returns the 1D numpy vector sum of predictions from each tree, with regards to bias
    """
    vector_sum = np.sum(predictions[0], axis=1)  # sum columns

    bias = get_bias(model)
    num_trees = predictions[1]
    vector_sum -= bias * (num_trees - 1)

    return vector_sum


def predict(model, features_tuple, test_set):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    features_tuple: tuple of features --> e.g. (0, 1) would be only trees with x1x2
        Assuming "containing x1x2" means there's a split on x1 followed by split on x2 or vice versa


    Returns: prediction (1D numpy array) using relevant trees in 'ranges' from model
    """
    # list of ranges
    range_list = get_filtered_tree_list_ranges_from_tuple(model, features_tuple)
    # print(f"range_list: {range_list}")
    # 2D numpy array of predictions
    relevant_predictions = get_split_tree_predictions(model, test_set, range_list)
    # predicted y vector
    y_pred = sum_split_tree_predictions(model, relevant_predictions)
    return y_pred


def predict_sum_of_all_trees(model, test_set):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    test_set: Pandas dataframe

    Returns: prediction (1D numpy vector) by summing all individual trees in model and accounting for bias
        kind of pointless, because we don't generate combinations of combined terms like f(x1x2)
    """
    all_features = tuple(feature_num for feature_num in range(test_set.shape[1]))
    return predict(model, all_features, test_set)


##### LOAD/SAVE REGRESSOR FUNCTIONS #####
def save_filtered_trees_indices(model, tree_indices, output_name):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    tree_indices: list of indices that we to keep
    name: "name of output file"

    Saves a json file (that is the original regressor containing the trees specified by 'tree_indices' & corresponding parameters)
        Does this by editing the originally saved xgboost json file
    Also saves original model in the process as original_model.json
    """
    # Sanity check
    if not output_name.endswith(".json"):
        output_name += ".json"
    # save model as json file
    model.save_model("original_model.json")

    # Load in json file into Python Dictionary for editing purposes (manipulate json file directly)
    with open("original_model.json", "r") as file:
        original_model = json.load(file)

    # Edit num_trees
    original_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(len(tree_indices))
    # Edit iteration_indptr
    original_model["learner"]["gradient_booster"]["model"]["iteration_indptr"] = [
        i for i in range(len(tree_indices) + 1)
    ]
    # Edit tree_info
    original_model["learner"]["gradient_booster"]["model"]["tree_info"] = [
        0 for _ in range(len(tree_indices))
    ]

    # trees = data["learner"]["gradient_booster"]["model"]["trees"]
    new_trees = []
    id_count = 0
    for i in tree_indices:
        new_trees.append(
            original_model["learner"]["gradient_booster"]["model"]["trees"][i]
        )
        new_trees[id_count]["id"] = id_count
        id_count += 1
    original_model["learner"]["gradient_booster"]["model"]["trees"] = new_trees

    # Write modified dictinoary new model to json file as output_name.json
    with open(str(output_name), "w") as file:
        json.dump(original_model, file)


def save_filtered_trees(model, ranges, output_name):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    ranges: list of ranges (representing the trees we care about)
        Output of get_filtered_tree_list_ranges_from_tuple
    name: "name of output file"

    Saves a json file (that is the original regressor minus the irrelevant trees & corresponding parameters)
        Does this by editing the originally saved xgboost json file
    Also saves original model in the process as original_model.json
    """
    tree_indices = []
    for rng in ranges:
        tree_indices.append(rng[0])
    save_filtered_trees_indices(model, tree_indices, output_name)


def filter_and_save(model, output_name, features_tuple=None):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    output_name (string): name we want to save the file as (ends with .json)
    features_tuple: tuple of features --> e.g. (0, 1) would be only trees with x1x2
        Assuming "containing x1x2" means there's a split on x1 followed by split on x2 or vice versa

    Saves a json file (that is the original regressor minus the irrelevant trees & corresponding parameters)
        Does this by editing the originally saved xgboost json file
    """
    ranges = get_filtered_tree_list_ranges_from_tuple(model, features_tuple)
    save_filtered_trees(model, ranges, output_name)


def filter_save_load(
    model, output_file_names, output_model_names, features_tuple_list=None
):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    output_file_names (list of strings): names we want to save the file as (ends with .json)
        MUST BE IN CORRESPONDING ORDER WITH FEATURES_TUPLE
    features_tuple_list: list of tuples of features --> e.g. [(0,), (0, 1)] would be trees with x1 (for first file) and trees with x1x2 (for second file)
        Assuming "containing x1x2" means there's a split on x1 followed by split on x2 or vice versa


    saves original model in json format as "original_model.json"
    saves filtered models json files in features_tuple
    loads filtered models into corresponding vars with corresponding output_model_names
    """
    output_models = []
    for i in range(len(output_model_names)):
        output_file_name = output_file_names[i]
        output_model_name = output_model_names[i]
        features_tuple = features_tuple_list[i]

        filter_and_save(model, output_file_name, features_tuple)
        output_model_name = xgb.XGBRegressor()
        output_model_name.load_model(output_file_name)
        output_models.append(output_model_name)

    return output_models


def save_new_trees_indices(model_file_name, tree_indices, leaf_vals, output_name):
    """
    model_file_name: name of xgb regressor json file(model = xgb.XGBRegressor())
        e.g. "model_tree_1.json"
    tree_indices: list of indices that we to keep
    leaf_vals: values we want to assign the leaves of the tree to
    name: "name of output file"

    Saves a new xgboost regressor model containing all the original trees, plus a new tree with leaf vals indicated by leaf_vals
    """
    # Open file and assign file object to f
    # Read contents of file as json object and store in model_file variable
    # Converts json structure into Python data structure
    with open(model_file_name) as f:
        model_file = json.load(f)


if __name__ == "__main__":
    ########################
    ##### INITIALIZING #####
    ########################

    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 1000)
    # x1 = np.arange(0, 100, 0.1)
    y = 10 * x1 + 2

    X = pd.DataFrame({"x1": x1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=100,  # 100 trees
        max_depth=1,  # depth of 2
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.8,
    )
    model.fit(X_train, y_train)

    #####################
    ##### CENTERING #####
    #####################

    # Get mean of output
    y_pred = model.predict(X_test)
    mean_pred = np.mean(y_pred)
    # Get one tree ("unsure_1.json" file contains model with first tree)
    save_filtered_trees_indices(model, [0], "model_tree_1.json")
    # load model with first tree for editing
    with open("model_tree_1.json", "r") as f:
        model_tree_1 = json.load(f)
    # lists of children
    tree = model_tree_1["learner"]["gradient_booster"]["model"]["trees"][0]
    left_children = tree["left_children"]
    right_children = tree["right_children"]
    # get leaf indices (index = -1 for left and right children)
    leaf_indices = [
        i
        for i, (left_child_val, right_child_val) in enumerate(
            zip(left_children, right_children)
        )
        if left_child_val == -1 and right_child_val == -1
    ]
    # subtract mean from leaf values of that one tree
    for i in leaf_indices:
        tree["base_weights"][i] = float(tree["base_weights"][i] - mean_pred)
    # save modified model to json file
    with open("model_tree_1_centered.json", "w") as f:
        json.dump(model_tree_1, f)

    ####################
    ##### PLOTTING #####
    ####################
    x_step = np.arange(0, 100, 0.1)
    X_test = pd.DataFrame({"x1": x_step})

    model_tree_1_centered = xgb.XGBRegressor()
    model_tree_1_centered.load_model("model_tree_1_centered.json")
    z_pred_model_tree_1_centered = model_tree_1_centered.predict(X_test)
    z_pred = model.predict(X_test)

    plot = go.Figure()

    plot.add_trace(
        go.Scatter(
            x=X_test["x1"],
            y=z_pred_model_tree_1_centered,
            mode="lines",
            name="z_pred_model_tree_1_centered: y = 10 * x1 + 2",
            line=dict(color="red"),
        )
    )

    plot.add_trace(
        go.Scatter(
            x=X_test["x1"],
            y=z_pred,
            mode="lines",
            name="z_pred: y = 10 * x1 + 2",
            line=dict(color="blue"),
        )
    )
    plot.show()
