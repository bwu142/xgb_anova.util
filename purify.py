# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import plotly.express as px
import os
import plotly.graph_objects as go
import io


##### TREE HELPER FUNCTIONS #####
def get_model_file(
    model, input_file_name="original_model.json", folder="loaded_models"
):
    """
    model: xgb.train(...)

    Returns file (dictionary) version of model that can be used for editing
        Saves original model to "original_model.json" in the process
    """
    # Sanity check
    if not input_file_name.endswith(".json"):
        input_file_name += ".json"
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    # Save Model
    file_path = os.path.join(folder, input_file_name)
    model.save_model(file_path)
    # Open model file (dictionary version) for editing
    with open(file_path, "r") as f:
        model_file = json.load(f)
    return model_file


def get_model(model_file, output_file_name="new_model.json", folder="loaded_models"):
    """
    model_file: file (dictionary) version of model

    Returns model (Booster object) that can be used for predictions
        Saves model_file to "model.json" in the process
    """
    # Sanity check
    if not output_file_name.endswith(".json"):
        output_file_name += ".json"
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Save file as json file
    output_path = os.path.join(folder, output_file_name)
    with open(output_path, "w") as file:
        json.dump(model_file, file)

    # Load model
    new_model = xgb.Booster()
    new_model.load_model(output_path)
    return new_model


def get_bias(model, input_file_name="model_bias", folder="loaded_models"):
    """
    model: xgb.train(...)
    Returns bias (even if unspecified in model params)
    """
    bias_model_file = get_model_file(model, input_file_name, folder)
    return float(bias_model_file["learner"]["learner_model_param"]["base_score"])


def get_leaf_indices(tree):
    """
    tree: a tree (dictionary) in loaded in json model file

    Returns indices of the leaves in json_tree (depth of 2)
    """
    left_children = tree["learner"]["gradient_booster"]["model"]["trees"][
        "left_children"
    ]
    right_children = tree["learner"]["gradient_booster"]["model"]["trees"][
        "right_children"
    ]
    leaf_indices = []
    for index, (left, right) in enumerate(zip(left_children, right_children)):
        if left == -1 and right == -1:
            leaf_indices.append(index)

    return leaf_indices


def get_ordered_leaf_indices(tree):
    """
    tree: a tree (dictionary) in loaded in json model file

    Returns List --> For a 7 node depth two tree with four leaves, returns list of leaf indices in order from left to right
    """
    # Left Subtree
    left_index = tree["left_children"][0]
    left_left_index = tree["left_children"][left_index]
    left_right_index = tree["right_children"][left_index]

    # Right Subtree
    right_index = tree["right_children"][0]
    right_left_index = tree["left_children"][right_index]
    right_right_index = tree["right_children"][right_index]

    return [left_left_index, left_right_index, right_left_index, right_right_index]


def get_filtered_tree_indices(model, feature_tuple=None):
    """
    model: xgb.train(...)
    feature_tuple: 0-indexing
        (0, ) means trees only with f(x1)
        (0, 1) means trees only with f(x1x2)
    Returns list of tree indices with splits corresponding to feature_tuple
        [0, 1, 4] means that trees 0, 1, and 4 in model contain features in that feature_tuple
    """

    def get_features_used(node, features=None):
        """
        node: tree/subtree (originally the root node) -- dictionary

        Returns set of all features used in one tree (represented as ints w/0-indexing)
            {0, 1} means x1 and x2 were used (subsequently?) in the tree

        """
        if features is None:
            features = set()
        if "split" in node:
            features.add(int(node["split"][1:]) - 1)
            # Only recurse if not a leaf
            if "children" in node:
                for child in node["children"]:
                    get_features_used(child, features)
        return features

    # tree_dump returns trees as JSON strings ['{'node_id': 0, 'depth' = 1, etc.}', '{}', etc.]
    tree_dump = model.get_dump(dump_format="json")
    filtered_tree_indices = []

    # For each tree, check for set equality (features_used vs. feature_tuple)
    for (
        i,
        tree_str,
    ) in enumerate(tree_dump):
        tree = json.loads(tree_str)
        features_used = get_features_used(tree)
        features_needed_set = set(feature_tuple)
        if features_used == features_needed_set:
            filtered_tree_indices.append(i)
    return filtered_tree_indices


##### TREE FILTERING #####
def filter_save_load(
    model, feature_tuple=None, output_file_name="new_model.json", folder="loaded_models"
):
    """
    model: xgb.train(...)
    feature_tuple: 0-indexing
        (0, ) means trees only with f(x1)
        (0, 1) means trees only with f(x1x2)
    Returns new model that only contains trees with features specified by feature_tuple
    """
    ##### LOAD #####
    original_model_file = get_model_file(model, "original_model.json")
    tree_indices = get_filtered_tree_indices(model, feature_tuple)

    ##### EDIT TREES #####
    new_trees = []
    id_count = 0
    for i in tree_indices:
        new_trees.append(
            original_model_file["learner"]["gradient_booster"]["model"]["trees"][i]
        )
        new_trees[id_count]["id"] = id_count
        id_count += 1
    original_model_file["learner"]["gradient_booster"]["model"]["trees"] = new_trees

    ##### EDIT OTHER FILE PARAMS #####
    # Edit num_trees
    original_model_file["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(len(tree_indices))
    # Edit iteration_indptr
    original_model_file["learner"]["gradient_booster"]["model"]["iteration_indptr"] = [
        i for i in range(len(tree_indices) + 1)
    ]
    # Edit tree_info
    original_model_file["learner"]["gradient_booster"]["model"]["tree_info"] = [
        0 for _ in range(len(tree_indices))
    ]

    ##### SAVE #####
    new_model = get_model(original_model_file, output_file_name, folder)
    return new_model


def filter_save_load_list(model, feature_tuple_list=None, output_file_name_list=None):
    """
    model: xgb.train(...)
    feature_tuple: 0-indexing
        (0, ) means trees only with f(x1)
        (0, 1) means trees only with f(x1x2)
    output_file_name_list: corresponding list of file names to be saved, ending with json
    Returns new model that only contains trees with features specified by feature_tuple

    """
    if output_file_name_list is None:
        output_file_name_list = [
            "model" + str(i) + ".json" for i in range(1, len(feature_tuple_list) + 1)
        ]
    output_models = []
    for i in range(len(output_file_name_list)):
        output_file_name = output_file_name_list[i]
        features_tuple = output_file_name_list[i]

        output_models.append(filter_save_load(model, output_file_name, features_tuple))

    return output_models


##### APPENDING NEW TREES #####
def get_new_duplicate_tree(model_file, leaf_val, new_id):
    """
    model_file: loaded in file
        with open("original_model.json", "r") as f:
            original_model_file = json.load(f)
    leaf_val: float
    new_id: int

    Returns a tree in the structure of the first tree in model_file
        model_file must have AT LEAST ONE TREE
    """
    leaf_val = float(leaf_val)
    # Get arbitrary tree structure
    new_tree = model_file["learner"]["gradient_booster"]["model"]["trees"][0].copy()
    new_tree["base_weights"] = [leaf_val for _ in range(len(new_tree["base_weights"]))]
    new_tree["split_conditions"] = [
        leaf_val for _ in range(len(new_tree["base_weights"]))
    ]
    new_tree["id"] = new_id
    return new_tree


def save_load_new_trees(
    model,
    leaf_val,
    base_score,
    num_new_trees,
    output_file_name="new_model.json",
    folder="loaded_models",
):
    """
    model: xgb.train(...)
    leaf_val: float
    base_score: string
    num_new_trees: int (number of trees to add)

    Returns new model with additional trees specified
    """
    original_model_file = get_model_file(model)

    ##### ADD TREES #####
    original_num_trees = int(
        original_model_file["learner"]["gradient_booster"]["model"][
            "gbtree_model_param"
        ]["num_trees"]
    )
    cur_id_num = original_num_trees

    for _ in range(num_new_trees):
        additional_tree = get_new_duplicate_tree(
            original_model_file, leaf_val, cur_id_num
        )
        original_model_file["learner"]["gradient_booster"]["model"]["trees"].append(
            additional_tree
        )

        cur_id_num += 1

    ##### EDIT OTHER FILE PARAMS #####
    original_model_file["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(original_num_trees + num_new_trees)
    original_model_file["learner"]["gradient_booster"]["model"]["iteration_indptr"] = [
        i for i in range(original_num_trees + num_new_trees + 1)
    ]
    original_model_file["learner"]["gradient_booster"]["model"]["tree_info"] = [0] * (
        original_num_trees + num_new_trees
    )
    original_model_file["learner"]["learner_model_param"]["base_score"] = str(
        float(base_score)
    )

    new_model = get_model(original_model_file, output_file_name, folder)
    return new_model


##### PURIFICATION HELPER FUNCTIONS #####
def get_new_depth_one_tree(
    leaf_val_left, leaf_val_right, new_id, split_index, split_condition
):
    """
    leaf_val_left: float
    leaf_val_right: float
    new_id: int
    split_index: int (0-indexing)
    split_condition: float

    Returns a depth one tree (one split along feature specified by split_index) with leaf_vals
    """
    leaf_val_left = float(leaf_val_left)
    leaf_val_right = float(leaf_val_right)
    tree = {
        "base_weights": [-0.12812316, leaf_val_left, leaf_val_right],
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": [0, 0, 0],
        "id": new_id,
        "left_children": [1, -1, -1],
        "loss_changes": [2600824.5, 0.0, 0.0],
        "parents": [2147483647, 0, 0],
        "right_children": [2, -1, -1],
        "split_conditions": [split_condition, leaf_val_left, leaf_val_right],
        "split_indices": [split_index, 0, 0],
        "split_type": [0, 0, 0],
        "sum_hessian": [700.0, 336.0, 364.0],
        "tree_param": {
            "num_deleted": "0",
            "num_feature": str(split_index + 1),
            "num_nodes": "3",
            "size_leaf_vector": "1",
        },
    }
    return tree


def get_new_depth_two_tree_left():
    pass


def get_new_depth_two_tree_right():
    pass


def split_node(tree, leaf_index, node_index):
    """
    tree: tree (dictionary) from loaded in file (dictionary)
    leaf_index: index of leaf that will be replaced with a split node (int)
    node_index: index of node that contains info of what to split leaf with (int)

    Mutates tree by changing leaf to a depth-1 split
    """
    # properties of original leaf
    new_leaf_val = float(tree["base_weights"][leaf_index])
    leaf_hessian = float(tree["sum_hessian"][leaf_index])
    leaf_loss = float(tree["loss_changes"][leaf_index])

    # indices of new leaf nodes
    new_left_leaf_index = len(tree["base_weights"])  # 5
    new_right_leaf_index = new_left_leaf_index + 1  # 6

    # Add two new leaf nodes
    tree["base_weights"].extend([new_leaf_val, new_leaf_val])
    tree["left_children"].extend([-1, -1])
    tree["right_children"].extend([-1, -1])
    tree["split_indices"].extend([0, 0])
    tree["split_conditions"].extend([new_leaf_val, new_leaf_val])

    tree["parents"].extend([leaf_index, leaf_index])
    tree["default_left"].extend([0, 0])
    tree["loss_changes"].extend([leaf_loss / 2, leaf_loss / 2])
    tree["split_type"].extend([0, 0])
    tree["sum_hessian"].extend([leaf_hessian / 2, leaf_hessian / 2])

    # Update original leaf node into a split node
    tree["base_weights"][leaf_index] = 0
    tree["left_children"][leaf_index] = new_left_leaf_index
    tree["right_children"][leaf_index] = new_right_leaf_index
    tree["split_indices"][leaf_index] = tree["split_indices"][node_index]
    tree["split_conditions"][leaf_index] = tree["split_conditions"][node_index]

    # Update other parameters
    tree["tree_param"]["num_nodes"] = str(len(tree["base_weights"]))

    # Make sure necessary items are float types
    float_keys = ["base_weights", "split_conditions", "loss_changes", "sum_hessian"]
    for k in float_keys:
        tree[k] = [float(v) for v in tree[k]]

    return


def get_subtract_means_seven_nodes(tree, node_index, test_data):
    """
    tree: tree (dictionary) from loaded in file (dictionary)
    node_index (int): index of node we're interested in
    test_data: DMatrix

    Returns a tuple of (mean_1, mean_2):
        mean_1: mean weighted by data distribution on left split
        mean_2: mean weighted by data distribution on right split

    Mutates tree such that means are subtracted from corresponding leaves
    """
    # Extract test_data as numpy array
    X = test_data.get_data()
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Get specific leaf indices
    A_index, B_index, C_index, D_index = get_ordered_leaf_indices(tree)

    # Get number of test points that fall into each leaf
    num_A = 0
    num_B = 0
    num_C = 0
    num_D = 0

    root_split_index = tree["split_indices"][0]
    root_split_condition = tree["split_conditions"][0]

    root_left_index = tree["left_children"][0]
    root_right_index = tree["right_children"][0]

    root_left_split_index = tree["split_indices"][root_left_index]
    root_left_split_condition = tree["split_conditions"][root_left_index]
    root_right_split_index = tree["split_indices"][root_right_index]
    root_right_split_condition = tree["split_conditions"][root_right_index]

    for test_point in X:
        if test_point[root_split_index] < root_split_condition:
            if test_point[root_left_split_index] < root_left_split_condition:
                num_A += 1
            else:
                num_B += 1
        else:
            if test_point[root_right_split_index] < root_right_split_condition:
                num_C += 1
            else:
                num_D += 1

    # Split by root index (AB vs. CD)
    if node_index == 0:
        left_mean = (
            num_A * tree["base_weights"][A_index]
            + num_B * tree["base_weights"][B_index]
        ) / (num_A + num_B)
        right_mean = (
            num_C * tree["base_weights"][C_index]
            + num_D * tree["base_weights"][D_index]
        ) / (num_C + num_D)

        # Subtract means from nodes
        tree["base_weights"][A_index] -= left_mean
        tree["base_weights"][B_index] -= left_mean
        tree["base_weights"][C_index] -= right_mean
        tree["base_weights"][D_index] -= right_mean

    # Split by depth-1 index (AC vs. BD)
    else:
        left_mean = (
            num_A * tree["base_weights"][A_index]
            + num_C * tree["base_weights"][C_index]
        ) / (num_A + num_C)
        right_mean = (
            num_B * tree["base_weights"][B_index]
            + num_D * tree["base_weights"][D_index]
        ) / (num_B + num_D)

        # Subtract means from nodes
        tree["base_weights"][A_index] -= left_mean
        tree["base_weights"][B_index] -= right_mean
        tree["base_weights"][C_index] -= left_mean
        tree["base_weights"][D_index] -= right_mean

    return (left_mean, right_mean)

    # Use y_true??? or y_pred. Currently using y_pred


def get_leaf_count(model, test_data, tree_index):
    """
    model: xb.train(...)
    test_data: DMatrix
    tree_index: 0-indexing int representing tree we're interested in

    Returns dictionary:
        {leaf_index: # samples in test_data that map to leaf_index in tree specified by tree_index}
    """
    # shape: (n_samples x n_trees)
    leaf_indices = model.predict(test_data, pred_leaf=True)
    # shape: (n_samples x 1)
    leaf_indices = leaf_indices[:, tree_index]

    # Loop through leaf_indices
    leaf_counts = {}
    for leaf_index in leaf_indices:
        if leaf_index in leaf_counts:
            leaf_counts[leaf_index] += 1
        else:
            leaf_counts[leaf_index] = 1

    return leaf_counts


def get_means(tree, leaf_groups, leaf_count):
    """
    tree: tree (dictionary) from loaded in file (dictionary)
    leaf_groups: (list of lists)
        Each sublist contains indices of leaves that we want to get the mean of
    leaf_count (dictionary): {leaf_index: # samples in test_data that map to leaf_index in tree specified by tree_index}

    Returns (dictionary): Key: Tuple of leaf indices, Value: Mean of data distribution over those leaf indices
    """
    mean_leaf_indices = {}

    # loop through each group of indices
    numerator = 0
    num_data_points = 0

    for group in leaf_groups:
        for leaf_index in group:
            leaf_val = tree["base_weights"][leaf_index]

            num_data_points += leaf_count.get(leaf_index, 0)
            numerator += leaf_val * leaf_count.get(leaf_index, 0)

        group_tuple = tuple(group)
        mean_leaf_indices[group_tuple] = numerator / num_data_points

        # Reset
        numerator = 0
        num_data_points = 0

    return mean_leaf_indices


def subtract_means(tree, mean_leaf_indices):
    """
    tree: tree (dictionary) from loaded in file (dictionary)
    leaf_indices: (dictionary). Key: Tuple of leaf indices, Value: Mean to be subtracted from each leaf in the tuple of leaf indices

    Mutates tree to subtract means from corresponding leaves
    """
    for group, mean in mean_leaf_indices.items():
        for leaf_index in group:
            tree["base_weights"][leaf_index] -= mean
            tree["split_conditions"][leaf_index] -= mean
    return


##### PURIFICATION #####
def fANOVA_2D(model, dtest, output_file_name="new_model.json"):
    """
    model: xgb.train(...)
        saves as original_model.json
        max_depth = 2
        y = f(x1, x2)

    Returns new model that is the fANOVA Decomposition of model
        Same predictions as model
        Mean along every axis is 0
    """
    # FIRST SPLIT UP ALL DEPTH 2 TREES

    # Get indices of f(x1, x2) trees
    tree_indices_x1x2 = get_filtered_tree_indices(model, (0, 1))
    model_file = get_model_file(model)

    # Loop through and purify each f(x1, x2) tree
    for i in tree_indices_x1x2:
        tree = model_file["learner"]["gradient_booster"]["model"]["trees"][i]

        if len(tree["base_weights"]) == 5:
            # Mutate (purify) tree and add correction trees
            new_trees = purify_five_nodes(model_file, i, dtest)
            model_file["learner"]["gradient_booster"]["model"]["trees"].extend(
                new_trees
            )

            # Update model metadata
            num_trees = len(model_file["learner"]["gradient_booster"]["model"]["trees"])

            model_file["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ] = str(num_trees)

            model_file["learner"]["gradient_booster"]["model"]["iteration_indptr"] = (
                list(range(num_trees + 1))
            )
            model_file["learner"]["gradient_booster"]["model"]["tree_info"].extend(
                [0] * len(new_trees)
            )

    # Save and return new model
    new_model = get_model(model_file)
    return new_model


def purify_five_nodes(
    model_file, tree_index, test_data, folder="loaded_models", epsilon=1e-3, max_iter=10
):
    """
    model_file: loaded in file
            with open("original_model.json", "r") as f:
                original_model_file = json.load(f)
    tree_index: index of the tree we are purifying
    test_data: DMatrix
    epsilon: convergence parameter
    max_iter: max number of iterations

    Purifies one f(x1, x2) tree
        Returns new trees
        Mutates model_file
    """
    # Get parameters
    trees = model_file["learner"]["gradient_booster"]["model"]["trees"]
    tree = trees[tree_index]
    cur_id = len(trees)
    new_trees = []

    ##### MODIFY ORIGINAL TREE #####
    depth_one_node_indices = [tree["left_children"][0], tree["right_children"][0]]
    # Index of leaf at depth 1
    leaf_index = -1
    # Index of root node of subtree with two leaves
    node_index = -1

    for index in depth_one_node_indices:
        if tree["left_children"][index] == -1 and tree["right_children"][index] == -1:
            leaf_index = index
        else:
            node_index = index

    split_node(tree, leaf_index, node_index)

    ##### PURIFY #####
    iter_count = 0
    while iter_count < max_iter:
        total_change = 0

        ##### INTEGRATE OVER AXIS 1: ROOT NODE FEATURE #####
        mean_left, mean_right = get_subtract_means_seven_nodes(tree, 0, test_data)

        ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####
        split_index = tree["split_indices"][0]
        split_condition = tree["split_conditions"][0]

        additional_tree = get_new_depth_one_tree(
            mean_left, mean_right, cur_id, split_index, split_condition
        )

        new_trees.append(additional_tree)
        cur_id += 1

        ##### INTEGRATE OVER AXIS 2: OTHER FEATURE #####
        new_model = get_model(model_file)

        left_group = [
            tree["left_children"][leaf_index],
            tree["left_children"][node_index],
        ]
        right_group = [
            tree["right_children"][leaf_index],
            tree["right_children"][node_index],
        ]
        leaf_groups = [left_group, right_group]

        leaf_count = get_leaf_count(new_model, test_data, tree_index)
        mean_leaf_indices = get_means(tree, leaf_groups, leaf_count)
        total_change += sum(abs(mean) for mean in mean_leaf_indices.values())
        subtract_means(tree, mean_leaf_indices)

        mean_left, mean_right = get_subtract_means_seven_nodes(
            tree, tree["left_children"][0], test_data
        )

        ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####
        split_index = tree["split_indices"][0]
        split_condition = tree["split_conditions"][0]

        additional_tree = get_new_depth_one_tree(
            mean_left, mean_right, cur_id, split_index, split_condition
        )

        new_trees.append(additional_tree)
        cur_id += 1

        ##### CONVERGENCE CHECK #####
        if total_change < epsilon:
            break
        iter_count += 1

    return new_trees


if __name__ == "__main__":
    # # Set seed for reproducibility
    # np.random.seed(42)
    # x1 = np.random.uniform(0, 100, 10)
    # x2 = np.random.uniform(0, 100, 10)
    # y = 10 * x1 + 2 * x2 + 3 * x1 * x2 + 5

    # X = pd.DataFrame({"x1": x1, "x2": x2})
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.3, random_state=42
    # )

    # # Convert to DMatrix format required by xgb.train
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test, label=y_test)

    # # Convert to DMatrix format required by xgb.train
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test, label=y_test)

    # # Parameters (note different parameter names)
    # params = {
    #     "max_depth": 2,
    #     "learning_rate": 1.0,
    #     "objective": "reg:squarederror",
    #     "random_state": 42,
    # }

    # # Training with monitoring
    # model = xgb.train(
    #     params=params,
    #     dtrain=dtrain,
    #     num_boost_round=10,  # Equivalent to n_estimators
    #     evals=[(dtrain, "train"), (dtest, "test")],
    #     verbose_eval=True,
    # )

    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 10)
    x2 = np.random.uniform(0, 100, 10)
    y = 10 * x1 + 2 * x2 + 3 * x1 * x2 + 5

    X = pd.DataFrame({"x1": x1, "x2": x2})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Convert to DMatrix format required by xgb.train
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Parameters (note different parameter names)
    params = {
        "max_depth": 2,
        "learning_rate": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    # Training with monitoring
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5,  # Equivalent to n_estimators
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=True,
    )

    new_model = filter_save_load(model, (0, 1), "original_model.json")
    print(f"original_prediction: {new_model.predict(dtest)}")
    purified_new_model = fANOVA_2D(new_model, dtest)
    print(f"purified_prediction: {purified_new_model.predict(dtest)}")

    #### ONE ####
    # model_file = get_model_file(model)

    # new_model_file = purify_five_nodes(model_file, 0, dtest)
    # print(model.predict(dtest))

    # float_keys = ["base_weights", "split_conditions", "loss_changes", "sum_hessian"]
    # for tree in model_file["learner"]["gradient_booster"]["model"]["trees"]:
    #     for k in float_keys:
    #         tree[k] = [float(v) for v in tree[k]]

    # new_model = get_model(new_model_file)
    # print(model.predict(dtest))
    # print(new_model.predict(dtest))

    # # TTL
    # np.random.seed(42)

    # # Generate 50 features
    # n_features = 50
    # n_samples = 1000
    # X = pd.DataFrame(
    #     {f"x{i}": np.random.uniform(0, 100, n_samples) for i in range(n_features)}
    # )

    # # Create nonlinear target using interactions of a few features
    # y = 10 * X["x0"] + 5 * X["x1"] * X["x2"] + 3 * X["x3"] + 7 * X["x4"] * X["x5"]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.3, random_state=42
    # )

    # # Convert to DMatrix format
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test, label=y_test)

    # # XGBoost parameters
    # params = {
    #     "max_depth": 2,
    #     "learning_rate": 1.0,
    #     "objective": "reg:squarederror",
    #     "random_state": 42,
    # }

    # # Train model
    # model2 = xgb.train(
    #     params=params,
    #     dtrain=dtrain,
    #     num_boost_round=1000,
    #     evals=[(dtrain, "train"), (dtest, "test")],
    #     verbose_eval=False,
    # )

    # model2.save_model("model.json")
    # with open("model.json", "r") as f:
    #     model_2_file = json.load(model2)
    # trees = model_2_file["learner"]["gradient_booster"]["model"]["trees"]
    # for i, tree in enumerate(trees):
    #     features = set(trees["split_indices"])
    #     if len(features == 4):
    #         print("YES")
