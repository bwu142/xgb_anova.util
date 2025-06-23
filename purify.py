# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
import plotly.express as px
import os
import plotly.graph_objects as go


##### TREE HELPER FUNCTIONS #####
def get_model_file(
    model, input_file_name="original_model.json", folder="loaded_models"
):
    """
    Args:
        model (Booster): trained model
        input_file_name (string):
        folder (string):

    Returns:
        dictionary: file version of model that can be edited
    Saves: json file as "input_file_name" in "folder"
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
    Args:
        model_file (dictionary): file version of model that can be edited
        input_file_name (string):
        folder (string):

    Returns:
        Booster: model used for predictions
    Saves model_file as "output_file_name" in "folder"
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


def get_bias(model, input_file_name="original_model.json", folder="loaded_models"):
    """
    Args:
        model (Booster):

    Returns:
        float: model's bias/base score
    """
    bias_model_file = get_model_file(model, input_file_name, folder)
    return float(bias_model_file["learner"]["learner_model_param"]["base_score"])


def get_leaf_indices_seven_nodes(tree):
    """
    Args:
        tree (dictinoary): 7-node, depth-2 tree in model_file with 4 leaves

    Returns:
        list of ints: list of four leaf indices in order from left to right
    """
    # Left Subtree
    root_left_index = tree["left_children"][0]
    root_left_left_index = tree["left_children"][root_left_index]
    root_left_right_index = tree["right_children"][root_left_index]

    # Right Subtree
    root_right_index = tree["right_children"][0]
    root_right_left_index = tree["left_children"][root_right_index]
    root_right_right_index = tree["right_children"][root_right_index]

    return [
        root_left_left_index,
        root_left_right_index,
        root_right_left_index,
        root_right_right_index,
    ]


def get_filtered_tree_indices(model, feature_tuple=None):
    """
    Args:
        model (Booster): trained model
        feature_tuple (tuple): 0-indexing tuple representing features we want to filter for
            (0, ) --> trees only with x1 splits | f(x1)
            (0, 1) --> trees only with x1 AND x2 splits | f(x1, x2)

    Returns:
        list of ints: list of tree indices with splits corresopnding to feature_tuple
            [0, 1, 4] means that trees 0, 1, and 4 in "model" contain the exact features in "feature_tuple"
    """

    def get_features_used(node, features=None):
        """
        Args:
            node (dictionary): tree/subtree (originally root node)
        Returns:
            set of ints: all features used in tree "node" (represented as ints w/0-indexing)
                {0, 1} means that only x1 and x2 were used as splits in the tree
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
def get_filtered_model(
    model, feature_tuple=None, output_file_name="new_model.json", folder="loaded_models"
):
    """
    Args:
        model (Booster): trained model
        feature_tuple (tuple): 0-indexing tuple representing features we want to filter for
    Returns:
        Booster: new model that only contains trees with features specified by feature_tuple
    Saves new model as "output_file_name" in "folder"
    """
    ##### LOAD #####
    original_model_file = get_model_file(model, "original_model.json")
    tree_indices = get_filtered_tree_indices(model, feature_tuple)

    ##### FILTER TREES #####
    new_trees = []
    id_count = 0
    for i in tree_indices:
        new_trees.append(
            original_model_file["learner"]["gradient_booster"]["model"]["trees"][i]
        )
        new_trees[id_count]["id"] = id_count
        id_count += 1
    original_model_file["learner"]["gradient_booster"]["model"]["trees"] = new_trees

    ##### UPDATE MODEL METADATA #####
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


def get_filtered_model_list(model, feature_tuple_list=None, output_file_name_list=None):
    """
    Args:
        model (Booster): trained model
    feature_tuple_list (list of tuples): list of feature_tuples
    output_file_name_list (list of strings): corresponding list of file names to be saved, ending with json

    Returns:
        list: list of new_models corresponding to each feature_tuple in "feature_tuple_list"
    """
    if output_file_name_list is None:
        output_file_name_list = [
            "model" + str(i) + ".json" for i in range(1, len(feature_tuple_list) + 1)
        ]
    output_models = []
    for i in range(len(output_file_name_list)):
        output_file_name = output_file_name_list[i]
        features_tuple = output_file_name_list[i]

        output_models.append(
            get_filtered_model(model, output_file_name, features_tuple)
        )

    return output_models


##### CREATING NEW TREES #####
def new_three_node_tree(
    leaf_val_left, leaf_val_right, split_index, split_condition, new_id
):
    """
    Args:
        leaf_val_left (float):
        leaf_val_right (float):
        split_index (int): 0-indexing
        split_condition (float):
        new_id (int):

    Returns:
        dictionary: 2-node, depth-1 tree
    """
    leaf_val_left = float(leaf_val_left)
    leaf_val_right = float(leaf_val_right)
    split_condition = float(split_condition)

    new_tree = {
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
    return new_tree


def new_five_node_tree_left(
    root_split_index,
    root_split_condition,
    root_left_split_index,
    root_left_split_condition,
    leaf_val_left,
    leaf_val_right,
    new_id,
):
    """
    Args:
        root_split_index (int):
        root_split_condition (float):
        root_left_split_index (int):
        root_left_split_condition (float):
        leaf_val_left (float):
        leaf_Val_right (float):
        new_id (int):

    Returns:
        dictionary: five node tree skewed left
            left child of root node is the root of a depth-1 subtree
            right child of root node is a leaf of value 0.0
    """
    leaf_val_left = float(leaf_val_left)
    leaf_val_right = float(leaf_val_right)
    root_split_condition = float(root_split_condition)
    root_left_split_condition = float(root_left_split_condition)

    new_tree = {
        "base_weights": [0.26570892, -0.63801795, 0.0, leaf_val_left, leaf_val_right],
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": [0, 0, 0, 0, 0],
        "id": new_id,
        "left_children": [1, 3, -1, -1, -1],
        "loss_changes": [24.010551, 16.733408, 0.0, 0.0, 0.0],
        "parents": [2147483647, 0, 0, 1, 1],
        "right_children": [2, 4, -1, -1, -1],
        "split_conditions": [
            root_split_condition,
            root_left_split_condition,
            0.0,
            leaf_val_left,
            leaf_val_right,
        ],
        "split_indices": [root_split_index, root_left_split_index, 0, 0, 0],
        "split_type": [0, 0, 0, 0, 0],
        "sum_hessian": [7.0, 6.0, 1.0, 4.0, 2.0],
        "tree_param": {
            "num_deleted": "0",
            "num_feature": "2",
            "num_nodes": "5",
            "size_leaf_vector": "1",
        },
    }

    return new_tree


def new_five_node_tree_right(
    root_split_index,
    root_split_condition,
    root_right_split_index,
    root_right_split_condition,
    leaf_val_left,
    leaf_val_right,
    new_id,
):
    """
    root_split_index: (int)
    root_split_condition: (float)
    root_right_split_index: (int)
    root_right_split_condition: (float)
    leaf_val_left: (float)
    leaf_val_right: (float)
    new_id: (int)

    Returns five node tree (dictionary) skewed right

    Args:
        root_split_index (int):
        root_split_condition (float):
        root_right_split_index (int):
        root_right_split_condition (float):
        leaf_val_left (float):
        leaf_Val_right (float):
        new_id (int):

    Returns:
        dictionary: five node tree skewed right
            left child of root node is a leaf of value 0.0
            right child of root node is the root of a depth-1 subtree
    """
    leaf_val_left = float(leaf_val_left)
    leaf_val_right = float(leaf_val_right)
    root_split_condition = float(root_split_condition)
    root_right_split_condition = float(root_right_split_condition)

    new_tree = {
        "base_weights": [0.17801666, 0.0, 0.8273141, leaf_val_left, leaf_val_right],
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": [0, 0, 0, 0, 0],
        "id": new_id,
        "left_children": [1, -1, 3, -1, -1],
        "loss_changes": [14.073252, 0.0, 4.4261208, 0.0, 0.0],
        "parents": [2147483647, 0, 0, 2, 2],
        "right_children": [2, -1, 4, -1, -1],
        "split_conditions": [
            root_split_condition,
            0.0,
            root_right_split_condition,
            leaf_val_left,
            leaf_val_right,
        ],
        "split_indices": [root_split_index, 0, root_right_split_index, 0, 0],
        "split_type": [0, 0, 0, 0, 0],
        "sum_hessian": [7.0, 1.0, 6.0, 4.0, 2.0],
        "tree_param": {
            "num_deleted": "0",
            "num_feature": "2",
            "num_nodes": "5",
            "size_leaf_vector": "1",
        },
    }

    return new_tree


##### DEPTH-2 TREE PURIFICATION HELPER FUNCTIONS #####
def split_tree(tree):
    """
    Args:
        tree (dictinoary): tree in model_file

    Returns:
        list of dictionaries: list of two new 5-node trees that sum to "tree"
    """
    ##### GET TREE INFO #####

    # Depth 0
    root_split_index = tree["split_indices"][0]
    root_split_condition = tree["split_conditions"][0]

    # Depth 1
    root_left_index = tree["left_children"][0]
    root_right_index = tree["right_children"][0]

    root_left_split_index = tree["split_indices"][root_left_index]
    root_left_split_condition = tree["split_conditions"][root_left_index]
    root_right_split_index = tree["split_indices"][root_right_index]
    root_right_split_condition = tree["split_conditions"][root_right_index]

    # Depth 2
    A_index, B_index, C_index, D_index = get_leaf_indices_seven_nodes(tree)
    A_val, B_val, C_val, D_val = (
        tree["base_weights"][A_index],
        tree["base_weights"][B_index],
        tree["base_weights"][C_index],
        tree["base_weights"][D_index],
    )

    # Get new trees
    tree_left = new_five_node_tree_left(
        root_split_index,
        root_split_condition,
        root_left_split_index,
        root_left_split_condition,
        A_val,
        B_val,
        -1,
    )
    tree_right = new_five_node_tree_right(
        root_split_index,
        root_split_condition,
        root_right_split_index,
        root_right_split_condition,
        C_val,
        D_val,
        -1,
    )
    return [tree_left, tree_right]


def split_node(tree, leaf_index, node_index):
    """
    Args:
        tree (dictionary): tree from model_file
        leaf_index (int): index of leaf that will be replaced with a split node
        node_index (int): index of node that our added split will replicate

    Returns:
        None
    Mutates tree by changing leaf_index to a depth-1 split mimicing split at node_index
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
    A_index, B_index, C_index, D_index = get_leaf_indices_seven_nodes(tree)

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
        left_total = num_A + num_B
        right_total = num_C + num_D

        if left_total > 0:
            left_mean = (
                num_A * tree["base_weights"][A_index]
                + num_B * tree["base_weights"][B_index]
            ) / (num_A + num_B)
        else:
            left_mean = 0.0

        if right_total > 0:
            right_mean = (
                num_C * tree["base_weights"][C_index]
                + num_D * tree["base_weights"][D_index]
            ) / (num_C + num_D)
        else:
            right_mean = 0.0

        # Subtract means from leaf nodes
        tree["base_weights"][A_index] -= left_mean
        tree["base_weights"][B_index] -= left_mean
        tree["base_weights"][C_index] -= right_mean
        tree["base_weights"][D_index] -= right_mean

        tree["split_conditions"][A_index] -= left_mean
        tree["split_conditions"][B_index] -= left_mean
        tree["split_conditions"][C_index] -= right_mean
        tree["split_conditions"][D_index] -= right_mean

    # Split by depth-1 index (AC vs. BD)
    else:
        left_total = num_A + num_C
        right_total = num_B + num_D

        if left_total > 0:
            left_mean = (
                num_A * tree["base_weights"][A_index]
                + num_C * tree["base_weights"][C_index]
            ) / (num_A + num_C)
        else:
            left_mean = 0.0

        if right_total > 0:
            right_mean = (
                num_B * tree["base_weights"][B_index]
                + num_D * tree["base_weights"][D_index]
            ) / (num_B + num_D)
        else:
            right_mean = 0.0

        # Subtract means from leaf nodes
        tree["base_weights"][A_index] -= left_mean
        tree["base_weights"][B_index] -= right_mean
        tree["base_weights"][C_index] -= left_mean
        tree["base_weights"][D_index] -= right_mean

        tree["split_conditions"][A_index] -= left_mean
        tree["split_conditions"][B_index] -= right_mean
        tree["split_conditions"][C_index] -= left_mean
        tree["split_conditions"][D_index] -= right_mean
    return (left_mean, right_mean)

    # Use y_true??? or y_pred. Currently using y_pred


##### PURIFICATION #####
def purify_five_nodes(tree, cur_id, test_data, epsilon=1e-1, max_iter=10):
    """
    tree: loaded in tree (dictionary) from json file
    cur_id: int
    test_data: DMatrix
    epsilon: convergence parameter
    max_iter: max number of iterations

    Purifies one f(x1, x2) tree
        Returns new trees
        Mutates model_file
    """
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

    # Add additional split to tree
    split_node(tree, leaf_index, node_index)

    ##### PURIFY #####
    new_trees = []
    iter_count = 0
    while iter_count < max_iter:
        total_change = 0

        ##### INTEGRATE OVER AXIS 1: ROOT NODE FEATURE #####
        mean_left, mean_right = get_subtract_means_seven_nodes(tree, 0, test_data)
        total_change += abs(mean_left) + abs(mean_right)

        ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####
        split_index = tree["split_indices"][0]
        split_condition = tree["split_conditions"][0]

        additional_tree = new_three_node_tree(
            mean_left, mean_right, split_index, split_condition, cur_id
        )

        new_trees.append(additional_tree)
        cur_id += 1

        ##### INTEGRATE OVER AXIS 2: OTHER FEATURE #####
        root_left_index = tree["left_children"][0]

        mean_left, mean_right = get_subtract_means_seven_nodes(
            tree, root_left_index, test_data
        )
        # print((mean_left, mean_right))
        total_change += abs(mean_left) + abs(mean_right)

        ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####
        split_index = tree["split_indices"][root_left_index]
        split_condition = tree["split_conditions"][root_left_index]

        additional_tree = new_three_node_tree(
            mean_left, mean_right, split_index, split_condition, cur_id
        )

        new_trees.append(additional_tree)
        cur_id += 1

        ##### CONVERGENCE CHECK #####
        if total_change < epsilon:
            break
        iter_count += 1
        # print(iter_count) --> Usually converges around 6-7 iterations

    return new_trees


def fANOVA_2D(
    model, dtest, output_file_name="new_model.json", output_folder="loaded_models"
):
    """
    model: xgb.train(...)
        saves as original_model.json
        max_depth = 2
        y = f(x1, x2)

    Returns new model that is the fANOVA Decomposition of model
        Same predictions as model
        Mean along every axis is 0
    """
    ##### SPLIT UP 7-NODE f(x1, x2) TREES INTO TWO, 5-NODE f(x1, x2) TREES#####
    tree_indices_x1x2 = get_filtered_tree_indices(model, (0, 1))
    model_file = get_model_file(model)

    # Append equivalent 5-node trees
    original_tree_list = model_file["learner"]["gradient_booster"]["model"]["trees"]
    for i in tree_indices_x1x2:
        tree = original_tree_list[i]
        if int(tree["tree_param"]["num_nodes"]) == 7:
            new_trees = split_tree(tree)
            original_tree_list.extend(new_trees)

    # print(f"CHECKPOINT 1: original tree list: {original_tree_list}")
    # Remove 7-node trees and correct for id numbers
    new_tree_list = []
    cur_id = 0
    for tree in original_tree_list:
        if int(tree["tree_param"]["num_nodes"]) == 5:
            tree["id"] = cur_id
            new_tree_list.append(tree)
            cur_id += 1

    # print(f"CHECKPOINT 2: NEW tree list: {new_tree_list}")

    ##### PURIFY EACH f(x1, x2) TREE (each should be 5 nodes now) #####
    new_tree_list_length = len(new_tree_list)

    for i in range(new_tree_list_length):
        tree = model_file["learner"]["gradient_booster"]["model"]["trees"][i]
        cur_id = len(new_tree_list)
        new_trees = purify_five_nodes(tree, cur_id, dtest)
        new_tree_list.extend(new_trees)

    model_file["learner"]["gradient_booster"]["model"]["trees"] = new_tree_list

    ##### UPDATE MODEL METADATA #####
    num_trees = len(model_file["learner"]["gradient_booster"]["model"]["trees"])

    model_file["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(num_trees)
    model_file["learner"]["gradient_booster"]["model"]["iteration_indptr"] = list(
        range(num_trees + 1)
    )
    model_file["learner"]["gradient_booster"]["model"]["tree_info"] = [0] * num_trees

    ##### SAVE AND RETURN #####
    new_model = get_model(model_file, output_file_name, output_folder)
    return new_model


if __name__ == "__main__":
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
        num_boost_round=100,  # Equivalent to n_estimators
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=True,
    )

    new_model = get_filtered_model(model, (0, 1), "original_model.json")
    print(f"original_prediction: {new_model.predict(dtest)}")
    purified_new_model = fANOVA_2D(new_model, dtest)
    print(f"purified_prediction: {purified_new_model.predict(dtest)}")
