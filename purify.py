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


def get_ordered_leaves(tree, node_index, leaf_indices=None, leaf_vals=None):
    """
    Args:
        tree (dictionary): tree from model_file
    node_index (int): index of (initially root) node in tree
    leaf_indices (list):

    Returns:
        list: list of leaf_indices (ints) in left to right order
            preorder traversal (depth first, left before right)
    """
    if leaf_indices is None:
        leaf_indices, leaf_vals = [], []

    l = tree["left_children"][node_index]
    r = tree["right_children"][node_index]

    # node_index is a leaf
    if l == -1 and r == -1:
        leaf_indices.append(node_index)
        leaf_vals.append(tree["base_weights"][node_index])
    else:
        get_ordered_leaves(tree, l, leaf_indices, leaf_vals)
        get_ordered_leaves(tree, r, leaf_indices, leaf_vals)

    return (leaf_indices, leaf_vals)


def traverse_tree(tree, sample):
    """
    Args:
        tree (dictionary): tree from model_file
        sample (numpy array): Feature values for the sample
    Returns:
        int: index of the leaf node that sample falls into
    """
    node_index = 0
    while True:
        l = tree["left_children"][node_index]
        r = tree["right_children"][node_index]

        # leaf
        if l == -1 and r == -1:
            return node_index

        # other
        split_index = tree["split_indices"][node_index]
        split_condition = tree["split_conditions"][node_index]
        if sample[split_index] < split_condition:
            node_index = l
        else:
            node_index = r


##### TREE FILTERING #####
def get_filtered_tree_indices(model, feature_tuple=None):
    """
    Args:
        model (Booster): trained model
        feature_tuple (tuple): 0-indexing tuple representing features we want to filter for
            (0, ) --> trees only with x1 splits | f(x1)
            (0, 1) --> trees only with x1 AND x2 splits | f(x1, x2)

    Returns:
        set of ints: set of tree indices with splits corresopnding to feature_tuple
            {0, 1, 4} means that trees 0, 1, and 4 in "model" contain the exact features in "feature_tuple"
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
    filtered_tree_indices = set()

    # For each tree, check for set equality (features_used vs. feature_tuple)
    for (
        i,
        tree_str,
    ) in enumerate(tree_dump):
        tree = json.loads(tree_str)
        features_used = get_features_used(tree)
        features_needed_set = set(feature_tuple)
        if features_used == features_needed_set:
            filtered_tree_indices.add(i)
    return filtered_tree_indices


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
        list: list of new_models (Booster objects) corresponding to each feature_tuple in "feature_tuple_list"
    """
    # Default file names
    if output_file_name_list is None:
        output_file_name_list = [
            "model" + str(i) + ".json" for i in range(1, len(feature_tuple_list) + 1)
        ]

    # Add new models
    output_models = []
    for i in range(len(output_file_name_list)):
        output_file_name = output_file_name_list[i]
        features_tuple = feature_tuple_list[i]

        output_models.append(
            get_filtered_model(model, features_tuple, output_file_name)
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
        "is_compensation": True,
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
    A_index, B_index, C_index, D_index = get_ordered_leaves(tree, 0)[0]

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

    # indices of new leaf nodes
    new_left_leaf_index = len(tree["base_weights"])
    new_right_leaf_index = new_left_leaf_index + 1

    # Add two new leaf nodes
    tree["base_weights"].extend([new_leaf_val, new_leaf_val])
    tree["left_children"].extend([-1, -1])
    tree["right_children"].extend([-1, -1])
    tree["split_indices"].extend([0, 0])
    tree["split_conditions"].extend([new_leaf_val, new_leaf_val])

    tree["parents"].extend([leaf_index, leaf_index])
    tree["default_left"].extend([0, 0])
    tree["loss_changes"].extend([0.0, 0.0])
    tree["split_type"].extend([0, 0])
    tree["sum_hessian"].extend([0.0, 0.0])

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


def get_and_subtract_means_seven_nodes(tree, node_axis_index, dataset):
    """
    Args:
        tree (dictionary): tree from model_file
        node_axis_index (int): index of node we're interested in, tells us axis to integrate on
            0 --> integrate on root node's split_condition-axis
            else --> integrate on other axis
        dataset (Dmatrix): dataset of points (x-vals)

    Returns:
        tuple: (mean_1, mean_2)
            mean_1 (float): mean weighted by data distribution on left split
            mean_2 (float): mean weighted by data distribution on right split

    Mutates tree such that means are subtracted from corresponding leaves
    """
    # Extract dataset as numpy array
    X = dataset.get_data()
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Get number of test points that fall into each leaf
    A_index, B_index, C_index, D_index = get_ordered_leaves(tree, 0)[0]
    num_leaves = {A_index: 0, B_index: 0, C_index: 0, D_index: 0}

    for test_point in X:
        leaf_index = traverse_tree(tree, test_point)
        num_leaves[leaf_index] += 1

    num_A, num_B, num_C, num_D = [num for num in num_leaves.values()]

    # Split by root index (AB vs. CD)
    if node_axis_index == 0:
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


##### PURIFICATION #####
def purify_five_nodes_two_features(tree, dataset, epsilon=1e-1, max_iter=10):
    """
    Args:
        tree (dictionary): tree from model_file
        dataset (DMatrix): dataset of points (x-vals)
        epsilon (float): if change is less than epsilon, END EARLY
        max_iter (int): max number of iterations

    Returns:
        List: list of tree (dictionaries) that are 1-feature compensations for purification
    Mutates "tree" such that its axes have a mean of 0 (purification)
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

    ##### PURIFY ORIGINAL TREE #####
    new_trees = []
    iter_count = 0
    while iter_count < max_iter:
        total_change = 0

        ##### INTEGRATE OVER AXIS 1: ROOT NODE FEATURE #####
        mean_left, mean_right = get_and_subtract_means_seven_nodes(tree, 0, dataset)
        total_change += abs(mean_left) + abs(mean_right)

        ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####
        split_index = tree["split_indices"][0]
        split_condition = tree["split_conditions"][0]

        additional_tree = new_three_node_tree(
            mean_left, mean_right, split_index, split_condition, -1
        )

        new_trees.append(additional_tree)

        ##### INTEGRATE OVER AXIS 2: OTHER FEATURE #####
        root_left_index = tree["left_children"][0]
        mean_left, mean_right = get_and_subtract_means_seven_nodes(
            tree, root_left_index, dataset
        )
        total_change += abs(mean_left) + abs(mean_right)

        ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####
        split_index = tree["split_indices"][root_left_index]
        split_condition = tree["split_conditions"][root_left_index]

        additional_tree = new_three_node_tree(
            mean_left, mean_right, split_index, split_condition, -1
        )

        new_trees.append(additional_tree)

        ##### CONVERGENCE CHECK #####
        if total_change < epsilon:
            break
        iter_count += 1
        # print(iter_count) --> Usually converges around 6-7 iterations

    return new_trees


def purify_one_feature(tree, dataset):
    """
    Args:
        tree (dictionary): tree (max_depth of 2) from model_file
        dataset (DMatrix):
    Returns:
        float: correction mean prediction float to add to base_score
    Mutates tree such that mean prediction is zero across dataset points
        Subtracts mean prediction from all leaves
    """
    # Extract test data as numpy array
    X = dataset.get_data()
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Get tree and leaf info
    leaf_indices, _ = get_ordered_leaves(tree, 0)
    leaf_count = {leaf_index: 0 for leaf_index in leaf_indices}
    sample_count = 0

    # Count samples per leaf
    for test_point in X:
        leaf_index = traverse_tree(tree, test_point)
        sample_count += 1
        leaf_count[leaf_index] += 1

    # Edge Case
    if sample_count == 0:
        return 0.0

    # Get mean prediction
    sum = 0
    for leaf_idx, num_samples in leaf_count.items():
        sum += tree["base_weights"][leaf_idx] * num_samples
    mean = sum / sample_count

    # Subtract mean from leaves
    for leaf_index in leaf_indices:
        tree["base_weights"][leaf_index] -= mean
        tree["split_conditions"][leaf_index] -= mean

    return mean


def purify_2D(
    model, dataset, output_file_name="new_model.json", output_folder="loaded_models"
):
    """
    Args:
        model (Booster): max_depth = 2
        dataset (DMatrix): set of points (x-vals)
        output_file_name (string):
        output_folder (string):

    Returns:
        Booster: new model that is the fANOVA Decomposition of model
            Same predictions as model
            Mean along each axis is 0
    """
    ##### SPLIT UP 7-NODE f(x1, x2) TREES INTO TWO, 5-NODE f(x1, x2) TREES#####
    tree_indices_x1x2x3 = (
        get_filtered_tree_indices(model, (0, 1))
        | get_filtered_tree_indices(model, (1, 2))
        | get_filtered_tree_indices(model, (0, 2))
        | get_filtered_tree_indices(model, (0, 1, 2))
    )
    model_file = get_model_file(model)

    tree_list_all = model_file["learner"]["gradient_booster"]["model"]["trees"]
    bias_tree_vals = []
    tree_list_one_feature = []
    tree_list_two_features = []

    # Append equivalent 5-node trees
    for i, tree in enumerate(tree_list_all):
        # 0-feature tree
        if len(tree["base_weights"]) == 1:
            bias_tree_vals.append(tree["base_weights"][0])
        # two-feature tree
        elif i in tree_indices_x1x2x3:
            if int(tree["tree_param"]["num_nodes"]) == 7:
                new_trees = split_tree(tree)
                tree_list_two_features.extend(new_trees)
            else:
                tree_list_two_features.append(tree)
        # one-feature tree
        else:
            tree_list_one_feature.append(tree)

    ##### PURIFY EACH f(x1, x2) TREE (each should be 5 nodes now) #####
    for tree in tree_list_two_features:
        new_trees = purify_five_nodes_two_features(tree, dataset)
        tree_list_one_feature.extend(new_trees)

    ##### PURIFY EACH f(x1), f(x2), TREE #####
    new_base_score = float(model_file["learner"]["learner_model_param"]["base_score"])

    for tree in tree_list_one_feature:
        # if tree.get("is_compensation"):
        #     continue

        mean = purify_one_feature(tree, dataset)
        new_base_score += mean

    ##### ADD 0-feature trees to bias #####
    for bias_val in bias_tree_vals:
        new_base_score += bias_val

    ##### UPDATE TREES #####
    model_file["learner"]["gradient_booster"]["model"]["trees"] = (
        tree_list_two_features + tree_list_one_feature
    )

    model_file["learner"]["learner_model_param"]["base_score"] = str(new_base_score)

    ##### UPDATE MODEL METADATA #####
    num_trees = len(model_file["learner"]["gradient_booster"]["model"]["trees"])

    model_file["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(num_trees)
    model_file["learner"]["gradient_booster"]["model"]["iteration_indptr"] = list(
        range(num_trees + 1)
    )
    model_file["learner"]["gradient_booster"]["model"]["tree_info"] = [0] * num_trees

    for i, tree in enumerate(
        model_file["learner"]["gradient_booster"]["model"]["trees"]
    ):
        tree["id"] = i

    ##### SAVE AND RETURN #####
    new_model = get_model(model_file, output_file_name, output_folder)
    new_model.set_param({"base_score": new_base_score})
    return new_model


def fANOVA_2D(model, dataset):
    """
    Args:
        model (Booster): model from model_file (max_depth = 2)
        dataset (DMatrix)
    Returns:
        dictionary:
            key (string): "bias", "x1", "x2", "x1x2", "x1x3", etc.
            value (Booster): model
    """
    model_dict = {}

    # Purify Model
    purified_model = purify_2D(model, dataset)
    purified_model_file = get_model_file(model)

    # Filter Model
    filtered_model_list = get_filtered_model_list(
        purified_model, [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)]
    )
    bias = float(purified_model_file["learner"]["learner_model_param"]["base_score"])

    # note: could simplify with recursion
    model_names = ["x1", "x2", "x3", "x1x2", "x2x3", "x1x3", "x1x2x3"]

    for model_name, model in zip(model_names, filtered_model_list):
        # Reset bias to 0 (don't want to overcount)
        model.set_param({"base_score": 0.0})

        model_dict[model_name] = model

    return model_dict, bias


if __name__ == "__main__":
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 10)
    x2 = np.random.uniform(0, 100, 10)
    x3 = np.random.uniform(0, 100, 10)
    y = 10 * x1 + 2 * x2 + 3 * x1 * x2 + 5 + 4 * x3

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
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
        num_boost_round=1000,  # Equivalent to n_estimators
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=True,
    )

    filtered_model = get_filtered_model(model, (0, 1), "filtered_model.json")
    print(f"original_prediction (0, 1): {filtered_model.predict(dtest)}")
    purified_new_model = purify_2D(filtered_model, dtrain)
    print(f"purified_prediction: {purified_new_model.predict(dtest)}")

    model_file = get_model_file(model, "original_model.json")
    print(f"original_prediction: {model.predict(dtest)}")
    purified_model = purify_2D(model, dtrain)
    print(f"purified_prediction: {purified_model.predict(dtest)}")

    model_dict, bias = fANOVA_2D(model, dtrain)
    print(model_dict["x1x2x3"])
