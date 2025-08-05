# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
import os
import pandas as pd
from collections import defaultdict

# SET PARAM BASE IS AN INPUT
# LOAD THROUGH JSON AND ACTUALLY CHANGE BASE_SCORE KEY


##### CLASS TO RETURN #####
class fANOVA_Result:
    """
    Container for fANOVA 2D decomposition results.

    Args:
        original_model (xgb.Booster): Original unmodified XGBoost model.
        purified_model (xgb.Booster): Model with interaction effects removed.
        components (dict): Mapping from feature index tuples to submodels.
        bias (float): Global base score removed during purification.
    """

    def __init__(self, original_model, purified_model, purified_model_dict, bias):
        self.original_model = original_model
        self.purified_model = purified_model
        self.purified_model_dict = purified_model_dict
        self.bias = bias

    def predict_original(self, dmatrix):
        return self.original_model.predict(dmatrix)

    def predict_purified(self, dmatrix):
        return self.purified_model.predict(dmatrix)

    def predict_component(self, feature_tuple, dmatrix):
        return self.purified_model_dict[feature_tuple].predict(dmatrix)


##### TREE HELPER FUNCTIONS #####
def load_model_from_memory(file_path):
    """
    Args:
        file_path (str): Path to the .json file representing a saved XGBoost model.
            e.g. "loaded_models/new_model.json"
    Returns:
        xgb.Booster: Loaded Booster object.
    """
    new_model = xgb.Booster()
    new_model.load_model(file_path)
    return new_model


def get_model_file(
    model,
    save_to_disk=False,
    input_file_name="original_model.json",
    folder="loaded_models",
):
    """
    Args:
        model (Booster): trained model
        save_to_disk (bool): True --> saves model to disk. False --> uses in-memory stream
        input_file_name (string):
        folder (string):

    Returns:
        dictionary: file version of model that can be edited
    Saves: json file as "input_file_name" in "folder"
    """
    if save_to_disk:  # --- disk path ---
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
    else:  # --- in-memory ---
        raw = model.save_raw(raw_format="json")  # bytes
        return json.loads(raw.decode("utf-8"))  # dict


def get_model(
    model_file,
    save_to_disk=False,
    output_file_name="new_model.json",
    folder="loaded_models",
):
    """
    Args:
        model_file (dictionary): file version of model that can be edited
        save_to_disk: True --> saves model to disk. False --> uses in-memory stream
        input_file_name (string):
        folder (string):

    Returns:
        Booster: model used for predictions
    Saves model_file as "output_file_name" in "folder"
    """
    if save_to_disk:
        # Ensure file ends with .json
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

    else:
        json_bytes = json.dumps(model_file).encode("utf-8")  # <-- RAW BYTES
        booster = xgb.Booster()
        booster.load_model(bytearray(json_bytes))  # bytearray / bytes OK
        return booster


def update_metadata(
    model_file,
    base_score="0.0",
    save_to_disk=True,
    output_file_name="new_model.json",
    folder="loaded_models",
):
    """
    Updates model metadata based on the tree list in the model dictionary
    Args:
        model_file (dict):
    Returns:
        Booster: new model
    """
    # get tree info
    trees = model_file["learner"]["gradient_booster"]["model"]["trees"]
    num_trees = len(trees)

    # update model metadata
    model_file["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(num_trees)
    model_file["learner"]["gradient_booster"]["model"]["iteration_indptr"] = [
        i for i in range(num_trees + 1)
    ]
    model_file["learner"]["gradient_booster"]["model"]["tree_info"] = [
        0 for _ in range(num_trees)
    ]
    for i, tree in enumerate(trees):
        tree["id"] = i

    # set base score
    model_file["learner"]["learner_model_param"]["base_score"] = str(float(base_score))

    # Convert required items to floats
    float_requirements = [
        "base_weights",
        "loss_changes",
        "split_conditions",
        "sum_hessian",
    ]
    for tree in model_file["learner"]["gradient_booster"]["model"]["trees"]:
        for key in float_requirements:
            tree[key] = [float(val) for val in tree[key]]

    # Return updated model
    updated_model = get_model(model_file, save_to_disk, output_file_name, folder)
    updated_model.set_param({"base_score": base_score})
    return updated_model


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


def get_split_conditions_not_used(trees, feature_list, dataset):
    """
    Args:
        trees (list): list of json tree (dicts) from booster
        feature_list (list): list of all possible feature names/ids (order must match column index)
        dataset (DMatrix):
    Returns:
        dict:
            key: feature name (from feature_list)
            value:
                - numerical: sorted np.array of unique split thresholds
                - categorical: integer number of unique categories seen in splits
    """
    # ['c', 'float']
    feature_type_list = list(dataset.feature_types)
    feature_indices_categorical = [
        i for i, feature_type in enumerate(feature_type_list) if feature_type == "c"
    ]

    # Sets for collecting split info
    split_dict_numerical = defaultdict(set)
    split_dict_categorical = defaultdict(set)

    for tree in trees:
        split_indices = tree["split_indices"]
        split_conditions = tree["split_conditions"]
        left = tree["left_children"]
        right = tree["right_children"]

        # categorical split info
        cat_nodes = tree.get("categories_nodes", [])
        cat_segments = tree.get("categories_segments", [])
        cat_sizes = tree.get("categories_sizes", [])
        cats = tree.get("categories", [])

        cat_nodes_set = set(cat_nodes)  # for fast lookup

        # first: collect numerical split conditions
        for i in range(len(left)):
            # skip leaves
            if left[i] == -1 and right[i] == -1:
                continue
            feature_idx = split_indices[i]
            feat_type = feature_type_list[feature_idx]
            # categorical splits handled later
            if i in cat_nodes_set:
                continue
            if feat_type.startswith("float"):
                splits[feature_idx].add(split_conditions[i])

        # now handle categorical splits
        for j, node in enumerate(cat_nodes):
            feature_idx = split_indices[node]
            feat_type = feature_type_list[feature_idx]
            if feat_type.startswith("c") or feat_type.startswith("cat"):
                start = cat_segments[j]
                sz = cat_sizes[j]
                for cat in cats[start : start + sz]:
                    categories[feature_idx].add(cat)

    # build output dict
    final = {}
    for feat_idx in range(num_features):
        feat_name = feature_list[feat_idx]
        feat_type = feature_type_list[feat_idx]
        if feat_type.startswith("float"):
            arr = np.array(sorted(splits[feat_idx]))
            final[feat_name] = arr
        elif feat_type.startswith("c") or feat_type.startswith("cat"):
            final[feat_name] = len(categories[feat_idx])

    return final


def get_split_conditions(trees, dataset):
    """
    Args:
        trees (list): list of json tree (dicts) from booster
        feature_list (list): list of all possible feature names/ids (order must match column index)
        dataset (DMatrix):
    Returns:
        dict:
            key: feature name (from feature_list)
            value:
                - numerical: sorted np.array of unique split thresholds
                - categorical: sorted np.array of unique categories seen in splits
    """
    # ['c', 'float']
    feature_type_list = dataset.feature_types
    split_indices_categorical = [
        i for i, feature_type in enumerate(feature_type_list) if feature_type == "c"
    ]
    split_indices_categorical_set = set(split_indices_categorical)

    # Sets for collecting split info
    split_dict_numerical = defaultdict(set)
    split_dict_categorical = defaultdict(set)

    for tree in trees:
        # numerical
        for i in range(len(tree["left_children"])):
            split_index = tree["split_indices"][i]
            # skip leaf / categorical
            if (tree["left_children"][i] == -1 and tree["right_children"][i] == -1) or (
                split_index in split_indices_categorical_set
            ):
                continue
            split_condition = tree["split_conditions"][i]
            split_dict_numerical[split_index].add(split_condition)

        # categorical
        for split_index in split_indices_categorical:
            max_cat = int(max(get_data_col(dataset, split_index)))
            split_dict_categorical[split_index] = list(range(max_cat + 1))

    # build output dict
    split_dict = split_dict_numerical | split_dict_categorical
    split_dict = {
        split_index: np.sort(np.array(list(split_condition_set)))
        for split_index, split_condition_set in split_dict.items()
    }

    return split_dict


def get_split_indices(tree):
    """
    Args:
        tree (dict): json tree from booster
    Retunrs:
        set: values of all unique split_indices along axis
    """
    split_indices = set()

    for index in range(len(tree["left_children"])):
        if tree["left_children"][index] == -1 and tree["right_children"][index] == -1:
            continue
        else:
            split_indices.add(tree["split_indices"][index])

    return split_indices


def all_combinations(indices_list):
    """
    Recursively generate all non-empty subsets of a list of indices.
    Args:
        indices_list (list): list of ints representing indices (0-indexing) of features in dataset
    Returns:
        list: list of feature_tuples

    """
    if not indices_list:
        return []
    first, rest = indices_list[0], indices_list[1:]
    subsets_without_first = all_combinations(rest)
    subsets_with_first = [[first] + list(subset) for subset in subsets_without_first]
    # Add the singleton [first]
    result = [[first]] + subsets_with_first + subsets_without_first
    return [tuple(sub) for sub in result]


def get_data_col(dataset, column_index):
    """
    Args:
        dataset (DMatrix):
        column_indcx (int): column of DMatrix we want
    Returns:
        numpy array (N,): column of DMatrix in a 1D numpy array vector format (to avoid that weird .any error)
    """
    data_col = dataset.get_data()[:, column_index]
    # Check for scipy sparse matrix
    if hasattr(data_col, "toarray"):
        data_col = data_col.toarray().flatten()
    return data_col


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
        # Leaves don't have 'split' key
        if "split" in node:
            split_str = node["split"]
            # Add valid splits
            if split_str[0] == "f":
                split_index = int(split_str[1:])
                if split_index >= 0:
                    features.add(split_index)
            # Only recurse if not a leaf
            if "children" in node:
                for child in node["children"]:
                    get_features_used(child, features)

        return features

    # tree_dump returns trees as JSON strings ['{'node_id': 0, 'depth' = 1, etc.}', '{}', etc.]
    tree_dump = model.get_dump(dump_format="json")
    # print(tree_dump) --> 0 indexing (split: f0)
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
        # print(f"Tree {i}: uses features {features_used}")
    return filtered_tree_indices


def get_filtered_model(
    model,
    feature_tuple=None,
    save_to_disk=False,
    output_file_name="new_model.json",
    folder="loaded_models",
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
    original_model_file = get_model_file(model, save_to_disk)
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
    new_model = get_model(original_model_file, save_to_disk, output_file_name, folder)
    return new_model


def get_filtered_model_list(
    model, feature_tuple_list=None, save_to_disk=False, output_file_name_list=None
):
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
            get_filtered_model(model, features_tuple, save_to_disk, output_file_name)
        )

    return output_models


##### CREATING NEW TREES #####
def five_node_tree(
    skew,
    root_split_index,
    root_split_condition,
    root_skew_split_index,
    root_skew_split_condition,
    leaf_val_left,
    leaf_val_right,
    new_id,
):
    """
    Args:
        skew (str): 'left' if left half is depth-2, 'right' if right half is depth-2
        root_split_index (int):
        root_split_condition (float):
        root_skew_split_index (int):
        root_skew_split_condition (float):
        leaf_val_left (float):
        leaf_Val_right (float):
        new_id (int):
    Returns:
        dict: five node skewed tree
            'skew' child of root node is the root of a depth-1 subtree
             other child of root node is a leaf of value 0.0
    """
    # Convert necessary values to floats
    leaf_val_left = float(leaf_val_left)
    leaf_val_right = float(leaf_val_right)
    root_split_condition = float(root_split_condition)
    root_skew_split_condition = float(root_skew_split_condition)

    # Construct new skewed 5-node tree
    if skew == "left":
        new_tree = {
            "base_weights": [
                0.26570892,
                -0.63801795,
                0.0,
                leaf_val_left,
                leaf_val_right,
            ],
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
                root_skew_split_condition,
                0.0,
                leaf_val_left,
                leaf_val_right,
            ],
            "split_indices": [root_split_index, root_skew_split_index, 0, 0, 0],
            "split_type": [0, 0, 0, 0, 0],
            "sum_hessian": [7.0, 6.0, 1.0, 4.0, 2.0],
            "tree_param": {
                "num_deleted": "0",
                "num_feature": "2",
                "num_nodes": "5",
                "size_leaf_vector": "1",
            },
        }

    elif skew == "right":
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
                root_skew_split_condition,
                leaf_val_left,
                leaf_val_right,
            ],
            "split_indices": [root_split_index, 0, root_skew_split_index, 0, 0],
            "split_type": [0, 0, 0, 0, 0],
            "sum_hessian": [7.0, 1.0, 6.0, 4.0, 2.0],
            "tree_param": {
                "num_deleted": "0",
                "num_feature": "2",
                "num_nodes": "5",
                "size_leaf_vector": "1",
            },
        }
    else:
        assert "skew must be 'left' or 'right'!"

    return new_tree


def tree_from_grid(grid, split_values_x, split_values_y, feature_tuple, is_categorical):
    """
    Args:
        grid: 2D numpy array (num_bins_x, num_bins_y) of values (e.g., alphas)
        split_values_x: list of thresholds for feature_x (length = num_bins_x - 1)
        split_values_y: list of thresholds for feature_y (length = num_bins_y - 1)
        feature_tuple: tuple of (feature_x_index, feature_y_index)
        is_categorical (tuple): (bool, bool)
            True if feature_tuple[0] is categorical
            False if numerical

    Returns: A dictionary matching one of the entries in the "trees" list in XGBoost internal format
    """

    # queue way (level traversal, not post-order)
    if True:
        categories = []
        categories_segments = []
        categories_sizes = []
        categories_nodes = []
        is_categorical_x, is_categorical_y = is_categorical

        base_weights = []
        left_children = []
        right_children = []
        split_indices = []
        split_conditions = []
        parents = []
        default_left = []
        split_type = []
        loss_changes = []
        sum_hessian = []

        queue = []
        node_counter = 0
        node_id_map = {}

        # Enqueue root node info: (x_lo, x_hi, y_lo, y_hi, parent)
        queue.append((0, grid.shape[0], 0, grid.shape[1], -1))

        while queue:
            x_lo, x_hi, y_lo, y_hi, parent = queue.pop(0)

            cur_index = node_counter
            node_counter += 1
            node_id_map[(x_lo, x_hi, y_lo, y_hi)] = cur_index

            width = x_hi - x_lo
            height = y_hi - y_lo

            is_leaf = width == 1 and height == 1

            if is_leaf:
                # Leaf node
                base_weights.append(float(grid[x_lo, y_lo]))
                left_children.append(-1)
                right_children.append(-1)
                split_indices.append(0)  # dummy
                split_conditions.append(float(grid[x_lo, y_lo]))
                parents.append(parent)
                default_left.append(0)
                split_type.append(0)
                loss_changes.append(0)
                sum_hessian.append(0)
            else:
                # Internal node: decide axis by larger dimension
                if width >= height:
                    # Split on x axis
                    mid = (x_lo + x_hi) // 2
                    split_indices.append(feature_tuple[0])
                    parents.append(parent)
                    default_left.append(0)
                    base_weights.append(0.0)
                    loss_changes.append(0.0)
                    sum_hessian.append(1.0)

                    if is_categorical_x:
                        split_type.append(1)  # categorical split
                        # Categories going left: bins [x_lo:mid)
                        node_cats = list(range(x_lo, mid))
                        # Extend categories list
                        start = len(categories)
                        categories.extend(node_cats)
                        length = len(node_cats)
                        categories_segments.append(start)
                        categories_sizes.append(length)
                        categories_nodes.append(cur_index)
                        split_conditions.append(1e-45)  # dummy for cat split
                    else:
                        split_type.append(0)  # numeric split
                        # split_values_x length = number of splits, so index at mid-1
                        split_conditions.append(float(split_values_x[mid - 1]))

                    left_children.append(None)
                    right_children.append(None)
                    queue.append((x_lo, mid, y_lo, y_hi, cur_index))
                    queue.append((mid, x_hi, y_lo, y_hi, cur_index))
                else:
                    # Split on y axis
                    mid = (y_lo + y_hi) // 2
                    split_indices.append(feature_tuple[1])
                    parents.append(parent)
                    default_left.append(0)
                    base_weights.append(0.0)
                    loss_changes.append(0.0)
                    sum_hessian.append(1.0)

                    if is_categorical_y:
                        split_type.append(1)  # categorical split
                        node_cats = list(range(y_lo, mid))
                        start = len(categories)
                        categories.extend(node_cats)
                        length = len(node_cats)
                        categories_segments.append(start)
                        categories_sizes.append(length)
                        categories_nodes.append(cur_index)
                        split_conditions.append(1e-45)
                    else:
                        split_type.append(0)
                        split_conditions.append(float(split_values_y[mid - 1]))

                    left_children.append(None)
                    right_children.append(None)
                    queue.append((x_lo, x_hi, y_lo, mid, cur_index))
                    queue.append((x_lo, x_hi, mid, y_hi, cur_index))

        # Fix children indices now that all nodes assigned
        for (x_lo, x_hi, y_lo, y_hi), idx in node_id_map.items():
            width = x_hi - x_lo
            height = y_hi - y_lo
            if width == 1 and height == 1:
                continue  # leaf node has no children
            if width >= height:
                mid = (x_lo + x_hi) // 2
                left_children[idx] = node_id_map[(x_lo, mid, y_lo, y_hi)]
                right_children[idx] = node_id_map[(mid, x_hi, y_lo, y_hi)]
            else:
                mid = (y_lo + y_hi) // 2
                left_children[idx] = node_id_map[(x_lo, x_hi, y_lo, mid)]
                right_children[idx] = node_id_map[(x_lo, x_hi, mid, y_hi)]

        return {
            "base_weights": base_weights,
            "left_children": left_children,
            "right_children": right_children,
            "split_indices": split_indices,
            "split_conditions": split_conditions,
            "parents": parents,
            "default_left": default_left,
            "split_type": split_type,
            "loss_changes": loss_changes,
            "sum_hessian": sum_hessian,
            "categories": categories,
            "categories_segments": categories_segments,
            "categories_sizes": categories_sizes,
            "categories_nodes": categories_nodes,
            "id": 0,
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(max(feature_tuple) + 1),
                "num_nodes": str(len(base_weights)),
                "size_leaf_vector": "1",
            },
        }
    # queue way (level traversal, not post-order)
    if False:
        base_weights = []
        left_children = []
        right_children = []
        split_indices = []
        split_conditions = []
        parents = []
        default_left = []
        split_type = []
        loss_changes = []
        sum_hessian = []

        queue = []
        node_counter = 0
        node_id_map = {}

        # Enqueue root node info: (x_lo, x_hi, y_lo, y_hi, parent)
        queue.append((0, grid.shape[0], 0, grid.shape[1], -1))

        while queue:
            # Dequeue node from front of list
            x_lo, x_hi, y_lo, y_hi, parent = queue.pop(0)

            cur_index = node_counter
            node_counter += 1
            node_id_map[(x_lo, x_hi, y_lo, y_hi)] = cur_index

            width = x_hi - x_lo
            height = y_hi - y_lo

            is_leaf = width == 1 and height == 1

            if is_leaf:
                # Leaf node
                base_weights.append(float(grid[x_lo, y_lo]))
                left_children.append(-1)
                right_children.append(-1)
                split_indices.append(0)
                split_conditions.append(float(grid[x_lo, y_lo]))
                parents.append(parent)
                default_left.append(0)
                split_type.append(0)
                loss_changes.append(0)
                sum_hessian.append(0)
            else:
                # Internal node: decide axis by larger dimension
                if width >= height:
                    mid = (x_lo + x_hi) // 2
                    split_value = split_values_x[mid - 1]
                    base_weights.append(0.0)
                    left_children.append(None)  # placeholder
                    right_children.append(None)  # placeholder
                    split_indices.append(feature_tuple[0])
                    split_conditions.append(float(split_value))
                    parents.append(parent)
                    default_left.append(0)
                    split_type.append(0)
                    loss_changes.append(0.0)
                    sum_hessian.append(1.0)

                    # Enqueue children
                    queue.append((x_lo, mid, y_lo, y_hi, cur_index))
                    queue.append((mid, x_hi, y_lo, y_hi, cur_index))
                else:
                    mid = (y_lo + y_hi) // 2
                    split_value = split_values_y[mid - 1]
                    base_weights.append(0.0)
                    left_children.append(None)
                    right_children.append(None)
                    split_indices.append(feature_tuple[1])
                    split_conditions.append(float(split_value))
                    parents.append(parent)
                    default_left.append(0)
                    split_type.append(0)
                    loss_changes.append(0.0)
                    sum_hessian.append(1.0)

                    queue.append((x_lo, x_hi, y_lo, mid, cur_index))
                    queue.append((x_lo, x_hi, mid, y_hi, cur_index))

        # Fix internal node children indices
        for (x_lo, x_hi, y_lo, y_hi), idx in node_id_map.items():
            width = x_hi - x_lo
            height = y_hi - y_lo
            if width == 1 and height == 1:
                continue  # leaf has no children
            if width >= height:
                mid = (x_lo + x_hi) // 2
                left_children[idx] = node_id_map[(x_lo, mid, y_lo, y_hi)]
                right_children[idx] = node_id_map[(mid, x_hi, y_lo, y_hi)]
            else:
                mid = (y_lo + y_hi) // 2
                left_children[idx] = node_id_map[(x_lo, x_hi, y_lo, mid)]
                right_children[idx] = node_id_map[(x_lo, x_hi, mid, y_hi)]

        return {
            "base_weights": base_weights,
            "left_children": left_children,
            "right_children": right_children,
            "split_indices": split_indices,
            "split_conditions": split_conditions,
            "parents": parents,
            "default_left": default_left,
            "split_type": split_type,
            "loss_changes": loss_changes,
            "sum_hessian": sum_hessian,
            "categories": [],
            "categories_segments": [],
            "categories_sizes": [],
            "categories_nodes": [],
            "id": 0,
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(max(feature_tuple) + 1),
                "num_nodes": str(len(base_weights)),
                "size_leaf_vector": "1",
            },
        }


def tree_from_vector(vector, split_condition_vector, feature_index, is_categorical):
    """
    Args:
        vector (1D numpy array) (N,): should be vector_alphas (containing final values per slice)
        split_condition_vector (1D numpy array) (N + 1,):
        feature_index (int):
        is_categorical (bool): True if is_categorical
    Returns:
        dict: XGBoost tree (level traversal)
    """
    if True:
        base_weights = []
        left_children = []
        right_children = []
        split_indices = []
        split_conditions = []
        parents = []
        default_left = []
        split_type = []
        loss_changes = []
        sum_hessian = []

        categories = []
        categories_segments = []
        categories_sizes = []
        categories_nodes = []

        queue = []
        node_counter = 0
        node_id_map = {}

        # Enqueue root node (start index, end index, parent)
        queue.append((0, len(vector), -1))

        while queue:
            start, end, parent_index = queue.pop(0)

            cur_index = node_counter
            node_counter += 1
            node_id_map[(start, end)] = cur_index

            is_leaf = (end - start) == 1

            if is_leaf:
                # Leaf node
                base_weights.append(float(vector[start]))
                left_children.append(-1)
                right_children.append(-1)
                split_indices.append(0)  # dummy
                split_conditions.append(float(vector[start]))
                parents.append(parent_index)
                default_left.append(0)
                split_type.append(0)
                loss_changes.append(0.0)
                sum_hessian.append(1.0)
            else:
                mid = (start + end) // 2
                base_weights.append(0.0)
                left_children.append(None)
                right_children.append(None)
                split_indices.append(feature_index)
                parents.append(parent_index)
                default_left.append(1)
                loss_changes.append(0.0)
                sum_hessian.append(1.0)

                if is_categorical:
                    # Categorical split
                    split_type.append(1)
                    # Categories going left: bins [start:mid)
                    node_cats = list(range(start, mid))
                    start_cat_idx = len(categories)
                    categories.extend(node_cats)
                    categories_segments.append(start_cat_idx)
                    categories_sizes.append(len(node_cats))
                    categories_nodes.append(cur_index)

                    split_conditions.append(1e-45)
                else:
                    # Numerical split
                    split_type.append(0)
                    split_conditions.append(float(split_condition_vector[mid - 1]))

                # Enqueue children
                queue.append((start, mid, cur_index))
                queue.append((mid, end, cur_index))

        # Fix children references
        for (start, end), idx in node_id_map.items():
            if (end - start) == 1:
                continue
            mid = (start + end) // 2
            left_children[idx] = node_id_map[(start, mid)]
            right_children[idx] = node_id_map[(mid, end)]

        return {
            "base_weights": base_weights,
            "left_children": left_children,
            "right_children": right_children,
            "split_indices": split_indices,
            "split_conditions": split_conditions,
            "parents": parents,
            "default_left": default_left,
            "split_type": split_type,
            "loss_changes": loss_changes,
            "sum_hessian": sum_hessian,
            "categories": categories,
            "categories_segments": categories_segments,
            "categories_sizes": categories_sizes,
            "categories_nodes": categories_nodes,
            "id": 0,
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(feature_index + 1),
                "num_nodes": str(len(base_weights)),
                "size_leaf_vector": "1",
            },
        }


##### DEPTH-2 TREE PURIFICATION HELPER FUNCTIONS #####
def split_tree(tree):
    """
    Args:
        tree (dictionary): 7-node, depth-2 tree in model_file
    Returns:
        list: tree_left, tree_right (each are dicts)
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
    tree_left = five_node_tree(
        "left",
        root_split_index,
        root_split_condition,
        root_left_split_index,
        root_left_split_condition,
        A_val,
        B_val,
        -1,
    )
    tree_right = five_node_tree(
        "right",
        root_split_index,
        root_split_condition,
        root_right_split_index,
        root_right_split_condition,
        C_val,
        D_val,
        -1,
    )
    return [tree_left, tree_right]


##### PURIFICATION #####
def purify_two_features(
    submodel, dataset, split_conditions_dict, feature_tuple, epsilon=1e-1, max_iter=10
):
    """
    Args:
        submodel (Booster): XGBoost model
        dataset (DMatrix):
        split_conditions_dict (dict):
        feature_tuple (tuple):
        epsilon (float): if change is less than epsilon, END EARLY
        max_iter (int): max number of iterations
    Returns:
        Tuple:
            Tuple[0]: tuple of 2 numpy 1D arrays (alpha vectors for x1, x2)
            Tuple[1]: tree (dictionary) that represents the alpha grid
    """
    feature_type_list = dataset.feature_types
    is_categorical = (
        feature_type_list[feature_tuple[0]] == "c",
        feature_type_list[feature_tuple[1]] == "c",
    )

    num_bins = [0, 0]  # (B,)
    binned_indices_vectors = [None, None]  # (N, 1)
    data_cols = (
        get_data_col(dataset, feature_tuple[0]),
        get_data_col(dataset, feature_tuple[1]),
    )

    ##### BUILD GRIDS #####
    for i in range(2):
        split_condition_vector = split_conditions_dict[feature_tuple[i]]
        # Categorical
        if is_categorical[i]:
            num_bins[i] = len(split_condition_vector)
            binned_indices_vectors[i] = np.array([int(val) for val in data_cols[i]])

        # Numerical
        else:
            num_bins[i] = len(split_condition_vector) + 1
            binned_indices_vectors[i] = np.digitize(
                data_cols[i], split_condition_vector
            )

    predictions = submodel.predict(dataset)  # (N,)
    grid_alphas = np.zeros(tuple(num_bins))  # (Bx, By)

    # initialize lower order vectors
    vector_x = np.zeros(num_bins[0])
    vector_y = np.zeros(num_bins[1])

    ##### PURIFY ALONG EACH AXIS UNTIL CONVERGENCE #####
    def get_mean_vector(current_vals, binned_indices, num_bins):
        """
        Args:
            current_vals (N x 1) (1D numpy array): current prediction value (after subtracting corresopnding alpha) per point in dataset
            binned_indices (Bi x 1) (1D numpy array): binned values (split values)
            num_bins (int): number of bins

        Returns:
            1D numpy array: the mean of current_values for each unique bin index (weighted avg per bin)
        """
        sum_vector = np.zeros(num_bins)
        count_vector = np.zeros(num_bins)
        np.add.at(sum_vector, binned_indices, current_vals)
        np.add.at(count_vector, binned_indices, 1)
        # Avoid division by zero
        mean_vector = np.zeros(num_bins)
        nonzero = count_vector > 0
        mean_vector[nonzero] = sum_vector[nonzero] / count_vector[nonzero]
        return mean_vector

    for i in range(max_iter):
        prev_grid_alphas = grid_alphas.copy()

        # integrate over x-axis
        current_prediction_vector = (
            grid_alphas[binned_indices_vectors[0], binned_indices_vectors[1]]
            + predictions
        )
        # ^^ (N x 1) --> Each element in vector is (the original prediction for that point using the original model, plus the correction (mean-centering) prediction from grid_alphas
        row_means = get_mean_vector(
            current_prediction_vector, binned_indices_vectors[1], num_bins[1]
        )
        for j in range(num_bins[1]):
            grid_alphas[:, j] -= row_means[j]
        vector_y += row_means

        # integrate over y-axis
        current_prediction_vector = (
            grid_alphas[binned_indices_vectors[0], binned_indices_vectors[1]]
            + predictions
        )
        col_means = get_mean_vector(
            current_prediction_vector, binned_indices_vectors[0], num_bins[0]
        )
        for i in range(num_bins[0]):
            grid_alphas[i, :] -= col_means[i]
        vector_x += col_means

        # convergence check --> maybe do row_means and col_means < epsilon???
        diff = np.abs(grid_alphas - prev_grid_alphas).max()
        if diff < epsilon:
            # print("END EARLY")
            break

    ##### CREATE TREE #####
    alpha_tree = tree_from_grid(
        grid_alphas, vector_x, vector_y, feature_tuple, is_categorical
    )

    ##### RETURN LOWER ORDER VECTORS #####
    return ((vector_x, vector_y), alpha_tree)


def purify_one_feature(
    submodel,
    dataset,
    split_conditions_dict,
    alpha_vectors_dict,
    feature_tuple,
):
    """
    Args:
        submodel (Booster object): XGBoost model
        dataset (DMatrix):
        split_conditions_dict (dict):
        feature_tuple (tuple): length one feature tuple

    Returns:
        tuple:
            mean_offset (float):
            alpha_tree (dict): json XGBoost tree
    """
    # Figure out whether feature is categorical or numerical
    feature_type_list = dataset.feature_types
    is_categorical = feature_type_list[feature_tuple[0]] == "c"

    # get unique split values --> these divide up the axes
    split_condition_vector = split_conditions_dict[
        feature_tuple[0]
    ]  # len = Bx (cat), Bx - 1 (num)

    # initialize vector_alphas
    vector_alpha = alpha_vectors_dict[feature_tuple[0]]  # (Bx x 1)

    # get vector_predictions (prediction values from submodel)
    data_col = get_data_col(dataset, feature_tuple[0])

    # categorical
    if is_categorical:
        binned_indices = np.array([int(val) for val in data_col])
    # numerical
    else:
        binned_indices = np.digitize(data_col, split_condition_vector)

    predictions = submodel.predict(dataset)

    # get mean prediction
    current_vals = np.array(vector_alpha[binned_indices] + predictions)
    mean_offset = 0.0
    mean_offset = current_vals.mean()

    # construct alpha tree
    vector_alpha -= mean_offset
    alpha_tree = tree_from_vector(
        vector_alpha, split_condition_vector, feature_tuple[0], is_categorical
    )

    return mean_offset, alpha_tree


def purify_2D(
    model,
    dataset,
    save_to_disk=True,
    input_file_name="original_model.json",
    output_file_name="new_model.json",
    output_folder="loaded_models",
):
    """
    Args:
        model (Booster): max_depth = 2
        dataset (DMatrix): set of points (x-vals)
        save_to_disk (bool): True --> saves model to disk. False --> uses in-memory stream
        output_file_name (string):
        output_folder (string):

    Returns:
        Booster: new model that is the fANOVA Decomposition of model
            Same predictions as model
            Mean along each axis is 0
    """
    # Get all trees indices with interaction
    feature_tuples_main_effect = []
    feature_tuples_interaction = []
    feature_list = list(range(dataset.num_col()))

    all_feature_combinations = all_combinations(feature_list)
    for feature_tuple in all_feature_combinations:
        if len(feature_tuple) == 1:
            feature_tuples_main_effect.append(feature_tuple)
        else:
            feature_tuples_interaction.append(feature_tuple)

    tree_indices_interaction = set()
    for feature_tuple in feature_tuples_interaction:
        tree_indices_interaction |= get_filtered_tree_indices(model, feature_tuple)

    ##### SEPARATE TREES INTO BIAS, 1-FEATURE, 2-FEATURE INTERACTION #####
    model_file = get_model_file(model, True, input_file_name, output_folder)
    original_base_score = float(
        model_file["learner"]["learner_model_param"]["base_score"]
    )

    tree_list_all = model_file["learner"]["gradient_booster"]["model"]["trees"]
    bias_tree_vals = []
    tree_list_one_feature = []
    tree_list_two_features = []

    # Append trees to appropriate lists
    for i, tree in enumerate(tree_list_all):
        # 0-feature (depth-0, 1-node) tree
        if len(tree["base_weights"]) == 1:
            bias_tree_vals.append(tree["base_weights"][0])
        # interaction tree (2-3 features)
        elif i in tree_indices_interaction:
            # Split up 7-node f(x_i, x_j, x_k) trees into two, 5-node f(x_i, x_j) trees
            if (
                int(tree["tree_param"]["num_nodes"]) == 7
                and len(get_split_indices(tree)) == 3
            ):
                new_trees = split_tree(tree)
                tree_list_two_features.extend(new_trees)
            # 5-node 2-feature or 7-node 2-feature f(x_i, x_j, x_j)
            else:
                tree_list_two_features.append(tree)
        # main effect tree (1 feature)
        else:
            tree_list_one_feature.append(tree)

    # make new model consistenting of new 1-feature and 2-feature trees
    model_file["learner"]["gradient_booster"]["model"]["trees"] = (
        tree_list_one_feature + tree_list_two_features
    )
    updated_model = update_metadata(model_file, "0.0", False)
    alpha_tree_list = []

    # get bins for each feature
    split_conditions_dict = get_split_conditions(
        tree_list_one_feature + tree_list_two_features, dataset
    )
    # split_conditions_dict = get_split_conditions(updated_model, dataset)
    alpha_vectors_dict = {}
    for feature_index, feature_type in zip(feature_list, dataset.feature_types):
        if feature_type == "c":
            num_categories = len(split_conditions_dict[feature_index])
            alpha_vectors_dict[feature_index] = np.zeros(num_categories)
        else:
            # print(split_conditions_dict)
            num_splits = len(split_conditions_dict[feature_index])
            alpha_vectors_dict[feature_index] = np.zeros(num_splits + 1)

    ##### PURIFY EACH f(x_i, x_j) TREE #####
    for feature_tuple in feature_tuples_interaction:
        submodel = get_filtered_model(updated_model, feature_tuple, False)
        alpha_vectors, alpha_tree_two = purify_two_features(
            submodel, dataset, split_conditions_dict, feature_tuple
        )

        # POSSIBLE ERROR: check if feature_tuple corresponds to correct alpha vector
        alpha_tree_list.append(alpha_tree_two)
        for feature, alpha_vector in zip(feature_tuple, alpha_vectors):
            alpha_vectors_dict[feature] += alpha_vector

    ##### PURIFY EACH f(x1), f(x2), TREE #####
    for feature_tuple in feature_tuples_main_effect:
        submodel = get_filtered_model(updated_model, feature_tuple, False)
        mean, alpha_tree_one = purify_one_feature(
            submodel, dataset, split_conditions_dict, alpha_vectors_dict, feature_tuple
        )

        alpha_tree_list.append(alpha_tree_one)

    ##### ADD 0-feature trees to bias #####
    new_base_score = original_base_score + mean
    for bias_val in bias_tree_vals:
        new_base_score += bias_val

    ##### UPDATE TREES #####
    model_file["learner"]["gradient_booster"]["model"]["trees"] = (
        tree_list_two_features + tree_list_one_feature + alpha_tree_list
    )

    ##### UPDATE MODEL METADATA #####
    new_model = update_metadata(
        model_file,
        str(float(new_base_score)),
        save_to_disk,
        output_file_name,
        output_folder,
    )

    ##### SAVE AND RETURN #####
    return new_model


def fANOVA_2D(
    use_cached,
    prefix="",
    model=None,
    dataset=None,
    save_to_disk=True,
    output_folder="loaded_models",
):
    """
    Args:
        use_cached (bool): True if purifed submodels already exist
        model (Booster): model from model_file (max_depth = 2)
        dataset (DMatrix):
        save_to_disk (bool):
        prefix (str): tag for the names of all submodels
        output_folder (str):
    Returns:
        fANOVA Result Object:
            original_model
            purified_model
            components (dict):
                key: feature_tuple
                val: Booster object
            bias (float)
    """

    def load_cached_fANOVA_models(prefix, output_folder="loaded_models"):
        """
        Loads the original model, purified model, component models, and bias from disk.

        Assumes files are named:
        - <prefix>_original_model.json       <- original model
        - <prefix>_purified_model.json       <-  purified model
        - <prefix>_component_(i,).json       <- component models

        Args:
            file_path_prefix (str): prefix used to save the models
            output_folder (str): folder containing the saved models

        Returns:
            original_model (xgb.Booster)
            purified_model (xgb.Booster)
            model_dict (dict): keys = feature index tuples, values = Booster models
            bias (float)
        """
        # Load original model
        original_path = os.path.join(output_folder, f"{prefix}_original_model.json")
        original_model = xgb.Booster()
        original_model.load_model(original_path)

        # Load purified model
        purified_path = os.path.join(output_folder, f"{prefix}_purified_model.json")
        purified_model = xgb.Booster()
        purified_model.load_model(purified_path)

        # Extract bias
        with open(purified_path, "r") as f:
            model_file = json.load(f)
        bias = float(model_file["learner"]["learner_model_param"]["base_score"])

        # --- Load component models ---
        model_dict = {}
        for fname in os.listdir(output_folder):
            if fname.startswith(f"{prefix}_component_") and fname.endswith(".json"):
                # remove prefix and ".json"
                subset_str = fname[len(f"{prefix}_component_") : -5]
                subset = eval(subset_str)  # safely turn "(0, 1)" into tuple
                component_path = os.path.join(output_folder, fname)
                component_model = xgb.Booster()
                component_model.load_model(component_path)
                model_dict[subset] = component_model

        return original_model, purified_model, model_dict, bias

    if use_cached:
        original_model, purified_model, model_dict, bias = load_cached_fANOVA_models(
            prefix, output_folder="loaded_models"
        )
        return fANOVA_Result(original_model, purified_model, model_dict, bias)

    else:
        # copy original model (for reference)
        original_model = model.copy()

        # Get all features
        num_features = dataset.num_col()
        feature_indices = list(range(num_features))

        # Purify Model
        purified_model = purify_2D(
            model,
            dataset,
            save_to_disk,
            f"{prefix}_original_model.json",
            f"{prefix}_purified_model.json",
            output_folder,
        )
        purified_model_file = get_model_file(
            purified_model,
            save_to_disk,
            f"{prefix}_purified_model.json",
            output_folder,
        )

        # Get Bias
        bias = float(
            purified_model_file["learner"]["learner_model_param"]["base_score"]
        )

        # Filter Model into submodels
        all_nonempty_subsets = all_combinations(feature_indices)
        filtered_model_list = get_filtered_model_list(
            purified_model,
            all_nonempty_subsets,
            save_to_disk,
            [prefix + "_component_" + str(tup) for tup in all_nonempty_subsets],
        )

        # Add submodels to submodel_dict
        submodel_dict = {}

        for subset, submodel in zip(all_nonempty_subsets, filtered_model_list):
            # Reset bias to 0 (don't want to overcount)
            submodel.set_param({"base_score": 0.0})
            submodel_file = get_model_file(
                submodel, save_to_disk, prefix + "_component_" + str(subset)
            )
            submodel_file["learner"]["learner_model_param"]["base_score"] = "0.0"
            submodel = get_model(
                submodel_file, save_to_disk, prefix + "_component_" + str(subset)
            )

            submodel_dict[subset] = submodel

        return fANOVA_Result(original_model, purified_model, submodel_dict, bias)


if __name__ == "__main__":
    # lei's example
    if False:
        np.random.seed(42)
        n = 50000
        grades = np.random.choice([1, 2, 3, 4], size=n)
        ltv = np.random.normal(loc=130, scale=15, size=n)
        error = np.random.normal(loc=0, scale=0.5, size=n)
        loss = []
        for g, l, e in zip(grades, ltv, error):
            if g == 1:
                val = max(10 + 0 * l + e, 0)
            elif g == 2:
                val = max(12 - 0.2 * (l - 130) + e, 0)
            elif g == 3:
                val = max(15 - 0.3 * (l - 130) + e, 0)
            elif g == 4:
                val = max(20 - 0.6 * (l - 130) + e, 0)
            loss.append(val)

        df = pd.DataFrame({"grade": grades, "ltv": ltv, "loss": loss})
        # print(df)
        X = df[["grade", "ltv"]].values
        # print(X)
        y = df["loss"].values

        # my stuff
        df = pd.DataFrame(
            {
                "grade": pd.Series(grades, dtype="category"),  # or
                # "grade": pd.Categorical(grades),
                "ltv": ltv,
                "loss": loss,
            }
        )
        dtrain = xgb.DMatrix(
            df[["grade", "ltv"]], label=df["loss"], enable_categorical=True
        )

        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",  # fast splits
            "max_depth": 2,
        }
        model = xgb.train(params, dtrain, num_boost_round=50)

        purified_model = purify_2D(model, dtrain)
        print(model.predict(dtrain)[:5])
        print(purified_model.predict(dtrain)[:5])

    # Synthetic data binary
    if True:
        N = 500
        x2 = np.random.rand(N)
        x1 = np.random.choice(["A", "B"], size=N)
        y = np.where(x1 == "A", 2 * x2, x2**2)

        # Prepare DataFrame
        df = pd.DataFrame({"x1": x1, "x2": x2})
        df["x1"] = df["x1"].astype("category")

        # Build DMatrix
        dtrain = xgb.DMatrix(df, label=y, enable_categorical=True)

        # Training parameters
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": 2,
        }
        model = xgb.train(params, dtrain, num_boost_round=1)

        # Predict
        y_pred = model.predict(dtrain)
        # model_file = get_model_file(model, True, "XXX".json")

        new_model = purify_2D(
            model, dtrain, True, "original_model.json", "new_model.json"
        )
        print(model.predict(dtrain)[:10])
        print(model.predict(dtrain)[:10])

    # Synthetic Data multiclass
    if False:
        # === A: Synthetic feature generation with 5 categories ===
        N = 100
        x2 = np.random.rand(N)
        x1 = np.random.choice(list("ABCDE"), size=N)

        # piecewise target: different functions for each A–E label
        y = np.select(
            [
                x1 == "A",
                x1 == "B",
                x1 == "C",
                x1 == "D",
                x1 == "E",
            ],
            [
                2.0 * x2,
                x2**2,
                -0.5 * x2 + 1.0,
                np.sin(2 * np.pi * x2),
                0.3 * (x2**3),
            ],
            default=np.nan,
        )

        # === B: Build DataFrame, mark x1 as categorical ===
        df = pd.DataFrame({"x1": x1, "x2": x2})
        df["x1"] = df["x1"].astype("category")

        # === C: Create DMatrix with categorical support enabled ===
        dtrain = xgb.DMatrix(df, label=y, enable_categorical=True)

        # === D: Train with histogram method and control one-hot threshold ===
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            # ensures categories ≤ 5 are one-hot split (helps interpretability)
            "max_depth": 2,
        }
        model = xgb.train(params, dtrain, num_boost_round=10)

        # === E: Predict & inspect ===
        y_pred = model.predict(dtrain)
        # print(get_data_col(dtrain, 0))
        new_model = purify_2D(model, dtrain, df)

        # print(model.predict(dtrain)[:5])
        # print(new_model.predict(dtrain)[:5])

    # perplexity LOL
    if False:
        # Prepare data
        df["grade"] = df["grade"].astype("category")
        X = df[["grade", "ltv"]]
        y = df["loss"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create DMatrix (with enable_categorical!) for both train and test
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Set parameters for regression with categorical support
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "enable_categorical": True,
            "max_depth": 5,
            "seed": 42,
        }

        # Train Booster
        model = xgb.train(params, dtrain, num_boost_round=100)
        model_file = get_model_file(model, True, "hello.json")
