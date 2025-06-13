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
def get_bias(model, output_file_name="model_bias.json", folder="loaded_models"):
    """
    model: xgb.train(...)
    Returns bias (even if unspecified in model params)
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, output_file_name)
    model.save_model(file_path)

    with open(file_path, "r") as f:
        bias_model_file = json.load(f)
    return float(bias_model_file["learner"]["learner_model_param"]["base_score"])


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
            features.add(int(node["split"][1]) - 1)
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


##### FILTERING AND SAVING TREES #####


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
    tree_indices = get_filtered_tree_indices(model, feature_tuple)
    # Sanity check
    if not output_file_name.endswith(".json"):
        output_file_name += ".json"

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    original_model_path = os.path.join(folder, "original_model.json")
    output_path = os.path.join(folder, output_file_name)

    # Save model as json file in the specified folder
    model.save_model(original_model_path)

    # Load in json file into Python Dictionary for editing purposes (manipulate json file directly)
    with open(original_model_path, "r") as file:
        original_model = json.load(file)

    ##### EDIT TREES #####
    new_trees = []
    id_count = 0
    for i in tree_indices:
        new_trees.append(
            original_model["learner"]["gradient_booster"]["model"]["trees"][i]
        )
        new_trees[id_count]["id"] = id_count
        id_count += 1
    original_model["learner"]["gradient_booster"]["model"]["trees"] = new_trees

    ##### EDIT OTHER FILE PARAMS #####
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

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, output_file_name)

    ##### SAVE #####
    with open(output_path, "w") as file:
        json.dump(original_model, file)

    new_model = xgb.Booster()
    new_model.load_model(output_path)
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
    # Sanity Check
    if not output_file_name.endswith(".json"):
        output_name += ".json"

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    original_model_path = os.path.join(folder, "original_model.json")
    output_path = os.path.join(folder, output_file_name)

    # Save model as json file in the specified folder
    model.save_model(original_model_path)

    # Load in json file into Python Dictionary for editing purposes (manipulate json file directly)
    with open(original_model_path, "r") as file:
        original_model_file = json.load(file)

    def create_new_tree(model_file, leaf_val, new_id):
        """
        model_file: loaded in file
            with open("original_model.json", "r") as f:
                original_model_file = json.load(f)
        leaf_val: float
        new_id: int

        Returns a tree in the structure of the trees within model_file
            model_file must have AT LEAST ONE TREE
        """
        leaf_val = float(leaf_val)
        # Get arbitrary tree structure
        new_tree = model_file["learner"]["gradient_booster"]["model"]["trees"][0].copy()
        new_tree["base_weights"] = [
            leaf_val for _ in range(len(new_tree["base_weights"]))
        ]
        new_tree["split_conditions"] = [
            leaf_val for _ in range(len(new_tree["base_weights"]))
        ]
        new_tree["id"] = new_id
        return new_tree

    ##### ADD TREES #####
    original_num_trees = int(
        original_model_file["learner"]["gradient_booster"]["model"][
            "gbtree_model_param"
        ]["num_trees"]
    )
    cur_id_num = original_num_trees

    for _ in range(num_new_trees):
        additional_tree = create_new_tree(original_model_file, leaf_val, cur_id_num)
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

    ##### SAVE #####
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, output_file_name)

    ##### SAVE #####
    with open(output_path, "w") as file:
        json.dump(original_model_file, file)

    new_model = xgb.Booster()
    new_model.load_model(output_path)
    return new_model


if __name__ == "__main__":
    # Set seed for reproducibility
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
        num_boost_round=1,  # Equivalent to n_estimators
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=True,
    )

    model.save_model("WHY.json")
    # print(model.get_dump(dump_format="json"))

    print(get_bias(model))
    # print(model.predict(dtest))
    print(len(get_filtered_tree_indices(model, (0,))))
    print(len(get_filtered_tree_indices(model, (1,))))
    print(len(get_filtered_tree_indices(model, (0, 1))))
    print(len(get_filtered_tree_indices(model, ())))

    new_model = filter_save_load(model, (0,))
    print(new_model.predict(dtest))

    the_list = filter_save_load_list(model, [(0,), (1,), (0, 1), ()])
    print(the_list[0].predict(dtest))

    a = save_load_new_trees(model, 4.0, "5", 2)
    print(a.predict(dtest))
