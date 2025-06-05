# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import json


###### TREE HELPER FUNCTIONS ######
def get_bias(model):
    bias = model.get_booster().attr("base_score")
    if bias is not None:
        return float(bias)
    return float(model.base_score)  # Fallback


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
        if features_needed.issubset(features_used):
            filtered_trees[0].append((i, i + 1))
            filtered_trees[1].append(tree)

    # print(f"number of original trees: {len(trees_json)}\nnumber of filtered trees: {len(filtered_trees[1])}")

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
    ranges: list of ranges of relevant trees (output of get_filtered_tree_list_tuple_ranges)


    Returns: prediction (1D numpy array) using relevant trees in 'ranges' from model
    """
    # list of ranges
    range_list = get_filtered_tree_list_ranges_from_tuple(model, features_tuple)
    # 2D numpy array of predictions
    relevant_predictions = get_split_tree_predictions(model, test_set, range_list)
    # predicted y vector
    y_pred = sum_split_tree_predictions(model, relevant_predictions)
    return y_pred


if __name__ == "__main__":

    #####################################
    ########### y = 10*x1 + 2 ###########
    #####################################

    ####### GENERATE DATA #######

    # Generate 100 random vars sampled from normal distribution
    np.random.seed(42)
    x1 = np.random.randn(100)
    y = 10 * x1 + 2  # true function (no noise)

    # Split data into training (70%) and testing sets (30%)
    X1 = pd.DataFrame({"x1": x1})  # create tabular format of x1
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y, test_size=0.3, random_state=42
    )

    ###### FIT XGBOOST REGRESSOR ######

    # Fit XGBoost regressor with 3 trees of depth 1
    model1 = xgb.XGBRegressor(
        n_estimators=3,  # 3 trees
        max_depth=2,  # depth of 1
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.8,
    )
    model1.fit(X1_train, y1_train)
    # model1.save_model("model1_3trees.json")  # save model1_3tree

    ####### TESTING #######
    x1 = X1_test["x1"].values  # test input x-values
    y_true = y1_test  # True y values

    # y value predicted from entire default tree
    y_pred = model1.predict(X1_test, output_margin=True)

    # y value predicted from summing individual trees
    y_pred_sum = predict(model1, (0,), X1_test)

    plt.xlabel("x1")
    plt.ylabel("y")
    plt.scatter(x1, y_true, label="True y", color="green", marker="x")
    plt.scatter(
        x1, y_pred, label="Default boosted tree y-prediction", color="red", marker="*"
    )
    plt.scatter(
        x1, y_pred_sum, label="Manual tree SUM y-prediction", color="pink", marker="."
    )
    plt.show()

    print(get_filtered_tree_list_ranges_from_tuple(model1, (0,)))

    print("helo")
