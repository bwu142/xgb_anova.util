# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import json

##########################################
#############  y = 10*x1 + 2 #############
##########################################


def generate_simple_data(num_samples, y_val, test_size=0.3):
    """"""
    pass


####### GENERATE DATA #######

# Generate 100 random samples sampled from normal distribution
np.random.seed(42)
x1 = np.random.randn(100)
y1 = 10 * x1 + 10  # true function (no noise)

# Split data into training and testing sets
X1 = pd.DataFrame({"x1": x1})  # create tabular format of x1
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.3, random_state=42
)

###### FIT XGBOOST REGRESSOR ######

# 3 trees of depth 1
model = xgb.XGBRegressor(
    n_estimators=3,  # 3 tree
    max_depth=1,  # depth of 1
    learning_rate=1.0,
    objective="reg:squarederror",
    random_state=42,
    base_score=0.8,
)
model.fit(X1_train, y1_train)

model.save_model("model_3trees.json")  # save model_3trees


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


def get_filtered_tree_list(model, features_needed=None):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    features_needed: set of features (represented as strings, e.g. 'x1')

    Returns list of all trees (represented as dictionaries) from model that contain the intersection of all features_needed as splits
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
    trees_json = model.get_booster().getdump(dump_format="json")
    filtered_trees = []

    # Loop through each tree, and append to filtered_trees if tree contains features
    for tree_str in trees_json:
        tree = json.loads(tree_str)
        features_used = get_features_used(tree)
        if features_needed.issubset(features_used):
            filtered_trees.append(tree)

    # print(f"number of original trees: {len(trees_json)}\nnumber of filtered trees: {len(filtered_trees)}")

    return filtered_trees


def get_leaf_val(tree, sample):
    """
    tree: tree dictionary
    sample: x-vector

    Returns leaf value obtained by traversing the singlular 'tree' with input 'sample'
    """
    node = tree
    while "leaf" not in node:
        feature = node["split"]
        decider = node["split_condition"]
        if sample[feature] < decider:
            node = node["children"][0]
        else:
            node = node["children"][1]
    return node["leaf"]


def sum_specific_outputs(test_set, trees, bias):
    """
    test_set: Pandas Dataframe of x-values (testing input set)
    trees: List of (filtered) tree dictionaries
    bias: initial bias from boosted_trees (starting estimated value)

    predict y-values from corresponding x-values in input test_set BY SUMMING output values from TREES in 'trees'
    """
    partial_predictions = []
    for _, row in test_set.iterrows():
        prediction = bias  # bias?
        for tree in trees:
            prediction += get_leaf_val(tree, row)
        partial_predictions.append(prediction)
    return partial_predictions


###### PLOTTING HELPER FUNCTIONS ######
def scatter_plot():
    """"""
    pass


####### TESTING #######
# Note: unsure about output_margin = True

# Note: unsure about output_margin here
x = X1_test["x1"].values  # test input x-values
y_true = y1_test  # True y values

# y value predicted from entire default tree
y_pred = np.array(model.predict(X1_test, output_margin=True))

# predicted y value from manually summing predictions from each tree
all_trees_list = get_all_tree_list(model)
bias = get_bias(model)
print(bias)
y_pred_alt = np.array(sum_specific_outputs(X1_test, all_trees_list, bias))

# PLOT
plt.scatter(x, y_true, label="True y", color="green", marker="o")
plt.scatter(
    x, y_pred, label="Default boosted tree y-prediction", color="blue", marker="."
)
plt.scatter(
    x, y_pred_alt, label="Manual tree sum y-prediction", color="red", marker="."
)
plt.xlabel("x1")
plt.ylabel("y")
plt.legend()
plt.show()

# Optional Test: print out second and third y-vals to see if identical for a single test point x
