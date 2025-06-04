# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import json

#####################################
########### y = 10*x1 + 2 ###########
#####################################


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


def get_split_tree_predictions(model, test_set):
    """
    model: xgb regressor (model = xgb.XGBRegressor())
    test_set: Pandas Dataframe of x-values (testing input set)

    Returns 2D numpy array of predictions, each prediction from one tree
        col: prediction vector from one tree
        row: test point
    """
    num_trees = model.n_estimators
    arr = np.zeros(
        (test_set.shape[0], num_trees)
    )  # arr is of dimenision (num test points x num trees)

    # put prediction values vectors into columns of arr
    for i in range(num_trees):
        new_col = np.array(
            model.predict(test_set, iteration_range=(i, i + 1), output_margin=False)
        )
        arr[:, i] = new_col

    return arr


def sum_split_tree_predictions(model, predictions):
    """
    predictions: 2D numpy array of predictions that get_split_tree_predictions returns

    Returns the 1D numpy vector sum of predictions from each tree, with regards to bias
    """
    vector_sum = np.sum(predictions, axis=1)  # sum columns

    bias = get_bias(model)
    num_trees = model.n_estimators

    vector_sum -= bias * (num_trees - 1)

    return vector_sum


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
    max_depth=1,  # depth of 1
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
predictions = get_split_tree_predictions(model1, X1_test)
y_pred_sum = sum_split_tree_predictions(model1, predictions)

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


########## OLD STUFF ##########
# model1_t1 = model1.predict(X1_test, iteration_range=(0, 1), output_margin=False)
# model1_t2 = model1.predict(X1_test, iteration_range=(1, 2), output_margin=False)
# model1_t3 = model1.predict(X1_test, iteration_range=(2, 3), output_margin=False)
# y_tree1 = np.array(model1_t1)
# print(y_tree1)
# y_tree2 = np.array(model1_t2)
# y_tree3 = np.array(model1_t3)
# y1_sum = y_tree1 + y_tree2 + y_tree3 - 2 * bias
# print(y1_sum[:10])
# print(y1_3[:10])
# bias = get_bias(model1)

# EACH INDIVIDUAL TREE (INCLUDES BIAS EACH TIME):
# plt.scatter(x1, y_tree1, label="tree 1 predicted y", color="blue", marker=".")
# plt.scatter(x1, y_tree2, label="tree 2 predicted y", color="brown", marker=".")
# plt.scatter(x1, y_tree3, label="tree 3 predicted y", color="purple", marker=".")
