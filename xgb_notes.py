# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import tree_filtering

##### SMALL TREE PREDICTOR #####
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import json
import matplotlib.pyplot as plt


##### PURIFY FIVE NODES #####
def purify_five_nodes(model_file, tree_index, test_data, folder="loaded_models"):
    """
    model_file: loaded in file
            with open("original_model.json", "r") as f:
                original_model_file = json.load(f)
    tree_index: index of the tree we are purifying
    test_data: DMatrix

    Purifies f(x1, x2) with respect to the easier variable
        Modifies and Returns model_file
    """
    trees = model_file["learner"]["gradient_booster"]["model"]["trees"]
    tree = trees[tree_index]

    ##### MODIFY ORIGINAL TREE #####
    depth_one_node_indices = [tree["left_children"][0], tree["right_children"][0]]
    # Index of leaf at depth 1
    depth_one_leaf_index = -1
    # Index of root node of subtree with two leaves
    depth_two_subtree_root_index = -1

    for index in depth_one_node_indices:
        if tree["left_children"][index] == -1 and tree["right_children"][index] == -1:
            depth_one_leaf_index = index
        else:
            depth_two_subtree_root_index = index

    # leaf value of the two new leaves we are adding
    new_leaf_val = tree["base_weights"][depth_one_leaf_index]

    # Add two new leaf nodes
    new_left_leaf_index = len(tree["base_weights"])
    new_right_leaf_index = new_left_leaf_index + 1

    tree["base_weights"].extend([new_leaf_val, new_leaf_val])
    tree["left_children"].extend([-1, -1])
    tree["right_children"].extend([-1, -1])
    tree["split_indices"].extend([0, 0])
    tree["split_conditions"].extend([new_leaf_val, new_leaf_val])

    # Update original leaf node into a split like other subtree
    depth_two_subtree_split_index = tree["split_indices"][depth_two_subtree_root_index]
    depth_two_subtree_split_condition = tree["split_conditions"][
        depth_two_subtree_root_index
    ]

    tree["base_weights"][depth_one_leaf_index] = 0
    tree["left_children"][depth_one_leaf_index] = new_left_leaf_index
    tree["right_children"][depth_one_leaf_index] = new_right_leaf_index
    tree["split_indices"][depth_one_leaf_index] = depth_two_subtree_split_index
    tree["split_conditions"][depth_one_leaf_index] = depth_two_subtree_split_condition

    ##### INTEGRATE OVER AXIS 1: ROOT NODE FEATURE #####

    ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####

    ##### INTEGRATE OVER AXIS 2: OTHER FEATURE #####
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, "original_model.json")

    # Save model as json file in the specified folder
    with open(output_path, "w") as file:
        json.dump("model.json", file)

    new_model = xgb.Booster()
    new_model.load_model(output_path)

    # Get means
    left_leaf_index_one = new_left_leaf_index
    left_leaf_index_two = tree["left_children"][depth_two_subtree_root_index]
    right_leaf_index_one = new_right_leaf_index
    right_leaf_index_two = tree["right_children"][depth_two_subtree_root_index]

    leaf_counts = get_leaf_count(new_model, test_data, tree_index)

    num_left = 0
    sum_left = 0
    num_right = 0
    sum_right = 0

    for leaf_index, count in leaf_counts.items():
        if leaf_index == left_leaf_index_one:
            num_left += count
            sum_left += tree["base_weights"][left_leaf_index_one]
        elif leaf_index == left_leaf_index_two:
            num_left += count
            sum_left += tree["base_weights"][left_leaf_index_two]
        elif leaf_index == right_leaf_index_one:
            num_right += count
            sum_right += tree["base_weights"][right_leaf_index_one]
        else:
            num_right += count
            sum_right += tree["base_weights"][right_leaf_index_one]

    mean_left = sum_left / num_left
    mean_right = sum_right / num_right

    # Subtract means from respective leaves
    tree["base_weights"][left_leaf_index_one] -= mean_left
    tree["split_conditions"][left_leaf_index_one] -= mean_left
    tree["base_weights"][left_leaf_index_two] -= mean_left
    tree["split_conditions"][left_leaf_index_two] -= mean_left
    tree["base_weights"][right_leaf_index_one] -= mean_right
    tree["split_conditions"][right_leaf_index_one] -= mean_right
    tree["base_weights"][right_leaf_index_two] -= mean_right
    tree["split_conditions"][right_leaf_index_two] -= mean_right

    ##### COMPENSATE WITH ADDITIONAL ONE-FEATURE-TREE #####
    cur_id = len(trees)
    additional_tree = create_depth_one_tree(
        model_file, mean_left, cur_id, other_subtree_split_index
    )
    trees.append(additional_tree)
    cur_id += 1
    additional_tree = create_depth_one_tree(
        model_file, mean_right, cur_id, other_subtree_split_index
    )
    trees.append(additional_tree)
    cur_id += 1

    return model_file


def generate_data_and_model(
    n_samples=1000,
    rho=0.5,
    b1=1,
    b2=1,
    b3=1,
    test_size=0.2,
    random_state=42,
    xgb_params=None,
):
    """
    Generate training and testing sets from 3 features (sampled from standard normal distribution) -- currently 1000 samples
    Return training set, testing set, xgb regressor

    """
    # 1. Generate correlated normal variables using Cholesky decomposition
    cov_matrix = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])
    L = np.linalg.cholesky(cov_matrix)
    uncorrelated = np.random.normal(size=(n_samples, 3))
    correlated = uncorrelated @ L.T

    # 2. Create DataFrame with interaction terms
    df = pd.DataFrame(correlated, columns=["x1", "x2", "x3"])
    df["y"] = b1 * df.x1 * df.x3 + b2 * df.x2 * df.x3 + b3 * df.x1 * df.x2

    # 3. Split data into features/target
    X = df[["x1", "x2", "x3"]]
    y = df["y"]

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. Set default XGBoost parameters if none provided
    if xgb_params is None:
        xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "n_estimators": 1,
            "learning_rate": 0.1,
            "random_state": random_state,
        }

    # 6. Train XGBoost model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


if __name__ == "__main__":
    # Generate data and train model
    X_train, X_test, y_train, y_test, model = generate_data_and_model(
        rho=0.3, b1=0.5, b2=1.2, b3=-0.8
    )

    # Make predictions
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Feature importances: {model.feature_importances_}")

    # === Dump all trees as JSON strings ===
    trees_json = model.get_booster().get_dump(dump_format="json")
    print(trees_json)

    # Define subset of features to filter by
    chosen_features = {"x1", "x2", "x3"}  # Example: only keep trees using x3

    ### AND
    def get_features_used(node, features=None):
        """get all features starting from root node of tree. Node is a dictionary"""
        if features is None:
            features = set()
        if "split" in node:
            features.add(node["split"])
            # Don't recurse if leaf
            if "children" in node:
                for child in node["children"]:
                    get_features_used(child, features)
        return features

    # Generate list of trees that contain all the chosen features
    filtered_trees = []
    for tree_str in trees_json:
        tree = json.loads(tree_str)
        features_used = get_features_used(tree)
        if chosen_features.issubset(features_used):
            filtered_trees.append(tree)

    # Generate list of all trees
    all_trees = [json.loads(tree_str) for tree_str in trees_json]

    print(f"Original number of trees: {len(trees_json)}")
    print(f"Number of trees using chosen features: {len(filtered_trees)}")

    print(f"filtered_trees: {filtered_trees}")

    ################### UNSURE ####################
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    # ASSUMMING THERES NO EASY WAY TO CREATE A NEW MODEL WITH FILTERED TREES.
    # ALTERNATIVE: MANUALLY SUMMING UP PREDICTION LEAF VALUES FROM ALL FILTERED TREES THEN PLOTTING SELECTED VAR VS Y

    def get_leaf_val(tree, sample):
        """traverse single tree from given sample & return leaf value"""
        node = tree
        while "leaf" not in node:
            feature = node["split"]
            decider = node["split_condition"]
            if sample[feature] < decider:
                node = node["children"][0]
            else:
                node = node["children"][1]
        return node["leaf"]

    def sum_specific_outputs(test_set, filtered_trees):
        """sum outputs from specific trees for each sample
        test_set: Pandas Dataframe of x-values (testing input set)
        filtered_trees: List of (filtered) tree dictionaries (containing only trees with specified feature(s))

        """
        partial_predictions = []
        for _, row in test_set.iterrows():
            prediction = 0
            for tree in filtered_trees:
                prediction += get_leaf_val(tree, row)
            partial_predictions.append(prediction)
        return partial_predictions

    y_hat = sum_specific_outputs(X_test, all_trees)
    # list vector of predicted y-vals from test set x-vals (testing input set).
    # predicted y from summing leaf vals from trees with specific x value
    y_alt_hat = sum_specific_outputs(X_test, filtered_trees)

    # Get numpy arrays to plot
    x = X_test["x1"].values  # test input x vals
    y = y_test.values  # true y vals
    y_alt_hat = np.array(y_alt_hat)  # predicted y vals using subset of trees
    y_hat = np.array(y_hat)  # predicted y vals using all trees

    # plot
    plt.scatter(x, y, label="True y", alpha=0.7)
    plt.scatter(x, y_alt_hat, label="Predicted y_alt_hat", color="red", marker="x")
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    plt.scatter(x, y_hat, label="Predicted y_hat", alpha=0.7)
    plt.scatter(x, y_alt_hat, label="Predicted y_alt_hat", color="red", marker="x")
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.legend()
    plt.show()


##### TESTING FILTER_AND_SAVE #####

# # "reasonability" check
# filter_and_save(model, "test_model_10_trees_x1", (0,))
# filter_and_save(model, "test_model_10_trees_x2", (1,))
# assert os.path.exists("test_model_10_trees_x2.json"), "File not created!"

# # prediction check
# new_model = xgb.XGBRegressor()
# new_model.load_model("test_model_10_trees_x2.json")
# new_model.load_model("test_model_10_trees_x1.json")

# # After saving
# print(new_model.predict(X_test))

##### CONTOUR PLOT HELPER FUNCTIONS #####


# Optional: Input -- list of variables --> generates all combinations recursively first
def plot_contour_two_vars(model, equation):
    """
    model: XGBRegressor
    equation: string


    """
    x1 = np.arange(0, 100, 0.1)
    x2 = np.arange(0, 100, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    all_pairs = np.column_stack([X1.flatten(), X2.flatten()])
    # convert all pairs to DataFrame structure
    X_test = pd.DataFrame({"x1": all_pairs[:, 0], "x2": all_pairs[:, 1]})
    # get predictions
    z_pred_x1 = tree_filtering.predict(model, (0,), X_test)
    z_pred_x2 = tree_filtering.predict(model, (1,), X_test)
    z_pred_x1x2 = tree_filtering.predict(model, (0, 1), X_test)
    z_pred = model.predict(X_test, output_margin=True)  # vector

    # reshape Z3 vector into 2D numpy array for contour plotting (rows: x2, cols: x1)
    Z_Pred_x1 = z_pred_x1.reshape(len(x2), len(x1))
    Z_Pred_x2 = z_pred_x2.reshape(len(x2), len(x1))
    Z_Pred_x1x2 = z_pred_x1x2.reshape(len(x2), len(x1))
    Z_Pred_Default = z_pred.reshape(len(x2), len(x1))

    # Z_Pred_x1x2
    contour_plot_Z_Pred_x1x2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1x2,
            x=x1,
            y=x2,
            colorscale="sunset",
            contours=dict(
                start=-4, end=10, size=1, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_x1x2.update_layout(
        title=f"Z_Pred_x1x2: {equation}",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    contour_plot_Z_Pred_x1x2.show()


###### TREE ADDITIVITY TESTS #####
def test_additivity_one_tree():
    """Test to see if prediction from summing predictions from individual trees (accounting for bias) equals prediction from original model"""

    # Generate 100 random samples sampled from normal distribution
    np.random.seed(42)
    x1 = np.random.randn(100)
    y = 10 * x1 + 10  # true function (no noise)

    # Split data into training and testing sets
    X = pd.DataFrame({"x1": x1})  # create tabular format of x1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ###### FIT XGBOOST REGRESSOR ######

    # 3 trees of depth 1
    model = xgb.XGBRegressor(
        n_estimators=3,  # 3 tree
        max_depth=1,  # depth of 1
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.5,
    )
    model.fit(X_train, y_train)

    x = X_test["x1"].values  # test input x-values
    y_true = y_test  # True y values
    # y values predicted from entire default tree
    print(X_test)
    y_pred = model.predict(X_test, output_margin=True)
    y_pred_sum = tree_filtering.predict_sum_of_all_trees(model, X_test)

    # print(f"y_pred: {y_pred}")
    # print(f"y_pred_sum: {y_pred_sum}")
    assert np.allclose(np.round(y_pred, 3), np.round(y_pred_sum, 3))

    # # PLOT
    # plt.scatter(x, y_true, label="True y", color="green", marker="o")
    # plt.scatter(
    #     x, y_pred, label="Default boosted tree y-prediction", color="blue", marker="."
    # )
    # plt.scatter(
    #     x, y_pred_sum, label="Manual tree sum y-prediction", color="red", marker="."
    # )
    # plt.xlabel("x1")
    # plt.ylabel("y")
    # plt.legend()
    # plt.show()


###### TREE FILTER TESTS #####
def test_filter_two_vars_1():
    """
    test filtering of two_vars: y = 10 * x1 + 2 * x2
        No intercept
        Tree depth 1
        1000 trees
    """
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 1000)
    x2 = np.random.uniform(0, 100, 1000)
    y = 10 * x1 + 2 * x2  # true function (no noise)

    # Split data into training (70%) and testing sets (30%)
    X = pd.DataFrame({"x1": x1, "x2": x2})  # create tabular format of x1, x2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ###### FIT XGBOOST REGRESSOR ######

    # Fit XGBoost regressor with 1000 trees of depth 1
    model = xgb.XGBRegressor(
        n_estimators=1000,  # 1000 trees
        max_depth=2,  # depth of 1
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.8,
    )
    model.fit(X_train, y_train)

    # # y value predicted from summing individual trees
    # trees_feature_x1 = tree_filtering.get_filtered_tree_list_ranges_from_tuple(
    #     model, (0,)
    # )
    # trees_feature_x2 = tree_filtering.get_filtered_tree_list_ranges_from_tuple(
    #     model, (1,)
    # )
    # print(f"num trees_with_feature_x1: {len(trees_feature_x1)}")
    # print(f"num trees_with_feature_x2: {len(trees_feature_x2)}")
    # print(f"trees_with_feature_x1: {trees_feature_x1}")
    # print(f"trees_with_feature_x2: {trees_feature_x2}")

    y_true = y_test  # True y vals

    # y value predicted from entire default tree
    y_pred = model.predict(X_test, output_margin=True)

    # GENERATE TEST SETS
    x1_ver1 = np.arange(0, 100, 0.1)  # 0.1 step, not including 100
    x2_ver1 = np.random.uniform(0, 100, size=len(x1))
    # X_test1 = pd.DataFrame({"x1": x1, "x2": x2})
    X_test1 = pd.DataFrame({"x1": x1_ver1, "x2": x2_ver1})

    x1_ver2 = np.random.uniform(0, 100, size=len(x1))
    x2_ver2 = np.arange(0, 100, 0.1)
    # X_test2 = pd.DataFrame({"x1": x1, "x2": x2})
    X_test2 = pd.DataFrame({"x1": x1_ver2, "x2": x2_ver2})

    y_pred_x1 = tree_filtering.predict(model, (0,), X_test1)  # 9 trees
    y_pred_x2 = tree_filtering.predict(model, (1,), X_test2)  # 1 tree

    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(
        X_test1["x1"],
        y_pred_x1,
        label="Default boosted tree y-prediction",
        color="red",
        marker="*",
    )
    plt.scatter(
        X_test2["x2"],
        y_pred_x2,
        label="Manual tree SUM y-prediction",
        color="pink",
        marker=".",
    )
    plt.show()
    # assert False


def test_filter_two_vars_2():
    """
    y = 10 * x1 + 2 * x2 + intercept

    Nonzero intercept
    Tree depth 2
    Contour Plot

    Total num trees: 1000
        num x2 trees: 149
        nm x2 trees: 96
        num x1x2 trees: 755
    """
    # Generate Data
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 1000)
    x2 = np.random.uniform(0, 100, 1000)
    intercept = 5
    y = 10 * x1 + 2 * x2 + intercept
    X = pd.DataFrame({"x1": x1, "x2": x2})  # create tabular format of x1, x2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit Regressor
    model = xgb.XGBRegressor(
        n_estimators=1000,  # 1000 trees
        max_depth=2,  # max_depth of 2
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.5,
    )
    model.fit(X_train, y_train)

    # z1 = tree_filtering.get_filtered_tree_list_ranges_from_tuple(model, (0,))
    # z2 = tree_filtering.get_filtered_tree_list_ranges_from_tuple(model, (1,))
    # z3 = tree_filtering.get_filtered_tree_list_ranges_from_tuple(
    #     model,
    #     (0, 1),
    # )
    # print(f"num x1 trees: {len(z1)}")
    # print(f"num x2 trees: {len(z2)}")
    # print(f"num x1x2 trees: {len(z3)}")

    ##### GENERATE PREDICTIONS #####

    # Make DataFrame with all combinations of pairs (x1, x2), each var from [0, 100) by .1 increments
    x1 = np.arange(0, 100, 0.1)  # make x1 vals
    x2 = np.arange(0, 100, 0.1)  # x2 vals
    # both shape (len(x1) * len(x1)) --> to generate all pairs
    X1, X2 = np.meshgrid(x1, x2)
    all_pairs = np.column_stack([X1.flatten(), X2.flatten()])  # generate all pairs
    # convert all pairs to DataFrame structure
    X_test = pd.DataFrame({"x1": all_pairs[:, 0], "x2": all_pairs[:, 1]})

    # get prediction
    z_pred_x1 = tree_filtering.predict(model, (0,), X_test)
    z_pred_x2 = tree_filtering.predict(model, (1,), X_test)
    z_pred_x1x2 = tree_filtering.predict(model, (0, 1), X_test)
    z_pred_Default = model.predict(X_test, output_margin=True)  # vector

    # reshape Z3 vector into 2D numpy array for contour plotting (rows: x2, cols: x1)
    Z_Pred_x1 = z_pred_x1.reshape(len(x2), len(x1))
    Z_Pred_x2 = z_pred_x2.reshape(len(x2), len(x1))
    Z_Pred_x1x2 = z_pred_x1x2.reshape(len(x2), len(x1))
    Z_Pred_Default = z_pred_Default.reshape(len(x2), len(x1))

    ##### PLOT #####

    # Z_Pred_x1
    contour_plot_Z_Pred_x1 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1,
            x=x1,
            y=x2,
            colorscale="blues",
            contours=dict(
                start=0, end=1200, size=50, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )
    contour_plot_Z_Pred_x1.update_layout(
        title="Z_Pred_x1: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    )

    # Z_Pred_x2
    contour_plot_Z_Pred_x2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x2,
            x=x1,
            y=x2,
            colorscale="blues",
            contours=dict(
                start=-50, end=50, size=5, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_x2.update_layout(
        title="Z_Pred_x2: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    )

    # Z_Pred_x1x2
    contour_plot_Z_Pred_x1x2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1x2,
            x=x1,
            y=x2,
            colorscale="sunset",
            contours=dict(
                start=-200, end=200, size=25, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_x1x2.update_layout(
        title="Z_Pred_x1x2: y = 10 * x1 + 2 * x2 + 5 ",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    # # Z_Pred_Default
    contour_plot_Z_Pred_Default = go.Figure(
        data=go.Contour(
            z=Z_Pred_Default,
            x=x1,
            y=x2,
            colorscale="Greens",
            contours=dict(
                start=0, end=1200, size=50, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_Default.update_layout(
        title="Z_Pred_Default: y = 10 * x1 + 2 * x2 + 5 ",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    contour_plot_Z_Pred_x1.show()
    contour_plot_Z_Pred_x2.show()
    contour_plot_Z_Pred_x1x2.show()
    contour_plot_Z_Pred_Default.show()


def test_filter_two_vars_3():
    """
    y = 10x1 + 5x1x2 + 3
        1000 Trees --- Depth 2
    """
    ##### Generate Data #####
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 1000)
    x2 = np.random.uniform(0, 100, 1000)
    y = 10 * x1 + 5 * x1 * x2 + 3
    X = pd.DataFrame({"x1": x1, "x2": x2})  # create tabular format of x1, x2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ##### FIT REGRESSOR  #####
    model = xgb.XGBRegressor(
        n_estimators=1000,  # 1000 trees
        max_depth=2,  # max_depth of 2
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.5,
    )
    model.fit(X_train, y_train)

    ##### GENERATE PREDICTIONS #####

    x1 = np.arange(0, 100, 0.1)
    x2 = np.arange(0, 100, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    all_pairs = np.column_stack([X1.flatten(), X2.flatten()])
    # convert all pairs to DataFrame structure
    X_test = pd.DataFrame({"x1": all_pairs[:, 0], "x2": all_pairs[:, 1]})

    # get predictions
    z_pred_x1 = tree_filtering.predict(model, (0,), X_test)
    z_pred_x2 = tree_filtering.predict(model, (1,), X_test)
    z_pred_x1x2 = tree_filtering.predict(model, (0, 1), X_test)
    z_pred = model.predict(X_test, output_margin=True)  # vector

    # reshape Z3 vector into 2D numpy array for contour plotting (rows: x2, cols: x1)
    Z_Pred_x1 = z_pred_x1.reshape(len(x2), len(x1))
    Z_Pred_x2 = z_pred_x2.reshape(len(x2), len(x1))
    Z_Pred_x1x2 = z_pred_x1x2.reshape(len(x2), len(x1))
    Z_Pred_Default = z_pred.reshape(len(x2), len(x1))

    ##### PLOT #####
    # Z_Pred_x1
    contour_plot_Z_Pred_x1 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1,
            x=x1,
            y=x2,
            colorscale="blues",
            contours=dict(
                start=-3000, end=3000, size=250, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )
    contour_plot_Z_Pred_x1.update_layout(
        title="Z_Pred_x1: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    )

    # Z_Pred_x2
    contour_plot_Z_Pred_x2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x2,
            x=x1,
            y=x2,
            colorscale="blues",
            contours=dict(
                start=-3000, end=3000, size=250, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_x2.update_layout(
        title="Z_Pred_x2: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    )

    # Z_Pred_x1x2
    contour_plot_Z_Pred_x1x2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1x2,
            x=x1,
            y=x2,
            colorscale="sunset",
            contours=dict(
                start=-5000, end=55000, size=5000, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_x1x2.update_layout(
        title="Z_Pred_x1x2: y = 10 * x1 + 2 * x2 + 5 ",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    # # Z_Pred_Default
    contour_plot_Z_Pred_Default = go.Figure(
        data=go.Contour(
            z=Z_Pred_Default,
            x=x1,
            y=x2,
            colorscale="Greens",
            contours=dict(
                start=-5000, end=55000, size=5000, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_Default.update_layout(
        title="Z_Pred_Default: y = 10 * x1 + 2 * x2 + 5 ",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    contour_plot_Z_Pred_x1.show()
    contour_plot_Z_Pred_x2.show()
    contour_plot_Z_Pred_x1x2.show()
    contour_plot_Z_Pred_Default.show()


def test_filter_two_vars_4():
    """
    y = 10 * x1 + 5 * x1 * x2 + 3
        WITH CORRELATION BETWEEN x1/x2
    """
    ##### GENERATE DATA #####
    np.random.seed(42)

    # Specify means and covariance matrix
    mean = [0, 0]  # Means for x1 and x2
    corr = 0  # Desired correlation between x1 and x2 (rho)
    std_x1 = 1  # Standard deviation for x1
    std_x2 = 1  # Standard deviation for x2

    # Covariance matrix
    cov = [[std_x1**2, corr * std_x1 * std_x2], [corr * std_x1 * std_x2, std_x2**2]]

    # Generate correlated normal data
    x1, x2 = np.random.multivariate_normal(mean, cov, 1000).T

    # Compute y using your equation
    y = 10 * x1 + 2 * x2 + 3

    # Prepare DataFrame and split
    X = pd.DataFrame({"x1": x1, "x2": x2})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ##### FIT REGRESSOR #####
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=1,
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.8,
    )
    model.fit(X_train, y_train)
    # print(tree_filtering.get_bias(model))

    # trees_json = model.get_booster().get_dump(dump_format="json")
    # print(trees_json)

    #### 2D LINES #####
    x1_ver1 = np.arange(-3, 3, 0.01)  # 0.1 step, not including 100
    x2_ver1 = np.random.uniform(0, len(x1_ver1), size=len(x1))
    # X_test1 = pd.DataFrame({"x1": x1, "x2": x2})
    X_test1 = pd.DataFrame({"x1": x1_ver1, "x2": x2_ver1})

    x1_ver2 = np.random.uniform(0, 100, size=len(x1))
    x2_ver2 = np.arange(0, 100, 0.1)
    # X_test2 = pd.DataFrame({"x1": x1, "x2": x2})
    X_test2 = pd.DataFrame({"x1": x2_ver1, "x2": x1_ver1})

    z_pred_x1 = tree_filtering.predict(model, (0,), X_test1)
    z_pred_x2 = tree_filtering.predict(model, (1,), X_test2)

    x1_plot = px.scatter(
        x=X_test1["x1"],
        y=z_pred_x1,
        title="Predicted z vs x1",
        labels={"x": "x", "y": "Predicted z"},
    )
    x1_plot.show()

    x2_plot = px.scatter(
        x=X_test1["x2"],
        y=z_pred_x2,
        title="Predicted z vs x2",
        labels={"x": "x", "y": "Predicted z"},
    )
    x2_plot.show()


def test_filter_two_vars_5():
    """
    y = log(x1 * x2)
        1000 Trees --- depth 2


    """
    ##### Generate Data #####
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 1000)
    x2 = np.random.uniform(0, 100, 1000)
    y = np.log(x1 * x2)
    X = pd.DataFrame({"x1": x1, "x2": x2})  # create tabular format of x1, x2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ##### FIT REGRESSOR  #####
    model = xgb.XGBRegressor(
        n_estimators=1000,  # 1000 trees
        max_depth=2,  # max_depth of 2
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.5,
    )
    model.fit(X_train, y_train)

    ##### GENERATE PREDICTIONS #####
    x1 = np.arange(0, 100, 0.1)
    x2 = np.arange(0, 100, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    all_pairs = np.column_stack([X1.flatten(), X2.flatten()])
    # convert all pairs to DataFrame structure
    X_test = pd.DataFrame({"x1": all_pairs[:, 0], "x2": all_pairs[:, 1]})

    # get predictions
    z_pred_x1 = tree_filtering.predict(model, (0,), X_test)
    z_pred_x2 = tree_filtering.predict(model, (1,), X_test)
    z_pred_x1x2 = tree_filtering.predict(model, (0, 1), X_test)
    z_pred = model.predict(X_test, output_margin=True)  # vector

    # reshape Z3 vector into 2D numpy array for contour plotting (rows: x2, cols: x1)
    Z_Pred_x1 = z_pred_x1.reshape(len(x2), len(x1))
    Z_Pred_x2 = z_pred_x2.reshape(len(x2), len(x1))
    Z_Pred_x1x2 = z_pred_x1x2.reshape(len(x2), len(x1))
    Z_Pred_Default = z_pred.reshape(len(x2), len(x1))

    ##### PLOT #####
    # Z_Pred_x1
    contour_plot_Z_Pred_x1 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1,
            x=x1,
            y=x2,
            colorscale="blues",
            contours=dict(
                start=0, end=1, size=0.1, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )
    contour_plot_Z_Pred_x1.update_layout(
        title="Z_Pred_x1: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    )

    # Z_Pred_x2
    contour_plot_Z_Pred_x2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x2,
            x=x1,
            y=x2,
            colorscale="blues",
            contours=dict(
                start=-2, end=2, size=0.25, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_x2.update_layout(
        title="Z_Pred_x2: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    )

    # Z_Pred_x1x2
    contour_plot_Z_Pred_x1x2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1x2,
            x=x1,
            y=x2,
            colorscale="sunset",
            contours=dict(
                start=-4, end=10, size=1, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_x1x2.update_layout(
        title="Z_Pred_x1x2: y = 10 * x1 + 2 * x2 + 5 ",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    # Z_Pred_Default
    contour_plot_Z_Pred_Default = go.Figure(
        data=go.Contour(
            z=Z_Pred_Default,
            x=x1,
            y=x2,
            colorscale="Greens",
            contours=dict(
                start=-2, end=10, size=2, coloring="heatmap", showlabels=True
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )

    contour_plot_Z_Pred_Default.update_layout(
        title="Z_Pred_Default: y = 10 * x1 + 2 * x2 + 5 ",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    contour_plot_Z_Pred_x1.show()
    contour_plot_Z_Pred_x2.show()
    contour_plot_Z_Pred_x1x2.show()
    contour_plot_Z_Pred_Default.show()


def test_filter_three_vars_1():
    """
    y = 10x1 + 8x2 + 7x3 + 2x1x2 + 9x1x3 + 3
        1000 Trees ---- depth 2
    """
    ##### GENERATE DATA #####
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 1000)
    x2 = np.random.uniform(0, 100, 1000)
    x3 = np.random.uniform(0, 100, 1000)
    y = 10 * x1 + 8 * x2 + 7 * x3 + 2 * x1 * x2 + 9 * x1 * x3 + 3
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    ##### FIT REGRESSOR #####
    model = xgb.XGBRegressor(
        n_estimators=1000,  # 1000 trees
        max_depth=2,  # max_depth of 2
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.5,
    )
    model.fit(X_train, y_train)

    ##### GENERATE PREDICTIONS #####
    # For visualization, use a smaller grid to avoid memory issues
    x1 = np.arange(0, 100, 1)
    x2 = np.arange(0, 100, 1)
    x3 = np.arange(0, 100, 1)
    X1, X2, X3 = np.meshgrid(x1, x2, x3)
    all_points = np.column_stack([X1.flatten(), X2.flatten(), X3.flatten()])
    X_test = pd.DataFrame(
        {"x1": all_points[:, 0], "x2": all_points[:, 1], "x3": all_points[:, 2]}
    )

    # get predictions
    pred_x1 = tree_filtering.predict(model, (0,), X_test)
    pred_x2 = tree_filtering.predict(model, (1,), X_test)
    pred_x3 = tree_filtering.predict(model, (2,), X_test)
    pred_x1x2 = tree_filtering.predict(model, (0, 1), X_test)
    pred_x1x3 = tree_filtering.predict(model, (0, 2), X_test)
    pred_x2x3 = tree_filtering.predict(model, (1, 2), X_test)
    pred_x1x2x3 = tree_filtering.predict(model, (0, 1, 2), X_test)
    pred = model.predict(X_test, output_margin=True)  # vector

    # reshape Z3 vector into 2D numpy array for contour plotting (rows: x2, cols: x1)
    Pred_x1 = pred_x1.reshape(len(x2), len(x1), len(x1))
    Pred_x2 = pred_x2.reshape(len(x2), len(x1), len(x1))
    Pred_x3 = pred_x3.reshape(len(x3), len(x1), len(x1))
    Pred_x1x2 = pred_x1x2.reshape(len(x3), len(x1), len(x1))
    Pred_x1x3 = pred_x1x3.reshape(len(x3), len(x1), len(x1))
    Pred_x2x3 = pred_x2x3.reshape(len(x3), len(x1), len(x1))
    Pred_x1x2x3 = pred_x1x2x3.reshape(len(x3), len(x1), len(x1))
    Z_Pred_Default = pred.reshape(len(x2), len(x1), len(x1))

    ##### PLOT #####
    pred_x1 = go.Figure(
        data=go.Scatter3d(
            x=x1,
            y=x2,
            z=x3,
            mode="markers",
            marker=dict(
                size=3,
                cmin=100,
                cmax=150,
                color=Pred_x1,  # Color by predicted y
                colorscale="Viridis",
                colorbar=dict(title="Predicted y"),
            ),
        )
    )

    pred_x1.update_layout(
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x3"),
        title="pred_x1: y = 10x1 + 8x2 + 7x3 + 2x1x2 + 9x1x3 + 3",
    )

    pred = go.Figure(
        data=go.Scatter3d(
            x=x1,
            y=x2,
            z=x3,
            mode="markers",
            marker=dict(
                size=3,
                cmin=100,
                cmax=150,
                color=pred,  # Color by predicted y
                colorscale="Viridis",
                colorbar=dict(title="Predicted y"),
            ),
        )
    )
    pred.update_layout(
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x3"),
        title="4D Visualization: Predicted y as color",
    )

    pred_x1.show()
    pred.show()


if __name__ == "__main__":
    pass


# contour_plot = go.Figure(
#     data=go.Contour(
#         z=Z3,
#         x=x1,
#         y=x2,
#         colorscale="Greens",
#         contours=dict(
#             start=-1, end=1, size=0.2, coloring="heatmap", showlabels=True
#         ),
#         line=dict(width=2),
#         colorbar=dict(title="Z value"),
#     )
# )

# contour_plot.update_layout(
#     title="y = 10 * x1 + 2 * x2 + 5", xaxis_title="x1", yaxis_title="x2"
# )

# contour_plot.show()

##########################

# total_contour = go.Figure()

# total_contour.add_trace(
#     go.Contour(
#         z=Z1,
#         x=x1,
#         y=x2,
#         colorscale="Blues",
#         contours=dict(showlabels=True),
#         line=dict(width=2),
#         name="z1",
#         showscale=False,  # Hide extra colorbar
#     )
# )

# total_contour.add_trace(
#     go.Contour(
#         z=Z2,
#         x=x1,
#         y=x2,
#         colorscale="Reds",
#         contours=dict(showlabels=True),
#         line=dict(width=2, dash="dash"),
#         name="z2",
#         showscale=False,  # Hide extra colorbar
#     )
# )
# total_contour.add_trace(
#     go.Contour(
#         z=Z3,
#         x=x1,
#         y=x2,
#         colorscale="Greens",
#         contours=dict(showlabels=True),
#         line=dict(width=2),
#         name="z3",
#         colorbar=dict(title="Z value"),  # Only one colorbar
#     )
# )
# total_contour.show()


####### TESTING #######

# plt.xlabel("x1")
# plt.ylabel("y")
# plt.scatter(x1, y_true, label="True y", color="green", marker="x")
# plt.scatter(
#     x1, y_pred, label="Default boosted tree y-prediction", color="red", marker="*"
# )
# plt.scatter(
#     x1, y_pred_sum, label="Manual tree SUM y-prediction", color="pink", marker="."
# )
# plt.show()

# print(f"y_pred: {y_pred}")
# print(f"y_pred_sum: {y_pred_sum}")

# print(get_filtered_tree_list_ranges_from_tuple(model1, (0,)))

# ########################################################################################################
# ########### TEST 2: y = b1 * x1 + b2 * x1 * x2 + b3 * x2 + b4 * x3**2 + b5 * x4 * x5 + noise ###########
# ########################################################################################################

# # Set random seed for reproducibility
# np.random.seed(42)

# # Number of samples
# n_samples = 100

# # Coefficients
# b1 = 10
# b2 = 20
# b3 = 5
# b4 = 3
# b5 = -7

# # Generate random features
# x1 = np.random.uniform(-5, 5, n_samples)
# x2 = np.random.uniform(-5, 5, n_samples)
# x3 = np.random.uniform(-5, 5, n_samples)
# x4 = np.random.uniform(-5, 5, n_samples)
# x5 = np.random.uniform(-5, 5, n_samples)

# # Generate target variable with cross terms and noise
# noise = np.random.normal(0, 2, n_samples)
# y = b1 * x1 + b2 * x1 * x2 + b3 * x2 + b4 * x3**2 + b5 * x4 * x5 + noise

# # Stack features into a matrix
# X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})

# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Create and configure the XGBoost regressor
# model = xgb.XGBRegressor(
#     objective="reg:squarederror",  # Standard regression objective
#     n_estimators=1000,  # Number of boosting rounds (trees)
#     learning_rate=0.1,  # Step size shrinkage
#     max_depth=5,  # Maximum tree depth
#     subsample=0.8,  # Fraction of samples per tree
#     colsample_bytree=0.8,  # Fraction of features per tree
#     random_state=42,
#     base_score=0.5,
# )

# # Fit the model to your training data
# model.fit(X_train, y_train)

# # Predict on the test set
# y_true = y
# y_pred = model.predict(X_test)
# y_pred_sum = predict(
#     model,
#     (0, 1, 2),
#     X_test,
# )

# print(f"y_pred: {y_pred}")
# print(f"y_pred_sum: {y_pred_sum}")


########## OLD ONEFEATURE_METHOD1 STUFF ##########
# predictions = get_all_split_tree_predictions(model1, X1_test)
# y_pred_sum = sum_split_tree_predictions(model1, predictions)


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
