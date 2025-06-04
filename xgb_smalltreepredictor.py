import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import json
import matplotlib.pyplot as plt


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

    # print(filtered_trees)

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
