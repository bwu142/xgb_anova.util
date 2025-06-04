# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import json


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
            "n_estimators": 100,
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

    # === Define your chosen subset of features ===
    chosen_features = {"x1", "x3"}  # Example: only keep trees using x1 AND/OR x3

    filtered_trees = []

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

    for tree_str in trees_json:
        tree = json.loads(tree_str)
        features_used = get_features_used(tree)
        if chosen_features.issubset(features_used):
            filtered_trees.append(tree)

    ### OR
    # # === Helper function to recursively check for feature usage in a tree ===
    # def tree_uses_chosen_feature(node, chosen_features):
    #     if "split" in node:
    #         if node["split"] in chosen_features:
    #             return True
    #         # Recursively check children
    #             tree_uses_chosen_feature(child, chosen_features)
    #             for child in node["children"]
    #         )
    #     return False  # Leaf node

    # # === Filter trees ===
    # for tree_str in trees_json:
    #     tree = json.loads(tree_str)
    #     if tree_uses_chosen_feature(tree, chosen_features):
    #         filtered_trees.append(tree)

    print(f"Original number of trees: {len(trees_json)}")
    print(f"Number of trees using chosen features: {len(filtered_trees)}")

    # # Optionally: Save filtered trees for further analysis
    # with open("filtered_trees.json", "w") as f:
    #     json.dump(filtered_trees, f, indent=2)
