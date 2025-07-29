# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import purify


##### HELPER FUNCTIONS #####
def new_model(trees, base_score, file_name="test_model.json"):
    """
    Args:
        trees (list): list of XGBoost trees (dicts)
        base_score (str):
    """
    # print(trees)
    new_model_file = {
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": str(len(trees)),
                    },
                    "iteration_indptr": [0, 1, 2, 3, 4, 5, 6],
                    "tree_info": [0],
                    "trees": trees,
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": "0.0",
                "boost_from_average": "0",
                "num_class": "0",
                "num_feature": "2",
                "num_target": "1",
            },
            "objective": {
                "name": "reg:squarederror",
                "reg_loss_param": {"scale_pos_weight": "1"},
            },
        },
        "version": [3, 0, 2],
    }
    # print(new_model_file)
    base_score = str(float(base_score))
    new_model = purify.update_metadata(new_model_file, base_score)

    model_file = purify.get_model_file(new_model, True, "test_model.json")
    # print(model_file["learner"]["gradient_booster"]["model"]["trees"])
    return new_model


def alpha_tree_predict(alpha_tree, dataset):
    alpha_tree_model = new_model([alpha_tree], "0.0", "alpha_tree_model.json")
    prediction = alpha_tree_model.predict(dataset)
    return prediction


##### MANUAL CHECK-TESTS #####
def test_vector_to_tree(
    alpha_vector, split_condition_vector, feature_index, dataset, true_prediction
):
    tree_from_vector = purify.tree_from_vector(
        alpha_vector, split_condition_vector, feature_index
    )

    test_model = new_model([tree_from_vector], "0.0")
    test_model_prediction = test_model.predict(dataset)
    print(f"TEST_VECTOR_PREDICTION: {test_model_prediction}")

    assert np.allclose(test_model_prediction, true_prediction, atol=1e-4, rtol=0.0)


def test_grid_to_tree(
    grid, split_vector_x, split_vector_y, feature_tuple, dataset, true_prediction
):
    tree_from_grid = purify.tree_from_grid(
        grid, split_vector_x, split_vector_y, feature_tuple
    )
    test_model = new_model([tree_from_grid], "0.0")

    test_model_prediction = test_model.predict(dataset)
    print(f"TEST_GRID_PREDICTION: {test_model_prediction}")

    assert np.allclose(test_model_prediction, true_prediction, atol=1e-4, rtol=0.0)


##### REGRESSION TESTS #####
def test_purify_two_features_equality(
    alpha_tree, dataset, original_prediction, split_conditions_dict
):

    alpha_prediction = alpha_tree_predict(alpha_tree, dataset)
    data = dataset.get_data()


def test_purify_two_features_all_pairs(model, dataset):
    """
    Test purification for all feature pairs (2-way interactions).

    Args:
        model: XGBoost Booster trained on dataset
        dataset: xgb.DMatrix with the data used by model

    Raises:
        AssertionError if purity or prediction equality checks fail.
    """
    model_file = purify.get_model_file(model, False)
    tree_list = model_file["learner"]["gradient_booster"]["model"]["trees"]

    num_features = dataset.num_col()
    feature_tuples = purify.all_combinations(list(range(num_features)))

    # Extract split conditions for all features used in the model
    # Assumes you have a helper that returns dict {feature_idx: split_values}
    split_conditions_dict = purify.get_split_conditions(
        tree_list,
        list(range(num_features)),
    )

    for pair in feature_tuples:
        # Extract submodel sensitive only to these two features
        submodel = purify.get_filtered_model(model, pair, use_inplace=False)

        # Run purification
        (vector_x, vector_y), alpha_tree = purify.purify_two_features(
            submodel, dataset, split_conditions_dict, pair
        )

        # Reconstruct alpha grid from purified tree (you need this util)
        grid_alphas = purify.predict_grid_from_tree(
            alpha_tree, split_conditions_dict, pair
        )

        # 1) Purity check: means along each axis should be near zero
        mean_x = np.mean(grid_alphas, axis=0)
        mean_y = np.mean(grid_alphas, axis=1)
        assert np.allclose(
            mean_x, 0, atol=1e-6
        ), f"Interaction grid not pure on x-axis for features {pair}: {mean_x}"
        assert np.allclose(
            mean_y, 0, atol=1e-6
        ), f"Interaction grid not pure on y-axis for features {pair}: {mean_y}"

        # 2) Prediction equality check
        original_preds = submodel.predict(dataset)
        alpha_tree_preds = alpha_tree_predict(alpha_tree, dataset)

        data = dataset.get_data()
        x_bins = np.digitize(data[:, pair[0]], split_conditions_dict[pair[0]])
        y_bins = np.digitize(data[:, pair[1]], split_conditions_dict[pair[1]])

        vector_contrib = vector_x[x_bins] + vector_y[y_bins]
        purified_preds = alpha_tree_preds + vector_contrib

        assert np.allclose(
            original_preds, purified_preds, atol=1e-5
        ), f"Predictions mismatch for features {pair}: max diff = {(np.abs(original_preds - purified_preds)).max()}"


def test_purify_one_feature():
    pass


def test_purity(model, dataset):
    # Run purification
    new_model = purify.purify_2D(model, dataset, save_to_disk=False)

    # Extract trees and related data (you may need custom code)
    trees = purify.get_model_file(new_model, True)["learner"]["gradient_booster"][
        "model"
    ]["trees"]
    for tree in trees:
        base_weights = np.array(tree["base_weights"])
        if len(base_weights.shape) == 1:
            # For 1D main effect: mean should be zero
            assert np.isclose(base_weights.mean(), 0, atol=1e-8)
        elif len(base_weights.shape) == 2:
            # For 2D interaction: mean zero across axes
            assert np.allclose(base_weights.mean(axis=0), 0, atol=1e-8)
            assert np.allclose(base_weights.mean(axis=1), 0, atol=1e-8)


def test_equal_predictions(model, dataset):
    original_prediction = model.predict(dataset)
    new_model = purify.purify_2D(model, dataset, True, "XYZ.json")
    purified_prediction = new_model.predict(dataset)
    print(f"original prediction: {original_prediction[:5]}")
    print(f"purified prediction: {purified_prediction[:5]}")
    assert np.allclose(original_prediction, purified_prediction, atol=1e-3, rtol=0.0)


if __name__ == "__main__":

    ##### test_vector_to_tree #####
    if False:
        # test 1
        X_test = np.array([[5.0], [15.0], [25.0]])
        dtest = xgb.DMatrix(X_test)
        alpha_vector = np.array([1.0, 2.0, 3.0])  # leaf values for bins
        split_condition_vector = np.array([10.0, 20.0])
        true_prediction = np.array([1.0, 2.0, 3.0])
        test_vector_to_tree(
            alpha_vector, split_condition_vector, 0, dtest, true_prediction
        )
        # test 2
        X_test = np.array([[1, 5.0], [1, 15.0], [1, 25.0], [1, 35.0]])
        # X_test = np.array([[5.0], [15.0], [25.0], [35.0]])
        dtest = xgb.DMatrix(X_test)
        alpha_vector = np.array([1.0, 2.0, 3.0, 4.0])
        split_condition_vector = np.array([10.0, 20.0, 30.0])
        true_prediction = np.array([1.0, 2.0, 3.0, 4.0])
        test_vector_to_tree(
            alpha_vector, split_condition_vector, 1, dtest, true_prediction
        )
        # test 3
        X_test = np.array([[1, 5.0], [1, 15.0], [1, 25.0], [1, 35.0], [1, 45.0]])
        # X_test = np.array([[5.0], [15.0], [25.0], [35.0]])
        dtest = xgb.DMatrix(X_test)
        alpha_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        split_condition_vector = np.array([10.0, 20.0, 30.0, 40.0])
        true_prediction = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_vector_to_tree(
            alpha_vector, split_condition_vector, 1, dtest, true_prediction
        )

    ##### test_grid_to_tree #####
    if False:
        # test one
        split_condition_vector_x = np.array([10.0, 20.0])
        split_condition_vector_y = np.array([10.0, 30.0])
        grid_alphas = np.array(
            [
                [-300.0, -200.0, -100.0],
                [-30.0, -20.0, -10.0],
                [-3.0, -2.0, -1.0],
            ]
        )
        feature_tuple = (0, 1)
        X_test = np.array([[15, 5], [5, 25]])
        dtest = xgb.DMatrix(X_test)
        true_prediction = np.array([-30.0, -200.0])
        test_grid_to_tree(
            grid_alphas,
            split_condition_vector_x,
            split_condition_vector_y,
            feature_tuple,
            dtest,
            true_prediction,
        )

        # test two
        split_condition_vector_x = np.array([10.0, 20.0, 30.0, 40.0])
        split_condition_vector_y = np.array([10.0, 20.0, 30.0])
        grid_alphas = np.array(
            [
                [-40000, -30000, -20000, -10000],
                [-4000, -3000, -2000, -1000],
                [-400, -300, -200, -100],
                [-40, -30, -20, -10],
                [-4, -3, -2, -1],
            ]
        )
        feature_tuple = (0, 1)
        X_test = np.array([[15, 25], [100, 25]])
        dtest = xgb.DMatrix(X_test)
        true_prediction = np.array([-2000, -2])
        test_grid_to_tree(
            grid_alphas,
            split_condition_vector_x,
            split_condition_vector_y,
            feature_tuple,
            dtest,
            true_prediction,
        )

        # test theee
        split_condition_vector_x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        split_condition_vector_y = np.array([15.0, 45.0])
        grid_alphas = np.array(
            [
                [-300000, -200000, -100000],
                [-30000, -20000, -10000],
                [-3000, -2000, -1000],
                [-300, -200, -100],
                [-30, -20, -10],
                [-3, -2, -1],
            ]
        )
        feature_tuple = (0, 1)
        X_test = np.array([[0, 20], [35, 50]])
        dtest = xgb.DMatrix(X_test)
        true_prediction = np.array([-200000, -100])
        test_grid_to_tree(
            grid_alphas,
            split_condition_vector_x,
            split_condition_vector_y,
            feature_tuple,
            dtest,
            true_prediction,
        )

    ##### test_purify_two_features #####

    ##### test_purify_one_features #####

    ##### test_purity #####
    if False:
        tree = {
            "base_weights": [
                1.9408799e-7,
                -2.3663945,
                2.4047706,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "categories": [],
            "categories_nodes": [],
            "categories_segments": [],
            "categories_sizes": [],
            "default_left": [0, 0, 0, 0, 0, 0, 0],
            "id": 0,
            "left_children": [1, 3, 5, -1, -1, -1, -1],
            "loss_changes": [2.610693e5, 6.0227062e4, 5.802592e4, 0, 0, 0, 0],
            "parents": [2147483647, 0, 0, 1, 1, 2, 2],
            "right_children": [2, 4, 6, -1, -1, -1, -1],
            "split_conditions": [
                10.0,
                5.0,
                15.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "split_indices": [0, 1, 1, 0, 0, 0, 0],
            "split_type": [0, 0, 0, 0, 0, 0, 0],
            "sum_hessian": [
                4.5875e4,
                2.3122e4,
                2.2753e4,
                1.284e4,
                1.0282e4,
                1.0986e4,
                1.1767e4,
            ],
            "tree_param": {
                "num_deleted": "0",
                "num_feature": "2",
                "num_nodes": "7",
                "size_leaf_vector": "1",
            },
        }
        model = new_model([tree], "10.0")

        X_train = np.array([[5, 2.5], [15, 20]])
        dtrain = xgb.DMatrix(X_train)
        original_prediction = model.predict(dtrain)
        print(f"Original predictions: {original_prediction}")

        split_x = np.array([10.0])
        split_y = np.array([5.0, 15.0])
        split_conditions_dict = {0: split_x, 1: split_y}
        feature_tuple = (0, 1)

        new_model = purify.purify_2D(model, dtrain, True)
        print(f"Purified (New) Prediction: {new_model.predict(dtrain)}")

    ##### test_equal_predictions #####
    if True:
        n = 1 << 16
        rho_val = 0  # 0, .5, .9, .999, 1
        b1, b2, b3 = 3, 2, 10
        cov_mat = np.identity(2)
        cov_mat[0, 1] = cov_mat[1, 0] = rho_val
        X = np.random.multivariate_normal(np.zeros(2), cov_mat, n)
        yf = lambda x1, x2: b1 * x1 + b2 * x2 + b3
        y_true = yf(X[:, 0], X[:, 1])

        np.random.seed(42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.3, random_state=42
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.save_binary("dtrain.buffer")
        # dtrain = xgb.DMatrix("dtrain.buffer")
        dtest = xgb.DMatrix(X_test, label=y_test)

        if True:
            params = {
                "max_depth": 2,
                "learning_rate": 1.0,
                "objective": "reg:squarederror",
                "random_state": 42,
            }

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=10,  # Equivalent to n_estimators
                evals=[(dtrain, "train"), (dtest, "test")],
                verbose_eval=True,
            )
        test_equal_predictions(model, dtrain)
