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
def new_model(trees, base_score):
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

    model_file = purify.get_model_file(new_model, True, "ZZZ.json")
    # print(model_file["learner"]["gradient_booster"]["model"]["trees"])
    return new_model


##### ACTUAL TESTS #####
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


def test_purify_two_features():
    pass


def test_purify_one_feature():
    pass


def test_purity():
    pass


def test_equal_predictions():
    pass


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

    ##### test_grid_to_tree
    if True:
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
