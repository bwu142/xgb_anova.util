# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import purify
from pygam import LinearGAM, s, f


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
    new_model = purify.update_metadata(new_model_file, base_score, True, file_name)

    model_file = purify.get_model_file(new_model, True, file_name)
    # print(model_file["learner"]["gradient_booster"]["model"]["trees"])
    return new_model


def alpha_tree_predict(alpha_tree, dataset):
    alpha_tree_model = new_model([alpha_tree], "0.0", "alpha_tree_model.json")
    prediction = alpha_tree_model.predict(dataset)
    return prediction


##### HELPER FUNCTION TESTS #####
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


##### UNUSED TESTS #####
def test_purity(model, dataset):
    # Run purification
    result = purify.fANOVA_2D(False, "purity", model, dataset)

    # Get all split thresholds once for all features from the whole model
    feature_list = list(range(dataset.num_col()))
    tree_list = purify.get_model_file(model, save_to_disk=False)["learner"][
        "gradient_booster"
    ]["model"]["trees"]
    split_conditions_dict = purify.get_split_conditions(tree_list, feature_list)

    # Iterate through all combinations (main effects and pairwise interactions)
    for feature_tuple, submodel in result.purified_model_dict.items():
        # main effect
        if len(feature_tuple) == 1:
            main_effect_prediction = submodel.predict(dataset)
            mean = np.mean(main_effect_prediction)
            assert abs(mean) < 1e-3
        # interaction
        else:
            if True:
                split_x = split_conditions_dict[feature_tuple[0]]
                split_y = split_conditions_dict[feature_tuple[1]]

                data_x_col = purify.get_data_col(dataset, feature_tuple[0])
                data_y_col = purify.get_data_col(dataset, feature_tuple[1])

                x_bins = np.digitize(data_x_col, split_x)
                y_bins = np.digitize(data_y_col, split_y)
                predictions = submodel.predict(dataset)

                num_bins_x = len(split_x) + 1
                num_bins_y = len(split_y) + 1

                grid_sums = np.zeros((num_bins_x, num_bins_y))
                grid_counts = np.zeros((num_bins_x, num_bins_y))

                for xb, yb, pred in zip(x_bins, y_bins, predictions):
                    grid_sums[xb, yb] += pred
                    grid_counts[xb, yb] += 1

                # Data-weighted mean across each axis (skip empty bins)
                mean_along_x = np.zeros(num_bins_x)
                mean_along_y = np.zeros(num_bins_y)

                # Along x (columns: for each y_bin, sum over x, weighted)
                for yb in range(num_bins_y):
                    valid = grid_counts[:, yb] > 0
                    if valid.any():
                        means = np.zeros(num_bins_x)
                        means[valid] = grid_sums[valid, yb] / grid_counts[valid, yb]
                        weights = grid_counts[valid, yb]
                        mean_along_x[yb] = (
                            np.sum(means[valid] * weights) / weights.sum()
                        )

                # Along y (rows: for each x_bin, sum over y, weighted)
                for xb in range(num_bins_x):
                    valid = grid_counts[xb, :] > 0
                    if valid.any():
                        means = np.zeros(num_bins_y)
                        means[valid] = grid_sums[xb, valid] / grid_counts[xb, valid]
                        weights = grid_counts[xb, valid]
                        mean_along_y[xb] = (
                            np.sum(means[valid] * weights) / weights.sum()
                        )

                assert np.all(
                    np.abs(mean_along_x) < 1e-3
                ), f"Interaction (weighted, x-axis) not zero: {mean_along_x}"
                assert np.all(
                    np.abs(mean_along_y) < 1e-3
                ), f"Interaction (weighted, y-axis) not zero: {mean_along_y}"
            if False:
                split_condition_vector_x = split_conditions_dict[feature_tuple[0]]
                split_condition_vector_y = split_conditions_dict[feature_tuple[1]]
                data_x_col = purify.get_data_col(dataset, feature_tuple[0])
                data_y_col = purify.get_data_col(dataset, feature_tuple[1])

                x_binned_indices = np.digitize(
                    data_x_col, split_condition_vector_x
                )  # (N x 1)
                y_binned_indices = np.digitize(
                    data_y_col, split_condition_vector_y
                )  # (N x 1)

                predictions = submodel.predict(dataset)

                num_bins_x = len(split_condition_vector_x) + 1
                num_bins_y = len(split_condition_vector_y) + 1
                sum_vector_x = np.zeros(num_bins_x)
                sum_vector_y = np.zeros(num_bins_y)
                count_vector_x = np.zeros(num_bins_x)
                count_vector_y = np.zeros(num_bins_y)

                np.add.at(sum_vector_x, x_binned_indices, predictions)
                np.add.at(sum_vector_y, y_binned_indices, predictions)
                np.add.at(count_vector_x, x_binned_indices, predictions)
                np.add.at(count_vector_y, y_binned_indices, predictions)

                # check purity across rows
                nonzero = count_vector_x > 0
                mean_vector_x = np.zeros(num_bins_x)
                mean_vector_x[nonzero] = sum_vector_x[nonzero] / count_vector_x[nonzero]
                print("mean_vector_x: ", mean_vector_x)
                assert np.any(mean_vector_x > 1e-3) is True

                # check purity across cols
                nonzero = count_vector_y > 0
                mean_vector_y = np.zeros(num_bins_y)
                mean_vector_y[nonzero] = sum_vector_y[nonzero] / count_vector_y[nonzero]
                print("mean_vector_y: ", mean_vector_y)
                assert np.any(mean_vector_y > 1e-3) is True


def plot_diag(model, dataset):
    """
    Args:
        result (fANOVA_Result object):
        dataset (dataframe)
        rho_val (float between -1, 1).
        yf (function)
    """
    # purify model
    result = purify.fANOVA_2D(False, "gph_diag", model, xgb.DMatrix(dataset))

    # theoretical purified interaction component
    rho_over2 = rho / (1 + rho**2)
    ref_x1x2 = lambda x: b3 * (
        x[:, 0] * x[:, 1]
        - rho_over2 * (x[:, 0] ** 2 + x[:, 1] ** 2)
        + rho_over2 * (1 - rho**2)
    )

    # Create evaluation grid along diagonal (x1 = x2)
    gridn = 100
    plot_range = (-1.5, 1.5)
    x_vals = np.linspace(*plot_range, num=gridn)
    diag_grid = np.column_stack([x_vals, x_vals])  # x1 = x2

    # Get predictions
    diag_grid_dm = xgb.DMatrix(diag_grid)
    pred_pure = result.purified_model_dict[(0, 1)].predict(diag_grid_dm)
    pred_orig = result.original_model.predict(diag_grid_dm)
    true_pure = ref_x1x2(diag_grid)

    original_model = result.original_model
    original_model_x1x2 = purify.get_filtered_model(original_model, (0, 1))
    pred_orig = original_model_x1x2.predict(diag_grid_dm)

    # Create plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pred_orig,
            mode="lines",
            name="Original Prediction",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pred_pure,
            mode="lines",
            name="Purified Prediction",
            line=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=true_pure,
            mode="lines",
            name="Reference",
            line=dict(color="red", dash="dash"),
        )
    )

    if False:
        # Add error metrics to title
        abs_error = np.abs(pred_pure - true_pure)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

    fig.update_layout(
        title=f"Diagonal Comparison (x1 = x2)",
        xaxis_title="x value (x1 = x2)",
        yaxis_title="Interaction Component Value",
        height=500,
    )

    fig.show()


##### MAIN TESTS #####


# Equal Predictions
def test_equal_predictions(model, dataset):
    original_prediction = model.predict(dataset)
    new_model = purify.purify_2D(model, dataset, True, "XYZ.json")
    purified_prediction = new_model.predict(dataset)
    print(f"original prediction: {original_prediction[:5]}")
    print(f"purified prediction: {purified_prediction[:5]}")
    assert np.allclose(original_prediction, purified_prediction, atol=1e-3, rtol=0.0)


def test_fANOVA_1(model, dataset):
    result = purify.fANOVA_2D(False, "test", model, dataset, True)
    dataset.save_binary("dataset.buffer")
    submodel_sum = result.bias
    for submodel in result.purified_model_dict.values():
        submodel_sum += submodel.predict(dataset)
    original_prediction = result.predict_original(dataset)

    print(f"original prediction: {original_prediction[:5]}")
    print(f"purified prediction: {submodel_sum[:5]}")
    assert np.allclose(original_prediction, submodel_sum, atol=1e-3, rtol=0.0)


def test_fANOVA_2(model, dataset):
    result = purify.fANOVA_2D(True, "test")
    submodel_sum = result.bias
    for submodel in result.purified_model_dict.values():
        submodel_sum += submodel.predict(dataset)
    original_prediction = result.predict_original(dataset)

    print(f"original prediction: {original_prediction[:5]}")
    print(f"purified prediction: {submodel_sum[:5]}")
    assert np.allclose(original_prediction, submodel_sum, atol=1e-3, rtol=0.0)


# Plot Comparisons
def plot_pairwise(
    model, dataset, rho, b3, plot_x_range=(-1.5, 1.5), plot_y_range=(-5, 30)
):
    """
    Generates two comparison plots:
    1. Diagonal (x1 = x2) comparison
    2. Correlation structure (x2 = rho*x1) comparison
    """
    if type(dataset) != xgb.DMatrix:
        dataset = xgb.DMatrix(dataset)
    # Purify model
    result = purify.fANOVA_2D(False, "gph_pair", model, dataset)
    orig_model = result.original_model
    orig_filt_model = purify.get_filtered_model(orig_model, (0, 1))

    # Theoretical components
    rho_over2 = rho / (1 + rho**2)

    # 1. Diagonal comparison components
    def ref_diagonal(x):
        """Theoretical interaction along diagonal (x1=x2)"""
        return b3 * (x**2 - 2 * rho_over2 * x**2 + rho_over2 * (1 - rho**2))

    # 2. Correlation structure components
    def ref_correlation(x):
        """Theoretical interaction along x2 = rho*x1"""
        return np.full_like(x, b3 * rho_over2 * (1 - rho**2))

    # Create evaluation grid
    gridn = 100
    x_vals = np.linspace(*plot_x_range, num=gridn)

    # ===== Plot 1: Diagonal (x1 = x2) ===== #
    diag_grid = np.column_stack([x_vals, x_vals])
    diag_dm = xgb.DMatrix(diag_grid)

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=x_vals,
            y=result.purified_model_dict[(0, 1)].predict(diag_dm),
            mode="lines",
            name="Purified",
            line=dict(color="blue"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=x_vals,
            y=ref_diagonal(x_vals),
            mode="lines",
            name="Theoretical",
            line=dict(color="green", dash="dash"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=x_vals,
            y=orig_filt_model.predict(diag_dm),
            mode="lines",
            name="Original (Unpurified)",
            line=dict(color="red"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=x_vals,
            y=np.full_like(x_vals, result.bias),
            mode="lines",
            name="bias",
            line=dict(color="purple"),
        )
    )
    fig1.update_layout(
        title=f"Diagonal Comparison (x1 = x2). rho = {rho}, b3 = {b3}",
        xaxis_title="x value",
        yaxis_title="Interaction Component",
        xaxis=dict(range=list(plot_x_range)),
        yaxis=dict(range=list(plot_y_range)),
    )
    fig1.show()

    # ===== Plot 2: Correlation Structure (x2 = rho*x1) =====
    rho_grid = np.column_stack([x_vals, rho * x_vals])
    rho_dm = xgb.DMatrix(rho_grid)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=x_vals,
            y=result.purified_model_dict[(0, 1)].predict(rho_dm),
            mode="lines",
            name="Purified",
            line=dict(color="blue"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=x_vals,
            y=ref_correlation(x_vals),
            mode="lines",
            name="Theoretical",
            line=dict(color="green", dash="dash"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=x_vals,
            y=orig_filt_model.predict(rho_dm),
            mode="lines",
            name="Original (Unpurified)",
            line=dict(color="red"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=x_vals,
            y=np.full_like(x_vals, result.bias),
            mode="lines",
            name="bias",
            line=dict(color="purple"),
        )
    )
    fig2.update_layout(
        title=f"Correlation Structure (x2 = {rho:.2f}x1). rho = {rho}, b3 = {b3}",
        xaxis_title="x1 value",
        yaxis_title="Interaction Component",
        xaxis=dict(range=list(plot_x_range)),
        yaxis=dict(range=list(plot_y_range)),
    )
    fig2.show()

    return fig1, fig2


def plot_marginal(
    model, dataset, rho, b1, b2, b3, plot_x_range=(-1.5, 1.5), plot_y_range=(-5, 30)
):
    """
    Complete marginal effect analysis for both features.
    """
    if type(dataset) != xgb.DMatrix:
        dataset = xgb.DMatrix(dataset)
    # Purify model
    result = purify.fANOVA_2D(False, "gph_main", model, dataset)
    orig_model = result.original_model
    orig_x1_model = purify.get_filtered_model(orig_model, (0,))
    orig_x2_model = purify.get_filtered_model(orig_model, (1,))

    # Theoretical term
    rho_term = rho / (1 + rho**2)

    # Create evaluation grid
    gridn = 100
    x_vals = np.linspace(*plot_x_range, num=gridn)

    # ---- Plot 1: Marginal Effect of x1 (x2=0) ----
    x1_grid = np.column_stack([x_vals, np.zeros_like(x_vals)])
    x1_dm = xgb.DMatrix(x1_grid)

    def ref_x1(x):
        return b1 * x + b3 * rho_term * (x**2 - 1)

    fig_x1 = go.Figure()
    fig_x1.add_trace(
        go.Scatter(
            x=x_vals,
            y=result.purified_model_dict[(0,)].predict(x1_dm),
            name="Purified",
            line=dict(color="blue"),
        )
    )
    fig_x1.add_trace(
        go.Scatter(
            x=x_vals,
            y=orig_x1_model.predict(x1_dm),
            name="Original (Unpurified)",
            line=dict(color="red"),
        )
    )
    fig_x1.add_trace(
        go.Scatter(
            x=x_vals,
            y=ref_x1(x_vals),
            name="Theoretical",
            line=dict(color="green", dash="dash"),
        )
    )
    fig_x1.update_layout(
        title=f"Marginal Effect of x1 (x2=0) | ρ={rho:.2f}",
        xaxis_title="x1 value",
        yaxis=dict(range=plot_y_range),
    )

    fig_x1.show()
    # ---- Plot 2: Marginal Effect of x2 (x1=0) ----
    x2_grid = np.column_stack([np.zeros_like(x_vals), x_vals])
    x2_dm = xgb.DMatrix(x2_grid)

    def ref_x2(x):
        return b2 * x + b3 * rho_term * (x**2 - 1)

    fig_x2 = go.Figure()
    fig_x2.add_trace(
        go.Scatter(
            x=x_vals,
            y=result.purified_model_dict[(1,)].predict(x2_dm),
            name="Purified",
            line=dict(color="blue"),
        )
    )
    fig_x2.add_trace(
        go.Scatter(
            x=x_vals,
            y=orig_x2_model.predict(x2_dm),
            name="Original (Unpurified)",
            line=dict(color="red"),
        )
    )
    fig_x2.add_trace(
        go.Scatter(
            x=x_vals,
            y=ref_x2(x_vals),
            name="Theoretical",
            line=dict(color="green", dash="dash"),
        )
    )
    fig_x2.update_layout(
        title=f"Marginal Effect of x2 (x1=0) | ρ={rho:.2f}",
        xaxis_title="x2 value",
        yaxis=dict(range=plot_y_range),
    )

    fig_x2.show()

    return fig_x1, fig_x2


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

    ##### test_equal_predictions #####
    if False:
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
        test_fANOVA_1(model, dtrain)
        test_fANOVA_2(model, dtrain)
        test_purity(model, dtrain)

    ##### plot against theoretical #####

    # Generate correlated data
    seed = 42
    n = 1 << 16
    rho = 0.5
    b1, b2, b3 = 3, 2, 10
    cov_mat = np.array([[1, rho], [rho, 1]])

    X = np.random.multivariate_normal(mean=[0, 0], cov=cov_mat, size=n)
    yf = lambda x1, x2: b1 * x1 + b2 * x2 + b3 * x1 * x2
    y = yf(X[:, 0], X[:, 1])

    # train model
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "reg:squarederror",
        "seed": seed,
        "max_depth": 2,
        "eta": 0.1,
        "verbosity": 2,
    }
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtrain, "train")])

    plot_pairwise(model, X, rho, b3)
    plot_marginal(model, X, rho, b1, b2, b3)

    ##### OLD UNUSED STUFF #####
    if False:
        seed = 42
        n = 1 << 14
        rho = -0.3
        b1, b2, b3 = 3, 2, 10
        cov_mat = np.array([[1, rho], [rho, 1]])

        # Generate correlated data
        X = np.random.multivariate_normal(mean=[0, 0], cov=cov_mat, size=n)
        yf = lambda x1, x2: b1 * x1 + b2 * x2 + b3 * x1 * x2
        y = yf(X[:, 0], X[:, 1])

        # Theoretical purified interaction component
        rho_over2 = rho / (1 + rho**2)
        ref_x1x2 = lambda x: b3 * (
            x[:, 0] * x[:, 1]
            - rho_over2 * (x[:, 0] ** 2 + x[:, 1] ** 2)
            + rho_over2 * (1 - rho**2)
        )

        # Create evaluation grid along diagonal (x1 = x2)
        gridn = 100
        plot_range = (-1.5, 1.5)
        x_vals = np.linspace(*plot_range, num=gridn)
        diag_grid = np.column_stack([x_vals, x_vals])  # x1 = x2

        # Train XGBoost model
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:squarederror",
            "seed": seed,
            "max_depth": 2,
            "eta": 0.1,
            "verbosity": 2,
        }
        model = xgb.train(
            params, dtrain, num_boost_round=1000, evals=[(dtrain, "train")]
        )

        # Purify the model
        result = purify.fANOVA_2D(False, "gph", model, dtrain, True)

        # Get predictions
        diag_grid_dm = xgb.DMatrix(diag_grid)
        pred_pure = result.purified_model_dict[(0, 1)].predict(diag_grid_dm)
        pred_orig = result.original_model.predict(diag_grid_dm)
        true_pure = ref_x1x2(diag_grid)

        original_model = result.original_model
        original_model_x1x2 = purify.get_filtered_model(original_model, (0, 1))
        pred_orig = original_model_x1x2.predict(diag_grid_dm)

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_orig,
                mode="lines",
                name="Original Prediction",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_pure,
                mode="lines",
                name="Purified Prediction",
                line=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=true_pure,
                mode="lines",
                name="Reference",
                line=dict(color="red", dash="dash"),
            )
        )

        # Add error metrics to title
        abs_error = np.abs(pred_pure - true_pure)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

        fig.update_layout(
            title=f"Diagonal Comparison (x1 = x2)<br>Max Error: {max_error:.4f}, Mean Error: {mean_error:.4f}",
            xaxis_title="x value (x1 = x2)",
            yaxis_title="Interaction Component Value",
            height=500,
        )

    if False:
        seed = 42
        n = 1 << 14
        rho = -0.3
        b1, b2, b3 = 3, 2, 10
        cov_mat = np.array([[1, rho], [rho, 1]])

        DataType = "x1x2"
        name = f"{DataType}_{rho:.2f}"

        if DataType == "x1x2":
            X = generate_x("gaussian", n, 2, seed=seed, cov_mat=cov_mat)
            yf = lambda x1, x2: b1 * x1 + b2 * x2 + b3 * x1 * x2
            y = yf(X[:, 0], X[:, 1])

            rho_over2 = rho / (1 + rho**2)
            ref_x1x2 = lambda x: b3 * (
                x[:, 0] * x[:, 1]
                - rho_over2 * (x[:, 0] ** 2 + x[:, 1] ** 2)
                + rho_over2 * (1 - rho**2)
            )
            ref_x1x2_rho = lambda x: np.full(len(x), b3 * rho_over2 * (1 - rho**2))
            ref_x1 = lambda x: b1 * x[:, 0] + b3 * rho_over2 * (x[:, 0] ** 2 - 1)
            ref_x2 = lambda x: b2 * x[:, 1] + b3 * rho_over2 * (x[:, 1] ** 2 - 1)
            ref_bias = b3 * rho

        def plot_pairwise(x1_grid, puremodel, model_dict, ref_x1x2):
            x1_grid_dm = xgb.DMatrix(x1_grid, enable_categorical=True)
            # yyy, univ_contrib, biv_contrib = tuple(k.numpy() for k in puremodel.predict_batch_numerical(x1_grid)[:3])

            biv_fig = go.Figure()
            # biv_fig.add_trace(go.Scatter(x = x1_grid[:, 0], y = biv_contrib[0, :], mode = 'lines', name = 'puregam'))
            biv_fig.add_trace(
                go.Scatter(
                    x=x1_grid[:, 0],
                    y=model_dict[(0, 1)].predict(x1_grid_dm),
                    mode="lines",
                    name="Purified_xgb",
                )
            )
            biv_fig.add_trace(
                go.Scatter(
                    x=x1_grid[:, 0],
                    y=ref_x1x2(x1_grid),
                    mode="lines",
                    name="Ref formula",
                )
            )

            biv_fig.show()
            return biv_fig

        gridn = 100
        plot_start, plot_end = -1.5, 1.5
        x1_vals = np.linspace(plot_start, plot_end, num=gridn)
        x2_vals = np.linspace(plot_start, plot_end, num=gridn)
        x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
        x1_grid = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])

        # my stuff lol
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:squarederror",
            "seed": seed,
            "max_depth": 2,  # moderate depth to avoid overfitting
            "eta": 0.1,  # learning rate
            "verbosity": 1,
        }
        model = new_model([tree], "10.0")
        num_boost_round = 10000
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        X_train = np.array([[5, 2.5], [15, 20]])
        dtrain = xgb.DMatrix(X_train)
        original_prediction = model.predict(dtrain)
        print(f"Original predictions: {original_prediction}")
        result = purify.fANOVA_2D(True, "gph", model, dtrain, True)

        split_x = np.array([10.0])
        split_y = np.array([5.0, 15.0])
        split_conditions_dict = {0: split_x, 1: split_y}
        feature_tuple = (0, 1)
        # end my stuff lol
        plot_pairwise(x1_grid, model, result.purified_model_dict, ref_x1x2)
        rho_grid = x1_grid.copy()
        rho_grid[:, 1] *= rho
        plot_pairwise(rho_grid, model, result.purified_model_dict, ref_x1x2_rho)
