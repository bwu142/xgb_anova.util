# Copyright (c) 2025 Ben Wu <benjamin.x.wu@gmail.com>
# Distributed under the BSD 3-Clause License

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import filter
import purify
from sklearn.metrics import r2_score
import pygam
import purify_first_attempt as fa


##############################
##### PURIFICATION TESTS #####
##############################
def test_independence_2(rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true):
    """
    Check that pairwise functions produce constant output when their features are fixed.
        e.g. For g12(x1,x2): Set x1,x2 to random constants → output is constant regardless of other features.
    """
    # Setup model and data
    model, dtrain, dtest = setup_model(
        rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true
    )
    # Decompose and purify model
    _, purified_components, _ = purify.fANOVA_2D(model, dtrain)

    # Convert D_matrix to Dataframe for mutability
    D_all = get_random_input_set(dtrain, dtest)
    X_all = D_all.get_data()
    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()
    feature_names = D_all.feature_names
    X_df_all = pd.DataFrame(X_all, columns=feature_names)

    # Loop through each feature
    for feature, feature_model in purified_components.items():
        # Testing pairwise-interaction independence only
        if feature in {"x1", "x2", "x3", "x1x2x3"}:
            continue

        # Mutate Dataframe by fixing feature to constant C for all test points
        X_df_fixed = X_df_all.copy()
        C = 50  # constant value

        component_features = [f"x{c}" for c in feature if c.isdigit()]
        for feature in component_features:
            X_df_fixed[feature] = C

        D_fixed = xgb.DMatrix(X_df_fixed, feature_names=feature_names)
        prediction = feature_model.predict(D_fixed)
        assert len(set(prediction)) == 1


def test_accuracy():
    """
    check f is a good fit to calculating metrics on the training and test data set, e.g. r^2
    """
    model, dtrain, dtest = setup_model()

    # Get true labels
    y_train = dtrain.get_label()
    y_test = dtest.get_label()

    # Get predictions
    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)

    # Compute R^2 scores
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"R^2 on training set: {r2_train:.4f}")
    print(f"R^2 on test set: {r2_test:.4f}")

    # Assert reasonable goodness of fit (can be tuned)
    assert r2_train > 0.9, "Model underfits training set"
    assert r2_test > 0.7, "Model may overfit or generalize poorly"


def test_equal_predictions_1(rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true):
    """
    test that f (original model) and g (purified model) are identical for each data point in a sample of random input set from both training and test data
        i.e. check that f(X) = g(X)
    """
    model, dtrain, dtest = setup_model(
        rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true
    )
    random_input_set = get_random_input_set(dtrain, dtest)

    model_prediction = model.predict(random_input_set)
    purified_model = purify.purify_2D(model, dtrain)
    purified_model_prediction = purified_model.predict(random_input_set)

    assert np.allclose(model_prediction, purified_model_prediction, atol=0.1)


def test_equal_predictions_2(rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true):
    """
    test that prediction from purify_2D (entire purified model) equals prediction from summing up predictions from components of purified_model
        i.e. g(X) = g0 + g1(x1) + g2(x2) + g12(x1, x2) + g13(x1, x3) + g23(x2, x3) + g123(x1, x2, x3)
    """
    model, dtrain, dtest = setup_model(
        rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true
    )
    random_input_set = get_random_input_set(dtrain, dtest)

    # Purified model total prediction
    purified_model = purify.purify_2D(model, dtrain)
    purified_model_prediction = purified_model.predict(random_input_set)

    # Purified model sum-to-total prediction
    _, purified_model_dict, bias = purify.fANOVA_2D(model, dtrain)
    num_samples = random_input_set.num_row()
    purified_prediction_sum = np.zeros(num_samples)

    # Make into new function
    for purified_model in purified_model_dict.values():
        purified_prediction_sum += purified_model.predict(random_input_set)
    purified_prediction_sum += bias

    print(purified_model_prediction[:10])
    print(purified_prediction_sum[:10])
    assert np.allclose(
        purified_prediction_sum, purified_model_prediction, atol=0.01, rtol=0
    )


def test_equal_predictions_3(rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true):
    """
    check f(X) = g0 + g1(x1) + g2(x2) + g12(x1, x2) + g13(x1, x3) + g23(x2, x3) + g123(x1, x2, x3)
    """
    model, dtrain, dtest = setup_model(
        rho_val, b1_val, b2_val, b3_val, X_dataframe, y_true
    )
    random_input_set = get_random_input_set(dtrain, dtest)
    model_prediction = model.predict(random_input_set)  # model prediction

    _, purified_model_dict, bias = purify.fANOVA_2D(model, dtrain)

    num_samples = random_input_set.num_row()
    purified_prediction_sum = np.zeros(num_samples)

    for purified_model in purified_model_dict.values():
        purified_prediction_sum += purified_model.predict(random_input_set)
    purified_prediction_sum += bias

    print(purified_prediction_sum[:10])
    print(model_prediction[:10])
    assert np.allclose(purified_prediction_sum, model_prediction, atol=0.01)


def test_independence(X, y):
    """
    Check that each purified, main-effect component depends only on its intended features.
        e.g. For g1(x1): Fix x1 and randomize other features → output remains constant.
    """
    # Setup model and data
    model, dtrain, dtest = setup_model(X, y)

    # Decompose and purify model
    _, purified_components, _ = purify.fANOVA_2D(model, dtrain)

    # Convert D_matrix to Dataframe for mutability
    D_all = get_random_input_set(dtrain, dtest)
    X_all = D_all.get_data()
    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()
    feature_names = D_all.feature_names
    X_df_all = pd.DataFrame(X_all, columns=feature_names)

    # Loop through each feature
    for feature_tuple, feature_model in purified_components.items():
        # Testing main effect independence only
        if len(feature_tuple) == 1:
            # Mutate Dataframe by fixing feature to constant C for all test points
            X_df_fixed = X_df_all.copy()
            C = 50  # constant value
            feature_index = feature_tuple[0]

            X_df_fixed[feature_index] = C
            D_fixed = xgb.DMatrix(X_df_fixed, feature_names=feature_names)
            prediction = feature_model.predict(D_fixed)
            assert len(set(prediction)) == 1
        elif len(feature_tuple) == 2:
            # Mutate Dataframe by fixing feature to constant C for all test points
            X_df_fixed = X_df_all.copy()
            C = 50  # constant value

            for feature in feature_tuple:
                X_df_fixed[feature] = C

            D_fixed = xgb.DMatrix(X_df_fixed, feature_names=feature_names)
            prediction = feature_model.predict(D_fixed)
            assert len(set(prediction)) == 1


####################
#### MAIN TESTS ####
####################
def setup_model(X, y, num_trees=100):
    # Setup
    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 2,
        "learning_rate": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_trees,  # Equivalent to n_estimators
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=True,
    )
    return model, dtrain, dtest


def get_random_input_set(dtrain, dtest):
    """Get random input set (DMatrix) from dtrain and dtest"""
    # Get numpy arrays from DMatrix
    X_train = dtrain.get_data()
    X_test = dtest.get_data()

    if hasattr(X_train, "toarray"):  # Handle sparse matrices
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    # Combine and shuffle
    X_combined = np.vstack([X_train, X_test])
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(X_combined))

    # Take a random subset, e.g., 100 samples
    sample_size = min(100, len(X_combined))
    X_sample = X_combined[indices[:sample_size]]

    # Include feature names to avoid prediction errors
    return xgb.DMatrix(X_sample, feature_names=dtrain.feature_names)


def test_equal_predictions(X, y, num_trees=100):
    """
    check f(X) = g0 + g1(x1) + g2(x2) + g12(x1, x2) + g13(x1, x3) + g23(x2, x3) + g123(x1, x2, x3)
    """
    model, dtrain, dtest = setup_model(X, y, num_trees)
    # random_input_set = get_random_input_set(dtrain, dtest)
    # model_prediction = model.predict(random_input_set)

    purified_model, purified_model_dict, bias = purify.fANOVA_2D(model, dtrain)

    # Original Model
    model_prediction = model.predict(dtrain)
    # Purified Model
    purified_model_prediction = purified_model.predict(dtrain)
    # Purified Prediction Sum
    num_samples = dtrain.num_row()
    purified_prediction_sum = np.zeros(num_samples)
    for purified_model in purified_model_dict.values():
        purified_prediction_sum += purified_model.predict(dtrain)
    purified_prediction_sum += bias

    # Comparisons
    print(f"original_model_prediction: {model_prediction}")
    print(f"purified_model_prediction: {purified_model_prediction}")
    print(f"purified_subset_sum_prediction: {purified_prediction_sum}")

    assert np.allclose(
        model_prediction, purified_model_prediction, atol=1e-5, rtol=0.0
    ), "model_prediction vs. purified_model_prediction not precise!"
    assert np.allclose(
        model_prediction, purified_prediction_sum, atol=1e-5, rtol=0.0
    ), "model_prediction vs. purified_prediction_sum not precise!"
    assert np.allclose(
        purified_model_prediction, purified_prediction_sum, atol=1e-5, rtol=0.0
    ), "purified_model_prediction vs. purified_prediction_sum not precise!"


def plot_against_true(X, y, b1, b2, b3, rho, plot_start=-1.5, plot_end=1.5):
    # Fit and purify model
    model, dtrain, dtest = setup_model(X, y)
    purified_model, purified_model_dict, bias = purify.fANOVA_2D(model, dtrain)

    # Generate meshgrid
    x = np.linspace(plot_start, plot_end, 100)
    x1, x2 = np.meshgrid(x, x)
    grid = np.column_stack((x1.ravel(), x2.ravel()))

    # True purified interaction f12
    term1 = x1 * x2
    term2 = (rho / (1 + rho**2)) * (x1**2 + x2**2)
    term3 = rho * (1 - rho**2) / (1 + rho**2)
    z_true = b3 * (term1 - term2 + term3)

    # Predicted purified interaction f12 from XGBoost model
    dgrid = xgb.DMatrix(grid)
    z_pred = purified_model_dict[(0, 1)].predict(dgrid).reshape(x1.shape)

    # Plot both surfaces
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            z=z_true, x=x1, y=x2, colorscale="Viridis", opacity=0.6, name="True f₁₂"
        )
    )
    fig.add_trace(
        go.Surface(
            z=z_pred,
            x=x1,
            y=x2,
            colorscale="Cividis",
            opacity=0.6,
            name="Purified f₁₂ (XGBoost)",
        )
    )

    fig.update_layout(
        title="True vs XGBoost Purified f₁₂(x₁, x₂)",
        scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f₁₂"),
        width=800,
        height=700,
    )
    fig.show()


def plot_components_ben(purified_model_dict, feature_tuple, x_vals, y_ref_func):
    model = purified_model_dict[feature_tuple]

    if feature_tuple == (0,):
        # f1
        X = np.zeros((len(x_vals), 2))
        X[:, 0] = x_vals
        dX = xgb.DMatrix(X)

        y_model = model.predict(dX)
        y_true = y_ref_func(x_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_true, mode="lines", name="True f1"))
        fig.add_trace(go.Scatter(x=x_vals, y=y_model, mode="lines", name="Purified f1"))
        fig.update_layout(
            title="f1: Purified vs True", xaxis_title="x1", yaxis_title="value"
        )
        fig.show()

    elif feature_tuple == (1,):
        # f2
        X = np.zeros((len(x_vals), 2))
        X[:, 1] = x_vals
        dX = xgb.DMatrix(X)

        y_model = model.predict(dX)
        y_true = y_ref_func(x_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_true, mode="lines", name="True f2"))
        fig.add_trace(go.Scatter(x=x_vals, y=y_model, mode="lines", name="Purified f2"))
        fig.update_layout(
            title="f2: Purified vs True", xaxis_title="x2", yaxis_title="value"
        )
        fig.show()

    elif feature_tuple == (0, 1):
        # f12
        x1, x2 = x_vals
        X_grid = np.column_stack([x1.ravel(), x2.ravel()])
        dX = xgb.DMatrix(X_grid)

        y_model = model.predict(dX).reshape(x1.shape)
        y_true = y_ref_func(x1, x2)

        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                z=y_true,
                x=x1[0],
                y=x2[:, 0],
                colorscale="Blues",
                contours_coloring="lines",
                name="True f12",
            )
        )
        fig.add_trace(
            go.Contour(
                z=y_model,
                x=x1[0],
                y=x2[:, 0],
                colorscale="Reds",
                contours_coloring="lines",
                name="Purified f12",
            )
        )
        fig.update_layout(
            title="f12: Purified vs True", xaxis_title="x1", yaxis_title="x2"
        )
        fig.show()


def plot_components(y_pred_func, x_vals, y_ref_func):
    """
    y_pred_func: submodel from purified_model_dict

    x_vals:
    y_ref_func: lambda x1, x2: b1 * x1 + b2 * x2 + b3 * x1 * x2
    """
    # Construct input data for purified model prediction
    X1 = np.zeros((len(x_vals), 2))
    X1[:, 0] = x_vals
    X2 = np.zeros((len(x_vals), 2))
    X2[:, 1] = x_vals
    dX1 = xgb.DMatrix(X1)
    dX2 = xgb.DMatrix(X2)

    model_pred = y_pred_func.predict(dX1)
    ref_pred = y_ref_func(x_vals)

    true_f1

    # True component functions (centered)
    true_f1 = (
        b1 * x_vals + b3 * rho / (1 + rho**2) * x_vals**2 - b3 * rho / (1 + rho**2)
    )
    true_f2 = (
        b2 * x_vals + b3 * rho / (1 + rho**2) * x_vals**2 - b3 * rho / (1 + rho**2)
    )

    # Construct input data for purified model prediction
    X1 = np.zeros((len(x_vals), 2))
    X1[:, 0] = x_vals
    X2 = np.zeros((len(x_vals), 2))
    X2[:, 1] = x_vals
    dX1 = xgb.DMatrix(X1)
    dX2 = xgb.DMatrix(X2)

    # Predicted component functions
    pred_f1 = purified_model_dict[(0,)].predict(dX1)
    pred_f2 = purified_model_dict[(1,)].predict(dX2)

    # Plot f1
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_vals, y=true_f1, mode="lines", name="True f1"))
    fig1.add_trace(
        go.Scatter(
            x=x_vals,
            y=pred_f1,
            mode="lines",
            name="Purified f1",
            line=dict(dash="dash"),
        )
    )
    fig1.update_layout(
        title="True vs Purified f1(x1)", xaxis_title="x1", yaxis_title="f1(x1)"
    )

    # Plot f2
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_vals, y=true_f2, mode="lines", name="True f2"))
    fig2.add_trace(
        go.Scatter(
            x=x_vals,
            y=pred_f2,
            mode="lines",
            name="Purified f2",
            line=dict(dash="dash"),
        )
    )
    fig2.update_layout(
        title="True vs Purified f2(x2)", xaxis_title="x2", yaxis_title="f2(x2)"
    )

    fig1.show()
    fig2.show()

    return fig1, fig2


def test_purified_mean_zero(X, y, epsilon=1, C=0):
    """
    Check that f(x1, x2) component has mean 0 over a slice of data where one feature is ~ constant
    """
    ##### SETUP #####
    model, dtrain, dtest = setup_model(X, y)
    _, purified_components, _ = purify.fANOVA_2D(model, dtrain)

    # Convert to DataFrame
    X_train = dtrain.get_data()
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    n_points, n_features = X_train.shape

    ##### CONSTRUCT TEST SET WITH FIXED FEATURE #####

    for varied_index in range(n_features):
        for fixed_index in range(n_features):
            if fixed_index == varied_index:
                continue
            mask = np.abs(X_train[:, fixed_index] - C) < epsilon
            if not np.any(mask):
                print(f"No points found with x{fixed_index+1} ≈ {C}")
                continue

            X_slice = X_train[mask]
            d_slice = xgb.DMatrix(X_slice)

            tuple_key = tuple(sorted([varied_index, fixed_index]))
            pred = purified_components[tuple_key].predict(d_slice)
            mean_val = np.mean(pred)
            print(
                f"mean f_{varied_index},{fixed_index} with x{fixed_index} ≈ {C}: {mean_val:.4f}"
            )
            assert np.abs(mean_val) < 0.1


# should just take in function and
def test_purity(X, y):
    """
    for training data, check the means of component functions are zero. Check the means with the test data - the means should be close to zero.
    """

    TRAIN_ATOL = 1e-2  # very tight for training data
    TEST_ATOL = 1e-2  # looser for unseen data

    def _zero_mean_assert(vec, atol, name, split):
        """Assert that `vec` is mean-zero within tolerance."""
        m = vec.mean()
        print(m)
        assert np.allclose(
            m, 0.0, atol=atol
        ), f"{name} not pure on {split} set: mean={m:.3e}, tol={atol}"

    # Build data & model
    model, dtrain, dtest = setup_model(X, y)

    # Decompose
    _, purified_model_dict, _ = purify.fANOVA_2D(
        model, dtrain
    )  # returns {name: Booster}, bias

    # Loop through every component (skip bias)
    for feature_tuple, booster in purified_model_dict.items():
        # Training set
        pred_train = booster.predict(dtrain)
        _zero_mean_assert(pred_train, TRAIN_ATOL, feature_tuple, "train")

        # # Test set
        # pred_test = booster.predict(dtest)
        # _zero_mean_assert(pred_test, TEST_ATOL, name, "test")


def test_plot_1(X, y):
    """
    Plot uniform distrubition on 1 factor (main effects vs. grid)
    This should just plot
    """
    ##### SETUP #####
    model, dtrain, dtest = setup_model(X, y)
    _, purified_components, bias = purify.fANOVA_2D(model, dtrain)

    # Convert to DataFrame
    X_sample = dtrain.get_data()
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()
    n_features = X_sample.shape[1]

    ##### PLOT BIAS AND MAIN EFFECTS OVER UNIFORM GRID #####
    x_range = np.linspace(-50, 50, 300)
    fig = go.Figure()

    # Bias
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[bias] * len(x_range),
            mode="lines",
            name="base_score",
            line=dict(dash="dash", width=2, color="black"),
        )
    )

    # Main Effects
    for feature_tuple, feature_model in purified_components.items():
        if len(feature_tuple) != 1:
            continue
        feature_index = feature_tuple[0]
        X_grid = np.zeros((len(x_range), n_features))
        X_grid[:, feature_index] = x_range

        dgrid = xgb.DMatrix(X_grid)
        y_pred = feature_model.predict(dgrid)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode="lines",
                name=f"{feature_tuple}",
            )
        )

    # Layout
    fig.update_layout(
        title="fANOVA 1st-Order Main Effects (Uniform Grid)",
        xaxis_title="Feature Value",
        yaxis_title="f_i(x_i) --> Main Effect Prediction",
        template="plotly_white",
        width=900,
        height=500,
        legend=dict(x=0.01, y=0.99),
    )
    fig.show()


def test_plot_all(
    X_dataframe, y_true, epsilon=0.05, C=0, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5
):
    """
    Plot Main Effects and Interactions (fixing one feature) over data distribution
    """
    ##### SETUP #####
    model, dtrain, dtest = setup_model(X_dataframe, y_true)
    _, purified_components, bias = purify.fANOVA_2D(model, dtrain)

    # Convert to DataFrame
    X_sample = dtrain.get_data()
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()
    n_points, n_features = X_sample.shape

    ##### PLOT BIAS AND 1-FEATURE AGAINST DISTRIBUTION #####

    # Plot bias and main effects
    fig = go.Figure()
    x_vals_all = X_sample[:, 0]
    fig.add_trace(
        go.Scatter(
            x=np.sort(x_vals_all),
            y=[bias] * len(x_vals_all),
            mode="lines",
            name="bias (base_score)",
            line=dict(dash="dash", width=2, color="black"),
        )
    )

    for feature_tuple, feature_model in purified_components.items():
        # Testing main effect independence only
        if len(feature_tuple) == 1:
            feature_index = feature_tuple[0]

            y_vals = feature_model.predict(dtrain)
            x_vals = X_sample[:, feature_index]

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=f"{feature_tuple}",
                    # line=dict(dash="dash", width=2),
                )
            )

    # Final layout
    fig.update_layout(
        title="fANOVA Bias & 1st-Order Main Effects (Against Distribution)",
        xaxis_title="Feature Value",
        yaxis_title="Main Effect Prediction",
        template="plotly_white",
        width=800,
        height=500,
        legend=dict(x=0.01, y=0.99),
    )
    fig.show()

    ##### PLOT 2-FEATURE AGAINST DISTRIBUTION (HOLDING ONE-VAR CONSTANT) #####
    # Grid settings
    fixed_vals = [-1, -0.5, 0, 0.5, 1]
    for feature_tuple, feature_model in purified_components.items():
        # Interaction terms only
        if len(feature_tuple) != 2:
            continue

        index_1, index_2 = feature_tuple
        name_1, name_2 = f"x{index_1 + 1}", f"x{index_2 + 1}"

        # Fix x1, vary x2
        fig1 = go.Figure()
        for C in fixed_vals:
            mask = np.abs(X_sample[:, index_1] - C) < epsilon
            if mask.sum() < 10:  # too few points → skip
                print("Not enough points!")
                continue
            X_slice = X_sample[mask]
            x2_vals = X_slice[:, index_2]
            y_pred = feature_model.predict(xgb.DMatrix(X_slice))

            sorted_index = np.argsort(x2_vals)
            x_sorted = x2_vals[sorted_index]
            y_sorted = y_pred[sorted_index]

            fig1.add_trace(
                go.Scatter(x=x_sorted, y=y_sorted, mode="markers", name=f"{name_1}≈{C}")
            )

        fig1.update_layout(
            title=f"{name_1},{name_2}: vary {name_2}  |  {name_1}≈C (ε={epsilon})",
            xaxis_title=name_2,
            yaxis_title=f"f_{name_1},{name_2}",
            template="plotly_white",
            width=900,
            height=500,
        )
        fig1.update_xaxes(range=[x_min, x_max])
        fig1.update_yaxes(range=[y_min, y_max])
        fig1.show()

        # Fix x2, vary x1
        fig2 = go.Figure()
        for C in fixed_vals:
            mask = np.abs(X_sample[:, index_2] - C) < epsilon
            if mask.sum() < 10:  # too few points → skip
                print("Not enough points!")
                continue
            X_slice = X_sample[mask]
            x1_vals = X_slice[:, index_1]
            y_pred = feature_model.predict(xgb.DMatrix(X_slice))

            sorted_index = np.argsort(x1_vals)
            x_sorted = x1_vals[sorted_index]
            y_sorted = y_pred[sorted_index]

            fig2.add_trace(
                go.Scatter(x=x_sorted, y=y_sorted, mode="markers", name=f"{name_2}≈{C}")
            )
        fig2.update_layout(
            title=f"{name_1},{name_2}: vary {name_1}  |  {name_2}≈C (ε={epsilon})",
            xaxis_title=name_1,
            yaxis_title=f"f_{name_1},{name_2}",
            template="plotly_white",
            width=900,
            height=500,
        )
        fig2.update_xaxes(range=[x_min, x_max])
        fig2.update_yaxes(range=[y_min, y_max])
        fig2.show()


###### COMPARISONS ######
def plot_pairwise(model, dataset, rho, b3, plot_range=(-1.5, 1.5)):
    """
    Generates two comparison plots:
    1. Diagonal (x1 = x2) comparison
    2. Correlation structure (x2 = rho*x1) comparison
    """
    # Purify model
    result = fa.fANOVA_2D(False, model, xgb.DMatrix(dataset), True, "old")
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
    x_vals = np.linspace(*plot_range, num=gridn)

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
    fig1.update_layout(
        title="Diagonal Comparison (x1 = x2)",
        xaxis_title="x value",
        yaxis_title="Interaction Component",
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
    fig1.add_trace(
        go.Scatter(
            x=x_vals,
            y=orig_filt_model.predict(rho_dm),
            mode="lines",
            name="Original (Unpurified)",
            line=dict(color="red"),
        )
    )
    fig2.update_layout(
        title=f"Correlation Structure (x2 = {rho:.2f}x1)",
        xaxis_title="x1 value",
        yaxis_title="Interaction Component",
    )
    fig2.show()

    return fig1, fig2


if __name__ == "__main__":
    # # TEST 1
    # n = 1 << 16
    # rho_val = 0  # 0, .5, .9, .999, 1
    # b1, b2, b3 = 3, 2, 10
    # cov_mat = np.identity(2)
    # cov_mat[0, 1] = cov_mat[1, 0] = rho_val
    # X = np.random.multivariate_normal(np.zeros(2), cov_mat, n)
    # yf = lambda x1, x2: b1 * x1 + b2 * x2 + b3
    # y_true = yf(X[:, 0], X[:, 1])

    # test_equal_predictions(X, y_true)
    # test_independence(X, y_true)
    # test_purified_mean_zero(X, y_true, 0.001, 0) # works for all!
    # test_purity(X, y_true)
    # test_plot_1(X, y_true)
    # test_plot_all(X, y_true, 0.05, 0, -3, 3, -2, 2)

    # # TEST 2
    # # Parameters
    # n = 1 << 16
    # rho_val = 0  # Correlation between x1 and x2

    # # Covariance matrix for bivariate normal
    # cov_mat = np.identity(2)
    # cov_mat[0, 1] = cov_mat[1, 0] = rho_val

    # # Generate standard bivariate normal data
    # X_raw = np.random.multivariate_normal(mean=np.zeros(2), cov=cov_mat, size=n)

    # # Shift both x1 and x2 to ensure x1 * x2 > 0
    # shift = 5.0  # Empirically safe for most values to ensure product > 0
    # X = X_raw + shift

    # # Calculate target: y = 2 + log(x1 * x2)
    # product = X[:, 0] * X[:, 1]
    # assert np.all(product > 0), "Some x1 * x2 values are not positive!"

    # y_true = 2 + np.log(product)

    # test_equal_predictions(X, y_true)
    # test_independence(X, y_true)
    # test_purified_mean_zero(X, y_true, 0.0001, 0)
    # test_purity(X, y_true)
    # test_plot_1(X, y_true)
    # test_plot_all(X, y_true, 0.5, 5)

    # # TEST 3
    # n = 1 << 18  # number of data points
    # rho_val = 0  # correlation between x1 and x2 (can try 0, 0.5, 1.0)
    # b1, b2, b3 = 3.0, 2.0, 10.0  # coefficients

    # # Covariance matrix
    # cov_mat = np.identity(2)
    # cov_mat[0, 1] = cov_mat[1, 0] = rho_val

    # # Sample from multivariate normal
    # X = np.random.multivariate_normal(np.zeros(2), cov_mat, size=n)
    # x1, x2 = X[:, 0], X[:, 1]

    # # Define target
    # y_true = b1 * x1 + b2 * x2 + b3 * x1 * x2 + 2

    # test_equal_predictions(X, y_true)
    # test_independence(X, y_true)
    # test_purified_mean_zero(X, y_true, 0.001, 0)
    # test_purity(X, y_true)
    # test_plot_1(X, y_true)
    # test_plot_all(X, y_true, 0.05, 0, -2, 2, -2, 2)

    ####### TESTS NOW ######
    # seed = 42
    # n = 1 << 16
    # rho = 0.5
    # b1, b2, b3 = 3, 2, 10
    # cov_mat = np.identity(2)
    # cov_mat[0, 1] = cov_mat[1, 0] = rho
    # DataType = "x1x2"

    # if DataType == "x1x2":
    #     np.random.seed(seed)
    #     X = np.random.multivariate_normal(np.zeros(2), cov_mat, n)
    #     yf = lambda x1, x2: b1 * x1 + b2 * x2 + b3 * x1 * x2
    #     y = yf(X[:, 0], X[:, 1])
    #     plot_start, plot_end = -1.5, 1.5
    #     Num_identity = 0
    # elif DataType == "x1x2L":
    #     np.random.seed(seed)
    #     X = np.random.multivariate_normal(np.zeros(2), cov_mat, n)
    #     yf = lambda x1, x2: b1 * x1 + b2 * x2 + b3
    #     y = yf(X[:, 0], X[:, 1])
    #     plot_start, plot_end = -1.5
    #     Num_identity = 0
    # elif DataType == "x1x2x3":
    #     np.random.seed(seed)
    #     cov_mat3 = np.full((3, 3), rho)
    #     np.fill_diagonal(cov_mat3, 1)

    # model, dtrain, dtest = setup_model(X, y, 100)
    # my_f = purify.fANOVA_2D(False, model, dtrain, True, "test_py")
    # x_vals = np.linspace(-1.5, 1.5, 200)
    # plot_components_ben(my_f.purified_model_dict, (0,), x_vals, yf)

    if False:
        seed = 42
        n = 1 << 14
        rho = 0  # 0, -.3, .7
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
        result = fa.fANOVA_2D(True, model, dtrain, True, "fa")

        # Get predictions
        diag_grid_dm = xgb.DMatrix(diag_grid)
        pred_pure = result.purified_model_dict[(0, 1)].predict(diag_grid_dm)
        true_pure = ref_x1x2(diag_grid)

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_pure,
                mode="lines",
                name="Purified prediction",
                line=dict(color="blue"),
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

        fig.show()

    # Generate correlated data
    seed = 42
    n = 1 << 14
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

    # plot_diag(model, X)
    plot_pairwise(model, X, rho, b3)
