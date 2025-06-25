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


##### CONTOUR PLOT HELPER FUNCTIONS #####
def plot_2D_one_feature(model, feature_names, equation):
    """
    model: xgb.Booster from xgb.train()
    feature_names: list of feature names, e.g. ['x1', 'x2']
    equation: string, for plot title
    """
    # Generate x_i vectors
    x_step = np.arange(0, 100, 0.1)
    x_random = np.random.uniform(0, 100, size=len(x_step))

    # Pandas DataFrames
    X_test1 = pd.DataFrame({feature_names[0]: x_step, feature_names[1]: x_random})
    X_test2 = pd.DataFrame({feature_names[0]: x_random, feature_names[1]: x_step})

    # Convert to DMatrix for xgb.train
    dtest1 = xgb.DMatrix(X_test1)
    dtest2 = xgb.DMatrix(X_test2)

    # Get predicted values
    model_x1 = purify.filter_save_load(model, (0,))
    model_x2 = purify.filter_save_load(model, (1,))
    y_pred_x1 = model_x1.predict(dtest1)
    y_pred_x2 = model_x2.predict(dtest2)

    # Plot
    plot = go.Figure()

    plot.add_trace(
        go.Scatter(
            x=X_test1[feature_names[0]],
            y=y_pred_x1,
            mode="lines",
            name=f"z_pred_{feature_names[0]}",
            line=dict(color="red"),
        )
    )

    plot.add_trace(
        go.Scatter(
            x=X_test2[feature_names[1]],
            y=y_pred_x2,
            mode="lines",
            name=f"z_pred_{feature_names[1]}",
            line=dict(color="blue"),
        )
    )

    plot.update_layout(
        title=f"{equation}",
        xaxis_title=f"{feature_names[0]}, {feature_names[1]}",
        yaxis_title="Predicted z",
    )

    plot.show()


def plot_countour_one_feature(
    model, feature_names, equation, start_val, stop_val, step_val
):
    x1 = np.arange(0, 100, 0.1)
    x2 = np.arange(0, 100, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    all_pairs = np.column_stack([X1.flatten(), X2.flatten()])
    X_test = pd.DataFrame(
        {feature_names[0]: all_pairs[:, 0], feature_names[1]: all_pairs[:, 1]}
    )
    dtest = xgb.DMatrix(X_test)

    model_x1 = purify.filter_save_load(model, (0,))
    model_x2 = purify.filter_save_load(model, (1,))
    z_pred_x1 = model_x1.predict(dtest)
    z_pred_x2 = model_x2.predict(dtest)
    Z_Pred_x1 = z_pred_x1.reshape(len(x2), len(x1))
    Z_Pred_x2 = z_pred_x2.reshape(len(x2), len(x1))

    # Contour for x1
    fig1 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1,
            x=x1,
            y=x2,
            colorscale="sunset",
            contours=dict(
                start=start_val,
                end=stop_val,
                size=step_val,
                coloring="heatmap",
                showlabels=True,
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )
    fig1.update_layout(
        title=f"Z_Pred_{feature_names[0]}: {equation}",
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
    )
    fig1.show()

    # Contour for x2
    fig2 = go.Figure(
        data=go.Contour(
            z=Z_Pred_x2,
            x=x1,
            y=x2,
            colorscale="sunset",
            contours=dict(
                start=start_val,
                end=stop_val,
                size=step_val,
                coloring="heatmap",
                showlabels=True,
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )
    fig2.update_layout(
        title=f"Z_Pred_{feature_names[1]}: {equation}",
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
    )
    fig2.show()


def plot_contour_two_features(
    model, feature_names, equation, start_val, stop_val, step_val
):
    x1 = np.arange(0, 100, 0.1)
    x2 = np.arange(0, 100, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    all_pairs = np.column_stack([X1.flatten(), X2.flatten()])
    X_test = pd.DataFrame(
        {feature_names[0]: all_pairs[:, 0], feature_names[1]: all_pairs[:, 1]}
    )
    dtest = xgb.DMatrix(X_test)

    model_x1x2 = purify.filter_save_load(model, (0, 1))
    z_pred_x1x2 = model_x1x2.predict(dtest)
    Z_Pred_x1x2 = z_pred_x1x2.reshape(len(x2), len(x1))

    fig = go.Figure(
        data=go.Contour(
            z=Z_Pred_x1x2,
            x=x1,
            y=x2,
            colorscale="sunset",
            contours=dict(
                start=start_val,
                end=stop_val,
                size=step_val,
                coloring="heatmap",
                showlabels=True,
            ),
            line=dict(width=2),
            colorbar=dict(title="Z value"),
        )
    )
    fig.update_layout(
        title=f"Z_Pred_{feature_names[0]}{feature_names[1]}: {equation}",
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
    )
    fig.show()

    """CENTERING"""
    # Generate simple data
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 1000)
    y = 10 * x1 + 2
    X = pd.DataFrame({"x1": x1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train with xgb.train
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {
        "max_depth": 1,
        "eta": 1.0,
        "objective": "reg:squarederror",
        "base_score": 0.8,
        "seed": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=1000)

    # Get original predictions and mean
    y_pred = model.predict(dtest)
    mean_pred = np.mean(y_pred)
    original_base_score = filter.get_bias(model)
    new_base_score = round(float(mean_pred) + original_base_score, 6)
    new_base_score_str = "{:.6f}".format(new_base_score)

    # Call save_load_new_trees to add a tree with leaf value -mean_pred and update base_score
    new_model = filter.save_load_new_trees(
        model,
        leaf_val=-mean_pred,
        base_score=new_base_score_str,
        num_new_trees=1,
        output_file_name="model_one_var_centered.json",
        folder="loaded_models",
    )

    # Predict with the new model
    y_pred_centered = new_model.predict(dtest)

    # Assert predictions are close
    assert np.allclose(
        np.round(y_pred, 3), np.round(y_pred_centered, 3), atol=0.1
    ), "TEST_SAVE_LOAD_NEW_TREES FAILED"


##############################
##### PURIFICATION TESTS #####
##############################
def setup_model():
    # Setup
    np.random.seed(42)
    x1 = np.random.uniform(0, 100, 10)
    x2 = np.random.uniform(0, 100, 10)
    x3 = np.random.uniform(0, 100, 10)
    y = 10 * x1 + 2 * x2 + 3 * x1 * x2 + 5 + 4 * x3

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
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
        num_boost_round=10000,  # Equivalent to n_estimators
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


def test_equal_predictions_1():
    """
    test that f (original model) and g (purified model) are identical for each data point in a sample of random input set from both training and test data
        i.e. check that f(X) = g(X)
    """
    model, dtrain, dtest = setup_model()
    random_input_set = get_random_input_set(dtrain, dtest)

    model_prediction = model.predict(random_input_set)
    purified_model = purify.purify_2D(model, dtrain)
    purified_model_prediction = purified_model.predict(random_input_set)

    assert np.allclose(model_prediction, purified_model_prediction, atol=0.1)


def test_equal_predictions_2():
    """
    test that prediction from purify_2D (entire purified model) equals prediction from summing up predictions from components of purified_model
        i.e. g(X) = g0 + g1(x1) + g2(x2) + g12(x1, x2) + g13(x1, x3) + g23(x2, x3) + g123(x1, x2, x3)
    """
    model, dtrain, dtest = setup_model()
    random_input_set = get_random_input_set(dtrain, dtest)

    # Purified model total prediction
    purified_model = purify.purify_2D(model, dtrain)
    purified_model_prediction = purified_model.predict(random_input_set)

    # Purified model sum-to-total prediction
    purified_model_dict, bias = purify.fANOVA_2D(model, dtrain)
    num_samples = random_input_set.num_row()
    purified_prediction_sum = np.zeros(num_samples)

    for purified_model in purified_model_dict.values():
        purified_prediction_sum += purified_model.predict(random_input_set)
    purified_prediction_sum += bias

    assert np.allclose(purified_prediction_sum, purified_model_prediction, atol=0.1)


def test_equal_predictions_3():
    """
    check f(X) = g0 + g1(x1) + g2(x2) + g12(x1, x2) + g13(x1, x3) + g23(x2, x3) + g123(x1, x2, x3)
    """
    model, dtrain, dtest = setup_model()
    random_input_set = get_random_input_set(dtrain, dtest)
    model_prediction = model.predict(random_input_set)  # model prediction

    purified_model_dict, bias = purify.fANOVA_2D(model, dtrain)

    num_samples = random_input_set.num_row()
    purified_prediction_sum = np.zeros(num_samples)

    for purified_model in purified_model_dict.values():
        purified_prediction_sum += purified_model.predict(random_input_set)
    purified_prediction_sum += bias

    assert np.allclose(purified_prediction_sum, model_prediction, atol=0.1)


def test_independence_1():
    """
    Check that each purified, main-effect component depends only on its intended features.
        e.g. For g1(x1): Fix x1 and randomize other features → output remains constant.
    """
    # Setup model and data
    model, dtrain, dtest = setup_model()
    # Decompose and purify model
    purified_components, _ = purify.fANOVA_2D(model, dtrain)

    # Convert D_matrix to Dataframe for mutability
    D_all = get_random_input_set(dtrain, dtest)
    X_all = D_all.get_data()
    if hasattr(X_all, "toarray"):
        X_all = X_all.toarray()
    feature_names = D_all.feature_names
    X_df_all = pd.DataFrame(X_all, columns=feature_names)

    # Loop through each feature
    for feature, feature_model in purified_components.items():
        # Testing main effect independence only
        if feature not in {"x1", "x2", "x3"}:
            continue
        # Mutate Dataframe by fixing feature to constant C for all test points
        X_df_fixed = X_df_all.copy()
        C = 50  # constant value

        X_df_fixed[feature] = C
        print(X_df_fixed)
        D_fixed = xgb.DMatrix(X_df_fixed, feature_names=feature_names)
        prediction = feature_model.predict(D_fixed)
        assert len(set(prediction)) == 1


def test_independence_2():
    """
    Check that pairwise functions produce constant output when their features are fixed.
        e.g. For g12(x1,x2): Set x1,x2 to random constants → output is constant regardless of other features.
    """
    # Setup model and data
    model, dtrain, dtest = setup_model()
    # Decompose and purify model
    purified_components, _ = purify.fANOVA_2D(model, dtrain)

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


def test_plot_1():
    """
    Display g0, and line charts for 1st order functions
    """
    # Setup model and data
    model, dtrain, dtest = setup_model()
    # Decompose and purify model
    purified_components, bias = purify.fANOVA_2D(model, dtrain)

    # Plot setup
    x_step = np.arange(0, 100, 0.1)
    plot = go.Figure()

    # Add bias
    plot.add_trace(
        go.Scatter(
            x=x_step,
            y=[bias] * len(x_step),
            mode="lines",
            name="base_score",
            line=dict(dash="dash", width=2),
        )
    )

    # Plot x_step vs main_effects
    for feature, feature_model in purified_components.items():
        # Testing main effect independence only
        if feature not in {"x1", "x2", "x3"}:
            continue

        plot.add_trace(
            go.Scatter(
                x=x_step,
                y=feature_model.predict(dtest),
                mode="lines",
                name=feature,
                line=dict(dash="dash", width=2),
            )
        )

    plot.show()


def test_plot_2():
    """
    Plot 2D charts for x12(x1, x2) etc. preferably three sets of ridge charts by varying x1 and x2 in turn and along the diagonal x1 = x2
    """


def test_purity():
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
    model, dtrain, dtest = setup_model()

    # Decompose
    comp_dict, _ = purify.fANOVA_2D(model, dtrain)  # returns {name: Booster}, bias

    # Loop through every component (skip bias)
    for name, booster in comp_dict.items():
        # Training set
        pred_train = booster.predict(dtrain)
        _zero_mean_assert(pred_train, TRAIN_ATOL, name, "train")

        # Test set
        pred_test = booster.predict(dtest)
        _zero_mean_assert(pred_test, TEST_ATOL, name, "test")


if __name__ == "__main__":
    pass
    # 6
    # 7
