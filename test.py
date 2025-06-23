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


##### PURIFICATION TESTS #####
def test_purification_1():
    """5 Nodes"""
    pass


if __name__ == "__main__":
    pass
