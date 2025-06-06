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

##### CONTOUR PLOT HELPER FUNCTIONS #####


def plot_linear_one_var(model, equation):
    """ """
    # Generate x_i vectors
    x_step = np.arange(0, 100, 0.1)  # 0.1 step, not including 100
    x_random = np.random.uniform(0, 100, size=len(x_step))

    # Pandas Dataframes
    X_test1 = pd.DataFrame({"x1": x_step, "x2": x_random})
    X_test2 = pd.DataFrame({"x1": x_random, "x2": x_step})

    # Get predicted values
    y_pred_x1 = tree_filtering.predict(model, (0,), X_test1)
    y_pred_x2 = tree_filtering.predict(model, (1,), X_test2)

    # Plot
    plot = go.Figure()

    plot.add_trace(
        go.Scatter(
            x=X_test1["x1"],
            y=y_pred_x1,
            mode="lines",
            name="z_pred_x1",
            line=dict(color="red"),  # specify color here
        )
    )

    plot.add_trace(
        go.Scatter(
            x=X_test2["x2"],
            y=y_pred_x2,
            mode="lines",
            name="z_pred_x2",
            line=dict(color="blue"),  # different color here
        )
    )

    plot.update_layout(
        title=f"{equation}", xaxis_title="x1 / x2", yaxis_title="Predicted z"
    )

    plot.show()


# Optional: Input -- list of variables --> generates all combinations recursively first
def plot_contour_two_vars(model, equation, start_val, stop_val, step_val):
    """
    model: XGBRegressor
    equation: string

    Makes a contour plot specifically from predictions from f(x1x2)
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
    ##### GENERATE DATA #####
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
        max_depth=1,  # depth of 1
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=42,
        base_score=0.8,
    )
    model.fit(X_train, y_train)

    ###### PLOT ######
    plot_linear_one_var(model, "y = 10 * x1 + 2 * x2")


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
    y = 10 * x1 + 2 * x2 + 5
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

    # contour_plot_Z_Pred_x1.show()
    # contour_plot_Z_Pred_x2.show()
    # contour_plot_Z_Pred_x1x2.show()
    # contour_plot_Z_Pred_Default.show()
    plot_contour_two_vars(model, "y = 10 * x1 + 2 * x2 + 5", -200, 200, 20)


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

    ##### GENERATE PREDICTIONS #####

    # x1 = np.arange(0, 100, 0.1)
    # x2 = np.arange(0, 100, 0.1)
    # X1, X2 = np.meshgrid(x1, x2)
    # all_pairs = np.column_stack([X1.flatten(), X2.flatten()])
    # # convert all pairs to DataFrame structure
    # X_test = pd.DataFrame({"x1": all_pairs[:, 0], "x2": all_pairs[:, 1]})

    # # get predictions
    # z_pred_x1 = tree_filtering.predict(model, (0,), X_test)
    # z_pred_x2 = tree_filtering.predict(model, (1,), X_test)
    # z_pred_x1x2 = tree_filtering.predict(model, (0, 1), X_test)
    # z_pred = model.predict(X_test, output_margin=True)  # vector

    # # reshape Z3 vector into 2D numpy array for contour plotting (rows: x2, cols: x1)
    # Z_Pred_x1 = z_pred_x1.reshape(len(x2), len(x1))
    # Z_Pred_x2 = z_pred_x2.reshape(len(x2), len(x1))
    # Z_Pred_x1x2 = z_pred_x1x2.reshape(len(x2), len(x1))
    # Z_Pred_Default = z_pred.reshape(len(x2), len(x1))

    # ##### PLOT #####
    # # Z_Pred_x1
    # contour_plot_Z_Pred_x1 = go.Figure(
    #     data=go.Contour(
    #         z=Z_Pred_x1,
    #         x=x1,
    #         y=x2,
    #         colorscale="blues",
    #         contours=dict(
    #             start=0, end=100, size=5, coloring="heatmap", showlabels=True
    #         ),
    #         line=dict(width=2),
    #         colorbar=dict(title="Z value"),
    #     )
    # )
    # contour_plot_Z_Pred_x1.update_layout(
    #     title="Z_Pred_x1: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    # )

    # # Z_Pred_x2
    # contour_plot_Z_Pred_x2 = go.Figure(
    #     data=go.Contour(
    #         z=Z_Pred_x2,
    #         x=x1,
    #         y=x2,
    #         colorscale="blues",
    #         contours=dict(
    #             start=0, end=1, size=0.5, coloring="heatmap", showlabels=True
    #         ),
    #         line=dict(width=2),
    #         colorbar=dict(title="Z value"),
    #     )
    # )

    # contour_plot_Z_Pred_x2.update_layout(
    #     title="Z_Pred_x2: y = 10 * x1 + 2 * x2 + 5 ", xaxis_title="x1", yaxis_title="x2"
    # )

    # # Z_Pred_x1x2
    # contour_plot_Z_Pred_x1x2 = go.Figure(
    #     data=go.Contour(
    #         z=Z_Pred_x1x2,
    #         x=x1,
    #         y=x2,
    #         colorscale="sunset",
    #         contours=dict(
    #             start=0, end=100, size=5, coloring="heatmap", showlabels=True
    #         ),
    #         line=dict(width=2),
    #         colorbar=dict(title="Z value"),
    #     )
    # )

    # contour_plot_Z_Pred_x1x2.update_layout(
    #     title="Z_Pred_x1x2: y = 10 * x1 + 2 * x2 + 5 ",
    #     xaxis_title="x1",
    #     yaxis_title="x2",
    # )

    # # # Z_Pred_Default
    # contour_plot_Z_Pred_Default = go.Figure(
    #     data=go.Contour(
    #         z=Z_Pred_Default,
    #         x=x1,
    #         y=x2,
    #         colorscale="Greens",
    #         contours=dict(
    #             start=0, end=100, size=5, coloring="heatmap", showlabels=True
    #         ),
    #         line=dict(width=2),
    #         colorbar=dict(title="Z value"),
    #     )
    # )

    # contour_plot_Z_Pred_Default.update_layout(
    #     title="Z_Pred_Default: y = 10 * x1 + 2 * x2 + 5 ",
    #     xaxis_title="x1",
    #     yaxis_title="x2",
    # )

    # contour_plot_Z_Pred_x1.show()
    # contour_plot_Z_Pred_x2.show()
    # contour_plot_Z_Pred_x1x2.show()
    # contour_plot_Z_Pred_Default.show()


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
    plot_contour_two_vars(model, "y = 10 * x1 + 2 * x2 + 5")
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
