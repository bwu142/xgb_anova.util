import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

#####################################
###########  y = 10*x1  #############
#####################################


###### TREE HELPER FUNCTIONS ######
def get_bias(model):
    bias = model.get_booster().attr("base_score")
    if bias is not None:
        return float(bias)
    return float(model.base_score)  # Fallback


####### GENERATE DATA #######

# arbitrary random seed (starting point for generating seq of random numbers)
np.random.seed(42)
# LIST of 100 random numbers generated from standard normal distribution (1D numpy array)
x1 = np.random.randn(100)
y = 10 * x1 + 2  # true function (no noise)

# Split data into training and testing sets
X1 = pd.DataFrame({"x1": x1})  # create tabular format of x1
# split data into: 30% of data used for testing, 70% for training
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y, test_size=0.3, random_state=42
)

###### FIT XGBOOST REGRESSOR ######

# Fit XGBoost regressor with 3 tree of depth 1
model1 = xgb.XGBRegressor(
    n_estimators=3,  # 3 trees
    max_depth=1,  # depth of 1
    learning_rate=1.0,
    objective="reg:squarederror",
    random_state=42,
    base_score=0.8,
)
model1.fit(X1_train, y1_train)

model1.save_model("model1_3trees.json")  # save model1_3tree

# WRAP IN FUNCTION?
model1_t1 = model1.predict(X1_test, iteration_range=(0, 1), output_margin=True)
model1_t2 = model1.predict(X1_test, iteration_range=(1, 2), output_margin=True)
model1_t3 = model1.predict(X1_test, iteration_range=(2, 3), output_margin=True)

# Get the total prediction (all trees + base_score)
total_pred = model1.predict(X1_test, output_margin=True)
base_score = (
    model1.get_booster.base_score
)  # get booster accesses underlying booster object

# Plot each tree's prediction vs total prediction
x1 = X1_test["x1"].values
y1 = y1_test
y1_3 = total_pred
y1_t1 = model1_t1
y1_t2 = model1_t2
y1_t3 = model1_t3

plt.xlabel("x1")
plt.ylabel("Prediction y value")
plt.scatter(x1, y1, label="true y", color="green", marker="x")
plt.scatter(x1, y1_3, label="all trees predicted y", color="red", marker="*")
plt.scatter(x1, y1_t1, label="tree 1 predicted y", color="blue", marker=".")
plt.scatter(x1, y1_t2, label="tree 2 predicted y", color="brown", marker=".")
plt.scatter(x1, y1_t3, label="tree 3 predicted y", color="purple", marker=".")

plt.show()


# # 4. Predict and compute R^2
# y_train_pred = model1.predict(X1_train)
# y_test_pred = model1.predict(X1_test)
# r2_train = r2_score(y1_train, y_train_pred)
# r2_test = r2_score(y1_test, y_test_pred)
# print(f"Train R^2: {r2_train:.3f}")
# print(f"Test R^2: {r2_test:.3f}")

# # 5. Plot results
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(X1_train["x1"], y1_train, label="True y (train)", color="blue")
# plt.scatter(
#     X1_train["x1"], y_train_pred, label="Predicted y (train)", color="red", marker="x"
# )
# plt.title("Training Set")
# plt.xlabel("x1")
# plt.ylabel("y")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.scatter(X1_test["x1"], y1_test, label="True y (test)", color="blue")
# plt.scatter(
#     X1_test["x1"], y_test_pred, label="Predicted y (test)", color="red", marker="x"
# )
# plt.title("Test Set")
# plt.xlabel("x1")
# plt.ylabel("y")
# plt.legend()

# plt.tight_layout()
# plt.show()

# # 6. Optional: Show the tree structure
# print("\nTree dump (JSON):")
# print(model1.get_booster().get_dump(dump_format="json")[0])

# ####################################
# ####################################
# ########## y = 10x1 + 5x2 ##########
# ####################################
# ####################################

# # Generate Data
# np.random.seed(42)
# x1 = np.random.randn(100)
# x2 = np.random.randn(100)
# y = 10 * x1 + 5 * x2

# # Train/test split
# X2 = pd.DataFrame({"x1": x1, "x2": x2})
# X2_train, X2_test, y2_train, y2_test = train_test_split(
#     X2, y, test_size=0.3, random_state=42
# )

# # Fit XGBoost regressor
# model2 = xgb.XGBRegressor(
#     n_estimators=2,
#     max_depth=1,
#     learning_rate=1.0,
#     objective="reg:squarederror",
#     random_state=42,
# )
# model2.fit(X2_train, y2_train)

# model2.save_model("model_y_10x1_5x2.json")  # save model

# # Test XGBoost regressor & get error
# y_train_pred = model2.predict(X2_train)
# y_test_pred = model2.predict(X2_test)
# r2_train = r2_score(y2_train, y_train_pred)
# r2_test = r2_score(y2_test, y_test_pred)
# print(f"Train R^2: {r2_train:.3f}")
# print(f"Test R^2: {r2_test:.3f}")

# # 5. Plot results: True y vs. Predicted y
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.scatter(y2_train, y_train_pred, color="red", alpha=0.7)
# plt.plot(
#     [y2_train.min(), y2_train.max()], [y2_train.min(), y2_train.max()], "k--", lw=2
# )
# plt.title("Training Set")
# plt.xlabel("True y")
# plt.ylabel("Predicted y")

# plt.subplot(1, 2, 2)
# plt.scatter(y2_test, y_test_pred, color="blue", alpha=0.7)
# plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], "k--", lw=2)
# plt.title("Test Set")
# plt.xlabel("True y")
# plt.ylabel("Predicted y")

# plt.tight_layout()
# plt.show()

# # 6. Optional: Show the tree structures
# print("\nTree dump (JSON):")
# for i, tree_json in enumerate(model2.get_booster().get_dump(dump_format="json")):
#     print(f"Tree {i}:\n{tree_json}\n")

# ########## MODEL 2 ALTERNATIVE ##########
# #########################################
# #########################################

# # Fit XGBoost Regressor
# model2_alt = xgb.XGBRegressor(
#     n_estimators=1,
#     max_depth=1,
#     learning_rate=1.0,
#     objective="reg:squarederror",
#     random_state=42,
# )
# model2_alt.fit(X2_train, y2_train)

# model2.save_model("model_y_10x1_5x2_alt.json")  # save model

# # Test XGBoost regressor & get error
# y_train_pred = model2_alt.predict(X2_train)
# y_test_pred = model2_alt.predict(X2_test)
# r2_alt_train = r2_score(y2_train, y_train_pred)
# r2_alt_test = r2_score(y2_test, y_test_pred)
# print(f"Train R^2_alt: {r2_alt_train:.3f}")
# print(f"Test R^2_alt: {r2_alt_test:.3f}")

# ################# OTHER STUFF #################

# # --- Load models into new instances ---
# model1_loaded = xgb.XGBRegressor()
# model1_loaded.load_model("model_y_10x1.json")

# model2_loaded = xgb.XGBRegressor()
# model2_loaded.load_model("model_y_10x1_5x2.json")

# # For y = 10x1
# y_test_pred_orig = model1.predict(X1_test)
# y_test_pred_loaded = model1_loaded.predict(X1_test)
# print("Max abs diff (y=10x1):", np.max(np.abs(y_test_pred_orig - y_test_pred_loaded)))

# # For y = 10x1 + 5x2
# y_test_pred_orig2 = model2.predict(X2_test)
# y_test_pred_loaded2 = model2_loaded.predict(X2_test)
# print(
#     "Max abs diff (y=10x1+5x2):",
#     np.max(np.abs(y_test_pred_orig2 - y_test_pred_loaded2)),
# )


# plt.figure(figsize=(6, 6))
# plt.scatter(y_test_pred_orig, y_test_pred_loaded, alpha=0.7)
# plt.plot(
#     [y_test_pred_orig.min(), y_test_pred_orig.max()],
#     [y_test_pred_orig.min(), y_test_pred_orig.max()],
#     "k--",
# )
# plt.xlabel("Original prediction")
# plt.ylabel("Loaded prediction")
# plt.title("Original vs. Loaded Model Predictions")
# plt.show()
