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
    """
    Generates correlated normal variables and XGBoost regression model

    Args:
        n_samples: Number of data samples
        rho: Correlation coefficient between variables
        b1,b2,b3: Coefficients for interaction terms
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        xgb_params: Dictionary of XGBoost parameters

    Returns:
        X_train, X_test, y_train, y_test: Split data
        model: Trained XGBoost model
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
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": random_state,
        }

    # 6. Train XGBoost model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model


# Example usage
if __name__ == "__main__":
    # Generate data and train model
    X_train, X_test, y_train, y_test, model = generate_data_and_model(
        rho=0.3, b1=0.5, b2=1.2, b3=-0.8
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Feature importances: {model.feature_importances_}")

    # # === Dump model config to a Python dict ===
    # model_json = model.get_booster().save_config()

    # model_dict = json.loads(model_json)
    # print("Model config as dict:")
    # print(model_dict)

    # Dump the full model (including trees) to a JSON string
    model_json = model.get_booster().save_model("xgb_model.json")

    # Now, read the file back in as a dictionary
    with open("xgb_model.json", "r") as f:
        tree_dict = json.load(f)

    # Now you can inspect tree_dict
    print(tree_dict)

    # # Alternative

    dump = model.get_booster().get_dump(dump_format="text")
    for i, tree in enumerate(dump):
        print(f"Tree {i}:\n{tree}\n")
