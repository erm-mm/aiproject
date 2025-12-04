import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare the Household Power Consumption dataset.
    Handles pandas warnings and missing values robustly.
    Returns:
        X: 2D numpy array of features
        y: 1D numpy array of target values
    """
    # 1. Read CSV without parse_dates to avoid mixed-type warnings
    df = pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values=["?"]
    )

    # 2. Combine Date and Time columns safely into a single datetime column
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,     # dataset uses day-first format, e.g. 16/12/2006
        errors="coerce"    # invalid dates become NaT
    )

    # 3. Drop rows with any NA (including NaT in Datetime)
    df = df.dropna()

    # 4. Target variable
    y = df["Global_active_power"].astype(float).to_numpy()

    # 5. Feature columns
    feature_cols = [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    X = df[feature_cols].astype(float).to_numpy()

    return X, y


def train_regression_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Train a simple linear regression model and return the fitted estimator.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def run(path: str = "tests/data/household_power_consumption.txt"):
    """
    Main routine: load data, split into train/test (no shuffle), train the model,
    and return test ground-truth and predictions.
    Returns:
        y_test, y_pred
    """
    X, y = load_dataset(path)

    # Train / test split without shuffling to preserve time order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = train_regression_model(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred


# Standalone run
if __name__ == "__main__":
    y_test, y_pred = run()
    print("Example predictions:", y_pred[:10])
    print("Example ground-truth values:", y_test[:10])
