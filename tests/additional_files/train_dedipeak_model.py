import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Simple Dataset implementation for DeDiPeak.
    """
    def __init__(self, path: str, n_rows: int = None):
        self.data = pd.read_csv(
            path,
            sep=";",
            low_memory=False,
            na_values=["?"]
        )
        # Limit the number of rows if needed
        if n_rows is not None:
            self.data = self.data.head(n_rows)

        # Combine Date and Time into a single datetime column
        self.data["Datetime"] = pd.to_datetime(
            self.data["Date"] + " " + self.data["Time"],
            dayfirst=True,
            errors="coerce"
        )
        self.data = self.data.dropna()


def load_dataset(path: str, n_rows: int = None):
    """
    Loads and prepares the Household Power Consumption dataset.
    """
    dataset = Dataset(path, n_rows=n_rows)

    y = dataset.data["Global_active_power"].astype(float).values
    feature_cols = [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    X = dataset.data[feature_cols].astype(float).values
    return X, y


def train_regression_model(X_train: np.ndarray, y_train: np.ndarray):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def run(path: str = "tests/data/household_power_consumption.txt", n_rows: int = 1000):
    """
    Main function: loads data, splits into train/test,
    trains a regression model and returns predictions and ground truth.
    """
    X, y = load_dataset(path, n_rows=n_rows)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = train_regression_model(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred


if __name__ == "__main__":
    y_test, y_pred = run()
    print("Prediction examples:", y_pred[:10])
    print("Ground truth examples:", y_test[:10])
