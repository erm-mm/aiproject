import numpy as np
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric_registry


def mae(model, X, dataset, y_pred):
    """Mean Absolute Error."""
    y_true = dataset.y
    value = np.mean(np.abs(y_true - y_pred))
    return value, {}


def mse(model, X, dataset, y_pred):
    """Mean Squared Error."""
    y_true = dataset.y
    value = np.mean((y_true - y_pred) ** 2)
    return value, {}


def r2(model, X, dataset, y_pred):
    """R-squared (coefficient of determination)."""
    y_true = dataset.y
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    value = 1 - ss_res / ss_tot
    return value, {}


# Register metrics in the global registry
prediction_metric_registry.register(func=mae, name="mae")
prediction_metric_registry.register(func=mse, name="mse")
prediction_metric_registry.register(func=r2, name="r2")
