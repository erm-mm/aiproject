import numpy as np
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric_registry


def find_peaks_simple(y: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Find local peaks: index t is a peak if y[t] > y[t-k] and y[t] > y[t+k] for all 1 <= k <= n.
    Returns the indices of the peaks as a numpy array of ints.
    """
    y = np.asarray(y).ravel()
    peaks = []
    N = len(y)
    for t in range(n, N - n):
        window = y[t - n : t + n + 1]
        # a peak is the center element and it must be the unique maximum in the window
        if y[t] == window.max() and np.argmax(window) == n:
            peaks.append(t)
    return np.array(peaks, dtype=int)


def P3E(
    model,
    X,
    dataset=None,
    y_pred: np.ndarray = None,
    y_true: np.ndarray = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    n: int = 1,
):
    """
    DeDiPeak metric P3E: symmetric sum of minimal distances between true and predicted peaks,
    combining temporal distance and amplitude difference.

    Returns:
        (value, metadata_dict) where metadata_dict contains 'peaks_true' and 'peaks_pred' indices.
    """
    # Obtain true signal
    if y_true is None:
        if dataset is None:
            raise ValueError("Either 'dataset' or 'y_true' must be provided.")
        # assume the signal is in the first column of dataset.data (pandas DataFrame)
        y_true = dataset.data.iloc[:, 0].to_numpy()

    if y_pred is None:
        raise ValueError("y_pred must be provided.")

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    peaks_true = find_peaks_simple(y_true, n=n)
    peaks_pred = find_peaks_simple(y_pred, n=n)

    # If no peaks in true or predicted signal, return NaN and metadata
    if len(peaks_true) == 0 or len(peaks_pred) == 0:
        return np.nan, {"peaks_true": peaks_true, "peaks_pred": peaks_pred}

    # EE(y, ŷ): for every true peak t find minimal cost to any predicted peak tp
    total = sum(
        min(alpha * (t - tp) ** 2 + beta * (y_true[t] - y_pred[tp]) ** 2 for tp in peaks_pred)
        for t in peaks_true
    ) / len(peaks_true)

    # EE(ŷ, y): symmetric term for predicted peaks to true peaks
    total2 = sum(
        min(alpha * (tp - t) ** 2 + beta * (y_pred[tp] - y_true[t]) ** 2 for t in peaks_true)
        for tp in peaks_pred
    ) / len(peaks_pred)

    value = total + total2
    return value, {"peaks_true": peaks_true, "peaks_pred": peaks_pred}


def P3sw(
    model,
    X,
    dataset=None,
    y_pred: np.ndarray = None,
    y_true: np.ndarray = None,
    T: int = 1,
    n: int = 1,
):
    """
    DeDiPeak metric P3sw: sliding-window based peak error.
    For each true peak t, compares the true peak amplitude at t with the maximum predicted value
    in a window [t-T, t+T], and vice versa. Returns symmetric sum of squared differences normalized
    by number of peaks.
    """
    # Obtain true signal
    if y_true is None:
        if dataset is None:
            raise ValueError("Either 'dataset' or 'y_true' must be provided.")
        y_true = dataset.data.iloc[:, 0].to_numpy()

    if y_pred is None:
        raise ValueError("y_pred must be provided.")

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    peaks_true = find_peaks_simple(y_true, n=n)
    peaks_pred = find_peaks_simple(y_pred, n=n)

    if len(peaks_true) == 0 or len(peaks_pred) == 0:
        return np.nan, {"peaks_true": peaks_true, "peaks_pred": peaks_pred}

    def esw(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for t in peaks_true:
            start = max(0, t - T)
            end = min(len(b), t + T + 1)
            local_max = b[start:end].max()
            total += (a[t] - local_max) ** 2
        return total / len(peaks_true)

    value = esw(y_true, y_pred) + esw(y_pred, y_true)
    return value, {"peaks_true": peaks_true, "peaks_pred": peaks_pred}


# Register metrics
prediction_metric_registry.register(func=P3E, name="dedipeak_p3e")
prediction_metric_registry.register(func=P3sw, name="dedipeak_p3sw")
