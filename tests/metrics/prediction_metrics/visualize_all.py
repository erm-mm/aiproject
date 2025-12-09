import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -----------------------------
# Project setup
# -----------------------------
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "tests/additional_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Imports
# -----------------------------
from tests.additional_files.train_regression_model import run as run_regression
from tests.additional_files.train_dedipeak_model import run as run_dedipeak
from a4s_eval.metrics.prediction_metrics import regression_metrics
from a4s_eval.metrics.prediction_metrics.dedipeak_metric import P3E, P3sw

# -----------------------------
# Settings
# -----------------------------
N_ROWS = 1000  # number of points to visualize

# -----------------------------
# Sampling helper
# -----------------------------
def sample_n_rows(y_true, y_pred, n_rows):
    total = len(y_true)
    if total <= n_rows:
        return y_true, y_pred
    indices = np.linspace(0, total-1, n_rows, dtype=int)
    return y_true[indices], y_pred[indices]

# -----------------------------
# Compute regression metrics
# -----------------------------
def compute_regression_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    r2 = 1 - (np.sum((y_true - y_pred)**2) /
              np.sum((y_true - np.mean(y_true))**2))
    return mae, mse, r2

# -----------------------------
# Super dashboard (3x2)
# -----------------------------
def create_super_dashboard(y_true_reg, y_pred_reg, mae, mse, r2,
                           y_true_peak, y_pred_peak, P3E_val, P3sw_val,
                           out_path):
    error = y_pred_reg - y_true_reg

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle("Unified Regression + DeDiPeak Dashboard", fontsize=20)

    # -------------------------
    # (1,1) Regression scatter
    # -------------------------
    ax = axes[0,0]
    ax.scatter(y_true_reg, y_pred_reg, alpha=0.5, label="Predicted vs True")
    min_val = min(y_true_reg.min(), y_pred_reg.min())
    max_val = max(y_true_reg.max(), y_pred_reg.max())
    padding = (max_val - min_val) * 0.05
    x_line = np.linspace(min_val-padding, max_val+padding, 500)
    ax.plot(x_line, x_line, color="red", linewidth=2, label="y = x")
    ax.set_title("Regression: True vs Predicted")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.grid(True)
    ax.legend()

    # -------------------------
    # (1,2) Regression error over time
    # -------------------------
    ax = axes[0,1]
    ax.plot(error, alpha=0.6)
    ax.set_title("Regression: Error over Time")
    ax.set_xlabel("Index")
    ax.set_ylabel("Error")
    ax.grid(True)

    # -------------------------
    # (2,1) Regression metrics bar chart
    # -------------------------
    ax = axes[1,0]
    metrics = ["MAE", "MSE", "R²"]
    values = [mae, mse, r2]
    ax.bar(metrics, values, color=["C0","C1","C2"])
    ax.set_title("Regression Metrics")
    ax.grid(axis='y')

    # -------------------------
    # (2,2) Regression error histogram
    # -------------------------
    ax = axes[1,1]
    ax.hist(error, bins=40, alpha=0.7, color="purple")
    ax.set_title("Regression: Error Distribution")
    ax.set_xlabel("Error value")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    # -------------------------
    # (3,1) DeDiPeak true vs predicted
    # -------------------------
    ax = axes[2,0]
    ax.plot(y_true_peak, label="y_true")
    ax.plot(y_pred_peak, label="y_pred", alpha=0.7)
    # Peaks
    for metric_func, color in zip([P3E, P3sw], ["green","orange"]):
        val, info = metric_func(model=None, X=None, dataset=None,
                                y_true=y_true_peak, y_pred=y_pred_peak)
        true_peaks = info.get("peaks_true", [])
        pred_peaks = info.get("peaks_pred", [])
        if len(true_peaks) > 0:
            ax.scatter(true_peaks, y_true_peak[true_peaks],
                       label=f"{metric_func.__name__} true peaks", marker="o", color=color)
        if len(pred_peaks) > 0:
            ax.scatter(pred_peaks, y_pred_peak[pred_peaks],
                       label=f"{metric_func.__name__} predicted peaks", marker="x", color=color)
    ax.set_title("DeDiPeak: True vs Predicted Peaks")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    # -------------------------
    # (3,2) P3E/P3sw vs peak range
    # -------------------------
    ax = axes[2,1]
    peak_range = np.max(y_true_peak) - np.min(y_true_peak)
    metrics_vals = [P3E_val, P3sw_val, peak_range]
    metrics_names = ["P3E","P3sw","Peak Range"]
    ax.bar(metrics_names, metrics_vals, color=["C0","C1","C2"])
    for i, v in enumerate(metrics_vals[:-1]):
        percent = v/peak_range*100
        ax.text(i, v+peak_range*0.02, f"{percent:.1f}%", ha='center')
    ax.set_title("Peak Metrics vs Signal Range")
    ax.set_ylabel("Value (same units as signal)")
    ax.grid(axis='y')

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()  # закрываем фигуру, чтобы не показывалась в интерактивном окне
    print(f"Dashboard saved to {out_path}")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Regression
    y_true_reg, y_pred_reg = run_regression()
    y_true_reg_vis, y_pred_reg_vis = sample_n_rows(y_true_reg, y_pred_reg, N_ROWS)
    mae, mse, r2 = compute_regression_metrics(y_true_reg_vis, y_pred_reg_vis)

    # DeDiPeak
    y_true_peak, y_pred_peak = run_dedipeak(n_rows=N_ROWS)
    P3E_val, _ = P3E(model=None, X=None, dataset=None, y_true=y_true_peak, y_pred=y_pred_peak)
    P3sw_val, _ = P3sw(model=None, X=None, dataset=None, y_true=y_true_peak, y_pred=y_pred_peak)

    # Dashboard
    out_path = OUTPUT_DIR / "super_dashboard.png"
    create_super_dashboard(y_true_reg_vis, y_pred_reg_vis, mae, mse, r2,
                           y_true_peak, y_pred_peak, P3E_val, P3sw_val, out_path)
