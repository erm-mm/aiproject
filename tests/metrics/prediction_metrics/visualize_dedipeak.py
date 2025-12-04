import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --- Add project root and tests folder to sys.path ---
project_root = Path(__file__).resolve().parents[3]  # root of a4s-eval
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'tests'))

# --- Imports ---
from additional_files.train_dedipeak_model import run
from a4s_eval.metrics.prediction_metrics.dedipeak_metric import P3E, P3sw

# --- Settings ---
n_rows = 1000  # number of rows for a quick test
metric_funcs = [P3E, P3sw]

# --- Load data ---
y_true, y_pred = run(n_rows=n_rows)

# --- Plotting ---
plt.figure(figsize=(14, 6))
plt.plot(y_true, label='y_true', color='blue')
plt.plot(y_pred, label='y_pred', color='orange', alpha=0.7)

# --- Mark peaks for each metric ---
for metric_func in metric_funcs:
    value, peaks = metric_func(model=None, X=None, dataset=None, y_true=y_true, y_pred=y_pred)
    true_peaks = peaks.get('peaks_true', [])
    pred_peaks = peaks.get('peaks_pred', [])

    plt.scatter(true_peaks, y_true[true_peaks], color='green', marker='o', label=f'{metric_func.__name__} true peaks')
    plt.scatter(pred_peaks, y_pred[pred_peaks], color='red', marker='x', label=f'{metric_func.__name__} pred peaks')

    print(f"\n=== {metric_func.__name__} ===")
    print(f"Metric value: {value}")
    print(f"True peaks: {true_peaks}")
    print(f"Predicted peaks: {pred_peaks}")

plt.title("DeDiPeak: True vs Predicted Peaks")
plt.xlabel("Index")
plt.ylabel("Value") 
plt.legend()
plt.grid(True)
plt.show()