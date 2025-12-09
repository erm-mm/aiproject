# A4S Evaluation module
# Regression Metrics (MAE, MSE, R2) + DeDePeak metric

## Link to the GitHub repository

https://github.com/erm-mm/aiproject

## Setup

To clone and set up the repository locally:
```bash
git clone git@github.com:erm-mm/aiproject.git
cd a4s-eval
uv sync
```

To make sure all required packages for testing and visualization are installed:
```bash
uv add pandas
uv add matplotlib
uv sync
```

## Running Tests

To run all prediction metric tests, execute the following command in the terminal:

```bash
uv run pytest tests/metrics/prediction_metrics/test_regression_metrics_real.py -s
```
— verifies basic regression metrics (MAE, MSE, R²) on a subset of the Household Power Consumption dataset.

```bash
uv run pytest tests/metrics/prediction_metrics/test_dedipeak_metric_real.py -s
```
— checks DeDiPeak metrics (P3E, P3sw) for correct peak detection on the same dataset.

```bash
uv run pytest tests/metrics/prediction_metrics/test_execute.py -s
```
— sequentially executes all main prediction metric tests to ensure the metrics work together as expected.


## Visualization

To generate a unified visualization dashboard (regression metrics + DeDiPeak metrics):
```bash
uv run tests/metrics/prediction_metrics/visualize_all.py
```
This generates all figures (scatter plots, error plots, metric bar charts, peak comparisons) in a single file:
```bash
tests/additional_files/super_dashboard.png
```
— includes:
- Regression: True vs Predicted, Error over Time, Error Distribution, MAE/MSE/R² bar chart
- DeDiPeak: True vs Predicted Peaks, P3E/P3sw relative to peak range

## Explanation of Metrics

- Regression Metrics (MAE, MSE, R²) measure general prediction accuracy across all points. They are implemented in a4s_eval/metrics/prediction_metrics/regression_metrics.py.

- DeDiPeak Metrics (P3E, P3sw) focus specifically on peak detection, combining timing and amplitude errors. These are implemented in a4s_eval/metrics/prediction_metrics/dedipeak_metric.py.

These metrics complement each other: regression metrics assess overall accuracy, while DeDiPeak metrics highlight errors in peak prediction, which is critical for time-series datasets like household power consumption.

##  Limitations

Data and model scope: The metrics are designed for regression tasks and time-series data, as demonstrated with the Household Power Consumption dataset. They may not be directly applicable to classification tasks or other types of data without modification. DeDiPeak metrics are particularly suited for signals with well-defined peaks; their performance may decrease on smoothed or noisy data.

Data size and quality: Tests use a subset of the dataset (1000 rows for DeDiPeak, full file for regression). Performance and accuracy may vary on larger datasets. Closely spaced or low-amplitude peaks may be detected inaccurately by P3E and P3sw metrics.

Methodological constraints: Regression metrics (MAE, MSE, R²) measure overall error across the signal, while DeDiPeak metrics focus specifically on peak errors. They complement each other but cannot be directly compared to claim that one is “better.” Metrics require one-dimensional model outputs; multi-dimensional or categorical outputs are not supported without modification.

Visualization and testing: The visualize_dedipeak.py script and tests.ipynb notebook are experimental, with limited functionality, and may be extended in future updates.

## Project Files

The following files were created or modified as part of this project:

/a4s-eval/a4s_eval/metrics/prediction_metrics/regression_metrics.py
Implementation and registration of basic regression metrics (MAE, MSE, R²).

/a4s-eval/a4s_eval/metrics/prediction_metrics/dedipeak_metric.py
DeDiPeak metrics for peak-aware error measurement. Implements P3E (symmetric minimal distance combining timing and amplitude) and P3sw (sliding-window peak comparison), plus a simple local-peak detector.

/a4s-eval/tests/additional_files/train_regression_model.py
Utility script to load the Household Power Consumption dataset, train a simple LinearRegression model, and return test ground-truth and predicted values (used for metric testing).

/a4s-eval/tests/additional_files/train_dedipeak_model.py
Utility module for loading the Household Power Consumption dataset, training a simple regression model, and generating predictions for testing DeDiPeak metrics.

/a4s-eval/tests/metrics/prediction_metrics/test_regression_metrics_real.py
Tests regression metrics (MAE, MSE, R²) on real Household Power Consumption data using the trained model predictions.

/a4s-eval/tests/metrics/prediction_metrics/test_dedipeak_metric_real.py
Tests DeDiPeak metrics (P3E, P3sw) on the first 1000 rows of real Household Power Consumption data to verify correct peak detection and metric computation.

/a4s-eval/tests/metrics/prediction_metrics/test_execute.py
Executes all main prediction metric tests (regression and DeDiPeak) sequentially using pytest, with a placeholder test to prevent collection errors.

/a4s-eval/tests/metrics/prediction_metrics/visualize_all.py
Generates a unified dashboard (super_dashboard.png) containing regression and DeDiPeak figures for quick inspection.

/a4s-eval/tests/data/household_power_consumption.txt
Subset of the UCI Household Power Consumption dataset, used for testing regression and DeDiPeak metrics on real time-series data. Only a small portion is included to speed up tests. Link

/a4s-eval/tests/additional_files/__init__.py, /a4s-eval/tests/__init__.py, /a4s-eval/tests/metrics/prediction_metrics/__init__.py
Empty placeholder files for correct Python package structure.
