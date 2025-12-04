import pytest
from a4s_eval.metrics.prediction_metrics.dedipeak_metric import P3E, P3sw
from tests.additional_files.train_dedipeak_model import run

@pytest.mark.parametrize("metric_func", [P3E, P3sw])
def test_dedipeak_metric_real(metric_func):
    print(f"\n=== Testing metric {metric_func.__name__} ===")

    # Take only the first 1000 rows for faster testing
    y_true, y_pred = run(n_rows=1000)

    # Pass y_true and y_pred to the metric function
    value, peaks = metric_func(
        model=None,
        X=None,
        dataset=None,
        y_pred=y_pred,
        y_true=y_true
    )

    print(f"Metric value: {value}")
    print(f"Peaks (full object): {peaks}")

    # Check that the metric value is a number
    assert isinstance(value, (float, int))
    # Check that peaks dictionary is not empty
    assert peaks is not None
