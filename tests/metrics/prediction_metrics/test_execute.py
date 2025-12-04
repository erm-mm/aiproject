import subprocess
import sys

# List of test files to run
test_files = [
    "tests/metrics/prediction_metrics/test_dedipeak_metric_real.py",
    "tests/metrics/prediction_metrics/test_regression_metrics_real.py"
]

for test_file in test_files:
    print(f"\n=== Running tests: {test_file} ===\n")
    # Call pytest for the specific file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-s"],
        capture_output=False  # set True to capture output in a variable
    )
    # Check pytest return code
    if result.returncode != 0:
        print(f"\n!!! Tests in {test_file} failed !!!\n")
    else:
        print(f"\nTests in {test_file} passed successfully!\n")


def test_placeholder():
    """Empty test to prevent pytest from complaining about 0 collected items."""
    pass
