from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pacmath.experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        experiment_name="smoke",
        calibration_n=config.SMOKE_CALIBRATION_N,
        test_n=config.SMOKE_TEST_N,
    )
