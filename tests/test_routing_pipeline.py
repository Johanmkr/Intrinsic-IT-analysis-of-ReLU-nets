"""Consistency checks of step 2 (run_estimate.py / FunctionalQuotientEstimator).

Like test_parx_partitions.py, reads the ./run.sh smoke checkpoints under
``./run.sh test`` (SMOKE=1) and the step-1 outputs/ under ``--full``.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src_experiment.estimators import ESTIMATORS  # noqa: E402
from src_experiment.functional_quotient import FunctionalQuotientEstimator  # noqa: E402
from src_experiment.probe_loader import make_composite_insample  # noqa: E402
from src_experiment.routing_estimator import (  # noqa: E402
    cumulative_pattern_hashes,
    forward_activation_patterns,
    routing_information,
)
from src_experiment.smoke import LAST_EPOCH, SMOKE  # noqa: E402

DATA_ROOT = REPO_ROOT / "smoke" if SMOKE else REPO_ROOT
COMPOSITE_H5 = DATA_ROOT / "outputs/composite_label_noise/n0.0_[9, 9, 9]/seed_101.h5"

pytestmark = pytest.mark.skipif(
    not COMPOSITE_H5.exists(),
    reason=f"needs {DATA_ROOT / 'outputs'}: run ./run.sh {'smoke' if SMOKE else 'step1'} first",
)


def test_composite_insample_ends_with_stored_test_split():
    insample = make_composite_insample(42)
    with h5py.File(COMPOSITE_H5, "r") as f:
        points, labels = f["points"][:], f["labels"][:]
    assert len(insample.y_probe) == 5 * len(labels)
    assert np.array_equal(insample.X_probe[-len(labels):], points.astype(np.float32))
    assert np.array_equal(insample.y_probe[-len(labels):], labels)


@pytest.fixture(scope="module")
def last_epoch_rows():
    est = FunctionalQuotientEstimator(COMPOSITE_H5)
    return est, est.evaluate_epoch(LAST_EPOCH, epsilons=(0.0, 0.3, 2.0))


def test_raw_estimates_match_routing_information(last_epoch_rows):
    est, rows = last_epoch_rows
    W, b = est.routing._load_weights(LAST_EPOCH)
    patterns = forward_activation_patterns(W, b, est.points)
    for row in rows:
        omega = cumulative_pattern_hashes(patterns, row["layer"])
        plug_in, mm, R, H_Y = routing_information(omega, est.labels)
        assert row["num_regions"] == R
        assert row["plug_in_bits"] == pytest.approx(plug_in, abs=1e-12)
        assert row["miller_madow_bits"] == pytest.approx(mm, abs=1e-12)
        assert row["H_Y_bits"] == pytest.approx(H_Y, abs=1e-12)


def test_quotient_merges_regions_and_never_adds_information(last_epoch_rows):
    _, rows = last_epoch_rows
    for row in rows:
        assert 1 <= row["num_quotient"] <= row["num_regions"]
        assert 0 < row["rho_func"] <= 1
        # Merging regions coarsens the partition: plug-in MI cannot increase.
        assert row["plug_in_func_bits"] <= row["plug_in_bits"] + 1e-12
        assert row["plug_in_bits"] <= row["H_Y_bits"] + 1e-12
    by_layer = {}
    for row in rows:
        by_layer.setdefault(row["layer"], []).append(row["num_quotient"])
    for sizes in by_layer.values():  # larger ε merges at least as much
        assert sizes == sorted(sizes, reverse=True)


def test_row_has_every_estimator_and_sane_diagnostics(last_epoch_rows):
    _, rows = last_epoch_rows
    for row in rows:
        for name in ESTIMATORS:
            assert f"{name}_bits" in row and f"{name}_func_bits" in row
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["rho"] == pytest.approx(row["num_regions"] / row["N"])
        assert 0.0 <= row["singleton_region_frac"] <= 1.0
