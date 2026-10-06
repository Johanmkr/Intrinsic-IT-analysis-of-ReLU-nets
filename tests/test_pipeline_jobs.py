"""Steps 2–4 job functions end to end on a tiny network, plus result I/O and PCA determinism."""
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

import run_baselines
import run_diagnostics
import run_estimate
from src_experiment.dataset import _pca_project
from src_experiment.estimators import ESTIMATORS
from src_experiment.results import write_table

EPSILONS = [0.0, 0.3, 1.0]


def job(h5, csv, **extra):
    return {"h5": h5, "csv": csv, "sweep": "toy", "dataset": "composite", "arch_str": "[6, 6, 6]",
            "target_dim": -1, "noise_level": 0.0, "permute_labels": False, "global_seed": 42,
            "protocol": "heldout", "labels": "true_labels", **extra}


@pytest.fixture(scope="module")
def step2(tiny_h5, tmp_path_factory):
    csv = tmp_path_factory.mktemp("step2") / "routing.csv"
    run_estimate.run_one(job(tiny_h5, csv), EPSILONS)
    return pd.read_csv(csv)


def test_step2_writes_one_row_per_epoch_layer_epsilon(step2):
    assert len(step2) == 2 * 3 * len(EPSILONS)
    assert {"protocol", "labels", "sweep", "arch_str"} <= set(step2.columns)
    assert all(f"{e}_bits" in step2 and f"{e}_func_bits" in step2 for e in ESTIMATORS)
    assert (step2["N"] == 400).all()


def test_step3_baselines_on_the_same_points(tiny_h5, tmp_path):
    csv = tmp_path / "baselines.csv"
    run_baselines.run_one(job(tiny_h5, csv), all_epochs=False)
    df = pd.read_csv(csv)
    assert list(df["layer"]) == [1, 2, 3] and (df["epoch"] == 1).all() and (df["N"] == 400).all()
    assert df["d_T"].tolist() == [6, 6, 6]


def test_step4_order_zero_reproduces_step2(tiny_h5, tmp_path):
    j = job(tiny_h5, tmp_path / "routing.csv", ordering=True,
            sizes_csv=tmp_path / "sizes.csv", ordering_csv=tmp_path / "ordering.csv")
    run_diagnostics.run_one(j)
    sizes, order = pd.read_csv(j["sizes_csv"]), pd.read_csv(j["ordering_csv"])
    points = (sizes["region_size"] * sizes["num_regions"]).groupby(sizes["layer"]).sum()
    assert (points == 400).all()  # every point is in exactly one region
    for eps in run_diagnostics.ORDER_EPSILONS:
        o0 = order[(order["order"] == 0) & np.isclose(order["epsilon"], eps)].set_index("layer")
        s2 = run_estimate.FunctionalQuotientEstimator(tiny_h5).evaluate_epoch(1, epsilons=[eps])
        for r in s2:
            assert o0.loc[r["layer"], "num_quotient"] == r["num_quotient"]
            assert o0.loc[r["layer"], "miller_madow_func_bits"] == pytest.approx(r["miller_madow_func_bits"], abs=1e-12)
    assert order["order"].nunique() == run_diagnostics.N_ORDERS + 1


def test_write_table_is_deterministic(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    write_table(df, tmp_path / "x.csv.gz")
    first = (tmp_path / "x.csv.gz").read_bytes()
    write_table(df, tmp_path / "x.csv.gz")
    assert (tmp_path / "x.csv.gz").read_bytes() == first
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "x.csv.gz"), df)


def test_pca_projection_is_independent_of_the_thread_count():
    rng = np.random.default_rng(0)
    X_train, X_test = rng.normal(size=(3000, 120)), rng.normal(size=(500, 120))
    with threadpool_limits(limits=1):
        a = _pca_project(X_train, X_test, 10, 42)
    with threadpool_limits(limits=4):
        b = _pca_project(X_train, X_test, 10, 42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
