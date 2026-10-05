"""Sweep constants shared by the pipeline, with a reduced smoke-test variant.

``./run.sh smoke`` sets ``SMOKE=1``: one seed, 11 training epochs and two PCA
dimensions, so the whole pipeline (train → estimate → baselines → plot) runs in
minutes. The figures it produces only check that the code runs; they are not
the paper's figures.
"""

import os

SMOKE = os.environ.get("SMOKE") == "1"

SEEDS = [101] if SMOKE else [101, 102, 103, 104, 105]
EPOCHS = 11 if SMOKE else 151
LAST_EPOCH = EPOCHS - 1
MNIST_DIMS_B = [2, 10] if SMOKE else [2, 3, 4, 5, 10, 15, 20]
