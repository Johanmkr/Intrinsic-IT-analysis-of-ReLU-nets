# Intrinsic Information Theoretic Analysis of ReLU Nets — code

Experiment suite for the NeurIPS 2026 paper: trains every network, computes every
estimate (routing information with six estimators, functional quotient, MI baselines,
diagnostics), stores all results in `results/`, and draws the figures from them.
Everything runs from this directory with one command per step.

**Project page:** <https://johanmkr.github.io/Intrinsic_IT_analysis_of_ReLU_nets/>
(abstract, links and interactive demos; source in `site/`).

---

## Requirements

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
  uv installs the pinned Python (3.13, `.python-version`) and the locked packages
  (`uv.lock`) itself. No GPU and no Julia.
- Internet access on the **first run** only: WBC is fetched from the UCI repository
  (cached as CSV in `data/uci_17/`) and MNIST is downloaded via torchvision
  (`./run.sh setup` does both up front).
- Or only Docker: the image (see [Docker](#c-docker)) contains the environment,
  both datasets and the stored results, and runs without internet.
- All estimators are implemented in this repository. `infomeasure` (AGPL-3.0, used for
  the rebuttal) is only needed to run the cross-check in `tests/test_estimators.py`:
  `uv sync --group rebuttal`; without it that comparison is skipped.

---

## Quick start

Get the code (all commands below run from the repository root):

```bash
git clone https://github.com/Johanmkr/Intrinsic_IT_analysis_of_ReLU_nets.git
cd Intrinsic_IT_analysis_of_ReLU_nets
```

Then either redraw the figures from the stored results (A) or rerun every
experiment from scratch (B), with uv on your machine or with Docker (C).

### A. Redraw the figures from the stored results (minutes)

All results of steps 1–4 are in git under `results/` (gzipped CSVs), so the
figures can be drawn without training anything or downloading MNIST:

```bash
uv sync            # install the environment
./run.sh step5     # results/*.csv.gz → figures/*.pdf + figures/*.png
```

Each figure script can also be run on its own, e.g.
`uv run python scripts/plot_calibration_scatter.py` (the table under Step 5
lists them). Figure 1 trains its tiny example network in a few seconds the
first time and caches the weights in `.cache/`.

### B. Rerun every experiment from scratch (hours)

```bash
./run.sh setup     # uv sync + download MNIST and WBC into data/
./run.sh smoke     # optional: whole pipeline on a tiny sweep in smoke/ (~5 min)
./run.sh all       # steps 1–5: train → estimate → baselines → diagnostics → figures
```

On a fresh clone `outputs/` is empty, so step 1 trains all 230 networks and
steps 2–4 recompute every estimate and overwrite `results/`. Wall time is
dominated by training. Every step skips work whose output already exists, so if
`outputs/` holds an earlier run, use `./run.sh all --force` to recompute
everything (or move `outputs/` aside first). Since the gzipped tables are
written deterministically, `git status results/` afterwards lists only the
tables whose contents changed (plus `provenance.json`, which records the run).

### C. Docker

Needs only Docker (and git for the clone). The `Dockerfile` builds an image
with the locked environment, MNIST, WBC and the stored results; its entry point
is `./run.sh`, so it takes the same commands. Run these from the root of the
clone, which mounts the folders the pipeline writes to (`--user` makes the
files yours, not root's):

```bash
docker build -t intrinsic-it-relu-nets .

# A: figures from the stored results → ./figures
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/figures:/app/figures" \
    intrinsic-it-relu-nets step5

# B: rerun every experiment → ./outputs, ./results, ./figures, ./logs
mkdir -p outputs results figures logs
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$PWD/outputs:/app/outputs" -v "$PWD/results:/app/results" \
    -v "$PWD/figures:/app/figures" -v "$PWD/logs:/app/logs" \
    intrinsic-it-relu-nets all --workers 8
git status results/    # which result tables differ from the stored ones
```

To check the code in the image, run the smoke pipeline and the tests in one
container (the tests read the smoke checkpoints):
`docker run --rm --entrypoint sh intrinsic-it-relu-nets -c "./run.sh smoke && ./run.sh test"`.
Set `--workers` to the number of cores you give the container: inside a
container the CPU count often reports the whole host. Everything runs on the
CPU; the environment uses the CPU-only PyTorch build.

### Notes

`./run.sh step1` … `./run.sh step5` run single steps; `all` and steps 1–4 accept
`--force` to recompute and `--workers N` to limit parallelism (default:
#CPUs − 2). `./run.sh test` runs the unit tests on the smoke checkpoints
(~4 min; `--full`: on `outputs/`).

The smoke run uses one seed, 11 epochs and PCA dims {2, 10} (`src_experiment/smoke.py`)
and writes everything under `smoke/`, never touching `outputs/`, `results/` or `figures/`.
Its figures only show that the code runs; they are not the paper's figures.

### Run times

Measured on a 22-core x86-64 workstation (AVX2) with `--workers 20`. Each network
and each estimation job runs single-threaded, so wall time scales roughly with
the number of workers; the CPU-hours column is wall time × workers.

| Command | Wall time | ≈ CPU-hours |
|---|---|---|
| `./run.sh setup` (environment + MNIST/WBC download) | 1–2 min | — |
| `./run.sh step5` (figures + `results/summary/` from `results/`) | 40 s | — |
| `./run.sh smoke` | 5 min | — |
| `./run.sh test` (on the smoke checkpoints) | 4–5 min | — |
| `./run.sh step1` (train 230 networks) | 71 min | 24 |
| `./run.sh step2` (routing MI, 350 jobs) | 39 min | 13 |
| `./run.sh step3` (baselines, 270 jobs) | 8 min | 3 |
| `./run.sh step4` (diagnostics, 350 jobs) | 6 min | 2 |
| `./run.sh all` (steps 1–5) | **2 h 05 min** | ≈ 42 |

Most of the training time is the 150 MNIST networks (about 3.5 min each on
one core). The Docker image builds in about one minute (1.6 GB).

---

## Pipeline overview

```
configs/generate_*.py   → YAML configs
        ↓
step1_train.sh          → outputs/<sweep>/<experiment>/seed_<seed>.h5, results/training_curves.csv.gz
        ↓
step2_estimate.sh       → results/routing_<sweep>.csv.gz   (routing MI, all estimators/protocols)
        ↓
step3_baselines.sh      → results/baselines_<sweep>.csv.gz  (binning / k-means / KSG grid)
        ↓
step4_diagnostics.sh    → results/region_sizes_<sweep>.csv.gz, results/ordering_<sweep>.csv.gz
        ↓
step5_plot.sh           → figures/*.pdf + figures/*.png, results/summary/ (tables + numbers)

results/provenance.json — git commit, settings and row counts of steps 2–4
```

Each step is **resumable**: already-completed work is detected and skipped.
Pass `--force` to recompute from scratch.

`results/` (gzipped CSVs, read directly by `pandas.read_csv`, plus
`provenance.json`) is tracked in git, and step 5 reads only `results/`, so the
figures can be redrawn without `outputs/` (quick start A). `outputs/`
(checkpoints and per-job CSVs) is not tracked.

---

## Step 1 — Train models (`step1_train.sh`)

Trains four experiment sweeps, writing one HDF5 per (architecture, seed) to
`outputs/<sweep>/`:

| Sweep | Architectures | Labels | Seeds | Models |
|---|---|---|---|---|
| `composite_label_noise` | [5,5,5], [5,5,5,5,5], [9,9,9], [9,9,9,9,9], [25,25,25], [25,25,25,25,25] | clean | 101–105 | 30 |
| `wbc_label_noise` | same 6 | clean | 101–105 | 30 |
| `mnist_capacity` | [3,3,3], [5,5,5] at PCA-10; [7,7,7], [15,15,15], [25,25,25], [50,50,50] at PCA {2,3,4,5,10,15,20} | clean | 101–105 | 150 |
| `label_permutation` | [5,5,5], [25,25,25] on Composite and WBC | training labels randomly permuted (memorization control) | 101–105 | 20 |

**Total: 230 models.** Each trains for 151 epochs (SGD, lr=0.001, momentum=0.9, batch=32),
saving checkpoints at epochs 0, 1, 2, 3, 4, 6, 8, 10, 20, 30, …, 150.

Config YAML files are generated by `configs/generate_*.py` into
`configs/<sweep>/`. `run_training.py --sweeps …` trains every config whose HDF5
does not exist yet, in parallel (`--workers`, default #CPUs − 2; one
single-threaded process per network, log in `logs/train/<sweep>/`). Each HDF5
is written under a temporary name and renamed when training finishes, so an
interrupted run is simply resumed. `run_training.py <config.yaml>` trains one
network.

For `label_permutation`, `src_experiment/dataset.py:permute_labels` shuffles
the training labels with the data seed after the train/test split; test labels
stay true.

### HDF5 layout

Each `seed_<seed>.h5` stores:
- `metadata/` — experiment config as HDF5 attributes
- `epochs/epoch_N/l{i}.weight`, `l{i}.bias` — weights of every layer at each
  saved epoch: hidden layers `i = 1..L`, output layer `i = L+1` (PyTorch
  convention: `W[i].shape == (n_{i+1}, n_i)`)
- `epochs/epoch_N` attributes — the six loss/accuracy values at that epoch
- `training_results/` — the same six curves for every epoch; step 1 exports
  them for all networks to `results/training_curves.csv.gz` (used for App. F)
- `points[N, d]`, `labels[N]` — the test split (never trained on; the `heldout` protocol)

---

## Step 2 — Routing information (`step2_estimate.sh` → `run_estimate.py`)

For every trained network and every estimation protocol,
`src_experiment/functional_quotient.py` computes, at every saved epoch and
hidden layer:

- the data-supported regions Ω_D (MD5 hash of the cumulative activation
  pattern of each point) and ρ = |Ω_D| / N, plus the fraction of singleton
  regions;
- the routing information Î(Y; Ω_D) with six estimators
  (`src_experiment/estimators.py`): plug-in, Miller–Madow, Grassberger,
  Chao–Shen, Chao–Wang–Jost and ANSB;
- for each ε in {0, 0.01, 0.05, 0.1, …, 0.9, 1.0, 1.5, 2.0}, the functional
  quotient (regions merged when their active linear maps are within relative
  Frobenius distance ε; bias excluded; regions visited in first-encounter order)
  and the same six estimators on it, plus ρ_func;
- the network's accuracy on the evaluated points.

Protocols:

| Protocol | Composite | WBC | MNIST |
|---|---|---|---|
| `heldout` — the 20 % test split stored with each network (never trained on) | N = 2 000 | N = 114 | N = 10 000 (MNIST test set) |
| `insample` — all points, train split followed by test split | N = 10 000 | N = 569 | — |

For the `label_permutation` sweep step 2 also evaluates the train split alone
(protocol `train`), against both the true labels and the permuted labels the
network was trained on (column `labels` = `true_labels` / `train_labels`);
every other row has `labels` = `true_labels`.

The figures use `heldout` with `true_labels` (`src_experiment/results.py`).

Output: one CSV per (network, protocol) next to its HDF5, aggregated into
`results/routing_<sweep>.csv.gz` (one row per network × protocol × epoch × layer
× ε; estimator columns `<estimator>_bits` and `<estimator>_func_bits`), and
`results/provenance.json` (git commit, settings, row counts). Jobs run in
parallel (`--workers`, default #CPUs − 2).

---

## Step 3 — MI baselines (`step3_baselines.sh` → `run_baselines.py`)

For every clean network (Composite, WBC, MNIST), every step-2 protocol — so a
baseline and the routing estimate it is compared with see the same points —
and every hidden layer at the last epoch (`--all-epochs` for all), on the
pre-activations T:

| Family | Grid | Columns |
|---|---|---|
| Uniform per-neuron binning over [−max\|T\|, max\|T\|] | K ∈ {2, 4, 8, 16, 30} | `binning<K>_<estimator>_bits`, `binning<K>_num_cells` |
| k-means (n_init = 10, seeded by the network seed) | K ∈ {\|Y\|, 2\|Y\|, 4\|Y\|, 16, 64, 256} (labels `KY`, `2KY`, `4KY`, `16`, …) | `kmeans<label>_<estimator>_bits`, `kmeans<label>_num_cells` |
| KSG / Ross (2014), mixed continuous–discrete | k ∈ {3, 5, 10} | `ksg<k>_bits` (clipped at 0), `ksg<k>_signed_bits` |

Binning and k-means are discrete partitions of T, so they are scored with the
same six estimators as step 2. The figures use binning K = 8 and k-means
K = |Y| with Miller–Madow, and KSG k = 3.

Output: one CSV per (network, protocol), aggregated into
`results/baselines_<sweep>.csv.gz`; settings in `results/provenance.json`.

---

## Step 4 — Diagnostics (`step4_diagnostics.sh` → `run_diagnostics.py`)

At the last epoch:

- **Region sizes** — for every network, protocol and hidden layer, the number
  of data-supported regions holding 1, 2, 3, … points
  (`results/region_sizes_<sweep>.csv.gz`, long format: `region_size`, `num_regions`).
- **Quotient ordering sensitivity** — the functional quotient visits regions in
  first-encounter order (`order` = 0). For every clean network, on the held-out
  points, every hidden layer and ε ∈ {0.1, 0.3, 0.5, 1.0}, it is recomputed
  under 15 random visiting orders (`order` = k uses random seed k), with
  `num_quotient` and every `<estimator>_func_bits` (`results/ordering_<sweep>.csv.gz`).
  Order 0 reproduces step 2 exactly.

---

## Step 5 — Figures (`step5_plot.sh`)

All figures are written to `figures/` as both `.pdf` and `.png`. Step 5 reads
only `results/`, so it needs neither the trained networks nor the datasets.

| File | Script | Description |
|---|---|---|
| `pedagogical_figure1` | `scripts/plot_figure1_pedagogy.py` | Trains a tiny 2→4→4→2 network on moons; shows hyperplane partition and activation-pattern encoding |
| `calibration_scatter_raw` | `scripts/plot_calibration_scatter.py` | Each baseline (x) vs plug-in routing MI (y), one point per clean network (n = 75: same held-out points, last layer, last epoch on both axes); Pearson r annotated |
| `layer_profile_last_epoch` | `scripts/plot_layer_profile_last_epoch.py` | Bits vs layer depth at last epoch; averaged over archs and seeds; ±1σ shading; three datasets |
| `mnist_capacity_bars_per_arch` | `scripts/plot_mnist_capacity_bars.py` | Grouped bars: I_raw and I_func(ε) vs PCA dim; 2×2 grid of arch panels |
| `mnist_rho_vs_eps` | `scripts/plot_mnist_functional_pca_sweep.py --type rho` | ρ_func vs ε at last epoch; 1×4 arch panels (one per width); lines per PCA dim |
| `rho_func_layerwise` | `scripts/plot_rho_func_layerwise.py` | ρ_func by layer for ε ∈ {0, 0.1, 0.3, 0.5, 1.0, 2.0}; three dataset panels |
| `composite_dataset` | `scripts/plot_composite_dataset.py` | App. B: the Composite training split after scaling, coloured by class |
| `training_curves_{composite,wbc,mnist}` | `scripts/plot_training_curves.py` | App. F: test accuracy and loss over epochs (mean ± std over seeds) |

The figures use the `heldout` protocol with `true_labels`.

Step 5 also runs `scripts/summarize_results.py`, which writes the appendix
tables and every number quoted in the text to `results/summary/` (tracked in
git):

| File | Content |
|---|---|
| `calibration.csv` | Fig. 2: n and Pearson r per baseline |
| `bias_corrections_by_cell.csv`, `bias_corrections_by_rho.csv` | Miller–Madow, Grassberger, Chao–Shen, Chao–Wang–Jost and ANSB: mean, seed-std and correction per (dataset, architecture, PCA dim, layer), and averaged over ρ ranges |
| `heldout_vs_insample.csv` | held-out vs in-sample routing MI and ρ (Composite, WBC) |
| `baseline_sensitivity.csv` | Pearson r of the routing MI against every binning / k-means / KSG setting of step 3 |
| `occupancy.csv` | ρ and singleton fractions per configuration (deepest layer) |
| `label_permutation.csv` | clean vs label-permuted networks: raw and quotient MI (ε = 0.3), ρ, accuracy |
| `ordering_by_cell.csv`, `ordering_by_epsilon.csv` | spread of the functional quotient over 16 visiting orders |
| `numbers.json`, `SUMMARY.md` | every number quoted in the text, under a stable key, with the selection behind it |

Unless a file says otherwise these use the clean networks, the last epoch, the
held-out points with true labels and the mean over seeds.

---

## Tests and CI

```bash
./run.sh test              # all tests; checkpoint tests on the smoke run (after ./run.sh smoke)
./run.sh test --full       # all tests; checkpoint tests on outputs/ (after step 1)
uv run pytest tests        # same as --full; in a fresh clone (no outputs/) only the fast tests run (~10 s)
```

The fast tests need no data or checkpoints: they build a tiny trained network
(`tests/conftest.py`) and check the estimators, the functional quotient, the
baselines, the step 2–4 jobs end to end (step 4's first-encounter order
reproduces step 2), deterministic result files and the thread-independent MNIST
PCA. The checkpoint tests cross-check the region code against `parx` and the
step-2 rows against the routing estimator.

GitHub Actions (`.github/workflows/ci.yml`) runs the fast tests (with the
`infomeasure` cross-check) and step 5 on every push and pull request; step 5
must reproduce `results/summary/` exactly. On `main` (and on manual dispatch)
it also runs the smoke pipeline plus the tests on it, and builds the Docker
image and runs step 5 inside it without network. The `site` job checks that the
project page's JavaScript estimators reproduce the pipeline's region counts,
quotient sizes and MI exactly on the exported networks.

---

## File structure

```
intrinsic_IT_analysis_of_relu_nets/
├── README.md                      ← this file
├── .github/workflows/ci.yml       ← GitHub Actions: tests, step 5, smoke, Docker, site tests
├── .github/workflows/pages.yml    ← publishes site/ to GitHub Pages
├── site/                          ← project page with interactive demos (see site/README.md)
├── run.sh                         ← entry point (setup/test/smoke/all/stepN)
├── Dockerfile                     ← reproduction image (entry point ./run.sh)
├── run_all.sh                     ← steps 1–5 end-to-end
├── step1_train.sh                 ← train 230 models
├── step2_estimate.sh              ← compute routing MI
├── step3_baselines.sh             ← compute MI baselines
├── step4_diagnostics.sh           ← region sizes, quotient ordering sensitivity
├── step5_plot.sh                  ← generate figures
│
├── configs/
│   ├── generate_composite.py      ← generates configs/composite_label_noise/
│   ├── generate_wbc.py            ← generates configs/wbc_label_noise/
│   ├── generate_mnist.py          ← generates configs/mnist_capacity/
│   └── generate_label_permutation.py ← generates configs/label_permutation/
│
├── run_training.py                ← step 1: training (one config or whole sweeps, parallel)
├── run_estimate.py                ← step 2: routing MI, all sweeps/protocols
├── run_baselines.py               ← step 3: MI baselines, all clean networks/protocols
├── run_diagnostics.py             ← step 4: region sizes, ordering sensitivity
│
├── scripts/
│   ├── plot_figure1_pedagogy.py
│   ├── plot_calibration_scatter.py
│   ├── plot_layer_profile_last_epoch.py
│   ├── plot_mnist_capacity_bars.py
│   ├── plot_mnist_functional_pca_sweep.py
│   ├── plot_rho_func_layerwise.py
│   ├── plot_composite_dataset.py
│   ├── plot_training_curves.py
│   ├── summarize_results.py       ← appendix tables + quoted numbers → results/summary/
│   └── export_site_data.py        ← demo data for the project page → site/data/
│
├── src_experiment/                ← Python package (estimators + training)
│   ├── dataset.py                 ← data loading (composite, WBC, MNIST)
│   ├── train_models.py            ← SGD training loop with epoch callbacks
│   ├── run_experiment.py          ← orchestrates training + HDF5 saving
│   ├── utils.py                   ← NeuralNet, savefig
│   ├── routing_estimator.py       ← regions Ω_D: forward pass + pattern hashing
│   ├── estimators.py              ← plug-in, MM, Grassberger, Chao–Shen, CWJ, ANSB
│   ├── results.py                 ← loads results/{routing,baselines}_<sweep>.csv.gz by protocol
│   ├── functional_quotient.py     ← ε-functional quotient + per-network estimates (step 2)
│   ├── probe_loader.py            ← in-sample / probe sets per dataset
│   ├── paths.py                   ← figure output path (→ figures/)
│   ├── smoke.py                   ← seeds/epochs/PCA dims (full vs ./run.sh smoke)
│   └── baselines/
│       ├── activations.py         ← forward-pass activation extraction
│       └── mi_baselines.py        ← per-neuron binning, KSG
│
├── tests/                         ← ./run.sh test (smoke checkpoints) / --full (outputs/)
│   ├── conftest.py                ← tiny trained network in the step-1 HDF5 layout
│   ├── test_estimators.py         ← estimators vs hand values, mpmath and infomeasure
│   ├── test_quotient.py           ← active subnetwork, ε-clustering, first-encounter order
│   ├── test_baselines.py          ← binning, KSG (vs scikit-learn), activations (vs PyTorch), grid
│   ├── test_pipeline_jobs.py      ← steps 2–4 jobs end to end, result I/O, PCA determinism
│   ├── test_routing_pipeline.py   ← step 2 consistency (regions, quotient, protocols)
│   ├── test_label_permutation.py  ← step 2 rebuilds the permuted training labels
│   └── test_parx_partitions.py    ← region code cross-checked against parx
├── outputs/                       ← created by step 1 (HDF5 checkpoints)
├── results/                       ← created by steps 2–4 (aggregated CSVs + provenance.json)
├── figures/                       ← created by step 5 (PDF + PNG)
└── logs/                          ← timestamped log files per step
```

---

## Design notes

- **No Julia required.** The routing estimator computes activation patterns
  directly from network weights via a standard ReLU forward pass; no polytope
  enumeration is needed. (`tests/test_parx_partitions.py` cross-checks the
  regions against the `parx` partition package.)

- **HDF5 as checkpoint format.** The training loop saves weights, curves and
  the test split to HDF5. Every later step reads only these files (plus the
  regenerated in-sample sets), so steps 2–5 can be rerun without retraining.

- **Determinism.** Global seed 42 controls all data splits and shuffles.
  Model seed (101–105) controls weight initialisation only. Both seeds are
  set independently across PyTorch, NumPy, and Python random. Each network
  trains in a single thread, and the MNIST PCA runs with one BLAS thread
  (`dataset._pca_project`): its rounding otherwise depends on the thread
  count, i.e. on the machine. Retraining is then bit-identical on the same
  kind of CPU; across CPU types (e.g. AVX2 vs AVX-512) the BLAS kernels can
  differ in the last digits.
- **Pinned environment.** `uv.lock` pins every package; PyTorch is the
  CPU-only build of 2.11.0, which retrains the networks bit-identically to
  the CUDA build. No GPU is used.

- **Protocols are explicit.** Every result row records its `protocol`
  (`heldout`, `insample`, `train`) and `labels`, and the figure scripts select
  them through `src_experiment/results.py`, so results from different point
  sets are never mixed. Merges between steps are checked one-to-one.

- **Parallel and resumable.** Steps 1–4 run one job per network (and protocol)
  in parallel and write each result atomically, so an interrupted step resumes
  where it stopped; `--force` recomputes. `results/provenance.json` records the
  git commit and settings that produced the aggregated results.

- **Smoke mode.** `src_experiment/smoke.py` holds the sweep sizes; with
  `SMOKE=1` (`./run.sh smoke`) every step runs on one seed, 11 epochs and two
  PCA dimensions under `smoke/`, and `./run.sh test` checks the code against it.
