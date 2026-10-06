# Project page — design and development notes

The project page for the NeurIPS 2026 paper, served by GitHub Pages from this
folder (deployed by `.github/workflows/pages.yml`). Plain HTML/CSS/JavaScript,
no build step and no external dependencies: the demos compute everything in
the browser from small JSON files exported from the pipeline.

## Page structure

1. **Header** — title, authors and affiliations (from `CITATION.cff`), venue
   (NeurIPS 2026), buttons: Paper (PDF), arXiv, OpenReview, Code.
2. **Teaser** — the pedagogical figure (partition → activation pattern → region).
3. **Abstract** — the camera-ready abstract (filled in by task `web-content`).
4. **Demos** — four interactive sections (below), each with a two-line
   explanation and the quantities it displays.
5. **BibTeX** — copy button.
6. **Footer** — links to the code, the reproduction instructions, licence.

Responsive single column, light and dark mode (`prefers-color-scheme`), all
figures drawn on `<canvas>`.

## Demos

All four use real trained networks from the paper's pipeline (Composite, seed
101, last epoch unless an epoch slider is shown) and the same 2,000 held-out
Composite points the paper's estimates use.

| # | Demo | Controls | Shows (live) | Data |
|---|---|---|---|---|
| 1 | **Linear regions and routing information** | network ([5,5,5] / [9,9,9] / [25,25,25]), layer 1–3, training epoch (22 checkpoints) | the input square coloured by data-supported region Ω_D at that layer; data points by class; \|Ω_D\|, ρ = \|Ω_D\|/N, Î(Y; Ω_D), H(Y) | weights at every saved epoch; points + labels |
| 2 | **Functional quotient** | network, layer, ε ∈ [0, 2] | regions merged into quotient classes (first-encounter order, relative Frobenius criterion, linear part only); \|Ω_func\|, ρ_func, Î_func vs Î | same weights (last epoch) |
| 3 | **Plug-in bias as ρ → 1** | sample size N, number of possible regions K, true dependence | simulated routing: plug-in and Miller–Madow vs the true MI as ρ grows; the paper's 138 network cells (ρ, Î, Ĩ_MM) as reference points | `results/summary/bias_corrections_by_cell.csv` (subset) |
| 4 | **Label permutation** | network ([5,5,5] / [25,25,25]), clean vs permuted | both partitions side by side; test accuracy, Î and Î_func(ε = 0.3) with the true labels — high routing information without learned structure | clean + permuted weights (last epoch) |

Demos 1, 2 and 4 recompute everything from the weights: a ReLU forward pass on
a pixel grid and on the 2,000 points, activation patterns, region hashing, the
plug-in and Miller–Madow MI, the active subnetwork Ã of every data-supported
region and the ε-clustering. Nothing is precomputed per ε or per pixel.

## Data (`site/data/`, tracked)

Written by `uv run python scripts/export_site_data.py` from the trained
networks in `outputs/` (step 1) and `results/`:

- `points.json` — the 2,000 held-out Composite points (scaled to [−1, 1]²) and labels.
- `networks.json` — per network: architecture, condition (clean / permuted),
  seed, and the weights at each exported epoch (float32 values).
- `reference.json` — the Python pipeline's values for the same networks
  (regions, quotient sizes and MI per layer and ε, from `results/routing_*.csv.gz`),
  used by the JavaScript tests.
- `bias_cells.json` — ρ, Î, Ĩ_MM, H(Y) of the 138 network cells (demo 3).

## Code

- `index.html`, `style.css` — the page (light/dark via `prefers-color-scheme`, one column below 720 px).
- `js/relu.js` — forward pass (pre-activations rounded to float32), activation
  patterns, region ids in first-encounter order, active subnetwork.
- `js/mi.js` — plug-in and Miller–Madow MI; `js/quotient.js` — ε-clustering in
  first-encounter order (a port of `src_experiment/functional_quotient.py`).
- `js/view.js` — a network evaluated on the pixel grid and on the data points.
- `js/plot.js` — canvas partition maps and charts; `js/ui.js` — small UI helpers.
- `js/demo-*.js` — one module per demo; `js/main.js` loads the data and starts
  each demo when it scrolls into view.
- `tests/` — Node tests (`cd site && npm test`) that run the JS estimators on
  the exported networks and compare them with `reference.json`: region counts,
  quotient sizes and accuracies are identical, and the MI agrees to rounding
  error. CI runs them on every push.

## Local preview

```bash
python -m http.server -d site 8000    # then open http://localhost:8000
```

## Deployment

`.github/workflows/pages.yml` publishes `site/` on every push to `main` that
touches it. GitHub Pages needs the repository to be public (or a paid plan):
enable it under Settings → Pages → Source: GitHub Actions once the repository
is public (task `code-release`).
