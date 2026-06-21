# Scripts

Helper scripts for data acquisition, signature measurement, graph generation, and experiment analysis. All scripts are run from the repo root via `python scripts/<name>.py`.

## Data Acquisition

### `get_data.py`
Downloads the AIFB knowledge graph from Figshare and saves it as `aifb.ttl`. One-shot utility; no arguments.

## Signature Measurement

### `measure_signature.py`
Computes the full graph signature (Blocks A–F) for a single KG file and writes block plots, JSON results, and a text summary to `sig_out/<graph>/` (or `--output-dir`).

```
python scripts/measure_signature.py data/graphs/aids/AIDS.nt
python scripts/measure_signature.py mygraph.ttl --output-dir out/ --format pdf
```

### `measure_signature_reduced.py`
Same as above but for the reduced (non-over-determined) signature. Writes a `signature/` directory next to the graph file by default, matching the `data/graphs/<name>/` corpus layout.

```
python scripts/measure_signature_reduced.py data/graphs/aids/AIDS.nt
```

### `measure_block_e.py`
Backfills Block E (motif counts, G5 colour-coding) into an already-measured corpus under `data/graphs/`. Skips graphs where `block_e.json` already exists; use `--force` to recompute.

```
python scripts/measure_block_e.py                # all graphs
python scripts/measure_block_e.py aids swdf      # named graphs only
python scripts/measure_block_e.py --force
```

### `measure_all_raw.py`
Batch-measures the full corpus under `data/graphs/` by calling the per-graph measurement scripts as subprocesses. Supports `--reduced` to run the reduced signature and `--blocks` to re-measure only a subset of blocks.

```
python scripts/measure_all_raw.py
python scripts/measure_all_raw.py --reduced
python scripts/measure_all_raw.py --blocks e    # re-measure Block E only
```

## Graph Generation & Round-trip

### `sample_signature.py`
Draws a novel reduced signature from the measured corpus using the `UniformRangeSampler` (each feature sampled uniformly over its corpus range ±10 %). Prints JSON to stdout or `--out`.

```
python scripts/sample_signature.py --seed 42
python scripts/sample_signature.py --out sampled.json
```

### `signature_roundtrip.py`
Full pipeline test: loads a target reduced signature for a named graph, runs Stage 3 to generate a synthetic graph, re-measures it, and compares the result to the target. Useful for end-to-end validation.

```
python scripts/signature_roundtrip.py aids
python scripts/signature_roundtrip.py wn18rr_v4 --seed 7 --rewire-budget 5000
python scripts/signature_roundtrip.py --kg-file path/to/graph.ttl
```

## Experiment Sweeps

### `sweep_collect.py`
Sweeps Stage 3 hyperparameters (`rewire_budget × remeasure_interval × seed`) and saves per-run synthetic signatures to `experiments/<graph>.jsonl` for later analysis. The target signature is written once to `experiments/<graph>_target.json`.

```
python scripts/sweep_collect.py fb237_v4_ind
python scripts/sweep_collect.py fb237_v4_ind --budgets 500 2000 5000 --intervals 200 2000 --seeds 0 1 2 3 4
python scripts/sweep_collect.py fb237_v4_ind --append
```

### `cc_variance.py`
Measures colour-coding estimator variance for Block E motif counts. Runs the `CCMotifCounter` with N seeds on one graph, computes the exact ground truth once via `ExactMotifCounter`, and produces a grid of boxplots (rows = features, columns = `n_samples`) with CV annotations. Output goes to `experiments/cc_variance_sweeps/`.

```
python scripts/cc_variance.py wn18rr_v4
python scripts/cc_variance.py fb237_v4_ind --n-runs 100 --n-samples 10000 50000
python scripts/cc_variance.py wn18rr_v4 --n-colorings 1 4 16 64 --n-samples 1000 10000 100000
```

## Visualisation

### `convergence_plot.py`
Plots Stage 3 convergence curves from one or more CSV files produced by the `refine()` loop. All metric columns (relative errors per feature) are plotted against a 0-reference line. Useful for diagnosing convergence speed and motif-count accuracy during generation.

```
python scripts/convergence_plot.py experiments/conv_a.csv experiments/conv_b.csv
python scripts/convergence_plot.py experiments/conv_a.csv --features tri_err cc_err --out fig.png
python scripts/convergence_plot.py experiments/conv_a.csv --list-features
```

### `sweep_viz.py`
Visualises per-feature relative-error distributions from a sweep JSONL file produced by `sweep_collect.py`. Supports box and violin plots; can list available features.

```
python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl --features triangle_count four_cycle_count
python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl --kind violin --out fig.png
python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl --list-features
```

### `plot_signature_distributions.py`
Loads all `signature.json` files from a corpus and plots component-wise value distributions across graphs (one figure per block). Reads the full signature from `sig_out/` by default; `--reduced` reads the reduced signature from `data/graphs/`.

```
python scripts/plot_signature_distributions.py
python scripts/plot_signature_distributions.py --reduced
python scripts/plot_signature_distributions.py --source my_sigs/ --out my_plots/
```
