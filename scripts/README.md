# Scripts

Helper scripts for data acquisition, signature measurement, graph generation, and experiment analysis. All scripts are run from the repo root via `python scripts/<name>.py`.

> For the three core operations — measure a KG, generate a synthetic one, compare two — prefer the
> installed CLI (`pip install -e .`):
>
> ```
> kgsynth measure  data/graphs/swdf/swdf.nt
> kgsynth generate swdf --seed 42 --rewire-budget 50000
> kgsynth compare  graph_a.ttl graph_b.ttl
> ```
>
> The scripts below remain the place for research workflows the CLI deliberately doesn't cover:
> parameter sweeps, Stage-3 convergence logging, and diagnostic plots.

## Data Acquisition

### `get_data.py`
Downloads the AIFB knowledge graph from Figshare and saves it as `aifb.ttl`. One-shot utility; no arguments.

## Signature Measurement

### `measure_signature.py`
Computes the graph signature (Blocks A–F) for a single KG file and writes block plots, JSON results, and a text summary to a `signature/` directory next to the graph file (i.e. `data/graphs/<name>/signature/`), or to `--output-dir`. Equivalent to the installed `kgsynth measure` CLI.

```
python scripts/measure_signature.py data/graphs/aids/AIDS.nt
python scripts/measure_signature.py mygraph.ttl --output-dir out/ --format pdf
```

### `measure_block_e.py`
Backfills Block E (motif counts, G5 colour-coding) into an already-measured corpus under `data/graphs/`. Skips graphs where `block_e.json` already exists; use `--force` to recompute.

```
python scripts/measure_block_e.py                # all graphs
python scripts/measure_block_e.py aids swdf      # named graphs only
python scripts/measure_block_e.py --force
```

### `measure_all_raw.py`
Batch-measures every graph under `data/graphs/` **and** the held-out test corpus `data/test_graphs/` by calling `measure_signature.py` as a subprocess per graph. Each graph's `signature/` directory is written next to its graph file. `--blocks` re-measures only a subset of blocks. `--graphs` restricts the run to specific graphs by directory name.

```
python scripts/measure_all_raw.py
python scripts/measure_all_raw.py --blocks e             # re-measure Block E only
python scripts/measure_all_raw.py --graphs aids fb237_v4  # only these graphs
```

## Graph Generation & Round-trip

### `sample_signature.py`
Draws a novel reduced signature from the measured corpus using the `UniformRangeSampler` (each feature sampled uniformly over its corpus range ±10 %). Prints JSON to stdout or `--out`.

```
python scripts/sample_signature.py --seed 42
python scripts/sample_signature.py --out sampled.json
```

### `signature_roundtrip.py`
Full pipeline test: loads a target reduced signature for a named graph, runs Stage 3 to generate a synthetic graph, re-measures it, and compares the result to the target. Useful for end-to-end validation. `--convergence-log` records the Stage-3 convergence CSV (auto-named into `experiments/convergence_logs/`; plot with `convergence_plot.py`); `--swap-log` records one row per evaluated Stage-3 swap proposal — per-motif deltas, Δloss, accepted — auto-named into `experiments/swap_delta_logs/` (plot with `swap_delta_viz.py`). `--skip-c5` / `--skip-c6` force 5-/6-cycle steering off in Stage 3 (`use_c5` / `use_c6` = False), dropping that cycle size's per-swap delta and loss term regardless of the target count. During the Stage-3 rewiring loop, pressing **ESC** or **`q`** (checked every 10 steps) breaks out early and returns the best graph found so far — useful for cutting a long run short without losing progress (no-op when stdout isn't an interactive terminal). On early escape, auto-named `--convergence-log` / `--swap-log` filenames — which otherwise encode the planned `--rewire-budget` as `rb<N>` — are renamed to the number of steps actually executed, so the filename reflects the real run, not the requested one.

```
python scripts/signature_roundtrip.py aids
python scripts/signature_roundtrip.py wn18rr_v4 --seed 7 --rewire-budget 5000
python scripts/signature_roundtrip.py wn18rr_v4 --swap-log
python scripts/signature_roundtrip.py wn18rr_v4 --skip-c5 --skip-c6
python scripts/signature_roundtrip.py --kg-file path/to/graph.ttl
```

## Experiment Sweeps

### `sweep_collect.py`
Sweeps Stage 3 hyperparameters (`rewire_budget × seed`) and saves per-run synthetic signatures to `experiments/<graph>.jsonl` for later analysis. The target signature is written once to `experiments/<graph>_target.json`.

```
python scripts/sweep_collect.py fb237_v4_ind
python scripts/sweep_collect.py fb237_v4_ind --budgets 500 2000 5000 --intervals 200 2000 --seeds 0 1 2 3 4
python scripts/sweep_collect.py fb237_v4_ind --append
```

### `sweep_adaptive_weight_scale.py`
Sweeps Stage 3's `ADAPTIVE_WEIGHT_SCALE` and reports the value that minimises the accumulated **unweighted** error across all steered motifs/metrics after a fixed rewire budget. Stages 1–2 are run once to build a fixed pre-Stage-3 graph; each candidate scale then runs Stage 3 (`adaptive_weights=True`) from a fresh copy of that graph, so only the loss-weighting scheme differs. Comparison metric is `stage3_best_unweighted_error_sum` (independent of the scale itself, so runs are directly comparable). Output CSV via `--out`.

```
python scripts/sweep_adaptive_weight_scale.py wn18rr_v4
python scripts/sweep_adaptive_weight_scale.py wn18rr_v4 --scales 1 5 10 20 30 50 75 100 --rewire-budget 100000
python scripts/sweep_adaptive_weight_scale.py wn18rr_v4 --seed 7 --out sweep.csv
```

### `cc_variance.py`
Collects the **exact-vs-CC counter benchmark** (accuracy + runtime) per motif size for Block E motif counts. Runs `CCMotifCounter` with N seeds over an `n_samples × n_colorings` grid, recording per seed the estimated counts **and the wall-clock time of each family call** (`runtime_triangle_s`, `runtime_motif4_s`, `runtime_motif5_s`, `runtime_motif6_s`, `runtime_stars_s`). The exact ground-truth counts and per-family exact runtimes are computed once via `ExactMotifCounter` and stored in the `_meta.json` sidecar. Covers triangle (k=3), 4-node motifs (k=4), 5-cycle (k=5), 6-cycle (k=6, exact via the ESCAPE enumerator), and stars k=2..10. Output goes to `experiments/cc_variance_sweeps/`. The exact-baseline phase logs per-family progress (`[1/4] triangle … [4/4] stars`). Two degree guards keep that phase tractable on hub graphs and leave the affected ground-truth values `None` (CC estimates are still swept): `--exact-max-degree` (default 100) gates exact c5/c6 ESCAPE enumeration, and a fixed degree-50 guard (`_STAR_EXACT_MAX_DEGREE`, matching the counter's `_HUB_THRESH`) skips exact stars when any hub would trigger the intractable `C(d,k)` subset enumeration.

Caveats: the triangle is counted with `list_triangles` in *both* counters (exact in each — not a real sampler race, variance 0); stars are counted jointly (one call yields k=2..10), so `runtime_stars_s` is a single value for the family while accuracy stays per-k. `--n-timings N` averages the exact runtime over N repeats (the exact counter has no seed axis); `--exact-max-degree D` (default 100) raises the ESCAPE degree guard so an isolated hub — e.g. wn18rr_v4's single degree-68 node — doesn't suppress the exact c5/c6 baseline (exact c6 on wn18rr_v4 takes ~2.5 min). `--skip-exact` bypasses the exact ground-truth phase entirely (truth counts and exact per-family runtimes are all recorded as `None`); use it when exact enumeration is intractable or only the CC variance/runtime is of interest.

```
python scripts/cc_variance.py wn18rr_v4
python scripts/cc_variance.py fb237_v4_ind --n-runs 100 --n-samples 10000 50000
python scripts/cc_variance.py wn18rr_v4 --n-colorings 1 4 16 64 --n-samples 1000 10000 100000
python scripts/cc_variance.py wn18rr_v4 --n-timings 5            # smooth exact runtime
python scripts/cc_variance.py wn18rr_v4 --skip-exact            # CC sweep only, no exact baseline
```

### `cc_variance_viz.py`
Renders the data collected by `cc_variance.py`. Writes two figures next to the CSV: `<csv>.png` — accuracy boxplots (rows = features, columns = `n_samples`, one box per `n_colorings`, exact ground truth as a horizontal line) with a coefficient-of-variation table printed to the console; and `<csv>_runtime.png` (when the timing columns are present) — mean CC runtime vs `n_samples` per motif family (one line per `n_colorings`, log-log) with the exact per-family runtime as a reference line.

```
python scripts/cc_variance_viz.py experiments/cc_variance_sweeps/wn18rr_v4_sweep.csv
python scripts/cc_variance_viz.py <csv> --out fig.png --meta path/to/meta.json
```

### `profile_stage3_deltas.py`
Profiles the **per-swap incremental delta cost** of Stage 3 on a graph's Stage-2 synthetic output (diagnoses slow `refine()` runs, e.g. fb237_v4). Rebuilds the exact Stage-2 graph a `signature_roundtrip.py` run feeds into `refine()` (same derived seeds from `--seed`), replays Stage-3's uniform swap-proposal sampling, and times `_triangle_node_delta`, `_motif4_delta`, and `_cycle_delta` (k=5 and k=6 separately, unguarded, plus the node-level-guarded k5+k6 call Stage 3 actually runs with the current `CYCLE_DELTA_MAX_DEGREE`) per proposal, each bounded by `--timeout` seconds via SIGALRM (timed-out costs are censored at the cap, so aggregates are lower bounds). Writes per-proposal timings + endpoint degrees (`proposals_<graph>_seed<seed>.csv`), the Stage-2 degree distribution (`degree_stats_<graph>_seed<seed>.csv`), and a regenerated `summary.md` aggregating every profiled graph, all to `experiments/stage3_delta_profiling/`.

```
python scripts/profile_stage3_deltas.py fb237_v4 wn18rr_v4
python scripts/profile_stage3_deltas.py fb237_v4 --proposals 300 --timeout 5
```

### `estimator_variance.py`
Characterises the **variance of the Horvitz–Thompson neighbour-subsampling estimator** for the induced 5-/6-cycle count (the "approximate hub delta" idea) as a function of endpoint node degree, for several sample counts `K`, and fits a power law `rel_std = a·deg^b` per `K`. For each hub swap it computes the exact cycle set through the four changed pairs and Monte-Carlo simulates the estimator over it (the estimator is unbiased, so relative std is what decides usability). `--metric count` (default) measures the count estimator; `--metric delta` the far noisier after−before delta. Writes per-(proposal, K) CSVs and a log-log rel-std-vs-degree scatter with fitted curves (one per `K`) to `experiments/estimator_variance/`, and prints the fit parameters. Conclusion from the fb237 run is recorded in `docs/notes/stage3_steering_analysis.md` §4 (variance grows steeply with degree — the estimator is unusable on the hubs where it would be needed).

```
python scripts/estimator_variance.py fb237_v4
python scripts/estimator_variance.py fb237_v4 --k 5 6 --samples 8 16 32 64 --per-bin 18
python scripts/estimator_variance.py fb237_v4 --metric delta --bins 20 50 100 200 450
```

### `edge_multiplicity.py`
Surveys the directed→simple **edge-multiplicity (pair-overlap)** gap between a graph's original and its Stage-2 synthetic output (built as `signature_roundtrip` with zero refinement), per graph or across the corpus. Reports ρ = directed/distinct-undirected, the parallel (multi-relational) and bidirectional factors, and the synthetic edge inflation. `--orig-only` skips the (slow) Stage-2 build for a fast survey of how much overlap the targets demand. Output to `experiments/edge_multiplicity/`. Diagnoses the root cause behind the fb237 motif overshoot (see `docs/notes/motif_reachability_and_edge_multiplicity.md`).

```
python scripts/edge_multiplicity.py                    # all corpus graphs
python scripts/edge_multiplicity.py fb237_v4 wn18rr_v4
python scripts/edge_multiplicity.py --orig-only
```

### `relation_reciprocity.py`
Surveys **per-relation reciprocity** and forward/inverse-CS symmetry — testing whether bidirectionality is carried by a per-relation "symmetric vs asymmetric" split (it is, nearly bimodally). Per relation reports the same-relation edge reciprocity and `|S_r∩O_r|/|S_r∪O_r|`; per graph the edge-weighted overall reciprocity, symmetric-edge fraction, mid-band fraction (bimodality) and CS↔inv-CS Jaccard. Writes per-relation CSVs + a summary to `experiments/relation_reciprocity/`. Findings in `docs/notes/relation_reciprocity_and_bidirectionality.md`.

```
python scripts/relation_reciprocity.py fb237_v4 wn18rr_v4
python scripts/relation_reciprocity.py            # all corpus graphs
python scripts/relation_reciprocity.py --top 15
```

## Visualisation

### `convergence_plot.py`
Plots Stage 3 convergence curves from one or more CSV files produced by the `refine()` loop. All metric columns (relative errors per feature) are plotted against a 0-reference line. Useful for diagnosing convergence speed and motif-count accuracy during generation.

```
python scripts/convergence_plot.py experiments/conv_a.csv experiments/conv_b.csv
python scripts/convergence_plot.py experiments/conv_a.csv --features tri_err cc_err --out fig.png
python scripts/convergence_plot.py experiments/conv_a.csv --list-features
```

### `convergence_plot_grid.py`
Like `convergence_plot.py`, but always renders a fixed 2×2 grid of the same four `*_err` columns (triangle, diamond, c6, paw) instead of an arbitrary `--features` list — a fixed side-by-side view for comparing runs (e.g. fixed-weight vs. adaptive-weight) without re-specifying `--features`. Same convergence-CSV input.

```
python scripts/convergence_plot_grid.py experiments/conv_a.csv
python scripts/convergence_plot_grid.py experiments/conv_a.csv experiments/conv_b.csv --out experiments/convergence_grid.png
```

### `signature_error_boxplot.py`
Per-block error boxplot for one graph's roundtrip: reconstructs each block's distribution and reports **Wasserstein-1** distance for distributional features and plain relative error for standalone scalars, on one comparable scale — showing *which* blocks carry the roundtrip error (the per-block companion to `signature_roundtrip.py`'s scalar + W1 tables). A couple of extreme outliers (`b:a_obj`, `d:cs_freq` W1) are excluded by default; `--exclude` disables the exclusion.

```
python scripts/signature_error_boxplot.py wn18rr_v4
python scripts/signature_error_boxplot.py wn18rr_v4 --synth-dir signature_synth_20260706_184120
python scripts/signature_error_boxplot.py wn18rr_v4 --out data/graph_population/error_boxplot.png
```

### `plot_out_degree_standalone.py`
Renders a standalone out-degree distribution panel from a measured `block_b.json`, reusing `BlockB._plot_degree_hist` as a single-axes figure with poster-matched colours. Useful for pulling one publication-ready degree plot out of the full block_b diagnostic grid.

```
python scripts/plot_out_degree_standalone.py data/test_graphs/wn18rr_v4/signature/block_b.json --out out_degree_dist.png
```

### `viz_sampling_approaches.py`
Conceptual (non-data) illustration contrasting two signature-sampling strategies over a toy signature space: **Signature Sampling** (one fitted joint density, which smears mass into empty gaps under p ≫ n) vs. **Signature Varying** (a Gaussian bump anchored on each measured signature, keeping every draw near a real graph). Datapoints are illustrative, not measured graphs.

```
python scripts/viz_sampling_approaches.py
```

### `sweep_viz.py`
Visualises per-feature relative-error distributions from a sweep JSONL file produced by `sweep_collect.py`. Supports box and violin plots; can list available features. With no `--features`, all non-NaN features are plotted. A final `mean |rel err|` panel aggregates the shown features — per (config, seed) it averages the absolute relative error across all shown features, then box-plots those per-seed means across seeds in the same style — giving one overall error level per config (also printed in the console mean ± std table).

```
python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl  # all features
python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl --features triangle_count four_cycle_count
python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl --kind violin --out fig.png
python scripts/sweep_viz.py experiments/fb237_v4_ind.jsonl --list-features
```

### `swap_delta_viz.py`
Analyses a Stage-3 swap-proposal log (from `signature_roundtrip.py --swap-log` / `refine(swap_log=…)`). Writes five outputs next to the CSV: `<csv>.png` — per-motif histograms of nonzero deltas (accepted vs rejected overlaid, a grey Δ=0 bar, zero-delta fraction in each panel title); `<csv>_leverage.png` — per-motif |delta| vs max endpoint degree scatters plus a cumulative-leverage curve (share of total |delta| carried by the top-x% of proposals); `<csv>_metrics.csv` — per-motif summary (zero-delta %, |delta| percentiles, top-1 %/10 % leverage shares, accept rates); `<csv>_loss.png` — swap *usefulness*: the signed loss-Δ distribution (accepted vs rejected; mass left of 0 is useful since loss is minimised) and a cumulative-usefulness curve; `<csv>_loss_metrics.csv` — useful % (accepted & Δloss<0), improving/neutral/harmful %, accept rates per class, and improvement concentration. Metrics also print to the console. Guard-dropped delta cells are excluded from stats and counted in the panel titles. Built to assess whether an approximate hub delta is viable and how many attempted swaps actually help.

```
python scripts/swap_delta_viz.py experiments/swap_delta_logs/swaps_wn18rr_v4_seed42_rb5000.csv
python scripts/swap_delta_viz.py <csv> --motifs d_c4 d_c6 --out fig.png
```

### `plot_signature_distributions.py`
Loads all `signature.json` files from a corpus and plots component-wise value distributions across graphs (one figure per block, A–F). Reads from `data/graphs/` and writes to `data/graph_population/`.

```
python scripts/plot_signature_distributions.py
python scripts/plot_signature_distributions.py --source my_sigs/ --out my_plots/
```

### `plot_signature_pca.py`
Fits a 2D PCA basis on the corpus's `signature.json` files (grey dots = real graphs), then projects one or more `signature_roundtrip.py` original/synthetic pairs into that space as a highlighted, arrow-connected pair — the arrow shows how far Stage 3 output drifts from its target. `--size-agnostic` drops raw size-dependent features (entity/motif counts, degree extrema) so the projection reflects structural shape rather than graph size. Provides `_find_corpus_signatures`/`_load_signature_json`/`_build_matrix`/`_fit_pca_2d`/`_project`/`_PAIR_COLOURS`, reused by `plot_sweep_pca.py` and `signature_pca_trajectory.py` below (the shared PCA-fitting math lives here once).

```
python scripts/plot_signature_pca.py wn18rr_v4
python scripts/plot_signature_pca.py wn18rr_v4 fb237_v4_ind --size-agnostic
```

### `plot_sweep_pca.py`
Projects a `sweep_collect.py` run (many synthetic seeds vs. one target) into the same corpus-fit PCA space as `plot_signature_pca.py`, showing the spread of independent generator draws around the target as a cloud rather than a single pair. Multiple graphs get distinct colours; multiple rewire budgets get distinct marker shapes. `--exclude` drops named corpus graphs from the PCA fit and the plotted cloud.

```
python scripts/plot_sweep_pca.py wn18rr_v4
python scripts/plot_sweep_pca.py wn18rr_v4 swdf --budget 50000
```

### `signature_pca_trajectory.py`
Runs one `signature_roundtrip.py`-style generation, but snapshots the graph right after Stage 2 and at evenly-spaced points through Stage 3's rewire budget (via `Generator.sample(checkpoint_steps=…, checkpoint_callback=…)` / `stage3.refine`'s matching parameters), measures each snapshot, and plots the resulting path through the corpus-fit PCA space toward the target. Also prints a per-step, per-block standardized-distance table (full feature space, not just the lossy 2D projection) so you can see which block is actually driving convergence.

```
python scripts/signature_pca_trajectory.py wn18rr_v4
python scripts/signature_pca_trajectory.py wn18rr_v4 --num-checkpoints 5 --rewire-budget 20000
```

## Maintenance / one-off

### `rerender_signatures.py`
Re-renders the `block_<x>.png` plots for every `block_<x>.json` already collected under `data/` (loads each via `from_serializable` and re-writes the plot). Use after changing `visualize()` or a plot helper, to refresh figures **without** re-running the (slow) measurements. `--blocks` restricts to specific blocks, `--fmt` picks the image format, `--dry-run` lists without writing.

```
python scripts/rerender_signatures.py                  # all blocks
python scripts/rerender_signatures.py --blocks b c d
python scripts/rerender_signatures.py --fmt pdf --dry-run
```

### `patch_block_b_degree_stats.py`
**One-off migration.** Backfills the Block B degree-stat fields (`out_degree_max`, `out_degree_p90`, `in_degree_max`, `in_degree_p90`) into every `block_b.json` / `signature.json` under the given roots, computing them from the `_out_degrees` / `_in_degrees` arrays already stored in each `block_b.json`. Superseded by any full corpus re-measurement (which emits these fields directly); retained only for patching a pre-existing corpus in place. `--dry-run` lists targets without writing.

```
python scripts/patch_block_b_degree_stats.py
python scripts/patch_block_b_degree_stats.py --roots data/graphs data/test_graphs
python scripts/patch_block_b_degree_stats.py --dry-run
```
