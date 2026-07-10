# BPC-Data-Engeneering: Scaling Laws of Cardinality Estimators

TUM *Beyond Practical Course* (BPC) Data Engineering, SS 2026 — Topic 2.
Supervisor: Tim Schwabe.

## The research question

Learned cardinality estimators for knowledge graphs (KGs) — models that predict how many
results a query pattern will match without executing it — can generalize to KGs they were
never trained on. What's unclear is **how much training data that generalization actually
needs**: does accuracy keep improving with more triples/graphs and more model parameters,
or does it plateau? This project studies that data/model scaling trade-off empirically,
aiming to reproduce a "generalization error vs. training-data size" scaling curve for a
learned estimator.

Studying this properly needs many KGs that are realistic (statistically close to real-world
graphs) but also **numerous and diverse** (real labeled KG corpora are small and hard to get
in bulk, and not all of the graphs are available as open-source). That's the gap this
repository fills: a **synthetic KG generator** that takes a measurement of a real graph and
produces new graphs that are statistically close to it — so a scaling study can have as
much training data as it wants, on demand.

## The four project tasks

| Task | Status | What it is |
|---|---|---|
| **T1** | ✅ done | Build a synthetic KG generator that reproduces a real KG's statistical fingerprint |
| **T2** | in progress | Make generated graphs match real-world KG distributions well (degree distribution, predicate frequency, motif structure, …) |
| **T3** | todo | Train cardinality-estimator models of different sizes (10K–1M params) on varying amounts of generated data |
| **T4** | todo | Analyze the resulting generalization-error-vs-data-size curves — the actual scaling law |

T1 and T2 (everything in this repo so far) exist to *feed* T3/T4 with training data whose
statistical properties are known and controllable — that's the whole point of generating
graphs instead of only using the handful of real ones available.

## The core idea: measure → generate → compare

You can't directly ask "is this synthetic graph realistic?" without first deciding what
"realistic" means numerically. So the project is built around one central object, the
**signature**:

```
   real KG  --[measure]-->  signature (117 numbers)  --[generate]-->  synthetic KG
                                    ^                                      |
                                    |                                      |
                                    +---------------[re-measure]-----------+
                                          (compare: did we hit the target?)
```

1. **Measure** — reduce a real KG to a compact, non-redundant vector of statistics (its
   *signature*): how many entities/relations, how skewed relation usage is, per-relation
   fan-out/fan-in shape, schema/type co-occurrence structure, characteristic-set (predicate
   bundle) reuse, motif counts (triangles, 4-/5-/6-node subgraphs), and global connectivity
   (components, clustering, assortativity, shortest paths).
2. **Generate** — sample a brand-new graph that *targets* that signature, without copying
   the original graph's actual entities or edges.
3. **Compare** — re-measure the synthetic graph's signature and check how close it landed
   to the target, feature by feature. This round-trip (`scripts/signature_roundtrip.py`) is
   the main way progress is validated.

Repeating step 2 with different random seeds turns one real graph into an arbitrarily large
*family* of statistically-similar synthetic graphs — the training-data supply T3/T4 need.

### Why "signature" and not "just copy the stats"?

The tricky part is that KG statistics are highly **inter-dependent** — e.g. a node's
out-degree is just the sum of how many objects it reaches through each relation it uses, so
storing *both* "degree distribution" and "per-relation fan-out distribution" as independent
targets would be storing the same information twice (over-determined) in some places, while
in others two statistics that *look* related are actually not exactly derivable from each
other and both need to be kept. Working out which of ~130 candidate KG statistics are
algebraically/statistically redundant, and which must be kept as independent targets, is
the subject of **[docs/signature.md](docs/signature.md)** — the design document for the
reduced, 117-feature signature actually implemented in `src/kgsynth/signature/`. Start there for the
full reasoning (it's substantial: the *why* behind every kept and dropped feature).

## The three-stage generator

Given a target signature, `src/kgsynth/generator/` builds a graph in three stages, each responsible
for a different layer of structure:

1. **Stage 1 — schema sampler** (`stage1.py`): decides the *abstract* shape — how many
   entities/relations/types, how skewed relation usage is (Zipf), the type→relation
   co-occurrence structure, and per-relation multiplicity (fan-out/fan-in) shapes. No actual
   graph yet, just the "recipe."
2. **Stage 2 — CS-first instantiation** (`stage2.py`): wires up actual entities and edges.
   "CS" = *characteristic set*, the bundle of predicates one entity uses — Stage 2 assigns
   each entity a characteristic set first (matching the target's schema reuse patterns),
   then fills in concrete edges to hit the target degree distribution, connectivity, and
   path-length structure.
3. **Stage 3 — Maslov–Sneppen refinement** (`stage3.py`): the graph from Stage 2 already has
   the right degree sequence and schema, but its finer-grained structure (triangle count,
   4-/5-/6-node motifs, clustering, assortativity) is whatever fell out incidentally. Stage 3
   nudges it closer to the target via **degree-preserving double-edge swaps** — pick two
   same-relation edges, swap their endpoints — accepted or rejected via **simulated
   annealing** against a loss function that measures how far every motif/connectivity target
   still is. This never touches what Stage 1/2 already fixed (relation labels, degree
   sequence); it only rewires *which* nodes each relation connects.

```python
from kgsynth import Signature, Generator

target = Signature.from_file("some_real_graph.ttl")   # measure
synthetic = Generator(target).sample(seed=42, rewire_budget=100_000)  # generate
```

Full algorithm walkthrough, which signature field drives which step, and the (still
evolving) known gaps: **[docs/generator.md](docs/generator.md)**.

### An active tuning knob: adaptive Stage-3 loss weights

Stage 3's loss function is a weighted sum of per-metric relative errors (one term per
motif/connectivity target). By default each term's weight is a fixed constant, so a metric
that's already converged pulls just as hard as one still far off. `stage3.refine(...,
adaptive_weights=True)` instead rescales each term's weight by its *own* current error
(`weight = base_weight * ADAPTIVE_WEIGHT_SCALE * error`), so the annealer automatically
concentrates rewiring pressure on whichever metric is currently worst. `--adaptive-weights`
on `scripts/signature_roundtrip.py` turns this on; `scripts/sweep_adaptive_weight_scale.py`
searches for the best `ADAPTIVE_WEIGHT_SCALE`; `scripts/convergence_plot_grid.py` plots
fixed-vs-adaptive convergence curves side by side. See the "Adaptive weights" section of
`docs/generator.md` for the current tuning findings and trade-offs (it doesn't uniformly
win — it trades some metrics for others).

## Repository layout

```
src/kgsynth/          the installable package (pip install -e .)
  signature/          Block A–F measurement code (the "measure" step)
  generator/          Stage 1/2/3 synthetic-graph generation (the "generate" step)
  motif_counter/      Exact / color-coding / hybrid subgraph-counting backends
  kg_io.py            Load/save KGs (.ttl, .nt, …)
  corpus.py           Locate + load cached signatures from data/graphs/
scripts/
  measure_signature_reduced.py   measure a real graph -> data/<name>/signature/
  signature_roundtrip.py         measure -> generate -> re-measure -> compare
  sweep_*.py, *_plot*.py         parameter sweeps and diagnostic plots
data/
  graphs/, test_graphs/          the real-KG corpus (measured + cached signatures)
docs/
  signature.md        the signature design & reasoning (start here)
  generator.md         the generator algorithm, step by step
  notes/               deep-dive analyses (why X is hard, what was tried, corpus surveys)
  plan/                forward-looking / not-yet-implemented plans
tests/                pytest suite for signature blocks and generator stages
```

See **[docs/README.md](docs/README.md)** for the full documentation map, including the
per-topic investigation notes (edge multiplicity, relation reciprocity, Stage-3 steering
limits) that explain *why* certain design choices were made and where the current
approach still falls short of the real-graph targets.

## Getting started

```bash
pip install -e .
```

That installs the `kgsynth` package and its CLI:

```bash
# Measure a real graph into a signature/ dir next to it
kgsynth measure data/graphs/swdf/swdf.nt

# Generate a synthetic graph from a cached target signature
kgsynth generate swdf --seed 42 --rewire-budget 50000 --output swdf_synth.ttl

# Compare two graphs feature by feature across the full 124-value signature
kgsynth compare data/graphs/swdf/swdf.nt swdf_synth.ttl
```

From Python, per the project proposal:

```python
from kgsynth import Signature, Generator

target = Signature.from_file("some_real_graph.ttl")                    # measure
synthetic = Generator(target).sample(seed=42, rewire_budget=100_000)   # generate
```

The full measure → generate → re-measure → compare round-trip, with Stage-3 convergence
logging and per-block Wasserstein distances, lives in
`python scripts/signature_roundtrip.py <graph_name> --rewire-budget 5000`.

Run the test suite with `pytest`.
