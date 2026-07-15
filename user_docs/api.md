# API reference

A single-stop reference for using `kgsynth` as a Python library and as a CLI. It
describes **how to call things** — signatures, parameters, return types, minimal
examples. For **why** each piece is built the way it is, follow the links out to
[signature.md](signature.md), [generator.md](generator.md), [dataset.md](dataset.md)
and [transform.md](transform.md), which this document does not duplicate.

```bash
pip install -e .          # from the repo root; installs the `kgsynth` console script too
```

```python
from kgsynth import Signature, Generator

target = Signature.from_file("some_real_graph.ttl")   # measure
synthetic = Generator(target).sample(seed=42)          # generate
```

Runnable, more fully-commented versions of this loop live in
[`examples/`](../examples/) (`measure_a_kg.py`, `generate_and_compare.py`).

---

## Package layout

| Import | Covers | Docs |
|---|---|---|
| `kgsynth` | Top-level re-exports: `Signature`, `Generator`, blocks, transforms, `load_kg`/`save_kg` | this page |
| `kgsynth.signature` | Blocks A–F measurement | [signature.md](signature.md), [§ Measuring](#measuring-kgsynthsignature) |
| `kgsynth.generator` | `Signature`, `Generator`, Stage 1/2/3 internals | [generator.md](generator.md), [§ Generating](#generating-kgsynthgenerator) |
| `kgsynth.transform` | Feature-dict perturbations | [transform.md](transform.md), [§ Transforms](#perturbing-kgsynthtransform) |
| `kgsynth.dataset` | Perturb-and-generate batch pipeline | [dataset.md](dataset.md), [§ Datasets](#batch-generation-kgsynthdataset) |
| `kgsynth.corpus` | Locate/load the cached `data/graphs/` corpus | [§ Corpus helpers](#corpus-helpers-kgsynthcorpus) |
| `kgsynth.kg_io` | Turtle/N-Triples ⇄ `igraph.Graph` | [§ Graph I/O](#graph-io-kgsynthkg_io) |
| `kgsynth.motif_counter` | Pluggable subgraph-counting backends | [§ Motif counters](#motif-counters-kgsynthmotif_counter) |
| `kgsynth.signature_sampler` | Draw a novel signature from the measured population | [§ Population sampler](#population-sampler-kgsynthsignature_sampler) |
| `kgsynth.cli` | The `kgsynth` console script | [§ CLI reference](#cli-reference) |

Everything under `kgsynth.*._private_module` (leading underscore) is internal;
import from the package or subpackage `__init__.py` instead — that is the
surface each `__all__` publishes and the surface this document covers.

---

## The graph object

`kgsynth.kg_io.load_kg` returns a directed `igraph.Graph` — vertices carry
`name` (URI or `_:blank` id), `is_literal`, `literal_value`/`literal_datatype`/
`literal_lang`; edges carry `predicate` (URI string). Every function below that
takes or returns "a graph" means this object. See [§ Graph I/O](#graph-io-kgsynthkg_io).

---

## Measuring (`kgsynth.signature`)

### `Signature` — the recommended entry point

`kgsynth.Signature` (re-exported from `kgsynth.generator.pipeline`) is the object
most code should use: it holds all six blocks together and is what `Generator`
consumes.

```python
from kgsynth import Signature

sig = Signature.from_file("graph.ttl")        # load_kg + measure all six blocks
sig = Signature.from_graph(g)                  # measure from an already-loaded igraph.Graph
sig = Signature.from_config("target.yaml")     # load a hand-edited/round-tripped YAML target
sig.to_config("target.yaml")                   # write it back out

feats = sig.as_features()                      # flat {name: value} dict (135 keys)
sig2 = Signature.from_features(feats)           # rebuild a generator-usable Signature from it
```

| Method | Signature | Returns |
|---|---|---|
| `Signature.from_graph(g, skip_stars_and_paths=False, skip_shortest_paths=False)` | classmethod | Measures all six blocks from an `igraph.Graph`. The two `skip_*` flags speed up Block E/F for sweep work. |
| `Signature.from_file(path)` | classmethod | `load_kg(path)` then `from_graph`. |
| `Signature.from_config(path)` | classmethod | Loads a YAML file with one top-level key per block letter (`a`..`f`), each holding that block's `to_serializable()` state. All six blocks required. Raises `KeyError` naming the missing block. |
| `sig.to_config(path)` | instance | Inverse of `from_config` — writes YAML readable by it. |
| `sig.as_features()` | instance | Flattens to the public feature dict (block order A→F). Requires every block present, including E. |
| `Signature.from_features(feats)` | classmethod | Inverse of `as_features()`; rebuilds fit objects from their named features. **Cannot** `visualize()` afterwards — raw plotting arrays aren't in the vector. Persist the feature dict, not a re-serialized block, for anything built this way. Raises `KeyError` on a missing key. |

Fields: `sig.a, sig.b, sig.c, sig.d, sig.e, sig.f` — the six block instances.
`a, b, c, d, f` are mandatory; `e` is `Optional[BlockE]` because
[`kgsynth.corpus.load_target_from_corpus`](#corpus-helpers-kgsynthcorpus) may
legitimately omit it for Stage-1/2-only callers.

### Measuring without a full `Signature` — `compute_reduced_signature`

For selective-block measurement (skip expensive blocks) or when you want the
un-wrapped `ReducedGraphSignature`, use `kgsynth.signature` directly:

```python
from kgsynth.signature import compute_reduced_signature, write_signature_outputs

sig = compute_reduced_signature("graph.ttl", blocks=["a", "c", "f"], verbose=True)
# sig.b, sig.d, sig.e are None; sig.as_vector() is still full-length (NaN-filled)

written = write_signature_outputs(sig, "data/graphs/mygraph/signature", source="graph.ttl")
```

| Function | Signature | Notes |
|---|---|---|
| `compute_reduced_signature(path, *, blocks=None, verbose=False)` | → `ReducedGraphSignature` | `blocks` defaults to all of `("a","b","c","d","e","f")`. Raises `ValueError` on an unknown block letter. |
| `write_signature_outputs(sig, out_dir, source, fmt="png", show=False, merge=True)` | → `list[Path]` | Writes `block_<x>.<fmt>` + `block_<x>.json` per computed block, plus a combined `summary.txt` and `signature.json`. With `merge=True` (default), blocks already on disk in `out_dir` are folded into the two aggregate files, so re-measuring a subset amends rather than truncates them. |
| `load_signature_dir(out_dir)` | → `ReducedGraphSignature` | Reconstructs from `block_<x>.json` files already on disk; absent blocks stay `None`. |

`ReducedGraphSignature` (the un-wrapped, `Optional`-block sibling of `Signature`
— `Signature` always requires Block E; this one doesn't) exposes `.a`..`.f`,
`.as_vector()` (NaN-filled for `None` blocks) and `.as_dict()`.

### Individual blocks

Every block (`BlockA`..`BlockF`, all importable from `kgsynth`) shares one
lifecycle (the class-pattern guide with the full rationale has since been pruned from this tree;
git history has it):

```python
from kgsynth import BlockB

b = BlockB().calculate(g)          # run computation, returns self
b.as_dict()                        # {feature_name: value}
b.as_vector()                      # same values, fixed-length list
b.visualize()                      # matplotlib figure
b.visualize(mode="text")           # CLI summary
b.visualize(mode="text", path="out.txt")   # write instead of display/print
b.to_serializable()                # JSON-safe dict of full internal state (needs calculate() first)
BlockB.from_serializable(data)     # reconstruct from that dict
BlockB.feature_names()             # classmethod — names in as_vector() order
BlockB.get_na_vec()                # classmethod — same-length all-NaN vector
```

Accessing any result attribute before `calculate()` raises `RuntimeError`.
Blocks B, C, D additionally expose `distribution_fits()` — `(name, fit, kind)`
triples used for Wasserstein-1 distance computations (see
`kgsynth.signature._distance`).

`kgsynth.QUANTILE_LEVELS` is the shared `(0, .1, .25, .5, .75, .9, 1)` tuple every
quantile-function feature is evaluated at.

---

## Generating (`kgsynth.generator`)

### `Generator`

```python
from kgsynth import Generator

gen = Generator(target)                 # target: a Signature with a, b, c, d, f (e optional)
graph = gen.sample(seed=42)             # full 3-stage pipeline -> igraph.Graph
```

`Generator.sample(...)` — keyword-only, all optional:

| Param | Default | Meaning |
|---|---|---|
| `seed` | `0` | Master seed. Stage 1 uses `seed`, Stage 2 `seed+1`, Stage 3 `seed+2` — the whole pipeline is reproducible from one integer. |
| `relation_zipf_exponent` | `2.0` | Fallback relation-frequency skew, used only when the target has no relations to fit `rel_freq_logq` from. |
| `rewire_budget` | `50_000` | Stage-3 rewiring attempts. |
| `initial_temp`, `cooling_rate` | `0.05`, `0.99993` | Simulated-annealing schedule, tuned for a ~100k budget; raise `cooling_rate` (e.g. ~0.998) for a much smaller one. |
| `skip_c5`, `skip_c6` | `False` | Force off 5-/6-cycle steering regardless of the target count. |
| `adaptive_weights` | `False` | Rescale each Stage-3 loss term by its own current error instead of a fixed weight. |
| `convergence_log` | `None` | Path — write a per-metric relative-error CSV during Stage 3. |
| `swap_log` | `None` | Path — write one CSV row per evaluated Stage-3 swap proposal. |
| `checkpoint_steps`, `checkpoint_callback` | `None`, `None` | Snapshot the walk's graph at given step indices (`0` = post-Stage-2, pre-rewiring); `checkpoint_callback(step, graph)` fires once per step. |

Returns an `igraph.Graph` with the same attribute schema as `kg_io.load_kg`
output, so it can be fed straight back into `Signature.from_graph` or `save_kg`.

`gen.sample_pre_refine(*, seed=0, relation_zipf_exponent=2.0)` runs Stages 1–2
only (bit-identical to what `sample()` hands to Stage 3 for the same seed) —
useful for inspecting the pre-refinement graph without paying for the annealing
loop.

For which target fields drive which stage, and the full stage-by-stage
algorithm, see [generator.md](generator.md). The lower-level pieces
(`sample_schema`, `instantiate`, `refine`, `Schema`) are also re-exported from
`kgsynth` for direct use, but `Generator.sample()` is the entry point almost
every caller wants.

---

## Perturbing (`kgsynth.transform`)

Transforms map a `Signature`'s flat feature dict to another one — for building a
synthetic-KG corpus or an OFAT sensitivity sweep:

```python
import numpy as np
from kgsynth import Signature, Generator, Perturb, FeatureSpec

feats = Signature.from_file("swdf.ttl").as_features()
feats, report = Perturb({"mean_degree": FeatureSpec(sigma=0.15)}).apply(
    feats, np.random.default_rng(0)
)
graph = Generator(Signature.from_features(feats)).sample(seed=1)
```

| Class | Role |
|---|---|
| `Perturb(specs)` | Joint design — every configured feature jittered at once, each by its own draw. `specs: dict[str, FeatureSpec]`. |
| `PerturbOne(feature, level, spec)` | OFAT design — exactly one feature moved by a fixed `level` (multiplier for multiplicative dists, offset for additive ones); everything else stays baseline. |
| `Identity()` | The null transform — unperturbed baseline. |
| `FeatureSpec(dist="lognormal", sigma=0.1, lo=None, hi=None, levels=(), clamp=None)` | Per-feature perturbation spec. `dist` is `"lognormal"` (multiplicative, default), `"normal"` (additive), `"uniform"` or `"loguniform"` (both need `lo`/`hi`). `levels` is only read by an OFAT sweep (`kgsynth.dataset`), ignored by `Perturb` itself. `clamp` overrides the feature's default domain. |

Every transform exposes `.apply(feats, rng) -> (dict[str, float], ClampReport)`
and `.describe() -> dict` (a JSON-safe summary). `ClampReport.clamped` maps
feature → `(requested, actual)` for every value a domain clamped back;
`.absorbed` maps a coupled group → fraction clamped; `.saturated()` lists groups
where that fraction exceeds 50%.

Coupled feature groups (the four quantile functions plus the six
`recip_symmetric_frac_bin*`) move together by one shared factor no matter which
member is named — `kgsynth.transform.group_of(name)` returns the full group.

`kgsynth.transform.validate(names)` checks a feature-name iterable against the
generator's perturbation surface: raises `ValueError` on a name that isn't a
signature feature, or one the generator never reads at all; returns warning
strings (not exceptions) for names that are on the surface but perturb
degenerately (`INERT`, `CONSTANT`, `SIZE_FEATURES` — see
[transform.md](transform.md) for what each category means and why).
`kgsynth.transform.SURFACE` is the frozenset of perturbable feature names.

`TRANSFORMS = {"joint": Perturb, "ofat": PerturbOne}` is the registry
`kgsynth.dataset`'s YAML `design:` key resolves against.

---

## Batch generation (`kgsynth.dataset`)

The programmatic form of `kgsynth dataset` — generate many synthetic KGs from
one perturbed baseline, one process per graph. Usually driven by a YAML config
(see [dataset.md](dataset.md) for the full schema and
[`examples/perturb_dataset.yaml`](../examples/perturb_dataset.yaml) for a
commented one), but every step is callable directly:

```python
from kgsynth.dataset import DatasetConfig, build_units, run, describe

config = DatasetConfig.from_yaml("run.yaml")   # parses + validates everything up front
print(describe(config))                        # the --dry-run table, as a string
failed = run(config, workers=8, force=False)    # returns the failed-unit count
```

| Function / class | Signature | Notes |
|---|---|---|
| `DatasetConfig.from_yaml(path)` | classmethod → `DatasetConfig` | Raises `ValueError` naming the offending key on anything invalid — a typo must not surface 40 graphs in. Requires the `base` graph to have a cached `block_e.json` (measuring Block E per worker is slow and not reproducible across processes). |
| `build_units(config)` | → `list[WorkUnit]` | Pure, no I/O — the `joint`/`ofat` design maps to a concrete unit list. |
| `run(config, *, workers=None, force=False)` | → `int` | Runs the plan across a `ProcessPoolExecutor`, one graph per worker process. Skips units whose `meta.json` already exists unless `force=True`. Writes `manifest.jsonl` incrementally. Returns the number of failed units. |
| `describe(config)` | → `str` | The plan table `--dry-run` prints, without generating anything. |
| `load_manifest(out_dir)` | → `list[dict]` | Reads a (possibly in-progress) run's `manifest.jsonl`. |
| `run_unit(unit)` | → `UnitResult` | Generates one `WorkUnit`; never raises — failures come back as `UnitResult(ok=False, error=...)`. What `run()` submits to each worker process. |

`DatasetConfig` fields: `base` (corpus graph name), `design` (`"joint"`/`"ofat"`),
`specs` (`dict[str, FeatureSpec]`), `out_dir`, `seed`, `num_graphs` (joint only),
`measure` (also re-measure + record distances), `generator_opts` (forwarded
verbatim to `Generator.sample`).

Output layout, resumability semantics, and the two things that mislead a naive
reading of the results (clamped perturbations, Block E estimator variance) are
covered in [dataset.md](dataset.md) — not repeated here.

---

## Corpus helpers (`kgsynth.corpus`)

Locate and load the cached measured-KG corpus (`data/graphs/`, `data/test_graphs/`):

```python
from kgsynth.corpus import load_target_from_corpus, corpus_graph_names

names = corpus_graph_names()                          # every cached graph name
sig, blocks, graph_dir = load_target_from_corpus("swdf")
```

| Function | Signature | Notes |
|---|---|---|
| `load_target_from_corpus(graph_name, search_dirs=None, with_block_e=True)` | → `(Signature, blocks_dict, graph_dir)` | Loads cached A/B/C/D/F; loads Block E from `block_e.json` if present, else measures it from the graph file. `with_block_e=False` skips it entirely (`sig.e is None`) — for Stage-1/2-only callers, since it's the expensive block. Raises `SystemExit` (not an exception a caller is expected to catch) if the graph or a required cached block is missing. |
| `corpus_graph_names(search_dirs=None)` | → `list[str]` | Every graph name across the searched corpora, sorted, de-duplicated. |
| `graph_dir(name, search_dirs=None)` | → `Path \| None` | The corpus directory for one graph name. |
| `iter_corpus_graphs(names=None, search_dirs=None)` | → `list[Path]` | One source graph file per corpus directory, smallest first. |
| `find_graph_file(d)` | → `Path \| None` | The first non-synthetic `.nt`/`.ttl`(`.gz`) file directly in directory `d`. |
| `load_block(cls, path)` | → block instance | `cls.from_serializable(json.loads(path.read_text()))`. |

`DEFAULT_SEARCH_DIRS = [REPO_ROOT/"data"/"graphs", REPO_ROOT/"data"/"test_graphs"]`;
`REPO_ROOT` is the repo root resolved from this module's file location (valid for
an editable install).

---

## Graph I/O (`kgsynth.kg_io`)

```python
from kgsynth import load_kg, save_kg   # top-level re-export

g = load_kg("graph.ttl")               # auto-detects Turtle vs N-Triples from content
save_kg(g, "out.ttl", fmt="turtle")    # fmt: "turtle"/"ttl" or "nt"/"n-triples"/"ntriples"
```

`load_kg` deduplicates edges by `(subject, predicate, object)` and inserts
triples in **sorted** order, so vertex/edge indices are a deterministic function
of file content — required for reproducibility of every seeded sampler
downstream (Block E colour-coding, Block F path sampling, Stage 1–3). Format is
detected from content, not the file extension, so extensionless dumps load
fine; invalid content raises `ValueError`.

`save_kg` raises `ValueError` on an unsupported `fmt`.

---

## Motif counters (`kgsynth.motif_counter`)

Pluggable subgraph-counting backends shared by Block E measurement and Stage 3
rewiring:

```python
from kgsynth.motif_counter import HybridMotifCounter

counts = HybridMotifCounter().count(g)   # exact for k<=3, colour-coding for k>=4
```

| Class | Strategy |
|---|---|
| `ExactMotifCounter` | Exact enumeration for `k <= 6` (ESCAPE algorithm for k=5/6). Slow on hub-heavy graphs. |
| `CCMotifCounter` | Colour-coding sampling estimator (Bressan et al. 2021) — approximate, fast, has variance. |
| `HybridMotifCounter` | Exact for `k <= 3` (triangles), colour-coding for `k >= 4`. What Block E and Stage 3 use by default. |

All three implement the `MotifCounter` abstract interface. Function-level
primitives (`cc_run`, `cc_run_stars`, `count_motifs5_escape`,
`count_motifsk_escape`) are also exported for direct use; see the module
docstring (`src/kgsynth/motif_counter/__init__.py`) for the full list. Per-swap
incremental delta helpers used by Stage 3's simulated annealing live separately
in `kgsynth.generator.local_updates` (not part of this subpackage).
[`notes/counter_benchmark.md`](../developer_docs/notes/counter_benchmark.md) has the exact-vs-CC
accuracy/speed comparison.

---

## Population sampler (`kgsynth.signature_sampler`)

Draws a **novel** reduced signature (not a perturbation of one specific graph)
from the distribution of the measured corpus — see
[plan/stage1_population_sampler.md](../developer_docs/plan/stage1_population_sampler.md) for the
design and its current limitations.

```python
from kgsynth.signature_sampler import UniformRangeSampler

sampler = UniformRangeSampler.load_corpus()     # data/graphs/*/signature/signature.json
feats = sampler.sample(seed=0)                  # 107-key dict: Blocks A/B/C/D/F (no Block E)
sampler.write("sampled.json", feats)
```

`SignatureSampler` is the reusable ABC (`load_corpus`, `sample`, `to_json`,
`write`); `UniformRangeSampler` is the only concrete implementation currently —
each feature drawn independently from `Uniform` over its observed corpus range,
widened ±10%. Output excludes Block E (motifs are raw, size-dependent counts,
out of scope for this uniform baseline); add the missing Block E keys yourself
before calling `Signature.from_features()` on the result.

---

## CLI reference

Installed as the `kgsynth` console script (`pip install -e .`); every
subcommand is a thin wrapper over the library calls above.

```bash
kgsynth measure graph.ttl [--output-dir DIR] [--blocks a,b,c,...] [--format png|pdf|svg] [--show]
kgsynth generate <graph-name> [--output PATH] [--seed N] [--rewire-budget N] [--graphs-dir DIR]
kgsynth generate --config target.yaml [--output PATH] [--seed N] [--rewire-budget N]
kgsynth compare left.ttl right.ttl
kgsynth dataset run.yaml [--workers N] [--measure] [--force] [--dry-run] [--out-dir DIR]
```

| Command | Library equivalent |
|---|---|
| `measure` | `compute_reduced_signature(...)` + `write_signature_outputs(...)` |
| `generate` | `Signature.from_config(...)` or `corpus.load_target_from_corpus(...)`, then `Generator(target).sample(...)` |
| `compare` | `Signature.from_graph(load_kg(...))` on both files, printed feature-by-feature per block |
| `dataset` | `DatasetConfig.from_yaml(...)` + `dataset.run(...)` / `dataset.describe(...)` — see [dataset.md](dataset.md) |

`-v`/`--verbose` (top-level flag, before the subcommand) logs progress to
stderr; `dataset` always logs regardless, since an unmonitored multi-hour run
would otherwise be silent. Every subcommand's `--help` documents its flags.
