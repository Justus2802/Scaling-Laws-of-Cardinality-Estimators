# Perturb-and-generate datasets

Generate *many* synthetic KGs by perturbing one measured signature. One process per graph.

```bash
kgsynth dataset examples/perturb_dataset.yaml --dry-run     # review the plan, generate nothing
kgsynth dataset examples/perturb_dataset.yaml --workers 8
kgsynth dataset examples/perturb_dataset.yaml --measure     # + re-measure each graph
```

Per graph: load the baseline signature from the corpus → perturb its feature dict
([transform.md](transform.md)) → rebuild a target (`Signature.from_features`) → generate → write.

---

## Two designs

**`joint`** — every configured feature is jittered at once, each by its own draw. A diverse cloud of
signatures around the baseline. This is the corpus builder: use it when you want *N* varied synthetic
KGs.

**`ofat`** — *one factor at a time*. One baseline graph, then one graph per `(knob, level)`: exactly
one feature moves, everything else stays at baseline. Because only one thing changed, any difference
in the generated graph is **attributable to that feature**. This is the sensitivity analysis — it
tells you which of the signature's features actually do anything, which is the evidence the project's
standing guardrail (*propose removing a feature with no proven downstream effect*) has so far been
missing.

OFAT enumerates **knobs, not features**: the seven `obj_mult_alpha_q*` keys are one knob (they must
move together to stay a valid quantile function). So a full sweep is `1 + Σ|levels|` graphs, not
`4 × 79`.

---

## Config

See [`examples/perturb_dataset.yaml`](../examples/perturb_dataset.yaml) for a commented one.

```yaml
base: fb237_v4              # corpus graph whose signature is the baseline
design: joint               # joint | ofat
num_graphs: 8               # joint only
seed: 20260712              # master seed; every graph's sub-seeds derive from it
measure: false              # true -> also re-measure each graph, record distances
out_dir: data/synthetic/fb237_v4_joint_v1

generator:                  # forwarded verbatim to Generator.sample()
  rewire_budget: 20000

features:
  mean_degree:
    dist: lognormal         # multiplicative: v * exp(N(0, sigma))
    sigma: 0.15
    levels: [0.8, 1.2]      # ofat only: multipliers (or offsets, for additive dists)
  degree_assortativity:
    dist: normal            # additive: v + N(0, sigma)
    sigma: 0.05
    clamp: [-1.0, 1.0]      # optional override of the feature's default domain
```

`dist` is `lognormal` (the default — multiplicative, so `V × 1.1`, not `V + 1.1`), `normal`,
`uniform` or `loguniform` (the last two need `lo` + `hi`).

**Everything is validated at parse time**, before a single graph is generated: unknown feature names,
off-surface features, missing `levels` under `ofat`, a `base` with no cached Block E. A run is tens of
minutes *per graph* — a typo must not surface on graph 40.

---

## Output

```
data/synthetic/fb237_v4_joint_v1/
  manifest.jsonl               # one line per completed unit, appended as it lands
  graph_0000/
    graph.ttl
    target.json                # {source, features} -- the perturbed signature
    meta.json                  # seeds, transform, clamp report, timings, corpus SHA
    achieved.json              # --measure only: the generated graph's own signature
    distance.json              # --measure only: per-block target-vs-achieved distances
```

`target.json` is the **flat feature dict**, not a `block_*.json` tree. A signature rebuilt from
features has no plot arrays, so `to_serializable()` would emit a file that *looks* measured but
silently is not. The flat dict is exactly what `from_features` consumes, so it round-trips cleanly.

The run is **resumable**: a unit is skipped when its `meta.json` exists. Since that file is written
last, a directory killed mid-generation is correctly treated as unfinished and redone. `--force`
regenerates everything.

Seeds come from `numpy.random.SeedSequence(seed).spawn(n)`, so a unit's result depends **only on its
index** — never on worker count or completion order. Two runs with the same config produce the same
dataset, whether on 1 worker or 16.

---

## Runtime, and what it means for scale

Generation is the cost, and it is substantial: on `fb237_v4` (V=4707, E≈34k) a single graph takes
**tens of minutes**, dominated by Stage 3. Four graphs on four workers is roughly an hour of wall
clock. Plan accordingly:

- Parallelism is per-graph, so wall time ≈ `ceil(units / workers) × per-graph time`.
- `--dry-run` prints the unit table first. Use it before committing to a 169-unit OFAT sweep.
- Lower `rewire_budget` while iterating on a config; raise it for the real run.
- Each worker is single-threaded Python, so the runner pins `OMP_NUM_THREADS=1` etc. before forking —
  otherwise every worker's BLAS spawns its own threads and oversubscribes the cores.

A failed unit does **not** take the pool down: it is recorded in the manifest with its error, the run
continues, and the process exits non-zero.

---

## Two things that will bite a naive reading of the results

**Clamped perturbations.** A perturbation pushed outside its feature's domain is clamped back — so the
config says "perturbed" while the generator sees (nearly) the baseline value, and OFAT reads it as
*"this knob has no effect"*. Every unit's `meta.json` carries a `clamp_report`, and the runner warns
when a knob's jitter was mostly absorbed. `swdf`'s `obj_mult_alpha_q*` already sits at its `[1.4,
3.0]` ceiling, so scaling that group **up** is 57 % absorbed. See [transform.md](transform.md).

**Block E is estimated, not exact.** `HybridMotifCounter` uses colour-coding for k ≥ 4, so a Block E
distance in `distance.json` is part generator error and part *estimator variance*. `distance.json`
carries `block_e_estimated: true` for exactly this reason. An OFAT effect on motif counts smaller than
the estimator's own spread is unreadable, not absent — characterise that spread with
`scripts/cc_variance.py` before drawing conclusions from a small Block E difference.

Related: `meta.json` records `corpus_regenerated: false`. The tracked corpus has not been re-measured
since the pinned-`xmin` fit change, so datasets built before and after that regeneration are **not
comparable**. The flag is there so two of them can never be silently mixed.

---

## Adding a design

`build_units` maps a config to a list of `WorkUnit`s — pure, no IO, so a 169-unit plan is testable in
milliseconds. A new design is a new branch there plus a transform in
`kgsynth.transform.TRANSFORMS`. The runner, the worker and the output layout do not change; this is
how `ScaleTo` (resize the graph, holding shape) will arrive.
