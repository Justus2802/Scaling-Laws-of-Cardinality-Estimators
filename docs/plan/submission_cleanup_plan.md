# Submission cleanup: ship `kgsynth`

## Context

The project enters its submission phase. The science is sound — 271 passing tests, six
signature blocks cross-validated against library oracles, a working three-stage generator.
What is missing is the **packaging**: `scaling_laws_student_project.pdf` §3.4/§5 promises a
*pip-installable `kgsynth` package* with `from kgsynth import Signature, Generator`, a CLI
entry point, examples and a README. None of those four exist. Instead `src/` is a loose
directory reachable only via `sys.path.insert(...)` repeated across **33 files**, with no
`pyproject.toml` and a one-line README.

A fresh checkout is also **broken**: `matplotlib` is imported at module scope by five block
modules but is absent from `requirements.txt`, so `pip install -r requirements.txt` followed
by `import signature` raises `ModuleNotFoundError`.

Scope decisions taken with the user:
- **Phase 1 only.** The scaling-law study (query generation, QLever labeling, FICE training
  grid, `Qerror(N)` fitting) is out of scope; documented as future work.
- **Full repackage** to `kgsynth`, dropping all `sys.path` hacks.
- **Remove** the disabled Stage-2 path-length steering rather than reviving it.
- **No** `sample(num_triples=…)`, **no** per-feature standard errors in the signature, **no**
  derived Block-A features in the export. Each is a deliberate deviation, to be documented
  both inline and in a summary table — not silently omitted.
- PCA manifold work **exists but is unpushed**; integrate rather than rebuild.
- **Standing guardrail (user instruction, added mid-project):** if a signature feature turns out
  to be unnecessary — unsteered, high-error, or with no proven downstream effect — propose
  removing it rather than carrying it forward for its own sake. 2.11 and 2.12 below are the
  first two instances found this way; more may surface while working Stage 2/3.

Intended outcome: a repo a grader can `pip install -e .`, run `pytest` on green, drive from a
CLI, and read a README whose claims all hold.

---

## ⚠️ Working-tree state — read first

**Stage 0 was partially executed before plan mode engaged.** The tree is *not* clean:

```
src/generator/schema.py        |   3 -
src/generator/stage1.py        |  11 ----
src/generator/stage2.py        | 134 +----------------------------
tests/test_generator_stage1.py |  26 ------
tests/test_generator_stage2.py |  31 ------
```

Path-length steering is already removed: `PATH_STEERING_ENABLED`, `_steer_path_lengths` and its
call site, the `Schema.path_mean_target` / `path_hi_target` fields, the Stage-1 code that
populated them, and the two test classes. No references remain in `src/` or `tests/`.

**The suite has not been re-run since.** Step 0.1 is mandatory before anything else.

Also: the PCA/UMAP manifold code is **not in this checkout** — not in the working tree, not on
any of the seven local branches, not in `git log --all`. It must be brought in from wherever it
lives (see 3.2).

---

## Stage 0 — Unbreak the checkout (½ day)

0.1 **Re-run the suite first**: `.venv/bin/python -m pytest -q` (~4 min). Prior baseline was
`266 passed, 1 failed`; the failure was `test_steer_hi_caps_diameter`, now deleted, so expect
green. Verify rather than assume.

0.2 Finish the steering removal — orphaned by the deletions above:
- [tests/test_generator_stage1.py:1](tests/test_generator_stage1.py#L1) — `import math`, now unused.
- [tests/test_generator_stage1.py:268](tests/test_generator_stage1.py#L268) — `_make_block_f` is
  now dead (only its own `def` references it). Remove it, and drop `BlockF` from the line-9 import.
- `math` in `stage2.py` is still used (line 249) — keep.

0.3 Add `matplotlib>=3.7` to [requirements.txt](requirements.txt) — the actual fresh-checkout
breakage. Add `PyYAML` (for `Signature.from_config`, 3.1).

0.4 Clean the repo root: five `conv_log_*.csv`, `modified_annealing.csv` (517 KB),
`test_star_steering.csv`, `check_after_adaptive_sampling.csv`, any leftover `main.{aux,out,pdf}`
LaTeX artifacts, the empty `signature/` directory, tracked 1 MB `graph_comparison.png`, and
`data/test_graphs/wn18rr_v4/signature/block_b copy.png`.
(The poster `main.tex` has been removed by the user — nothing to preserve here.)

---

## Stage 1 — The `kgsynth` package (1–2 days)

The headline deliverable. **Moves and import rewrites only, no logic changes.** The test suite
is the safety net: green before, green after.

1.1 Create `pyproject.toml` (setuptools, `src/` layout, `[project.scripts]`), migrating the
pins from `requirements.txt`.

1.2 `git mv` everything under one package root:

```
src/kgsynth/
  __init__.py          # re-export Signature, Generator, BlockA…BlockF, load_kg, save_kg
  cli.py               # new
  kg_io.py             # was src/kg_io.py
  signature_sampler.py # was src/signature_sampler.py
  signature/  generator/  motif_counter/
```

1.3 Rewrite cross-package absolute imports as relative: `from signature import BlockA` in
[generator/pipeline.py](src/generator/pipeline.py) → `from ..signature import BlockA`; likewise
the deferred `from kg_io import load_kg` inside `Signature.from_file`. Intra-package relative
imports already correct.

1.4 **Delete every `sys.path.insert(...)`** — 33 files across `tests/` and `scripts/`. Replace
with `pip install -e .` plus a `tests/conftest.py`. Rewrite imports to `from kgsynth.signature
import …`. Note [scripts/signature_roundtrip.py](scripts/signature_roundtrip.py) is imported
*as a module* by a sibling script (`from signature_roundtrip import _DEFAULT_SEARCH_DIRS`) —
that cross-script import needs its own handling, likely by promoting the shared helpers into
the package.

1.5 `kgsynth/__init__.py` exporting the public API, so the proposal's
`from kgsynth import Signature, Generator` works verbatim.

1.6 `kgsynth/cli.py` with `measure` / `generate` / `compare` subcommands wired to
`[project.scripts] kgsynth = "kgsynth.cli:main"`. **Reuse** the existing argparse bodies from
`scripts/measure_signature_reduced.py` and `scripts/signature_roundtrip.py`; don't write new ones.

**Do not start Stage 2 until Stage 1 is green.** Everything downstream rebases across this move.

---

## Stage 2 — Deduplicate and reconcile code (1 day)

Each item is an internal contradiction a grader could find.

2.1 [scripts/measure_signature.py](scripts/measure_signature.py) and
[scripts/measure_signature_reduced.py](scripts/measure_signature_reduced.py) are now
functionally identical — both call `compute_reduced_signature`. Delete one; fold the survivor
into `kgsynth measure`.

2.2 [scripts/measure_all_raw.py:73](scripts/measure_all_raw.py#L73) — the `--full` flag
advertises "the original full signature," which no longer exists, and silently emits reduced
output. **Remove it**; it lies.

2.3 `HybridMotifCounter` is documented three incompatible ways: its module docstring
([hybrid_motif_counter.py:1](src/motif_counter/hybrid_motif_counter.py#L1)) says "exact for
k≤3, ESCAPE for k=5, CC for k=4 and k≥6"; the class docstring differs; and
[motif_counter/__init__.py](src/motif_counter/__init__.py) says "exact for k ≤ 5 (ESCAPE), CC
for k ≥ 6". Read the dispatch, make all three match it.

2.4 [block_e.py:9](src/signature/block_e.py#L9) says star counts are "no longer measured", but
`data/graphs/*/signature/block_e.json` contains `_star_counts`. Resolve the contradiction. Also
document *why* the unused `count_stars` helper is retained (it is exercised by tests and
`scripts/cc_variance.py`).

2.5 Signature JSON leaks private field names (`_triangle_count`, …). Strip the leading
underscore on export in `signature/_serialize.py`. **Caution:** `signature_sampler.py` reads
these keys and the tracked corpus JSON uses them — update both, and regenerate the corpus (4.3).

2.6 Dangling reference: `docs/signature_redesign.md` does not exist but is cited by
[signature/__init__.py:8](src/signature/__init__.py#L8),
[block_b.py:41](src/signature/block_b.py#L41), [_fits.py:13](src/signature/_fits.py#L13).
Repoint to `docs/signature.md`.

2.7 [signature/__init__.py:3](src/signature/__init__.py#L3) calls itself "a coexisting
alternative to the `signature` package". It *is* that package — leftover from the rename.

2.8 Collapse the three near-duplicate `_logging.py` modules into one `kgsynth/_logging.py`.

2.9 Move the 7 `print()` calls in `src/` to the logger. The 199 in `scripts/` are fine.

2.10 **Delete the stale-signature back-compat fallbacks.** *(User instruction.)* Error handling
that exists only to keep *old* signature JSONs loadable inflates the code and hides real
breakage. There is exactly one signature format — the one this repo emits. If a value the
generator needs is missing, **raise**; do not silently substitute a "legacy" default. A loud
`KeyError`/`AttributeError` is the correct behaviour and is strictly more informative than a
synthetic graph quietly built from neutral defaults.

Remove, at minimum:
- [stage1.py:264](src/generator/stage1.py#L264) — `getattr(c, "type_relation_conditional", None) or {}`,
  guarding against Block C objects that predate the field. Read `c.type_relation_conditional`.
- [stage1.py:386-390](src/generator/stage1.py#L386) — the `_ratio()` helper's
  `try/except → 1.0` for Block C multiplicity, whose comment literally says
  *"NaN / not-measured (stale signatures) → 1.0, the neutral legacy near-simple graph"*.
- [schema.py](src/generator/schema.py) — the `Schema` fields whose defaults are documented as
  reproducing "legacy behaviour" rather than as genuine optionality:
  `relation_reciprocity=None` ("→ all-asymmetric (legacy behaviour)"),
  `target_out_degrees=None` ("→ legacy PA fallback, no degree steering"),
  `mean_functionality=1.0` ("out-side fallback only"), and the
  `cs_size_mean=0.0` / `cs_num_templates=0` sentinels.
- The 6 `getattr(obj, name, default)` calls across `src/generator/` and `src/signature/`.

**Distinguish two cases before deleting** — they look identical and are not:
- *Back-compat fallback* → **remove**, let it raise. The field always exists in a
  current-format signature.
- *Genuine optionality* → **keep**. `Signature` legitimately treats `b, d, f` as optional
  blocks (`docs/generator.md`), and `_adapters.py`'s NaN-fit paths encode "the fit did not
  converge on this graph", which is a real measurement outcome, not a stale file. Likewise a
  Block C with too few classes to fit a power-law
  ([stage1.py:253](src/generator/stage1.py#L253)) is a real graph, not an old signature.

Do this **after** 2.5 (underscore-stripped JSON keys) and the corpus regeneration, so the
tracked corpus is already in the one true format when the fallbacks disappear. Then a stale
file fails loudly instead of being silently mis-generated. Regenerating the corpus is what
makes this safe: it guarantees no in-repo signature relies on a removed fallback.

2.11 **Delete Block E's path/tree template *steering* — dead code — but keep the features.**
*(User instruction, revised.)* The path/tree template features
(`path_template_zipf_k{2..10}`, `path_template_entropy_k{2..10}`, `tree_template_zipf`,
`tree_template_entropy` — 20 of Block E's 27) **stay in the signature** as measured/emergent
statistics of the generated graph; only the Stage 3 machinery that pretends to steer them comes
out. Stage 3 has a full incremental-delta steering path for them
([stage3.py:456-516](src/generator/stage3.py#L456),
`_tree_entropy_delta`/`_path_entropy_delta` in
[local_updates.py:594-654](src/generator/local_updates.py#L594)), but it is permanently inert:
`LOSS_WEIGHT_TREE_ENTROPY = 0` and `LOSS_WEIGHT_PATH_ENTROPY = 0`
([stage3.py:112-113](src/generator/stage3.py#L112)) gate `use_tree_entropy`/`use_path_entropy`
to always-False, so every branch guarded by them — `_pair_freq`/`_path_freqs` construction
(456-516), the `_error_terms`/`_base_weights` entries (550-572), the incremental-delta calls in
the swap loop (968-989, 1051-1066) — never executes; `tree_h`/`path_h` sit at `0.0` for the
whole run. This is the same dead-code shape as the already-removed path-length steering (Stage
0): a mechanism nobody enabled, not a feature being used lightly. Remove:
- Stage 3: `use_tree_entropy`/`use_path_entropy` and everything they gate — the setup blocks
  ([stage3.py:456-516](src/generator/stage3.py#L456)), the two `_error_terms`/`_base_weights`
  entries (550-572), the `_SAState.tree_h`/`.path_h` fields (212-213) and all their read/write
  sites (600-601, 710-713, 968-989, 1051-1066), `LOSS_WEIGHT_TREE_ENTROPY`,
  `LOSS_WEIGHT_PATH_ENTROPY`, and the module docstring's "tree/path template entropy" line (14).
- `local_updates.py`: `_tree_entropy_delta`, `_path_entropy_delta`, `_entropy_from_freq` (check
  first whether anything else still calls it), and the corresponding lines in the module
  docstring (25-26).
- Doc references describing Block E path/tree templates as steered: `docs/generator.md`,
  `docs/report_outline.md`, `docs/notes/assumptions.md`, `docs/notes/generation_algorithm_fit.md`,
  `docs/plan/generation_implementation_plan.md`, `docs/block-refactoring-guide.md`. Reword to
  match how 4.3 already documents Block F path lengths: measured, not steered.

**No corpus regeneration needed** — Block E's vector stays 27-wide, only *how a synthetic graph
is built* changes, not what gets measured on any graph (real or synthetic).

2.12 **`a_obj`/`a_subj` — flagged, kept for now.** *(User instruction: keep, revisit later.)*
Block B's G2b cs-size offset exponents (`fit_cs_size_offset`,
[block_b.py:236,244](src/signature/block_b.py#L236)) were initially raised for removal — high
fit error, no proven effect on Stage 2 output — but the user wants them kept in the signature
and in Stage 2's per-relation degree weighting
([stage2.py:714-762](src/generator/stage2.py#L714)) for now. Not a Stage 2 action item; noted
here per the standing guardrail (Context section) so it isn't re-litigated from scratch — if it
comes back up, the removal shape is: Block B attrs/properties + `as_vector()`/`feature_names()`
entries (B33 → B31), `fit_cs_size_offset` in `_fits.py`, `Schema.a_obj`/`.a_subj`
([schema.py:84,86](src/generator/schema.py#L84)), the Stage 1 population
([stage1.py:358-359,458-461](src/generator/stage1.py#L358)), the Stage 2 weighting itself, and a
corpus regeneration.

---

## Stage 3 — Small code additions (½ day)

3.1 `Signature.from_config(path)` reading YAML, per §3.4. Add to
[generator/pipeline.py](src/generator/pipeline.py) beside `from_graph` / `from_file`. Also the
natural backing for `kgsynth generate --config`.

3.2 **~~Integrate~~ Audit and consolidate the PCA manifold work.** *Landed via merge `7976815`*
as three scripts totalling 709 lines: [plot_signature_pca.py](scripts/plot_signature_pca.py)
(256), [plot_sweep_pca.py](scripts/plot_sweep_pca.py) (191),
[signature_pca_trajectory.py](scripts/signature_pca_trajectory.py) (262). Uses `np.linalg.svd`
— library primitive, good. Remaining work:
- Check the three don't triplicate a corpus loader; they should reuse `SignatureSampler`'s
  (`signature_sampler.py` loads `data/graphs/<name>/signature/signature.json`).
- Reconcile with the underscore-stripped keys from 2.5.
- Document all three in `scripts/README.md` (see 4.5).

Explicitly **not** doing: `sample(num_triples=…)`, per-feature standard errors, derived Block-A
features on export. All three become documented deviations (Stage 4).

---

## Stage 4 — Documentation consistency pass (1 day)

The docs currently contradict the code and each other. Ground truth, from `get_na_vec()`:

> **124 features = A3 + B33 + C29 + D25 + E27 + F7**

4.1 Fix the four mutually-inconsistent feature counts, all wrong:

| Source | Claims | Fix |
|---|---|---|
| [docs/signature.md:46](docs/signature.md#L46) | 117 (A3+B26+C27+D25+E27+F9) | → 124; B, C, F wrong |
| [docs/notes/signature_size_dependence.md:3](docs/notes/signature_size_dependence.md#L3) | 99 | → 124 |
| [src/signature_sampler.py:18](src/signature_sampler.py#L18) | "88-value feature dict" | → 124 |
| [docs/notes/signature_measurement_plan.md:33](docs/notes/signature_measurement_plan.md#L33) | 69 (A3+B18+C23+D16+F9) | historical note; mark as superseded |

4.2 [docs/signature.md:53](docs/signature.md#L53): "The original full signature (`signature/`,
`scripts/measure_signature.py`) is unchanged and still runs" — **false**. The root `signature/`
dir is empty and that script runs the reduced signature. Same "coexisting module" staleness in
[docs/README.md:24,56](docs/README.md#L24) and [docs/signature.md:18](docs/signature.md#L18).

4.3 **Deviations from the proposal — inline notes *and* a summary table** (user's choice). New
section in `docs/signature.md`, cross-linked from the README, with a per-row *what the proposal
asks / what we do / why*:
- Block A `density`, `triples_per_entity`, `relation_reuse` — dropped as algebraically
  derivable (the derivability criterion already argued in `docs/signature.md`).
- Block E induced star counts — dropped; pinned by characteristic sets (Block D) instead.
- Per-feature standard errors (§3.3 step 2) — not stored on the signature; estimator variance
  is characterised once in `scripts/cc_variance.py` / `estimator_variance.py` and reported in
  the writeup.
- `Generator.sample(num_triples=…)` (§3.4) — not implemented. Size is pinned by Block A
  (`num_triples = round(V × mean_degree)`, [stage1.py:227](src/generator/stage1.py#L227)).
  Honoring an arbitrary `num_triples` needs a rescaling law for the *extensive* features
  (Block E raw motif counts, `|R|`, `|T|`, `num_distinct_cs`, `num_components` — enumerated in
  `docs/notes/signature_size_dependence.md`); that is the conditional-on-size model in
  `docs/plan/stage1_population_sampler.md`, blocked on data, and needed only by Phase 2.
- Phase 2 (scaling-law study) — out of scope.
- Block F path lengths — unsteered since the Stage-2 steering removal (0.2).

Mirror each as a short inline note at the corresponding block/function.

4.4 [docs/generator.md](docs/generator.md) (lines ~52, ~82–83) and
[docs/notes/path_length_steering.md](docs/notes/path_length_steering.md) describe the removed
steering. Turn the latter into a historical note headed with "removed; see §Deviations".

4.5 `scripts/README.md` documents 18 scripts; **28** exist after merge `7976815`. Missing:
`patch_block_b_degree_stats.py`, `plot_out_degree_standalone.py`, `rerender_signatures.py`,
`viz_sampling_approaches.py`, `convergence_plot_grid.py`, `plot_signature_pca.py`,
`plot_sweep_pca.py`, `signature_error_boxplot.py`, `signature_pca_trajectory.py`,
`sweep_adaptive_weight_scale.py`. Add them (or delete the one-off `patch_*` script).

4.6 [docs/README.md](docs/README.md) files `plan/generation_implementation_plan.md` under
"Plans (future)" though the generator is implemented. Re-file as implemented/historical.

4.7 Normalize docstrings to **numpydoc**, already the majority; Google-style `Args:` survives in
`kg_io.py`, `signature_sampler.py`, `block_f.py`, `_logging.py`. CLAUDE.md mandates one style.

---

## Stage 5 — Submission surface (1 day)

5.1 A real [README.md](README.md) — currently one line. Contents: what `kgsynth` is;
`pip install -e .`; a quickstart matching the proposal's sketch; the CLI; corpus description;
how to reproduce the figures; and a **Limitations** section linking the 4.3 deviations table,
the unsteered Block F path lengths, and the loosely-bounded CC diamond estimator.

5.2 `examples/` — §5 asks for them. Two scripts: *measure a real KG*, and *generate a synthetic
KG from a target signature and compare per-block distances*.

5.3 Emit `data/signatures/<kg_name>.json` in the flat layout §3.3 step 3 names, via an
**aggregator** over the existing per-graph `data/graphs/<name>/signature/` tree — that layout
was a deliberate choice and stays. Regenerate the corpus so it picks up the public JSON keys
from 2.5.

5.4 Add `ruff` (line-length 100; ~70 lines exceed it today).

5.5 Add `tests/conftest.py`; scope the global 360s `pytest.ini` timeout to the one slow oracle
test (`test_signature_block_e_vs_library.py`) via a marker instead of applying it suite-wide.

5.6 Resolve the `TODO: fix the CC diamond estimator` at
[tests/test_hybrid_motif_counter.py:48](tests/test_hybrid_motif_counter.py#L48), where the test
only loosely bounds the estimator. Fix it, or record it as a known limitation in 5.1.

---

## Stage 5b — Repo hygiene, discovered post-merge (needs user sign-off)

Found while executing Stage 0; all introduced by merge `7976815`, so **not mine to delete
unilaterally**.

5b.1 **`sig_out/` is tracked at the repo root** (7 files under `sig_out/59622641_signature/`).
Contradicts the established convention that signatures live at `data/graphs/<name>/signature/`
with no top-level `sig_out/`. Likely an accidental `git add`. → `git rm -r --cached`, gitignore.

5b.2 **Four TTL graph blobs committed to `tests/graphs/`** — 69,881 lines
(`59621618.ttl` 33,916; `59622641.ttl` 9,842; `59622641_synth.ttl` 13,610;
`test_generated.ttl` 12,513). `.gitignore:7` already carries an ad-hoc
`tests/graphs/59410577.ttl`, so this recurs. Decide: keep as tracked fixtures (they'd fix 5b.3)
or ignore by pattern.

5b.3 **The Block E library cross-check silently skips every graph.**
`tests/block_e_verification_graphs.csv` lists paths under `graphs/data/…`, a prefix that has
never existed (the repo uses `graph_data/`), and `graph_data/` is itself **gitignored**
(`.gitignore:4`). So `test_motif_counts_match_library` — the strongest correctness guarantee in
the repo — resolves nothing, skips all 5 subtests, and reports green. It can never run in a
fresh clone. The graphs the manifest wants (`59622641`, `59621618`) were just committed to
`tests/graphs/` by 5b.2, so repointing the manifest there would make the oracle actually run.
**High value: a passing-but-vacuous test is worse than a missing one.**

5b.4 Untracked and undecided: `data/GRAPH_SIZES.md`, `scaling_laws_student_project.pdf`,
`data/test_graphs/wn18rr_v4/signature/block_b copy.png` (an accidental duplicate).

5b.6 **`load_kg` is not reproducible across processes.** *(Found while fixing 5b.3; not fixed —
it changes every seeded result.)* [kg_io.py:91](src/kg_io.py#L91) claims it "preserve[s]
insertion order for stable vertex indices", but the insertion order is rdflib's iteration
order, which is hash-ordered. With `PYTHONHASHSEED` unset (Python's default) the same file
yields a **different vertex numbering on every run**. Verified: `PYTHONHASHSEED=0` twice →
identical edge list; unset twice → two different ones. Exact motif counts are invariant under
vertex relabelling, so every exact test passes and nothing flags it. But **every seeded sampler
is affected**: measuring the same KG twice with `seed=1` gives different Block E CC estimates
(diamond 720 / 706 / 708 on wn18rr_v4 across three processes). This contradicts
[docs/generator.md:27](docs/generator.md#L27) ("`sample()` derives sub-seeds so the whole
pipeline is reproducible") and `docs/report_outline.md:110` ("reproducible from one integer"),
and means the tracked signature corpus cannot be regenerated bit-for-bit.
*Fix* is one line (sort `node_index` insertion, or sort the triples before the loop), but it
renumbers vertices → changes all seeded outputs → **requires regenerating the corpus**. User's
call. The 5b.3 test works around it locally by canonicalising vertex order before counting.

5b.7 **`HybridMotifCounter` silently stopped using ESCAPE for k=5.** Commit `c8fdd4e`
("Split cc_variance into collector + viz; **commit pending working-tree changes**") flipped
`self._exact.count_motifsk(g, 5)` → `self._cc.count_motifsk(g, 5)` inside a `try`, leaving a
`try`/`except RuntimeError` whose two arms are now identical — the fingerprint of an unintended
edit, and the commit message admits it "bundles other pending changes: … motif_counter edits."
So k=5 is CC-estimated, not exact, contradicting all four places that document it:
[hybrid_motif_counter.py:1](src/motif_counter/hybrid_motif_counter.py#L1) and its class
docstring, `motif_counter/__init__.py`, and (previously) `block_e.py`'s `MOTIF_COUNTER` comment.
Restoring the exact path changes `five_cycle_count` across the tracked corpus, so this is
**not** a drive-by fix. Supersedes the doc-only reconciliation in plan item 2.3 — the code
drifted, not the docs.

5b.5 **Not** deleting the root scratch CSVs (`conv_log_*.csv`, `modified_annealing.csv`, …).
They are gitignored, so they never reach a grader's clone, and they are the user's experiment
outputs. Removing them buys nothing and risks data loss.

---

## Stage 6 — Optional

- **Wikidata subset** — the one proposal-named target KG missing from the corpus (§4.3 names
  DBpedia, Wikidata, SWDF, FB15k-237; three of four present).
- YAGO (explicitly optional in the proposal).

---

## Verification

Run after **every** stage:

```bash
.venv/bin/python -m pytest -q          # ~4 min; must stay green
```

Stage-specific end-to-end checks:

- **Stage 0:** clean venv, `pip install -r requirements.txt`, then `python -c "import signature"`
  — must not raise. This is the fresh-checkout bug.
- **Stage 1** (critical — touches every file):
  ```bash
  pip install -e .
  python -c "from kgsynth import Signature, Generator; print(Signature, Generator)"
  kgsynth --help && kgsynth measure data/graphs/swdf/swdf.nt --output-dir /tmp/sig
  grep -rn "sys.path.insert" src tests scripts   # must return nothing
  ```
- **Stage 2:** re-measure one small graph; diff its `signature.json` against the committed one.
  Only key names (underscore strip) should differ — any changed *value* is a regression.
- **Stage 3:** `Signature.from_config` round-trips a YAML target; `manifold_projection.py`
  renders a 2D plot with the six real KGs labeled.
- **Stage 4:** `grep -rn "117\|99 reduced\|88-value\|69 " docs/ src/` returns nothing stale;
  every file path cited in the docs resolves.
- **Stage 5:** in a fresh clone + venv, follow the README verbatim, top to bottom. Anything that
  doesn't work is a bug.

The riskiest step is **1.4** (deleting 33 `sys.path` hacks). Do the move and the import rewrite
as one commit and run the suite immediately; a green suite after 1.4 is what makes the rest of
the plan safe.

---

## Execution workflow

**One stage per session, one session per context.** Each stage is sized to finish inside a
single context window with the test suite run at both ends. Start each session fresh (`/clear`)
with:

> Execute Stage N of `docs/plan/submission_cleanup_plan.md`.
> Run the suite first to confirm the baseline, then work the stage, then verify.

Do **not** batch stages. Stage 1 renames every import in the repo; anything begun before it
lands has to be rebased across the move.

### The loop, per stage

1. `.venv/bin/python -m pytest -q` — establish the baseline *before* editing (~4 min). If it's
   already red, stop and say so rather than layering changes on a broken tree.
2. Work the stage's numbered items.
3. `pytest` again, plus the stage's end-to-end check from the Verification section.
4. `/code-review` (medium) on stages **1, 2, 3** — they carry logic risk. Skip it for stage 4
   (docs) and use `/simplify` instead on stage 5.
5. Append a `## Unreleased` entry to [CHANGELOG.md](CHANGELOG.md) per the workflow that file
   documents — a title plus terse *what and why* bullets, grouped under
   `Added`/`Changed`/`Fixed`/`Removed`.
6. Commit. Use the CHANGELOG entry as the commit body, then **delete that entry** from the file
   (the repo convention: only genuinely uncommitted work stays there).

### Commit granularity

| Stage | Commits | Notes |
|---|---|---|
| 0 | 1 | Includes the already-made steering removal; write its CHANGELOG entry retroactively. |
| 1 | 2 | (a) `pyproject` + `git mv` + import rewrite + `sys.path` removal + `conftest` — **must be atomic**, the tree does not import between the move and the rewrite. (b) the CLI. |
| 2 | 1 per item, or 1 | Items are independent; 2.5 (JSON keys) touches the tracked corpus, so keep it alone. |
| 3 | 2 | `from_config`; PCA integration. |
| 4 | 1 | Docs only. |
| 5 | 2–3 | README + examples; tooling (`ruff`, `conftest`); the CC diamond TODO. |

Use `git mv` in 1.2 so file history survives the move — a reviewer can still `git log --follow`
a block module.

### What to parallelize, and how

Most stages are strictly sequential. Three pieces are genuinely independent and can run in
parallel **if you want them off the critical path**:

- **5.6** — the CC diamond estimator investigation. Self-contained, open-ended, a good fit for a
  subagent: "figure out why the estimator is biased and whether the loose bound in
  `test_hybrid_motif_counter.py:48` can be tightened."
- **6** — Wikidata subset acquisition. Pure data work, no code dependency.
- **5.4** — `ruff` config. Trivial, but it will churn every file, so land it *last*, after all
  code motion.

**Do not run these as parallel sessions in this working tree.** Concurrent sessions share it and
`git stash` is off the table here; use `git worktree` (or an agent with
`isolation: "worktree"`) so each has its own checkout.

### Guardrails

- **Never** let stage 1 and stage 2 be in flight simultaneously.
- After 2.5 (underscore-stripped JSON keys), `signature_sampler.py`, the tracked corpus, and the
  Stage-3 PCA script all read those keys. Change all three in the same commit or the corpus
  silently mismatches its readers.
- The suite takes ~4 min. That is cheap relative to a bad merge — run it every time, don't infer
  from "the edit looked right."
- Per [CLAUDE.md](CLAUDE.md): brief comments only where intent isn't evident, numpydoc
  docstrings, and `docs/` updated in the same session as the code it describes. Stage 4 is the
  backstop for docs, not the excuse to defer them.
