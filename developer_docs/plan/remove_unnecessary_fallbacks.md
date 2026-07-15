# Plan: remove unnecessary fallbacks from the generator

Status: **planned, not started**. Companion to
[powerlaw_truncation.md](powerlaw_truncation.md) — do **this** plan first (it is pure deletion, no
feature-vector change, no corpus regeneration), then that one.

## Motivation

The generator carries a large amount of degraded-mode code that handles signature data which, in
practice, is *always* measurable on a real graph. The clearest example: Stage 2 contains two
complete parallel implementations of characteristic-set assignment — "template mode" (Block D
present) and "per-entity mode" (Block D absent) — plus a third legacy `_sample_cs_for_type` path.
Block D is never actually absent.

Blocks B, D and F are declared optional on `Signature` and `sample_schema`, but **no production
caller ever omits them**:

- `Signature.from_graph` measures all six blocks.
- `Signature.from_features` reconstructs all six.
- `corpus.load_signature` raises `SystemExit` when any cached `block_*.json` is missing.
- Only `tests/test_generator_stage1.py` calls `sample_schema(a, c, seed=0)` without B/D/F.

The optionality therefore buys nothing and costs an entire second code path per feature.

## Evidence: which NaNs are real

Every NaN feature across all 9 corpus signatures (`data/signatures/*.json`) was enumerated. The
result partitions the fallbacks cleanly:

| NaN feature group | Occurs in | Root cause | Verdict |
|---|---|---|---|
| `relation_zipf_*`, `obj/subj_mult_alpha_q*`, `subj/obj_row_entropy_q*` | aids (R=5), wn18rr (R=9) | fewer than `MIN_SAMPLES_FOR_FIT`=10 **relations** | **KEEP** — the legitimate "missing relation" fallback |
| `recip_symmetric_frac_bin*` | aids, codex_l, hetionet, wn18rr | empty frequency bin (small R) | **KEEP** — already handled by nearest-bin borrow |
| `recip_symmetric_value` | hetionet, swdf | genuinely **no symmetric relation exists** | **KEEP** — real measurement outcome |
| `cs_freq_*`, `inv_cs_freq_*` | aids only | small R → too few distinct CSs to fit | **KEEP** (relation-driven) |
| `class_size_*`, `type_rel_spectrum_*`, `per_type_entropy_*` | 7 of 9 KGs | `num_classes == 0` — **untyped KGs are the majority case** | **KEEP** — the normal path, not a fallback |
| `path_template_*_k*` | most | no path of that length exists | **KEEP** — real measurement outcome |
| **anything entity-count-driven** | **never** | — | **DELETE all guards** |

Never NaN on any corpus KG: `cs_size_q`, `inv_cs_size_q`, `num_distinct_cs`, `inv_num_distinct_cs`,
`out/in_degree_fit`, `out/in_degree_p90`, `out/in_degree_max`, `a_obj`, `a_subj`, `subj_cooc_exp`,
`obj_cooc_exp`. **Every fallback keyed on these is unreachable code.**

`signature_sampler` propagates corpus NaNs but never invents new ones, so this holds for sampled
signatures too. A "too few entities to fit" fallback is not a real scenario — the smallest corpus KG
has orders of magnitude more than `MIN_SAMPLES_FOR_FIT = 10`.

---

## Step 1 — Make Blocks B, D, F mandatory

- **[../../src/kgsynth/generator/pipeline.py](../../src/kgsynth/generator/pipeline.py)**
  - `Signature.b/d/f` lose `| None = None`.
  - `from_config` requires all six letters — drop the `elif letter in ("a", "c", "e")` special case.
  - `as_features` (line 122) drops the `block.as_vector() if block is not None else cls.get_na_vec()`
    branch.
  - **Keep `Signature.e` nullable.** `corpus.load_signature(with_block_e=False)` legitimately skips
    the expensive Block E for Stage-1/2-only consumers. Type it `BlockE | None` honestly rather than
    leaving the current mismatch (declared required, passed `None`).
- **[../../src/kgsynth/generator/stage1.py](../../src/kgsynth/generator/stage1.py)**
  - `sample_schema(a, c, *, d, b, f)` — required keyword arguments.
  - Delete every `if b is not None` / `if d is not None` / `f is not None` guard: lines 191, 249–270,
    272–298, 303–317, 330–331, 360.
  - Delete the `DEFAULT_NUM_COMPONENTS` / `DEFAULT_LCC` constants (39–40).
- **[../../src/kgsynth/generator/stage3.py](../../src/kgsynth/generator/stage3.py)**
  - `target_f` becomes required; drop the `if target_f is not None` guards at 468, 516, 1244–1245.
- **[../../src/kgsynth/signature/__init__.py](../../src/kgsynth/signature/__init__.py)** — lines 73
  and 84 carry the same None-branch; remove.
- **Tests.** `tests/test_generator_stage1.py` calls `sample_schema(self.a, self.c, seed=0)` in ~10
  places — the only callers exercising the optional path. Give them real B/D/F blocks measured from
  the same fixture graph the A/C blocks come from.

## Step 2 — Delete the unreachable Stage-2 fallbacks

With Block D mandatory and `cs_size_q` never NaN, `cs_num_templates == 0` cannot occur. In
**[../../src/kgsynth/generator/stage2.py](../../src/kgsynth/generator/stage2.py)** delete:

- The **per-entity forward-CS branch** (413–421) and the **per-entity inverse-CS branch** (498–505).
- The **type-based template branch** (436–461). Unreachable: `subj_group_probs` is non-`None`
  whenever R ≥ 3 (the exp-decay fit needs 3 rank points), and the smallest corpus KG has R = 5.
  Replace with an explicit `raise` (Step 4), not a silent fallback.
- The **legacy `_sample_cs_for_type` branch** (462–465) and the `_sample_cs_for_type` helper (315–323).
- The `entity_inv_cs is None` → "every object eligible for every relation" path: `objects_by_rel is
  None`, `O_r is None`, `obj_ids = all_objs` (720–721), and the `if O_r is not None` guard (775).
- The budget-derived CS-size Poisson fallback: `fallback_cs_mean` (206–209), `objects_per_slot` (205),
  the `FALLBACK_CS_MEAN_FLOOR` constant (33), and the `else rng.poisson(...)` arm of `_draw_size`
  (211–215). `_draw_size` becomes a straight `sample_quantiles_trunc` call.
- The `tgt_out is None` / `tgt_in is None` guards (603–633, 736, 759, 769, 783, 1041, 1046) —
  `sample_degree_sequence` can no longer return `None`, because `p90` / `max` are never NaN.

### Cascading deletions

- **stage1.py**: `cs_size_mean`, `mean_functionality`, `FUNCTIONALITY_FLOOR`, and the `_safe(fn)`
  RuntimeError-catching helper (283–288) — a back-compat shim for stale signature files that no
  longer exist.
- **[../../src/kgsynth/generator/_adapters.py](../../src/kgsynth/generator/_adapters.py)**:
  `_functionality_from_alpha` (its only consumer is the deleted `mean_functionality`); `_quantile_mean`
  (its only consumer is the deleted `cs_size_mean` — delete it and its two tests in
  `tests/test_signature_distance.py`).
- **[../../src/kgsynth/generator/schema.py](../../src/kgsynth/generator/schema.py)**: drop the
  `cs_size_mean` and `mean_functionality` fields. Make `subj/obj_group_probs`,
  `subj/obj_group_weights`, `target_out/in_degrees`, `cs_num_templates`, `inv_cs_num_templates` and
  `relation_reciprocity` **required** fields rather than `None` / `0` / `_NAN_Q`-defaulted — a
  `Schema` should not be constructible in a degraded state.
- **schema.py `degree_mechanism`** (line 80): **fully dead code**. Nothing in `src/`, `tests/`,
  `scripts/` or `examples/` ever sets it away from `"capacity"`, so the `"chunglu"` expected-degree
  path has never run. Delete the field and both `== "chunglu"` branches in stage2 (737–739, 771–772);
  keep the capacity path unconditionally.
- **stage3.py `CYCLE_DELTA_MAX_DEGREE`**: per `CHANGELOG.md`, its auto-derive-from-percentile branch
  checks `< 0` but the constant defaults to `inf`, so the cycle-delta guard is off by default and the
  branch is unreachable. Delete it (or wire it up — but deletion is consistent with this plan).

## Step 3 — Keep exactly the legitimate fallbacks

These survive, with their comments rewritten to state *why* they are legitimate, so the remaining
fallbacks read as deliberate rather than defensive:

- `DEFAULT_ZIPF_EXPONENT` when `relation_zipf.exponent` or `cs_freq_fit.alpha` is NaN — **small R**.
- The NaN-α → flat-weight path in `_relation_alpha` (stage2.py:713–716) — **small R**.
- The nearest-non-empty-bin borrow for `recip_symmetric_frac` (stage1.py:366–380) — **small R**;
  and `symmetric_value` NaN → `0.9` (hetionet/swdf have no symmetric relation at all).
- Uniform type weights when `class_size_fit.alpha` is NaN, and the whole `num_types == 0` path —
  **untyped KGs are 7 of 9 in the corpus**, so this is a first-class case, not a fallback.
- `_reconstruct_singular_values` returning empty on a NaN exp-decay fit, *for the type-relation
  spectrum only* (untyped KGs). The `subj/obj_cooc_exp` spectra are never NaN → their empty-return
  path becomes a `raise` (Step 4).

## Step 4 — Replace the deleted fallbacks with loud validation

Deleting a fallback must not mean "crash mysteriously three stages later". Add
`_validate_target(a, b, c, d, f)` at the Stage-1 entry point, raising `ValueError` that **names the
offending feature** when a must-be-finite quantity is NaN:

```
num_entities, mean_degree, num_relations,
cs_size_q, inv_cs_size_q, num_distinct_cs, inv_num_distinct_cs,
out_degree_fit.alpha, in_degree_fit.alpha,
out_degree_p90, out_degree_max, in_degree_p90, in_degree_max,
a_obj, a_subj,
subj_cooc_exp, obj_cooc_exp
```

The KEEP list from Step 3 stays explicitly permitted to be NaN.

## Step 5 — Documentation

- **[../../user_docs/generator.md](../../user_docs/generator.md) — the main doc casualty.** It currently documents behaviour
  that will no longer exist. Remove:
  - the per-entity / legacy CS-mode descriptions in the Stage-2 section;
  - the "optional" language for Blocks B/D/F throughout (Inputs table, Stage-1 §, Stage-2 §, Design
    notes);
  - the `degree_mechanism` capacity/chunglu description;
  - the Known-limitations entries that referred to removed fallbacks (including the
    `CYCLE_DELTA_MAX_DEGREE` caveat).

  Add: a "target signature must be complete" precondition section documenting the `_validate_target`
  contract and the short list of features that are *legitimately* allowed to be NaN (with the reason
  — small R / untyped KG / no symmetric relation).
- **[../../user_docs/signature.md](../../user_docs/signature.md)**: note which features are guaranteed finite on any real graph
  versus which may legitimately be NaN, mirroring the evidence table above.
- **`CHANGELOG.md`**: append under `## Unreleased` with `Changed` / `Removed` subheadings.

## Verification

1. **Reachability proof before deleting.** For each fallback slated for removal, add a temporary
   `assert` at its trigger condition, run all 9 corpus KGs through Stages 1–2, and confirm no assert
   fires. Only then delete. This turns "I believe it's dead" into a measurement, and catches any
   fallback this analysis got wrong.
2. `pytest` — must pass once the Stage-1 fixtures are given real B/D/F blocks.
3. `ruff check` (line-length 100, per the repo config).
4. Generate one graph per corpus KG at a fixed seed and confirm the signature-comparison error does
   not regress (`scripts/signature_roundtrip.py`, `scripts/signature_error_boxplot.py`). Behaviour on
   real inputs should be **bit-identical** — only unreachable branches were removed.
5. Report the **net LOC delta**. The point of the exercise is that this is a substantial deletion,
   not a refactor that moves code around.
