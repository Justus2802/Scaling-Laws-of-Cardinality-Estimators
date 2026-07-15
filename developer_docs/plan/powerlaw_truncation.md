# Plan: make every power law consistently truncated (fit **and** sample)

Status: **executed 2026-07-13** (see `CHANGELOG.md`). Companion to
[remove_unnecessary_fallbacks.md](remove_unnecessary_fallbacks.md), which landed first (`953eb90`).

Two findings that contradict what is written below, recorded here so the plan is not read as
retrospectively correct:

1. **The truncated α is *shallower*, not steeper** (Verification §3 predicted steeper). An
   unbounded MLE must normalise over an infinite tail, which forces `α > 1` and biases it upward;
   the truncated one is free of that. Corpus degree exponents fell from ≈2.2–2.9 to ≈1.0–1.8.
   Confirmed against a brute-force MLE, so this is the real optimum, not a fitter artifact.
2. **Not every feature improved** (Verification §2 asked for no regression). `in_degree_max` got
   worse (0.038 → 0.239 on fb237_v4) because the old *unbounded* multiplicity draw emitted
   occasional enormous weights that concentrated in-edges onto a single object — it was hitting the
   in-degree hub target by accident. The overall error still fell sharply (mean 0.641 → 0.182), and
   the clamp's runaway CS reuse (max 683 vs a target of 156) is gone.

## Motivation

The repo's power-law handling is internally inconsistent in two independent ways:

1. **Fit side.** `fit_truncated_powerlaw` pins `xmin`/`xmax` and stores both bounds (`v_min`,
   `v_max`), but `_fit_powerlaw` — used for degree distributions, class sizes and per-relation
   multiplicity α — pins `xmin=1` with **no `xmax`**, i.e. an unbounded MLE over an inherently
   bounded quantity.
2. **Sampling side.** `sample_powerlaw` draws from an unbounded `[1, ∞)` inverse-CDF, and
   `_assign_templates` then applies `np.minimum(fit, reuse_vmax)`. **A clamp is not a truncated
   draw** — it deposits an atom of probability mass exactly at `v_max` instead of redistributing it
   across the bounded support. The distribution being sampled is therefore not the distribution that
   was fitted.

Meanwhile a *correct* truncated inverse-CDF already exists in the codebase — `_trunc_powerlaw`,
buried as a nested closure inside `sample_degree_sequence`. It just isn't reused.

## Current state (verified against the code and the 9-KG corpus)

| Fit | Fitted how | Bound stored | Sampled how | Consistent? |
|---|---|---|---|---|
| `cs_freq_fit`, `inv_cs_freq_fit` (Block D) | `fit_truncated_powerlaw` → `powerlaw.Fit(xmin, xmax)` | ✅ `v_min`, `v_max` | `sample_powerlaw` unbounded, then `np.minimum(·, v_max)` **clamp** | ❌ atom at `v_max`; `v_min` ignored |
| `two_step_fit` (Block D) | truncated | ✅ | not sampled (validation only) | — |
| `out_degree_fit`, `in_degree_fit` (Block B) | `_fit_powerlaw`, `xmin=1`, **no `xmax`** | ❌ (`p90` / `max` stored as separate features) | `_trunc_powerlaw` on `[1, p90]` | ❌ unbounded α used for a truncated draw |
| `obj_alpha_q`, `subj_alpha_q` (Block B) | per-relation `_fit_powerlaw`, no `xmax` | ❌ no max-multiplicity feature exists | `sample_powerlaw` **unbounded**, no cap at all | ❌ wrong at both ends |
| `class_size_fit` (Block C) | `_fit_powerlaw`, no `xmax` | ❌ | `_zipf_weights` (rank law over finite n) | ⚠️ different object — see note |
| `relation_zipf` (Block B) | `_fit_powerlaw`, no `xmax` | ❌ | `_zipf_weights` (rank law over finite n) | ⚠️ different object — see note |

**Note on the two ⚠️ rows.** `_zipf_weights` builds `rank^(−α)` over a finite `n` and normalises, so
it is bounded *by construction* — the sampling side is fine. The inconsistency is that α was
estimated as an unbounded value-law MLE and is then applied as a rank-law exponent. Fixing the fit
side (below) makes the exponent at least internally consistent.

Also worth recording: the degree **tail** in `sample_degree_sequence` deliberately does *not* use the
fitted α — it uses an extreme-value-matched `α_tail = 1 + ln(n_tail)/ln(max/p90)` so the expected
maximum lands on the target max. The fitted α is used only for the `[1, p90]` body. That is a
deliberate choice, documented in the function, and this plan keeps it; but it means the fitted α
should be a **truncated** MLE over the body range, which is exactly what the fit-side change gives.

---

## Change 1 — Sampling side

**[../../src/kgsynth/generator/_adapters.py](../../src/kgsynth/generator/_adapters.py)**

- **Hoist** the nested `_trunc_powerlaw` (lines 134–143, currently a closure inside
  `sample_degree_sequence`) to a module-level
  `sample_powerlaw_trunc(alpha, lo, hi, n, rng) -> np.ndarray`. It is already a correct truncated
  inverse-CDF (`(lo^a1 + u·(hi^a1 − lo^a1))^(1/a1)` with `a1 = 1 − α`); this is **reuse, not new
  math**. Keep `sample_degree_sequence` calling it.
- **Delete `sample_powerlaw`** (lines 70–82 — the unbounded `[1, ∞)` draw) and replace all three
  call sites:

| Call site | Replace with | Bounds source |
|---|---|---|
| `_assign_templates` ([stage2.py:371](../../src/kgsynth/generator/stage2.py)) | `sample_powerlaw_trunc(α, v_min, v_max, n_p, rng)` | `cs_freq_fit` / `inv_cs_freq_fit` — **Block D already stores both bounds** |
| `w_out` ([stage2.py:733](../../src/kgsynth/generator/stage2.py)) | `sample_powerlaw_trunc(α, 1, obj_mult_max, n_sr, rng)` | new Block B feature (Change 3) |
| `w_in` ([stage2.py:768](../../src/kgsynth/generator/stage2.py)) | `sample_powerlaw_trunc(α, 1, subj_mult_max, n_or, rng)` | new Block B feature (Change 3) |

- **Remove the `np.minimum(fit, reuse_vmax)` clamp** in `_assign_templates` (stage2.py:372–373) — it
  is superseded by the truncated draw and is the source of the `v_max` atom.
- Thread `cs_freq_vmin` / `inv_cs_freq_vmin` through `Schema` alongside the existing
  `cs_template_vmax` / `inv_cs_template_vmax` (Stage 1 currently reads only `v_max` from the fit and
  discards `v_min`).
- Keep `_zipf_weights` ([stage1.py:55](../../src/kgsynth/generator/stage1.py)) unchanged — bounded by
  construction.

## Change 2 — Fit side

**[../../src/kgsynth/signature/_utils.py](../../src/kgsynth/signature/_utils.py)**

`_fit_powerlaw` currently calls `powerlaw.Fit(positive, discrete=True, xmin=1, verbose=False)`. Add
`xmax=positive.max()` so the MLE is the **truncated** one, matching the bounded law that is sampled
from. One change, and degree α, class-size α and per-relation multiplicity α all become consistent
with their samplers.

Update the `_fit_powerlaw` docstring — it currently states the fit is "the MLE over the full positive
support", which will no longer be true.

⚠️ **This changes measured feature values** → corpus regeneration (Change 4).

## Change 3 — New Block B features: `obj_mult_max` / `subj_mult_max`

Per-relation object/subject multiplicity is inherently bounded, but Block B stores no bound, so its
draws cannot be truncated at all today.

**[../../src/kgsynth/signature/block_b.py](../../src/kgsynth/signature/block_b.py)**

- Measure max object-multiplicity and max subject-multiplicity. The loops at 205–212 already build
  `subj_obj_count` / `obj_subj_count`; take the max over those counts (guard the empty case → NaN,
  which lands in the legitimate "no relations" fallback category).
- Add: two `@property` accessors, two `as_vector` entries, two `feature_names` entries, two
  `_state_from_features` keys. Bump `get_na_vec` from
  `6 + 2·n_q + 2 + 4 + (n_q − 1) + 1` to `… + 2`.
- **Block B: 33 → 35 features. Total feature vector: 124 → 126.**

Consumers to update:

| File | Change |
|---|---|
| `generator/schema.py` | carry `obj_mult_max` / `subj_mult_max` |
| `generator/stage1.py` | read them from Block B into the `Schema` |
| `generator/stage2.py` | pass them as the `hi` bound at lines 733 / 768 |
| `dataset/worker.py` | validate `≥ 1` in `_validate_signature` |
| `_domains.py` | count-like → int-cast on `from_features` |
| `tests/test_signature_reduced_blocks.py` | vector-length assertions (124 → 126) |

## Change 4 — Corpus regeneration

Changes 2 and 3 both alter the feature vector (values and length), so all 9 signatures must be
re-measured — rewriting `data/graphs/<kg>/signature/block_*.json` and the flat
`data/signatures/<kg>.json`. Any previously generated dataset built on the old vector is invalidated.

## Change 5 — Documentation

- **[../../user_docs/generator.md](../../user_docs/generator.md)**: document the **truncated-everywhere power-law contract** —
  a short table of which quantity is drawn from which bounded law and where its bounds come from
  (`cs_freq_fit.v_min/v_max`, `[1, p90]` + extreme-value-matched tail for degrees, `[1, mult_max]`
  for multiplicity). Remove any prose implying unbounded draws or post-hoc clamping, and update the
  Inputs table with the two new Block B parameters.
- **[../../user_docs/signature.md](../../user_docs/signature.md)**: the two new Block B features, the 126-length vector, and the
  change of `_fit_powerlaw` to a truncated MLE (what it means for the reported α, and that old
  signature files are not comparable to new ones).
- **`CHANGELOG.md`**: append under `## Unreleased` with `Added` / `Changed` / `Fixed`.

## Verification

1. **The atom test.** Generate for one KG (e.g. `hetionet`) at a fixed seed before and after
   Change 1, and plot the realised CS-template reuse histogram against the target's. The **spike at
   `v_max` should disappear** and the reuse tail should track the target more closely. This is the
   direct, visible symptom of the clamp bug and the clearest proof the fix landed.
2. **No feature regresses.** `scripts/signature_roundtrip.py` and `scripts/signature_error_boxplot.py`
   already do target-vs-generated comparison across the corpus — run both and confirm no per-feature
   error gets worse. `cs_freq_alpha`, `obj_mult_alpha_*` and the degree features should improve.
3. **Fit sanity.** After Change 2, spot-check that the truncated α differs from the old unbounded α
   in the expected direction (a truncated MLE over a finite range is generally *steeper* than the
   unbounded one for the same data).
4. `pytest` and `ruff check` (line-length 100).
