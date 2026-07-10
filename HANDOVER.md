# Handover — Stages 0–4 done; Stage 2 remainder + Stage 5 next

Plan: [docs/plan/submission_cleanup_plan.md](docs/plan/submission_cleanup_plan.md). Branch: `main`.
Suite: **264 passed, 55 subtests, 0 failed, ~4 min** — green before and after every commit below.

> Shell note: this environment's zsh profile fails with `zsh: no such file or directory:
> /mnt/data/claude_tmp/claude-*cwd`, so **every** Bash call exits 1 and background tasks report
> "failed" even when the actual command (incl. pytest) succeeded. Read the real output, not the
> exit code. The suite genuinely passes.

## What's landed (newest first)

- **`d672084` Stage 4 (docs consistency)** — docs + docstrings only, no logic. Ground truth
  from `feature_names()`: **124 = A3 + B33 + C29 + D25 + E27 + F7**. Fixed the stale feature
  counts everywhere (117/99/88/69 → correct; note the population **sampler** is **97**, not 124,
  because it excludes Block E). Also reconciled a representation drift the plan didn't enumerate
  but which was equally false: Block F shortest-path is `max/mean/var`, not the "skew-normal"
  the docs claimed, and multiplicity-α / row-entropy / CS-size are quantile functions, not
  skew-normal. Added the "Deviations from the proposal" table (4.3), documented the 7 missing
  scripts (4.5), re-filed the implemented generation plan (4.6), numpydoc'd the last Google-style
  docstrings (4.7). Skipped `/code-review` per the plan (docs-only).
  - **Deliberately not done, flagged:** `docs/plan/stage1_population_sampler.md` and
    `docs/notes/data_source_evaluation.md` still use an old "69-feature" / "58 of 69"
    population-fit accounting. Their whole analysis is built on that subset and the sampler
    excludes Block E (→97), so they can't be mechanically renumbered to 124 in a docs pass —
    reconcile when that (blocked-on-data, Phase-2) work resumes.
- **`7a8c3fc` Stage 3**, **`56b741b` Stage 2 (partial)**, **`f52132f` and earlier** — see git log.

## Facts the next session needs

- **Corpus state still gates Stage 2 remainder and Stage 5.3.** 2.5 (underscore-strip JSON keys),
  2.10 (delete stale-signature fallbacks), 2.12 (a_obj/a_subj — kept, revisit) and 2.13 (already
  coded) all converge on **one** corpus regeneration, planned once at **5.3** alongside 2.5's key
  rename. Do not regen twice. **Verify the corpus is actually present and in the one true format
  before relying on any of these being unblocked** — the prior handover flagged that the parallel
  corpus-regen session's artifacts had vanished from this checkout; confirm current state first.
- **Two open Stage-1 code-review notes, still minor/by-design:** (1) `cli.py generate --graphs-dir
  X` with no `--output` writes `<graph>_synth.ttl` into the corpus source dir for the corpus-graph
  path; (2) `corpus.py:_REPO = parents[2]` only resolves under an editable install.
- **`load_kg` is not reproducible across processes (5b.6)** — one-line fix, but it renumbers
  vertices → changes all seeded output → needs a corpus regen. Bundle with 5.3 if the user wants
  bit-for-bit reproducibility. Still bounds Stage 5's reproducibility claim.
- **Pre-existing float nondeterminism** (Block C co-occurrence params vary in the last 1–2 ULP,
  BLAS reduction order) — still uninvestigated; also bounds bit-for-bit regen.

## Next up (per plan)

**Stage 5 — submission surface** (README, `examples/`, the flat `data/signatures/<kg>.json`
aggregator + the single corpus regen at 5.3, `ruff`, `conftest`/timeout scoping, the CC diamond
TODO). Use `/simplify` (not `/code-review`) per the workflow. **Stage 2 remainder (2.5/2.10/2.12)**
folds into 5.3's single regen — land their *code* changes as their own commits, do the actual
regen once.

**Stage 5b hygiene items** need user sign-off (tracked `sig_out/`, TTL blobs in `tests/graphs/`,
the vacuous Block E oracle 5b.3, the `load_kg` reproducibility fix 5b.6, the k=5 ESCAPE regression
5b.7). Several change the corpus → coordinate with the 5.3 regen.

This file is scratch — update or delete as the next session sees fit, per the repo's CHANGELOG
workflow (only genuinely uncommitted/unfinished work stays tracked here).
