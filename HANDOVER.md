# Handover — submission cleanup plan essentially complete; only the corpus regen (+ optional Stage 6) remains

Plan: [docs/plan/submission_cleanup_plan.md](docs/plan/submission_cleanup_plan.md). Branch: `main`.
Suite: **264 passed, 55 subtests, 0 failed, ~7 min** — green before and after every commit below.
`ruff check` — **All checks passed!** project-wide.

## Standing decisions (user sign-off, still in force)

- **The single corpus regen is DEFERRED as a discrete follow-up.** All the *code* that changes
  what a re-measure would emit has landed (2.5 keys, 2.13 pinned-`xmin`); the actual re-measurement
  of the tracked corpus has **not** been run. Until it is, `data/graphs/*/signature/` (and the flat
  `data/signatures/*.json` copied from it in 5.3) still carry **old fit values + underscore keys**.
  The tolerant `decode_state` (2.5) keeps them loading. This is expected, not a bug.
- **TTL fixtures kept** (5b.2/5b.3) — and it turned out the Block E oracle was *already* wired to
  them and running (see below), so nothing was needed there this session beyond verifying it.

## Landed this session (all committed, newest first)

- **`5423d33` Stage 5.4 — ruff (line-length 100), clean check project-wide.** Added `[tool.ruff]`
  (select E4/E7/E9/F/E501) + a `dev` extra pinning ruff. 11 safe auto-fixes (7 dead imports, 2
  f-string/2 semicolon), then a mechanical reflow of 96 long lines + 35 one-line compound
  statements + 1 `l` rename across ~37 files (done by four parallel subagents, each self-verifying
  `ruff check`/compile; I spot-checked the core files). 3 F821 forward-ref annotations resolved via
  `TYPE_CHECKING` (stage3) / `import igraph` (a script). E402 per-file-ignored for `scripts/` only
  (the `_REPO`/sys.path anchor legitimately precedes imports there). Landed last, as planned.
- **`7a59f04` Stage 5.1 — README Limitations section.** Folded in the 4.3 deviations table link,
  the unsteered Block F path lengths, the CC diamond bias (5.6), and the non-bit-for-bit
  reproducibility caveat (5b.6, until the regen lands).
- **`e787cbb` Stage 5.3 — flat `data/signatures/<kg>.json` aggregator.** New
  `scripts/aggregate_signatures.py` copies each graph's combined `signature.json` into the flat
  §3.3-step-3 layout; committed the 9 emitted files (each byte-semantically identical to its
  committed per-graph source). **The aggregator ran over the *current* corpus** — re-run it after
  the deferred regen to refresh the flat copies.
- **`e4fbdd1` Stage 5.2 — `examples/`.** `measure_a_kg.py` and `generate_and_compare.py` (public
  API only), plus `examples/README.md`. Both default to `tests/graphs/test_generated.ttl` and take
  a graph arg. Runtime note in each: full-fidelity measurement is dominated by Block E's 100k-walk
  colour-coding sampler (several minutes on a few-thousand-vertex graph) — the genuine one-time
  cost, which is why the corpus is measured offline and cached.
- **`8aec7ce` Stage 5.5 + 5.6.** 5.5: moved the (inert — `pytest-timeout` wasn't installed)
  suite-wide `pytest.ini` timeout onto the one slow oracle test via
  `@pytest.mark.timeout(360, method="thread")`; installing the plugin + scoping removed the 2 config
  warnings. 5.6: the CC diamond over-count is now documented as a known limitation (target and
  re-measure share the biased estimator, so the round-trip is self-consistent; Stage 3's per-swap
  diamond delta is exact) rather than a `TODO: fix`.
- **`385e80d` Stage 2.10 — remove stale-signature back-compat fallbacks (conservative scope).**
  *User picked the conservative option after I flagged that the plan's literal site-list conflicts
  with its own "distinguish two cases" rule.* Removed only true stale-file guards safe against the
  un-regenerated corpus: `_ratio()`'s and the Block B recip `try/except` (kept their NaN clamps —
  NaN is a real zero-edge-graph outcome, not a stale file), and the dead full-block
  `type_relation_conditional` path + `_build_type_rel_probs_from_measured` helper (reduced BlockC
  never has that attr). **Kept** the Schema optional-block defaults and `getattr(target_e,…,0)`
  reads (genuine optionality). Completed the two test fixtures (`_make_block_c` /`_make_block_b`)
  that were incomplete — the removed fallbacks had been silently tolerating them. Validated: every
  tracked `block_b/c.json` carries the fields read directly, `sample_schema` runs clean on cached
  blocks, suite green, `/code-review` (high) found nothing.
- **Verified already-done (no work needed): 5b.2/5b.3** — the Block E library oracle already points
  at committed `tests/graphs/` fixtures, canonicalises vertex order (works around 5b.6), and
  actually runs (4 tests / 6 subtests pass in ~72s, confirmed). A prior session executed it
  equivalently to the plan's intent.

Earlier work (`62de16f` Stage 2 remainder code, `d672084` Stage 4 docs, `7a8c3fc` Stage 3,
`56b741b` Stage 2 partial, and before) — see git log.

## What remains

1. **THE deferred corpus regen (5.3's regen half; the pivot).** Re-measure the 6 population graphs
   (`aids codex_l dbpedia100k fb237_v4 hetionet swdf`) + the 3 held-out (`fb237_v4_ind wn18rr_v4
   wn18rr_v4_ind`) via `scripts/measure_all_raw.py`, re-emit their per-graph `signature/` tree, then
   re-run `scripts/aggregate_signatures.py` to refresh `data/signatures/*.json`. This picks up 2.5's
   clean keys + 2.13's pinned-`xmin` values **in one pass**. Blocked only on the source graph files:
   the corpus `.nt`/`.ttl` sources are **not present in this checkout** (gitignored / external) — a
   regen needs them restored first. Decide with the user whether to also bundle:
   - **5b.6** (`load_kg` vertex-order reproducibility, one-line fix, renumbers vertices → changes all
     seeded output → must ride with a regen), and
   - **5b.7** (restore exact ESCAPE for k=5, changes `five_cycle_count`).
   Doing 5b.6/5b.7 *with* this regen is the only way to avoid a second regen later.
2. **2.10 revalidation against the regen** — once the corpus is regenerated in the one-true format,
   the conservative-scope call from this session holds, but a re-run of the suite against the fresh
   corpus is the final confirmation.
3. **Stage 6 (optional, per plan):** Wikidata subset (the one proposal-named target KG missing), and
   YAGO (explicitly optional). Pure data-acquisition work, no code dependency.

## Facts worth keeping

- **Environment staleness, not code bugs:** this `.venv` needed `pip install -e .` (kgsynth wasn't
  installed) and was missing `pytest-timeout` and `ruff` — all declared in `pyproject.toml`
  (`[test]`/`[dev]` extras). A fresh `pip install -e .[test,dev]` gets everything. If a session sees
  red collection or "Unknown config option" warnings, install first.
- **Pre-existing float nondeterminism** (Block C co-occurrence params, last 1–2 ULP, BLAS reduction
  order) — still uninvestigated; also bounds bit-for-bit regen. Noted in the README Limitations.
- **Two long-standing minor code-review notes, by design:** `cli.py generate --graphs-dir X` with
  no `--output` writes into the corpus source dir (the `--config` path added in Stage 3 defaults to
  cwd, so only the corpus-graph path is affected); `corpus.py:_REPO = parents[2]` only resolves
  under an editable install.

This file is scratch — update or delete as the next session sees fit.
