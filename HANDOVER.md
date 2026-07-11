# Handover — Stages 0–4 done; Stage 2 remainder in progress; Stage 5 next

Plan: [docs/plan/submission_cleanup_plan.md](docs/plan/submission_cleanup_plan.md). Branch: `main`.
Suite: **264 passed, 55 subtests, 0 failed, ~4 min** — green before and after every commit below.

## Decisions taken this session (user sign-off)

- **Corpus regen is DEFERRED entirely.** Land the 2.5 / 2.13 *code* changes now (done, below), but
  do **not** regenerate the tracked corpus this session. The single regen (5.3, with 2.5's clean
  keys + 2.13's pinned-`xmin` values, and optionally 5b.6/5b.7) is handed off as a **discrete
  follow-up**. Consequence: the tracked `data/graphs/*/signature/` files still carry **old values
  and underscore keys** until that regen runs — this is expected, not a bug.
- **Keep the TTL fixtures (5b.2/5b.3).** Keep the 4 TTL blobs in `tests/graphs/` as tracked
  fixtures and repoint the Block E oracle manifest at them so the currently-vacuous library
  cross-check actually runs. **Not yet done — next up.**

## Landed this session (Stage 2 remainder, code only — regen deferred)

- **2.13 — pinned `xmin=1` in `_fit_powerlaw`** ([src/kgsynth/signature/_utils.py](src/kgsynth/signature/_utils.py#L36)).
  The fitted α now describes the whole positive range its consumers sample from, not an
  auto-searched CSN tail. `*_xmin` features become constant `1.0` (removal candidates under the
  standing guardrail). Values in the tracked corpus change **only on the deferred regen**.
- **2.5 — stripped the leading `_` from exported block-JSON keys**
  ([src/kgsynth/signature/_serialize.py](src/kgsynth/signature/_serialize.py)). `encode_state`
  strips one leading underscore; `decode_state` re-adds it **tolerantly** (keys already starting
  with `_` pass through), so the still-underscore tracked corpus keeps loading until regen. The
  flat `signature.json` and its readers (`signature_sampler.py`, the PCA scripts) were **already**
  clean-keyed — no reader changes were needed, contrary to the plan's caution.
- **2.10 — NOT done.** Deleting the stale-signature fallbacks was explored (the `stage1.py:264`
  `getattr(c, "type_relation_conditional", None) or {}` site and the others the plan lists) but
  **not applied** — the session was cut short here. Verify each removed fallback is a true
  back-compat guard (not genuine optionality — see plan 2.10's "distinguish two cases") and that
  the tracked corpus still loads after removal, since the regen that would guarantee that is
  deferred.
- **2.12 — no action** (a_obj/a_subj kept, by standing user instruction).

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

- **The single corpus regen (5.3) is the pivot, and it is DEFERRED.** It must pick up: 2.5's clean
  JSON keys, 2.13's pinned-`xmin` values, and (per user's later call) optionally 5b.6 (`load_kg`
  reproducibility) and 5b.7 (k=5 exact ESCAPE). Do it **once**, not twice. Until it runs, the
  tracked `data/graphs/*/signature/` corpus carries **old values + underscore keys**; the tolerant
  `decode_state` (2.5) keeps it loading. Regen command lives in the aggregator work (5.3) — measure
  each of the 6 corpus graphs (`aids codex_l dbpedia100k fb237_v4 hetionet swdf`) and re-emit their
  per-graph `signature/` tree, then the flat `data/signatures/<kg>.json`.
- **2.10 must be validated against the deferred regen.** Because the corpus is NOT regenerated,
  deleting a fallback that the *current* tracked corpus actually relies on would break loading.
  Before removing each fallback, confirm the field is present in every tracked `block_*.json`
  (or accept that 2.10 lands only *with* the regen). Distinguish back-compat guards (remove) from
  genuine optionality (keep) per plan 2.10.
- **Two open Stage-1 code-review notes, still minor/by-design:** (1) `cli.py generate --graphs-dir
  X` with no `--output` writes `<graph>_synth.ttl` into the corpus source dir; (2)
  `corpus.py:_REPO = parents[2]` only resolves under an editable install.
- **Two open Stage-1 code-review notes, still minor/by-design:** (1) `cli.py generate --graphs-dir
  X` with no `--output` writes `<graph>_synth.ttl` into the corpus source dir for the corpus-graph
  path; (2) `corpus.py:_REPO = parents[2]` only resolves under an editable install.
- **`load_kg` is not reproducible across processes (5b.6)** — one-line fix, but it renumbers
  vertices → changes all seeded output → needs a corpus regen. Bundle with 5.3 if the user wants
  bit-for-bit reproducibility. Still bounds Stage 5's reproducibility claim.
- **Pre-existing float nondeterminism** (Block C co-occurrence params vary in the last 1–2 ULP,
  BLAS reduction order) — still uninvestigated; also bounds bit-for-bit regen.

## Next up — remaining todo (ordered; this session's working plan)

Use `/simplify` (not `/code-review`) on Stage 5 per the workflow. Run the suite (~4 min) before
and after each item; commit per item.

1. **2.10 — delete stale-signature back-compat fallbacks.** *In progress, not applied.* Sites the
   plan lists: [stage1.py:264](src/kgsynth/generator/stage1.py#L264)
   (`getattr(c, "type_relation_conditional", None) or {}` → read the attr directly),
   [stage1.py:386-390](src/kgsynth/generator/stage1.py#L386) (`_ratio()` try/except → 1.0),
   [schema.py](src/kgsynth/generator/schema.py) legacy-default fields
   (`relation_reciprocity`, `target_out_degrees`, `mean_functionality`, `cs_size_mean`/
   `cs_num_templates` sentinels), and the 6 `getattr(obj, name, default)` calls across
   `generator/` + `signature/`. **Keep** genuine optionality (b/d/f optional blocks, `_adapters.py`
   NaN-fit paths, too-few-classes-to-fit at [stage1.py:253](src/kgsynth/generator/stage1.py#L253)).
   Validate each removal against the deferred/un-regenerated corpus (see Facts above).

2. **5b.2/5b.3 — make the Block E oracle real.** Keep the 4 TTL blobs in `tests/graphs/` tracked;
   repoint `tests/block_e_verification_graphs.csv` from the never-existing `graphs/data/…` prefix
   to the committed `tests/graphs/{59622641,59621618}.ttl`, so `test_motif_counts_match_library`
   stops silently skipping all subtests. Canonicalise vertex order before counting (works around
   5b.6). Confirm it now actually runs (not skips) and passes.

3. **5.5 — scope the pytest timeout.** Move the suite-wide `pytest.ini` `timeout=360` onto just the
   slow oracle test (`test_signature_block_e_vs_library.py`) via a marker; drop the global timeout.
   `tests/conftest.py` already exists.

4. **5.6 — CC diamond estimator TODO** at [tests/test_hybrid_motif_counter.py:48](tests/test_hybrid_motif_counter.py#L48).
   Either tighten the loose bound (investigate the bias — a good subagent task,
   `isolation: "worktree"`) or record it as a known limitation in the README §Limitations.

5. **5.2 — `examples/`.** Two scripts: (a) measure a real KG; (b) generate a synthetic KG from a
   target signature and compare per-block distances. §5 of the proposal asks for these.

6. **5.3 — flat `data/signatures/<kg>.json` aggregator + THE deferred regen.** Build an aggregator
   over the per-graph `data/graphs/<name>/signature/` tree emitting the flat layout (§3.3 step 3).
   The aggregator itself can run over the current corpus now; **the actual regen is the deferred
   handoff** — do it once, picking up 2.5 keys + 2.13 values (+ 5b.6/5b.7 if the user opts in).

7. **5.4 — `ruff`** (line-length 100; ~70 lines exceed today). **Land LAST** — it churns every file.

8. **README §Limitations** (5.1, README already substantial): link the 4.3 deviations table, the
   unsteered Block F path lengths, the CC diamond bound (5.6), and — until the regen lands — the
   non-bit-for-bit reproducibility (5b.6) as a stated limitation.

**Remaining Stage 5b, still needing user sign-off before action:** `sig_out/` untracking (5b.1),
`load_kg` reproducibility fix (5b.6 — changes all seeded output, bundle with the regen), k=5 exact
ESCAPE (5b.7 — changes `five_cycle_count`, bundle with the regen). This session's user calls were:
regen deferred, TTL fixtures kept.

This file is scratch — update or delete as the next session sees fit, per the repo's CHANGELOG
workflow (only genuinely uncommitted/unfinished work stays tracked here).
