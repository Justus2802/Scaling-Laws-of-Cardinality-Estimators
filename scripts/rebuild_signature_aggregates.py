#!/usr/bin/env python3
"""Rebuild each corpus graph's aggregate ``signature.json`` / ``summary.txt`` from its blocks.

**Why this exists.** Four of the six population graphs (``aids``, ``codex_l``,
``dbpedia100k``, ``hetionet``) shipped an aggregate ``signature.json`` that is
~101–109 of 124 features NaN — ``aids``'s says ``num_entities: nan`` while its
``block_a.json`` says ``254207``. The per-block files are correct; only the
aggregate is wrong. It was written by a partial re-measure back when the writer
treated the blocks it was handed as the whole truth, so the blocks it was *not*
handed were NaN-filled over.

``write_signature_outputs``'s ``merge=True`` (now the default) fixed the cause —
it reloads whatever blocks are already on disk and folds them in — but nothing
went back and repaired the files written before it. This script does that, reusing
the same ``load_signature_dir`` merge path.

It **does not re-measure**: it only re-derives the aggregates from the
``block_*.json`` already on disk, so it needs neither the source graphs (absent
from this checkout) nor the deferred corpus regeneration. Those remain outstanding
and will change the block *values*; this only makes the aggregates agree with
whatever the blocks currently say.

Who was reading the broken files: ``SignatureSampler.load_corpus`` (so the
population sampler was fitting per-feature ranges over 2 graphs instead of 6,
silently — it drops NaN per feature), the PCA scripts, and
``data/signatures/<name>.json``, the flat public export. Re-run
``scripts/aggregate_signatures.py`` after this to refresh that export.

Usage (from the repo root)::

    python scripts/rebuild_signature_aggregates.py --check     # report, write nothing
    python scripts/rebuild_signature_aggregates.py             # repair all
    python scripts/rebuild_signature_aggregates.py aids swdf   # repair only these
"""

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # block.visualize(mode="text") must not need a display

from kgsynth.corpus import DEFAULT_SEARCH_DIRS  # noqa: E402
from kgsynth.signature import _BLOCK_CLASSES, load_signature_dir  # noqa: E402

# Signature width, derived rather than hard-coded so it tracks the blocks.
_N_FEATURES = sum(len(c.feature_names()) for c in _BLOCK_CLASSES.values())


def _nan_count(features: dict) -> int:
    """Number of NaN entries in a feature dict."""
    return sum(1 for v in features.values() if isinstance(v, float) and math.isnan(v))


def _discover(names: set[str] | None) -> dict[str, Path]:
    """Map graph name -> its ``signature/`` directory, across both corpus roots."""
    found: dict[str, Path] = {}
    for root in DEFAULT_SEARCH_DIRS:
        if not root.is_dir():
            continue
        for sig_dir in sorted(root.glob("*/signature")):
            name = sig_dir.parent.name
            if names and name not in names:
                continue
            if not any(sig_dir.glob("block_*.json")):
                continue  # nothing measured here
            found.setdefault(name, sig_dir)
    return found


def _source_of(sig_dir: Path) -> str:
    """Preserve the existing ``source`` field, or fall back to the graph directory."""
    existing = sig_dir / "signature.json"
    if existing.exists():
        with contextlib.suppress(Exception):
            return json.loads(existing.read_text())["source"]
    return str(sig_dir.parent)


def rebuild(sig_dir: Path, *, write: bool) -> tuple[int, int]:
    """Rebuild one graph's aggregates from its ``block_*.json``.

    :param sig_dir: The graph's ``signature/`` directory.
    :param write: If false, compute and report but write nothing.
    :returns: ``(nan_before, nan_after)`` — NaN counts in the aggregate feature dict.
    """
    aggregate = load_signature_dir(sig_dir)
    features = aggregate.as_dict()
    after = _nan_count(features)

    json_path = sig_dir / "signature.json"
    before = after
    if json_path.exists():
        with contextlib.suppress(Exception):
            before = _nan_count(json.loads(json_path.read_text())["features"])

    if not write:
        return before, after

    json_path.write_text(
        json.dumps({"source": _source_of(sig_dir), "features": features}, indent=2)
    )

    # summary.txt is derived from the same block set, so it was truncated identically.
    sections: list[str] = []
    for block in aggregate._blocks():
        if block is None:
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            block.visualize(mode="text", path=None)
        sections.append(buf.getvalue().rstrip())
    (sig_dir / "summary.txt").write_text("\n\n".join(sections) + "\n")

    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graphs", nargs="*", help="Graph names (default: all).")
    parser.add_argument("--check", action="store_true",
                        help="Report what would change; write nothing. Exits 1 if stale.")
    args = parser.parse_args()

    targets = _discover(set(args.graphs) or None)
    if not targets:
        raise SystemExit(f"No signature dirs found for {args.graphs or 'any graph'}.")

    stale = 0
    print(f"{'graph':<16} {'signature.json':>16} {'from blocks':>14}   status")
    for name, sig_dir in sorted(targets.items()):
        before, after = rebuild(sig_dir, write=not args.check)
        is_stale = before != after
        stale += is_stale
        status = ("REPAIRED" if not args.check else "STALE") if is_stale else "ok"
        print(f"{name:<16} {f'{before}/{_N_FEATURES} NaN':>16} "
              f"{f'{after}/{_N_FEATURES} NaN':>14}   {status}")

    if args.check:
        print(f"\n{stale} of {len(targets)} aggregates are stale.")
        sys.exit(1 if stale else 0)

    print(f"\nRepaired {stale} of {len(targets)}. "
          "Now re-run scripts/aggregate_signatures.py to refresh data/signatures/.")


if __name__ == "__main__":
    main()
