#!/usr/bin/env python3
"""Aggregate the per-graph signature tree into the flat ``data/signatures/`` layout.

The measured corpus stores each graph's signature under
``data/graphs/<name>/signature/`` (population graphs) and
``data/test_graphs/<name>/signature/`` (held-out graphs) — a per-graph directory
of block plots, per-block JSON, a text summary, and the combined
``signature.json``. That nested layout is the working format and stays.

This script emits the *public*, flat layout the proposal (§3.3 step 3) names:
one ``data/signatures/<name>.json`` per graph, each the graph's combined
``{"source", "features"}`` signature — so a downstream consumer can read every
KG's 126-feature vector from a single directory without walking the corpus tree.

It is a pure copy/normalise over what ``kgsynth measure`` already wrote; it does
**not** re-measure anything. (Re-measuring the corpus — to pick up 2.5's public
JSON keys and 2.13's pinned-xmin values — is a separate, deliberately deferred
regeneration pass; run this aggregator again afterwards to refresh the flat
copies.)

Usage (from the repo root)::

    python scripts/aggregate_signatures.py                 # all corpus graphs
    python scripts/aggregate_signatures.py swdf fb237_v4   # only these
    python scripts/aggregate_signatures.py --out-dir /tmp/sigs
"""

import argparse
import json
from pathlib import Path

from kgsynth.corpus import DEFAULT_SEARCH_DIRS, REPO_ROOT

_DEFAULT_OUT = REPO_ROOT / "data" / "signatures"


def _discover(names: set[str] | None) -> dict[str, Path]:
    """Map graph name -> its ``signature/signature.json`` across both corpora.

    :param names: If given, keep only these graph names; else take every graph
        that has a combined ``signature.json``.
    :returns: ``{graph_name: signature_json_path}``, first match per name wins.
    """
    found: dict[str, Path] = {}
    for corpus in DEFAULT_SEARCH_DIRS:
        if not corpus.is_dir():
            continue
        for graph_dir in sorted(corpus.iterdir()):
            sig = graph_dir / "signature" / "signature.json"
            if sig.is_file() and graph_dir.name not in found:
                if names is None or graph_dir.name in names:
                    found[graph_dir.name] = sig
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graphs", nargs="*", default=None, metavar="NAME",
                        help="Graph names to aggregate (default: every graph with a "
                             "cached signature.json under data/graphs/ or data/test_graphs/).")
    parser.add_argument("--out-dir", default=None,
                        help=f"Destination directory (default: {_DEFAULT_OUT}).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _DEFAULT_OUT
    selected = set(args.graphs) if args.graphs else None
    graphs = _discover(selected)

    if not graphs:
        roots = " or ".join(str(c) for c in DEFAULT_SEARCH_DIRS)
        target = f" matching {sorted(selected)}" if selected else ""
        print(f"No cached signatures{target} found under {roots}.")
        return 1

    # Warn about names that matched nothing, so a typo doesn't pass silently.
    if selected is not None:
        for missing in sorted(selected - set(graphs)):
            print(f"!!! No cached signature for {missing!r}; skipping.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sig_path in graphs.items():
        # Round-trip through json so the flat copy is normalised (stable 2-space
        # indent) regardless of how the source file was written.
        data = json.loads(sig_path.read_text())
        dest = out_dir / f"{name}.json"
        dest.write_text(json.dumps(data, indent=2))
        n_features = len(data.get("features", {}))
        print(f"  {name:<16} -> {dest}  ({n_features} features)")

    print(f"\nAggregated {len(graphs)} signatures into {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
