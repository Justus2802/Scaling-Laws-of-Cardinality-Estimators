"""Add reduced Block E (motifs, G5) to already-measured corpus signatures.

For each graph under ``data/graphs/<name>/`` this computes Block E and writes
``block_e.json`` + ``block_e.png`` into ``<name>/signature/``, then merges Block
E's named features into ``signature.json`` and appends its text summary to
``summary.txt`` — **without** disturbing the already-measured A/B/C/D/F outputs.

The corpus was measured before reduced Block E existed; this backfills it.
Block E uses color coding (expensive on large graphs), so graphs are processed
one at a time and outputs are written immediately, making the run re-runnable
(existing ``block_e.json`` is skipped unless ``--force``).

Usage
-----
    python scripts/measure_block_e.py                 # all graphs under data/graphs
    python scripts/measure_block_e.py aids swdf        # named graphs only
    python scripts/measure_block_e.py --force          # recompute even if present
"""

import argparse
import contextlib
import io
import json
import logging
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
from kgsynth.kg_io import load_kg
from kgsynth.signature import BlockE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _find_graph_file(d: Path) -> Path | None:
    """Return the first non-synthetic .nt/.ttl graph file in directory ``d`` (None if absent)."""
    for pattern in ("*.nt", "*.ttl", "*.nt.gz", "*.ttl.gz"):
        hits = sorted(p for p in d.glob(pattern) if not p.stem.endswith("_synth"))
        if hits:
            return hits[0]
    return None


def process(graph_dir: Path, fmt: str = "png", force: bool = False) -> None:
    """Compute Block E for one corpus graph and merge it into the signature dir."""
    sig_dir = graph_dir / "signature"
    if not sig_dir.is_dir():
        print(f"  skip {graph_dir.name}: no signature/ dir")
        return
    e_json = sig_dir / "block_e.json"
    if e_json.exists() and not force:
        print(f"  skip {graph_dir.name}: block_e.json exists (use --force)")
        return
    graph_file = _find_graph_file(graph_dir)
    if graph_file is None:
        print(f"  skip {graph_dir.name}: no graph file found")
        return

    print(f"== {graph_dir.name}: loading {graph_file.name} …", flush=True)
    g = load_kg(graph_file)
    print(f"   {g.vcount():,} nodes  {g.ecount():,} edges; computing Block E …", flush=True)
    e = BlockE().calculate(g)

    # 1. Serialized block state (for reconstruction by the roundtrip / sampler).
    e_json.write_text(json.dumps(e.to_serializable(), indent=2))

    # 2. Diagnostic plot.
    e.visualize(mode="plot", path=str(sig_dir / f"block_e.{fmt}"))

    # 3. Merge named features into signature.json (other blocks untouched).
    sig_path = sig_dir / "signature.json"
    if sig_path.exists():
        payload = json.loads(sig_path.read_text())
        payload.setdefault("features", {}).update(e.as_dict())
        sig_path.write_text(json.dumps(payload, indent=2))

    # 4. Append the Block E text summary (idempotent).
    summary_path = sig_dir / "summary.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e.visualize(mode="text", path=None)
    text = buf.getvalue().rstrip()
    if summary_path.exists():
        existing = summary_path.read_text()
        if "Block E" not in existing:
            summary_path.write_text(existing.rstrip() + "\n\n" + text + "\n")
    else:
        summary_path.write_text(text + "\n")

    print(f"   wrote block_e.json + block_e.{fmt}; merged signature.json + summary.txt",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", help="Graph names (default: all under --graphs-dir)")
    parser.add_argument("--graphs-dir", default=str(_REPO / "data" / "graphs"))
    parser.add_argument("--format", default="png", dest="fmt", choices=["png", "pdf", "svg"])
    parser.add_argument("--force", action="store_true", help="Recompute even if block_e.json exists")
    args = parser.parse_args()

    root = Path(args.graphs_dir)
    names = args.graphs or sorted(p.name for p in root.iterdir() if p.is_dir())
    for name in names:
        process(root / name, fmt=args.fmt, force=args.force)


if __name__ == "__main__":
    main()
