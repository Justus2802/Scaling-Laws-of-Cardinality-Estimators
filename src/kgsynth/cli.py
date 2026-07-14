"""Command-line interface: ``kgsynth measure | generate | compare``.

A thin dispatch layer over the package's public API — every subcommand delegates
to functions that already exist and are exercised by the test suite:

``measure``   :func:`kgsynth.signature.compute_reduced_signature` + :func:`write_signature_outputs`
``generate``  :class:`kgsynth.Generator` fed by :func:`kgsynth.corpus.load_target_from_corpus`
``compare``   per-block feature vectors of two signatures, via each block's ``feature_names()``

The scripts under ``scripts/`` remain the place for research workflows (sweeps,
convergence logging, plots); this CLI covers the three operations the package
promises to a user who has only ``pip install``ed it.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from .corpus import DEFAULT_SEARCH_DIRS, load_target_from_corpus
from .dataset import cli as dataset_cli
from .generator import Generator, Signature
from .kg_io import load_kg, save_kg
from .signature import _ALL_BLOCKS, compute_reduced_signature, write_signature_outputs

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _cmd_measure(args: argparse.Namespace) -> None:
    """Measure a KG file's reduced signature and write the block outputs to a directory.

    :param args: Parsed CLI arguments (``kg_file``, ``output_dir``, ``blocks``, ``fmt``, ``show``).
    """
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    # Default to a 'signature/' dir next to the graph, matching data/graphs/<name>/.
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.kg_file).parent / "signature"

    sig = compute_reduced_signature(args.kg_file, verbose=True, blocks=blocks)
    written = write_signature_outputs(sig, out_dir, source=str(args.kg_file),
                                      fmt=args.fmt, show=args.show)
    print(f"Done. {len(written)} files written to {out_dir}/")
    print(f"Feature count: {len(sig.as_dict())}")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate a synthetic KG from a target signature and save it.

    The target comes from either a corpus graph name (``args.graph``, loaded via
    :func:`load_target_from_corpus`) or a YAML config (``args.config``, loaded via
    :meth:`Signature.from_config`) — :func:`build_parser`'s mutually exclusive
    group enforces that exactly one of the two is given.

    :param args: Parsed CLI arguments (``graph``, ``config``, ``output``, ``seed``,
        ``rewire_budget``).
    """
    if args.config:
        target = Signature.from_config(args.config)
        default_out = Path(f"{Path(args.config).stem}_synth.ttl")
    else:
        search_dirs = [Path(args.graphs_dir)] if args.graphs_dir else DEFAULT_SEARCH_DIRS
        target, _blocks, graph_dir = load_target_from_corpus(args.graph, search_dirs)
        default_out = graph_dir / f"{args.graph}_synth.ttl"

    g = Generator(target).sample(seed=args.seed, rewire_budget=args.rewire_budget)

    out_path = Path(args.output) if args.output else default_out
    save_kg(g, out_path)
    print(f"Saved: {out_path}  ({g.vcount()} vertices, {g.ecount()} edges)")


def _cmd_compare(args: argparse.Namespace) -> None:
    """Compare two KG files feature by feature across the full reduced signature.

    :param args: Parsed CLI arguments (``left``, ``right``).
    """
    left = Signature.from_graph(load_kg(Path(args.left)))
    right = Signature.from_graph(load_kg(Path(args.right)))

    print(f"  {'Metric':<38}  {'Left':>14}  {'Right':>14}  {'Rel err':>8}")
    print("  " + "─" * 80)
    lv, rv = [], []
    for label, lb, rb in [
        ("A — size & vocabulary", left.a, right.a),
        ("B — relation frequency & multiplicity", left.b, right.b),
        ("C — schema & co-occurrence", left.c, right.c),
        ("D — characteristic sets & two-step", left.d, right.d),
        ("E — motifs & templates", left.e, right.e),
        ("F — connectivity", left.f, right.f),
    ]:
        print(f"\n  Block {label}")
        for name, a, b in zip(lb.feature_names(), lb.as_vector(), rb.as_vector()):
            err = abs(a - b) / max(abs(a), 1e-9)
            print(f"  {name:<38}  {a:>14.4f}  {b:>14.4f}  {err:>8.3f}")
        lv += lb.as_vector()
        rv += rb.as_vector()

    lv, rv = np.array(lv, dtype=float), np.array(rv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(lv - rv) / np.maximum(np.abs(lv), 1e-9)
    print(f"\n  Vector length    : {len(lv)}")
    print(f"  Mean  rel error  : {np.nanmean(rel):.3f}")
    print(f"  Median rel error : {np.nanmedian(rel):.3f}")
    print(f"  Max   rel error  : {np.nanmax(rel):.3f}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``kgsynth`` argument parser.

    :returns: The configured top-level parser with its three subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="kgsynth",
        description="Measure a KG's statistical signature, generate synthetic KGs from it, "
                    "and compare them.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log progress to stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="Measure a KG's reduced signature")
    m.add_argument("kg_file", help="Path to the input KG (.ttl or .nt)")
    m.add_argument("--output-dir", default=None,
                   help="Where to write results (default: a 'signature/' dir next to the graph)")
    m.add_argument("--blocks", default=",".join(_ALL_BLOCKS),
                   help=f"Comma-separated blocks to compute (default: {','.join(_ALL_BLOCKS)})")
    m.add_argument("--format", choices=["png", "pdf", "svg"], default="png", dest="fmt",
                   help="Image format for block plots (default: png)")
    m.add_argument("--show", action="store_true", help="Display each block's plot after saving")
    m.set_defaults(func=_cmd_measure)

    g = sub.add_parser("generate", help="Generate a synthetic KG from a target signature")
    g_target = g.add_mutually_exclusive_group(required=True)
    g_target.add_argument("graph", nargs="?", default=None,
                           help="Corpus graph name (e.g. 'swdf').")
    g_target.add_argument("--config", default=None,
                           help="YAML target signature (see Signature.from_config).")
    g.add_argument("--output", default=None,
                   help="Output path (default: <graph>_synth.ttl next to the corpus graph, "
                        "or <config-stem>_synth.ttl in the current directory for --config)")
    g.add_argument("--seed", type=int, default=42, help="Master seed (default: 42)")
    g.add_argument("--rewire-budget", type=int, default=50_000,
                   help="Stage-3 rewiring attempts (default: 50000)")
    g.add_argument("--graphs-dir", default=None,
                   help="Override the corpus search dirs (default: data/graphs, data/test_graphs). "
                        "Ignored with --config.")
    g.set_defaults(func=_cmd_generate)

    c = sub.add_parser("compare", help="Compare two KG files across the full signature")
    c.add_argument("left", help="First KG file")
    c.add_argument("right", help="Second KG file")
    c.set_defaults(func=_cmd_compare)

    dataset_cli.add_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``kgsynth`` console script.

    Exits non-zero when a subcommand reports failures (``dataset`` returns its
    failed-unit count), so a partially-failed run is visible to a caller.

    :param argv: Argument list; defaults to ``sys.argv[1:]``.
    """
    args = build_parser().parse_args(argv)
    # A dataset run logs its progress; without -v it would be silent for hours.
    if args.verbose or args.command == "dataset":
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
