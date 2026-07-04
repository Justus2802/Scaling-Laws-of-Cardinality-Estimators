"""Signature round-trip: load a target reduced signature, generate a synthetic
graph from it, save it, re-measure it, and compare.

By default the target signature is searched in ``data/graphs/`` first, then in
``data/test_graphs/`` (smaller graphs excluded from the population fit).  Within
each directory the per-block ``block_*.json`` files written by
``measure_signature_reduced.py`` are used — no recomputation of the original.
Block E is not part of the corpus yet; if ``block_e.json`` is absent it is
measured on demand from the graph file in that directory.  Pass ``--kg-file`` to
measure the full target signature from a graph file instead.

Usage
-----
    python scripts/signature_roundtrip.py aids
    python scripts/signature_roundtrip.py wn18rr_v4          # from data/test_graphs
    python scripts/signature_roundtrip.py aids --seed 7 --rewire-budget 5000
    python scripts/signature_roundtrip.py --kg-file path/to/graph.ttl
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from generator import Generator, Signature
from kg_io import load_kg, save_kg
from signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF
from signature import ReducedGraphSignature, write_signature_outputs
from signature import _distance
import signature.block_e as _block_e
from motif_counter import HybridMotifCounter

# Sample budget for the synthetic graph's final re-measurement (Block E motif CC
# sampling + path/tree walks). Lower than the 100k Block-E default to keep the
# roundtrip fast; targets are read from cache, so only the synthetic side uses it.
_FINAL_SAMPLE_BUDGET = 20_000

# Surface generator + signature-measurement progress and errors in the console.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Reduced block letter → class, in signature order.
_BLOCK_CLASSES = {"a": BlockA, "b": BlockB, "c": BlockC, "d": BlockD, "f": BlockF}

# Directories searched (in order) when --graphs-dir is not set explicitly.
_DEFAULT_SEARCH_DIRS: list[Path] = [
    _REPO / "data" / "graphs",
    _REPO / "data" / "test_graphs",
]

# Where auto-named Stage 3 convergence / swap-proposal CSVs are written.
_CONVERGENCE_LOG_DIR = _REPO / "experiments" / "convergence_logs"
_SWAP_LOG_DIR = _REPO / "experiments" / "swap_delta_logs"


def _auto_log_path(log_dir: Path, prefix: str, graph_label: str, args,
                   run_ts: str) -> Path:
    """Build an auto-named Stage-3 log path under ``log_dir``.

    The filename encodes the graph name, the run options (seed, rewire budget,
    and the skip-templates flag) and the run timestamp so distinct runs — even
    with identical options — don't overwrite each other.

    :param log_dir: Destination directory (e.g. ``experiments/convergence_logs``).
    :param prefix: Filename prefix (``conv`` or ``swaps``).
    :param graph_label: Name of the source graph (corpus name or file stem).
    :param args: Parsed CLI namespace supplying the run options.
    :param run_ts: Run timestamp string (``YYYYmmdd_HHMMSS``).
    :returns: Destination path for the CSV.
    """
    parts = [graph_label, f"seed{args.seed}", f"rb{args.rewire_budget}"]
    if args.skip_templates:
        parts.append("skiptmpl")
    parts.append(run_ts)
    return log_dir / (f"{prefix}_" + "_".join(parts) + ".csv")


def _resolve_log_path(value: "str | None", log_dir: Path, prefix: str,
                      graph_label: str, args, run_ts: str) -> "Path | None":
    """Resolve a log CLI value: ``"AUTO"`` → auto-named path (dir created),
    explicit string → that path, ``None`` → no log."""
    if value == "AUTO":
        path = _auto_log_path(log_dir, prefix, graph_label, args, run_ts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return Path(value) if value else None


def _load_block(cls, path: Path):
    """Reconstruct a reduced block from its serialized ``block_*.json``."""
    return cls.from_serializable(json.loads(path.read_text()))


def _find_graph_file(d: Path) -> Path | None:
    """Return the first non-synthetic .nt/.ttl graph file in directory ``d`` (None if absent)."""
    for pattern in ("*.nt", "*.ttl", "*.nt.gz", "*.ttl.gz"):
        hits = sorted(p for p in d.glob(pattern) if not p.stem.endswith("_synth"))
        if hits:
            return hits[0]
    return None


def _load_target_from_corpus(graph_name: str, search_dirs: list[Path]):
    """Load the cached reduced target signature for ``graph_name``.

    Searches each directory in ``search_dirs`` for ``<graph_name>/signature/``
    and loads blocks A/B/C/D/F from the first match. Block E is loaded from
    ``block_e.json`` if present, else measured from the graph file. Returns
    ``(Signature, blocks_dict)``.
    """
    graph_dir = sig_dir = None
    for graphs_dir in search_dirs:
        candidate = graphs_dir / graph_name
        if (candidate / "signature").is_dir():
            graph_dir = candidate
            sig_dir = candidate / "signature"
            break

    if sig_dir is None:
        available: list[str] = []
        for d in search_dirs:
            if d.is_dir():
                available += sorted(p.name for p in d.iterdir() if p.is_dir())
        raise SystemExit(
            f"'{graph_name}' not found in {[str(d) for d in search_dirs]}. "
            f"Available graphs: {sorted(set(available))}"
        )

    blocks: dict[str, object] = {}
    for letter, cls in _BLOCK_CLASSES.items():
        path = sig_dir / f"block_{letter}.json"
        if not path.exists():
            raise SystemExit(f"Missing cached block: {path}")
        blocks[letter] = _load_block(cls, path)
        print(f"  Loaded : {path.name}")

    e_path = sig_dir / "block_e.json"
    if e_path.exists():
        blocks["e"] = _load_block(BlockE, e_path)
        print(f"  Loaded : {e_path.name}")
    else:
        graph_file = _find_graph_file(graph_dir)
        if graph_file is None:
            raise SystemExit(
                f"block_e.json absent and no graph file in {graph_dir} to measure it from."
            )
        print(f"  block_e.json absent — measuring Block E from {graph_file.name} …")
        blocks["e"] = BlockE().calculate(load_kg(graph_file))

    sig = Signature(
        a=blocks["a"], b=blocks["b"], c=blocks["c"],
        d=blocks["d"], e=blocks["e"], f=blocks["f"],
    )
    return sig, blocks, graph_dir


def _measure_target_from_file(kg_file: Path, skip_templates: bool = False):
    """Measure the full reduced target signature from a graph file."""
    print(f"Loading   : {kg_file}")
    g = load_kg(kg_file)
    print(f"  {g.vcount():,} nodes  {g.ecount():,} edges")
    print("Measuring : blocks A–F on original graph …")
    blocks = {
        "a": BlockA().calculate(g), "b": BlockB().calculate(g),
        "c": BlockC().calculate(g), "d": BlockD().calculate(g),
        "e": BlockE().calculate(g, skip_stars_and_paths=skip_templates),
        "f": BlockF().calculate(g),
    }
    sig = Signature(
        a=blocks["a"], b=blocks["b"], c=blocks["c"],
        d=blocks["d"], e=blocks["e"], f=blocks["f"],
    )
    return sig, blocks


def _fmt(v):
    if isinstance(v, float):
        return "nan" if (v != v) else f"{v:.4f}"   # nan check without math import
    return str(v)


def _row(label, tv, sv):
    tv_s, sv_s = _fmt(tv), _fmt(sv)
    try:
        tv_f, sv_f = float(tv), float(sv)
        if tv_f != tv_f or sv_f != sv_f:       # either NaN
            err_s = "NaN"
        elif abs(tv_f) < 1e-9:
            err_s = "—" if abs(sv_f) < 1e-9 else ">100%"
        else:
            err_s = f"{abs(tv_f - sv_f) / abs(tv_f) * 100:.1f}%"
    except (TypeError, ValueError):
        err_s = ""
    return f"  {label:<38}  {tv_s:>14}  {sv_s:>14}  {err_s:>8}"


def _header(title):
    return f"\n  {'── ' + title + ' ':-<42}{'':->18}{'':->18}{'':->10}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graph", nargs="?", default=None,
        help="Graph name in the corpus (e.g. 'aids' or 'wn18rr_v4'); its cached "
             "target signature is loaded from <graphs-dir>/<graph>/signature/. "
             "Omit when using --kg-file.",
    )
    parser.add_argument(
        "--graphs-dir", default=None,
        help="Corpus root holding <graph>/signature/. "
             "Default: searches data/graphs/ then data/test_graphs/ in order.",
    )
    parser.add_argument(
        "--kg-file", default=None,
        help="Measure the full target signature from this graph file instead of "
             "loading it from the corpus.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Where to save the synthetic graph (.ttl or .nt). "
             "Default: <graph>_synth_<timestamp>.ttl next to the source.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rewire-budget", type=int, default=5_000)
    parser.add_argument("--convergence-log", nargs="?", const="AUTO", default=None,
                        help="Write Stage 3 convergence CSV. With no value, the file "
                             "is auto-named from the graph name and run options and "
                             "written to experiments/convergence_logs/. Pass an "
                             "explicit path to override.")
    parser.add_argument("--swap-log", nargs="?", const="AUTO", default=None,
                        help="Write a Stage 3 swap-proposal CSV (one row per evaluated "
                             "proposal: per-motif deltas, Δloss, accepted). With no "
                             "value, auto-named into experiments/swap_delta_logs/. "
                             "Pass an explicit path to override. Plot with "
                             "scripts/swap_delta_viz.py.")
    parser.add_argument("--skip-templates", action="store_true",
                        help="Skip path/tree template and star measurement on both "
                             "target and synthetic (saves ~8 min on medium graphs).")
    args = parser.parse_args()

    # Single per-run timestamp stamped into every auto-named output (graph,
    # synthetic signature dir, convergence/swap logs) so repeated runs — even
    # with identical options — don't overwrite each other.
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: obtain the target signature ──────────────────────────────────
    if args.kg_file:
        kg_path = Path(args.kg_file)
        target_sig, tblocks = _measure_target_from_file(kg_path, skip_templates=args.skip_templates)
        default_out = kg_path.with_name(f"{kg_path.stem}_synth_{run_ts}.ttl")
        graph_label = kg_path.stem
        graph_dir = kg_path.parent
    elif args.graph:
        search_dirs = [Path(args.graphs_dir)] if args.graphs_dir else _DEFAULT_SEARCH_DIRS
        print(f"Loading   : cached target signature for '{args.graph}' from {[str(d) for d in search_dirs]}")
        target_sig, tblocks, found_graph_dir = _load_target_from_corpus(args.graph, search_dirs)
        default_out = found_graph_dir / f"{args.graph}_synth_{run_ts}.ttl"
        graph_label = args.graph
        graph_dir = found_graph_dir
    else:
        parser.error("provide a corpus graph name or --kg-file")

    # Resolve the log destinations: AUTO → generated path under the experiments
    # dir; an explicit value → that path; absent → no log.
    conv_log_path = _resolve_log_path(
        args.convergence_log, _CONVERGENCE_LOG_DIR, "conv", graph_label, args, run_ts)
    swap_log_path = _resolve_log_path(
        args.swap_log, _SWAP_LOG_DIR, "swaps", graph_label, args, run_ts)

    ta, tb, tc, td, te, tf = (
        tblocks["a"], tblocks["b"], tblocks["c"], tblocks["d"], tblocks["e"], tblocks["f"],
    )
    out_path = Path(args.out) if args.out else default_out

    # ── Step 2: generate synthetic graph ─────────────────────────────────────
    print(f"Generating: seed={args.seed}, rewire_budget={args.rewire_budget} …")
    g_synth = Generator(target_sig).sample(
        seed=args.seed,
        rewire_budget=args.rewire_budget,
        convergence_log=conv_log_path,
        swap_log=swap_log_path,
    )
    print(f"  {g_synth.vcount():,} nodes  {g_synth.ecount():,} edges")
    if conv_log_path is not None:
        print(f"  convergence log → {conv_log_path}")
    if swap_log_path is not None:
        print(f"  swap log → {swap_log_path}")
    print(f"  best loss {g_synth['stage3_best_loss']:.6f} reached at accepted swap {g_synth['stage3_best_accepted']}")

    # ── Step 3: save synthetic graph ─────────────────────────────────────────
    save_kg(g_synth, out_path)
    print(f"Saved     : {out_path}")

    # ── Step 4: measure all six blocks for the synthetic graph ───────────────
    print("Measuring : blocks A–F on synthetic graph …")
    # Use the lighter sample budget for Block E's CC motif/star counting (the
    # module-level counter is rebound) and its path/tree walks (sample_budget).
    _block_e.MOTIF_COUNTER = HybridMotifCounter(n_samples=_FINAL_SAMPLE_BUDGET, seed=1)
    sa = BlockA().calculate(g_synth)
    sb = BlockB().calculate(g_synth)
    sc = BlockC().calculate(g_synth)
    sd = BlockD().calculate(g_synth)
    se = BlockE().calculate(g_synth, sample_budget=_FINAL_SAMPLE_BUDGET,
                            skip_stars_and_paths=args.skip_templates)
    sf = BlockF().calculate(g_synth, skip_shortest_paths=True)

    # ── Step 4b: dump the synthetic signature (same layout as measured graphs) ─
    # Write plots, per-block JSON, summary and combined JSON to a
    # 'signature_synth/' dir next to the source graph, mirroring the measured
    # 'signature/' dir so the two are directly comparable / drop-in for readers.
    synth_sig = ReducedGraphSignature(a=sa, b=sb, c=sc, d=sd, e=se, f=sf)
    synth_dir = graph_dir / f"signature_synth_{run_ts}"
    synth_written = write_signature_outputs(
        synth_sig, synth_dir, source=str(out_path)
    )
    print(f"Saved     : synthetic signature ({len(synth_written)} files) → {synth_dir}/")

    # ── Step 5: full reduced-signature comparison ────────────────────────────
    # Every feature of every block (the complete reduced signature vector), via each
    # block's feature_names() × as_vector(), so nothing is summarised away.
    print()
    print(f"  {'Metric':<38}  {'Original':>14}  {'Synthetic':>14}  {'Rel err':>8}")
    print("  " + "─" * 80)

    block_pairs = [
        ("Block A — size & vocabulary", ta, sa),
        ("Block B — relation frequency & multiplicity", tb, sb),
        ("Block C — schema & co-occurrence", tc, sc),
        ("Block D — characteristic sets & two-step", td, sd),
        ("Block E — motifs & templates", te, se),
        ("Block F — connectivity", tf, sf),
    ]
    for title, tblk, sblk in block_pairs:
        print(_header(title))
        for name, tv, sv in zip(tblk.feature_names(), tblk.as_vector(), sblk.as_vector()):
            print(_row(name, tv, sv))

    # ── Aggregate vector error ────────────────────────────────────────────────
    tv_vec = np.array(
        ta.as_vector() + tb.as_vector() + tc.as_vector()
        + td.as_vector() + te.as_vector() + tf.as_vector(), dtype=float)
    sv_vec = np.array(
        sa.as_vector() + sb.as_vector() + sc.as_vector()
        + sd.as_vector() + se.as_vector() + sf.as_vector(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(tv_vec - sv_vec) / np.maximum(np.abs(tv_vec), 1e-9)

    print()
    print(f"  Vector length    : {len(tv_vec)}")
    print(f"  Mean  rel error  : {np.nanmean(rel_err):.3f}")
    print(f"  Median rel error : {np.nanmedian(rel_err):.3f}")
    print(f"  Max   rel error  : {np.nanmax(rel_err):.3f}")
    print()

    # ── Distribution Wasserstein-1 ────────────────────────────────────────────
    # Per reported distribution, the W1 distance between the target and synthetic
    # fits (measures distribution mismatch directly, unlike the per-parameter
    # relative error above). Normalised W1 = W1 / target IQR is comparable across
    # distributions on different scales. Blocks B/C/D expose distribution_fits();
    # blocks without distributional features are skipped.
    print(_header("Distribution Wasserstein-1"))
    print(f"  {'Distribution':<28}  {'W1':>12}  {'W1 / target IQR':>16}")
    print("  " + "─" * 60)
    w1_norms: list[float] = []
    for prefix, tblk, sblk in [("B", tb, sb), ("C", tc, sc), ("D", td, sd)]:
        for (name, tfit, kind), (_, sfit, _) in zip(
            tblk.distribution_fits(), sblk.distribution_fits()
        ):
            w1 = _distance.wasserstein1(tfit, sfit, kind)
            iqr = _distance.reconstructed_iqr(tfit, kind)
            w1_norm = w1 / iqr if (iqr is not None and iqr > 0) else float("nan")
            if not np.isnan(w1_norm):
                w1_norms.append(w1_norm)
            print(f"  {prefix}:{name:<26}  {w1:>12.4f}  {w1_norm:>16.3f}")
    if w1_norms:
        print()
        print(f"  Mean   norm W1   : {np.nanmean(w1_norms):.3f}")
        print(f"  Median norm W1   : {np.nanmedian(w1_norms):.3f}")
    print()


if __name__ == "__main__":
    main()
