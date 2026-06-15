"""Signature round-trip: measure a KG, generate a synthetic one, save it, compare.

Usage
-----
    python scripts/signature_roundtrip.py path/to/graph.ttl
    python scripts/signature_roundtrip.py path/to/graph.ttl --out synth.ttl --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import Generator, Signature
from kg_io import load_kg, save_kg
from signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF


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
    parser.add_argument("kg_file", help="Input KG file (.ttl or .nt)")
    parser.add_argument(
        "--out", default=None,
        help="Where to save the synthetic graph (.ttl or .nt). "
             "Default: <input_stem>_synth.ttl next to the input file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rewire-budget", type=int, default=5_000)
    args = parser.parse_args()

    in_path = Path(args.kg_file)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_synth.ttl")

    # ── Step 1: measure all six blocks for the original graph ────────────────
    print(f"Loading   : {in_path}")
    g_orig = load_kg(in_path)
    print(f"  {g_orig.vcount():,} nodes  {g_orig.ecount():,} edges")

    print("Measuring : blocks A–F on original graph …")
    ta = BlockA().calculate(g_orig)
    tb = BlockB().calculate(g_orig)
    tc = BlockC().calculate(g_orig)
    td = BlockD().calculate(g_orig)
    te = BlockE().calculate(g_orig)
    tf = BlockF().calculate(g_orig)

    # ── Step 2: generate synthetic graph ─────────────────────────────────────
    print(f"Generating: seed={args.seed}, rewire_budget={args.rewire_budget} …")
    target_sig = Signature(a=ta, b=tb, c=tc, d=td, e=te, f=tf)
    g_synth = Generator(target_sig).sample(
        seed=args.seed,
        rewire_budget=args.rewire_budget,
    )
    print(f"  {g_synth.vcount():,} nodes  {g_synth.ecount():,} edges")

    # ── Step 3: save synthetic graph ─────────────────────────────────────────
    save_kg(g_synth, out_path)
    print(f"Saved     : {out_path}")

    # ── Step 4: measure all six blocks for the synthetic graph ───────────────
    print("Measuring : blocks A–F on synthetic graph …")
    sa = BlockA().calculate(g_synth)
    sb = BlockB().calculate(g_synth)
    sc = BlockC().calculate(g_synth)
    sd = BlockD().calculate(g_synth)
    se = BlockE().calculate(g_synth)
    sf = BlockF().calculate(g_synth)

    # ── Step 5: comparison table ─────────────────────────────────────────────
    print()
    print(f"  {'Metric':<38}  {'Original':>14}  {'Synthetic':>14}  {'Rel err':>8}")
    print("  " + "─" * 80)

    print(_header("Block A — size & density"))
    print(_row("num_entities",       ta.num_entities,       sa.num_entities))
    print(_row("num_triples",        ta.num_triples,        sa.num_triples))
    print(_row("num_relations",      ta.num_relations,      sa.num_relations))
    print(_row("density",            ta.density,            sa.density))
    print(_row("triples_per_entity", ta.triples_per_entity, sa.triples_per_entity))
    print(_row("relation_reuse",     ta.relation_reuse,     sa.relation_reuse))

    print(_header("Block B — degree structure"))
    print(_row("out_degree_fit.alpha",     tb.out_degree_fit.alpha,       sb.out_degree_fit.alpha))
    print(_row("out_degree_fit.xmin",      tb.out_degree_fit.xmin,        sb.out_degree_fit.xmin))
    print(_row("in_degree_fit.alpha",      tb.in_degree_fit.alpha,        sb.in_degree_fit.alpha))
    print(_row("in_degree_fit.xmin",       tb.in_degree_fit.xmin,         sb.in_degree_fit.xmin))
    t_func  = list(tb.functionality.values())
    s_func  = list(sb.functionality.values())
    t_ifunc = list(tb.inverse_functionality.values())
    s_ifunc = list(sb.inverse_functionality.values())
    print(_row("functionality (mean)",     np.mean(t_func)  if t_func  else float("nan"),
                                           np.mean(s_func)  if s_func  else float("nan")))
    print(_row("inv_functionality (mean)", np.mean(t_ifunc) if t_ifunc else float("nan"),
                                           np.mean(s_ifunc) if s_ifunc else float("nan")))

    print(_header("Block C — schema & co-occurrence"))
    print(_row("num_classes",          tc.num_classes,              sc.num_classes))
    print(_row("class_size_zipf_exp",  tc.class_size_zipf_exponent, sc.class_size_zipf_exponent))
    print(_row("subj_cooc_density",    tc.subj_cooc_density,        sc.subj_cooc_density))
    print(_row("obj_cooc_density",     tc.obj_cooc_density,         sc.obj_cooc_density))
    for i in range(min(5, len(tc.subj_singular_values))):
        print(_row(f"subj_sv[{i}]", tc.subj_singular_values[i], sc.subj_singular_values[i]))

    print(_header("Block D — characteristic sets"))
    print(_row("num_distinct_cs",     td.num_distinct_cs,         sd.num_distinct_cs))
    print(_row("cs_size_mean",        td.cs_size_mean,            sd.cs_size_mean))
    print(_row("cs_size_median",      td.cs_size_median,          sd.cs_size_median))
    print(_row("inv_num_distinct_cs", td.inv_num_distinct_cs,     sd.inv_num_distinct_cs))
    print(_row("inv_cs_size_mean",    td.inv_cs_size_mean,        sd.inv_cs_size_mean))

    print(_header("Block E — motifs & structural patterns"))
    print(_row("triangle_count",      te.triangle_count,        se.triangle_count))
    print(_row("four_cycle_count",    te.four_cycle_count,      se.four_cycle_count))
    print(_row("five_cycle_count",    te.five_cycle_count,      se.five_cycle_count))
    print(_row("six_cycle_count",     te.six_cycle_count,       se.six_cycle_count))
    print(_row("diamond_count",       te.diamond_count,         se.diamond_count))
    print(_row("k4_count",            te.k4_count,              se.k4_count))
    print(_row("tailed_triangle",     te.tailed_triangle_count, se.tailed_triangle_count))
    for k in range(2, 11):
        print(_row(f"star[{k}]", te.star_counts.get(k, 0), se.star_counts.get(k, 0)))
    print(_row("tree_template_zipf",    te.tree_template_zipf,    se.tree_template_zipf))
    print(_row("tree_template_entropy", te.tree_template_entropy, se.tree_template_entropy))

    print(_header("Block F — connectivity"))
    print(_row("num_components",             tf.num_components,              sf.num_components))
    print(_row("largest_component_fraction", tf.largest_component_fraction,  sf.largest_component_fraction))
    print(_row("avg_shortest_path_length",   tf.avg_shortest_path_length,    sf.avg_shortest_path_length))
    print(_row("clustering_coefficient",     tf.clustering_coefficient,      sf.clustering_coefficient))
    print(_row("degree_assortativity",       tf.degree_assortativity,        sf.degree_assortativity))

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


if __name__ == "__main__":
    main()
