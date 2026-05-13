"""Signature round-trip: measure a KG, generate a synthetic one, compare."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import Generator, Signature
from kg_io import load_kg
from signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kg_file",
        nargs="?",
        default=str(Path(__file__).parent.parent / "tests/fixtures/aifb.ttl"),
        help="Path to the input KG (.ttl or .nt). Defaults to aifb.ttl.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rewire-budget", type=int, default=5_000)
    parser.add_argument("--v-noise", type=float, default=0.05)
    parser.add_argument("--e-noise", type=float, default=0.05)
    args = parser.parse_args()

    # ── Step 1: measure all six blocks for the target graph ─────────────────
    print(f"Loading  : {args.kg_file}")
    g_orig = load_kg(args.kg_file)

    print("Measuring target blocks A–F …")
    ta = BlockA().calculate(g_orig)
    tb = BlockB().calculate(g_orig)
    tc = BlockC().calculate(g_orig)
    td = BlockD().calculate(g_orig)
    te = BlockE().calculate(g_orig)
    tf = BlockF().calculate(g_orig)

    # ── Step 2: generate synthetic graph ────────────────────────────────────
    print(f"Generating (seed={args.seed}, rewire_budget={args.rewire_budget}) …")
    target_sig = Signature(a=ta, b=tb, c=tc, d=td, e=te)
    g_synth = Generator(target_sig).sample(
        seed=args.seed,
        v_noise=args.v_noise,
        e_noise=args.e_noise,
        rewire_budget=args.rewire_budget,
    )

    # ── Step 3: measure all six blocks for the synthetic graph ───────────────
    print("Measuring synthetic blocks A–F …")
    sa = BlockA().calculate(g_synth)
    sb = BlockB().calculate(g_synth)
    sc = BlockC().calculate(g_synth)
    sd = BlockD().calculate(g_synth)
    se = BlockE().calculate(g_synth)
    sf = BlockF().calculate(g_synth)

    # ── Step 4: print full signature comparison ─────────────────────────────
    def _row(label, tv, sv):
        tv_s, sv_s = _fmt(tv), _fmt(sv)
        try:
            tv_f, sv_f = float(tv), float(sv)
            if np.isnan(tv_f) or np.isnan(sv_f):
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

    print()
    print(f"  {'Metric':<38}  {'Target':>14}  {'Synthetic':>14}  {'Rel err':>8}")
    print("  " + "─" * 76)

    # Block A
    print(_header("Block A — size & density"))
    print(_row("num_entities",       ta.num_entities,       sa.num_entities))
    print(_row("num_triples",        ta.num_triples,        sa.num_triples))
    print(_row("num_relations",      ta.num_relations,      sa.num_relations))
    print(_row("density",            ta.density,            sa.density))
    print(_row("triples_per_entity", ta.triples_per_entity, sa.triples_per_entity))
    print(_row("relation_reuse",     ta.relation_reuse,     sa.relation_reuse))

    # Block B
    print(_header("Block B — degree structure"))
    print(_row("out_degree_fit.alpha",     tb.out_degree_fit.alpha,       sb.out_degree_fit.alpha))
    print(_row("out_degree_fit.xmin",      tb.out_degree_fit.xmin,        sb.out_degree_fit.xmin))
    print(_row("out_degree_fit.ks",        tb.out_degree_fit.ks,          sb.out_degree_fit.ks))
    print(_row("in_degree_fit.alpha",      tb.in_degree_fit.alpha,        sb.in_degree_fit.alpha))
    print(_row("in_degree_fit.xmin",       tb.in_degree_fit.xmin,         sb.in_degree_fit.xmin))
    print(_row("in_degree_fit.ks",         tb.in_degree_fit.ks,           sb.in_degree_fit.ks))
    t_func  = list(tb.functionality.values())
    s_func  = list(sb.functionality.values())
    t_ifunc = list(tb.inverse_functionality.values())
    s_ifunc = list(sb.inverse_functionality.values())
    print(_row("functionality (mean)",     float(np.mean(t_func))  if t_func  else float("nan"),
                                           float(np.mean(s_func))  if s_func  else float("nan")))
    print(_row("functionality (min)",      float(np.min(t_func))   if t_func  else float("nan"),
                                           float(np.min(s_func))   if s_func  else float("nan")))
    print(_row("functionality (max)",      float(np.max(t_func))   if t_func  else float("nan"),
                                           float(np.max(s_func))   if s_func  else float("nan")))
    print(_row("inv_functionality (mean)", float(np.mean(t_ifunc)) if t_ifunc else float("nan"),
                                           float(np.mean(s_ifunc)) if s_ifunc else float("nan")))
    print(_row("inv_functionality (min)",  float(np.min(t_ifunc))  if t_ifunc else float("nan"),
                                           float(np.min(s_ifunc))  if s_ifunc else float("nan")))
    print(_row("inv_functionality (max)",  float(np.max(t_ifunc))  if t_ifunc else float("nan"),
                                           float(np.max(s_ifunc))  if s_ifunc else float("nan")))

    # Block C
    print(_header("Block C — schema & co-occurrence"))
    print(_row("num_classes",          tc.num_classes,              sc.num_classes))
    print(_row("class_size_zipf_exp",  tc.class_size_zipf_exponent, sc.class_size_zipf_exponent))
    print(_row("subj_cooc_density",    tc.subj_cooc_density,        sc.subj_cooc_density))
    print(_row("obj_cooc_density",     tc.obj_cooc_density,         sc.obj_cooc_density))
    for i in range(len(tc.subj_singular_values)):
        print(_row(f"subj_sv[{i}]", tc.subj_singular_values[i], sc.subj_singular_values[i]))
    for i in range(len(tc.obj_singular_values)):
        print(_row(f"obj_sv[{i}]",  tc.obj_singular_values[i],  sc.obj_singular_values[i]))

    # Block D
    print(_header("Block D — characteristic sets"))
    print(_row("num_distinct_cs",     td.num_distinct_cs,          sd.num_distinct_cs))
    print(_row("cs_freq_stats.alpha", td.cs_freq_stats.alpha,      sd.cs_freq_stats.alpha))
    print(_row("cs_size_mean",        td.cs_size_mean,             sd.cs_size_mean))
    print(_row("cs_size_median",      td.cs_size_median,           sd.cs_size_median))
    print(_row("cs_size_p90",         td.cs_size_p90,              sd.cs_size_p90))
    print(_row("inv_num_distinct_cs", td.inv_num_distinct_cs,      sd.inv_num_distinct_cs))
    print(_row("inv_cs_freq_alpha",   td.inv_cs_freq_stats.alpha,  sd.inv_cs_freq_stats.alpha))
    print(_row("inv_cs_size_mean",    td.inv_cs_size_mean,         sd.inv_cs_size_mean))
    print(_row("inv_cs_size_median",  td.inv_cs_size_median,       sd.inv_cs_size_median))
    print(_row("inv_cs_size_p90",     td.inv_cs_size_p90,          sd.inv_cs_size_p90))
    print(_row("pair_freq_alpha",     td.pair_freq_stats.alpha,    sd.pair_freq_stats.alpha))
    for i, (tv_pf, sv_pf) in enumerate(zip(td.top_pair_freqs, sd.top_pair_freqs)):
        print(_row(f"top_pair_freq[{i}]", float(tv_pf), float(sv_pf)))

    # Block E
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
    for k in range(2, 11):
        print(_row(f"path_zipf[{k}]",
                   te.path_template_zipf.get(k, float("nan")),
                   se.path_template_zipf.get(k, float("nan"))))
    for k in range(2, 11):
        print(_row(f"path_entropy[{k}]",
                   te.path_template_entropy.get(k, float("nan")),
                   se.path_template_entropy.get(k, float("nan"))))
    print(_row("tree_template_zipf",    te.tree_template_zipf,    se.tree_template_zipf))
    print(_row("tree_template_entropy", te.tree_template_entropy, se.tree_template_entropy))

    # Block F
    print(_header("Block F — connectivity"))
    print(_row("num_components",             tf.num_components,              sf.num_components))
    print(_row("largest_component_fraction", tf.largest_component_fraction,  sf.largest_component_fraction))
    print(_row("avg_shortest_path_length",   tf.avg_shortest_path_length,    sf.avg_shortest_path_length))
    print(_row("avg_spl_se",                 tf.avg_shortest_path_length_se, sf.avg_shortest_path_length_se))
    print(_row("clustering_coefficient",     tf.clustering_coefficient,      sf.clustering_coefficient))
    print(_row("degree_assortativity",       tf.degree_assortativity,        sf.degree_assortativity))

    # Aggregate error over full A+B+C+D+E+F vector
    print()
    tv_vec = np.array(
        ta.as_vector() + tb.as_vector() + tc.as_vector()
        + td.as_vector() + te.as_vector() + tf.as_vector(), dtype=float)
    sv_vec = np.array(
        sa.as_vector() + sb.as_vector() + sc.as_vector()
        + sd.as_vector() + se.as_vector() + sf.as_vector(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(tv_vec - sv_vec) / np.maximum(np.abs(tv_vec), 1e-9)
    print(f"  Vector length   : {len(tv_vec)}")
    print(f"  Mean rel error  : {np.nanmean(rel_err):.3f}")
    print(f"  Median rel error: {np.nanmedian(rel_err):.3f}")
    print(f"  Max  rel error  : {np.nanmax(rel_err):.3f}")


if __name__ == "__main__":
    main()
