"""Signature round-trip: measure a KG, generate a synthetic one, compare."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import Generator, Signature


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kg_file",
        nargs="?",
        default=str(Path(__file__).parent.parent / "tests/fixtures/academic_kg.ttl"),
        help="Path to the input KG (.ttl or .nt). Defaults to the academic fixture.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rewire-budget", type=int, default=5_000)
    parser.add_argument("--v-noise", type=float, default=0.05)
    parser.add_argument("--e-noise", type=float, default=0.05)
    args = parser.parse_args()

    # ── Step 1: measure target signature ────────────────────────────────────
    print(f"Loading  : {args.kg_file}")
    target = Signature.from_file(args.kg_file)

    # ── Step 2: generate synthetic graph ────────────────────────────────────
    print(f"Generating (seed={args.seed}, rewire_budget={args.rewire_budget}) …")
    gen = Generator(target)
    g_synth = gen.sample(
        seed=args.seed,
        v_noise=args.v_noise,
        e_noise=args.e_noise,
        rewire_budget=args.rewire_budget,
    )

    # ── Step 3: measure synthetic signature ─────────────────────────────────
    synth = Signature.from_graph(g_synth)

    # ── Step 4: side-by-side comparison ─────────────────────────────────────
    rows = [
        # label, target value, synthetic value
        ("── Block A ──────────────────────────", "", ""),
        ("  num_entities",        target.a.num_entities,           synth.a.num_entities),
        ("  num_triples",         target.a.num_triples,            synth.a.num_triples),
        ("  num_relations",       target.a.num_relations,          synth.a.num_relations),
        ("  density",             target.a.density,                synth.a.density),
        ("  triples_per_entity",  target.a.triples_per_entity,     synth.a.triples_per_entity),
        ("── Block C ──────────────────────────", "", ""),
        ("  num_classes",         target.c.num_classes,            synth.c.num_classes),
        ("  class_size_zipf_exp", target.c.class_size_zipf_exponent, synth.c.class_size_zipf_exponent),
        ("  subj_sv[0]",          target.c.subj_singular_values[0], synth.c.subj_singular_values[0]),
        ("  subj_sv[1]",          target.c.subj_singular_values[1], synth.c.subj_singular_values[1]),
        ("  subj_cooc_density",   target.c.subj_cooc_density,      synth.c.subj_cooc_density),
        ("── Block E ──────────────────────────", "", ""),
        ("  triangle_count",      target.e.triangle_count,         synth.e.triangle_count),
        ("  four_cycle_count",    target.e.four_cycle_count,       synth.e.four_cycle_count),
        ("  five_cycle_count",    target.e.five_cycle_count,       synth.e.five_cycle_count),
        ("  diamond_count",       target.e.diamond_count,          synth.e.diamond_count),
        ("  tailed_triangle",     target.e.tailed_triangle_count,  synth.e.tailed_triangle_count),
        ("  star[2]",             target.e.star_counts.get(2, 0),  synth.e.star_counts.get(2, 0)),
        ("  star[3]",             target.e.star_counts.get(3, 0),  synth.e.star_counts.get(3, 0)),
    ]

    col_w = 36
    print()
    print(f"{'Metric':<{col_w}}  {'Target':>14}  {'Synthetic':>14}  {'Rel err':>8}")
    print("─" * (col_w + 44))
    for label, tv, sv in rows:
        if sv == "":
            print(f"\n{label}")
            continue
        tv_s = _fmt(tv)
        sv_s = _fmt(sv)
        try:
            tv_f, sv_f = float(tv), float(sv)
            if np.isnan(tv_f) or np.isnan(sv_f):
                err_s = "  NaN"
            elif abs(tv_f) < 1e-9:
                err_s = "  —" if abs(sv_f) < 1e-9 else ">100%"
            else:
                err_s = f"{abs(tv_f - sv_f) / abs(tv_f) * 100:7.1f}%"
        except (TypeError, ValueError):
            err_s = ""
        print(f"  {label:<{col_w - 2}}  {tv_s:>14}  {sv_s:>14}  {err_s:>8}")

    # ── Step 5: aggregate vector error ──────────────────────────────────────
    tv_vec = np.array(target.a.as_vector() + target.c.as_vector() + target.e.as_vector(), dtype=float)
    sv_vec = np.array(synth.a.as_vector() + synth.c.as_vector() + synth.e.as_vector(), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(tv_vec - sv_vec) / np.maximum(np.abs(tv_vec), 1e-9)

    print()
    print(f"Vector length  : {len(tv_vec)}")
    print(f"Mean rel error : {np.nanmean(rel_err):.3f}")
    print(f"Median rel error: {np.nanmedian(rel_err):.3f}")
    print(f"Max  rel error : {np.nanmax(rel_err):.3f}")


if __name__ == "__main__":
    main()
