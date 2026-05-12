"""Signature round-trip: measure a KG, generate a synthetic one, compare."""

import argparse
import sys
from pathlib import Path

import igraph
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import Generator, Signature
from kg_io import load_kg

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_TYPE_COLOURS = {
    "Person":       "#4e79a7",
    "Paper":        "#f28e2b",
    "Organization": "#59a14f",
    "Venue":        "#e15759",
    "Topic":        "#76b7b2",
    "Class":        "#b07aa1",
}
_SYNTH_ENTITY = "#4e79a7"
_SYNTH_TYPE   = "#f28e2b"


def _short(uri: str) -> str:
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


def _type_colour(name: str) -> str:
    for key, col in _TYPE_COLOURS.items():
        if key.lower() in name.lower():
            return col
    return "#aaaaaa"


def _draw_graph(ax, g: igraph.Graph, title: str, synthetic: bool = False):
    n = g.vcount()
    if n == 0:
        ax.set_title(title)
        ax.axis("off")
        return

    g_und = g.as_undirected()
    layout = g_und.layout("fr", niter=500)
    coords = np.array(layout.coords)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    coords = (coords - lo) / span

    for e in g.es:
        x = [coords[e.source, 0], coords[e.target, 0]]
        y = [coords[e.source, 1], coords[e.target, 1]]
        is_type = e["predicate"] == _RDF_TYPE
        ax.plot(x, y, color="#cccccc" if is_type else "#888888",
                linewidth=0.4 if is_type else 0.6, zorder=1)

    is_lit = g.vs["is_literal"]
    names  = g.vs["name"] if "name" in g.vertex_attributes() else [""] * n

    if synthetic:
        type_targets = {e.target for e in g.es if e["predicate"] == _RDF_TYPE}
        node_colours = [
            (_SYNTH_TYPE if i in type_targets else _SYNTH_ENTITY)
            for i in range(n)
        ]
    else:
        node_colours = [
            ("#dddddd" if is_lit[i] else _type_colour(names[i] or ""))
            for i in range(n)
        ]

    sizes = [max(20, min(300, 20 + g_und.degree(i) * 8)) for i in range(n)]
    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=node_colours,
               edgecolors="#333333", linewidths=0.4, zorder=2)

    deg_threshold = max(4, int(np.percentile([g_und.degree(i) for i in range(n)], 80)))
    for i, name in enumerate(names):
        if name and g_und.degree(i) >= deg_threshold:
            ax.text(coords[i, 0], coords[i, 1] + 0.03, _short(name),
                    fontsize=5, ha="center", va="bottom", zorder=3, color="#222222")

    n_content = sum(1 for e in g.es if e["predicate"] != _RDF_TYPE)
    n_type_e  = g.ecount() - n_content
    predicates = len({e["predicate"] for e in g.es if e["predicate"] != _RDF_TYPE})
    ax.text(0.01, 0.01,
            f"V={n}  E={n_content} content + {n_type_e} rdf:type  R={predicates}",
            transform=ax.transAxes, fontsize=7, va="bottom", color="#444444")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


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
    parser.add_argument("--output", default="graph_comparison.png",
                        help="Output image path (PNG/PDF/SVG).")
    args = parser.parse_args()

    # ── Step 1: measure target signature ────────────────────────────────────
    print(f"Loading  : {args.kg_file}")
    g_orig = load_kg(args.kg_file)
    target = Signature.from_graph(g_orig)

    # ── Step 2: generate synthetic graph ────────────────────────────────────
    print(f"Generating (seed={args.seed}, rewire_budget={args.rewire_budget}) …")
    g_synth = Generator(target).sample(
        seed=args.seed,
        v_noise=args.v_noise,
        e_noise=args.e_noise,
        rewire_budget=args.rewire_budget,
    )

    # ── Step 3: measure synthetic signature ─────────────────────────────────
    synth = Signature.from_graph(g_synth)

    # ── Step 4: print comparison table ──────────────────────────────────────
    rows = [
        ("── Block A ──────────────────────────", "", ""),
        ("  num_entities",        target.a.num_entities,              synth.a.num_entities),
        ("  num_triples",         target.a.num_triples,               synth.a.num_triples),
        ("  num_relations",       target.a.num_relations,             synth.a.num_relations),
        ("  density",             target.a.density,                   synth.a.density),
        ("  triples_per_entity",  target.a.triples_per_entity,        synth.a.triples_per_entity),
        ("── Block C ──────────────────────────", "", ""),
        ("  num_classes",         target.c.num_classes,               synth.c.num_classes),
        ("  class_size_zipf_exp", target.c.class_size_zipf_exponent,  synth.c.class_size_zipf_exponent),
        ("  subj_sv[0]",          target.c.subj_singular_values[0],   synth.c.subj_singular_values[0]),
        ("  subj_sv[1]",          target.c.subj_singular_values[1],   synth.c.subj_singular_values[1]),
        ("  subj_cooc_density",   target.c.subj_cooc_density,         synth.c.subj_cooc_density),
        ("── Block E ──────────────────────────", "", ""),
        ("  triangle_count",      target.e.triangle_count,            synth.e.triangle_count),
        ("  four_cycle_count",    target.e.four_cycle_count,          synth.e.four_cycle_count),
        ("  five_cycle_count",    target.e.five_cycle_count,          synth.e.five_cycle_count),
        ("  diamond_count",       target.e.diamond_count,             synth.e.diamond_count),
        ("  tailed_triangle",     target.e.tailed_triangle_count,     synth.e.tailed_triangle_count),
        ("  star[2]",             target.e.star_counts.get(2, 0),     synth.e.star_counts.get(2, 0)),
        ("  star[3]",             target.e.star_counts.get(3, 0),     synth.e.star_counts.get(3, 0)),
    ]

    col_w = 36
    print()
    print(f"{'Metric':<{col_w}}  {'Target':>14}  {'Synthetic':>14}  {'Rel err':>8}")
    print("─" * (col_w + 44))
    for label, tv, sv in rows:
        if sv == "":
            print(f"\n{label}")
            continue
        tv_s, sv_s = _fmt(tv), _fmt(sv)
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

    tv_vec = np.array(target.a.as_vector() + target.c.as_vector() + target.e.as_vector(), dtype=float)
    sv_vec = np.array(synth.a.as_vector() + synth.c.as_vector() + synth.e.as_vector(), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(tv_vec - sv_vec) / np.maximum(np.abs(tv_vec), 1e-9)
    print()
    print(f"Vector length   : {len(tv_vec)}")
    print(f"Mean rel error  : {np.nanmean(rel_err):.3f}")
    print(f"Median rel error: {np.nanmedian(rel_err):.3f}")
    print(f"Max  rel error  : {np.nanmax(rel_err):.3f}")

    # ── Step 5: plot both graphs ─────────────────────────────────────────────
    print("\nPlotting …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("KG Signature Round-Trip", fontsize=13, fontweight="bold", y=1.01)

    _draw_graph(axes[0], g_orig,  "Original KG",  synthetic=False)
    _draw_graph(axes[1], g_synth, "Synthetic KG", synthetic=True)

    axes[0].legend(
        handles=[mpatches.Patch(color=c, label=l) for l, c in _TYPE_COLOURS.items()]
                + [mpatches.Patch(color="#dddddd", label="Literal")],
        loc="upper right", fontsize=6, framealpha=0.8, title="Type", title_fontsize=7,
    )
    axes[1].legend(
        handles=[
            mpatches.Patch(color=_SYNTH_ENTITY, label="Entity node"),
            mpatches.Patch(color=_SYNTH_TYPE,   label="Type-class node"),
        ],
        loc="upper right", fontsize=6, framealpha=0.8,
    )

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved     : {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
