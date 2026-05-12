"""Plot original KG and synthetic graph side by side."""

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

# Colour palette per entity kind (original graph) and generic (synthetic)
_TYPE_COLOURS = {
    "Person":       "#4e79a7",
    "Paper":        "#f28e2b",
    "Organization": "#59a14f",
    "Venue":        "#e15759",
    "Topic":        "#76b7b2",
    "Class":        "#b07aa1",
}
_DEFAULT_COLOUR = "#aaaaaa"
_SYNTH_ENTITY   = "#4e79a7"
_SYNTH_TYPE     = "#f28e2b"


def _short(uri: str) -> str:
    """Return the local name of a URI."""
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


def _type_colour(name: str) -> str:
    for key, col in _TYPE_COLOURS.items():
        if key.lower() in name.lower():
            return col
    return _DEFAULT_COLOUR


def _draw_graph(ax, g: igraph.Graph, title: str, synthetic: bool = False):
    """Draw *g* on *ax* using a Fruchterman-Reingold layout."""
    n = g.vcount()
    if n == 0:
        ax.set_title(title)
        ax.axis("off")
        return

    # Layout (use undirected copy for better spreading)
    g_und = g.as_undirected()
    layout = g_und.layout("fr", niter=500)
    coords = np.array(layout.coords)

    # Normalise to [0, 1]
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    coords = (coords - lo) / span

    # ── Edges ────────────────────────────────────────────────────────────────
    is_type_edge = [e["predicate"] == _RDF_TYPE for e in g.es]
    for i, e in enumerate(g.es):
        x = [coords[e.source, 0], coords[e.target, 0]]
        y = [coords[e.source, 1], coords[e.target, 1]]
        colour = "#cccccc" if is_type_edge[i] else "#888888"
        lw     = 0.4        if is_type_edge[i] else 0.6
        ax.plot(x, y, color=colour, linewidth=lw, zorder=1)

    # ── Vertices ─────────────────────────────────────────────────────────────
    is_lit = g.vs["is_literal"]
    names  = g.vs["name"] if "name" in g.vertex_attributes() else [""] * n

    # Detect type-class nodes (no incoming non-type edge in synthetic graphs)
    if synthetic:
        # Type-class nodes are those that only receive rdf:type edges
        type_targets = {e.target for e in g.es if e["predicate"] == _RDF_TYPE}
        entity_set   = set(range(n)) - type_targets
        node_colours = [
            (_SYNTH_TYPE if i in type_targets else _SYNTH_ENTITY)
            for i in range(n)
        ]
    else:
        node_colours = [
            ("#dddddd" if is_lit[i] else _type_colour(names[i] or ""))
            for i in range(n)
        ]

    sizes = []
    for i in range(n):
        deg = g_und.degree(i)
        sizes.append(max(20, min(300, 20 + deg * 8)))

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes,
        c=node_colours,
        edgecolors="#333333",
        linewidths=0.4,
        zorder=2,
    )

    # Labels for high-degree nodes only (avoid clutter)
    deg_threshold = max(4, int(np.percentile([g_und.degree(i) for i in range(n)], 80)))
    for i, name in enumerate(names):
        if name and g_und.degree(i) >= deg_threshold:
            ax.text(
                coords[i, 0], coords[i, 1] + 0.03,
                _short(name),
                fontsize=5, ha="center", va="bottom", zorder=3,
                color="#222222",
            )

    # ── Stats annotation ─────────────────────────────────────────────────────
    n_content = sum(1 for e in g.es if e["predicate"] != _RDF_TYPE)
    n_type_e  = sum(1 for e in g.es if e["predicate"] == _RDF_TYPE)
    predicates = len({e["predicate"] for e in g.es if e["predicate"] != _RDF_TYPE})
    ax.text(
        0.01, 0.01,
        f"V={n}  E={n_content} content + {n_type_e} rdf:type  R={predicates}",
        transform=ax.transAxes, fontsize=7, va="bottom", color="#444444",
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kg_file",
        nargs="?",
        default=str(Path(__file__).parent.parent / "tests/fixtures/academic_kg.ttl"),
        help="Input KG file (.ttl / .nt). Defaults to the academic fixture.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rewire-budget", type=int, default=5_000)
    parser.add_argument("--output", default="graph_comparison.png",
                        help="Output image path (PNG/PDF/SVG).")
    args = parser.parse_args()

    # ── Load and generate ────────────────────────────────────────────────────
    print(f"Loading   : {args.kg_file}")
    g_orig = load_kg(args.kg_file)

    print("Measuring signature …")
    target = Signature.from_graph(g_orig)

    print(f"Generating (seed={args.seed}, rewire_budget={args.rewire_budget}) …")
    g_synth = Generator(target).sample(seed=args.seed, rewire_budget=args.rewire_budget)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("KG Signature Round-Trip", fontsize=13, fontweight="bold", y=1.01)

    _draw_graph(axes[0], g_orig,  "Original KG",   synthetic=False)
    _draw_graph(axes[1], g_synth, "Synthetic KG",  synthetic=True)

    # Shared legend for original graph
    legend_handles = [
        mpatches.Patch(color=col, label=label)
        for label, col in _TYPE_COLOURS.items()
    ] + [mpatches.Patch(color="#dddddd", label="Literal")]
    axes[0].legend(
        handles=legend_handles, loc="upper right",
        fontsize=6, framealpha=0.8, title="Type", title_fontsize=7,
    )

    # Legend for synthetic graph
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
