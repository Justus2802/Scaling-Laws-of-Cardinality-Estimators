"""Project signature vectors to 2D via PCA and compare original vs. synthetic graphs.

A single signature vector has no meaningful "shape" on its own, so the PCA basis
is fit on the corpus of original-graph signatures (``data/graphs/`` and
``data/test_graphs/``, i.e. the same population used elsewhere for population
fits). Every corpus graph is plotted as a small grey dot to show the spread of
real graphs in signature space. One or more roundtrip pairs (an original graph
and its ``signature_synth`` counterpart, as produced by
``scripts/signature_roundtrip.py``) are then projected into that same space and
drawn as a highlighted, labeled pair connected by an arrow — the arrow length
and direction visualise how far Stage 3 output drifts from its target in
signature space.

Missing features (NaN, e.g. star counts or CS-frequency fit params that some
blocks skip) are mean-imputed per column using the corpus mean before fitting,
so graphs with partially-missing blocks don't break the projection.

``--size-agnostic`` drops every feature whose raw value scales with graph size
(``num_entities``, raw motif/CS counts, raw max/percentile degrees, ...) and
keeps only already-normalized, scale-free descriptors (power-law exponents,
entropies over a fixed vocabulary, densities, ratios, assortativity, ...). Size
so dominates the corpus's variance that an unfiltered PCA mostly separates
graphs by "how big" rather than "what shape" — see
``developer_docs/notes/signature_size_dependence.md`` for the per-feature analysis this
list is based on. Without the flag, all features are used (raw mode).

Usage
-----
    python scripts/plot_signature_pca.py wn18rr_v4
    python scripts/plot_signature_pca.py wn18rr_v4 fb237_v4_ind wn18rr_v4_ind
    python scripts/plot_signature_pca.py wn18rr_v4 --size-agnostic
    python scripts/plot_signature_pca.py wn18rr_v4 --out data/graph_population/pca.png
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from kgsynth.corpus import DEFAULT_SEARCH_DIRS, REPO_ROOT


# Corpus directories scanned for original-graph signatures (PCA basis + grey dots).

# Distinct colours for multiple roundtrip pairs.
_PAIR_COLOURS = ["#C44E52", "#4C72B0", "#55A868", "#8172B3", "#DD8452", "#937860"]

# Features whose raw value scales with graph size (entity/edge/vocabulary
# count), excluded in --size-agnostic mode. Classification per
# developer_docs/notes/signature_size_dependence.md: strictly-size-dependent counts plus
# the "weakly size-dependent" threshold/extremum family (xmin, vmax, max/p90
# degree, shortest-path max/mean), since those still drift with log-scale
# growth rather than being fixed-range ratios or exponents.
_SIZE_DEPENDENT_FEATURES = {
    # Block A — vocabulary size.
    "num_entities", "num_relations",
    # Block B — degree thresholds/extrema (drift with size, not fixed range).
    "out_degree_xmin", "in_degree_xmin", "relation_zipf_xmin",
    "out_degree_max", "out_degree_p90", "in_degree_max", "in_degree_p90",
    # Block C — type vocabulary size and its threshold.
    "num_classes", "class_size_xmin",
    # Block D — characteristic-set counts and frequency extrema.
    "num_distinct_cs", "inv_num_distinct_cs",
    "cs_freq_vmax", "inv_cs_freq_vmax",
    "two_step_vmax",
    # Block E — raw motif/graphlet counts (deliberately unnormalized by design).
    "triangle_count", "four_cycle_count", "five_cycle_count", "six_cycle_count",
    "diamond_count", "k4_count", "tailed_triangle_count",
    # Block F — component count and shortest-path length scale (~log V growth).
    "num_components", "shortest_path_max", "shortest_path_mean", "shortest_path_var",
}


def _load_signature_json(path: Path) -> dict[str, float]:
    """Return the ``features`` dict of a ``signature.json`` file."""
    return json.loads(path.read_text())["features"]


def _find_corpus_signatures() -> dict[str, Path]:
    """Map graph name -> its ``signature/signature.json`` path, across corpus dirs."""
    found: dict[str, Path] = {}
    for corpus_dir in DEFAULT_SEARCH_DIRS:
        if not corpus_dir.is_dir():
            continue
        for graph_dir in sorted(corpus_dir.iterdir()):
            sig_path = graph_dir / "signature" / "signature.json"
            if sig_path.is_file():
                found[graph_dir.name] = sig_path
    return found


def _find_pair(graph_name: str, corpus_signatures: dict[str, Path]) -> tuple[dict, dict]:
    """Load (target_features, synthetic_features) for a graph name.

    The target is the corpus signature; the synthetic counterpart is read from
    the sibling ``signature_synth/signature.json`` written by
    ``signature_roundtrip.py``.
    """
    if graph_name not in corpus_signatures:
        raise SystemExit(
            f"'{graph_name}' has no cached corpus signature. Available: "
            f"{sorted(corpus_signatures)}"
        )
    target_path = corpus_signatures[graph_name]
    synth_path = target_path.parent.parent / "signature_synth" / "signature.json"
    if not synth_path.is_file():
        raise SystemExit(
            f"No synthetic signature for '{graph_name}' at {synth_path}. "
            f"Run: python scripts/signature_roundtrip.py {graph_name}"
        )
    return _load_signature_json(target_path), _load_signature_json(synth_path)


def _build_matrix(
    feature_dicts: list[dict[str, float]], size_agnostic: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Align feature dicts to a shared column order, returning (matrix, feature_names).

    Uses the union of keys across all inputs (some graphs skip star counts /
    CS-frequency fits); missing keys become NaN, later mean-imputed. If
    ``size_agnostic``, columns in ``_SIZE_DEPENDENT_FEATURES`` are dropped.
    """
    names = sorted(set().union(*(d.keys() for d in feature_dicts)))
    if size_agnostic:
        names = [n for n in names if n not in _SIZE_DEPENDENT_FEATURES]
    mat = np.array(
        [[d.get(name, np.nan) for name in names] for d in feature_dicts], dtype=float
    )
    return mat, names


def _fit_pca_2d(
    mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean-impute, z-score, and PCA-project a feature matrix to 2D.

    :param mat: ``(n_samples, n_features)`` raw feature matrix, NaNs allowed.
    :returns: ``(coords_2d, impute, mean, std, components)`` — ``impute`` is the
        per-column value NaNs were filled with (raw scale, for projecting new
        points), ``mean``/``std`` are the standardization stats (post-impute
        scale), and ``components`` has shape ``(2, n_features)``.
    """
    # Columns that are NaN across the whole corpus (e.g. a fit param some block
    # skips everywhere) trigger "mean of empty slice"; nanmean still correctly
    # returns NaN for them, which the next line falls back to 0.0 for.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        impute = np.nanmean(mat, axis=0)
    impute = np.where(np.isnan(impute), 0.0, impute)
    filled = np.where(np.isnan(mat), impute, mat)

    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    standardized = (filled - mean) / std_safe

    # SVD-based PCA: rows of Vt are principal axes, sorted by decreasing variance.
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[:2]
    coords = standardized @ components.T
    return coords, impute, mean, std_safe, components


def _project(feature_dict: dict, names: list[str], impute: np.ndarray, mean: np.ndarray,
             std: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project a single feature dict into the fitted 2D PCA space."""
    vec = np.array([feature_dict.get(name, np.nan) for name in names], dtype=float)
    vec = np.where(np.isnan(vec), impute, vec)
    standardized = (vec - mean) / std
    return standardized @ components.T


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "graphs", nargs="+",
        help="Corpus graph name(s) with a cached signature_synth/ roundtrip result "
             "(e.g. from scripts/signature_roundtrip.py).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output image path (default: data/graph_population/signature_pca.png, or "
             "signature_pca_size_agnostic.png with --size-agnostic)",
    )
    parser.add_argument(
        "--size-agnostic", action="store_true",
        help="Drop features whose raw value scales with graph size (entity/edge/motif "
             "counts, degree extrema, ...) and fit PCA on scale-free structural "
             "descriptors only (exponents, entropies, densities, ratios, ...).",
    )
    args = parser.parse_args()

    corpus_signatures = _find_corpus_signatures()
    if not corpus_signatures:
        roots = [str(d) for d in DEFAULT_SEARCH_DIRS]
        raise SystemExit(f"No corpus signatures found under {roots}")

    print(f"Corpus    : {len(corpus_signatures)} graphs — {sorted(corpus_signatures)}")
    corpus_names = sorted(corpus_signatures)
    corpus_features = [_load_signature_json(corpus_signatures[name]) for name in corpus_names]

    pairs = [_find_pair(name, corpus_signatures) for name in args.graphs]

    # Fit PCA on the corpus only, so the basis reflects real cross-graph variance
    # and isn't distorted by the (possibly still-converging) synthetic points.
    mat, feature_names = _build_matrix(corpus_features, size_agnostic=args.size_agnostic)
    coords, impute, mean, std, components = _fit_pca_2d(mat)

    mode = "size-agnostic (structural only)" if args.size_agnostic else "raw (all features)"
    print(f"Mode      : {mode}")
    print(f"Features  : {len(feature_names)} (union across corpus)")

    fig, ax = plt.subplots(figsize=(9, 7))

    # Corpus cloud.
    ax.scatter(coords[:, 0], coords[:, 1], c="#B0B0B0", s=40, alpha=0.7,
               label="corpus (original graphs)", zorder=2)
    for name, (x, y) in zip(corpus_names, coords):
        ax.annotate(name, (x, y), fontsize=8, color="#808080",
                    xytext=(4, 4), textcoords="offset points")

    # Roundtrip pairs.
    for i, (graph_name, (target_feats, synth_feats)) in enumerate(zip(args.graphs, pairs)):
        colour = _PAIR_COLOURS[i % len(_PAIR_COLOURS)]
        t_xy = _project(target_feats, feature_names, impute, mean, std, components)
        s_xy = _project(synth_feats, feature_names, impute, mean, std, components)

        ax.annotate(
            "", xy=s_xy, xytext=t_xy,
            arrowprops=dict(arrowstyle="->", color=colour, lw=1.8, alpha=0.9),
            zorder=3,
        )
        ax.scatter(*t_xy, c=colour, s=140, marker="o", edgecolor="black",
                   linewidth=1.2, label=f"{graph_name} — original", zorder=4)
        ax.scatter(*s_xy, c=colour, s=140, marker="^", edgecolor="black",
                   linewidth=1.2, label=f"{graph_name} — synthetic", zorder=4)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    title_suffix = " — size-agnostic" if args.size_agnostic else ""
    ax.set_title(
        f"Signature vectors in PCA space: original vs. synthetic (roundtrip){title_suffix}"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    default_name = "signature_pca_size_agnostic.png" if args.size_agnostic else "signature_pca.png"
    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "data" / "graph_population" / default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved     : {out_path}")


if __name__ == "__main__":
    main()
