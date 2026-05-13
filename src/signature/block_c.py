"""Block C — Schema and relation correlation features."""

from collections import defaultdict
from typing import Any

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ._logging import get_logger
from ._utils import RDF_TYPE, _fit_powerlaw

log = get_logger(__name__)

_TOP_K_SV = 10  # number of singular values to keep

_NOT_CALCULATED = object()


class BlockC:
    """Block C — Schema and relation correlation features of a KG.

    Captures how relations co-occur on subjects and objects (via co-occurrence
    matrices summarised by singular values, density, and row entropy), plus
    type-level statistics derived from rdf:type triples.

    Usage::

        c = BlockC().calculate(g)
        c.as_vector()                      # fixed-length comparison vector
        c.visualize()                      # interactive matplotlib figure
        c.visualize(mode="text")           # CLI summary
        c.visualize(path="out.png")        # save plot to file
    """

    def __init__(self) -> None:
        self._subj_singular_values = _NOT_CALCULATED
        self._subj_cooc_density = _NOT_CALCULATED
        self._subj_row_entropies = _NOT_CALCULATED
        self._obj_singular_values = _NOT_CALCULATED
        self._obj_cooc_density = _NOT_CALCULATED
        self._obj_row_entropies = _NOT_CALCULATED
        self._num_classes = _NOT_CALCULATED
        self._class_size_zipf_exponent = _NOT_CALCULATED
        self._class_sizes = _NOT_CALCULATED
        self._type_relation_conditional = _NOT_CALCULATED

    def _require(self, name: str, value: object) -> Any:
        if value is _NOT_CALCULATED:
            raise RuntimeError(f"Call calculate() before accessing {name}")
        return value

    @property
    def subj_singular_values(self) -> np.ndarray:
        return self._require("subj_singular_values", self._subj_singular_values)

    @property
    def subj_cooc_density(self) -> float:
        return self._require("subj_cooc_density", self._subj_cooc_density)

    @property
    def subj_row_entropies(self) -> np.ndarray:
        return self._require("subj_row_entropies", self._subj_row_entropies)

    @property
    def obj_singular_values(self) -> np.ndarray:
        return self._require("obj_singular_values", self._obj_singular_values)

    @property
    def obj_cooc_density(self) -> float:
        return self._require("obj_cooc_density", self._obj_cooc_density)

    @property
    def obj_row_entropies(self) -> np.ndarray:
        return self._require("obj_row_entropies", self._obj_row_entropies)

    @property
    def num_classes(self) -> int:
        return self._require("num_classes", self._num_classes)

    @property
    def class_size_zipf_exponent(self) -> float:
        return self._require("class_size_zipf_exponent", self._class_size_zipf_exponent)

    @property
    def class_sizes(self) -> dict[str, int]:
        return self._require("class_sizes", self._class_sizes)

    @property
    def type_relation_conditional(self) -> dict[str, dict[str, float]]:
        return self._require("type_relation_conditional", self._type_relation_conditional)

    def calculate(self, g: igraph.Graph) -> "BlockC":
        """Compute Block C (schema and relation correlation) of the graph signature.

        Builds subject-side and object-side relation co-occurrence matrices, summarises
        them by top-10 singular values, density, and per-row entropy, then extracts
        type statistics from rdf:type triples.
        """
        predicates: list[str] = g.es["predicate"] if g.ecount() > 0 else []
        unique_rels: list[str] = sorted(set(predicates))
        rel_idx: dict[str, int] = {r: i for i, r in enumerate(unique_rels)}
        num_relations: int = len(unique_rels)

        subj_to_rels: defaultdict[int, set[int]] = defaultdict(set)
        obj_to_rels: defaultdict[int, set[int]] = defaultdict(set)
        for e in g.es:
            ri = rel_idx[e["predicate"]]
            subj_to_rels[e.source].add(ri)
            obj_to_rels[e.target].add(ri)

        M_subj = self._build_cooc_matrix(subj_to_rels, num_relations)
        M_obj = self._build_cooc_matrix(obj_to_rels, num_relations)

        self._subj_singular_values, self._subj_cooc_density, self._subj_row_entropies = self._cooc_stats(M_subj)
        self._obj_singular_values, self._obj_cooc_density, self._obj_row_entropies = self._cooc_stats(M_obj)

        subj_types: defaultdict[int, set[str]] = defaultdict(set)
        for e in g.es:
            if e["predicate"] == RDF_TYPE:
                type_name: str = g.vs[e.target]["name"]
                subj_types[e.source].add(type_name)

        class_counts: defaultdict[str, int] = defaultdict(int)
        for types in subj_types.values():
            for t in types:
                class_counts[t] += 1

        self._class_sizes = dict(class_counts)
        self._num_classes = len(self._class_sizes)
        self._class_size_zipf_exponent = _fit_powerlaw(
            np.array(list(self._class_sizes.values()), dtype=float)
        ).alpha

        type_rel_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        for subj_vid, types in subj_types.items():
            rels_used: list[str] = [g.es[eid]["predicate"] for eid in g.incident(subj_vid, mode="out")]
            for t in types:
                for r in rels_used:
                    type_rel_counts[t][r] += 1

        type_relation_conditional: dict[str, dict[str, float]] = {}
        for t, rel_counts in type_rel_counts.items():
            total = sum(rel_counts.values())
            type_relation_conditional[t] = {r: cnt / total for r, cnt in rel_counts.items()}
        self._type_relation_conditional = type_relation_conditional

        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 29-vector for cross-KG comparison.

        Layout (in order):
          - subj_singular_values: 10 floats (zero-padded)
          - subj_cooc_density, mean subj row entropy, std subj row entropy → 3 floats
          - obj_singular_values: 10 floats (zero-padded)
          - obj_cooc_density, mean obj row entropy, std obj row entropy → 3 floats
          - num_classes, class_size_zipf_exponent, mean type-relation entropy → 3 floats
        """
        subj_ent = float(np.mean(self.subj_row_entropies)) if self.subj_row_entropies.size else 0.0
        subj_ent_std = float(np.std(self.subj_row_entropies)) if self.subj_row_entropies.size else 0.0
        obj_ent = float(np.mean(self.obj_row_entropies)) if self.obj_row_entropies.size else 0.0
        obj_ent_std = float(np.std(self.obj_row_entropies)) if self.obj_row_entropies.size else 0.0

        type_rel_entropies = []
        for dist in self.type_relation_conditional.values():
            p = np.array(list(dist.values()), dtype=float)
            p = p[p > 0]
            if p.size:
                type_rel_entropies.append(-float(np.sum(p * np.log(p))))
        mean_type_rel_ent = float(np.mean(type_rel_entropies)) if type_rel_entropies else 0.0

        return (
            list(self.subj_singular_values)
            + [self.subj_cooc_density, subj_ent, subj_ent_std]
            + list(self.obj_singular_values)
            + [self.obj_cooc_density, obj_ent, obj_ent_std]
            + [float(self.num_classes), self.class_size_zipf_exponent, mean_type_rel_ent]
        )

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for this block's computed features.

        Args:
            mode: "plot" for a matplotlib figure, "text" for a CLI summary.
            path: if given, write output to this file path instead of
                  displaying interactively (savefig for plot, write for text).
        """
        if mode == "text":
            self._visualize_text(path)
        elif mode == "plot":
            self._visualize_plot(path)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'plot' or 'text'.")

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _short_uri(uri: str) -> str:
        return uri.split("/")[-1].split("#")[-1]

    def _visualize_text(self, path: str | None) -> None:
        lines: list[str] = []
        lines.append("=== Block C: Schema and Relation Correlation ===\n")

        lines.append("--- Subject-side co-occurrence ---")
        lines.append(f"  density:       {self.subj_cooc_density:.4f}")
        lines.append(f"  top SVs:       {', '.join(f'{v:.3f}' for v in self.subj_singular_values if v > 0) or '(none)'}")
        if self.subj_row_entropies.size:
            lines.append(
                f"  row entropy:   mean={np.mean(self.subj_row_entropies):.4f}"
                f"  std={np.std(self.subj_row_entropies):.4f}"
            )

        lines.append("\n--- Object-side co-occurrence ---")
        lines.append(f"  density:       {self.obj_cooc_density:.4f}")
        lines.append(f"  top SVs:       {', '.join(f'{v:.3f}' for v in self.obj_singular_values if v > 0) or '(none)'}")
        if self.obj_row_entropies.size:
            lines.append(
                f"  row entropy:   mean={np.mean(self.obj_row_entropies):.4f}"
                f"  std={np.std(self.obj_row_entropies):.4f}"
            )

        lines.append(f"\n--- Type statistics ({self.num_classes} classes) ---")
        lines.append(f"  class_size_zipf_exponent: {self.class_size_zipf_exponent:.4f}")
        if self.class_sizes:
            top_classes = sorted(self.class_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("  top classes by size:")
            for uri, cnt in top_classes:
                lines.append(f"    {self._short_uri(uri):<40s}  {cnt}")

        if self.type_relation_conditional:
            lines.append("\n--- P(r | type) — top relations per type ---")
            for t_uri, dist in sorted(self.type_relation_conditional.items()):
                top_rels = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
                rel_str = ", ".join(f"{self._short_uri(r)}={p:.2f}" for r, p in top_rels)
                lines.append(f"  {self._short_uri(t_uri):<30s}  {rel_str}")

        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            # top-left: singular values for both sides
            ax = axes[0, 0]
            x = np.arange(1, _TOP_K_SV + 1)
            ax.bar(x - 0.2, self.subj_singular_values, width=0.4, label="subject-side", color="steelblue")
            ax.bar(x + 0.2, self.obj_singular_values,  width=0.4, label="object-side",  color="darkorange", alpha=0.8)
            ax.set_xlabel("rank")
            ax.set_ylabel("singular value")
            ax.set_title("Top singular values of co-occurrence matrices")
            ax.legend()

            # top-right: row entropy distributions
            ax = axes[0, 1]
            if self.subj_row_entropies.size:
                ax.hist(self.subj_row_entropies, bins=20, alpha=0.7, label="subject-side", color="steelblue")
            if self.obj_row_entropies.size:
                ax.hist(self.obj_row_entropies,  bins=20, alpha=0.5, label="object-side",  color="darkorange")
            ax.set_xlabel("row entropy (nats)")
            ax.set_ylabel("count (relations)")
            ax.set_title("Row entropy distribution of co-occurrence matrices")
            ax.legend()

            # bottom-left: class size distribution
            ax = axes[1, 0]
            if self.class_sizes:
                sizes = sorted(self.class_sizes.values(), reverse=True)
                ax.bar(range(1, len(sizes) + 1), sizes, color="mediumseagreen")
                ax.set_xlabel("class rank")
                ax.set_ylabel("number of entities")
                ax.set_title(f"Class size distribution ({self.num_classes} classes)")
            else:
                ax.set_title("Class size distribution (no rdf:type triples)")
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

            # bottom-right: mean P(r|type) entropy per type
            ax = axes[1, 1]
            if self.type_relation_conditional:
                entropies = []
                labels = []
                for t_uri, dist in sorted(self.type_relation_conditional.items()):
                    p = np.array(list(dist.values()), dtype=float)
                    p = p[p > 0]
                    entropies.append(-float(np.sum(p * np.log(p))) if p.size else 0.0)
                    labels.append(self._short_uri(t_uri))
                order = np.argsort(entropies)[::-1]
                top_n = min(20, len(order))
                ax.barh(
                    range(top_n),
                    [entropies[i] for i in order[:top_n]],
                    color="mediumpurple",
                )
                ax.set_yticks(range(top_n))
                ax.set_yticklabels([labels[i] for i in order[:top_n]], fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("H(r | type) (nats)")
                ax.set_title("Per-type relation entropy (top 20)")
            else:
                ax.set_title("Per-type relation entropy (no rdf:type triples)")
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block C: plot failed: %s", exc, exc_info=True)
            plt.close("all")

    @staticmethod
    def _build_cooc_matrix(
        entity_to_rels: dict[int, set[int]],
        num_relations: int,
    ) -> scipy.sparse.csr_matrix:
        """Build a relation co-occurrence count matrix from a mapping entity -> {rel_idx}."""
        rows, cols, data = [], [], []
        for rel_set in entity_to_rels.values():
            rel_list = list(rel_set)
            for ri in rel_list:
                for rj in rel_list:
                    rows.append(ri)
                    cols.append(rj)
                    data.append(1)
        if not rows:
            return scipy.sparse.csr_matrix((num_relations, num_relations), dtype=np.int32)
        return scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_relations, num_relations),
            dtype=np.int32,
        )

    @staticmethod
    def _cooc_stats(
        M: scipy.sparse.csr_matrix,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Return (top-k singular values padded to _TOP_K_SV, density, row entropies)."""
        n_rows, n_cols = M.shape
        total_cells = n_rows * n_cols
        density = M.nnz / total_cells if total_cells > 0 else 0.0

        k = min(_TOP_K_SV, min(n_rows, n_cols) - 1)
        svs = np.zeros(_TOP_K_SV)
        if k > 0 and M.nnz > 0:
            computed = scipy.sparse.linalg.svds(
                M.astype(float), k=k, return_singular_vectors=False
            )
            computed = np.sort(computed)[::-1]
            svs[:len(computed)] = computed

        row_entropies = np.zeros(n_rows)
        for i in range(n_rows):
            row = np.asarray(M.getrow(i).todense(), dtype=float).ravel()
            s = row.sum()
            if s > 0:
                p = row / s
                p = p[p > 0]
                row_entropies[i] = -np.sum(p * np.log(p))

        return svs, density, row_entropies
