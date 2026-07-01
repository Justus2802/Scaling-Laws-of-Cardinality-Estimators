"""Reduced Block C — Schema: co-occurrence & type-relation structure (G3).

Replaces the raw 10 singular values with the **exponential-decay** parameters of
each spectrum (rate λ, scale A), and the moment summaries of row entropy with a
**quantile function**. Adds the ``P(r|t)`` type-relation spectrum as its own exp-decay
curve (the original conflated it with ``M``). Co-occurrence density and per-type
relation entropy are kept as targets — functionals the lossy spectrum does not
pin. The matrix/SVD/row-entropy machinery is reused from the original Block C.

The ``M`` co-occurrence spectrum is **V-normalised** (the singular values are
divided by the entity count, i.e. ``M/V``) so its ``scale`` is size-free; the
normalised entries are the empirical joint ``P(i,j)`` = fraction of entities
using both relations. This mirrors — but is distinct from — the *row*-normalised
``P(r|t)`` spectrum (whose scale is already bounded); see
``docs/notes/signature_size_dependence.md``.

The unsummarised spectra and entropy samples are kept on the object so
``visualize`` can overlay each fit on the data it was computed from.
"""

from collections import defaultdict

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._utils import RDF_TYPE, PowerLawStats, _fit_powerlaw, _nan_power_law_stats
from ._orig_block_c import BlockC as _OrigBlockC, _TOP_K_SV
from ._fits import (
    ExpDecayFit,
    QuantileFit,
    QUANTILE_LEVELS,
    QUANTILE_SUFFIXES,
    fit_exp_decay_rank,
    fit_quantiles,
    nan_exp_decay,
    nan_quantiles,
)
from ._plot_helpers import overlay_exp_decay_rank, overlay_quantiles
from . import _distance

log = get_logger(__name__)


class BlockC(SignatureBlock):
    """Reduced Block C — schema, co-occurrence and type-relation structure.

    Usage::

        c = BlockC().calculate(g)
        c.as_vector()                # fixed-length comparison vector
        c.as_dict()                  # named key-value pairs
        c.visualize(mode="text")     # CLI summary
        c.visualize(path="out.png")  # save plot to file
    """

    def __init__(self) -> None:
        self._class_size_fit = _NOT_CALCULATED
        self._num_classes = _NOT_CALCULATED
        self._subj_cooc_exp = _NOT_CALCULATED
        self._subj_cooc_density = _NOT_CALCULATED
        self._subj_row_entropy_q = _NOT_CALCULATED
        self._obj_cooc_exp = _NOT_CALCULATED
        self._obj_cooc_density = _NOT_CALCULATED
        self._obj_row_entropy_q = _NOT_CALCULATED
        self._type_rel_spectrum_exp = _NOT_CALCULATED
        self._per_type_entropy_exp = _NOT_CALCULATED
        # unsummarised data kept for visualization
        self._subj_singular_values = _NOT_CALCULATED
        self._obj_singular_values = _NOT_CALCULATED
        self._type_rel_singular_values = _NOT_CALCULATED
        self._subj_row_entropies = _NOT_CALCULATED
        self._obj_row_entropies = _NOT_CALCULATED
        self._per_type_entropies = _NOT_CALCULATED

    # ── properties ────────────────────────────────────────────────────────────
    # Exp-decay / quantile fits are NamedTuples re-wrapped on access so they
    # survive the JSON round-trip (which restores plain tuples).

    @property
    def class_size_fit(self) -> PowerLawStats:
        return self._require("class_size_fit", self._class_size_fit)

    @property
    def num_classes(self) -> int:
        return self._require("num_classes", self._num_classes)

    @property
    def subj_cooc_exp(self) -> ExpDecayFit:
        return ExpDecayFit(*self._require("subj_cooc_exp", self._subj_cooc_exp))

    @property
    def subj_cooc_density(self) -> float:
        return self._require("subj_cooc_density", self._subj_cooc_density)

    @property
    def subj_row_entropy_q(self) -> QuantileFit:
        return QuantileFit(*self._require("subj_row_entropy_q", self._subj_row_entropy_q))

    @property
    def obj_cooc_exp(self) -> ExpDecayFit:
        return ExpDecayFit(*self._require("obj_cooc_exp", self._obj_cooc_exp))

    @property
    def obj_cooc_density(self) -> float:
        return self._require("obj_cooc_density", self._obj_cooc_density)

    @property
    def obj_row_entropy_q(self) -> QuantileFit:
        return QuantileFit(*self._require("obj_row_entropy_q", self._obj_row_entropy_q))

    @property
    def type_rel_spectrum_exp(self) -> ExpDecayFit:
        return ExpDecayFit(*self._require("type_rel_spectrum_exp", self._type_rel_spectrum_exp))

    @property
    def per_type_entropy_exp(self) -> ExpDecayFit:
        return ExpDecayFit(*self._require("per_type_entropy_exp", self._per_type_entropy_exp))

    # ── core ──────────────────────────────────────────────────────────────────

    def calculate(self, g: igraph.Graph) -> "BlockC":
        """Compute reduced Block C (schema & co-occurrence).

        Builds subject/object relation co-occurrence matrices and summarises each
        by an exp-decay spectrum + density + quantile-function row entropy; fits the
        class-size power-law; and builds the type→relation matrix to summarise
        ``P(r|t)`` by its own spectrum plus a per-type entropy rank curve.
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

        # Reuse the original matrix builder + SVD/density/row-entropy summary.
        M_subj = _OrigBlockC._build_cooc_matrix(subj_to_rels, num_relations)
        M_obj = _OrigBlockC._build_cooc_matrix(obj_to_rels, num_relations)
        subj_svs, self._subj_cooc_density, subj_ent = _OrigBlockC._cooc_stats(M_subj)
        obj_svs, self._obj_cooc_density, obj_ent = _OrigBlockC._cooc_stats(M_obj)

        # Normalise the spectra by V so the exp-decay `scale` is size-free: M holds
        # raw entity counts (top singular value ∝ V), so we divide by the entity
        # count. SVD is linear, so svds(M/V) == svds(M)/V — scaling the returned
        # singular values is identical to normalising M and avoids copying the
        # sparse matrix. M/V entries are the empirical joint P(i,j) (fraction of
        # entities using both relations i and j). Only `scale` shifts; the decay
        # `rate` (log-rank slope) is invariant to this rescale. Density and row
        # entropy come from the unnormalised M and are already size-free, so they
        # are left as-is. See docs/notes/signature_size_dependence.md.
        num_entities = len(g.vs.select(is_literal_eq=False))
        norm = float(num_entities) if num_entities > 0 else 1.0
        self._subj_singular_values = subj_svs / norm
        self._obj_singular_values = obj_svs / norm
        self._subj_row_entropies = subj_ent
        self._obj_row_entropies = obj_ent
        self._subj_cooc_exp = fit_exp_decay_rank(self._subj_singular_values)
        self._obj_cooc_exp = fit_exp_decay_rank(self._obj_singular_values)
        self._subj_row_entropy_q = fit_quantiles(subj_ent) if subj_ent.size else nan_quantiles()
        self._obj_row_entropy_q = fit_quantiles(obj_ent) if obj_ent.size else nan_quantiles()

        # --- Types ---
        subj_types: defaultdict[int, set[str]] = defaultdict(set)
        for e in g.es:
            if e["predicate"] == RDF_TYPE:
                subj_types[e.source].add(g.vs[e.target]["name"])

        class_counts: defaultdict[str, int] = defaultdict(int)
        for types in subj_types.values():
            for t in types:
                class_counts[t] += 1
        self._num_classes = len(class_counts)
        self._class_size_fit = (
            _fit_powerlaw(np.array(list(class_counts.values()), dtype=float))
            if class_counts else _nan_power_law_stats()
        )

        # --- P(r|t) spectrum + per-type relation entropy ---
        svs, spectrum_exp, per_type_exp, per_type_entropies = self._type_relation_stats(
            g, subj_types, rel_idx, num_relations
        )
        self._type_rel_singular_values = svs
        self._type_rel_spectrum_exp = spectrum_exp
        self._per_type_entropy_exp = per_type_exp
        self._per_type_entropies = per_type_entropies

        log.info(
            "Block C: classes=%d, subj_cooc(rate=%.3f), type_rel(rate=%.3f)",
            self._num_classes, self._subj_cooc_exp.rate, self._type_rel_spectrum_exp.rate,
        )
        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 27-vector for cross-KG comparison.

        Layout: class power-law (alpha, xmin); num_classes; subj co-occurrence
        (rate, scale, density); obj co-occurrence (rate, scale, density); subj
        row-entropy quantile function (7); obj row-entropy quantile function (7);
        P(r|t) spectrum (rate, scale); per-type entropy curve (rate, scale).

        Attributes absent from stale serialized data are emitted as NaN.
        """
        n_q = len(QUANTILE_LEVELS)
        return [
            self._safe_scalar(lambda: self.class_size_fit.alpha),
            self._safe_scalar(lambda: self.class_size_fit.xmin),
            self._safe_scalar(lambda: self.num_classes),
            self._safe_scalar(lambda: self.subj_cooc_exp.rate),
            self._safe_scalar(lambda: self.subj_cooc_exp.scale),
            self._safe_scalar(lambda: self.subj_cooc_density),
            self._safe_scalar(lambda: self.obj_cooc_exp.rate),
            self._safe_scalar(lambda: self.obj_cooc_exp.scale),
            self._safe_scalar(lambda: self.obj_cooc_density),
            *self._safe_iter(lambda: self.subj_row_entropy_q, n_q),
            *self._safe_iter(lambda: self.obj_row_entropy_q, n_q),
            self._safe_scalar(lambda: self.type_rel_spectrum_exp.rate),
            self._safe_scalar(lambda: self.type_rel_spectrum_exp.scale),
            self._safe_scalar(lambda: self.per_type_entropy_exp.rate),
            self._safe_scalar(lambda: self.per_type_entropy_exp.scale),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = [
            "class_size_alpha", "class_size_xmin",
            "num_classes",
            "subj_cooc_rate", "subj_cooc_scale", "subj_cooc_density",
            "obj_cooc_rate", "obj_cooc_scale", "obj_cooc_density",
        ]
        for side in ("subj", "obj"):
            names += [f"{side}_row_entropy_{suffix}" for suffix in QUANTILE_SUFFIXES]
        names += [
            "type_rel_spectrum_rate", "type_rel_spectrum_scale",
            "per_type_entropy_rate", "per_type_entropy_scale",
        ]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a NaN vector the same length as as_vector()."""
        return [float("nan")] * (9 + 2 * len(QUANTILE_LEVELS) + 4)

    def distribution_fits(self) -> list[tuple[str, object, str]]:
        """Return ``(name, fit, kind)`` for each reportable distribution.

        Used by the roundtrip to compute a Wasserstein-1 distance per
        distribution between this block and a re-measured one.
        """
        return [
            ("class_size", self.class_size_fit, _distance.POWERLAW),
            ("subj_cooc_spectrum", self.subj_cooc_exp, _distance.EXP_DECAY),
            ("obj_cooc_spectrum", self.obj_cooc_exp, _distance.EXP_DECAY),
            ("subj_row_entropy", self.subj_row_entropy_q, _distance.QUANTILE),
            ("obj_row_entropy", self.obj_row_entropy_q, _distance.QUANTILE),
            ("type_rel_spectrum", self.type_rel_spectrum_exp, _distance.EXP_DECAY),
            ("per_type_entropy", self.per_type_entropy_exp, _distance.EXP_DECAY),
        ]

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for reduced Block C.

        Args:
            mode: "plot" for a matplotlib figure, "text" for a CLI summary.
            path: write to this file instead of displaying interactively.
        """
        if mode == "text":
            self._visualize_text(path)
        elif mode == "plot":
            self._visualize_plot(path)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'plot' or 'text'.")

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _type_relation_stats(
        g: igraph.Graph,
        subj_types: dict[int, set[str]],
        rel_idx: dict[str, int],
        num_relations: int,
    ) -> tuple[np.ndarray, ExpDecayFit, ExpDecayFit, np.ndarray]:
        """Build the type→relation matrix and summarise P(r|t).

        Returns the raw singular values of the ``T×R`` matrix, their exp-decay
        fit, the exp-decay fit of the per-type relation-entropy rank curve, and
        the raw per-type entropies.
        """
        empty = np.array([], dtype=float)
        if not subj_types or num_relations == 0:
            return empty, nan_exp_decay(), nan_exp_decay(), empty

        types: list[str] = sorted({t for ts in subj_types.values() for t in ts})
        type_idx: dict[str, int] = {t: i for i, t in enumerate(types)}

        # Type×relation usage counts: for each typed subject, the relations it uses.
        counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for subj_vid, ts in subj_types.items():
            rels_used = [g.es[eid]["predicate"] for eid in g.incident(subj_vid, mode="out")]
            for t in ts:
                ti = type_idx[t]
                for r in rels_used:
                    counts[(ti, rel_idx[r])] += 1
        if not counts:
            return empty, nan_exp_decay(), nan_exp_decay(), empty

        rows = np.fromiter((ti for ti, _ in counts), dtype=int, count=len(counts))
        cols = np.fromiter((ri for _, ri in counts), dtype=int, count=len(counts))
        data = np.fromiter(counts.values(), dtype=float, count=len(counts))
        M = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(types), num_relations))

        # P(r|t) = row-normalised type×relation matrix; summarise by its spectrum.
        row_sums = np.asarray(M.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        P = M.multiply(1.0 / row_sums[:, None]).tocsr()
        k = min(_TOP_K_SV, min(P.shape) - 1)
        if k > 0 and P.nnz > 0:
            svs = np.sort(scipy.sparse.linalg.svds(
                P.astype(float), k=k, return_singular_vectors=False))[::-1]
        else:
            svs = empty
        spectrum = fit_exp_decay_rank(svs)

        # Per-type relation entropy H(r|t), as a rank curve (top = most diffuse).
        entropies: list[float] = []
        P_lil = P.tolil()
        for i in range(len(types)):
            p = np.asarray(P_lil.getrow(i).todense(), dtype=float).ravel()
            p = p[p > 0]
            if p.size:
                entropies.append(-float(np.sum(p * np.log(p))))
        per_type_entropies = np.array(entropies, dtype=float)
        per_type_exp = fit_exp_decay_rank(per_type_entropies)

        return svs, spectrum, per_type_exp, per_type_entropies

    def _visualize_text(self, path: str | None) -> None:
        s, o = self.subj_row_entropy_q, self.obj_row_entropy_q
        lines = [
            "=== Reduced Block C: Schema & Co-occurrence (G3) ===",
            f"  num_classes        : {self.num_classes}",
            f"  class size power-law: alpha={self.class_size_fit.alpha:.4f}  xmin={self.class_size_fit.xmin}",
            f"  subj co-occurrence : exp(rate={self.subj_cooc_exp.rate:.3f}, scale={self.subj_cooc_exp.scale:.3f})  density={self.subj_cooc_density:.4f}",
            f"  obj  co-occurrence : exp(rate={self.obj_cooc_exp.rate:.3f}, scale={self.obj_cooc_exp.scale:.3f})  density={self.obj_cooc_density:.4f}",
            f"  subj row entropy   : quantiles(median={s.q50:.3f}, IQR=[{s.q25:.3f},{s.q75:.3f}])",
            f"  obj  row entropy   : quantiles(median={o.q50:.3f}, IQR=[{o.q25:.3f},{o.q75:.3f}])",
            f"  P(r|t) spectrum    : exp(rate={self.type_rel_spectrum_exp.rate:.3f}, scale={self.type_rel_spectrum_exp.scale:.3f})",
            f"  per-type entropy   : exp(rate={self.per_type_entropy_exp.rate:.3f}, scale={self.per_type_entropy_exp.scale:.3f})",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))

            # Row 0: spectra (raw singular values + exp-decay fit).
            for ax, svs, fit, title in [
                (axes[0, 0], self._subj_singular_values, self.subj_cooc_exp, "Subject co-occurrence spectrum"),
                (axes[0, 1], self._obj_singular_values, self.obj_cooc_exp, "Object co-occurrence spectrum"),
                (axes[0, 2], self._type_rel_singular_values, self.type_rel_spectrum_exp, "P(r|t) spectrum"),
            ]:
                drew = overlay_exp_decay_rank(ax, self._require("svs", svs), fit, label="singular values")
                if not drew:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("rank")
                ax.set_ylabel("singular value")
                ax.set_title(title)

            # Row 1: row entropies (quantiles) and per-type entropy (exp-decay).
            for ax, ent, fit, title, color in [
                (axes[1, 0], self._subj_row_entropies, self.subj_row_entropy_q, "Subject row entropy", "steelblue"),
                (axes[1, 1], self._obj_row_entropies, self.obj_row_entropy_q, "Object row entropy", "darkorange"),
            ]:
                drew = overlay_quantiles(ax, self._require("ent", ent), fit, color=color)
                if not drew:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("row entropy (nats)")
                ax.set_ylabel("count (relations)")
                ax.set_title(title)

            ax = axes[1, 2]
            drew = overlay_exp_decay_rank(ax, self._require("per_type", self._per_type_entropies),
                                          self.per_type_entropy_exp, label="H(r|type)")
            if not drew:
                ax.text(0.5, 0.5, "no rdf:type", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("type rank")
            ax.set_ylabel("H(r | type) (nats)")
            ax.set_title("Per-type relation entropy")

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block C: plot failed: %s", exc, exc_info=True)
            plt.close("all")
