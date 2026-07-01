"""Reduced Block B — Relation frequency & per-relation multiplicity (G1/G2/G2b).

Stores the *shape* of each relation's fan-out/fan-in rather than redundant
moments: the spread of per-relation power-law exponents as a quantile function, the
relation-usage frequency as a Zipf exponent, and the CS-size→multiplicity offset
``a`` (G2b) that injects the out-degree-shaping correlation the marginals
discard. Aggregate out/in-degree power-laws are kept as *targets* (a compound
sum the marginals do not pin). ``functionality`` and the multiplicity *scale*
are dropped — both are guaranteed by the stored law and edge conservation.

The unsummarised inputs to the fits (the per-relation exponents and the degree
sequences) are kept on the object so ``visualize`` can overlay each fit on the
data it came from.
"""

from collections import defaultdict

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._utils import MIN_SAMPLES_FOR_FIT, PowerLawStats, _fit_powerlaw
from ._orig_block_b import BlockB as _OrigBlockB
from ._fits import (
    QuantileFit,
    QUANTILE_LEVELS,
    QUANTILE_SUFFIXES,
    ZipfFit,
    fit_quantiles,
    fit_zipf,
    fit_cs_size_offset,
    nan_zipf,
)
from ._plot_helpers import overlay_quantiles
from . import _distance

log = get_logger(__name__)

# Per-relation multiplicity exponents are confined to this range for generation;
# stored as the quantile-function cutoffs (q@0 / q@1; docs/signature_redesign.md G2).
_ALPHA_LO = 1.4
_ALPHA_HI = 3.0


class BlockB(SignatureBlock):
    """Reduced Block B — relation frequency and per-relation multiplicity.

    Usage::

        b = BlockB().calculate(g)
        b.as_vector()                # fixed-length comparison vector
        b.as_dict()                  # named key-value pairs
        b.visualize(mode="text")     # CLI summary
        b.visualize(path="out.png")  # save plot to file
    """

    def __init__(self) -> None:
        self._out_degree_fit = _NOT_CALCULATED          # target
        self._in_degree_fit = _NOT_CALCULATED           # target
        self._relation_zipf = _NOT_CALCULATED           # G1
        self._obj_alpha_q = _NOT_CALCULATED             # G2 object side
        self._subj_alpha_q = _NOT_CALCULATED            # G2 subject side
        self._a_obj = _NOT_CALCULATED                   # G2b object-side offset
        self._a_subj = _NOT_CALCULATED                  # G2b subject-side offset
        # unsummarised data kept for visualization
        self._obj_alphas = _NOT_CALCULATED
        self._subj_alphas = _NOT_CALCULATED
        self._out_degrees = _NOT_CALCULATED
        self._in_degrees = _NOT_CALCULATED

    # ── properties ────────────────────────────────────────────────────────────
    # The quantile / Zipf fits are NamedTuples; the JSON round-trip restores
    # them as plain tuples, so each accessor re-wraps to recover attribute access.

    @property
    def out_degree_fit(self) -> PowerLawStats:
        return self._require("out_degree_fit", self._out_degree_fit)

    @property
    def in_degree_fit(self) -> PowerLawStats:
        return self._require("in_degree_fit", self._in_degree_fit)

    @property
    def relation_zipf(self) -> ZipfFit:
        return ZipfFit(*self._require("relation_zipf", self._relation_zipf))

    @property
    def obj_alpha_q(self) -> QuantileFit:
        return QuantileFit(*self._require("obj_alpha_q", self._obj_alpha_q))

    @property
    def subj_alpha_q(self) -> QuantileFit:
        return QuantileFit(*self._require("subj_alpha_q", self._subj_alpha_q))

    @property
    def a_obj(self) -> float:
        return self._require("a_obj", self._a_obj)

    @property
    def a_subj(self) -> float:
        return self._require("a_subj", self._a_subj)

    # ── core ──────────────────────────────────────────────────────────────────

    def calculate(self, g: igraph.Graph) -> "BlockB":
        """Compute reduced Block B (relation frequency & multiplicity).

        One edge pass builds, per predicate ``r``: the object-multiplicity
        ``m_obj(s,r)`` (distinct objects per subject) and subject-multiplicity
        ``m_subj(o,r)`` (distinct subjects per object), plus each entity's
        characteristic-set size on both sides. From those it derives the
        per-relation power-law exponents, the across-relation quantile function
        of those exponents, the relation-usage Zipf exponent, and the
        CS-size→multiplicity OLS offsets.
        """
        non_lit_vs = g.vs.select(is_literal_eq=False)
        if len(non_lit_vs):
            out_degrees = np.array(g.degree(non_lit_vs, mode="out"), dtype=int)
            in_degrees = np.array(g.degree(non_lit_vs, mode="in"), dtype=int)
        else:
            out_degrees = np.array([], dtype=int)
            in_degrees = np.array([], dtype=int)
        self._out_degrees = out_degrees
        self._in_degrees = in_degrees
        self._out_degree_fit = _fit_powerlaw(out_degrees)
        self._in_degree_fit = _fit_powerlaw(in_degrees)

        is_literal: list[bool] = g.vs["is_literal"] if g.vcount() else []

        # Per-relation (subject→#objects) and (object→#subjects) counts, plus the
        # forward/inverse CS of every entity, in a single edge pass.
        subj_obj_count: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        obj_subj_count: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        cs_of: defaultdict[int, set[str]] = defaultdict(set)
        inv_cs_of: defaultdict[int, set[str]] = defaultdict(set)
        rel_edge_counts: defaultdict[str, int] = defaultdict(int)
        for e in g.es:
            r: str = e["predicate"]
            subj_obj_count[r][e.source] += 1
            cs_of[e.source].add(r)
            rel_edge_counts[r] += 1
            if not is_literal[e.target]:
                obj_subj_count[r][e.target] += 1
                inv_cs_of[e.target].add(r)

        # --- G1: relation-usage frequency (Zipf over per-predicate edge counts) ---
        self._relation_zipf = (
            fit_zipf(np.fromiter(rel_edge_counts.values(), dtype=float, count=len(rel_edge_counts)))
            if rel_edge_counts else nan_zipf()
        )

        # --- G2: per-relation exponents, then quantile function across relations ---
        obj_alphas: list[float] = []
        for subj_map in subj_obj_count.values():
            counts = np.fromiter(subj_map.values(), dtype=int, count=len(subj_map))
            obj_alphas.append(_fit_powerlaw(counts).alpha)
        subj_alphas: list[float] = []
        for obj_map in obj_subj_count.values():
            counts = np.fromiter(obj_map.values(), dtype=int, count=len(obj_map))
            subj_alphas.append(_fit_powerlaw(counts).alpha)

        self._obj_alphas = np.array([a for a in obj_alphas if np.isfinite(a)], dtype=float)
        self._subj_alphas = np.array([a for a in subj_alphas if np.isfinite(a)], dtype=float)
        self._obj_alpha_q = fit_quantiles(self._obj_alphas, lo=_ALPHA_LO, hi=_ALPHA_HI)
        if np.isnan(self._obj_alpha_q.q50):
            log.warning(
                "Block B: obj_alpha quantile fit skipped — only %d finite per-relation "
                "alphas (need ≥ %d); obj_mult_alpha metrics will be NaN. "
                "Most likely cause: most relations have constant object-multiplicity (all 1s), "
                "so per-relation power-law fits are degenerate.",
                self._obj_alphas.size, MIN_SAMPLES_FOR_FIT,
            )
        self._subj_alpha_q = fit_quantiles(self._subj_alphas, lo=_ALPHA_LO, hi=_ALPHA_HI)
        if np.isnan(self._subj_alpha_q.q50):
            log.warning(
                "Block B: subj_alpha quantile fit skipped — only %d finite per-relation "
                "alphas (need ≥ %d); subj_mult_alpha metrics will be NaN.",
                self._subj_alphas.size, MIN_SAMPLES_FOR_FIT,
            )

        # --- G2b: CS-size→multiplicity offset (OLS slope per side) ---
        obj_cs_sizes: list[int] = []
        obj_mults: list[int] = []
        for subj_map in subj_obj_count.values():
            for s, m in subj_map.items():
                obj_cs_sizes.append(len(cs_of[s]))
                obj_mults.append(m)
        self._a_obj = fit_cs_size_offset(obj_cs_sizes, obj_mults)

        subj_cs_sizes: list[int] = []
        subj_mults: list[int] = []
        for obj_map in obj_subj_count.values():
            for o, m in obj_map.items():
                subj_cs_sizes.append(len(inv_cs_of[o]))
                subj_mults.append(m)
        self._a_subj = fit_cs_size_offset(subj_cs_sizes, subj_mults)

        log.info(
            "Block B: rel_zipf=%.3f, obj_alpha(median=%.3f), a_obj=%.3f, a_subj=%.3f",
            self._relation_zipf.exponent, self._obj_alpha_q.q50,
            self._a_obj, self._a_subj,
        )
        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 22-vector for cross-KG comparison.

        Layout: out-degree (alpha, xmin); in-degree (alpha, xmin); relation Zipf
        (exponent, x_min); object-α quantile function (7 levels); subject-α
        quantile function (7 levels); offsets a_obj, a_subj.

        Attributes absent from stale serialized data are emitted as NaN.
        """
        n_q = len(QUANTILE_LEVELS)
        return [
            self._safe_scalar(lambda: self.out_degree_fit.alpha),
            self._safe_scalar(lambda: self.out_degree_fit.xmin),
            self._safe_scalar(lambda: self.in_degree_fit.alpha),
            self._safe_scalar(lambda: self.in_degree_fit.xmin),
            self._safe_scalar(lambda: self.relation_zipf.exponent),
            self._safe_scalar(lambda: self.relation_zipf.x_min),
            *self._safe_iter(lambda: self.obj_alpha_q, n_q),
            *self._safe_iter(lambda: self.subj_alpha_q, n_q),
            self._safe_scalar(lambda: self.a_obj),
            self._safe_scalar(lambda: self.a_subj),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = [
            "out_degree_alpha", "out_degree_xmin",
            "in_degree_alpha", "in_degree_xmin",
            "relation_zipf_exponent", "relation_zipf_xmin",
        ]
        for side in ("obj", "subj"):
            names += [f"{side}_mult_alpha_{suffix}" for suffix in QUANTILE_SUFFIXES]
        names += ["a_obj", "a_subj"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a NaN vector the same length as as_vector()."""
        return [float("nan")] * (6 + 2 * len(QUANTILE_LEVELS) + 2)

    def distribution_fits(self) -> list[tuple[str, object, str]]:
        """Return ``(name, fit, kind)`` for each reportable distribution.

        Used by the roundtrip to compute a Wasserstein-1 distance per
        distribution between this block and a re-measured one.
        """
        return [
            ("out_degree", self.out_degree_fit, _distance.POWERLAW),
            ("in_degree", self.in_degree_fit, _distance.POWERLAW),
            ("relation_freq", self.relation_zipf, _distance.ZIPF),
            ("obj_mult_alpha", self.obj_alpha_q, _distance.QUANTILE),
            ("subj_mult_alpha", self.subj_alpha_q, _distance.QUANTILE),
        ]

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for reduced Block B.

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

    def _visualize_text(self, path: str | None) -> None:
        s = self.obj_alpha_q
        ss = self.subj_alpha_q
        lines = [
            "=== Reduced Block B: Relation Frequency & Multiplicity (G1/G2/G2b) ===",
            f"  out-degree fit : alpha={self.out_degree_fit.alpha:.4f}  xmin={self.out_degree_fit.xmin}",
            f"  in-degree fit  : alpha={self.in_degree_fit.alpha:.4f}  xmin={self.in_degree_fit.xmin}",
            f"  relation Zipf  : exponent={self.relation_zipf.exponent:.4f}  xmin={self.relation_zipf.x_min}",
            f"  obj  mult-alpha quantiles: median={s.q50:.3f} IQR=[{s.q25:.3f},{s.q75:.3f}] cutoffs=[{s.q0:.2f},{s.q100:.2f}]",
            f"  subj mult-alpha quantiles: median={ss.q50:.3f} IQR=[{ss.q25:.3f},{ss.q75:.3f}] cutoffs=[{ss.q0:.2f},{ss.q100:.2f}]",
            f"  CS-size offset : a_obj={self.a_obj:.4f}  a_subj={self.a_subj:.4f}",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            obj_alphas = self._require("_obj_alphas", self._obj_alphas)
            subj_alphas = self._require("_subj_alphas", self._subj_alphas)
            out_degrees = self._require("_out_degrees", self._out_degrees)
            in_degrees = self._require("_in_degrees", self._in_degrees)
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            # Degree targets: raw histogram + fitted power-law (reused from original).
            _OrigBlockB._plot_degree_hist(axes[0, 0], out_degrees, self.out_degree_fit,
                                          "Out-degree distribution (target)", False)
            _OrigBlockB._plot_degree_hist(axes[0, 1], in_degrees, self.in_degree_fit,
                                          "In-degree distribution (target)", False)

            # Per-relation exponents: raw histogram + stored quantile markers.
            ax = axes[1, 0]
            if not overlay_quantiles(ax, obj_alphas, self.obj_alpha_q,
                                     label="per-relation α", color="steelblue"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("object-multiplicity exponent α")
            ax.set_ylabel("count (relations)")
            ax.set_title("Object-multiplicity α (fit: quantiles)")

            ax = axes[1, 1]
            if not overlay_quantiles(ax, subj_alphas, self.subj_alpha_q,
                                     label="per-relation α", color="darkorange"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("subject-multiplicity exponent α")
            ax.set_ylabel("count (relations)")
            ax.set_title("Subject-multiplicity α (fit: quantiles)")

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block B: plot failed: %s", exc, exc_info=True)
            plt.close("all")
