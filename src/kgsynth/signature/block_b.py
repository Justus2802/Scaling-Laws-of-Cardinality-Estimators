"""Reduced Block B — Relation frequency & per-relation multiplicity (G1/G2/G2b).

Stores the *shape* of each relation's fan-out/fan-in rather than redundant
moments: the spread of per-relation power-law exponents as a quantile function, the
relation-usage frequency as a Zipf exponent, and the CS-size→multiplicity offset
``a`` (G2b) that injects the out-degree-shaping correlation the marginals
discard. Aggregate out/in-degree power-laws are kept as *targets* (a compound
sum the marginals do not pin). ``functionality`` and the multiplicity *scale*
are dropped — both are guaranteed by the stored law and edge conservation.

The unsummarised inputs to the fits (the per-relation exponents, the degree
sequences and the per-relation edge counts) are kept on the object so
``visualize`` can overlay each fit on the data it came from.
"""

from collections import defaultdict

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from .._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._utils import MIN_SAMPLES_FOR_FIT, RDF_TYPE, PowerLawStats, _fit_powerlaw
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
from ._plot_helpers import overlay_quantiles, overlay_zipf
from . import _distance

log = get_logger(__name__)

# Per-relation multiplicity exponents are confined to this range for generation;
# stored as the quantile-function cutoffs (q@0 / q@1; docs/signature.md §G2).
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
        self._obj_mult_max = _NOT_CALCULATED            # G2 upper bound (object side)
        self._subj_mult_max = _NOT_CALCULATED           # G2 upper bound (subject side)
        self._a_obj = _NOT_CALCULATED                   # G2b object-side offset
        self._a_subj = _NOT_CALCULATED                  # G2b subject-side offset
        self._recip_symmetric_frac = _NOT_CALCULATED    # P(symmetric) per edge-freq bin
        self._recip_symmetric_value = _NOT_CALCULATED   # symmetric-mode reciprocity magnitude
        # high-end degree statistics (explicit targets for hub steering)
        self._out_degree_max = _NOT_CALCULATED
        self._out_degree_p90 = _NOT_CALCULATED
        self._in_degree_max = _NOT_CALCULATED
        self._in_degree_p90 = _NOT_CALCULATED
        # unsummarised data kept for visualization
        self._obj_alphas = _NOT_CALCULATED
        self._subj_alphas = _NOT_CALCULATED
        self._out_degrees = _NOT_CALCULATED
        self._in_degrees = _NOT_CALCULATED
        self._rel_edge_counts = _NOT_CALCULATED
        self._recips = _NOT_CALCULATED                  # per-relation reciprocity values

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
    def obj_mult_max(self) -> float:
        return self._require("obj_mult_max", self._obj_mult_max)

    @property
    def subj_mult_max(self) -> float:
        return self._require("subj_mult_max", self._subj_mult_max)

    @property
    def a_obj(self) -> float:
        return self._require("a_obj", self._a_obj)

    @property
    def a_subj(self) -> float:
        return self._require("a_subj", self._a_subj)

    @property
    def recip_symmetric_frac(self) -> np.ndarray:
        return np.asarray(self._require("recip_symmetric_frac", self._recip_symmetric_frac))

    @property
    def recip_symmetric_value(self) -> float:
        return self._require("recip_symmetric_value", self._recip_symmetric_value)

    @property
    def out_degree_max(self) -> int:
        return self._require("out_degree_max", self._out_degree_max)

    @property
    def out_degree_p90(self) -> float:
        return self._require("out_degree_p90", self._out_degree_p90)

    @property
    def in_degree_max(self) -> int:
        return self._require("in_degree_max", self._in_degree_max)

    @property
    def in_degree_p90(self) -> float:
        return self._require("in_degree_p90", self._in_degree_p90)

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
        self._out_degree_max = int(out_degrees.max()) if out_degrees.size else 0
        self._out_degree_p90 = float(np.percentile(out_degrees, 90)) if out_degrees.size else 0.0
        self._in_degree_max = int(in_degrees.max()) if in_degrees.size else 0
        self._in_degree_p90 = float(np.percentile(in_degrees, 90)) if in_degrees.size else 0.0

        is_literal: list[bool] = g.vs["is_literal"] if g.vcount() else []

        # Per-relation (subject→#objects) and (object→#subjects) counts, plus the
        # forward/inverse CS of every entity, in a single edge pass.
        subj_obj_count: defaultdict[str, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        obj_subj_count: defaultdict[str, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        cs_of: defaultdict[int, set[str]] = defaultdict(set)
        inv_cs_of: defaultdict[int, set[str]] = defaultdict(set)
        rel_edge_counts: defaultdict[str, int] = defaultdict(int)
        # Per-relation directed pairs, over entity–entity content edges only, for
        # the reciprocity feature below.
        rel_pairs: defaultdict[str, set] = defaultdict(set)
        for e in g.es:
            r: str = e["predicate"]
            subj_obj_count[r][e.source] += 1
            cs_of[e.source].add(r)
            rel_edge_counts[r] += 1
            tgt_lit = is_literal[e.target]
            if not tgt_lit:
                obj_subj_count[r][e.target] += 1
                inv_cs_of[e.target].add(r)
            if (r != RDF_TYPE and e.source != e.target
                    and not tgt_lit and not is_literal[e.source]):
                rel_pairs[r].add((e.source, e.target))

        # --- G1: relation-usage frequency (Zipf over per-predicate edge counts) ---
        self._rel_edge_counts = (
            np.fromiter(rel_edge_counts.values(), dtype=float, count=len(rel_edge_counts))
            if rel_edge_counts else np.array([], dtype=float)
        )
        self._relation_zipf = (
            fit_zipf(self._rel_edge_counts) if self._rel_edge_counts.size else nan_zipf()
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

        # Upper bound of each multiplicity law: the largest object-/subject-
        # multiplicity any (subject, relation) / (object, relation) reaches. The
        # per-relation alphas are truncated MLEs over a bounded range, so Stage 2
        # needs that bound to draw from the same bounded law instead of an
        # unbounded one. NaN when no relation has any (in-)edges to count — the
        # same "nothing measured" outcome as the alpha quantiles below.
        self._obj_mult_max = float(
            max((max(m.values()) for m in subj_obj_count.values()), default=float("nan"))
        )
        self._subj_mult_max = float(
            max((max(m.values()) for m in obj_subj_count.values()), default=float("nan"))
        )

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

        # --- Per-relation reciprocity: fraction of a relation's directed pairs whose
        # reverse also exists via the same relation (drives bidirectionality). Nearly
        # BIMODAL across the corpus (relations are symmetric ≈1 or asymmetric ≈0, with
        # ~0% in between — see docs/notes/relation_reciprocity_and_bidirectionality.md),
        # and STRONGLY tied to relation frequency (which relation is symmetric matters
        # for generation, not just how many are). A plain quantile function over
        # relations discards that frequency pairing, so instead we bin relations by
        # cumulative EDGE-fraction rank (reusing QUANTILE_LEVELS as fixed bin edges —
        # weights bins by edge mass, so a few huge relations get their own resolution
        # instead of being diluted by a long tail of tiny ones) and store, per bin, the
        # fraction of relations that are symmetric — a mixing weight, not a curve,
        # because the bimodality collapses the general conditional distribution
        # P(reciprocity | frequency) to a single number per bin. The symmetric-mode
        # magnitude (~0.9, not exactly 1) is stored once as a separate scalar. ---
        rel_recip: dict[str, float] = {
            r: sum(1 for (s, o) in pairs if (o, s) in pairs) / len(pairs)
            for r, pairs in rel_pairs.items() if pairs
        }
        self._recips = np.array(list(rel_recip.values()), dtype=float)
        n_bins = len(QUANTILE_LEVELS) - 1
        frac_symmetric = np.full(n_bins, np.nan)
        symmetric_value = float("nan")
        if rel_recip:
            order = sorted(rel_recip, key=lambda r: -rel_edge_counts[r])
            counts = np.array([rel_edge_counts[r] for r in order], dtype=float)
            cum_frac = np.cumsum(counts) / counts.sum()
            recip_ordered = np.array([rel_recip[r] for r in order], dtype=float)
            bin_idx = np.clip(np.searchsorted(QUANTILE_LEVELS[1:], cum_frac, side="left"),
                              0, n_bins - 1)
            for b in range(n_bins):
                in_bin = recip_ordered[bin_idx == b]
                if in_bin.size:
                    frac_symmetric[b] = float(np.mean(in_bin > 0.5))
            symmetric_mode = self._recips[self._recips > 0.5]
            if symmetric_mode.size:
                symmetric_value = float(symmetric_mode.mean())
        self._recip_symmetric_frac = frac_symmetric
        self._recip_symmetric_value = symmetric_value

        log.info(
            "Block B: rel_zipf=%.3f, obj_alpha(median=%.3f), a_obj=%.3f, a_subj=%.3f",
            self._relation_zipf.exponent, self._obj_alpha_q.q50,
            self._a_obj, self._a_subj,
        )
        log.info(
            "Block B: per-relation reciprocity — frac_symmetric by edge-freq bin=%s, "
            "symmetric_value=%.3f (%d relations)",
            np.round(frac_symmetric, 3).tolist(), symmetric_value, self._recips.size,
        )
        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 35-vector for cross-KG comparison.

        Layout: out-degree (alpha, xmin); in-degree (alpha, xmin); relation Zipf
        (exponent, x_min); object-α quantile function (7 levels); subject-α
        quantile function (7 levels); object/subject multiplicity maxima (2 — the
        upper bounds of the two α laws); offsets a_obj, a_subj; out/in degree
        max/p90 (4); per-relation reciprocity — P(symmetric) per edge-frequency bin
        (6 levels) + the symmetric-mode reciprocity magnitude (1 scalar).

        Attributes absent from stale serialized data are emitted as NaN.
        """
        n_q = len(QUANTILE_LEVELS)
        n_bins = n_q - 1
        return [
            self._safe_scalar(lambda: self.out_degree_fit.alpha),
            self._safe_scalar(lambda: self.out_degree_fit.xmin),
            self._safe_scalar(lambda: self.in_degree_fit.alpha),
            self._safe_scalar(lambda: self.in_degree_fit.xmin),
            self._safe_scalar(lambda: self.relation_zipf.exponent),
            self._safe_scalar(lambda: self.relation_zipf.x_min),
            *self._safe_iter(lambda: self.obj_alpha_q, n_q),
            *self._safe_iter(lambda: self.subj_alpha_q, n_q),
            self._safe_scalar(lambda: self.obj_mult_max),
            self._safe_scalar(lambda: self.subj_mult_max),
            self._safe_scalar(lambda: self.a_obj),
            self._safe_scalar(lambda: self.a_subj),
            self._safe_scalar(lambda: self.out_degree_max),
            self._safe_scalar(lambda: self.out_degree_p90),
            self._safe_scalar(lambda: self.in_degree_max),
            self._safe_scalar(lambda: self.in_degree_p90),
            *self._safe_iter(lambda: self.recip_symmetric_frac, n_bins),
            self._safe_scalar(lambda: self.recip_symmetric_value),
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
        names += ["obj_mult_max", "subj_mult_max"]
        names += ["a_obj", "a_subj"]
        names += ["out_degree_max", "out_degree_p90", "in_degree_max", "in_degree_p90"]
        names += [f"recip_symmetric_frac_bin{i}" for i in range(len(QUANTILE_LEVELS) - 1)]
        names += ["recip_symmetric_value"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a NaN vector the same length as as_vector()."""
        n_q = len(QUANTILE_LEVELS)
        return [float("nan")] * (6 + 2 * n_q + 2 + 2 + 4 + (n_q - 1) + 1)

    @classmethod
    def _state_from_features(cls, feats: dict[str, float]) -> dict:
        """Rebuild Block B's state from the flat feature dict.

        ``PowerLawStats``'s ``ks`` / ``D_*`` fit-quality diagnostics are not in
        the feature vector and no consumer reads them, so they are filled NaN.
        """
        nan = float("nan")
        n_bins = len(QUANTILE_LEVELS) - 1
        return {
            "_out_degree_fit": PowerLawStats(
                feats["out_degree_alpha"], feats["out_degree_xmin"], nan, nan, nan, nan
            ),
            "_in_degree_fit": PowerLawStats(
                feats["in_degree_alpha"], feats["in_degree_xmin"], nan, nan, nan, nan
            ),
            "_relation_zipf": ZipfFit(
                feats["relation_zipf_exponent"], feats["relation_zipf_xmin"]
            ),
            "_obj_alpha_q": QuantileFit(
                *[feats[f"obj_mult_alpha_{s}"] for s in QUANTILE_SUFFIXES]
            ),
            "_subj_alpha_q": QuantileFit(
                *[feats[f"subj_mult_alpha_{s}"] for s in QUANTILE_SUFFIXES]
            ),
            "_obj_mult_max": cls._int(feats, "obj_mult_max"),
            "_subj_mult_max": cls._int(feats, "subj_mult_max"),
            "_a_obj": feats["a_obj"],
            "_a_subj": feats["a_subj"],
            "_out_degree_max": cls._int(feats, "out_degree_max"),
            "_out_degree_p90": feats["out_degree_p90"],
            "_in_degree_max": cls._int(feats, "in_degree_max"),
            "_in_degree_p90": feats["in_degree_p90"],
            "_recip_symmetric_frac": np.array(
                [feats[f"recip_symmetric_frac_bin{i}"] for i in range(n_bins)], dtype=float
            ),
            "_recip_symmetric_value": feats["recip_symmetric_value"],
        }

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
            f"  out-degree fit : alpha={self.out_degree_fit.alpha:.4f}  "
            f"xmin={self.out_degree_fit.xmin}",
            f"  in-degree fit  : alpha={self.in_degree_fit.alpha:.4f}  "
            f"xmin={self.in_degree_fit.xmin}",
            f"  relation Zipf  : exponent={self.relation_zipf.exponent:.4f}  "
            f"xmin={self.relation_zipf.x_min}",
            f"  obj  mult-alpha quantiles: median={s.q50:.3f} "
            f"IQR=[{s.q25:.3f},{s.q75:.3f}] cutoffs=[{s.q0:.2f},{s.q100:.2f}]",
            f"  subj mult-alpha quantiles: median={ss.q50:.3f} "
            f"IQR=[{ss.q25:.3f},{ss.q75:.3f}] cutoffs=[{ss.q0:.2f},{ss.q100:.2f}]",
            f"  mult maxima    : obj={self.obj_mult_max:.0f}  subj={self.subj_mult_max:.0f}",
            f"  CS-size offset : a_obj={self.a_obj:.4f}  a_subj={self.a_subj:.4f}",
            f"  out-degree     : max={self.out_degree_max}  p90={self.out_degree_p90:.1f}",
            f"  in-degree      : max={self.in_degree_max}  p90={self.in_degree_p90:.1f}",
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
            fig, axes = plt.subplots(2, 3, figsize=(18, 9))

            # Degree targets: raw histogram + fitted power-law.
            self._plot_degree_hist(axes[0, 0], out_degrees, self.out_degree_fit,
                                   "Out-degree distribution (target)", False)
            self._plot_degree_hist(axes[0, 1], in_degrees, self.in_degree_fit,
                                   "In-degree distribution (target)", False)

            # Relation-usage frequency: raw per-predicate edge counts + Zipf tail.
            ax = axes[0, 2]
            if self._rel_edge_counts is _NOT_CALCULATED:
                ax.text(0.5, 0.5, "not in serialized data\n(re-run measurement)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8)
            elif not overlay_zipf(ax, self._rel_edge_counts, self.relation_zipf,
                                  label="relation usage", color="teal"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("edge count per relation")
            ax.set_ylabel("P(X ≥ x)")
            ax.set_title("Relation-usage frequency (fit: Zipf, CCDF)")

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

            axes[1, 2].axis("off")  # spare cell in the 2×3 grid

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block B: plot failed: %s", exc, exc_info=True)
            plt.close("all")

    @staticmethod
    def _plot_degree_hist(ax, degrees: np.ndarray, fit: PowerLawStats, title: str, log_scale: bool,
                          dot_color: str = "C0", line_color: str = "red") -> None:
        """Plot a degree histogram with an overlaid fitted power-law tail."""
        pos = degrees[degrees > 0]
        if pos.size == 0:
            ax.set_title(f"{title} (no data)")
            return

        if log_scale:
            bins = np.logspace(np.log10(pos.min()), np.log10(pos.max() + 1), 30)
        else:
            bins = np.linspace(pos.min(), pos.max() + 1, 30)
        counts, edges = np.histogram(pos, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        mask = counts > 0
        bars = ax.bar(centers[mask], counts[mask], width=widths[mask], align="center",
                      color=dot_color, label="data")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        handles = [bars]
        if not np.isnan(fit.alpha) and not np.isnan(fit.xmin):
            xmin = max(int(fit.xmin), 1)
            x_fit = np.arange(xmin, pos.max() + 1, dtype=float)
            y_fit = x_fit ** (-fit.alpha)
            # normalize scale to histogram counts above xmin
            tail_counts, _ = np.histogram(pos[pos >= xmin], bins=bins)
            total = tail_counts.sum()
            if total > 0:
                y_fit = y_fit / y_fit.sum() * total
            line, = ax.plot(x_fit, y_fit, "-", color=line_color, linewidth=1.5,
                            label=f"powerlaw α={fit.alpha:.2f}")
            handles.append(line)

        ax.set_xlabel("degree")
        ax.set_ylabel("count")
        ax.set_title(title)
        # explicit handle order: legend's default ordering puts the line before the
        # bar container, which reads backwards against the data-then-fit plot order.
        ax.legend(handles=handles, fontsize=8)
