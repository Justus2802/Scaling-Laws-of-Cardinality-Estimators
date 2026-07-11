"""Reduced Block D — Characteristic sets, inverse CS & two-step targets (G3).

Replaces the original mean/median/p90 CS-size summaries with a **quantile
function** fit, and stores CS-frequency, inverse-CS-frequency and the two-step
pair **path-count** distributions as **truncated power-laws** (free α on the
observed bounded range). Recurrence counts are inherently bounded by the entity
count, and pinning the range removes the free-``xmin`` instability that made
the fits incomparable across graphs (and let α < 2 fits explode the roundtrip
W1 distance). Inverse-CS size is kept as a target (object-side wiring
aggregation, not given by subject multiplicity). The CS / inverse-CS /
path-count computations are done by this block's own ``_compute_cs`` /
``_compute_inv_cs`` / ``_two_step_pair_counts`` helpers.

The unsummarised CS sizes, CS-frequency counts and path counts are kept on the
object so ``visualize`` can overlay each fit on the data it was computed from.
"""

from collections import Counter, defaultdict

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from .._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._fits import (
    QuantileFit,
    QUANTILE_LEVELS,
    QUANTILE_SUFFIXES,
    TruncPowerLawFit,
    fit_quantiles,
    fit_truncated_powerlaw,
    nan_quantiles,
    nan_trunc_powerlaw,
)
from ._plot_helpers import overlay_quantiles, overlay_truncated_powerlaw
from . import _distance

log = get_logger(__name__)

_TOP_K_PAIRS = 20  # top path-count pairs retained for visualization


class BlockD(SignatureBlock):
    """Reduced Block D — characteristic-set, inverse-CS and two-step features.

    Usage::

        d = BlockD().calculate(g)
        d.as_vector()                # fixed-length comparison vector
        d.as_dict()                  # named key-value pairs
        d.visualize(mode="text")     # CLI summary
        d.visualize(path="out.png")  # save plot to file
    """

    def __init__(self) -> None:
        self._num_distinct_cs = _NOT_CALCULATED
        self._cs_freq_fit = _NOT_CALCULATED
        self._cs_size_q = _NOT_CALCULATED
        self._inv_num_distinct_cs = _NOT_CALCULATED
        self._inv_cs_freq_fit = _NOT_CALCULATED
        self._inv_cs_size_q = _NOT_CALCULATED
        self._two_step_fit = _NOT_CALCULATED
        # unsummarised data kept for visualization
        self._cs_sizes = _NOT_CALCULATED
        self._inv_cs_sizes = _NOT_CALCULATED
        self._cs_freq_counts = _NOT_CALCULATED
        self._inv_cs_freq_counts = _NOT_CALCULATED
        self._pair_counts = _NOT_CALCULATED
        self._top_pairs = _NOT_CALCULATED

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def num_distinct_cs(self) -> int:
        return self._require("num_distinct_cs", self._num_distinct_cs)

    @property
    def cs_freq_fit(self) -> TruncPowerLawFit:
        return TruncPowerLawFit(*self._require("cs_freq_fit", self._cs_freq_fit))

    @property
    def cs_size_q(self) -> QuantileFit:
        return QuantileFit(*self._require("cs_size_q", self._cs_size_q))

    @property
    def inv_num_distinct_cs(self) -> int:
        return self._require("inv_num_distinct_cs", self._inv_num_distinct_cs)

    @property
    def inv_cs_freq_fit(self) -> TruncPowerLawFit:
        return TruncPowerLawFit(*self._require("inv_cs_freq_fit", self._inv_cs_freq_fit))

    @property
    def inv_cs_size_q(self) -> QuantileFit:
        return QuantileFit(*self._require("inv_cs_size_q", self._inv_cs_size_q))

    @property
    def two_step_fit(self) -> TruncPowerLawFit:
        return TruncPowerLawFit(*self._require("two_step_fit", self._two_step_fit))

    # ── core ──────────────────────────────────────────────────────────────────

    def calculate(self, g: igraph.Graph) -> "BlockD":
        """Compute reduced Block D (CS, inverse CS, two-step path counts)."""
        cs_of = self._compute_cs(g)
        inv_cs_of = self._compute_inv_cs(g)

        cs_sizes = np.fromiter((len(cs) for cs in cs_of.values()), dtype=float, count=len(cs_of)) \
            if cs_of else np.array([], dtype=float)
        inv_cs_sizes = (
            np.fromiter(
                (len(cs) for cs in inv_cs_of.values()), dtype=float, count=len(inv_cs_of)
            )
            if inv_cs_of
            else np.array([], dtype=float)
        )
        self._cs_sizes = cs_sizes
        self._inv_cs_sizes = inv_cs_sizes

        self._num_distinct_cs = len(set(cs_of.values())) if cs_of else 0
        self._inv_num_distinct_cs = len(set(inv_cs_of.values())) if inv_cs_of else 0
        self._cs_size_q = fit_quantiles(cs_sizes) if cs_sizes.size else nan_quantiles()
        self._inv_cs_size_q = fit_quantiles(inv_cs_sizes) if inv_cs_sizes.size else nan_quantiles()

        # CS frequency: how often each distinct (inverse-)CS recurs → truncated
        # power-law on the observed range (recurrence is bounded by |entities|).
        if cs_of:
            freq = Counter(cs_of.values())
            self._cs_freq_counts = np.fromiter(freq.values(), dtype=float, count=len(freq))
            self._cs_freq_fit = fit_truncated_powerlaw(self._cs_freq_counts)
        else:
            self._cs_freq_counts = np.array([], dtype=float)
            self._cs_freq_fit = nan_trunc_powerlaw()
        if inv_cs_of:
            inv_freq = Counter(inv_cs_of.values())
            self._inv_cs_freq_counts = np.fromiter(
                inv_freq.values(), dtype=float, count=len(inv_freq)
            )
            self._inv_cs_freq_fit = fit_truncated_powerlaw(self._inv_cs_freq_counts)
        else:
            self._inv_cs_freq_counts = np.array([], dtype=float)
            self._inv_cs_freq_fit = nan_trunc_powerlaw()

        # Two-step path counts → truncated power-law over the full value set.
        pair_counts, top_pairs = self._two_step_pair_counts(g)
        self._pair_counts = pair_counts
        self._top_pairs = top_pairs
        self._two_step_fit = (
            fit_truncated_powerlaw(pair_counts) if pair_counts.size else nan_trunc_powerlaw()
        )

        log.info(
            "Block D: num_distinct_cs=%d, cs_freq(alpha=%.3f), inv_num_distinct_cs=%d, "
            "inv_cs_freq(alpha=%.3f), two_step(alpha=%.3f)",
            self._num_distinct_cs, self._cs_freq_fit.alpha, self._inv_num_distinct_cs,
            self._inv_cs_freq_fit.alpha, self._two_step_fit.alpha,
        )
        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 25-vector for cross-KG comparison.

        Layout (forward then inverse, symmetric): num_distinct_cs; CS-frequency
        truncated power-law (alpha, v_min, v_max); CS-size quantile function (7);
        inv_num_distinct_cs; inverse-CS-frequency truncated power-law (alpha,
        v_min, v_max); inverse-CS-size quantile function (7); two-step truncated
        power-law (alpha, v_min, v_max).

        Attributes absent from stale serialized data are emitted as NaN.
        """
        n_q = len(QUANTILE_LEVELS)
        return [
            self._safe_scalar(lambda: self.num_distinct_cs),
            *self._safe_iter(lambda: self.cs_freq_fit, 3),
            *self._safe_iter(lambda: self.cs_size_q, n_q),
            self._safe_scalar(lambda: self.inv_num_distinct_cs),
            *self._safe_iter(lambda: self.inv_cs_freq_fit, 3),
            *self._safe_iter(lambda: self.inv_cs_size_q, n_q),
            *self._safe_iter(lambda: self.two_step_fit, 3),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = ["num_distinct_cs", "cs_freq_alpha", "cs_freq_vmin", "cs_freq_vmax"]
        names += [f"cs_size_{s}" for s in QUANTILE_SUFFIXES]
        names += ["inv_num_distinct_cs", "inv_cs_freq_alpha", "inv_cs_freq_vmin",
                  "inv_cs_freq_vmax"]
        names += [f"inv_cs_size_{s}" for s in QUANTILE_SUFFIXES]
        names += ["two_step_alpha", "two_step_vmin", "two_step_vmax"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a NaN vector the same length as as_vector()."""
        return [float("nan")] * (4 + len(QUANTILE_LEVELS) + 4 + len(QUANTILE_LEVELS) + 3)

    def distribution_fits(self) -> list[tuple[str, object, str]]:
        """Return ``(name, fit, kind)`` for each reportable distribution.

        Used by the roundtrip to compute a Wasserstein-1 distance per
        distribution between this block and a re-measured one.
        """
        return [
            ("cs_freq", self.cs_freq_fit, _distance.TRUNC_POWERLAW),
            ("cs_size", self.cs_size_q, _distance.QUANTILE),
            ("inv_cs_freq", self.inv_cs_freq_fit, _distance.TRUNC_POWERLAW),
            ("inv_cs_size", self.inv_cs_size_q, _distance.QUANTILE),
            ("two_step", self.two_step_fit, _distance.TRUNC_POWERLAW),
        ]

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for reduced Block D.

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
    def _compute_cs(g: igraph.Graph) -> dict[int, frozenset[str]]:
        """Single g.es pass → cs_of[v_idx] = frozenset of outgoing predicates."""
        cs_of: defaultdict[int, set[str]] = defaultdict(set)
        for e in g.es:
            cs_of[e.source].add(e["predicate"])
        return {v: frozenset(preds) for v, preds in cs_of.items()}

    @staticmethod
    def _compute_inv_cs(g: igraph.Graph) -> dict[int, frozenset[str]]:
        """Single g.es pass → inv_cs_of[v_idx] = frozenset of incoming predicates
        (non-literals only)."""
        inv_cs_of: defaultdict[int, set[str]] = defaultdict(set)
        is_literal: list[bool] = g.vs["is_literal"]
        for e in g.es:
            if not is_literal[e.target]:
                inv_cs_of[e.target].add(e["predicate"])
        return {v: frozenset(preds) for v, preds in inv_cs_of.items()}

    @staticmethod
    def _two_step_pair_counts(g: igraph.Graph) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
        """Return the full path-count value set and the top pairs (for viz).

        Path count per ``(in_pred q, out_pred p)`` is
        ``Σ_x deg_in(x,q)·deg_out(x,p)`` — the multiplicity-weighted 2-hop count.
        Returns the *complete* count array (so the truncated power-law sees
        every value), not just the top-k.
        """
        out_deg: defaultdict[int, Counter] = defaultdict(Counter)
        in_deg: defaultdict[int, Counter] = defaultdict(Counter)
        is_literal: list[bool] = g.vs["is_literal"] if g.vcount() else []
        for e in g.es:
            out_deg[e.source][e["predicate"]] += 1
            if not is_literal[e.target]:
                in_deg[e.target][e["predicate"]] += 1

        pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for v in set(out_deg) & set(in_deg):
            in_c, out_c = in_deg[v], out_deg[v]
            for q, cq in in_c.items():
                for p, cp in out_c.items():
                    pair_counts[(q, p)] += cq * cp

        if not pair_counts:
            return np.array([], dtype=float), []
        ordered = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
        counts = np.fromiter((c for _, c in ordered), dtype=float, count=len(ordered))
        top_pairs = [(q, p, c) for (q, p), c in ordered[:_TOP_K_PAIRS]]
        return counts, top_pairs

    def _visualize_text(self, path: str | None) -> None:
        cs, inv = self.cs_size_q, self.inv_cs_size_q
        cf, icf = self.cs_freq_fit, self.inv_cs_freq_fit
        ts = self.two_step_fit
        lines = [
            "=== Reduced Block D: Characteristic Sets & Two-step (G3) ===",
            f"  num_distinct_cs    : {self.num_distinct_cs}",
            f"  CS frequency       : trunc. power-law(alpha={cf.alpha:.4f}, "
            f"range=[{cf.v_min:.0f},{cf.v_max:.0f}])",
            f"  CS size            : quantiles(median={cs.q50:.1f}, "
            f"IQR=[{cs.q25:.1f},{cs.q75:.1f}], range=[{cs.q0:.1f},{cs.q100:.1f}])",
            f"  inv_num_distinct_cs: {self.inv_num_distinct_cs}",
            f"  inverse-CS freq    : trunc. power-law(alpha={icf.alpha:.4f}, "
            f"range=[{icf.v_min:.0f},{icf.v_max:.0f}])",
            f"  inverse-CS size    : quantiles(median={inv.q50:.1f}, "
            f"IQR=[{inv.q25:.1f},{inv.q75:.1f}], range=[{inv.q0:.1f},{inv.q100:.1f}])",
            f"  two-step path count: trunc. power-law(alpha={ts.alpha:.4f}, "
            f"range=[{ts.v_min:.0f},{ts.v_max:.0f}])",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            cs_sizes = self._require("_cs_sizes", self._cs_sizes)
            inv_cs_sizes = self._require("_inv_cs_sizes", self._inv_cs_sizes)
            pair_counts = self._require("_pair_counts", self._pair_counts)
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))

            ax = axes[0, 0]
            if not overlay_quantiles(ax, cs_sizes, self.cs_size_q, label="|CS|", color="steelblue"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("|CS| (predicate count)")
            ax.set_ylabel("entity count")
            ax.set_title("Forward CS size (fit: quantiles)")

            ax = axes[0, 1]
            if not overlay_quantiles(ax, inv_cs_sizes, self.inv_cs_size_q,
                                     label="inverse |CS|", color="darkorange"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("inverse |CS| (in-predicate count)")
            ax.set_ylabel("entity count")
            ax.set_title("Inverse CS size (fit: quantiles)")

            ax = axes[0, 2]
            if not overlay_truncated_powerlaw(
                ax, pair_counts, self.two_step_fit, label="path counts"
            ):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("path count per (q, p)")
            ax.set_ylabel("P(X ≥ x)")
            ax.set_title("Two-step path counts (fit: trunc. power-law, CCDF)")

            _missing = "not in serialized data\n(re-run measurement)"
            ax = axes[1, 0]
            if self._cs_freq_counts is _NOT_CALCULATED:
                ax.text(0.5, 0.5, _missing, ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            elif not overlay_truncated_powerlaw(ax, self._cs_freq_counts, self.cs_freq_fit,
                                                label="CS freq"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("recurrence count per distinct CS")
            ax.set_ylabel("P(X ≥ x)")
            ax.set_title("Forward CS frequency (fit: trunc. power-law, CCDF)")

            ax = axes[1, 1]
            if self._inv_cs_freq_counts is _NOT_CALCULATED:
                ax.text(0.5, 0.5, _missing, ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            elif not overlay_truncated_powerlaw(ax, self._inv_cs_freq_counts,
                                                self.inv_cs_freq_fit, label="inverse CS freq"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("recurrence count per distinct inverse CS")
            ax.set_ylabel("P(X ≥ x)")
            ax.set_title("Inverse CS frequency (fit: trunc. power-law, CCDF)")

            axes[1, 2].axis("off")  # spare cell in the 2×3 grid

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block D: plot failed: %s", exc, exc_info=True)
            plt.close("all")
