"""Reduced Block D — Characteristic sets, inverse CS & two-step targets (G3).

Replaces the original mean/median/p90 CS-size summaries with a **skew-normal**
fit, keeps CS-frequency as a power-law, and stores the two-step pair
**path-count** distribution as a **truncated power-law** (free α on a bounded
range). Inverse-CS size is kept as a target (object-side wiring aggregation, not
given by subject multiplicity). The CS / inverse-CS / path-count computations
are reused from the original Block D.

The unsummarised CS sizes and path counts are kept on the object so ``visualize``
can overlay each fit on the data it was computed from.
"""

from collections import Counter, defaultdict

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._utils import PowerLawStats, _fit_powerlaw, _nan_power_law_stats
from ._orig_block_d import BlockD as _OrigBlockD
from ._fits import (
    SkewNormFit,
    TruncPowerLawFit,
    fit_skewnorm,
    fit_truncated_powerlaw,
    nan_skewnorm,
    nan_trunc_powerlaw,
)
from ._plot_helpers import overlay_skewnorm, overlay_truncated_powerlaw

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
        self._cs_size_skew = _NOT_CALCULATED
        self._inv_num_distinct_cs = _NOT_CALCULATED
        self._inv_cs_freq_fit = _NOT_CALCULATED
        self._inv_cs_size_skew = _NOT_CALCULATED
        self._two_step_fit = _NOT_CALCULATED
        # unsummarised data kept for visualization
        self._cs_sizes = _NOT_CALCULATED
        self._inv_cs_sizes = _NOT_CALCULATED
        self._pair_counts = _NOT_CALCULATED
        self._top_pairs = _NOT_CALCULATED

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def num_distinct_cs(self) -> int:
        return self._require("num_distinct_cs", self._num_distinct_cs)

    @property
    def cs_freq_fit(self) -> PowerLawStats:
        return self._require("cs_freq_fit", self._cs_freq_fit)

    @property
    def cs_size_skew(self) -> SkewNormFit:
        return SkewNormFit(*self._require("cs_size_skew", self._cs_size_skew))

    @property
    def inv_num_distinct_cs(self) -> int:
        return self._require("inv_num_distinct_cs", self._inv_num_distinct_cs)

    @property
    def inv_cs_freq_fit(self) -> PowerLawStats:
        return self._require("inv_cs_freq_fit", self._inv_cs_freq_fit)

    @property
    def inv_cs_size_skew(self) -> SkewNormFit:
        return SkewNormFit(*self._require("inv_cs_size_skew", self._inv_cs_size_skew))

    @property
    def two_step_fit(self) -> TruncPowerLawFit:
        return TruncPowerLawFit(*self._require("two_step_fit", self._two_step_fit))

    # ── core ──────────────────────────────────────────────────────────────────

    def calculate(self, g: igraph.Graph) -> "BlockD":
        """Compute reduced Block D (CS, inverse CS, two-step path counts)."""
        cs_of = _OrigBlockD._compute_cs(g)
        inv_cs_of = _OrigBlockD._compute_inv_cs(g)

        cs_sizes = np.fromiter((len(cs) for cs in cs_of.values()), dtype=float, count=len(cs_of)) \
            if cs_of else np.array([], dtype=float)
        inv_cs_sizes = np.fromiter((len(cs) for cs in inv_cs_of.values()), dtype=float, count=len(inv_cs_of)) \
            if inv_cs_of else np.array([], dtype=float)
        self._cs_sizes = cs_sizes
        self._inv_cs_sizes = inv_cs_sizes

        self._num_distinct_cs = len(set(cs_of.values())) if cs_of else 0
        self._inv_num_distinct_cs = len(set(inv_cs_of.values())) if inv_cs_of else 0
        self._cs_size_skew = fit_skewnorm(cs_sizes) if cs_sizes.size else nan_skewnorm()
        self._inv_cs_size_skew = fit_skewnorm(inv_cs_sizes) if inv_cs_sizes.size else nan_skewnorm()

        # CS frequency: how often each distinct (inverse-)CS recurs → power-law.
        if cs_of:
            freq = Counter(cs_of.values())
            self._cs_freq_fit = _fit_powerlaw(
                np.fromiter(freq.values(), dtype=int, count=len(freq))
            )
        else:
            self._cs_freq_fit = _nan_power_law_stats()
        if inv_cs_of:
            inv_freq = Counter(inv_cs_of.values())
            self._inv_cs_freq_fit = _fit_powerlaw(
                np.fromiter(inv_freq.values(), dtype=int, count=len(inv_freq))
            )
        else:
            self._inv_cs_freq_fit = _nan_power_law_stats()

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
        """Flatten to a fixed-length 19-vector for cross-KG comparison.

        Layout (forward then inverse, symmetric): num_distinct_cs; CS-frequency
        power-law (alpha, xmin); CS-size skew-normal (5); inv_num_distinct_cs;
        inverse-CS-frequency power-law (alpha, xmin); inverse-CS-size skew-normal
        (5); two-step truncated power-law (alpha, v_min, v_max).

        Attributes absent from stale serialized data are emitted as NaN.
        """
        return [
            self._safe_scalar(lambda: self.num_distinct_cs),
            self._safe_scalar(lambda: self.cs_freq_fit.alpha),
            self._safe_scalar(lambda: self.cs_freq_fit.xmin),
            *self._safe_iter(lambda: self.cs_size_skew, 5),
            self._safe_scalar(lambda: self.inv_num_distinct_cs),
            self._safe_scalar(lambda: self.inv_cs_freq_fit.alpha),
            self._safe_scalar(lambda: self.inv_cs_freq_fit.xmin),
            *self._safe_iter(lambda: self.inv_cs_size_skew, 5),
            *self._safe_iter(lambda: self.two_step_fit, 3),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = ["num_distinct_cs", "cs_freq_alpha", "cs_freq_xmin"]
        names += [f"cs_size_{s}" for s in ("loc", "scale", "shape", "lo", "hi")]
        names += ["inv_num_distinct_cs", "inv_cs_freq_alpha", "inv_cs_freq_xmin"]
        names += [f"inv_cs_size_{s}" for s in ("loc", "scale", "shape", "lo", "hi")]
        names += ["two_step_alpha", "two_step_vmin", "two_step_vmax"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 19-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 19

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
    def _two_step_pair_counts(g: igraph.Graph) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
        """Return the full path-count value set and the top pairs (for viz).

        Path count per ``(in_pred q, out_pred p)`` is
        ``Σ_x deg_in(x,q)·deg_out(x,p)`` — the multiplicity-weighted 2-hop count.
        Mirrors the original Block D computation but returns the *complete* count
        array (so the truncated power-law sees every value), not just the top-k.
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
        cs, inv = self.cs_size_skew, self.inv_cs_size_skew
        ts = self.two_step_fit
        lines = [
            "=== Reduced Block D: Characteristic Sets & Two-step (G3) ===",
            f"  num_distinct_cs    : {self.num_distinct_cs}",
            f"  CS frequency       : power-law(alpha={self.cs_freq_fit.alpha:.4f}, xmin={self.cs_freq_fit.xmin})",
            f"  CS size            : skew-normal(loc={cs.loc:.3f}, scale={cs.scale:.3f}, shape={cs.shape:.3f}, cutoffs=[{cs.lo:.1f},{cs.hi:.1f}])",
            f"  inv_num_distinct_cs: {self.inv_num_distinct_cs}",
            f"  inverse-CS freq    : power-law(alpha={self.inv_cs_freq_fit.alpha:.4f}, xmin={self.inv_cs_freq_fit.xmin})",
            f"  inverse-CS size    : skew-normal(loc={inv.loc:.3f}, scale={inv.scale:.3f}, shape={inv.shape:.3f}, cutoffs=[{inv.lo:.1f},{inv.hi:.1f}])",
            f"  two-step path count: trunc. power-law(alpha={ts.alpha:.4f}, range=[{ts.v_min:.0f},{ts.v_max:.0f}])",
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
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            ax = axes[0]
            if not overlay_skewnorm(ax, cs_sizes, self.cs_size_skew, label="|CS|", color="steelblue"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("|CS| (predicate count)")
            ax.set_ylabel("entity count")
            ax.set_title("Forward CS size (fit: skew-normal)")

            ax = axes[1]
            if not overlay_skewnorm(ax, inv_cs_sizes, self.inv_cs_size_skew,
                                    label="inverse |CS|", color="darkorange"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("inverse |CS| (in-predicate count)")
            ax.set_ylabel("entity count")
            ax.set_title("Inverse CS size (fit: skew-normal)")

            ax = axes[2]
            if not overlay_truncated_powerlaw(ax, pair_counts, self.two_step_fit, label="path counts"):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("path count per (q, p)")
            ax.set_ylabel("number of pairs")
            ax.set_title("Two-step path counts (fit: trunc. power-law)")

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block D: plot failed: %s", exc, exc_info=True)
            plt.close("all")
