"""Block D — Characteristic set features."""

from collections import Counter, defaultdict

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._utils import PowerLawStats, _fit_powerlaw, _nan_power_law_stats

log = get_logger(__name__)

_TOP_K_PAIRS = 20  # top pair frequencies kept in as_vector(); mirrors _TOP_K_SV


class BlockD(SignatureBlock):
    """Block D — Characteristic set features of a KG.

    A characteristic set (CS) of a subject entity is the set of predicates it
    uses as outgoing edges. The inverse CS is the set of incoming predicates for
    an object entity. Two-step pairs count directed 2-hop paths per (in_pred,
    out_pred) label — ``Σ_x deg_in(x,q)·deg_out(x,p)`` — which predicts path-2
    query selectivity.

    Usage::

        d = BlockD().calculate(g)
        d.as_vector()                      # fixed-length comparison vector
        d.as_dict()                        # named key-value pairs
        d.visualize()                      # interactive matplotlib figure
        d.visualize(mode="text")           # CLI summary
        d.visualize(path="out.png")        # save plot to file
    """

    def __init__(self) -> None:
        # Forward CS
        self._num_distinct_cs = _NOT_CALCULATED
        self._cs_freq_stats = _NOT_CALCULATED
        self._cs_size_mean = _NOT_CALCULATED
        self._cs_size_median = _NOT_CALCULATED
        self._cs_size_p90 = _NOT_CALCULATED
        # Inverse CS
        self._inv_num_distinct_cs = _NOT_CALCULATED
        self._inv_cs_freq_stats = _NOT_CALCULATED
        self._inv_cs_size_mean = _NOT_CALCULATED
        self._inv_cs_size_median = _NOT_CALCULATED
        self._inv_cs_size_p90 = _NOT_CALCULATED
        # Two-step pairs
        self._top_pair_freqs = _NOT_CALCULATED
        self._pair_freq_stats = _NOT_CALCULATED
        self._top_pairs = _NOT_CALCULATED
        # visualization-only
        self._cs_sizes = _NOT_CALCULATED
        self._inv_cs_sizes = _NOT_CALCULATED

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def num_distinct_cs(self) -> int:
        return self._require("num_distinct_cs", self._num_distinct_cs)

    @property
    def cs_freq_stats(self) -> PowerLawStats:
        return self._require("cs_freq_stats", self._cs_freq_stats)

    @property
    def cs_size_mean(self) -> float:
        return self._require("cs_size_mean", self._cs_size_mean)

    @property
    def cs_size_median(self) -> float:
        return self._require("cs_size_median", self._cs_size_median)

    @property
    def cs_size_p90(self) -> float:
        return self._require("cs_size_p90", self._cs_size_p90)

    @property
    def inv_num_distinct_cs(self) -> int:
        return self._require("inv_num_distinct_cs", self._inv_num_distinct_cs)

    @property
    def inv_cs_freq_stats(self) -> PowerLawStats:
        return self._require("inv_cs_freq_stats", self._inv_cs_freq_stats)

    @property
    def inv_cs_size_mean(self) -> float:
        return self._require("inv_cs_size_mean", self._inv_cs_size_mean)

    @property
    def inv_cs_size_median(self) -> float:
        return self._require("inv_cs_size_median", self._inv_cs_size_median)

    @property
    def inv_cs_size_p90(self) -> float:
        return self._require("inv_cs_size_p90", self._inv_cs_size_p90)

    @property
    def top_pair_freqs(self) -> np.ndarray:
        return self._require("top_pair_freqs", self._top_pair_freqs)

    @property
    def pair_freq_stats(self) -> PowerLawStats:
        return self._require("pair_freq_stats", self._pair_freq_stats)

    @property
    def top_pairs(self) -> list[tuple[str, str, int]]:
        return self._require("top_pairs", self._top_pairs)

    # ── core methods ──────────────────────────────────────────────────────────

    def calculate(self, g: igraph.Graph) -> "BlockD":
        """Compute Block D (characteristic sets) of the graph signature."""
        cs_of = self._compute_cs(g)
        inv_cs_of = self._compute_inv_cs(g)

        n_cs, cs_freq, cs_mean, cs_med, cs_p90 = self._cs_scalar_stats(cs_of)
        self._num_distinct_cs = n_cs
        self._cs_freq_stats = cs_freq
        self._cs_size_mean = cs_mean
        self._cs_size_median = cs_med
        self._cs_size_p90 = cs_p90
        log.info(
            "Block D: computed forward CS stats (num_distinct=%d, alpha=%.4f, mean=%.2f, p90=%.2f)",
            n_cs, cs_freq.alpha, cs_mean, cs_p90,
        )

        n_inv, inv_freq, inv_mean, inv_med, inv_p90 = self._cs_scalar_stats(inv_cs_of)
        self._inv_num_distinct_cs = n_inv
        self._inv_cs_freq_stats = inv_freq
        self._inv_cs_size_mean = inv_mean
        self._inv_cs_size_median = inv_med
        self._inv_cs_size_p90 = inv_p90
        log.info(
            "Block D: computed inverse CS stats (num_distinct=%d, alpha=%.4f, mean=%.2f, p90=%.2f)",
            n_inv, inv_freq.alpha, inv_mean, inv_p90,
        )

        top_freqs, pair_freq, top_pairs = self._two_step_pair_stats(g)
        self._top_pair_freqs = top_freqs
        self._pair_freq_stats = pair_freq
        self._top_pairs = top_pairs
        log.info(
            "Block D: computed two-step pair stats (n_distinct_pairs=%d, alpha=%.4f)",
            len(top_pairs), pair_freq.alpha,
        )

        self._cs_sizes = (
            np.fromiter((len(cs) for cs in cs_of.values()), dtype=float, count=len(cs_of))
            if cs_of else np.array([], dtype=float)
        )
        self._inv_cs_sizes = (
            np.fromiter((len(cs) for cs in inv_cs_of.values()), dtype=float, count=len(inv_cs_of))
            if inv_cs_of else np.array([], dtype=float)
        )

        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 34-vector for cross-KG comparison."""
        return [
            # --- Forward CS (6 values) ---
            float(self.num_distinct_cs),
            self.cs_freq_stats.alpha,
            self.cs_freq_stats.ks,
            self.cs_size_mean,
            self.cs_size_median,
            self.cs_size_p90,
            # --- Inverse CS (6 values) ---
            float(self.inv_num_distinct_cs),
            self.inv_cs_freq_stats.alpha,
            self.inv_cs_freq_stats.ks,
            self.inv_cs_size_mean,
            self.inv_cs_size_median,
            self.inv_cs_size_p90,
            # --- Two-step pair frequencies (_TOP_K_PAIRS + 2 values) ---
            *self.top_pair_freqs.tolist(),
            self.pair_freq_stats.alpha,
            self.pair_freq_stats.ks,
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = [
            "num_distinct_cs", "cs_freq_alpha", "cs_freq_ks",
            "cs_size_mean", "cs_size_median", "cs_size_p90",
            "inv_num_distinct_cs", "inv_cs_freq_alpha", "inv_cs_freq_ks",
            "inv_cs_size_mean", "inv_cs_size_median", "inv_cs_size_p90",
        ]
        names += [f"pair_freq_top_{i:02d}" for i in range(1, _TOP_K_PAIRS + 1)]
        names += ["pair_freq_alpha", "pair_freq_ks"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 34-element NaN vector (same length as as_vector())."""
        return [float("nan")] * (12 + _TOP_K_PAIRS + 2)

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for this block's computed features.

        Args:
            mode: "plot" for a 2x2 matplotlib figure, "text" for a CLI summary.
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
        lines.append("=== Block D: Characteristic Set Features ===\n")

        for label, n, freq, mean, median, p90 in [
            ("Forward CS",
             self.num_distinct_cs, self.cs_freq_stats,
             self.cs_size_mean, self.cs_size_median, self.cs_size_p90),
            ("Inverse CS",
             self.inv_num_distinct_cs, self.inv_cs_freq_stats,
             self.inv_cs_size_mean, self.inv_cs_size_median, self.inv_cs_size_p90),
        ]:
            lines.append(f"{label}:")
            lines.append(f"  Distinct types: {n}")
            lines.append(
                f"  Freq fit: alpha={freq.alpha:.4f}  ks={freq.ks:.4f}"
            )
            lines.append(
                f"  Size stats: mean={mean:.2f}  median={median:.2f}  p90={p90:.2f}"
            )

        lines.append("\nTwo-step pairs (top pairs):")
        if self.top_pairs:
            lines.append(f"  {'in_pred':<35s}  {'out_pred':<35s}  count  freq")
            for i, (q, p, cnt) in enumerate(self.top_pairs[:10]):
                freq_val = self.top_pair_freqs[i]
                lines.append(
                    f"  {self._short_uri(q):<35s}  {self._short_uri(p):<35s}  {cnt:>5}  {freq_val:.4f}"
                )
        else:
            lines.append("  (no two-step pairs)")
        lines.append(
            f"\nPair freq fit: alpha={self.pair_freq_stats.alpha:.4f}  ks={self.pair_freq_stats.ks:.4f}"
        )

        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            cs_sizes: np.ndarray = self._require("_cs_sizes", self._cs_sizes)  # type: ignore[assignment]
            inv_cs_sizes: np.ndarray = self._require("_inv_cs_sizes", self._inv_cs_sizes)  # type: ignore[assignment]
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            self._plot_cs_size_hist(axes[0, 0], cs_sizes, "Forward CS size distribution")
            self._plot_cs_size_hist(axes[0, 1], inv_cs_sizes, "Inverse CS size distribution")
            self._plot_pair_freqs(axes[1, 0], self.top_pair_freqs, self.top_pairs)
            self._plot_cs_size_comparison(axes[1, 1], cs_sizes, inv_cs_sizes)

            plt.tight_layout()
            if path is None:
                plt.show()
            else:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            log.warning("Block D: plot failed: %s", exc, exc_info=True)
            plt.close("all")

    @staticmethod
    def _plot_cs_size_hist(ax, sizes: np.ndarray, title: str) -> None:
        if sizes.size == 0:
            ax.set_title(f"{title} (no data)")
            return
        max_size = int(sizes.max())
        bins = np.arange(0.5, max_size + 1.5, 1.0)
        ax.hist(sizes, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("|CS| (predicate count)")
        ax.set_ylabel("entity count")
        ax.set_title(title)

    @staticmethod
    def _plot_pair_freqs(ax, freqs: np.ndarray, top_pairs: list[tuple[str, str, int]]) -> None:
        nonzero = int(np.count_nonzero(freqs))
        if nonzero == 0:
            ax.set_title("Top-K pair frequencies (no data)")
            return
        x = np.arange(nonzero)
        labels = [
            f"({BlockD._short_uri(q)},{BlockD._short_uri(p)})"
            for q, p, _ in top_pairs[:nonzero]
        ]
        ax.bar(x, freqs[:nonzero], color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("normalised frequency")
        ax.set_title(f"Top-{nonzero} two-step pair frequencies")

    @staticmethod
    def _plot_cs_size_comparison(ax, cs_sizes: np.ndarray, inv_cs_sizes: np.ndarray) -> None:
        fwd = cs_sizes[cs_sizes > 0] if cs_sizes.size else np.array([])
        inv = inv_cs_sizes[inv_cs_sizes > 0] if inv_cs_sizes.size else np.array([])
        data = [fwd, inv]
        labels = ["Forward CS", "Inverse CS"]
        has_data = [d.size > 1 for d in data]
        if any(has_data):
            ax.violinplot(
                [data[i] for i in range(2) if has_data[i]],
                positions=[i + 1 for i in range(2) if has_data[i]],
                showmedians=True,
            )
        for pos, d in enumerate(data, start=1):
            clean = d[~np.isnan(d)] if d.size else np.array([])
            if clean.size:
                ax.scatter([pos] * len(clean), clean, color="black", s=15, zorder=3, alpha=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel("|CS| (predicate count)")
        ax.set_title("CS size distribution: forward vs inverse")

    @staticmethod
    def _compute_cs(g: igraph.Graph) -> dict[int, frozenset[str]]:
        """Single g.es pass → cs_of[v_idx] = frozenset of outgoing predicates."""
        cs_of: defaultdict[int, set[str]] = defaultdict(set)
        for e in g.es:
            cs_of[e.source].add(e["predicate"])
        return {v: frozenset(preds) for v, preds in cs_of.items()}

    @staticmethod
    def _compute_inv_cs(g: igraph.Graph) -> dict[int, frozenset[str]]:
        """Single g.es pass → inv_cs_of[v_idx] = frozenset of incoming predicates (non-literals only)."""
        inv_cs_of: defaultdict[int, set[str]] = defaultdict(set)
        is_literal: list[bool] = g.vs["is_literal"]
        for e in g.es:
            if not is_literal[e.target]:
                inv_cs_of[e.target].add(e["predicate"])
        return {v: frozenset(preds) for v, preds in inv_cs_of.items()}

    @staticmethod
    def _cs_scalar_stats(
        cs_of: dict[int, frozenset[str]],
    ) -> tuple[int, PowerLawStats, float, float, float]:
        """Derive scalar summary from a cs_of / inv_cs_of mapping."""
        if not cs_of:
            return 0, _nan_power_law_stats(), float("nan"), float("nan"), float("nan")

        cs_values: list[frozenset[str]] = list(cs_of.values())
        num_distinct: int = len(set(cs_values))

        freq_counter: defaultdict[frozenset[str], int] = defaultdict(int)
        for cs in cs_values:
            freq_counter[cs] += 1
        freq_arr = np.fromiter(freq_counter.values(), dtype=int, count=len(freq_counter))
        freq_stats = _fit_powerlaw(freq_arr)

        sizes = np.fromiter((len(cs) for cs in cs_values), dtype=float, count=len(cs_values))
        size_mean = float(np.mean(sizes))
        size_median = float(np.median(sizes))
        size_p90 = float(np.percentile(sizes, 90))

        return num_distinct, freq_stats, size_mean, size_median, size_p90

    @staticmethod
    def _two_step_pair_stats(
        g: igraph.Graph,
    ) -> tuple[np.ndarray, PowerLawStats, list[tuple[str, str, int]]]:
        """Count directed 2-hop paths per (in_pred, out_pred) label, in one g.es pass.

        For each bridge entity ``x`` we accumulate the per-predicate in- and
        out-degree, then the number of paths ``s -q-> x -p-> o`` labelled (q, p) is
        ``path_count(q, p) = Σ_x deg_in(x, q) · deg_out(x, p)``. This is the
        multiplicity-weighted count that predicts path-2 query selectivity — unlike a
        bridge-node count, it accounts for how many edges actually meet at ``x``.
        """
        # Per-predicate in/out *degree* per entity (Counter, not set) so the product
        # below counts edge instances, not just whether a predicate is present.
        out_deg: defaultdict[int, Counter] = defaultdict(Counter)
        in_deg: defaultdict[int, Counter] = defaultdict(Counter)
        is_literal: list[bool] = g.vs["is_literal"]
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
            return (
                np.zeros(_TOP_K_PAIRS, dtype=float),
                _nan_power_law_stats(),
                [],
            )

        sorted_pairs: list[tuple[tuple[str, str], int]] = sorted(
            pair_counts.items(), key=lambda kv: kv[1], reverse=True
        )
        all_counts: np.ndarray = np.fromiter(
            (cnt for _, cnt in sorted_pairs), dtype=int, count=len(sorted_pairs)
        )
        total: int = int(all_counts.sum())

        top_k = sorted_pairs[:_TOP_K_PAIRS]
        freqs: np.ndarray = np.zeros(_TOP_K_PAIRS, dtype=float)
        for i, (_, cnt) in enumerate(top_k):
            freqs[i] = cnt / total

        top_pairs: list[tuple[str, str, int]] = [(q, p, cnt) for (q, p), cnt in top_k]
        freq_stats: PowerLawStats = _fit_powerlaw(all_counts)

        return freqs, freq_stats, top_pairs
