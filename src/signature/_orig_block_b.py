"""Block B — Degree structure features."""

from collections import defaultdict

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._utils import (
    MIN_SAMPLES_FOR_FIT,
    PowerLawStats,
    _fit_powerlaw,
    _nan_power_law_stats,
    _summarize_values,
)

log = get_logger(__name__)

_DEGREE_HIST_LOG_SCALE: bool = False  # set False for linear axes on degree histograms


class BlockB(SignatureBlock):
    """Block B — Degree structure features of a KG.

    Aggregate features fit the in/out-degree distributions (over non-literal
    vertices) with the `powerlaw` package. Per-relation features quantify how
    multi-valued each predicate is, distinguishing functional relations like
    `bornIn` from many-to-many ones like `friend`/`type` — they have very
    different join selectivities.

    Usage::

        b = BlockB().calculate(g)
        b.as_vector()     # fixed-length comparison vector
        b.as_dict()       # named key-value pairs
        b.visualize()     # interactive matplotlib figure
        b.visualize(mode="text")          # CLI summary
        b.visualize(path="out.png")       # save plot to file
    """

    def __init__(self) -> None:
        self._out_degree_fit = _NOT_CALCULATED
        self._in_degree_fit = _NOT_CALCULATED
        self._object_multiplicity = _NOT_CALCULATED
        self._subject_multiplicity = _NOT_CALCULATED
        self._functionality = _NOT_CALCULATED
        self._inverse_functionality = _NOT_CALCULATED
        self._out_degrees = _NOT_CALCULATED
        self._in_degrees = _NOT_CALCULATED

    @property
    def out_degree_fit(self) -> PowerLawStats:
        return self._require("out_degree_fit", self._out_degree_fit)

    @property
    def in_degree_fit(self) -> PowerLawStats:
        return self._require("in_degree_fit", self._in_degree_fit)

    @property
    def object_multiplicity(self) -> dict[str, PowerLawStats]:
        return self._require("object_multiplicity", self._object_multiplicity)

    @property
    def subject_multiplicity(self) -> dict[str, PowerLawStats]:
        return self._require("subject_multiplicity", self._subject_multiplicity)

    @property
    def functionality(self) -> dict[str, float]:
        return self._require("functionality", self._functionality)

    @property
    def inverse_functionality(self) -> dict[str, float]:
        return self._require("inverse_functionality", self._inverse_functionality)

    @property
    def out_degrees(self) -> np.ndarray:
        return self._require("out_degrees", self._out_degrees)

    @property
    def in_degrees(self) -> np.ndarray:
        return self._require("in_degrees", self._in_degrees)

    def calculate(self, g: igraph.Graph) -> "BlockB":
        """Compute Block B (degree structure) of the graph signature.

        Degree distributions are taken over non-literal vertices only (matching
        Block A's |V| definition); literals can only appear as RDF objects and
        would always have d_out=0. Self-loops contribute 1 to each side, which is
        the RDF-correct count of triples-as-subject and triples-as-object.

        The aggregate power-law fits use `powerlaw.Fit` (KS-optimized x_min,
        discrete-aware, with alternative-distribution KS distances); per-relation
        fits reuse the same helper. See `_fit_powerlaw` for the short-circuit on
        small samples.
        """
        non_lit_vs: igraph.VertexSeq = g.vs.select(is_literal_eq=False)
        if len(non_lit_vs):
            out_degrees: np.ndarray = np.array(g.degree(non_lit_vs, mode="out"), dtype=int)
            in_degrees: np.ndarray = np.array(g.degree(non_lit_vs, mode="in"), dtype=int)
        else:
            out_degrees = np.array([], dtype=int)
            in_degrees = np.array([], dtype=int)

        self._out_degrees = out_degrees
        self._in_degrees = in_degrees
        self._out_degree_fit = _fit_powerlaw(out_degrees)
        log.info(
            "Block B: computed out_degree_fit (alpha=%.4f, xmin=%s, ks=%.4f, n=%d)",
            self._out_degree_fit.alpha, self._out_degree_fit.xmin,
            self._out_degree_fit.ks, out_degrees.size,
        )
        self._in_degree_fit = _fit_powerlaw(in_degrees)
        log.info(
            "Block B: computed in_degree_fit (alpha=%.4f, xmin=%s, ks=%.4f, n=%d)",
            self._in_degree_fit.alpha, self._in_degree_fit.xmin,
            self._in_degree_fit.ks, in_degrees.size,
        )

        obj_mult, subj_mult, func, inv_func = self._per_relation_features(g)
        self._object_multiplicity = obj_mult
        log.info(
            "Block B: computed object_multiplicity (n_relations=%d)", len(obj_mult)
        )
        self._subject_multiplicity = subj_mult
        log.info(
            "Block B: computed subject_multiplicity (n_relations=%d)", len(subj_mult)
        )
        self._functionality = func
        log.info(
            "Block B: computed functionality (n_relations=%d)", len(func)
        )
        self._inverse_functionality = inv_func
        log.info(
            "Block B: computed inverse_functionality (n_relations=%d)", len(inv_func)
        )

        return self

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 22-vector for cross-KG comparison.

        Layout (in order):
          - out_degree_fit: alpha, ks → 2 floats
          - in_degree_fit:  alpha, ks → 2 floats
          - For object_multiplicity: (mean, std, median) of alpha and
            (mean, std, median) of ks over per-relation values → 6 floats
          - Same for subject_multiplicity → 6 floats
          - (mean, std, median) of functionality.values() → 3 floats
          - (mean, std, median) of inverse_functionality.values() → 3 floats

        Per-relation dicts are summarized rather than emitted directly so the
        vector length stays fixed across KGs with any number of predicates.
        """
        om_alpha = _summarize_values(v.alpha for v in self.object_multiplicity.values())
        om_ks    = _summarize_values(v.ks    for v in self.object_multiplicity.values())
        sm_alpha = _summarize_values(v.alpha for v in self.subject_multiplicity.values())
        sm_ks    = _summarize_values(v.ks    for v in self.subject_multiplicity.values())
        func     = _summarize_values(self.functionality.values())
        inv_func = _summarize_values(self.inverse_functionality.values())

        return [
            # aggregate degree fits
            self.out_degree_fit.alpha, self.out_degree_fit.ks,
            self.in_degree_fit.alpha,  self.in_degree_fit.ks,
            # per-relation object multiplicity (alpha, then ks)
            om_alpha.mean, om_alpha.std, om_alpha.median,
            om_ks.mean,    om_ks.std,    om_ks.median,
            # per-relation subject multiplicity (alpha, then ks)
            sm_alpha.mean, sm_alpha.std, sm_alpha.median,
            sm_ks.mean,    sm_ks.std,    sm_ks.median,
            # functionality / inverse functionality
            func.mean,     func.std,     func.median,
            inv_func.mean, inv_func.std, inv_func.median,
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        names = [
            "out_degree_alpha", "out_degree_ks",
            "in_degree_alpha", "in_degree_ks",
        ]
        for prefix in (
            "obj_multiplicity_alpha", "obj_multiplicity_ks",
            "subj_multiplicity_alpha", "subj_multiplicity_ks",
            "functionality", "inverse_functionality",
        ):
            names += [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_median"]
        return names

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 22-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 22

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
        lines.append("=== Block B: Degree Structure ===\n")

        for label, arr, fit in [
            ("Out-degree", self.out_degrees, self.out_degree_fit),
            ("In-degree ", self.in_degrees,  self.in_degree_fit),
        ]:
            if arr.size:
                lines.append(
                    f"{label}: n={arr.size}  min={arr.min()}  max={arr.max()}"
                    f"  mean={arr.mean():.2f}  median={np.median(arr):.1f}"
                )
            else:
                lines.append(f"{label}: n=0 (no data)")
            lines.append(
                f"  fit: alpha={fit.alpha:.4f}  xmin={fit.xmin}  ks={fit.ks:.4f}"
            )

        lines.append("\n--- Functionality (sorted desc) ---")
        for rel, val in sorted(self.functionality.items(), key=lambda x: x[1], reverse=True):
            inv = self.inverse_functionality.get(rel, float("nan"))
            lines.append(
                f"  {self._short_uri(rel):<40s}  func={val:.3f}  inv_func={inv:.3f}"
            )

        lines.append("\n--- Per-relation powerlaw fits ---")
        lines.append(f"  {'relation':<40s}  {'om_alpha':>9s}  {'om_xmin':>7s}  {'sm_alpha':>9s}  {'sm_xmin':>7s}")
        for r in sorted(self.object_multiplicity.keys()):
            om = self.object_multiplicity[r]
            sm = self.subject_multiplicity.get(r)
            sm_a = f"{sm.alpha:.4f}" if sm else "n/a"
            sm_x = str(sm.xmin) if sm else "n/a"
            lines.append(
                f"  {self._short_uri(r):<40s}  {om.alpha:>9.4f}  {om.xmin:>7}  {sm_a:>9}  {sm_x:>7}"
            )

        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            self._plot_degree_hist(axes[0, 0], self.out_degrees, self.out_degree_fit, "Out-degree distribution", _DEGREE_HIST_LOG_SCALE)
            self._plot_degree_hist(axes[0, 1], self.in_degrees,  self.in_degree_fit,  "In-degree distribution",  _DEGREE_HIST_LOG_SCALE)

            # violin of powerlaw alpha values across all relations
            ax = axes[1, 0]
            obj_alphas  = [v.alpha for v in self.object_multiplicity.values()  if not np.isnan(v.alpha)]
            subj_alphas = [v.alpha for v in self.subject_multiplicity.values() if not np.isnan(v.alpha)]
            data = [obj_alphas or [float("nan")], subj_alphas or [float("nan")]]
            if any(len(d) > 1 for d in data):
                ax.violinplot([d for d in data if len(d) > 1],
                              positions=[i + 1 for i, d in enumerate(data) if len(d) > 1],
                              showmedians=True)
            for pos, d in enumerate(data, start=1):
                clean = [v for v in d if not np.isnan(v)]
                if clean:
                    ax.scatter([pos] * len(clean), clean, color="black", s=15, zorder=3, alpha=0.6)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["object_multiplicity\n(alpha)", "subject_multiplicity\n(alpha)"])
            ax.set_ylabel("alpha")
            ax.set_title("Per-relation powerlaw alpha values")

            # binned histogram of functionality values
            ax = axes[1, 1]
            func_vals = [v for v in self.functionality.values()         if not np.isnan(v)]
            inv_vals  = [v for v in self.inverse_functionality.values() if not np.isnan(v)]
            if func_vals:
                ax.hist(func_vals, bins=20, range=(0.0, 1.0), alpha=0.7,
                        label="functionality", color="steelblue")
            if inv_vals:
                ax.hist(inv_vals,  bins=20, range=(0.0, 1.0), alpha=0.5,
                        label="inverse_functionality", color="darkorange")
            ax.set_xlabel("value")
            ax.set_ylabel("count (relations)")
            ax.set_title("Functionality distribution across relations")
            ax.legend()

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
    def _plot_degree_hist(ax, degrees: np.ndarray, fit: PowerLawStats, title: str, log_scale: bool) -> None:
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
        plot_fn = ax.loglog if log_scale else ax.plot
        plot_fn(centers[counts > 0], counts[counts > 0], "o", markersize=4, label="data")

        if not np.isnan(fit.alpha) and not np.isnan(fit.xmin):
            xmin = max(int(fit.xmin), 1)
            x_fit = np.arange(xmin, pos.max() + 1, dtype=float)
            y_fit = x_fit ** (-fit.alpha)
            # normalize scale to histogram counts above xmin
            tail_counts, _ = np.histogram(pos[pos >= xmin], bins=bins)
            total = tail_counts.sum()
            if total > 0:
                y_fit = y_fit / y_fit.sum() * total
            plot_fn(x_fit, y_fit, "-", color="red", linewidth=1.5,
                    label=f"powerlaw α={fit.alpha:.2f}")

        ax.set_xlabel("degree")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    @staticmethod
    def _per_relation_features(
        g: igraph.Graph,
    ) -> tuple[dict[str, PowerLawStats], dict[str, PowerLawStats], dict[str, float], dict[str, float]]:
        """Build per-relation multiplicity / functionality features in one edge pass.

        For each predicate r, computes:
          - object_multiplicity[r]: PowerLawStats over (#distinct objects per subject)
          - subject_multiplicity[r]: PowerLawStats over (#distinct subjects per object)
          - functionality[r]: fraction of subjects whose object-multiplicity == 1
          - inverse_functionality[r]: fraction of objects whose subject-multiplicity == 1

        Assumes `kg_io.load_kg`'s contract: at most one edge per (subject, predicate,
        object) triple. Under that invariant, the count of edges with fixed (r, s)
        equals the number of distinct objects, so plain integer counters suffice.
        Per-relation `_fit_powerlaw` calls short-circuit to all-NaN for relations
        with fewer than MIN_SAMPLES_FOR_FIT distinct subjects (or objects).
        """
        subj_obj_count: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        obj_subj_count: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        for e in g.es:
            r: str = e["predicate"]
            subj_obj_count[r][e.source] += 1
            obj_subj_count[r][e.target] += 1

        object_multiplicity: dict[str, PowerLawStats] = {}
        subject_multiplicity: dict[str, PowerLawStats] = {}
        functionality: dict[str, float] = {}
        inverse_functionality: dict[str, float] = {}

        for r, subj_map in subj_obj_count.items():
            obj_counts = np.fromiter(subj_map.values(), dtype=int, count=len(subj_map))
            object_multiplicity[r] = _fit_powerlaw(obj_counts)
            functionality[r] = float(np.mean(obj_counts == 1)) if obj_counts.size else float("nan")

        for r, obj_map in obj_subj_count.items():
            subj_counts = np.fromiter(obj_map.values(), dtype=int, count=len(obj_map))
            subject_multiplicity[r] = _fit_powerlaw(subj_counts)
            inverse_functionality[r] = float(np.mean(subj_counts == 1)) if subj_counts.size else float("nan")

        return object_multiplicity, subject_multiplicity, functionality, inverse_functionality
