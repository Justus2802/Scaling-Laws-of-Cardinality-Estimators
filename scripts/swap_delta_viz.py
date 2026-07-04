"""Visualize Stage-3 swap-proposal delta logs (``refine(..., swap_log=…)``).

Input CSV (written by ``signature_roundtrip.py --swap-log`` /
``stage3.refine(swap_log=…)``), one row per evaluated proposal::

    step, targeted, deg_s1, deg_o1, deg_s2, deg_o2, deg_max4,
    d_tri, [d_c4, d_diamond, d_k4, d_paw,] [d_c5,] [d_c6,] d_loss, accepted

Motif-delta cells are empty when a degree guard dropped that delta.

Outputs (next to the CSV unless ``--out`` overrides the first figure):

* ``<csv>.png``           — per-motif distribution of nonzero deltas, accepted vs
  rejected overlaid, plus a grey bar at delta=0 for the zero-delta proposals
  (y capped at the nonzero max and its true height annotated when it would
  otherwise dominate); panel title carries the zero-delta fraction (the headline
  number for approximate-delta viability).
* ``<csv>_leverage.png``  — per-motif |delta| vs max endpoint degree (log-log)
  and a cumulative-leverage curve: share of total |delta| carried by the top-x%
  of proposals (Lorenz-style; steep = leverage concentrated in few swaps).
* ``<csv>_metrics.csv``   — per-motif summary metrics (also printed to console):
  rows logged/computed, zero-delta %, |delta| mean/median/p90/max, share of
  total |delta| from the top 1 % / 10 % of proposals, accept rate for zero- vs
  nonzero-delta proposals.
* ``<csv>_loss.png``      — swap *usefulness*: (left) signed loss-delta
  distribution, accepted vs rejected, with a grey Δloss=0 bar — since loss is
  minimised, mass to the left of 0 is useful; (right) cumulative usefulness curve
  (share of total loss reduction carried by the top-x% of all proposals).
* ``<csv>_loss_metrics.csv`` — graph-level usefulness metrics (also printed):
  useful % (accepted & Δloss<0), improving/neutral/harmful %, accept rates per
  class, achieved-improvement magnitude, top-1 %/10 % improvement concentration.

Usage
-----
    python scripts/swap_delta_viz.py experiments/swap_delta_logs/swaps_wn18rr_v4_seed42_rb5000.csv
    python scripts/swap_delta_viz.py <csv> --motifs d_c4 d_c6 --out fig.png
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# Fixed categorical order (Paul Tol "bright", colourblind-safe): accepted/rejected
# always use the first two; the leverage curves assign the rest per motif in
# column order — never cycled.
_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
_ACCEPTED_C, _REJECTED_C, _ZERO_C = _COLORS[0], _COLORS[1], "#999999"


def _load(path: Path) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    """Read the swap log.

    :returns: (motif column names, {motif: float array with NaN for guard-dropped
        cells}, accepted bool array, deg_max4 array, d_loss array).  Arrays share
        the row order of the CSV.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    motifs = [c for c in rows[0] if c.startswith("d_") and c != "d_loss"]
    deltas = {
        m: np.array([float(r[m]) if r[m] not in ("", None) else np.nan for r in rows])
        for m in motifs
    }
    accepted = np.array([r["accepted"] == "1" for r in rows])
    deg_max4 = np.array([float(r["deg_max4"]) for r in rows])
    d_loss = np.array([float(r["d_loss"]) for r in rows])
    return motifs, deltas, accepted, deg_max4, d_loss


def _hist_bins(vals: np.ndarray) -> np.ndarray:
    """Bin edges for a delta histogram: integer bins when the range is narrow,
    symmetric log-spaced magnitude bins otherwise (deltas are integers, |d| ≥ 1
    here since zeros are filtered before plotting)."""
    lo, hi = vals.min(), vals.max()
    if hi - lo <= 60:
        return np.arange(math.floor(lo) - 0.5, math.ceil(hi) + 1.5, 1.0)
    mx = max(abs(lo), abs(hi))
    pos = np.logspace(0, math.log10(mx + 1), 24)
    edges = np.concatenate([-pos[::-1], pos]) if lo < 0 else pos
    return np.unique(edges)


def _grid(n_panels: int):
    """(rows, cols) for a compact panel grid."""
    cols = min(4, n_panels)
    return math.ceil(n_panels / cols), cols


def _metrics(motifs, deltas, accepted) -> list[dict]:
    """Per-motif summary metrics over computed (non-NaN) rows."""
    out = []
    for m in motifs:
        d = deltas[m]
        comp = ~np.isnan(d)
        dc = d[comp]
        nz = dc != 0
        a = np.abs(dc[nz])
        a_sorted = np.sort(a)[::-1]
        tot = a_sorted.sum()

        def _top_share(frac: float) -> float:
            k = max(1, int(round(frac * len(dc))))
            return float(a_sorted[:k].sum() / tot) if tot > 0 else float("nan")

        acc_c = accepted[comp]
        out.append({
            "motif": m,
            "rows": len(d),
            "computed": int(comp.sum()),
            "zero_frac": round(float((~nz).mean()) if len(dc) else float("nan"), 4),
            "mean_abs": round(float(np.abs(dc).mean()) if len(dc) else float("nan"), 4),
            "median_abs_nonzero": round(float(np.median(a)) if len(a) else float("nan"), 4),
            "p90_abs_nonzero": round(float(np.percentile(a, 90)) if len(a) else float("nan"), 4),
            "max_abs": round(float(a.max()) if len(a) else 0.0, 4),
            "top1pct_share": round(_top_share(0.01), 4),
            "top10pct_share": round(_top_share(0.10), 4),
            "accept_rate_zero": round(float(acc_c[~nz].mean()) if (~nz).any() else float("nan"), 4),
            "accept_rate_nonzero": round(float(acc_c[nz].mean()) if nz.any() else float("nan"), 4),
        })
    return out


def _signed_log_bins(vals: np.ndarray, n: int = 22) -> np.ndarray:
    """Symmetric log-magnitude bin edges spanning the signed range of ``vals``,
    with a small linear cell across zero.  Loss deltas are tiny floats of both
    signs, so linear bins collapse them; log magnitude on each side separates the
    ‘barely useful’ from the ‘very useful’ tail."""
    a = np.abs(vals[vals != 0])
    if a.size == 0:
        return np.array([-1.0, 1.0])
    lo, hi = np.log10(a.min()), np.log10(a.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pos = np.logspace(lo, hi, n)
    lin = pos.min()  # linear cell half-width around 0
    return np.unique(np.concatenate([-pos[::-1], [-lin, lin], pos]))


def _loss_metrics(d_loss: np.ndarray, accepted: np.ndarray) -> dict:
    """Graph-level swap-usefulness metrics from the per-proposal loss deltas.

    Loss is minimised, so a proposal is *useful* when it both lowers the loss
    (``d_loss < 0``) and was accepted.  Also reports the improving/neutral/harmful
    split (independent of acceptance), acceptance rates per class, the magnitude
    of the achieved improvement, and how concentrated the total loss reduction is.
    """
    n = len(d_loss)
    improving = d_loss < 0
    neutral = d_loss == 0
    harmful = d_loss > 0
    useful = improving & accepted
    imp = -d_loss[useful]                       # positive loss reductions actually taken
    imp_sorted = np.sort(imp)[::-1]
    tot = imp_sorted.sum()

    def _share(frac: float) -> float:
        k = max(1, int(round(frac * n)))
        return round(float(imp_sorted[:k].sum() / tot), 4) if tot > 0 else float("nan")

    def _pct(mask) -> float:
        return round(100.0 * mask.mean(), 2)

    return {
        "proposals": n,
        "useful_pct": _pct(useful),               # accepted & Δloss<0
        "improving_pct": _pct(improving),         # Δloss<0 (any accept)
        "neutral_pct": _pct(neutral),             # Δloss==0
        "harmful_pct": _pct(harmful),             # Δloss>0
        "accepted_pct": _pct(accepted),
        "accept_rate_improving": round(float(accepted[improving].mean()), 4) if improving.any() else float("nan"),
        "accept_rate_harmful": round(float(accepted[harmful].mean()), 4) if harmful.any() else float("nan"),
        "mean_improvement_useful": round(float(imp.mean()), 6) if imp.size else float("nan"),
        "median_improvement_useful": round(float(np.median(imp)), 6) if imp.size else float("nan"),
        "max_improvement_useful": round(float(imp.max()), 6) if imp.size else float("nan"),
        "top1pct_improvement_share": _share(0.01),
        "top10pct_improvement_share": _share(0.10),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="swap-log CSV from refine(swap_log=…)")
    parser.add_argument("--motifs", nargs="+", default=None,
                        help="restrict to these delta columns (e.g. d_c4 d_c6)")
    parser.add_argument("--out", type=Path, default=None,
                        help="path for the distribution figure "
                             "(default: <csv>.png; the leverage figure and metrics "
                             "CSV are always named off the input stem)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    motifs, deltas, accepted, deg_max4, d_loss = _load(args.csv)
    if args.motifs:
        missing = [m for m in args.motifs if m not in motifs]
        if missing:
            raise SystemExit(f"unknown motif columns {missing}; available: {motifs}")
        motifs = args.motifs

    # ── metrics (console + CSV) ───────────────────────────────────────────────
    metrics = _metrics(motifs, deltas, accepted)
    metrics_path = args.csv.with_name(args.csv.stem + "_metrics.csv")
    with open(metrics_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    cols = list(metrics[0])
    widths = [max(len(c), *(len(str(r[c])) for r in metrics)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in metrics:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))
    print(f"metrics → {metrics_path}")

    # ── loss-usefulness metrics (console + CSV) ──────────────────────────────
    loss_m = _loss_metrics(d_loss, accepted)
    loss_metrics_path = args.csv.with_name(args.csv.stem + "_loss_metrics.csv")
    with open(loss_metrics_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(loss_m))
        w.writeheader()
        w.writerow(loss_m)
    print("\n  swap usefulness (loss Δ):")
    for k, v in loss_m.items():
        print(f"    {k:<28} {v}")
    print(f"loss metrics → {loss_metrics_path}")

    # ── figure 1: per-motif delta distributions, accepted vs rejected ────────
    nrows, ncols = _grid(len(motifs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(motifs):]:
        ax.set_visible(False)
    for i, m in enumerate(motifs):
        ax = axes.flat[i]
        d = deltas[m]
        comp = ~np.isnan(d)
        nz = comp & (d != 0)
        met = metrics[i]
        n_comp = int(comp.sum())
        zero_pct = met["zero_frac"] * 100.0
        max_nz = 0.0  # tallest nonzero bar (%), for y-axis capping
        if nz.any():
            bins = _hist_bins(d[nz])
            symlog = bins[-1] - bins[0] > 60
            # Weight each proposal by 100/(computed proposals) so bar heights read
            # as a percent of this motif's computed proposals.
            w = np.full(int(nz.sum()), 100.0 / n_comp)
            n_acc, _, _ = ax.hist(d[nz & accepted], bins=bins, weights=w[accepted[nz]],
                                  color=_ACCEPTED_C, alpha=0.75, label="accepted")
            n_rej, _, _ = ax.hist(d[nz & ~accepted], bins=bins, weights=w[~accepted[nz]],
                                  histtype="step", lw=1.5, color=_REJECTED_C, label="rejected")
            max_nz = float(max(n_acc.max(), n_rej.max()))
            if symlog:
                ax.set_xscale("symlog", linthresh=1)
        else:
            symlog = False
        # Grey bar at delta=0 for the zero-delta proposals (kept out of the
        # histogram so the nonzero shape stays legible).  It is usually the
        # tallest bar by far, so cap the y-axis at the nonzero max and annotate
        # the zero bar's true height rather than let it flatten everything.
        if zero_pct > 0:
            w0 = 1.4 if symlog else 0.8
            ax.bar(0, zero_pct, width=w0, color=_ZERO_C, alpha=0.85,
                   zorder=0, label="zero-delta")
            if max_nz > 0 and zero_pct > 1.6 * max_nz:
                ax.set_ylim(0, 1.18 * max_nz)
                ax.annotate(f"{zero_pct:.0f}%", (0, 1.10 * max_nz), ha="center",
                            va="bottom", fontsize=8, color=_ZERO_C, weight="bold")
        dropped = len(d) - int(comp.sum())
        sub = f", {dropped} guard-dropped" if dropped else ""
        ax.set_title(f"{m} — {met['zero_frac'] * 100:.0f}% zero "
                     f"(n={met['computed']}{sub})", fontsize=10)
        ax.set_xlabel("delta per proposal")
        ax.set_ylabel("proposals (%)")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Swap-proposal motif deltas — {args.csv.stem}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dist_png = args.out or args.csv.with_suffix(".png")
    dist_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dist_png, dpi=150, bbox_inches="tight")
    print(f"distributions → {dist_png}")

    # ── figure 2: leverage — |delta| vs endpoint degree + cumulative share ───
    nrows2, ncols2 = _grid(len(motifs) + 1)
    fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(4.2 * ncols2, 3.4 * nrows2),
                               squeeze=False)
    for ax in axes2.flat[len(motifs) + 1:]:
        ax.set_visible(False)
    for i, m in enumerate(motifs):
        ax = axes2.flat[i]
        d = deltas[m]
        nz = ~np.isnan(d) & (d != 0)
        if nz.any():
            ax.scatter(deg_max4[nz], np.abs(d[nz]), s=10, alpha=0.35,
                       color=_COLORS[i % len(_COLORS)], edgecolors="none")
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("max endpoint degree")
        ax.set_ylabel("|delta|")
        ax.grid(True, alpha=0.25)
    # Cumulative leverage: proposals sorted by |delta| descending; y = share of
    # Σ|delta| carried by the top-x% of *all computed* proposals.
    ax = axes2.flat[len(motifs)]
    for i, m in enumerate(motifs):
        d = deltas[m]
        a = np.abs(d[~np.isnan(d)])
        if a.sum() == 0:
            continue
        a = np.sort(a)[::-1]
        x = np.arange(1, len(a) + 1) / len(a) * 100
        ax.plot(x, np.cumsum(a) / a.sum() * 100, lw=2,
                color=_COLORS[i % len(_COLORS)], label=m)
    ax.set_xlabel("top x% of proposals by |delta|")
    ax.set_ylabel("share of total |delta| (%)")
    ax.set_title("cumulative leverage", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig2.suptitle(f"Swap leverage vs endpoint degree — {args.csv.stem}", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    lev_png = dist_png.with_name(dist_png.stem + "_leverage.png")
    fig2.savefig(lev_png, dpi=150, bbox_inches="tight")
    print(f"leverage → {lev_png}")

    # ── figure 3: loss-Δ distribution + cumulative usefulness ────────────────
    # How many attempted swaps are useful (accepted & Δloss<0) and how useful.
    fig3, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    improving = d_loss < 0
    nzl = d_loss != 0
    # Left: signed loss-Δ distribution, accepted vs rejected (percent of proposals).
    n = len(d_loss)
    if nzl.any():
        bins = _signed_log_bins(d_loss[nzl])
        wl = np.full(int(nzl.sum()), 100.0 / n)
        nL_acc, _, _ = axL.hist(d_loss[nzl & accepted], bins=bins, weights=wl[accepted[nzl]],
                                color=_ACCEPTED_C, alpha=0.75, label="accepted")
        nL_rej, _, _ = axL.hist(d_loss[nzl & ~accepted], bins=bins, weights=wl[~accepted[nzl]],
                                histtype="step", lw=1.5, color=_REJECTED_C, label="rejected")
        axL.set_xscale("symlog", linthresh=float(np.abs(d_loss[nzl]).min()))
        max_nzl = float(max(nL_acc.max(), nL_rej.max(), 1e-9))
    else:
        max_nzl = 0.0
    zero_pct = 100.0 * (d_loss == 0).mean()
    if zero_pct > 0:
        lt = float(np.abs(d_loss[nzl]).min()) if nzl.any() else 1.0
        axL.bar(0, zero_pct, width=1.4 * lt, color=_ZERO_C, alpha=0.85, zorder=0,
                label="Δloss=0")
        if max_nzl > 0 and zero_pct > 1.6 * max_nzl:
            axL.set_ylim(0, 1.18 * max_nzl)
            axL.annotate(f"{zero_pct:.0f}%", (0, 1.10 * max_nzl), ha="center",
                         va="bottom", fontsize=8, color=_ZERO_C, weight="bold")
    axL.axvline(0, color="#444444", lw=0.8, zorder=1)
    axL.set_xlabel("loss delta (Δloss)  ← useful | harmful →")
    axL.set_ylabel("proposals (%)")
    axL.set_title(f"useful {loss_m['useful_pct']:.0f}% (accepted & Δloss<0) · "
                  f"improving {loss_m['improving_pct']:.0f}% · harmful {loss_m['harmful_pct']:.0f}%",
                  fontsize=10)
    axL.grid(True, alpha=0.25)
    axL.legend(fontsize=8)
    # Right: cumulative usefulness — accepted improvements sorted largest-first,
    # x = top x% of *all* proposals, y = share of total loss reduction taken.
    imp = -d_loss[improving & accepted]
    if imp.sum() > 0:
        imp = np.sort(imp)[::-1]
        x = np.arange(1, len(imp) + 1) / n * 100
        axR.plot(x, np.cumsum(imp) / imp.sum() * 100, lw=2, color=_ACCEPTED_C)
        axR.axhline(100, color="#BBBBBB", lw=0.8, ls="--")
    axR.set_xlabel("top x% of all proposals by improvement")
    axR.set_ylabel("share of total loss reduction (%)")
    axR.set_title("cumulative usefulness", fontsize=10)
    axR.grid(True, alpha=0.25)
    fig3.suptitle(f"Swap usefulness (loss Δ) — {args.csv.stem}", fontsize=12)
    fig3.tight_layout(rect=(0, 0, 1, 0.94))
    loss_png = dist_png.with_name(dist_png.stem + "_loss.png")
    fig3.savefig(loss_png, dpi=150, bbox_inches="tight")
    print(f"loss → {loss_png}")


if __name__ == "__main__":
    main()
