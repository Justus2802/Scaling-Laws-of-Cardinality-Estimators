"""Conceptual visualization of two signature-sampling strategies.

Both panels render the same toy "signature space" as an x-z plane with measured
real-world signatures as dots on the baseplane; the vertical axis is probability
density.  The two panels contrast how each strategy turns those few measured
points into a sampleable density:

* **Signature Sampling** (panel 1) -- fit a single joint density (here a
  multivariate Gaussian, the textbook "fit a distribution and sample") over the
  measured cloud.  With p >> n the fitted blob smears probability mass into the
  empty gaps between real signatures, so draws land in regions no real graph
  occupies and the inter-component correlations are only crudely captured.

* **Signature Varying** (panel 2) -- anchor an equal-height Gaussian bump on each
  measured signature and sample from the resulting mixture.  Mass collapses to
  ~zero between anchors, so every draw stays near a real signature; correlations
  are preserved implicitly because each bump sits on a correlation-consistent
  real point.  Trades coverage / distributional realism for proximity to reality.

The datapoints are chosen purely for a clear illustration (a correlated ridge
plus a couple of off-ridge points); they do not represent real measured graphs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# Toy measured signatures in the 2D (x, z) projection of signature space.
# Laid out along a correlated ridge (x ~ z) with mild spread, so a single
# fitted Gaussian visibly fails to honour the discrete structure.
MEASURED = np.array(
    [
        [-2.0, -1.6],
        [-0.4, -0.2],
        [1.0, 1.3],
        [1.6, -1.0],   # off-ridge point: breaks the clean correlation
    ]
)


def _grid(lo: float = -3.5, hi: float = 3.5, n: int = 220):
    """Return an (X, Z) meshgrid plus the flattened (N, 2) coordinate stack."""
    ax = np.linspace(lo, hi, n)
    xx, zz = np.meshgrid(ax, ax)
    pts = np.column_stack([xx.ravel(), zz.ravel()])
    return xx, zz, pts


def _gaussian2d(pts: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Evaluate an un-normalised 2D Gaussian density at ``pts`` (N, 2)."""
    inv = np.linalg.inv(cov)
    d = pts - mean
    quad = np.einsum("ni,ij,nj->n", d, inv, d)
    return np.exp(-0.5 * quad)


def fitted_joint_density(pts: np.ndarray) -> np.ndarray:
    """Approach 1: kernel density estimate fitted over the measured cloud.

    A KDE is the generic "fit a joint density over the points" recipe.  With
    only a handful of points the bandwidth smears probability mass into the
    empty gaps between real signatures, so draws land where no real graph sits.
    """
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(MEASURED.T, bw_method=0.55)
    dens = kde(pts.T)
    return dens / dens.max()


def varying_mixture_density(pts: np.ndarray, sigma: float = 0.35) -> np.ndarray:
    """Approach 2: equal-height narrow Gaussian bump per measured signature.

    Each bump is individually max-normalised to height 1 then the per-point
    bumps are combined with a max (envelope), so every anchor peaks at the same
    height and the surface decays to ~zero between anchors.
    """
    cov = (sigma ** 2) * np.eye(2)
    env = np.zeros(pts.shape[0])
    for mu in MEASURED:
        bump = _gaussian2d(pts, mu, cov)
        env = np.maximum(env, bump)  # equal-height envelope of all bumps
    return env


def _panel(ax, xx, zz, dens, title, cmap):
    """Render one 3D panel: density surface + measured points on the floor."""
    surf = dens.reshape(xx.shape)
    ax.plot_surface(
        xx, zz, surf,
        cmap=cmap, linewidth=0, antialiased=True,
        rstride=2, cstride=2, alpha=0.6,  # semi-transparent so floor dots read through
    )
    # Measured signatures as red dots lying in the floor plane (density = 0).
    ax.scatter(
        MEASURED[:, 0], MEASURED[:, 1], np.zeros(len(MEASURED)),
        color="red", s=70, depthshade=False,
        edgecolor="white", linewidth=1.0, zorder=20,
    )

    if title:
        ax.set_title(title, fontsize=12, pad=12)
    # Signature axes are an abstract projection: no scale/labels on the floor.
    ax.set_xlabel("signature axis")
    ax.set_ylabel("signature axis")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zlabel("probability density")
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=28, azim=-60)
    ax.set_box_aspect((1, 1, 0.55))


def _standalone(xx, zz, dens, title, cmap, out: Path) -> None:
    """Render a single approach as its own square figure and save it."""
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _panel(ax, xx, zz, dens, title, cmap)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out}")


def _marginals_figure(xx, zz, dens1, out: Path) -> None:
    """Third figure: 1D marginals of each signature axis (Signature Sampling).

    Marginalising = integrating the joint density over the other axis (here a
    sum over the grid).  The point of this figure is the correlation problem:
    the per-axis marginals do **not** reconstruct the joint (the measured
    points' x and z are correlated), so sampling the two axes independently from
    their marginals would land draws off the real-signature ridge.
    """
    axis = xx[0, :]  # shared 1D coordinate for both signature axes
    s1 = dens1.reshape(xx.shape)

    # Marginal over an axis = sum the joint over the other axis; max-normalised.
    def _norm(v):
        return v / v.max()

    marg_x = _norm(s1.sum(axis=0))  # over z -> f(x)
    marg_z = _norm(s1.sum(axis=1))  # over x -> f(z)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [
        (axes[0], "signature axis x (first axis)", marg_x, MEASURED[:, 0]),
        (axes[1], "signature axis z (second axis)", marg_z, MEASURED[:, 1]),
    ]
    for ax, xlabel, m, anchors in panels:
        ax.plot(axis, m, color="C0", lw=2)
        ax.fill_between(axis, m, color="C0", alpha=0.15)
        # Measured points projected onto this axis (rug).
        ax.plot(anchors, np.zeros_like(anchors), "k|", ms=14, mew=2,
                label="measured signatures")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("marginal density (normalised)")
        ax.set_ylim(0, 1.08)
        ax.set_xticklabels([])
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("developer_docs/notes/figures"),
        help="output directory for the figures",
    )
    p.add_argument("--sigma", type=float, default=0.35,
                   help="bump width for the Signature-Varying mixture")
    args = p.parse_args()

    xx, zz, pts = _grid()
    dens1 = fitted_joint_density(pts)
    dens2 = varying_mixture_density(pts, sigma=args.sigma)

    title1 = "Signature Sampling"
    title2 = "Signature Varying"

    # Shared colour scheme so the two approaches are visually comparable.
    # Light-at-the-floor map (Blues) keeps the red anchor dots high-contrast.
    cmap = cm.Blues

    # Standalone figure per approach (no title; the slide supplies the header).
    _standalone(xx, zz, dens1, "", cmap,
                args.out_dir / "sampling_signature_sampling.png")
    _standalone(xx, zz, dens2, "", cmap,
                args.out_dir / "sampling_signature_varying.png")

    # Combined side-by-side figure.
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _panel(ax1, xx, zz, dens1, "Approach 1 - " + title1, cmap)
    _panel(ax2, xx, zz, dens2, "Approach 2 - " + title2, cmap)
    fig.tight_layout()
    combined = args.out_dir / "sampling_approaches.png"
    combined.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(combined, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {combined}")

    # Third figure: per-axis marginals (correlation point).
    _marginals_figure(xx, zz, dens1,
                      args.out_dir / "sampling_marginals.png")


if __name__ == "__main__":
    main()
