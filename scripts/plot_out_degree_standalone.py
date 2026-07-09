"""Render a standalone Out-degree distribution panel from a measured block_b.json.

Reuses ``BlockB._plot_degree_hist`` (the same panel embedded in the full block_b
diagnostic grid) but as a single-axes figure, with colors matched to the poster's
Stage 3 convergence figure (matplotlib tab:blue dots / tab:orange fit line).

Usage
-----
    python scripts/plot_out_degree_standalone.py \\
        data/test_graphs/wn18rr_v4/signature/block_b.json \\
        --out 6a4a3f10023384faa65f5a1e/figures/out_degree_dist.png
"""

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from signature import BlockB  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("block_b_json", type=Path, help="Path to a measured block_b.json")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--dot-color", default="C0", help="Data-point color (default: matplotlib tab:blue)")
    parser.add_argument("--line-color", default="C1", help="Power-law fit line color (default: matplotlib tab:orange)")
    parser.add_argument("--figsize", nargs=2, type=float, default=(6, 3.2), metavar=("W", "H"),
                        help="Figure size in inches (default: 6x3.2, a flattened aspect ratio)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    raw = json.loads(args.block_b_json.read_text())
    b = BlockB.from_serializable(raw)
    out_degrees = b._require("_out_degrees", b._out_degrees)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    BlockB._plot_degree_hist(ax, out_degrees, b.out_degree_fit, "Out-degree distribution (target)",
                             False, dot_color=args.dot_color, line_color=args.line_color)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
