# The `SignatureBlock` Pattern

Every block is a **single class** in `block_<x>.py` that owns its state, computation, and
presentation:

```
class BlockX               # owns state, computation, and presentation
```

This is deliberately one class rather than a dataclass plus free functions: a class can hold
partially-computed state (the `_NOT_CALCULATED` sentinel below), keep the pre-fit raw data for
`visualize`, and expose one uniform lifecycle across all six blocks.

Three public methods form the full lifecycle:

| Method | Role |
|--------|------|
| `calculate(g)` | Runs all computation, stores results internally, returns `self` |
| `as_vector()` | Projects stored results to a fixed-length float list |
| `get_na_vec()` | Classmethod — returns a same-length list of `float("nan")`; used when a block is skipped |
| `visualize(mode, path)` | CLI text summary or matplotlib figure; saves to file if `path` given |

---

## Key Decisions

**Sentinel instead of `None`** — a module-level `_NOT_CALCULATED = object()` is used as the default for all private attributes. This distinguishes "not yet run" from a legitimately `None`/falsy result.

**Property guards** — every result attribute is private (`_x`) with a `@property` that calls a shared `_require()` helper. Accessing any attribute before `calculate()` raises `RuntimeError("Call calculate() before accessing <name>")`.

**`calculate()` returns `self`** — enables the one-liner `b = BlockX().calculate(g)` while still allowing mutation after the fact.

**Only store what `visualize()` actually needs** — intermediate arrays are added as attributes only when a plot genuinely requires the raw data. Results already captured in the output attributes (dicts of `PowerLawStats`, floats) are reused directly in `visualize()` — no redundant storage.

**Private helpers → `@staticmethod`** — module-level helpers that are only called by this class move inside as `@staticmethod`, keeping the module namespace clean.

**`visualize()` split** — the method dispatches to `_visualize_text()` and `_visualize_plot()`. Plot-specific geometry (histogram helpers, axis setup) goes into further `@staticmethod`s.

**Only plot distributions, not unrelated scalars** — `_visualize_plot()` should only produce output when there is meaningful distribution data (histograms, violin plots, multi-point profiles). If all features for a block are unrelated scalars (e.g. Block A's size/density counts), `_visualize_plot()` is a no-op and the text mode is the primary output. Related scalar profiles where the shape across a parameter matters (e.g. k-star counts vs k, motif counts across motif types) are still worth plotting.

---

## Skeleton

```python
from collections import ...
from typing import Any

import igraph
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from ._logging import get_logger
from ._utils import PowerLawStats, _fit_powerlaw, _summarize_values, ...

log = get_logger(__name__)

_NOT_CALCULATED = object()


class BlockX:
    """Block X — <description>.

    Usage::

        b = BlockX().calculate(g)
        b.as_vector()                      # fixed-length comparison vector
        b.visualize()                      # interactive matplotlib figure
        b.visualize(mode="text")           # CLI summary
        b.visualize(path="out.png")        # save plot to file
    """

    def __init__(self) -> None:
        self._result_a = _NOT_CALCULATED
        self._result_b = _NOT_CALCULATED
        # ... one private attribute per result field
        # add visualization-only arrays here if needed

    def _require(self, name: str, value: object) -> Any:
        if value is _NOT_CALCULATED:
            raise RuntimeError(f"Call calculate() before accessing {name}")
        return value

    @property
    def result_a(self) -> ...:
        return self._require("result_a", self._result_a)

    # ... one @property per result field

    def calculate(self, g: igraph.Graph) -> "BlockX":
        """Compute Block X of the graph signature."""
        # run computation
        self._result_a = ...
        # store visualization-only arrays if needed
        return self

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a N-element NaN vector (same length as as_vector())."""
        return [float("nan")] * N

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length N-vector for cross-KG comparison."""
        return [
            self.result_a.alpha, self.result_a.ks,
            # ...
        ]

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics.

        Args:
            mode: "plot" for matplotlib, "text" for CLI summary.
            path: write to file instead of displaying interactively.
        """
        if mode == "text":
            self._visualize_text(path)
        elif mode == "plot":
            self._visualize_plot(path)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'plot' or 'text'.")

    def _visualize_text(self, path: str | None) -> None:
        lines: list[str] = []
        # build text output
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        fig, axes = plt.subplots(...)
        # populate axes
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def _some_helper(g: igraph.Graph) -> ...:
        """Module-level helpers that only serve this class move here."""
        ...
```

---

## Logging

Every block module imports the package logger via:

```python
from ._logging import get_logger
log = get_logger(__name__)
```

This produces a child logger named `signature.<block_module>` (e.g. `signature.block_f`) parented under the `signature` root logger. Consumers control verbosity with standard `logging` configuration:

```python
import logging
logging.getLogger("signature").setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
```

Log levels used across the package:

| Level | When to use |
|-------|-------------|
| `INFO` | Emit one line per computed signature/sub-feature inside `calculate()`, including headline result values. No start/end markers. |
| `WARNING` | Plot-failure fallback inside `_visualize_plot()`; otherwise reserved for genuine fallback paths. |
| `ERROR` | Unexpected failures that fall back to NaN / default values |

The per-signature `INFO` convention means each named feature in a block's signature vector (e.g. `out_degree_fit`, `cs_freq_stats`, `path_template_zipf`) produces a single `log.info(...)` line at the point it is assigned, with the feature's key values formatted into the message. Block B is the orientation reference for this style. Example output:

```
2026-07-01 14:23:11,204 INFO signature.block_b: Block B: computed out_degree_fit (alpha=2.3147, xmin=3, ks=0.0421, n=12034)
2026-07-01 14:23:11,251 INFO signature.block_b: Block B: computed in_degree_fit (alpha=2.1985, xmin=2, ks=0.0387, n=12034)
2026-07-01 14:23:11,298 INFO signature.block_b: Block B: computed object_multiplicity (n_relations=47)
...
```

---

## Selective Block Computation

`compute_reduced_signature()` accepts an optional `blocks` list that controls which blocks are run:

```python
# compute only blocks A and F
sig = compute_reduced_signature("graph.ttl", blocks=["a", "f"])

# sig.b, sig.c, sig.d, sig.e are None
# sig.as_vector() still returns the full-length vector — skipped positions are NaN
vec = sig.as_vector()
```

`ReducedGraphSignature` fields are `Optional[BlockX]` (default `None`). `as_vector()` calls
`BlockX.get_na_vec()` for each `None` field, keeping the vector length fixed regardless
of which blocks were computed.

---

## Adding or invoking a block

A block is constructed and run in one call, then queried:

```python
from kgsynth.signature import BlockX
b = BlockX().calculate(g)
```

`compute_reduced_signature()` in `src/kgsynth/signature/__init__.py` wires each block this way;
a new block is registered by adding its class to `_BLOCK_CLASSES` / `_ALL_BLOCKS` there.
