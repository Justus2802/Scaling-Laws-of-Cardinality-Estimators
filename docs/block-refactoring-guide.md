# Block Refactoring Guide

## Pattern: Dataclass + Free Function → Single Class

### Before
```
@dataclass BlockX          # holds results, requires all fields at construction
def block_x(g) -> BlockX   # computes and returns a fully-populated BlockX
helper_fn(g) -> ...        # module-level private helper
```

### After
```
class BlockX               # owns state, computation, and presentation
```

Three public methods form the full lifecycle:

| Method | Role |
|--------|------|
| `calculate(g)` | Runs all computation, stores results internally, returns `self` |
| `as_vector()` | Unchanged logic — projects stored results to a fixed-length float list |
| `visualize(mode, path)` | New — CLI text summary or matplotlib figure; saves to file if `path` given |

---

## Key Decisions

**Sentinel instead of `None`** — a module-level `_NOT_CALCULATED = object()` is used as the default for all private attributes. This distinguishes "not yet run" from a legitimately `None`/falsy result.

**Property guards** — every result attribute is private (`_x`) with a `@property` that calls a shared `_require()` helper. Accessing any attribute before `calculate()` raises `RuntimeError("Call calculate() before accessing <name>")`.

**`calculate()` returns `self`** — enables the one-liner `b = BlockX().calculate(g)` while still allowing mutation after the fact.

**Only store what `visualize()` actually needs** — intermediate arrays are added as attributes only when a plot genuinely requires the raw data. Results already captured in the output attributes (dicts of `PowerLawStats`, floats) are reused directly in `visualize()` — no redundant storage.

**Private helpers → `@staticmethod`** — module-level helpers that are only called by this class move inside as `@staticmethod`, keeping the module namespace clean.

**`visualize()` split** — the method dispatches to `_visualize_text()` and `_visualize_plot()`. Plot-specific geometry (histogram helpers, axis setup) goes into further `@staticmethod`s.

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
logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
```

Log levels used across the package:

| Level | When to use |
|-------|-------------|
| `DEBUG` | Per-step intermediates: sample counts, matrix sizes, loop indices |
| `INFO` | Block start/end, key scalar results (mean path length, component count, …) |
| `WARNING` | Degenerate inputs: empty graph, too few samples for a reliable estimate |
| `ERROR` | Unexpected failures that fall back to NaN / default values |

---

## Call Site Updates

Replace all occurrences of the old pattern:

```python
from signature import BlockX, block_x
b = block_x(g)
```

with:

```python
from signature import BlockX
b = BlockX().calculate(g)
```

Files to check per block: `src/signature/__init__.py`, `tests/test_signature_block_x.py`, `scripts/`.

In `__init__.py`:
- Remove `block_x` from the import line and from `__all__`
- Change `b=block_x(g)` → `b=BlockX().calculate(g)` in `compute_signature()`
