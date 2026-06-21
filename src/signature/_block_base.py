"""Abstract base class shared by all signature blocks.

Provides the ``_NOT_CALCULATED`` sentinel, the ``_require`` guard, the
serialization round-trip (``to_serializable`` / ``from_serializable``), and
the concrete ``as_dict`` method.  All domain logic stays in the individual
block subclasses.
"""

from abc import ABC, abstractmethod
from typing import Any

import igraph

from ._serialize import encode_state, decode_state

# Sentinel that distinguishes "not yet calculated" from any legitimate value,
# including None, 0, or False.
_NOT_CALCULATED = object()


class SignatureBlock(ABC):
    """Abstract base for every graph signature block.

    Subclasses must implement:
      - ``calculate(g, **kwargs) -> "SignatureBlock"``
      - ``as_vector() -> list[float]``
      - ``feature_names() -> list[str]``  (classmethod)
      - ``get_na_vec() -> list[float]``   (classmethod)
      - ``visualize(mode, path) -> None``
    """

    def _require(self, name: str, value: object) -> Any:
        """Return *value*, raising if it is still the un-calculated sentinel."""
        if value is _NOT_CALCULATED:
            raise RuntimeError(f"Call calculate() before accessing {name}")
        return value

    @staticmethod
    def _safe_scalar(fn) -> float:
        """Call *fn* and return its float result, or NaN if calculate() was never called."""
        try:
            return float(fn())
        except RuntimeError:
            return float("nan")

    @staticmethod
    def _safe_iter(fn, n: int) -> list[float]:
        """Call *fn* and unpack its *n*-element result, or NaN×n if not calculated."""
        try:
            return [float(x) for x in fn()]
        except RuntimeError:
            return [float("nan")] * n

    @abstractmethod
    def calculate(self, g: igraph.Graph, **kwargs: Any) -> "SignatureBlock":
        """Compute this block's features from the igraph directed graph *g*."""

    @abstractmethod
    def as_vector(self) -> list[float]:
        """Flatten computed features to a fixed-length float list."""

    @classmethod
    @abstractmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""

    @classmethod
    @abstractmethod
    def get_na_vec(cls) -> list[float]:
        """Return an all-NaN vector of the same length as :meth:`as_vector`."""

    @abstractmethod
    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for this block.

        Args:
            mode: ``"plot"`` for a matplotlib figure, ``"text"`` for a CLI summary.
            path: write to this file instead of showing interactively.
        """

    # ── concrete helpers ──────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, float]:
        """Return feature names mapped to values from :meth:`as_vector`.

        Useful for producing named JSON output and for debugging.
        """
        return dict(zip(self.feature_names(), self.as_vector()))

    def to_serializable(self) -> dict:
        """Return a JSON-serializable dict of this block's full internal state.

        Preserves numpy arrays, ``PowerLawStats`` tuples, nested dicts, and
        other non-JSON-native types via tagged encoding.  Reload with
        :meth:`from_serializable`.  Call :meth:`calculate` first.
        """
        return encode_state(self.__dict__)

    @classmethod
    def from_serializable(cls, data: dict) -> "SignatureBlock":
        """Reconstruct a block instance from :meth:`to_serializable` output.

        Calls ``__init__`` first so that attributes added after serialization
        receive their ``_NOT_CALCULATED`` defaults rather than being absent.
        """
        obj = cls.__new__(cls)
        obj.__init__()
        obj.__dict__.update(decode_state(data))
        return obj
