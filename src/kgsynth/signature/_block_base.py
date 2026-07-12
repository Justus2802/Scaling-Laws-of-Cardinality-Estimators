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
        """Call *fn* and return its float result, or NaN if calculate() was never called.

        TypeError also maps to NaN: stale serialized data can hold a fit of a
        superseded shape (e.g. a 6-field PowerLawStats where a 3-field
        TruncPowerLawFit is expected), which fails on unpacking.
        """
        try:
            return float(fn())
        except (RuntimeError, TypeError):
            return float("nan")

    @staticmethod
    def _safe_iter(fn, n: int) -> list[float]:
        """Call *fn* and unpack its *n*-element result, or NaN×n if not calculated.

        TypeError (stale serialized fit of a superseded shape) also yields NaN×n.
        """
        try:
            return [float(x) for x in fn()]
        except (RuntimeError, TypeError):
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

    @classmethod
    def from_features(cls, feats: dict[str, float]) -> "SignatureBlock":
        """Rebuild a block from the flat feature dict (the inverse of :meth:`as_dict`).

        Reconstructs the fit objects each block stores from their named features,
        so the result reproduces every value the generator reads. Subclasses
        implement :meth:`_state_from_features`; this method wraps it in the same
        ``__new__`` + ``__init__`` rebuild :meth:`from_serializable` uses, so
        attributes absent from the feature vector keep their ``_NOT_CALCULATED``
        defaults rather than being missing.

        The raw sample arrays kept for plotting (degree lists, class sizes,
        singular-value spectra) are **not** in the feature vector and are not
        restored: a block rebuilt this way cannot :meth:`visualize`. See
        ``Signature.from_features`` for the full contract.

        :param feats: Feature name → value; must hold this block's feature names.
        :returns: A populated block instance.
        :raises KeyError: If one of this block's features is missing from *feats*.
        """
        obj = cls.__new__(cls)
        obj.__init__()
        obj.__dict__.update(cls._state_from_features(feats))
        return obj

    @classmethod
    def _state_from_features(cls, feats: dict[str, float]) -> dict:
        """Map this block's features back to its internal state dict.

        :param feats: Feature name → value.
        :returns: The ``__dict__`` entries to restore (private attribute names).
        """
        raise NotImplementedError(f"{cls.__name__} does not support from_features()")

    @staticmethod
    def _int(feats: dict[str, float], name: str) -> int:
        """Read a count-like feature as an ``int``.

        Features round-trip through ``float``; a float count propagates into
        ``range()`` and array shapes downstream, so counts are cast explicitly.
        NaN (a genuinely unmeasured count) maps to 0.

        :param feats: Feature name → value.
        :param name: The feature to read.
        :returns: The value as an int.
        """
        value = feats[name]
        return 0 if value != value else int(round(value))

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
