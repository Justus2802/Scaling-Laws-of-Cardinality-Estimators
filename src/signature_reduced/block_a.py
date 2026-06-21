"""Reduced Block A — Size and vocabulary (G0).

The over-determined original stored ``num_triples``, ``density`` and
``relation_reuse``, all exact functions of ``num_entities``, mean degree and
``num_relations``. The reduced block keeps only the independent root parameters:
``num_entities``, ``num_relations`` and **mean degree** ``E/V`` (the size-stable
edge-budget handle). ``num_classes`` is reported by Block C, which already scans
``rdf:type``.
"""

import igraph

from signature._logging import get_logger
from signature._block_base import SignatureBlock, _NOT_CALCULATED

log = get_logger(__name__)


class BlockA(SignatureBlock):
    """Reduced Block A — size and vocabulary root parameters.

    Usage::

        b = BlockA().calculate(g)
        b.as_vector()                # fixed-length comparison vector
        b.as_dict()                  # named key-value pairs
        b.visualize(mode="text")     # CLI summary
    """

    def __init__(self) -> None:
        self._num_entities = _NOT_CALCULATED
        self._num_relations = _NOT_CALCULATED
        self._mean_degree = _NOT_CALCULATED

    @property
    def num_entities(self) -> int:
        return self._require("num_entities", self._num_entities)

    @property
    def num_relations(self) -> int:
        return self._require("num_relations", self._num_relations)

    @property
    def mean_degree(self) -> float:
        """Edge-budget handle ``E/V`` (old ``triples_per_entity``)."""
        return self._require("mean_degree", self._mean_degree)

    def calculate(self, g: igraph.Graph) -> "BlockA":
        """Compute reduced Block A (size & vocabulary).

        Literals are excluded from the entity count, matching the original
        |V| = distinct subjects ∪ objects excluding RDF literals. Mean degree is
        ``E / V`` and, with V, fixes the edge budget E; ``density`` and
        ``relation_reuse`` are intentionally not stored (derivable).
        """
        num_entities = len(g.vs.select(is_literal_eq=False))
        num_triples = g.ecount()
        self._num_entities = num_entities
        self._num_relations = len(set(g.es["predicate"])) if num_triples > 0 else 0
        self._mean_degree = num_triples / num_entities if num_entities > 0 else 0.0
        log.info(
            "Block A: V=%d, R=%d, mean_degree=%.4f",
            self._num_entities, self._num_relations, self._mean_degree,
        )
        return self

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        return ["num_entities", "num_relations", "mean_degree"]

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 3-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 3

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 3-vector for cross-KG comparison.

        Attributes absent from stale serialized data are emitted as NaN.
        """
        return [
            self._safe_scalar(lambda: self.num_entities),
            self._safe_scalar(lambda: self.num_relations),
            self._safe_scalar(lambda: self.mean_degree),
        ]

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for reduced Block A.

        Args:
            mode: "text" for a CLI summary; "plot" is a no-op (all scalars).
            path: write to this file instead of stdout (text mode only).
        """
        if mode == "text":
            self._visualize_text(path)
        elif mode == "plot":
            # All features are unrelated scalars — nothing to plot.
            return
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'plot' or 'text'.")

    def _visualize_text(self, path: str | None) -> None:
        lines = [
            "=== Reduced Block A: Size & Vocabulary (G0) ===",
            f"  num_entities : {self.num_entities}",
            f"  num_relations: {self.num_relations}",
            f"  mean_degree  : {self.mean_degree:.4f}",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")
