"""Reduced Block A — Size and vocabulary (G0).

The over-determined original stored ``num_triples``, ``density`` and
``relation_reuse``, all exact functions of ``num_entities``, mean degree and
``num_relations``. The reduced block keeps only the independent root parameters:
``num_entities``, ``num_relations``, **mean degree** ``E/V`` (the size-stable
edge-budget handle) and ``type_edge_frac`` (the rdf:type share of that budget,
which splits it into type vs content edges). ``num_classes`` is reported by
Block C, which already scans ``rdf:type``.
"""

import igraph

from .._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED
from ._utils import RDF_TYPE

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
        self._type_edge_frac = _NOT_CALCULATED

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

    @property
    def type_edge_frac(self) -> float:
        """Fraction of E that is ``rdf:type`` — splits the budget into type vs content."""
        return self._require("type_edge_frac", self._type_edge_frac)

    def calculate(self, g: igraph.Graph) -> "BlockA":
        """Compute reduced Block A (size & vocabulary).

        Literals are excluded from the entity count, matching the original
        |V| = distinct subjects ∪ objects excluding RDF literals. Mean degree is
        ``E / V`` and, with V, fixes the edge budget E; ``density`` and
        ``relation_reuse`` are intentionally not stored (derivable).

        ``type_edge_frac`` splits that budget: rdf:type edges are wired outside the
        content-edge budget (Stage 2 emits them separately), and Block B's degree
        fits exclude them, so ``content_E = E·(1 − type_edge_frac)`` is the mean the
        degree distributions actually describe. Without it the stored ``mean_degree``
        and the degree fits would describe different populations.
        """
        num_entities = len(g.vs.select(is_literal_eq=False))
        num_triples = g.ecount()
        n_type = sum(1 for p in g.es["predicate"] if p == RDF_TYPE) if num_triples else 0
        self._num_entities = num_entities
        self._num_relations = len(set(g.es["predicate"])) if num_triples > 0 else 0
        self._mean_degree = num_triples / num_entities if num_entities > 0 else 0.0
        self._type_edge_frac = n_type / num_triples if num_triples > 0 else 0.0
        log.info(
            "Block A: V=%d, R=%d, mean_degree=%.4f, type_edge_frac=%.4f",
            self._num_entities, self._num_relations, self._mean_degree,
            self._type_edge_frac,
        )
        return self

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        return ["num_entities", "num_relations", "mean_degree", "type_edge_frac"]

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 4-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 4

    @classmethod
    def _state_from_features(cls, feats: dict[str, float]) -> dict:
        """Rebuild Block A's state from the flat feature dict."""
        return {
            "_num_entities": cls._int(feats, "num_entities"),
            "_num_relations": cls._int(feats, "num_relations"),
            "_mean_degree": feats["mean_degree"],
            # Pre-type_edge_frac signatures round-trip as untyped (the value the
            # old budget math implied for a graph with no rdf:type edges).
            "_type_edge_frac": feats.get("type_edge_frac", 0.0),
        }

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 4-vector for cross-KG comparison.

        Attributes absent from stale serialized data are emitted as NaN.
        """
        return [
            self._safe_scalar(lambda: self.num_entities),
            self._safe_scalar(lambda: self.num_relations),
            self._safe_scalar(lambda: self.mean_degree),
            self._safe_scalar(lambda: self.type_edge_frac),
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
            f"  num_entities  : {self.num_entities}",
            f"  num_relations : {self.num_relations}",
            f"  mean_degree   : {self.mean_degree:.4f}",
            f"  type_edge_frac: {self.type_edge_frac:.4f}",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")
