"""Block A — Size and density features."""

import igraph

from ._logging import get_logger
from ._block_base import SignatureBlock, _NOT_CALCULATED

log = get_logger(__name__)


class BlockA(SignatureBlock):
    """Block A — Size and density features of a KG.

    Usage::

        b = BlockA().calculate(g)
        b.as_vector()                # fixed-length comparison vector
        b.as_dict()                  # named key-value pairs
        b.visualize()                # interactive matplotlib figure
        b.visualize(mode="text")     # CLI summary
        b.visualize(path="out.png")  # save plot to file
    """

    def __init__(self) -> None:
        self._num_entities = _NOT_CALCULATED
        self._num_triples = _NOT_CALCULATED
        self._num_relations = _NOT_CALCULATED
        self._density = _NOT_CALCULATED
        self._triples_per_entity = _NOT_CALCULATED
        self._relation_reuse = _NOT_CALCULATED

    @property
    def num_entities(self) -> int:
        return self._require("num_entities", self._num_entities)

    @property
    def num_triples(self) -> int:
        return self._require("num_triples", self._num_triples)

    @property
    def num_relations(self) -> int:
        return self._require("num_relations", self._num_relations)

    @property
    def density(self) -> float:
        return self._require("density", self._density)

    @property
    def triples_per_entity(self) -> float:
        return self._require("triples_per_entity", self._triples_per_entity)

    @property
    def relation_reuse(self) -> float:
        return self._require("relation_reuse", self._relation_reuse)

    def calculate(self, g: igraph.Graph) -> "BlockA":
        """Compute Block A (size and density) of the graph signature.

        Literals are excluded from the entity count, matching the definition
        |V| = distinct subjects ∪ objects excluding RDF literals.
        """
        self._num_entities = len(g.vs.select(is_literal_eq=False))
        log.info("Block A: computed num_entities (%d)", self._num_entities)
        self._num_triples = g.ecount()
        log.info("Block A: computed num_triples (%d)", self._num_triples)
        self._num_relations = len(set(g.es["predicate"])) if self._num_triples > 0 else 0
        log.info("Block A: computed num_relations (%d)", self._num_relations)

        self._density = self._num_triples / (self._num_entities ** 2) if self._num_entities > 0 else 0.0
        log.info("Block A: computed density (%.6g)", self._density)
        self._triples_per_entity = self._num_triples / self._num_entities if self._num_entities > 0 else 0.0
        log.info("Block A: computed triples_per_entity (%.4f)", self._triples_per_entity)
        self._relation_reuse = self._num_triples / self._num_relations if self._num_relations > 0 else 0.0
        log.info("Block A: computed relation_reuse (%.4f)", self._relation_reuse)

        return self

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return feature names in the same order as :meth:`as_vector`."""
        return [
            "num_entities",
            "num_triples",
            "num_relations",
            "density",
            "triples_per_entity",
            "relation_reuse",
        ]

    @classmethod
    def get_na_vec(cls) -> list[float]:
        """Return a 6-element NaN vector (same length as as_vector())."""
        return [float("nan")] * 6

    def as_vector(self) -> list[float]:
        """Flatten to a fixed-length 6-vector for cross-KG comparison."""
        return [
            float(self.num_entities),
            float(self.num_triples),
            float(self.num_relations),
            self.density,
            self.triples_per_entity,
            self.relation_reuse,
        ]

    def visualize(self, mode: str = "plot", path: str | None = None) -> None:
        """Display or save diagnostics for Block A.

        Args:
            mode: "plot" for a matplotlib bar chart, "text" for a CLI summary.
            path: if given, write output to this file path instead of
                  displaying interactively.
        """
        if mode == "text":
            self._visualize_text(path)
        elif mode == "plot":
            self._visualize_plot(path)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'plot' or 'text'.")

    def _visualize_text(self, path: str | None) -> None:
        lines: list[str] = [
            "=== Block A: Size and Density ===",
            f"  num_entities      : {self.num_entities}",
            f"  num_triples       : {self.num_triples}",
            f"  num_relations     : {self.num_relations}",
            f"  density           : {self.density:.6g}",
            f"  triples_per_entity: {self.triples_per_entity:.4f}",
            f"  relation_reuse    : {self.relation_reuse:.4f}",
        ]
        text = "\n".join(lines)
        if path is None:
            print(text)
        else:
            with open(path, "w") as f:
                f.write(text + "\n")

    def _visualize_plot(self, path: str | None) -> None:
        # All Block A features are unrelated scalars — no meaningful distribution to plot.
        # Use visualize(mode="text") for a summary.
        return
