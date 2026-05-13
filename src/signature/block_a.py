"""Block A — Size and density features."""

from typing import Any

import igraph

from ._logging import get_logger

log = get_logger(__name__)

_NOT_CALCULATED = object()


class BlockA:
    """Block A — Size and density features of a KG.

    Usage::

        b = BlockA().calculate(g)
        b.as_vector()                # fixed-length comparison vector
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

    def _require(self, name: str, value: object) -> Any:
        if value is _NOT_CALCULATED:
            raise RuntimeError(f"Call calculate() before accessing {name}")
        return value

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
        num_entities: int = len(g.vs.select(is_literal_eq=False))
        num_triples: int = g.ecount()
        num_relations: int = len(set(g.es["predicate"])) if num_triples > 0 else 0

        self._num_entities = num_entities
        self._num_triples = num_triples
        self._num_relations = num_relations
        self._density = num_triples / (num_entities ** 2) if num_entities > 0 else 0.0
        self._triples_per_entity = num_triples / num_entities if num_entities > 0 else 0.0
        self._relation_reuse = num_triples / num_relations if num_relations > 0 else 0.0

        return self

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
