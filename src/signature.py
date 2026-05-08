"""Graph signature measurement for KGs loaded via kg_io.load_kg."""

from dataclasses import dataclass

import igraph


@dataclass
class BlockA:
    """Block A — Size and density features of a KG."""
    num_entities: int       # |V|  distinct non-literal nodes
    num_triples: int        # |E|  total triples
    num_relations: int      # |R|  distinct predicates
    density: float          # |E| / |V|^2
    triples_per_entity: float  # |E| / |V|
    relation_reuse: float   # |E| / |R|

    def as_vector(self) -> list[float]:
        return [
            float(self.num_entities),
            float(self.num_triples),
            float(self.num_relations),
            self.density,
            self.triples_per_entity,
            self.relation_reuse,
        ]


def block_a(g: igraph.Graph) -> BlockA:
    """Compute Block A (size and density) of the graph signature.

    Literals are excluded from the entity count, matching the definition
    |V| = distinct subjects ∪ objects excluding RDF literals.
    """
    num_entities = sum(1 for v in g.vs if not v["is_literal"])
    num_triples = g.ecount()
    num_relations = len(set(g.es["predicate"])) if num_triples > 0 else 0

    density = num_triples / (num_entities ** 2) if num_entities > 0 else 0.0
    triples_per_entity = num_triples / num_entities if num_entities > 0 else 0.0
    relation_reuse = num_triples / num_relations if num_relations > 0 else 0.0

    return BlockA(
        num_entities=num_entities,
        num_triples=num_triples,
        num_relations=num_relations,
        density=density,
        triples_per_entity=triples_per_entity,
        relation_reuse=relation_reuse,
    )
