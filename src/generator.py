"""kgsynth Stages 1 & 2 — Schema sampler and graph instantiation.

Stage 1 (sample_schema): builds the abstract schema (relations, types,
type-relation probability table) from a measured BlockA + BlockC target.

Stage 2 (instantiate): turns a Schema into an igraph.Graph by
  - sampling actual |V| and |E| from Gaussian distributions centred on the
    Schema targets (so two calls with different seeds produce different graphs
    even for the same target signature),
  - assigning types to entities via the Schema's type_weights,
  - sampling each entity's characteristic set from P(r | type) so the
    co-occurrence structure matches the target,
  - wiring edges with preferential attachment to reproduce heavy-tailed
    in-degree distributions,
  - adding rdf:type edges for all typed entities,
  - throttling content edges down to the sampled |E| budget if needed.

Design decisions:
  - Relation frequency weights are sampled from a Zipf distribution whose
    exponent is a tunable parameter; the spec requires it but it is not
    directly available from Blocks A or C (Block B would supply it).
  - The type-relation probability table P(r|t) is constructed via a low-rank
    random factorisation whose singular values match the Block C target, so
    the co-occurrence structure of the generated schema resembles the real KG.
  - All randomness goes through a single np.random.Generator seeded at call
    time, making every output fully reproducible.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import igraph
import numpy as np

from signature import BlockA, BlockC, BlockE, block_e as _measure_block_e

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


# ---------------------------------------------------------------------------
# Schema dataclass
# ---------------------------------------------------------------------------


@dataclass
class Schema:
    """Stage 1 output: abstract schema for a synthetic KG.

    Passed directly to Stage 2 (instantiate) to build the actual graph.

    Attributes
    ----------
    relations : list[str]
        |R| synthetic relation URIs, e.g. "http://kgsynth.org/rel/0".
    relation_weights : np.ndarray, shape (|R|,)
        Normalized frequency weights (sum to 1); controls how often each
        relation appears relative to the others.
    types : list[str]
        |T| synthetic type URIs.  Empty when Block C reports no classes.
    type_weights : np.ndarray, shape (|T|,)
        Normalized type-size weights (sum to 1); governs how many entities
        each type receives in Stage 2.
    type_relation_probs : np.ndarray, shape (|T|, |R|)
        P(r | t) table — for each type, the probability distribution over
        outgoing relations.  Rows sum to 1.  Shape is (0, |R|) when |T| = 0.
    num_entities : int
        Target |V| copied from Block A; used by Stage 2 to size the graph.
    num_triples : int
        Target |E| copied from Block A; used by Stage 2 to size the graph.
    """

    relations: list
    relation_weights: np.ndarray
    types: list
    type_weights: np.ndarray
    type_relation_probs: np.ndarray
    num_entities: int
    num_triples: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zipf_weights(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n normalized frequency weights from a Zipf distribution.

    Weights are shuffled so that relation indices carry no implicit rank
    ordering — the Zipf shape is preserved in the *distribution* of weights,
    not in their index order.
    """
    if n == 0:
        return np.array([], dtype=float)
    ranks = np.arange(1, n + 1, dtype=float)
    weights = ranks ** (-exponent)
    weights /= weights.sum()
    rng.shuffle(weights)
    return weights


def _sample_type_relation_probs(
    num_types: int,
    num_relations: int,
    relation_weights: np.ndarray,
    target_singular_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a P(r|t) matrix whose singular spectrum matches the Block C target.

    Construction (low-rank random factorisation):
      1. Determine rank k = number of nonzero target singular values,
         capped by min(|T|, |R|).
      2. Draw random orthonormal U (|T|×k) and V (|R|×k) via QR.
      3. Form logits = U @ diag(sigma_normalised) @ V^T.
      4. Multiply each row element-wise by the global relation_weights so
         frequently-used relations are more likely to appear across all types.
      5. Row-normalise with softmax to produce valid probability rows.

    Falls back to tiling relation_weights uniformly across types when the
    singular value information is insufficient for a low-rank construction.
    """
    if num_types == 0 or num_relations == 0:
        return np.zeros((num_types, num_relations), dtype=float)

    nonzero_svs = target_singular_values[target_singular_values > 0]
    rank = min(len(nonzero_svs), num_types, num_relations)

    if rank == 0:
        # No co-occurrence signal: every type gets the same global relation weights
        return np.tile(relation_weights, (num_types, 1))

    # Random orthonormal factors
    U = np.linalg.qr(rng.standard_normal((num_types, rank)))[0]      # (T, k)
    V = np.linalg.qr(rng.standard_normal((num_relations, rank)))[0]  # (R, k)
    sigma = nonzero_svs[:rank] / nonzero_svs[:rank].sum()            # normalised

    logits = U @ np.diag(sigma) @ V.T   # (T, R)

    # Shift for numerical stability before exponentiation
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)

    # Bias each row toward globally frequent relations
    P *= relation_weights[np.newaxis, :]

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P /= row_sums
    return P


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_schema(
    a: BlockA,
    c: BlockC,
    *,
    relation_zipf_exponent: float = 2.0,
    seed: int = 0,
) -> Schema:
    """Stage 1: derive an abstract schema from a target BlockA + BlockC.

    Parameters
    ----------
    a : BlockA
        Measured size/density signature of the target KG.
        |V|, |E|, |R| are used directly.
    c : BlockC
        Measured schema/correlation signature of the target KG.
        num_classes, class_size_zipf_exponent, and subj_singular_values
        guide the type structure and co-occurrence reconstruction.
    relation_zipf_exponent : float
        Zipf exponent for relation frequency weights.  Controls how skewed
        relation usage is; real KGs typically fall in [1.5, 2.5].  Block B
        (per-relation multiplicity fits) would supply this in a full pipeline;
        here it is an explicit tuning knob.
    seed : int
        RNG seed; the same seed + inputs always produce the same schema.

    Returns
    -------
    Schema
        Abstract schema ready to be handed to Stage 2 (instantiate).
    """
    rng = np.random.default_rng(seed)

    num_relations = max(1, a.num_relations)
    num_types = max(0, c.num_classes)

    # --- Relations ---
    relations = [f"http://kgsynth.org/rel/{i}" for i in range(num_relations)]
    relation_weights = _zipf_weights(num_relations, relation_zipf_exponent, rng)

    # --- Types ---
    types = [f"http://kgsynth.org/type/{i}" for i in range(num_types)]

    if num_types > 0:
        type_zipf = c.class_size_zipf_exponent
        if not np.isnan(type_zipf) and type_zipf > 0:
            type_weights = _zipf_weights(num_types, type_zipf, rng)
        else:
            # Block C could not fit a Zipf (too few classes): fall back to uniform
            type_weights = np.full(num_types, 1.0 / num_types)
    else:
        type_weights = np.array([], dtype=float)

    # --- Type-relation probability table ---
    type_relation_probs = _sample_type_relation_probs(
        num_types,
        num_relations,
        relation_weights,
        c.subj_singular_values,
        rng,
    )

    return Schema(
        relations=relations,
        relation_weights=relation_weights,
        types=types,
        type_weights=type_weights,
        type_relation_probs=type_relation_probs,
        num_entities=a.num_entities,
        num_triples=a.num_triples,
    )


# ---------------------------------------------------------------------------
# Stage 2 — CS-aware graph instantiation
# ---------------------------------------------------------------------------


def instantiate(
    schema: Schema,
    *,
    v_noise: float = 0.05,
    e_noise: float = 0.05,
    pa_exponent: float = 0.5,
    seed: int = 0,
) -> igraph.Graph:
    """Stage 2: instantiate a KG from a Schema.

    Parameters
    ----------
    schema : Schema
        Output of Stage 1 (sample_schema).
    v_noise : float
        Relative standard deviation for sampling actual |V| around the Schema
        target.  0.05 = 5 % noise.  Set to 0 to reproduce the exact target.
    e_noise : float
        Same for |E|.
    pa_exponent : float
        Preferential-attachment exponent for object selection.  0 = uniform
        random; 1 = linear PA (rich-get-richer); 0.5 is a good default that
        produces moderate hubs without full scale-free extremes.
    seed : int
        RNG seed — fully determines the output given the same schema and
        parameters.  Pass different seeds to get structurally different graphs
        from the same target signature.

    Returns
    -------
    igraph.Graph
        Directed graph with vertex attributes (name, is_literal, …) and edge
        attribute (predicate) matching the kg_io.load_kg contract, so it can
        be passed directly to compute_signature().

    Notes
    -----
    Vertex layout::

        0 .. actual_V − 1          entity nodes  (is_literal = False)
        actual_V .. actual_V + |T| type-class nodes (is_literal = False)

    Content edges (schema.relations) and rdf:type edges are both included;
    their combined count approximates the sampled |E| target.
    """
    rng = np.random.default_rng(seed)
    num_relations = len(schema.relations)
    num_types = len(schema.types)

    # ------------------------------------------------------------------
    # 1. Sample actual |V| and |E| with Gaussian noise
    # ------------------------------------------------------------------
    actual_V = max(2, int(round(
        rng.normal(schema.num_entities, max(1.0, schema.num_entities * v_noise))
    )))
    actual_E_target = max(1, int(round(
        rng.normal(schema.num_triples, max(1.0, schema.num_triples * e_noise))
    )))

    # rdf:type edges (one per entity when types exist) come out of the budget
    n_type_edges = actual_V if num_types > 0 else 0
    content_E_target = max(0, actual_E_target - n_type_edges)

    # CS size mean derived so expected content edges ≈ content_E_target
    cs_size_mean = content_E_target / actual_V if actual_V > 0 else 1.0

    # ------------------------------------------------------------------
    # 2. Assign a type to every entity
    # ------------------------------------------------------------------
    if num_types > 0:
        entity_types = rng.choice(num_types, size=actual_V, p=schema.type_weights)
    else:
        entity_types = np.full(actual_V, -1, dtype=int)

    # ------------------------------------------------------------------
    # 3. Sample a characteristic set (CS) for each entity
    #    CS = subset of relations drawn from P(r | type), size ~ Poisson
    # ------------------------------------------------------------------
    entity_cs: list[np.ndarray] = []
    for v in range(actual_V):
        if num_relations == 0:
            entity_cs.append(np.array([], dtype=int))
            continue

        t = int(entity_types[v])
        if t >= 0:
            probs = schema.type_relation_probs[t].copy()
        else:
            probs = schema.relation_weights.copy()

        # Guard: renormalise in case of floating-point drift
        s = probs.sum()
        if s <= 0:
            probs = schema.relation_weights.copy()
            s = probs.sum()
        probs /= s

        nonzero = int((probs > 0).sum())
        k = min(nonzero, max(0, int(rng.poisson(max(0.1, cs_size_mean)))))
        if k == 0:
            entity_cs.append(np.array([], dtype=int))
            continue

        entity_cs.append(rng.choice(num_relations, size=k, replace=False, p=probs))

    # ------------------------------------------------------------------
    # 4. Wire content edges with preferential attachment
    #    Object picked proportional to current in_degree ^ pa_exponent;
    #    Laplace smoothing (start at 1) ensures every vertex is reachable.
    # ------------------------------------------------------------------
    in_degrees = np.ones(actual_V, dtype=float)
    seen: set[tuple[int, int, str]] = set()
    content_edges: list[tuple[int, int, str]] = []

    for v, cs in enumerate(entity_cs):
        for rel_idx in cs:
            predicate = schema.relations[int(rel_idx)]
            weights = in_degrees ** pa_exponent
            weights[v] = 0.0        # no self-loops
            total = weights.sum()
            if total == 0.0:
                continue
            weights /= total
            obj = int(rng.choice(actual_V, p=weights))
            triple = (v, obj, predicate)
            if triple not in seen:
                seen.add(triple)
                content_edges.append(triple)
                in_degrees[obj] += 1.0

    # ------------------------------------------------------------------
    # 5. Throttle content edges down to budget if over target
    # ------------------------------------------------------------------
    if len(content_edges) > content_E_target > 0:
        keep = rng.choice(len(content_edges), size=content_E_target, replace=False)
        keep_set = set(keep.tolist())
        content_edges = [e for i, e in enumerate(content_edges) if i in keep_set]

    # ------------------------------------------------------------------
    # 6. Build rdf:type edges
    #    Type-class nodes sit at indices actual_V .. actual_V + num_types - 1
    # ------------------------------------------------------------------
    type_edges: list[tuple[int, int, str]] = []
    for v in range(actual_V):
        t = int(entity_types[v])
        if t >= 0:
            type_edges.append((v, actual_V + t, _RDF_TYPE))

    # ------------------------------------------------------------------
    # 7. Assemble igraph.Graph
    # ------------------------------------------------------------------
    total_V = actual_V + num_types
    g = igraph.Graph(n=total_V, directed=True)

    g.vs["name"] = (
        [f"http://kgsynth.org/entity/{i}" for i in range(actual_V)]
        + list(schema.types)
    )
    g.vs["is_literal"] = [False] * total_V
    g.vs["literal_value"] = [None] * total_V
    g.vs["literal_datatype"] = [None] * total_V
    g.vs["literal_lang"] = [None] * total_V

    all_edges = content_edges + type_edges
    if all_edges:
        g.add_edges([(s, o) for s, o, _ in all_edges])
        g.es["predicate"] = [p for _, _, p in all_edges]

    return g


# ---------------------------------------------------------------------------
# Stage 3 — Maslov-Sneppen rewiring with simulated annealing
# ---------------------------------------------------------------------------


def _adj_inc(adj: list, u: int, v: int) -> None:
    """Increment undirected adjacency count for edge (u, v)."""
    adj[u][v] = adj[u].get(v, 0) + 1
    adj[v][u] = adj[v].get(u, 0) + 1


def _adj_dec(adj: list, u: int, v: int) -> None:
    """Decrement undirected adjacency count for edge (u, v)."""
    adj[u][v] -= 1
    if adj[u][v] == 0:
        del adj[u][v]
    adj[v][u] -= 1
    if adj[v][u] == 0:
        del adj[v][u]


def _triangle_delta(adj: list, s1: int, o1: int, s2: int, o2: int) -> int:
    """Compute change in triangle count from swapping o1↔o2 for edges s1→o1, s2→o2.

    Temporarily removes both edges from adj to isolate the common-neighbour
    calculation, then restores them.  Returns gained_triangles - lost_triangles.
    """
    lost1 = len(set(adj[s1]) & set(adj[o1]))
    lost2 = len(set(adj[s2]) & set(adj[o2]))

    _adj_dec(adj, s1, o1)
    _adj_dec(adj, s2, o2)

    gained1 = len(set(adj[s1]) & set(adj[o2]))
    gained2 = len(set(adj[s2]) & set(adj[o1]))

    _adj_inc(adj, s1, o1)
    _adj_inc(adj, s2, o2)

    return (gained1 + gained2) - (lost1 + lost2)


def refine(
    g: igraph.Graph,
    target_e: "BlockE",
    *,
    budget: int = 10_000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.999,
    seed: int = 0,
) -> igraph.Graph:
    """Stage 3: Maslov-Sneppen rewiring + simulated annealing.

    Rewires content edges (never rdf:type edges) using the Maslov-Sneppen
    double-edge swap to preserve per-relation degree sequences while moving the
    triangle count toward target_e.triangle_count.  Metropolis-Hastings
    acceptance allows uphill moves early on; temperature decays geometrically.

    The best graph seen during the walk (lowest |triangle_count - target|) is
    returned, so the result is robust even if SA gets stuck.

    Parameters
    ----------
    g : igraph.Graph
        Output of Stage 2 (instantiate).
    target_e : BlockE
        Block E signature of the target KG — supplies target triangle_count.
    budget : int
        Maximum number of rewiring attempts.
    initial_temp : float
        Starting temperature for the Metropolis criterion.
    cooling_rate : float
        Geometric decay factor applied after each accepted swap.
    seed : int
        RNG seed.

    Returns
    -------
    igraph.Graph
        Best graph encountered during the annealing walk.
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    target_tri = int(target_e.triangle_count)

    # Extract all edges as a mutable list; igraph edges are immutable.
    # type_edges stay fixed; content_edges are rewired.
    type_edge_data: list[tuple[int, int, str]] = []
    content_edge_data: list[tuple[int, int, str]] = []
    for e in g.es:
        entry = (e.source, e.target, e["predicate"])
        if e["predicate"] == _RDF_TYPE:
            type_edge_data.append(entry)
        else:
            content_edge_data.append(entry)

    if len(content_edge_data) < 2:
        return g

    # Group content edge *list indices* by predicate
    rel_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, p) in enumerate(content_edge_data):
        rel_to_idxs[p].append(i)

    swappable_rels = [r for r, lst in rel_to_idxs.items() if len(lst) >= 2]
    if not swappable_rels:
        return g

    # Build dict-based undirected adjacency for fast local triangle counting
    n = g.vcount()
    adj: list[dict] = [{} for _ in range(n)]
    for s, o, _ in content_edge_data:
        _adj_inc(adj, s, o)

    def _count_triangles() -> int:
        total = 0
        for u, nbrs in enumerate(adj):
            for v in nbrs:
                if v > u:
                    total += len(set(adj[u]) & set(adj[v]))
        return total // 3

    current_tri = _count_triangles()
    current_dist = abs(current_tri - target_tri)

    best_dist = current_dist
    best_content = list(content_edge_data)  # snapshot of best-seen layout

    temp = initial_temp

    for _ in range(budget):
        rel = swappable_rels[int(rng.integers(len(swappable_rels)))]
        pool = rel_to_idxs[rel]
        pi1, pi2 = rng.choice(len(pool), size=2, replace=False)
        i1, i2 = pool[pi1], pool[pi2]

        s1, o1, p1 = content_edge_data[i1]
        s2, o2, _  = content_edge_data[i2]

        # Skip swaps that create self-loops
        if s1 == o2 or s2 == o1:
            continue

        delta = _triangle_delta(adj, s1, o1, s2, o2)
        new_tri = current_tri + delta
        new_dist = abs(new_tri - target_tri)

        if new_dist < current_dist:
            accept = True
        else:
            diff = new_dist - current_dist
            accept = bool(rng.random() < math.exp(-diff / max(temp, 1e-10)))

        if accept:
            # Commit the swap in the mutable edge list and adjacency
            content_edge_data[i1] = (s1, o2, p1)
            content_edge_data[i2] = (s2, o1, p1)

            _adj_dec(adj, s1, o1)
            _adj_dec(adj, s2, o2)
            _adj_inc(adj, s1, o2)
            _adj_inc(adj, s2, o1)

            current_tri = new_tri
            current_dist = new_dist
            temp *= cooling_rate

            if current_dist < best_dist:
                best_dist = current_dist
                best_content = list(content_edge_data)

    # Rebuild igraph from best snapshot, preserving vertex attributes
    all_best = best_content + type_edge_data
    g_out = igraph.Graph(n=g.vcount(), directed=True)
    for attr in g.vertex_attributes():
        g_out.vs[attr] = list(g.vs[attr])
    if all_best:
        g_out.add_edges([(s, o) for s, o, _ in all_best])
        g_out.es["predicate"] = [p for _, _, p in all_best]

    return g_out


# ---------------------------------------------------------------------------
# High-level API: Signature + Generator
# ---------------------------------------------------------------------------


@dataclass
class Signature:
    """Compact target signature used by Generator (Blocks A, C, E only).

    Block A supplies size/density targets; Block C supplies schema/class
    structure; Block E supplies motif counts that Stage 3 optimises toward.
    """

    a: "BlockA"
    c: "BlockC"
    e: "BlockE"

    @classmethod
    def from_graph(cls, g: igraph.Graph) -> "Signature":
        from signature import block_a, block_c
        return cls(
            a=block_a(g),
            c=block_c(g),
            e=_measure_block_e(g),
        )

    @classmethod
    def from_file(cls, path) -> "Signature":
        from kg_io import load_kg
        return cls.from_graph(load_kg(Path(path)))


class Generator:
    """Full three-stage KG generator.

    Usage
    -----
    >>> sig = Signature.from_file("target.ttl")
    >>> gen = Generator(sig)
    >>> g = gen.sample(seed=42)          # reproducible
    >>> g2 = gen.sample(seed=99)         # structurally different

    Parameters
    ----------
    target : Signature
        Measured signature of the target KG.  All three stages read from it.
    """

    def __init__(self, target: Signature) -> None:
        self.target = target

    def sample(
        self,
        *,
        seed: int = 0,
        relation_zipf_exponent: float = 2.0,
        v_noise: float = 0.05,
        e_noise: float = 0.05,
        pa_exponent: float = 0.5,
        rewire_budget: int = 10_000,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.999,
    ) -> igraph.Graph:
        """Generate one synthetic KG from the target signature.

        Parameters
        ----------
        seed : int
            Master seed; all three stages derive sub-seeds from it so the
            entire pipeline is reproducible from a single integer.
        relation_zipf_exponent : float
            Passed to Stage 1; controls skewness of relation frequency.
        v_noise, e_noise : float
            Relative Gaussian noise for |V| and |E| in Stage 2.
        pa_exponent : float
            Preferential-attachment exponent in Stage 2.
        rewire_budget : int
            Number of rewiring attempts in Stage 3.
        initial_temp, cooling_rate : float
            Simulated-annealing parameters for Stage 3.

        Returns
        -------
        igraph.Graph
            Synthetic KG with the same vertex/edge attribute schema as a
            graph loaded by kg_io.load_kg.
        """
        schema = sample_schema(
            self.target.a,
            self.target.c,
            relation_zipf_exponent=relation_zipf_exponent,
            seed=seed,
        )
        g = instantiate(
            schema,
            v_noise=v_noise,
            e_noise=e_noise,
            pa_exponent=pa_exponent,
            seed=seed + 1,
        )
        return refine(
            g,
            self.target.e,
            budget=rewire_budget,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            seed=seed + 2,
        )
