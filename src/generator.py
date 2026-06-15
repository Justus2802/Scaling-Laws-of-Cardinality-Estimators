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

from signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Lazily discovered mapping: 4-node degree-sequence tuple → igraph motifs_randesu index.
_MOTIF4_IDX: dict[tuple, int] | None = None


def _get_motif4_idx() -> dict[tuple, int]:
    """Discover igraph's 4-node motif index mapping once via small test graphs."""
    global _MOTIF4_IDX
    if _MOTIF4_IDX is not None:
        return _MOTIF4_IDX
    test_cases = [
        ([(0, 1), (1, 2), (2, 3)],                             (1, 1, 2, 2)),  # P4
        ([(0, 1), (0, 2), (0, 3)],                             (1, 1, 1, 3)),  # star K_{1,3}
        ([(0, 1), (1, 2), (2, 3), (3, 0)],                     (2, 2, 2, 2)),  # C4
        ([(0, 1), (1, 2), (2, 0), (0, 3)],                     (1, 2, 2, 3)),  # paw
        ([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],             (2, 2, 3, 3)),  # diamond
        ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],     (3, 3, 3, 3)),  # K4
    ]
    mapping: dict[tuple, int] = {}
    for edges, deg_seq in test_cases:
        g_test = igraph.Graph(n=4, edges=edges)
        counts = g_test.motifs_randesu(size=4)
        for idx, c in enumerate(counts):
            if c == 1:
                mapping[deg_seq] = idx
                break
    _MOTIF4_IDX = mapping
    return mapping


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
    # Block D-derived CS structure (defaults = legacy behaviour)
    cs_size_mean: float = 0.0       # 0 → derive from E/V budget at instantiate time
    cs_num_templates: int = 0       # 0 → per-entity independent sampling
    cs_template_zipf: float = 2.0   # Zipf exponent for template frequency
    # Block B-derived edge multiplicity and degree distribution
    mean_functionality: float = 1.0      # 1.0 → single object per (s,p) pair
    in_pa_exponent: float = 0.5          # PA exponent for object selection → in-degree shape
    mean_inv_functionality: float = 1.0  # 1.0 → no cap on subjects per (predicate, object)
    max_in_degree: int = 0               # 0 → uncapped; limits hub formation


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


def _build_type_rel_probs_from_measured(
    type_relation_conditional: dict[str, dict[str, float]],
    num_types: int,
    num_relations: int,
    relation_weights: np.ndarray,
) -> np.ndarray:
    """Build P(r|t) from measured type_relation_conditional via rank mapping.

    Sorts real types by activity (descending) and real relations by aggregate
    frequency (descending), then maps them positionally onto schema indices.
    This preserves co-occurrence structure without requiring URI alignment.
    """
    if not type_relation_conditional or num_types == 0 or num_relations == 0:
        return np.tile(relation_weights, (num_types, 1))

    real_types = sorted(
        type_relation_conditional,
        key=lambda t: sum(type_relation_conditional[t].values()),
        reverse=True,
    )
    rel_agg: dict[str, float] = {}
    for probs in type_relation_conditional.values():
        for rel, p in probs.items():
            rel_agg[rel] = rel_agg.get(rel, 0.0) + p
    real_rels = sorted(rel_agg, key=lambda r: rel_agg[r], reverse=True)

    P = np.zeros((num_types, num_relations), dtype=float)
    for t_idx in range(num_types):
        real_t = real_types[t_idx % len(real_types)]
        src = type_relation_conditional[real_t]
        for r_idx in range(num_relations):
            real_r = real_rels[r_idx % len(real_rels)]
            P[t_idx, r_idx] = src.get(real_r, 0.0)

    row_sums = P.sum(axis=1, keepdims=True)
    zero_mask = (row_sums.ravel() == 0)
    P[zero_mask] = relation_weights
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
    d: BlockD = None,
    b: BlockB = None,
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
    d : BlockD, optional
        Characteristic-set statistics.  When provided, Stage 2 will use
        d.cs_size_mean and d.num_distinct_cs to build a realistic pool of
        reusable CS templates instead of sampling every entity independently.
        This fixes the co-occurrence density and num_distinct_cs deviations.
    b : BlockB, optional
        Degree-structure statistics.  When provided, the mean relation
        functionality is used to sample more than one object per (s,p) pair,
        matching the target's edge multiplicity.
    relation_zipf_exponent : float
        Zipf exponent for relation frequency weights.  Controls how skewed
        relation usage is; real KGs typically fall in [1.5, 2.5].
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
    # Use measured P(r|t) directly when available; fall back to low-rank synthesis.
    try:
        trc = c.type_relation_conditional
    except Exception:
        trc = {}
    if trc:
        type_relation_probs = _build_type_rel_probs_from_measured(
            trc, num_types, num_relations, relation_weights,
        )
    else:
        type_relation_probs = _sample_type_relation_probs(
            num_types, num_relations, relation_weights, c.subj_singular_values, rng,
        )

    # --- CS structure from Block D ---
    if d is not None and d.cs_size_mean > 0:
        cs_size_mean = float(d.cs_size_mean)
        cs_num_templates = max(1, int(d.num_distinct_cs))
        cs_template_zipf = (
            float(d.cs_freq_stats.alpha)
            if not math.isnan(d.cs_freq_stats.alpha) else 2.0
        )
    else:
        cs_size_mean = 0.0   # signal instantiate to derive from E/V budget
        cs_num_templates = 0
        cs_template_zipf = 2.0

    # --- Edge multiplicity, PA exponent, inverse functionality from Block B ---
    if b is not None and b.functionality:
        mean_functionality = float(np.clip(np.mean(list(b.functionality.values())), 0.1, 1.0))
    else:
        mean_functionality = 1.0

    if b is not None:
        alpha_in = b.in_degree_fit.alpha
        # Dorogovtsev-Mendes relation: α = 2 + 1/β → β = 1/(α−2)
        # α must be > 2 for a finite-mean power law; clamp β to [0.1, 2.0].
        if not math.isnan(alpha_in) and alpha_in > 2.0:
            in_pa_exponent = float(np.clip(1.0 / (alpha_in - 2.0), 0.1, 2.0))
        else:
            in_pa_exponent = 0.5
        # Expected maximum in-degree: n^(1/(α−1)) (extreme-value statistic)
        n_ent = a.num_entities
        if not math.isnan(alpha_in) and alpha_in > 1.1 and n_ent > 0:
            max_in_degree = max(10, int(round(n_ent ** (1.0 / (alpha_in - 1.0)))))
        else:
            max_in_degree = 0
        inv_funcs = b.inverse_functionality
        mean_inv_functionality = (
            float(np.clip(np.mean(list(inv_funcs.values())), 0.01, 1.0))
            if inv_funcs else 1.0
        )
    else:
        in_pa_exponent = 0.5
        mean_inv_functionality = 1.0
        max_in_degree = 0

    return Schema(
        relations=relations,
        relation_weights=relation_weights,
        types=types,
        type_weights=type_weights,
        type_relation_probs=type_relation_probs,
        num_entities=a.num_entities,
        num_triples=a.num_triples,
        cs_size_mean=cs_size_mean,
        cs_num_templates=cs_num_templates,
        cs_template_zipf=cs_template_zipf,
        mean_functionality=mean_functionality,
        in_pa_exponent=in_pa_exponent,
        mean_inv_functionality=mean_inv_functionality,
        max_in_degree=max_in_degree,
    )


# ---------------------------------------------------------------------------
# Stage 2 — CS-aware graph instantiation
# ---------------------------------------------------------------------------


def _connect_components(
    content_edges: list,
    actual_V: int,
    schema: "Schema",
    rng: "np.random.Generator",
    seen: set,
    in_degrees: "np.ndarray",
) -> None:
    """Bridge isolated entity components into one weakly connected component.

    Adds one directed edge from each isolated component to the largest
    component (giant).  Uses manual BFS to avoid igraph cluster API overhead.
    """
    if actual_V < 2:
        return
    # Build undirected adjacency for connectivity check
    adj: list[list[int]] = [[] for _ in range(actual_V)]
    for s, o, _ in content_edges:
        if s < actual_V and o < actual_V and s != o:
            adj[s].append(o)
            adj[o].append(s)

    visited = [False] * actual_V
    comps: list[list[int]] = []
    for start in range(actual_V):
        if not visited[start]:
            comp: list[int] = []
            stack = [start]
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                comp.append(v)
                stack.extend(u for u in adj[v] if not visited[u])
            comps.append(comp)

    if len(comps) <= 1:
        return

    giant = max(comps, key=len)
    for comp in comps:
        if comp is giant:
            continue
        src = comp[0]
        # Pick a random node in the giant to avoid creating a star hub
        bridge = giant[int(rng.integers(len(giant)))]
        pred = schema.relations[int(rng.integers(len(schema.relations)))]
        triple = (src, bridge, pred)
        if triple not in seen:
            seen.add(triple)
            content_edges.append(triple)
            in_degrees[bridge] += 1.0


def instantiate(
    schema: Schema,
    *,
    seed: int = 0,
) -> igraph.Graph:
    """Stage 2: instantiate a KG from a Schema.

    Parameters
    ----------
    schema : Schema
        Output of Stage 1 (sample_schema).
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
    their combined count matches the schema's num_triples target exactly.
    """
    rng = np.random.default_rng(seed)
    num_relations = len(schema.relations)
    num_types = len(schema.types)

    # ------------------------------------------------------------------
    # 1. Use schema targets exactly — no noise
    # ------------------------------------------------------------------
    actual_V = max(2, schema.num_entities)
    actual_E_target = max(1, schema.num_triples)

    # rdf:type edges (one per entity when types exist) come out of the budget
    n_type_edges = actual_V if num_types > 0 else 0
    content_E_target = max(0, actual_E_target - n_type_edges)

    effective_cs_size = (
        schema.cs_size_mean if schema.cs_size_mean > 0
        else (content_E_target / actual_V if actual_V > 0 else 1.0)
    )

    # ------------------------------------------------------------------
    # 2. Assign a type to every entity
    # ------------------------------------------------------------------
    if num_types > 0:
        entity_types = rng.choice(num_types, size=actual_V, p=schema.type_weights)
    else:
        entity_types = np.full(actual_V, -1, dtype=int)

    # ------------------------------------------------------------------
    # 3. Sample characteristic sets (CS) for all entities
    #
    #  Template mode (Block D available):
    #    Build schema.cs_num_templates reusable CS templates per type,
    #    then assign each entity to a template via Zipf weights.  This
    #    reproduces the target num_distinct_cs and co-occurrence sparsity.
    #
    #  Legacy mode (no Block D):
    #    Sample each entity's CS independently (original behaviour).
    # ------------------------------------------------------------------

    def _sample_cs_for_type(t: int, size: float) -> np.ndarray:
        """Draw one CS from the distribution appropriate for type t."""
        if num_relations == 0:
            return np.array([], dtype=int)
        if t >= 0:
            probs = schema.type_relation_probs[t].copy()
        else:
            probs = schema.relation_weights.copy()
        s = probs.sum()
        if s <= 0:
            probs = schema.relation_weights.copy()
            s = probs.sum()
        probs /= s
        nonzero = int((probs > 0).sum())
        k = min(nonzero, max(0, int(rng.poisson(max(0.1, size)))))
        if k == 0:
            return np.array([], dtype=int)
        return rng.choice(num_relations, size=k, replace=False, p=probs)

    entity_cs: list[np.ndarray] = []

    if schema.cs_num_templates > 0 and num_relations > 0:
        # --- Template-based CS sampling ---
        # Generate a pool of templates per type, sized proportionally.
        type_templates: list[list[np.ndarray]] = []
        for t in range(num_types):
            n_t = max(1, round(schema.cs_num_templates * float(schema.type_weights[t])))
            type_templates.append([
                _sample_cs_for_type(t, effective_cs_size) for _ in range(n_t)
            ])
        # Template pool for untyped entities (or when num_types == 0)
        untyped_templates = [
            _sample_cs_for_type(-1, effective_cs_size)
            for _ in range(max(1, schema.cs_num_templates))
        ]

        # Zipf weights for selecting a template (popular templates reused more)
        def _zipf_pick(pool: list[np.ndarray]) -> np.ndarray:
            n = len(pool)
            if n == 1:
                return pool[0]
            ranks = np.arange(1, n + 1, dtype=float)
            w = ranks ** (-schema.cs_template_zipf)
            w /= w.sum()
            return pool[int(rng.choice(n, p=w))]

        for v in range(actual_V):
            t = int(entity_types[v])
            pool = type_templates[t] if (t >= 0 and num_types > 0) else untyped_templates
            entity_cs.append(_zipf_pick(pool))
    else:
        # --- Legacy: per-entity independent sampling ---
        for v in range(actual_V):
            entity_cs.append(_sample_cs_for_type(int(entity_types[v]), effective_cs_size))

    # ------------------------------------------------------------------
    # 4. Wire content edges with preferential attachment
    #    Object picked proportional to current in_degree ^ in_pa_exponent
    #    (derived from Block B's in-degree power-law fit).
    #    Laplace smoothing (start at 1) ensures every vertex is reachable.
    #
    #    When mean_functionality < 1, each (s, p) slot produces more than
    #    one object on average — matching Block B's edge multiplicity.
    #    Inverse functionality cap: limits how many subjects can share the
    #    same (predicate, object) pair, from Block B's inverse_functionality.
    # ------------------------------------------------------------------
    in_degrees = np.ones(actual_V, dtype=float)
    # unique_src_count[o] = number of distinct subjects that point to o (any pred)
    # Used as proxy for undirected in-degree under the simplification step.
    unique_src_count = np.zeros(actual_V, dtype=int)
    seen: set[tuple[int, int, str]] = set()
    content_edges: list[tuple[int, int, str]] = []
    # seen_src[o] = set of sources that already point to o (for unique_src_count)
    seen_src: list[set] = [set() for _ in range(actual_V)]

    # Inverse functionality cap: only active when mean_inv_func < 0.7 AND
    # ceil(1/x) >= 2, so the cap actually allows sharing.
    # (round(1/0.77) = 1 would block all sharing; that bug is fixed here.)
    max_subj_per_po: int | None = None
    if schema.mean_inv_functionality < 0.7:
        cap = math.ceil(1.0 / schema.mean_inv_functionality)
        if cap >= 2:
            max_subj_per_po = cap
    po_subj_counts: dict[tuple[str, int], int] = {}

    for v, cs in enumerate(entity_cs):
        for rel_idx in cs:
            predicate = schema.relations[int(rel_idx)]
            # Geometric(p) has mean 1/p; p = mean_functionality gives
            # mean 1/mean_functionality objects per (subject, predicate) slot.
            n_obj = int(rng.geometric(schema.mean_functionality)) if schema.mean_functionality < 1.0 else 1
            for _ in range(n_obj):
                weights = in_degrees ** schema.in_pa_exponent
                # Cap on unique sources to match the undirected in-degree target
                if schema.max_in_degree > 0:
                    weights[unique_src_count >= schema.max_in_degree] = 0.0
                weights[v] = 0.0
                total = weights.sum()
                if total == 0.0:
                    break
                weights /= total
                obj = int(rng.choice(actual_V, p=weights))
                # Skip if this (predicate, object) pair already has too many subjects
                if max_subj_per_po is not None:
                    po_key = (predicate, obj)
                    if po_subj_counts.get(po_key, 0) >= max_subj_per_po:
                        continue
                    po_subj_counts[po_key] = po_subj_counts.get(po_key, 0) + 1
                triple = (v, obj, predicate)
                if triple not in seen:
                    seen.add(triple)
                    content_edges.append(triple)
                    in_degrees[obj] += 1.0
                    if v not in seen_src[obj]:
                        seen_src[obj].add(v)
                        unique_src_count[obj] += 1

    # ------------------------------------------------------------------
    # 5. Throttle content edges down to budget if over target
    # ------------------------------------------------------------------
    if len(content_edges) > content_E_target > 0:
        keep = rng.choice(len(content_edges), size=content_E_target, replace=False)
        keep_set = set(keep.tolist())
        content_edges = [e for i, e in enumerate(content_edges) if i in keep_set]

    # ------------------------------------------------------------------
    # 5b. Connectivity guarantee: bridge any isolated components to the giant
    # ------------------------------------------------------------------
    _connect_components(content_edges, actual_V, schema, rng, seen, in_degrees)

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
    target_f: "BlockF | None" = None,
    budget: int = 10_000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.999,
    remeasure_interval: int = 200,
    seed: int = 0,
) -> igraph.Graph:
    """Stage 3: Maslov-Sneppen rewiring + simulated annealing.

    Rewires content edges (never rdf:type edges) using degree-preserving
    double-edge swaps.  The SA objective is a weighted sum of relative errors
    across multiple targets:

    * Triangle count (exact, incremental via _triangle_delta)
    * 4-node motif counts — C4, diamond, K4, paw (remeasured every
      ``remeasure_interval`` accepted swaps via igraph.motifs_randesu)
    * Degree assortativity (exact, incremental — degree sequence is invariant
      under double-edge swaps, so only the cross-product sum Q changes)

    Parameters
    ----------
    g : igraph.Graph
        Output of Stage 2 (instantiate).
    target_e : BlockE
        Block E signature — supplies triangle_count and 4-node motif targets.
    target_f : BlockF, optional
        Block F signature — supplies degree_assortativity target.
    budget : int
        Maximum number of rewiring attempts.
    initial_temp : float
        Starting SA temperature.
    cooling_rate : float
        Geometric decay per accepted swap.
    remeasure_interval : int
        Accepted-swap interval between full 4-node motif remeasurements.
    seed : int
        RNG seed.

    Returns
    -------
    igraph.Graph
        Best graph encountered during the annealing walk.
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ setup
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

    rel_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, p) in enumerate(content_edge_data):
        rel_to_idxs[p].append(i)

    swappable_rels = [r for r, lst in rel_to_idxs.items() if len(lst) >= 2]
    if not swappable_rels:
        return g

    n = g.vcount()
    adj: list[dict] = [{} for _ in range(n)]
    for s, o, _ in content_edge_data:
        _adj_inc(adj, s, o)

    # -------------------------------------------------------- triangle counter
    def _count_triangles() -> int:
        total = 0
        for u, nbrs in enumerate(adj):
            for v in nbrs:
                if v > u:
                    total += len(set(adj[u]) & set(adj[v]))
        return total // 3

    target_tri = int(target_e.triangle_count)
    current_tri = _count_triangles()

    # ----------------------------------------- 4-node motif targets & counter
    _motif4_targets: dict[tuple, int] = {}
    for deg_seq, attr in [
        ((2, 2, 2, 2), "four_cycle_count"),
        ((2, 2, 3, 3), "diamond_count"),
        ((3, 3, 3, 3), "k4_count"),
        ((1, 2, 2, 3), "tailed_triangle_count"),
    ]:
        val = getattr(target_e, attr, 0)
        if val and val > 0:
            _motif4_targets[deg_seq] = int(val)

    def _measure_motifs4() -> dict[tuple, int]:
        """Rebuild a temporary undirected igraph graph and count 4-node motifs."""
        edge_set: set[tuple[int, int]] = set()
        und_edges: list[tuple[int, int]] = []
        for s, o, _ in content_edge_data:
            key = (min(s, o), max(s, o))
            if key not in edge_set:
                edge_set.add(key)
                und_edges.append(key)
        g_tmp = igraph.Graph(n=n)
        if und_edges:
            g_tmp.add_edges(und_edges)
        idx_map = _get_motif4_idx()
        raw = g_tmp.motifs_randesu(size=4)
        result: dict[tuple, int] = {}
        for ds, idx in idx_map.items():
            c = raw[idx] if idx < len(raw) else 0
            result[ds] = 0 if (isinstance(c, float) and math.isnan(c)) else int(c)
        return result

    current_motifs4 = _measure_motifs4() if _motif4_targets else {}

    # ------------------------------------------------- assortativity tracking
    # Double-edge swaps preserve degree sequences, so only the cross-product
    # sum Q = Σ_e d_u*d_v changes.  S and T are constant throughout the walk.
    target_r = float(target_f.degree_assortativity) if target_f is not None else float("nan")
    use_assort = not math.isnan(target_r)

    und_deg = [0] * n
    for s, o, _ in content_edge_data:
        und_deg[s] += 1
        und_deg[o] += 1
    M_e = len(content_edge_data)
    S_deg = float(sum(und_deg[s] + und_deg[o] for s, o, _ in content_edge_data))
    T_deg = float(sum(und_deg[s] ** 2 + und_deg[o] ** 2 for s, o, _ in content_edge_data))
    Q_deg = float(sum(und_deg[s] * und_deg[o] for s, o, _ in content_edge_data))

    def _assort_from_Q(Q: float) -> float:
        denom = M_e * T_deg - S_deg ** 2 / 2.0
        if denom == 0.0:
            return 0.0
        return (2.0 * M_e * Q - S_deg ** 2 / 2.0) / denom

    # ------------------------------------------------------- loss function
    def _loss(tri: int, motifs: dict, Q: float) -> float:
        loss = abs(tri - target_tri) / max(1, target_tri)
        for ds, tgt in _motif4_targets.items():
            loss += abs(motifs.get(ds, 0) - tgt) / tgt
        if use_assort:
            loss += abs(_assort_from_Q(Q) - target_r)
        return loss

    current_loss = _loss(current_tri, current_motifs4, Q_deg)
    best_loss = current_loss
    best_content = list(content_edge_data)

    # -------------------------------------------- triangle-targeting indexes
    # edge_tgt[o] = set of content_edge_data indices whose target is o
    # edge_src_by_pred[p][s] = list of indices with source s and predicate p
    # Sources never change in a double-edge swap, so edge_src_by_pred is static.
    # Only edge_tgt needs updating after each accepted swap.
    edge_tgt: dict[int, set[int]] = {}
    edge_src_by_pred: dict[str, dict[int, list[int]]] = {}
    for i, (s, o, p) in enumerate(content_edge_data):
        edge_tgt.setdefault(o, set()).add(i)
        edge_src_by_pred.setdefault(p, {}).setdefault(s, []).append(i)

    def _targeted_swap():
        """Find a swap that closes an open wedge (u–w–v with no u–v edge).

        Picks a random node w with ≥2 neighbours, finds two unconnected
        neighbours u and v, then looks for edges (u→x, y→v) with the same
        predicate.  After the swap u→v is created, closing the triangle u–v–w.
        Returns (i1, i2, s1, o1, s2, o2, p1) or None if no candidate found.
        """
        w = int(rng.integers(n))
        nbrs = list(adj[w].keys())
        if len(nbrs) < 2:
            return None
        pi, pj = rng.choice(len(nbrs), size=2, replace=False)
        u_t, v_t = nbrs[pi], nbrs[pj]
        if v_t in adj[u_t]:
            return None  # already connected; no new triangle
        # Want: (u_t→o1, s2→v_t) same pred → swap → (u_t→v_t, s2→o1)
        tgt_pool = list(edge_tgt.get(v_t, ()))
        if not tgt_pool:
            return None
        i2 = tgt_pool[int(rng.integers(len(tgt_pool)))]
        s2, _, pred = content_edge_data[i2]
        src_pool = edge_src_by_pred.get(pred, {}).get(u_t, [])
        if not src_pool:
            return None
        i1 = src_pool[int(rng.integers(len(src_pool)))]
        s1, o1, p1 = content_edge_data[i1]
        o2 = v_t
        if s1 == o2 or s2 == o1 or i1 == i2:
            return None
        return i1, i2, s1, o1, s2, o2, p1

    temp = initial_temp
    accepted = 0

    for _ in range(budget):
        # Attempt targeted triangle-creating swap when triangles are below target.
        # The probability scales with how large the deficit is (max 50%).
        tri_deficit = target_tri - current_tri
        p_targeted = float(min(0.5, tri_deficit / max(1, target_tri)))
        targeted = tri_deficit > 0 and rng.random() < p_targeted
        if targeted:
            result = _targeted_swap()
            if result is None:
                targeted = False
        if not targeted:
            rel = swappable_rels[int(rng.integers(len(swappable_rels)))]
            pool = rel_to_idxs[rel]
            pi1, pi2 = rng.choice(len(pool), size=2, replace=False)
            i1, i2 = pool[pi1], pool[pi2]
            s1, o1, p1 = content_edge_data[i1]
            s2, o2, _  = content_edge_data[i2]
            if s1 == o2 or s2 == o1:
                continue
        else:
            i1, i2, s1, o1, s2, o2, p1 = result

        tri_delta = _triangle_delta(adj, s1, o1, s2, o2)
        new_tri = current_tri + tri_delta
        # Assortativity delta: only Q changes (degrees are invariant)
        dQ = float(und_deg[s1] * und_deg[o2] + und_deg[s2] * und_deg[o1]
                   - und_deg[s1] * und_deg[o1] - und_deg[s2] * und_deg[o2])
        new_Q = Q_deg + dQ
        new_loss = _loss(new_tri, current_motifs4, new_Q)

        if new_loss < current_loss:
            accept = True
        else:
            diff = new_loss - current_loss
            accept = bool(rng.random() < math.exp(-diff / max(temp, 1e-10)))

        if accept:
            content_edge_data[i1] = (s1, o2, p1)
            content_edge_data[i2] = (s2, o1, p1)

            # Update edge_tgt: target of i1 changes o1→o2; target of i2 changes o2→o1
            edge_tgt.setdefault(o1, set()).discard(i1)
            edge_tgt.setdefault(o2, set()).add(i1)
            edge_tgt.setdefault(o2, set()).discard(i2)
            edge_tgt.setdefault(o1, set()).add(i2)

            _adj_dec(adj, s1, o1)
            _adj_dec(adj, s2, o2)
            _adj_inc(adj, s1, o2)
            _adj_inc(adj, s2, o1)

            current_tri = new_tri
            Q_deg = new_Q
            current_loss = new_loss
            temp *= cooling_rate
            accepted += 1

            # Periodically remeasure 4-node motifs (they have no cheap delta)
            if _motif4_targets and accepted % remeasure_interval == 0:
                current_motifs4 = _measure_motifs4()
                current_loss = _loss(current_tri, current_motifs4, Q_deg)

            if current_loss < best_loss:
                best_loss = current_loss
                best_content = list(content_edge_data)

    # Re-connect any components that SA swaps may have disconnected
    seen_best: set[tuple[int, int, str]] = set(best_content)
    in_deg_best = np.ones(n, dtype=float)
    for s, o, _ in best_content:
        in_deg_best[o] += 1.0
    # Borrow a schema-like object to pass relations; extract from type_edge_data or content
    all_preds = list({p for _, _, p in best_content if p != _RDF_TYPE})
    if not all_preds:
        all_preds = ["http://kgsynth.org/rel/0"]

    class _FakeSchema:
        relations = all_preds

    _connect_components(best_content, n, _FakeSchema(), rng, seen_best, in_deg_best)

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
    """Target signature used by Generator (Blocks A, B, C, D, E).

    Block A supplies size/density targets; Block B supplies edge multiplicity
    and degree-distribution PA exponents; Block C supplies schema/class
    structure; Block D supplies CS statistics that enable template-based CS
    reuse; Block E supplies motif counts that Stage 3 optimises toward; Block F
    supplies degree assortativity that Stage 3 also targets.
    """

    a: "BlockA"
    c: "BlockC"
    e: "BlockE"
    b: "BlockB | None" = None   # optional: enables multi-object edges + data-driven PA
    d: "BlockD | None" = None   # optional: enables CS template reuse
    f: "BlockF | None" = None   # optional: enables assortativity targeting in Stage 3

    @classmethod
    def from_graph(cls, g: igraph.Graph) -> "Signature":
        return cls(
            a=BlockA().calculate(g),
            b=BlockB().calculate(g),
            c=BlockC().calculate(g),
            d=BlockD().calculate(g),
            e=BlockE().calculate(g),
            f=BlockF().calculate(g),
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
        rewire_budget: int = 50_000,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.9999,
    ) -> igraph.Graph:
        """Generate one synthetic KG from the target signature.

        Parameters
        ----------
        seed : int
            Master seed; all three stages derive sub-seeds from it so the
            entire pipeline is reproducible from a single integer.
        relation_zipf_exponent : float
            Passed to Stage 1; controls skewness of relation frequency.
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
            d=self.target.d,
            b=self.target.b,
            relation_zipf_exponent=relation_zipf_exponent,
            seed=seed,
        )
        g = instantiate(schema, seed=seed + 1)
        return refine(
            g,
            self.target.e,
            target_f=self.target.f,
            budget=rewire_budget,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            seed=seed + 2,
        )
