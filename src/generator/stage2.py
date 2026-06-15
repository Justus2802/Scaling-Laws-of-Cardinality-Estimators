"""Stage 2 — CS-aware graph instantiation.

Turns a Schema into an igraph.Graph by
  - using the Schema's |V| and |E| targets,
  - assigning types to entities via the Schema's type_weights,
  - sampling each entity's characteristic set from P(r | type) so the
    co-occurrence structure matches the target,
  - wiring edges with preferential attachment to reproduce heavy-tailed
    in-degree distributions,
  - adding rdf:type edges for all typed entities,
  - throttling content edges down to the sampled |E| budget if needed.
"""

import math

import igraph
import numpy as np

from ._constants import _RDF_TYPE
from .schema import Schema


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

    # Target edges per entity divided by average objects per CS slot.
    # cs_size_mean counts relation slots; each slot emits geometric(mean_func)
    # objects on average, so we scale up to hit the edge budget.
    edges_per_entity = content_E_target / actual_V if actual_V > 0 else 1.0
    objects_per_slot = 1.0 / schema.mean_functionality if schema.mean_functionality < 1.0 else 1.0
    effective_cs_size = max(0.1, edges_per_entity / objects_per_slot)

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
