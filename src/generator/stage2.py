"""Stage 2 — CS-aware graph instantiation.

Turns a Schema into an igraph.Graph by
  - using the Schema's |V| and |E| targets,
  - assigning types to entities via the Schema's type_weights,
  - sampling each entity's characteristic set from P(r | type) so the
    co-occurrence structure matches the target,
  - wiring edges with preferential attachment to reproduce heavy-tailed
    in-degree distributions,
  - adding rdf:type edges for all typed entities,
  - bridging isolated components so the graph is (weakly) connected.
"""

import math

import igraph
import numpy as np

from ._adapters import sample_powerlaw, sample_skewnorm_trunc
from ._constants import _RDF_TYPE
from ._logging import get_logger
from .schema import Schema

log = get_logger(__name__)

# ── Tuning constants (Stage-2 wiring) — adjust here ─────────────────────────────
MAX_PAIR_RETRY = 16            # stub-pairing attempts before an edge is dropped
CAP_REDISTRIBUTE_PASSES = 8    # bounded passes when redistributing capped allocations
SIZE_ESCAPE_FAILS = 32         # consecutive template collisions before growing min CS size
TEMPLATE_ATTEMPT_FLOOR = 64    # floor on rejection-sampling attempts per template pool
TEMPLATE_ATTEMPT_FACTOR = 20   # rejection-sampling attempts per requested distinct template
FALLBACK_CS_MEAN_FLOOR = 0.5   # floor on the budget-derived CS-size Poisson mean (no Block D)


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

    # CS size (number of relation slots per entity) comes from the measured cs_size
    # skew-normal; when unavailable, fall back to a budget-derived Poisson mean. Edge
    # counts are NOT set here — the per-relation multinomial allocation below owns the
    # |E| budget, so CS size only sets relation *membership*.
    objects_per_slot = 1.0 / schema.mean_functionality if schema.mean_functionality < 1.0 else 1.0
    fallback_cs_mean = max(
        FALLBACK_CS_MEAN_FLOOR,
        (content_E_target / actual_V if actual_V > 0 else 1.0) / objects_per_slot,
    )

    def _draw_size(size_skew) -> int:
        """Draw one (forward or inverse) CS size from a skew-normal, else budget Poisson."""
        vals = sample_skewnorm_trunc(size_skew, 1, rng)
        size = float(vals[0]) if vals is not None else float(rng.poisson(fallback_cs_mean))
        return max(1, int(round(size)))

    def _cap_redistribute(m: np.ndarray, cap: int, w: np.ndarray) -> None:
        """Cap each count at ``cap`` and redistribute the overflow by weights ``w``.

        Used on both sides: an object takes ≤ |S_r| distinct subjects, a subject
        reaches ≤ |O_r| distinct objects. Bounded passes; tiny residual is dropped.
        """
        if m.size == 0 or cap <= 0:
            return
        for _ in range(CAP_REDISTRIBUTE_PASSES):
            overflow = int(np.maximum(m - cap, 0).sum())
            if overflow == 0:
                break
            np.minimum(m, cap, out=m)
            free = np.where(m < cap)[0]
            if free.size == 0:
                break
            wf = w[free]
            swf = wf.sum()
            pf = wf / swf if swf > 0 else np.full(free.size, 1.0 / free.size)
            m[free] += rng.multinomial(overflow, pf)
        np.minimum(m, cap, out=m)

    log.info(
        "Stage 2: instantiating (seed=%d) V=%d, content-edge target=%d (+%d type edges), "
        "cs_size source=%s", seed, actual_V, content_E_target, n_type_edges,
        "skew-normal" if not math.isnan(schema.cs_size_skew[0]) else "budget-derived",
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

    def _cs_probs(t: int) -> tuple[np.ndarray, int]:
        """Return (normalised relation probabilities, #nonzero) for type t (-1 = untyped).

        Typed entities draw relations from their P(r|t) row; untyped from the global
        relation frequency. Falls back to relation frequency for an empty row.
        """
        probs = schema.type_relation_probs[t].copy() if t >= 0 else schema.relation_weights.copy()
        s = probs.sum()
        if s <= 0:
            probs = schema.relation_weights.copy()
            s = probs.sum()
        if s > 0:
            probs = probs / s
        return probs, int((probs > 0).sum())

    def _sample_cs_for_type(t: int) -> np.ndarray:
        """Draw one forward CS (relation membership) for type t; size from cs_size_skew."""
        if num_relations == 0:
            return np.array([], dtype=int)
        probs, nonzero = _cs_probs(t)
        k = min(nonzero, _draw_size(schema.cs_size_skew))
        if k == 0:
            return np.array([], dtype=int)
        return rng.choice(num_relations, size=k, replace=False, p=probs)

    def _build_distinct(probs: np.ndarray, nonzero: int, size_skew, n_target: int) -> list[np.ndarray]:
        """Rejection-sample up to ``n_target`` DISTINCT relation-sets from ``probs``.

        Sizes come from ``size_skew``; deduping by relation-set steers the distinct-CS
        count (a plain pool collides heavily), and a size-escape raises the minimum size
        once small combos saturate — bounded by the ``nonzero`` support and an attempt
        cap. Used for both forward CS (per-type P(r|t)) and inverse CS (object side,
        relation frequency).
        """
        if num_relations == 0 or n_target <= 0 or nonzero == 0:
            return [np.array([], dtype=int)]
        seen: set[frozenset] = set()
        pool: list[np.ndarray] = []
        attempts = consec_fail = 0
        min_k = 1
        max_attempts = max(TEMPLATE_ATTEMPT_FLOOR, n_target * TEMPLATE_ATTEMPT_FACTOR)
        while len(pool) < n_target and attempts < max_attempts:
            attempts += 1
            k = min(nonzero, max(min_k, _draw_size(size_skew)))
            cs = rng.choice(num_relations, size=k, replace=False, p=probs)
            key = frozenset(int(x) for x in cs)
            if key in seen:
                consec_fail += 1
                if consec_fail >= SIZE_ESCAPE_FAILS and min_k < nonzero:
                    min_k += 1          # small combos saturated → explore larger CSs
                    consec_fail = 0
                continue
            seen.add(key)
            pool.append(cs)
            consec_fail = 0
        return pool

    def _assign_templates(entities: list[int], pool: list[np.ndarray], reuse_zipf: float,
                          target: list) -> None:
        """Assign entities to templates: floor each template at ≥1 entity (so every distinct
        (inverse-)CS is realised), then distribute the rest by a power-law(reuse_zipf) reuse
        tail. Writes the chosen relation-set into ``target[v]``. Empty pool → empty set."""
        n_p = len(pool)
        if n_p == 0:
            for v in entities:
                target[v] = np.array([], dtype=int)
            return
        order = rng.permutation(len(entities))
        fit = sample_powerlaw(reuse_zipf, n_p, rng)
        sfit = fit.sum()
        fit = fit / sfit if sfit > 0 else np.full(n_p, 1.0 / n_p)
        for rank, oi in enumerate(order):
            v = int(entities[oi])
            idx = rank if rank < n_p else int(rng.choice(n_p, p=fit))
            target[v] = pool[idx]

    # --- 3a. Forward CS membership (out-relations per entity) ---
    entity_cs: list = [None] * actual_V
    if schema.cs_num_templates > 0 and num_relations > 0:
        # DISTINCT templates per type (drawn from each type's P(r|t)), sized proportionally.
        type_templates: list[list[np.ndarray]] = []
        for t in range(num_types):
            probs, nz = _cs_probs(t)
            n_t = max(1, round(schema.cs_num_templates * float(schema.type_weights[t])))
            type_templates.append(_build_distinct(probs, nz, schema.cs_size_skew, n_t))
        if num_types > 0:
            buckets: dict[int, list[int]] = {}
            for v in range(actual_V):
                buckets.setdefault(int(entity_types[v]), []).append(v)
            for t in range(num_types):
                _assign_templates(buckets.get(t, []), type_templates[t],
                                  schema.cs_template_zipf, entity_cs)
        else:
            probs, nz = _cs_probs(-1)
            untyped = _build_distinct(probs, nz, schema.cs_size_skew, max(1, schema.cs_num_templates))
            _assign_templates(list(range(actual_V)), untyped, schema.cs_template_zipf, entity_cs)
        used = len({frozenset(int(x) for x in entity_cs[v]) for v in range(actual_V) if len(entity_cs[v])})
        log.info("Stage 2: forward CS (target %d distinct, realised %d)", schema.cs_num_templates, used)
    else:
        log.info("Stage 2: forward CS in per-entity mode (no Block D templates)")
        for v in range(actual_V):
            entity_cs[v] = _sample_cs_for_type(int(entity_types[v]))

    # --- 3b. Inverse CS membership (in-relations per entity), symmetric to forward ---
    # No inverse templates → entity_inv_cs is None → every object is eligible for every
    # relation (today's behaviour) and the a_subj factor stays inert.
    entity_inv_cs: list | None = None
    if schema.inv_cs_num_templates > 0 and num_relations > 0:
        inv_probs = schema.relation_weights.copy()
        s = inv_probs.sum()
        inv_probs = inv_probs / s if s > 0 else np.full(num_relations, 1.0 / num_relations)
        inv_nz = int((inv_probs > 0).sum())
        inv_templates = _build_distinct(inv_probs, inv_nz, schema.inv_cs_size_skew,
                                        max(1, schema.inv_cs_num_templates))
        entity_inv_cs = [None] * actual_V
        _assign_templates(list(range(actual_V)), inv_templates,
                          schema.inv_cs_template_zipf, entity_inv_cs)
        inv_used = len({frozenset(int(x) for x in entity_inv_cs[v])
                        for v in range(actual_V) if len(entity_inv_cs[v])})
        log.info("Stage 2: inverse CS (target %d distinct, realised %d)",
                 schema.inv_cs_num_templates, inv_used)

    # ------------------------------------------------------------------
    # 4. Wire content edges: per-relation multiplicity-then-PA with edge conservation,
    #    matched within S_r × O_r (forward-CS subjects × inverse-CS objects). For each
    #    relation r: allocate |edges_r| across subjects (out-side: power-law(α_obj) tail ×
    #    cs_size^a_obj, floor ≥1, cap at |O_r|) and across objects (in-side: power-law(α_subj)
    #    tail × in_degree^pa × inv_cs_size^a_subj, cap at |S_r|), then pair the stubs.
    # ------------------------------------------------------------------
    in_degrees = np.ones(actual_V, dtype=float)
    # unique_src_count[o] = number of distinct subjects that point to o (any pred)
    # Used as proxy for undirected in-degree under the simplification step.
    unique_src_count = np.zeros(actual_V, dtype=int)
    seen: set[tuple[int, int, str]] = set()
    content_edges: list[tuple[int, int, str]] = []
    # seen_src[o] = set of sources that already point to o (for unique_src_count)
    seen_src: list[set] = [set() for _ in range(actual_V)]

    # Subject pool S_r (forward CS) and object pool O_r (inverse CS) per relation.
    subjects_by_rel: dict[int, list[int]] = {}
    for v, cs in enumerate(entity_cs):
        for rel_idx in cs:
            subjects_by_rel.setdefault(int(rel_idx), []).append(v)
    objects_by_rel: dict[int, list[int]] | None = None
    if entity_inv_cs is not None:
        objects_by_rel = {}
        for v, inv in enumerate(entity_inv_cs):
            for rel_idx in inv:
                objects_by_rel.setdefault(int(rel_idx), []).append(v)

    # Per-relation edge budget over relations that can be wired (subjects present, and
    # objects present when the inverse CS restricts them); renormalised to ~content_E.
    if objects_by_rel is not None:
        present = sorted(r for r in subjects_by_rel if objects_by_rel.get(r))
    else:
        present = sorted(subjects_by_rel)
    if present:
        w_present = np.array([schema.relation_weights[r] for r in present], dtype=float)
        w_sum = w_present.sum()
        w_present = w_present / w_sum if w_sum > 0 else np.full(len(present), 1.0 / len(present))
        edge_budget = {r: int(round(content_E_target * w_present[i])) for i, r in enumerate(present)}
    else:
        edge_budget = {}

    all_objs = np.arange(actual_V)

    def _relation_alpha(skew) -> float:
        """One per-relation exponent drawn from a multiplicity-α skew-normal (NaN → flat)."""
        vals = sample_skewnorm_trunc(skew, 1, rng)
        return float(vals[0]) if vals is not None else float("nan")

    for rel_idx in present:
        S_r = subjects_by_rel[rel_idx]
        O_r = objects_by_rel[rel_idx] if objects_by_rel is not None else None
        obj_ids = np.asarray(O_r, dtype=np.int64) if O_r is not None else all_objs
        n_sr, n_or = len(S_r), int(obj_ids.shape[0])
        # An edge needs a distinct (subject, object) pair → at most |S_r|·|O_r| of them.
        edges_r = min(edge_budget.get(rel_idx, 0), n_sr * n_or)
        if edges_r <= 0 or n_sr == 0 or n_or == 0:
            continue
        predicate = schema.relations[rel_idx]

        # Out-side: edges per subject = power-law(α_obj) tail × cs_size^a_obj (G2b). Floor each
        # subject at ≥1 (object-multiplicity ≥1 when r ∈ CS), distribute the surplus, then cap
        # at |O_r| (a subject reaches ≤ |O_r| distinct objects) + redistribute.
        w_out = sample_powerlaw(_relation_alpha(schema.obj_alpha_skew), n_sr, rng)
        cs_sizes = np.array([len(entity_cs[s]) for s in S_r], dtype=float)
        w_out = w_out * np.power(np.maximum(cs_sizes, 1.0), schema.a_obj)
        sw_out = w_out.sum()
        w_out = w_out / sw_out if sw_out > 0 else np.full(n_sr, 1.0 / n_sr)
        if edges_r >= n_sr:
            m_obj = np.ones(n_sr, dtype=np.int64) + rng.multinomial(edges_r - n_sr, w_out)
        else:
            m_obj = np.zeros(n_sr, dtype=np.int64)
            m_obj[rng.choice(n_sr, size=edges_r, replace=False, p=w_out)] = 1
        _cap_redistribute(m_obj, n_or, w_out)

        # In-side: edges per object (over O_r) = power-law(α_subj) subject-multiplicity tail ×
        # in_degree^pa hub preference × inv_cs_size^a_subj (G2b), masked by max_in_degree, then
        # cap at |S_r| (≤ |S_r| distinct subjects per object) + redistribute. Replaces the old
        # hard inverse-functionality cap: the object-stub multiset *is* the subject-mult law.
        w_in = sample_powerlaw(_relation_alpha(schema.subj_alpha_skew), n_or, rng)
        w_in = w_in * (in_degrees[obj_ids] ** schema.in_pa_exponent)
        if O_r is not None and schema.a_subj != 0.0:
            inv_sizes = np.array([len(entity_inv_cs[o]) for o in O_r], dtype=float)
            w_in = w_in * np.power(np.maximum(inv_sizes, 1.0), schema.a_subj)
        if schema.max_in_degree > 0:
            w_in[unique_src_count[obj_ids] >= schema.max_in_degree] = 0.0
        sw_in = w_in.sum()
        if sw_in <= 0.0:
            continue
        m_in = rng.multinomial(edges_r, w_in / sw_in)
        _cap_redistribute(m_in, n_sr, w_in)

        # Pair subject-stubs with object-stubs within S_r × O_r (configuration model). Each
        # object stub is consumed once (preserving m_in); on a self-loop or duplicate (s, o)
        # we swap in another still-pending object stub (retry) so the edge is re-routed.
        subj_stubs = np.repeat(np.asarray(S_r, dtype=np.int64), m_obj)
        obj_stubs = np.repeat(obj_ids, m_in)
        rng.shuffle(obj_stubs)
        placed_pairs: set[tuple[int, int]] = set()
        n_stubs = min(int(subj_stubs.shape[0]), int(obj_stubs.shape[0]))
        for i in range(n_stubs):
            s = int(subj_stubs[i])
            for attempt in range(MAX_PAIR_RETRY):
                j = i if attempt == 0 else int(rng.integers(i, n_stubs))
                o = int(obj_stubs[j])
                if o == s or (s, o) in placed_pairs:
                    continue
                obj_stubs[i], obj_stubs[j] = obj_stubs[j], obj_stubs[i]  # consume stub at i
                placed_pairs.add((s, o))
                content_edges.append((s, o, predicate))
                seen.add((s, o, predicate))
                in_degrees[o] += 1.0
                if s not in seen_src[o]:
                    seen_src[o].add(s)
                    unique_src_count[o] += 1
                break
            # else: no valid object found within retries → drop this stub (rare)

    # ------------------------------------------------------------------
    log.info("Stage 2: wired %d content edges", len(content_edges))

    # ------------------------------------------------------------------
    # 5. Connectivity guarantee: bridge any isolated components to the giant
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

    log.info(
        "Stage 2: built graph V=%d, E=%d (%d content + %d type)",
        total_V, len(all_edges), len(content_edges), len(type_edges),
    )
    return g
