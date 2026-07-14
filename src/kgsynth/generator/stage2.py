"""Stage 2 — CS-aware graph instantiation.

Turns a Schema into an igraph.Graph by
  - using the Schema's |V| and |E| targets,
  - deriving each entity's type from its realised characteristic set (post-hoc
    argmax over P(r|t)),
  - sampling each entity's characteristic set from its co-occurrence group
    prototype (Block C subj_cooc_exp/obj_cooc_exp) so the co-occurrence
    structure matches the target, reusing a pool of Block-D-sized CS templates,
  - wiring edges toward per-entity target degrees sampled from the measured
    degree distribution,
  - adding rdf:type edges for all typed entities,
  - selectively bridging isolated components to match target num_components / LCC fraction.
"""

from collections import defaultdict, deque

import igraph
import numpy as np

from ._adapters import (
    _DEGSEQ_TAIL_FRACTION as DEGSEQ_TAIL_FRACTION,
    repair_degree_sum,
    sample_powerlaw_trunc,
    sample_quantiles_trunc,
)
from ._constants import _RDF_TYPE
from ._ipf import build_support, fit_stubs, solve_edge_budget
from .._logging import get_logger
from .schema import Schema

log = get_logger(__name__)

# ── Tuning constants (Stage-2 wiring) — adjust here ─────────────────────────────
# (DEGREE_QUOTA_SLACK is gone: it traded deficit-recovery volume against degree fidelity,
#  and the IPF allocation leaves no deficit to trade against.)
MAX_PAIR_RETRY = 16            # stub-pairing attempts before an edge is dropped
SIZE_ESCAPE_FAILS = 32         # consecutive template collisions before growing min CS size
TEMPLATE_ATTEMPT_FLOOR = 64    # floor on rejection-sampling attempts per template pool
TEMPLATE_ATTEMPT_FACTOR = 20   # rejection-sampling attempts per requested distinct template


def _connect_components(
    content_edges: list,
    actual_V: int,
    schema: "Schema",
    rng: "np.random.Generator",
    seen: set,
    in_degrees: "np.ndarray",
    target_nc: int = 1,
    target_lcc: float = 1.0,
    objects_by_rel: "dict | None" = None,
) -> "np.ndarray":
    """Bridge isolated entity components, targeting a specific component count and LCC fraction.

    Selects satellites (components left disconnected) so that their combined
    size is as close as possible to ``(1 - target_lcc) * actual_V``, subject
    to keeping at most ``target_nc - 1`` satellites.  All remaining components
    are bridged to the giant with one directed edge each.

    Uses manual BFS to avoid igraph cluster API overhead.

    Parameters
    ----------
    target_nc : int
        Desired number of weakly-connected components.  1 → fully connect.
    target_lcc : float
        Desired fraction of entity nodes in the largest component.

    Returns
    -------
    np.ndarray
        Boolean mask, length ``actual_V``, True for nodes left in a
        deliberately-unbridged satellite component. Callers that add further
        edges after this (deficit recovery) must not touch these nodes — doing
        so would silently reconnect a satellite and undo the ``target_nc`` /
        ``target_lcc`` guarantee established here.
    """
    is_satellite = np.zeros(actual_V, dtype=bool)
    if actual_V < 2:
        return is_satellite

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
        return is_satellite

    comps.sort(key=len, reverse=True)
    giant = comps[0]
    satellites_to_keep: set[int] = set()

    max_satellites = target_nc - 1
    if max_satellites > 0 and len(comps) > 1:
        # Sort satellite candidates smallest-first: prefix sums over the j smallest
        # give fine-grained control and match typical KG structure (many tiny isolates).
        sats_asc = sorted(comps[1:], key=len)
        k = min(max_satellites, len(sats_asc))
        sat_budget = (1.0 - target_lcc) * actual_V

        prefix = [0] * (k + 1)
        for i, s in enumerate(sats_asc[:k]):
            prefix[i + 1] = prefix[i] + len(s)

        # j* minimises |prefix[j] - sat_budget| — "as near as possible".
        best_j = min(range(k + 1), key=lambda j: abs(prefix[j] - sat_budget))

        if best_j == 0 and sat_budget > 0:
            log.warning(
                "_connect_components: target_nc=%d target_lcc=%.4f but only %d natural "
                "components available — nc will be 1",
                target_nc, target_lcc, len(comps),
            )
        satellites_to_keep = {id(c) for c in sats_asc[:best_j]}
        for c in sats_asc[:best_j]:
            for v in c:
                is_satellite[v] = True

    for comp in comps[1:]:
        if id(comp) in satellites_to_keep:
            continue
        src = comp[0]
        bridge = giant[int(rng.integers(len(giant)))]
        # Respect inv-CS: only use a relation that bridge is eligible to receive.
        if objects_by_rel is not None:
            eligible = [r for r in range(len(schema.relations))
                        if bridge in (objects_by_rel.get(r) or [])]
            if not eligible:
                eligible = list(range(len(schema.relations)))
            rel_idx = eligible[int(rng.integers(len(eligible)))]
        else:
            rel_idx = int(rng.integers(len(schema.relations)))
        pred = schema.relations[rel_idx]
        triple = (src, bridge, pred)
        if triple not in seen:
            seen.add(triple)
            content_edges.append(triple)
            in_degrees[bridge] += 1.0

    return is_satellite


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

    # rdf:type edges come out of the budget, sized by the measured rdf:type share of E
    # (Block A's type_edge_frac) rather than assumed to be one per entity — a target
    # graph may type only some of its entities. Capped at |V| since Stage 2 gives an
    # entity at most one type.
    n_type_edges = (min(actual_V, int(round(actual_E_target * schema.type_edge_frac)))
                    if num_types > 0 else 0)
    content_E_target = max(0, actual_E_target - n_type_edges)

    # CS size (number of relation slots per entity) comes from the measured cs_size
    # quantile fit. Edge counts are NOT set here — the per-relation multinomial
    # allocation below owns the |E| budget, so CS size only sets relation *membership*.
    def _draw_size(size_q) -> int:
        """Draw one (forward or inverse) CS size from a quantile fit."""
        return max(1, int(round(float(sample_quantiles_trunc(size_q, 1, rng)[0]))))

    log.info(
        "Stage 2: instantiating (seed=%d) V=%d, content-edge target=%d (+%d type edges)",
        seed, actual_V, content_E_target, n_type_edges,
    )

    # ------------------------------------------------------------------
    # 2. Allocate the entity->type map (all untyped).
    #    Types are *derived from* the realised CS, post-hoc, in step 5b below —
    #    the co-occurrence-group CS path never reads a type. Entities stay -1
    #    (untyped) when T=0, which is also the final state in that case.
    # ------------------------------------------------------------------
    entity_types = np.full(actual_V, -1, dtype=int)

    # ------------------------------------------------------------------
    # 3. Sample characteristic sets (CS) for all entities.
    #    Build schema.cs_num_templates reusable CS templates per co-occurrence
    #    group, then assign each entity to a template via Zipf weights.  This
    #    reproduces the target num_distinct_cs and co-occurrence sparsity.
    # ------------------------------------------------------------------

    def _allocate_quotas(weights: np.ndarray, total: int) -> list[int]:
        """Largest-remainder integer allocation: distribute ``total`` slots among groups
        proportional to ``weights``, with a floor of 1 per group, summing exactly to
        ``max(total, len(weights))``.  Prevents the old max(1,round(total*w_g)) per-group
        pattern from inflating the total when many groups have small weights.
        """
        n = len(weights)
        if n == 0:
            return []
        budget = max(total, n)  # at least 1 per group
        # Start with floor(budget * w_g) per group, then give leftover slots to the
        # groups with the largest fractional parts (largest-remainder method).
        raw = np.asarray(weights, dtype=float) * budget
        floors = np.maximum(1, np.floor(raw).astype(int))
        leftover = budget - int(floors.sum())
        if leftover > 0:
            fracs = raw - np.floor(raw)
            order = np.argsort(fracs)[::-1]
            for i in range(min(leftover, n)):
                floors[order[i]] += 1
        return floors.tolist()

    def _build_distinct(probs: np.ndarray, nonzero: int, size_q, n_target: int) -> list[np.ndarray]:
        """Rejection-sample up to ``n_target`` DISTINCT relation-sets from ``probs``.

        Sizes come from ``size_q``; deduping by relation-set steers the distinct-CS
        count (a plain pool collides heavily), and a size-escape raises the minimum size
        once small combos saturate — bounded by the ``nonzero`` support and an attempt
        cap. Used for both forward CS (subject co-occurrence group prototypes) and
        inverse CS (object co-occurrence group prototypes).
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
            k = min(nonzero, max(min_k, _draw_size(size_q)))
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
                          target: list, reuse_vmin: float = float("nan"),
                          reuse_vmax: float = float("nan")) -> None:
        """Assign entities to templates: floor each template at ≥1 entity (so every distinct
        (inverse-)CS is realised), then distribute the rest by a power-law(reuse_zipf) reuse
        tail. The tail is drawn *truncated* to ``[reuse_vmin, reuse_vmax]`` — the support the
        measured cs_freq law was fitted over — so no template's share exceeds the target's
        observed max recurrence, and none of the mass a clamp would pile up at the bound
        survives. Writes the chosen relation-set into ``target[v]``. Empty pool → empty set."""
        n_p = len(pool)
        if n_p == 0:
            for v in entities:
                target[v] = np.array([], dtype=int)
            return
        order = rng.permutation(len(entities))
        fit = sample_powerlaw_trunc(reuse_zipf, reuse_vmin, reuse_vmax, n_p, rng)
        sfit = fit.sum()
        fit = fit / sfit if sfit > 0 else np.full(n_p, 1.0 / n_p)
        for rank, oi in enumerate(order):
            v = int(entities[oi])
            idx = rank if rank < n_p else int(rng.choice(n_p, p=fit))
            target[v] = pool[idx]

    # --- 3a. Forward CS membership (out-relations per entity) ---
    # Group-based forward CS: assign each entity to a co-occurrence group drawn from
    # the exp-decay spectrum weights, then build a pool of schema.cs_num_templates
    # reusable CS templates per group and assign entities to templates via Zipf
    # weights. subj_group_probs and cs_num_templates are always populated by
    # sample_schema's _validate_target guard (see docs/generator.md), so this is
    # the only forward-CS path — no per-entity or per-type fallback.
    entity_cs: list = [None] * actual_V
    n_sg = schema.subj_group_probs.shape[0]
    entity_subj_group = rng.choice(n_sg, size=actual_V, p=schema.subj_group_weights)

    # One template pool per group, sized ∝ group weight. Use largest-remainder
    # allocation so the per-group counts sum exactly to cs_num_templates (a plain
    # max(1,round(...)) per-group pattern would inflate the total).
    _fwd_quotas = _allocate_quotas(schema.subj_group_weights, schema.cs_num_templates)
    group_fwd_pools: list[list[np.ndarray]] = []
    for g in range(n_sg):
        probs_g = schema.subj_group_probs[g].copy()
        nz_g = int((probs_g > 0).sum())
        group_fwd_pools.append(
            _build_distinct(probs_g, nz_g, schema.cs_size_q, _fwd_quotas[g])
        )
    buckets_sg: dict[int, list[int]] = {}
    for v in range(actual_V):
        buckets_sg.setdefault(int(entity_subj_group[v]), []).append(v)
    for g in range(n_sg):
        _assign_templates(buckets_sg.get(g, []), group_fwd_pools[g],
                          schema.cs_template_zipf, entity_cs,
                          reuse_vmin=schema.cs_template_vmin,
                          reuse_vmax=schema.cs_template_vmax)
    used = len({frozenset(int(x) for x in entity_cs[v])
                for v in range(actual_V) if entity_cs[v] is not None and len(entity_cs[v])})
    log.info("Stage 2: group forward CS (target %d templates, realised %d)",
             schema.cs_num_templates, used)

    # Post-hoc type assignment: score each entity's realised CS against P(r|t)
    # and assign the highest-likelihood type.  This makes type labels emerge from
    # relation usage (the real causal direction) rather than being set independently.
    # entity_cs[v] is never empty here: every subj_group_probs row is strictly positive
    # (softmax(logits) * relation_weights, both >0 for every relation), every group's
    # template quota is floored at ≥1 by _allocate_quotas, and _build_distinct/
    # _assign_templates given a positive quota and nonzero support always produce a
    # non-empty template — so there is no "no CS to score" case to fall back from.
    if num_types > 0:
        log_ptr = np.log(np.maximum(schema.type_relation_probs, 1e-12))  # (T, R)
        best_t = np.empty(actual_V, dtype=int)
        best_score = np.empty(actual_V, dtype=float)
        for v in range(actual_V):
            scores = log_ptr[:, entity_cs[v]].sum(axis=1)
            best_t[v] = int(np.argmax(scores))
            best_score[v] = float(scores[best_t[v]])
        # Only n_type_edges entities are typed — the measured rdf:type share of E need
        # not cover every entity. Type the best-scoring ones (the entities whose CS most
        # clearly identifies a type); the rest stay untyped and emit no rdf:type edge.
        typed = np.argsort(-best_score, kind="stable")[:n_type_edges]
        entity_types[typed] = best_t[typed]
        log.info("Stage 2: post-hoc type assignment from CS — %d/%d entities typed",
                 len(typed), actual_V)

    # --- 3b. Inverse CS membership (in-relations per entity), symmetric to forward ---
    # Group-based inverse CS, symmetric to the forward group path above. obj_group_probs
    # and inv_cs_num_templates are always populated (see 3a), so entity_inv_cs is always
    # built here — no "every object eligible for every relation" fallback.
    n_og = schema.obj_group_probs.shape[0]
    entity_obj_group = rng.choice(n_og, size=actual_V, p=schema.obj_group_weights)

    entity_inv_cs: list = [None] * actual_V
    _inv_quotas = _allocate_quotas(schema.obj_group_weights, schema.inv_cs_num_templates)
    group_inv_pools: list[list[np.ndarray]] = []
    for g in range(n_og):
        probs_g = schema.obj_group_probs[g].copy()
        nz_g = int((probs_g > 0).sum())
        group_inv_pools.append(
            _build_distinct(probs_g, nz_g, schema.inv_cs_size_q, _inv_quotas[g])
        )
    buckets_og: dict[int, list[int]] = {}
    for v in range(actual_V):
        buckets_og.setdefault(int(entity_obj_group[v]), []).append(v)
    for g in range(n_og):
        _assign_templates(buckets_og.get(g, []), group_inv_pools[g],
                          schema.inv_cs_template_zipf, entity_inv_cs,
                          reuse_vmin=schema.inv_cs_template_vmin,
                          reuse_vmax=schema.inv_cs_template_vmax)
    inv_used = len({frozenset(int(x) for x in entity_inv_cs[v])
                    for v in range(actual_V)
                    if entity_inv_cs[v] is not None and len(entity_inv_cs[v])})
    log.info("Stage 2: group inverse CS (target %d templates, realised %d)",
             schema.inv_cs_num_templates, inv_used)

    # --- 3b2. Overlap subject/object pools for reciprocal relations ---
    # A bidirectional (mutual) pair a↔b needs both a and b to *emit* and *receive* the
    # relation, i.e. to sit in both S_r (subject pool) and O_r (object pool). Forward
    # and inverse CS are assigned independently above, so S_r ∩ O_r is tiny even for a
    # relation Stage-1 marked symmetric (ρ_r≈1), starving the mutual-pair construction
    # in the wiring loop. Here we make a ρ_r fraction of the entities that *emit* r also
    # *receive* it: add r to their inverse CS (swapping out one existing entry so the
    # inverse-CS *size* — and the §3c degree rank-matching / Block D inv_cs_size_q — is
    # preserved; only which relations they receive changes, which is correct for a
    # symmetric relation). This runs before objects_by_rel (= O_r) is derived from
    # entity_inv_cs below, so the change propagates into O_r.
    # No-op when no reciprocity target is set (small-R fallback — see sample_schema).
    if schema.relation_reciprocity is not None:
        emitters_of: dict[int, list[int]] = defaultdict(list)
        receivers_of: dict[int, set[int]] = defaultdict(set)
        for v in range(actual_V):
            cs_v = entity_cs[v]
            if cs_v is not None:
                for r in cs_v:
                    emitters_of[int(r)].append(v)
            inv_v = entity_inv_cs[v]
            if inv_v is not None:
                for r in inv_v:
                    receivers_of[int(r)].add(v)
        n_shared = 0
        for r_idx in range(num_relations):
            rho_r = float(schema.relation_reciprocity[r_idx])
            if rho_r <= 0.0:
                continue
            already = receivers_of.get(r_idx, ())
            for v in emitters_of.get(r_idx, ()):
                if v in already or rng.random() >= rho_r:
                    continue
                inv_v = entity_inv_cs[v]
                if inv_v is not None and len(inv_v) > 0:
                    inv_v = np.asarray(inv_v).copy()
                    inv_v[int(rng.integers(len(inv_v)))] = r_idx  # size-preserving swap
                    entity_inv_cs[v] = inv_v
                else:
                    entity_inv_cs[v] = np.array([r_idx], dtype=int)
                n_shared += 1
        log.info("Stage 2: pool overlap — added %d reciprocal in-relations "
                 "(entities now emitting+receiving a symmetric relation)", n_shared)

    # ------------------------------------------------------------------
    # 3c. Per-entity target degrees.
    #     Sampled target values are rank-matched to (inverse-)CS size so entities
    #     with larger characteristic sets receive the larger degree targets —
    #     preserving the CS-size↔degree correlation (G2b) and keeping the
    #     ≥1-edge-per-CS-relation floor feasible.  Both sides are repaired to the
    #     SAME budget, so Σ tgt_out == Σ tgt_in: these become the *row margins* of
    #     the IPF allocation below, and a transportation problem is only solvable
    #     when its two margins agree.
    # ------------------------------------------------------------------
    quota_budget = content_E_target
    # Hub entries the sum repair must not touch — they carry the p90/max targets.
    n_hub = max(1, int(round(actual_V * DEGSEQ_TAIL_FRACTION)))

    def _sample_target_degrees(
        samples: np.ndarray, rank_scores: np.ndarray, *, floor: np.ndarray | None = None,
    ) -> np.ndarray:
        """Rank-match sampled degree targets to per-entity scores (descending), apply an
        optional per-entity floor, then repair the sum to ``quota_budget`` **exactly**.

        Exactly is not a nicety here. These are the row margins of the IPF allocation in
        §4, and a transportation problem is solvable only when its two margins agree: if
        ``Σ tgt_out ≠ Σ tgt_in`` the wiring inflates or starves by the difference (an early
        version of this overshot wn18rr_v4's edge count by 9%). So the repair escalates
        rather than returning a residual:

        1. trim the body only, preserving the hub tail that carries p90/max;
        2. if the budget is still unreachable that way, **drop the CS-size floor** and fall
           back to the unfloored degree law, which already sums to the budget by
           construction (Stage 1 built it that way).

        The hub tail is never bent. The two obvious alternatives to step 2 were both tried
        against the corpus and both are worse:

        * *Let the trim touch the hubs, keeping the floor.* Costs the max-degree target
          exactly where it is hardest to hit — wn18rr_v4's realised max out-degree halved
          (28 → 14) and fb237_v4's slipped off target (195 → 192).
        * *Re-repair the floored sequence with ``floor=1``.* Far worse: the trim is weighted
          by headroom above the floor, so the hubs — which have by far the most — absorb
          nearly all of it, and swdf's max out-degree *target* collapsed 623 → 18.

        So the ordering is deliberate: the degree law outranks the CS floor. An entity
        emitting no edge for one relation in its CS costs a little Block-D fidelity; a
        flattened degree sequence costs the whole of Block B. And the floor is not lost
        outright — ``fit_stubs`` still honours it per relation wherever that relation's
        budget can afford it.

        Step 2 fires whenever the floor plus the frozen hub tail exceeds the budget, which
        includes but is not limited to the floor being outright impossible
        (``Σ|CS(v)| > content_E`` — swdf asks for 606 500 edges against a budget of 242 256).
        """
        vals = np.sort(np.asarray(samples, dtype=np.int64))[::-1]
        if vals.size < actual_V:
            vals = np.concatenate([vals, rng.choice(vals, size=actual_V - vals.size)])
        vals = vals[:actual_V]
        order = np.argsort(-rank_scores, kind="stable")
        unfloored = np.empty(actual_V, dtype=np.int64)
        unfloored[order] = vals

        def _hub_mask(seq: np.ndarray) -> np.ndarray:
            keep = np.ones(actual_V, dtype=bool)
            keep[np.argsort(-seq, kind="stable")[:n_hub]] = False
            return keep

        tgt = np.maximum(unfloored, floor) if floor is not None else unfloored.copy()
        tgt = repair_degree_sum(tgt, quota_budget, rng, floor=floor,
                                adjustable=_hub_mask(tgt))
        if int(tgt.sum()) != quota_budget:
            log.info(
                "Stage 2: the ≥1-edge-per-CS-relation floor (Σ|CS|=%d) does not fit the "
                "edge budget (content_E=%d) alongside the degree tail — dropping it; "
                "fit_stubs still applies it per relation where affordable",
                int(np.sum(floor)) if floor is not None else 0, quota_budget,
            )
            tgt = repair_degree_sum(unfloored, quota_budget, rng,
                                    adjustable=_hub_mask(unfloored))
        return tgt

    cs_sizes_all = np.array(
        [len(entity_cs[v]) if entity_cs[v] is not None else 0 for v in range(actual_V)],
        dtype=np.int64,
    )
    # No rdf:type correction on either side: Block B measures entity *content* degrees
    # (type edges and class nodes excluded), and Stage 1 sampled against the content mean.
    # floor=cs_sizes_all keeps the ≥1-edge-per-CS-relation floor feasible on the out-side.
    samples_out = np.asarray(schema.target_out_degrees, dtype=np.int64)
    tgt_out = _sample_target_degrees(samples_out, cs_sizes_all.astype(float), floor=cs_sizes_all)

    in_scores = np.array(
        [len(entity_inv_cs[v]) if entity_inv_cs[v] is not None else 0 for v in range(actual_V)],
        dtype=float,
    )
    in_scores = in_scores + rng.random(actual_V)  # random tiebreak within equal inverse-CS sizes
    tgt_in = _sample_target_degrees(np.asarray(schema.target_in_degrees, dtype=np.int64), in_scores)

    # The IPF allocation reads these as its two row margins, and a transportation problem
    # with disagreeing margins has no solution — the wiring would silently inflate or
    # starve by the difference rather than failing. Cheap to check, so check it.
    assert int(tgt_out.sum()) == int(tgt_in.sum()) == quota_budget, (
        f"degree-target margins disagree: Σout={int(tgt_out.sum())} "
        f"Σin={int(tgt_in.sum())} budget={quota_budget}"
    )

    log.info(
        "Stage 2: degree targets — out(max=%d, p90=%.1f) in(max=%d, p90=%.1f)",
        int(tgt_out.max()), np.percentile(tgt_out, 90),
        int(tgt_in.max()), np.percentile(tgt_in, 90),
    )

    # ------------------------------------------------------------------
    # 4. Wire content edges: per-relation multiplicity-then-PA with edge conservation,
    #    matched within S_r × O_r (forward-CS subjects × inverse-CS objects). For each
    #    relation r: allocate |edges_r| across subjects (out-side: power-law(α_obj) tail ×
    #    cs_size^a_obj, floor ≥1, cap at |O_r|) and across objects (in-side: power-law(α_subj)
    #    tail × in_degree^pa × inv_cs_size^a_subj, cap at |S_r|), then pair the stubs.
    # ------------------------------------------------------------------
    in_degrees = np.ones(actual_V, dtype=float)
    out_degrees = np.zeros(actual_V, dtype=np.int64)   # total out-edges placed so far
    # unique_src_count[o] = number of distinct subjects that point to o (any pred)
    # Used as proxy for undirected in-degree under the simplification step.
    unique_src_count = np.zeros(actual_V, dtype=int)
    seen: set[tuple[int, int, str]] = set()
    content_edges: list[tuple[int, int, str]] = []
    # seen_src[o] = set of sources that already point to o (for unique_src_count)
    seen_src: list[set] = [set() for _ in range(actual_V)]

    # ── Pair-level edge-multiplicity (overlap) steering (Block C targets) ────────
    # Real graphs pack directed content edges onto shared pairs; the default wiring
    # scatters them (~simple graph), inflating the undirected simple graph the
    # motifs are counted on. We bias *which* pending object stub a subject pairs
    # with — toward a pair it already links to (parallel/multi-relational) or one
    # that already links to it (bidirectional) — which preserves the m_obj/m_in
    # degree allocations exactly (degree- and budget-neutral) and only correlates
    # the pairing. Global (cross-relation) neighbour indices:
    out_targets: list[set[int]] = [set() for _ in range(actual_V)]  # o's that s → (parallel test)
    in_neighbours: list[set[int]] = [set() for _ in range(actual_V)]  # o's that → s (bidir test)
    # Target overlap-edge counts from ρ = edge_multiplicity·bidirectional_ratio:
    #   parallel edges (extra relation on an existing directed pair) = E·(1 − 1/em)
    #   bidirectional edges (the reverse-direction edge of a pair)    = E·(b−1)/(em·b)
    _em = max(1.0, float(schema.edge_multiplicity))
    _bd = max(1.0, float(schema.bidirectional_ratio))
    n_parallel_target = int(round(content_E_target * (1.0 - 1.0 / _em)))
    n_bidir_target = int(round(content_E_target * (_bd - 1.0) / (_em * _bd)))
    n_parallel = 0
    n_bidir = 0
    # Per-relation reciprocity ρ_r drives the shared-pool bidirectional construction
    # (Phase A in the pairing loop); None → all-asymmetric (legacy).
    rel_recip = schema.relation_reciprocity

    # Subject pool S_r (forward CS) and object pool O_r (inverse CS) per relation.
    subjects_by_rel: dict[int, list[int]] = {}
    for v, cs in enumerate(entity_cs):
        for rel_idx in cs:
            subjects_by_rel.setdefault(int(rel_idx), []).append(v)
    objects_by_rel: dict[int, list[int]] = {}
    for v, inv in enumerate(entity_inv_cs):
        for rel_idx in inv:
            objects_by_rel.setdefault(int(rel_idx), []).append(v)

    # Relations that can be wired at all (subjects and inverse-CS-eligible objects both
    # present). A relation missing either pool must carry exactly zero edges.
    present = sorted(r for r in subjects_by_rel if objects_by_rel.get(r))

    def _relation_alpha(alpha_q) -> float:
        """One per-relation exponent drawn from a multiplicity-α quantile fit (NaN → flat)."""
        vals = sample_quantiles_trunc(alpha_q, 1, rng)
        return float(vals[0]) if vals is not None else float("nan")

    # ── Joint stub allocation (IPF) ─────────────────────────────────────────────
    # Decide every entity's out- and in-stub count *per relation*, for all relations at
    # once, subject to both margins: the per-entity degree targets (rows) and the
    # per-relation edge budget (columns). See generator/_ipf.py for why this is a
    # transportation problem and not a sequence of independent draws.
    #
    # This replaces the old per-relation `Multinomial(edges_r, w)` pair. Those hit the
    # column margin and said nothing about the rows, so the degree target had to be
    # bolted on afterwards as a hard cap — and capping each side independently is exactly
    # what broke the column margin again, leaving the two sides with unequal stub counts
    # and dumping the difference into a uniform-random deficit pass.
    #
    # The seed weights are the ones the loop used to compute inline: a per-relation
    # power-law multiplicity draw times the G2b CS-size offset. IPF only rescales whole
    # rows and columns, so their cross-ratios survive the fit exactly — the multiplicity
    # law and the CS-size coupling are preserved, and only the margins are forced.
    cs_by_rel = [np.array([r for r in cs if r in subjects_by_rel and objects_by_rel.get(r)],
                          dtype=np.int64) for cs in entity_cs]
    inv_by_rel = [np.array([r for r in inv if r in subjects_by_rel and objects_by_rel.get(r)],
                           dtype=np.int64) for inv in entity_inv_cs]
    out_rows, out_cols = build_support(cs_by_rel, num_relations)
    in_rows, in_cols = build_support(inv_by_rel, num_relations)

    # Per-relation multiplicity exponents, drawn once each (was: once per loop iteration).
    alpha_obj = {r: _relation_alpha(schema.obj_alpha_q) for r in present}
    alpha_subj = {r: _relation_alpha(schema.subj_alpha_q) for r in present}

    # Seed weights, in the support's column-major order. `build_support` sorts by
    # (relation, entity), and subjects_by_rel[r] / objects_by_rel[r] are likewise built by
    # ascending entity, so a column's slice lines up with that relation's pool entry for
    # entry — which is what lets a column slice be used directly as its stub vector below.
    if present:
        out_val = np.concatenate([
            sample_powerlaw_trunc(alpha_obj[r], 1.0, schema.obj_mult_max,
                                  len(subjects_by_rel[r]), rng)
            for r in present
        ]) * np.power(np.maximum(cs_sizes_all[out_rows], 1.0), schema.a_obj)
        inv_sizes_all = np.array(
            [len(entity_inv_cs[v]) if entity_inv_cs[v] is not None else 0
             for v in range(actual_V)], dtype=float)
        in_val = np.concatenate([
            sample_powerlaw_trunc(alpha_subj[r], 1.0, schema.subj_mult_max,
                                  len(objects_by_rel[r]), rng)
            for r in present
        ]) * np.power(np.maximum(inv_sizes_all[in_rows], 1.0), schema.a_subj)
    else:
        out_val = in_val = np.array([], dtype=float)

    # Starting budget from the relation-frequency fit; the solver renegotiates it where a
    # relation's pools cannot absorb it. An edge needs a distinct (subject, object) pair,
    # so |S_r|·|O_r| is a hard ceiling.
    w_rel = np.zeros(num_relations, dtype=float)
    for r in present:
        w_rel[r] = schema.relation_weights[r]
    w_sum = w_rel.sum()
    w_rel = (w_rel / w_sum if w_sum > 0
             else np.array([1.0 / len(present) if r in present else 0.0
                            for r in range(num_relations)]))
    col_cap = np.zeros(num_relations, dtype=float)
    for r in present:
        col_cap[r] = len(subjects_by_rel[r]) * len(objects_by_rel[r])

    edge_budget_arr = solve_edge_budget(
        out_rows, out_cols, out_val, tgt_out,
        in_rows, in_cols, in_val, tgt_in,
        w_rel * content_E_target, actual_V, num_relations, col_cap=col_cap,
    )

    # Final fit against the solved budget. Column sums come out exactly equal to it on both
    # sides — that is what makes the two sides' stubs pairable. The out side carries the
    # ≥1-edge-per-CS-relation floor; the in side does not (an entity's inverse CS is
    # completed later, best-effort, by the redirect pass in §4b).
    stub_out = fit_stubs(out_rows, out_cols, out_val, tgt_out, edge_budget_arr,
                         actual_V, num_relations, rng, floor=True)
    stub_in = fit_stubs(in_rows, in_cols, in_val, tgt_in, edge_budget_arr,
                        actual_V, num_relations, rng)

    # Column slices: `*_cols` are sorted, so relation r owns one contiguous run.
    out_starts = np.concatenate(([0], np.cumsum(np.bincount(out_cols, minlength=num_relations))))
    in_starts = np.concatenate(([0], np.cumsum(np.bincount(in_cols, minlength=num_relations))))

    _shortfall = int(np.abs(edge_budget_arr - (w_rel * content_E_target)).sum() / 2)
    log.info(
        "Stage 2: IPF stub allocation — %d/%d out/in support entries, budget %d edges "
        "(%d redistributed off bottlenecked relations)",
        out_rows.size, in_rows.size, int(edge_budget_arr.sum()), _shortfall,
    )

    for rel_idx in present:
        S_r = subjects_by_rel[rel_idx]
        O_r = objects_by_rel[rel_idx]
        obj_ids = np.asarray(O_r, dtype=np.int64)
        n_sr, n_or = len(S_r), int(obj_ids.shape[0])
        edges_r = int(edge_budget_arr[rel_idx])
        if edges_r <= 0 or n_sr == 0 or n_or == 0:
            continue
        predicate = schema.relations[rel_idx]
        # Stub counts for this relation, straight off the joint allocation. Both sum to
        # edges_r by construction — no cap, no truncation, no imbalance.
        m_obj = stub_out[out_starts[rel_idx]:out_starts[rel_idx + 1]].copy()
        m_in = stub_in[in_starts[rel_idx]:in_starts[rel_idx + 1]].copy()


        # Reciprocity: guarantee both an out-stub AND an in-stub of r for a ρ_r-sized
        # subset of the entities eligible for both (S_r ∩ O_r, populated by the §3b2
        # CS-overlap pass above) — the mutual-pair construction below can only pair
        # entities that actually have nonzero remaining stubs on *both* sides, and
        # m_obj/m_in are otherwise independent draws that rarely give the same entity
        # both by chance. This reservation steals one stub from the current max-count
        # entity on each side, so sum(m_obj)==edges_r / sum(m_in)==edges_r exactly —
        # budget-neutral, applied last so capping above can't undo it.
        rho_r = float(rel_recip[rel_idx]) if rel_recip is not None else 0.0
        if rho_r > 0.0:
            both_r = list(set(S_r) & set(O_r))
            n_mutual_target = int(round(rho_r * edges_r / 2.0))
            n_reserve = min(len(both_r) // 2 * 2, 2 * n_mutual_target, n_sr, n_or)
            if n_reserve > 0:
                rng.shuffle(both_r)
                reserved = both_r[:n_reserve]
                pos_out = {int(s): i for i, s in enumerate(S_r)}
                pos_in = {int(o): i for i, o in enumerate(obj_ids.tolist())}

                def _reserve(m: np.ndarray, pos: dict) -> None:
                    """Give a stub to each reserved entity that has none, taking it from an
                    entity that has one to spare. Column-sum-preserving, so the per-relation
                    stub balance the IPF allocation establishes survives.

                    Donors are drawn **uniformly among the entities with a surplus**. Taking
                    from ``argmax(m)`` instead — as this did originally — robs the same entry
                    over and over, and that entry is by definition the hub carrying the
                    max/p90 degree target. It was harmless while ``m_obj`` came from
                    ``ones + multinomial`` (no zeros, so nothing to reserve), but the IPF
                    allocation puts real hub mass in ``m``, and on a highly reciprocal graph
                    with tens of thousands of reservations it flattened the peak completely:
                    aids' realised max out-degree collapsed to 4 against a target of 11.
                    """
                    idx = np.fromiter((pos[e] for e in reserved if e in pos),
                                      dtype=np.int64, count=-1)
                    if idx.size == 0:
                        return
                    starving = idx[m[idx] == 0]
                    surplus = np.maximum(m - 1, 0)
                    donors = np.where(surplus > 0)[0]
                    take = min(starving.size, int(surplus.sum()))
                    if take <= 0 or donors.size == 0:
                        return
                    cut = np.bincount(rng.choice(donors, size=take, replace=True),
                                      minlength=m.size)
                    cut = np.minimum(cut, surplus)     # a donor cannot give more than it has
                    moved = int(cut.sum())
                    m -= cut
                    m[starving[:moved]] += 1

                _reserve(m_obj, pos_out)
                _reserve(m_in, pos_in)

        # Pair subjects with objects within S_r × O_r (configuration model), holding
        # the per-entity out/in stub multiplicities as `remaining_out`/`remaining_in`
        # counts (== m_obj/m_in) so both overlap phases below decrement them and the
        # degree allocation is preserved exactly. `_place` does all placement
        # bookkeeping and classifies the edge as parallel / bidirectional overlap.
        remaining_out: dict[int, int] = defaultdict(int)
        for idx, s in enumerate(S_r):
            remaining_out[int(s)] += int(m_obj[idx])
        remaining_in: dict[int, int] = defaultdict(int)
        for idx, o in enumerate(obj_ids.tolist()):
            remaining_in[int(o)] += int(m_in[idx])
        placed_pairs: set[tuple[int, int]] = set()

        def _place(s: int, o: int) -> bool:
            """Place directed edge (s→o) if the stubs/pair are available, updating all
            bookkeeping (degree allocation, neighbour indices, overlap counters)."""
            nonlocal n_parallel, n_bidir
            if (s == o or remaining_out.get(s, 0) <= 0 or remaining_in.get(o, 0) <= 0
                    or (s, o) in placed_pairs):
                return False
            if o in out_targets[s]:
                n_parallel += 1                      # (s,o) already exists (other relation)
            elif s in out_targets[o]:
                n_bidir += 1                         # (o,s) exists → this is the reverse edge
            remaining_out[s] -= 1
            remaining_in[o] -= 1
            placed_pairs.add((s, o))
            content_edges.append((s, o, predicate))
            seen.add((s, o, predicate))
            out_targets[s].add(o)
            in_neighbours[o].add(s)
            in_degrees[o] += 1.0
            out_degrees[s] += 1
            if s not in seen_src[o]:
                seen_src[o].add(s)
                unique_src_count[o] += 1
            return True

        # Phase A — reciprocity-driven bidirectional construction: this relation's
        # target reciprocity ρ_r (Block B, sampled per relation in Stage 1) says a
        # ρ_r fraction of its directed edges should be part of a mutual pair, i.e.
        # build ~ρ_r·edges_r/2 mutual (e1↔e2) pairs. Draw both endpoints from the
        # relation's *shared* pool `both = S_r ∩ O_r` (entities that are both a pending
        # subject and object of r) — for a symmetric relation the forward and inverse
        # CS coincide, so this is the correct pool. `_place` keeps it degree/budget-
        # neutral and counts the reverse edge as bidirectional.
        # An entity with multiple stubs (edges_r/n_sr is typically ≫1) can supply more
        # than one mutual pair, so `pool` is drawn from *with replacement* rather than
        # walked once — an entity stays in the pool (swap-removed only once BOTH its
        # remaining_out and remaining_in are exhausted) so its full stub budget is used.
        rho_r = (float(rel_recip[rel_idx]) if rel_recip is not None else 0.0)
        n_mutual_target = int(round(rho_r * edges_r / 2.0))
        if n_mutual_target > 0:
            pool = [e for e in remaining_out
                    if remaining_in.get(e, 0) > 0 and remaining_out[e] > 0]
            built = 0
            max_attempts = 4 * n_mutual_target + 20   # bound: a few stale-pick misses per pair
            attempts = 0
            while built < n_mutual_target and len(pool) >= 2 and attempts < max_attempts:
                attempts += 1
                i1 = int(rng.integers(len(pool)))
                i2 = int(rng.integers(len(pool)))
                if i2 == i1:
                    continue
                e1, e2 = pool[i1], pool[i2]
                if (remaining_out.get(e1, 0) > 0 and remaining_in.get(e2, 0) > 0
                        and remaining_out.get(e2, 0) > 0 and remaining_in.get(e1, 0) > 0
                        and (e1, e2) not in placed_pairs and (e2, e1) not in placed_pairs):
                    if _place(e1, e2):               # forward
                        _place(e2, e1)               # reverse → counted as bidir in _place
                        built += 1
                # Swap-remove any entity no longer a valid mutual-pair candidate — it
                # needs BOTH remaining_out>0 and remaining_in>0 to serve as either side
                # of a future pair, so either hitting 0 disqualifies it (largest index
                # first so popping doesn't invalidate the other's position).
                for idx, e in sorted(((i1, e1), (i2, e2)), reverse=True):
                    if remaining_out.get(e, 0) <= 0 or remaining_in.get(e, 0) <= 0:
                        pool[idx] = pool[-1]
                        pool.pop()

        # Phase B — pair the remaining stubs. Draw objects from a shuffled reservoir
        # (unbiased configuration model), but when behind on the parallel target first
        # try an object s already links to (multi-relational overlap).
        subj_seq = [e for e, c in remaining_out.items() for _ in range(c)]
        obj_seq = [e for e, c in remaining_in.items() for _ in range(c)]
        rng.shuffle(subj_seq)
        rng.shuffle(obj_seq)
        order: deque = deque(obj_seq)

        def _valid(s: int, o: int) -> bool:
            return o != s and remaining_in.get(o, 0) > 0 and (s, o) not in placed_pairs

        for s in subj_seq:
            if remaining_out.get(s, 0) <= 0:
                continue                             # already consumed by phase A
            placed = False
            if n_parallel < n_parallel_target:       # multi-relational overlap first
                for o in out_targets[s]:
                    if _valid(s, o) and _place(s, o):
                        placed = True
                        break
            if placed:
                continue
            for _ in range(MAX_PAIR_RETRY):          # default unbiased draw
                if not order:
                    break
                cand = order.popleft()
                if remaining_in.get(cand, 0) <= 0:
                    continue                         # exhausted by an overlap/phase-A draw
                if _valid(s, cand) and _place(s, cand):
                    break
                order.append(cand)                   # valid object, wrong for this s — requeue

    log.info("Stage 2: wired %d content edges (overlap: parallel=%d/%d, bidir=%d/%d)",
             len(content_edges), n_parallel, n_parallel_target, n_bidir, n_bidir_target)

    # ------------------------------------------------------------------
    # 4b. Inv-CS template completion: redirect existing edges so every object
    #     node receives at least one in-edge per predicate in its assigned
    #     template.  Runs after main wiring but before deficit recovery /
    #     bridging (which use objects_by_rel and therefore stay inv-CS-aware).
    #     For each missing predicate r on node o, finds a donor edge (s',o',r)
    #     where o' has r from ≥2 edges (can spare one) and redirects to
    #     (s',o,r).  No net edge-count change.
    #     Limitation: in sparse graphs (mean degree ~2.5, 9 relations) only
    #     ~44% of (node,rel) pairs have ≥2 in-edges, so not all gaps can be
    #     filled — this is a structural density constraint, not a code issue.
    # ------------------------------------------------------------------
    if subjects_by_rel:
        pred_to_idx = {r: i for i, r in enumerate(schema.relations)}
        actual_in_preds: list[set[int]] = [set() for _ in range(actual_V)]
        for s, o, pred in content_edges:
            if o < actual_V and pred in pred_to_idx:
                actual_in_preds[o].add(pred_to_idx[pred])

        edges_by_rel: dict[int, list[int]] = {}
        in_count: dict[tuple[int, int], int] = {}
        for ei, (s, o, pred) in enumerate(content_edges):
            if pred in pred_to_idx:
                ri = pred_to_idx[pred]
                edges_by_rel.setdefault(ri, []).append(ei)
                in_count[(o, ri)] = in_count.get((o, ri), 0) + 1

        redirected = 0
        for o in range(actual_V):
            tmpl = entity_inv_cs[o]
            if tmpl is None or len(tmpl) == 0:
                continue
            for rel_idx in tmpl:
                if rel_idx in actual_in_preds[o]:
                    continue
                pred = schema.relations[rel_idx]
                candidates = list(edges_by_rel.get(rel_idx, []))
                rng.shuffle(candidates)
                for ei in candidates:
                    s2, o2, _ = content_edges[ei]
                    if o2 == o or s2 == o:
                        continue
                    if in_count.get((o2, rel_idx), 0) < 2:
                        continue
                    if (s2, o, pred) in seen:
                        continue
                    seen.discard((s2, o2, pred))
                    seen.add((s2, o, pred))
                    content_edges[ei] = (s2, o, pred)
                    in_count[(o2, rel_idx)] = in_count.get((o2, rel_idx), 1) - 1
                    in_count[(o, rel_idx)] = in_count.get((o, rel_idx), 0) + 1
                    actual_in_preds[o].add(rel_idx)
                    redirected += 1
                    break

        if redirected:
            log.info("Stage 2: inv-CS template completion redirected %d edges", redirected)

    # ------------------------------------------------------------------
    # 5. Connectivity guarantee: bridge isolated components to the giant,
    #     selectively — keeps up to (target_nc - 1) satellite components
    #     unbridged to hit target_lcc, bridges the rest.  Runs *first* among
    #     the connectivity-affecting steps so the two passes below (which
    #     sample edge endpoints freely and would otherwise reconnect a
    #     deliberately-kept-isolated satellite) can be told which nodes to
    #     avoid via the returned mask.  See _connect_components.
    #     objects_by_rel is passed so bridging edges respect inv-CS templates.
    # ------------------------------------------------------------------
    is_satellite = _connect_components(
        content_edges, actual_V, schema, rng, seen, in_degrees,
        target_nc=schema.target_num_components,
        target_lcc=schema.target_lcc,
        objects_by_rel=objects_by_rel,
    )
    non_satellite = ~is_satellite

    # ------------------------------------------------------------------
    # 5a. Deficit recovery: per-relation budget vs degree-quota misalignment
    #     can leave part of the edge budget unplaced.  Places the remainder
    #     by sampling (subject, object) pairs weighted by remaining quota.
    #     Uses objects_by_rel so only inv-CS-eligible objects are chosen.
    # ------------------------------------------------------------------
    deficit = content_E_target - len(content_edges)
    if deficit > 0 and present:
        rel_w = np.array([schema.relation_weights[r] for r in present], dtype=float)
        s_rw = rel_w.sum()
        rel_w = rel_w / s_rw if s_rw > 0 else np.full(len(present), 1.0 / len(present))
        placed = 0
        for _ in range(deficit * MAX_PAIR_RETRY):
            if placed >= deficit:
                break
            rel_idx = present[int(rng.choice(len(present), p=rel_w))]
            subj_pool = np.asarray(subjects_by_rel[rel_idx], dtype=np.int64)
            subj_pool = subj_pool[non_satellite[subj_pool]]
            obj_pool = np.asarray(objects_by_rel[rel_idx], dtype=np.int64)
            obj_pool = obj_pool[non_satellite[obj_pool]]
            if subj_pool.size == 0 or obj_pool.size == 0:
                continue
            w_s = np.maximum(tgt_out[subj_pool] - out_degrees[subj_pool], 0) + 1e-3
            s = int(rng.choice(subj_pool, p=w_s / w_s.sum()))
            w_o = np.maximum(tgt_in[obj_pool] - (in_degrees[obj_pool] - 1.0), 0) + 1e-3
            o = int(rng.choice(obj_pool, p=w_o / w_o.sum()))
            predicate = schema.relations[rel_idx]
            if s == o or (s, o, predicate) in seen:
                continue
            seen.add((s, o, predicate))
            content_edges.append((s, o, predicate))
            in_degrees[o] += 1.0
            out_degrees[s] += 1
            if s not in seen_src[o]:
                seen_src[o].add(s)
                unique_src_count[o] += 1
            placed += 1
        log.warning("Stage 2: pairing residual — recovered %d/%d edges (%.2f%% of budget)",
                    placed, deficit, 100.0 * deficit / max(content_E_target, 1))

    # ------------------------------------------------------------------
    # 5b. Trim back to the edge budget.  _connect_components appends its bridging edges
    #     *on top* of whatever the main loop placed.  That used to be invisible, because
    #     the old wiring always finished short and the bridges landed in the shortfall;
    #     the IPF allocation saturates the budget, so those bridges now push |E| over it
    #     (aids overshot by 456).  Give the bridges room by removing an equal number of
    #     edges from elsewhere.
    #
    #     Only **non-bridge** edges (in the undirected sense) are eligible: removing one
    #     cannot split a component, so the target_nc / target_lcc structure that
    #     _connect_components just established survives untouched.  Among those, drop the
    #     edges whose endpoints most *exceed* their degree targets — the excess has to
    #     come off somewhere, and taking it from over-served nodes improves degree
    #     fidelity rather than degrading it.
    # ------------------------------------------------------------------
    excess = len(content_edges) - content_E_target
    if excess > 0:
        tmp = igraph.Graph(n=actual_V, directed=False)
        tmp.add_edges([(s, o) for s, o, _ in content_edges if s != o])
        # `bridges()` indexes into tmp's edge list, which skips self-loops — map back.
        idx_map = [i for i, (s, o, _) in enumerate(content_edges) if s != o]
        is_bridge = np.zeros(len(content_edges), dtype=bool)
        for ei in tmp.bridges():
            is_bridge[idx_map[ei]] = True

        over = np.zeros(len(content_edges), dtype=float)
        for i, (s, o, _) in enumerate(content_edges):
            over[i] = (max(out_degrees[s] - tgt_out[s], 0)
                       + max(in_degrees[o] - 1.0 - tgt_in[o], 0))
        cand = np.where(~is_bridge)[0]
        if cand.size < excess:
            log.warning("Stage 2: only %d non-bridge edges for an excess of %d — "
                        "|E| will overshoot by %d", cand.size, excess, excess - cand.size)
        # Most over-served first; random tiebreak so a run of equal-scored edges is not
        # taken in index order (which would concentrate the trim on one relation).
        rank = over[cand] + rng.random(cand.size)
        drop = set(cand[np.argsort(-rank)[:excess]].tolist())
        for i in sorted(drop, reverse=True):
            s, o, _ = content_edges[i]
            out_degrees[s] -= 1
            in_degrees[o] -= 1.0
            del content_edges[i]
        log.info("Stage 2: trimmed %d edges to make room for bridging", len(drop))

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
