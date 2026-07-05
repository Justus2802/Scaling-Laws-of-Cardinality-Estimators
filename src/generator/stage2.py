"""Stage 2 — CS-aware graph instantiation.

Turns a Schema into an igraph.Graph by
  - using the Schema's |V| and |E| targets,
  - assigning types to entities via the Schema's type_weights,
  - sampling each entity's characteristic set from P(r | type) so the
    co-occurrence structure matches the target,
  - wiring edges toward per-entity target degrees sampled from the measured
    degree distribution (no degree steering when unavailable),
  - adding rdf:type edges for all typed entities,
  - selectively bridging isolated components to match target num_components / LCC fraction.
"""

import math

import igraph
import numpy as np

from ._adapters import sample_powerlaw, sample_quantiles_trunc
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
PATH_STEERING_ENABLED = False  # Stage-2 path-length steering (shortcut injection); can only
                               # shorten paths, so disabled for now — see _steer_path_lengths


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
        edges after this (deficit recovery, path-length steering) must not
        touch these nodes — doing so would silently reconnect a satellite
        and undo the ``target_nc`` / ``target_lcc`` guarantee established here.
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


def _steer_path_lengths(
    content_edges: list,
    actual_V: int,
    schema: "Schema",
    rng: "np.random.Generator",
    seen: set,
    in_degrees: "np.ndarray",
    is_satellite: "np.ndarray | None" = None,
) -> None:
    """Steer mean path length and diameter toward Block F targets via hub shortcuts.

    Builds a temporary undirected igraph entity graph for efficient C-backend
    diameter and distance estimation; adds shortcut edges to both the igraph
    object and content_edges so each round re-estimates on the updated graph.
    Runs at most 4 estimation + injection rounds.

    No-ops when schema.path_mean_target is NaN and schema.path_hi_target is 0.

    Parameters
    ----------
    is_satellite : np.ndarray, optional
        Boolean mask (from ``_connect_components``) marking nodes in a
        deliberately-unbridged satellite component. All BFS-source and
        shortcut-endpoint sampling is restricted to the complement (the giant
        component) so this pass cannot silently reconnect a satellite.
    """
    path_mean_target = schema.path_mean_target
    path_hi_target = schema.path_hi_target
    has_mean = not math.isnan(path_mean_target)
    has_hi = path_hi_target > 0

    if not PATH_STEERING_ENABLED:
        if has_mean or has_hi:
            log.warning("Stage 2: path steering disabled (PATH_STEERING_ENABLED=False); "
                        "Block F path targets (mean=%s, diameter=%s) will not be steered",
                        f"{path_mean_target:.2f}" if has_mean else "—",
                        str(path_hi_target) if has_hi else "—")
        return
    if (not has_mean and not has_hi) or actual_V < 3:
        return

    ig = igraph.Graph(n=actual_V, directed=False)
    ig.add_edges([(s, o) for s, o, _ in content_edges
                  if s < actual_V and o < actual_V and s != o])

    giant_nodes = (np.where(~is_satellite)[0] if is_satellite is not None
                   else np.arange(actual_V))
    n_giant = giant_nodes.size
    if n_giant < 3:
        return

    def estimate_stats(k: int = 50) -> tuple[int, float]:
        diam = ig.diameter(directed=False, unconn=True)
        srcs = rng.choice(giant_nodes, size=min(k, n_giant), replace=False)
        tot = cnt = 0
        for src in srcs:
            for d in ig.distances(source=[int(src)], mode="all")[0]:
                if 0 < d < float("inf"):
                    tot += d; cnt += 1
        return int(diam), (tot / cnt if cnt > 0 else float("nan"))

    def add_shortcut(u: int, v: int) -> bool:
        if u == v:
            return False
        pred = schema.relations[int(rng.integers(len(schema.relations)))]
        triple = (u, v, pred)
        if triple in seen:
            return False
        seen.add(triple)
        content_edges.append(triple)
        in_degrees[v] += 1.0
        ig.add_edge(u, v)
        return True

    diam0, mean0 = estimate_stats()

    for _ in range(4):
        diam, mean_path = estimate_stats()
        hi_ok = not has_hi or diam <= path_hi_target
        mean_ok = not has_mean or math.isnan(mean_path) or mean_path <= path_mean_target + 0.5
        if hi_ok and mean_ok:
            break

        if not hi_ok:
            # Add several shortcuts per round: sample different remote pairs so the
            # diameter converges from multiple directions simultaneously.
            n_hi = max(1, (diam - path_hi_target + 1) // 2)
            for _ in range(n_hi):
                src = int(rng.choice(giant_nodes))
                row = ig.distances(source=[src], mode="all")[0]
                far = int(max(giant_nodes,
                              key=lambda v, r=row: r[v] if r[v] < float("inf") else -1))
                add_shortcut(src, far)

        if not mean_ok:
            deg = np.array(ig.degree(), dtype=np.float64)[giant_nodes]
            deg_sum = deg.sum()
            if deg_sum > 0:
                p = deg / deg_sum
                n_sc = max(1, round(n_giant ** 0.5 * (mean_path - path_mean_target) / mean_path))
                added = 0
                for _ in range(n_sc * 8):
                    if added >= n_sc:
                        break
                    if add_shortcut(int(rng.choice(giant_nodes, p=p)),
                                    int(rng.choice(giant_nodes, p=p))):
                        added += 1

    diam1, mean1 = estimate_stats(k=30)
    log.info(
        "Stage 2: path steering — diameter %d→%d (target %s), mean %.2f→%.2f (target %s)",
        diam0, diam1,
        str(path_hi_target) if has_hi else "—",
        mean0 if not math.isnan(mean0) else float("nan"),
        mean1 if not math.isnan(mean1) else float("nan"),
        f"{path_mean_target:.2f}" if has_mean else "—",
    )


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
    # quantile fit; when unavailable, fall back to a budget-derived Poisson mean. Edge
    # counts are NOT set here — the per-relation multinomial allocation below owns the
    # |E| budget, so CS size only sets relation *membership*.
    objects_per_slot = 1.0 / schema.mean_functionality if schema.mean_functionality < 1.0 else 1.0
    fallback_cs_mean = max(
        FALLBACK_CS_MEAN_FLOOR,
        (content_E_target / actual_V if actual_V > 0 else 1.0) / objects_per_slot,
    )

    def _draw_size(size_q) -> int:
        """Draw one (forward or inverse) CS size from a quantile fit, else budget Poisson."""
        vals = sample_quantiles_trunc(size_q, 1, rng)
        size = float(vals[0]) if vals is not None else float(rng.poisson(fallback_cs_mean))
        return max(1, int(round(size)))

    def _cap_redistribute(m: np.ndarray, cap, w: np.ndarray, hard_cap: np.ndarray | None = None) -> None:
        """Cap each count and redistribute overflow by weights ``w``.

        ``cap`` is a scalar upper bound (used for |S_r| / |O_r| side caps).
        ``hard_cap``, if given, is a per-node integer array of remaining capacity;
        it overrides ``cap`` element-wise with ``min(cap, hard_cap[i])``.
        Used on both sides: an object takes ≤ |S_r| distinct subjects, a subject
        reaches ≤ |O_r| distinct objects. Bounded passes; tiny residual is dropped.
        """
        if m.size == 0:
            return
        caps = np.minimum(cap, hard_cap) if hard_cap is not None else np.full(m.shape, cap, dtype=np.int64)
        if (caps <= 0).all():
            m[:] = 0
            return
        for _ in range(CAP_REDISTRIBUTE_PASSES):
            overflow = int(np.maximum(m - caps, 0).sum())
            if overflow == 0:
                break
            np.minimum(m, caps, out=m)
            free = np.where(m < caps)[0]
            if free.size == 0:
                break
            wf = w[free]
            swf = wf.sum()
            pf = wf / swf if swf > 0 else np.full(free.size, 1.0 / free.size)
            m[free] += rng.multinomial(overflow, pf)
        np.minimum(m, caps, out=m)

    log.info(
        "Stage 2: instantiating (seed=%d) V=%d, content-edge target=%d (+%d type edges), "
        "cs_size source=%s", seed, actual_V, content_E_target, n_type_edges,
        "quantiles" if not math.isnan(schema.cs_size_q[0]) else "budget-derived",
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
        """Draw one forward CS (relation membership) for type t; size from cs_size_q."""
        if num_relations == 0:
            return np.array([], dtype=int)
        probs, nonzero = _cs_probs(t)
        k = min(nonzero, _draw_size(schema.cs_size_q))
        if k == 0:
            return np.array([], dtype=int)
        return rng.choice(num_relations, size=k, replace=False, p=probs)

    def _build_distinct(probs: np.ndarray, nonzero: int, size_q, n_target: int) -> list[np.ndarray]:
        """Rejection-sample up to ``n_target`` DISTINCT relation-sets from ``probs``.

        Sizes come from ``size_q``; deduping by relation-set steers the distinct-CS
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
                          target: list, reuse_vmax: float = float("nan")) -> None:
        """Assign entities to templates: floor each template at ≥1 entity (so every distinct
        (inverse-)CS is realised), then distribute the rest by a power-law(reuse_zipf) reuse
        tail. ``reuse_vmax`` truncates the raw reuse draws, mirroring the measured truncated
        power-law's upper bound so no template dominates beyond the target's observed max
        recurrence; NaN → unbounded. Writes the chosen relation-set into ``target[v]``.
        Empty pool → empty set."""
        n_p = len(pool)
        if n_p == 0:
            for v in entities:
                target[v] = np.array([], dtype=int)
            return
        order = rng.permutation(len(entities))
        fit = sample_powerlaw(reuse_zipf, n_p, rng)
        if np.isfinite(reuse_vmax) and reuse_vmax >= 1.0:
            fit = np.minimum(fit, reuse_vmax)
        sfit = fit.sum()
        fit = fit / sfit if sfit > 0 else np.full(n_p, 1.0 / n_p)
        for rank, oi in enumerate(order):
            v = int(entities[oi])
            idx = rank if rank < n_p else int(rng.choice(n_p, p=fit))
            target[v] = pool[idx]

    # --- 3a. Forward CS membership (out-relations per entity) ---
    entity_cs: list = [None] * actual_V
    if schema.subj_group_probs is not None and num_relations > 0:
        # Group-based forward CS: assign each entity to a co-occurrence group drawn
        # from the exp-decay spectrum weights, then build CS from that group's prototype.
        n_sg = schema.subj_group_probs.shape[0]
        entity_subj_group = rng.choice(n_sg, size=actual_V, p=schema.subj_group_weights)

        if schema.cs_num_templates > 0:
            # One template pool per group, sized ∝ group weight.
            # Use largest-remainder allocation so the per-group counts sum exactly to
            # cs_num_templates (previously max(1,round(...)) inflated the total, causing
            # more distinct CSes than the target).
            _fwd_quotas = _allocate_quotas(schema.subj_group_weights, schema.cs_num_templates)
            group_fwd_pools: list[list[np.ndarray]] = []
            for g in range(n_sg):
                probs_g = schema.subj_group_probs[g].copy()
                nz_g = int((probs_g > 0).sum())
                group_fwd_pools.append(_build_distinct(probs_g, nz_g, schema.cs_size_q, _fwd_quotas[g]))
            buckets_sg: dict[int, list[int]] = {}
            for v in range(actual_V):
                buckets_sg.setdefault(int(entity_subj_group[v]), []).append(v)
            for g in range(n_sg):
                _assign_templates(buckets_sg.get(g, []), group_fwd_pools[g],
                                  schema.cs_template_zipf, entity_cs,
                                  reuse_vmax=schema.cs_template_vmax)
            used = len({frozenset(int(x) for x in entity_cs[v])
                        for v in range(actual_V) if entity_cs[v] is not None and len(entity_cs[v])})
            log.info("Stage 2: group forward CS (target %d templates, realised %d)",
                     schema.cs_num_templates, used)
        else:
            # Per-entity sampling directly from each entity's group prototype.
            for v in range(actual_V):
                probs_g = schema.subj_group_probs[int(entity_subj_group[v])].copy()
                nz_g = int((probs_g > 0).sum())
                k = min(nz_g, _draw_size(schema.cs_size_q))
                entity_cs[v] = (rng.choice(num_relations, size=k, replace=False, p=probs_g)
                                if k > 0 else np.array([], dtype=int))
            log.info("Stage 2: group forward CS (per-entity mode)")

        # Post-hoc type assignment: score each entity's realised CS against P(r|t)
        # and assign the highest-likelihood type.  This makes type labels emerge from
        # relation usage (the real causal direction) rather than being set independently.
        if num_types > 0:
            log_ptr = np.log(np.maximum(schema.type_relation_probs, 1e-12))  # (T, R)
            for v in range(actual_V):
                cs = entity_cs[v]
                if cs is None or len(cs) == 0:
                    entity_types[v] = int(rng.choice(num_types, p=schema.type_weights))
                else:
                    entity_types[v] = int(np.argmax(log_ptr[:, cs].sum(axis=1)))
            log.info("Stage 2: post-hoc type assignment from CS (log P(CS|type) argmax)")

    elif schema.cs_num_templates > 0 and num_relations > 0:
        # DISTINCT templates per type (drawn from each type's P(r|t)), sized proportionally.
        type_templates: list[list[np.ndarray]] = []
        for t in range(num_types):
            probs, nz = _cs_probs(t)
            n_t = max(1, round(schema.cs_num_templates * float(schema.type_weights[t])))
            type_templates.append(_build_distinct(probs, nz, schema.cs_size_q, n_t))
        if num_types > 0:
            buckets: dict[int, list[int]] = {}
            for v in range(actual_V):
                buckets.setdefault(int(entity_types[v]), []).append(v)
            for t in range(num_types):
                _assign_templates(buckets.get(t, []), type_templates[t],
                                  schema.cs_template_zipf, entity_cs,
                                  reuse_vmax=schema.cs_template_vmax)
        else:
            probs, nz = _cs_probs(-1)
            untyped = _build_distinct(probs, nz, schema.cs_size_q, max(1, schema.cs_num_templates))
            _assign_templates(list(range(actual_V)), untyped, schema.cs_template_zipf,
                              entity_cs, reuse_vmax=schema.cs_template_vmax)
        used = len({frozenset(int(x) for x in entity_cs[v]) for v in range(actual_V) if len(entity_cs[v])})
        log.info("Stage 2: forward CS (target %d distinct, realised %d)", schema.cs_num_templates, used)
    else:
        log.info("Stage 2: forward CS in per-entity mode (no Block D templates)")
        for v in range(actual_V):
            entity_cs[v] = _sample_cs_for_type(int(entity_types[v]))

    # --- 3b. Inverse CS membership (in-relations per entity), symmetric to forward ---
    # No inverse templates and no obj groups → entity_inv_cs is None → every object is
    # eligible for every relation (today's behaviour) and the a_subj factor stays inert.
    entity_inv_cs: list | None = None
    if schema.obj_group_probs is not None and num_relations > 0:
        # Group-based inverse CS, symmetric to the forward group path above.
        n_og = schema.obj_group_probs.shape[0]
        entity_obj_group = rng.choice(n_og, size=actual_V, p=schema.obj_group_weights)

        entity_inv_cs = [None] * actual_V
        if schema.inv_cs_num_templates > 0:
            _inv_quotas = _allocate_quotas(schema.obj_group_weights, schema.inv_cs_num_templates)
            group_inv_pools: list[list[np.ndarray]] = []
            for g in range(n_og):
                probs_g = schema.obj_group_probs[g].copy()
                nz_g = int((probs_g > 0).sum())
                group_inv_pools.append(_build_distinct(probs_g, nz_g, schema.inv_cs_size_q, _inv_quotas[g]))
            buckets_og: dict[int, list[int]] = {}
            for v in range(actual_V):
                buckets_og.setdefault(int(entity_obj_group[v]), []).append(v)
            for g in range(n_og):
                _assign_templates(buckets_og.get(g, []), group_inv_pools[g],
                                  schema.inv_cs_template_zipf, entity_inv_cs,
                                  reuse_vmax=schema.inv_cs_template_vmax)
            inv_used = len({frozenset(int(x) for x in entity_inv_cs[v])
                            for v in range(actual_V) if entity_inv_cs[v] is not None and len(entity_inv_cs[v])})
            log.info("Stage 2: group inverse CS (target %d templates, realised %d)",
                     schema.inv_cs_num_templates, inv_used)
        else:
            for v in range(actual_V):
                probs_g = schema.obj_group_probs[int(entity_obj_group[v])].copy()
                nz_g = int((probs_g > 0).sum())
                k = min(nz_g, _draw_size(schema.inv_cs_size_q))
                entity_inv_cs[v] = (rng.choice(num_relations, size=k, replace=False, p=probs_g)
                                    if k > 0 else np.array([], dtype=int))
            log.info("Stage 2: group inverse CS (per-entity mode)")

    elif schema.inv_cs_num_templates > 0 and num_relations > 0:
        inv_probs = schema.relation_weights.copy()
        s = inv_probs.sum()
        inv_probs = inv_probs / s if s > 0 else np.full(num_relations, 1.0 / num_relations)
        inv_nz = int((inv_probs > 0).sum())
        inv_templates = _build_distinct(inv_probs, inv_nz, schema.inv_cs_size_q,
                                        max(1, schema.inv_cs_num_templates))
        entity_inv_cs = [None] * actual_V
        _assign_templates(list(range(actual_V)), inv_templates,
                          schema.inv_cs_template_zipf, entity_inv_cs,
                          reuse_vmax=schema.inv_cs_template_vmax)
        inv_used = len({frozenset(int(x) for x in entity_inv_cs[v])
                        for v in range(actual_V) if len(entity_inv_cs[v])})
        log.info("Stage 2: inverse CS (target %d distinct, realised %d)",
                 schema.inv_cs_num_templates, inv_used)

    # ------------------------------------------------------------------
    # 3c. Per-entity target degrees (replace the old global max-degree caps).
    #     Sampled target values are rank-matched to (inverse-)CS size so entities
    #     with larger characteristic sets receive the larger degree targets —
    #     preserving the CS-size↔degree correlation (G2b) and keeping the
    #     ≥1-edge-per-CS-relation floor feasible.  A multinomial top-up ensures
    #     Σ targets covers the content-edge budget so capacity caps cannot
    #     starve edge conservation.
    # ------------------------------------------------------------------

    def _assign_degree_targets(samples: np.ndarray, rank_scores: np.ndarray) -> np.ndarray:
        """Rank-match sampled degree targets to per-entity scores (descending)."""
        vals = np.sort(np.asarray(samples, dtype=np.int64))[::-1]
        if vals.size < actual_V:
            vals = np.concatenate([vals, rng.choice(vals, size=actual_V - vals.size)])
        vals = vals[:actual_V]
        order = np.argsort(-rank_scores, kind="stable")
        out = np.empty(actual_V, dtype=np.int64)
        out[order] = vals
        return out

    def _cover_edge_budget(tgt: np.ndarray) -> np.ndarray:
        """Top up targets multinomially so Σ targets ≥ content-edge budget."""
        shortfall = content_E_target - int(tgt.sum())
        if shortfall > 0:
            tot = float(tgt.sum())
            p = tgt / tot if tot > 0 else np.full(actual_V, 1.0 / actual_V)
            tgt = tgt + rng.multinomial(shortfall, p)
        return tgt

    cs_sizes_all = np.array(
        [len(entity_cs[v]) if entity_cs[v] is not None else 0 for v in range(actual_V)],
        dtype=np.int64,
    )
    tgt_out: np.ndarray | None = None
    if schema.target_out_degrees is not None and actual_V > 0:
        samples_out = np.asarray(schema.target_out_degrees, dtype=np.int64)
        if num_types > 0:
            # The measured out-degree includes each typed entity's rdf:type edge,
            # which is wired separately from the content budget.
            samples_out = np.maximum(samples_out - 1, 0)
        tgt_out = _assign_degree_targets(samples_out, cs_sizes_all.astype(float))
        tgt_out = np.maximum(tgt_out, cs_sizes_all)   # keep per-CS-relation floor feasible
        tgt_out = _cover_edge_budget(tgt_out)

    tgt_in: np.ndarray | None = None
    if schema.target_in_degrees is not None and actual_V > 0:
        if entity_inv_cs is not None:
            in_scores = np.array([len(entity_inv_cs[v]) if entity_inv_cs[v] is not None else 0
                                  for v in range(actual_V)], dtype=float)
            # Random tiebreak within equal inverse-CS sizes.
            in_scores = in_scores + rng.random(actual_V)
        else:
            in_scores = rng.random(actual_V)
        tgt_in = _assign_degree_targets(np.asarray(schema.target_in_degrees, dtype=np.int64),
                                        in_scores)
        tgt_in = _cover_edge_budget(tgt_in)

    if tgt_out is not None or tgt_in is not None:
        log.info(
            "Stage 2: degree targets (%s) — out(max=%s, p90=%s) in(max=%s, p90=%s)",
            schema.degree_mechanism,
            int(tgt_out.max()) if tgt_out is not None else "—",
            int(np.percentile(tgt_out, 90)) if tgt_out is not None else "—",
            int(tgt_in.max()) if tgt_in is not None else "—",
            int(np.percentile(tgt_in, 90)) if tgt_in is not None else "—",
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
        # Largest-remainder allocation: floor each share, then give the remaining
        # integer edge(s) to relations with the biggest fractional parts.  This
        # guarantees sum(edge_budget.values()) == content_E_target exactly.
        raw = [content_E_target * float(w_present[i]) for i in range(len(present))]
        floored = [int(r) for r in raw]
        shortfall = content_E_target - sum(floored)
        order = sorted(range(len(present)), key=lambda i: raw[i] - floored[i], reverse=True)
        for i in order[:shortfall]:
            floored[i] += 1
        edge_budget = {r: floored[i] for i, r in enumerate(present)}
    else:
        edge_budget = {}

    all_objs = np.arange(actual_V)

    def _relation_alpha(alpha_q) -> float:
        """One per-relation exponent drawn from a multiplicity-α quantile fit (NaN → flat)."""
        vals = sample_quantiles_trunc(alpha_q, 1, rng)
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
        subj_ids = np.asarray(S_r, dtype=np.int64)
        w_out = sample_powerlaw(_relation_alpha(schema.obj_alpha_q), n_sr, rng)
        cs_sizes = np.array([len(entity_cs[s]) for s in S_r], dtype=float)
        w_out = w_out * np.power(np.maximum(cs_sizes, 1.0), schema.a_obj)
        if tgt_out is not None:
            if schema.degree_mechanism == "chunglu":
                # Expected-degree weighting: allocation ∝ sampled target degree.
                w_out = w_out * np.maximum(tgt_out[subj_ids], 1.0)
            else:
                # Capacity weighting: allocation ∝ remaining quota (target − placed).
                w_out = w_out * np.maximum(tgt_out[subj_ids] - out_degrees[subj_ids], 0.0)
        sw_out = w_out.sum()
        w_out = w_out / sw_out if sw_out > 0 else np.full(n_sr, 1.0 / n_sr)
        if edges_r >= n_sr:
            m_obj = np.ones(n_sr, dtype=np.int64) + rng.multinomial(edges_r - n_sr, w_out)
        else:
            m_obj = np.zeros(n_sr, dtype=np.int64)
            nz = int((w_out > 0).sum())
            if nz >= edges_r:
                m_obj[rng.choice(n_sr, size=edges_r, replace=False, p=w_out)] = 1
            else:
                # Capacity exhausted for most subjects: take every positive-weight
                # subject and fill the remainder uniformly from the zero-weight pool.
                m_obj[w_out > 0] = 1
                zero_pool = np.where(w_out <= 0)[0]
                m_obj[rng.choice(zero_pool, size=edges_r - nz, replace=False)] = 1
        _cap_redistribute(m_obj, n_or, w_out)
        if tgt_out is not None and schema.degree_mechanism == "capacity":
            # Hard per-subject quota: never exceed the sampled target degree.
            out_cap = np.maximum(tgt_out[subj_ids] - out_degrees[subj_ids], 0).astype(np.int64)
            _cap_redistribute(m_obj, n_or, w_out, hard_cap=out_cap)

        # In-side: edges per object (over O_r) = power-law(α_subj) subject-multiplicity tail ×
        # degree-target weighting (capacity or expected-degree) × inv_cs_size^a_subj (G2b),
        # then cap at |S_r| (≤ |S_r| distinct subjects per object) + redistribute.
        # The object-stub multiset *is* the subject-mult law.
        w_in = sample_powerlaw(_relation_alpha(schema.subj_alpha_q), n_or, rng)
        if tgt_in is not None:
            # in_degrees starts at ones, so placed edges = in_degrees − 1.
            if schema.degree_mechanism == "chunglu":
                w_in = w_in * np.maximum(tgt_in[obj_ids], 1.0)
            else:
                w_in = w_in * np.maximum(tgt_in[obj_ids] - (in_degrees[obj_ids] - 1.0), 0.0)
        if O_r is not None and schema.a_subj != 0.0:
            inv_sizes = np.array([len(entity_inv_cs[o]) for o in O_r], dtype=float)
            w_in = w_in * np.power(np.maximum(inv_sizes, 1.0), schema.a_subj)
        sw_in = w_in.sum()
        if sw_in <= 0.0:
            continue
        m_in = rng.multinomial(edges_r, w_in / sw_in)
        _cap_redistribute(m_in, n_sr, w_in)
        if tgt_in is not None and schema.degree_mechanism == "capacity":
            # Hard per-object quota: never exceed the sampled target in-degree.  The
            # multinomial above can still over-allocate to a single node within one
            # relation pass (all edges placed before in_degrees is updated), so the
            # excess is redistributed proportionally to nodes with remaining quota.
            global_cap = np.maximum(tgt_in[obj_ids] - (in_degrees[obj_ids] - 1.0), 0).astype(np.int64)
            _cap_redistribute(m_in, n_sr, w_in, hard_cap=global_cap)

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
                out_degrees[s] += 1
                if s not in seen_src[o]:
                    seen_src[o].add(s)
                    unique_src_count[o] += 1
                break
            # else: no valid object found within retries → drop this stub (rare)

    log.info("Stage 2: wired %d content edges", len(content_edges))

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
    if entity_inv_cs is not None and subjects_by_rel:
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
            obj_pool = (np.asarray(objects_by_rel[rel_idx], dtype=np.int64)
                        if objects_by_rel is not None else all_objs)
            obj_pool = obj_pool[non_satellite[obj_pool]]
            if subj_pool.size == 0 or obj_pool.size == 0:
                continue
            if tgt_out is not None:
                w_s = np.maximum(tgt_out[subj_pool] - out_degrees[subj_pool], 0) + 1e-3
            else:
                w_s = np.ones(subj_pool.shape[0], dtype=float)
            s = int(rng.choice(subj_pool, p=w_s / w_s.sum()))
            if tgt_in is not None:
                w_o = np.maximum(tgt_in[obj_pool] - (in_degrees[obj_pool] - 1.0), 0) + 1e-3
            else:
                w_o = np.ones(obj_pool.shape[0], dtype=float)
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
        log.info("Stage 2: deficit recovery placed %d/%d missing edges", placed, deficit)

    # ------------------------------------------------------------------
    # 5b. Path-length steering (diameter cap + mean compression via shortcuts).
    #     Restricted to non-satellite nodes for the same reason as 5a.
    # ------------------------------------------------------------------
    _steer_path_lengths(content_edges, actual_V, schema, rng, seen, in_degrees,
                        is_satellite=is_satellite)

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
