"""Stage 3 — Maslov-Sneppen rewiring with simulated annealing.

Rewires content edges (never rdf:type edges) using degree-preserving
double-edge swaps to drive the graph's motif counts and degree assortativity
toward the Block E / Block F targets.
"""

import math
from collections import defaultdict

import igraph
import numpy as np

from ._constants import _RDF_TYPE
from ._logging import get_logger
from .stage2 import _connect_components

log = get_logger(__name__)

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
        log.warning("Stage 3: <2 content edges — skipping refinement, returning input graph")
        return g

    rel_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, p) in enumerate(content_edge_data):
        rel_to_idxs[p].append(i)

    swappable_rels = [r for r, lst in rel_to_idxs.items() if len(lst) >= 2]
    if not swappable_rels:
        log.warning("Stage 3: no relation has ≥2 edges to swap — skipping refinement")
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
    log.info(
        "Stage 3: refining (seed=%d, budget=%d) — target triangles=%d, motif4 targets=%s, "
        "assortativity=%s; initial loss=%.4f (triangles=%d)",
        seed, budget, target_tri, sorted(_motif4_targets),
        f"{target_r:.4f}" if use_assort else "off", current_loss, current_tri,
    )

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

    log.info(
        "Stage 3: done — accepted %d/%d swaps, best loss=%.4f, triangles=%d (target %d)",
        accepted, budget, best_loss, current_tri, target_tri,
    )

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
