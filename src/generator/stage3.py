"""Stage 3 — Maslov-Sneppen rewiring with simulated annealing.

Rewires content edges (never rdf:type edges) using degree-preserving
double-edge swaps to steer the graph toward Block E / Block F targets:

* triangle_count               — exact, incremental
* four_cycle / diamond / K4 / tailed-triangle counts  — remeasured every
  ``remeasure_interval`` accepted swaps via the colour-coding sampler
* five_cycle_count / six_cycle_count (induced/chordless) — exact, incremental
  when ``USE_INCREMENTAL_CYCLES`` is set; otherwise remeasured every
  ``remeasure_interval`` accepted swaps via CC sampling (k=5, k=6)
* CC_avg (average local clustering coefficient)       — exact, incremental;
  C(k_v, 2) denominators are invariant under degree-preserving swaps
* degree_assortativity         — exact, incremental; only the cross-product
  sum Q changes (degree sequence is invariant)

After the SA walk, the best-seen snapshot is returned and component
connectivity is restored to match the Block F targets.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import igraph
import numpy as np

from ._constants import _RDF_TYPE
from motif_counter import ExactMotifCounter, HybridMotifCounter, MotifCounter  # CCMotifCounter available as swap-in
from .local_updates import _adj_inc, _adj_dec, _triangle_node_delta, _motif4_delta, _cycle_delta
from ._logging import get_logger
from .stage2 import _connect_components

# ── Tuning constants (Stage-3 refinement) — adjust here ─────────────────────────
MAX_TARGETED_SWAP_PROB = 0.5  # cap on the probability of attempting a triangle-closing swap
TEMP_FLOOR = 1e-10             # numerical floor on the SA temperature in the accept test

# ── Loss function weights — one per component ────────────────────────────────
# Triangle and motif terms are normalised by target value (relative error in [0,∞]).
# Assortativity is absolute (r ∈ [−1,1]).  CC_avg is absolute (∈ [0,1]) but its
# absolute errors (~0.03-0.05) are much smaller than normalised motif terms
# (~0.3-1.0), so the default weight of 5 gives it comparable influence in the loss.
LOSS_WEIGHT_TRIANGLES:     float = 1.0
LOSS_WEIGHT_C4:            float = 1.0  # 4-cycle      (2,2,2,2)
LOSS_WEIGHT_DIAMOND:       float = 1.0  # diamond      (2,2,3,3)
LOSS_WEIGHT_K4:            float = 1.0  # complete K4  (3,3,3,3)
LOSS_WEIGHT_PAW:           float = 0  # paw          (1,2,2,3)
LOSS_WEIGHT_C5:            float = 0
LOSS_WEIGHT_C6:            float = 0
LOSS_WEIGHT_ASSORTATIVITY: float = 1.0
LOSS_WEIGHT_CC_AVG:        float = 1.0

# Lookup table used by _loss; keeps the function body concise.
_MOTIF4_WEIGHTS: dict[tuple, float] = {
    (2, 2, 2, 2): LOSS_WEIGHT_C4,
    (2, 2, 3, 3): LOSS_WEIGHT_DIAMOND,
    (3, 3, 3, 3): LOSS_WEIGHT_K4,
    (1, 2, 2, 3): LOSS_WEIGHT_PAW,
}
# Samples for the colour-coding motif estimator used during SA remeasure.
# Lower than block_e's measurement budget (which scales up to n*20) — steering
# only needs a rough signal, not a precise count.
CC4_SAMPLES = 10_000
# Samples for the 5/6-cycle CC remeasure (run every remeasure_interval accepted swaps).
# Fewer samples than 4-node because cycle estimation is higher-variance and the
# SA signal just needs a direction, not a precise count.
CC_CYCLE_SAMPLES = 5_000
# True = O(deg²) incremental delta per swap; False = CC sampler every remeasure_interval swaps.
USE_INCREMENTAL_MOTIF4 = True
# True = exact O(Δ^(k-1)) induced 5-/6-cycle delta per swap; False = CC remeasure every
# remeasure_interval swaps.  Incremental is exact but costs O(Δ⁴)/O(Δ⁵) per attempted swap,
# so it is best left off for high-degree graphs.  Only active when a cycle target is set.
USE_INCREMENTAL_CYCLES = True
# Accepted-swap interval between convergence CSV rows; 0 disables logging entirely.
CONVERGENCE_LOG_INTERVAL: int = 1000
# Motif counter used for the initial measurement at the start of the SA walk.
#INITIAL_MOTIF_COUNTER: MotifCounter = CCMotifCounter(n_samples=CC4_SAMPLES, seed=42)
INITIAL_MOTIF_COUNTER: MotifCounter = HybridMotifCounter()

# Motif counter used for periodic remeasurement every remeasure_interval accepted swaps.
# HybridMotifCounter: exact for k≤4, CC with CC_CYCLE_SAMPLES for k=5/6.
#REMEASURE_MOTIF_COUNTER: MotifCounter = CCMotifCounter(n_samples=CC4_SAMPLES, seed=43)
REMEASURE_MOTIF_COUNTER: MotifCounter = HybridMotifCounter(n_samples=CC_CYCLE_SAMPLES, seed=43)



log = get_logger(__name__)


def refine(
    g: igraph.Graph,
    target_e: "BlockE",
    *,
    target_f: "BlockF | None" = None,
    budget: int = 10_000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.999,
    remeasure_interval: int = 2000,
    seed: int = 0,
    convergence_log: "Path | str | None" = None,
) -> igraph.Graph:
    """Stage 3: Maslov-Sneppen rewiring + simulated annealing.

    Rewires content edges (never rdf:type edges) using degree-preserving
    double-edge swaps.  The SA objective is a weighted sum of relative errors
    across multiple targets:

    * Triangle count (exact, incremental via _triangle_node_delta)
    * Average local clustering coefficient CC_avg (exact, incremental —
      denominator C(k_v,2) is invariant; per-node Δt_v drives Δ(CC_avg))
    * 4-node motif counts — C4, diamond, K4, paw (remeasured every
      ``remeasure_interval`` accepted swaps via colour-coding sampler)
    * 5-cycle and 6-cycle counts (remeasured every ``remeasure_interval``
      accepted swaps via CC sampling; only when target > 0 in Block E)
    * Degree assortativity (exact, incremental — degree sequence is invariant
      under double-edge swaps, so only the cross-product sum Q changes)

    Parameters
    ----------
    g : igraph.Graph
        Output of Stage 2 (instantiate).
    target_e : BlockE
        Block E signature — supplies triangle_count, 4-node motif targets,
        and 5/6-cycle counts.
    target_f : BlockF, optional
        Block F signature — supplies degree_assortativity and
        clustering_coefficient targets.
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
    convergence_log : Path or str, optional
        If given, write a CSV with per-metric relative errors every
        ``CONVERGENCE_LOG_INTERVAL`` accepted swaps.  Columns are determined
        by which targets are active (unsteered motifs produce no columns).

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

    # -------------------------------------- CC_avg state (invariant denominators)
    # sim_deg[v] = number of *distinct* neighbours (simple undirected graph degree).
    # This differs from und_deg which counts directed/multi edges; CC_avg uses sim_deg.
    sim_deg = np.array([len(adj[v]) for v in range(n)], dtype=np.int64)
    # denom[v] = C(sim_deg[v], 2), floored at 1 to avoid division by zero.
    denom = np.maximum(sim_deg * (sim_deg - 1) // 2, 1).astype(np.float64)

    # Initialise per-node triangle counts t_node from igraph's local CC:
    #   t_v = CC_local_v * C(k_v, 2)   (exact for k_v >= 2, 0 for k_v < 2)
    _und_edges_init: list[tuple[int, int]] = []
    _seen_und: set[tuple[int, int]] = set()
    for s, o, _ in content_edge_data:
        key = (min(s, o), max(s, o))
        if key not in _seen_und:
            _seen_und.add(key)
            _und_edges_init.append(key)
    _g_init = igraph.Graph(n=n)
    if _und_edges_init:
        _g_init.add_edges(_und_edges_init)
    _cc_local_init = np.array(_g_init.transitivity_local_undirected(mode="zero"), dtype=np.float64)
    t_node = np.round(_cc_local_init * denom).astype(np.int64)

    target_cc = float(target_f.clustering_coefficient) if target_f is not None else float("nan")
    use_cc = not math.isnan(target_cc)
    cc_current = float(np.sum(t_node / denom) / n)

    def _build_und_graph() -> igraph.Graph:
        """Build a simple undirected igraph.Graph from current content edges."""
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
        return g_tmp

    target_tri = int(target_e.triangle_count)
    current_tri = INITIAL_MOTIF_COUNTER.count_triangles(_build_und_graph())

    # ----------------------------------------- 4-node motif targets & counter
    # Only track a motif type when it has both a positive target AND a nonzero
    # loss weight — a zero-weight term never enters the loss, so computing its
    # (O(Δ³)) per-swap delta would be wasted work.
    _motif4_targets: dict[tuple, int] = {}
    for deg_seq, attr in [
        ((2, 2, 2, 2), "four_cycle_count"),
        ((2, 2, 3, 3), "diamond_count"),
        ((3, 3, 3, 3), "k4_count"),
        ((1, 2, 2, 3), "tailed_triangle_count"),
    ]:
        val = getattr(target_e, attr, 0)
        if val and val > 0 and _MOTIF4_WEIGHTS.get(deg_seq, 0) > 0:
            _motif4_targets[deg_seq] = int(val)

    current_motifs4 = INITIAL_MOTIF_COUNTER.count_motifs4(_build_und_graph()) if _motif4_targets else {}

    # ----------------------------------------- 5/6-cycle targets
    # As with 4-node motifs, only steer a cycle size when its loss weight is
    # nonzero — the cycle delta is the costliest per-swap update (O(Δ⁴)/O(Δ⁵)).
    _target_c5 = int(getattr(target_e, "five_cycle_count", 0) or 0)
    _target_c6 = int(getattr(target_e, "six_cycle_count", 0) or 0)
    use_c5 = _target_c5 > 0 and LOSS_WEIGHT_C5 > 0
    use_c6 = _target_c6 > 0 and LOSS_WEIGHT_C6 > 0

    def _measure_cycles(g_und: igraph.Graph) -> tuple[int, int]:
        """Estimate 5- and 6-cycle counts via the remeasure counter."""
        return REMEASURE_MOTIF_COUNTER.count_cycles(g_und, k5=use_c5, k6=use_c6)

    _g_init_und = _build_und_graph() if (use_c5 or use_c6) else None
    current_c5, current_c6 = _measure_cycles(_g_init_und) if _g_init_und is not None else (0, 0)

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
        """Newman degree-assortativity from the cross-product sum Q.

        r = (2M·Q − (S/2)²) / (M·T − (S/2)²)
        where M = edge count, S = Σ_e(d_u+d_v), T = Σ_e(d_u²+d_v²).
        S and T are invariant; only Q changes with each swap.
        """
        denom = M_e * T_deg - S_deg ** 2 / 2.0
        if denom == 0.0:
            return 0.0
        return (2.0 * M_e * Q - S_deg ** 2 / 2.0) / denom

    # ------------------------------------------------------- loss function
    def _loss(tri: int, motifs: dict, Q: float, cc: float, c5: int, c6: int) -> float:
        """SA objective: weighted sum of relative errors across all active targets.

        All terms use |current − target| / |target|.  A floor of 1e-9 guards
        against division by zero when a target is exactly 0.
        """
        loss = LOSS_WEIGHT_TRIANGLES * abs(tri - target_tri) / max(1, target_tri)
        for ds, tgt in _motif4_targets.items():
            loss += _MOTIF4_WEIGHTS.get(ds, 1.0) * abs(motifs.get(ds, 0) - tgt) / tgt
        if use_c5:
            loss += LOSS_WEIGHT_C5 * abs(c5 - _target_c5) / _target_c5
        if use_c6:
            loss += LOSS_WEIGHT_C6 * abs(c6 - _target_c6) / _target_c6
        if use_assort:
            loss += LOSS_WEIGHT_ASSORTATIVITY * abs(_assort_from_Q(Q) - target_r) / max(1e-9, abs(target_r))
        if use_cc:
            loss += LOSS_WEIGHT_CC_AVG * abs(cc - target_cc) / max(1e-9, target_cc)
        return loss

    current_loss = _loss(current_tri, current_motifs4, Q_deg, cc_current, current_c5, current_c6)
    best_loss = current_loss
    best_content = list(content_edge_data)
    best_accepted = 0
    temp = initial_temp
    accepted = 0

    # ------------------------------------------------- convergence CSV setup
    _DS_NAME = {(2, 2, 2, 2): "c4_err", (2, 2, 3, 3): "diamond_err",
                (3, 3, 3, 3): "k4_err",  (1, 2, 2, 3): "paw_err"}
    # Block E attribute name → (deg_seq key into _motif4_targets, sig column name)
    _SIG_ATTRS = [
        ("triangle_count",       None,            "sig_tri_err"),
        ("four_cycle_count",     (2, 2, 2, 2),    "sig_c4_err"),
        ("diamond_count",        (2, 2, 3, 3),    "sig_diamond_err"),
        ("k4_count",             (3, 3, 3, 3),    "sig_k4_err"),
        ("tailed_triangle_count",(1, 2, 2, 3),    "sig_paw_err"),
    ]

    _conv_fields = ["accepted", "loss", "tri_err"]
    _conv_fields += [_DS_NAME[ds] for ds in _motif4_targets if ds in _DS_NAME]
    if use_c5:
        _conv_fields.append("c5_err")
    if use_c6:
        _conv_fields.append("c6_err")
    if use_cc:
        _conv_fields.append("cc_err")
    if use_assort:
        _conv_fields.append("assort_err")
    # sig_ columns: triangle always included; 4-motif only when the target is active
    _conv_fields.append("sig_tri_err")
    for attr, ds, col in _SIG_ATTRS[1:]:
        if ds in _motif4_targets:
            _conv_fields.append(col)

    _conv_fh = open(convergence_log, "w", newline="") if convergence_log else None  # noqa: SIM115
    _conv_writer = csv.DictWriter(_conv_fh, fieldnames=_conv_fields) if _conv_fh else None
    if _conv_writer:
        _conv_writer.writeheader()

    def _write_conv_row() -> None:
        if not _conv_writer:
            return
        row: dict = {
            "accepted": accepted,
            "loss":     round(current_loss, 6),
            "tri_err":  round(abs(current_tri - target_tri) / max(1, target_tri), 6),
        }
        for ds, name in _DS_NAME.items():
            if name in _conv_fields:
                row[name] = round(
                    abs(current_motifs4.get(ds, 0) - _motif4_targets[ds])
                    / max(1, _motif4_targets[ds]), 6
                )
        if use_c5:
            row["c5_err"] = round(abs(current_c5 - _target_c5) / max(1, _target_c5), 6)
        if use_c6:
            row["c6_err"] = round(abs(current_c6 - _target_c6) / max(1, _target_c6), 6)
        if use_cc:
            row["cc_err"] = round(abs(cc_current - target_cc) / max(1e-9, target_cc), 6)
        if use_assort:
            row["assort_err"] = round(abs(_assort_from_Q(Q_deg) - target_r) / max(1e-9, abs(target_r)), 6)

        # Ground-truth errors via periodic counter (same strategy as remeasure)
        _g_sig = _build_und_graph()
        tri_sig = REMEASURE_MOTIF_COUNTER.count_triangles(_g_sig)
        row["sig_tri_err"] = round(abs(tri_sig - target_tri) / max(1, target_tri), 6)
        if any(col in _conv_fields for _, _, col in _SIG_ATTRS[1:]):
            _sig_motifs4 = REMEASURE_MOTIF_COUNTER.count_motifs4(_g_sig)
            for _, ds, col in _SIG_ATTRS[1:]:
                if col in _conv_fields:
                    tgt = _motif4_targets[ds]
                    row[col] = round(abs(_sig_motifs4.get(ds, 0) - tgt) / max(1, tgt), 6)

        _conv_writer.writerow(row)

    _write_conv_row()  # row 0: baseline before any swap

    log.info(
        "Stage 3: refining (seed=%d, budget=%d) — target triangles=%d, motif4 targets=%s, "
        "5-cycle=%s, 6-cycle=%s, assortativity=%s, cc_avg=%s; "
        "initial loss=%.4f (triangles=%d, cc_avg=%.4f)",
        seed, budget, target_tri, sorted(_motif4_targets),
        _target_c5 if use_c5 else "off",
        _target_c6 if use_c6 else "off",
        f"{target_r:.4f}" if use_assort else "off",
        f"{target_cc:.4f}" if use_cc else "off",
        current_loss, current_tri, cc_current,
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

    for step in range(budget):
        if step > 0 and step % 5_000 == 0:
            log.info(
                "Stage 3: step %d/%d — loss=%.4f, tri=%d (target %d), accepted=%d",
                step, budget, current_loss, current_tri, target_tri, accepted,
            )
        # Attempt targeted triangle-creating swap when triangles are below target.
        # The probability scales with how large the deficit is (max 50%).
        tri_deficit = target_tri - current_tri
        p_targeted = float(min(MAX_TARGETED_SWAP_PROB, tri_deficit / max(1, target_tri)))
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

        tri_delta, node_delta = _triangle_node_delta(adj, s1, o1, s2, o2)
        new_tri = current_tri + tri_delta
        # CC_avg delta: weighted sum of per-node triangle changes (denom fixed)
        new_cc = cc_current + sum(dt / denom[v] for v, dt in node_delta.items()) / n
        # Assortativity delta: only Q changes (degrees are invariant)
        dQ = float(und_deg[s1] * und_deg[o2] + und_deg[s2] * und_deg[o1]
                   - und_deg[s1] * und_deg[o1] - und_deg[s2] * und_deg[o2])
        new_Q = Q_deg + dQ
        if USE_INCREMENTAL_MOTIF4 and _motif4_targets:
            _m4d = _motif4_delta(adj, s1, o1, s2, o2)
            new_motifs4 = {k: current_motifs4.get(k, 0) + _m4d.get(k, 0) for k in _motif4_targets}
        else:
            new_motifs4 = current_motifs4
        # 5/6-cycle counts: exact incremental delta when enabled, else carry the
        # current values (remeasured every remeasure_interval).
        if USE_INCREMENTAL_CYCLES and (use_c5 or use_c6):
            _dc5, _dc6 = _cycle_delta(adj, s1, o1, s2, o2, k5=use_c5, k6=use_c6)
            new_c5 = current_c5 + _dc5
            new_c6 = current_c6 + _dc6
        else:
            new_c5, new_c6 = current_c5, current_c6
        new_loss = _loss(new_tri, new_motifs4, new_Q, new_cc, new_c5, new_c6)

        if new_loss < current_loss:
            accept = True
        else:
            diff = new_loss - current_loss
            accept = bool(rng.random() < math.exp(-diff / max(temp, TEMP_FLOOR)))

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
            for v, dt in node_delta.items():
                t_node[v] += dt
            # Recompute from t_node to prevent float drift from accumulated deltas.
            cc_current = float(np.sum(t_node / denom) / n)
            current_loss = new_loss
            temp *= cooling_rate
            accepted += 1

            if USE_INCREMENTAL_MOTIF4 and _motif4_targets:
                current_motifs4 = new_motifs4
            if USE_INCREMENTAL_CYCLES and (use_c5 or use_c6):
                current_c5, current_c6 = new_c5, new_c6

            do_remeasure = accepted % remeasure_interval == 0
            if not USE_INCREMENTAL_MOTIF4 and _motif4_targets and do_remeasure:
                current_motifs4 = REMEASURE_MOTIF_COUNTER.count_motifs4(_build_und_graph())
                current_loss = _loss(current_tri, current_motifs4, Q_deg, cc_current, current_c5, current_c6)
            if not USE_INCREMENTAL_CYCLES and (use_c5 or use_c6) and do_remeasure:
                current_c5, current_c6 = _measure_cycles(_build_und_graph())
                current_loss = _loss(current_tri, current_motifs4, Q_deg, cc_current, current_c5, current_c6)

            if CONVERGENCE_LOG_INTERVAL > 0 and accepted % CONVERGENCE_LOG_INTERVAL == 0:
                _write_conv_row()

            if current_loss < best_loss:
                best_loss = current_loss
                best_content = list(content_edge_data)
                best_accepted = accepted

    if _conv_fh:
        _conv_fh.close()

    log.info(
        "Stage 3: done — accepted %d/%d swaps, best loss=%.4f at accepted=%d, "
        "triangles=%d (target %d), cc_avg=%.4f (target %s)",
        accepted, budget, best_loss, best_accepted, current_tri, target_tri,
        cc_current, f"{target_cc:.4f}" if use_cc else "off",
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

    _connect_components(
        best_content, n, _FakeSchema(), rng, seen_best, in_deg_best,
        target_nc=int(target_f.num_components) if target_f is not None else 1,
        target_lcc=float(target_f.largest_component_fraction) if target_f is not None else 1.0,
    )

    # Rebuild igraph from best snapshot, preserving vertex attributes
    all_best = best_content + type_edge_data
    g_out = igraph.Graph(n=g.vcount(), directed=True)
    for attr in g.vertex_attributes():
        g_out.vs[attr] = list(g.vs[attr])
    if all_best:
        g_out.add_edges([(s, o) for s, o, _ in all_best])
        g_out.es["predicate"] = [p for _, _, p in all_best]

    g_out["stage3_best_accepted"] = best_accepted
    g_out["stage3_best_loss"] = round(best_loss, 6)

    return g_out
