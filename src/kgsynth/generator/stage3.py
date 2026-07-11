"""Stage 3 — Maslov-Sneppen rewiring with simulated annealing.

Rewires content edges (never rdf:type edges) using degree-preserving
double-edge swaps to steer the graph toward Block E / Block F targets.  Every
tracked target is updated *incrementally* per swap via an exact O(Δ^k) delta:

* triangle_count               — exact, incremental
* four_cycle / diamond / K4 / tailed-triangle counts — exact, incremental
* five_cycle_count / six_cycle_count (induced/chordless) — exact, incremental
* CC_avg (average local clustering coefficient)       — exact, incremental;
  C(k_v, 2) denominators are invariant under degree-preserving swaps
* degree_assortativity         — exact, incremental; only the cross-product
  sum Q changes (degree sequence is invariant)
* tree/path template entropy — exact, incremental

After the SA walk, the best-seen snapshot is returned and component
connectivity is restored to match the Block F targets.
"""

import csv
import math
import select
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:          # for the quoted forward-ref annotations only (no runtime import)
    from ..signature import BlockE, BlockF

try:                       # POSIX-only: single-key stdin reads for manual early-exit
    import termios
    import tty
except ImportError:        # non-POSIX (e.g. Windows) — escape watcher becomes a no-op
    termios = None
    tty = None

import igraph
import numpy as np

from ._constants import _RDF_TYPE
from ..motif_counter import HybridMotifCounter, MotifCounter  # CCMotifCounter available as swap-in
from .local_updates import (
    _adj_inc,
    _adj_dec,
    _triangle_node_delta,
    _motif4_delta,
    _cycle_delta,
    _tree_entropy_delta,
    _path_entropy_delta,
    _entropy_from_freq,
)
from .._logging import get_logger
from .stage2 import _connect_components

# ── Tuning constants (Stage-3 refinement) — adjust here ─────────────────────────
MAX_TARGETED_SWAP_PROB = 0.5   # cap on the probability of attempting a triangle-closing swap
TEMP_FLOOR = 1e-10             # numerical floor on the SA temperature in the accept test
ESCAPE_CHECK_INTERVAL = 10     # poll stdin for a manual-exit keypress every N rewiring steps
ADAPTIVE_WEIGHT_SCALE = 50.0   # high-strength multiplier for adaptive_weights' linear term:
                                # weight = base_weight * ADAPTIVE_WEIGHT_SCALE * error


class _EscapeWatcher:
    """Non-blocking stdin watcher for manual early-exit of the rewiring loop.

    On entry it puts an interactive TTY into cbreak mode so single keypresses
    register without Enter; :meth:`pressed` then reports (without blocking)
    whether ESC or ``q`` is waiting on stdin.  A no-op when stdin is not an
    interactive TTY (piped input, non-POSIX platform) so batch/CI runs are
    unaffected.  Use as a context manager to guarantee the terminal is restored.
    """

    def __init__(self):
        self._enabled = False
        self._fd = None
        self._old_attrs = None

    def __enter__(self) -> "_EscapeWatcher":
        if termios is None or not (sys.stdin and sys.stdin.isatty()):
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except (termios.error, ValueError, OSError):
            self._enabled = False  # e.g. not a real terminal — stay a no-op
        return self

    def pressed(self) -> bool:
        """Return True if ESC or ``q`` is waiting on stdin (drains the buffer)."""
        if not self._enabled:
            return False
        hit = False
        # Drain everything currently buffered so held/enter keys don't accumulate.
        while select.select([self._fd], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch in ("\x1b", "q", "Q"):
                hit = True
        return hit

    def __exit__(self, *exc) -> bool:
        if self._enabled and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        return False

# ── Loss function weights — one per component ────────────────────────────────
# Triangle and motif terms are normalised by target value (relative error in [0,∞]).
# Assortativity is absolute (r ∈ [−1,1]).  CC_avg is absolute (∈ [0,1]) but its
# absolute errors (~0.03-0.05) are much smaller than normalised motif terms
# (~0.3-1.0), so the default weight of 5 gives it comparable influence in the loss.
LOSS_WEIGHT_TRIANGLES:     float = 1.0
LOSS_WEIGHT_C4:            float = 1.0  # 4-cycle      (2,2,2,2)
LOSS_WEIGHT_DIAMOND:       float = 1.0  # diamond      (2,2,3,3)
LOSS_WEIGHT_K4:            float = 1.0  # complete K4  (3,3,3,3)
LOSS_WEIGHT_PAW:           float = 1.0  # paw          (1,2,2,3)
LOSS_WEIGHT_C5:            float = 1
LOSS_WEIGHT_C6:            float = 1
LOSS_WEIGHT_ASSORTATIVITY: float = 1.0
LOSS_WEIGHT_CC_AVG:        float = 1.0
LOSS_WEIGHT_TREE_ENTROPY:  float = 0
LOSS_WEIGHT_PATH_ENTROPY:  float = 0

# Lookup table used by _loss; keeps the function body concise.
_MOTIF4_WEIGHTS: dict[tuple, float] = {
    (2, 2, 2, 2): LOSS_WEIGHT_C4,
    (2, 2, 3, 3): LOSS_WEIGHT_DIAMOND,
    (3, 3, 3, 3): LOSS_WEIGHT_K4,
    (1, 2, 2, 3): LOSS_WEIGHT_PAW,
}
# Convergence-column stem per 4-node motif degree sequence ("c4" → "c4_err").
_DS_STEM: dict[tuple, str] = {
    (2, 2, 2, 2): "c4",
    (2, 2, 3, 3): "diamond",
    (3, 3, 3, 3): "k4",
    (1, 2, 2, 3): "paw",
}
# Samples for the 5/6-cycle CC estimator used for the initial cycle baseline and
# the optional ground-truth convergence remeasure.  Fewer samples than the 4-node
# budget because cycle estimation is higher-variance and only a direction is needed.
CC_CYCLE_SAMPLES = 20000
# Samples for the CC estimator behind the initial triangle/motif4 baseline.
# Raised from HybridMotifCounter's default (10000) to match CC_CYCLE_SAMPLES.
MOTIF4_INITIAL_SAMPLES = 20000
# Degree guard for the incremental cycle delta, applied to **every node the
# induced-path DFS expands** — swap endpoints and path interiors alike (an
# interior hub explodes the O(Δ^(k-2)) search even when all four endpoints are
# small; see experiments/stage3_delta_profiling/).  On the first node whose
# simple degree exceeds this, the delta is dropped for that swap and the
# 5/6-cycle counts carry over unchanged, so the cycle loss terms are identical
# before and after and neither favour nor penalise the swap.
# Set to float("inf") to disable the guard (always compute the exact delta).  When
# left at the sentinel below, ``refine()`` derives a per-graph threshold instead
# (see ``CYCLE_DELTA_MAX_DEGREE_PERCENTILE``) — profiling on wn18rr_v4 showed
# this single delta accounting for >50% of Stage 3 wall-clock with the guard off.
CYCLE_DELTA_MAX_DEGREE: float = float("inf") # sentinel: auto-derive from degree percentile
# Percentile of the simple-degree distribution used to auto-derive
# CYCLE_DELTA_MAX_DEGREE when it is left at its sentinel (-1.0).  Swaps touching
# the top (100 - this) % of nodes by degree skip the exact cycle delta; those
# hub-heavy swaps are also the ones the O(Δ^(k-2)) cost explodes on, so this
# trades a small amount of exactness on rare hub swaps for a large constant-factor
# speedup on typical ones.
CYCLE_DELTA_MAX_DEGREE_PERCENTILE: float = 100
# Degree guard for the incremental 4-node motif delta, applied to the four swap
# endpoints only — unlike the cycle DFS, _motif4_delta's cost is fully determined
# by the endpoint neighbourhoods (candidates come from N(a)∪N(b), so a max
# endpoint degree D bounds the work at O(D²) per pair; no interior explosion).
# Swaps whose max endpoint degree exceeds this skip the delta and carry the
# motif4 counts over unchanged (loss terms cancel in the accept test).
# Profiled on fb237_v4 (experiments/stage3_delta_profiling/): unguarded mean
# ~57 ms/proposal (max 1.2 s on hub swaps); at 200, 88 % of proposals still
# compute the delta with a ~47 ms worst case (~11× less total motif4 work).
# Set to float("inf") to disable the guard (always compute the exact delta).
MOTIF4_DELTA_MAX_DEGREE = float("inf")
# Proposal-swap interval between convergence CSV rows; 0 disables logging entirely.
# Counts every evaluated proposal (accepted or rejected), not just accepted swaps.
CONVERGENCE_LOG_INTERVAL: int = 100
# True = also log ground-truth errors (sig_* columns) by remeasuring the full graph
# via the measure counter on every convergence row. False = log only the locally
# updated (incrementally tracked) errors, avoiding the expensive global remeasurement.
CONVERGENCE_LOG_GLOBAL_REMEASURE: bool = False
# Stage 3 measurement counters are built per run so they follow refine()'s
# ``seed`` (the pipeline's seed+2) rather than a fixed seed — keeping the whole
# generation reproducible from the single master seed. Swap the counter type /
# n_samples here; the seed is always supplied by refine() at generation time.


def _make_initial_motif_counter(seed: int) -> MotifCounter:
    """Counter for the initial triangle/motif4 baseline at the start of the walk."""
    # return CCMotifCounter(n_samples=CC_CYCLE_SAMPLES, seed=seed)  # CC-only alternative
    return HybridMotifCounter(n_samples=MOTIF4_INITIAL_SAMPLES, seed=seed)


def _make_measure_counter(seed: int) -> MotifCounter:
    """Counter for full-graph measurement — the initial 5/6-cycle baseline and the
    optional ground-truth convergence remeasure (``CONVERGENCE_LOG_GLOBAL_REMEASURE``).

    HybridMotifCounter: exact for k≤4, CC with CC_CYCLE_SAMPLES for k=5/6.
    """
    # return CCMotifCounter(n_samples=CC_CYCLE_SAMPLES, seed=seed)  # CC-only alternative
    return HybridMotifCounter(n_samples=CC_CYCLE_SAMPLES, seed=seed)


log = get_logger(__name__)


@dataclass
class _SAState:
    """Bundle of every loss-tracked metric value at one point in the walk.

    Shared by the SA loss (weighted sum of relative errors) and the convergence
    logger (unweighted dump) so each error term is defined in exactly one place.
    """

    tri: int
    motifs4: dict
    Q: float
    cc: float
    c5: int
    c6: int
    tree_h: float
    path_h: float


def _materialize_graph(
    g: igraph.Graph,
    content_edge_data: list[tuple[int, int, str]],
    type_edge_data: list[tuple[int, int, str]],
) -> igraph.Graph:
    """Build an independent ``igraph.Graph`` snapshot from edge-tuple lists.

    Shared by the final-output rebuild and checkpoint snapshots so both
    produce structurally identical graphs (same vertex attributes, same edge
    attribute schema) from whatever edge-tuple state is passed in.

    Parameters
    ----------
    g : igraph.Graph
        Original Stage-2 graph — supplies vertex count and attributes.
    content_edge_data : list of (int, int, str)
        Current content-edge ``(source, target, predicate)`` tuples.
    type_edge_data : list of (int, int, str)
        Unchanging ``rdf:type`` edge tuples.

    Returns
    -------
    igraph.Graph
        A new graph with ``g``'s vertex attributes and the given edges.
    """
    all_edges = content_edge_data + type_edge_data
    snapshot = igraph.Graph(n=g.vcount(), directed=True)
    for attr in g.vertex_attributes():
        snapshot.vs[attr] = list(g.vs[attr])
    if all_edges:
        snapshot.add_edges([(s, o) for s, o, _ in all_edges])
        snapshot.es["predicate"] = [p for _, _, p in all_edges]
    return snapshot


def refine(
    g: igraph.Graph,
    target_e: "BlockE",
    *,
    target_f: "BlockF | None" = None,
    budget: int = 10_000,
    initial_temp: float = 0.05,
    cooling_rate: float = 0.99993,
    seed: int = 0,
    skip_c5: bool = False,
    skip_c6: bool = False,
    adaptive_weights: bool = False,
    convergence_log: "Path | str | None" = None,
    swap_log: "Path | str | None" = None,
    checkpoint_steps: "list[int] | None" = None,
    checkpoint_callback: "Callable[[int, igraph.Graph], None] | None" = None,
) -> igraph.Graph:
    """Stage 3: Maslov-Sneppen rewiring + simulated annealing.

    Rewires content edges (never rdf:type edges) using degree-preserving
    double-edge swaps.  The SA objective is a weighted sum of relative errors
    across multiple targets, all tracked exactly and incrementally per swap:

    * Triangle count (via _triangle_node_delta)
    * Average local clustering coefficient CC_avg (denominator C(k_v,2) is
      invariant; per-node Δt_v drives Δ(CC_avg))
    * 4-node motif counts — C4, diamond, K4, paw (via _motif4_delta)
    * 5-cycle and 6-cycle counts (via _cycle_delta; only when target > 0)
    * Degree assortativity (degree sequence is invariant under double-edge
      swaps, so only the cross-product sum Q changes)
    * Tree/path template entropy

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
        Starting SA temperature. Default 0.05 ≈ a few× the typical per-swap
        |Δloss| (~0.008 on wn18rr_v4), giving ~85% initial uphill acceptance.
        Re-derive per graph when the loss scale differs (fb237's is larger).
    cooling_rate : float
        Geometric decay per accepted swap. Default 0.99993 sweeps the temperature
        from ~0.05 to ~0.001 over ~55k accepted swaps (≈ a 100k-attempt budget),
        so the final ~20% of the walk is effectively greedy. For a much smaller
        budget, cool faster (e.g. ~0.998 for a 5k budget) or the walk never cools.
    seed : int
        RNG seed.
    skip_c5, skip_c6 : bool
        Force 5-/6-cycle steering off regardless of the target count and loss
        weight. Suppresses the (costly) per-swap cycle delta for that size so
        its contribution to the loss is dropped entirely.
    adaptive_weights : bool
        If True, each term's loss weight is scaled linearly by its own current
        error magnitude, with a high fixed multiplier (``weight = base_weight *
        ADAPTIVE_WEIGHT_SCALE * error``) instead of held fixed at
        ``base_weight``. Terms that are already close to target contribute
        little to the loss; terms still far off are pushed harder,
        proportionally to their error. Recomputed every accepted
        swap from ``current`` (the live SA state), not the
        candidate being scored, so a single swap can't move its own weight.
    convergence_log : Path or str, optional
        If given, write a CSV with per-metric relative errors every
        ``CONVERGENCE_LOG_INTERVAL`` *proposals* (accepted or rejected).  The
        ``step`` column is the proposal index (x-axis) and ``accepted`` the
        accepted-swap count so far.  Columns are otherwise determined by which
        targets are active (unsteered motifs produce no columns).
    swap_log : Path or str, optional
        If given, write one CSV row per *evaluated* swap proposal with its
        per-motif deltas and accept decision: proposal context (``step``,
        ``targeted``, endpoint simple degrees, ``deg_max4``), ``d_tri``,
        ``d_c4``/``d_diamond``/``d_k4``/``d_paw`` (only for active motif4
        targets), ``d_c5``/``d_c6`` (only when steered), ``d_loss`` and
        ``accepted``.  Delta cells are empty when a degree guard dropped the
        delta for that proposal.  Proposals discarded before any delta is
        computed (self-loop guard) produce no row.
    checkpoint_steps : list of int, optional
        Loop indices at which to materialize a graph snapshot and invoke
        ``checkpoint_callback`` with it. ``0`` means the pre-loop graph,
        before any swap is attempted. A step at or beyond where the loop
        actually stopped (``budget``, or earlier on a manual escape) fires
        with the same *best-so-far* graph this call returns — every other
        step fires with the walk's *live current* state at that point (which
        may be worse than the best seen, since SA accepts uphill moves).
        Ignored if ``checkpoint_callback`` is ``None``, and never fires at all
        if either early-return guard below is hit (too few content edges, or
        no relation has ≥2 edges to swap) — in both cases the input graph
        ``g`` is returned unrefined and no rewiring occurs to snapshot.
    checkpoint_callback : callable, optional
        ``(step: int, graph: igraph.Graph) -> None``, called once per step in
        ``checkpoint_steps`` (in ascending order) with an ``igraph.Graph``
        snapshot — for tracing how the graph evolves through the annealing
        run (see ``scripts/signature_pca_trajectory.py``). Each snapshot is
        an independent graph; mutating it does not affect the ongoing
        rewiring, *except* that a trailing/final-step snapshot is the same
        object this call returns (not a copy) — treat it as read-only if you
        also use this function's return value.

    Returns
    -------
    igraph.Graph
        Best graph encountered during the annealing walk. Carries
        ``stage3_best_accepted``, ``stage3_best_loss``,
        ``stage3_best_unweighted_error_sum`` (``Σ|error|`` over ``_error_terms``
        at the best snapshot — unlike ``stage3_best_loss`` this ignores
        ``adaptive_weights``/``ADAPTIVE_WEIGHT_SCALE``, so it's the metric to
        use when comparing runs across different weighting schemes), and
        ``stage3_executed_steps`` (< ``budget`` if a manual escape stopped the
        loop early) as graph attributes.
    """
    rng = np.random.default_rng(seed)
    # Measurement counters follow the same run seed as the rewiring RNG.
    initial_motif_counter = _make_initial_motif_counter(seed)
    measure_counter = _make_measure_counter(seed)

    # ------------------------------------------------------------------ setup
    type_edge_data: list[tuple[int, int, str]] = []
    content_edge_data: list[tuple[int, int, str]] = []
    for e in g.es:
        entry = (e.source, e.target, e["predicate"])
        if e["predicate"] == _RDF_TYPE:
            type_edge_data.append(entry)
        else:
            content_edge_data.append(entry)

    # Steps at which to fire checkpoint_callback with a snapshot of the walk's
    # current state; step 0 (pre-loop, before any swap) fires immediately.
    _checkpoints = sorted(set(checkpoint_steps)) if checkpoint_callback and checkpoint_steps else []
    if _checkpoints and _checkpoints[0] == 0:
        checkpoint_callback(0, _materialize_graph(g, content_edge_data, type_edge_data))
        _checkpoints = _checkpoints[1:]

    if len(content_edge_data) < 2:
        log.warning("Stage 3: <2 content edges — skipping refinement, returning input graph")
        for _step in _checkpoints:
            checkpoint_callback(_step, g)
        return g

    rel_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, p) in enumerate(content_edge_data):
        rel_to_idxs[p].append(i)

    swappable_rels = [r for r, lst in rel_to_idxs.items() if len(lst) >= 2]
    if not swappable_rels:
        log.warning("Stage 3: no relation has ≥2 edges to swap — skipping refinement")
        for _step in _checkpoints:
            checkpoint_callback(_step, g)
        return g

    n = g.vcount()
    adj: list[dict] = [{} for _ in range(n)]
    for s, o, _ in content_edge_data:
        _adj_inc(adj, s, o)

    def _und_edges() -> list[tuple[int, int]]:
        """Deduplicated simple undirected edge list from current content edges."""
        edge_set: set[tuple[int, int]] = set()
        und_edges: list[tuple[int, int]] = []
        for s, o, _ in content_edge_data:
            key = (min(s, o), max(s, o))
            if key not in edge_set:
                edge_set.add(key)
                und_edges.append(key)
        return und_edges

    def _build_und_graph() -> igraph.Graph:
        """Build a simple undirected igraph.Graph from current content edges."""
        g_tmp = igraph.Graph(n=n)
        und_edges = _und_edges()
        if und_edges:
            g_tmp.add_edges(und_edges)
        return g_tmp

    # -------------------------------------- CC_avg state (invariant denominators)
    # sim_deg[v] = number of *distinct* neighbours (simple undirected graph degree).
    # This differs from und_deg which counts directed/multi edges; CC_avg uses sim_deg.
    sim_deg = np.array([len(adj[v]) for v in range(n)], dtype=np.int64)
    # denom[v] = C(sim_deg[v], 2), floored at 1 to avoid division by zero.
    denom = np.maximum(sim_deg * (sim_deg - 1) // 2, 1).astype(np.float64)

    # Auto-derive the cycle-delta degree guard from this graph's degree
    # distribution when CYCLE_DELTA_MAX_DEGREE is left at its sentinel (-1.0).
    # Doing this per-graph (rather than a fixed constant) keeps the guard
    # meaningful across graphs of very different scale/density.
    cycle_delta_max_degree = CYCLE_DELTA_MAX_DEGREE
    if cycle_delta_max_degree < 0:
        cycle_delta_max_degree = float(np.percentile(sim_deg, CYCLE_DELTA_MAX_DEGREE_PERCENTILE))

    # Initialise per-node triangle counts t_node from igraph's local CC:
    #   t_v = CC_local_v * C(k_v, 2)   (exact for k_v >= 2, 0 for k_v < 2)
    _g_init = igraph.Graph(n=n)
    _und_edges_init = _und_edges()
    if _und_edges_init:
        _g_init.add_edges(_und_edges_init)
    _cc_local_init = np.array(_g_init.transitivity_local_undirected(mode="zero"), dtype=np.float64)
    t_node = np.round(_cc_local_init * denom).astype(np.int64)

    target_cc = float(target_f.clustering_coefficient) if target_f is not None else float("nan")
    use_cc = not math.isnan(target_cc)
    cc_current = float(np.sum(t_node / denom) / n)

    target_tri = int(target_e.triangle_count)
    current_tri = initial_motif_counter.count_triangles(_build_und_graph())

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

    # Motif types steered this run — passed to _motif4_delta so it can take the
    # fast O(Δ²) path (valid when the paw is not among them).
    _m4_types = frozenset(_motif4_targets)

    current_motifs4 = (
        initial_motif_counter.count_motifs4(_build_und_graph()) if _motif4_targets else {}
    )

    # ----------------------------------------- 5/6-cycle targets
    # As with 4-node motifs, only steer a cycle size when its loss weight is
    # nonzero — the cycle delta is the costliest per-swap update (O(Δ⁴)/O(Δ⁵)).
    _target_c5 = int(getattr(target_e, "five_cycle_count", 0) or 0)
    _target_c6 = int(getattr(target_e, "six_cycle_count", 0) or 0)
    use_c5 = _target_c5 > 0 and LOSS_WEIGHT_C5 > 0 and not skip_c5
    use_c6 = _target_c6 > 0 and LOSS_WEIGHT_C6 > 0 and not skip_c6

    def _measure_cycles(g_und: igraph.Graph) -> tuple[int, int]:
        """Estimate 5- and 6-cycle counts via the full-graph measure counter."""
        return measure_counter.count_cycles(g_und, k5=use_c5, k6=use_c6)

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

    # ----------------------------------------- depth-2 tree template entropy
    # rel_out[v] = list of outgoing relation labels from v (content edges only).
    # pair_freq[(r1, r2)] = number of root→child(r1)→grandchild(r2) observations.
    _target_tree_entropy = float(getattr(target_e, "tree_template_entropy", float("nan")))
    use_tree_entropy = (
        not math.isnan(_target_tree_entropy)
        and _target_tree_entropy > 0
        and LOSS_WEIGHT_TREE_ENTROPY > 0
    )

    rel_out: list[list[str]] = [[] for _ in range(n)]
    for s, o, p in content_edge_data:
        rel_out[s].append(p)

    # Build initial (r1, r2) pair frequency dict from all root→child→grandchild triples.
    _pair_freq: dict[tuple, int] = {}
    if use_tree_entropy:
        for s, o, p in content_edge_data:
            for r2 in rel_out[o]:
                key = (p, r2)
                _pair_freq[key] = _pair_freq.get(key, 0) + 1

    tree_entropy_current = _entropy_from_freq(_pair_freq) if use_tree_entropy else 0.0

    # ----------------------------------------- k-hop path template entropy (k=2,3)
    # out_edges[v] = list of (relation, target) pairs for directed outgoing edges.
    # path_freqs[k][(r1,...,rk)] = count of first-hop-anchored k-hop paths starting
    # with the first swapped relation.  We track k=2 and k=3 only.
    # path_template_entropy is a dict {k: float} on BlockE; use k=3 as the target.
    _pte_dict = getattr(target_e, "path_template_entropy", None) or {}
    _target_path_entropy_k3 = float(
        _pte_dict.get(3, float("nan")) if isinstance(_pte_dict, dict) else float("nan")
    )
    use_path_entropy = (
        not math.isnan(_target_path_entropy_k3)
        and _target_path_entropy_k3 > 0
        and LOSS_WEIGHT_PATH_ENTROPY > 0
    )

    # out_edges is needed for both tree (via rel_out already built) and path entropy.
    # For path entropy we need (rel, target) pairs, not just labels.
    out_edges: list[list] = [[] for _ in range(n)]
    for s, o, p in content_edge_data:
        out_edges[s].append((p, o))

    _path_freqs: dict[int, dict] = {}
    if use_path_entropy:
        # k=2: (r1, r2) sequences, anchored at first-hop relation r1
        freq2: dict[tuple, int] = {}
        for s, o, p in content_edge_data:
            for r2, _ in out_edges[o]:
                key = (p, r2)
                freq2[key] = freq2.get(key, 0) + 1
        # k=3: (r1, r2, r3) sequences
        freq3: dict[tuple, int] = {}
        for s, o, p in content_edge_data:
            for r2, mid in out_edges[o]:
                for r3, _ in out_edges[mid]:
                    key = (p, r2, r3)
                    freq3[key] = freq3.get(key, 0) + 1
        _path_freqs = {2: freq2, 3: freq3}

    path_entropy_current = _entropy_from_freq(_path_freqs.get(3, {})) if use_path_entropy else 0.0

    # ------------------------------------------------------- error terms & loss
    def _error_terms(st: _SAState, *, signed: bool = False) -> dict[str, float]:
        """Per-target relative error for every active target, keyed by convergence-
        column stem (``tri`` → ``tri_err``).

        By default returns the unsigned magnitude ``|current − target| / |target|``.
        With ``signed=True`` the same quantity keeps its sign
        ``(current − target) / |target|`` (negative = under target, positive = over)
        — used for the convergence CSV so the direction of each miss is visible.

        Single source of truth: the SA loss is the weighted sum over the unsigned
        dict and the convergence CSV is the rounded, unweighted signed dump.  A floor
        of 1e-9 guards divisions where the absolute target can approach 0.  Insertion
        order matches the loss summation order.
        """
        def _rel(cur: float, tgt: float, denom: float) -> float:
            d = (cur - tgt) / denom
            return d if signed else abs(d)

        terms: dict[str, float] = {
            "tri": _rel(st.tri, target_tri, max(1, target_tri)),
        }
        for ds, tgt in _motif4_targets.items():
            terms[_DS_STEM[ds]] = _rel(st.motifs4.get(ds, 0), tgt, tgt)
        if use_c5:
            terms["c5"] = _rel(st.c5, _target_c5, _target_c5)
        if use_c6:
            terms["c6"] = _rel(st.c6, _target_c6, _target_c6)
        if use_assort:
            terms["assort"] = _rel(_assort_from_Q(st.Q), target_r, max(1e-9, abs(target_r)))
        if use_cc:
            terms["cc"] = _rel(st.cc, target_cc, max(1e-9, target_cc))
        if use_tree_entropy:
            terms["tree_entropy"] = _rel(
                st.tree_h, _target_tree_entropy, max(1e-9, _target_tree_entropy)
            )
        if use_path_entropy:
            terms["path_entropy_k3"] = _rel(
                st.path_h, _target_path_entropy_k3, max(1e-9, _target_path_entropy_k3)
            )
        return terms

    # Base weight per active error-term stem; parallels _error_terms so the loss is a
    # plain weighted sum over the shared term dict.
    _base_weights: dict[str, float] = {"tri": LOSS_WEIGHT_TRIANGLES}
    for ds in _motif4_targets:
        _base_weights[_DS_STEM[ds]] = _MOTIF4_WEIGHTS.get(ds, 1.0)
    if use_c5:
        _base_weights["c5"] = LOSS_WEIGHT_C5
    if use_c6:
        _base_weights["c6"] = LOSS_WEIGHT_C6
    if use_assort:
        _base_weights["assort"] = LOSS_WEIGHT_ASSORTATIVITY
    if use_cc:
        _base_weights["cc"] = LOSS_WEIGHT_CC_AVG
    if use_tree_entropy:
        _base_weights["tree_entropy"] = LOSS_WEIGHT_TREE_ENTROPY
    if use_path_entropy:
        _base_weights["path_entropy_k3"] = LOSS_WEIGHT_PATH_ENTROPY

    # _term_weights holds the weights actually used by _loss. In the fixed-weight
    # case (default) it never changes. In adaptive mode it is refreshed once per
    # accepted swap from the *current* (pre-swap) SA state — see _refresh_weights
    # below — so scoring a candidate never lets it influence its own weight.
    _term_weights: dict[str, float] = dict(_base_weights)

    def _refresh_weights(st: "_SAState") -> None:
        """Rescale each term's weight linearly by its own current error, with a
        high fixed multiplier: weight = base * ADAPTIVE_WEIGHT_SCALE * error.

        No-op when adaptive_weights is off (weights stay at their base value).
        A term already at its target (error 0) drops to weight 0; a term far off
        gets pushed harder, proportionally to its error. Mutates _term_weights
        in place.
        """
        if not adaptive_weights:
            return
        for name, err in _error_terms(st).items():
            _term_weights[name] = _base_weights[name] * ADAPTIVE_WEIGHT_SCALE * err

    def _loss(st: "_SAState") -> float:
        """SA objective: weighted sum of the shared per-target relative errors."""
        return sum(_term_weights[name] * err for name, err in _error_terms(st).items())

    current = _SAState(
        tri=current_tri, motifs4=current_motifs4, Q=Q_deg, cc=cc_current,
        c5=current_c5, c6=current_c6, tree_h=tree_entropy_current,
        path_h=path_entropy_current,
    )
    _refresh_weights(current)
    current_loss = _loss(current)
    best_loss = current_loss
    best_content = list(content_edge_data)
    best_state = current
    best_accepted = 0
    temp = initial_temp
    accepted = 0

    # ------------------------------------------------- convergence CSV setup
    # Block E attribute name → (deg_seq key into _motif4_targets, sig column name)
    _SIG_ATTRS = [
        ("triangle_count",       None,            "sig_tri_err"),
        ("four_cycle_count",     (2, 2, 2, 2),    "sig_c4_err"),
        ("diamond_count",        (2, 2, 3, 3),    "sig_diamond_err"),
        ("k4_count",             (3, 3, 3, 3),    "sig_k4_err"),
        ("tailed_triangle_count",(1, 2, 2, 3),    "sig_paw_err"),
    ]

    # Error columns mirror the active terms exactly (one per _error_terms key).
    # ``step`` = proposals evaluated so far (x-axis); ``accepted`` = accepted swaps so far.
    _conv_fields = ["step", "accepted", "loss"] + [f"{name}_err" for name in _error_terms(current)]
    # weight_ columns track each term's live loss weight; only meaningful (and only
    # logged) in adaptive mode since fixed weights never move.
    if adaptive_weights:
        _conv_fields += [f"weight_{name}" for name in _error_terms(current)]
    # sig_ columns are only logged when the global remeasurement is enabled; they
    # hold ground-truth errors from remeasuring the full graph (see _write_conv_row).
    if CONVERGENCE_LOG_GLOBAL_REMEASURE:
        # sig_ columns: triangle always included; 4-motif only when the target is active
        _conv_fields.append("sig_tri_err")
        for attr, ds, col in _SIG_ATTRS[1:]:
            if ds in _motif4_targets:
                _conv_fields.append(col)
        # Ground-truth 5-cycle error: global (induced) 5-cycle count measured on the
        # full graph (HybridMotifCounter) vs the target — validates the incremental
        # cycle delta.
        _conv_fields.append("sig_c5_err")

    _conv_fh = open(convergence_log, "w", newline="") if convergence_log else None  # noqa: SIM115
    _conv_writer = csv.DictWriter(_conv_fh, fieldnames=_conv_fields) if _conv_fh else None
    if _conv_writer:
        _conv_writer.writeheader()

    def _write_conv_row(step: int) -> None:
        if not _conv_writer:
            return
        # Locally tracked errors: the same terms that drive the loss (single source),
        # dumped signed so each miss's direction (under/over target) is visible.
        row: dict = {"step": step, "accepted": accepted, "loss": round(current_loss, 6)}
        for name, err in _error_terms(current, signed=True).items():
            row[f"{name}_err"] = round(err, 6)
        if adaptive_weights:
            for name in _error_terms(current):
                row[f"weight_{name}"] = round(_term_weights[name], 6)

        # Ground-truth errors via a full-graph remeasurement (expensive); only when
        # enabled. Otherwise the row carries just the locally tracked errors above.
        if CONVERGENCE_LOG_GLOBAL_REMEASURE:
            # Signed relative errors, matching the locally tracked columns above.
            _g_sig = _build_und_graph()
            tri_sig = measure_counter.count_triangles(_g_sig)
            row["sig_tri_err"] = round((tri_sig - target_tri) / max(1, target_tri), 6)
            if any(col in _conv_fields for _, _, col in _SIG_ATTRS[1:]):
                _sig_motifs4 = measure_counter.count_motifs4(_g_sig)
                for _, ds, col in _SIG_ATTRS[1:]:
                    if col in _conv_fields:
                        tgt = _motif4_targets[ds]
                        row[col] = round((_sig_motifs4.get(ds, 0) - tgt) / max(1, tgt), 6)

            # Ground-truth 5-cycle relative error: global (induced/chordless) count
            # measured via the hybrid counter vs target (raw count when target is 0).
            c5_sig = measure_counter.count_cycles(_g_sig, k5=True, k6=False)[0]
            row["sig_c5_err"] = round((c5_sig - _target_c5) / max(1, _target_c5), 6)

        _conv_writer.writerow(row)

    _write_conv_row(0)  # row 0: baseline before any swap

    # ------------------------------------------------- swap-proposal CSV setup
    # One row per evaluated proposal: context, per-motif deltas (columns only for
    # actively steered motifs, mirroring the convergence log), Δloss and the
    # accept decision.  Guard-dropped deltas leave their cells empty.
    _swap_fields = ["step", "targeted", "deg_s1", "deg_o1", "deg_s2", "deg_o2",
                    "deg_max4", "d_tri"]
    _swap_fields += [f"d_{_DS_STEM[ds]}" for ds in _motif4_targets]
    if use_c5:
        _swap_fields.append("d_c5")
    if use_c6:
        _swap_fields.append("d_c6")
    _swap_fields += ["d_loss", "accepted"]
    _swap_fh = open(swap_log, "w", newline="") if swap_log else None  # noqa: SIM115
    _swap_writer = csv.DictWriter(_swap_fh, fieldnames=_swap_fields) if _swap_fh else None
    if _swap_writer:
        _swap_writer.writeheader()

    log.info(
        "Stage 3: refining (seed=%d, budget=%d, adaptive_weights=%s) — target triangles=%d, "
        "motif4 targets=%s, "
        "5-cycle=%s, 6-cycle=%s, assortativity=%s, cc_avg=%s, tree_entropy=%s, path_entropy_k3=%s, "
        "cycle_delta_max_degree=%s; "
        "initial loss=%.4f (triangles=%d, cc_avg=%.4f, tree_entropy=%.4f, path_entropy=%.4f)",
        seed, budget, adaptive_weights, target_tri, sorted(_motif4_targets),
        _target_c5 if use_c5 else "off",
        _target_c6 if use_c6 else "off",
        f"{target_r:.4f}" if use_assort else "off",
        f"{target_cc:.4f}" if use_cc else "off",
        f"{_target_tree_entropy:.4f}" if use_tree_entropy else "off",
        f"{_target_path_entropy_k3:.4f}" if use_path_entropy else "off",
        f"{cycle_delta_max_degree:.1f}" if (use_c5 or use_c6) else "n/a",
        current_loss, current.tri, current.cc, current.tree_h, current.path_h,
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

    # hub_nodes: nodes with undirected degree ≥ min star k, sorted by degree descending.
    # Used by _targeted_star_swap (unused now that star steering is disabled; kept
    # as a helper for future use) to find triangle-rich centers to de-cluster.
    _min_star_k = 2
    _hub_nodes = sorted(
        [v for v in range(n) if len(adj[v]) >= _min_star_k],
        key=lambda v: -len(adj[v]),
    )
    # Keep only the top fraction of hubs (the ones with most to gain).
    _hub_pool = _hub_nodes[:max(1, len(_hub_nodes) // 5)]

    def _targeted_star_swap():
        """Find a swap that breaks a triangle among the neighbors of a high-degree hub.

        Picks a hub node v (high undirected degree), finds two neighbors u1, u2 that
        are mutually connected (forming a triangle with v), then proposes redirecting
        an edge *into* u2 from some other source s2 to a non-neighbor target instead.
        After the swap, u1 and u2 are no longer connected, increasing v's induced
        k-star contributions.

        Specifically: pick edge s2→u2 (= i2) and an edge u1→o1 (= i1) with the same
        predicate p.  The swap produces u1→u2 and s2→o1 — wait, that *closes* u1-u2.
        Instead we want to *break* the u1-u2 edge by using it as i1 and pairing with
        a random edge i2 that redirects u2 away from u1.

        Strategy: pick edge i1 = (u1→u2) (the triangle edge to break), find any edge
        i2 = (s2→o2) with the same predicate where o2 ∉ adj[u1] and s2 ≠ u2.
        After swap: (u1→o2, s2→u2).  The u1–u2 undirected link is broken (u1 loses u2
        as neighbor, s2 gains u2 as neighbor).  This increases v's induced star count.

        Returns (i1, i2, s1, o1, s2, o2, p1) or None if no candidate found.
        """
        if not _hub_pool:
            return None
        v = _hub_pool[int(rng.integers(len(_hub_pool)))]
        nbrs_v = list(adj[v].keys())
        if len(nbrs_v) < 2:
            return None
        # Find a pair of mutually connected neighbors (triangle through v).
        rng.shuffle(nbrs_v)
        u1, u2 = None, None
        for i, nb in enumerate(nbrs_v):
            for nb2 in nbrs_v[i + 1:]:
                if nb2 in adj[nb]:
                    u1, u2 = nb, nb2
                    break
            if u1 is not None:
                break
        if u1 is None:
            return None  # no triangle through v
        # Find a directed edge u1→u2 (i1 to break).
        u1_edges = []
        for p_cand, src_map in edge_src_by_pred.items():
            idxs = src_map.get(u1, [])
            for idx in idxs:
                _, tgt, _ = content_edge_data[idx]
                if tgt == u2:
                    u1_edges.append((idx, p_cand))
        if not u1_edges:
            # No directed edge u1→u2; try u2→u1 instead.
            for p_cand, src_map in edge_src_by_pred.items():
                idxs = src_map.get(u2, [])
                for idx in idxs:
                    _, tgt, _ = content_edge_data[idx]
                    if tgt == u1:
                        u1_edges.append((idx, p_cand))
                        u1, u2 = u2, u1  # swap roles so i1 is u1→u2
                        break
                if u1_edges:
                    break
        if not u1_edges:
            return None
        i1, pred = u1_edges[int(rng.integers(len(u1_edges)))]
        s1, o1, p1 = content_edge_data[i1]  # s1=u1, o1=u2
        # Find a swap partner i2 = (s2→o2) with same predicate, where:
        #   o2 ∉ adj[s1] (so we don't just recreate another inner edge)
        #   s2 ≠ o1 and o2 ≠ s1 (no self-loops)
        pool = rel_to_idxs.get(pred, [])
        if len(pool) < 2:
            return None
        for _ in range(10):  # up to 10 tries to find a valid partner
            i2 = pool[int(rng.integers(len(pool)))]
            if i2 == i1:
                continue
            s2, o2, _ = content_edge_data[i2]
            if s2 == o1 or o2 == s1 or o2 in adj[s1]:
                continue
            return i1, i2, s1, o1, s2, o2, p1
        return None

    # Guard bookkeeping: per delta family, proposals whose delta was computed vs
    # dropped because a degree guard fired (cycles: any DFS-expanded node above
    # cycle_delta_max_degree; motif4: max endpoint degree above MOTIF4_DELTA_MAX_DEGREE).
    cycle_delta_computed = 0
    cycle_delta_dropped = 0
    motif4_delta_computed = 0
    motif4_delta_dropped = 0

    # Triangle-steering attribution: how much of the accepted triangle *increase*
    # comes from the biased _targeted_swap proposals vs the random swaps.  A
    # "triangle steer" = an accepted proposal with tri_delta > 0 (targeting only
    # fires below target, so upward moves are the steering signal).  Counted over
    # evaluated proposals only (those that reach the accept test), matching the
    # swap-log rows.
    evaluated_proposals = 0       # proposals reaching the accept test (swap-log rows)
    targeted_proposals = 0        # of those, from a biased _targeted_swap
    targeted_accepted = 0         # of the targeted, accepted
    tri_up_targeted = 0           # accepted tri-up swaps from targeted proposals
    tri_up_untargeted = 0         # accepted tri-up swaps from random proposals
    tri_gain_targeted = 0         # Σ tri_delta of the targeted tri-up swaps
    tri_gain_untargeted = 0       # Σ tri_delta of the random  tri-up swaps

    # Steps actually attempted — equals `budget` unless a manual escape (below)
    # breaks out early; exposed on the returned graph so callers (e.g. the
    # auto-named log filenames in signature_roundtrip.py) can reflect what
    # actually ran instead of what was requested.
    executed_steps = 0

    with _EscapeWatcher() as _esc_watch:
        for step in range(budget):
            executed_steps = step + 1
            # Poll stdin every ESCAPE_CHECK_INTERVAL steps so the rewiring can be
            # aborted manually (ESC or 'q') without killing the process — the
            # best-so-far graph found up to here is still returned below.
            if step % ESCAPE_CHECK_INTERVAL == 0 and _esc_watch.pressed():
                log.info("Stage 3: manual escape at step %d/%d — stopping rewiring "
                         "early (best loss=%.4f at accepted=%d)",
                         step, budget, best_loss, best_accepted)
                break
            if step > 0 and step % 5_000 == 0:
                log.info(
                    "Stage 3: step %d/%d — loss=%.4f, tri=%d (target %d), accepted=%d, "
                    "deltas computed/dropped (degree guards) — motif4 %d/%d, cycles %d/%d",
                    step, budget, current_loss, current.tri, target_tri, accepted,
                    motif4_delta_computed, motif4_delta_dropped,
                    cycle_delta_computed, cycle_delta_dropped,
                )
            # Log convergence state every N *proposals* (accepted or not); step 0 is the
            # pre-loop baseline written above.
            if CONVERGENCE_LOG_INTERVAL > 0 and step > 0 and step % CONVERGENCE_LOG_INTERVAL == 0:
                _write_conv_row(step)
            # _checkpoints is sorted ascending and step increases monotonically, so the
            # next due checkpoint (if any) is always at index 0.
            if _checkpoints and step == _checkpoints[0]:
                checkpoint_callback(step, _materialize_graph(g, content_edge_data, type_edge_data))
                _checkpoints = _checkpoints[1:]
            # Attempt targeted triangle-creating swap when triangles are below target.
            # The probability scales with how large the deficit is (max 50%).
            tri_deficit = target_tri - current.tri
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

            evaluated_proposals += 1
            targeted_proposals += int(targeted)
            tri_delta, node_delta = _triangle_node_delta(adj, s1, o1, s2, o2)
            new_tri = current.tri + tri_delta
            # CC_avg delta: weighted sum of per-node triangle changes (denom fixed)
            new_cc = current.cc + sum(dt / denom[v] for v, dt in node_delta.items()) / n
            # Assortativity delta: only Q changes (degrees are invariant)
            dQ = float(und_deg[s1] * und_deg[o2] + und_deg[s2] * und_deg[o1]
                       - und_deg[s1] * und_deg[o1] - und_deg[s2] * und_deg[o2])
            new_Q = current.Q + dQ
            if _motif4_targets:
                # Endpoint-degree guard: _motif4_delta enumerates candidate pairs from
                # the endpoints' neighbourhoods (O(Δ²) per pair), so hub endpoints make
                # it the dominant per-swap cost on dense graphs.  Skipped swaps carry
                # the motif4 counts over unchanged — the motif4 loss terms cancel in
                # the accept test, as with the cycle guard.
                if (
                    max(len(adj[s1]), len(adj[o1]), len(adj[s2]), len(adj[o2]))
                    > MOTIF4_DELTA_MAX_DEGREE
                ):
                    motif4_delta_dropped += 1
                    _m4d = None  # guard-dropped: swap log leaves the motif4 cells empty
                    new_motifs4 = current.motifs4
                else:
                    motif4_delta_computed += 1
                    _m4d = _motif4_delta(adj, s1, o1, s2, o2, types=_m4_types)
                    new_motifs4 = {
                        k: current.motifs4.get(k, 0) + _m4d.get(k, 0) for k in _motif4_targets
                    }
            else:
                _m4d = None
                new_motifs4 = current.motifs4
            # 5/6-cycle counts: exact incremental delta when steered.
            if use_c5 or use_c6:
                # The O(Δ^(k-2)) cycle delta guards every node its path DFS expands
                # (endpoints and interiors — an interior hub explodes the search even
                # between low-degree endpoints) against cycle_delta_max_degree (the
                # resolved threshold: CYCLE_DELTA_MAX_DEGREE, or its per-graph
                # percentile-derived value at the sentinel) and returns None on the
                # first hub encountered.  Carrying the counts over unchanged leaves
                # the cycle loss terms identical before and after, so they cancel in
                # the accept test (no positive/negative contribution for these swaps).
                _dc = _cycle_delta(adj, s1, o1, s2, o2, k5=use_c5, k6=use_c6,
                                   max_degree=cycle_delta_max_degree)
                if _dc is None:
                    cycle_delta_dropped += 1
                    new_c5, new_c6 = current.c5, current.c6
                else:
                    cycle_delta_computed += 1
                    new_c5, new_c6 = current.c5 + _dc[0], current.c6 + _dc[1]
            else:
                new_c5, new_c6 = current.c5, current.c6
            # Tree template entropy: exact incremental update via _tree_entropy_delta.
            # rel_out is kept in sync with content_edge_data on accept.
            if use_tree_entropy:
                new_tree_h, new_pair_freq = _tree_entropy_delta(
                    rel_out, _pair_freq, s1, o1, p1, s2, o2
                )
            else:
                new_tree_h, new_pair_freq = current.tree_h, _pair_freq
            # Path template entropy: exact incremental update via _path_entropy_delta.
            # out_edges[v] = [(rel, target), ...]; updated on accept when targets change.
            if use_path_entropy:
                _new_path_ents, _new_path_freqs = _path_entropy_delta(
                    out_edges, _path_freqs, s1, o1, p1, s2, o2
                )
                new_path_h = _new_path_ents.get(3, 0.0)
            else:
                _new_path_ents, _new_path_freqs = {}, _path_freqs
                new_path_h = current.path_h

            candidate = _SAState(
                tri=new_tri, motifs4=new_motifs4, Q=new_Q, cc=new_cc,
                c5=new_c5, c6=new_c6, tree_h=new_tree_h, path_h=new_path_h,
            )
            new_loss = _loss(candidate)

            if new_loss < current_loss:
                accept = True
            else:
                diff = new_loss - current_loss
                accept = bool(rng.random() < math.exp(-diff / max(temp, TEMP_FLOOR)))

            if _swap_writer:
                # Degrees are read pre-swap (adj is only mutated below on accept),
                # matching what the guards saw for this proposal.
                _degs = (len(adj[s1]), len(adj[o1]), len(adj[s2]), len(adj[o2]))
                _row = {
                    "step": step, "targeted": int(targeted),
                    "deg_s1": _degs[0], "deg_o1": _degs[1],
                    "deg_s2": _degs[2], "deg_o2": _degs[3],
                    "deg_max4": max(_degs), "d_tri": tri_delta,
                    "d_loss": round(new_loss - current_loss, 6), "accepted": int(accept),
                }
                if _m4d is not None:
                    for ds in _motif4_targets:
                        _row[f"d_{_DS_STEM[ds]}"] = _m4d.get(ds, 0)
                if (use_c5 or use_c6) and _dc is not None:
                    if use_c5:
                        _row["d_c5"] = _dc[0]
                    if use_c6:
                        _row["d_c6"] = _dc[1]
                _swap_writer.writerow(_row)

            if accept:
                targeted_accepted += int(targeted)
                if tri_delta > 0:
                    if targeted:
                        tri_up_targeted += 1
                        tri_gain_targeted += tri_delta
                    else:
                        tri_up_untargeted += 1
                        tri_gain_untargeted += tri_delta

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

                for v, dt in node_delta.items():
                    t_node[v] += dt
                # Recompute cc from t_node to prevent float drift from accumulated deltas;
                # this drift-corrected value (not the incremental new_cc used in the loss)
                # becomes the baseline for the next swap.
                candidate.cc = float(np.sum(t_node / denom) / n)

                if use_tree_entropy:
                    # A same-relation swap leaves source nodes' relation labels unchanged
                    # (s1 still has p1, just to a new object) — only _pair_freq changes.
                    _pair_freq = new_pair_freq
                if use_path_entropy:
                    # Update out_edges: s1 now points to o2 (was o1), s2 now points to o1 (was o2).
                    # The relation p1 stays the same; only the target changes.
                    for i, (r, t) in enumerate(out_edges[s1]):
                        if r == p1 and t == o1:
                            out_edges[s1][i] = (p1, o2)
                            break
                    for i, (r, t) in enumerate(out_edges[s2]):
                        if r == p1 and t == o2:
                            out_edges[s2][i] = (p1, o1)
                            break
                    _path_freqs = _new_path_freqs

                current = candidate
                # Rescale weights from the *new* current state before recomputing
                # current_loss, so current_loss and best_loss stay comparable
                # under the same weight snapshot (adaptive mode only; no-op otherwise).
                _refresh_weights(current)
                current_loss = _loss(current)
                temp *= cooling_rate
                accepted += 1

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_content = list(content_edge_data)
                    best_state = current
                    best_accepted = accepted

    if _conv_fh:
        _conv_fh.close()
    if _swap_fh:
        _swap_fh.close()

    log.info(
        "Stage 3: done — accepted %d/%d swaps, best loss=%.4f at accepted=%d, "
        "triangles=%d (target %d), cc_avg=%.4f (target %s)",
        accepted, budget, best_loss, best_accepted, current.tri, target_tri,
        current.cc, f"{target_cc:.4f}" if use_cc else "off",
    )
    # Triangle-steering attribution — what share of the accepted triangle increase
    # the biased _targeted_swap proposals actually delivered vs the random swaps.
    _tri_up = tri_up_targeted + tri_up_untargeted
    _tri_gain = tri_gain_targeted + tri_gain_untargeted
    log.info(
        "Stage 3: triangle steering — %d targeted proposals (%.1f%% of all), accept rate "
        "%.3f (vs %.3f random); of %d accepted triangle-up swaps %d (%.1f%%) were targeted, "
        "contributing %d/%d (%.1f%%) of the total +%d triangle gain",
        targeted_proposals,
        100.0 * targeted_proposals / max(1, evaluated_proposals),
        targeted_accepted / max(1, targeted_proposals),
        (accepted - targeted_accepted) / max(1, evaluated_proposals - targeted_proposals),
        _tri_up, tri_up_targeted, 100.0 * tri_up_targeted / max(1, _tri_up),
        tri_gain_targeted, _tri_gain, 100.0 * tri_gain_targeted / max(1, _tri_gain),
        _tri_gain,
    )
    if use_c5 or use_c6:
        _n_cyc = cycle_delta_computed + cycle_delta_dropped
        log.info(
            "Stage 3: 5/6-cycle deltas — computed %d, dropped %d of %d proposals (%.1f%%) "
            "by the degree guard (cycle_delta_max_degree=%s); dropped swaps carried their "
            "cycle counts over unchanged",
            cycle_delta_computed, cycle_delta_dropped, _n_cyc,
            100.0 * cycle_delta_dropped / max(1, _n_cyc), cycle_delta_max_degree,
        )
    if _motif4_targets:
        _n_m4 = motif4_delta_computed + motif4_delta_dropped
        log.info(
            "Stage 3: motif4 deltas — computed %d, dropped %d of %d proposals (%.1f%%) "
            "by the endpoint-degree guard (MOTIF4_DELTA_MAX_DEGREE=%s); dropped swaps "
            "carried their motif4 counts over unchanged",
            motif4_delta_computed, motif4_delta_dropped, _n_m4,
            100.0 * motif4_delta_dropped / max(1, _n_m4), MOTIF4_DELTA_MAX_DEGREE,
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
    g_out = _materialize_graph(g, best_content, type_edge_data)

    # Unweighted error sum at the best snapshot: independent of adaptive_weights /
    # ADAPTIVE_WEIGHT_SCALE, so it's the fair metric for comparing runs across
    # different weighting schemes (unlike best_loss, which bakes the weights in).
    best_unweighted_error_sum = sum(_error_terms(best_state).values())

    g_out["stage3_executed_steps"] = executed_steps
    g_out["stage3_best_accepted"] = best_accepted
    g_out["stage3_best_loss"] = round(best_loss, 6)
    g_out["stage3_best_unweighted_error_sum"] = round(best_unweighted_error_sum, 6)

    # Any checkpoint step at or beyond where the loop actually stopped (the
    # requested budget, or earlier on a manual escape) fires with the same
    # graph this call returns — after the stage3_* attributes above are set,
    # so a synchronous reader sees them too.
    for _step in _checkpoints:
        checkpoint_callback(_step, g_out)

    return g_out
