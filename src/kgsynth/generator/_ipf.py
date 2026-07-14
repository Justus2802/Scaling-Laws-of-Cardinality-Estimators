"""Joint (entity × relation) stub allocation by iterative proportional fitting.

Stage 2 must decide, for every entity ``v`` and every relation ``r`` it is eligible for,
how many out-stubs ``X[v, r]`` and in-stubs ``Y[v, r]`` it gets. Two families of
constraint bear on that choice at once:

* **rows** — ``Σ_r X[v, r] = tgt_out[v]``: the per-entity degree targets sampled from
  Block B's degree law;
* **columns** — ``Σ_v X[v, r] = Σ_v Y[v, r] = e_r``: the per-relation edge budget, which
  must be the *same* on both sides or the relation's stubs cannot be paired at all.

The old wiring loop drew each relation's stubs independently
(``m_obj ~ Multinomial(edges_r, w)``), which hits the column margin and says nothing
about the rows, then bolted the row constraint on afterwards as a *cap*. Capping is what
broke the column margin again: the two sides were truncated independently against the
global remaining quota of their own pools, so they ended up with different sums, and
whatever could not be paired fell through to a uniform-random deficit pass. On ``aids``
that was a third of the content edges.

This module solves for both margins at once instead. Given a weight matrix ``W`` on the
sparse support (the same power-law × CS-size weights the loop already computed), IPF —
Sinkhorn's algorithm, Deming–Stephan's iterative proportional fitting — alternately
rescales rows and columns until both margins hold. Every operation multiplies a whole row
or column by one positive scalar, so the fitted matrix is ``A[v, r] = W[v, r]·u[v]·w[r]``:
the search is over ``V + R`` numbers, not ``nnz`` of them.

That form is what makes this the right tool rather than merely a working one. All
cross-ratios of ``W`` survive exactly ::

    A[v,r]·A[v',r'] / (A[v,r']·A[v',r])  ==  W[v,r]·W[v',r'] / (W[v,r']·W[v',r])

— the ``u``/``w`` factors cancel. The multiplicity law and the G2b CS-size coupling come
out untouched; only the margins are forced. Formally, ``A`` is the I-projection of ``W``
onto the transportation polytope: the closest matrix to ``W`` in KL divergence with the
required margins, i.e. the maximum-entropy allocation consistent with both.

Infeasibility is a *feature* here, not a failure mode. Convergence holds iff a matrix with
that support and those margins exists; when it does not, ending the loop on a row step
still spends every entity's degree quota exactly, and the achieved column sums report what
the supports could actually deliver. See :func:`solve_edge_budget`.
"""

import numpy as np

from .._logging import get_logger

log = get_logger(__name__)

IPF_ITERS = 60          # inner Sinkhorn sweeps; convergence is geometric, 60 is ample
OUTER_ITERS = 12        # outer sweeps reconciling the two sides' column margins
_EPS = 1e-12


def build_support(memberships: list, num_relations: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-entity relation memberships into a column-major sparse support.

    ``memberships[v]`` is the relation-index array for entity ``v`` (its CS on the out
    side, its inverse CS on the in side). Returns ``(rows, cols)`` sorted by
    ``(col, row)``, so the slice of a column holds its entities in **ascending index
    order** — exactly the order ``subjects_by_rel[r]`` / ``objects_by_rel[r]`` are built
    in, which lets a column slice be used as that relation's stub vector directly.

    :param memberships: per-entity arrays of eligible relation indices.
    :param num_relations: total relation count (unused members are simply absent).
    :returns: ``(rows, cols)`` int64 arrays of length ``nnz = Σ_v |memberships[v]|``.
    """
    rows_l, cols_l = [], []
    for v, rels in enumerate(memberships):
        if rels is None:
            continue
        for r in rels:
            rows_l.append(v)
            cols_l.append(int(r))
    if not rows_l:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    rows = np.asarray(rows_l, dtype=np.int64)
    cols = np.asarray(cols_l, dtype=np.int64)
    order = np.lexsort((rows, cols))          # primary: col, secondary: row
    return rows[order], cols[order]


def ipf(
    rows: np.ndarray,
    cols: np.ndarray,
    val: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    n_rows: int,
    n_cols: int,
    iters: int = IPF_ITERS,
) -> np.ndarray:
    """Fit ``val`` to the given row/column margins by alternating rescaling.

    Ends on a **row** step, so the row margins hold exactly on return and the achieved
    column sums are whatever the support could deliver. That asymmetry is deliberate: the
    row margins are the degree law (which must not bend) and the column margins are the
    relation budget (which :func:`solve_edge_budget` is allowed to renegotiate).

    :param rows: row index per non-zero.
    :param cols: column index per non-zero.
    :param val: seed weight per non-zero (the modelled attractiveness of that pairing).
    :param row_target: desired sum per row, length ``n_rows``.
    :param col_target: desired sum per column, length ``n_cols``.
    :param n_rows: number of rows (entities).
    :param n_cols: number of columns (relations).
    :param iters: Sinkhorn sweeps.
    :returns: fitted non-negative values per non-zero, with exact row sums.
    """
    if rows.size == 0:
        return np.array([], dtype=float)
    v = np.maximum(np.asarray(val, dtype=float), _EPS)
    rt = np.asarray(row_target, dtype=float)
    ct = np.asarray(col_target, dtype=float)
    u = np.ones(n_rows)
    w = np.ones(n_cols)
    for _ in range(iters):
        a = v * u[rows] * w[cols]
        rs = np.bincount(rows, a, minlength=n_rows)
        u *= rt / np.maximum(rs, _EPS)

        a = v * u[rows] * w[cols]
        cs = np.bincount(cols, a, minlength=n_cols)
        w *= ct / np.maximum(cs, _EPS)

    # Final row step: row margins exact on return.
    a = v * u[rows] * w[cols]
    rs = np.bincount(rows, a, minlength=n_rows)
    u *= rt / np.maximum(rs, _EPS)
    return v * u[rows] * w[cols]


def _largest_remainder(values: np.ndarray, total: int) -> np.ndarray:
    """Round ``values`` to non-negative integers summing exactly to ``total``."""
    if values.size == 0:
        return np.array([], dtype=np.int64)
    floors = np.floor(values).astype(np.int64)
    short = int(total) - int(floors.sum())
    if short > 0:
        order = np.argsort(-(values - floors), kind="stable")
        floors[order[:short]] += 1
    elif short < 0:
        # Trim from the entries with the least fractional claim that still have a unit.
        order = np.argsort(values - floors, kind="stable")
        for i in order:
            if short == 0:
                break
            if floors[i] > 0:
                floors[i] -= 1
                short += 1
    return floors


def round_to_columns(
    cols: np.ndarray, a: np.ndarray, col_target: np.ndarray, n_cols: int
) -> np.ndarray:
    """Round ``a`` to integers whose **column sums equal ``col_target`` exactly**.

    Largest-remainder within each column. The column margins are the per-relation stub
    counts, so preserving them exactly through the rounding is what makes the two sides
    pairable — a rounding scheme that only preserved the total would reintroduce the very
    imbalance this module exists to remove. Row sums may drift by ±1 as a result; the
    degree targets are the softer of the two constraints (see the module docstring).

    :param cols: column index per non-zero (must be sorted by column).
    :param a: fractional values per non-zero.
    :param col_target: exact integer sum required per column.
    :param n_cols: number of columns.
    :returns: int64 stub counts per non-zero.
    """
    if cols.size == 0:
        return np.array([], dtype=np.int64)
    tgt = np.asarray(col_target, dtype=np.int64)

    # Rescale each column to its target *before* flooring. This is what makes the
    # remainder step provably sufficient: once Σ_i b_i == e_c exactly, Σ_i floor(b_i) ≤
    # floor(Σ_i b_i) == e_c, so the units still owed are never negative and never exceed
    # the column's entry count. Rounding the raw values instead can leave a column already
    # over its target, with no well-defined entry to take the excess back from.
    colsum = np.bincount(cols, a, minlength=n_cols)
    scale = np.where(colsum > _EPS, tgt / np.maximum(colsum, _EPS), 0.0)
    b = a * scale[cols]

    out = np.floor(b).astype(np.int64)
    frac = b - out
    need = tgt - np.bincount(cols, out, minlength=n_cols).astype(np.int64)

    # Rank each entry within its column by descending fractional part; the top `need[c]`
    # entries take the column's remaining units. `cols` is sorted, so a column occupies a
    # contiguous slice and its start offset is its exclusive prefix count.
    counts = np.bincount(cols, minlength=n_cols)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    order = np.lexsort((-frac, cols))            # within each column: largest frac first
    within = np.arange(cols.size) - starts[cols[order]]
    out[order[within < need[cols[order]]]] += 1
    return out


def solve_edge_budget(
    out_rows, out_cols, out_val, tgt_out,
    in_rows, in_cols, in_val, tgt_in,
    edge_budget: np.ndarray, n_entities: int, n_relations: int,
    iters: int = OUTER_ITERS,
) -> np.ndarray:
    """Find per-relation edge counts both sides can actually realise.

    The two sides must agree on the column margins, and which side binds differs by graph
    (on ``aids``/``fb237_v4`` the in side is the bottleneck; on ``wn18rr_v4`` the out
    side). So the shared column vector ``e`` is solved for by coordinate descent:

    1. run :func:`ipf` on each side against the current ``e`` — each ends on a row step,
       so each spends its entities' degree quota exactly and reports the column sums its
       support can actually deliver;
    2. take ``e ← min(achieved_out, achieved_in)`` — a relation can only carry as many
       edges as its *weaker* side supports;
    3. give the shortfall back to the relations with headroom, in proportion to that
       headroom.

    Because each row step conserves every entity's quota, the achieved columns always sum
    to ``Σ tgt_out = Σ tgt_in = content_E``. So the returned budget is fully spendable:
    **there is no deficit, by construction.** Where a relation's pool cannot absorb its
    target budget, its column comes out short and the surplus has already flowed to
    relations whose pools have room — which is the "shrink ``e_r``, redistribute" policy,
    obtained for free rather than as a separate mechanism.

    :param edge_budget: the target per-relation edge counts (from the relation-frequency
        fit) — the starting point, not a guarantee.
    :returns: int64 per-relation edge counts summing to ``Σ tgt_out``.
    """
    total = int(np.sum(tgt_out))
    e = np.asarray(edge_budget, dtype=float).copy()
    if n_relations == 0 or total <= 0:
        return np.zeros(n_relations, dtype=np.int64)

    for _ in range(iters):
        e_safe = np.maximum(e, _EPS)
        a_out = ipf(out_rows, out_cols, out_val, tgt_out, e_safe, n_entities, n_relations)
        a_in = ipf(in_rows, in_cols, in_val, tgt_in, e_safe, n_entities, n_relations)
        got_out = np.bincount(out_cols, a_out, minlength=n_relations)
        got_in = np.bincount(in_cols, a_in, minlength=n_relations)

        # A relation can carry only as many edges as its *weaker* side supports.
        new_e = np.minimum(got_out, got_in)
        short = total - new_e.sum()
        if short > _EPS:
            # Give the shortfall back only to relations that are NOT bottlenecked — ones
            # where both sides met the budget they were asked for. A bottlenecked relation
            # would simply hand the units straight back on the next sweep (that is what
            # made it the minimum), so topping it up cannot converge.
            free = (got_out >= e - 0.5) & (got_in >= e - 0.5)
            head = np.where(free, np.maximum(new_e, _EPS), 0.0)
            if head.sum() <= _EPS:            # everything is bottlenecked — spread evenly
                head = np.maximum(new_e, _EPS)
            new_e = new_e + short * head / head.sum()
        if np.allclose(new_e, e, rtol=1e-3, atol=0.5):
            e = new_e
            break
        e = new_e

    budget = _largest_remainder(np.maximum(e, 0.0), total)
    return budget
