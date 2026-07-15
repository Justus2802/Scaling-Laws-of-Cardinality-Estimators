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
    rt = np.asarray(row_target, dtype=float)
    ct = np.asarray(col_target, dtype=float)

    # Rescale the values themselves rather than accumulating the row/column multipliers
    # ``u``/``w``. The two are algebraically identical (``a = W·u·w``), but the multiplier
    # form **overflows**: a column whose target far exceeds what its support can supply
    # drives ``w`` up without bound, and once one entry hits ``inf`` the next product is
    # ``nan`` and the whole allocation is silently destroyed (swdf did exactly this and came
    # out with a 170-edge budget). Scaling ``a`` in place cannot overflow — every row sweep
    # renormalises it back to ``rt``, so it stays bounded by the margins throughout.
    a = np.maximum(np.asarray(val, dtype=float), _EPS)
    for _ in range(iters):
        a *= (rt / np.maximum(np.bincount(rows, a, minlength=n_rows), _EPS))[rows]
        a *= (ct / np.maximum(np.bincount(cols, a, minlength=n_cols), _EPS))[cols]

    # Final row step: row margins exact on return (see the docstring — this asymmetry is
    # what lets solve_edge_budget read the achievable column sums off the result).
    a *= (rt / np.maximum(np.bincount(rows, a, minlength=n_rows), _EPS))[rows]
    return a


def _fill_to_total(
    e: np.ndarray, total: int, cap: np.ndarray, live: np.ndarray
) -> np.ndarray:
    """Push ``e`` up (or down) to sum to ``total``, never crossing ``cap``.

    The per-relation ceilings mean a single proportional rescale is not enough: scaling up
    pins some relations at their cap, which leaves the total short again. So the surplus is
    poured into the remaining headroom, repeatedly, until it is all placed or every relation
    is full.

    This is what keeps the returned budget *spendable*. ``Σ e`` must equal ``Σ tgt_out``, or
    the graph either cannot place all its stubs (a deficit) or is asked to place stubs that
    do not exist (an overshoot).
    """
    out = np.clip(np.asarray(e, dtype=float), 0.0, cap)
    out[~live] = 0.0
    for _ in range(16):
        short = total - out.sum()
        if abs(short) < 0.5:
            break
        if short > 0:
            room = np.where(live, np.maximum(cap - out, 0.0), 0.0)
            if room.sum() <= _EPS:
                break                       # every relation is at its ceiling
            out = np.minimum(out + short * room / room.sum(), cap)
        else:
            scale = total / max(out.sum(), _EPS)
            out = out * scale
    return out


def _largest_remainder(values: np.ndarray, total: int) -> np.ndarray:
    """Round ``values`` to non-negative integers summing exactly to ``total``.

    This *rounds*; it does not *scale*. It can move each entry by at most one, so it only
    reaches ``total`` when ``values`` already sums to within ``len(values)`` of it — call
    :func:`_fill_to_total` first. (Handing it a badly-scaled vector silently returns a sum
    of ``len(values)``: an early version did exactly that and produced a 170-edge budget
    for swdf's 170 relations, against a target of 242 256.)
    """
    if values.size == 0:
        return np.array([], dtype=np.int64)
    floors = np.floor(values).astype(np.int64)
    short = int(total) - int(floors.sum())
    if abs(short) > values.size:
        log.warning("IPF: budget vector is off by %d over %d relations — rounding cannot "
                    "reach the total", short, values.size)
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


def _clip_to_entry_caps(
    cols: np.ndarray, x: np.ndarray, entry_cap: np.ndarray, n_cols: int,
    rng: np.random.Generator, passes: int = 8,
) -> int:
    """Clip each entry at ``entry_cap``, moving the excess **within its own column**.

    An object can be reached by at most ``|S_r|`` *distinct* subjects, and a subject can
    reach at most ``|O_r|`` distinct objects, because a relation may not carry the same
    ``(s, o)`` pair twice. An allocation that hands one entity more stubs of ``r`` than that
    is not merely hard to place — it is **unrealisable**, and no pairing algorithm will ever
    satisfy it. (fb237_v4's in-hub was allocated 119 in-stubs of relation 178 from a pool of
    117 subjects.)

    The excess is redistributed to entries in the same column that still have headroom, in
    proportion to that headroom, so ``Σ_v X[v, r]`` is untouched and the two sides' stub
    counts stay equal. ``solve_edge_budget`` bounds ``e_r`` by ``|S_r|·|O_r|``, which is
    exactly the column's total capacity here, so the redistribution always has somewhere to
    go.

    :returns: the number of stubs moved.
    """
    if cols.size == 0:
        return 0
    counts = np.bincount(cols, minlength=n_cols)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    moved = 0
    for _ in range(passes):
        over = np.maximum(x - entry_cap, 0)
        if not over.any():
            break
        x -= over
        moved += int(over.sum())
        per_col = np.bincount(cols, over, minlength=n_cols).astype(np.int64)
        for c in np.where(per_col > 0)[0]:
            lo, hi = int(starts[c]), int(starts[c] + counts[c])
            xs, cp = x[lo:hi], entry_cap[lo:hi]
            head = np.maximum(cp - xs, 0)
            total = int(head.sum())
            if total <= 0:
                continue                       # column is saturated; caller's cap was wrong
            take = min(int(per_col[c]), total)
            add = np.minimum(rng.multinomial(take, head / total), head)
            xs += add
    return moved


def fit_stubs(
    rows: np.ndarray, cols: np.ndarray, val: np.ndarray,
    row_target: np.ndarray, col_target: np.ndarray,
    n_rows: int, n_cols: int, rng: np.random.Generator,
    floor: bool = False,
    entry_cap: "np.ndarray | None" = None,
) -> np.ndarray:
    """Fit integer stub counts to both margins, optionally with a ≥1-per-entry floor.

    With ``floor``, every support entry is guaranteed at least one stub. ``r ∈ CS(v)``
    means "v emits r", so v should emit at least one r-edge — otherwise the realised
    characteristic sets drift from the assigned ones and Block D's CS statistics stop
    describing the graph.

    The floor is imposed by **substitution**, not by repair: fit ``X'`` to the reduced
    margins ``(tgt_out[v] − |elig(v)|, e_r − |S_r|)`` and return ``X = 1 + X'``. Both
    margins then come out exact, because the constant 1 contributes ``|elig(v)|`` to each
    row and ``|S_r|`` to each column, which is precisely what was subtracted.

    Doing it the other way round — fit first, then hand a stub to each starved entry by
    taking one from a heavy entry in the same column — is what a first version did, and it
    *decapitates the hubs*: the entries with the most to give are the ones carrying the
    ``max``/``p90`` degree targets, so the trim lands squarely on them (it cost fb237_v4's
    max out-degree 195 → 166 and aids' 11 → 5). The floor has to be in the margins.

    A column with fewer edges than eligible subjects (``e_r < |S_r|``) cannot give every
    subject an edge; the floor is simply dropped for that column and its entries fall back
    to the plain fit.

    :param floor: guarantee ≥1 stub per support entry where the column can afford it.
    :param entry_cap: per-entry ceiling — the opposite pool's size, i.e. the number of
        *distinct* partners this entity can have under relation ``r``. Without it the fit
        can allocate an entity more stubs of a relation than there are partners to spend
        them on, which no pairing can realise. See :func:`_clip_to_entry_caps`.
    :returns: int64 stub count per non-zero, with column sums exactly ``col_target``.
    """
    if rows.size == 0:
        return np.array([], dtype=np.int64)
    ct = np.asarray(col_target, dtype=np.int64)
    rt = np.asarray(row_target, dtype=np.int64)

    def _capped(x: np.ndarray) -> np.ndarray:
        if entry_cap is not None:
            _clip_to_entry_caps(cols, x, np.asarray(entry_cap, dtype=np.int64),
                                n_cols, rng)
        return x

    if not floor:
        a = ipf(rows, cols, val, rt, np.maximum(ct, _EPS), n_rows, n_cols)
        return _capped(round_to_columns(cols, a, ct, n_cols))

    per_col = np.bincount(cols, minlength=n_cols)
    affordable = ct >= per_col                       # column can seat one stub per entry
    base = affordable[cols].astype(np.int64)         # the "1" of X = 1 + X'
    rt2 = rt - np.bincount(rows, base, minlength=n_rows).astype(np.int64)
    ct2 = ct - np.bincount(cols, base, minlength=n_cols).astype(np.int64)

    # tgt_out[v] ≥ |CS(v)| ≥ |elig(v)| normally (that is what floor=cs_sizes_all buys in
    # §3c), so rt2 is already non-negative. It can dip below zero only when that floor had
    # to be relaxed because the CS sizes over-determined the edge budget; clip, then put
    # the clipped units back so the two margins still agree — they must, or IPF has no
    # solution at all.
    if (rt2 < 0).any():
        rt2 = np.maximum(rt2, 0)
    gap = int(ct2.sum()) - int(rt2.sum())
    if gap > 0:
        has = np.where(np.bincount(rows, minlength=n_rows) > 0)[0]
        rt2[has] += rng.multinomial(gap, np.full(has.size, 1.0 / has.size))
    elif gap < 0:
        rt2 = _largest_remainder(rt2.astype(float), int(ct2.sum()))

    a = ipf(rows, cols, val, rt2, np.maximum(ct2, _EPS), n_rows, n_cols)
    return _capped(round_to_columns(cols, a, ct2, n_cols) + base)


def solve_edge_budget(
    out_rows, out_cols, out_val, tgt_out,
    in_rows, in_cols, in_val, tgt_in,
    edge_budget: np.ndarray, n_entities: int, n_relations: int,
    col_cap: "np.ndarray | None" = None,
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
    :param col_cap: hard per-relation ceiling, e.g. ``|S_r|·|O_r|`` (an edge needs a
        distinct subject/object pair, so a relation cannot carry more than that many).
    :returns: int64 per-relation edge counts summing to ``Σ tgt_out``.
    """
    total = int(np.sum(tgt_out))
    if n_relations == 0 or total <= 0:
        return np.zeros(n_relations, dtype=np.int64)

    # A relation with no eligible subject *or* no eligible object cannot carry an edge at
    # all. It must be held at exactly zero throughout: hand it even one unit of budget and
    # that unit is unplaceable — a deficit reintroduced by the back door.
    live = (
        (np.bincount(out_cols, minlength=n_relations) > 0)
        & (np.bincount(in_cols, minlength=n_relations) > 0)
    )
    # Cap defaults to the whole budget (a relation cannot carry more edges than exist) and
    # is clipped to it. A finite ceiling is required, not cosmetic: _fill_to_total pours the
    # shortfall in proportion to the remaining headroom, and an infinite headroom makes that
    # ratio inf/inf = nan.
    cap = (np.full(n_relations, float(total)) if col_cap is None
           else np.minimum(np.asarray(col_cap, dtype=float), float(total)))
    cap = np.where(live, cap, 0.0)
    if cap.sum() < total:
        log.warning(
            "IPF: relation capacity %d < edge budget %d — the graph cannot hold its "
            "edge count within the CS pools", int(cap.sum()), total,
        )

    e = np.minimum(np.where(live, np.asarray(edge_budget, dtype=float), 0.0), cap)

    for _ in range(iters):
        e_safe = np.maximum(e, _EPS)
        a_out = ipf(out_rows, out_cols, out_val, tgt_out, e_safe, n_entities, n_relations)
        a_in = ipf(in_rows, in_cols, in_val, tgt_in, e_safe, n_entities, n_relations)
        got_out = np.bincount(out_cols, a_out, minlength=n_relations)
        got_in = np.bincount(in_cols, a_in, minlength=n_relations)

        # A relation can carry only as many edges as its *weaker* side supports; whatever
        # that leaves unspent is pushed onto the relations that still have room.
        new_e = _fill_to_total(np.minimum(np.minimum(got_out, got_in), cap),
                              total, cap, live)
        if np.allclose(new_e, e, rtol=1e-3, atol=0.5):
            e = new_e
            break
        e = new_e

    budget = _largest_remainder(e, total)
    # Rounding to the total can nudge a relation past its ceiling; move any such unit to a
    # relation that still has room, so the |S_r|·|O_r| bound is never violated.
    over = np.maximum(budget - np.floor(cap).astype(np.int64), 0)
    spill = int(over.sum())
    if spill > 0:
        budget -= over
        room = (np.floor(cap).astype(np.int64) - budget) * live
        for _ in range(spill):
            i = int(np.argmax(room))
            if room[i] <= 0:
                log.warning("IPF: %d edges cannot be placed within the relation caps", spill)
                break
            budget[i] += 1
            room[i] -= 1
    return budget
