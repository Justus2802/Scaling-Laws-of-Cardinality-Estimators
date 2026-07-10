# Path-length steering — problem analysis and open options

> **Historical note.** The Stage-2 implementation described below
> (`_steer_path_lengths`, `PATH_STEERING_ENABLED`, and the `Schema.path_mean_target` /
> `path_hi_target` fields) has been **removed** from the code. It was one-sided — shortcut
> injection can only shorten paths — and had been disabled behind its flag. Block F's path
> statistics are now measured but not targeted; see
> [generator.md § Path-length steering](../generator.md#path-length-steering--removed).
> This document is retained for its root-cause analysis and for
> § "Option B — Stage 3 SA loss term", which remains the way forward if path-length targeting
> is ever revisited. Sections "Current implementation" and "File map" describe code that no
> longer exists.

Block F stores three path-length statistics: `shortest_path_max` (diameter),
`shortest_path_mean`, and `shortest_path_var` (variance).  The generator
consumed `max` as the diameter cap target and `mean` as the mean path-length
target; `var` is validation-only.

These were the hardest Block F targets to hit.

---

## Root cause: structural undershoot

The synthetic graph almost always has **shorter paths** than the real KG, even when
density (`E/V`) matches. Observed roundtrip example:

```
path steering — diameter 7→7 (target 9), mean 2.73→2.77 (target 4.07)
```

The structural reasons:

1. **Hub formation.** Degree steering (historically `in_pa_exponent` preferential
   attachment, since replaced by per-entity target degree sequences) gives high-degree nodes a
   disproportionate share of incoming edges. Hubs act as relay points: any two nodes can reach
   each other in 2–3 hops via a hub. This compresses paths regardless of global density.

2. **Large CS sizes.** Many entities having large characteristic sets means many entities
   share many relations → dense cross-connections between "neighbourhoods" → short paths.

3. **`_connect_components` bridge edges.** Every bridged isolated component is connected to
   the giant with a direct edge, flattening the remaining longest paths.

These three forces combine to push mean path length well below what real KGs show, even
when global `E` and `V` are matched.

---

## What is controllable vs emergent

| Feature | Status | Notes |
|---|---|---|
| `shortest_path_max` (diameter) | Partially controllable | Can be capped via shortcuts; cannot be increased post-hoc without removing edges |
| `shortest_path_mean` | Partially controllable | Can be compressed via shortcuts; difficult to increase post-hoc |
| `shortest_path_var` | Emergent | Determined by density + degree structure; not directly steered |

---

## Current implementation

`_steer_path_lengths` in `src/generator/stage2.py`, called after `_connect_components` (step 7b).

**Mechanism:**
- Builds a temporary undirected igraph entity graph from `content_edges`.
- Up to 4 rounds of: estimate current diameter + mean → inject shortcuts.
- **Diameter (hi):** BFS from a random source to its farthest node; adds
  `⌈(diam − hi_target)/2⌉` such shortcuts per round so convergence is fast.
- **Mean:** hub-to-hub shortcuts sampled ∝ degree, count
  `max(1, round(√V · (mean − mean_target) / mean))` per round.
- Shortcuts are added to the igraph object in-place so subsequent rounds see the
  updated graph without rebuilding.

**Fundamental limitation — one-sided only:**
Shortcuts can only *reduce* mean and diameter. They cannot increase them. When the
synthetic graph undershoots the target (the common case), this pass is a no-op.

In the roundtrip above, both targets are already below the measured values:
- `hi_ok = (7 ≤ 9)` → True, no hi shortcuts added
- `mean_ok = (2.73 ≤ 4.07 + 0.5)` → True, no mean shortcuts added

The 2.73→2.77 drift is BFS sampling noise.

---

## Options evaluated

### Option A — Accept loc/scale/shape as emergent, target only hi (current partial fix)

**What it does:** Shortcut injection for hi only; mean/loc/scale/shape are validated but
not steered.

**Verdict:** Correct for the hi cap (useful when synthetic diameter > target). Useless when
the synthetic graph undershoots (common case).

---

### Option B — Stage 3 SA loss term (designed, not implemented)

Add `current_mean_path` to the `_loss()` function in `refine()`, with periodic BFS
estimation on the already-maintained `adj` dict.

**BFS helper (no igraph rebuild needed):**
```python
def _sample_mean_path(k=50) -> float:
    srcs = rng.choice(n, size=min(k, n), replace=False)
    tot = cnt = 0
    for src in srcs:
        dist = {int(src): 0}; queue = [int(src)]; head = 0
        while head < len(queue):
            v = queue[head]; head += 1
            for u in adj[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1; queue.append(u)
        for d in dist.values():
            if d > 0: tot += d; cnt += 1
    return tot / cnt if cnt > 0 else float("nan")
```

Update at the same `remeasure_interval` as 4-node motifs. Add to loss:
```python
if use_path_mean and not math.isnan(current_mean_path):
    loss += abs(current_mean_path - path_mean_target) / path_mean_target
```

**What Stage 3 can do:** Degree-preserving swaps steer mean path length by breaking or
creating hub shortcuts. To increase mean (the common need), the SA accepts swaps that
move edges from hub-to-hub positions to peripheral-peripheral positions.

**Ceiling on improvement:** The degree sequence is fixed after Stage 2. Mean path length
is approximately determined by the degree sequence (Chung–Lu: `L ≈ log V / log k̄`).
Stage 3 can move the mean by roughly 0.3–0.5 units in favourable cases; it cannot close a
1.3-unit gap (2.77 → 4.07) without upstream changes.

**Implementation cost:** ~30 lines added to `refine()` in `stage3.py`. `_steer_path_lengths`
in Stage 2 can be removed (or kept as a cheap pre-pass for the hi cap only).

---

### Option C — Stage 2 swap bias (not implemented)

During CS assignment or edge wiring, bias toward configurations that produce longer paths:
- Smaller CS sizes → fewer relation memberships → sparser cross-connections
- Lower PA exponent → flatter degree distribution → fewer strong hubs

This is not a post-hoc fix; it requires changing the wiring parameters themselves and
conflicts with Block B/D targets.

---

### Option D — Stage 2 edge removal (ruled out)

Remove content edges post-hoc to lengthen paths. Ruled out because:
- Content edges encode the degree sequence and CS structure (Block B/D targets)
- Removing an edge changes two node degrees
- Can disconnect the graph

The only safely removable edges are bridge edges from `_connect_components` and shortcuts
from `_steer_path_lengths` itself. Removing those could marginally increase mean/diameter
but has small practical effect.

---

## Recommended next steps

**Short path to improvement (Stage 3 integration, Option B):**
1. Remove `_steer_path_lengths` from Stage 2 (or keep only the hi-cap logic).
2. Add `_sample_mean_path()` closure to `refine()` using the existing `adj` dict.
3. Add `use_path_mean` flag + loss term to `_loss()`.
4. Update at `remeasure_interval` alongside 4-node motifs.
5. Pass `path_mean_target` from `Schema` into `refine()` (already on Schema; need to wire
   through `Generator.sample()` → `refine()` call).

Expected outcome: Stage 3 can close maybe 30–40% of the observed gap.

**Structural fix (Stage 1/2, harder):**
The remaining gap requires one or more of:
- Reducing `cs_size_mean` (smaller CS → sparser connections)
- Flattening the target degree sequences (fewer strong hubs; was `in_pa_exponent` before degree-sequence targeting)
- A feedback loop that adjusts these parameters based on the observed path overshoot

Neither has a direct Block F input signal; they would need to be inferred from the path
distribution itself (inverse problem).

---

## File map

| File | Relevant code |
|---|---|
| `src/generator/schema.py` | `path_mean_target`, `path_hi_target` fields |
| `src/generator/stage1.py` | Reads `f.shortest_path_mean` and `f.shortest_path_max` directly |
| `src/generator/stage2.py` | `_steer_path_lengths` (current implementation; one-sided) |
| `src/generator/stage3.py` | `refine()` — target site for Option B |
| `src/signature/block_f.py` | `shortest_path_max`, `shortest_path_mean`, `shortest_path_var` properties |
