# Approximate-hub-delta estimator experiments

Results behind `developer_docs/notes/stage3_steering_analysis.md` §4 — whether a
Horvitz–Thompson neighbour-subsampling estimator for the induced 5-/6-cycle delta
could replace dropping expensive hub swaps in Stage 3. Verdict: **no** (unbiased
but far too high variance on the hubs where it would be needed).

## Files

| File | Produced by | What it holds |
|---|---|---|
| `fb237_v4_estimator_variance_count_k{5,6}.csv` | `scripts/estimator_variance.py` | per-(proposal, K) relative std of the estimated cycle **count** vs endpoint degree |
| `fb237_v4_estimator_variance_count_k{5,6}.png` | `scripts/estimator_variance.py` | log-log rel-std-vs-degree scatter + power-law fit, one curve per sample count K |
| `fb237_v4_variance_fit_summary.txt` | `scripts/estimator_variance.py` (console) | fitted `rel_std = a·deg^b` params per K, both cycle sizes |
| `fb237_v4_hub_delta_validation.txt` | scratch `estimator_test.py` (console) | the 47-swap offline validation: count/delta bias+variance and the downstream net-loss sign-flip rate (sections A/B/C) |

## Headline results (fb237_v4, k=6)

- **Unbiased** (correctness check): count-estimator bias ~0 at K=32.
- **Variance scales ≈ `deg^0.9` with a ~`1/K` prefactor** (K ≥ 32), i.e.
  `rel_std ≈ C·deg/K`. Count relative std at K=32 grows 14 % → ~228 % across
  degrees 20 → 600; keeping it under ~20 % at deg 200 needs K ≳ 230 ≈ the whole
  neighbourhood (no speedup left).
- **Delta far worse** (difference of two large noisy counts): 45 % → 9 000–28 000 %
  relative std; frequent wrong sign.
- **Downstream:** feeding one estimated c5/c6 draw into the loss (tri+motif4 exact)
  flips the net-loss sign on 25–33 % of deg ≥ 50 hub proposals.

## Reproduce

```
python scripts/estimator_variance.py fb237_v4 --k 5 6 --samples 8 16 32 64 --per-bin 18 --bins 20 50 100 200 450
```

The `estimator_test.py` downstream sign-flip check is a one-off scratch script
(not committed); its console output is captured in `fb237_v4_hub_delta_validation.txt`.
