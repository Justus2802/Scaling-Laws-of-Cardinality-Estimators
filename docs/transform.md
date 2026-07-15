# Signature transforms

A **transform** maps a signature's flat feature dict to another one:

```python
from kgsynth import Signature, Generator, Perturb, FeatureSpec
import numpy as np

feats = Signature.from_file("swdf.nt").as_features()          # 134 keys
feats, report = Perturb({"mean_degree": FeatureSpec(sigma=0.15)}).apply(
    feats, np.random.default_rng(0)
)
graph = Generator(Signature.from_features(feats)).sample(seed=1)
```

Transforms work on the **public feature dict** (`as_features()` / `from_features()`, see
[generator.md](generator.md)), never on block internals, so they do not depend on any block's private
attribute names.

| Transform | What it does |
|---|---|
| `Perturb` | Jitter every configured feature at once (*joint*) — a diverse cloud of signatures around the baseline. For building a synthetic-KG corpus. |
| `PerturbOne` | Move exactly one feature by a fixed level (*OFAT*) — everything else stays at baseline, so any difference in the generated graph is attributable to that feature. A sensitivity analysis. |
| `Identity` | The null transform — the unperturbed baseline graph an OFAT sweep compares against. |
| `ScaleTo` | Resize the graph. **Not implemented** — see [`transform/scale.py`](../src/kgsynth/transform/scale.py) for what it needs and why it is blocked. |

---

## The perturbation surface — 79 of 127 features

Only **87** of the signature's 134 features are read by the generator. Perturbing any of the other 47
is a silent no-op: the graph comes out identical. `kgsynth.transform.validate()` therefore **raises**
on an off-surface feature rather than accepting it — a duplicate graph produced by a knob that does
nothing is the worst outcome for a sensitivity study.

| Block | Reached / total | Not reached |
|---|---|---|
| A | 4 / 4 | — |
| B | 32 / 35 | the three `*_xmin` — only `.alpha` / `.exponent` is read off those fits |
| C | 10 / 29 | `class_size_xmin`; both `cooc_density`; the 14 `row_entropy_q*`; `per_type_entropy_rate/scale` |
| D | 22 / 25 | `two_step_alpha/vmin/vmax` |
| E | 7 / 27 | the 20 `path_template_*` / `tree_template_*` — their Stage-3 steering is gated off |
| F | 4 / 7 | `shortest_path_max/mean/var` — unsteered |
| | **87 / 134** | **47** |

These counts are asserted by `test_transform.py::TestSurface`, so a block gaining or losing a feature
fails there rather than silently changing what a sweep covers.

Two subsets are on the surface but perturb degenerately. `validate()` accepts them **with a warning**:

**`INERT` — read, but cannot change the output.** `subj_cooc_scale`, `obj_cooc_scale`,
`type_rel_spectrum_scale`. `_reconstruct_singular_values` returns `scale · exp(−rate · r)` and both
consumers normalise immediately (`svs / svs.sum()`), so a constant factor cancels exactly. Their only
surviving role is the `isnan(scale)` presence check. Pinned by
`test_transform.py::TestInertFeatures`: scaling any of them 100× leaves the `Schema` identical.

Note the consequence for Block C: with `scale` inert, the whole synthesised `T × R` type-relation
conditional is steered through the **single scalar** `type_rel_spectrum_rate`.

**`CONSTANT` — pinned by the fitter, not measured.** `obj/subj_mult_alpha_q00` and `_q100`.
`fit_quantiles(..., lo=1.4, hi=3.0)` *pins* the min/max levels to those bounds, so they are the same
value on every graph and carry no information about the graph they came from. They still act as the
generator's sampling cutoffs, so perturbing them does something — it moves a design constant.

---

## Coupled groups

Six feature groups must move **together**, and the transforms enforce it: naming any member perturbs
the whole group by **one shared factor**.

The four quantile functions (`obj_mult_alpha_q*`, `subj_mult_alpha_q*`, `cs_size_q*`,
`inv_cs_size_q*`) back an *invertible CDF for inverse-transform sampling*. Jittering their seven
levels independently can invert them and corrupt the sampler. A shared factor preserves the ordering
by construction; the result is re-sorted defensively afterwards, since clamping can still flatten a
group whose entries straddle a bound. `recip_symmetric_frac_bin0..5` is a 6-bin probability vector and
gets the same treatment.

This is why an OFAT sweep enumerates **knobs, not features**: the seven `obj_mult_alpha_q*` keys are
one knob.

---

## Clamping, and why it is reported

Every feature has a domain (`kgsynth._domains`): `degree_assortativity ∈ [-1, 1]`,
`edge_multiplicity ≥ 1`, `bidirectional_ratio ∈ [1, 2]`, the multiplicity exponents ∈ `[1.4, 3.0]`,
counts integral. A perturbation is clamped back into range.

**A clamped perturbation is a no-op wearing a costume.** The feature reads as "perturbed" in the
config, but the generator sees the same value — and an OFAT sweep reports "this knob has no effect"
when in fact it never moved. So `apply()` returns a `ClampReport` alongside the features, recording
which values were clamped and what fraction of each coupled group was absorbed.

This is not hypothetical. `swdf`'s `obj_mult_alpha_q*` already sits at the `[1.4, 3.0]` window's
ceiling for its top three levels:

```
baseline   [1.40  2.08  2.22  2.61  3.00  3.00  3.00]
× 1.2  ->  [1.68  2.49  2.66  3.00  3.00  3.00  3.00]   57% absorbed -> SATURATED
× 0.8  ->  [1.40  1.66  1.77  2.09  2.40  2.40  2.40]   14% absorbed
```

Scaling that group **up** is mostly swallowed by the clamp; scaling it **down** is not. The report
flags the first case, so the asymmetry is visible instead of silently biasing the sweep.

---

## NaN is data, not a gap

An all-NaN fit is a real measurement outcome — "this fit did not converge on this graph" — not a
missing value. `aids` has only 5 relations, too few for the per-relation α-quantile fit, so its
`obj_mult_alpha_q*` is all-NaN. Transforms **leave NaN features untouched** rather than perturbing
them into a number, and `from_features` reproduces them as NaN so the generator's own NaN fallbacks
fire.

---

## Adding a transform

Implement two methods — no base class, `SignatureTransform` is a `Protocol`:

```python
@dataclass(frozen=True)          # frozen: it crosses a process boundary
class MyTransform:
    def apply(self, feats, rng) -> tuple[dict[str, float], ClampReport]: ...
    def describe(self) -> dict: ...          # goes into the dataset's meta.json
```

Then add it to `kgsynth.transform.TRANSFORMS`. It must not mutate the dict it is given — workers share
a baseline. `ScaleTo` is the worked example of what the next one looks like.
