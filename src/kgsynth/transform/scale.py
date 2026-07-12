"""Size rescaling — the designed-for next transform. **Not implemented.**

The shape of the problem, and why it is not a perturbation
---------------------------------------------------------
:class:`~kgsynth.transform.Perturb` holds graph size fixed and jitters shape. A
*scaling* transform does the opposite: it moves ``num_entities`` to a target ``V'``
and asks what the rest of the signature becomes.

That is not a matter of scaling one feature. The signature's features split in two
(``docs/notes/signature_size_dependence.md``, encoded as
:data:`kgsynth._domains.EXTENSIVE` / :data:`~kgsynth._domains.WEAKLY_EXTENSIVE`):

- **Intensive** — exponents, ratios, shapes, ``mean_degree``. Size-free by design;
  a scaling transform carries them across unchanged. This is most of the signature.
- **Extensive** — ``num_relations``, ``num_classes``, ``num_distinct_cs``,
  ``num_components``, and the seven raw motif counts. These scale with ``V`` or
  ``E``, and *how* they scale is exactly the open question.

Simply moving ``num_entities`` and leaving the extensive features at their measured
values asks the generator for a target no real graph satisfies — a graph with
``swdf``'s triangle count but a tenth of its vertices. That is why ``num_entities``
is flagged by :func:`kgsynth.transform._surface.validate` rather than silently
perturbed.

What implementing this needs
----------------------------
A **rescaling law per extensive feature**: ``f(V') = f(V) · g(V'/V)`` for some
``g``. Motif counts grow super-linearly in ``E`` (a triangle count roughly follows
a power of the edge count), while ``num_relations`` saturates — a bigger subset of
DBpedia does not keep inventing predicates. So ``g`` differs per feature and must
be *measured*, not assumed.

The data to fit it does not exist yet: it needs the same KG measured at several
sizes (nested subsets), which is the conditional-on-size model sketched in
``docs/plan/stage1_population_sampler.md`` and blocked on data acquisition.

This is also the deviation ``docs/signature.md`` records against the proposal's
``Generator.sample(num_triples=…)``: size is currently pinned by Block A, and
honouring an arbitrary size needs precisely this law.

Implementation sketch
---------------------
Once the law exists, ``ScaleTo`` is a :class:`~kgsynth.transform.SignatureTransform`
like any other, and composes with ``Perturb`` in a list::

    @dataclass(frozen=True)
    class ScaleTo:
        num_entities: int
        laws: dict[str, Callable[[float, float], float]]   # feature -> g(ratio)

        def apply(self, feats, rng):
            ratio = self.num_entities / feats["num_entities"]
            out = dict(feats)
            out["num_entities"] = float(self.num_entities)
            for name in EXTENSIVE - {"num_entities"}:
                out[name] = self.laws[name](feats[name], ratio)
            # intensive features carry across untouched
            return out, ClampReport()

The transform interface, the runner, the process pool and the dataset layout all
work unchanged — adding ``ScaleTo`` is a new class plus one entry in
:data:`kgsynth.transform.TRANSFORMS`.
"""

__all__: list[str] = []
