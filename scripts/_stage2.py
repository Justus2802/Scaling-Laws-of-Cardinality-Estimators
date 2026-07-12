"""Shared helper for the diagnostic scripts that study the pre-refinement graph.

``profile_stage3_deltas``, ``edge_multiplicity`` and ``estimator_variance`` all need
the graph Stage 3 *starts from* for a named corpus graph. This wraps the cached-target
lookup plus :meth:`Generator.sample_pre_refine`, which derives the same sub-seeds as a
full ``Generator.sample(seed=…)`` — so the Stage-2 graph studied here is bit-for-bit
the one a ``signature_roundtrip.py`` run at that seed would refine.

Previously each script re-implemented this and imported it from its neighbour, which
only worked because ``python scripts/x.py`` happens to put ``scripts/`` on ``sys.path``.
"""

import igraph

from kgsynth.corpus import load_target_from_corpus
from kgsynth.generator import Generator


def build_stage2_graph(graph_name: str, seed: int) -> igraph.Graph:
    """Build the post-Stage-2, pre-refinement graph for a named corpus graph.

    Block E is not loaded: Stages 1–2 never read it, and it is the one block that is
    expensive to measure when uncached.

    :param graph_name: Corpus name of the target graph (e.g. ``fb237_v4``).
    :param seed: Master seed, derived exactly as ``Generator.sample(seed=…)`` does.
    :returns: The Stage-2 graph that ``refine()`` would be handed.
    """
    target, _blocks, _dir = load_target_from_corpus(graph_name, with_block_e=False)
    return Generator(target).sample_pre_refine(seed=seed)
