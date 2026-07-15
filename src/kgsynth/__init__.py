"""kgsynth — measure a real knowledge graph's statistical signature, generate synthetic KGs from it.

The public API mirrors the measure → generate → compare loop:

>>> from kgsynth import Signature, Generator
>>> target = Signature.from_file("some_real_graph.ttl")       # measure
>>> synthetic = Generator(target).sample(seed=42)             # generate

Subpackages
-----------
``kgsynth.signature``      Block A–F measurement (the "measure" step).
``kgsynth.generator``      Stage 1/2/3 synthetic-graph generation (the "generate" step).
``kgsynth.motif_counter``  Exact / colour-coding / hybrid subgraph-counting backends.

See ``user_docs/signature.md`` for the signature design and ``user_docs/generator.md`` for the
generator algorithm.
"""

from .generator import Generator, Schema, Signature, instantiate, refine, sample_schema
from .kg_io import load_kg, save_kg
from .signature import (
    QUANTILE_LEVELS,
    BlockA,
    BlockB,
    BlockC,
    BlockD,
    BlockE,
    BlockF,
    ReducedGraphSignature,
    compute_reduced_signature,
    write_signature_outputs,
)
from .transform import FeatureSpec, Identity, Perturb, PerturbOne, SignatureTransform

__version__ = "0.1.0"

__all__ = [
    # measure
    "BlockA",
    "BlockB",
    "BlockC",
    "BlockD",
    "BlockE",
    "BlockF",
    "ReducedGraphSignature",
    "compute_reduced_signature",
    "write_signature_outputs",
    "QUANTILE_LEVELS",
    # generate
    "Signature",
    "Generator",
    "Schema",
    "sample_schema",
    "instantiate",
    "refine",
    # transform (Signature feature-dict -> Signature feature-dict)
    "SignatureTransform",
    "Perturb",
    "PerturbOne",
    "Identity",
    "FeatureSpec",
    # io
    "load_kg",
    "save_kg",
    "__version__",
]
