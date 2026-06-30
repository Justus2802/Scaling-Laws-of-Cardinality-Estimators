"""kgsynth generator — three-stage synthetic KG generation.

Consumes the **reduced** signature (``signature``): Stage 1 samples an
abstract :class:`Schema` from a measured BlockA/BlockC (optionally BlockB/BlockD),
Stage 2 instantiates it into an ``igraph.Graph``, and Stage 3 rewires that graph
toward the Block E / Block F targets.

The package is split by stage; this module re-exports the public API so existing
imports (``from generator import Schema, sample_schema, Signature, Generator``)
keep working.
"""

from .schema import Schema
from .stage1 import sample_schema
from .stage2 import instantiate
from .stage3 import refine
from .pipeline import Signature, Generator

__all__ = [
    "Schema",
    "sample_schema",
    "instantiate",
    "refine",
    "Signature",
    "Generator",
]
