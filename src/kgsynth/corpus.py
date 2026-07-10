"""Locating and loading cached signatures from the measured-KG corpus.

The corpus lives at ``data/graphs/<name>/signature/`` (population graphs) and
``data/test_graphs/<name>/signature/`` (smaller graphs held out of the population
fit), each holding the per-block ``block_*.json`` written by ``kgsynth measure``.

These helpers were previously private to ``scripts/signature_roundtrip.py`` and
imported across scripts via a ``sys.path`` hack; they live here so every consumer
(the CLI, the sweep scripts, the PCA plots) shares one implementation.
"""

import json
from pathlib import Path

from .generator import Signature
from .kg_io import load_kg
from .signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF

# Repo root: src/kgsynth/corpus.py → parents[2]. Valid for an editable install,
# which is how the corpus-facing scripts are run.
_REPO = Path(__file__).resolve().parents[2]

# Reduced block letter → class, in signature order. Block E is loaded separately:
# it may be absent from the corpus and measured on demand.
_BLOCK_CLASSES = {"a": BlockA, "b": BlockB, "c": BlockC, "d": BlockD, "f": BlockF}

# Directories searched (in order) when no explicit graphs dir is given.
DEFAULT_SEARCH_DIRS: list[Path] = [
    _REPO / "data" / "graphs",
    _REPO / "data" / "test_graphs",
]


def load_block(cls, path: Path):
    """Reconstruct a reduced block from its serialized ``block_*.json``.

    :param cls: The block class to reconstruct (e.g. :class:`BlockA`).
    :param path: Path to the block's serialized JSON.
    :returns: A populated block instance.
    """
    return cls.from_serializable(json.loads(path.read_text()))


def find_graph_file(d: Path) -> Path | None:
    """Return the first non-synthetic .nt/.ttl graph file in directory ``d`` (None if absent).

    :param d: Directory to search.
    :returns: The graph file, or ``None`` when the directory holds none.
    """
    for pattern in ("*.nt", "*.ttl", "*.nt.gz", "*.ttl.gz"):
        hits = sorted(p for p in d.glob(pattern) if not p.stem.endswith("_synth"))
        if hits:
            return hits[0]
    return None


def load_target_from_corpus(graph_name: str, search_dirs: list[Path]):
    """Load the cached reduced target signature for ``graph_name``.

    Searches each directory in ``search_dirs`` for ``<graph_name>/signature/``
    and loads blocks A/B/C/D/F from the first match. Block E is loaded from
    ``block_e.json`` if present, else measured from the graph file.

    :param graph_name: Corpus name of the graph (the directory name).
    :param search_dirs: Directories to search, in order.
    :returns: ``(Signature, blocks_dict, graph_dir)``.
    :raises SystemExit: If the graph, a cached block, or a measurable Block E is missing.
    """
    graph_dir = sig_dir = None
    for graphs_dir in search_dirs:
        candidate = graphs_dir / graph_name
        if (candidate / "signature").is_dir():
            graph_dir = candidate
            sig_dir = candidate / "signature"
            break

    if sig_dir is None:
        available: list[str] = []
        for d in search_dirs:
            if d.is_dir():
                available += sorted(p.name for p in d.iterdir() if p.is_dir())
        raise SystemExit(
            f"'{graph_name}' not found in {[str(d) for d in search_dirs]}. "
            f"Available graphs: {sorted(set(available))}"
        )

    blocks: dict[str, object] = {}
    for letter, cls in _BLOCK_CLASSES.items():
        path = sig_dir / f"block_{letter}.json"
        if not path.exists():
            raise SystemExit(f"Missing cached block: {path}")
        blocks[letter] = load_block(cls, path)
        print(f"  Loaded : {path.name}")

    e_path = sig_dir / "block_e.json"
    if e_path.exists():
        blocks["e"] = load_block(BlockE, e_path)
        print(f"  Loaded : {e_path.name}")
    else:
        graph_file = find_graph_file(graph_dir)
        if graph_file is None:
            raise SystemExit(
                f"block_e.json absent and no graph file in {graph_dir} to measure it from."
            )
        print(f"  block_e.json absent — measuring Block E from {graph_file.name} …")
        blocks["e"] = BlockE().calculate(load_kg(graph_file))

    sig = Signature(
        a=blocks["a"], b=blocks["b"], c=blocks["c"],
        d=blocks["d"], e=blocks["e"], f=blocks["f"],
    )
    return sig, blocks, graph_dir
