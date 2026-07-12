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

from ._logging import get_logger
from .generator import Signature
from .kg_io import load_kg
from .signature import BlockA, BlockB, BlockC, BlockD, BlockE, BlockF

log = get_logger(__name__)

# Repo root: src/kgsynth/corpus.py → parents[2]. Valid for an editable install,
# which is how the corpus-facing scripts are run. Exported as REPO_ROOT so the
# scripts anchor their data/ and experiments/ paths here instead of each
# re-deriving one from __file__.
_REPO = REPO_ROOT = Path(__file__).resolve().parents[2]

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


def graph_dir(name: str, search_dirs: list[Path] | None = None) -> Path | None:
    """Return the corpus directory for graph ``name`` (None when absent).

    :param name: Corpus name of the graph (the directory name).
    :param search_dirs: Directories to search, in order (default: :data:`DEFAULT_SEARCH_DIRS`).
    :returns: The graph's directory, or ``None`` when no corpus holds it.
    """
    for root in search_dirs or DEFAULT_SEARCH_DIRS:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def corpus_graph_names(search_dirs: list[Path] | None = None) -> list[str]:
    """Return every graph name across the corpora, sorted and de-duplicated.

    :param search_dirs: Directories to scan (default: :data:`DEFAULT_SEARCH_DIRS`).
    :returns: Sorted graph (directory) names.
    """
    names: set[str] = set()
    for root in search_dirs or DEFAULT_SEARCH_DIRS:
        if root.is_dir():
            names |= {p.name for p in root.iterdir() if p.is_dir()}
    return sorted(names)


def iter_corpus_graphs(
    names: set[str] | None = None,
    search_dirs: list[Path] | None = None,
) -> list[Path]:
    """Return one graph file per corpus directory, smallest first.

    Each ``<corpus>/<name>/`` directory holds a single source ``.nt``/``.ttl`` graph
    alongside its ``signature/`` output; synthetic (``*_synth``) outputs are ignored.
    Sorted by file size so quick wins land first.

    :param names: If given, only these graph (directory) names are returned.
    :param search_dirs: Directories to scan (default: :data:`DEFAULT_SEARCH_DIRS`).
    :returns: Graph file paths, smallest first.
    """
    graphs: list[Path] = []
    for root in search_dirs or DEFAULT_SEARCH_DIRS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or (names is not None and d.name not in names):
                continue
            found = find_graph_file(d)
            if found is not None:
                graphs.append(found)
    graphs.sort(key=lambda p: p.stat().st_size)
    return graphs


def load_target_from_corpus(
    graph_name: str,
    search_dirs: list[Path] | None = None,
    with_block_e: bool = True,
):
    """Load the cached reduced target signature for ``graph_name``.

    Searches each directory in ``search_dirs`` for ``<graph_name>/signature/``
    and loads blocks A/B/C/D/F from the first match. Block E is loaded from
    ``block_e.json`` if present, else measured from the graph file.

    :param graph_name: Corpus name of the graph (the directory name).
    :param search_dirs: Directories to search, in order (default: :data:`DEFAULT_SEARCH_DIRS`).
    :param with_block_e: Load Block E (default). Pass ``False`` for consumers that only
        drive Stages 1–2 — Block E is the expensive block to measure when uncached, and
        the schema sampler and instantiator never read it. Block E is then ``None``.
    :returns: ``(Signature, blocks_dict, graph_dir)``.
    :raises SystemExit: If the graph, a cached block, or a measurable Block E is missing.
    """
    search_dirs = search_dirs or DEFAULT_SEARCH_DIRS
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
        log.info("Loaded : %s", path.name)

    if not with_block_e:
        blocks["e"] = None
    elif (e_path := sig_dir / "block_e.json").exists():
        blocks["e"] = load_block(BlockE, e_path)
        log.info("Loaded : %s", e_path.name)
    else:
        graph_file = find_graph_file(graph_dir)
        if graph_file is None:
            raise SystemExit(
                f"block_e.json absent and no graph file in {graph_dir} to measure it from."
            )
        log.info("block_e.json absent — measuring Block E from %s …", graph_file.name)
        blocks["e"] = BlockE().calculate(load_kg(graph_file))

    sig = Signature(
        a=blocks["a"], b=blocks["b"], c=blocks["c"],
        d=blocks["d"], e=blocks["e"], f=blocks["f"],
    )
    return sig, blocks, graph_dir
