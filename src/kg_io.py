"""Knowledge Graph I/O: load/save Turtle and N-Triples files as igraph objects.

Loading detects the serialization from file content (not the extension) so
extensionless dumps work; saving takes the format as an explicit argument.
"""

from pathlib import Path

import igraph
import rdflib
from rdflib import BNode, Literal, URIRef

# Maps caller-facing format names (and common aliases) to rdflib parser names.
_FORMAT_ALIASES = {
    "turtle": "turtle",
    "ttl": "turtle",
    "nt": "nt",
    "ntriples": "nt",
    "n-triples": "nt",
}


def _rdf_node_id(node: rdflib.term.Identifier) -> str:
    """Return a stable string key for an RDF node."""
    if isinstance(node, BNode):
        return f"_:{node}"
    return str(node)


def _looks_like_ntriples(text: str) -> bool:
    """Return True if every content line is a one-line N-Triples statement.

    N-Triples is a strict line-based subset of Turtle: each statement sits on a
    single line, starts with an absolute IRI (`<...>`) or blank node (`_:...`)
    and is terminated by `.`. Any other shape (prefix directives, prefixed
    names, `;`/`,` predicate lists) means the document needs the Turtle parser.
    Requires at least one statement so empty/comment-only input is not N-Triples.
    """
    saw_statement = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not (line.startswith(("<", "_:")) and line.endswith(".")):
            return False
        saw_statement = True
    return saw_statement


def _detect_rdf_format(text: str) -> str:
    """Guess the RDF serialization ('nt' or 'turtle') from file content.

    Because N-Triples is a subset of Turtle, the Turtle parser also accepts
    N-Triples; we therefore default to Turtle and only return 'nt' for input
    that is unambiguously line-shaped N-Triples. Invalid RDF is left for the
    parser to reject.
    """
    return "nt" if _looks_like_ntriples(text) else "turtle"


def load_kg(path: str | Path) -> igraph.Graph:
    """Parse a Turtle or N-Triples file and return a directed igraph Graph.

    The serialization is detected from the file *content*, not its extension,
    so extensionless files (common for raw LOD dumps) load directly. Input that
    is not valid Turtle/N-Triples raises a ValueError via the rdflib parser.

    Edges are deduplicated by (subject, predicate, object): the resulting
    graph contains at most one edge per distinct RDF triple. rdflib's `Graph`
    already enforces this at parse time, but the loader also guards against
    duplicates explicitly so the contract holds for any input source.

    Vertices carry:
        name         – URI string (or blank-node id like "_:b0")
        is_literal   – True for RDF literal objects
        literal_value, literal_datatype, literal_lang  – set when is_literal

    Edges carry:
        predicate    – URI string of the RDF predicate
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    fmt = _detect_rdf_format(text)

    rdf_graph = rdflib.Graph()
    try:
        rdf_graph.parse(data=text, format=fmt)
    except Exception as exc:
        raise ValueError(f"'{path}' is not valid Turtle or N-Triples content") from exc

    # Collect unique nodes (preserve insertion order for stable vertex indices)
    node_index: dict[str, int] = {}
    vertex_attrs: list[dict] = []

    def _ensure_vertex(node: rdflib.term.Identifier) -> int:
        key = _rdf_node_id(node)
        if key not in node_index:
            idx = len(node_index)
            node_index[key] = idx
            attrs: dict = {
                "name": key,
                "is_literal": False,
                "literal_value": None,
                "literal_datatype": None,
                "literal_lang": None,
            }
            if isinstance(node, Literal):
                attrs["is_literal"] = True
                attrs["literal_value"] = str(node)
                attrs["literal_datatype"] = str(node.datatype) if node.datatype else None
                attrs["literal_lang"] = node.language
            vertex_attrs.append(attrs)
        return node_index[key]

    seen_triples: set[tuple[int, int, str]] = set()
    edges: list[tuple[int, int, str]] = []
    for s, p, o in rdf_graph:
        si = _ensure_vertex(s)
        oi = _ensure_vertex(o)
        triple = (si, oi, str(p))
        if triple in seen_triples:
            continue
        seen_triples.add(triple)
        edges.append(triple)

    g = igraph.Graph(directed=True)
    g.add_vertices(len(node_index))
    for attr in ("name", "is_literal", "literal_value", "literal_datatype", "literal_lang"):
        g.vs[attr] = [v[attr] for v in vertex_attrs]

    if edges:
        g.add_edges([(s, o) for s, o, _ in edges])
        g.es["predicate"] = [p for _, _, p in edges]

    return g


def save_kg(graph: igraph.Graph, path: str | Path, fmt: str = "turtle") -> None:
    """Serialize an igraph Graph (produced by load_kg) to a file.

    Because the output format cannot be inferred from content that does not yet
    exist, it is given explicitly via ``fmt`` rather than the path's extension.

    Args:
        graph: Graph with vertex attribute 'name' and edge attribute
            'predicate' as set by load_kg.
        path: Destination file path (its extension is not interpreted).
        fmt: Output serialization — 'turtle'/'ttl' or 'nt'/'n-triples'.

    Raises:
        ValueError: if ``fmt`` is not a supported serialization.
    """
    path = Path(path)
    rdf_fmt = _FORMAT_ALIASES.get(fmt.lower())
    if rdf_fmt is None:
        raise ValueError(f"Unsupported format '{fmt}'. Use 'turtle' or 'nt'")

    rdf_graph = rdflib.Graph()

    def _to_rdf_node(v: igraph.Vertex) -> rdflib.term.Identifier:
        name: str = v["name"]
        if v["is_literal"]:
            datatype = v["literal_datatype"]
            lang = v["literal_lang"]
            value = v["literal_value"] if v["literal_value"] is not None else name
            if lang:
                return Literal(value, lang=lang)
            if datatype:
                return Literal(value, datatype=URIRef(datatype))
            return Literal(value)
        if name.startswith("_:"):
            return BNode(name[2:])
        return URIRef(name)

    for edge in graph.es:
        s_node = _to_rdf_node(graph.vs[edge.source])
        o_node = _to_rdf_node(graph.vs[edge.target])
        predicate = URIRef(edge["predicate"])
        rdf_graph.add((s_node, predicate, o_node))

    rdf_graph.serialize(destination=str(path), format=rdf_fmt)
