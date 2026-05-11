"""Knowledge Graph I/O: load/save .ttl and .nt files as igraph objects."""

from pathlib import Path

import igraph
import rdflib
from rdflib import BNode, Literal, URIRef

_FORMAT_MAP = {".ttl": "turtle", ".nt": "nt"}


def _rdf_node_id(node: rdflib.term.Identifier) -> str:
    """Return a stable string key for an RDF node."""
    if isinstance(node, BNode):
        return f"_:{node}"
    return str(node)


def load_kg(path: str | Path) -> igraph.Graph:
    """Parse a .ttl or .nt file and return a directed igraph Graph.

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
    fmt = _FORMAT_MAP.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"Unsupported file extension '{path.suffix}'. Use .ttl or .nt")

    rdf_graph = rdflib.Graph()
    rdf_graph.parse(str(path), format=fmt)

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


def save_kg(graph: igraph.Graph, path: str | Path) -> None:
    """Serialize an igraph Graph (produced by load_kg) to a .ttl or .nt file.

    The graph must have vertex attribute 'name' and edge attribute 'predicate'
    as set by load_kg.
    """
    path = Path(path)
    fmt = _FORMAT_MAP.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"Unsupported file extension '{path.suffix}'. Use .ttl or .nt")

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

    rdf_graph.serialize(destination=str(path), format=fmt)
