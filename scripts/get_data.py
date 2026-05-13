from rdflib import Graph

g = Graph()
g.parse("https://ndownloader.figshare.com/files/1118822", format="n3")

g.serialize("aifb.ttl", format="turtle")

print(f"Done — {len(g)} triples")