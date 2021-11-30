from networkx import Graph

from dppy.graph.uniform_spanning_tree import UST

# Build graph
g = Graph()
edges = [(0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
g.add_edges_from(edges)

# Initialize UST object
ust = UST(g)
