from dppy.exotic_dpps import *

# Build graph
g = nx.Graph()
edges = [(0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
g.add_edges_from(edges)

# Initialize UST object
ust = UST(g)
# Display original graph
ust.plot_graph()
# Display some samples
for _ in range(3):
    ust.sample()
    ust.plot()
# Display underlyin kernel i.e. transfer current matrix
ust.plot_kernel()
