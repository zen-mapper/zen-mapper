"""
G-Mapper
--------

Data
====
We will start by generating a two circle data set like that from Figure 6 of
the paper.
"""

import matplotlib.pyplot as plt
import numpy as np

from kaiju_mapper.datasets import sphere

rng = np.random.default_rng(seed=0xDEADBEEF)

data = sphere(
    dim=1,
    radius=[0.4, 1],
    num_samples=10_000,
    seed=rng,
)

# Add some noise to our samples
data += rng.normal(0, 0.03, data.shape)

# Plot the data set
plt.scatter(data[:, 0], data[:, 1], s=1)
plt.gca().axis("equal")
plt.show()

# %%
# Lens function
# =============
# Next we will construct our lens function. In the paper they compute the sum
# of the x and y coordinate then normalize the result to lie in the interval
# [0, 1]. We will do the same.

projection = data[:, 0] + data[:, 1]

# Normalization
a, b = np.min(projection), np.max(projection)
projection -= a
projection /= b - a

# Double check that we normalized
assert np.isclose(np.min(projection), 0)
assert np.isclose(np.max(projection), 1)

# Visualize
plt.scatter(data[:, 0], data[:, 1], c=projection, s=1)
plt.gca().axis("equal")
plt.show()

# %%
# Computing the mapper graph
# ==========================
import networkx as nx
from sklearn.cluster import DBSCAN

from kaiju_mapper import GMapperCoverScheme, mapper
from kaiju_mapper.adapters import sk_learn, to_networkx

cover = GMapperCoverScheme(
    ad_threshold=10,
    g_overlap=0.1,
    max_intervals=5,
    iterations=100,
    seed=rng,
)

clusterer = sk_learn(DBSCAN(eps=0.1, min_samples=5))

result = mapper(
    data=data,
    projection=projection,
    cover_scheme=cover,
    clusterer=clusterer,
    dim=1,
)

# Convert to a networkx graph for visualization
graph = to_networkx(result.nerve)

# Compute the average projection value for each node
nodes = (result.nodes[node] for node in graph.nodes)
colors = [np.mean(projection[node]) for node in nodes]

# Compute the layout of the graph
kk = nx.kamada_kawai_layout(graph)

# Center each connected component, not necessary but makes the image look nicer
for component in nx.connected_components(graph):
    mean = np.zeros(2)
    for node in component:
        mean += kk[node]
    mean /= len(component)
    for node in component:
        kk[node] -= mean

# Draw the graph, using the average projection value as our color value
nx.draw(graph, node_color=colors, node_size=1_150, pos=kk)
plt.show()
# sphinx_gallery_thumbnail_number = -1
