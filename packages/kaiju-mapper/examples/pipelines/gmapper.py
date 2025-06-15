"""
G-Mapper
--------

G-Mapper, introduced in Alvarado, et al. [#gmapper]_, is a scheme for producing a
cover for Mapper inspired by the G-means clustering algorithm [#gmeans]_. In summary,
the algorithm iteratival splits intervals until the data contained within that
interval looks "normal enough" (the :code:`ad_threshold`). Instead of doing this split
naïvely the interval is split such that the resulting two intervals look as
normal as possible (using a gaussian mixture model).

This example will recreate the analysis represented in Figure 6 from the `arxiv
version <https://arxiv.org/abs/2309.06634>`_ of the paper. At this exact moment
in time we only implement the `BFS` strategy from their paper, all their
analyses used `DFS`, so some liberties were taken with the parameters.
"""

# %%
# Data
# ====
# We will start by generating a two circle data set like that from Figure 6 of
# the paper. To do this we will use Kajiu Mapper's :func:`sphere dataset
# <kaiju_mapper.datasets.sphere>`.

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
# We now have all the pieces in place to actually compute the mapper graph! We
# will use Kajiu Mappers :func:`sk_learn <kaiju_mapper.adapters.sk_learn>` adapter
# along with sk learns `DBSCAN` clutering algorithm to provide the clustering.
from sklearn.cluster import DBSCAN

from kaiju_mapper import GMapperCoverScheme, mapper
from kaiju_mapper.adapters import sk_learn

cover = GMapperCoverScheme(
    ad_threshold=10,
    g_overlap=0.1,
    max_intervals=6,
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

# %%
# Visualizing the mapper graph
# ============================
# At the present time Kaiju Mapper really does not provide much assistance with
# visualization, sorry about that. This highlights how to use the
# :func:`networkx <kaiju_mapper.adapters.to_networkx>` adapter to produce a
# networkx graph which can be used for visualization.
import networkx as nx
from kaiju_mapper.adapters import to_networkx

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

# %%
# References
# ==========
# .. [#gmapper] E. Alvarado, R. Belton, E. Fischer, K. J. Lee, S. Palande, S.
#        Percival, and E. Purvine,  "G-mapper: Learning a cover in the
#        mapper construction" 2025.
#
# .. [#gmeans] G. Hamerly, C. Elkan, "Learning the k in k-means" 2003.
