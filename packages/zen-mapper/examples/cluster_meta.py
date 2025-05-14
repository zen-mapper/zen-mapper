"""
Getting clustering metadata
---------------------------

This example will go over how to extract metadata which a clustering algorithm
may have generated.
"""

import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2 * np.pi, 100)
data = np.c_[np.cos(theta), np.sin(theta)]
plt.scatter(data[:, 0], data[:, 1])
plt.show()

# %%
# Projecting our data
# ===================
projection = data[:, 0]
plt.scatter(data[:, 0], data[:, 1], label="data")
plt.scatter(projection, np.zeros_like(projection), label="projection")
plt.show()

# %%
# Covering our data
# =================
from zen_mapper.cover import Width_Balanced_Cover

cover_scheme = Width_Balanced_Cover(n_elements=3, percent_overlap=0.4)
cover = cover_scheme(projection)

# %%
# Defining a clusterer
# ====================
from sklearn.cluster import AffinityPropagation

from zen_mapper.cluster import sk_learn

sk = AffinityPropagation()
clusterer = sk_learn(sk)

# %%
# Computing the mapper graph
# ==========================
from zen_mapper import mapper

result = mapper(
    data=data,
    projection=projection,
    cover_scheme=cover_scheme,
    clusterer=clusterer,
    dim=1,
)

# %%
# Plotting using cluster_metadata
# ===============================
import networkx as nx

from zen_mapper.adapters import to_networkx

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(data[:, 0], data[:, 1], label="data")
pos = dict()
for cover_element, cluster_meta in zip(result.cover, result.cluster_metadata):
    for node, center in zip(cover_element, cluster_meta.cluster_centers_):
        pos[node] = center


G = to_networkx(result.nerve)
nx.draw(G, pos=pos, ax=ax2)
plt.tight_layout()
plt.show()
