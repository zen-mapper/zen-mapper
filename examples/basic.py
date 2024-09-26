"""
Basic Use
---------
"""

# %%
# Generating data
# ===============
import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2 * np.pi, 50)
data = np.c_[np.cos(theta), np.sin(theta)]
plt.scatter(data[:, 0], data[:, 1])
plt.show()

# %%
# Projecting our data
# ===================
projection = data[:, 0]
plt.scatter(data[:, 0], data[:, 1], label="data")
plt.scatter(projection, np.zeros_like(projection), label="projection")
plt.legend()
plt.show()

# %%
# Covering our data
# =================
from zen_mapper.cover import Width_Balanced_Cover

cover_scheme = Width_Balanced_Cover(n_elements=3, percent_overlap=0.4)
cover = cover_scheme(projection)


for i, c in enumerate(cover):
    plt.scatter(
        projection[c],
        np.full_like(c, i),
        label=f"Cover element: {i+1}",
    )

plt.legend()
plt.show()

# %%
# Pulling back
# ============
for i, c in enumerate(cover):
    plt.scatter(
        data[c, 0],
        data[c, 1] + 0.5 * i,
        label=f"Cover element: {i+1}",
    )

plt.legend()
plt.show()

# %%
# Defining a clusterer
# ====================
from sklearn.cluster import AgglomerativeClustering

from zen_mapper.cluster import sk_learn

sk = AgglomerativeClustering(
    linkage="single",
    n_clusters=None,
    distance_threshold=0.2,
)
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
# Visualizing the mapper graph
# ============================
import networkx as nx

from zen_mapper.adapters import to_networkx

graph = to_networkx(result.nerve)
nx.draw(graph)
