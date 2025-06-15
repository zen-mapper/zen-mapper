"""
G-Mapper
------
"""

# %%
# Data
# ====
# We will start by generating a two circle data set like that from Figure 6 of
# the paper.

import matplotlib.pyplot as plt
import numpy as np

from kaiju_mapper.datasets import sphere

NUM_SAMPLES = 5000
rng = np.random.default_rng(seed=0xDEADBEEF)
data = np.full((NUM_SAMPLES, 2), fill_value=np.nan, dtype=float)

# Sample half of our points from a big circle
data[: NUM_SAMPLES // 2, :] = sphere(
    dim=1,
    radius=1,
    num_samples=NUM_SAMPLES // 2,
    seed=rng,
)
# Sample half of our points from a small circle
data[NUM_SAMPLES // 2 :, :] = sphere(
    dim=1,
    radius=0.3,
    num_samples=NUM_SAMPLES // 2,
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
# Lets doit
# =========
from sklearn.cluster import DBSCAN
import networkx as nx

from kaiju_mapper import GMapperCoverScheme, mapper
from kaiju_mapper.adapters import sk_learn, to_networkx

clusterer = sk_learn(DBSCAN(eps=0.1, min_samples=5))

cover = GMapperCoverScheme(
    ad_threshold=10,
    g_overlap=0.1,
    max_intervals=8,
    iterations=100,
)

result = mapper(
    data=data,
    projection=projection,
    cover_scheme=cover,
    clusterer=clusterer,
    dim=1,
)

for interval in cover.intervals:
    plt.scatter(data[interval.members,0], data[interval.members, 1])

plt.show()


# %%
# And now, a graph
# ================
graph = to_networkx(result.nerve)
nx.draw(graph)
plt.show()
