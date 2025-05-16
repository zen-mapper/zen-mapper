"""
Creating a custom clusterer
---------------------------

This example will go over the API for creating a clusterer. We will implement a
density-based clusterer (no real thought was put into how useful of a clusterer
this is). Given an epsilon > 0, we calculate the number of neighbors a point
has within an epsilon ball. We then cluster the dataset
into k (num_clusters) sets, aiming to equally divide the number of possible
neighbors. Note that two points being in
the same cluster tells us nothing about their Euclidean distance.
"""
# %%
# Creating a custom clusterer
# ===========================
#
# We will start by defining an epsilon neighbor function. Note that zen-mapper
# expects the clusterer to be an iterator that returns arrays. Each array
# corresponds to a cluster, and the elements of the array are the indices for
# the data points in that cluster. Outliers can be removed if necessary.

from collections.abc import Iterator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from zen_mapper import mapper
from zen_mapper.adapters import to_networkx
from zen_mapper.cover import Width_Balanced_Cover


def neighbor_counts(epsilon: float, data: np.ndarray) -> np.ndarray:
    assert epsilon > 0, "Epsilon must be greater than 0."
    if data.size == 0:
        return np.array([])  # return an empty array if given empty data
    n_points = data.shape[0]

    # initialize neighbor count
    neighbor_count = np.zeros(n_points, dtype=int)

    for i in range(n_points):
        distances = np.linalg.norm(data - data[i], axis=1)
        neighbor_count[i] = np.sum(distances <= epsilon) - 1

    return neighbor_count


def get_clusters(
    num_clusters: int, neighbor_counts: np.ndarray
) -> Iterator[np.ndarray]:
    max_neighbors = np.max(neighbor_counts) if np.any(neighbor_counts) else 0

    clusters_per_cluster = max_neighbors // num_clusters
    remainder = max_neighbors % num_clusters

    start = 0
    for i in range(num_clusters):
        end = start + clusters_per_cluster + (1 if i < remainder else 0)
        # distributes the remainder across initial clusters
        indices = np.flatnonzero((neighbor_counts > start) & (neighbor_counts <= end))
        # does not include isolated points

        yield indices  # yields the current cluster
        start = end  # moves the start index for the next cluster


def epsilon_density_clusterer(
    epsilon: float, num_clusters: int, data: np.ndarray
) -> Iterator[np.ndarray]:
    """
    Performs density-based clustering using an epsilon neighborhood.

    Parameters:
    - epsilon: A positive float defining the radius for neighbor counting.
    - k: An integer specifying the number of desired clusters.
    - data: A 2D NumPy array of shape (n_samples, n_features) representing the dataset.

    Returns:
    - An iterator yielding clusters,
    where each cluster is represented as a
    NumPy array of indices.
    """
    # calculate the density counts for the given data
    density_counts = neighbor_counts(epsilon=epsilon, data=data)

    # generate and return clusters based on the density counts
    return get_clusters(num_clusters=num_clusters, neighbor_counts=density_counts)


# make our clusterer passable to zen-mapper:
def clusterer(data):
    return epsilon_density_clusterer(epsilon, num_clusters, data)


# clustering parameters
epsilon = 0.1
num_clusters = 4

# %%
# we can then visualize to see if this clusterer acts as we would expect.

# dataset parameters
n_points = 4000
noise_level = 0.1  # Noise level for the radius

# generate angles uniformly between 0 and 2*pi
angles = np.linspace(0, 2 * np.pi, n_points)

# generate radii close to 1 with some noise
radii = 1 + noise_level * np.random.randn(n_points)

# convert polar coordinates to Cartesian coordinates
x = radii * np.cos(angles)
y = radii * np.sin(angles)


# stack x and y into a 2D array for clustering
data = np.column_stack((x, y))

clusters = list(
    epsilon_density_clusterer(epsilon=epsilon, num_clusters=num_clusters, data=data)
)

# create an array for cluster labels
cluster_labels = np.full(data.shape[0], -1)  # use default label for noise points
for cluster_id, indices in enumerate(clusters):
    cluster_labels[indices] = cluster_id

# plotting
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    data[:, 0], data[:, 1], c=cluster_labels, cmap="viridis", s=30, alpha=0.75
)
plt.colorbar(scatter, label="Cluster ID")
plt.title("Density Clustering")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()


# %%
# Using with mapper
# =================
#
# Now we have a clusterer compatible with zen-mapper.

cover_scheme = Width_Balanced_Cover(n_elements=5, percent_overlap=0.25)
cover = cover_scheme(data)
projection = data[:, 0]

result = mapper(
    data=data,
    projection=projection,
    cover_scheme=cover_scheme,
    clusterer=clusterer,
    dim=1,
)

graph = to_networkx(result.nerve)

# plot the mapper graph
nx.draw_kamada_kawai(graph)
plt.show()


# %%
# Coloring the nodes
# ==================

density_counts = neighbor_counts(epsilon=epsilon, data=data)

node_densities = {}

for node_id, indices in enumerate(result.nodes):
    # calculate the average density for the current cluster
    node_densities[node_id] = np.mean(density_counts[indices])

# create a color map based on node densities
node_colors = [node_densities[node] for node in graph.nodes]

# set up figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# plot the mapper graph with nodes colored by density
pos = nx.kamada_kawai_layout(graph)
sm = nx.draw_networkx_nodes(
    graph,
    pos,
    node_color=node_colors,
    node_size=100,
    ax=ax,
)
nx.draw_networkx_edges(graph, pos)

# add the color bar for average density
cbar = fig.colorbar(sm, ax=ax, label="Average Density")

plt.title("Mapper Graph Colored by Average Density")
plt.show()
# %%
