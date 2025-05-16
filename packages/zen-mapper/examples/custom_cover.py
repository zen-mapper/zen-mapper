"""
Creating a custom cover
-----------------------

This example will go over the API for creating a custom cover scheme. We will
be implementing the epsilon nets outlined in https://arxiv.org/abs/1901.07410.
This is meant to illustrate the expected api, not much thought was put into the
actual implementation of the cover itself and there may be better ways.
"""

# %%
# Constructing a cover
# ====================
#
# We will start by defining an epsilon-net function. This will take a list of
# centers, an epsilon, and a data array and return a list of numpy arrays.

import numpy as np

from zen_mapper.cover import Cover


def epsilon_net(centers, epsilon, data) -> Cover:
    if len(data.shape) == 1:
        data.reshape(-1, 1)

    result = []
    for center in centers:
        current = []
        for i, datum in enumerate(data):
            if np.sum((center - datum) ** 2) < epsilon**2:
                current.append(i)
        result.append(np.array(current, dtype=int))

    return result


# %%
# We can then visualize that this cover acts as we hope it would

import matplotlib.pyplot as plt

data = np.c_[np.arange(11), np.arange(11)]
centers = np.array([[5, 5], [3, 3]])
epsilon = 2

ax = plt.gca()
ax.scatter(data[:, 0], data[:, 1])
cover = epsilon_net(centers, epsilon, data)

for element in cover:
    ax.scatter(data[element, 0], data[element, 1])

for center in centers:
    circ = plt.Circle(center, epsilon, color="r", fill=False)
    ax.add_patch(circ)

ax.set_aspect("equal")
plt.show()

# %%
# Creating a covering scheme
# ==========================
#
# The mapper function in zen_mapper expects a `covering_scheme` which is simply
# a function which takes data and returns a cover. I find pythons partial
# function support to be kind of awkward. So instead of defining a function we
# will define a class whith the `__call__` method which python will treat like
# a function.


class Greedy_Epsilon_Net:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.centers = None

    def __call__(self, data: np.ndarray) -> Cover:
        to_cover = set(range(len(data)))

        centers = []

        while to_cover:
            new_center = to_cover.pop()
            centers.append(data[new_center])
            covered = set()

            for point in to_cover:
                if np.sum((data[new_center] - data[point]) ** 2) < self.epsilon**2:
                    covered.add(point)

            to_cover -= covered

        self.centers = centers

        return epsilon_net(centers, self.epsilon, data)


# %%
# we can then visualize the covering scheme in much the same way we visualized the cover

data = np.c_[np.arange(11), np.arange(11)]
scheme = Greedy_Epsilon_Net(epsilon=2)

ax = plt.gca()
ax.scatter(data[:, 0], data[:, 1])
cover = scheme(data)

for element in cover:
    ax.scatter(data[element, 0], data[element, 1])

for center in scheme.centers:
    circ = plt.Circle(center, epsilon, color="r", fill=False)
    ax.add_patch(circ)

ax.set_aspect("equal")
plt.show()

# %%
# Using with mapper
# =================
#
# Now that we have a cover we will demonstrate how to plug it into zen mapper
# to duplicate the analysis done in the original paper. We will be using a
# similar window dataset to the one they used. We start by generating our data.


def window(num_samples: int) -> np.ndarray:
    data = np.random.rand(num_samples, 2)
    data[:, 1] *= 9
    data[:, 0] += 4 * np.random.randint(0, 3, size=num_samples)
    choice = np.random.randint(0, 2, size=(num_samples, 1))
    return choice * data + (1 - choice) * data[:, [1, 0]]


np.random.seed(42)
data = window(1000)
plt.scatter(data[:, 0], data[:, 1])
plt.gca().set_aspect("equal")
plt.show()

# %%
# We also need to define a clustering algorithm. In this paper they don't
# actually cluster anything, instead any datapoint in the ball is deemed to be
# connected. So we define the following trivial clusterer.


def trivial(data: np.ndarray):
    yield np.arange(len(data))


# %%
# With this we are able to call mapper and get a very similar graph to the one
# from the paper.

import networkx as nx

from zen_mapper import mapper
from zen_mapper.adapters import to_networkx

result = mapper(
    data=data,
    projection=data,
    cover_scheme=Greedy_Epsilon_Net(1),
    clusterer=trivial,
    dim=1,
)

g = to_networkx(result.nerve)
nx.draw_kamada_kawai(g)
plt.show()
