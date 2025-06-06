"""
Simplex
-------

You can sample an arbitrary simplex using the :func:`~kaiju_mapper.datasets.simplex` dataset.
"""

import matplotlib.pyplot as plt

from kaiju_mapper.datasets import simplex

closed = simplex(
    simplex=[[0, 0], [2, 0], [2, 2]],
    num_samples=500,
    seed=0xDEADBEEF,
)

plt.scatter(closed[:, 0], closed[:, 1])
plt.plot([0, 2, 2, 0], [0, 0, 2, 0])
plt.gca().axis("equal")
plt.show()

# %%
# Open Simplex
# ============
# By default this simplex is closed, that is it includes its boundary. You can
# instead only sample the interior by setting `closed = False`.

open = simplex(
    simplex=[[0, 0], [2, 0], [2, 2]],
    closed=False,
    num_samples=500,
    seed=0xDEADBEEF,
)

plt.scatter(open[:, 0], open[:, 1])
plt.plot([0, 2, 2, 0], [0, 0, 2, 0])
plt.gca().axis("equal")
plt.show()

# %%
# It's a computer... they're going to look very similar. If we inspect we can
# see that they're not the same thing.

import numpy as np

np.max(abs(open - closed))

# %%
# Unit Simplex
# ------------
#
# If for some reason you want the unit simplex specifically there is
# :func:`~kaiju_mapper.datasets.unit_simplex` just for you. It is slightly
# faster as it avoids some matrix multiplication.

from kaiju_mapper.datasets import unit_simplex

closed = unit_simplex(
    dim=2,
    num_samples=500,
    seed=0xDEADBEEF,
)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(closed[:, 0], closed[:, 1], closed[:, 2])
ax.plot([0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0])
ax.axis("equal")
ax.view_init(elev=20.0, azim=22.5)
plt.show()

# %%
# It also supports sampling the interior

open = unit_simplex(
    dim=2,
    closed=False,
    num_samples=500,
    seed=0xDEADBEEF,
)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(open[:, 0], open[:, 1], open[:, 2])
ax.plot([0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0])
ax.axis("equal")
ax.view_init(elev=20.0, azim=22.5)
plt.show()
