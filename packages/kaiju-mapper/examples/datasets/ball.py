"""
Annulus
-------
"""

import matplotlib.pyplot as plt

from kaiju_mapper.datasets import annulus

data = annulus(minor_radius=1, major_radius=2, num_samples=500, seed=0xDEADBEEF)
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(data[:, 0], data[:, 1])
ax.axis("equal")
plt.show()


# %%
# Like every other method this works with arbitrary dimension... it's just a
# little tricky to showcase. Here we will hide every data point where :math:`z
# > 0` to try and highlight what's going on.
data = annulus(
    dim=2,
    minor_radius=1,
    major_radius=2,
    num_samples=2000,
    seed=0xDEADBEEF,
)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

mask = data[:, 2] <= 0
ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2])
ax.axis("equal")
plt.show()

# %%
# Well that didn't really work. Until we have interactive plots to peruse you
# will probably just have to trust us on this one.
