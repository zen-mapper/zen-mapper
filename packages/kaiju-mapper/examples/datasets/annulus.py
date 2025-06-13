"""
Annulus
----
"""

import matplotlib.pyplot as plt

from kaiju_mapper.datasets import annulus

data = annulus(
    dim=1,
    minor_radius=1.5,
    major_radius=2.0,
    num_samples=1_000,
    seed=0xDEADBEEF,
)
plt.scatter(data[:, 0], data[:, 1])
plt.gca().axis("equal")
plt.show()

# %%
# we also allow for higher dimensional annuli if that's your jam
data = annulus(
    dim=2,
    minor_radius=1.5,
    major_radius=2.0,
    num_samples=500,
    seed=0xDEADBEEF,
)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.axis("equal")
plt.show()
