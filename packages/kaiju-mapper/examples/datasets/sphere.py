"""
Sphere
------
"""

import matplotlib.pyplot as plt

from kaiju_mapper.datasets import sphere

data = sphere(dim=2, radius=1.5, num_samples=250, seed=0xDEADBEEF)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.axis("equal")
plt.show()

# %%
# turns out a circle is just a one sphere, look at that
data = sphere(dim=1, radius=1.5, num_samples=250, seed=0xDEADBEEF)
plt.scatter(data[:, 0], data[:, 1])
plt.gca().axis("equal")
plt.show()

# %%
# Nested Spheres
# ==============
# You can sample from multiple spheres by specify multiple radii.
data = sphere(dim=1, radius=[2, 4], num_samples=250, seed=0xDEADBEEF)
plt.scatter(data[:, 0], data[:, 1])
plt.gca().axis("equal")
plt.show()
