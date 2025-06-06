"""
Ball
----
"""

import matplotlib.pyplot as plt

from kaiju_mapper.datasets import ball

data = ball(dim=2, radius=1.5, num_samples=1_000, seed=0xDEADBEEF)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.axis("equal")
plt.show()

# %%
# turns out a disk is just a one ball, look at that
data = ball(dim=1, radius=1.5, num_samples=1_000, seed=0xDEADBEEF)
plt.scatter(data[:, 0], data[:, 1])
plt.gca().axis("equal")
plt.show()
