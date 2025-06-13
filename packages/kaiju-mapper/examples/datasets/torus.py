"""
Torus
----

The standard embedding of the torus in ℝ³, for higher dimensional versions
see :func:`kaiju_mapper.datasets.flat_torus`.
"""

import matplotlib.pyplot as plt

from kaiju_mapper.datasets import torus

data = torus(
    minor_radius=1.0,
    major_radius=2.0,
    num_samples=1_000,
    seed=0xDEADBEEF,
)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.axis("equal")
plt.show()
