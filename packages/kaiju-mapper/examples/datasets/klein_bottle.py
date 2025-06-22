"""
Klein Bottle
-----------

Provides a dataset sampled from a standard embedding of the n-dimensional Klein bottle in (n+2)-dimensions.
The Klein bottle is a non-orientable surface without boundary, similar to a Möbius strip but closed.
This example shows a 3D projection of the 4D parameterization of the Klein bottle.

For the mathematical construction, see Davis [1]_.

.. [1] Donald M. Davis, "n-dimensional Klein bottles", arXiv:1706.03704 [math.AT], 2017.
"""

import matplotlib.pyplot as plt

from kaiju_mapper.datasets import klein_bottle

data = klein_bottle(
    dim=2,
    num_samples=5000,
    scale=2.0,
    seed=0xDEADBEEF,
)


print("My suspicion for why the projection is not closed is...")
print("in the remarks section of: https://arxiv.org/pdf/0909.5354")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.scatter(
    data[:, 0],
    data[:, 1],
    data[:, 2],
    c=data[:, 3],
    cmap="viridis",
    alpha=0.2,
)
plt.show()
