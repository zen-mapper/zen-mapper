"""
Logging Example
---------------
In this example we demonstrate how to enable logging in zen mapper. The bulk of
the example is taken from :doc:`/examples/basic`, see that for discussion about
the mapper bit.
"""

# %%
# Enable logging
# ==============
# We use the `builtin python logging
# library <https://docs.python.org/3/library/logging.html>`_ to log
# internal activity. The simplest way to enable logging is to use
# `basicConfig`. A complete breakdown of the logging library is beyond the
# scope of this example however the official logging documentation is rather
# complete. Zen mapper logs to the `zen_mapper` logger.

import logging

logging.basicConfig(level=logging.INFO)

# %%
# Generating data
# ===============
import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2 * np.pi, 50)
data = np.c_[np.cos(theta), np.sin(theta)]
plt.scatter(data[:, 0], data[:, 1])
plt.show()

# %%
# Running mapper
# ===============
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

from zen_mapper import mapper, width_balanced_cover
from zen_mapper.adapters import sk_learn, to_networkx

projection = data[:, 0]

cover, _ = width_balanced_cover(
    n_elements=3,
    percent_overlap=0.4,
    data=projection,
)

sk = AgglomerativeClustering(
    linkage="single",
    n_clusters=None,
    distance_threshold=0.2,
)
clusterer = sk_learn(sk)


result = mapper(
    data=data,
    cover=cover,
    clusterer=clusterer,
    dim=1,
)

# %%
# Visualizing the mapper graph
# ============================

graph = to_networkx(result.nerve)
nx.draw(graph)
