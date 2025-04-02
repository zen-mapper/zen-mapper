"""
Logging Example
---------------

In this example we demonstrate how to enable logging in zen mapper. The bulk of
the example is taken from the basic example, see that example for a more
discussion about the mapper bit.
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
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 50)
data = np.c_[np.cos(theta), np.sin(theta)]
plt.scatter(data[:, 0], data[:, 1])
plt.show()

# %%
# Running mapper
# ===============
import networkx as nx
from zen_mapper.adapters import to_networkx
from sklearn.cluster import AgglomerativeClustering
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
from zen_mapper import mapper

projection = data[:, 0]

cover_scheme = Width_Balanced_Cover(n_elements=3, percent_overlap=0.4)
cover = cover_scheme(projection)

sk = AgglomerativeClustering(
    linkage="single",
    n_clusters=None,
    distance_threshold=0.2,
)
clusterer = sk_learn(sk)


result = mapper(
    data=data,
    projection=projection,
    cover_scheme=cover_scheme,
    clusterer=clusterer,
    dim=1,
)

# %%
# Visualizing the mapper graph
# ============================

graph = to_networkx(result.nerve)
nx.draw(graph)
