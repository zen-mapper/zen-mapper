import logging
from collections.abc import Iterator
from typing import Protocol

import numpy as np

logger = logging.getLogger("zen_mapper")
"""Logs clustering process, will note if points are determined to be 'noise' and are subsequently removed.
"""


class Clusterer(Protocol):
    """A function which partitions a data set.

    In particular, it is a function which takes a data array and returns an
    iterator of arrays of indices that partition the data into disjoint clusters.

    A simple example of such a clusterer is given as follows:

    .. code-block:: python
        :linenos:

        import numpy as np
        from typing import Iterator

        def simple_clusterer(data: np.ndarray) -> Iterator[np.ndarray]:
            # Here, we define a basic clusterer which partitions our dataset into two sets:
            # Cluster 1: All data points where the first feature is < 0
            # Cluster 2: All data points where the first feature is >= 0
            
            # Get the indices for the two clusters
            cluster_1 = np.flatnonzero(data[:, 0] < 0)  # Indices where the first feature < 0
            cluster_2 = np.flatnonzero(data[:, 0] >= 0)  # Indices where the first feature >= 0
            
            # Yield the clusters (disjoint sets of indices)
            yield cluster_1
            yield cluster_2

        # Example data
        data = np.array([
            [-1.0, 2],
            [3, 0.4],
            [-0.5, -6]
        ])

        # Use the simple clusterer to partition the data
        for cluster in simple_clusterer(data):
            print(f"Cluster: {cluster}")

    The above will return [0, 2]  and  [1].

    """

    def __call__(self, data: np.ndarray) -> Iterator[np.ndarray]: 
        """Calls the clusterer to partition the dataset into clusters.

        Parameters
        ----------
        data : np.ndarray
            The dataset to be partitioned into clusters.

        Returns
        -------
        Iterator[np.ndarray]
            An iterator over arrays of indices, where each array represents the indices
            of data points belonging to a specific cluster.
        """
        ...


class sk_learn:
    """Wraps an sk-learn clusterer for use with zen-mapper.
    
    Acts as an adapter to allow scikit-learn clustering algorithms to be used within ZenMapper. It implements the callable interface.

    Parameters
    ----------
    clusterer : object
        An instance of a clustering algorithm with implements the fit_predict method. Examples include KMeans, DBSCAN, etc.

    Methods
    -------
    __call__(data: np.ndarray) -> Iterator[np.ndarray]:
        Takes in a dataset and returns an iterator over clusters represented as arrays of indices which correspond to input data.
    """

    def __init__(self, clusterer):
        self.clusterer = clusterer
        """A clusterer implementing the sk-learn api"""

    def __call__(self, data: np.ndarray) -> Iterator[np.ndarray]:
        """
        Notes
        -----

        Fits the clusterer to the input data and yields indices of the data points that belong to each cluster. 
        - Noise points, if any, will be discarded.
        - If the input data is empty an empty array is yielded.
            
        Warnings
        --------

        If the clusterer identifies any points as noise (indicated by a label of -1), 
        a warning is logged and those points are excluded from output clusters.
        """
        if len(data) == 0:
            yield np.array([], dtype=int)
            return

        labels = np.unique(self.clusterer.fit_predict(data))

        if -1 in labels:
            logger.warn(
                "the clusterer has labeled some points as noise, "
                "they are being discarded"
            )

        labels = labels[
            labels != -1
        ]  # -1 indicates noise, we don't do anything with it
        c = self.clusterer.labels_ == labels[:, np.newaxis]
        yield from (np.flatnonzero(x) for x in c)
