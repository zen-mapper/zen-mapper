from collections.abc import Iterator
from typing import Protocol

import numpy as np


class Clusterer(Protocol):
    def __call__(self, data: np.ndarray) -> Iterator[np.ndarray]:
        ...


class sk_learn:
    """A wrapper for sk-learn clusterers"""

    def __init__(self, clusterer):
        """Wraps an sk-learn clusterer for use with zen-mapper

        Parameters
        ----------
        clusterer:
            A clusterer implementing the sk-learn api
        Returns
        -------
        clusterer:
            A clusterer implementing the zen-mapper api
        """
        self.clusterer = clusterer

    def __call__(self, data: np.ndarray) -> Iterator[np.ndarray]:
        if len(data) == 0:
            yield np.array([])
            return

        labels = np.unique(self.clusterer.fit_predict(data))
        labels = labels[
            labels != -1
        ]  # -1 indicates noise, we don't do anything with it
        c = self.clusterer.labels_ == labels[:, np.newaxis]
        yield from (np.flatnonzero(x) for x in c)
