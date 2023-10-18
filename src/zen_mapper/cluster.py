from collections.abc import Iterator
from typing import Protocol

import numpy as np


class Clusterer(Protocol):
    """A function which partitions a data set

    In particular it is a function which takes a data array and returns an
    iterator of arrays of indices into that array which are disjoint.
    """

    def __call__(self, data: np.ndarray) -> Iterator[np.ndarray]:
        ...


class sk_learn:
    """Wraps an sk-learn clusterer for use with zen-mapper"""

    def __init__(self, clusterer):
        self.clusterer = clusterer
        """A clusterer implementing the sk-learn api"""

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
