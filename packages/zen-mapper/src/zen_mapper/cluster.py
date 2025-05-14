import logging
from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar

import numpy as np

logger = logging.getLogger("zen_mapper")

M = TypeVar("M", covariant=True)


class Clusterer(Protocol[M]):
    """A callable that partitions a dataset.

    A clusterer takes a dataset and divides it into distinct groups,
    known as a partition. A partition is a collection of arrays, where each array
    contains indices referencing the original dataset. These index arrays
    must collectively include all data points from the original dataset,
    and no data point should belong to more than one array (i.e., the
    arrays are disjoint and cover the entire dataset).

    For example, for a dataset with 6 elements, `[[1, 2, 3], [0, 4], [5]]` is a
    valid partition. However, `[[1, 2, 3], [4], [5]]` is not valid because the
    element at index `0` is missing. Similarly, `[[1, 2, 3], [0, 4], [0, 5]]`
    is not valid because the element at index `0` appears in multiple arrays.

    In addition to the partition, the clusterer also produces metadata.
    The clusterer, when called, must return a tuple containing two elements:
    the partition (an iterable of NumPy arrays, where each array holds indices)
    and the associated metadata. If there is no meaningful metadata it should
    return None.
    """

    def __call__(self, data: np.ndarray) -> tuple[Iterable[np.ndarray], M]: ...


try:
    import sklearn as sk

    C = TypeVar("C")

    class sk_learn(Generic[C]):
        """Wraps an sk-learn clusterer for use with zen-mapper"""

        def __init__(self, clusterer: C):
            self.clusterer = clusterer
            """A clusterer implementing the sk-learn api"""

        def __call__(self, data: np.ndarray) -> tuple[Iterable[np.ndarray], C]:
            self.clusterer: C = sk.clone(self.clusterer)  # type: ignore
            if len(data) <= 1:
                return (np.arange(len(data)),), self.clusterer

            labels = np.unique(self.clusterer.fit_predict(data))  # type: ignore

            if -1 in labels:
                logger.warning(
                    "the clusterer has labeled some points as noise, "
                    "they are being discarded"
                )

            labels = labels[
                labels != -1
            ]  # -1 indicates noise, we don't do anything with it
            c = self.clusterer.labels_ == labels[:, np.newaxis]  # type: ignore
            return (np.flatnonzero(x) for x in c), self.clusterer
except ImportError:
    ...
