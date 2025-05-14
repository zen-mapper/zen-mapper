import logging
from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar

import numpy as np

logger = logging.getLogger("zen_mapper")

M = TypeVar("M", covariant=True)


class Clusterer(Protocol[M]):
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
