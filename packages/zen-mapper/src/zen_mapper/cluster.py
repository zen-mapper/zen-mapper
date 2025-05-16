import logging
from collections.abc import Iterable
from typing import Protocol, TypeVar

import numpy as np

__all__ = ["Clusterer", "sk_learn"]

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


C = TypeVar("C")


def sk_learn(base_clusterer: C) -> Clusterer[C]:
    """Wraps a scikit-learn clusterer for use with zen-mapper.

    This function acts as an adapter, allowing scikit-learn's clustering
    algorithms to be integrated into the zen-mapper pipeline. Note: any
    datapoints which are considered noise by the base clusterer are ignored.

    Parameters
    ----------
    base_clusterer : C
        An instance of a scikit-learn compatible clustering algorithm.
        This object should have a `fit_predict` method and a `labels_`
        attribute after fitting, which is standard for scikit-learn
        clusterers.

    Returns
    -------
    Clusterer[C]
        An object conforming to the zen-mapper `Clusterer` protocol, which
        wraps the provided `clusterer`. This allows zen-mapper to use the
        scikit-learn clusterer's `fit_predict` methods within its pipeline. A
        copy of the fitted base clusterer is also returned as metadata allowing
        for inspection of the fitted model (e.g., centroids, parameters,
        dendrograms) after the mapper pipeline.
    """

    try:
        import sklearn as sk
    except ImportError as e:
        raise ImportError(
            "sk-learn needs to be installed to use the sk_learn adapter"
        ) from e

    def inner(data: np.ndarray) -> tuple[Iterable[np.ndarray], C]:
        clusterer: C = sk.clone(base_clusterer)  # type: ignore
        if len(data) <= 1:
            return (np.arange(len(data)),), clusterer

        labels = np.unique(clusterer.fit_predict(data))  # type: ignore

        if -1 in labels:
            logger.warning(
                "the clusterer has labeled some points as noise, "
                "they are being discarded"
            )

        labels = labels[
            labels != -1
        ]  # -1 indicates noise, we don't do anything with it
        c = clusterer.labels_ == labels[:, np.newaxis]  # type: ignore
        return map(np.flatnonzero, c), clusterer

    return inner
