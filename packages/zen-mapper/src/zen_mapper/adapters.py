"""Adapters for converting between zen-mapper types and 3rd party types"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import numpy.typing as npt

from zen_mapper.types import Clusterer, Komplex

__all__ = ["to_networkx", "sk_learn"]

logger = logging.getLogger("zen_mapper")

if TYPE_CHECKING:
    import networkx as nx


def to_networkx(komplex: Komplex) -> nx.Graph:
    """Convert a zen-mapper komplex to a networkx graph

    This function takes a `Komplex` object, which represents a simplicial complex,
    and converts it into a `networkx.Graph` object. The vertices of the `Komplex`
    become the nodes in the `networkx` graph, and the 1-simplices (edges) of
    the `Komplex` become the edges in the `networkx` graph.

    Args:
        komplex: The `Komplex` object to convert. This object is expected to
            have a `vertices` attribute and support indexing for its simplices
            (e.g., `komplex[1]` for 1-simplices).

    Returns:
        A graph representing the 0 and 1-dimensional structure of the input
        `Komplex`.

    Raises:
        ImportError: If the `networkx` library is not installed.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is needed to export to networkx") from e

    G = nx.Graph()
    G.add_nodes_from(komplex.vertices)
    # Type checking is mad here because it can't determine that all the tuples
    # in `komplex[1]` are of dimension 2. I can't think of an obvious way to
    # remedy this right now.
    G.add_edges_from(komplex[1])  # type: ignore
    return G


C = TypeVar("C")


def sk_learn(
    base_clusterer: C,
    precomputed: bool | None = None,
) -> Clusterer[np.ndarray, C]:
    """Wrap a scikit-learn clusterer for use with zen-mapper.

    This function acts as an adapter, allowing scikit-learn's clustering
    algorithms to be integrated into the zen-mapper pipeline. Note: any
    datapoints which are considered noise by the base clusterer are ignored.

    Args:
        base_clusterer: An instance of a scikit-learn compatible clustering
            algorithm. This object should have a `fit_predict` method and a
            `labels_` attribute after fitting, which is standard for scikit-learn
            clusterers.

        precomputed: True if the scikit-learn algorithm is expecting a distance
            matrix. If not specified the adapter attempts to detect this from
            `base_clusterer`.

    Returns:
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

    if precomputed is None:
        precomputed = getattr(base_clusterer, "metric", "") == "precomputed"

    def inner(
        data: npt.ArrayLike,
        elements: np.ndarray,
    ) -> tuple[Collection[np.ndarray], C]:
        _data = np.asarray(data)

        clusterer: C = sk.clone(base_clusterer)  # type: ignore

        if precomputed:
            masked_data = _data[np.ix_(elements, elements)]
        else:
            masked_data = _data[elements]

        if len(masked_data) <= 1:
            return (np.arange(len(masked_data)),), clusterer

        labels = np.unique(clusterer.fit_predict(masked_data))  # type: ignore

        # -1 indicates noise, we don't do anything with it
        if -1 in labels:
            noise_points = labels == -1
            logger.warning(
                "the clusterer has labeled %d points as noise, "
                "they are being discarded",
                noise_points.size,
            )

            labels = labels[~noise_points]

        c = clusterer.labels_ == labels[:, np.newaxis]  # type: ignore
        return [np.flatnonzero(idx) for idx in c], clusterer

    return inner


def sk_learn_node(
    clusterer: C,
    data: npt.ArrayLike,
    elements: np.ndarray,
    precomputed: bool | None = None,
) -> tuple[list[np.ndarray], C]:
    """Wrap a scikit-learn clusterer for use with zen-mapper.

    This function acts as an adapter, allowing scikit-learn's clustering
    algorithms to be integrated into the zen-mapper pipeline. Note: any
    datapoints which are considered noise by the base clusterer are ignored.

    Args:
        clusterer: An instance of a scikit-learn compatible clustering
            algorithm. This object should have a `fit_predict` method and a
            `labels_` attribute after fitting, which is standard for scikit-learn
            clusterers.

        precomputed: True if the scikit-learn algorithm is expecting a distance
            matrix. If not specified the adapter attempts to detect this from
            `clusterer`.

    Returns:
        A tuple `(clusters, clusterer)` where `clusters` is the list of
        computed clusters and `clusterer` is the `sklearn` object used for
        fitting. Which allows for introspection should you want it.
    """

    try:
        import sklearn as sk
    except ImportError as e:
        raise ImportError(
            "sk-learn needs to be installed to use the sk_learn adapter"
        ) from e

    if precomputed is None:
        precomputed = getattr(clusterer, "metric", "") == "precomputed"

    _data = np.asarray(data)

    _clusterer: C = sk.clone(clusterer)  # type: ignore

    if precomputed:
        masked_data = _data[np.ix_(elements, elements)]
    else:
        masked_data = _data[elements]

    if len(masked_data) <= 1:
        return [np.arange(len(masked_data))], _clusterer

    labels = np.unique(_clusterer.fit_predict(masked_data))  # type: ignore

    # -1 indicates noise, we don't do anything with it
    if -1 in labels:
        noise_points = labels == -1
        logger.warning(
            "the clusterer has labeled %d points as noise, they are being discarded",
            noise_points.size,
        )

        labels = labels[~noise_points]

    c = _clusterer.labels_ == labels[:, np.newaxis]  # type: ignore
    ind = (np.flatnonzero(idx) for idx in c)
    return [elements[mask] for mask in ind], _clusterer
