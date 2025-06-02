"""Adapters for converting between zen-mapper types and 3rd party types"""

import logging
from collections.abc import Iterable
from typing import TypeVar

import numpy as np

from zen_mapper.types import Clusterer, Komplex

__all__ = ["to_networkx", "sk_learn"]

logger = logging.getLogger("zen_mapper")


def to_networkx(komplex: Komplex):
    """Converts a zen-mapper komplex to a networkx graph

    This function takes a `Komplex` object, which represents a simplicial complex,
    and converts it into a `networkx.Graph` object. The vertices of the `Komplex`
    become the nodes in the `networkx` graph, and the 1-simplices (edges) of
    the `Komplex` become the edges in the `networkx` graph.

    Parameters
    ----------
    komplex : Komplex
        The `Komplex` object to convert. This object is expected to
        have a `vertices` attribute and support indexing for its
        simplices (e.g., `komplex[1]` for 1-simplices).

    Returns
    -------
    networkx.Graph
        A `networkx.Graph` object representing the 0- and 1-dimensional
        structure of the input `Komplex`.

    Raises
    ------
    ImportError
        If the `networkx` library is not installed.
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
