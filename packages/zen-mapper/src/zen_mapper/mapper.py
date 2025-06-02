import logging
from collections.abc import Iterable
from itertools import chain, combinations, count
from typing import TypeVar

import numpy as np

from .types import Clusterer, CoverScheme, Komplex, MapperResult, Simplex

__all__ = ["mapper"]

logger = logging.getLogger("zen_mapper")

M = TypeVar("M")


def mapper(
    data: np.ndarray,
    projection: np.ndarray,
    cover_scheme: CoverScheme,
    clusterer: Clusterer[M],
    dim: int | None,
    min_intersection: int = 1,
) -> MapperResult[M]:
    """
    Constructs a simplicial complex representation of the data.

    Parameters
    ----------
    data: np.ndarray
    projection: np.ndarray
        The output of the lens/filter function on the data. Must have the same
        number of elements as data.
    cover_scheme: CoverScheme
        For cover generation. Should be a callable object that takes a
        numpy array and returns a list of list(indices).
    clusterer: Clusterer
        A callable object that takes in a dataset and returns an iterator of
        numpy arrays which contain indices for clustered points.
    dim: int
        The highest dimension of the mapper complex to compute.
    min_intersection: int
        The minimum intersection required between clusters to make a simplex.

    Returns
    -------
    MapperResult
        An object containing:
        - nodes: List of clusters where each cluster is a list of data indices.
        - nerve: A complete list of simplices.
        - cover: List of list(indices) corresponding to elements of the cover.
    """
    assert len(data) == len(projection), (
        "the entries in projection have to correspond to entries in data"
    )

    nodes = list()
    cover_id = list()
    metadata = list()

    cover_elements = map(np.array, cover_scheme(projection))

    for i, element in enumerate(cover_elements):
        logger.info("Clustering cover element %d", i)
        clusters, meta = clusterer(data[element])
        metadata.append(meta)
        new_nodes = [element[cluster] for cluster in clusters]
        logger.info("Found %d clusters", len(new_nodes))
        if new_nodes:
            m = len(nodes)
            n = len(new_nodes)
            cover_id.append(list(range(m, m + n)))
            nodes.extend(new_nodes)
        else:
            cover_id.append(list())

    return MapperResult(
        nodes=nodes,
        nerve=compute_nerve(nodes, dim=dim, min_intersection=min_intersection),
        cover=cover_id,
        cluster_metadata=metadata,
    )


def compute_nerve(
    nodes,
    dim: int | None = 1,
    min_intersection: int = 1,
) -> Komplex:
    """Helper function to find edges of the overlapping clusters.

    Parameters
    ----------
    nodes:
        A dictionary with entries `{node id}:{list of ids in node}`
    dim:
        An optional int, specifies the maximal dimension simplex. A value of
        `None` puts no bound on the dimension. `dim = 0` returns only the nodes
        of the complex. Default: 1
    min_intersection:
        How many points of intersection two covers should have to count as
        connected. Default: 1
    Returns
    -------
    simplices:
        Complete list of simplices
    """
    assert dim is None or dim >= 0, "dim must be at least 0"
    assert min_intersection >= 1, "min_intersection must be at least 1"
    logger.info("Computing the nerve")
    n = len(nodes)
    komplex = Komplex(Simplex((i,)) for i in range(n))
    logger.info("Found %d 0-complexes", n)

    if dim == 0:
        return komplex

    _nodes = [frozenset(node) for node in nodes]

    dimensions = range(1, dim + 1) if dim else count(1)

    prev = set(Simplex((i,)) for i in range(n))
    for current_dim in dimensions:
        logger.info("Searching for %d-complexes", current_dim)

        candidates = _get_candidates(prev, current_dim)
        prev = set()

        for candidate in candidates:
            elements = map(lambda x: _nodes[x], candidate)
            if len(frozenset.intersection(*elements)) >= min_intersection:
                prev.add(candidate)
                komplex.add(candidate)

        logger.info("Found %d %d-complexes", len(prev), current_dim)

        if not prev:
            # No k-simplices were found, there are no k+1 simplices either
            break

    return komplex


def _get_candidates(prev: Iterable[Simplex], dim) -> Iterable[Simplex]:
    """Given previously found simplices generate new candidates simplices

    A k-simplex must have all its faces. We look for sets of k, (k-1) simplices
    that potentially border a k-simplex to generate candidate k-simplices.

    Params
    ------
    prev: a collection of simplices

    dim: the dimension of simplex we are searching for

    Yields
    -------
    simplices
    """
    for x in combinations(prev, dim + 1):
        candidate = Simplex(set(chain(*x)))
        if candidate.dim == dim:
            yield candidate
