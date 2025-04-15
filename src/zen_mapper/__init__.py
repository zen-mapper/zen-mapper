import logging
from dataclasses import dataclass

import numpy as np

from zen_mapper.cluster import Clusterer

from .cover import CoverScheme
from .komplex import Komplex, compute_nerve

logger = logging.getLogger("zen_mapper")
logger.addHandler(logging.NullHandler())


@dataclass
class MapperResult:
    nodes: list[np.ndarray]
    nerve: Komplex
    cover: list[list[int]]


def mapper(
    data: np.ndarray,
    projection: np.ndarray,
    cover_scheme: CoverScheme,
    clusterer: Clusterer,
    dim: int | None,
    min_intersection: int = 1,
) -> MapperResult:
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

    cover_elements = map(np.array, cover_scheme(projection))

    for i, element in enumerate(
        filter(lambda element: element.size != 0, cover_elements)
    ):
        logger.info("Clustering cover element %d", i)
        clusters = clusterer(data[element])
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
    )
