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
    """Holds the results of the mapper function.
    """


    nodes: list[np.ndarray] 
    nerve: Komplex                   
    cover: list[list[int]] 


def mapper(
    data: np.ndarray,
    projection: np.ndarray,
    cover_scheme: CoverScheme,  # type: ignore
    clusterer: Clusterer,       # type: ignore
    dim: int | None,
) -> MapperResult:
    """Performs the Mapper algorithm on the given data.

    This function clusters the data using the specified cover scheme and clusterer,
    and computes the nerve complex based on the resulting clusters.

    Parameters
    ----------
    data : np.ndarray
        The input data array where each row represents a data point.
    projection : np.ndarray
        The projected data to be used for clustering. It should correspond to the data points.
    cover_scheme : CoverScheme
        A callable that defines the cover scheme for the Mapper algorithm.
    clusterer : Clusterer
        A callable that partitions the data into clusters.
    dim : int | None
        The dimension of the nerve complex. If None, the dimension will not be restricted.

    Returns
    -------
    MapperResult
        An instance of `MapperResult` containing the nodes, the nerve complex, and the cover information.

    Raises
    ------
    AssertionError
        If the number of entries in `projection` does not match the number of entries in `data`.

    Notes
    -----
    This function logs information about the clustering process for each cover element,
    including the number of clusters found.
    """
    
    assert len(data) == len(
        projection
    ), "the entries in projection have to correspond to entries in data"

    nodes = list()
    cover_id = list()

    for i, element in enumerate(cover_scheme(projection)):
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
        nerve=compute_nerve(nodes, dim=dim),
        cover=cover_id,
    )
