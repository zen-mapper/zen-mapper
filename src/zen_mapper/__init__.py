import logging
from dataclasses import dataclass

import numpy as np

from zen_mapper.cluster import Clusterer

from .cover import CoverScheme
from .simplex import SimplexTree

logger = logging.getLogger("zen_mapper")
logger.addHandler(logging.NullHandler())


@dataclass
class MapperResult:
    nodes: list[np.ndarray]
    nerve: SimplexTree
    cover: list[list[int]]


def mapper(
    data: np.ndarray,
    projection: np.ndarray,
    cover_scheme: CoverScheme,
    clusterer: Clusterer,
    dim: int | None,
    min_intersection: int = 1,
) -> MapperResult:
    assert len(data) == len(projection), (
        "the entries in projection have to correspond to entries in data"
    )

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

    nerve = SimplexTree()
    nerve.compute_nerve(covers=nodes, dim=dim, min_intersection=min_intersection)

    return MapperResult(
        nodes=nodes,
        nerve=nerve,
        cover=cover_id,
    )
