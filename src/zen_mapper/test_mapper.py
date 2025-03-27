import numpy as np
from sklearn.cluster import DBSCAN

from . import mapper
from .cluster import sk_learn
from .cover import Width_Balanced_Cover
from .simplex import get_skeleton


def test_mapper():
    theta = np.linspace(
        -np.pi,
        np.pi,
        1000,
        endpoint=False,
    )
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    db = DBSCAN(eps=0.1, min_samples=2)
    clusterer = sk_learn(db)
    cover_scheme = Width_Balanced_Cover(3, 0.4)
    result = mapper(
        data=data,
        projection=data[:, 0],
        cover_scheme=cover_scheme,
        clusterer=clusterer,
        dim=1,
    )
    assert result.nerve.dimension == 1
    assert len(result.nerve.get_simplices()) == 8

    vertices = [s for s in get_skeleton(result.nerve, 0) if len(s) == 1]
    assert len(vertices) == 4

    edges = [s for s in get_skeleton(result.nerve, 1) if len(s) == 2]
    assert len(edges) == 4
