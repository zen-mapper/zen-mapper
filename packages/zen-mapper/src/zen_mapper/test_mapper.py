import numpy as np
from sklearn.cluster import DBSCAN

from .adapters import sk_learn
from .cover import width_balanced_cover
from .mapper import mapper


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
    projection = data[:, 0]
    cover, _ = width_balanced_cover(3, 0.4, projection)
    result = mapper(
        data=data,
        cover=cover,
        clusterer=clusterer,
        dim=1,
    )

    assert result.nerve.dim == 1
    assert len(list(result.nerve)) == 8
    assert len(list(result.nerve[0])) == 4
    assert len(list(result.nerve[1])) == 4
