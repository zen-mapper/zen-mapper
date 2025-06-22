import numpy as np
from sklearn.cluster import DBSCAN

from zen_mapper import mapper
from zen_mapper.cover import precomputed_cover

from .adapters import sk_learn


def test_sk_learn():
    N_SAMPLES = 1000
    theta = np.linspace(-np.pi, np.pi, N_SAMPLES)
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    db = DBSCAN(eps=0.1, min_samples=2)
    clusterer = sk_learn(db)
    clusters, _ = clusterer(data, np.arange(N_SAMPLES))
    clusters = list(map(np.asarray, clusters))
    assert len(clusters) == 1
    assert len(clusters[0]) == len(data)


def test_empty_dtype():
    db = DBSCAN(eps=0.1, min_samples=2)
    clusterer = sk_learn(db)
    clusters, _ = clusterer(np.array([]), np.arange(0))
    clusters = list(map(np.asarray, clusters))
    for cluster in clusters:
        assert cluster.dtype == "int64"


def test_non_array_cluster():
    """Clusterers should be able to return array like objects"""

    def _trivial_cluster(data, elements):
        return [range(len(elements))], None

    N_SAMPLES = 100
    data = np.empty(N_SAMPLES)
    mapper(
        data=data,
        projection=data,
        cover_scheme=precomputed_cover([np.arange(N_SAMPLES)]),
        clusterer=_trivial_cluster,
        dim=0,
    )
