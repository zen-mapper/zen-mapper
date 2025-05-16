from zen_mapper.adapters import to_networkx
from zen_mapper.komplex import Komplex, Simplex


def test_to_network_x():
    komplex = [
        [0],
        [1],
        [2],
        [0, 1],
        [1, 2],
        [2, 0],
    ]

    komplex = Komplex(Simplex(v) for v in komplex)
    G = to_networkx(komplex)
    assert set(G.nodes) == {0, 1, 2}
    assert set(G.edges) == {(1, 0), (2, 1), (2, 0)}
