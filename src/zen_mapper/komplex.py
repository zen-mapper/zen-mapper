from collections.abc import Iterable
from itertools import chain, combinations, count
from typing import Self


class Simplex:
    def __init__(self: Self, vertices: Iterable[int]) -> None:
        _simplex = sorted(vertices)
        assert len(_simplex) == len(
            set(_simplex)
        ), "A simplex must not have repeated elements"
        assert len(_simplex) != 0, "A simplex must have at least one vertex"

        self._simplex: tuple[int, ...] = tuple(_simplex)

    @property
    def dim(self: Self) -> int:
        return len(self._simplex) - 1

    @property
    def faces(self: Self):
        for i in range(1, len(self._simplex) + 1):
            yield from map(Simplex, combinations(self._simplex, i))

    def __eq__(self: Self, other: Self) -> bool:
        return self._simplex == other._simplex

    def __hash__(self: Self) -> int:
        return hash(self._simplex)

    def __iter__(self: Self):
        yield from self._simplex

    def __str__(self: Self) -> str:
        return str(self._simplex)

    def __repr__(self: Self) -> str:
        return str(self._simplex)


class Komplex:
    def __init__(self: Self, simplices: Iterable[Simplex] | None = None) -> None:
        self._simplices: set[Simplex] = set(simplices) if simplices else set()

    def add(self: Self, simplex: Simplex) -> None:
        self._simplices.add(simplex)

    @property
    def dim(self: Self) -> int:
        try:
            return max(simplex.dim for simplex in self._simplices)
        except ValueError:
            return 0

    def __contains__(self: Self, simplex: Simplex) -> bool:
        return simplex in self._simplices

    def __getitem__(self: Self, ind: int) -> Iterable[Simplex]:
        yield from (simplex for simplex in self._simplices if simplex.dim == ind)

    def __iter__(self: Self):
        yield from self._simplices


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
    n = len(nodes)
    komplex = Komplex(Simplex((i,)) for i in range(n))

    if dim == 0:
        return komplex

    _nodes = [frozenset(node) for node in nodes]

    dimensions = range(1, dim + 1) if dim else count(1)

    prev = set(Simplex((i,)) for i in range(n))
    for current_dim in dimensions:
        if not prev:
            # No k-simplices were found, there are no k+1 simplices either
            break

        candidates = _get_candidates(prev, current_dim)
        prev = set()

        for candidate in candidates:
            elements = map(lambda x: _nodes[x], candidate)
            if len(frozenset.intersection(*elements)) >= min_intersection:
                prev.add(candidate)
                komplex.add(candidate)

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
