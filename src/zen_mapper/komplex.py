import logging
import sys
from collections.abc import Iterable
from itertools import chain, combinations, count

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

logger = logging.getLogger("zen_mapper")


class Simplex(tuple[int, ...]):
    def __new__(cls, vertices: Iterable[int]):
        _simplex = sorted(vertices)
        assert len(_simplex) == len(
            set(_simplex)
        ), "A simplex must not have repeated elements"
        assert len(_simplex) != 0, "A simplex must have at least one vertex"
        return super().__new__(cls, tuple(_simplex))

    @property
    def dim(self: Self) -> int:
        """The dimension of the simplex

        The dimension of a simplex is defined to be one less than the number of
        elements in the simplex. Thus a 0-simplex (a vertex) is comprised of a
        single point, a 1-simplex (an edge) is comprised of two points, and so
        on.
        """
        return len(self) - 1

    @property
    def faces(self: Self) -> Iterable[Self]:
        """All the faces of a simplex

        A simplex θ is a face of τ if and only if θ ⊆ τ. Note that as τ ⊆ τ
        that τ is a face of τ!

        Yields
        ------
        simplex
            a face of the simplex
        """
        for i in range(1, len(self) + 1):
            yield from map(Simplex, combinations(self, i))

    @property
    def vertices(self: Self) -> Iterable[int]:
        yield from self


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

    @property
    def vertices(self: Self) -> Iterable[int]:
        for simplex in self[0]:
            yield from simplex.vertices


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
