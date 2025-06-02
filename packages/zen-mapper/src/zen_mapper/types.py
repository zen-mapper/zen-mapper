from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import Generic, Protocol, Self, TypeVar

import numpy as np
import numpy.typing as npt

__all__ = [
    "Clusterer",
    "Cover",
    "CoverScheme",
    "Komplex",
    "MapperResult",
    "Simplex",
]

M = TypeVar("M", covariant=True)


class Cover(Protocol):
    def __len__(self: Self) -> int: ...

    def __iter__(self: Self) -> Iterator[npt.ArrayLike]: ...


class CoverScheme(Protocol):
    def __call__(self: Self, data: np.ndarray) -> Cover: ...


class Simplex(tuple[int, ...]):
    def __new__(cls, vertices: Iterable[int]):
        _simplex = sorted(vertices)
        assert len(_simplex) == len(set(_simplex)), (
            "A simplex must not have repeated elements"
        )
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
    def faces(self: Self) -> Iterable["Simplex"]:
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


class Clusterer(Protocol[M]):
    """A callable that partitions a dataset.

    A clusterer takes a dataset and divides it into distinct groups,
    known as a partition. A partition is a collection of arrays, where each array
    contains indices referencing the original dataset. These index arrays
    must collectively include all data points from the original dataset,
    and no data point should belong to more than one array (i.e., the
    arrays are disjoint and cover the entire dataset).

    For example, for a dataset with 6 elements, `[[1, 2, 3], [0, 4], [5]]` is a
    valid partition. However, `[[1, 2, 3], [4], [5]]` is not valid because the
    element at index `0` is missing. Similarly, `[[1, 2, 3], [0, 4], [0, 5]]`
    is not valid because the element at index `0` appears in multiple arrays.

    In addition to the partition, the clusterer also produces metadata.
    The clusterer, when called, must return a tuple containing two elements:
    the partition (an iterable of NumPy arrays, where each array holds indices)
    and the associated metadata. If there is no meaningful metadata it should
    return None.
    """

    def __call__(self, data: np.ndarray) -> tuple[Iterable[np.ndarray], M]: ...


@dataclass
class MapperResult(Generic[M]):
    nodes: list[np.ndarray]
    nerve: Komplex
    cover: list[list[int]]
    cluster_metadata: list[M]
