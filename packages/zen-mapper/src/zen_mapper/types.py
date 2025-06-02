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
    """
    Protocol for a cover

    A `Cover` object must be iterable, yielding elements that are array-like
    (e.g., NumPy arrays), and must also have a defined length.

    Methods
    -------
    __len__()
        Returns the number of sets in the cover.
    __iter__()
        Returns an iterator over the sets in the cover.
    """

    def __len__(self: Self) -> int:
        """
        Returns the number of sets in the cover.

        Returns
        -------
        int
            The number of sets in the cover.
        """
        ...

    def __iter__(self: Self) -> Iterator[npt.ArrayLike]:
        """
        Returns an iterator over the sets in the cover.

        Each element yielded by the iterator is expected to be array-like,
        suitable for conversion to a NumPy array. The elements of these arrays
        are indices into the original data set

        Yields
        ------
        npt.ArrayLike
            An array-like object representing a set in the cover.
        """
        ...


class CoverScheme(Protocol):
    """
    Protocol for a cover scheme

    A cover scheme is a function or callable object that takes
    some data and produces a `Cover` object.

    Methods
    -------
    __call__(data)
        Generates a `Cover` from the input data.
    """

    def __call__(self: Self, data: np.ndarray) -> Cover:
        """
        Generates a `Cover` from the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data

        Returns
        -------
        Cover
            A `Cover` object representing the generated cover of the data.
        """
        ...


class Simplex(tuple[int, ...]):
    """
    A thin representation of a simplex

    A simplex is an ordered collection of vertex ids without repetition. This
    implementation is essentially a python tuple with some convenience methods
    bolted on.

    Parameters
    ----------
    vertices : Iterable[int]
        Vertex ids for the simplex

    Raises
    ------
    ValueError
        If `vertices` contains repeated elements (a simplex must have unique vertices).
    ValueError:
        If `vertices` is empty (a simplex must have at least one vertex).

    Examples
    --------
    >>> s1 = Simplex([1, 0, 2])
    >>> s1
    (0, 1, 2)
    >>> s2 = Simplex((5,))
    >>> s2
    (5,)
    """

    def __new__(cls, vertices: Iterable[int]):
        _simplex = sorted(vertices)
        if __debug__:
            if len(_simplex) != len(set(_simplex)):
                raise ValueError("A simplex must not have repeated elements")
            if len(_simplex) == 0:
                raise ValueError("A simplex must have at least one vertex")
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


# TODO: this class needs some tlc
class Komplex:
    """
    A basic simplicial complex

    A `Komplex` is a collection of simplices. It provides methods
    for adding simplices, checking for their presence, iterating over them,
    and querying their properties like dimension and vertices. At this time we
    do not enforce closure propreties for these simplicial complexes. This
    class is optimized for construction, querying it is slow. If you seek to do
    much with it we recomend converting it to something else.

    Parameters
    ----------
    simplices : Iterable[Simplex], optional
        An initial collection of `Simplex` objects to populate the complex.
        If `None`, the complex starts empty.

    Methods
    -------
    add(simplex)
        Adds a single simplex to the complex.
    dim
        Returns the highest dimension of any simplex in the complex.
    __contains__(simplex)
        Checks if a given simplex is present in the complex.
    __getitem__(ind)
        Yields all simplices of a specific dimension.
    __iter__()
        Iterates over all simplices in the complex.
    vertices
        Yields all unique vertices (0-simplices) in the complex.
    """

    def __init__(self: Self, simplices: Iterable[Simplex] | None = None) -> None:
        self._simplices: set[Simplex] = set(simplices) if simplices else set()

    def add(self: Self, simplex: Simplex) -> None:
        """
        Adds a simplex to the simplicial complex.

        Parameters
        ----------
        simplex : Simplex
            The `Simplex` object to add to the complex.
        """
        self._simplices.add(simplex)

    @property
    def dim(self: Self) -> int:
        """
        Returns the dimension of the simplicial complex.

        The dimension of the complex is the highest dimension of any simplex it
        contains. An empty complex has a dimension of 0.

        Returns
        -------
        int
            The dimension of the complex.
        """
        try:
            return max(simplex.dim for simplex in self._simplices)
        except ValueError:
            return 0

    def __len__(self: Self) -> int:
        return self._simplices.__len__()

    def __contains__(self: Self, simplex: Simplex) -> bool:
        """
        Checks if a given simplex is present in the complex.

        Parameters
        ----------
        simplex : Simplex
            The `Simplex` object to check for existence in the complex.

        Returns
        -------
        bool
            `True` if the simplex is in the complex, `False` otherwise.
        """
        return simplex in self._simplices

    def __getitem__(self: Self, ind: int) -> Iterable[Simplex]:
        """
        Yields all simplices of a specific dimension from the complex.

        Parameters
        ----------
        ind : int
            The dimension of the simplices to retrieve (e.g., 0 for vertices,
            1 for edges, 2 for triangles, etc.).

        Yields
        ------
        Simplex
            A simplex of the specified dimension.
        """
        yield from (simplex for simplex in self._simplices if simplex.dim == ind)

    def __iter__(self: Self):
        """
        Iterates over all simplices contained within the complex.

        Yields
        ------
        Simplex
            Each simplex present in the complex.
        """
        yield from self._simplices

    @property
    def vertices(self: Self) -> Iterable[int]:
        """
        Yields all unique vertex identifiers (0-simplices) present in the complex.

        Returns
        -------
        Iterable[int]
            An iterable of integer vertex identifiers.
        """
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
    """
    Output of a Mapper computation

    This dataclass encapsulates all the output generated by applying the Mapper
    algorithm. It includes information about the Mapper complex, the
    constructed cover, and any metadata associated with the clusters of the
    complex.

    Attributes
    ----------
    nodes : list[np.ndarray]
        A list of NumPy arrays, where each array represents the points that
        belong to a specific node in the Mapper graph. Each `np.ndarray`
        corresponds to a cluster formed by the algorithm.
    nerve : Komplex
        A `Komplex` object representing the nerve of the cover. This `Komplex`
        object defines the topological structure  of the Mapper graph, where
        vertices correspond to the `nodes` in this result.
    cover : list[list[int]]
        Each inner list contains the indices of the original data points that
        fall into a specific cover element. This provides a mapping from the
        original dataset to the cover.
    cluster_metadata : list[M]
        A list of metadata objects, where each object corresponds to a
        cluster (node) in the Mapper graph. The type `M` is generic,
        allowing for flexible storage of any additional information
        relevant to each cluster, such as cluster statistics, labels,
        or other derived properties.
    """

    nodes: list[np.ndarray]
    nerve: Komplex
    cover: list[list[int]]
    cluster_metadata: list[M]
