from collections.abc import Collection, Iterable, Iterator
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

# Clusterer Metadata
M = TypeVar("M", covariant=True)

# High dimensional data type
H = TypeVar("H", contravariant=True)


class Cover(Protocol):
    """
    Protocol for a cover

    A set is represented as a numpy array of indices into the original data
    set. For instance `[0, 4, 3]` represents the set with the 0th, 4th, and 3rd
    elements from the original dataset. A cover is a collection of these sets.
    It is expected that these cover the dataset however no effort is made to
    enforce this constraint.

    Specifically we require that you can iterate over the sets in the cover and
    that you can report the number of cover elements. In particular this means
    a list of arrays or a set of arrays will work. It is unlikely that you will
    actually implement this protocol, what you probably want is
    :class:`CoverScheme`.
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

    A cover scheme is a function or callable object which takes the projected
    data and produces a `Cover` object. See the example
    :doc:`/examples/custom_cover` for a more detailed look at how to create a
    custom covering scheme.
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


class Clusterer(Protocol[H, M]):
    """
    A protocol defining a callable that partitions a dataset into disjoint groups.

    A `Clusterer` takes a dataset and a subset of its indices, returning a
    partition and associated metadata.

    Methods
    -------
    __call__(data, elements)
        Partition the specified elements of the dataset.

    Notes
    -----
    It is assumed that the returned partition satisfies the following properties:

    1. **Disjointness**: No index from `elements` appears in more than one
       partition array
    2. **Exhaustiveness**: The union of all partition arrays exactly
       equals the set of indices into `elements`
    3. **Non-Empty**: No partition element is empty

    For a dataset with 6 elements, `[[1, 2, 3], [0, 4], [5]]` is a valid
    partition. While the following are considered invalid:

    - `[[1, 2, 3], [4], [5]]` (missing index 0)
    - `[[1, 2, 3], [0, 4], [0, 5]]` (index 0 is duplicated)
    - `[[1, 2, 3], [0, 4], [], [5]]` (there is an empty partition element)

    If no meaningful metadata is produced by the clustering algorithm,
    the second element of the returned tuple should be `None`.

    See Also
    --------
    :doc:`/examples/custom_clusterer` : A narrated example of implementing a clusterer

    :func:`~zen_mapper.adapters.sk_learn` : An example clusterer defined in `zen_mapper`
    """

    def __call__(
        self,
        data: H,
        elements: np.ndarray,
    ) -> tuple[Collection[npt.ArrayLike], M]:
        """Partition a subset of the dataset into disjoint groups.

        Parameters
        ----------
        data : H
            The full dataset object.
        elements : np.ndarray
            An array of indices referencing the data points within `data`
            that are to be clustered.

        Returns
        -------
        partition : Collection[npt.ArrayLike]
            A Collection of NumPy array-like things. Each array contains
            indices into `elements`. The collection must form a partition of
            `elements`.
        metadata : M
            Associated metadata produced by the clustering process. Returns
            None if no meaningful metadata is generated.

        See Also
        --------
        Clusterer : The protocol defining the expected behavior and
            partitioning constraints.

        Examples
        --------
        >>> # If elements is [0, 2, 4, 6, 8]
        >>> # A valid return value might look like:
        >>> ([np.array([1, 2]), np.array([0, 4]), np.array([3])], None)
        """
        ...


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
    cluster_metadata : list[M | None]
        A list of clustering metadata objects. The object `cluster_metadata[i]`
        corresponds to whatever metadata the clusterer produced on cover
        element `i`. If cover element `i` was empty this will
        `cluster_metadata[i]` is `None`.
    """

    nodes: list[np.ndarray]
    nerve: Komplex
    cover: list[list[int]]
    cluster_metadata: list[M | None]
