import logging
from collections.abc import Iterable
from itertools import chain, combinations, count
from typing import Self

logger = logging.getLogger("zen_mapper")
"""Initializes our logger.
"""

class Simplex(tuple[int, ...]):
    """Represents a simplex as defined by its vertices.

    - A simplex is initialized as a list of distinct integers. 
        - Elements of the list are considered verticies of the simplex. 
    
    - The Simplex class allows one to create a new simplex
        - takes as input an iterable (list or set) of integers. 
        - The new simplex is a **non-empty and sorted** list of integers (vertices). 
        - The `tuple(_simplex)` below converts the sorted list of vertices back into a tuple before passing it to `super().__new__.` 
            - This is done to make the vertex set immutable.

    .. code-block:: python
        :linenos:

            def __new__(cls, vertices: Iterable[int]):
                _simplex = sorted(vertices) # Initializes the simplex 
                assert len(_simplex) == len(set(_simplex)), "A simplex must not have repeated elements"
                assert len(_simplex) != 0, "A simplex must have at least one vertex"
                return super().__new__(cls, tuple(_simplex))
    """

    def __new__(cls, vertices: Iterable[int]):
        """Create a new Simplex instance.

        Parameters
        ----------
        vertices : Iterable[int]
            An iterable containing unique vertex indices for the simplex. 
            The simplex must not be empty.

        Returns
        -------
        Simplex
            A new simplex instance.
        """
        _simplex = sorted(vertices)
        assert len(_simplex) == len(set(_simplex)), "A simplex must not have repeated elements"
        assert len(_simplex) != 0, "A simplex must have at least one vertex"
        return super().__new__(cls, tuple(_simplex))

    @property
    def dim(self: Self) -> int:
        """The dimension of a simplex is defined to be one less than the number of vertices. Thus a 0-simplex (a vertex) is comprised of a single point :math:`(v)`, a 1-simplex (an edge) is comprised of two points, :math:`(v, w)`, and so on.

        Returns
        -------
        int
            The dimension of the simplex.
        """
        return len(self) - 1

    @property
    def faces(self: Self) -> Iterable[Self]:
        """Generates all faces of the simplex as subsets of the vertices. For example, if we are given a 2-dimensional simplex :math:`\\Delta^{2} = \\left(0, 1, 2 \\right)` then the generated faces will be:

        .. math::
            (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), \\text{ and } (0, 1, 2)

            
        .. code-block:: python
            :linenos:
            
            def faces(self: Self) -> Iterable[Self]:
                for i in range(1, len(self) + 1):
                    yield from map(Simplex, combinations(self, i))


        Yields
        ------
        simplex
            Each face of the simplex, including the simplex itself.
        """
        for i in range(1, len(self) + 1):
            yield from map(Simplex, combinations(self, i))

    @property
    def vertices(self: Self) -> Iterable[int]:
        """Obtains the vertices of the simplex. For example, :math:`(0, 1, 2)` will return :math:`0`, then :math:`1`, then :math:`2`.

        Here is an example usage:

        .. code-block:: python
            :linenos:

            from zen_mapper import komplex

            X = [0,3,1,2]
            S = komplex.Simplex(X)


            # Iterate over the generator to print each vertex
            for vertex in S.vertices:
                print(vertex)

        Which will print: 0  1  2  3.

        Note that :math:`X = [0,0,1,2]` will throw an error since a simplex cannot have repeated vertices.

        Yields
        ------
        int
            The vertex indices of the simplex.
        """
        yield from self


class Komplex:
    """The Komplex class represents a collection of simplices, which together form a simplicial complex (topological space). 
    
    Either spellings of (K/C)omplex are valid. For us, komplex will refer to an object we are actively computing and complex will refer to the mathematical definitions.

    This class provides methods for:

        - Adding a simplex to the komplex.

        - Obtaining dimension of the komplex.

        - Checking if a given simplex is contained in the komplex.

        - Retrieving all simplicies of a given dimension.
    """

    def __init__(self: Self, simplices: Iterable[Simplex] | None = None) -> None:
        """Initialize a Komplex instance.

        Parameters
        ----------
        simplices : Iterable[Simplex] | None, optional
            An iterable of Simplex instances to initialize the Komplex. 
            Defaults to an empty Komplex if None.
        """
        self._simplices: set[Simplex] = set(simplices) if simplices else set()

    def add(self: Self, simplex: Simplex) -> None:
        """Adds a simplex to the komplex.

        Parameters
        ----------
        simplex : Simplex
            The simplex to be added.
        """
        self._simplices.add(simplex)

    @property
    def dim(self: Self) -> int:
        """Determines the maximum dimension of the simplices in the komplex.

        Returns
        -------
        int
            The maximum dimension of the simplices, or :math:`0` if the komplex is empty.
        """
        try:
            return max(simplex.dim for simplex in self._simplices)
        except ValueError:
            return 0

    def __contains__(self: Self, simplex: Simplex) -> bool:
        """Checks for the presence of a simplex in the komplex. Since the komplex is a set of simplicies this just checks if the given simplex is in the set.

        Parameters
        ----------
        simplex : Simplex
            The simplex to check.

        Returns
        -------
        bool
            True if the simplex is in the komplex, False otherwise.
        """
        return simplex in self._simplices

    def __getitem__(self: Self, ind: int) -> Iterable[Simplex]:
        """Get all simplices of a specific dimension.

        Parameters
        ----------
        ind : int
            The dimension of simplices to retrieve.

        Yields
        ------
        Simplex
            The simplices of the specified dimension.
        """
        yield from (simplex for simplex in self._simplices if simplex.dim == ind)

    def __iter__(self: Self):
        """Iterates over the simplices in the komplex by returning elements from the komplex set. 

        Yields
        ------
        Simplex
            The simplices in the komplex.
        """
        yield from self._simplices

    @property
    def vertices(self: Self) -> Iterable[int]:
        """Yields all vertices of the simplices in the komplex. Goes through the komplex set and obtains the vertices of each 0-simplex.

        Yields
        ------
        int
            The vertex indices of all 0-simplices in the komplex.
        """
        for simplex in self[0]:
            yield from simplex.vertices

def _get_candidates(prev: Iterable[Simplex], dim) -> Iterable[Simplex]:
    """Generate candidate simplices from previously found simplices.

    A :math:`(k+1)`-simplex must contain all math:`k+1` of its faces. Candidate simplices are constructed by using previously found
    k-simplices. A candidate is yielded only if it has the correct dimension.

    .. code-block:: python
        :linenos:

        def _get_candidates(prev: Iterable[Simplex], dim) -> Iterable[Simplex]:
            for x in combinations(prev, dim + 1):       # For each collection of dim-simplicies with size dim + 1
                candidate = Simplex(set(chain(*x)))     # Creates a candidate simplex
            if candidate.dim == dim:                    # Check if the candidate has the right dimension
                yield candidate

    Parameters
    ----------
    prev : Iterable[Simplex]
        A collection of previously found simplices.
    dim : int
        The dimension of the simplex to generate candidates for.

    Yields
    -------
    Simplex
        Candidate simplices that may form new k-simplices.
    """


                
    for x in combinations(prev, dim + 1): 
        candidate = Simplex(set(chain(*x))) 
        if candidate.dim == dim: 
            yield candidate

def compute_nerve(
    nodes,
    dim: int | None = 1,
    min_intersection: int = 1,
) -> Komplex:
    """
    Returns a simplicial complex which is an approximation of the nerve.

    The nerve of an open cover is a simplicial complex in which each vertex corresponds to an element of the cover and each simplex represents a non-empty intersection of the corresponding sets in the cover.

    In our case, each vertex :math:`x_{i}` represents a cluster :math:`C_{i}` in our dataset obtained by the mapper algorithm. We include the face :math:`\{x_{i_{1}}, \dots, x_{i_{n}}\}` if 
    
    .. math::
        \\vert\\cap_{k \\geq 1}^{n} C_{i_{k}} \\vert \\geq \\text{min_intersection}


        


    Parameters
    ----------
    nodes : dict
        A dictionary mapping node (vertex) ids to lists of connected node ids.
    dim : int | None, optional
        Specifies the maximal dimension simplex to compute. If None, 
        there is no upper limit on the dimension. Default is 1.
    min_intersection : int, optional
        The minimum number of intersecting nodes required for two covers 
        to be considered connected. Default is 1.

    Returns
    -------
    Komplex
        A Komplex containing the computed simplices representing node connectivity.


    .. code-block:: python
        :linenos:
        
        def compute_nerve(
            nodes,
            dim: int | None = 1,
            min_intersection: int = 1,
        ) -> Komplex:
            assert dim is None or dim >= 0, "dim must be at least 0"

            n = len(nodes) # Number of clusters

            ##################################
            # We initialize a komplex with just 0-simplices
            ##################################
            
            komplex = Komplex(Simplex((i,) for i in range(n)))

            ##################################
            # compute faces from dimensions 1 to dim
            ##################################

            _nodes = [frozenset(node) for node in nodes]

            dimensions = range(1, dim + 1) if dim else count(1)


            prev = set(Simplex((i,) for i in range(n)))

            ##################################
            # prev = collection of previously found simplicies
            ##################################

            for current_dim in dimensions:

                candidates = _get_candidates(prev, current_dim)
                prev = set()

        
            #################################
            # For each candidate simplex 

            # elements = list of clusters which is indexed by nodes

            # elements* = unpacks the list to calculate intersection between the clusters
            ##################################

            for candidate in candidates:
                elements = map(lambda x: _nodes[x], candidate)
                if len(frozenset.intersection(*elements)) >= min_intersection:  # If the clusters from vertices satisfies intersection condition
                    prev.add(candidate)                                         # add candidate simplex to previously found simplices
                    komplex.add(candidate)                                      # add candidate to the nerve komplex

            if not prev:
                # No k-simplices were found, implying no k+1 simplices exist
                break

        return komplex
    """
    assert dim is None or dim >= 0, "dim must be at least 0"
    assert min_intersection >= 1, "min_intersection must be at least 1"
    logger.info("Computing the nerve")
    n = len(nodes) #Number of clusters

    ##################################
    # We initialize a komplex with just 0-simplices for each cluster.
    ##################################
    komplex = Komplex(Simplex((i,) for i in range(n)))
    logger.info("Found %d 0-complexes", n)

    ##################################
    # If the maximal dimension of the komplex is set to 0 then we are done!
    ##################################
    if dim == 0:
        return komplex
    

    ##################################
    # Make nodes immutable then compute faces from dimension 1 to dim
    ##################################

    _nodes = [frozenset(node) for node in nodes]

    dimensions = range(1, dim + 1) if dim else count(1)


    prev = set(Simplex((i,) for i in range(n)))
    ##################################
    # prev = collection of previously found simplicies
    ##################################
    for current_dim in dimensions:
        logger.info("Searching for %d-complexes", current_dim)

        candidates = _get_candidates(prev, current_dim)
        prev = set()

        
        #################################
        # For each candidate simplex 

        # elements = list of clusters which is indexed by nodes

        # elements* = unpacks the list to calculate intersection between the clusters
        ##################################
        for candidate in candidates:
            elements = map(lambda x: _nodes[x], candidate)
            if len(frozenset.intersection(*elements)) >= min_intersection:  # If the clusters from vertices satisfies intersection condition
                prev.add(candidate)                                         # add candidate simplex to previously found simplices
                komplex.add(candidate)                                      # add candidate to the nerve komplex

        logger.info("Found %d %d-complexes", len(prev), current_dim)

        if not prev:
            # No k-simplices were found, implying no k+1 simplices exist
            break

    return komplex


