from functools import reduce
from itertools import chain, combinations, permutations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from zen_mapper.simplex import SimplexNode, SimplexTree, get_skeleton


def count_simplices(st):
    """Count all simplices in the tree."""
    return len(st.get_simplices())


def insert_full_simplex(st, simplex):
    """Insert a simplex and all its faces into the tree."""
    simplex = sorted(simplex)
    for r in range(1, len(simplex) + 1):
        for face in combinations(simplex, r):
            st.insert(face)
    return st


def test_empty_tree():
    st = SimplexTree()
    assert st.dimension == -1
    assert count_simplices(st) == 0
    assert not st.root.children


def test_simplex_node():
    """Test SimplexNode functionality."""
    node = SimplexNode(vertex=5)
    assert node.vertex == 5
    assert node.children == {}
    assert node.parents == set()


def test_empty_simplex_insertion():
    st = SimplexTree()
    assert st.insert([]) == st.root


def test_simplex_insertion():
    st = SimplexTree()

    st.insert([1])
    assert st.dimension == 0
    assert count_simplices(st) == 1

    st.insert([1, 2])
    assert st.dimension == 1
    assert count_simplices(st) == 2

    st.insert([1, 2, 3])
    assert st.dimension == 2
    assert count_simplices(st) == 3


def test_find_simplex():
    st = SimplexTree()
    st.insert([1, 2, 3])

    assert [1, 2, 3] in st
    assert [3, 2, 1] in st

    assert [1, 2, 4] not in st


def test_insert_full_simplex():
    st = SimplexTree()
    insert_full_simplex(st, [1, 2, 3])

    assert [1] in st
    assert [2] in st
    assert [3] in st
    assert [1, 2] in st
    assert [1, 3] in st
    assert [2, 3] in st
    assert [1, 2, 3] in st

    assert count_simplices(st) == 7


def test_get_simplices():
    """Test retrieving all simplices."""
    st = SimplexTree()
    insert_full_simplex(st, [1, 2, 3, 4])  # Insert tetrahedron with all faces

    all_simplices = st.get_simplices()
    assert len(all_simplices) == 15

    vertices = st.get_simplices(dim=0)
    assert len(vertices) == 4
    assert sorted(vertices) == [(1,), (2,), (3,), (4,)]

    edges = st.get_simplices(dim=1)
    assert len(edges) == 6

    triangles = st.get_simplices(dim=2)
    assert len(triangles) == 4


def test_get_skeleton():
    st = SimplexTree()
    insert_full_simplex(st, [1, 2, 3, 4])  # insert tetrahedron

    skeleton0 = get_skeleton(st, 0)
    assert len(skeleton0) == 4
    assert sorted([s for s in skeleton0 if len(s) == 1]) == [(1,), (2,), (3,), (4,)]

    skeleton1 = get_skeleton(st, 1)
    assert len(skeleton1) == 10  # 6 edges

    skeleton2 = get_skeleton(st, 2)
    assert len(skeleton2) == 14  # 4 triangles


def test_string_representation():
    """Test string representation of SimplexTree."""
    st = SimplexTree()
    st.insert([1, 2, 3])

    tree_str = str(st)

    assert "1" in tree_str
    assert "2" in tree_str
    assert "3" in tree_str


@given(st.sets(st.integers(min_value=1, max_value=100), min_size=1, max_size=10))
def test_dimension_property(vertices):
    st = SimplexTree()
    simplex = list(vertices)
    st.insert(simplex)
    assert st.dimension == len(simplex) - 1


@given(st.sets(st.integers(min_value=1, max_value=100), min_size=1, max_size=5))
def test_permutation_invariance(vertices):
    st = SimplexTree()
    simplex = list(vertices)
    st.insert(simplex)

    for perm in permutations(simplex):
        assert list(perm) in st


@given(
    st.lists(
        elements=st.sets(
            elements=st.integers(min_value=1, max_value=20), min_size=1, max_size=5
        ),
        min_size=1,
        max_size=5,
    )
)
def test_insert_multiple_simplices(simplices):
    """Test inserting multiple simplices."""
    st = SimplexTree()

    for simplex in simplices:
        st.insert(list(simplex))

    for simplex in simplices:
        assert list(simplex) in st


def test_nerve_empty_cover():
    """Test nerve computation with an empty cover."""
    st = SimplexTree()
    st.compute_nerve([])
    assert st.dimension == -1
    assert count_simplices(st) == 0


def test_nerve_basic():
    cover = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}]

    st = SimplexTree()
    st.compute_nerve(covers=cover)

    for i in range(3):
        assert [i] in st

    assert [0, 1] in st
    assert [1, 2] in st
    assert [0, 2] in st

    assert [0, 1, 2] in st


def test_nerve_with_empty_intersections():
    cover = [{1, 2}, {3, 4}, {4, 5}, {1, 5}]

    st = SimplexTree()
    st.compute_nerve(cover)

    for i in range(4):
        assert [i] in st

    assert [0, 1] not in st
    assert [1, 2] in st
    assert [0, 3] in st
    assert [2, 3] in st

    assert [0, 1, 2] not in st
    assert [0, 2, 3] not in st
    assert [1, 2, 3] not in st


def test_nerve_with_numpy_arrays():
    """Test nerve computation with numpy array inputs."""
    cover = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([3, 4, 5])]

    st = SimplexTree()
    st.compute_nerve(cover)

    assert [0, 1] in st
    assert [1, 2] in st
    assert [0, 2] in st

    assert [0, 1, 2] in st


def test_nerve_dimension_limit():
    cover = [{1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 4, 5}, {1, 3, 4, 5}, {2, 3, 4, 5}]

    st = SimplexTree()
    st.compute_nerve(cover, dim=1)

    # All vertices and edges should exist
    for i in range(5):
        assert [i] in st

    for i, j in combinations(range(5), 2):
        assert [i, j] in st

    # no triangles
    for i, j, k in combinations(range(5), 3):
        assert [i, j, k] not in st


def test_nerve_min_intersection():
    """Test nerve computation with minimum intersection size."""
    cover = [{1, 2, 3, 4}, {1, 2, 5, 6}, {1, 7, 8, 9}]

    # With min_intersection=1, all three sets form a triangle
    st1 = SimplexTree()
    st1.compute_nerve(cover, min_intersection=1)
    assert [0, 1, 2] in st1

    # With min_intersection=2, only sets 0 and 1 form an edge
    st2 = SimplexTree()
    st2.compute_nerve(cover, min_intersection=2)
    assert [0, 1] in st2
    assert [0, 2] not in st2
    assert [1, 2] not in st2


elements_strategy = st.frozensets(
    st.integers(min_value=0, max_value=20), min_size=1, max_size=10
)


@given(st.sets(elements_strategy, max_size=5))
def test_nerve_property(elements):
    """Comprehensive property-based test for nerve computation."""
    element_list = list(elements)

    nerve_complex = SimplexTree()
    nerve_complex.compute_nerve(element_list, dim=None)

    vertices = range(len(elements))
    candidates = chain.from_iterable(
        combinations(vertices, i) for i in range(1, len(elements) + 1)
    )

    for candidate in candidates:
        intersection = reduce(
            lambda prev, new: prev.intersection(new),
            (element_list[i] for i in candidate),
            set(element_list[candidate[0]]) if candidate else set(),
        )

        if intersection:
            assert list(candidate) in nerve_complex
        else:
            assert list(candidate) not in nerve_complex
