import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, sets

from zen_mapper.simplex import SimplexTree


def test_empty_tree():
    tree = SimplexTree()
    assert tree.dimension == -1
    assert tree.num_simplices == 1  # counts empty face


def test_invalid_simplex():
    tree = SimplexTree()
    with pytest.raises(ValueError):
        tree.insert_simplex([2, 1, 3]) 
    with pytest.raises(ValueError):
        tree.insert_simplex([1, 2, 2])


@given(sets(integers(min_value=0, max_value=100), min_size=1, max_size=5))
def test_insert_find_simplex(vertices):
    tree = SimplexTree()
    simplex = sorted(vertices)
    tree.insert_simplex(simplex)
    assert tree.find_simplex(simplex) is not None

    # test non-existent simplex
    non_existent = list(range(max(vertices) + 1, max(vertices) + len(vertices) + 1))
    assert tree.find_simplex(non_existent) is None


@given(sets(integers(min_value=0, max_value=100), min_size=1, max_size=15))
def test_dimension(vertices):
    tree = SimplexTree()
    simplex = sorted(vertices)
    tree.insert_simplex(simplex)
    assert tree.dimension == len(simplex) - 1


@given(lists(
        integers(min_value=0, max_value=100), 
        min_size=1, 
        max_size=15, 
        unique=True
        ))
def test_facets(vertices):
    tree = SimplexTree()
    simplex = sorted(vertices)
    tree.insert_simplex(simplex)

    # get all facets
    facets = tree.locate_facets(simplex)

    # verify facets
    for facet in facets:
        assert len(facet) == len(simplex) - 1
        assert set(facet).issubset(set(simplex))

    # verify number of facets
    assert len(facets) == len(simplex)


@given(sets(integers(min_value=0, max_value=100), min_size=1, max_size=15))
def test_cofaces(vertices):
    tree = SimplexTree()
    simplex = sorted(vertices)
    tree.insert_full_simplex(simplex)

    vertex = [min(vertices)]
    cofaces = tree.locate_cofaces(vertex)

    # Verify each coface contains the vertex
    for coface in cofaces:
        assert vertex[0] in coface
        assert tree.find_simplex(coface) is not None


def test_elementary_collapse():
    tree = SimplexTree()
    # Create a simple example with a free pair
    face = [1, 2]
    coface = [1, 2, 3]
    tree.insert_full_simplex(coface)

    # Verify collapse
    assert tree.elementary_collapse(face, coface)

    # check thta simplices are removed
    assert tree.find_simplex(face) is None
    assert tree.find_simplex(coface) is None


def test_edge_contraction():
    tree = SimplexTree()
    tree.insert_full_simplex([1, 2, 3, 4])

    tree.edge_contraction(1, 2)

    assert tree.find_simplex([1, 2]) is None
    assert tree.find_simplex([1, 3]) is not None
    assert tree.find_simplex([1, 4]) is not None


@given(sets(integers(min_value=0, max_value=50), min_size=1, max_size=10))
def test_skeleton(vertices):
    tree = SimplexTree()
    simplex = sorted(vertices)
    tree.insert_full_simplex(simplex)

    # Get k-skeleton for each k up to dimension
    dim = len(simplex) - 1
    for k in range(dim + 1):
        skeleton = tree.get_skeleton(k)
        # No simplices of dimension > k
        assert all(len(s) <= k + 1 for s in skeleton)


@given(sets(integers(min_value=0, max_value=100), min_size=1, max_size=15))
def test_star_link(vertices):
    tree = SimplexTree()
    simplex = sorted(vertices)
    tree.insert_full_simplex(simplex)

    # convert to tuples (not sure if lists are a good choice but...sigh)
    vertex = tuple([min(vertices)])
    star = {tuple(s) for s in tree.get_star(list(vertex))} 
    link = {tuple(s) for s in tree.get_link(list(vertex))}

    # Containment properties
    assert all(vertex[0] in s for s in star)
    assert all(vertex[0] not in s for s in link)

    # Star = vertex * Link ( * = join operation)
    computed_star = set()
    for l in link:
        new_simplex = tuple(sorted([vertex[0]] + list(l)))
        computed_star.add(new_simplex)
    # add the vertex to star
    computed_star.add(vertex)
    assert star == computed_star

    # Link consists of faces that complete star simplices
    for s in star:
        if len(s) > 1:
            # Remove vertex to get link simplex
            link_simplex = tuple(sorted(x for x in s if x != vertex[0]))
            assert link_simplex in link

    # Simplices exist in original complex
    all_simplices = {tuple(s) for s in tree.get_skeleton(tree.dimension)}
    assert all(s in all_simplices for s in star)
    assert all(s in all_simplices for s in link)

    # Verify dimensions
    if len(vertices) > 1:
        assert max(len(s) - 1 for s in link) < max(len(s) - 1 for s in star)


def test_remove_simplex():
    tree = SimplexTree()
    simplex = [1, 2, 3]
    tree.insert_full_simplex(simplex)

    tree.remove_simplex(simplex)
    assert tree.find_simplex(simplex) is None

    # verify the faces still exist
    for i in range(len(simplex)):
        face = simplex[:i] + simplex[i+1:]
        if face:  # skip empty face
            assert tree.find_simplex(face) is not None


def test_persistence():
    tree = SimplexTree()
    simplex = [1, 2, 3]
    tree.insert_full_simplex(simplex)

    initial_simplices = set(tree.get_skeleton(2))
    assert (1, 2, 3) in initial_simplices

    # Remove [1, 2] -- this should also remove [1,2,3]
    tree.remove_simplex([1, 2])

    # Verify only [1,2] and [1,2,3] are removed
    after_removal = set(tree.get_skeleton(2))
    assert (1, 2) not in after_removal
    assert (1, 2, 3) not in after_removal
    assert (1,) in after_removal
    assert (2,) in after_removal
    assert (3,) in after_removal
    assert (1, 3) in after_removal
    assert (2, 3) in after_removal

    # Insert [1,2]
    tree.insert_simplex([1, 2])

    # Verify:
        # [1,2] is back
        # [1,2,3] is still gone

    final_simplices = set(tree.get_skeleton(2))
    assert (1, 2) in final_simplices
    assert (1, 2, 3) not in final_simplices


def test_cleanup():
    tree = SimplexTree()

    tree.insert_full_simplex([0, 1, 2])
    tree.insert_full_simplex([1, 2, 3])
    tree.insert_full_simplex([2, 3, 4, 5])

    all_simplices = tree.get_skeleton(tree.dimension)
    for s in all_simplices:
        print(f"  {s} (dim {len(s)-1})")

    assert tree.dimension == 3

    simplices = tree.get_skeleton(tree.dimension)
    simplices = [s for s in simplices if s] # ignores empty face

    print("\nRemoving simplices in this order:")
    for simplex in sorted(simplices, key=len, reverse=True):
        tree.remove_simplex(list(simplex))
        
    final_simplices = tree.get_skeleton(max(0, tree.dimension))
    print("Remaining simplices:")
    for s in final_simplices:
        print(f"  {s} (dim {len(s)-1})")

    # verify cleanup
    assert tree.dimension == -1
    assert tree.num_simplices == 1  # only empty face remains






