import networkx as nx

from zen_mapper.adapters import simplex_tree_to_nx, to_networkx
from zen_mapper.simplex import SimplexTree


def test_to_network_x():
    st = SimplexTree()
    faces = [[0], [1], [2], [0, 1], [1, 2], [2, 0]]

    for f in faces:
        st.insert(f)

    G = to_networkx(st)

    # Check that the edges match exactly what we expect
    edge_sets = {frozenset(edge) for edge in G.edges}
    expected_edges = {frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])}
    assert edge_sets == expected_edges


def test_empty_tree():
    """Test conversion of an empty SimplexTree."""
    st = SimplexTree()
    G = simplex_tree_to_nx(st)

    # Should have just the root node
    assert G.number_of_nodes() == 1
    assert "root" in G.nodes
    assert G.nodes["root"]["label"] == "∅"
    assert G.nodes["root"]["type"] == "root"
    assert G.nodes["root"]["depth"] == 0


def test_single_vertex():
    st = SimplexTree()
    st.insert([1])
    G = simplex_tree_to_nx(st)

    # Should have root and one vertex
    assert G.number_of_nodes() == 2

    # Find the vertex node
    vertex_node = [n for n in G.nodes if n != "root"][0]
    assert G.nodes[vertex_node]["vertex"] == 1
    assert G.nodes[vertex_node]["simplex"] == (1,)
    assert G.nodes[vertex_node]["depth"] == 1


def test_edge():
    st = SimplexTree()
    st.insert([1, 2])
    G = simplex_tree_to_nx(st)

    # Should have root + two vertex nodes
    assert G.number_of_nodes() == 3

    # Find node with simplex (1, 2)
    edge_node = None
    for node, data in G.nodes(data=True):
        if "simplex" in data and data["simplex"] == (1, 2):
            edge_node = node
            break

    assert edge_node is not None
    assert G.nodes[edge_node]["depth"] == 2


def test_triangle():
    st = SimplexTree()
    st.insert([1, 2, 3])
    G = simplex_tree_to_nx(st)

    # Check all simplices are represented
    simplices = {data["simplex"] for _, data in G.nodes(data=True) if "simplex" in data}
    expected = {(1,), (1, 2), (1, 2, 3)}
    assert simplices == expected

    # Find the leaf node (triangle)
    leaf_node = None
    for node, out_degree in G.out_degree():
        if out_degree == 0 and node != "root":
            leaf_node = node
            break

    assert leaf_node is not None
    path = nx.shortest_path(G, "root", leaf_node)
    assert len(path) == 4  # root -> 1 -> 2 -> 3


def test_multiple_simplices():
    st = SimplexTree()
    st.insert([1, 2, 3])
    st.insert([1, 4])
    st.insert([5, 6])

    G = simplex_tree_to_nx(st)

    # all simplices?
    simplices = {data["simplex"] for _, data in G.nodes(data=True) if "simplex" in data}
    expected = {(1,), (1, 2), (1, 2, 3), (1, 4), (5,), (5, 6)}
    assert simplices == expected

    # exactly one parent
    for node in G.nodes():
        if node != "root":
            assert len(list(G.predecessors(node))) == 1


def test_shared_faces():
    st = SimplexTree()

    st.insert([1, 2, 3])
    st.insert([1, 2, 4])

    G = simplex_tree_to_nx(st)

    simplices = {data["simplex"] for _, data in G.nodes(data=True) if "simplex" in data}
    expected = {(1,), (1, 2), (1, 2, 3), (1, 2, 4)}
    assert simplices == expected

    # (1,2) should appear only once
    edge_nodes = [
        n for n, d in G.nodes(data=True) if "simplex" in d and d["simplex"] == (1, 2)
    ]
    assert len(edge_nodes) == 1


def test_attributes():
    st = SimplexTree()
    st.insert([1, 2])
    G = simplex_tree_to_nx(st)

    for node, data in G.nodes(data=True):
        if node == "root":
            assert data["type"] == "root"
            assert data["depth"] == 0
        else:
            assert data["type"] == "vertex"
            assert "vertex" in data
            assert "label" in data
            assert "simplex" in data


def test_sort_order():
    st = SimplexTree()
    st.insert([3, 1, 2])  # Insert in unsorted order
    G = simplex_tree_to_nx(st)

    # Find node
    full_node = None
    for node, data in G.nodes(data=True):
        if "simplex" in data and len(data["simplex"]) == 3:
            full_node = node
            break

    assert full_node is not None

    # stored in sorted order?
    assert G.nodes[full_node]["simplex"] == (1, 2, 3)

    # Trace path and check vertex ordering
    path = nx.shortest_path(G, "root", full_node)
    vertices = [G.nodes[n].get("vertex") for n in path[1:]]
    assert vertices == [1, 2, 3]
