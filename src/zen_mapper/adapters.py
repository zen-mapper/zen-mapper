from zen_mapper.simplex import SimplexTree


def to_networkx(simplex_tree: SimplexTree):
    """
    Exports the mapper graph to a networkx graph.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is needed to export to networkx") from e

    G = nx.Graph()

    vertices = [s[0] for s in simplex_tree.get_simplices(dim=0)]
    edges = simplex_tree.get_simplices(dim=1)

    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    return G


def simplex_tree_to_nx(simplex_tree):
    """
    Convert a SimplexTree structure to a NetworkX directed graph.

    Creates a graph representation of the SimplexTree where:
        - Each node in the tree becomes a node in the graph
        - Each parent-child relationship becomes a directed edge
        - Node attribute 'simplex' stores what simplex the path from root to node is
        - Paths from the root correspond to simplices in the mapper complex

    Parameters:
        simplex_tree (SimplexTree): The SimplexTree to convert

    Returns:
        networkx.DiGraph: A directed graph representing the tree structure
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("NetworkX is required.") from e

    G = nx.DiGraph()

    G.add_node("root", label="∅", type="root", depth=0)

    # Map to keep track of node IDs
    node_map = {simplex_tree.root: "root"}
    node_count = 0

    def traverse_tree(node, parent_id, depth=0):
        nonlocal node_count

        for vertex, child in sorted(node.children.items()):
            # Create ID
            node_id = f"node_{node_count}"
            node_count += 1

            # Add node with its vertex label and depth
            G.add_node(
                node_id, label=str(vertex), vertex=vertex, type="vertex", depth=depth
            )

            G.add_edge(parent_id, node_id)

            # Map this SimplexNode to its graph node ID
            node_map[child] = node_id

            # Recurse it
            traverse_tree(child, node_id, depth + 1)

    traverse_tree(simplex_tree.root, "root", 1)

    # Add simplex path information to each node
    for node_id in G.nodes():
        if node_id != "root":
            # Compute the path from node to root (excluding root)
            path = []
            current = node_id

            while current != "root":
                node_data = G.nodes[current]
                if "vertex" in node_data:
                    path.append(node_data["vertex"])
                predecessors = list(G.predecessors(current))
                if predecessors:
                    current = predecessors[0]
                else:
                    break

            # Store path as node attribute -- reversed for consistency
            G.nodes[node_id]["simplex"] = tuple(reversed(path))

    return G
