from zen_mapper.komplex import Komplex


def to_networkx(komplex: Komplex):
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is needed to export to networkx") from e

    G = nx.Graph()
    G.add_nodes_from(komplex.vertices)
    # Type checking is mad here because it can't determine that all the tuples
    # in `komplex[1]` are of dimension 2. I can't think of an obvious way to
    # remedy this right now.
    G.add_edges_from(komplex[1])  # type: ignore
    return G
