from zen_mapper.komplex import Komplex


def to_networkx(komplex: Komplex):
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is needed to export to networkx") from e

    G = nx.Graph()
    G.add_nodes_from(komplex.vertices)
    G.add_edges_from(komplex[1])
    return G
