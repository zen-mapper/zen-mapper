from zen_mapper.komplex import Komplex


def to_networkx(komplex: Komplex):
    """
    Converts a :class:`Komplex` object to a :class:`networkx.Graph`.

    Takes a :class:`Komplex` object, representing a simplicial complex,
    and exports it as a :class:`networkx.Graph`. The vertices of the Komplex object are added
    as nodes in the graph, and the edges from the 1-dimensional simplices (pairs of vertices)
    are added as edges between those nodes in the :class:`networkx.Graph`.

    .. code-block:: python
        :linenos:

        def to_networkx(komplex: Komplex):
            try:
                import networkx as nx
            except ImportError as e:
                raise ImportError("networkx is needed to export to networkx") from e

            G = nx.Graph()
            G.add_nodes_from(komplex.vertices)
            G.add_edges_from(komplex[1])
            return G

    Parameters
    ----------
    komplex : :class:`~zen_mapper.komplex.Komplex`
        A Komplex object that contains vertices and edges, which will be mapped to a
        :class:`networkx.Graph`.

    Returns
    -------
    :class:`networkx.Graph`
        A :class:`networkx.Graph` object constructed from the vertices and edges of the Komplex object.

    Raises
    ------
    ImportError
        If :mod:`networkx` is not installed, the function will raise an ImportError explaining that
        :mod:`networkx` is required for this conversion.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is needed to export to networkx") from e

    G = nx.Graph()
    G.add_nodes_from(komplex.vertices)
    G.add_edges_from(komplex[1])
    return G
