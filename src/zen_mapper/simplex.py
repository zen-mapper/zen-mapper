from functools import lru_cache


class SimplexNode:
    """
    A node in the SimplexTree.

    Attributes:
        vertex: The vertex identifier represented by this node
        children (dict): Maps vertex identifiers to child SimplexNode objects
        parents (set): Set of parent SimplexNode objects
    """

    def __init__(self, vertex):
        self.vertex = vertex
        self.children = {}
        self.parents = set()


class SimplexTree:
    """
    A data structure for dealing with simplicial complexes.
    Uses a tree where each node (decorated by a vertex label) represents a simplex.
    The tree is organized such that the root represents the empty simplex,
    and each path to level k representing a (k-1)-simplices.

    Attributes
    -----------
        root : SimplexNode
            Root node of the tree representing the empty simplex.
        simplex_map : dict
            Maps frozenset of vertices to corresponding SimplexNode
        dimension : int
            Maximum dimension of any simplex in the tree.
            An empty tree has dimension -1.
        _covers : list
            Cache for covers to be used during nerve computation
    """

    def __init__(self):
        self.root = SimplexNode(vertex=None)
        self.simplex_map = {}
        self.dimension = -1
        self._covers = None

    def insert(self, simplex):
        """
        Insert a simplex into the tree and return the corresponding node.

        Parameters
        ----------
        simplex : list or iterable
            The vertices of the simplex to insert. An empty list represents
            the empty simplex (root node).

        Returns
        -------
        SimplexNode
            The node whose path from root represents the inserted simplex.
        """
        if not simplex:
            return self.root

        simplex = sorted(simplex)
        node = self.root

        # insert
        for vertex in simplex:
            if vertex not in node.children:
                child = SimplexNode(vertex)
                node.children[vertex] = child
                child.parents.add(node)
            node = node.children[vertex]

        self.dimension = max(self.dimension, len(simplex) - 1)

        self.simplex_map[frozenset(simplex)] = node
        return node

    def find(self, simplex):
        """
        Find a simplex in the tree.
        """
        if not simplex:
            return self.root
        return self.simplex_map.get(frozenset(simplex))

    def __contains__(self, simplex):
        """Checks if a given simplex exists in the tree."""
        return self.find(simplex) is not None

    def get_simplices(self, dim=None):
        """Get all simplices in the complex, optionally filtered by dimension."""
        simplices = []

        # DFS
        # Stack items are (node, path_to_node)
        stack = [(self.root, [])]

        while stack:
            node, path = stack.pop()

            # If not the root, add the current vertex to the path
            if node != self.root:
                current_simplex = path + [node.vertex]

                if dim is None or len(current_simplex) - 1 == dim:
                    simplices.append(tuple(current_simplex))
            else:
                current_simplex = path

            # Add children to the stack (in reverse order)
            for _, child in sorted(node.children.items(), reverse=True):
                stack.append((child, current_simplex.copy()))

        return simplices

    def compute_nerve(self, covers, dim=None, min_intersection=1):
        """
        Compute the nerve complex with caching for intersection checks.

        Parameters
        ----------
        covers : list
            List of cover elements. Each cover element should be a collection
            (set, list, array) of points.
        dim : int, optional
            Maximum dimension of simplices to include. If None, all possible dimensions
            are computed.
        min_intersection : int, default=1
            Minimum number of points required in the intersection of cover elements
            to create a simplex. Default is 1 (non-empty intersection).

        Notes
        -----
        - Halts when no new simplices can be added at the current dimension
        or when the maximum dimension is reached.
        """

        if not covers:
            return

        n_covers = len(covers)
        self._covers = covers

        @lru_cache(maxsize=1024)
        def has_intersection(*indices):
            if not indices or len(indices) == 1:
                return len(indices) == 1  # empty returns False, singleton returns True

            # Convert all cover elements to sets
            sets_to_intersect = [set(self._covers[i]) for i in indices]

            result = sets_to_intersect[0]
            for s in sets_to_intersect[1:]:
                result = result.intersection(s)
                if len(result) < min_intersection:
                    return False

            return True

        max_dim = min(n_covers - 1, dim if dim is not None else float("inf"))

        # Insert vertices
        for i in range(n_covers):
            self.insert([i])

        # Do edges first
        if max_dim >= 1:
            for i in range(n_covers):
                set_i = set(self._covers[i])
                for j in range(i + 1, n_covers):
                    set_j = set(self._covers[j])

                    # can now terminate early
                    count = 0
                    # go through the smaller set
                    smaller_set = set_i if len(set_i) < len(set_j) else set_j
                    larger_set = set_j if len(set_i) < len(set_j) else set_i

                    for element in smaller_set:
                        if element in larger_set:
                            count += 1
                            if count >= min_intersection:
                                self.insert([i, j])
                                break

            # If no edges were added, we're done
            if not self.get_simplices(dim=1):
                has_intersection.cache_clear()
                return

        # Handle higher dimensions
        for current_dim in range(2, max_dim + 1):
            prev_dim_simplices = self.get_simplices(dim=current_dim - 1)
            if not prev_dim_simplices:
                break

            # Group simplices by "prefix" (all vertices except the last)
            prefix_groups = {}
            for simplex in prev_dim_simplices:
                prefix = simplex[:-1]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(simplex[-1])

            # Find higher dimensional simplices
            new_simplices_added = False
            for prefix, last_vertices in prefix_groups.items():
                for i, v1 in enumerate(last_vertices):
                    for v2 in last_vertices[i + 1 :]:
                        candidate = tuple(sorted(prefix + (v1, v2)))

                        all_faces_exist = True
                        for j in range(len(candidate)):
                            face = candidate[:j] + candidate[j + 1 :]
                            if face not in self:
                                all_faces_exist = False
                                break

                        if all_faces_exist and has_intersection(*candidate):
                            self.insert(list(candidate))
                            new_simplices_added = True

            if not new_simplices_added:
                break

        # Clean up cache
        has_intersection.cache_clear()

    def __str__(self):
        """String representation of the simplex tree"""
        result = []

        def traverse(node, depth, path):
            if node != self.root:
                path.append(node.vertex)
                indent = "  " * depth
                result.append(f"{indent}{node.vertex} - {path}")

            for child in sorted(node.children.keys()):
                traverse(node.children[child], depth + 1, path.copy())

        traverse(self.root, 0, [])
        return "\n".join(result)


def get_skeleton(tree: SimplexTree, dim: int):
    """
    Get all simplices up to the specified dimension (the k-skeleton) from a SimplexTree.

    Parameters
    ----------
    tree : SimplexTree
        The simplex tree to extract the skeleton from.
    dim : int
        Maximum dimension of simplices to include in the skeleton.

    Returns
    -------
    list of tuples
        List of simplices (tuples of vertices), sorted by increasing dimension.
    """
    simplices = []

    def collect_to_dim(node, current_simplex, current_dim):
        if node != tree.root:
            current_simplex = current_simplex + [node.vertex]
            current_dim = len(current_simplex) - 1
            if current_dim <= dim:
                simplices.append(tuple(current_simplex))

        if current_dim < dim:
            for child in node.children.values():
                collect_to_dim(child, current_simplex.copy(), current_dim)

    collect_to_dim(tree.root, [], -1)
    return simplices
