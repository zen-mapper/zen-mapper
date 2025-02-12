
# Yoinked from paper https://arxiv.org/pdf/2001.02581

class SimplexNode:
    def __init__(self, label: int=None):
        self.label = label          # Vertex label
        self.children = {}          # Dictionary of child nodes
        self.parent = None          # Pointer to parent node
        self.siblings = []          # List of nodes at the same depth


class SimplexTree:
    def __init__(self):
        self.root = SimplexNode()   # Root node is the empty face
        self.top_nodes = []         # Array of top level nodes (0-simplices)
        self.dimension = -1          # Dimension of the complex
        self.num_simplices = 1      # Number of simplices including empty face


    def _validate_simplex(self, simplex: list[int]) -> bool:
        """Check if simplex is valid (sorted + unique vertices)."""
        if not simplex:
            return True
        return all(simplex[i] < simplex[i+1] for i in range(len(simplex)-1))


    def find_simplex(self, simplex: list[int]) -> SimplexNode | None:
        """Find a simplex in the tree."""
        if not self._validate_simplex(simplex):
            raise ValueError("Simplex is not valid...")
        
        current_node = self.root
        for vertex in simplex:
            if vertex not in current_node.children:
                return None
            current_node = current_node.children[vertex]
        return current_node


    def insert_simplex(self, simplex: list[int]) -> bool:
        """
        Insert a simplex into the tree.
        Returns True if insertion was successful, False otherwise.
        """
        if not simplex:
            return False

        if not self._validate_simplex(simplex):
            raise ValueError("Simplex is not valid...")

        sorted_simplex = sorted(simplex)

        if self.find_simplex(sorted_simplex):
            return False

        # Update dimension and number of simplices once
        self.dimension = max(self.dimension, len(sorted_simplex) - 1)
        self.num_simplices += 1

        current_node = self.root
        for i, vertex in enumerate(sorted_simplex):
            if vertex not in current_node.children:
                # create a new node
                new_node = SimplexNode(label=vertex)
                new_node.parent = current_node
                current_node.children[vertex] = new_node

                # Update top nodes reference if this is a first-level vertex
                if i == 0:
                    if len(self.top_nodes) <= vertex:
                        self.top_nodes.extend(
                            [None] * (vertex - len(self.top_nodes) + 1)
                            )
                    self.top_nodes[vertex] = new_node

            current_node = current_node.children[vertex]

        return True

                


    def insert_full_simplex(self, simplex: list[int]) -> None:
        """Insert a simplex and all its faces into the tree."""
        if not simplex:
            return

        sorted_simplex = sorted(simplex)
        n = len(sorted_simplex)

        # Generate all possible faces of size 1 to n
        for size in range(1, n + 1):
            # Generate all combinations of given size
            def generate_faces(start: int, current_face: list[int]):
                if len(current_face) == size:
                    self.insert_simplex(current_face[:])
                    return

                for i in range(start, n):
                    current_face.append(sorted_simplex[i])
                    generate_faces(i + 1, current_face)
                    current_face.pop()

            generate_faces(0, [])


    def _recompute_dimension(self) -> None:
        """
        Recompute the dimension of the complex.
        The dimension is the maximum length of any simplex path minus 1.
        """
        if not self.root.children:
            self.dimension = -1 
            return

        # Find the longest path in the tree
        max_depth = -1              # Start at -1 to account for empty simplex
        stack = [(self.root, -1)]   # Start root at -1 since it's empty simplex

        while stack:
            node, depth = stack.pop()
            if node.children:  # Only update max_depth if node has children
                for child in node.children.values():
                    stack.append((child, depth + 1))
            else:  # Leaf node
                max_depth = max(max_depth, depth)

        self.dimension = max_depth


    def remove_simplex(self, simplex: list[int]) -> bool:
        """Remove a simplex from the complex while preserving its faces."""
        if not simplex:
            return False

        sorted_simplex = sorted(simplex)
        if not self.find_simplex(sorted_simplex):
            return False

        # Traverse to the node
        current = self.root
        path = []  # Keep track of path to node
        for label in sorted_simplex:
            if label not in current.children:
                return False
            path.append((current, label))
            current = current.children[label]

        # Remove only this specific simplex
        parent, last_label = path[-1]
        del parent.children[last_label]
        self.num_simplices -= 1

        # Clean up empty nodes in the path, but only if they have no other children
        # and are not part of a face of the removed simplex
        for i, (parent, label) in enumerate(reversed(path[:-1])):
            node = parent.children[label]
            if not node.children:
                # Check if this node is part of a face we want to keep
                is_face = False
                face = sorted_simplex[:-(i+2)]  # Get the face at this level
                if face:  # If it's not empty
                    # Check if this face should exist
                    face_node = self.find_simplex(face)
                    is_face = face_node is not None

                if not is_face:
                    del parent.children[label]
                    self.num_simplices -= 1

        self._recompute_dimension()
        return True


    def get_skeleton(self, k:int) -> list[list[int]]:
        """ Get all simplicies in the complex with dimension <= k."""
        skeleton = []

        def collect_simplices(node: SimplexNode, current_path: list[int]):
            if len(current_path) - 1 <= k:
                skeleton.append(tuple(current_path))
            if len(current_path) - 1 < k:
                for label, child in sorted(node.children.items()):
                    current_path.append(label)
                    collect_simplices(child, current_path)
                    current_path.pop()

        collect_simplices(self.root, [])
        return skeleton


    def locate_cofaces(self, simplex: list[int]) -> list[list[int]]:
        """Find all cofaces of given simplex."""
        if not simplex:
            return []

        cofaces = []

        def find_cofaces_recursive(
                node: SimplexNode, 
                current_path: list[int], 
                remaining: set[int]
                ):

            if not remaining:  # Found all vertices of the simplex
                cofaces.append(current_path[:])
                # Continue searching for higher dimensional cofaces
                for label, child in sorted(node.children.items()):
                    current_path.append(label)
                    find_cofaces_recursive(child, current_path, set())
                    current_path.pop()
            else:
                # Still need to find some vertices
                for label, child in sorted(node.children.items()):
                    if label in remaining:
                        current_path.append(label)
                        find_cofaces_recursive(child, current_path, remaining - {label})
                        current_path.pop()

        target_set = set(sorted(simplex))
        find_cofaces_recursive(self.root, [], target_set)
        return [sorted(coface) for coface in cofaces]


    def locate_facets(self, simplex: list[int]) -> list[list[int]]:
        """Find all facets (faces of codimension 1) of a simplex."""
        if not simplex:
            return []

        # vertices return the empty simplex as its only facet
        if len(simplex) == 1:
            return [[]]

        # generate facets by removing one vertex at a time
        facets = []
        for i in range(len(simplex)):
            facet = simplex[:i] + simplex[i+1:]  # remove i-th vertex
            facets.append(facet)

        return facets


    def elementary_collapse(self, face: list[int], coface: list[int]) -> bool:
        """ Perform elementary collapse of a free pair (face, coface)."""
        # verify simplices exist
        face_node = self.find_simplex(face)
        coface_node = self.find_simplex(coface)
        if not face_node or not coface_node:
            return False

        # verify face is a proper face of coface
        if not (set(face).issubset(set(coface)) and len(coface)==len(face) + 1):
            return False

        # get all cofaces of the face using locate_cofaces()
        face_cofaces = []

        def collect_cofaces(node: SimplexNode, current_path: list[int]):
            if len(current_path) > len(face):
                face_cofaces.append(sorted(current_path))
            for label, child in sorted(node.children.items()):
                current_path.append(label)
                collect_cofaces(child, current_path)
                current_path.pop()

        collect_cofaces(face_node, face[:])

        # σ must be the only coface of τ
        if len(face_cofaces) != 1 or face_cofaces[0] != coface:
            return False

        # According to the paper: 
        # "either the node representing τ in the simplex tree is a leaf or 
        # it has the node representing σ as its unique child"
        if len(face_node.children) > 1:
            return False

        # Remove both simplices as specified in the paper
        self.remove_simplex(coface)
        self.remove_simplex(face)
        return True

    
    def edge_contraction(self, v1: int, v2: int) -> None:
        """
        Contract the edge [v1, v2] by removing v2 and updating incident simplices.
        """
        if not self.find_simplex([min(v1, v2), max(v1, v2)]):
            raise ValueError("Edge does not exist in complex")

        # Find all simplices containing v2
        to_remove = []
        to_add = []

        def collect_simplices(node: SimplexNode, path: list[int]):
            if node.label == v2:
                simplex = path + [v2]
                to_remove.append(simplex)
                # Replace v2 with v1 and sort
                new_simplex = sorted(set(path + [v1]))
                if new_simplex not in to_add:
                    to_add.append(new_simplex)
            for child in node.children.values():
                collect_simplices(
                    node=child, 
                    path=path + [node.label] if node.label is not None else path
                    )

        collect_simplices(self.root, [])

        # Remove simplices containing v2
        for simplex in sorted(to_remove, key=len, reverse=True):
            self.remove_simplex(simplex)

        # Add new simplices with v1
        for simplex in sorted(to_add, key=len):
            if not self.find_simplex(simplex):
                self.insert_simplex(simplex)

        # Make sure to remove the edge [v1, v2]
        self.remove_simplex([min(v1, v2), max(v1, v2)])

    
    
    def get_star(self, simplex: list[int]) -> list[list[int]]:
        """Get all cofaces of a given simplex."""
        return self.locate_cofaces(simplex)

    
    def get_link(self, simplex: list[int]) -> list[list[int]]:
        """
        Get the link of a given simplex. The link is the set of all simplices 
        in a complex that are disjoint from a given simplex.
        """
        star = set(tuple(coface) for coface in self.get_star(simplex))
        link = []
        simplex_set = set(simplex)
        for coface in star:
            # A face in the link contains no vertices from the simplex
            face = [v for v in coface if v not in simplex_set]
            if face:
                link.append(face)
        return link
    
    
