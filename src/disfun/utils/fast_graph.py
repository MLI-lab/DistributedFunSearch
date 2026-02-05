"""FastGraph: Memory-efficient graph with NetworkX-compatible API.

Uses C++ implementation (fast_graph_cpp) if available, falls back to pure Python.

Usage:
    G = load_graph_from_lmdb(path)
    G.neighbors(node)  # returns tuple of neighbors
    G.degree(node)     # returns int (also G.degree[node] for NetworkX compat)
    G.nodes            # returns tuple of all nodes

LLM-generated priority functions work unchanged:
    def priority(node, G, n, s):
        return G.degree(node) + sum(G.degree(nb) for nb in G.neighbors(node))
"""


class DegreeView:
    """NetworkX-compatible degree view supporting both G.degree(node) and G.degree[node]."""

    __slots__ = ('_graph',)

    def __init__(self, graph):
        self._graph = graph

    def __call__(self, node=None):
        """G.degree(node) -> int, G.degree() -> dict"""
        return self._graph._degree_lookup(node)

    def __getitem__(self, node):
        """G.degree[node] -> int (NetworkX compatibility)"""
        return self._graph._degree_lookup(node)

    def __iter__(self):
        """Iterate over (node, degree) pairs."""
        for node in self._graph.nodes:
            yield node, self._graph._degree_lookup(node)

    def __len__(self):
        return self._graph.number_of_nodes()


class AdjacencyView:
    """NetworkX-compatible adjacency view supporting G.adj[node] and G[node]."""

    __slots__ = ('_graph',)

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, node):
        """G.adj[node] -> neighbors dict-like"""
        return {nb: {} for nb in self._graph.neighbors(node)}

    def __iter__(self):
        return iter(self._graph.nodes)

    def __len__(self):
        return self._graph.number_of_nodes()


# Try C++ implementation first
# Supports FASTGRAPH_CPP_PATH env var for architecture-specific builds (e.g., HPC hetjobs)
import os
import importlib.util

def _load_cpp_module():
    """Load fast_graph_cpp from custom path or default location."""
    custom_path = os.environ.get('FASTGRAPH_CPP_PATH')
    if custom_path:
        # Load from custom path (for HPC with different CPU architectures)
        import glob
        so_files = glob.glob(os.path.join(custom_path, 'fast_graph_cpp*.so'))
        if so_files:
            spec = importlib.util.spec_from_file_location('fast_graph_cpp', so_files[0])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    # Default: import from package
    from disfun.utils import fast_graph_cpp
    return fast_graph_cpp

try:
    _cpp_module = _load_cpp_module()
    _FastGraphCppBase = _cpp_module.FastGraphCpp
    _load_graph_from_lmdb_cpp = _cpp_module.load_graph_from_lmdb
    USING_CPP = True
except (ImportError, AttributeError, FileNotFoundError):
    USING_CPP = False


if USING_CPP:
    # Wrapper around C++ class to add NetworkX-compatible degree property
    class FastGraphCpp:
        """Wrapper around C++ FastGraph with NetworkX-compatible G.degree[node] support."""

        __slots__ = ('_cpp_graph', '_degree_view', '_adj_view')

        def __init__(self, cpp_graph):
            self._cpp_graph = cpp_graph
            self._degree_view = DegreeView(self)
            self._adj_view = AdjacencyView(self)

        @property
        def nodes(self):
            return self._cpp_graph.nodes

        def neighbors(self, node):
            return self._cpp_graph.neighbors(node)

        @property
        def degree(self):
            """Returns DegreeView supporting both G.degree(node) and G.degree[node]."""
            return self._degree_view

        def _degree_lookup(self, node=None):
            """Internal method for DegreeView."""
            return self._cpp_graph.degree(node)

        def number_of_nodes(self):
            return self._cpp_graph.number_of_nodes()

        def number_of_edges(self):
            return self._cpp_graph.number_of_edges()

        def has_node(self, node):
            return self._cpp_graph.has_node(node)

        def has_edge(self, u, v):
            """Check if edge exists (NetworkX compatibility)."""
            return v in self._cpp_graph.neighbors(u)

        def order(self):
            """Number of nodes (NetworkX alias)."""
            return self._cpp_graph.number_of_nodes()

        def size(self):
            """Number of edges (NetworkX alias)."""
            return self._cpp_graph.number_of_edges()

        def __contains__(self, node):
            return node in self._cpp_graph

        def __len__(self):
            return len(self._cpp_graph)

        def __iter__(self):
            return iter(self._cpp_graph)

        def __getitem__(self, node):
            """G[node] -> dict of neighbors (NetworkX compatibility)."""
            return {nb: {} for nb in self._cpp_graph.neighbors(node)}

        @property
        def adj(self):
            """G.adj[node] -> neighbors (NetworkX compatibility)."""
            return self._adj_view

        @property
        def edges(self):
            """Iterate over edges as (u, v) tuples."""
            seen = set()
            for node in self._cpp_graph.nodes:
                for nb in self._cpp_graph.neighbors(node):
                    edge = (node, nb) if node < nb else (nb, node)
                    if edge not in seen:
                        seen.add(edge)
                        yield edge

        def greedy_independent_set(self, priorities):
            return self._cpp_graph.greedy_independent_set(priorities)

    def load_graph_from_lmdb(graph_path: str) -> FastGraphCpp:
        """Load graph from LMDB and wrap with NetworkX-compatible API."""
        return FastGraphCpp(_load_graph_from_lmdb_cpp(graph_path))

else:
    # Pure Python fallback

    class FastGraphCpp:
        """Graph with NetworkX-like API but efficient tuple/dict storage."""

        __slots__ = ('_nodes', '_node_set', '_neighbors', '_degrees', '_num_edges', '_degree_view', '_adj_view', '_edges')

        def __init__(self, nodes: list, edges: list):
            """Build graph from node list and edge list."""
            self._nodes = tuple(nodes)
            self._node_set = frozenset(nodes)

            neighbors = {n: [] for n in nodes}
            for u, v in edges:
                neighbors[u].append(v)
                neighbors[v].append(u)

            self._neighbors = {n: tuple(nbs) for n, nbs in neighbors.items()}
            self._degrees = {n: len(nbs) for n, nbs in self._neighbors.items()}
            self._num_edges = len(edges)
            self._edges = tuple(edges)
            self._degree_view = DegreeView(self)
            self._adj_view = AdjacencyView(self)

        @property
        def nodes(self):
            return self._nodes

        def neighbors(self, node) -> tuple:
            return self._neighbors[node]

        @property
        def degree(self):
            """Returns DegreeView supporting both G.degree(node) and G.degree[node]."""
            return self._degree_view

        def _degree_lookup(self, node=None):
            """Internal method for DegreeView."""
            if node is None:
                return self._degrees
            return self._degrees[node]

        def number_of_nodes(self) -> int:
            return len(self._nodes)

        def number_of_edges(self) -> int:
            return self._num_edges

        def has_node(self, node) -> bool:
            return node in self._node_set

        def has_edge(self, u, v) -> bool:
            """Check if edge exists (NetworkX compatibility)."""
            return v in self._neighbors.get(u, ())

        def order(self) -> int:
            """Number of nodes (NetworkX alias)."""
            return len(self._nodes)

        def size(self) -> int:
            """Number of edges (NetworkX alias)."""
            return self._num_edges

        def __contains__(self, node) -> bool:
            return node in self._node_set

        def __len__(self) -> int:
            return len(self._nodes)

        def __iter__(self):
            return iter(self._nodes)

        def __getitem__(self, node):
            """G[node] -> dict of neighbors (NetworkX compatibility)."""
            return {nb: {} for nb in self._neighbors[node]}

        @property
        def adj(self):
            """G.adj[node] -> neighbors (NetworkX compatibility)."""
            return self._adj_view

        @property
        def edges(self):
            """Return edges as tuple."""
            return self._edges

        def greedy_independent_set(self, priorities: dict) -> list:
            """Compute greedy independent set given priority dict."""
            nodes_sorted = sorted(self._nodes, key=lambda x: (-priorities.get(x, 0), x))

            independent_set = []
            removed = set()

            for node in nodes_sorted:
                if node in removed:
                    continue
                independent_set.append(node)
                removed.add(node)
                removed.update(self._neighbors[node])

            return independent_set


    def load_graph_from_lmdb(graph_path: str) -> FastGraphCpp:
        """Load graph from LMDB database."""
        import lmdb
        import ujson

        env = lmdb.open(graph_path, readonly=True, lock=False, readahead=True, max_readers=126)

        nodes = []
        edges = []

        with env.begin(buffers=True) as txn:
            for key, value in txn.cursor():
                node = bytes(key).decode()
                nodes.append(node)
                for neighbor in ujson.loads(bytes(value).decode()):
                    if node < neighbor:
                        edges.append((node, neighbor))

        env.close()

        return FastGraphCpp(nodes, edges)


# Alias for backwards compatibility
FastGraph = FastGraphCpp
