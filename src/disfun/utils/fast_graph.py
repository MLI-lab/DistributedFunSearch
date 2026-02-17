"""FastGraph: Memory-efficient graph with NetworkX-compatible API.

Uses C++ implementation (fast_graph_cpp) if available, falls back to pure Python.

Usage:
    G = load_graph_from_lmdb(path)
    G.neighbors(node)  # returns tuple of neighbors
    G.degree(node)     # returns int (also G.degree[node] for NetworkX compat)
    G.nodes            # returns tuple of all nodes (also G.nodes() for compat)

LLM-generated priority functions work unchanged:
    def priority(node, G, n, s):
        return G.degree(node) + sum(G.degree(nb) for nb in G.neighbors(node))
"""


class NodeView:
    """NetworkX-compatible node view supporting both G.nodes and G.nodes()."""

    __slots__ = ('_nodes',)

    def __init__(self, nodes):
        self._nodes = nodes

    def __call__(self, data=False, default=None):
        """G.nodes() or G.nodes(data=True) -> nodes with optional attributes."""
        if data:
            return [(n, {}) for n in self._nodes]
        return self._nodes

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, node):
        return node in self._nodes

    def __getitem__(self, idx):
        return self._nodes[idx]

    def keys(self):
        """G.nodes.keys() -> nodes"""
        return self._nodes

    def values(self):
        """G.nodes.values() -> empty attribute dicts"""
        return [{} for _ in self._nodes]

    def items(self):
        """G.nodes.items() -> (node, attr_dict) pairs"""
        return [(n, {}) for n in self._nodes]

    def data(self, data=True, default=None):
        """G.nodes.data() -> (node, attr_dict) pairs"""
        return [(n, {}) for n in self._nodes]

    def __repr__(self):
        return f"NodeView({self._nodes})"


class EdgesView:
    """NetworkX-compatible edges view supporting both G.edges and G.edges()."""

    __slots__ = ('_graph', '_cached_edges')

    def __init__(self, graph, cached_edges=None):
        self._graph = graph
        self._cached_edges = cached_edges  # For pure Python version

    def __call__(self, nbunch=None, data=False, default=None):
        """G.edges() -> iterator of edges (NetworkX compatibility)."""
        return iter(self)

    def __iter__(self):
        if self._cached_edges is not None:
            return iter(self._cached_edges)
        # For C++ version, generate on the fly
        seen = set()
        for node in self._graph._nodes_tuple:
            for nb in self._graph.neighbors(node):
                edge = (node, nb) if node < nb else (nb, node)
                if edge not in seen:
                    seen.add(edge)
                    yield edge

    def __len__(self):
        return self._graph.number_of_edges()

    def __contains__(self, edge):
        u, v = edge
        return self._graph.has_edge(u, v)

    def __repr__(self):
        return f"EdgesView({self._graph})"


class DegreeView:
    """NetworkX-compatible degree view supporting both G.degree(node) and G.degree[node]."""

    __slots__ = ('_graph',)

    def __init__(self, graph):
        self._graph = graph

    def __call__(self, node=None, weight=None):
        """G.degree(node) -> int, G.degree() -> dict. Weight param ignored (unweighted graph)."""
        return self._graph._degree_lookup(node)

    def __getitem__(self, node):
        """G.degree[node] -> int (NetworkX compatibility)"""
        return self._graph._degree_lookup(node)

    def __iter__(self):
        """Iterate over (node, degree) pairs."""
        for node in self._graph._nodes_tuple:
            yield node, self._graph._degree_lookup(node)

    def __len__(self):
        return self._graph.number_of_nodes()

    def get(self, node, default=None):
        """G.degree.get(node) -> degree or default (NetworkX compatibility)."""
        if node in self._graph:
            return self._graph._degree_lookup(node)
        return default

    def values(self):
        """G.degree.values() -> iterator of degrees."""
        for node in self._graph._nodes_tuple:
            yield self._graph._degree_lookup(node)

    def keys(self):
        """G.degree.keys() -> iterator of nodes."""
        return iter(self._graph._nodes_tuple)

    def items(self):
        """G.degree.items() -> iterator of (node, degree) pairs."""
        return iter(self)


class AdjacencyView:
    """NetworkX-compatible adjacency view supporting G.adj[node] and G[node]."""

    __slots__ = ('_graph',)

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, node):
        """G.adj[node] -> neighbors dict-like"""
        return {nb: {} for nb in self._graph.neighbors(node)}

    def __iter__(self):
        return iter(self._graph._nodes_tuple)

    def __len__(self):
        return self._graph.number_of_nodes()

    def keys(self):
        """G.adj.keys() -> nodes"""
        return self._graph._nodes_tuple

    def values(self):
        """G.adj.values() -> list of neighbor dicts"""
        return [self[node] for node in self._graph._nodes_tuple]

    def items(self):
        """G.adj.items() -> (node, neighbors_dict) pairs"""
        return [(node, self[node]) for node in self._graph._nodes_tuple]

    def get(self, node, default=None):
        """G.adj.get(node) -> neighbors dict or default"""
        if node in self._graph:
            return self[node]
        return default


class SubGraphView:
    """Read-only subgraph view for NetworkX compatibility.

    Wraps a FastGraphCpp and filters to only include specified nodes.
    Provides same interface as FastGraphCpp but limited to subgraph nodes.
    """

    __slots__ = ('_graph', '_nodes_set', '_nodes_tuple')

    def __init__(self, graph, nodes):
        self._graph = graph
        # Convert to set for O(1) membership, keep only nodes that exist
        if isinstance(nodes, set):
            self._nodes_set = nodes & set(graph._nodes_tuple)
        else:
            self._nodes_set = set(nodes) & set(graph._nodes_tuple)
        self._nodes_tuple = tuple(self._nodes_set)

    @property
    def nodes(self):
        """Returns NodeView for subgraph nodes."""
        return NodeView(self._nodes_tuple)

    def neighbors(self, node):
        """Return neighbors of node that are also in subgraph."""
        if node not in self._nodes_set:
            raise KeyError(node)
        return tuple(nb for nb in self._graph.neighbors(node) if nb in self._nodes_set)

    @property
    def degree(self):
        """Returns DegreeView for subgraph."""
        return DegreeView(self)

    def _degree_lookup(self, node=None):
        """Internal method for DegreeView."""
        if node is None:
            return {n: len(self.neighbors(n)) for n in self._nodes_tuple}
        return len(self.neighbors(node))

    def number_of_nodes(self):
        return len(self._nodes_tuple)

    def number_of_edges(self):
        count = 0
        for node in self._nodes_tuple:
            count += len(self.neighbors(node))
        return count // 2  # Each edge counted twice

    def has_node(self, node):
        return node in self._nodes_set

    def has_edge(self, u, v, key=None):
        """Check if edge exists. Key param ignored (not a multigraph)."""
        if u not in self._nodes_set or v not in self._nodes_set:
            return False
        return self._graph.has_edge(u, v)

    def __contains__(self, node):
        return node in self._nodes_set

    def __len__(self):
        return len(self._nodes_tuple)

    def __iter__(self):
        return iter(self._nodes_tuple)

    def __getitem__(self, node):
        """G[node] -> dict of neighbors in subgraph."""
        return {nb: {} for nb in self.neighbors(node)}

    @property
    def adj(self):
        """G.adj[node] -> neighbors (NetworkX compatibility)."""
        return AdjacencyView(self)

    def get(self, node, default=None):
        """G.get(node) -> neighbors dict or default."""
        if node in self._nodes_set:
            return {nb: {} for nb in self.neighbors(node)}
        return default

    def subgraph(self, nodes):
        """Create nested subgraph."""
        return SubGraphView(self._graph, set(nodes) & self._nodes_set)

    def nbunch_iter(self, nbunch=None):
        """Iterate over nodes in subgraph."""
        if nbunch is None:
            return iter(self._nodes_tuple)
        if nbunch in self._nodes_set:
            return iter([nbunch])
        return (n for n in nbunch if n in self._nodes_set)

    def is_directed(self):
        """Returns False (graph is undirected)."""
        return False

    def is_multigraph(self):
        """Returns False (no parallel edges)."""
        return False

    def adjacency(self):
        """G.adjacency() -> iterator of (node, neighbors_dict) pairs."""
        for node in self._nodes_tuple:
            yield node, {nb: {} for nb in self.neighbors(node)}

    @property
    def _adj(self):
        """Internal adjacency dict (NetworkX compatibility for algorithms)."""
        return AdjacencyView(self)

    def copy(self):
        """G.copy() -> self (subgraph views are read-only)."""
        return self


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

        __slots__ = ('_cpp_graph', '_degree_view', '_adj_view', '_node_view', '_edges_view', '_nodes_tuple')

        def __init__(self, cpp_graph):
            self._cpp_graph = cpp_graph
            self._nodes_tuple = cpp_graph.nodes  # Cache for internal use
            self._degree_view = DegreeView(self)
            self._adj_view = AdjacencyView(self)
            self._node_view = NodeView(self._nodes_tuple)
            self._edges_view = EdgesView(self)

        @property
        def nodes(self):
            """Returns NodeView supporting both G.nodes and G.nodes()."""
            return self._node_view

        @property
        def node(self):
            """Deprecated alias for nodes (NetworkX compatibility)."""
            return {n: {} for n in self._nodes_tuple}

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

        def has_edge(self, u, v, key=None):
            """Check if edge exists (NetworkX compatibility). Key param ignored (not a multigraph)."""
            return v in self._cpp_graph.neighbors(u)

        def order(self):
            """Number of nodes (NetworkX alias)."""
            return self._cpp_graph.number_of_nodes()

        def size(self):
            """Number of edges (NetworkX alias)."""
            return self._cpp_graph.number_of_edges()

        def is_directed(self):
            """Returns False (graph is undirected)."""
            return False

        def is_multigraph(self):
            """Returns False (no parallel edges)."""
            return False

        def adjacency(self):
            """G.adjacency() -> iterator of (node, neighbors_dict) pairs."""
            for node in self._nodes_tuple:
                yield node, {nb: {} for nb in self._cpp_graph.neighbors(node)}

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
        def _adj(self):
            """Internal adjacency dict (NetworkX compatibility for algorithms)."""
            return self._adj_view

        @property
        def edges(self):
            """Returns EdgesView supporting both G.edges and G.edges()."""
            return self._edges_view

        def greedy_independent_set(self, priorities):
            return self._cpp_graph.greedy_independent_set(priorities)

        def get(self, node, default=None):
            """G.get(node) -> neighbors dict or default (NetworkX compatibility)."""
            if node in self._cpp_graph:
                return {nb: {} for nb in self._cpp_graph.neighbors(node)}
            return default

        def subgraph(self, nodes):
            """G.subgraph(nodes) -> SubGraphView (NetworkX compatibility).

            Returns a read-only view that filters to only the specified nodes.
            """
            return SubGraphView(self, nodes)

        def nbunch_iter(self, nbunch=None):
            """G.nbunch_iter(nbunch) -> iterator over nodes (NetworkX compatibility).

            If nbunch is None, iterate over all nodes.
            If nbunch is a single node, yield just that node.
            If nbunch is iterable, yield nodes that exist in graph.
            """
            if nbunch is None:
                return iter(self._nodes_tuple)
            if nbunch in self:
                return iter([nbunch])
            return (n for n in nbunch if n in self)

        def copy(self):
            """G.copy() -> self (graphs are read-only, no need to copy)."""
            return self

        def values(self):
            """G.values() -> adjacency dicts (for LLMs that treat G as a dict)."""
            return (self[node] for node in self._nodes_tuple)

        def keys(self):
            """G.keys() -> nodes (for LLMs that treat G as a dict)."""
            return self._nodes_tuple

        def items(self):
            """G.items() -> (node, neighbors_dict) pairs (for LLMs that treat G as a dict)."""
            return ((node, self[node]) for node in self._nodes_tuple)

        @property
        def graph(self):
            """G.graph -> empty dict (graph-level attributes, NetworkX compatibility)."""
            return {}

        def reverse(self, copy=True):
            """G.reverse() -> self (undirected graph, reverse is no-op)."""
            return self

        def to_undirected(self, as_view=False):
            """G.to_undirected() -> self (already undirected)."""
            return self

        def to_directed(self, as_view=False):
            """G.to_directed() -> self (treat as directed for compatibility)."""
            return self

        def out_degree(self, nbunch=None, weight=None):
            """G.out_degree() -> same as degree for undirected graph."""
            return self._degree_view(nbunch)

        def in_degree(self, nbunch=None, weight=None):
            """G.in_degree() -> same as degree for undirected graph."""
            return self._degree_view(nbunch)

        def out_edges(self, nbunch=None, data=False, default=None):
            """G.out_edges(node) -> edges from node (same as edges for undirected)."""
            if nbunch is None:
                return self._edges_view
            if nbunch in self:
                return [(nbunch, nb) for nb in self._cpp_graph.neighbors(nbunch)]
            return [(n, nb) for n in nbunch if n in self for nb in self._cpp_graph.neighbors(n)]

        def in_edges(self, nbunch=None, data=False, default=None):
            """G.in_edges(node) -> edges to node (same as out_edges for undirected)."""
            return self.out_edges(nbunch, data, default)

        def successors(self, node):
            """G.successors(node) -> neighbors (same as neighbors for undirected)."""
            return self._cpp_graph.neighbors(node)

        def predecessors(self, node):
            """G.predecessors(node) -> neighbors (same as neighbors for undirected)."""
            return self._cpp_graph.neighbors(node)

    def load_graph_from_lmdb(graph_path: str) -> FastGraphCpp:
        """Load graph from LMDB and wrap with NetworkX-compatible API."""
        return FastGraphCpp(_load_graph_from_lmdb_cpp(graph_path))

else:
    # Pure Python fallback

    class FastGraphCpp:
        """Graph with NetworkX-like API but efficient tuple/dict storage."""

        __slots__ = ('_nodes_tuple', '_node_set', '_neighbors', '_degrees', '_num_edges', '_degree_view', '_adj_view', '_edges_tuple', '_node_view', '_edges_view')

        def __init__(self, nodes: list, edges: list):
            """Build graph from node list and edge list."""
            self._nodes_tuple = tuple(nodes)
            self._node_set = frozenset(nodes)

            neighbors = {n: [] for n in nodes}
            for u, v in edges:
                neighbors[u].append(v)
                neighbors[v].append(u)

            self._neighbors = {n: tuple(nbs) for n, nbs in neighbors.items()}
            self._degrees = {n: len(nbs) for n, nbs in self._neighbors.items()}
            self._num_edges = len(edges)
            self._edges_tuple = tuple(edges)
            self._degree_view = DegreeView(self)
            self._adj_view = AdjacencyView(self)
            self._node_view = NodeView(self._nodes_tuple)
            self._edges_view = EdgesView(self, self._edges_tuple)

        @property
        def nodes(self):
            """Returns NodeView supporting both G.nodes and G.nodes()."""
            return self._node_view

        @property
        def node(self):
            """Deprecated alias for nodes (NetworkX compatibility)."""
            return {n: {} for n in self._nodes_tuple}

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
            return len(self._nodes_tuple)

        def number_of_edges(self) -> int:
            return self._num_edges

        def has_node(self, node) -> bool:
            return node in self._node_set

        def has_edge(self, u, v, key=None) -> bool:
            """Check if edge exists (NetworkX compatibility). Key param ignored (not a multigraph)."""
            return v in self._neighbors.get(u, ())

        def order(self) -> int:
            """Number of nodes (NetworkX alias)."""
            return len(self._nodes_tuple)

        def size(self) -> int:
            """Number of edges (NetworkX alias)."""
            return self._num_edges

        def is_directed(self):
            """Returns False (graph is undirected)."""
            return False

        def is_multigraph(self):
            """Returns False (no parallel edges)."""
            return False

        def adjacency(self):
            """G.adjacency() -> iterator of (node, neighbors_dict) pairs."""
            for node in self._nodes_tuple:
                yield node, {nb: {} for nb in self._neighbors[node]}

        def __contains__(self, node) -> bool:
            return node in self._node_set

        def __len__(self) -> int:
            return len(self._nodes_tuple)

        def __iter__(self):
            return iter(self._nodes_tuple)

        def __getitem__(self, node):
            """G[node] -> dict of neighbors (NetworkX compatibility)."""
            return {nb: {} for nb in self._neighbors[node]}

        @property
        def adj(self):
            """G.adj[node] -> neighbors (NetworkX compatibility)."""
            return self._adj_view

        @property
        def _adj(self):
            """Internal adjacency dict (NetworkX compatibility for algorithms)."""
            return self._adj_view

        @property
        def edges(self):
            """Returns EdgesView supporting both G.edges and G.edges()."""
            return self._edges_view

        def greedy_independent_set(self, priorities: dict) -> list:
            """Compute greedy independent set given priority dict."""
            nodes_sorted = sorted(self._nodes_tuple, key=lambda x: (-priorities.get(x, 0), x))

            independent_set = []
            removed = set()

            for node in nodes_sorted:
                if node in removed:
                    continue
                independent_set.append(node)
                removed.add(node)
                removed.update(self._neighbors[node])

            return independent_set

        def get(self, node, default=None):
            """G.get(node) -> neighbors dict or default (NetworkX compatibility)."""
            if node in self._node_set:
                return {nb: {} for nb in self._neighbors[node]}
            return default

        def subgraph(self, nodes):
            """G.subgraph(nodes) -> SubGraphView (NetworkX compatibility).

            Returns a read-only view that filters to only the specified nodes.
            """
            return SubGraphView(self, nodes)

        def nbunch_iter(self, nbunch=None):
            """G.nbunch_iter(nbunch) -> iterator over nodes (NetworkX compatibility).

            If nbunch is None, iterate over all nodes.
            If nbunch is a single node, yield just that node.
            If nbunch is iterable, yield nodes that exist in graph.
            """
            if nbunch is None:
                return iter(self._nodes_tuple)
            if nbunch in self:
                return iter([nbunch])
            return (n for n in nbunch if n in self)

        def copy(self):
            """G.copy() -> self (graphs are read-only, no need to copy)."""
            return self

        def values(self):
            """G.values() -> adjacency dicts (for LLMs that treat G as a dict)."""
            return (self[node] for node in self._nodes_tuple)

        def keys(self):
            """G.keys() -> nodes (for LLMs that treat G as a dict)."""
            return self._nodes_tuple

        def items(self):
            """G.items() -> (node, neighbors_dict) pairs (for LLMs that treat G as a dict)."""
            return ((node, self[node]) for node in self._nodes_tuple)

        @property
        def graph(self):
            """G.graph -> empty dict (graph-level attributes, NetworkX compatibility)."""
            return {}

        def reverse(self, copy=True):
            """G.reverse() -> self (undirected graph, reverse is no-op)."""
            return self

        def to_undirected(self, as_view=False):
            """G.to_undirected() -> self (already undirected)."""
            return self

        def to_directed(self, as_view=False):
            """G.to_directed() -> self (treat as directed for compatibility)."""
            return self

        def out_degree(self, nbunch=None, weight=None):
            """G.out_degree() -> same as degree for undirected graph."""
            return self._degree_view(nbunch)

        def in_degree(self, nbunch=None, weight=None):
            """G.in_degree() -> same as degree for undirected graph."""
            return self._degree_view(nbunch)

        def out_edges(self, nbunch=None, data=False, default=None):
            """G.out_edges(node) -> edges from node (same as edges for undirected)."""
            if nbunch is None:
                return self._edges_view
            if nbunch in self:
                return [(nbunch, nb) for nb in self._neighbors[nbunch]]
            return [(n, nb) for n in nbunch if n in self for nb in self._neighbors[n]]

        def in_edges(self, nbunch=None, data=False, default=None):
            """G.in_edges(node) -> edges to node (same as out_edges for undirected)."""
            return self.out_edges(nbunch, data, default)

        def successors(self, node):
            """G.successors(node) -> neighbors (same as neighbors for undirected)."""
            return self._neighbors[node]

        def predecessors(self, node):
            """G.predecessors(node) -> neighbors (same as neighbors for undirected)."""
            return self._neighbors[node]


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
