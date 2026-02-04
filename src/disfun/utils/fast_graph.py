"""FastGraph: Memory-efficient graph with NetworkX-compatible API.

Uses C++ implementation (fast_graph_cpp) if available, falls back to pure Python.

Usage:
    G = load_graph_from_lmdb(path)
    G.neighbors(node)  # returns tuple of neighbors
    G.degree(node)     # returns int
    G.nodes            # returns tuple of all nodes

LLM-generated priority functions work unchanged:
    def priority(node, G, n, s):
        return G.degree(node) + sum(G.degree(nb) for nb in G.neighbors(node))
"""

# Try C++ implementation first
try:
    from disfun.utils.fast_graph_cpp import FastGraphCpp, load_graph_from_lmdb
    USING_CPP = True
except ImportError:
    USING_CPP = False


if not USING_CPP:
    # Pure Python fallback

    class FastGraphCpp:
        """Graph with NetworkX-like API but efficient tuple/dict storage."""

        __slots__ = ('_nodes', '_node_set', '_neighbors', '_degrees', '_num_edges')

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

        @property
        def nodes(self):
            return self._nodes

        def neighbors(self, node) -> tuple:
            return self._neighbors[node]

        def degree(self, node=None):
            if node is None:
                return self._degrees
            return self._degrees[node]

        def number_of_nodes(self) -> int:
            return len(self._nodes)

        def number_of_edges(self) -> int:
            return self._num_edges

        def has_node(self, node) -> bool:
            return node in self._node_set

        def __contains__(self, node) -> bool:
            return node in self._node_set

        def __len__(self) -> int:
            return len(self._nodes)

        def __iter__(self):
            return iter(self._nodes)

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
