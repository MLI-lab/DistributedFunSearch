"""Evaluation script using precomputed graphs (works for any ECC problem type).

PROBLEM TYPE: Generic - works with both Deletions and IDS
The edge condition is baked into the precomputed graph files.

Graph is pre-loaded by evaluator and passed via params. Uses FastGraph with
NetworkX-compatible API: G.neighbors(node), G.degree(node), G.nodes
"""

import hashlib
import math
import itertools
from collections import Counter
import numpy as np
import networkx as nx
import random


# Wrappers to make nx.function(G, ...) work with FastGraph
def _is_fastgraph(G):
    return hasattr(G, '_cpp_graph') or hasattr(G, '_degrees')

_nx_degree = nx.degree
def _degree_wrapper(G, node=None, weight=None):
    if _is_fastgraph(G):
        if node is None:
            return G.degree
        return G.degree(node)
    return _nx_degree(G, node, weight)
nx.degree = _degree_wrapper

_nx_neighbors = nx.neighbors
def _neighbors_wrapper(G, node):
    if _is_fastgraph(G):
        return G.neighbors(node)
    return _nx_neighbors(G, node)
nx.neighbors = _neighbors_wrapper

_nx_number_of_nodes = nx.number_of_nodes
def _number_of_nodes_wrapper(G):
    if _is_fastgraph(G):
        return G.number_of_nodes()
    return _nx_number_of_nodes(G)
nx.number_of_nodes = _number_of_nodes_wrapper

_nx_number_of_edges = nx.number_of_edges
def _number_of_edges_wrapper(G):
    if _is_fastgraph(G):
        return G.number_of_edges()
    return _nx_number_of_edges(G)
nx.number_of_edges = _number_of_edges_wrapper

_nx_nodes = nx.nodes
def _nodes_wrapper(G):
    if _is_fastgraph(G):
        return G.nodes
    return _nx_nodes(G)
nx.nodes = _nodes_wrapper

_nx_edges = nx.edges
def _edges_wrapper(G, nbunch=None):
    if _is_fastgraph(G):
        return G.edges
    return _nx_edges(G, nbunch)
nx.edges = _edges_wrapper

_nx_adj = getattr(nx, 'adj', None)
if _nx_adj:
    def _adj_wrapper(G):
        if _is_fastgraph(G):
            return G.adj
        return _nx_adj(G)
    nx.adj = _adj_wrapper

# nx.clustering - compute locally for FastGraph
_nx_clustering = nx.clustering
def _clustering_wrapper(G, nodes=None, weight=None):
    if _is_fastgraph(G):
        # Simple clustering coefficient: triangles / possible triangles
        def local_clustering(node):
            neighbors = list(G.neighbors(node))
            k = len(neighbors)
            if k < 2:
                return 0.0
            triangles = 0
            for i, u in enumerate(neighbors):
                for v in neighbors[i+1:]:
                    if G.has_edge(u, v):
                        triangles += 1
            return (2.0 * triangles) / (k * (k - 1))

        if nodes is None:
            return {node: local_clustering(node) for node in G.nodes}
        elif hasattr(nodes, '__iter__') and not isinstance(nodes, str):
            return {node: local_clustering(node) for node in nodes}
        else:
            return local_clustering(nodes)
    return _nx_clustering(G, nodes, weight)
nx.clustering = _clustering_wrapper


def hash_priority_mapping(priorities, nodes):
    """Generate a hash based on the mapping of nodes to their priority scores."""
    mapping_str = ','.join(f'{node}:{priorities[node]}' for node in sorted(nodes))
    return hashlib.sha256(mapping_str.encode()).hexdigest()


def evaluate(params, graph_dir):
    n, s, q, G = params
    independent_set, hash_value = solve(n, s, q, G)
    return (len(independent_set), hash_value)


def solve(n, s, q, G):
    """Find a large independent set."""
    # Seed random for deterministic evaluation
    random.seed(1)
    np.random.seed(1)

    # Compute priorities
    priorities = {node: priority(node, G, n, s) for node in G.nodes}

    # Sort nodes by priority descending, lexicographic tie breaking
    nodes_sorted = sorted(G.nodes, key=lambda x: (-priorities[x], x))

    # Greedy independent set construction
    independent_set = set()
    removed = set()
    for node in nodes_sorted:
        if node in removed:
            continue
        independent_set.add(node)
        removed.add(node)
        removed.update(G.neighbors(node))

    # Compute hash for deduplication only for smallest n
    hash_value = None
    if n == start_n:
        hash_value = hash_priority_mapping(priorities, G.nodes)

    return independent_set, hash_value


def priority(node, G, n, s) -> float:
    pass
