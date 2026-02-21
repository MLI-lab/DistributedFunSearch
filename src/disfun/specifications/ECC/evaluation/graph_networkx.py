"""Evaluation script using precomputed graphs (works for any ECC problem type).

Graph is pre-loaded by evaluator and passed via params. Uses FastGraph with
NetworkX-compatible API: G.neighbors(node), G.degree(node), G.nodes
"""

import hashlib
import math
from math import *  # Make log, exp, sqrt, etc. available directly for LLM code
import itertools
from itertools import combinations, product, permutations
from collections import Counter
import numpy as np
import networkx as nx
import random

# Common aliases LLMs use
mean = np.mean
inf = float('inf')


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
nx.algorithms.cluster.clustering = _clustering_wrapper  # LLMs sometimes use fully-qualified path

# nx.all_neighbors - trivial for undirected graph
_nx_all_neighbors = nx.all_neighbors
def _all_neighbors_wrapper(G, node):
    if _is_fastgraph(G):
        return iter(G.neighbors(node))
    return _nx_all_neighbors(G, node)
nx.all_neighbors = _all_neighbors_wrapper

# nx.non_neighbors
_nx_non_neighbors = nx.non_neighbors
def _non_neighbors_wrapper(G, node):
    if _is_fastgraph(G):
        nbrs = set(G.neighbors(node))
        nbrs.add(node)
        return (n for n in G.nodes if n not in nbrs)
    return _nx_non_neighbors(G, node)
nx.non_neighbors = _non_neighbors_wrapper

# nx.common_neighbors
_nx_common_neighbors = nx.common_neighbors
def _common_neighbors_wrapper(G, u, v):
    if _is_fastgraph(G):
        u_nbrs = set(G.neighbors(u))
        v_nbrs = set(G.neighbors(v))
        return iter(u_nbrs & v_nbrs)
    return _nx_common_neighbors(G, u, v)
nx.common_neighbors = _common_neighbors_wrapper

# nx.degree_centrality - simple: deg / (n-1)
_nx_degree_centrality = nx.degree_centrality
def _degree_centrality_wrapper(G):
    if _is_fastgraph(G):
        n = G.number_of_nodes()
        if n <= 1:
            return {node: 0.0 for node in G.nodes}
        return {node: G.degree(node) / (n - 1) for node in G.nodes}
    return _nx_degree_centrality(G)
nx.degree_centrality = _degree_centrality_wrapper
nx.algorithms.centrality.degree_centrality = _degree_centrality_wrapper  # LLMs sometimes use fully-qualified path

# nx.density
_nx_density = nx.density
def _density_wrapper(G):
    if _is_fastgraph(G):
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0
        return 2.0 * G.number_of_edges() / (n * (n - 1))
    return _nx_density(G)
nx.density = _density_wrapper

# Expensive centrality measures — build a real nx.Graph on demand, cache it
def _to_nx_graph(G):
    """Convert FastGraph to a real nx.Graph for algorithms that need it."""
    nxG = nx.Graph()
    nxG.add_nodes_from(G.nodes)
    nxG.add_edges_from(G.edges)
    return nxG

_nx_closeness_centrality = nx.closeness_centrality
def _closeness_centrality_wrapper(G, u=None, distance=None, wf_improved=True):
    if _is_fastgraph(G):
        return _nx_closeness_centrality(_to_nx_graph(G), u, distance, wf_improved)
    return _nx_closeness_centrality(G, u, distance, wf_improved)
nx.closeness_centrality = _closeness_centrality_wrapper
nx.algorithms.closeness_centrality = _closeness_centrality_wrapper  # LLMs sometimes use fully-qualified path

_nx_betweenness_centrality = nx.betweenness_centrality
def _betweenness_centrality_wrapper(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    if _is_fastgraph(G):
        return _nx_betweenness_centrality(_to_nx_graph(G), k, normalized, weight, endpoints, seed)
    return _nx_betweenness_centrality(G, k, normalized, weight, endpoints, seed)
nx.betweenness_centrality = _betweenness_centrality_wrapper

_nx_eigenvector_centrality = nx.eigenvector_centrality
def _eigenvector_centrality_wrapper(G, max_iter=100, tol=1e-06, nstart=None, weight=None):
    if _is_fastgraph(G):
        return _nx_eigenvector_centrality(_to_nx_graph(G), max_iter, tol, nstart, weight)
    return _nx_eigenvector_centrality(G, max_iter, tol, nstart, weight)
nx.eigenvector_centrality = _eigenvector_centrality_wrapper

_nx_eigenvector_centrality_numpy = getattr(nx, 'eigenvector_centrality_numpy', None)
if _nx_eigenvector_centrality_numpy:
    def _eigenvector_centrality_numpy_wrapper(G, weight=None, max_iter=50, tol=0):
        if _is_fastgraph(G):
            return _nx_eigenvector_centrality_numpy(_to_nx_graph(G), weight, max_iter, tol)
        return _nx_eigenvector_centrality_numpy(G, weight, max_iter, tol)
    nx.eigenvector_centrality_numpy = _eigenvector_centrality_numpy_wrapper

_nx_pagerank = nx.pagerank
def _pagerank_wrapper(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None):
    if _is_fastgraph(G):
        return _nx_pagerank(_to_nx_graph(G), alpha, personalization, max_iter, tol, nstart, weight, dangling)
    return _nx_pagerank(G, alpha, personalization, max_iter, tol, nstart, weight, dangling)
nx.pagerank = _pagerank_wrapper

# nx.shortest_path_length — used occasionally
_nx_shortest_path_length = nx.shortest_path_length
def _shortest_path_length_wrapper(G, source=None, target=None, weight=None, method='dijkstra'):
    if _is_fastgraph(G):
        return _nx_shortest_path_length(_to_nx_graph(G), source, target, weight, method)
    return _nx_shortest_path_length(G, source, target, weight, method)
nx.shortest_path_length = _shortest_path_length_wrapper

_nx_shortest_path = nx.shortest_path
def _shortest_path_wrapper(G, source=None, target=None, weight=None, method='dijkstra'):
    if _is_fastgraph(G):
        return _nx_shortest_path(_to_nx_graph(G), source, target, weight, method)
    return _nx_shortest_path(G, source, target, weight, method)
nx.shortest_path = _shortest_path_wrapper

# nx.connected_components
_nx_connected_components = nx.connected_components
def _connected_components_wrapper(G):
    if _is_fastgraph(G):
        return _nx_connected_components(_to_nx_graph(G))
    return _nx_connected_components(G)
nx.connected_components = _connected_components_wrapper

# nx.average_shortest_path_length
_nx_average_shortest_path_length = nx.average_shortest_path_length
def _average_shortest_path_length_wrapper(G, weight=None, method=None):
    if _is_fastgraph(G):
        return _nx_average_shortest_path_length(_to_nx_graph(G), weight, method)
    return _nx_average_shortest_path_length(G, weight, method)
nx.average_shortest_path_length = _average_shortest_path_length_wrapper


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
