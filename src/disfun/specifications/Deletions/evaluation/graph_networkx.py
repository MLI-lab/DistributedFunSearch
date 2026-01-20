"""Evaluation script for Deletions problem using NetworkX.

This is the simpler version for LLMs as NetworkX allows direct node access without conversion.
Slower than graph tool, but easier for LLMs to use correctly.
"""

import os
import hashlib
import ujson as json
import lmdb

# Imports available to priority function (must match imports/networkx.txt)
import networkx as nx
import math
import itertools
from collections import Counter


def load_graph(graph_db_path):
    """Load graph from LMDB database into NetworkX Graph."""
    # max_readers is concurrent read slots for parallel evaluators
    env = lmdb.open(graph_db_path, readonly=True, lock=False, readahead=True, max_readers=126)

    edges = []
    nodes = []

    with env.begin(buffers=True) as txn:
        for key, value in txn.cursor():
            node = bytes(key).decode()
            nodes.append(node)
            for neighbor in json.loads(bytes(value).decode()):
                if node < neighbor:  # Add each edge only once
                    edges.append((node, neighbor))

    env.close()

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def hash_priority_mapping(priorities, nodes):
    """Generate a hash based on the mapping of nodes to their priority scores."""
    mapping_str = ','.join(f'{node}:{priorities[node]}' for node in sorted(nodes))
    return hashlib.sha256(mapping_str.encode()).hexdigest()


def evaluate(params, graph_dir):
    n, s, q = params
    independent_set, hash_value = solve(n, s, q, graph_dir)
    return (len(independent_set), hash_value)


def solve(n, s, q, graph_dir):
    """Find a large independent set using NetworkX."""
    path = os.path.join(graph_dir, f"graph_d_s{s}_n{n}_q{q}.lmdb")
    G = load_graph(path)

    # Freeze graph to protect against LLM modifications (instant, no copy needed).
    # If LLM tries to modify G, it raises NetworkXError and evaluation fails.
    nx.freeze(G)

    # Compute priorities (G is frozen, read only)
    priorities = {node: priority(node, G, n, s) for node in G.nodes}

    # Sort nodes by priority (descending), lexicographic tie breaking
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

    # Compute hash for deduplication (only for smallest n)
    # Note: "n == start_n" gets replaced with actual value (e.g. "n == 6") at runtime by __main__.py
    hash_value = None
    if n == start_n:
        hash_value = hash_priority_mapping(priorities, G.nodes)

    return independent_set, hash_value


def priority(node, G, n, s) -> float:
    pass
