"""Evaluation script using precomputed graphs with native NetworkX.

Same as graph_fastgraph.py but receives a real nx.Graph instead of FastGraph.
All nx functions work natively without wrappers.
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
    # Freeze graph so LLM-generated priority() cannot mutate it
    nx.freeze(G)

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
