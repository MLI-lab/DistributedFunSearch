"""Evaluation script for DELETIONS problem (no_graph variant).

PROBLEM TYPE: Deletions only (binary strings, LCS-based neighbor check)
DO NOT use this for IDS - use no_graph_ids.py instead.

Neighbors are defined by: lcs_length(node1, node2) >= n - s
This variant computes neighbors on the fly without loading precomputed graphs.
"""

import hashlib

# Imports available to priority function (must match imports/no_graph.txt)
import math
import itertools
from collections import Counter
import numpy as np
import random


def lcs_length(s1, s2):
    """Compute longest common subsequence length using dynamic programming."""
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = 0
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev

    return prev[n]


def are_neighbors(node1, node2, n, s):
    """Check if two nodes are neighbors (share subsequence of length >= n minus s)."""
    return lcs_length(node1, node2) >= n - s


def hash_priority_mapping(priorities, nodes):
    """Generate a hash based on the mapping of nodes to their priority scores."""
    mapping_str = ','.join(f'{node}:{priorities[node]}' for node in sorted(nodes))
    return hashlib.sha256(mapping_str.encode()).hexdigest()


def evaluate(params, graph_dir):
    n, s, q = params
    independent_set, hash_value = solve(n, s, q, graph_dir)
    return (len(independent_set), hash_value)


def solve(n, s, q, graph_dir):
    """Find a large independent set, computes neighbors on the fly without loading graph files."""
    # Generate all q ary strings of length n
    nodes = [''.join(seq) for seq in itertools.product(map(str, range(q)), repeat=n)]

    # Seed random for deterministic evaluation (same code always gets same score)
    random.seed(1)
    np.random.seed(1)

    # Calculate priorities based only on node string properties (no graph passed)
    priorities = {node: priority(codeword=node, n=n, s=s) for node in nodes}

    # Sort nodes by priority (descending), lexicographic tie breaking
    nodes_sorted = sorted(nodes, key=lambda x: (-priorities[x], x))

    # Greedy independent set construction
    independent_set = set()
    removed = set()
    for node in nodes_sorted:
        if node in removed:
            continue
        independent_set.add(node)
        removed.add(node)
        # Remove all neighbors (computed on the fly)
        for other in nodes:
            if other not in removed and are_neighbors(node, other, n, s):
                removed.add(other)

    # Compute hash for deduplication (only for smallest n)
    # Note: "n == start_n" gets replaced with actual value (e.g. "n == 6") at runtime by __main__.py
    hash_value = None
    if n == start_n:
        hash_value = hash_priority_mapping(priorities, nodes)

    return independent_set, hash_value


def priority(codeword, n, s) -> float:
    pass
