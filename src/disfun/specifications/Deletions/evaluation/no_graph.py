"""Evaluation script for no_graph variant, priority function receives only node - n - s.

This variant computes neighbors on-the-fly without loading pre-computed graphs from disk.
Useful when graph loading is slow or graph files are not available.
"""

import sys
import itertools
import hashlib


def lcs_length(s1, s2):
    """Compute longest common subsequence length using dynamic programming."""
    m, n = len(s1), len(s2)
    # Use only two rows for space optimization
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = 0  # Reset first column for this row (base case)
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev  # Swap rows

    return prev[n]


def are_neighbors(node1, node2, n, s):
    """Check if two nodes are neighbors (share subsequence of length >= n-s)."""
    return lcs_length(node1, node2) >= n - s


def priority(node, n, s, q) -> float:
    pass


def hash_priority_mapping(priorities, sequences):
    """Generate a hash based on the mapping of sequences to their priority scores."""
    mapping = [(seq, priorities[seq]) for seq in sequences]
    mapping_sorted = sorted(mapping, key=lambda x: x[0])
    mapping_str = ','.join(f'{seq}:{score}' for seq, score in mapping_sorted)
    return hashlib.sha256(mapping_str.encode()).hexdigest()


def evaluate(params, graph_dir):
    n, s, q = params
    independent_set, hash_value = solve(n, s, q, graph_dir)
    return (len(independent_set), hash_value)


def solve(n, s, q, graph_dir):
    """Find a large independent set, computes neighbors on-the-fly without loading graph files."""
    print(f"Computing independent set on-the-fly for n={n}, s={s} (no graph loading)", file=sys.stderr)

    # Generate all q-ary strings of length n
    nodes = [''.join(seq) for seq in itertools.product(map(str, range(q)), repeat=n)]

    # Calculate priorities based only on node string properties (no graph passed)
    priorities = {node: priority(node, n, s, q) for node in nodes}

    # Sort nodes by priority (descending), Lexicographic tie-breaking (the second element x is the node string)
    nodes_sorted = sorted(nodes, key=lambda x: (-priorities[x], x))

    # Greedy algorithm: build independent set by checking neighbors on-the-fly
    independent_set = set()
    removed_nodes = set()

    for node in nodes_sorted:
        if node in removed_nodes:
            continue

        # Add node to independent set
        independent_set.add(node)
        removed_nodes.add(node)

        # Remove all neighbors (computed on-the-fly)
        for other_node in nodes:
            if other_node not in removed_nodes:
                if are_neighbors(node, other_node, n, s):
                    removed_nodes.add(other_node)

    # Compute hash for deduplication (only for start_n)
    # Note: "n == start_n" gets replaced with actual value (e.g. "n == 7") at runtime by __main__.py
    hash_value = None
    if n == start_n:
        hash_value = hash_priority_mapping(priorities, nodes)

    return independent_set, hash_value
