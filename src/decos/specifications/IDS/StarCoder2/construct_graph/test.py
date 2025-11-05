"""
Quick test script for IDS construct_graph specification.
Tests that the graph generation and independent set finding work correctly.
"""

import itertools
import hashlib
import numpy as np
import networkx as nx
import json
import Levenshtein


def generate_graph(n, s):
    """
    Generate a graph where nodes are binary strings of length n.
    Two nodes are connected if edit_distance(node1, node2) < 2s + 1.
    """
    G = nx.Graph()
    sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]

    # Add all nodes
    for seq in sequences:
        G.add_node(seq)

    # Add edges between nodes with insufficient edit distance
    threshold = 2 * s + 1
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if Levenshtein.distance(sequences[i], sequences[j]) < threshold:
                G.add_edge(sequences[i], sequences[j])

    return G


def hash_priority_mapping(priorities, sequences):
    """
    Generate a hash based on the mapping of sequences to their priority scores.
    """
    mapping = [(seq, priorities[seq]) for seq in sequences]
    mapping_sorted = sorted(mapping, key=lambda x: x[0])  # Sort by sequence
    mapping_str = ','.join(f'{seq}:{score}' for seq, score in mapping_sorted)
    return hashlib.sha256(mapping_str.encode()).hexdigest()


def evaluate(params):
    n, s = params
    independent_set, hash_value = solve(n, s)
    return (len(independent_set), hash_value)


def solve(n, s):
    G_original = generate_graph(n, s)
    G_for_priority = G_original.copy()  # Pass copy to priority function
    sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]
    priorities = {node: priority(node, G_for_priority, n, s) for node in G_original.nodes}
    nodes_sorted = sorted(G_original.nodes, key=lambda x: (-priorities[x], x))
    independent_set = set()
    for node in nodes_sorted:
        if node not in G_original:
            continue
        independent_set.add(node)
        neighbors = list(G_original.neighbors(node))
        G_original.remove_node(node)
        G_original.remove_nodes_from(neighbors)
    hash_value = None
    if n == 6:
        hash_value = hash_priority_mapping(priorities, sequences)
    return independent_set, hash_value


def priority(node, G, n, s):
    """
    Returns the priority with which we want to add `node` to independent set.
    """
    return 0.0


# Test the specification
if __name__ == "__main__":
    print("Testing IDS construct_graph specification...")
    print()

    # Test case 1: n=6, s=1 (requires min distance 3)
    params = (6, 1)
    result = evaluate(params)
    print(f"Test 1: n={params[0]}, s={params[1]} (min distance required: {2*params[1]+1})")
    print(f"  Independent set size: {result[0]}")
    print(f"  Hash value: {result[1]}")
    print()

    # Verify the independent set is valid
    n, s = params
    independent_set, _ = solve(n, s)
    print(f"  Verifying independent set validity...")
    min_distance = float('inf')
    for seq1 in independent_set:
        for seq2 in independent_set:
            if seq1 != seq2:
                dist = Levenshtein.distance(seq1, seq2)
                min_distance = min(min_distance, dist)

    required_distance = 2 * s + 1
    print(f"  Minimum pairwise distance in set: {min_distance}")
    print(f"  Required distance: {required_distance}")
    if min_distance >= required_distance:
        print(f"   Valid code (can correct {s} errors)")
    else:
        print(f"   Invalid code (cannot correct {s} errors)")
    print()

    # Test case 2: n=7, s=1
    params = (7, 1)
    result = evaluate(params)
    print(f"Test 2: n={params[0]}, s={params[1]} (min distance required: {2*params[1]+1})")
    print(f"  Independent set size: {result[0]}")
    print()

    print("Tests completed!")
