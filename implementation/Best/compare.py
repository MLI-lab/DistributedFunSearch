"""
Finds large independent set in graph G where nodes are binary strings of length n.
Nodes in G are connected if they share a subsequence of length at least length n-s.
"""
import itertools
import hashlib
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count

def generate_graph(n, s):
    G = nx.Graph()
    sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]
    for seq in sequences:
        G.add_node(seq)
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if has_common_subsequence(sequences[i], sequences[j], n, s):
                G.add_edge(sequences[i], sequences[j])
    return G

def has_common_subsequence(seq1, seq2, n, s):
    threshold = n - s
    if threshold <= 0:
        return True
    prev = [0] * (n + 1)
    current = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                current[j] = prev[j - 1] + 1
            else:
                current[j] = max(prev[j], current[j - 1])
            if current[j] >= threshold:
                return True
        prev, current = (current, prev)
    return False

def evaluate(params):
    """
    Main evaluation function.
    Returns the independent set, its size, and the hash value (if applicable).
    """
    n, s = params
    independent_set, hash_value = solve(n, s)
    return (independent_set, len(independent_set), hash_value)

def compute_priority_for_node(args):
    """
    Compute the priority for a single node. Used for parallel processing.
    """
    node, G, n, s = args
    return (node, priority(node, G, n, s))

def solve(n, s):
    """
    Solve for the largest independent set using a priority function.
    Returns the independent set and its hash value.
    """
    G_original = generate_graph(n, s)
    G_for_priority = G_original.copy()
    G_for_independent_set = G_original.copy()
    nodes = list(G_original.nodes)
    args = [(node, G_for_priority, n, s) for node in nodes]
    with Pool(processes=50) as pool:
        priority_results = pool.map(compute_priority_for_node, args)
    priorities = dict(priority_results)
    nodes_sorted = sorted(nodes, key=lambda x: priorities[x], reverse=True)
    independent_set = set()
    for node in nodes_sorted:
        if node not in G_for_independent_set:
            continue
        independent_set.add(node)
        neighbors = list(G_for_independent_set.neighbors(node))
        G_for_independent_set.remove_node(node)
        G_for_independent_set.remove_nodes_from(neighbors)
    hash_value = None
    return (independent_set, hash_value)

def priority(node, G, n, s):
    neighbors = [neighbor for neighbor in list(nx.all_neighbors(G, node)) if len(neighbor) >= n]
    mn = min([len(neigh) for neigh in neighbors], default=-float('inf'))
    return (-1 * (mn - s) * sum([(n - i) * (i + 1) * int(bit == '1') for i, bit in enumerate(reversed(list(node)), start=0)]) + sum([len(neig) / (n * 1 / 7) for neig in neighbors]), [neighbour for neighbour in sorted(list({n for n in neighbors}), key=lambda x: -len(x))])
if __name__ == '__main__':
    params = (7, 1)
    independent_set, size, hash_value = evaluate(params)
    print(f'Independent set size: {size}')
    print(f'Independent set: {list(independent_set)}')
    print(f'Hash value: {hash_value}')