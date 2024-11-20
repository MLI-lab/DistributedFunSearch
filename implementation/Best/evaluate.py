import itertools
import numpy as np
import networkx as nx
from multiprocessing import Pool
import time

def edge_task(args):
    """Task for determining if an edge exists between two nodes."""
    seq1, seq2, n, s = args
    if has_common_subsequence(seq1, seq2, n, s):
        return (seq1, seq2)
    return None

def priority_task(args):
    """Task for calculating priority of a node."""
    node, G, n, s = args
    return (node, priority(node, G, n, s))

def generate_graph(n, s, num_workers=4, chunk_size=1000):
    G = nx.Graph()
    sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]
    for seq in sequences:
        G.add_node(seq)
    total_combinations = len(sequences) * (len(sequences) - 1) // 2
    edge_args = ((sequences[i], sequences[j], n, s) for i in range(len(sequences)) for j in range(i + 1, len(sequences)))
    edges = []
    with Pool(num_workers) as pool:
        while True:
            chunk = list(itertools.islice(edge_args, chunk_size))
            if not chunk:
                break
            edges.extend(pool.map(edge_task, chunk))
    for edge in edges:
        if edge:
            G.add_edge(*edge)
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

def solve(n, s, num_workers=4, chunk_size=1000):
    G_original = generate_graph(n, s, num_workers, chunk_size)
    G_for_priority = G_original.copy()
    G_for_independent_set = G_original.copy()
    with Pool(num_workers) as pool:
        priorities = dict(pool.map(priority_task, [(node, G_for_priority, n, s) for node in G_original.nodes]))
    nodes_sorted = sorted(G_original.nodes, key=lambda x: priorities[x], reverse=True)
    independent_set = set()
    for node in nodes_sorted:
        if node not in G_for_independent_set:
            continue
        independent_set.add(node)
        neighbors = list(G_for_independent_set.neighbors(node))
        G_for_independent_set.remove_node(node)
        G_for_independent_set.remove_nodes_from(neighbors)
    return len(independent_set)

def priority(node, G, n, s):
    neighbors = [neighbor for neighbor in list(nx.all_neighbors(G, node)) if len(neighbor) == n]
    max_neighbor_length = min([len(neigh) for neigh in neighbors], default=-float('inf'))
    if max_neighbor_length > s:
        return -(max_neighbor_length - s) * sum([(n - i) * (i + 1) * int(bit == '1') for bit, i in zip(reversed(list(node)), range(len(node)))]) + sum([len(neigh) / (n * (1 / 7)) for neigh in neighbors])
num_workers = 20
chunk_size = 1000
for n in range(14, 17):
    start_time = time.time()
    size = solve(n, 1, num_workers, chunk_size)
    elapsed_time = time.time() - start_time
    print(f'Independent set size for N={n}, s=1: {size} (computed in {elapsed_time:.2f} seconds)')