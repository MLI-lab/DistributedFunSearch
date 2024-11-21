"""
Finds large independent set in graph G where nodes are binary strings of length n.
Nodes in G are connected if they share a subsequence of length at least length n-s. 
"""
import itertools
import hashlib
import numpy as np
import networkx as nx

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

def hash_priority_scores(scores):
    """Generate a hash from the priority scores."""
    score_str = ','.join(map(str, scores))
    return hashlib.sha256(score_str.encode()).hexdigest()

def evaluate(params):
    n, s = params
    independent_set, hash_value = solve(n, s)
    return (len(independent_set), hash_value)

def solve(n, s):
    G_original = generate_graph(n, s)
    G_for_priority = G_original.copy()
    G_for_independent_set = G_original.copy()
    priorities = {node: priority(node, G_for_priority, n, s) for node in G_original.nodes}
    nodes_sorted = sorted(G_original.nodes, key=lambda x: priorities[x], reverse=True)
    independent_set = set()
    for node in nodes_sorted:
        if node not in G_for_independent_set:
            continue
        independent_set.add(node)
        neighbors = list(G_for_independent_set.neighbors(node))
        G_for_independent_set.remove_node(node)
        G_for_independent_set.remove_nodes_from(neighbors)
    hash_value = None
    if n == 6:
        test_sequences = [''.join(seq) for seq in itertools.product('01', repeat=6)]
        priority_scores = [priorities[seq] for seq in test_sequences]
        print(priority_scores)
        hash_value = hash_priority_scores(priority_scores)
    return (independent_set, hash_value)

def priority(node, G, n, s):
    neighbors = [len(neigh) - 1 for neigh in nx.all_neighbors(G, node)]
    M = max(*neighbors, -float('Inf'))
    if M >= s:
        term = [(M - s) * ((n - i) * (i + 1) * int(j == '1')) for i, j in enumerate(reversed(list(node)))]
        P = -(sum(term) / (1 + s + s ** 2 + ((n - s) // 3) ** 3)) + sum(neighbors) / 60
    elif any([l <= len(node) - 2 for l in neighbors]):
        P = None
    else:
        P = len(node) / 15 + np.random.randint(1, 10 ** 7, dtype='uint8').item() / 10 ** 7
    return P
if __name__ == '__main__':
    params = (6, 1)
    result = evaluate(params)
    print(result)