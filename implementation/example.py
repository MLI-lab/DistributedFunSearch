"""
Finds large independent set in graph G where nodes are binary strings of length n.
Nodes in G are connected if they share a subsequence of length at least n-s.
"""
import itertools
import numpy as np
import networkx as nx

def priority_v0(node, G, n, s):
    """    
    Returns the priority with which we want to add `node` to the independent set.
    """
    neighbors = list(G.neighbors(node))
    deg_surr = sum([G.degree(neigh) for neigh in neighbors])
    score = node.count('01') + (deg_surr - len(neighbors)) / 2.0
    
    # We put some weight on high-scoring neighboring nodes because an order-2 or higher match will involve their presence too!
    weighted_deg_surr = [s * G.degree(edge_i) for edge_i in neighbors]
    return float(np.sum(weighted_deg_surr))

def priority_v1(node, G, n, s):
    """Improved version of `priority_v0`."""
    edges_with_node_as_source_or_target = [(u, v) for u, v in G.edges() if (u == node or v == node)]
    # print(edges_with_node_as_source_or_target)
    degrees_connected_nodes = map(lambda x: G.degree(x[0]) + G.degree(x[1]), edges_with_node_as_source_or_target)
    degree_sum_weighted = dict(zip(edges_with_node_as_source_or_target, degrees_connected_nodes))
    return max(degree_sum_weighted.values())    

def priority_v2(node, G, n, s):
    """Improved version of `priority_v1`."""

