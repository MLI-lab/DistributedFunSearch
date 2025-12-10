"""Evaluation script for Deletions problem using NetworkX.

This is the simpler version for LLMs as NetworkX allows direct node access without conversion.
Use this if LLMs keep making "object of too small" errors with graph-tool.

Slower than graph-tool, but easier for LLMs to use correctly.
"""

import os
import sys
import itertools
import hashlib
import networkx as nx
import ujson as json
import lmdb
import numpy as np
import math
import re
import random
import time
import binascii
import collections
import pandas as pd
from itertools import groupby, combinations, permutations, product
from functools import reduce, lru_cache
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from binascii import hexlify
from math import log, log2, sqrt, exp, ceil, floor, factorial, gcd
from numpy import zeros, ones, array

# Python 2 compatibility
xrange = range

# Binary string helpers
def str2int(s): return int(s, 2)
def int2str(n, length): return format(n, f'0{length}b')
def hamming_distance(s1, s2): return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# Aliases for common LLM naming variations
HammingDistance = hamming_distance
Hamming_Distance = hamming_distance
hammingDistance = hamming_distance

# Graph helper aliases (LLMs often expect these)
Graph = nx.Graph
DiGraph = nx.DiGraph

def degree(node, G):
    """Get degree of a node."""
    return G.degree(node)

def neighbors(node, G):
    """Get list of neighbors for a node."""
    return list(G.neighbors(node))

def edge_weight(G, u, v, default=1):
    """Get edge weight with fallback (graph has no weights by default)."""
    return G[u][v].get('weight', default) if G.has_edge(u, v) else default

def num_nodes(G):
    """Get number of nodes in graph."""
    return G.number_of_nodes()

def num_edges(G):
    """Get number of edges in graph."""
    return G.number_of_edges()

# Common variable aliases LLMs might reference
inf = float('inf')
INF = float('inf')
pi = math.pi
PI = math.pi
e = math.e
E = math.e

# Statistics helpers (LLMs often use these without importing)
def mean(x):
    """Calculate mean of a sequence."""
    lst = list(x)
    return sum(lst) / len(lst) if lst else 0.0

def std(x):
    """Calculate standard deviation of a sequence."""
    lst = list(x)
    if len(lst) < 2:
        return 0.0
    m = sum(lst) / len(lst)
    variance = sum((xi - m) ** 2 for xi in lst) / len(lst)
    return sqrt(variance)

# Safe division helper
def safe_div(a, b, default=0.0):
    """Safe division that returns default on zero division."""
    return a / b if b != 0 else default

# Global graph cache
_GRAPH_CACHE = {}


def load_graph(graph_db_path):
    """Load graph from LMDB database into NetworkX Graph."""


    graph_env = lmdb.open(graph_db_path, readonly=True, lock=False,
                          readahead=True, max_readers=126)

    edges = set()
    nodes_list = []

    with graph_env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            node = bytes(key).decode()
            nodes_list.append(node)
            neighbors = json.loads(bytes(value).decode())
            for neighbor in neighbors:
                if node < neighbor:  # Add each edge only once
                    edges.add((node, neighbor))

    graph_env.close()

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges)

    return G


def priority(node, G, n, s) -> float:
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
    """Find a large independent set using NetworkX (slower but simpler for LLMs)."""
    path = os.path.join(graph_dir, f"graph_d_s{s}_n{n}_q{q}.lmdb")
    cache_key = (s, n)

    cache_enabled = globals().get('CACHE_GRAPHS', False)
    cache_limit_gb = globals().get('CACHE_SIZE_LIMIT_GB', 2.0)

    if cache_enabled and cache_key in _GRAPH_CACHE:
        print(f"Using cached graph for s={s}, n={n}", file=sys.stderr)
        G = _GRAPH_CACHE[cache_key]
    else:
        print(f"Loading graph from: {path}", file=sys.stderr)
        G = load_graph(path)

        if cache_enabled:
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            estimated_size_bytes = (num_nodes * 100) + (num_edges * 50)
            estimated_size_gb = estimated_size_bytes / (1024**3)

            if estimated_size_gb < cache_limit_gb:
                print(f"Caching graph (estimated size: {estimated_size_gb:.2f} GB)", file=sys.stderr)
                _GRAPH_CACHE[cache_key] = G
            else:
                print(f"Graph too large to cache ({estimated_size_gb:.2f} GB > {cache_limit_gb} GB limit)", file=sys.stderr)

    # Freeze graph to protect against LLM modifications (instant, no copy needed)
    # If LLM tries to modify G, it raises NetworkXError and evaluation fails
    nx.freeze(G)

    # Compute priorities (G is frozen - read-only)
    # Handle None/non-numeric returns gracefully by defaulting to 0.0
    priorities = {}
    for node in G.nodes:
        try:
            p = priority(node, G, n, s)
            # Convert to float, default to 0.0 if None or non-numeric
            if p is None:
                priorities[node] = 0.0
            else:
                priorities[node] = float(p)
        except (TypeError, ValueError):
            priorities[node] = 0.0

    # Sort nodes by priority (descending), Lexicographic tie-breaking (the second element x is the node string)
    nodes_sorted = sorted(G.nodes, key=lambda x: (-priorities[x], x))

    # Greedy independent set construction (don't modify G - preserves cache)
    independent_set = set()
    removed = set()
    for node in nodes_sorted:
        if node in removed:
            continue

        independent_set.add(node)
        removed.add(node)
        for neighbor in G.neighbors(node):
            removed.add(neighbor)

    # Compute hash for deduplication (only for smallest n)
    hash_value = None
    start_n = globals().get('start_n', min(params[0] for params in [(n, s, q)]))
    if n == start_n:
        sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]
        hash_value = hash_priority_mapping(priorities, sequences)

    return independent_set, hash_value
