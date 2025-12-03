"""Evaluation script for IDS problem using NetworkX.

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

    # Make a copy for priority computation
    G_for_priority = G.copy()

    # Compute priorities
    priorities = {
        node: priority(node, G_for_priority, n, s)
        for node in G.nodes
    }

    # Sort nodes by priority (descending), Lexicographic tie-breaking (the second element x is the node string)
    nodes_sorted = sorted(G.nodes, key=lambda x: (-priorities[x], x))

    # Greedy independent set construction
    independent_set = set()
    for node in nodes_sorted:
        if node not in G:  # Already removed
            continue

        independent_set.add(node)
        neighbors = list(G.neighbors(node))
        G.remove_node(node)
        G.remove_nodes_from(neighbors)

    # Compute hash for deduplication (only for smallest n)
    hash_value = None
    start_n = globals().get('start_n', min(params[0] for params in [(n, s, q)]))
    if n == start_n:
        sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]
        hash_value = hash_priority_mapping(priorities, sequences)

    return independent_set, hash_value
