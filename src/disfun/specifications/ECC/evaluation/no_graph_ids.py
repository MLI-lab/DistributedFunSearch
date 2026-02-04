"""Evaluation script for IDS problem (no_graph variant).

PROBLEM TYPE: IDS only (q-ary strings, edit distance neighbor check)
DO NOT use this for Deletions - use no_graph_deletions.py instead.

Neighbors are defined by: Levenshtein.distance(node1, node2) < 2*s + 1
This variant computes neighbors on the fly without loading precomputed graphs.
"""

import sys
import itertools
import hashlib
import Levenshtein
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


def are_neighbors(node1, node2, s):
    r"""Check if two nodes are neighbors (edit distance < 2s + 1, for ids code d_min  \geq 2s+1)."""
    return Levenshtein.distance(node1, node2) < 2 * s + 1


def priority(node, n, s) -> float:
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
    print(f"Computing independent set on-the-fly for n={n}, s={s}, q={q} (no graph loading)", file=sys.stderr)

    # Generate all q-ary strings of length n
    alphabet = ''.join(str(i) for i in range(q))
    nodes = [''.join(seq) for seq in itertools.product(alphabet, repeat=n)]

    # Calculate priorities based only on node string properties (no graph passed)
    priorities = {node: priority(node, n, s) for node in nodes}

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

        # Remove all neighbors (computed on-the-fly using Levenshtein distance)
        for other_node in nodes:
            if other_node not in removed_nodes:
                if are_neighbors(node, other_node, s):
                    removed_nodes.add(other_node)

    # Compute hash for deduplication (only for start_n)
    # Note: "n == start_n" gets replaced with actual value (e.g. "n == 7") at runtime by __main__.py
    hash_value = None
    start_n = globals().get('start_n', min(params[0] for params in [(n, s, q)]))
    if n == start_n:
        hash_value = hash_priority_mapping(priorities, nodes)

    return independent_set, hash_value
