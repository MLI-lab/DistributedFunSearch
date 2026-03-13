#!/usr/bin/env python3
"""
Shared helper functions for analysis scripts.

This module contains common utilities used across multiple analysis scripts:
  Function signature detection
  LCS computation for neighbor detection
  Graph building utilities
  Common imports for priority function execution

VT code utilities are in vt_utils.py but re exported here for compatibility.
"""

import re
from typing import List, Set, Dict, Tuple

# Re export VT utilities for backward compatibility
from .vt_utils import (
    load_vt_codes,
    is_flat_vt_format,
    compute_vt_syndrome,
    compute_vt_complement_index,
    bitwise_complement,
    complement_codebook,
)


# Signature types

SIGNATURE_NO_GRAPH = 'no_graph'           # priority(node, n, s, q)
SIGNATURE_GRAPH_NETWORKX = 'graph_networkx'  # priority(node, G, n, s)


# Common imports for priority function execution

COMMON_IMPORTS = """
import math
from math import *
import itertools
from itertools import combinations, permutations, product
import hashlib
import re
import random
import collections
from collections import Counter, defaultdict

try:
    import numpy as np
    from numpy import zeros, ones, array
    mean = np.mean
except ImportError:
    np = None

try:
    import networkx as nx
except ImportError:
    nx = None

inf = float('inf')
"""


# Signature detection

def detect_signature(body: str, args: str = None) -> str:
    """
    Detect the function signature type from the body or args string.

    Args:
        body: The function body code
        args: Optional args string from the function definition

    Returns:
        One of: 'no_graph', 'graph_networkx'
    """
    # If we have the args string, use it directly
    if args:
        args_lower = args.lower()
        if 'g,' in args_lower or ', g)' in args_lower or 'g)' in args_lower.replace(' ', ''):
            return SIGNATURE_GRAPH_NETWORKX

    # Otherwise, analyze the function body for usage patterns
    body_lower = body.lower()

    # Check for NetworkX patterns (G.neighbors, G.degree, etc.)
    if re.search(r'\bg\.(neighbors|degree|nodes|edges|adj)\b', body_lower):
        return SIGNATURE_GRAPH_NETWORKX

    # Default to no_graph
    return SIGNATURE_NO_GRAPH


# LCS and neighbor detection

def lcs_length(s1: str, s2: str) -> int:
    """
    Compute longest common subsequence length using dynamic programming.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Length of the longest common subsequence
    """
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


def are_neighbors(node1: str, node2: str, n: int, s: int) -> bool:
    """
    Check if two nodes are neighbors (share subsequence of length >= n s).

    In deletion correcting codes, two codewords are neighbors if they
    could be confused after up to s deletions.

    Args:
        node1: First node (binary string)
        node2: Second node (binary string)
        n: Length of codewords
        s: Number of deletions to correct

    Returns:
        True if nodes are neighbors
    """
    return lcs_length(node1, node2) >= n - s


# Graph building

def build_graph_networkx(nodes: List[str], n: int, s: int):
    """
    Build NetworkX graph for the given nodes.

    Args:
        nodes: List of node strings
        n: Length of codewords
        s: Number of deletions

    Returns:
        NetworkX Graph object
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX required. Install with: pip install networkx")

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            if are_neighbors(node1, node2, n, s):
                G.add_edge(node1, node2)

    return G


# VT Code Utilities (re exported from vt_utils.py).
# See vt_utils.py for implementation.
