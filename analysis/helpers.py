#!/usr/bin/env python3
"""
Shared helper functions for analysis scripts.

This module contains common utilities used across multiple analysis scripts:
- Function signature detection
- LCS computation for neighbor detection
- Graph building utilities
- Common imports for priority function execution
"""

import re
from typing import List, Set, Dict, Tuple


# =============================================================================
# Signature Types
# =============================================================================

SIGNATURE_NO_GRAPH = 'no_graph'           # priority(node, n, s, q)
SIGNATURE_GRAPH_GT = 'graph_gt'           # priority(node, G_gt, node_to_vertex, vertex_to_node, n, s)
SIGNATURE_GRAPH_NETWORKX = 'graph_networkx'  # priority(node, G, n, s)


# =============================================================================
# Common Imports for Priority Function Execution
# =============================================================================

COMMON_IMPORTS = """
import math
import itertools
import hashlib
import re
import collections
from collections import Counter, defaultdict
from math import log, log2, sqrt, exp, ceil, floor, factorial, gcd
from itertools import combinations, permutations, product

try:
    import numpy as np
    from numpy import zeros, ones, array
except ImportError:
    np = None
"""


# =============================================================================
# Signature Detection
# =============================================================================

def detect_signature(body: str, args: str = None) -> str:
    """
    Detect the function signature type from the body or args string.

    Args:
        body: The function body code
        args: Optional args string from the function definition

    Returns:
        One of: 'no_graph', 'graph_gt', 'graph_networkx'
    """
    # If we have the args string, use it directly
    if args:
        args_lower = args.lower()
        if 'g_gt' in args_lower or 'node_to_vertex' in args_lower:
            return SIGNATURE_GRAPH_GT
        elif 'g,' in args_lower or ', g)' in args_lower or 'g)' in args_lower.replace(' ', ''):
            # Check if G is used (but not G_gt)
            if 'g_gt' not in args_lower:
                return SIGNATURE_GRAPH_NETWORKX

    # Otherwise, analyze the function body for usage patterns
    body_lower = body.lower()

    # Check for graph-tool specific patterns
    if 'g_gt' in body_lower or 'node_to_vertex' in body_lower or 'vertex_to_node' in body_lower:
        return SIGNATURE_GRAPH_GT

    # Check for NetworkX patterns (G.neighbors, G.degree, etc.)
    if re.search(r'\bg\.(neighbors|degree|nodes|edges|adj)\b', body_lower):
        return SIGNATURE_GRAPH_NETWORKX

    # Check for graph-tool method patterns
    if '.out_degree()' in body or '.in_degree()' in body or 'G.vertex(' in body:
        return SIGNATURE_GRAPH_GT

    # Default to no_graph
    return SIGNATURE_NO_GRAPH


# =============================================================================
# LCS and Neighbor Detection
# =============================================================================

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
    Check if two nodes are neighbors (share subsequence of length >= n-s).

    In deletion-correcting codes, two codewords are neighbors if they
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


# =============================================================================
# Graph Building Utilities
# =============================================================================

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


def build_graph_gt(nodes: List[str], n: int, s: int, graph_dir: str = None):
    """
    Build graph-tool graph for the given nodes.

    If graph_dir is provided, tries to load from LMDB first.

    Args:
        nodes: List of node strings
        n: Length of codewords
        s: Number of deletions
        graph_dir: Optional path to directory with pre-computed graphs (LMDB format)

    Returns:
        Tuple of (G_gt, node_to_vertex, vertex_to_node)
    """
    try:
        import graph_tool.all as gt
    except ImportError:
        raise ImportError("graph-tool required. Install with: conda install -c conda-forge graph-tool")

    # Try loading from LMDB if graph_dir provided
    if graph_dir:
        import os
        graph_path = os.path.join(graph_dir, f"n{n}_s{s}")
        if os.path.exists(graph_path):
            try:
                G, node_to_vertex, vertex_to_node, _ = load_graph_from_lmdb(graph_path)
                return G, node_to_vertex, vertex_to_node
            except Exception as e:
                import sys
                print(f"Warning: Failed to load graph from {graph_path}: {e}", file=sys.stderr)

    # Fall back to building on-the-fly
    node_to_vertex = {node: idx for idx, node in enumerate(nodes)}
    vertex_to_node = {idx: node for idx, node in enumerate(nodes)}

    G = gt.Graph(directed=False)
    G.add_vertex(len(nodes))

    # Add edges
    edges = []
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            if are_neighbors(node1, node2, n, s):
                edges.append((i, j))

    if edges:
        G.add_edge_list(edges)

    return G, node_to_vertex, vertex_to_node


def load_graph_from_lmdb(graph_db_path: str):
    """
    Load graph from LMDB database (pre-computed graphs).

    Args:
        graph_db_path: Path to LMDB database

    Returns:
        Tuple of (G_gt, node_to_vertex, vertex_to_node, nodes_list)
    """
    try:
        import lmdb
        import graph_tool.all as gt
    except ImportError as e:
        raise ImportError(f"Required library missing: {e}")

    graph_env = lmdb.open(
        graph_db_path,
        readonly=True,
        lock=False,
        readahead=True,
        max_readers=126
    )

    try:
        nodes_list = []
        with graph_env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                node = key.tobytes().decode("utf-8")
                nodes_list.append(node)

        node_to_vertex = {node: idx for idx, node in enumerate(nodes_list)}
        vertex_to_node = {idx: node for idx, node in enumerate(nodes_list)}

        G = gt.Graph(directed=False)
        G.add_vertex(len(nodes_list))

        edges = []
        with graph_env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                node = key.tobytes().decode("utf-8")
                neighbors = value.tobytes().decode("utf-8").split(",")
                v1 = node_to_vertex[node]
                for neighbor in neighbors:
                    if neighbor and neighbor in node_to_vertex:
                        v2 = node_to_vertex[neighbor]
                        if v1 < v2:
                            edges.append((v1, v2))

        if edges:
            G.add_edge_list(edges)

        return G, node_to_vertex, vertex_to_node, nodes_list
    finally:
        graph_env.close()


# =============================================================================
# VT Code Utilities
# =============================================================================

def compute_vt_syndrome(codeword: str) -> int:
    """
    Compute VT syndrome: sum_{i=1}^{n} i * x_i

    Args:
        codeword: Binary string like "01101"

    Returns:
        The weighted position sum
    """
    return sum((i + 1) * int(bit) for i, bit in enumerate(codeword))


def compute_vt_complement_index(n: int, a: int = 0) -> int:
    """
    Compute the VT index of the complement codebook.

    For VT_a, the complement is VT_b where b = (n(n+1)/2 - a) mod (n+1).

    This is because:
    - syndrome(complement(x)) = n(n+1)/2 - syndrome(x)
    - If x in VT_a, then syndrome(x) ≡ a (mod n+1)
    - So syndrome(complement(x)) ≡ n(n+1)/2 - a (mod n+1)

    Args:
        n: Code length
        a: Syndrome value (default 0)

    Returns:
        The complement VT index b
    """
    return (n * (n + 1) // 2 - a) % (n + 1)


def bitwise_complement(codeword: str) -> str:
    """Compute bitwise complement of a binary string."""
    return ''.join('1' if c == '0' else '0' for c in codeword)


def is_flat_vt_format(data: dict) -> bool:
    """
    Check if VT data is in old flat format (n -> codewords) vs nested (n -> a -> codewords).

    Args:
        data: VT codes dictionary

    Returns:
        True if flat format, False if nested
    """
    if not data:
        return False
    first_value = next(iter(data.values()))
    return isinstance(first_value, list)
