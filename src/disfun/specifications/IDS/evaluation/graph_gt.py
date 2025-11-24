"""Evaluation script for IDS problem using graph-tool.

Graph-tool is faster than NetworkX (see analysis/time_priority_function_graphtool.py and analysis/time_priority_function_networkx.py for time comparison).
However, LLMs must convert node strings to vertex indices before graph operations, which causes more execution errors in generated functions (e.g. "object of too small" errors).

"""

import os
import sys
import itertools
import hashlib
import numpy as np
import graph_tool.all as gt
import ujson as json
import lmdb

# Global graph cache
_GRAPH_CACHE = {}


def load_graph(graph_db_path):
    """Load graph from LMDB database into graph-tool Graph."""
    graph_env = lmdb.open(
        graph_db_path,
        readonly=True,
        lock=False,
        readahead=True,
        max_readers=126
    )

    try:
        # collect all nodes (keys) and assign indices
        nodes_list = []
        with graph_env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                # key is a buffer-like object, decode once here
                node = key.tobytes().decode("utf-8")
                nodes_list.append(node)

        # Build mappings once
        node_to_vertex = {node: idx for idx, node in enumerate(nodes_list)}
        vertex_to_node = {idx: node for idx, node in enumerate(nodes_list)}

        # Create graph with all vertices preallocated
        G = gt.Graph(directed=False)
        G.add_vertex(len(nodes_list))

        # Build edges directly as integer pairs
        edge_list = []
        with graph_env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                node = key.tobytes().decode("utf-8")
                src = node_to_vertex[node]

                neighbors = json.loads(value.tobytes().decode("utf-8"))
                for neighbor in neighbors:
                    # undirected dedupe by string ordering
                    if node < neighbor:
                        try:
                            dst = node_to_vertex[neighbor]
                        except KeyError:
                            # neighbor might not exist as a key; skip
                            continue
                        edge_list.append((src, dst))

        # Add edges 
        G.add_edge_list(edge_list)

        return G, node_to_vertex, vertex_to_node

    finally:
        graph_env.close()


def hash_priority_mapping(priorities, sequences):
    """Generate a hash based on the mapping of sequences to their priority scores."""
    mapping = [(seq, priorities[seq]) for seq in sequences]
    mapping_sorted = sorted(mapping, key=lambda x: x[0])
    mapping_str = ','.join(f'{seq}:{score}' for seq, score in mapping_sorted) # string e.g. seq1:score1,seq2:score2,seq3:score3 gets hashed 
    return hashlib.sha256(mapping_str.encode()).hexdigest()


def evaluate(params, graph_dir):
    n, s, q = params
    independent_set, hash_value = solve(n, s, q, graph_dir)
    return (len(independent_set), hash_value)


def solve(n, s, q, graph_dir):
    """Find a large independent set using graph-tool for speed and optional caching."""
    path = os.path.join(graph_dir, f"graph_d_s{s}_n{n}_q{q}.lmdb")
    cache_key = (s, n)

    cache_enabled = globals().get('CACHE_GRAPHS', False)
    cache_limit_gb = globals().get('CACHE_SIZE_LIMIT_GB', 2.0)

    if cache_enabled and cache_key in _GRAPH_CACHE:
        print(f"Using cached graph for s={s}, n={n}", file=sys.stderr)
        G, node_to_vertex, vertex_to_node = _GRAPH_CACHE[cache_key]
    else:
        print(f"Loading graph from: {path}", file=sys.stderr)
        G, node_to_vertex, vertex_to_node = load_graph(path)

        if cache_enabled:
            num_nodes = G.num_vertices()
            num_edges = G.num_edges()
            estimated_size_bytes = (num_nodes * 100) + (num_edges * 50)
            estimated_size_gb = estimated_size_bytes / (1024**3)

            if estimated_size_gb < cache_limit_gb:
                print(f"Caching graph (estimated size: {estimated_size_gb:.2f} GB)", file=sys.stderr)
                _GRAPH_CACHE[cache_key] = (G, node_to_vertex, vertex_to_node)
            else:
                print(f" Graph too large to cache ({estimated_size_gb:.2f} GB > {cache_limit_gb} GB limit)", file=sys.stderr)

    priorities = {
        node: priority(node, G, node_to_vertex, vertex_to_node, n, s, q)
        for node in vertex_to_node.values()
    }


    # Sort nodes by priority (descending), Lexicographic tie-breaking (the second element x is the node string)
    nodes_sorted = sorted(vertex_to_node.values(), key=lambda x: (-priorities[x], x))

    independent_set = set()
    removed_vertices = set()

    for node in nodes_sorted:
        v = node_to_vertex[node]
        if v in removed_vertices:
            continue

        independent_set.add(node)
        removed_vertices.add(v)

        neighbors = G.get_out_neighbors(v)
        for neighbor_v in neighbors:
            removed_vertices.add(int(neighbor_v))

    hash_value = None
    if n == start_n:
        sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]
        hash_value = hash_priority_mapping(priorities, sequences)

    return independent_set, hash_value


def priority(node, G, node_to_vertex, vertex_to_node, n, s, q):
    """ Placeholder, this function will be evolved. """
    pass