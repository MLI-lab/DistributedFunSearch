#!/usr/bin/env python3
"""Profile evaluation script to find bottlenecks."""

import time
import sys
sys.path.insert(0, '/workspace/DistributedFunSearch/src')

import lmdb
import ujson as json
import networkx as nx
import os

GRAPH_DIR = "/workspace/DistributedFunSearch/src/graphs"

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
                if node < neighbor:
                    edges.add((node, neighbor))

    graph_env.close()

    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges)
    return G


def simple_priority(node, G, n, s):
    """Simple O(1) priority."""
    return 0.0


def degree_priority(node, G, n, s):
    """O(degree) priority - access neighbors."""
    return -G.degree(node)


def expensive_priority(node, G, n, s):
    """O(n) priority - iterate all nodes."""
    return sum(1 for _ in G.nodes())


def solve_with_priority(G, n, s, priority_func):
    """Run the solve algorithm with timing breakdown."""
    G_work = G.copy()

    # Time priority computation
    t0 = time.perf_counter()
    priorities = {node: priority_func(node, G, n, s) for node in G.nodes}
    t_priority = time.perf_counter() - t0

    # Time sorting
    t0 = time.perf_counter()
    nodes_sorted = sorted(G.nodes, key=lambda x: (-priorities[x], x))
    t_sort = time.perf_counter() - t0

    # Time greedy construction
    t0 = time.perf_counter()
    independent_set = set()
    for node in nodes_sorted:
        if node not in G_work:
            continue
        independent_set.add(node)
        neighbors = list(G_work.neighbors(node))
        G_work.remove_node(node)
        G_work.remove_nodes_from(neighbors)
    t_greedy = time.perf_counter() - t0

    return len(independent_set), t_priority, t_sort, t_greedy


def main():
    print("=" * 60)
    print("Evaluation Profiling")
    print("=" * 60)

    for n in [6, 7, 8, 9, 10, 11]:
        s, q = 1, 2
        path = os.path.join(GRAPH_DIR, f"graph_d_s{s}_n{n}_q{q}.lmdb")

        if not os.path.exists(path):
            print(f"n={n}: Graph not found at {path}")
            continue

        # Time graph loading
        t0 = time.perf_counter()
        G = load_graph(path)
        t_load = time.perf_counter() - t0

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        print(f"\nn={n} ({num_nodes} nodes, {num_edges} edges)")
        print(f"  Graph load: {t_load*1000:.1f}ms")

        # Test different priority functions
        for name, func in [("simple", simple_priority),
                           ("degree", degree_priority),
                           ("expensive", expensive_priority)]:
            score, t_pri, t_sort, t_greedy = solve_with_priority(G, n, s, func)
            total = t_pri + t_sort + t_greedy
            print(f"  {name:12}: priority={t_pri*1000:7.1f}ms, sort={t_sort*1000:6.1f}ms, "
                  f"greedy={t_greedy*1000:6.1f}ms, total={total*1000:7.1f}ms, score={score}")


if __name__ == "__main__":
    main()
