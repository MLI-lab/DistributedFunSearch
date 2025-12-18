#!/usr/bin/env python3
"""Profile evaluation script to find bottlenecks."""

import time
import sys
sys.path.insert(0, '/workspace/DistributedFunSearch/src')

import itertools
import lmdb
import ujson as json
import networkx as nx
import os
import psutil

GRAPH_DIR = "/workspace/DistributedFunSearch/src/graphs"


# =============================================================================
# Priority functions for GRAPH-BASED evaluation (signature: node, G, n, s)
# =============================================================================
def graph_simple_priority(node, G, n, s):
    """O(1) - return constant."""
    return 0.0

def graph_degree_priority(node, G, n, s):
    """O(degree) - access neighbors."""
    return -G.degree(node)

def graph_expensive_priority(node, G, n, s):
    """O(nodes) - iterate all nodes."""
    return sum(1 for _ in G.nodes())

GRAPH_PRIORITY_FUNCS = [
    ("simple", graph_simple_priority),
    ("degree", graph_degree_priority),
    ("expensive", graph_expensive_priority),
]


# =============================================================================
# Priority functions for NO-GRAPH evaluation (signature: node, n, s, q)
# =============================================================================
def nograph_simple_priority(node, n, s, q):
    """O(1) - return constant."""
    return 0.0

def nograph_degree_priority(node, n, s, q):
    """O(n) - iterate over node string (analogous to O(degree))."""
    return sum(ord(c) for c in node)

def nograph_expensive_priority(node, n, s, q):
    """O(nodes) - iterate over all q^n nodes."""
    return sum(1 for _ in range(q ** n))

NOGRAPH_PRIORITY_FUNCS = [
    ("simple", nograph_simple_priority),
    ("degree", nograph_degree_priority),
    ("expensive", nograph_expensive_priority),
]


# =============================================================================
# Helper functions
# =============================================================================

# === COPIED FROM graph_networkx.py lines 101-126 ===
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


# === COPIED FROM no_graph.py lines 68-89 ===
def lcs_length(s1, s2):
    """Compute longest common subsequence length using dynamic programming."""
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

def are_neighbors(node1, node2, n, s):
    """Check if two nodes are neighbors (share subsequence of length >= n-s)."""
    return lcs_length(node1, node2) >= n - s
# === END COPIED FROM no_graph.py ===


def solve_with_priority(G, n, s, priority_func):
    """Run the solve algorithm with timing breakdown and CPU utilization.

    Code matches: src/disfun/specifications/Deletions/evaluation/graph_networkx.py
    (lines 174-206)
    """
    process = psutil.Process(os.getpid())

    # Get CPU time before entire solve
    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()

    # === COPIED FROM graph_networkx.py lines 174-191 ===
    # Freeze graph to protect against modifications (instant, no copy needed)
    # Note: nx.freeze() is O(1) - just sets a flag, doesn't affect profiling
    if not nx.is_frozen(G):
        nx.freeze(G)

    # Compute priorities (G is frozen/read-only)
    t0 = time.perf_counter()
    priorities = {}
    for node in G.nodes:
        try:
            p = priority_func(node, G, n, s)
            if p is None:
                priorities[node] = 0.0
            else:
                priorities[node] = float(p)
        except (TypeError, ValueError):
            priorities[node] = 0.0
    t_priority = time.perf_counter() - t0

    # Sort nodes by priority (descending), Lexicographic tie-breaking
    t0 = time.perf_counter()
    nodes_sorted = sorted(G.nodes, key=lambda x: (-priorities[x], x))
    t_sort = time.perf_counter() - t0

    # Greedy independent set construction (don't modify G - uses removed set)
    t0 = time.perf_counter()
    independent_set = set()
    removed = set()
    for node in nodes_sorted:
        if node in removed:
            continue
        independent_set.add(node)
        removed.add(node)
        for neighbor in G.neighbors(node):
            removed.add(neighbor)
    t_greedy = time.perf_counter() - t0
    # === END COPIED FROM graph_networkx.py ===

    # Get CPU time after
    cpu_end = process.cpu_times()
    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start
    cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_util = (cpu_time / wall_time * 100) if wall_time > 0 else 0

    return len(independent_set), t_priority, t_sort, t_greedy, cpu_time, cpu_util


def main():
    print("=" * 80)
    print("Evaluation Profiling (with CPU Utilization)")
    print("=" * 80)

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
        for name, func in GRAPH_PRIORITY_FUNCS:
            score, t_pri, t_sort, t_greedy, cpu_time, cpu_util = solve_with_priority(G, n, s, func)
            total = t_pri + t_sort + t_greedy
            print(f"  {name:12}: priority={t_pri*1000:7.1f}ms, sort={t_sort*1000:6.1f}ms, "
                  f"greedy={t_greedy*1000:6.1f}ms, total={total*1000:7.1f}ms, "
                  f"cpu={cpu_time*1000:6.1f}ms, cpu%={cpu_util:5.1f}%, score={score}")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("  - cpu% ~100%: Fully CPU-bound (good core utilization)")
    print("  - cpu% <80%:  Some overhead/idle time (may not need full core)")
    print("  - total <50ms: Very fast, process overhead may dominate")
    print("=" * 80)


def profile_no_graph():
    """Profile the no_graph evaluation.

    Code matches: src/disfun/specifications/Deletions/evaluation/no_graph.py
    (lines 115-139)
    """
    def solve_no_graph(n, s, q, priority_func):
        """Profile solve() from no_graph.py lines 110-139."""
        process = psutil.Process(os.getpid())
        cpu_start = process.cpu_times()
        wall_start = time.perf_counter()

        # === COPIED FROM no_graph.py lines 115-139 ===
        # Generate all q-ary strings of length n
        nodes = [''.join(seq) for seq in itertools.product(map(str, range(q)), repeat=n)]

        # Calculate priorities based only on node string properties (no graph passed)
        priorities = {node: priority_func(node, n, s, q) for node in nodes}

        # Sort nodes by priority (descending), Lexicographic tie-breaking
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

            # Remove all neighbors (computed on-the-fly)
            for other_node in nodes:
                if other_node not in removed_nodes:
                    if are_neighbors(node, other_node, n, s):
                        removed_nodes.add(other_node)
        # === END COPIED FROM no_graph.py ===

        cpu_end = process.cpu_times()
        wall_end = time.perf_counter()

        wall_time = wall_end - wall_start
        cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        cpu_util = (cpu_time / wall_time * 100) if wall_time > 0 else 0

        return len(independent_set), wall_time, cpu_time, cpu_util

    print("\n" + "=" * 90)
    print("NO_GRAPH Evaluation Profiling")
    print("Code copied from: src/disfun/specifications/Deletions/evaluation/no_graph.py")
    print("=" * 90)

    # Use globally defined priority functions
    priority_funcs = NOGRAPH_PRIORITY_FUNCS

    for n in [6, 7, 8, 9, 10, 11]:
        s, q = 1, 2
        num_nodes = q ** n
        print(f"\nn={n} ({num_nodes} nodes)")

        for name, func in priority_funcs:
            score, wall_time, cpu_time, cpu_util = solve_no_graph(n, s, q, func)
            print(f"  {name:12}: wall={wall_time*1000:8.1f}ms, cpu={cpu_time*1000:8.1f}ms, "
                  f"cpu%={cpu_util:5.1f}%, score={score}")

    print("\n" + "-" * 90)
    print("Priority function complexity (consistent with graph-based version):")
    print("  simple:    O(1)     - constant time")
    print("  degree:    O(n)     - iterate node string")
    print("  expensive: O(nodes) - iterate all q^n nodes")
    print("-" * 90)


if __name__ == "__main__":
    main()
    profile_no_graph()
