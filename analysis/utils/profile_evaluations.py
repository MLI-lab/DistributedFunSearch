#!/usr/bin/env python3
"""
Profile Evaluation Performance: Graph-based vs No-Graph

Compares two evaluation approaches:
1. Graph-based: Load pre-computed LMDB graph, fast neighbor lookups
2. No-graph: Compute neighbors on-the-fly using LCS (no disk I/O)

Measures both time and peak memory usage to help decide which approach fits your constraints.

Outputs (saved incrementally after each n):
- profile_s<s>_q<q>_<type>_<timestamp>.json  - Full results (programmatic access)
- profile_s<s>_q<q>_<type>_<timestamp>.txt   - Human-readable summary

Usage Examples:
    # Default: quaternary IDS (q=4, type=ids), saves to auto-generated filename
    python profile_evaluations.py

    # Binary deletion codes
    python profile_evaluations.py --q 2 --graph-type deletion

    # Quaternary deletion codes
    python profile_evaluations.py --q 4 --graph-type deletion

    # IDS codes (binary)
    python profile_evaluations.py --q 2 --graph-type ids

    # Specific n values
    python profile_evaluations.py --n-values 6,7,8,9

    # Custom output filename
    python profile_evaluations.py --n-values 6,7,8 -o ./my_profile

    # Console only (no file output)
    python profile_evaluations.py --n-values 6,7,8 --no-output

    # Only profile no-graph mode (skip graph loading)
    python profile_evaluations.py --no-graph-only

    # Only profile graph-based mode (skip no-graph)
    python profile_evaluations.py --graph-only

    # Custom graph directory
    python profile_evaluations.py --graph-dir /path/to/graphs

Arguments:
    --n-values        Comma-separated n values (default: 6,7,8,9,10,11)
    --s               Number of errors to correct (default: 1)
    --q               Alphabet size: 2=binary, 4=quaternary (default: 4)
    --graph-type      Graph type: 'deletion' or 'ids' (default: ids)
    --graph-dir       Directory with LMDB graphs (default: /mnt/Graphs)
    --output, -o      Output file path (default: auto-generated with timestamp)
    --no-output       Disable file output (console only)
    --no-graph-only   Only profile no-graph mode
    --graph-only      Only profile graph-based mode

Interpretation:
    TIMING:
    - Graph-based wins for repeated evaluations (load once, evaluate many)
    - No-graph wins for single evaluation if graph loading is slow

    MEMORY:
    - Graph-based: High memory (stores full adjacency list)
    - No-graph: Low memory (just node strings + temporary sets)
    - Use no-graph if memory-constrained
"""

import argparse
import time
import sys
import os
import gc
import json as std_json
import tracemalloc
from datetime import datetime

sys.path.insert(0, '/workspace/DistributedFunSearch/analysis')

import itertools
import psutil

from utils.helpers import are_neighbors
from utils.graph_paths import get_graph_path, DEFAULT_GRAPH_DIR


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def format_memory(mb):
    """Format memory size in GiB."""
    gib = mb / 1024
    return f"{gib:.3f} GiB"


def format_time(seconds):
    """Format time in seconds and minutes."""
    if seconds >= 60:
        mins = seconds / 60
        return f"{seconds:.2f}s ({mins:.2f}min)"
    else:
        return f"{seconds:.2f}s"

# Optional imports for graph-based mode
try:
    import lmdb
    import ujson as json
    import networkx as nx
    HAS_GRAPH_DEPS = True
except ImportError:
    HAS_GRAPH_DEPS = False


# =============================================================================
# Priority functions (simple O(1) for fair comparison)
# =============================================================================
def priority_simple(node, *args):
    """O(1) priority - focuses timing on graph operations, not priority computation."""
    return 0.0


# =============================================================================
# Graph-based evaluation (loads pre-computed graph)
# =============================================================================
def load_graph_with_memory(graph_db_path):
    """Load graph from LMDB database into NetworkX Graph. Returns (G, peak_memory_mb)."""
    # Force garbage collection and get baseline memory
    gc.collect()
    mem_before = get_process_memory_mb()

    # Start memory tracking
    tracemalloc.start()

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

    # Get peak memory from tracemalloc
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Also measure process memory delta
    mem_after = get_process_memory_mb()
    process_delta = mem_after - mem_before

    # Use the larger of the two measurements
    peak_mb = max(peak / (1024 * 1024), process_delta)

    return G, peak_mb, mem_after


def solve_with_graph(G, n, s):
    """Evaluate using pre-loaded graph. Returns (score, timings)."""
    process = psutil.Process(os.getpid())

    # Freeze graph
    if not nx.is_frozen(G):
        nx.freeze(G)

    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()

    # Priority computation
    t0 = time.perf_counter()
    priorities = {node: priority_simple(node) for node in G.nodes}
    t_priority = time.perf_counter() - t0

    # Sort
    t0 = time.perf_counter()
    nodes_sorted = sorted(G.nodes, key=lambda x: (-priorities[x], x))
    t_sort = time.perf_counter() - t0

    # Greedy with graph neighbor lookup
    t0 = time.perf_counter()
    independent_set = set()
    removed = set()
    for node in nodes_sorted:
        if node in removed:
            continue
        independent_set.add(node)
        removed.add(node)
        for neighbor in G.neighbors(node):  # O(1) lookup per neighbor
            removed.add(neighbor)
    t_greedy = time.perf_counter() - t0

    cpu_end = process.cpu_times()
    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start
    cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)

    return len(independent_set), {
        'priority': t_priority,
        'sort': t_sort,
        'greedy': t_greedy,
        'total': t_priority + t_sort + t_greedy,
        'cpu': cpu_time,
        'wall': wall_time,
    }


# =============================================================================
# No-graph evaluation (computes neighbors on-the-fly)
# =============================================================================
def solve_no_graph(n, s, q):
    """Evaluate without graph - computes LCS neighbors on-the-fly. Returns (score, timings, peak_memory_mb)."""
    process = psutil.Process(os.getpid())

    # Force garbage collection and get baseline memory
    gc.collect()
    mem_before = get_process_memory_mb()

    # Start memory tracking
    tracemalloc.start()

    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()

    # Generate all nodes
    t0 = time.perf_counter()
    nodes = [''.join(seq) for seq in itertools.product(map(str, range(q)), repeat=n)]
    t_generate = time.perf_counter() - t0

    # Priority computation
    t0 = time.perf_counter()
    priorities = {node: priority_simple(node) for node in nodes}
    t_priority = time.perf_counter() - t0

    # Sort
    t0 = time.perf_counter()
    nodes_sorted = sorted(nodes, key=lambda x: (-priorities[x], x))
    t_sort = time.perf_counter() - t0

    # Greedy with on-the-fly neighbor computation
    t0 = time.perf_counter()
    independent_set = set()
    removed = set()
    for node in nodes_sorted:
        if node in removed:
            continue
        independent_set.add(node)
        removed.add(node)
        # Compute neighbors on-the-fly using LCS
        for other in nodes:
            if other not in removed and are_neighbors(node, other, n, s):
                removed.add(other)
    t_greedy = time.perf_counter() - t0

    cpu_end = process.cpu_times()
    wall_end = time.perf_counter()

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_after = get_process_memory_mb()
    process_delta = mem_after - mem_before
    peak_mb = max(peak / (1024 * 1024), process_delta)

    wall_time = wall_end - wall_start
    cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)

    return len(independent_set), {
        'generate': t_generate,
        'priority': t_priority,
        'sort': t_sort,
        'greedy': t_greedy,
        'total': t_generate + t_priority + t_sort + t_greedy,
        'cpu': cpu_time,
        'wall': wall_time,
    }, peak_mb


# =============================================================================
# Main profiling
# =============================================================================
def profile_comparison(n_values, s, q, graph_type, graph_dir, skip_graph=False, skip_no_graph=False, output_path=None):
    """Profile both approaches and compare."""

    print("=" * 90)
    print("  Evaluation Profiling: Graph-based vs No-Graph")
    print("=" * 90)
    print(f"  Parameters: s={s}, q={q}, type={graph_type}")
    print(f"  Graph directory: {graph_dir}")
    print(f"  n values: {n_values}")
    if output_path:
        print(f"  Output: {output_path} (incremental saves)")
    print("=" * 90)

    results = []

    # Create writer if output path specified
    writer = ResultsWriter(output_path, s, q, graph_type) if output_path else None

    for n in n_values:
        num_nodes = q ** n
        print(f"\n{'='*80}")
        print(f"n={n} ({num_nodes:,} nodes)")
        print("=" * 80)

        result = {'n': n, 'num_nodes': num_nodes}

        # --- Graph-based evaluation ---
        if not skip_graph and HAS_GRAPH_DEPS:
            path = get_graph_path(graph_type, s, n, q, graph_dir=graph_dir)

            if path:
                print(f"\n[GRAPH-BASED] Loading: {path}")

                # Time graph loading with memory tracking
                t0 = time.perf_counter()
                G, peak_mem_load, mem_after_load = load_graph_with_memory(path)
                t_load = time.perf_counter() - t0

                num_edges = G.number_of_edges()
                print(f"  Graph load: {format_time(t_load)} ({num_edges:,} edges)")
                print(f"  Peak memory (load): {format_memory(peak_mem_load)}")
                print(f"  Process memory after load: {format_memory(mem_after_load)}")

                # Time evaluation
                score, timings = solve_with_graph(G, n, s)

                print(f"  Evaluation: priority={format_time(timings['priority'])}, "
                      f"sort={format_time(timings['sort'])}, greedy={format_time(timings['greedy'])}")
                print(f"  Total (eval only): {format_time(timings['total'])}")
                print(f"  Total (with load): {format_time(t_load + timings['total'])}")
                print(f"  Score: {score}")

                result['graph'] = {
                    'load': t_load,
                    'eval': timings['total'],
                    'total': t_load + timings['total'],
                    'score': score,
                    'edges': num_edges,
                    'peak_memory_mb': peak_mem_load,
                    'process_memory_mb': mem_after_load,
                }

                # Free the graph to not affect no-graph memory measurement
                del G
                gc.collect()
            else:
                print(f"\n[GRAPH-BASED] Graph not found for n={n}")
                result['graph'] = None
        elif skip_graph:
            print(f"\n[GRAPH-BASED] Skipped (--no-graph-only)")
            result['graph'] = None
        else:
            print(f"\n[GRAPH-BASED] Skipped (missing lmdb/networkx)")
            result['graph'] = None

        # --- No-graph evaluation ---
        if not skip_no_graph:
            print(f"\n[NO-GRAPH] Computing neighbors on-the-fly (LCS)")

            score, timings, peak_mem = solve_no_graph(n, s, q)

            print(f"  Generate nodes: {format_time(timings['generate'])}")
            print(f"  Evaluation: priority={format_time(timings['priority'])}, "
                  f"sort={format_time(timings['sort'])}, greedy={format_time(timings['greedy'])}")
            print(f"  Total: {format_time(timings['total'])}")
            print(f"  Peak memory: {format_memory(peak_mem)}")
            print(f"  Score: {score}")

            result['no_graph'] = {
                'total': timings['total'],
                'greedy': timings['greedy'],
                'score': score,
                'peak_memory_mb': peak_mem,
            }
        else:
            result['no_graph'] = None

        # --- Comparison ---
        if result.get('graph') and result.get('no_graph'):
            graph_total = result['graph']['total']
            no_graph_total = result['no_graph']['total']

            if graph_total < no_graph_total:
                speedup = no_graph_total / graph_total
                winner = "GRAPH-BASED"
            else:
                speedup = graph_total / no_graph_total
                winner = "NO-GRAPH"

            print(f"\n  >>> Winner: {winner} ({speedup:.1f}x faster)")
            result['winner'] = winner
            result['speedup'] = speedup

        results.append(result)

        # Save incrementally after each n
        if writer:
            writer.add_result(result)

    # --- Summary ---
    print("\n" + "=" * 110)
    print("  TIMING SUMMARY")
    print("=" * 110)
    print(f"{'n':>4} | {'Nodes':>10} | {'Graph (load+eval)':>22} | {'No-Graph':>22} | {'Winner':>12} | {'Speedup':>8}")
    print("-" * 100)

    for r in results:
        n = r['n']
        nodes = r['num_nodes']

        if r.get('graph'):
            graph_time = format_time(r['graph']['total'])
        else:
            graph_time = "N/A"

        if r.get('no_graph'):
            no_graph_time = format_time(r['no_graph']['total'])
        else:
            no_graph_time = "N/A"

        winner = r.get('winner', '-')
        speedup = f"{r.get('speedup', 0):.1f}x" if r.get('speedup') else "-"

        print(f"{n:>4} | {nodes:>10,} | {graph_time:>22} | {no_graph_time:>22} | {winner:>12} | {speedup:>8}")

    # --- Memory Summary ---
    print("\n" + "=" * 110)
    print("  MEMORY SUMMARY (Peak Memory Usage in GiB)")
    print("=" * 110)
    print(f"{'n':>4} | {'Nodes':>10} | {'Graph-based':>15} | {'No-Graph':>15} | {'Ratio':>10}")
    print("-" * 70)

    for r in results:
        n = r['n']
        nodes = r['num_nodes']

        if r.get('graph') and 'peak_memory_mb' in r['graph']:
            graph_mem = format_memory(r['graph']['peak_memory_mb'])
        else:
            graph_mem = "N/A"

        if r.get('no_graph') and 'peak_memory_mb' in r['no_graph']:
            no_graph_mem = format_memory(r['no_graph']['peak_memory_mb'])
        else:
            no_graph_mem = "N/A"

        # Calculate ratio
        if (r.get('graph') and 'peak_memory_mb' in r['graph'] and
            r.get('no_graph') and 'peak_memory_mb' in r['no_graph'] and
            r['no_graph']['peak_memory_mb'] > 0):
            ratio = r['graph']['peak_memory_mb'] / r['no_graph']['peak_memory_mb']
            ratio_str = f"{ratio:.1f}x"
        else:
            ratio_str = "-"

        print(f"{n:>4} | {nodes:>10,} | {graph_mem:>15} | {no_graph_mem:>15} | {ratio_str:>10}")

    print("=" * 110)

    return results


class ResultsWriter:
    """Incrementally writes profiling results to JSON and text files after each n."""

    def __init__(self, output_path, s, q, graph_type):
        # Ensure output path has .json extension
        if not output_path.endswith('.json'):
            output_path = output_path + '.json'

        self.json_path = output_path
        self.txt_path = output_path.replace('.json', '.txt')
        self.s = s
        self.q = q
        self.graph_type = graph_type
        self.results = []
        self.start_time = datetime.now().isoformat()

    def add_result(self, result):
        """Add a result and immediately write to files."""
        self.results.append(result)
        self._write_files()
        print(f"  [SAVED] Results updated: {self.json_path}")

    def _write_files(self):
        """Write current results to JSON and text files."""
        # Write JSON file
        output_data = {
            'timestamp': self.start_time,
            'last_updated': datetime.now().isoformat(),
            'parameters': {
                's': self.s,
                'q': self.q,
                'graph_type': self.graph_type,
            },
            'results': self.results,
        }

        with open(self.json_path, 'w') as f:
            std_json.dump(output_data, f, indent=2)

        # Write human-readable text summary
        with open(self.txt_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("  Evaluation Profiling Results\n")
            f.write("=" * 90 + "\n")
            f.write(f"  Started: {self.start_time}\n")
            f.write(f"  Last updated: {output_data['last_updated']}\n")
            f.write(f"  Parameters: s={self.s}, q={self.q}, type={self.graph_type}\n")
            f.write("=" * 90 + "\n\n")

            # Timing summary
            f.write("TIMING SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'n':>4} | {'Nodes':>12} | {'Graph':>22} | {'No-Graph':>22} | {'Winner':>12} | {'Speedup':>8}\n")
            f.write("-" * 100 + "\n")

            for r in self.results:
                n = r['n']
                nodes = r['num_nodes']

                graph_time = format_time(r['graph']['total']) if r.get('graph') else "N/A"
                no_graph_time = format_time(r['no_graph']['total']) if r.get('no_graph') else "N/A"
                winner = r.get('winner', '-')
                speedup = f"{r.get('speedup', 0):.1f}x" if r.get('speedup') else "-"

                f.write(f"{n:>4} | {nodes:>12,} | {graph_time:>22} | {no_graph_time:>22} | {winner:>12} | {speedup:>8}\n")

            f.write("\n")

            # Memory summary
            f.write("MEMORY SUMMARY (Peak GiB)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'n':>4} | {'Nodes':>12} | {'Graph (GiB)':>15} | {'No-Graph (GiB)':>15} | {'Ratio':>10}\n")
            f.write("-" * 80 + "\n")

            for r in self.results:
                n = r['n']
                nodes = r['num_nodes']

                if r.get('graph') and 'peak_memory_mb' in r['graph']:
                    graph_mem = format_memory(r['graph']['peak_memory_mb'])
                else:
                    graph_mem = "N/A"

                if r.get('no_graph') and 'peak_memory_mb' in r['no_graph']:
                    no_graph_mem = format_memory(r['no_graph']['peak_memory_mb'])
                else:
                    no_graph_mem = "N/A"

                # Calculate ratio
                if (r.get('graph') and 'peak_memory_mb' in r['graph'] and
                    r.get('no_graph') and 'peak_memory_mb' in r['no_graph'] and
                    r['no_graph']['peak_memory_mb'] > 0):
                    ratio = r['graph']['peak_memory_mb'] / r['no_graph']['peak_memory_mb']
                    ratio_str = f"{ratio:.1f}x"
                else:
                    ratio_str = "-"

                f.write(f"{n:>4} | {nodes:>12,} | {graph_mem:>15} | {no_graph_mem:>15} | {ratio_str:>10}\n")

            f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Profile evaluation: graph-based vs no-graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--n-values', type=str, default='6,7,8,9,10,11',
                        help='Comma-separated n values (default: 6,7,8)')
    parser.add_argument('--s', type=int, default=1,
                        help='Number of errors to correct (default: 1)')
    parser.add_argument('--q', type=int, default=4,
                        help='Alphabet size (default: 4 for quaternary)')
    parser.add_argument('--graph-type', choices=['deletion', 'ids'], default='ids',
                        help='Graph type (default: ids)')
    parser.add_argument('--graph-dir', default=DEFAULT_GRAPH_DIR,
                        help=f'Graph directory (default: {DEFAULT_GRAPH_DIR})')
    parser.add_argument('--no-graph-only', action='store_true',
                        help='Only profile no-graph mode (skip graph loading)')
    parser.add_argument('--graph-only', action='store_true',
                        help='Only profile graph-based mode (skip no-graph)')
    parser.add_argument('--output', '-o', type=str, default='auto',
                        help='Output file path for results (JSON format). '
                             'Also creates a .txt summary. Default: auto-generated filename')
    parser.add_argument('--no-output', action='store_true',
                        help='Disable file output (console only)')

    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(',')]

    # Determine output path
    if args.no_output:
        output_path = None
    elif args.output == 'auto':
        # Auto-generate filename based on parameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"./profile_s{args.s}_q{args.q}_{args.graph_type}_{timestamp}"
    else:
        output_path = args.output

    profile_comparison(
        n_values=n_values,
        s=args.s,
        q=args.q,
        graph_type=args.graph_type,
        graph_dir=args.graph_dir,
        skip_graph=args.no_graph_only,
        skip_no_graph=args.graph_only,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
