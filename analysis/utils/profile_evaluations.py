#!/usr/bin/env python3
"""
Profile Evaluation Performance

Compares three evaluation approaches:
1. NetworkX: Load LMDB,  NetworkX Graph, Python greedy (baseline)
2. FastGraph C++: Load LMDB, CSR format, C++ greedy (optimized)
3. No-graph: Generate nodes, compute LCS neighbors on-the-fly

Usage:
    python profile_evaluations.py                           # All methods, IDS q=4
    python profile_evaluations.py --q 2 --graph-type deletion  # Deletion binary
    python profile_evaluations.py --fastgraph-only          # Only FastGraph C++
    python profile_evaluations.py --n-values 6,7,8 --no-output  # Specific n, no file
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

# Optional import for C++ FastGraph
try:
    from disfun.utils.fast_graph import FastGraph, load_graph_from_lmdb, USING_CPP
    HAS_FASTGRAPH = USING_CPP
except ImportError:
    HAS_FASTGRAPH = False


# =============================================================================
# Priority functions (simple O(1) for fair comparison)
# =============================================================================
def priority_simple(node, *args):
    """O(1) priority - focuses timing on graph operations, not priority computation."""
    return 0.0


# =============================================================================
# C++ FastGraph evaluation (memory-efficient CSR format)
# =============================================================================
def load_fastgraph_with_memory(graph_db_path):
    """Load graph using C++ FastGraph. Returns (G, peak_memory_mb, mem_after)."""
    gc.collect()
    mem_before = get_process_memory_mb()
    tracemalloc.start()

    G = load_graph_from_lmdb(graph_db_path)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_after = get_process_memory_mb()
    process_delta = mem_after - mem_before
    peak_mb = max(peak / (1024 * 1024), process_delta)

    return G, peak_mb, mem_after


def solve_with_fastgraph(G, n, s):
    """Evaluate using C++ FastGraph with built-in greedy solver. Returns (score, timings)."""
    process = psutil.Process(os.getpid())

    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()

    # Priority computation
    t0 = time.perf_counter()
    priorities = {node: priority_simple(node) for node in G.nodes}
    t_priority = time.perf_counter() - t0

    # C++ greedy independent set (includes sort internally)
    t0 = time.perf_counter()
    independent_set = G.greedy_independent_set(priorities)
    t_greedy = time.perf_counter() - t0

    cpu_end = process.cpu_times()
    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start
    cpu_time = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)

    return len(independent_set), {
        'priority': t_priority,
        'sort': 0.0,  # Included in greedy
        'greedy': t_greedy,
        'total': t_priority + t_greedy,
        'cpu': cpu_time,
        'wall': wall_time,
    }


# =============================================================================
# NetworkX Graph-based evaluation (loads pre-computed graph)
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
def profile_comparison(n_values, s, q, graph_type, graph_dir, skip_graph=False, skip_no_graph=False, skip_fastgraph=False, output_path=None):
    """Profile all approaches and compare."""

    print("=" * 90)
    print("  Evaluation Profiling: NetworkX vs FastGraph (C++) vs No-Graph")
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

                # Free the graph to not affect other memory measurements
                del G
                gc.collect()
            else:
                print(f"\n[NETWORKX] Graph not found for n={n}")
                result['graph'] = None
        elif skip_graph:
            print(f"\n[NETWORKX] Skipped (--skip-networkx)")
            result['graph'] = None
        else:
            print(f"\n[NETWORKX] Skipped (missing lmdb/networkx)")
            result['graph'] = None

        # --- C++ FastGraph evaluation ---
        if not skip_fastgraph and HAS_FASTGRAPH:
            path = get_graph_path(graph_type, s, n, q, graph_dir=graph_dir)

            if path:
                print(f"\n[FASTGRAPH C++] Loading: {path}")

                # Time graph loading with memory tracking
                t0 = time.perf_counter()
                G, peak_mem_load, mem_after_load = load_fastgraph_with_memory(path)
                t_load = time.perf_counter() - t0

                num_edges = G.number_of_edges()
                print(f"  Graph load: {format_time(t_load)} ({num_edges:,} edges)")
                print(f"  Peak memory (load): {format_memory(peak_mem_load)}")
                print(f"  Process memory after load: {format_memory(mem_after_load)}")

                # Time evaluation
                score, timings = solve_with_fastgraph(G, n, s)

                print(f"  Evaluation: priority={format_time(timings['priority'])}, "
                      f"greedy={format_time(timings['greedy'])}")
                print(f"  Total (eval only): {format_time(timings['total'])}")
                print(f"  Total (with load): {format_time(t_load + timings['total'])}")
                print(f"  Score: {score}")

                result['fastgraph'] = {
                    'load': t_load,
                    'eval': timings['total'],
                    'total': t_load + timings['total'],
                    'score': score,
                    'edges': num_edges,
                    'peak_memory_mb': peak_mem_load,
                    'process_memory_mb': mem_after_load,
                }

                # Free the graph to not affect other memory measurements
                del G
                gc.collect()
            else:
                print(f"\n[FASTGRAPH C++] Graph not found for n={n}")
                result['fastgraph'] = None
        elif skip_fastgraph:
            print(f"\n[FASTGRAPH C++] Skipped (--skip-fastgraph)")
            result['fastgraph'] = None
        else:
            print(f"\n[FASTGRAPH C++] Skipped (C++ module not available)")
            result['fastgraph'] = None

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
    print("\n" + "=" * 130)
    print("  TIMING SUMMARY")
    print("=" * 130)
    print(f"{'n':>4} | {'Nodes':>10} | {'NetworkX':>18} | {'FastGraph C++':>18} | {'No-Graph':>18} | {'Best':>12}")
    print("-" * 130)

    for r in results:
        n = r['n']
        nodes = r['num_nodes']

        nx_time = format_time(r['graph']['total']) if r.get('graph') else "N/A"
        fg_time = format_time(r['fastgraph']['total']) if r.get('fastgraph') else "N/A"
        ng_time = format_time(r['no_graph']['total']) if r.get('no_graph') else "N/A"

        # Find best
        times = []
        if r.get('graph'): times.append(('NetworkX', r['graph']['total']))
        if r.get('fastgraph'): times.append(('FastGraph', r['fastgraph']['total']))
        if r.get('no_graph'): times.append(('No-Graph', r['no_graph']['total']))
        best = min(times, key=lambda x: x[1])[0] if times else "-"

        print(f"{n:>4} | {nodes:>10,} | {nx_time:>18} | {fg_time:>18} | {ng_time:>18} | {best:>12}")

    # --- Memory Summary ---
    print("\n" + "=" * 100)
    print("  MEMORY SUMMARY (Peak Memory Usage in GiB)")
    print("=" * 100)
    print(f"{'n':>4} | {'Nodes':>10} | {'NetworkX':>15} | {'FastGraph C++':>15} | {'No-Graph':>15}")
    print("-" * 100)

    for r in results:
        n = r['n']
        nodes = r['num_nodes']

        nx_mem = format_memory(r['graph']['peak_memory_mb']) if r.get('graph') and 'peak_memory_mb' in r['graph'] else "N/A"
        fg_mem = format_memory(r['fastgraph']['peak_memory_mb']) if r.get('fastgraph') and 'peak_memory_mb' in r['fastgraph'] else "N/A"
        ng_mem = format_memory(r['no_graph']['peak_memory_mb']) if r.get('no_graph') and 'peak_memory_mb' in r['no_graph'] else "N/A"

        print(f"{n:>4} | {nodes:>10,} | {nx_mem:>15} | {fg_mem:>15} | {ng_mem:>15}")

    print("=" * 100)

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
            f.write("-" * 120 + "\n")
            f.write(f"{'n':>4} | {'Nodes':>12} | {'NetworkX':>18} | {'FastGraph C++':>18} | {'No-Graph':>18} | {'Best':>12}\n")
            f.write("-" * 120 + "\n")

            for r in self.results:
                n = r['n']
                nodes = r['num_nodes']

                nx_time = format_time(r['graph']['total']) if r.get('graph') else "N/A"
                fg_time = format_time(r['fastgraph']['total']) if r.get('fastgraph') else "N/A"
                ng_time = format_time(r['no_graph']['total']) if r.get('no_graph') else "N/A"

                # Find best
                times = []
                if r.get('graph'): times.append(('NetworkX', r['graph']['total']))
                if r.get('fastgraph'): times.append(('FastGraph', r['fastgraph']['total']))
                if r.get('no_graph'): times.append(('No-Graph', r['no_graph']['total']))
                best = min(times, key=lambda x: x[1])[0] if times else "-"

                f.write(f"{n:>4} | {nodes:>12,} | {nx_time:>18} | {fg_time:>18} | {ng_time:>18} | {best:>12}\n")

            f.write("\n")

            # Memory summary
            f.write("MEMORY SUMMARY (Peak GiB)\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'n':>4} | {'Nodes':>12} | {'NetworkX':>15} | {'FastGraph C++':>15} | {'No-Graph':>15}\n")
            f.write("-" * 100 + "\n")

            for r in self.results:
                n = r['n']
                nodes = r['num_nodes']

                nx_mem = format_memory(r['graph']['peak_memory_mb']) if r.get('graph') and 'peak_memory_mb' in r['graph'] else "N/A"
                fg_mem = format_memory(r['fastgraph']['peak_memory_mb']) if r.get('fastgraph') and 'peak_memory_mb' in r['fastgraph'] else "N/A"
                ng_mem = format_memory(r['no_graph']['peak_memory_mb']) if r.get('no_graph') and 'peak_memory_mb' in r['no_graph'] else "N/A"

                f.write(f"{n:>4} | {nodes:>12,} | {nx_mem:>15} | {fg_mem:>15} | {ng_mem:>15}\n")

            f.write("=" * 100 + "\n")


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
    parser.add_argument('--skip-networkx', action='store_true',
                        help='Skip NetworkX profiling')
    parser.add_argument('--skip-fastgraph', action='store_true',
                        help='Skip FastGraph C++ profiling')
    parser.add_argument('--skip-no-graph', action='store_true',
                        help='Skip no-graph (on-the-fly LCS) profiling')
    parser.add_argument('--fastgraph-only', action='store_true',
                        help='Only profile FastGraph C++ mode')
    parser.add_argument('--output', '-o', type=str, default='auto',
                        help='Output file path for results (JSON format). '
                             'Also creates a .txt summary. Default: auto-generated filename')
    parser.add_argument('--no-output', action='store_true',
                        help='Disable file output (console only)')

    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(',')]

    # Determine which modes to run
    skip_graph = args.skip_networkx or args.fastgraph_only
    skip_fastgraph = args.skip_fastgraph
    skip_no_graph = args.skip_no_graph or args.fastgraph_only

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
        skip_graph=skip_graph,
        skip_no_graph=skip_no_graph,
        skip_fastgraph=skip_fastgraph,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
