#!/usr/bin/env python3
"""
Profiling script for all evaluation variants.

Profiles and compares:
- no_graph.py: On-the-fly neighbor computation
- graph_networkx.py: NetworkX graphs with simple API
- graph_gt.py: graph-tool with fast C++ backend

Configure the variants to profile and test parameters below, then run:
    python analysis/profile_evaluations.py
"""

import sys
import os
import time
import importlib.util
from pathlib import Path
import lmdb

# Add src to path so we can import evaluation scripts
sys.path.insert(0, '/workspace/DistributedFunSearch/src')

# ============================================================================
# Configuration
# ============================================================================

# Which variants to profile (set to True to enable)
PROFILE_NO_GRAPH = True
PROFILE_NETWORKX = True
PROFILE_GRAPH_TOOL = True

# Path to graph directory containing .lmdb files
GRAPH_DIR = "/workspace/DistributedFunSearch/src/graphs"

# Path to specification file containing priority function
# Use zero.txt for baseline (all nodes equal priority)
SPEC_PATH = "/workspace/DistributedFunSearch/src/disfun/specifications/IDS/initial_functions/no_graph/zero.txt"

# Problem parameters to test
S_VALUES = [1]           # Error correction parameter(s)
N_VALUES = [6,7,8,9,10,11]     # String length parameter(s)
Q = 4                   # Alphabet size

# Cache loaded graphs in memory (only applies to graph variants)
CACHE_GRAPHS = True

# ============================================================================


def load_evaluation_module(eval_script_path):
    """Dynamically load evaluation script as module."""
    spec = importlib.util.spec_from_file_location("eval_module", eval_script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_priority_function(spec_path):
    """Load priority function from specification file."""
    with open(spec_path, 'r') as f:
        spec_content = f.read()

    # Extract code from <code> tags if present
    import re
    code_match = re.search(r'<code>(.*?)</code>', spec_content, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = spec_content

    namespace = {}
    exec(code, namespace)

    return namespace['priority']


def profile_variant(variant_name, eval_module, priority_func, n, s, q, graph_dir):
    """Profile a single evaluation variant for given parameters."""
    print(f"\n{'='*80}")
    print(f"Profiling: {variant_name} | n={n}, s={s}, q={q}")
    print(f"{'='*80}")

    # Detect expected signature and wrap priority function if needed
    import inspect
    existing_priority = getattr(eval_module, 'priority', None)
    if existing_priority:
        sig = inspect.signature(existing_priority)
        num_params = len(sig.parameters)

        # Wrap priority function to match expected signature
        if num_params == 4:  # no_graph: (node, n, s, q)
            eval_module.priority = lambda node, n, s, q: priority_func(node, n, s, q)
        elif num_params == 5:  # graph_networkx: (node, G, n, s, q)
            eval_module.priority = lambda node, G, n, s, q: priority_func(node, n, s, q)
        elif num_params == 7:  # graph_gt: (node, G, ntv, vtn, n, s, q)
            eval_module.priority = lambda node, G, ntv, vtn, n, s, q: priority_func(node, n, s, q)
        else:
            # Unknown signature, just assign directly
            eval_module.priority = priority_func
    else:
        # No existing priority function, just assign
        eval_module.priority = priority_func

    # Inject start_n for hash computation
    eval_module.start_n = n

    # Inject cache configuration (for graph variants)
    if hasattr(eval_module, '_GRAPH_CACHE'):
        eval_module.CACHE_GRAPHS = CACHE_GRAPHS
        eval_module.CACHE_SIZE_LIMIT_GB = 2.0

    try:
        num_nodes = q ** n
        num_edges = None
        graph_load_time = 0.0

        # For graph variants, show file size
        if variant_name in ['graph_networkx', 'graph_gt']:
            path = os.path.join(graph_dir, f"graph_ids_s{s}_n{n}_q{q}.lmdb")

            if os.path.exists(path):
                # Get file size
                if os.path.isdir(path):
                    data_file = os.path.join(path, 'data.mdb')
                    if os.path.exists(data_file):
                        file_size_bytes = os.path.getsize(data_file)
                    else:
                        file_size_bytes = 0
                else:
                    file_size_bytes = os.path.getsize(path)

                if file_size_bytes >= 1024**3:
                    file_size_str = f"{file_size_bytes / (1024**3):.2f} GiB"
                else:
                    file_size_str = f"{file_size_bytes / (1024**2):.2f} MiB"

                print(f"  Graph file: {file_size_str}")
        else:
            print(f"  Nodes: {num_nodes} (q^n = {q}^{n})")

        # Time the full solve operation (includes graph loading for graph variants)
        solve_start = time.time()
        independent_set, hash_value = eval_module.solve(n, s, q, graph_dir)
        total_solve_time = time.time() - solve_start
        set_size = len(independent_set)

        # For graph variants, measure loading time separately to estimate algorithm-only time
        if variant_name in ['graph_networkx', 'graph_gt']:
            load_start = time.time()
            if variant_name == 'graph_networkx':
                G = eval_module.load_graph(path)
                num_edges = G.number_of_edges()
            elif variant_name == 'graph_gt':
                G, node_to_vertex, vertex_to_node = eval_module.load_graph(path)
                num_edges = G.num_edges()
            graph_load_time = time.time() - load_start

            # Algorithm time is approximately (solve_time - load_time)
            # This assumes solve() did similar loading work
            algorithm_time = max(0, total_solve_time - graph_load_time)

            print(f"  Graph loading: ~{graph_load_time:.4f}s")
            print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
            print(f"  Algorithm (priority + greedy): ~{algorithm_time:.4f}s")
            print(f"  Total time: {total_solve_time:.4f}s")
        else:
            # no_graph: no separate loading
            print(f"  Algorithm (priority + greedy): {total_solve_time:.4f}s")
            print(f"  Total time: {total_solve_time:.4f}s")

        print(f"  Independent set size: {set_size}")

        return {
            'variant': variant_name,
            'n': n,
            's': s,
            'q': q,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'set_size': set_size,
            'graph_load_time': graph_load_time,
            'algorithm_time': total_solve_time - graph_load_time if graph_load_time > 0 else total_solve_time,
            'total_time': total_solve_time,
            'success': True
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {
            'variant': variant_name,
            'n': n,
            's': s,
            'q': q,
            'set_size': None,
            'total_time': None,
            'success': False,
            'error': str(e)
        }


def main():
    print(f"\n{'#'*80}")
    print(f"# Evaluation Variants Performance Profiling")
    print(f"{'#'*80}")
    print(f"\nConfiguration:")
    print(f"  Specification: {SPEC_PATH}")
    print(f"  Graph directory: {GRAPH_DIR}")
    print(f"  s_values: {S_VALUES}")
    print(f"  n_values: {N_VALUES}")
    print(f"  q: {Q}")
    print(f"\nVariants to profile:")
    if PROFILE_NO_GRAPH:
        print(f" no_graph (on-the-fly computation)")
    if PROFILE_NETWORKX:
        print(f" graph_networkx (NetworkX graphs)")
    if PROFILE_GRAPH_TOOL:
        print(f" graph_gt (graph-tool C++ backend)")

    # Load evaluation modules
    eval_base = "/workspace/DistributedFunSearch/src/disfun/specifications/IDS/evaluation"
    modules = {}

    if PROFILE_NO_GRAPH:
        try:
            modules['no_graph'] = load_evaluation_module(f"{eval_base}/no_graph.py")
        except Exception as e:
            print(f"no_graph: Failed to load ({e})")

    if PROFILE_NETWORKX:
        try:
            modules['graph_networkx'] = load_evaluation_module(f"{eval_base}/graph_networkx.py")
        except Exception as e:
            print(f"graph_networkx: Failed to load ({e})")

    if PROFILE_GRAPH_TOOL:
        try:
            modules['graph_gt'] = load_evaluation_module(f"{eval_base}/graph_gt.py")
        except Exception as e:
            print(f"graph_gt: Failed to load ({e})")

    if not modules:
        print("\nError: No evaluation modules could be loaded. Check dependencies.")
        sys.exit(1)

    # Load priority function
    print(f"\nLoading priority function from: {SPEC_PATH}")
    priority_func = load_priority_function(SPEC_PATH)

    # Profile all combinations
    results = []
    for s in S_VALUES:
        for n in N_VALUES:
            for variant_name, eval_module in modules.items():
                result = profile_variant(
                    variant_name, eval_module, priority_func,
                    n, s, Q, GRAPH_DIR
                )
                results.append(result)

    # Print comparison table
    print(f"\n{'='*80}")
    print("Performance Comparison")
    print(f"{'='*80}\n")

    # Group by (n, s) for easy comparison
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        if r['success']:
            grouped[(r['n'], r['s'])].append(r)

    for (n, s), group in sorted(grouped.items()):
        print(f"\nn={n}, s={s}:")
        print(f"  {'Variant':<20} {'Set Size':<12} {'Time (s)':<12} {'Speedup':<10}")
        print(f"  {'-'*54}")

        # Sort by time (fastest first)
        group_sorted = sorted(group, key=lambda x: x['total_time'] if x['total_time'] else float('inf'))
        baseline_time = group_sorted[0]['total_time'] if group_sorted else None

        for r in group_sorted:
            speedup = f"{baseline_time / r['total_time']:.2f}x" if baseline_time and r['total_time'] else "n/a"
            print(f"  {r['variant']:<20} {r['set_size']:<12} {r['total_time']:<12.4f} {speedup:<10}")

    # Detailed results table
    print(f"\n{'='*80}")
    print("Detailed Timing Breakdown")
    print(f"{'='*80}\n")

    print(f"{'Variant':<20} {'n':<4} {'s':<4} {'Nodes':<8} {'Load (s)':<10} {'Algo (s)':<10} {'Total (s)':<10} {'Set':<5}")
    print("-" * 80)

    for r in results:
        if r['success']:
            load_str = f"{r['graph_load_time']:.4f}" if r['graph_load_time'] > 0 else "-"
            algo_str = f"{r['algorithm_time']:.4f}"
            print(f"{r['variant']:<20} {r['n']:<4} {r['s']:<4} {r['num_nodes']:<8} {load_str:<10} "
                  f"{algo_str:<10} {r['total_time']:<10.4f} {r['set_size']:<5}")
        else:
            print(f"{r['variant']:<20} {r['n']:<4} {r['s']:<4} {'Error':<8} {r.get('error', 'Unknown')[:40]}")

    print("\n" + "="*80 + "\n")

    # Generate LaTeX table
    print("="*80)
    print("LaTeX Table")
    print("="*80 + "\n")

    # Organize results by (variant, n, s)
    result_dict = {}
    for r in results:
        if r['success']:
            key = (r['variant'], r['n'], r['s'])
            result_dict[key] = r

    # Get unique n and s values
    n_values = sorted(set(r['n'] for r in results if r['success']))
    s_values = sorted(set(r['s'] for r in results if r['success']))
    variants = ['no_graph', 'graph_networkx', 'graph_gt']
    variant_names = ['Scratch', 'NX', 'C++']

    # Helper to get value or dash
    def get_value(variant, n, s, metric):
        key = (variant, n, s)
        if key not in result_dict:
            return '-'
        r = result_dict[key]
        if metric == 'load':
            return f"{r['graph_load_time']:.4f}" if r['graph_load_time'] > 0 else '-'
        elif metric == 'algo':
            return f"{r['algorithm_time']:.4f}"
        elif metric == 'total':
            return f"{r['total_time']:.4f}"
        return '-'

    # One comprehensive table
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\resizebox{\\columnwidth}{!}{%")

    # Build column spec: first column for metric name, then 3 columns per n value with vertical lines
    col_spec = "l|" + "|".join(["ccc"] * len(n_values)) + "|"
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\hline")

    # Header row 1: n values with multicolumn (include vertical line on right)
    header1 = ""
    for n in n_values:
        header1 += f" & \\multicolumn{{3}}{{c|}}{{n={n}}}"
    header1 += " \\\\"
    print(header1)

    # Header row 2: variant abbreviations
    header2 = ""
    for n in n_values:
        header2 += f" & {variant_names[0]} & {variant_names[1]} & {variant_names[2]}"
    header2 += " \\\\"
    print(header2)
    print("\\hline")

    # Data rows grouped by s
    for s in s_values:
        if s == 1:
            caption = "Single deletion correction (s=1)"
        elif s == 2:
            caption = "Two deletion correction (s=2)"
        else:
            caption = f"s={s}"

        print(f"\\multicolumn{{{len(n_values)*3+1}}}{{c}}{{{caption}}} \\\\")
        print("\\hline")

        # Loading row
        row = "Loading"
        for n in n_values:
            for variant in variants:
                row += f" & {get_value(variant, n, s, 'load')}"
        row += " \\\\"
        print(row)

        # Construction row
        row = "Construction"
        for n in n_values:
            for variant in variants:
                row += f" & {get_value(variant, n, s, 'algo')}"
        row += " \\\\"
        print(row)

        # Total row
        row = "Total"
        for n in n_values:
            for variant in variants:
                row += f" & {get_value(variant, n, s, 'total')}"
        row += " \\\\"
        print(row)

        if s != s_values[-1]:
            print("\\hline")

    print("\\hline")
    print("\\end{tabular}%")
    print("}")
    print("\\caption{Performance comparison (seconds) showing loading, construction, and total times for different variants and problem sizes.}")
    print("\\label{tab:performance}")
    print("\\end{table}")
    print()


if __name__ == "__main__":
    main()
