#!/usr/bin/env python3
"""
Step 3: Evaluate Functions on Extended Inputs

This script takes the deduplicated functions from extract_successful_functions.py
and evaluates them on larger inputs (n=6 through n=16).

It automatically detects the function signature and uses the appropriate evaluation:
- no_graph: priority(node, n, s, q)
- graph_gt: priority(node, G_gt, node_to_vertex, vertex_to_node, n, s)
- graph_networkx: priority(node, G, n, s)

Features:
- Incremental checkpoint saving after each function (crash-safe)
- Resume from checkpoint with --resume
- Real-time logging of codebook sizes

Usage:
    python test/evaluate.py <functions_json> [options]

Examples:
    python test/evaluate.py ./successful_functions/successful_functions.json --max-n 16
    python test/evaluate.py ./successful_functions/successful_functions.json --resume
"""

import argparse
import json
import sys
import itertools
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    SIGNATURE_NO_GRAPH,
    SIGNATURE_GRAPH_GT,
    SIGNATURE_GRAPH_NETWORKX,
    COMMON_IMPORTS,
    detect_signature,
    are_neighbors,
    build_graph_networkx,
    build_graph_gt,
)


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(checkpoint_path: Path, results: Dict[Tuple[int, int], Dict],
                    completed_pairs: Set[Tuple[int, int]], metadata: Dict) -> None:
    """
    Save checkpoint atomically (write to temp file, then rename).

    Args:
        checkpoint_path: Path to checkpoint file
        results: Dict mapping (func_index, n) -> result
        completed_pairs: Set of completed (func_index, n) pairs
        metadata: Dict with run metadata (n_values, s, q, etc.)
    """
    # Convert tuple keys to strings for JSON
    checkpoint_data = {
        'metadata': metadata,
        'completed_pairs': [list(p) for p in sorted(completed_pairs)],
        'results': {f"{k[0]}_{k[1]}": v for k, v in results.items()},
        'last_updated': datetime.now().isoformat(),
    }

    # Write atomically
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f)
    temp_path.rename(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> Tuple[Dict[Tuple[int, int], Dict], Set[Tuple[int, int]], Dict]:
    """
    Load checkpoint from file.

    Returns:
        (results, completed_pairs, metadata)
    """
    with open(checkpoint_path) as f:
        data = json.load(f)

    # Convert string keys back to tuples
    results = {}
    for k, v in data.get('results', {}).items():
        parts = k.split('_')
        key = (int(parts[0]), int(parts[1]))
        results[key] = v

    completed_pairs = set(tuple(p) for p in data.get('completed_pairs', []))
    metadata = data.get('metadata', {})

    return results, completed_pairs, metadata


def log_progress(log_path: Path, func_index: int, n: int, size: int,
                 elapsed: float, signature: str) -> None:
    """Append progress to log file for a single (func, n) evaluation."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] Func {func_index:4d} n={n:2d} | size={size:4d} | {elapsed:.1f}s | {signature}\n"

    with open(log_path, 'a') as f:
        f.write(log_line)


# =============================================================================
# Priority Function Creation
# =============================================================================

def create_priority_function(body: str, signature: str):
    """Create a callable priority function from body string with appropriate signature."""
    clean_body = body.strip()
    if not clean_body.startswith('    ') and not clean_body.startswith('\t'):
        clean_body = textwrap.indent(clean_body, '    ')

    if signature == SIGNATURE_NO_GRAPH:
        func_def = "def priority(node, n, s, q):"
    elif signature == SIGNATURE_GRAPH_GT:
        func_def = "def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):"
    elif signature == SIGNATURE_GRAPH_NETWORKX:
        func_def = "def priority(node, G, n, s):"
    else:
        raise ValueError(f"Unknown signature: {signature}")

    func_code = COMMON_IMPORTS + f"""
{func_def}
{clean_body}
"""

    namespace = {}
    exec(func_code, namespace)
    return namespace['priority']


def solve_no_graph(priority_func, nodes: List[str], n: int, s: int, q: int) -> Tuple[Set[str], Dict[str, float]]:
    """Solve using no_graph signature."""
    priorities = {}
    for node in nodes:
        try:
            p = priority_func(node, n, s, q)
            if p is None:
                p = 0.0
            priorities[node] = float(p)
        except Exception:
            priorities[node] = 0.0

    return greedy_independent_set(nodes, priorities, n, s)


def solve_graph_networkx(priority_func, nodes: List[str], n: int, s: int, q: int) -> Tuple[Set[str], Dict[str, float]]:
    """Solve using graph_networkx signature."""
    G = build_graph_networkx(nodes, n, s)

    priorities = {}
    for node in nodes:
        try:
            p = priority_func(node, G, n, s)
            if p is None:
                p = 0.0
            priorities[node] = float(p)
        except Exception:
            priorities[node] = 0.0

    return greedy_independent_set_with_graph_nx(nodes, priorities, G)


def solve_graph_gt(priority_func, nodes: List[str], n: int, s: int, q: int,
                   graph_dir: str = None) -> Tuple[Set[str], Dict[str, float]]:
    """Solve using graph_gt signature."""
    G_gt, node_to_vertex, vertex_to_node = build_graph_gt(nodes, n, s, graph_dir)

    priorities = {}
    for node in nodes:
        try:
            p = priority_func(node, G_gt, node_to_vertex, vertex_to_node, n, s)
            if p is None:
                p = 0.0
            priorities[node] = float(p)
        except Exception:
            priorities[node] = 0.0

    return greedy_independent_set_with_graph_gt(nodes, priorities, G_gt, node_to_vertex)


def greedy_independent_set(nodes: List[str], priorities: Dict[str, float],
                           n: int, s: int) -> Tuple[Set[str], Dict[str, float]]:
    """Build independent set using greedy algorithm (computing neighbors on-the-fly)."""
    nodes_sorted = sorted(nodes, key=lambda x: (-priorities[x], x))

    independent_set = set()
    removed_nodes = set()

    for node in nodes_sorted:
        if node in removed_nodes:
            continue

        independent_set.add(node)
        removed_nodes.add(node)

        for other_node in nodes:
            if other_node not in removed_nodes:
                if are_neighbors(node, other_node, n, s):
                    removed_nodes.add(other_node)

    return independent_set, priorities


def greedy_independent_set_with_graph_nx(nodes: List[str], priorities: Dict[str, float],
                                          G) -> Tuple[Set[str], Dict[str, float]]:
    """Build independent set using greedy algorithm with NetworkX graph."""
    nodes_sorted = sorted(nodes, key=lambda x: (-priorities[x], x))

    independent_set = set()
    removed_nodes = set()

    for node in nodes_sorted:
        if node in removed_nodes:
            continue

        independent_set.add(node)
        removed_nodes.add(node)

        # Remove neighbors using graph
        for neighbor in G.neighbors(node):
            removed_nodes.add(neighbor)

    return independent_set, priorities


def greedy_independent_set_with_graph_gt(nodes: List[str], priorities: Dict[str, float],
                                          G_gt, node_to_vertex: Dict) -> Tuple[Set[str], Dict[str, float]]:
    """Build independent set using greedy algorithm with graph-tool graph."""
    nodes_sorted = sorted(nodes, key=lambda x: (-priorities[x], x))

    vertex_to_node = {v: k for k, v in node_to_vertex.items()}
    independent_set = set()
    removed_nodes = set()

    for node in nodes_sorted:
        if node in removed_nodes:
            continue

        independent_set.add(node)
        removed_nodes.add(node)

        # Remove neighbors using graph
        v = node_to_vertex[node]
        for neighbor_v in G_gt.vertex(v).out_neighbors():
            neighbor_node = vertex_to_node[int(neighbor_v)]
            removed_nodes.add(neighbor_node)

    return independent_set, priorities


def solve_with_priority(priority_func, signature: str, n: int, s: int, q: int,
                        graph_dir: str = None) -> Tuple[Set[str], Dict[str, float]]:
    """
    Find independent set using given priority function with detected signature.

    Args:
        graph_dir: Optional path to directory with pre-computed graphs (LMDB format)

    Returns:
        - independent_set: Set of codewords
        - priorities: Dict mapping all nodes to their priority values
    """
    # Generate all q-ary strings of length n
    nodes = [''.join(seq) for seq in itertools.product(map(str, range(q)), repeat=n)]

    if signature == SIGNATURE_NO_GRAPH:
        return solve_no_graph(priority_func, nodes, n, s, q)
    elif signature == SIGNATURE_GRAPH_NETWORKX:
        return solve_graph_networkx(priority_func, nodes, n, s, q)
    elif signature == SIGNATURE_GRAPH_GT:
        return solve_graph_gt(priority_func, nodes, n, s, q, graph_dir)
    else:
        raise ValueError(f"Unknown signature: {signature}")


def evaluate_single_n(func_data: Dict, n: int, s: int = 1, q: int = 2,
                      graph_dir: str = None) -> Dict:
    """
    Evaluate a single priority function on a single n value.

    Args:
        func_data: Function data dict with 'body', 'args', etc.
        n: The n value to evaluate
        s, q: Problem parameters
        graph_dir: Optional path to pre-computed graphs

    Returns dict with result for this (func, n) pair.
    """
    body = func_data['body']
    args = func_data.get('args', '')
    func_id = func_data.get('priority_hash', 'unknown')
    func_index = func_data.get('func_index', -1)

    # Detect signature
    signature = detect_signature(body, args)

    result = {
        'func_id': func_id,
        'func_index': func_index,
        'n': n,
        'signature': signature,
        'codebook': [],
        'size': 0,
        'error': None,
    }

    try:
        priority_func = create_priority_function(body, signature)
        codebook, _ = solve_with_priority(priority_func, signature, n, s, q, graph_dir)
        result['codebook'] = sorted(codebook)
        result['size'] = len(codebook)
    except Exception as e:
        result['error'] = str(e)

    return result


def evaluate_single_n_worker(args):
    """Worker function for parallel (func, n) evaluation."""
    func_data, n, s, q, graph_dir = args
    return evaluate_single_n(func_data, n, s, q, graph_dir)


def evaluate_single_function(func_data: Dict, n_values: List[int],
                              s: int = 1, q: int = 2,
                              graph_dir: str = None) -> Dict:
    """
    Evaluate a single priority function on multiple n values.
    (Used for non-checkpointed mode for backward compatibility)

    Args:
        graph_dir: Optional path to directory with pre-computed graphs

    Returns dict with codebooks and sizes for each n.
    """
    body = func_data['body']
    args = func_data.get('args', '')
    func_id = func_data.get('priority_hash', 'unknown')

    # Detect signature
    signature = detect_signature(body, args)

    results = {
        'func_id': func_id,
        'func_index': func_data.get('func_index', -1),
        'signature': signature,
        'codebooks': {},
        'sizes': {},
        'error': None,
    }

    try:
        priority_func = create_priority_function(body, signature)

        for n in n_values:
            try:
                codebook, _ = solve_with_priority(priority_func, signature, n, s, q, graph_dir)
                results['codebooks'][n] = sorted(codebook)
                results['sizes'][n] = len(codebook)
            except Exception as e:
                results['codebooks'][n] = []
                results['sizes'][n] = 0
                if results['error'] is None:
                    results['error'] = f"n={n}: {str(e)}"

    except Exception as e:
        results['error'] = f"Function creation failed: {str(e)}"
        for n in n_values:
            results['codebooks'][n] = []
            results['sizes'][n] = 0

    return results


def evaluate_function_worker(args):
    """Worker function for parallel evaluation (all n values at once)."""
    func_data, n_values, s, q, graph_dir = args
    return evaluate_single_function(func_data, n_values, s, q, graph_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate priority functions on extended inputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('functions_json', help='Path to functions JSON file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--min-n', type=int, default=6,
                        help='Minimum n value (default: 6)')
    parser.add_argument('--max-n', type=int, default=16,
                        help='Maximum n value (default: 16)')
    parser.add_argument('--s', type=int, default=1,
                        help='s parameter (default: 1)')
    parser.add_argument('--q', type=int, default=2,
                        help='q parameter (default: 2)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of functions to evaluate')
    parser.add_argument('--force-signature', choices=['no_graph', 'graph_gt', 'graph_networkx'],
                        help='Force a specific signature instead of auto-detecting')
    parser.add_argument('--graph-dir', '-g', default=None,
                        help='Directory with pre-computed graphs (LMDB format, e.g., graphs/n10_s1)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save checkpoint every N functions (default: 1)')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable checkpoint saving (faster but no crash recovery)')

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.functions_json)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load functions
    print("=" * 70)
    print("  Evaluate Functions on Extended Inputs")
    print("=" * 70)
    print()
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"n range: {args.min_n} to {args.max_n}")
    print(f"Parameters: s={args.s}, q={args.q}")
    print()

    with open(input_path) as f:
        functions = json.load(f)

    if args.limit:
        functions = functions[:args.limit]
        print(f"Limited to {args.limit} functions")

    # Add index to each function
    for i, func in enumerate(functions):
        func['func_index'] = i

    # Detect signatures
    print("Detecting function signatures...")
    signature_counts = {SIGNATURE_NO_GRAPH: 0, SIGNATURE_GRAPH_GT: 0, SIGNATURE_GRAPH_NETWORKX: 0}
    for func in functions:
        if args.force_signature:
            sig = args.force_signature
        else:
            sig = detect_signature(func['body'], func.get('args', ''))
        func['detected_signature'] = sig
        signature_counts[sig] += 1

    print(f"  no_graph: {signature_counts[SIGNATURE_NO_GRAPH]} functions")
    print(f"  graph_gt: {signature_counts[SIGNATURE_GRAPH_GT]} functions")
    print(f"  graph_networkx: {signature_counts[SIGNATURE_GRAPH_NETWORKX]} functions")
    print()

    n_values = list(range(args.min_n, args.max_n + 1))
    print(f"Evaluating {len(functions)} functions on n values: {n_values}")
    if args.graph_dir:
        print(f"Using pre-computed graphs from: {args.graph_dir}")
    print()

    # Setup checkpoint and log paths
    checkpoint_path = output_dir / "evaluation_checkpoint_type1.json"
    log_path = output_dir / "evaluation_progress_type1.log"

    # Checkpoint metadata
    metadata = {
        'input_file': str(input_path),
        'n_values': n_values,
        's': args.s,
        'q': args.q,
        'total_functions': len(functions),
        'started_at': datetime.now().isoformat(),
    }

    # Check for existing checkpoint
    # Results keyed by (func_index, n)
    results_dict: Dict[Tuple[int, int], Dict] = {}
    completed_pairs: Set[Tuple[int, int]] = set()

    # All possible (func_index, n) pairs
    all_pairs = [(f['func_index'], n) for f in functions for n in n_values]
    total_pairs = len(all_pairs)

    if args.resume and checkpoint_path.exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        results_dict, completed_pairs, saved_metadata = load_checkpoint(checkpoint_path)

        # Validate checkpoint matches current run
        if saved_metadata.get('n_values') != n_values:
            print(f"  Warning: n_values mismatch! Checkpoint: {saved_metadata.get('n_values')}, Current: {n_values}")
            print(f"  Starting fresh evaluation...")
            results_dict = {}
            completed_pairs = set()
        else:
            print(f"  Resuming from checkpoint: {len(completed_pairs)}/{total_pairs} (func,n) pairs already completed")
    elif checkpoint_path.exists() and not args.resume:
        print(f"  Note: Checkpoint exists at {checkpoint_path}")
        print(f"  Use --resume to continue from checkpoint, or delete it to start fresh")

    # Filter out already-completed pairs
    remaining_pairs = [(fi, n) for fi, n in all_pairs if (fi, n) not in completed_pairs]
    print(f"(func, n) pairs to evaluate: {len(remaining_pairs)} (skipping {len(completed_pairs)} already done)")
    print()

    start_time = time.time()

    if not remaining_pairs:
        print("All (func, n) pairs already evaluated!")
    else:
        # Build func_index -> func lookup
        func_by_index = {f['func_index']: f for f in functions}

        # Parallel evaluation of (func, n) pairs
        workers = args.workers or min(cpu_count(), len(remaining_pairs))
        print(f"Using {workers} parallel workers")
        if not args.no_checkpoint:
            print(f"Checkpoint: saving every {args.checkpoint_interval} evaluation(s) to {checkpoint_path}")
            print(f"Progress log: {log_path}")
        print()

        # Prepare work items: (func_data, n, s, q, graph_dir)
        work_items = [(func_by_index[fi], n, args.s, args.q, args.graph_dir) for fi, n in remaining_pairs]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_single_n_worker, item): (item[0]['func_index'], item[1])
                       for item in work_items}

            newly_completed = 0
            for future in as_completed(futures):
                func_index, n = futures[future]
                result = future.result()

                # Store result
                results_dict[(func_index, n)] = result
                completed_pairs.add((func_index, n))
                newly_completed += 1

                # Log progress for this (func, n) pair
                if not args.no_checkpoint:
                    elapsed = time.time() - start_time
                    log_progress(log_path, func_index, n, result['size'], elapsed, result.get('signature', '?'))

                # Save checkpoint
                if not args.no_checkpoint and newly_completed % args.checkpoint_interval == 0:
                    save_checkpoint(checkpoint_path, results_dict, completed_pairs, metadata)

                # Print progress
                total_completed = len(completed_pairs)
                if newly_completed % 50 == 0 or total_completed == total_pairs:
                    elapsed = time.time() - start_time
                    rate = newly_completed / elapsed if elapsed > 0 else 0
                    eta = (len(remaining_pairs) - newly_completed) / rate if rate > 0 else 0

                    print(f"  Progress: {total_completed}/{total_pairs} pairs "
                          f"({100*total_completed/total_pairs:.1f}%) "
                          f"- {rate:.1f} evals/s - ETA: {eta:.0f}s "
                          f"- Latest: func {func_index} n={n} size={result['size']}",
                          file=sys.stderr)

        # Final checkpoint save
        if not args.no_checkpoint:
            save_checkpoint(checkpoint_path, results_dict, completed_pairs, metadata)
            print(f"\nFinal checkpoint saved to {checkpoint_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Aggregate (func_index, n) results into per-function results
    print("Aggregating results...")
    all_results = []
    func_indices = sorted(set(fi for fi, n in results_dict.keys()))

    for func_index in func_indices:
        func = functions[func_index]

        # Gather all n results for this function
        func_result = {
            'func_id': func.get('priority_hash', 'unknown'),
            'func_index': func_index,
            'signature': None,
            'codebooks': {},
            'sizes': {},
            'error': None,
        }

        for n in n_values:
            key = (func_index, n)
            if key in results_dict:
                r = results_dict[key]
                func_result['codebooks'][n] = r.get('codebook', [])
                func_result['sizes'][n] = r.get('size', 0)
                if func_result['signature'] is None:
                    func_result['signature'] = r.get('signature', '?')
                if r.get('error') and func_result['error'] is None:
                    func_result['error'] = f"n={n}: {r['error']}"
            else:
                # Missing result (incomplete evaluation)
                func_result['codebooks'][n] = []
                func_result['sizes'][n] = 0

        all_results.append(func_result)

    # Count errors
    errors = sum(1 for r in all_results if r['error'])
    if errors:
        print(f"Warning: {errors} functions had errors")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved evaluation results to {results_path}")

    # Save summary table
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Signature detection:\n")
        f.write(f"  no_graph: {signature_counts[SIGNATURE_NO_GRAPH]}\n")
        f.write(f"  graph_gt: {signature_counts[SIGNATURE_GRAPH_GT]}\n")
        f.write(f"  graph_networkx: {signature_counts[SIGNATURE_GRAPH_NETWORKX]}\n\n")

        # Header
        header = "Func#  | Sig      | " + " | ".join(f"n={n:2d}" for n in n_values)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for result in all_results:
            sig_short = (result.get('signature') or '?')[:8]
            row = f"{result['func_index']:5d}  | {sig_short:8s} | "
            row += " | ".join(f"{result['sizes'].get(n, 0):4d}" for n in n_values)
            if result['error']:
                row += "  [ERROR]"
            f.write(row + "\n")

        f.write("\n")
        f.write("Statistics per n:\n")
        for n in n_values:
            sizes = [r['sizes'].get(n, 0) for r in all_results]
            if sizes:
                f.write(f"  n={n:2d}: min={min(sizes):4d}, max={max(sizes):4d}, "
                        f"avg={sum(sizes)/len(sizes):.1f}\n")

    print(f"Saved summary to {summary_path}")

    # Also save codebooks in a separate file for VT analysis
    codebooks_path = output_dir / "codebooks.json"
    codebooks_data = {
        'n_values': n_values,
        's': args.s,
        'q': args.q,
        'functions': []
    }

    for result in all_results:
        func = functions[result['func_index']]
        codebooks_data['functions'].append({
            'func_index': result['func_index'],
            'func_id': result['func_id'],
            'signature': result['signature'],
            'body': func['body'],
            'priority_hash': func.get('priority_hash'),
            'duplicate_count': func.get('duplicate_count', 1),
            'codebooks': result['codebooks'],
            'sizes': result['sizes'],
            'error': result['error'],
        })

    with open(codebooks_path, 'w') as f:
        json.dump(codebooks_data, f, indent=2)
    print(f"Saved codebooks to {codebooks_path}")

    print()
    print("=" * 70)
    print("  Evaluation Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
