#!/usr/bin/env python3
"""
Evaluate extracted priority functions on extended inputs.

Takes the JSON from extract.py and evaluates each function on a range of n values
using the greedy independent set algorithm. Auto-detects function signature type.

Arguments:
  functions_json                 Path to JSON file from extract.py

  -o, --output DIR               Output directory (default: ./evaluate_<timestamp>/)
  --min-n N                      Minimum n value to evaluate (default: 6)
  --max-n N                      Maximum n value to evaluate (default: 16)
  --s S                          s parameter, i.e. deletion distance (default: 1)
  --q Q                          q parameter, i.e. alphabet size (default: 2)
  -w, --workers N                Number of parallel workers (default: all CPUs)
  -l, --limit N                  Only evaluate first N functions
  --force-signature TYPE         Force no_graph / graph_networkx
  -g, --graph-dir DIR            Directory with precomputed LMDB graphs
  --resume                       Resume from checkpoint if available
  --checkpoint-interval N        Save checkpoint every N evaluations (default: 100)
  --no-checkpoint                Disable checkpoint saving
  --no-codebooks                 Don't store codebook contents (only sizes)

Examples:
    # s=1 functions, evaluate up to n=16
    python functions/evaluate.py ./extract/successful_functions.json --max-n 16

    # s=2 functions, evaluate n=7..18 with 16 workers
    python functions/evaluate.py ./extract/top50_gap_functions.json \
        --s 2 --min-n 7 --max-n 18 --workers 16

    # Resume interrupted evaluation
    python functions/evaluate.py ./extract/successful_functions.json --resume
"""

import argparse
import json
import sys
import itertools
import textwrap
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    SIGNATURE_NO_GRAPH,
    SIGNATURE_GRAPH_NETWORKX,
    COMMON_IMPORTS,
    detect_signature,
    are_neighbors,
    build_graph_networkx,
)
from utils.graph_paths import get_graph_path as find_graph_path, DEFAULT_GRAPH_DIR

# Try to import FastGraph (C++ backend with nx compatible API)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from disfun.utils.fast_graph import load_graph_from_lmdb as _load_fastgraph, FastGraphCpp
    from disfun.utils.fast_graph import USING_CPP
    HAS_FASTGRAPH = True
except ImportError:
    HAS_FASTGRAPH = False
    USING_CPP = False

# Apply nx monkey patches so LLM generated functions using nx.clustering(G, node) etc.
# work with FastGraph objects. Mirrors graph_fastgraph.py from the evaluation pipeline.
try:
    _eval_graph_fastgraph_path = str(Path(__file__).parent.parent.parent / "src" / "disfun" / "specifications" / "ECC" / "evaluation")
    if _eval_graph_fastgraph_path not in sys.path:
        sys.path.insert(0, _eval_graph_fastgraph_path)
    import graph_fastgraph as _gfg  # noqa: F401 — side effect: monkey-patches nx.*
except ImportError:
    pass

# Per-worker caches (populated once per (n, s, q) per worker process)
_nodes_cache = {}       # (n, q) -> list of node strings
_neighbors_cache = {}   # (n, s, q, graph_dir) -> dict or None
_fastgraph_cache = {}   # (n, s, q, graph_dir) -> FastGraphCpp or None
_graph_nx_cache = {}    # (n, s, q) -> networkx Graph


def load_neighbors_from_lmdb(graph_path: str) -> Dict[str, Set[str]]:
    """
    Load neighbor dict from LMDB database for fast greedy algorithm.

    Returns:
        Dict mapping each node to its set of neighbors
    """
    try:
        import lmdb
    except ImportError:
        raise ImportError("lmdb required. Install with: pip install lmdb")

    neighbors = {}
    env = lmdb.open(graph_path, readonly=True, lock=False)
    with env.begin(buffers=True) as txn:
        for key, value in txn.cursor():
            node = bytes(key).decode()
            val_str = bytes(value).decode()
            if val_str.startswith('['):
                import json
                neighbor_list = json.loads(val_str)
            else:
                neighbor_list = [n for n in val_str.split(',') if n]
            neighbors[node] = set(neighbor_list)
    env.close()
    return neighbors


def get_graph_path(graph_dir: str, n: int, s: int, q: int) -> Optional[str]:
    """
    Find precomputed graph path for given parameters.

    Returns path if found, None otherwise.
    """
    if not graph_dir:
        return None

    # Try deletion graphs first, then IDS graphs
    path = find_graph_path("deletion", s, n, q, graph_dir=graph_dir)
    if path is None:
        path = find_graph_path("ids", s, n, q, graph_dir=graph_dir)
    return path


def _get_nodes(n: int, q: int) -> List[str]:
    """Return cached list of all q-ary strings of length n."""
    key = (n, q)
    if key not in _nodes_cache:
        _nodes_cache[key] = [''.join(seq) for seq in itertools.product(map(str, range(q)), repeat=n)]
    return _nodes_cache[key]


def _get_neighbors(n: int, s: int, q: int, graph_dir: str) -> Optional[Dict[str, Set[str]]]:
    """Return cached neighbor dict from LMDB, or None if unavailable.

    Note: Prefer _get_fastgraph() when FastGraph is available. It is faster
    for both loading and greedy IS computation.
    """
    key = (n, s, q, graph_dir)
    if key not in _neighbors_cache:
        graph_path = get_graph_path(graph_dir, n, s, q)
        if graph_path:
            try:
                _neighbors_cache[key] = load_neighbors_from_lmdb(graph_path)
            except Exception as e:
                print(f"Warning: Failed to load graph from {graph_path}: {e}", file=sys.stderr)
                _neighbors_cache[key] = None
        else:
            _neighbors_cache[key] = None
    return _neighbors_cache[key]


def _get_fastgraph(n: int, s: int, q: int, graph_dir: str):
    """Return cached FastGraph from LMDB, or None if unavailable.

    FastGraph provides:
    - NetworkX compatible API (G.neighbors, G.degree, G[node], etc.)
    - Built in greedy_independent_set() (C++ when available)
    - Faster LMDB loading (ujson + CSR format)
    """
    if not HAS_FASTGRAPH:
        return None
    key = (n, s, q, graph_dir)
    if key not in _fastgraph_cache:
        graph_path = get_graph_path(graph_dir, n, s, q)
        if graph_path:
            try:
                _fastgraph_cache[key] = _load_fastgraph(graph_path)
            except Exception as e:
                print(f"Warning: Failed to load FastGraph from {graph_path}: {e}", file=sys.stderr)
                _fastgraph_cache[key] = None
        else:
            _fastgraph_cache[key] = None
    return _fastgraph_cache[key]


# Checkpoint management

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


# Priority function creation

def create_priority_function(body: str, signature: str):
    """Create a callable priority function from body string with appropriate signature."""
    # Only strip trailing whitespace - preserve leading indentation from checkpoint
    clean_body = body.rstrip().expandtabs(4)
    if not clean_body.startswith('    '):
        clean_body = textwrap.indent(clean_body, '    ')

    if signature == SIGNATURE_NO_GRAPH:
        func_def = "def priority(node, n, s, q):"
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


def solve_no_graph(priority_func, nodes: List[str], n: int, s: int, q: int,
                   graph_dir: str = None,
                   neighbors: Dict[str, Set[str]] = None) -> Tuple[Set[str], Dict[str, float]]:
    """Solve using no_graph signature.

    Uses FastGraph built in greedy IS when available for speed.
    """
    priorities = {}
    for node in nodes:
        try:
            p = priority_func(node, n, s, q)
            if p is None:
                p = 0.0
            priorities[node] = float(p)
        except Exception:
            priorities[node] = 0.0

    # Use FastGraph built in greedy IS when available
    effective_graph_dir = graph_dir if graph_dir is not None else DEFAULT_GRAPH_DIR
    fg = _get_fastgraph(n, s, q, effective_graph_dir)
    if fg is not None:
        codebook = fg.greedy_independent_set(priorities)
        return set(codebook), priorities

    return greedy_independent_set(nodes, priorities, n, s, neighbors)


def solve_graph_networkx(priority_func, nodes: List[str], n: int, s: int, q: int,
                         graph_dir: str = None,
                         neighbors: Dict[str, Set[str]] = None) -> Tuple[Set[str], Dict[str, float]]:
    """Solve using graph_networkx signature.

    Prefers FastGraph (nx compatible API, C++ greedy IS) when available.
    Falls back to NetworkX if FastGraph is not installed.
    """
    cache_key = (n, s, q)

    # Try FastGraph first. It is nx compatible and has built in greedy IS.
    effective_graph_dir = graph_dir if graph_dir is not None else DEFAULT_GRAPH_DIR
    fg = _get_fastgraph(n, s, q, effective_graph_dir)
    if fg is not None:
        G = fg  # FastGraph has nx compatible API
    elif cache_key in _graph_nx_cache:
        G = _graph_nx_cache[cache_key]
    elif neighbors is not None:
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx")
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for node, neighs in neighbors.items():
            for neigh in neighs:
                if node < neigh:  # Avoid duplicate edges
                    G.add_edge(node, neigh)
        _graph_nx_cache[cache_key] = G
    else:
        G = build_graph_networkx(nodes, n, s)
        _graph_nx_cache[cache_key] = G

    priorities = {}
    for node in nodes:
        try:
            p = priority_func(node, G, n, s)
            if p is None:
                p = 0.0
            priorities[node] = float(p)
        except Exception:
            priorities[node] = 0.0

    # Use FastGraph built in greedy IS when available (C++ optimized)
    if fg is not None:
        codebook = fg.greedy_independent_set(priorities)
        return set(codebook), priorities

    return greedy_independent_set_with_graph_nx(nodes, priorities, G)



def greedy_independent_set(nodes: List[str], priorities: Dict[str, float],
                           n: int, s: int,
                           neighbors: Dict[str, Set[str]] = None) -> Tuple[Set[str], Dict[str, float]]:
    """
    Build independent set using greedy algorithm.

    If neighbors dict is provided, uses O(degree) neighbor lookup.
    Otherwise falls back to O(|V| * n²) on the fly computation (slow for large n).
    """
    nodes_sorted = sorted(nodes, key=lambda x: (-priorities[x], x))

    independent_set = set()
    removed_nodes = set()

    if neighbors is not None:
        # Fast path: use precomputed neighbors
        for node in nodes_sorted:
            if node in removed_nodes:
                continue
            independent_set.add(node)
            removed_nodes.add(node)
            removed_nodes.update(neighbors.get(node, set()))
    else:
        # Slow path: compute neighbors on the fly
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



def solve_with_priority(priority_func, signature: str, n: int, s: int, q: int,
                        graph_dir: str = None) -> Tuple[Set[str], Dict[str, float]]:
    """
    Find independent set using given priority function with detected signature.

    Args:
        graph_dir: Optional path to directory with precomputed graphs (LMDB format).
                   Defaults to DEFAULT_GRAPH_DIR if not specified.

    Returns:
        - independent_set: Set of codewords
        - priorities: Dict mapping all nodes to their priority values
    """
    # Use default graph dir if not specified
    effective_graph_dir = graph_dir if graph_dir is not None else DEFAULT_GRAPH_DIR

    # Generate all q-ary strings of length n (cached per worker)
    nodes = _get_nodes(n, q)

    # Load precomputed neighbors (only needed as fallback when FastGraph unavailable).
    # FastGraph loads directly from LMDB with faster ujson + CSR format.
    neighbors = None
    if not HAS_FASTGRAPH or _get_fastgraph(n, s, q, effective_graph_dir) is None:
        neighbors = _get_neighbors(n, s, q, effective_graph_dir)

    if signature == SIGNATURE_NO_GRAPH:
        return solve_no_graph(priority_func, nodes, n, s, q, effective_graph_dir, neighbors)
    elif signature == SIGNATURE_GRAPH_NETWORKX:
        return solve_graph_networkx(priority_func, nodes, n, s, q, effective_graph_dir, neighbors)
    else:
        raise ValueError(f"Unknown signature: {signature}")


def evaluate_single_n(func_data: Dict, n: int, s: int = 1, q: int = 2,
                      graph_dir: str = None, store_codebooks: bool = True) -> Dict:
    """
    Evaluate a single priority function on a single n value.

    Args:
        func_data: Function data dict with 'body', 'args', etc.
        n: The n value to evaluate
        s, q: Problem parameters
        graph_dir: Optional path to precomputed graphs
        store_codebooks: If False, don't store codebook contents (saves memory)

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
        result['codebook'] = sorted(codebook) if store_codebooks else []
        result['size'] = len(codebook)
    except Exception as e:
        result['error'] = str(e)

    return result


def evaluate_single_n_worker(args):
    """Worker function for parallel (func, n) evaluation."""
    func_data, n, s, q, graph_dir, store_codebooks = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_single_n(func_data, n, s, q, graph_dir, store_codebooks)


def evaluate_single_function(func_data: Dict, n_values: List[int],
                              s: int = 1, q: int = 2,
                              graph_dir: str = None) -> Dict:
    """
    Evaluate a single priority function on multiple n values.
    Used for non checkpointed mode for backward compatibility.

    Args:
        graph_dir: Optional path to directory with precomputed graphs

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
                        help='Output directory (default: ./evaluate_<timestamp>/)')
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
    parser.add_argument('--force-signature', choices=['no_graph', 'graph_networkx'],
                        help='Force a specific signature instead of auto-detecting')
    parser.add_argument('--graph-dir', '-g', default=DEFAULT_GRAPH_DIR,
                        help=f'Directory with precomputed graphs (LMDB format). '
                             f'Default: {DEFAULT_GRAPH_DIR}')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N evaluations (default: 100)')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable checkpoint saving (faster but no crash recovery)')
    parser.add_argument('--no-codebooks', action='store_true',
                        help='Don\'t store codebook contents (only sizes). Saves memory and disk.')
    parser.add_argument('--stagger-from', type=int, default=None, metavar='N',
                        help='Only evaluate functions that achieved max score at n-1 for n >= N. '
                             'E.g. --stagger-from 13 evaluates all functions for n<13, then filters.')

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.functions_json)
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./evaluate_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load functions
    print("Evaluate functions on extended inputs")
    print()
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"n range: {args.min_n} to {args.max_n}")
    print(f"Parameters: s={args.s}, q={args.q}")
    if args.stagger_from:
        print(f"Stagger filter: enabled from n>={args.stagger_from}")
    if HAS_FASTGRAPH:
        fg_backend = "C++" if USING_CPP else "Python fallback"
        print(f"FastGraph: available ({fg_backend})")
    else:
        print(f"FastGraph: not available (falling back to NetworkX/neighbor dict)")
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
    signature_counts = {SIGNATURE_NO_GRAPH: 0, SIGNATURE_GRAPH_NETWORKX: 0}
    for func in functions:
        if args.force_signature:
            sig = args.force_signature
        else:
            sig = detect_signature(func['body'], func.get('args', ''))
        func['detected_signature'] = sig
        signature_counts[sig] += 1

    print(f"  no_graph: {signature_counts[SIGNATURE_NO_GRAPH]} functions")
    print(f"  graph_networkx: {signature_counts[SIGNATURE_GRAPH_NETWORKX]} functions")
    print()

    n_values = list(range(args.min_n, args.max_n + 1))
    print(f"Evaluating {len(functions)} functions on n values: {n_values}")

    # Check which graphs are available
    print(f"Graph directory: {args.graph_dir}")
    graphs_found = []
    graphs_missing = []
    for n in n_values:
        graph_path = get_graph_path(args.graph_dir, n, args.s, args.q)
        if graph_path:
            graphs_found.append(n)
        else:
            graphs_missing.append(n)

    if graphs_found:
        print(f"  Pre-computed graphs found for n={graphs_found}")
    if graphs_missing:
        print(f"  Warning: No graphs for n={graphs_missing}, will compute neighbors on the fly (slow)")
    print()

    # Setup checkpoint and log paths
    checkpoint_path = output_dir / "evaluation_checkpoint.json"
    log_path = output_dir / "evaluation_progress.log"

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

    store_codebooks = not args.no_codebooks

    stagger_log = []  # (n, num_active, max_score_prev_n, prev_n)

    if not remaining_pairs:
        print("All (func, n) pairs already evaluated!")
    else:
        # Build func_index -> func lookup
        func_by_index = {f['func_index']: f for f in functions}

        # Determine workers
        workers = args.workers or min(cpu_count(), len(remaining_pairs))
        print(f"Using {workers} parallel workers")
        if not args.no_checkpoint:
            print(f"Checkpoint: saving every {args.checkpoint_interval} evaluation(s) to {checkpoint_path}")
            print(f"Progress log: {log_path}")
        if not store_codebooks:
            print(f"Codebook storage: disabled (--no-codebooks)")
        print()

        # Process n values sequentially (outer loop), functions in parallel (inner loop).
        # This ensures workers only cache one graph at a time. Destroying the executor
        # between n values kills workers and frees their cached graph memory.
        newly_completed = 0

        # Staggered filtering: track which functions are still active
        active_func_indices = set(func_by_index.keys())  # all functions initially

        for n in sorted(n_values):
            # Apply stagger filter: only keep max-score functions from previous n
            if args.stagger_from and n >= args.stagger_from and n > min(n_values):
                prev_n = n - 1
                if prev_n in n_values:
                    scores_at_prev = {}
                    for fi in active_func_indices:
                        key = (fi, prev_n)
                        if key in results_dict:
                            scores_at_prev[fi] = results_dict[key].get('size', 0)

                    if scores_at_prev:
                        max_score = max(scores_at_prev.values())
                        promoted = {fi for fi, s in scores_at_prev.items() if s == max_score}
                        filtered_count = len(active_func_indices) - len(promoted)
                        active_func_indices = promoted
                        stagger_log.append((n, len(promoted), max_score, prev_n))
                        print(f"  Stagger filter at n={n}: {len(promoted)} functions achieved "
                              f"max score {max_score} at n={prev_n} (filtered {filtered_count})",
                              file=sys.stderr)

            remaining_for_n = [(fi, n) for fi in sorted(active_func_indices)
                               if (fi, n) not in completed_pairs]
            if not remaining_for_n:
                continue

            print(f"  n={n}: evaluating {len(remaining_for_n)} functions ...", file=sys.stderr)

            n_workers = min(workers, len(remaining_for_n))

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                work_items = [(func_by_index[fi], n, args.s, args.q, args.graph_dir, store_codebooks)
                              for fi, _ in remaining_for_n]
                futures = {executor.submit(evaluate_single_n_worker, item): item[0]['func_index']
                           for item in work_items}

                for future in as_completed(futures):
                    func_index = futures[future]
                    result = future.result()

                    # Store result
                    results_dict[(func_index, n)] = result
                    completed_pairs.add((func_index, n))
                    newly_completed += 1

                    # Log progress
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

            # Executor destroyed here — workers die, cached graphs freed

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
                row += "  [error]"
            f.write(row + "\n")

        f.write("\n")
        f.write("Statistics per n:\n")
        for n in n_values:
            sizes = [r['sizes'].get(n, 0) for r in all_results]
            if sizes:
                f.write(f"  n={n:2d}: min={min(sizes):4d}, max={max(sizes):4d}, "
                        f"avg={sum(sizes)/len(sizes):.1f}\n")

        if args.stagger_from and stagger_log:
            f.write("\n")
            f.write("Stagger filter summary:\n")
            cutoff = args.stagger_from
            f.write(f"  n={min(n_values)}..{cutoff - 1}: {len(functions)} functions (all)\n")
            for n, num_active, max_score, prev_n in stagger_log:
                f.write(f"  n={n}: {num_active} functions (max score at n={prev_n}: {max_score})\n")

    print(f"Saved summary to {summary_path}")

    # Print stagger summary to stderr
    if args.stagger_from and stagger_log:
        print("\nStagger filter summary:", file=sys.stderr)
        cutoff = args.stagger_from
        print(f"  n={min(n_values)}..{cutoff - 1}: {len(functions)} functions (all)", file=sys.stderr)
        for n, num_active, max_score, prev_n in stagger_log:
            print(f"  n={n}: {num_active} functions (max score at n={prev_n}: {max_score})", file=sys.stderr)

    # Save codebooks in a separate file for VT analysis (skip if no codebooks)
    if store_codebooks:
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
    else:
        print("Skipping codebooks.json (--no-codebooks)")

    print()
    print("Evaluation complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
