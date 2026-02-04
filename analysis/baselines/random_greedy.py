#!/usr/bin/env python3
"""
Random Greedy Independent Set Baseline

Runs random greedy trials to find maximal independent sets. Uses C++ FastGraph
for speedup when --graph-dir is provided (recommended).

Usage:
    python random_greedy.py --n-values 6,7,8,9,10 --q 4 --graph-type ids --graph-dir /mnt/Graphs

Arguments:
    --n-values          Comma-separated n values (required)
    --s                 Number of errors to correct (default: 1)
    --q                 Alphabet size: 2=binary, 4=quaternary (default: 2)
    --graph-type        Graph type: 'deletion' or 'ids' (default: deletion)
    --graph-dir         Directory with LMDB graphs. Enables C++ mode (fastest)
    --trials            Number of random trials per n (default: 100000)
    --seed              Base random seed (default: 42)
    --output, -o        Output directory (default: ./greedy_results)

    Python-only options (ignored in C++ mode):
    --workers, -w       Parallel workers (default: cpu_count)
    --no-cpp            Disable C++ mode, use Python multiprocessing
    --memory-efficient  LMDB-backed Python mode for huge graphs

Outputs:
    greedy_<type>_s<s>_n<n>_q<q>_best.txt  - Best solution
    greedy_summary.json                     - Statistics

Performance (C++ mode with --graph-dir):
    IDS n=8 (65k nodes):   100k trials in ~4 min
    IDS n=9 (262k nodes):  100k trials in ~15 min
    IDS n=10 (1M nodes):   100k trials in ~1 hour
    IDS n=11 (4M nodes):   100k trials in ~4-5 hours (includes 75 min load time)
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from multiprocessing import Pool, cpu_count
from statistics import mean, stdev
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import are_neighbors
from utils.graph_paths import get_graph_path

# Try to import C++ FastGraph for massive speedup
try:
    from disfun.utils.fast_graph import load_graph_from_lmdb as load_fastgraph
    HAS_FASTGRAPH = True
except ImportError:
    HAS_FASTGRAPH = False


# Global variables for multiprocessing (initialized per worker)
_worker_neighbors: Dict[str, Set[str]] = {}
_worker_nodes: List[str] = []


def _init_worker(neighbors: Dict[str, Set[str]], nodes: List[str]):
    """Initialize worker process with shared data."""
    global _worker_neighbors, _worker_nodes
    _worker_neighbors = neighbors
    _worker_nodes = nodes


def build_neighbor_dict(nodes: List[str], n: int, s: int) -> Dict[str, Set[str]]:
    """Build dictionary mapping each node to its neighbors."""
    neighbors = {node: set() for node in nodes}
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            if are_neighbors(node1, node2, n, s):
                neighbors[node1].add(node2)
                neighbors[node2].add(node1)
    return neighbors


def load_graph_from_lmdb(graph_path: str) -> Dict[str, Set[str]]:
    """Load graph from LMDB database into memory."""
    try:
        import lmdb
        import ujson as json
    except ImportError:
        raise ImportError("lmdb and ujson required. Install with: pip install lmdb ujson")

    neighbors = {}
    env = lmdb.open(graph_path, readonly=True, lock=False)
    with env.begin(buffers=True) as txn:
        for key, value in txn.cursor():
            node = bytes(key).decode()
            # Try JSON format first (used by main codebase), fall back to CSV
            val_str = bytes(value).decode()
            if val_str.startswith('['):
                neighbor_list = json.loads(val_str)
            else:
                neighbor_list = [n for n in val_str.split(',') if n]
            neighbors[node] = set(neighbor_list)
    env.close()
    return neighbors


class LMDBGraph:
    """Memory-efficient graph backed by LMDB (memory-mapped, no Python copy)."""

    def __init__(self, graph_path: str):
        try:
            import lmdb
        except ImportError:
            raise ImportError("lmdb required. Install with: pip install lmdb")

        self.graph_path = graph_path
        self.env = lmdb.open(graph_path, readonly=True, lock=False,
                            readahead=False, meminit=False)
        self._nodes = None

    def get_neighbors(self, node: str) -> Set[str]:
        """Get neighbors of a node (reads from LMDB on demand)."""
        try:
            import ujson as json
        except ImportError:
            import json

        with self.env.begin(buffers=True) as txn:
            value = txn.get(node.encode())
            if value is None:
                return set()
            val_str = bytes(value).decode()
            if val_str.startswith('['):
                return set(json.loads(val_str))
            else:
                return set(n for n in val_str.split(',') if n)

    def get_nodes(self) -> List[str]:
        """Get all node names (cached after first call)."""
        if self._nodes is None:
            self._nodes = []
            with self.env.begin(buffers=True) as txn:
                for key, _ in txn.cursor():
                    self._nodes.append(bytes(key).decode())
        return self._nodes

    def close(self):
        self.env.close()


def random_greedy_mis(
    neighbors: Dict[str, Set[str]],
    nodes: List[str],
    seed: int,
    return_set: bool = False
) -> Tuple[int, Optional[List[str]]]:
    """
    Build maximal independent set using random greedy algorithm.

    Args:
        neighbors: Dict mapping each node to its neighbors
        nodes: List of all nodes (avoids repeated dict.keys() calls)
        seed: Random seed
        return_set: If True, return the actual set

    Returns:
        (size, set) if return_set else (size, None)
    """
    rng = random.Random(seed)
    shuffled = nodes.copy()
    rng.shuffle(shuffled)

    independent_set = []
    removed = set()

    for node in shuffled:
        if node in removed:
            continue
        independent_set.append(node)
        removed.add(node)
        removed.update(neighbors[node])

    if return_set:
        return len(independent_set), independent_set
    return len(independent_set), None


def _worker_trial(seed: int) -> Tuple[int, int, Optional[List[str]]]:
    """Worker function for parallel trials (in-memory graph). Returns (seed, size, solution)."""
    size, solution = random_greedy_mis(_worker_neighbors, _worker_nodes, seed, return_set=True)
    return seed, size, solution


# Global for LMDB-backed worker
_worker_graph_path: str = ""


def _init_lmdb_worker(graph_path: str, nodes: List[str]):
    """Initialize worker with LMDB path (each worker opens its own connection)."""
    global _worker_graph_path, _worker_nodes
    _worker_graph_path = graph_path
    _worker_nodes = nodes


def _lmdb_worker_trial(seed: int) -> Tuple[int, int, Optional[List[str]]]:
    """Worker function for LMDB-backed trials. Each worker opens its own LMDB."""
    graph = LMDBGraph(_worker_graph_path)

    rng = random.Random(seed)
    shuffled = _worker_nodes.copy()
    rng.shuffle(shuffled)

    independent_set = []
    removed = set()

    for node in shuffled:
        if node in removed:
            continue
        independent_set.append(node)
        removed.add(node)
        # Get neighbors on-demand from LMDB
        removed.update(graph.get_neighbors(node))

    graph.close()
    return seed, len(independent_set), independent_set


def run_trials(
    neighbors: Dict[str, Set[str]],
    num_trials: int,
    base_seed: int,
    num_workers: int = None
) -> Tuple[List[int], int, List[str]]:
    """
    Run multiple random greedy trials in parallel (in-memory graph).

    Args:
        neighbors: Dict mapping each node to its neighbors
        num_trials: Number of random trials to run
        base_seed: Base random seed (each trial uses base_seed + i)
        num_workers: Number of parallel workers (default: cpu_count)

    Returns:
        (sizes, best_size, best_solution)
    """
    if num_workers is None:
        num_workers = cpu_count()

    nodes = list(neighbors.keys())
    seeds = [base_seed + i for i in range(num_trials)]

    # Single worker: run sequentially without multiprocessing (avoids memory copy)
    if num_workers == 1:
        sizes = []
        best_size = 0
        best_solution = []
        for seed in tqdm(seeds, desc="Running trials", unit="trial"):
            size, solution = random_greedy_mis(neighbors, nodes, seed, return_set=True)
            sizes.append(size)
            if size > best_size:
                best_size = size
                best_solution = solution
        return sizes, best_size, best_solution

    # Multiple workers: use multiprocessing (copies data to each worker)
    with Pool(num_workers, initializer=_init_worker, initargs=(neighbors, nodes)) as pool:
        # Use imap_unordered with tqdm for progress bar
        results = list(tqdm(
            pool.imap_unordered(_worker_trial, seeds),
            total=num_trials,
            desc="Running trials",
            unit="trial"
        ))

    # Collect results
    sizes = []
    best_size = 0
    best_solution = []

    for seed, size, solution in results:
        sizes.append(size)
        if size > best_size:
            best_size = size
            best_solution = solution

    return sizes, best_size, best_solution


def run_trials_lmdb(
    graph_path: str,
    nodes: List[str],
    num_trials: int,
    base_seed: int,
    num_workers: int = None
) -> Tuple[List[int], int, List[str]]:
    """
    Run multiple random greedy trials in parallel (LMDB-backed, memory efficient).

    Each worker opens its own LMDB connection and reads neighbors on-demand.
    Uses ~O(nodes) memory per worker instead of O(edges).

    Args:
        graph_path: Path to LMDB graph database
        nodes: List of all node names
        num_trials: Number of random trials to run
        base_seed: Base random seed (each trial uses base_seed + i)
        num_workers: Number of parallel workers (default: cpu_count)

    Returns:
        (sizes, best_size, best_solution)
    """
    if num_workers is None:
        num_workers = cpu_count()

    seeds = [base_seed + i for i in range(num_trials)]

    # Run trials in parallel - each worker opens its own LMDB connection
    with Pool(num_workers, initializer=_init_lmdb_worker, initargs=(graph_path, nodes)) as pool:
        # Use imap_unordered with tqdm for progress bar
        results = list(tqdm(
            pool.imap_unordered(_lmdb_worker_trial, seeds),
            total=num_trials,
            desc="Running trials",
            unit="trial"
        ))

    # Collect results
    sizes = []
    best_size = 0
    best_solution = []

    for seed, size, solution in results:
        sizes.append(size)
        if size > best_size:
            best_size = size
            best_solution = solution

    return sizes, best_size, best_solution


def run_trials_cpp(
    graph_path: str,
    num_trials: int,
    base_seed: int,
) -> Tuple[List[int], int, List[str]]:
    """
    Run multiple random greedy trials using C++ FastGraph (fastest method).

    All trials run in C++ with no Python overhead. Much faster than Python multiprocessing.

    Args:
        graph_path: Path to LMDB graph database
        num_trials: Number of random trials to run
        base_seed: Base random seed (each trial uses base_seed + i)

    Returns:
        (sizes, best_size, best_solution)
    """
    if not HAS_FASTGRAPH:
        raise ImportError("C++ FastGraph not available. Install with: pip install -e .")

    # Load graph using C++ FastGraph
    G = load_fastgraph(graph_path)

    # Run all trials in C++ (returns list of sizes)
    print(f"  Running {num_trials:,} trials in C++...")
    sizes = list(G.random_greedy_trials(num_trials, base_seed))

    # Find best and get its solution
    best_size = max(sizes)
    best_idx = sizes.index(best_size)
    best_seed = base_seed + best_idx
    best_solution = list(G.random_greedy_independent_set(best_seed))

    return sizes, best_size, best_solution


def main():
    parser = argparse.ArgumentParser(
        description='Random greedy independent set baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--n-values', type=str, required=True,
                        help='Comma-separated n values (e.g., "6,7,8,9,10")')
    parser.add_argument('--s', type=int, default=1,
                        help='Number of deletions to correct (default: 1)')
    parser.add_argument('--q', type=int, default=2,
                        help='Alphabet size (default: 2 for binary)')
    parser.add_argument('--trials', type=int, default=100000,
                        help='Number of random trials per n (default: 100000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--graph-dir',
                        help='Directory with pre-computed LMDB graphs')
    parser.add_argument('--graph-type', choices=['deletion', 'ids'], default='deletion',
                        help='Graph type: "deletion" or "ids" (default: deletion)')
    parser.add_argument('--output', '-o', default='./greedy_results',
                        help='Output directory for solutions')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    parser.add_argument('--memory-efficient', action='store_true',
                        help='Use LMDB-backed mode (slower but uses much less memory). '
                             'Auto-enabled for n >= 15 with pre-computed graphs.')
    parser.add_argument('--cpp', action='store_true',
                        help='Use C++ FastGraph for massive speedup (recommended). '
                             'Requires pre-computed LMDB graphs.')
    parser.add_argument('--no-cpp', action='store_true',
                        help='Disable C++ even if available (use Python multiprocessing).')

    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = args.workers or cpu_count()

    # Determine if C++ will be used
    use_cpp = (args.cpp or HAS_FASTGRAPH) and not args.no_cpp and args.graph_dir

    print("=" * 70)
    print("  Random Greedy Baseline")
    print("=" * 70)
    print()
    print(f"n values: {n_values}")
    print(f"s={args.s}, q={args.q}, type={args.graph_type}")
    print(f"Trials per n: {args.trials}")
    if use_cpp:
        print(f"Mode: C++ FastGraph (fastest)")
    else:
        print(f"Workers: {num_workers}")
    print(f"Output: {output_dir}")
    if HAS_FASTGRAPH and not use_cpp:
        print(f"Note: C++ FastGraph available but not used. Add --graph-dir to enable.")
    print()

    results = []

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"Processing n={n}")
        print("=" * 60)

        # Determine mode: C++ (fastest) > LMDB-backed > in-memory
        use_cpp = (args.cpp or HAS_FASTGRAPH) and not args.no_cpp and args.graph_dir
        use_lmdb_mode = args.memory_efficient or (args.graph_dir and n >= 15)
        graph_path = None

        # Get graph path if using pre-computed graphs
        if args.graph_dir:
            graph_path = get_graph_path(args.graph_type, args.s, n, args.q, graph_dir=args.graph_dir)

        # Run trials with appropriate method
        if use_cpp and graph_path and HAS_FASTGRAPH:
            # C++ FastGraph: fastest method
            print(f"Using C++ FastGraph (fastest): {graph_path}")
            start_time = time.time()
            sizes, best_size, best_solution = run_trials_cpp(graph_path, args.trials, args.seed)
        elif graph_path:
            if use_lmdb_mode:
                # Memory-efficient: only load node names, not full graph
                print(f"Using LMDB-backed mode (memory efficient): {graph_path}")
                graph = LMDBGraph(graph_path)
                nodes = graph.get_nodes()
                graph.close()
                print(f"  Nodes: {len(nodes)}")
                print(f"  Running {args.trials} trials with {num_workers} workers...")
                start_time = time.time()
                sizes, best_size, best_solution = run_trials_lmdb(graph_path, nodes, args.trials, args.seed, num_workers)
            else:
                print(f"Loading graph into memory: {graph_path}")
                neighbors = load_graph_from_lmdb(graph_path)
                print(f"  Nodes: {len(neighbors)}")
                print(f"  Running {args.trials} trials with {num_workers} workers...")
                start_time = time.time()
                sizes, best_size, best_solution = run_trials(neighbors, args.trials, args.seed, num_workers)
        else:
            # Build graph on-the-fly (no pre-computed graph)
            if graph_path is None and args.graph_dir:
                print(f"Graph not found, building on-the-fly...")
            else:
                print("Building graph on-the-fly...")
            nodes = [''.join(seq) for seq in itertools.product(map(str, range(args.q)), repeat=n)]
            print(f"  Nodes: {len(nodes)}")
            neighbors = build_neighbor_dict(nodes, n, args.s)
            print(f"  Graph built")
            print(f"  Running {args.trials} trials with {num_workers} workers...")
            start_time = time.time()
            sizes, best_size, best_solution = run_trials(neighbors, args.trials, args.seed, num_workers)
        elapsed = time.time() - start_time

        # Statistics
        avg_size = mean(sizes)
        std_size = stdev(sizes) if len(sizes) > 1 else 0
        max_count = sizes.count(best_size)
        max_freq = 100 * max_count / len(sizes)

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Best: {best_size}")
        print(f"  Mean: {avg_size:.2f} +/- {std_size:.2f}")
        print(f"  Max frequency: {max_freq:.2f}%")

        # Append to log file (human-readable, incremental)
        type_prefix = "d" if args.graph_type == "deletion" else "ids"
        log_file = output_dir / f"greedy_{type_prefix}_s{args.s}_q{args.q}_log.txt"
        with open(log_file, 'a') as f:
            f.write(f"n={n}: best={best_size}, mean={avg_size:.2f}+/-{std_size:.2f}, time={elapsed:.1f}s\n")

        # Save best solution
        solution_file = output_dir / f"greedy_{type_prefix}_s{args.s}_n{n}_q{args.q}_best.txt"
        with open(solution_file, 'w') as f:
            for cw in sorted(best_solution):
                f.write(f"{cw}\n")
        print(f"  Saved: {solution_file}")

        results.append({
            'n': n,
            's': args.s,
            'q': args.q,
            'graph_type': args.graph_type,
            'trials': args.trials,
            'best': best_size,
            'mean': round(avg_size, 2),
            'std': round(std_size, 2),
            'max_freq_pct': round(max_freq, 2),
        })

        # Save summary after each n (incremental updates)
        summary_file = output_dir / "greedy_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Save final summary
    summary_file = output_dir / "greedy_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Print summary table
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"{'n':>4} | {'Best':>6} | {'Mean':>10} | {'Max Freq':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['n']:>4} | {r['best']:>6} | {r['mean']:>6.2f}+/-{r['std']:<4.2f} | {r['max_freq_pct']:>8.2f}%")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
