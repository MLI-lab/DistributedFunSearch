#!/usr/bin/env python3
"""
Random Greedy Independent Set Baseline

Runs multiple trials with different random seeds, building maximal independent
sets by shuffling node order and greedily selecting nodes.

Outputs solutions in text format (one codeword per line) for easy comparison
with VT codes using compare_to_vt.py.

Usage:
    # Run for specific n values
    python baselines/random_greedy.py --n-values 6,7,8,9,10 --trials 1000

    # Use pre-computed graphs
    python baselines/random_greedy.py --graph-dir /path/to/graphs --n-values 10,11,12

    # Build graphs on-the-fly (slower)
    python baselines/random_greedy.py --n-values 6,7,8 --trials 100
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import lcs_length, are_neighbors


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
    """Load graph from LMDB database."""
    try:
        import lmdb
    except ImportError:
        raise ImportError("lmdb required. Install with: pip install lmdb")

    neighbors = {}
    env = lmdb.open(graph_path, readonly=True, lock=False)
    with env.begin() as txn:
        for key, value in txn.cursor():
            node = key.decode()
            neighbor_list = value.decode().split(',')
            neighbors[node] = set(n for n in neighbor_list if n)
    env.close()
    return neighbors


def random_greedy_mis(
    neighbors: Dict[str, Set[str]],
    seed: int,
    return_set: bool = False
) -> Tuple[int, Optional[List[str]]]:
    """
    Build maximal independent set using random greedy algorithm.

    Args:
        neighbors: Dict mapping each node to its neighbors
        seed: Random seed
        return_set: If True, return the actual set

    Returns:
        (size, set) if return_set else size
    """
    rng = random.Random(seed)
    nodes = list(neighbors.keys())
    rng.shuffle(nodes)

    independent_set = []
    removed = set()

    for node in nodes:
        if node in removed:
            continue
        independent_set.append(node)
        removed.add(node)
        removed.update(neighbors[node])

    if return_set:
        return len(independent_set), independent_set
    return len(independent_set), None


def run_trials(
    neighbors: Dict[str, Set[str]],
    num_trials: int,
    base_seed: int
) -> Tuple[List[int], int, List[str]]:
    """
    Run multiple random greedy trials.

    Returns:
        (sizes, best_size, best_solution)
    """
    sizes = []
    best_size = 0
    best_solution = []

    for i in range(num_trials):
        size, solution = random_greedy_mis(neighbors, base_seed + i, return_set=(i < 100))
        sizes.append(size)

        if size > best_size:
            best_size = size
            if solution:
                best_solution = solution
            else:
                # Re-run to get the actual solution
                _, best_solution = random_greedy_mis(neighbors, base_seed + i, return_set=True)

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
    parser.add_argument('--trials', type=int, default=10000,
                        help='Number of random trials per n (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--graph-dir',
                        help='Directory with pre-computed LMDB graphs')
    parser.add_argument('--output', '-o', default='./greedy_results',
                        help='Output directory for solutions')

    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Random Greedy Baseline")
    print("=" * 70)
    print()
    print(f"n values: {n_values}")
    print(f"s={args.s}, q={args.q}")
    print(f"Trials per n: {args.trials}")
    print(f"Output: {output_dir}")
    print()

    results = []

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"Processing n={n}")
        print("=" * 60)

        # Load or build graph
        if args.graph_dir:
            graph_path = os.path.join(args.graph_dir, f"n{n}_s{args.s}")
            if not os.path.exists(graph_path):
                graph_path = os.path.join(args.graph_dir, f"graph_d_s{args.s}_n{n}_q{args.q}.lmdb")

            if os.path.exists(graph_path):
                print(f"Loading graph from: {graph_path}")
                neighbors = load_graph_from_lmdb(graph_path)
            else:
                print(f"Graph not found at {graph_path}, building on-the-fly...")
                nodes = [''.join(seq) for seq in itertools.product(map(str, range(args.q)), repeat=n)]
                neighbors = build_neighbor_dict(nodes, n, args.s)
        else:
            print("Building graph on-the-fly...")
            nodes = [''.join(seq) for seq in itertools.product(map(str, range(args.q)), repeat=n)]
            print(f"  Nodes: {len(nodes)}")
            neighbors = build_neighbor_dict(nodes, n, args.s)
            print(f"  Graph built")

        print(f"Nodes: {len(neighbors)}")

        # Run trials
        print(f"Running {args.trials} trials...")
        start_time = time.time()
        sizes, best_size, best_solution = run_trials(neighbors, args.trials, args.seed)
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

        # Save best solution
        solution_file = output_dir / f"greedy_s{args.s}_n{n}_q{args.q}_best.txt"
        with open(solution_file, 'w') as f:
            for cw in sorted(best_solution):
                f.write(f"{cw}\n")
        print(f"  Saved: {solution_file}")

        results.append({
            'n': n,
            's': args.s,
            'q': args.q,
            'trials': args.trials,
            'best': best_size,
            'mean': round(avg_size, 2),
            'std': round(std_size, 2),
            'max_freq_pct': round(max_freq, 2),
        })

    # Save summary
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
