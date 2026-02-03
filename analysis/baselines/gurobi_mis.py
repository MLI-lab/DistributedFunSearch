#!/usr/bin/env python3
"""
Gurobi-based Maximum Independent Set solver.

Finds large independent sets using Gurobi's ILP solver with time limits.
Even without proving optimality, Gurobi finds good feasible solutions quickly.

For very large graphs (n>=17), the model may be too large to build.
Use random_greedy.py with local search for those cases.

Usage:
    # Single run with 1 hour time limit
    python gurobi_mis.py --n 14 --s 3 --timeout 3600

    # Multiple n values
    python gurobi_mis.py --n-values 10,11,12,13,14 --s 3 --timeout 600

    # Use METIS file directly
    python gurobi_mis.py --metis-file /path/to/graph.metis --timeout 3600

Requirements:
    pip install gurobipy
    # Plus valid Gurobi license (gurobi.lic)
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Set
from datetime import datetime

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("WARNING: Gurobi not available. Install with: pip install gurobipy")

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from analysis.baselines.kamis.convert_lmdb_to_metis import get_metis_path
except ImportError:
    def get_metis_path(n, s, q, graph_type, graph_dir):
        """Fallback path construction."""
        if graph_type == "deletions":
            subdir = "deletion"
            prefix = "graph_d"
        else:
            subdir = graph_type
            prefix = f"graph_{graph_type}"
        alphabet = "binary" if q == 2 else "quaternary"
        filename = f"{prefix}_s{s}_n{n}_q{q}.metis"
        return Path(graph_dir) / subdir / alphabet / f"s{s}" / filename


def load_metis_graph(metis_file: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Load a METIS format graph file.

    Returns:
        (num_nodes, edge_list) where edge_list contains (u, v) pairs (0-indexed)
    """
    edges = []
    num_nodes = 0
    num_edges = 0

    print(f"Loading graph from {metis_file}...")
    start_time = time.time()

    with open(metis_file, 'r') as f:
        # Read header
        header = f.readline().strip()
        while header.startswith('%'):  # Skip comments
            header = f.readline().strip()

        parts = header.split()
        num_nodes = int(parts[0])
        num_edges = int(parts[1])

        print(f"  Nodes: {num_nodes:,}, Edges: {num_edges:,}")

        # Check if graph is too large
        if num_edges > 100_000_000:  # 100M edges
            print(f"  WARNING: Graph has {num_edges:,} edges - model may be too large!")
            print(f"  Consider using random_greedy.py for graphs this size.")

        # Read adjacency lists (1-indexed in METIS)
        for node_idx in range(num_nodes):
            line = f.readline().strip()
            if not line:
                continue
            neighbors = [int(x) - 1 for x in line.split()]  # Convert to 0-indexed
            for neighbor in neighbors:
                if neighbor > node_idx:  # Only add each edge once
                    edges.append((node_idx, neighbor))

    load_time = time.time() - start_time
    print(f"  Loaded {len(edges):,} edges in {load_time:.1f}s")

    return num_nodes, edges


def solve_mis_gurobi(
    num_nodes: int,
    edges: List[Tuple[int, int]],
    timeout: float = 3600,
    threads: int = 0,
    verbose: bool = True
) -> Tuple[List[int], float, float, str]:
    """
    Solve MIS using Gurobi ILP.

    Returns:
        (solution_nodes, objective_value, gap, status)
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi not available")

    print(f"\nBuilding Gurobi model...")
    print(f"  Variables: {num_nodes:,}")
    print(f"  Constraints: {len(edges):,}")

    build_start = time.time()

    # Create model
    model = gp.Model("MIS")
    model.Params.TimeLimit = timeout
    model.Params.Threads = threads
    if not verbose:
        model.Params.OutputFlag = 0

    # Binary variable for each node: 1 if in independent set
    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")

    # Objective: maximize sum of x[i]
    model.setObjective(gp.quicksum(x[i] for i in range(num_nodes)), GRB.MAXIMIZE)

    # Constraints: for each edge (u, v), at most one endpoint in IS
    print(f"  Adding {len(edges):,} edge constraints...")
    for u, v in edges:
        model.addConstr(x[u] + x[v] <= 1)

    build_time = time.time() - build_start
    print(f"  Model built in {build_time:.1f}s")

    # Solve
    print(f"\nSolving with {timeout}s time limit...")
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start

    # Extract solution
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    status = status_map.get(model.Status, f"STATUS_{model.Status}")

    solution = []
    obj_val = 0
    gap = float('inf')

    if model.SolCount > 0:
        obj_val = model.ObjVal
        gap = model.MIPGap if hasattr(model, 'MIPGap') else 0
        solution = [i for i in range(num_nodes) if x[i].X > 0.5]

    print(f"\nResults:")
    print(f"  Status: {status}")
    print(f"  Best solution: {len(solution)} nodes")
    print(f"  Best bound: {model.ObjBound:.1f}" if hasattr(model, 'ObjBound') else "")
    print(f"  Gap: {gap*100:.2f}%" if gap < float('inf') else "  Gap: N/A")
    print(f"  Solve time: {solve_time:.1f}s")

    return solution, obj_val, gap, status


def verify_independent_set(solution: List[int], edges: List[Tuple[int, int]]) -> bool:
    """Verify that solution is a valid independent set."""
    solution_set = set(solution)
    for u, v in edges:
        if u in solution_set and v in solution_set:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Gurobi MIS solver")
    parser.add_argument("--metis-file", type=str, help="Path to METIS graph file")
    parser.add_argument("--n", type=int, help="Single n value")
    parser.add_argument("--n-values", type=str, help="Comma-separated n values")
    parser.add_argument("--s", type=int, default=1, help="Number of deletions (default: 1)")
    parser.add_argument("--q", type=int, default=2, help="Alphabet size (default: 2)")
    parser.add_argument("--graph-type", type=str, default="deletions",
                        choices=["deletions", "ids"], help="Graph type")
    parser.add_argument("--graph-dir", type=str, default="/mnt/Graphs",
                        help="Directory with METIS graphs")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Time limit in seconds (default: 3600)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Number of threads (0=auto)")
    parser.add_argument("--output", "-o", type=str, default="./gurobi_results",
                        help="Output directory")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress Gurobi output")

    args = parser.parse_args()

    if not GUROBI_AVAILABLE:
        print("ERROR: Gurobi not available. Install with: pip install gurobipy")
        sys.exit(1)

    # Determine which graphs to solve
    if args.metis_file:
        metis_files = [(args.metis_file, "custom")]
    elif args.n:
        n_values = [args.n]
        metis_files = [(str(get_metis_path(n, args.s, args.q, args.graph_type, args.graph_dir)), f"n{n}")
                       for n in n_values]
    elif args.n_values:
        n_values = [int(x) for x in args.n_values.split(",")]
        metis_files = [(str(get_metis_path(n, args.s, args.q, args.graph_type, args.graph_dir)), f"n{n}")
                       for n in n_values]
    else:
        parser.error("Must specify --metis-file, --n, or --n-values")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results
    results = {
        "solver": "gurobi",
        "timeout": args.timeout,
        "threads": args.threads,
        "timestamp": datetime.now().isoformat(),
        "runs": []
    }

    print("=" * 70)
    print("  Gurobi MIS Solver")
    print("=" * 70)
    print(f"  Timeout: {args.timeout}s")
    print(f"  Threads: {args.threads if args.threads > 0 else 'auto'}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    for metis_file, label in metis_files:
        print(f"\n{'='*70}")
        print(f"  Processing: {label}")
        print(f"  File: {metis_file}")
        print("=" * 70)

        if not os.path.exists(metis_file):
            print(f"  ERROR: File not found: {metis_file}")
            continue

        try:
            # Load graph
            num_nodes, edges = load_metis_graph(metis_file)

            # Solve
            solution, obj_val, gap, status = solve_mis_gurobi(
                num_nodes, edges,
                timeout=args.timeout,
                threads=args.threads,
                verbose=not args.quiet
            )

            # Verify
            if solution:
                is_valid = verify_independent_set(solution, edges)
                print(f"  Valid IS: {is_valid}")
            else:
                is_valid = False

            # Save solution
            if solution:
                solution_file = output_dir / f"gurobi_{label}_solution.txt"
                with open(solution_file, 'w') as f:
                    for node in sorted(solution):
                        f.write(f"{node}\n")
                print(f"  Solution saved: {solution_file}")

            # Record results
            results["runs"].append({
                "label": label,
                "file": metis_file,
                "num_nodes": num_nodes,
                "num_edges": len(edges),
                "solution_size": len(solution),
                "gap": gap if gap < float('inf') else None,
                "status": status,
                "valid": is_valid
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results["runs"].append({
                "label": label,
                "file": metis_file,
                "error": str(e)
            })

    # Save summary
    summary_file = output_dir / "gurobi_results.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Label':<10} {'Nodes':>10} {'Edges':>15} {'IS Size':>10} {'Gap':>10} {'Status':<12}")
    print("-" * 70)
    for run in results["runs"]:
        if "error" in run:
            print(f"  {run['label']:<10} ERROR: {run['error']}")
        else:
            gap_str = f"{run['gap']*100:.1f}%" if run['gap'] is not None else "N/A"
            print(f"  {run['label']:<10} {run['num_nodes']:>10,} {run['num_edges']:>15,} "
                  f"{run['solution_size']:>10} {gap_str:>10} {run['status']:<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
