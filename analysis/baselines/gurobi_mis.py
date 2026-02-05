#!/usr/bin/env python3
"""
Gurobi-based Maximum Independent Set solver.

Solves MIS exactly via integer linear programming. Works well on dense graphs
where kernelization based methods (KaMIS) fail. Even when it times out, provides
both a feasible solution (lower bound on MIS size) and the dual bound (upper
bound on MIS size).

MIS-ILP formulation:
    maximize    Σ x_i              (maximize # nodes in IS)
    subject to  x_u + x_v ≤ 1      for each edge (u,v)
                x_i ∈ {0, 1}       for each node i

For very large graphs (n>=17), the model may be too large to build.
Use random_greedy.py with local search for those cases.

Input formats:
    --n/--s/--q:    Auto-constructs path: {graph-dir}/deletion/binary/s{s}/graph_d_s{s}_n{n}_q{q}.metis
    --metis-file:   Use any METIS file directly (bypasses auto-path construction)

METIS format requirements:
    Header: num_nodes num_edges [format]  (format indicator ignored)
    Lines: space separated neighbor indices (1 indexed)
    Unweighted only: vertex/edge weights are not supported. If your METIS file
    has weights (format=1,10,11), the parser will misinterpret them as neighbors.

Usage:
    # Autopath from parameters (looks in /mnt/Graphs by default)
    python gurobi_mis.py --n 14 --s 3
    python gurobi_mis.py --n-values 10,11,12,13,14 --s 3

    # Direct METIS file (any path, bypasses autopath construction)
    python gurobi_mis.py --metis-file /path/to/my_graph.metis

    # Save Gurobi solver logs (detailed progress, cuts, branching)
    python gurobi_mis.py --n 14 --s 3 --log-dir ./gurobi_logs

    # Custom timeout and threads
    python gurobi_mis.py --n 14 --s 3 --timeout 3600 --threads 16

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
    print("Gurobi not available. Install with: pip install gurobipy")

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


def load_metis_graph(metis_file: str) -> Tuple[int, List[Tuple[int, int]], dict]:
    """
    Load a METIS format graph file (unweighted only).

    Returns:
        (num_nodes, edge_list, adjacency) where:
            edge_list contains (u, v) pairs (0 indexed)
            adjacency is dict mapping node -> set of neighbors

    Note:
        Only supports unweighted METIS format. If the format indicator in the
        header suggests weights (1, 10, 11, etc.), a warning is printed.
    """
    edges = []
    adjacency = {}
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

        # Check for format indicator (weights)
        if len(parts) >= 3:
            fmt = int(parts[2])
            if fmt != 0:
                print(f"  Note: METIS format={fmt} suggests weights. This parser")
                print(f"        only supports unweighted graphs (format=0).")
                print(f"        Results may be incorrect if file contains weights.")

        print(f"  Nodes: {num_nodes:,}, Edges: {num_edges:,}")

        # Check if graph is too large
        if num_edges > 100_000_000:
            print(f"  Note: Graph has {num_edges:,} edges, model may be too large.")
            print(f"  Consider using random_greedy.py for graphs this size.")

        # Read adjacency lists (1 indexed in METIS)
        for node_idx in range(num_nodes):
            line = f.readline().strip()
            if not line:
                adjacency[node_idx] = set()
                continue
            neighbors = [int(x) - 1 for x in line.split()]  # Convert to 0 indexed
            adjacency[node_idx] = set(neighbors)
            for neighbor in neighbors:
                if neighbor > node_idx:  # Only add each edge once
                    edges.append((node_idx, neighbor))

    load_time = time.time() - start_time
    print(f"  Loaded {len(edges):,} edges in {load_time:.1f}s")

    return num_nodes, edges, adjacency


def solve_mis_gurobi(
    num_nodes: int,
    edges: List[Tuple[int, int]],
    timeout: float = 86400,
    threads: int = 16,
    mip_gap: float = 0.0,
    log_file: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Solve MIS using Gurobi ILP.

    Args:
        num_nodes: Number of nodes in graph
        edges: List of (u, v) edge tuples (0 indexed)
        timeout: Time limit in seconds (default: 86400 = 24 hours)
        threads: Number of threads (default: 16)
        mip_gap: MIP optimality gap (default: 0 = only stop when optimal or timeout)
        log_file: Path to save Gurobi log (optional)
        verbose: Print Gurobi output during solve

    Returns:
        Dictionary with:
            solution: List of node indices in the independent set
            lower_bound: Best feasible solution size (primal bound)
            upper_bound: MIP bound (dual bound), guaranteed upper bound on optimal
            gap: Relative MIP gap
            status: Solver status string
            solve_time: Total solve time in seconds
            build_time: Model build time in seconds
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi not available")

    print(f"\nBuilding Gurobi model...")
    print(f"  Variables: {num_nodes:,}")
    print(f"  Constraints: {len(edges):,}")

    build_start = time.time()

    # Create model with environment for log file
    env = gp.Env(empty=True)
    if log_file:
        env.setParam('LogFile', log_file)
    if not verbose:
        env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model("MIS", env=env)

    # Set parameters
    model.Params.TimeLimit = timeout
    model.Params.Threads = threads
    model.Params.MIPGap = mip_gap

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
    print(f"\nSolving with timeout={timeout}s, threads={threads}, MIPGap={mip_gap}...")
    if log_file:
        print(f"  Log file: {log_file}")
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
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    status = status_map.get(model.Status, f"STATUS_{model.Status}")

    solution = []
    lower_bound = 0
    upper_bound = float('inf')
    gap = float('inf')

    if model.SolCount > 0:
        lower_bound = int(model.ObjVal)
        solution = [i for i in range(num_nodes) if x[i].X > 0.5]

    # Get upper bound (dual/MIP bound)
    try:
        upper_bound = model.ObjBound
        if model.SolCount > 0:
            gap = model.MIPGap
    except:
        pass

    print(f"\n{'='*50}")
    print(f"  Results")
    print(f"{'='*50}")
    print(f"  Status:       {status}")
    print(f"  Lower bound:  {lower_bound} (best feasible IS size)")
    print(f"  Upper bound:  {upper_bound:.1f} (MIP bound)")
    print(f"  Gap:          {gap*100:.2f}%" if gap < float('inf') else "  Gap: N/A")
    print(f"  Build time:   {build_time:.1f}s")
    print(f"  Solve time:   {solve_time:.1f}s")
    print(f"{'='*50}")

    return {
        'solution': solution,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'gap': gap,
        'status': status,
        'solve_time': solve_time,
        'build_time': build_time
    }


def verify_independent_set(solution: List[int], adjacency: dict) -> bool:
    """
    Verify that solution is a valid independent set.

    Uses adjacency dict for O(|S| * avg_degree) verification instead of
    O(|E|) edge iteration. For small solutions on dense graphs this is
    much faster (e.g., 100 nodes vs 156M edges).
    """
    solution_set = set(solution)
    for node in solution:
        # Check if any neighbor of this node is also in the solution
        if adjacency.get(node, set()) & solution_set:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIS solver with exact ILP formulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single instance with 24h timeout (default)
    python gurobi_mis.py --n 14 --s 3

    # Multiple instances
    python gurobi_mis.py --n-values 10,11,12,13,14 --s 3

    # Direct METIS file input
    python gurobi_mis.py --metis-file /path/to/graph.metis

    # With log files
    python gurobi_mis.py --n 14 --s 3 --log-dir ./logs
        """
    )
    parser.add_argument("--metis-file", type=str, help="Path to METIS graph file")
    parser.add_argument("--n", type=int, help="Single n value")
    parser.add_argument("--n-values", type=str, help="Comma-separated n values")
    parser.add_argument("--s", type=int, default=1, help="Number of deletions (default: 1)")
    parser.add_argument("--q", type=int, default=2, help="Alphabet size (default: 2)")
    parser.add_argument("--graph-type", type=str, default="deletions",
                        choices=["deletions", "ids"], help="Graph type")
    parser.add_argument("--graph-dir", type=str, default="/mnt/Graphs",
                        help="Directory with METIS graphs")
    parser.add_argument("--timeout", type=float, default=86400,
                        help="Time limit in seconds (default: 86400 = 24 hours)")
    parser.add_argument("--threads", type=int, default=16,
                        help="Number of threads (default: 16)")
    parser.add_argument("--mip-gap", type=float, default=0.0,
                        help="MIP optimality gap (default: 0 = only stop when optimal)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory to save Gurobi log files (one per instance)")
    parser.add_argument("--output", "-o", type=str, default="./gurobi_results",
                        help="Output directory for solutions and results")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress Gurobi output during solve")

    args = parser.parse_args()

    if not GUROBI_AVAILABLE:
        print("Gurobi not available. Install with: pip install gurobipy")
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

    # Create log directory if specified
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = None

    # Results
    results = {
        "solver": "gurobi",
        "timeout": args.timeout,
        "threads": args.threads,
        "mip_gap": args.mip_gap,
        "timestamp": datetime.now().isoformat(),
        "runs": []
    }

    num_instances = len(metis_files)
    print("=" * 70)
    print("  Gurobi MIS Solver (ILP)")
    print("=" * 70)
    print(f"  Instances:   {num_instances}")
    print(f"  Timeout:     {args.timeout}s ({args.timeout/3600:.1f} hours)")
    print(f"  Threads:     {args.threads}")
    print(f"  MIP Gap:     {args.mip_gap}")
    print(f"  Output:      {output_dir}")
    if log_dir:
        print(f"  Log dir:     {log_dir}")
    print("=" * 70)

    for idx, (metis_file, label) in enumerate(metis_files, 1):
        print(f"\n{'='*70}")
        print(f"  [{idx}/{num_instances}] Processing: {label}")
        print(f"  File: {metis_file}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        if not os.path.exists(metis_file):
            print(f"  File not found: {metis_file}")
            results["runs"].append({
                "label": label,
                "file": metis_file,
                "error": "File not found"
            })
            continue

        try:
            # Load graph
            num_nodes, edges, adjacency = load_metis_graph(metis_file)

            # Compute density for reference
            max_edges = num_nodes * (num_nodes - 1) // 2
            density = len(edges) / max_edges if max_edges > 0 else 0
            print(f"  Density: {density*100:.1f}%")

            # Set up log file
            log_file = None
            if log_dir:
                log_file = str(log_dir / f"gurobi_{label}.log")

            # Solve
            result = solve_mis_gurobi(
                num_nodes, edges,
                timeout=args.timeout,
                threads=args.threads,
                mip_gap=args.mip_gap,
                log_file=log_file,
                verbose=not args.quiet
            )

            solution = result['solution']

            # Verify (uses adjacency for fast O(|S|) check instead of O(|E|))
            if solution:
                is_valid = verify_independent_set(solution, adjacency)
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
                "density": density,
                "lower_bound": result['lower_bound'],
                "upper_bound": result['upper_bound'] if result['upper_bound'] != float('inf') else None,
                "gap": result['gap'] if result['gap'] != float('inf') else None,
                "status": result['status'],
                "valid": is_valid,
                "build_time": result['build_time'],
                "solve_time": result['solve_time']
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
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
    print("\n" + "=" * 90)
    print("  Summary")
    print("=" * 90)
    print(f"  {'Label':<10} {'Nodes':>10} {'Density':>8} {'Lower':>8} {'Upper':>8} {'Gap':>8} {'Status':<12}")
    print("-" * 90)
    for run in results["runs"]:
        if "error" in run:
            print(f"  {run['label']:<10} error: {run['error']}")
        else:
            gap_str = f"{run['gap']*100:.1f}%" if run['gap'] is not None else "N/A"
            upper_str = f"{run['upper_bound']:.0f}" if run['upper_bound'] is not None else "N/A"
            density_str = f"{run['density']*100:.0f}%"
            print(f"  {run['label']:<10} {run['num_nodes']:>10,} {density_str:>8} "
                  f"{run['lower_bound']:>8} {upper_str:>8} {gap_str:>8} {run['status']:<12}")
    print("=" * 90)
    print(f"\nLower bound = best feasible IS found")
    print(f"Upper bound = MIP bound (proven upper bound on optimal MIS)")
    print(f"Gap = 0% means optimal solution found")


if __name__ == "__main__":
    main()
