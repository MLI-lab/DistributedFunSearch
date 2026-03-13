"""
Exact LP Upper Bounds for Deletion/Edit Correcting Codes

Computes the optimal fractional transversal bound by solving the LP relaxation.

From Kulkarni-Kiyavash (2013):
- LP gives: optimal fractional transversal (tighter than Theorem 4.1)
- Theorem 4.1 bound available in: kulkarni_kiyavash_upper_bound.py

Actual Code Size <= LP Bound <= Theorem 4.1 Bound

Usage (Deletion codes):
    # Single value (n=10, s=1, q=2)
    python upper_bound_LP.py --n 10

    # Single value with custom s and q
    python upper_bound_LP.py --n 10 --s 2 --q 2

    # Table for s=1 (default), n from 3 to 14
    python upper_bound_LP.py --table --max-n 14

    # Table for s=2, n from 4 to 12
    python upper_bound_LP.py --table --max-n 12 --s 2

Usage (Edit codes - deletion + substitution + insertion):
    # Single edit error, quaternary alphabet
    python upper_bound_LP.py --edit --n 6 --q 4

    # Edit code table
    python upper_bound_LP.py --edit --table --max-n 8 --q 4

    # Edit code with verbose output
    python upper_bound_LP.py --edit --n 5 --q 4 -v

Defaults:
    --s 1       Number of deletions (deletion mode only)
    --q 2       Alphabet size (binary)
    --max-n 12  Maximum n for table mode
    --solver auto  LP solver: 'auto' (gurobi if available), 'gurobi', 'scipy', 'ortools'
    --threads 0    Gurobi threads (0=auto)
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, save_npz, load_npz
from itertools import product, combinations
from typing import Tuple, List, Dict, Set, Optional
from pathlib import Path
import time

# Try to import OR-Tools for faster LP solving
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Try to import Gurobi (fastest commercial solver)
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


# =============================================================================
# Deletion-only functions
# =============================================================================

def deletion_set(x: Tuple[int, ...], s: int) -> Set[Tuple[int, ...]]:
    """
    Compute D_s(x): all distinct strings obtained by deleting s symbols from x.
    """
    n = len(x)
    if s >= n:
        return {()}
    if s == 0:
        return {x}

    result = set()
    for positions in combinations(range(n), s):
        new_str = tuple(x[i] for i in range(n) if i not in positions)
        result.add(new_str)
    return result


def build_hypergraph(n: int, s: int, q: int = 2,
                     cache_dir: Optional[str] = None,
                     checkpoint_interval: int = 100_000,
                     verbose: bool = False) -> Tuple[np.ndarray, List, List]:
    """
    Build the incidence matrix for the hypergraph H^D_{q,s,n}.

    Hypergraph structure:
    - Hyperedges: strings in F_q^n (potential codewords)
    - Vertices: strings in F_q^{n-s} (possible outputs after s deletions)
    - Incidence: vertex y is in hyperedge x iff y in D_s(x)

    Supports checkpointing: saves partial rows/cols to disk every
    checkpoint_interval hyperedges so the build can resume after interruption.
    """
    hyperedges = list(product(range(q), repeat=n))
    vertices = list(product(range(q), repeat=n-s))

    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    num_vertices = len(vertices)
    num_hyperedges = len(hyperedges)

    # Checkpoint paths
    ckpt_path = None
    if cache_dir:
        ckpt_dir = Path(cache_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"build_ckpt_deletion_n{n}_s{s}_q{q}.npz"

    # Resume from checkpoint if available
    start_j = 0
    rows, cols = [], []
    if ckpt_path and ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        start_j = int(ckpt['next_j'])
        rows = ckpt['rows'].tolist()
        cols = ckpt['cols'].tolist()
        if verbose:
            print(f"  Resuming from checkpoint: {start_j}/{num_hyperedges} "
                  f"({start_j/num_hyperedges*100:.1f}%), {len(rows)} non-zeros so far")

    for j in range(start_j, num_hyperedges):
        x = hyperedges[j]
        D_s_x = deletion_set(x, s)
        for y in D_s_x:
            if y in vertex_to_idx:
                i = vertex_to_idx[y]
                rows.append(i)
                cols.append(j)

        if verbose and (j + 1) % 1000 == 0:
            print(f"    Processed {j + 1}/{num_hyperedges} hyperedges...")

        if ckpt_path and (j + 1) % checkpoint_interval == 0:
            np.savez(ckpt_path,
                     rows=np.array(rows, dtype=np.int32),
                     cols=np.array(cols, dtype=np.int32),
                     next_j=j + 1)
            if verbose:
                print(f"    Checkpoint saved at {j + 1}/{num_hyperedges}")

    # Clean up checkpoint after successful completion
    if ckpt_path and ckpt_path.exists():
        ckpt_path.unlink()
        if verbose:
            print(f"  Build complete, checkpoint removed")

    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_vertices, num_hyperedges))

    return A, vertices, hyperedges


# =============================================================================
# Edit distance functions (deletion + substitution + insertion)
# =============================================================================

def edit_ball_1(x: Tuple[int, ...], q: int) -> Set[Tuple[int, ...]]:
    """
    Compute B_1(x): all strings within edit distance 1 from x.

    For x of length n, the 1-edit ball contains:
    - x itself (distance 0)
    - All 1-deletion results: length n-1
    - All 1-substitution results: length n
    - All 1-insertion results: length n+1

    Args:
        x: Input string as tuple of integers
        q: Alphabet size

    Returns:
        Set of tuples representing the 1-edit ball
    """
    n = len(x)
    result = set()

    # Include x itself
    result.add(x)

    # 1-deletions: remove one symbol at each position
    for i in range(n):
        deleted = x[:i] + x[i+1:]
        result.add(deleted)

    # 1-substitutions: change one symbol to each other symbol
    for i in range(n):
        for symbol in range(q):
            if symbol != x[i]:
                substituted = x[:i] + (symbol,) + x[i+1:]
                result.add(substituted)

    # 1-insertions: insert each symbol at each position
    for i in range(n + 1):
        for symbol in range(q):
            inserted = x[:i] + (symbol,) + x[i:]
            result.add(inserted)

    return result


def build_edit_hypergraph(n: int, q: int, verbose: bool = False,
                          cache_dir: Optional[str] = None,
                          checkpoint_interval: int = 100_000) -> Tuple[np.ndarray, List, List]:
    """
    Build the incidence matrix for 1-edit error correcting codes.

    Hypergraph structure:
    - Hyperedges: strings in F_q^n (potential codewords)
    - Vertices: F_q^{n-1} ∪ F_q^n ∪ F_q^{n+1} (all strings in edit balls)
    - Incidence: vertex y is in hyperedge x iff y ∈ B_1(x)

    Supports checkpointing: saves partial rows/cols to disk every
    checkpoint_interval hyperedges so the build can resume after interruption.

    Args:
        n: Codeword length
        q: Alphabet size
        verbose: Print progress
        cache_dir: Directory for checkpoint files (required for resume)
        checkpoint_interval: Save checkpoint every N hyperedges (default: 100k)

    Returns:
        A: Incidence matrix (num_vertices × num_hyperedges)
        vertices: List of vertex strings
        hyperedges: List of hyperedge strings
    """
    # Hyperedges: all strings of length n
    hyperedges = list(product(range(q), repeat=n))

    # Vertices: strings of length n-1, n, and n+1
    vertices_n_minus_1 = list(product(range(q), repeat=n-1)) if n > 1 else [()]
    vertices_n = list(product(range(q), repeat=n))
    vertices_n_plus_1 = list(product(range(q), repeat=n+1))

    vertices = vertices_n_minus_1 + vertices_n + vertices_n_plus_1
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    num_vertices = len(vertices)
    num_hyperedges = len(hyperedges)

    if verbose:
        print(f"  Vertices: {len(vertices_n_minus_1)} (len {n-1}) + "
              f"{len(vertices_n)} (len {n}) + {len(vertices_n_plus_1)} (len {n+1}) "
              f"= {num_vertices}")
        print(f"  Hyperedges: {num_hyperedges}")

    # Checkpoint paths
    ckpt_path = None
    if cache_dir:
        ckpt_dir = Path(cache_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"build_ckpt_edit_n{n}_q{q}.npz"

    # Resume from checkpoint if available
    start_j = 0
    rows, cols = [], []
    if ckpt_path and ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        start_j = int(ckpt['next_j'])
        rows = ckpt['rows'].tolist()
        cols = ckpt['cols'].tolist()
        if verbose:
            print(f"  Resuming from checkpoint: {start_j}/{num_hyperedges} "
                  f"({start_j/num_hyperedges*100:.1f}%), {len(rows)} non-zeros so far")

    # Build sparse incidence matrix
    for j in range(start_j, num_hyperedges):
        x = hyperedges[j]
        ball = edit_ball_1(x, q)
        for y in ball:
            if y in vertex_to_idx:
                i = vertex_to_idx[y]
                rows.append(i)
                cols.append(j)

        if verbose and (j + 1) % 1000 == 0:
            print(f"    Processed {j + 1}/{num_hyperedges} hyperedges...")

        if ckpt_path and (j + 1) % checkpoint_interval == 0:
            np.savez(ckpt_path,
                     rows=np.array(rows, dtype=np.int32),
                     cols=np.array(cols, dtype=np.int32),
                     next_j=j + 1)
            if verbose:
                print(f"    Checkpoint saved at {j + 1}/{num_hyperedges} "
                      f"({len(rows)} non-zeros)")

    # Clean up checkpoint after successful completion
    if ckpt_path and ckpt_path.exists():
        ckpt_path.unlink()
        if verbose:
            print(f"  Build complete, checkpoint removed")

    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_vertices, num_hyperedges))

    if verbose:
        print(f"  Matrix density: {len(rows) / (num_vertices * num_hyperedges):.4f}")
        print(f"  Non-zeros: {len(rows)}")

    return A, vertices, hyperedges


# =============================================================================
# LP solvers
# =============================================================================

def solve_lp_gurobi(A, use_primal: bool, verbose: bool = False,
                    threads: int = 0) -> Tuple[float, Optional[np.ndarray]]:
    """
    Solve LP using Gurobi (fastest commercial solver).

    Args:
        A: Incidence matrix (sparse), shape (num_vertices, num_hyperedges)
        use_primal: If True, solve primal (matching). If False, solve dual (transversal).
        verbose: Print progress
        threads: Number of threads (0 = auto)

    Returns:
        (optimal_value, solution_vector)
    """
    num_vertices, num_hyperedges = A.shape

    # Create model with suppressed output unless verbose
    env = gp.Env(empty=True)
    if not verbose:
        env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(env=env)
    model.Params.Threads = threads
    model.Params.Crossover = 0  # Skip crossover - barrier solution suffices for LP bound

    # Convert sparse matrix to CSC format for efficient column access
    A_csc = A.tocsc()

    if use_primal:
        # Primal: maximize sum z(x), s.t. A @ z <= 1, z >= 0
        if verbose:
            print(f"  [Gurobi] PRIMAL: {num_hyperedges} vars, {num_vertices} constraints")

        # Create variables
        z = model.addVars(num_hyperedges, lb=0, name="z")

        # Add constraints using sparse matrix structure
        A_csr = A.tocsr()
        for i in range(num_vertices):
            row_start, row_end = A_csr.indptr[i], A_csr.indptr[i+1]
            if row_end > row_start:
                cols = A_csr.indices[row_start:row_end]
                model.addConstr(gp.quicksum(z[j] for j in cols) <= 1)

        # Objective: maximize sum z[j]
        model.setObjective(gp.quicksum(z[j] for j in range(num_hyperedges)), GRB.MAXIMIZE)

        # Solve
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi solver failed with status {model.Status}")

        # Extract solution
        solution = np.array([z[j].X for j in range(num_hyperedges)])
        return model.ObjVal, solution

    else:
        # Dual: minimize sum w(y), s.t. A.T @ w >= 1, w >= 0
        if verbose:
            print(f"  [Gurobi] DUAL: {num_vertices} vars, {num_hyperedges} constraints")

        # Create variables
        w = model.addVars(num_vertices, lb=0, name="w")

        # Add constraints: for each hyperedge j, sum of w[i] where A[i,j]=1 >= 1
        for j in range(num_hyperedges):
            col_start, col_end = A_csc.indptr[j], A_csc.indptr[j+1]
            if col_end > col_start:
                rows = A_csc.indices[col_start:col_end]
                model.addConstr(gp.quicksum(w[i] for i in rows) >= 1)

        # Objective: minimize sum w[i]
        model.setObjective(gp.quicksum(w[i] for i in range(num_vertices)), GRB.MINIMIZE)

        # Solve
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi solver failed with status {model.Status}")

        # Extract solution
        solution = np.array([w[i].X for i in range(num_vertices)])
        return model.ObjVal, solution


def solve_lp_ortools(A, use_primal: bool, verbose: bool = False) -> Tuple[float, Optional[np.ndarray]]:
    """
    Solve LP using Google OR-Tools (faster than scipy for large problems).

    Args:
        A: Incidence matrix (sparse), shape (num_vertices, num_hyperedges)
        use_primal: If True, solve primal (matching). If False, solve dual (transversal).
        verbose: Print progress

    Returns:
        (optimal_value, None)  # OR-Tools doesn't easily return solution vector
    """
    num_vertices, num_hyperedges = A.shape

    # Create solver - use GLOP (Google's LP solver)
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError("Could not create OR-Tools GLOP solver")

    # Convert sparse matrix to COO format for efficient iteration
    A_coo = A.tocoo()

    if use_primal:
        # Primal: maximize sum z(x), s.t. A @ z <= 1, z >= 0
        if verbose:
            print(f"  [OR-Tools] PRIMAL: {num_hyperedges} vars, {num_vertices} constraints")

        # Create variables z[j] for each hyperedge
        z = [solver.NumVar(0, solver.infinity(), f'z_{j}') for j in range(num_hyperedges)]

        # Add constraints: for each vertex i, sum of z[j] where A[i,j]=1 <= 1
        # Build constraint matrix efficiently
        vertex_to_hyperedges = [[] for _ in range(num_vertices)]
        for i, j in zip(A_coo.row, A_coo.col):
            vertex_to_hyperedges[i].append(j)

        for i in range(num_vertices):
            if vertex_to_hyperedges[i]:
                constraint = solver.Constraint(0, 1)
                for j in vertex_to_hyperedges[i]:
                    constraint.SetCoefficient(z[j], 1)

        # Objective: maximize sum z[j]
        objective = solver.Objective()
        for j in range(num_hyperedges):
            objective.SetCoefficient(z[j], 1)
        objective.SetMaximization()

    else:
        # Dual: minimize sum w(y), s.t. A.T @ w >= 1, w >= 0
        if verbose:
            print(f"  [OR-Tools] DUAL: {num_vertices} vars, {num_hyperedges} constraints")

        # Create variables w[i] for each vertex
        w = [solver.NumVar(0, solver.infinity(), f'w_{i}') for i in range(num_vertices)]

        # Add constraints: for each hyperedge j, sum of w[i] where A[i,j]=1 >= 1
        hyperedge_to_vertices = [[] for _ in range(num_hyperedges)]
        for i, j in zip(A_coo.row, A_coo.col):
            hyperedge_to_vertices[j].append(i)

        for j in range(num_hyperedges):
            if hyperedge_to_vertices[j]:
                constraint = solver.Constraint(1, solver.infinity())
                for i in hyperedge_to_vertices[j]:
                    constraint.SetCoefficient(w[i], 1)

        # Objective: minimize sum w[i]
        objective = solver.Objective()
        for i in range(num_vertices):
            objective.SetCoefficient(w[i], 1)
        objective.SetMinimization()

    # Solve
    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"OR-Tools solver failed with status {status}")

    return solver.Objective().Value(), None


def solve_lp_scipy(A, use_primal: bool, verbose: bool = False) -> Tuple[float, np.ndarray]:
    """
    Solve LP using scipy (fallback when OR-Tools not available).
    """
    num_vertices, num_hyperedges = A.shape

    if use_primal:
        if verbose:
            print(f"  [scipy] PRIMAL: {num_hyperedges} vars, {num_vertices} constraints")
        c = -np.ones(num_hyperedges)
        A_ub = A
        b_ub = np.ones(num_vertices)
        bounds = [(0, None) for _ in range(num_hyperedges)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs-ds',
                         options={'presolve': True})

        if not result.success:
            raise RuntimeError(f"LP solver failed: {result.message}")

        return -result.fun, result.x
    else:
        if verbose:
            print(f"  [scipy] DUAL: {num_vertices} vars, {num_hyperedges} constraints")
        c = np.ones(num_vertices)
        A_ub = -A.T
        b_ub = -np.ones(num_hyperedges)
        bounds = [(0, None) for _ in range(num_vertices)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs-ds',
                         options={'presolve': True})

        if not result.success:
            raise RuntimeError(f"LP solver failed: {result.message}")

        return result.fun, result.x


def solve_lp_bound(A, verbose: bool = False, solver: str = 'auto',
                   threads: int = 0) -> Tuple[float, Optional[np.ndarray]]:
    """
    Solve LP to get the optimal fractional bound.

    Automatically selects the formulation with fewer variables:

    Primal (Fractional Matching) - when hyperedges < vertices:
        maximize    sum_x z(x)
        subject to  A @ z <= 1   (for all vertices y)
                    z(x) >= 0

    Dual (Fractional Transversal) - when vertices < hyperedges:
        minimize    sum_y w(y)
        subject to  A.T @ w >= 1   (for all hyperedges x)
                    w(y) >= 0

    By strong LP duality, both give the same optimal value.

    Args:
        A: Incidence matrix (sparse or dense), shape (num_vertices, num_hyperedges)
        verbose: Print which formulation is used
        solver: 'auto' (best available), 'gurobi', 'ortools', or 'scipy'
        threads: Number of threads for Gurobi (0 = auto)

    Returns:
        (optimal_value, optimal_solution)
    """
    num_vertices, num_hyperedges = A.shape

    # Choose solver (priority: gurobi > scipy > ortools for this problem type)
    if solver == 'auto':
        if GUROBI_AVAILABLE:
            solver = 'gurobi'
        else:
            solver = 'scipy'  # scipy highs-ipm is faster than ortools GLOP for our problem

    # Choose formulation: fewer variables for Gurobi/OR-Tools,
    # but for scipy prefer fewer constraints (dual) since HiGHS struggles
    # with very large constraint counts.
    if solver == 'scipy':
        use_primal = num_hyperedges < num_vertices and num_vertices < 1_000_000
    else:
        use_primal = num_hyperedges < num_vertices

    if solver == 'gurobi':
        if not GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi not available. Install with: pip install gurobipy")
        if verbose:
            print(f"  Solver: Gurobi")
        return solve_lp_gurobi(A, use_primal, verbose, threads)
    elif solver == 'ortools':
        if not ORTOOLS_AVAILABLE:
            raise RuntimeError("OR-Tools not available. Install with: pip install ortools")
        if verbose:
            print(f"  Solver: OR-Tools")
        return solve_lp_ortools(A, use_primal, verbose)
    else:
        if verbose:
            print(f"  Solver: scipy")
        return solve_lp_scipy(A, use_primal, verbose)


# =============================================================================
# Compute bounds
# =============================================================================

def compute_lp_bound(n: int, s: int, q: int = 2, verbose: bool = False,
                      solver: str = 'auto', threads: int = 0,
                      cache_dir: Optional[str] = None) -> Dict:
    """
    Compute the exact LP upper bound for deletion correcting codes.
    """
    if n <= s:
        return {'lp_bound': float(q**n), 'weights': None}

    num_vertices = q ** (n - s)
    num_hyperedges = q ** n

    # Try loading cached hypergraph
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"hypergraph_deletion_n{n}_s{s}_q{q}.npz"

    if cache_path and cache_path.exists():
        if verbose:
            print(f"Loading cached hypergraph from {cache_path}")
        start = time.time()
        A = load_npz(cache_path)
        build_time = time.time() - start
        if verbose:
            print(f"  Loaded in {build_time:.2f}s (shape: {A.shape}, nnz: {A.nnz})")
    else:
        if verbose:
            print(f"Building deletion hypergraph: n={n}, s={s}, q={q}")
            print(f"  Vertices: {num_vertices}, Hyperedges: {num_hyperedges}")

        start = time.time()
        A, vertices, hyperedges = build_hypergraph(n, s, q, cache_dir=cache_dir,
                                                    verbose=verbose)
        build_time = time.time() - start

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_npz(cache_path, A)
            if verbose:
                print(f"  Saved hypergraph to {cache_path}")

    if verbose:
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Solving LP...")

    start = time.time()
    lp_bound, weights = solve_lp_bound(A, verbose=verbose, solver=solver, threads=threads)
    solve_time = time.time() - start

    if verbose:
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  LP bound: {lp_bound:.6f}")

    return {
        'lp_bound': lp_bound,
        'weights': weights,
        'vertices': vertices,
        'build_time': build_time,
        'solve_time': solve_time
    }


def compute_edit_lp_bound(n: int, q: int = 2, verbose: bool = False,
                           solver: str = 'auto', threads: int = 0,
                           cache_dir: Optional[str] = None) -> Dict:
    """
    Compute the exact LP upper bound for 1-edit error correcting codes.

    Args:
        n: Codeword length
        q: Alphabet size
        verbose: Print progress
        solver: LP solver to use ('auto', 'ortools', 'scipy')
        cache_dir: Directory to cache/load hypergraph matrices

    Returns:
        Dictionary with lp_bound, weights, timing info
    """
    if n < 1:
        return {'lp_bound': 1.0, 'weights': None}

    # Try loading cached hypergraph
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"hypergraph_edit_n{n}_q{q}.npz"

    if cache_path and cache_path.exists():
        if verbose:
            print(f"Loading cached hypergraph from {cache_path}")
        start = time.time()
        A = load_npz(cache_path)
        build_time = time.time() - start
        if verbose:
            print(f"  Loaded in {build_time:.2f}s (shape: {A.shape}, nnz: {A.nnz})")
    else:
        if verbose:
            print(f"Building edit hypergraph: n={n}, q={q}")

        start = time.time()
        A, vertices, hyperedges = build_edit_hypergraph(n, q, verbose,
                                                         cache_dir=cache_dir)
        build_time = time.time() - start

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_npz(cache_path, A)
            if verbose:
                print(f"  Saved hypergraph to {cache_path}")

    if verbose:
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Solving LP...")

    start = time.time()
    lp_bound, weights = solve_lp_bound(A, verbose=verbose, solver=solver, threads=threads)
    solve_time = time.time() - start

    if verbose:
        print(f"  Solve time: {solve_time:.2f}s")
        print(f"  LP bound: {lp_bound:.6f}")

    return {
        'lp_bound': lp_bound,
        'weights': weights,
        'vertices': vertices,
        'build_time': build_time,
        'solve_time': solve_time
    }


# =============================================================================
# Print tables
# =============================================================================

def print_lp_table(max_n: int = 12, s: int = 1, q: int = 2, solver: str = 'auto',
                   threads: int = 0, cache_dir: Optional[str] = None):
    """Print table of LP bounds for deletion codes."""
    print(f"\nLP Upper Bounds for s={s}-deletion codes over F_{q}")
    print("=" * 50)
    print(f"{'n':>4} | {'LP Bound':>15} | {'Floor':>10}")
    print("-" * 50)

    for n in range(s + 2, max_n + 1):
        # No size limit - let the solver try (use --cache-dir for large instances)

        try:
            result = compute_lp_bound(n, s, q, verbose=False, solver=solver,
                                      threads=threads, cache_dir=cache_dir)
            lp_bound = result['lp_bound']
            print(f"{n:>4} | {lp_bound:>15.6f} | {int(lp_bound):>10}")
        except Exception as e:
            print(f"{n:>4} | Error: {e}")

    print("=" * 50)
    print("For Theorem 4.1 bound, see: kulkarni_kiyavash_upper_bound.py")


def print_edit_lp_table(max_n: int = 8, q: int = 2, solver: str = 'auto',
                        threads: int = 0, cache_dir: Optional[str] = None):
    """Print table of LP bounds for 1-edit error correcting codes."""
    print(f"\nLP Upper Bounds for 1-edit error codes over F_{q}")
    print("(1 edit = 1 deletion OR 1 substitution OR 1 insertion)")
    print("=" * 50)
    print(f"{'n':>4} | {'LP Bound':>15} | {'Floor':>10}")
    print("-" * 50)

    for n in range(2, max_n + 1):
        # Check if computation is feasible (relaxed limit for Gurobi)
        num_hyperedges = q**n
        if num_hyperedges > 100000000:  # 100M hyperedges limit
            print(f"{n:>4} | {'(too large)':>15} | {'-':>10}")
            continue

        try:
            result = compute_edit_lp_bound(n, q, verbose=False, solver=solver,
                                           threads=threads, cache_dir=cache_dir)
            lp_bound = result['lp_bound']
            print(f"{n:>4} | {lp_bound:>15.6f} | {int(lp_bound):>10}")
        except Exception as e:
            print(f"{n:>4} | Error: {e}")

    print("=" * 50)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute exact LP bounds for deletion/edit correcting codes'
    )
    parser.add_argument('--n', type=int, help='Codeword length')
    parser.add_argument('--s', type=int, default=1, help='Number of deletions (deletion mode)')
    parser.add_argument('--q', type=int, default=2, help='Alphabet size')
    parser.add_argument('--table', action='store_true', help='Print table')
    parser.add_argument('--max-n', type=int, default=12, help='Max n for table')
    parser.add_argument('--edit', action='store_true',
                        help='Use edit distance (del+sub+ins) instead of deletion-only')
    parser.add_argument('--solver', choices=['auto', 'gurobi', 'ortools', 'scipy'], default='auto',
                        help='LP solver: auto, gurobi, ortools, scipy (default: auto)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads for Gurobi (0=auto)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache hypergraph matrices (saves/loads .npz files)')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if args.edit:
        # Edit distance mode
        if args.table:
            print_edit_lp_table(args.max_n, args.q, solver=args.solver,
                                threads=args.threads, cache_dir=args.cache_dir)
        elif args.n:
            result = compute_edit_lp_bound(args.n, args.q, verbose=args.verbose,
                                           solver=args.solver, threads=args.threads,
                                           cache_dir=args.cache_dir)

            if args.json:
                import json
                output = {
                    'n': args.n,
                    'q': args.q,
                    'mode': 'edit',
                    'lp_bound': result['lp_bound'],
                    'lp_bound_floor': int(result['lp_bound'])
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"\nLP Upper Bound for 1-Edit Error Correcting Codes")
                print(f"{'='*50}")
                print(f"Parameters: n={args.n}, q={args.q}")
                print(f"Edit ball: 1 deletion OR 1 substitution OR 1 insertion")
                print(f"LP bound:   {result['lp_bound']:.6f}")
                print(f"Floor:      {int(result['lp_bound'])}")
        else:
            print("LP Upper Bounds for 1-Edit Error Correcting Codes")
            print("=" * 55)
            print("\nFor q=2 (binary):")
            print_edit_lp_table(max_n=10, q=2, cache_dir=args.cache_dir)
            print("\nFor q=4 (quaternary):")
            print_edit_lp_table(max_n=6, q=4, cache_dir=args.cache_dir)
    else:
        # Deletion-only mode (original behavior)
        if args.table:
            print_lp_table(args.max_n, args.s, args.q, solver=args.solver,
                           threads=args.threads, cache_dir=args.cache_dir)
        elif args.n:
            result = compute_lp_bound(args.n, args.s, args.q, verbose=args.verbose,
                                      solver=args.solver, threads=args.threads,
                                      cache_dir=args.cache_dir)

            if args.json:
                import json
                output = {
                    'n': args.n,
                    's': args.s,
                    'q': args.q,
                    'mode': 'deletion',
                    'lp_bound': result['lp_bound'],
                    'lp_bound_floor': int(result['lp_bound'])
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"\nLP Upper Bound (Optimal Fractional Transversal)")
                print(f"{'='*45}")
                print(f"Parameters: n={args.n}, s={args.s}, q={args.q}")
                print(f"LP bound:   {result['lp_bound']:.6f}")
                print(f"Floor:      {int(result['lp_bound'])}")
                print(f"\nFor Theorem 4.1 bound, see: kulkarni_kiyavash_upper_bound.py")
        else:
            print("Exact LP Upper Bounds for Deletion Correcting Codes")
            print("=" * 55)
            print("\nFor s=1:")
            print_lp_table(max_n=12, s=1, q=2, cache_dir=args.cache_dir)
            print("\nFor s=2:")
            print_lp_table(max_n=10, s=2, q=2, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
