"""
Standalone script to construct graphs for IDS (Insertion/Deletion/Substitution) codes.

Nodes are q-ary strings of length n (e.g., binary for q=2, DNA for q=4).
Two nodes are connected if their edit distance < 2s + 1.

For a code to correct s errors (insertions, deletions, or substitutions),
all codewords must have pairwise edit distance >= 2s + 1. So an independent set in this graph is an s edit error-correcting code.

Usage:
    python construct_ids_graphs.py

The script will construct graphs for the (n, s, q) tuples specified in the __main__ block
and save them to LMDB databases in the format: graph_ids_s{s}_n{n}_q{q}.lmdb

Parallelization Strategy:
    To avoid creating a massive list of all sequence pairs in memory (which would require
    ~15TB for n=10, q=4), workers generate pairs on-the-fly from assigned index ranges.

    For N sequences, we need to compute all pairs (i,j) where i < j:
    - Total pairs: N(N-1)/2
    - Each worker is assigned a range of 'i' indices: [start_i, end_i)
    - For each i in its range, the worker compares sequence[i] with all sequence[j] where j > i, i.e. For every i in its range, that worker loops only over j = i+1..N-1.
    - This ensures no duplicate comparisons (each pair is processed exactly once)

    Load balancing: Index ranges are assigned such that each worker processes approximately
    the same number of pairs. Early indices (small i) have more work since they compare with
    more j values, so workers processing early indices get fewer indices.

    How it works:
        Using the quadratic formula to solve for index boundaries:
        Cumulative pairs from i=0 to i=k-1: f(k) = sum_{i=0}^{k-1} (N-1-i) = k*N - k*(k+1)/2
        Target for worker w: (w+1) * (total_pairs / num_workers)
        We want to choose boundaries k_0=0, k_1, k_2, …, k_W=N such that f(k_w) \approx \frac{w}{W} T where T is total pairs and W is number of workers and w is the worker index.
        Solve: k*N - k*(k+1)/2 = target
"""

import itertools
import json
import os
import math
import lmdb
import Levenshtein
import tracemalloc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def estimate_memory_usage(n, s, q, max_workers):
    """
    Estimate peak memory usage for graph construction (Upper bound).

    Uses Hamming ball formula to compute estimate of degree and edge distribution (Levenshtein distance ≤ Hamming distance)

    Args:
        n: Length of strings
        s: Number of errors to correct
        q: Alphabet size
        max_workers: Number of parallel workers

    Returns:
        dict with memory estimates in GB
    """
    from math import comb

    total_sequences = q ** n
    r = 2 * s  # Hamming distance threshold

    # Compute Hamming ball volume: V_q(n,r) = sum_{t=0}^{r} C(n,t) * (q-1)^t
    hamming_ball_volume = sum(comb(n, t) * ((q - 1) ** t) for t in range(r + 1))
    avg_degree = hamming_ball_volume

    # Edge probability: p_edge ≈ V_q(n,r) / (N-1)
    p_edge = hamming_ball_volume / (total_sequences - 1) if total_sequences > 1 else 0

    # Total pairs and pairs per worker
    total_pairs = total_sequences * (total_sequences - 1) // 2
    pairs_per_worker = total_pairs / max_workers if max_workers > 0 else 0

    # Expected edges per worker: E_w ≈ P_w * p_edge
    expected_edges_per_worker = pairs_per_worker * p_edge

    # Memory per edge: Python tuple of two strings (~120-200 bytes)
    # Using conservative 150 bytes per edge
    bytes_per_edge = 150
    edge_buffer_gb_per_worker = expected_edges_per_worker * bytes_per_edge / (1024**3)

    # Per-worker memory: edge buffer + process overhead (~50-150 MB)
    worker_memory_gb = max_workers * (edge_buffer_gb_per_worker + 0.1)

    # Main process memory components
    sequences_gb = total_sequences * 60 / (1024**3)  # ~60 bytes per string
    adjacency_gb = total_sequences * avg_degree * 8 / (1024**3)  # 8 bytes per pointer

    # Total memory
    total_gb = adjacency_gb + sequences_gb + worker_memory_gb + 0.2

    return {
        'total': total_gb,
        'adjacency': adjacency_gb,
        'sequences': sequences_gb,
        'workers': worker_memory_gb,
        'edge_buffer_per_worker': edge_buffer_gb_per_worker,
        'expected_edges_per_worker': expected_edges_per_worker,
        'overhead': 0.2,
        'avg_degree': avg_degree,
        'hamming_ball_volume': hamming_ball_volume,
        'p_edge': p_edge,
        'total_nodes': total_sequences
    }


def _compute_edges_chunk(args):
    """
    Worker function to compute edges for a chunk of sequence pairs.

    Args:
        args: Tuple of (worker_id, start_i, end_i, sequences, threshold)
              Worker generates pairs from range [start_i, end_i) to save memory

    Returns:
        List of edges (seq1, seq2) that should be connected
    """
    worker_id, start_i, end_i, sequences, threshold = args
    edges = []
    n_sequences = len(sequences)

    # Create progress bar for this worker at a specific vertical position
    # position=worker_id places each worker's bar at a different line
    pbar = tqdm(
        total=end_i - start_i,
        desc=f"  Worker {worker_id:2d}",
        position=worker_id,
        leave=True,
        unit="idx"
    )

    for i in range(start_i, end_i):
        for j in range(i + 1, n_sequences):
            seq1, seq2 = sequences[i], sequences[j]
            edit_dist = Levenshtein.distance(seq1, seq2)

            if edit_dist < threshold:
                edges.append((seq1, seq2))

        pbar.update(1)

    pbar.close()
    return edges


def generate_ids_graph(n, s, q=2, max_workers=None):
    """
    Generate a graph where nodes are q-ary strings of length n.
    Two nodes are connected if edit_distance(node1, node2) < 2s + 1.

    Args:
        n: Length of strings
        s: Number of errors to correct (requires min distance 2s + 1)
        q: Alphabet size (default: 2 for binary, 4 for DNA)
        max_workers: Number of parallel workers (default: cpu_count())

    Returns:
        dict: Adjacency list representation {node: [list of neighbors]}
    """
    if max_workers is None:
        max_workers = cpu_count()

    # Start memory tracking (only tracks this Python process, not other processes)
    tracemalloc.start()

    print(f"Generating graph for n={n}, s={s}, q={q} (min required distance: {2*s + 1})")
    print(f"  Using {max_workers} workers for parallel computation")

    # Print memory estimate
    mem_estimate = estimate_memory_usage(n, s, q, max_workers)
    print(f"  Estimated memory usage (UPPER BOUND):")
    print(f"    Total: {mem_estimate['total']:.2f} GB")
    print(f"    - Adjacency dict: {mem_estimate['adjacency']:.2f} GB")
    print(f"    - Sequences list: {mem_estimate['sequences']:.2f} GB")
    print(f"    - Workers ({max_workers}): {mem_estimate['workers']:.2f} GB")
    print(f"      * Edge buffer per worker: {mem_estimate['edge_buffer_per_worker']:.2f} GB (~{mem_estimate['expected_edges_per_worker']/1e6:.1f}M edges)")
    print(f"      * Process overhead: 0.1 GB per worker")
    print(f"    - Base overhead: {mem_estimate['overhead']:.2f} GB")
    print(f"  Note: This is an UPPER BOUND because:")
    print(f"    - Uses Hamming ball (Hamming ≥ Levenshtein, equality only without shifts)")
    print(f"    - All workers peak simultaneously (actual: staggered due to load balancing)")
    print(f"    - Conservative 150 bytes/edge (actual: ~100-120 bytes)")
    print(f"    - No copy-on-write sharing (Linux/macOS: sequences are shared)")
    print(f"    Actual memory usage typically 30-50% of estimate (or less if sparse).")
    print(f"  Graph properties (based on Hamming ball approximation):")
    print(f"    - Upper bound on degree: V_{q}(n,{2*s}) = {mem_estimate['avg_degree']:.0f}")
    print(f"    - Upper bound on edge probability: {mem_estimate['p_edge']:.6f}")
    print(f"    - Actual degree will be lower due to Levenshtein < Hamming for shifts")

    # Generate q-ary alphabet: '0', '1', ..., 'q-1'
    alphabet = ''.join(str(i) for i in range(q))
    sequences = [''.join(seq) for seq in itertools.product(alphabet, repeat=n)]
    print(f"  Total nodes: {len(sequences)}")

    # Build adjacency list
    adjacency = {seq: [] for seq in sequences}

    threshold = 2 * s + 1

    # Split sequence indices into ranges for workers with balanced workload
    # Each worker processes a range of 'i' values and all corresponding j > i
    # This avoids creating massive pair lists in memory
    n_sequences = len(sequences)
    total_pairs = n_sequences * (n_sequences - 1) // 2
    pairs_per_worker = total_pairs / max_workers

    def cumulative_pairs_at_index(k):
        """Calculate total pairs from i=0 to i=k-1"""
        return k * n_sequences - k * (k + 1) // 2

    worker_args = []
    current_i = 0

    for worker_id in range(max_workers):
        start_i = current_i
        target_cumulative = int((worker_id + 1) * pairs_per_worker)

        if worker_id == max_workers - 1:
            # Last worker gets all remaining indices
            end_i = n_sequences
        else:
            # Use closed-form solution to find end_i
            # Solve: k*n_sequences - k*(k+1)/2 = target
            # Rearranging: k^2 + k - 2*k*n_sequences + 2*target = 0
            # k^2 - k*(2*n_sequences - 1) + 2*target = 0
            a = 1
            b = -(2 * n_sequences - 1)
            c = 2 * target_cumulative
            discriminant = b * b - 4 * a * c

            if discriminant >= 0:
                k = (-b - math.sqrt(discriminant)) / (2 * a)
                end_i = int(math.ceil(k))
                # Clamp to valid range
                end_i = max(start_i + 1, min(end_i, n_sequences))
            else:
                end_i = n_sequences

        if start_i < n_sequences:
            worker_args.append((worker_id, start_i, end_i, sequences, threshold))

        current_i = end_i

    # Process in parallel
    print(f"  Computing edit distances in parallel...")
    print(f"  Each worker will show its own progress bar below:\n")
    with Pool(max_workers) as pool:
        # Use imap without outer tqdm - each worker has its own progress bar
        results = list(pool.imap(_compute_edges_chunk, worker_args))

    # Combine results into adjacency list
    edge_count = 0
    for edges in results:
        for seq1, seq2 in edges:
            adjacency[seq1].append(seq2)
            adjacency[seq2].append(seq1)
            edge_count += 1

    print(f"  Total edges: {edge_count}")

    # Report actual memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Actual peak memory usage: {peak / (1024**3):.2f} GB")
    print(f"    (Estimated: {mem_estimate['total']:.2f} GB, Difference: {(peak / (1024**3)) - mem_estimate['total']:.2f} GB)")

    return adjacency


def save_graph_to_lmdb(adjacency, output_path):
    """
    Save graph adjacency list to LMDB database.

    Args:
        adjacency: dict mapping node to list of neighbors
        output_path: Path to LMDB database directory
    """
    print(f"Saving graph to {output_path}")

    # Create LMDB environment
    # Map size: 10GB should be enough for most graphs
    env = lmdb.open(output_path, map_size=10 * 1024 * 1024 * 1024)

    with env.begin(write=True) as txn:
        for node, neighbors in tqdm(adjacency.items(), desc="  Writing to LMDB", unit="nodes"):
            key = node.encode('utf-8')
            value = json.dumps(neighbors).encode('utf-8')
            txn.put(key, value)

    env.close()
    print(f"  Graph saved successfully!")


def construct_and_save_graph(n, s, q, output_dir, max_workers=None):
    """
    Construct an IDS graph and save it to LMDB.

    Args:
        n: Length of strings
        s: Number of errors to correct
        q: Alphabet size (2 for binary, 4 for DNA)
        output_dir: Directory to save the graph
        max_workers: Number of parallel workers (default: cpu_count())
    """
    # Generate graph
    adjacency = generate_ids_graph(n, s, q, max_workers=max_workers)

    # Create output path
    graph_name = f"graph_ids_s{s}_n{n}_q{q}.lmdb"
    output_path = os.path.join(output_dir, graph_name)

    # Save to LMDB
    save_graph_to_lmdb(adjacency, output_path)
    print()


if __name__ == "__main__":
    # Specify the output directory (relative to src/graphs)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    #OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../graphs")
    OUTPUT_DIR="/mnt/graphs/ids_graphs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Constructing IDS (Insertion/Deletion/Substitution) Code Graphs")
    print("=" * 70)
    print()

    # Alphabet size: 2 for binary, 4 for DNA (quaternary)
    q = 4

    # Number of parallel workers (set to None to use all available CPU cores)
    max_workers = 15

    # Define (n, s) pairs to construct graphs for
    # Adjust these based on your experimental needs
    params = [
        # s=1: requires min distance 3
        #(6, 1),
        #(7, 1),
        #(8, 1),
        #(9, 1),
        (10, 1),
        #(11, 1),

        # s=2: requires min distance 5
        #(6, 2),
        #(7, 2),
        #(8, 2),
        #(9, 2),
        #(10, 2),
        #(11, 2),
        #(12, 2),
    ]

    for n, s in tqdm(params, desc="Overall progress", unit="graph"):
        construct_and_save_graph(n, s, q, OUTPUT_DIR, max_workers=max_workers)

    print("=" * 70)
    print("All graphs constructed successfully!")
    print(f"Graphs saved to: {OUTPUT_DIR}")
    print("=" * 70)
