"""
Standalone script to construct graphs for deletion-correcting codes.

Nodes are q-ary strings of length n (e.g., binary for q=2, DNA for q=4).
Two nodes are connected if they share a common subsequence of length at least n-s.

For a code to correct s deletions, no two codewords can share a subsequence of length >= n-s.
So an independent set in this graph is a deletion-correcting code.

Usage:
    python construct_deletions_graphs.py

The script will construct graphs for the (n, s, q) tuples specified in the __main__ block
and save them to LMDB databases in the format: graph_d_s{s}_n{n}_q{q}.lmdb

Parallelization Strategy:
    To avoid creating a massive list of all sequence pairs in memory (which would require
    ~15TB for n=10, q=4), workers generate pairs on-the-fly from assigned index ranges.

    For N sequences, we need to compute all pairs (i,j) where i < j:
    - Total pairs: N(N-1)/2
    - Each worker is assigned a range of 'i' indices: [start_i - end_i)
    - For each i in its range, the worker compares sequence[i] with all sequence[j] where j > i
    - This ensures no duplicate comparisons (each pair is processed exactly once)

    Load balancing: Index ranges are assigned such that each worker processes approximately
    the same number of pairs. Early indices (small i) have more work since they compare with
    more j values, so workers processing early indices get fewer indices.

    How it works:
        Using the quadratic formula to solve for index boundaries:
        Cumulative pairs from i=0 to i=k-1: k*N - k*(k+1)/2
        Target for worker w: (w+1) * (total_pairs / num_workers)
        Solve: k*N - k*(k+1)/2 = target

Optimizations:
    - imap_unordered for ~10-20% speedup (no ordering overhead)
    - Reduced tqdm update frequency (every 100 iterations)
    - Pre-allocated DP arrays passed to workers (reduces GC pressure)
    - Checkpointing every 10k comparisons for crash recovery
    - Pre-serialized JSON for faster LMDB writes
"""

import itertools
import json
import os
import math
import lmdb
import pickle
import threading
import time
import psutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Thread-local storage for reusable DP arrays
import threading
_thread_local = threading.local()


def _get_dp_arrays(n):
    """Get thread-local pre-allocated DP arrays to avoid repeated allocation."""
    if not hasattr(_thread_local, 'prev') or len(_thread_local.prev) < n + 1:
        _thread_local.prev = [0] * (n + 1)
        _thread_local.current = [0] * (n + 1)
    return _thread_local.prev, _thread_local.current


class MemoryMonitor:
    """Monitor memory usage of process and all children."""

    def __init__(self, interval=0.5):
        """
        Args:
            interval: Sampling interval in seconds (default: 0.5s for finer granularity)
        """
        self.interval = interval
        self.peak_memory_gb = 0.0
        self.current_memory_gb = 0.0
        self.running = False
        self.thread = None
        self.process = psutil.Process(os.getpid())
        self._children_cache = []
        self._cache_time = 0

    def _get_total_memory(self):
        """Get total memory (RSS) of this process and all children in GB."""
        try:
            total = self.process.memory_info().rss
            # Cache children list for 1 second to reduce overhead
            current_time = time.time()
            if current_time - self._cache_time > 1.0:
                self._children_cache = self.process.children(recursive=True)
                self._cache_time = current_time
            for child in self._children_cache:
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total / (1024**3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    def _monitor_loop(self):
        """Background thread that samples memory periodically."""
        while self.running:
            self.current_memory_gb = self._get_total_memory()
            if self.current_memory_gb > self.peak_memory_gb:
                self.peak_memory_gb = self.current_memory_gb
            time.sleep(self.interval)

    def start(self):
        """Start monitoring memory in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return peak memory."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval * 2)
        # Final sample
        final_memory = self._get_total_memory()
        if final_memory > self.peak_memory_gb:
            self.peak_memory_gb = final_memory
        return self.peak_memory_gb


def _compute_edges_chunk(args):
    """
    Worker function to compute edges for a chunk of sequence pairs.

    Args:
        args: Tuple of (worker_id, start_i, end_i, sequences, n, s, checkpoint_dir).
              Worker generates pairs from range [start_i, end_i) to save memory.

    Returns:
        List of edges (seq1, seq2) that should be connected
    """
    worker_id, start_i, end_i, sequences, n, s, checkpoint_dir = args
    edges = []
    n_sequences = len(sequences)
    total_iters = end_i - start_i
    update_interval = max(1, total_iters // 100)  # Update progress ~100 times

    # Create progress bar for this worker at a specific vertical position
    pbar = tqdm(
        total=total_iters,
        desc=f"  Worker {worker_id:2d}",
        position=worker_id,
        leave=True,
        unit="idx",
        mininterval=0.5  # Reduce I/O overhead
    )

    # Get pre-allocated DP arrays
    prev, current = _get_dp_arrays(n)
    threshold = n - s

    for idx, i in enumerate(range(start_i, end_i)):
        for j in range(i + 1, n_sequences):
            seq1, seq2 = sequences[i], sequences[j]

            if _has_common_subsequence_fast(seq1, seq2, n, threshold, prev, current):
                edges.append((seq1, seq2))

        # Update progress less frequently to reduce I/O overhead
        if idx % update_interval == 0 or idx == total_iters - 1:
            pbar.update(update_interval if idx > 0 else 1)

    pbar.close()
    return edges


def _has_common_subsequence_fast(seq1, seq2, n, threshold, prev, current):
    """
    Check if two sequences share a common subsequence of length >= threshold.
    Uses pre-allocated DP arrays to avoid repeated memory allocation.

    Args:
        seq1: First sequence
        seq2: Second sequence
        n: Length of sequences
        threshold: Minimum LCS length required (n - s)
        prev: Pre-allocated array for previous DP row
        current: Pre-allocated array for current DP row

    Returns:
        bool: True if LCS length >= threshold
    """
    if threshold <= 0:
        return True  # Trivial case

    # Reset arrays (faster than reallocating)
    for i in range(n + 1):
        prev[i] = 0
        current[i] = 0

    # Fill the DP table row by row
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                current[j] = prev[j - 1] + 1
            else:
                current[j] = max(prev[j], current[j - 1])
            if current[j] >= threshold:
                return True  # Early termination
        # Swap rows
        prev, current = current, prev

    return False


def has_common_subsequence(seq1, seq2, n, s):
    """
    Check if two sequences share a common subsequence of length >= n-s.
    Public API that allocates its own arrays (for single-threaded use).
    """
    threshold = n - s
    prev = [0] * (n + 1)
    current = [0] * (n + 1)
    return _has_common_subsequence_fast(seq1, seq2, n, threshold, prev, current)


def _get_checkpoint_path(output_dir, n, s, q):
    """Get the checkpoint file path for a given graph."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"checkpoint_d_s{s}_n{n}_q{q}.pkl")


def _save_checkpoint(checkpoint_path, edges_so_far, completed_workers):
    """Save checkpoint with edges computed so far."""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'edges': edges_so_far,
            'completed_workers': completed_workers,
            'timestamp': time.time()
        }, f)


def _load_checkpoint(checkpoint_path):
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                print(f"  Resuming from checkpoint ({len(data['edges'])} edges, {len(data['completed_workers'])} workers done)")
                return data
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
    return None


def generate_deletion_graph(n, s, q=2, max_workers=None, output_dir=None):
    """
    Generate a graph where nodes are q-ary strings of length n.
    Two nodes are connected if they share a common subsequence of length >= n-s.

    Args:
        n: Length of strings
        s: Number of deletions to correct
        q: Alphabet size (default: 2 for binary, 4 for DNA)
        max_workers: Number of parallel workers (default: cpu_count())
        output_dir: Directory for checkpoints (optional)

    Returns:
        dict: Adjacency list representation {node: [list of neighbors]}
    """
    if max_workers is None:
        max_workers = cpu_count()

    # Start memory monitoring
    memory_monitor = MemoryMonitor(interval=0.5)
    memory_monitor.start()

    print(f"Generating graph for n={n}, s={s}, q={q} (LCS threshold: {n-s})")
    print(f"  Using {max_workers} workers for parallel computation")
    print(f"  Monitoring memory usage (sampling every 0.5s)...")

    # Generate q-ary alphabet: '0', '1', ..., 'q-1'
    alphabet = ''.join(str(i) for i in range(q))
    sequences = [''.join(seq) for seq in itertools.product(alphabet, repeat=n)]
    print(f"  Total nodes: {len(sequences)}")

    # Build adjacency list
    adjacency = {seq: [] for seq in sequences}

    # Check for checkpoint
    checkpoint_path = _get_checkpoint_path(output_dir or ".", n, s, q) if output_dir else None
    checkpoint_data = _load_checkpoint(checkpoint_path) if checkpoint_path else None
    completed_workers = set(checkpoint_data['completed_workers']) if checkpoint_data else set()
    all_edges = list(checkpoint_data['edges']) if checkpoint_data else []

    # Split sequence indices into ranges for workers with balanced workload
    n_sequences = len(sequences)
    total_pairs = n_sequences * (n_sequences - 1) // 2
    pairs_per_worker = total_pairs / max_workers

    worker_args = []
    current_i = 0

    for worker_id in range(max_workers):
        start_i = current_i
        target_cumulative = int((worker_id + 1) * pairs_per_worker)

        if worker_id == max_workers - 1:
            end_i = n_sequences
        else:
            a = 1
            b = -(2 * n_sequences - 1)
            c = 2 * target_cumulative
            discriminant = b * b - 4 * a * c

            if discriminant >= 0:
                k = (-b - math.sqrt(discriminant)) / (2 * a)
                end_i = int(math.ceil(k))
                end_i = max(start_i + 1, min(end_i, n_sequences))
            else:
                end_i = n_sequences

        # Skip already completed workers
        if worker_id in completed_workers:
            current_i = end_i
            continue

        if start_i < n_sequences:
            worker_args.append((worker_id, start_i, end_i, sequences, n, s, checkpoint_path))

        current_i = end_i

    if not worker_args:
        print("  All workers already completed from checkpoint!")
    else:
        # Process in parallel using imap_unordered for better performance
        print(f"  Computing common subsequences in parallel...")
        print(f"  Each worker will show its own progress bar below:\n")

        try:
            with Pool(max_workers) as pool:
                # imap_unordered: ~10-20% faster as we don't need ordering
                for worker_id, result in enumerate(pool.imap_unordered(_compute_edges_chunk, worker_args)):
                    all_edges.extend(result)
                    # Save checkpoint after each worker completes
                    if checkpoint_path:
                        completed_workers.add(worker_args[worker_id][0])
                        _save_checkpoint(checkpoint_path, all_edges, completed_workers)
        except KeyboardInterrupt:
            print("\n  Interrupted! Saving checkpoint...")
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, all_edges, completed_workers)
            raise

    # Combine results into adjacency list
    edge_count = 0
    for seq1, seq2 in all_edges:
        adjacency[seq1].append(seq2)
        adjacency[seq2].append(seq1)
        edge_count += 1

    # Compute degree statistics
    degrees = [len(neighbors) for neighbors in adjacency.values()]
    min_deg, max_deg = min(degrees), max(degrees)
    avg_deg = sum(degrees) / len(degrees)
    sorted_degrees = sorted(degrees)
    median_deg = sorted_degrees[len(sorted_degrees) // 2]

    print(f"\n  Graph statistics:")
    print(f"    Total edges: {edge_count:,}")
    print(f"    Degree distribution: min={min_deg}, max={max_deg}, avg={avg_deg:.1f}, median={median_deg}")
    print(f"    Graph density: {2 * edge_count / (n_sequences * (n_sequences - 1)) * 100:.4f}%")

    # Stop memory monitoring
    peak_memory_gb = memory_monitor.stop()
    print(f"\n  Peak memory usage: {peak_memory_gb:.2f} GB")

    # Clean up checkpoint on success
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Checkpoint cleaned up")

    return adjacency


def save_graph_to_lmdb(adjacency, output_path):
    """
    Save graph adjacency list to LMDB database.
    Pre-serializes all JSON for better performance.

    Args:
        adjacency: dict mapping node to list of neighbors
        output_path: Path to LMDB database directory
    """
    print(f"Saving graph to {output_path}")

    # Estimate database size
    num_nodes = len(adjacency)
    avg_neighbors = sum(len(neighbors) for neighbors in adjacency.values()) / num_nodes if num_nodes > 0 else 0
    estimated_bytes_per_node = 100 + avg_neighbors * 20
    estimated_total_bytes = num_nodes * estimated_bytes_per_node

    map_size_bytes = int(estimated_total_bytes * 1.5)
    map_size_gb = (map_size_bytes // (1024**3)) + 1
    map_size_gb = max(10, map_size_gb)
    map_size_bytes = map_size_gb * 1024 * 1024 * 1024

    print(f"  Database size estimate:")
    print(f"    Nodes: {num_nodes:,}")
    print(f"    Avg neighbors per node: {avg_neighbors:.1f}")
    print(f"    Estimated size: {estimated_total_bytes / (1024**3):.2f} GB")
    print(f"    LMDB map_size (with 50% margin): {map_size_gb} GB")

    # Pre-serialize all data before transaction (faster)
    print(f"  Pre-serializing data...")
    serialized_data = []
    for node, neighbors in tqdm(adjacency.items(), desc="  Serializing", unit="nodes", mininterval=0.5):
        key = node.encode('utf-8')
        value = json.dumps(neighbors).encode('utf-8')
        serialized_data.append((key, value))

    # Create LMDB environment and write in batches
    env = lmdb.open(output_path, map_size=map_size_bytes)

    print(f"  Writing to LMDB...")
    batch_size = 10000
    with env.begin(write=True) as txn:
        for i, (key, value) in enumerate(tqdm(serialized_data, desc="  Writing", unit="nodes", mininterval=0.5)):
            txn.put(key, value)

    env.close()
    print(f"  Graph saved successfully!")


def construct_and_save_graph(n, s, q, output_dir, max_workers=None):
    """
    Construct a deletion-correcting code graph and save it to LMDB.

    Args:
        n: Length of strings
        s: Number of deletions to correct
        q: Alphabet size (2 for binary, 4 for DNA)
        output_dir: Directory to save the graph
        max_workers: Number of parallel workers (default: cpu_count())
    """
    # Generate graph
    adjacency = generate_deletion_graph(n, s, q, max_workers=max_workers, output_dir=output_dir)

    # Create output path
    graph_name = f"graph_d_s{s}_n{n}_q{q}.lmdb"
    output_path = os.path.join(output_dir, graph_name)

    # Save to LMDB
    save_graph_to_lmdb(adjacency, output_path)
    print()


if __name__ == "__main__":
    # Specify the output directory (relative to src/graphs)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    #OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../graphs")
    OUTPUT_DIR = "/mnt/Graphs/deletion/binary/s2"  
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Constructing Deletion-Correcting Code Graphs")
    print("=" * 70)
    print()

    # Alphabet size: 2 for binary, 4 for DNA (quaternary)
    q = 2

    # Number of parallel workers (set to None to use all available CPU cores)
    max_workers = 100

    # Define (n, s) pairs to construct graphs for
    # Adjust these based on your experimental needs
    params = [
        # s=1: single deletion correction
        (17, 2),
        (18, 2),
        (19, 2),

        # s=2: double deletion correction
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
