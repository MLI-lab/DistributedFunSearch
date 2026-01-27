"""
Construct graphs for IDS (Insertion/Deletion/Substitution) codes.

Nodes are q-ary strings of length n. Two nodes are connected if their edit
distance < 2s+1. An independent set is an s-error-correcting code.

Usage:
    python construct_ids_graphs.py --n 8,9,10 --s 1 --q 4 --workers 50
    python construct_ids_graphs.py --n 12 --s 2 --q 2 --workers 100 --output /mnt/Graphs

Arguments:
    --n         Comma-separated string lengths (required)
    --s         Number of errors to correct, requires distance >= 2s+1 (required)
    --q         Alphabet size: 2=binary, 4=DNA (default: 4)
    --workers   Parallel workers (default: all CPUs)
    --output    Output directory (default: /mnt/Graphs/ids/{alphabet}/s{s})
    --stream    Force streaming mode (writes edges to disk instead of memory)
    --no-stream Force in-memory mode even for large n

Output: graph_ids_s{s}_n{n}_q{q}.lmdb

Parallelization: Workers process index ranges [start_i, end_i) with load balancing
(early indices have more pairs, so get fewer indices). Checkpoints after each worker.

Memory efficiency:
- For n < 19: In-memory mode (fast, holds all edges in RAM)
- For n >= 19: Streaming mode (writes edges to temp files, ~10-20% slower but O(1) edge memory)
"""

import itertools
import json
import os
import math
import lmdb
import Levenshtein
import pickle
import psutil
import shutil
import tempfile
import threading
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


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


def estimate_memory_usage(n, s, q, max_workers):
    """
    Estimate peak memory usage for graph construction (Upper bound).

    Uses Hamming ball formula to compute estimate of degree and edge distribution (Levenshtein distance <= Hamming distance)

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

    # Edge probability: p_edge ~ V_q(n,r) / (N-1)
    p_edge = hamming_ball_volume / (total_sequences - 1) if total_sequences > 1 else 0

    # Total pairs and pairs per worker
    total_pairs = total_sequences * (total_sequences - 1) // 2
    pairs_per_worker = total_pairs / max_workers if max_workers > 0 else 0

    # Expected edges per worker: E_w ~ P_w * p_edge
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
    Worker function to compute edges for a chunk of sequence pairs (in-memory mode).

    Args:
        args: Tuple of (worker_id, start_i, end_i, sequences, threshold).
              Worker generates pairs from range [start_i, end_i) to save memory.

    Returns:
        List of edges (seq1, seq2) that should be connected
    """
    worker_id, start_i, end_i, sequences, threshold = args
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

    for idx, i in enumerate(range(start_i, end_i)):
        for j in range(i + 1, n_sequences):
            seq1, seq2 = sequences[i], sequences[j]
            edit_dist = Levenshtein.distance(seq1, seq2)

            if edit_dist < threshold:
                edges.append((seq1, seq2))

        # Update progress less frequently to reduce I/O overhead
        if idx % update_interval == 0 or idx == total_iters - 1:
            pbar.update(update_interval if idx > 0 else 1)

    pbar.close()
    return edges


def _compute_edges_chunk_streaming(args):
    """
    Worker function to compute edges and write to disk (streaming mode).

    Args:
        args: Tuple of (worker_id, start_i, end_i, sequences, threshold, temp_dir).
              Worker writes edges to temp_dir/worker_{worker_id}.edges

    Returns:
        Tuple of (worker_id, edge_count, output_file_path)
    """
    worker_id, start_i, end_i, sequences, threshold, temp_dir = args
    n_sequences = len(sequences)
    total_iters = end_i - start_i
    update_interval = max(1, total_iters // 100)

    # Output file for this worker
    output_file = os.path.join(temp_dir, f"worker_{worker_id}.edges")
    edge_count = 0

    # Create progress bar
    pbar = tqdm(
        total=total_iters,
        desc=f"  Worker {worker_id:2d}",
        position=worker_id,
        leave=True,
        unit="idx",
        mininterval=0.5
    )

    # Write edges to file with buffering
    with open(output_file, 'w', buffering=8192*16) as f:  # 128KB buffer
        for idx, i in enumerate(range(start_i, end_i)):
            for j in range(i + 1, n_sequences):
                seq1, seq2 = sequences[i], sequences[j]
                edit_dist = Levenshtein.distance(seq1, seq2)

                if edit_dist < threshold:
                    f.write(f"{seq1}\t{seq2}\n")
                    edge_count += 1

            # Update progress
            if idx % update_interval == 0 or idx == total_iters - 1:
                pbar.update(update_interval if idx > 0 else 1)

    pbar.close()

    # Write done marker so resume logic can distinguish complete vs partial files
    done_file = os.path.join(temp_dir, f"worker_{worker_id}.done")
    with open(done_file, 'w') as df:
        df.write(str(edge_count))

    return (worker_id, edge_count, output_file)


def _get_checkpoint_path(output_dir, n, s, q):
    """Get the checkpoint file path for a given graph."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"checkpoint_ids_s{s}_n{n}_q{q}.pkl")


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


def generate_ids_graph(n, s, q=2, max_workers=None, output_dir=None, stream_to_disk=None):
    """
    Generate a graph where nodes are q-ary strings of length n.
    Two nodes are connected if edit_distance(node1, node2) < 2s + 1.

    Args:
        n: Length of strings
        s: Number of errors to correct (requires min distance 2s + 1)
        q: Alphabet size (default: 2 for binary, 4 for DNA)
        max_workers: Number of parallel workers (default: cpu_count())
        output_dir: Directory for checkpoints (optional)
        stream_to_disk: If True, write edges to temp files instead of memory.
                       If None, auto-enable for n >= 19.

    Returns:
        dict: Adjacency list representation {node: [list of neighbors]}
              OR tuple (temp_dir, edge_count, sequences) if stream_to_disk=True
    """
    if max_workers is None:
        max_workers = cpu_count()

    # Auto-enable streaming for large graphs
    if stream_to_disk is None:
        stream_to_disk = (n >= 19)

    # Start memory monitoring (tracks main process + all worker processes)
    memory_monitor = MemoryMonitor(interval=0.5)
    memory_monitor.start()

    mode_str = "STREAMING (disk)" if stream_to_disk else "IN-MEMORY"
    print(f"Generating graph for n={n}, s={s}, q={q} (min required distance: {2*s + 1})")
    print(f"  Mode: {mode_str}")
    print(f"  Using {max_workers} workers for parallel computation")
    print(f"  Monitoring memory usage (sampling every 0.5s)...")

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
    print(f"    - Uses Hamming ball (Hamming >= Levenshtein, equality only without shifts)")
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

    n_sequences = len(sequences)
    total_pairs = n_sequences * (n_sequences - 1) // 2
    pairs_per_worker = total_pairs / max_workers
    threshold = 2 * s + 1

    # Create temp directory for streaming mode (deterministic name for resume support)
    temp_dir = None
    total_resumed_edges = 0
    if stream_to_disk:
        if output_dir:
            temp_base = os.path.join(output_dir, "temp")
            os.makedirs(temp_base, exist_ok=True)
            temp_dir = os.path.join(temp_base, f"graph_edges_n{n}_s{s}_q{q}")
        else:
            temp_dir = os.path.join(tempfile.gettempdir(), f"graph_edges_n{n}_s{s}_q{q}")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"  Temp directory for edges: {temp_dir}")

    # Build worker arguments with balanced workload
    worker_args = []
    current_i = 0
    checkpoint_path = _get_checkpoint_path(output_dir or ".", n, s, q) if output_dir and not stream_to_disk else None

    # For streaming mode, check for resumable state via done marker files
    completed_workers = set()
    if stream_to_disk and temp_dir:
        metadata_file = os.path.join(temp_dir, "_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as mf:
                    metadata = json.load(mf)
                if metadata.get('max_workers') != max_workers:
                    print(f"  Worker count changed ({metadata.get('max_workers')} -> {max_workers}), clearing temp dir...")
                    for f in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, f))
                else:
                    # Scan for completed workers
                    for f in os.listdir(temp_dir):
                        if f.endswith('.done'):
                            wid = int(f.replace('worker_', '').replace('.done', ''))
                            try:
                                with open(os.path.join(temp_dir, f), 'r') as df:
                                    edge_count = int(df.read().strip())
                                total_resumed_edges += edge_count
                                completed_workers.add(wid)
                            except (ValueError, IOError):
                                pass
                    # Clean up partial edge files (no matching .done)
                    for f in os.listdir(temp_dir):
                        if f.endswith('.edges'):
                            wid = int(f.replace('worker_', '').replace('.edges', ''))
                            if wid not in completed_workers:
                                os.remove(os.path.join(temp_dir, f))
                                print(f"  Removed partial edge file: {f}")
                    if completed_workers:
                        print(f"  Resuming: {len(completed_workers)} workers already done ({total_resumed_edges:,} edges)")
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not read metadata: {e}, starting fresh")
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))

        # Write metadata for future resume
        with open(metadata_file, 'w') as mf:
            json.dump({'max_workers': max_workers}, mf)

    elif not stream_to_disk:
        if checkpoint_path:
            checkpoint_data = _load_checkpoint(checkpoint_path)
            if checkpoint_data:
                completed_workers = set(checkpoint_data['completed_workers'])

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
            if stream_to_disk:
                worker_args.append((worker_id, start_i, end_i, sequences, threshold, temp_dir))
            else:
                worker_args.append((worker_id, start_i, end_i, sequences, threshold))

        current_i = end_i

    if not worker_args:
        if stream_to_disk:
            print(f"  All workers already completed! ({total_resumed_edges:,} edges from previous run)")
            memory_monitor.stop()
            return (temp_dir, total_resumed_edges, sequences)
        else:
            print("  All workers already completed from checkpoint!")
            all_edges = []
            if checkpoint_path:
                checkpoint_data = _load_checkpoint(checkpoint_path)
                if checkpoint_data:
                    all_edges = checkpoint_data['edges']
    else:
        print(f"  Computing edit distances in parallel...")
        print(f"  Each worker will show its own progress bar below:\n")

        if stream_to_disk:
            # Streaming mode: workers write to temp files
            total_edge_count = total_resumed_edges
            edge_files = []

            try:
                with Pool(len(worker_args)) as pool:
                    for result in pool.imap_unordered(_compute_edges_chunk_streaming, worker_args):
                        worker_id, edge_count, output_file = result
                        total_edge_count += edge_count
                        edge_files.append(output_file)
            except KeyboardInterrupt:
                print("\n  Interrupted! Temp files preserved in:", temp_dir)
                raise

            # Stop memory monitoring
            peak_memory_gb = memory_monitor.stop()
            print(f"\n  Edge computation complete:")
            print(f"    Total edges: {total_edge_count:,}")
            if total_resumed_edges > 0:
                print(f"    (includes {total_resumed_edges:,} edges from resumed workers)")
            print(f"    Peak memory during computation: {peak_memory_gb:.2f} GB")
            print(f"    Estimated upper bound: {mem_estimate['total']:.2f} GB")

            # Return temp_dir info for streaming LMDB build
            return (temp_dir, total_edge_count, sequences)

        else:
            # In-memory mode: collect edges in list
            all_edges = []
            if checkpoint_path:
                checkpoint_data = _load_checkpoint(checkpoint_path)
                if checkpoint_data:
                    all_edges = list(checkpoint_data['edges'])

            try:
                with Pool(max_workers) as pool:
                    for worker_id, result in enumerate(pool.imap_unordered(_compute_edges_chunk, worker_args)):
                        all_edges.extend(result)
                        if checkpoint_path:
                            completed_workers.add(worker_args[worker_id][0])
                            _save_checkpoint(checkpoint_path, all_edges, completed_workers)
            except KeyboardInterrupt:
                print("\n  Interrupted! Saving checkpoint...")
                if checkpoint_path:
                    _save_checkpoint(checkpoint_path, all_edges, completed_workers)
                raise

    # Build adjacency list (in-memory mode only)
    adjacency = {seq: [] for seq in sequences}
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

    # Stop memory monitoring and report actual usage
    peak_memory_gb = memory_monitor.stop()
    print(f"\n  ACTUAL peak memory usage: {peak_memory_gb:.2f} GB (main + all {max_workers} workers)")
    print(f"    Estimated upper bound: {mem_estimate['total']:.2f} GB")
    print(f"    Actual vs estimate: {peak_memory_gb / mem_estimate['total'] * 100:.1f}% of upper bound")
    if peak_memory_gb < mem_estimate['total']:
        print(f"    Within estimate (saved {mem_estimate['total'] - peak_memory_gb:.2f} GB)")
    else:
        print(f"    WARNING: Exceeded estimate by {peak_memory_gb - mem_estimate['total']:.2f} GB")

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

    # Create LMDB environment and write
    env = lmdb.open(output_path, map_size=map_size_bytes)

    print(f"  Writing to LMDB...")
    with env.begin(write=True) as txn:
        for key, value in tqdm(serialized_data, desc="  Writing", unit="nodes", mininterval=0.5):
            txn.put(key, value)

    env.close()
    print(f"  Graph saved successfully!")


def save_graph_to_lmdb_streaming(temp_dir, sequences, output_path, chunked=None,
                                  nodes_per_chunk=32768):
    """
    Build LMDB from edge temp files (streaming mode).

    Two modes:
    - Standard (n < 19): Build full adjacency dict in memory, then write to LMDB
    - Chunked (n >= 19): Process nodes in chunks, scan edge files per chunk

    Args:
        temp_dir: Directory containing worker edge files
        sequences: List of all node sequences
        output_path: Path to LMDB database directory
        chunked: If True, use chunked mode. Auto-detected based on n if None.
        nodes_per_chunk: Nodes to process per chunk in chunked mode (default: 32K)
    """
    print(f"Saving graph to {output_path} (streaming mode)")

    # Get list of edge files
    edge_files_names = sorted([f for f in os.listdir(temp_dir) if f.endswith('.edges')])
    edge_files = [os.path.join(temp_dir, f) for f in edge_files_names]
    print(f"  Found {len(edge_files)} edge files in {temp_dir}")

    num_nodes = len(sequences)
    n = len(sequences[0]) if sequences else 0

    # Auto-detect chunked mode for n >= 19
    if chunked is None:
        chunked = (n >= 19)

    mode_str = "CHUNKED (low memory)" if chunked else "STANDARD (in-memory)"
    print(f"  LMDB build mode: {mode_str}")

    # Count total edges
    print(f"  Counting edges...")
    total_edges = 0
    for fpath in edge_files:
        with open(fpath, 'r') as f:
            total_edges += sum(1 for _ in f)
    print(f"  Total edges: {total_edges:,}")

    # Estimate LMDB size
    avg_degree = (2 * total_edges) / num_nodes if num_nodes > 0 else 0
    estimated_bytes_per_node = 100 + avg_degree * 20
    estimated_total_bytes = num_nodes * estimated_bytes_per_node
    map_size_gb = max(10, int(estimated_total_bytes * 1.5 / (1024**3)) + 1)
    map_size_bytes = map_size_gb * 1024 * 1024 * 1024

    print(f"  Database size estimate:")
    print(f"    Nodes: {num_nodes:,}")
    print(f"    Avg degree: {avg_degree:.1f}")
    print(f"    LMDB map_size: {map_size_gb} GB")

    # Track statistics
    degrees = []

    env = lmdb.open(output_path, map_size=map_size_bytes)

    if chunked:
        # === CHUNKED MODE ===
        # Process nodes in chunks, scanning edge files for each chunk.
        # Memory: O(nodes_per_chunk * avg_degree) instead of O(total_edges)
        # Time: O(num_chunks * edge_file_size) - multiple passes over edge files

        num_chunks = (num_nodes + nodes_per_chunk - 1) // nodes_per_chunk
        print(f"\n  Processing {num_nodes:,} nodes in {num_chunks} chunks of {nodes_per_chunk:,} nodes each")
        print(f"  (This requires {num_chunks} passes over edge files)\n")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * nodes_per_chunk
            end_idx = min(start_idx + nodes_per_chunk, num_nodes)
            chunk_sequences = sequences[start_idx:end_idx]
            chunk_set = set(chunk_sequences)

            # Build adjacency for this chunk only
            adjacency = {seq: [] for seq in chunk_sequences}

            # Scan all edge files
            for fpath in edge_files:
                with open(fpath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        seq1, seq2 = line.split('\t')
                        # Add edge if either endpoint is in this chunk
                        if seq1 in chunk_set:
                            adjacency[seq1].append(seq2)
                        if seq2 in chunk_set:
                            adjacency[seq2].append(seq1)

            # Write chunk to LMDB
            with env.begin(write=True) as txn:
                for node in chunk_sequences:
                    neighbors = adjacency[node]
                    degrees.append(len(neighbors))
                    key = node.encode('utf-8')
                    value = json.dumps(neighbors).encode('utf-8')
                    txn.put(key, value)

            print(f"  Chunk {chunk_idx + 1}/{num_chunks} done ({end_idx:,}/{num_nodes:,} nodes)")

    else:
        # === STANDARD MODE ===
        # Build full adjacency dict in memory (faster, but uses more RAM)

        print(f"  Building adjacency list from edge files...")
        adjacency = {seq: [] for seq in sequences}

        for fpath in tqdm(edge_files, desc="  Reading edges", unit="files"):
            with open(fpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seq1, seq2 = line.split('\t')
                        adjacency[seq1].append(seq2)
                        adjacency[seq2].append(seq1)

        # Write to LMDB
        print(f"  Writing to LMDB...")
        with env.begin(write=True) as txn:
            for node in tqdm(sequences, desc="  Writing", unit="nodes", mininterval=0.5):
                neighbors = adjacency[node]
                degrees.append(len(neighbors))
                key = node.encode('utf-8')
                value = json.dumps(neighbors).encode('utf-8')
                txn.put(key, value)

    env.close()

    # Compute final statistics
    min_deg = min(degrees) if degrees else 0
    max_deg = max(degrees) if degrees else 0
    avg_deg = sum(degrees) / len(degrees) if degrees else 0
    sorted_degrees = sorted(degrees)
    median_deg = sorted_degrees[len(sorted_degrees) // 2] if sorted_degrees else 0

    print(f"\n  Graph statistics:")
    print(f"    Total edges: {total_edges:,}")
    print(f"    Degree distribution: min={min_deg}, max={max_deg}, avg={avg_deg:.1f}, median={median_deg}")
    print(f"    Graph density: {2 * total_edges / (num_nodes * (num_nodes - 1)) * 100:.4f}%")

    # Clean up temp files
    print(f"  Cleaning up temp directory...")
    shutil.rmtree(temp_dir)

    print(f"  Graph saved successfully!")


def construct_and_save_graph(n, s, q, output_dir, max_workers=None, stream_to_disk=None):
    """
    Construct an IDS graph and save it to LMDB.

    Args:
        n: Length of strings
        s: Number of errors to correct
        q: Alphabet size (2 for binary, 4 for DNA)
        output_dir: Directory to save the graph
        max_workers: Number of parallel workers (default: cpu_count())
        stream_to_disk: If True, use streaming mode. If None, auto-enable for n >= 19.
    """
    # Create output path
    graph_name = f"graph_ids_s{s}_n{n}_q{q}.lmdb"
    output_path = os.path.join(output_dir, graph_name)

    # Check if LMDB already exists and is complete
    expected_entries = q ** n
    if os.path.exists(output_path):
        try:
            env = lmdb.open(output_path, readonly=True)
            with env.begin() as txn:
                actual = txn.stat()['entries']
            env.close()
            if actual == expected_entries:
                print(f"Graph n={n}, s={s}, q={q} already complete ({actual} entries), skipping.")
                print()
                return
            else:
                print(f"Graph n={n}, s={s}, q={q} has {actual}/{expected_entries} entries, rebuilding...")
                shutil.rmtree(output_path)
        except Exception as e:
            print(f"  Warning: Could not read existing LMDB: {e}, rebuilding...")
            shutil.rmtree(output_path, ignore_errors=True)

    # Generate graph
    result = generate_ids_graph(n, s, q, max_workers=max_workers, output_dir=output_dir,
                                stream_to_disk=stream_to_disk)

    # Handle streaming vs in-memory mode
    if isinstance(result, tuple):
        # Streaming mode: result is (temp_dir, edge_count, sequences)
        temp_dir, edge_count, sequences = result
        save_graph_to_lmdb_streaming(temp_dir, sequences, output_path)
    else:
        # In-memory mode: result is adjacency dict
        save_graph_to_lmdb(result, output_path)

    print()


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Construct IDS (Insertion/Deletion/Substitution) code graphs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build graphs for n=8,9,10 with s=1, quaternary alphabet (DNA)
    python construct_ids_graphs.py --n 8,9,10 --s 1 --q 4

    # Build graph for n=12 with s=2, binary, 50 workers
    python construct_ids_graphs.py --n 12 --s 2 --q 2 --workers 50

    # Custom output directory
    python construct_ids_graphs.py --n 10,11 --s 1 --q 4 --output /mnt/Graphs/ids

    # Force streaming mode for memory-limited systems
    python construct_ids_graphs.py --n 18 --s 1 --q 4 --stream

Memory efficiency:
    For n >= 19, streaming mode is auto-enabled. This writes edges to temp files
    instead of holding them all in memory, reducing peak memory by ~50-70%.
    Speed impact: ~10-20% slower on SSD, more on HDD.
        """
    )

    parser.add_argument('--n', type=str, required=True,
                        help='Comma-separated n values (string lengths), e.g., "8,9,10"')
    parser.add_argument('--s', type=int, required=True,
                        help='Number of errors to correct (requires min distance 2s+1)')
    parser.add_argument('--q', type=int, default=4,
                        help='Alphabet size (default: 4 for DNA/quaternary, use 2 for binary)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: all CPU cores)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: /mnt/Graphs/ids/{quaternary|binary}/s{s})')

    # Streaming mode options
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument('--stream', action='store_true',
                              help='Force streaming mode (writes edges to disk instead of memory)')
    stream_group.add_argument('--no-stream', action='store_true',
                              help='Force in-memory mode even for large n (requires lots of RAM)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse n values
    n_values = [int(x.strip()) for x in args.n.split(',')]

    # Set default output directory based on parameters
    if args.output:
        output_dir = args.output
    else:
        alphabet_name = "binary" if args.q == 2 else "quaternary" if args.q == 4 else f"q{args.q}"
        output_dir = f"/mnt/Graphs/ids/{alphabet_name}/s{args.s}"

    os.makedirs(output_dir, exist_ok=True)

    # Set workers
    max_workers = args.workers if args.workers else cpu_count()

    # Determine streaming mode
    if args.stream:
        stream_to_disk = True
    elif args.no_stream:
        stream_to_disk = False
    else:
        stream_to_disk = None  # Auto-detect based on n

    print("=" * 70)
    print("Constructing IDS (Insertion/Deletion/Substitution) Code Graphs")
    print("=" * 70)
    print()
    print(f"Parameters:")
    print(f"  n values: {n_values}")
    print(f"  s (errors): {args.s} (requires min distance {2*args.s + 1})")
    print(f"  q (alphabet): {args.q}")
    print(f"  workers: {max_workers}")
    print(f"  output: {output_dir}")
    if stream_to_disk is True:
        print(f"  mode: STREAMING (forced)")
    elif stream_to_disk is False:
        print(f"  mode: IN-MEMORY (forced)")
    else:
        print(f"  mode: AUTO (streaming for n >= 19)")
    print()

    # Build graphs
    params = [(n, args.s) for n in n_values]

    for n, s in tqdm(params, desc="Overall progress", unit="graph"):
        construct_and_save_graph(n, s, args.q, output_dir, max_workers=max_workers,
                                stream_to_disk=stream_to_disk)

    print("=" * 70)
    print("All graphs constructed successfully!")
    print(f"Graphs saved to: {output_dir}")
    print("=" * 70)
    print()
    print("SLURM Memory Reporting:")
    print("  To get SLURM's memory report after job completion, use:")
    print("    sacct -j <job_id> --format=JobID,MaxRSS,ReqMem,Elapsed")
    print("  MaxRSS shows the actual peak memory used by the job")
    print("=" * 70)
