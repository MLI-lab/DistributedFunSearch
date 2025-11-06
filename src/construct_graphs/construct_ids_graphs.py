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
"""

import itertools
import json
import os
import lmdb
import Levenshtein
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _compute_edges_chunk(args):
    """
    Worker function to compute edges for a chunk of sequence pairs.

    Args:
        args: Tuple of (pairs, sequences, threshold)

    Returns:
        List of edges (seq1, seq2) that should be connected
    """
    pairs, sequences, threshold = args
    edges = []

    for i, j in pairs:
        seq1, seq2 = sequences[i], sequences[j]
        edit_dist = Levenshtein.distance(seq1, seq2)

        if edit_dist < threshold:
            edges.append((seq1, seq2))

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

    print(f"Generating graph for n={n}, s={s}, q={q} (min required distance: {2*s + 1})")
    print(f"  Using {max_workers} workers for parallel computation")

    # Generate q-ary alphabet: '0', '1', ..., 'q-1'
    alphabet = ''.join(str(i) for i in range(q))
    sequences = [''.join(seq) for seq in itertools.product(alphabet, repeat=n)]
    print(f"  Total nodes: {len(sequences)}")

    # Build adjacency list
    adjacency = {seq: [] for seq in sequences}

    threshold = 2 * s + 1

    # Generate all pairs of indices
    total_pairs = len(sequences) * (len(sequences) - 1) // 2
    all_pairs = [(i, j) for i in range(len(sequences)) for j in range(i + 1, len(sequences))]

    # Split pairs into chunks for workers
    chunk_size = max(1, len(all_pairs) // max_workers)
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, len(all_pairs), chunk_size)]

    # Prepare arguments for workers
    worker_args = [(chunk, sequences, threshold) for chunk in chunks]

    # Process in parallel
    print(f"  Computing edit distances in parallel...")
    with Pool(max_workers) as pool:
        results = list(tqdm(
            pool.imap(_compute_edges_chunk, worker_args),
            total=len(chunks),
            desc="  Progress",
            unit="chunk"
        ))

    # Combine results into adjacency list
    edge_count = 0
    for edges in results:
        for seq1, seq2 in edges:
            adjacency[seq1].append(seq2)
            adjacency[seq2].append(seq1)
            edge_count += 1

    print(f"  Total edges: {edge_count}")
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
    max_workers = 16

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
