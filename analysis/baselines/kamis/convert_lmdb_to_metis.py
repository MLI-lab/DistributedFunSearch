#!/usr/bin/env python3
"""
Convert LMDB conflict graphs to METIS format for KaMIS.

Three conversion modes:
- In-memory (default for n < 15): Fast, loads full graph into memory
- Streaming (for n >= 15): Memory-efficient, stores node mappings only
- Ultra-efficient (for n >= 20): Computes mappings on-the-fly, O(1) memory for mappings

LMDB format: JSON adjacency lists with string node IDs
METIS format: Text file with integer node IDs (1-indexed), sorted neighborhoods

Output files (saved next to source LMDB by default, or in --output-dir if specified):
- graph_<type>_s<s>_n<n>_q<q>.metis   - METIS graph file
- graph_<type>_s<s>_n<n>_q<q>.mapping - Node ID mapping (JSON)

Example: If LMDB is at /mnt/Graphs/ids/quaternary/s1/graph_ids_s1_n6_q4.lmdb
         METIS saved to /mnt/Graphs/ids/quaternary/s1/graph_ids_s1_n6_q4.metis

Usage Examples:
    # Binary deletion codes (default: q=2, type=deletion)
    python convert_lmdb_to_metis.py --n-values 6,7,8,9,10

    # Quaternary deletion codes
    python convert_lmdb_to_metis.py --n-values 6,7,8,9,10 --q 4

    # IDS (Insertion/Deletion/Substitution) codes
    python convert_lmdb_to_metis.py --n-values 6,7,8,9,10 --graph-type ids

    # Quaternary IDS codes
    python convert_lmdb_to_metis.py --n-values 6,7,8,9,10,11 --q 4 --graph-type ids

    # Custom graph directory
    python convert_lmdb_to_metis.py --n-values 6,7,8 --graph-dir /path/to/graphs

    # Custom output directory
    python convert_lmdb_to_metis.py --n-values 6,7,8 --output-dir /path/to/output

    # Force memory-efficient mode (for large graphs)
    python convert_lmdb_to_metis.py --n-values 15,16,17 --memory-efficient

    # Force ultra-efficient mode (minimal memory, for very large graphs)
    python convert_lmdb_to_metis.py --n-values 20,21,22 --ultra-efficient

    # Force in-memory mode even for large graphs
    python convert_lmdb_to_metis.py --n-values 15,16 --no-memory-efficient

    # Overwrite existing METIS files
    python convert_lmdb_to_metis.py --n-values 6,7,8 --force

Arguments:
    --n-values        Comma-separated n values (required)
    --s               Number of errors to correct (default: 1)
    --q               Alphabet size: 2=binary, 4=quaternary (default: 2)
    --graph-type      Graph type: 'deletions' or 'ids' (default: deletions)
    --graph-dir       Directory with LMDB graphs (default: /mnt/Graphs)
    --output-dir      Output directory for METIS files (default: same as graph-dir)
    --memory-efficient  Force streaming mode (auto for n >= 15)
    --ultra-efficient   Force ultra mode (auto for n >= 20)
    --no-memory-efficient  Disable memory-efficient modes
    --force           Overwrite existing METIS files
"""

import argparse
import os
import sys
import json
import lmdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.graph_paths import DEFAULT_GRAPH_DIR, get_lmdb_path, get_metis_path

# Thresholds for automatic mode selection
MEMORY_EFFICIENT_THRESHOLD = 15
ULTRA_EFFICIENT_THRESHOLD = 20


def node_to_int_binary(node: str) -> int:
    """Convert binary string node to 1-indexed integer (sorted order)."""
    return int(node, 2) + 1


def int_to_node_binary(i: int, n: int) -> str:
    """Convert 1-indexed integer to binary string node of length n."""
    return format(i - 1, f'0{n}b')


def node_to_int_qary(node: str, q: int) -> int:
    """Convert q-ary string node to 1-indexed integer (sorted order)."""
    result = 0
    for char in node:
        result = result * q + int(char)
    return result + 1


def int_to_node_qary(i: int, n: int, q: int) -> str:
    """Convert 1-indexed integer to q-ary string node of length n."""
    val = i - 1
    chars = []
    for _ in range(n):
        chars.append(str(val % q))
        val //= q
    return ''.join(reversed(chars))


def convert_lmdb_to_metis_inmemory(graph_db_path: str, output_file: str, mapping_file: str):
    """
    Convert LMDB graph to METIS format by loading full graph into memory.

    Fast but uses O(|V| + |E|) memory. Suitable for smaller graphs (n < 15).

    Args:
        graph_db_path: Path to LMDB graph directory
        output_file: Path to output .metis file
        mapping_file: Path to output .mapping file (JSON)
    """
    print(f"  [In-memory mode] Reading adjacency lists from LMDB...")

    # First pass: Read all adjacency lists and collect nodes
    adjacency = {}
    env = lmdb.open(graph_db_path, readonly=True, lock=False)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            node = key.decode()
            neighbors = json.loads(value.decode())
            adjacency[node] = neighbors

    env.close()

    # Create sorted node list for consistent mapping
    sorted_nodes = sorted(adjacency.keys())
    num_nodes = len(sorted_nodes)

    # Create bidirectional mapping (1-indexed for METIS)
    str_to_int = {node: idx + 1 for idx, node in enumerate(sorted_nodes)}
    int_to_str = {idx + 1: node for idx, node in enumerate(sorted_nodes)}

    # Count edges (each edge counted twice in undirected graph)
    num_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2

    print(f"  Nodes: {num_nodes:,}, Edges: {num_edges:,}")
    print(f"  Writing METIS file...")

    # Write METIS file
    with open(output_file, "w") as f:
        # Header: <num_nodes> <num_edges>
        f.write(f"{num_nodes} {num_edges}\n")

        # For each node (in sorted order)
        for int_id in range(1, num_nodes + 1):
            str_id = int_to_str[int_id]

            # Convert neighbor string IDs to integer IDs and sort
            neighbor_int_ids = sorted([str_to_int[nbr] for nbr in adjacency[str_id]])

            # Write space-separated neighbor IDs
            if neighbor_int_ids:
                f.write(" ".join(map(str, neighbor_int_ids)))
            f.write("\n")

    # Write mapping file (for converting results back)
    print(f"  Writing mapping file...")
    with open(mapping_file, "w") as f:
        json.dump({
            "str_to_int": str_to_int,
            "int_to_str": {str(k): v for k, v in int_to_str.items()},  # JSON keys must be strings
            "num_nodes": num_nodes,
            "num_edges": num_edges
        }, f, indent=2)

    print(f"  METIS file: {output_file}")
    print(f"  Mapping file: {mapping_file}")


def convert_lmdb_to_metis_streaming(graph_db_path: str, output_file: str, mapping_file: str):
    """
    Convert LMDB graph to METIS format using streaming (memory-efficient).

    Uses O(|V|) memory for node mapping only, reads neighbors on-demand.
    Suitable for large graphs (n >= 15).

    Args:
        graph_db_path: Path to LMDB graph directory
        output_file: Path to output .metis file
        mapping_file: Path to output .mapping file (JSON)
    """
    print(f"  [Streaming mode] First pass: collecting node names...")

    env = lmdb.open(graph_db_path, readonly=True, lock=False,
                    readahead=False, meminit=False)

    # First pass: collect all node names (keys only, not values)
    nodes = []
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            nodes.append(bytes(key).decode())

    # Sort for consistent mapping
    sorted_nodes = sorted(nodes)
    num_nodes = len(sorted_nodes)

    # Create mapping (1-indexed for METIS)
    str_to_int = {node: idx + 1 for idx, node in enumerate(sorted_nodes)}
    int_to_str = {idx + 1: node for idx, node in enumerate(sorted_nodes)}

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Second pass: counting edges...")

    # Second pass: count edges
    num_edges = 0
    with env.begin(buffers=True) as txn:
        for node in sorted_nodes:
            value = txn.get(node.encode())
            if value:
                neighbors = json.loads(bytes(value).decode())
                num_edges += len(neighbors)
    num_edges //= 2  # Each edge counted twice

    print(f"  Edges: {num_edges:,}")
    print(f"  Third pass: writing METIS file...")

    # Third pass: write METIS file
    with open(output_file, "w") as f:
        # Header: <num_nodes> <num_edges>
        f.write(f"{num_nodes} {num_edges}\n")

        # For each node (in sorted order), read neighbors on-demand
        with env.begin(buffers=True) as txn:
            for int_id in range(1, num_nodes + 1):
                str_id = int_to_str[int_id]

                # Read neighbors from LMDB
                value = txn.get(str_id.encode())
                if value:
                    neighbors = json.loads(bytes(value).decode())
                    # Convert to integer IDs and sort
                    neighbor_int_ids = sorted([str_to_int[nbr] for nbr in neighbors])
                    if neighbor_int_ids:
                        f.write(" ".join(map(str, neighbor_int_ids)))
                f.write("\n")

                # Progress indicator for large graphs
                if int_id % 100000 == 0:
                    print(f"    Progress: {int_id:,}/{num_nodes:,} nodes written")

    env.close()

    # Write mapping file
    print(f"  Writing mapping file...")
    with open(mapping_file, "w") as f:
        json.dump({
            "str_to_int": str_to_int,
            "int_to_str": {str(k): v for k, v in int_to_str.items()},
            "num_nodes": num_nodes,
            "num_edges": num_edges
        }, f, indent=2)

    print(f"  METIS file: {output_file}")
    print(f"  Mapping file: {mapping_file}")


def convert_lmdb_to_metis_ultra(graph_db_path: str, output_file: str, mapping_file: str,
                                 n: int, q: int = 2):
    """
    Convert LMDB graph to METIS format with minimal memory usage.

    Computes node-to-integer mappings on-the-fly instead of storing them.
    Memory usage: O(max_degree) instead of O(|V|).

    For binary alphabet (q=2), sorted node order = binary integer order.
    For q-ary alphabet, sorted node order = base-q integer order.

    Args:
        graph_db_path: Path to LMDB graph directory
        output_file: Path to output .metis file
        mapping_file: Path to output .mapping file (JSON, minimal info only)
        n: Code length (number of characters per node)
        q: Alphabet size (default: 2 for binary)
    """
    print(f"  [Ultra-efficient mode] n={n}, q={q}")
    print(f"  Computing mappings on-the-fly (O(1) memory for mappings)")

    num_nodes = q ** n
    print(f"  Nodes: {num_nodes:,}")

    env = lmdb.open(graph_db_path, readonly=True, lock=False,
                    readahead=False, meminit=False)

    # Choose mapping functions based on alphabet size
    if q == 2:
        str_to_int = node_to_int_binary
        int_to_str = lambda i: int_to_node_binary(i, n)
    else:
        str_to_int = lambda s: node_to_int_qary(s, q)
        int_to_str = lambda i: int_to_node_qary(i, n, q)

    print(f"  First pass: counting edges...")

    # First pass: count edges
    num_edges = 0
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            neighbors = json.loads(bytes(value).decode())
            num_edges += len(neighbors)
            if num_edges % 10000000 == 0:
                print(f"    Counted {num_edges:,} edge endpoints...")
    num_edges //= 2  # Each edge counted twice

    print(f"  Edges: {num_edges:,}")
    print(f"  Second pass: writing METIS file...")

    # Second pass: write METIS file
    with open(output_file, "w") as f:
        # Header: <num_nodes> <num_edges>
        f.write(f"{num_nodes} {num_edges}\n")

        # For each node (in integer order = sorted string order)
        with env.begin(buffers=True) as txn:
            for int_id in range(1, num_nodes + 1):
                str_id = int_to_str(int_id)

                # Read neighbors from LMDB
                value = txn.get(str_id.encode())
                if value:
                    neighbors = json.loads(bytes(value).decode())
                    # Convert to integer IDs and sort
                    neighbor_int_ids = sorted([str_to_int(nbr) for nbr in neighbors])
                    if neighbor_int_ids:
                        f.write(" ".join(map(str, neighbor_int_ids)))
                f.write("\n")

                # Progress indicator
                if int_id % 1000000 == 0:
                    pct = 100 * int_id / num_nodes
                    print(f"    Progress: {int_id:,}/{num_nodes:,} nodes ({pct:.1f}%)")

    env.close()

    # Write minimal mapping file (just metadata, no full mappings)
    print(f"  Writing mapping file (metadata only)...")
    with open(mapping_file, "w") as f:
        json.dump({
            "mode": "ultra-efficient",
            "n": n,
            "q": q,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "note": "Mappings computed on-the-fly: str_to_int = base-q integer + 1"
        }, f, indent=2)

    print(f"  METIS file: {output_file}")
    print(f"  Mapping file: {mapping_file}")


def convert_lmdb_to_metis(graph_db_path: str, output_file: str, mapping_file: str,
                          memory_efficient: bool = False, ultra_efficient: bool = False,
                          n: int = None, q: int = 2):
    """
    Convert LMDB graph to METIS format.

    Args:
        graph_db_path: Path to LMDB graph directory
        output_file: Path to output .metis file
        mapping_file: Path to output .mapping file (JSON)
        memory_efficient: If True, use streaming mode (O(|V|) memory)
        ultra_efficient: If True, use ultra mode (O(max_degree) memory)
        n: Code length (required for ultra_efficient mode)
        q: Alphabet size (default: 2)
    """
    if ultra_efficient:
        if n is None:
            raise ValueError("n parameter required for ultra_efficient mode")
        convert_lmdb_to_metis_ultra(graph_db_path, output_file, mapping_file, n, q)
    elif memory_efficient:
        convert_lmdb_to_metis_streaming(graph_db_path, output_file, mapping_file)
    else:
        convert_lmdb_to_metis_inmemory(graph_db_path, output_file, mapping_file)


def get_metis_path_local(graph_dir: str, graph_type: str, s: int, n: int, q: int) -> str:
    """Get METIS file path using centralized graph_paths module."""
    return get_metis_path(graph_type, s, n, q, base_dir=graph_dir)


def get_lmdb_path_local(graph_dir: str, graph_type: str, s: int, n: int, q: int) -> str:
    """Get LMDB file path using centralized graph_paths module."""
    return get_lmdb_path(graph_type, s, n, q, base_dir=graph_dir)


def convert_graph(
    graph_db_path: str,
    output_dir: str,
    n: Optional[int] = None,
    q: int = 2,
    memory_efficient: Optional[bool] = None,
    ultra_efficient: Optional[bool] = None,
    force: bool = False
) -> bool:
    """
    Convert a single LMDB graph to METIS format.

    Args:
        graph_db_path: Path to LMDB graph directory
        output_dir: Directory to save METIS files
        n: Code length (used for auto mode selection and ultra-efficient mode)
        q: Alphabet size (default: 2)
        memory_efficient: Override for memory-efficient mode (None = auto-detect)
        ultra_efficient: Override for ultra-efficient mode (None = auto-detect)
        force: If True, overwrite existing METIS file

    Returns:
        True if conversion was performed, False if skipped
    """
    # Extract graph name from path
    graph_name = os.path.basename(graph_db_path).replace(".lmdb", "")

    # Output files
    metis_file = os.path.join(output_dir, f"{graph_name}.metis")
    mapping_file = os.path.join(output_dir, f"{graph_name}.mapping")

    # Check if METIS file already exists
    if os.path.exists(metis_file) and not force:
        print(f"\n[SKIP] {graph_name}: METIS file already exists at {metis_file}")
        print(f"       Use --force to overwrite")
        return False

    print(f"\n[INFO] Converting: {graph_name}")
    print(f"[INFO] Loading from: {graph_db_path}")

    # Determine conversion mode (ultra > memory-efficient > in-memory)
    use_ultra = False
    use_streaming = False

    if ultra_efficient is True:
        use_ultra = True
        print(f"[INFO] Using ultra-efficient mode (forced)")
    elif memory_efficient is True:
        use_streaming = True
        print(f"[INFO] Using memory-efficient streaming mode (forced)")
    elif ultra_efficient is None and memory_efficient is None:
        # Auto-detect based on n
        if n is not None and n >= ULTRA_EFFICIENT_THRESHOLD:
            use_ultra = True
            print(f"[INFO] Auto-enabled ultra-efficient mode (n={n} >= {ULTRA_EFFICIENT_THRESHOLD})")
        elif n is not None and n >= MEMORY_EFFICIENT_THRESHOLD:
            use_streaming = True
            print(f"[INFO] Auto-enabled memory-efficient mode (n={n} >= {MEMORY_EFFICIENT_THRESHOLD})")
    elif memory_efficient is False and ultra_efficient is None:
        # Explicit in-memory mode
        pass

    # Convert
    convert_lmdb_to_metis(
        graph_db_path, metis_file, mapping_file,
        memory_efficient=use_streaming,
        ultra_efficient=use_ultra,
        n=n,
        q=q
    )

    print(f"[INFO] Conversion complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert LMDB graphs to METIS format for KaMIS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--n-values', type=str, required=True,
                        help='Comma-separated n values (e.g., "6,7,8,9,10")')
    parser.add_argument('--s', type=int, default=1,
                        help='Number of deletions to correct (default: 1)')
    parser.add_argument('--q', type=int, default=2,
                        help='Alphabet size (default: 2 for binary)')
    parser.add_argument('--graph-type', choices=['deletions', 'ids'], default='deletions',
                        help='Graph type (default: deletions)')
    parser.add_argument('--graph-dir', default=DEFAULT_GRAPH_DIR,
                        help=f'Directory with LMDB graphs (default: {DEFAULT_GRAPH_DIR})')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as graph-dir)')
    parser.add_argument('--memory-efficient', action='store_true',
                        help='Force memory-efficient streaming mode (auto for n >= 15)')
    parser.add_argument('--ultra-efficient', action='store_true',
                        help='Force ultra-efficient mode with on-the-fly mappings (auto for n >= 20)')
    parser.add_argument('--no-memory-efficient', action='store_true',
                        help='Disable memory-efficient modes even for large graphs')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing METIS files')

    args = parser.parse_args()

    # Parse n values
    n_values = [int(x.strip()) for x in args.n_values.split(',')]

    # Output directory: if specified, use it; otherwise save next to source LMDB
    custom_output_dir = args.output_dir  # None if not specified
    if custom_output_dir:
        os.makedirs(custom_output_dir, exist_ok=True)

    # Determine mode overrides
    if args.no_memory_efficient:
        memory_efficient = False
        ultra_efficient = False
    elif args.ultra_efficient:
        memory_efficient = None
        ultra_efficient = True
    elif args.memory_efficient:
        memory_efficient = True
        ultra_efficient = False
    else:
        memory_efficient = None  # Auto-detect per graph
        ultra_efficient = None

    print("=" * 60)
    print("  LMDB to METIS Converter")
    print("=" * 60)
    print(f"n values: {n_values}")
    print(f"s={args.s}, q={args.q}, type={args.graph_type}")
    print(f"Graph directory: {args.graph_dir}")
    if custom_output_dir:
        print(f"Output directory: {custom_output_dir}")
    else:
        print(f"Output: Same directory as source LMDB files")
    if args.no_memory_efficient:
        print(f"Mode: in-memory (forced)")
    elif args.ultra_efficient:
        print(f"Mode: ultra-efficient (forced)")
    elif args.memory_efficient:
        print(f"Mode: memory-efficient streaming (forced)")
    else:
        print(f"Mode: auto (n<15: in-memory, n>=15: streaming, n>=20: ultra)")
    print("=" * 60)

    # Track results
    converted = []
    skipped = []
    not_found = []

    # Convert all specified graphs
    output_dirs_used = set()
    for n in n_values:
        lmdb_path = get_lmdb_path_local(args.graph_dir, args.graph_type, args.s, n, args.q)

        if not os.path.isdir(lmdb_path):
            print(f"\n[WARNING] Graph not found: {lmdb_path}, skipping n={n}")
            not_found.append(n)
            continue

        # Determine output directory: custom or same as LMDB source
        if custom_output_dir:
            output_dir = custom_output_dir
        else:
            # Save next to the source LMDB file
            output_dir = os.path.dirname(lmdb_path)

        output_dirs_used.add(output_dir)

        result = convert_graph(
            lmdb_path,
            output_dir,
            n=n,
            q=args.q,
            memory_efficient=memory_efficient,
            ultra_efficient=ultra_efficient,
            force=args.force
        )

        if result:
            converted.append(n)
        else:
            skipped.append(n)

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print("=" * 60)
    if converted:
        print(f"Converted: n={converted}")
    if skipped:
        print(f"Skipped (already exist): n={skipped}")
    if not_found:
        print(f"Not found: n={not_found}")
    if len(output_dirs_used) == 1:
        print(f"METIS files saved to: {list(output_dirs_used)[0]}")
    else:
        print(f"METIS files saved to source directories")
    print("=" * 60)


if __name__ == "__main__":
    main()
