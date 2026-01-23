#!/usr/bin/env python3
"""
Step 1 & 2: Extract and Deduplicate Successful Functions

This script:
1. Extracts all priority functions from checkpoint(s) that achieve a target signature
2. Deduplicates them based on the priority values they assign (semantic deduplication)

Supports:
- Single checkpoint file
- Multiple checkpoint files
- Folders containing checkpoints (scans recursively for .pkl files)
- Mix of files and folders

Usage:
    python test/extract.py <path1> [path2 ...] [options]

Examples:
    python test/extract.py /path/to/checkpoint.pkl
    python test/extract.py /exp1/checkpoints/ /exp2/checkpoints/ --output ./results/
"""

import argparse
import pickle
import hashlib
import itertools
import sys
import json
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    SIGNATURE_NO_GRAPH,
    SIGNATURE_GRAPH_GT,
    SIGNATURE_GRAPH_NETWORKX,
    COMMON_IMPORTS,
    detect_signature,
    are_neighbors,
)


# Default target signature (VT-optimal for s=1, q=2)
DEFAULT_TARGET = {
    (6, 1, 2): 10,
    (7, 1, 2): 16,
    (8, 1, 2): 30,
    (9, 1, 2): 52,
    (10, 1, 2): 94,
    (11, 1, 2): 172,
}


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint from pickle file."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def find_checkpoint_files(paths: List[str], pattern: str = "*.pkl") -> List[Path]:
    """
    Find all checkpoint files from a list of paths.

    Args:
        paths: List of file paths or directory paths
        pattern: Glob pattern for checkpoint files (default: *.pkl)

    Returns:
        List of Path objects to checkpoint files
    """
    checkpoint_files = []

    for path_str in paths:
        path = Path(path_str)

        if path.is_file():
            # Direct file path
            if path.suffix == '.pkl':
                checkpoint_files.append(path)
            else:
                print(f"Warning: Skipping non-.pkl file: {path}", file=sys.stderr)

        elif path.is_dir():
            # Directory - scan for .pkl files
            found = list(path.glob(f"**/{pattern}"))
            if found:
                checkpoint_files.extend(found)
                print(f"Found {len(found)} checkpoint(s) in {path}", file=sys.stderr)
            else:
                print(f"Warning: No .pkl files found in {path}", file=sys.stderr)

        else:
            print(f"Warning: Path does not exist: {path}", file=sys.stderr)

    # Remove duplicates and sort by modification time (newest first)
    unique_files = list(set(checkpoint_files))
    unique_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return unique_files


def parse_signature_string(sig_str: str) -> Dict[tuple, int]:
    """Parse a signature string into a dictionary."""
    sig_str = sig_str.strip()

    # Try direct eval if it looks like a dict
    if sig_str.startswith('{'):
        try:
            return eval(sig_str)
        except:
            pass

    # Try parsing colon-separated format
    try:
        result = {}
        pattern = r'\(([^)]+)\)\s*:\s*(\d+)'
        matches = re.findall(pattern, sig_str)
        for tuple_content, value in matches:
            key = eval(f"({tuple_content})")
            result[key] = int(value)
        if result:
            return result
    except:
        pass

    raise ValueError(f"Could not parse signature string: {sig_str}")


def extract_matching_functions(checkpoint: Dict[str, Any],
                                target_signature: Dict[tuple, int],
                                source_path: str = None) -> List[Dict]:
    """Extract all functions that match the target signature exactly."""
    target_tuple = tuple(sorted(target_signature.items()))
    matching_functions = []

    for island_id, island_state in enumerate(checkpoint['islands_state']):
        for signature_str, cluster_data in island_state['clusters'].items():
            scores_per_test = cluster_data.get('scores_per_test', {})

            # Parse scores_per_test keys
            parsed_scores = {}
            for k, v in scores_per_test.items():
                key = eval(k) if isinstance(k, str) else k
                parsed_scores[key] = v

            current_tuple = tuple(sorted(parsed_scores.items()))

            if current_tuple == target_tuple:
                programs = cluster_data.get('programs', [])
                for prog_idx, program_dict in enumerate(programs):
                    matching_functions.append({
                        'source_checkpoint': source_path,
                        'island_id': island_id,
                        'cluster_signature': signature_str,
                        'cluster_score': cluster_data['score'],
                        'scores_per_test': parsed_scores,
                        'program_index': prog_idx,
                        'body': program_dict.get('body', ''),
                        'docstring': program_dict.get('docstring', ''),
                        'thinking': program_dict.get('thinking', ''),
                        'thought': program_dict.get('thought', ''),
                        'hash_value': program_dict.get('hash_value', ''),
                        'name': program_dict.get('name', 'priority'),
                        'args': program_dict.get('args', ''),
                    })

    return matching_functions


def build_graph_for_dedup(nodes: List[str], n: int, s: int, signature: str):
    """Build appropriate graph structure for deduplication based on signature."""
    if signature == SIGNATURE_GRAPH_NETWORKX:
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(nodes)
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    if are_neighbors(node1, node2, n, s):
                        G.add_edge(node1, node2)
            return G, None, None
        except ImportError:
            return None, None, None

    elif signature == SIGNATURE_GRAPH_GT:
        try:
            import graph_tool.all as gt
            node_to_vertex = {node: idx for idx, node in enumerate(nodes)}
            vertex_to_node = {idx: node for idx, node in enumerate(nodes)}
            G = gt.Graph(directed=False)
            G.add_vertex(len(nodes))
            edges = []
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    if are_neighbors(node1, node2, n, s):
                        edges.append((i, j))
            if edges:
                G.add_edge_list(edges)
            return G, node_to_vertex, vertex_to_node
        except ImportError:
            return None, None, None

    return None, None, None


def compute_priority_hash(body: str, args: str = '', n: int = 6, s: int = 1, q: int = 2) -> str:
    """
    Execute the priority function and compute a hash based on the
    priority values it assigns to all nodes.

    This provides semantic deduplication - functions that produce the
    same priority ordering will have the same hash.

    Automatically detects and handles different function signatures.
    """
    # Generate all q-ary strings of length n
    nodes = [''.join(seq) for seq in itertools.product(map(str, range(q)), repeat=n)]

    # Detect signature
    signature = detect_signature(body, args)

    # Create the priority function from body
    clean_body = body.strip()

    # Build the appropriate function definition
    if signature == SIGNATURE_NO_GRAPH:
        func_def = "def priority(node, n, s, q):"
    elif signature == SIGNATURE_GRAPH_GT:
        func_def = "def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):"
    elif signature == SIGNATURE_GRAPH_NETWORKX:
        func_def = "def priority(node, G, n, s):"
    else:
        func_def = "def priority(node, n, s, q):"

    func_code = COMMON_IMPORTS + f"""
{func_def}
{textwrap.indent(clean_body, '    ')}
"""

    try:
        # Execute in isolated namespace
        namespace = {}
        exec(func_code, namespace)
        priority_func = namespace['priority']

        # Build graph if needed
        G, node_to_vertex, vertex_to_node = None, None, None
        if signature in (SIGNATURE_GRAPH_GT, SIGNATURE_GRAPH_NETWORKX):
            G, node_to_vertex, vertex_to_node = build_graph_for_dedup(nodes, n, s, signature)
            if G is None:
                # Fall back to code hash if graph library not available
                return hashlib.sha256(body.encode()).hexdigest()[:16]

        # Compute priorities for all nodes
        priorities = {}
        for node in nodes:
            try:
                if signature == SIGNATURE_NO_GRAPH:
                    p = priority_func(node, n, s, q)
                elif signature == SIGNATURE_GRAPH_GT:
                    p = priority_func(node, G, node_to_vertex, vertex_to_node, n, s)
                elif signature == SIGNATURE_GRAPH_NETWORKX:
                    p = priority_func(node, G, n, s)
                else:
                    p = priority_func(node, n, s, q)

                if p is None:
                    p = 0.0
                priorities[node] = float(p)
            except Exception:
                priorities[node] = 0.0

        # Create hash from sorted priority mapping
        mapping = [(node, priorities[node]) for node in sorted(nodes)]
        mapping_str = ','.join(f'{node}:{score:.10f}' for node, score in mapping)
        return hashlib.sha256(mapping_str.encode()).hexdigest()[:16]

    except Exception as e:
        # If execution fails, fall back to code hash
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def deduplicate_by_priority(functions: List[Dict],
                            dedup_n: int = 6,
                            dedup_s: int = 1,
                            dedup_q: int = 2,
                            verbose: bool = True) -> List[Dict]:
    """
    Deduplicate functions based on the priority values they assign.

    Functions that produce identical priority mappings are considered duplicates.
    Automatically handles different function signatures (no_graph, graph_gt, graph_networkx).
    """
    seen_hashes = {}
    unique_functions = []

    # Group by detected signature for reporting
    signature_counts = {SIGNATURE_NO_GRAPH: 0, SIGNATURE_GRAPH_GT: 0, SIGNATURE_GRAPH_NETWORKX: 0}

    for i, func in enumerate(functions):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processing function {i+1}/{len(functions)}...", file=sys.stderr)

        body = func['body']
        args = func.get('args', '')

        # Detect and store signature
        signature = detect_signature(body, args)
        func['detected_signature'] = signature
        signature_counts[signature] += 1

        # Compute semantic hash (handles different signatures automatically)
        priority_hash = compute_priority_hash(body, args, dedup_n, dedup_s, dedup_q)

        if priority_hash not in seen_hashes:
            seen_hashes[priority_hash] = len(unique_functions)
            func['priority_hash'] = priority_hash
            func['duplicate_count'] = 1
            unique_functions.append(func)
        else:
            # Increment duplicate count for the original
            orig_idx = seen_hashes[priority_hash]
            unique_functions[orig_idx]['duplicate_count'] += 1

    if verbose:
        print(f"  Signature distribution:", file=sys.stderr)
        print(f"    no_graph: {signature_counts[SIGNATURE_NO_GRAPH]}", file=sys.stderr)
        print(f"    graph_gt: {signature_counts[SIGNATURE_GRAPH_GT]}", file=sys.stderr)
        print(f"    graph_networkx: {signature_counts[SIGNATURE_GRAPH_NETWORKX]}", file=sys.stderr)

    return unique_functions


def save_functions(functions: List[Dict], output_dir: str, prefix: str = "successful"):
    """Save extracted functions to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert tuple keys to strings for JSON serialization
    json_safe_functions = []
    for func in functions:
        func_copy = func.copy()
        if 'scores_per_test' in func_copy:
            # Convert {(6,1,2): 10} to {"(6, 1, 2)": 10}
            func_copy['scores_per_test'] = {
                str(k): v for k, v in func_copy['scores_per_test'].items()
            }
        json_safe_functions.append(func_copy)

    # Save all functions as JSON
    json_path = output_path / f"{prefix}_functions.json"
    with open(json_path, 'w') as f:
        json.dump(json_safe_functions, f, indent=2, default=str)
    print(f"Saved {len(functions)} functions to {json_path}")

    # Save just the function bodies as Python file
    py_path = output_path / f"{prefix}_functions.py"
    with open(py_path, 'w') as f:
        f.write('"""Extracted priority functions that achieved target signature."""\n\n')
        for i, func in enumerate(functions):
            f.write(f"# Function {i+1}\n")
            f.write(f"# Priority hash: {func.get('priority_hash', 'N/A')}\n")
            f.write(f"# Duplicate count: {func.get('duplicate_count', 1)}\n")
            if func.get('source_checkpoint'):
                f.write(f"# Source: {Path(func['source_checkpoint']).name}\n")
            f.write(f"# Island: {func['island_id']}, Score: {func['cluster_score']}\n")
            f.write(f"# Scores: {func['scores_per_test']}\n")
            f.write(f"def priority_{i+1}(node, n, s, q):\n")
            # Indent body properly
            body = func['body']
            if not body.startswith('    '):
                body = '\n'.join('    ' + line if line.strip() else line
                                for line in body.split('\n'))
            f.write(body)
            f.write('\n\n')
    print(f"Saved function bodies to {py_path}")

    # Save summary
    summary_path = output_path / f"{prefix}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Extraction Summary\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Total unique functions: {len(functions)}\n")
        f.write(f"Total duplicates removed: {sum(func.get('duplicate_count', 1) - 1 for func in functions)}\n\n")

        # Group by source checkpoint
        by_source = defaultdict(list)
        for func in functions:
            source = func.get('source_checkpoint', 'unknown')
            if source:
                source = Path(source).name
            by_source[source].append(func)

        if len(by_source) > 1:
            f.write("Distribution by source checkpoint:\n")
            for source in sorted(by_source.keys()):
                f.write(f"  {source}: {len(by_source[source])} functions\n")
            f.write("\n")

        # Group by duplicate count
        by_dup_count = defaultdict(list)
        for func in functions:
            by_dup_count[func.get('duplicate_count', 1)].append(func)

        f.write("Distribution by duplicate count:\n")
        for count in sorted(by_dup_count.keys(), reverse=True):
            f.write(f"  {count} duplicates: {len(by_dup_count[count])} functions\n")
    print(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and deduplicate successful functions from checkpoint(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('paths', nargs='+',
                        help='Checkpoint file(s) or folder(s) containing checkpoints')
    parser.add_argument('--target', '-t',
                        help='Target signature (default: VT-optimal for s=1,q=2)')
    parser.add_argument('--output', '-o', default='./successful_functions/',
                        help='Output directory')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Skip deduplication step')
    parser.add_argument('--dedup-n', type=int, default=6,
                        help='n value to use for deduplication (default: 6)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--latest-only', action='store_true',
                        help='Only use the latest checkpoint from each folder')

    args = parser.parse_args()

    # Parse target signature
    if args.target:
        target_signature = parse_signature_string(args.target)
    else:
        target_signature = DEFAULT_TARGET

    print("=" * 70)
    print("  Extract Successful Functions")
    print("=" * 70)
    print()
    print(f"Input paths: {args.paths}")
    print(f"Target signature: {target_signature}")
    print(f"Output directory: {args.output}")
    print()

    # Find all checkpoint files
    print("Scanning for checkpoint files...", file=sys.stderr)
    checkpoint_files = find_checkpoint_files(args.paths)

    if not checkpoint_files:
        print("No checkpoint files found!")
        return 1

    # If --latest-only, keep only the newest checkpoint per parent folder
    if args.latest_only:
        by_parent = defaultdict(list)
        for f in checkpoint_files:
            by_parent[f.parent].append(f)

        latest_files = []
        for parent, files in by_parent.items():
            # Already sorted by mtime, first is newest
            latest_files.append(files[0])
        checkpoint_files = latest_files
        print(f"Using latest checkpoint from each folder: {len(checkpoint_files)} file(s)")

    print(f"\nFound {len(checkpoint_files)} checkpoint file(s) to process")
    print()

    # Extract from all checkpoints
    all_functions = []
    for i, ckpt_path in enumerate(checkpoint_files):
        print(f"[{i+1}/{len(checkpoint_files)}] Processing {ckpt_path.name}...", file=sys.stderr)

        try:
            checkpoint = load_checkpoint(str(ckpt_path))
            functions = extract_matching_functions(checkpoint, target_signature, str(ckpt_path))

            if functions:
                print(f"  Found {len(functions)} matching functions", file=sys.stderr)
                all_functions.extend(functions)
            else:
                if args.verbose:
                    print(f"  No matching functions", file=sys.stderr)

        except Exception as e:
            print(f"  Error loading checkpoint: {e}", file=sys.stderr)
            continue

    print(f"\nTotal functions found across all checkpoints: {len(all_functions)}")

    if not all_functions:
        print("No functions found with the target signature!")
        print("Use checkpoint_inspector.py --list-signatures to see available signatures.")
        return 1

    # Deduplicate
    if not args.no_dedup:
        print(f"\nDeduplicating based on priority values (using n={args.dedup_n})...",
              file=sys.stderr)
        unique_functions = deduplicate_by_priority(
            all_functions,
            dedup_n=args.dedup_n,
            verbose=args.verbose
        )
        print(f"After deduplication: {len(unique_functions)} unique functions")
        print(f"Removed {len(all_functions) - len(unique_functions)} duplicates")
    else:
        unique_functions = all_functions
        for func in unique_functions:
            func['priority_hash'] = 'not_computed'
            func['duplicate_count'] = 1

    # Save results
    print(f"\nSaving results to {args.output}...")
    save_functions(unique_functions, args.output)

    print()
    print("=" * 70)
    print("  Extraction Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
