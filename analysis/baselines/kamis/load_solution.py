#!/usr/bin/env python3
"""
Load independent set solutions and report their size.

Can load solutions from:
- KaMIS result files (.result format)
- Plain text files (one codeword per line)

Usage:
    python load_solution.py path/to/solution.result --mapping path/to/mapping.json
    python load_solution.py path/to/solution.txt --type text
"""

import argparse
import os
import json
from typing import List, Optional


def load_node_mapping(mapping_file: str) -> dict:
    """Load node ID mapping from JSON file."""
    with open(mapping_file, "r") as f:
        return json.load(f)


def get_kamis_solution_size(result_file: str) -> int:
    """
    Get the size of a KaMIS solution (fast, no mapping needed).

    Args:
        result_file: Path to .result file (one 0/1 per line)

    Returns:
        Number of nodes in the independent set
    """
    count = 0
    with open(result_file, "r") as f:
        for line in f:
            if line.strip() == "1":
                count += 1
    return count


def int_to_node_binary(i: int, n: int) -> str:
    """Convert 1-indexed integer to binary string node of length n."""
    return format(i - 1, f'0{n}b')


def int_to_node_qary(i: int, n: int, q: int) -> str:
    """Convert 1-indexed integer to q-ary string node of length n."""
    val = i - 1
    chars = []
    for _ in range(n):
        chars.append(str(val % q))
        val //= q
    return ''.join(reversed(chars))


def load_kamis_result(result_file: str, mapping_file: Optional[str] = None) -> List[str]:
    """
    Load KaMIS result file and return independent set.

    Args:
        result_file: Path to .result file (one 0/1 per line)
        mapping_file: Optional path to .mapping file (to convert integer IDs to strings)

    Returns:
        List of node IDs in the independent set
    """
    with open(result_file, "r") as f:
        lines = f.readlines()

    # Parse: 1 means node is in set, 0 means not
    independent_set_int = []
    for node_id, line in enumerate(lines, start=1):
        if line.strip() == "1":
            independent_set_int.append(node_id)

    # Convert back to original string IDs if mapping provided
    if mapping_file and os.path.exists(mapping_file):
        mapping = load_node_mapping(mapping_file)

        # Check if this is an ultra-efficient mapping (has n and q, no int_to_str)
        if mapping.get("mode") == "ultra-efficient":
            n = mapping["n"]
            q = mapping.get("q", 2)
            if q == 2:
                independent_set = [int_to_node_binary(node_id, n) for node_id in independent_set_int]
            else:
                independent_set = [int_to_node_qary(node_id, n, q) for node_id in independent_set_int]
            return independent_set
        elif "int_to_str" in mapping:
            # Standard mapping with full int_to_str dict
            int_to_str = mapping["int_to_str"]
            independent_set = [int_to_str[str(node_id)] for node_id in independent_set_int]
            return independent_set

    # No mapping - return integer IDs as strings
    return [str(node_id) for node_id in independent_set_int]


def load_text_solution(solution_file: str) -> List[str]:
    """
    Load solution from plain text file (one codeword per line).

    Args:
        solution_file: Path to text file with codewords

    Returns:
        List of codewords
    """
    with open(solution_file, "r") as f:
        codewords = [line.strip() for line in f if line.strip()]
    return codewords


def load_solution(
    solution_file: str,
    mapping_file: Optional[str] = None,
    solution_type: str = "kamis",
    show_codewords: bool = False
) -> List[str]:
    """
    Load and report independent set solution size.

    Args:
        solution_file: Path to solution file
        mapping_file: Optional path to .mapping file (for KaMIS results)
        solution_type: "kamis" or "text"
        show_codewords: Whether to print sample codewords

    Returns:
        List of codewords in the solution
    """
    print(f"\n{'='*60}")
    print(f"Loading solution: {solution_file}")
    print(f"{'='*60}\n")

    # Check if file exists
    if not os.path.exists(solution_file):
        print(f"[ERROR] File not found: {solution_file}")
        print(f"{'='*60}\n")
        return []

    # Load solution
    if solution_type == "kamis":
        solution = load_kamis_result(solution_file, mapping_file)
    else:
        solution = load_text_solution(solution_file)

    # Report size
    print(f"[RESULT] Solution size: {len(solution)}")

    # Show sample codewords if requested
    if show_codewords and len(solution) > 0:
        print(f"\n[INFO] Sample codewords (first 10):")
        for cw in solution[:10]:
            print(f"  {cw}")
        if len(solution) > 10:
            print(f"  ... ({len(solution) - 10} more)")

    print(f"\n{'='*60}\n")

    return solution


def main():
    parser = argparse.ArgumentParser(
        description='Load and inspect independent set solutions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('solution_file', help='Path to solution file')
    parser.add_argument('--mapping', '-m', default=None,
                        help='Path to mapping file (for KaMIS results)')
    parser.add_argument('--type', '-t', choices=['kamis', 'text'], default='kamis',
                        help='Solution type: kamis (0/1 format) or text (one codeword per line)')
    parser.add_argument('--show-codewords', '-s', action='store_true',
                        help='Show sample codewords')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file to save codewords (one per line)')

    args = parser.parse_args()

    # Auto-detect mapping file if not provided
    if args.mapping is None and args.type == 'kamis':
        potential_mapping = args.solution_file.replace('.result', '.mapping')
        # Try to find mapping in same directory or graph directory
        if os.path.exists(potential_mapping):
            args.mapping = potential_mapping
            print(f"[INFO] Auto-detected mapping file: {args.mapping}")

    # Load solution
    solution = load_solution(
        solution_file=args.solution_file,
        mapping_file=args.mapping,
        solution_type=args.type,
        show_codewords=args.show_codewords
    )

    # Save to output file if requested
    if args.output and solution:
        with open(args.output, 'w') as f:
            for cw in sorted(solution):
                f.write(f"{cw}\n")
        print(f"[INFO] Saved {len(solution)} codewords to: {args.output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
