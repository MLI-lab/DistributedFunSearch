#!/usr/bin/env python3
"""
Generate VT (Varshamov-Tenengolts) Codes

VT codes are single-deletion-correcting codes. VT_a(n) is defined as:
    VT_a(n) = {x in {0,1}^n : sum_{i=1}^{n} i * x_i ≡ a (mod n+1)}

For a=0 (VT_0), the weighted position sum is divisible by n+1.

This script:
1. Generates VT codes for specified n and a values
2. Updates the vt_solutions.json file with new codes

Output format (nested structure):
{
  "6": {
    "0": ["000000", "001011", ...],
    "1": ["000001", "001010", ...],
    ...
  },
  "7": {
    "0": [...],
    ...
  }
}

Usage:
    # Generate only VT_0 (default)
    python generate_vt_codes.py --max-n 20

    # Generate VT_0 and VT_1
    python generate_vt_codes.py --max-n 20 --a-values 0,1

    # Generate all a values (0 to n for each n)
    python generate_vt_codes.py --max-n 20 --all-a

    # Preview without saving
    python generate_vt_codes.py --max-n 25 --all-a --preview
"""

import argparse
import json
import sys
import itertools
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.vt_utils import compute_vt_syndrome, is_flat_vt_format


def generate_vt_code(n: int, a: int = 0) -> List[str]:
    """
    Generate VT_a(n) code.

    Args:
        n: Length of codewords
        a: Syndrome value (default 0 for VT_0)

    Returns:
        List of binary strings in VT_a(n)
    """
    modulus = n + 1
    codewords = []

    # Generate all binary strings of length n
    for bits in itertools.product('01', repeat=n):
        codeword = ''.join(bits)
        syndrome = compute_vt_syndrome(codeword)

        if syndrome % modulus == a:
            codewords.append(codeword)

    return sorted(codewords)


def migrate_flat_to_nested(flat_data: dict) -> dict:
    """
    Migrate old flat format to nested format.
    Assumes flat format contains VT_0 codes only.
    """
    nested = {}
    for n_str, codewords in flat_data.items():
        nested[n_str] = {"0": codewords}
    return nested


def load_existing_vt_codes(path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load existing VT codes from JSON file.
    Handles both flat (legacy) and nested formats.
    """
    if not Path(path).exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    # Check format and migrate if needed
    if is_flat_vt_format(data):
        print("  (Detected legacy flat format, will migrate to nested format)")
        return migrate_flat_to_nested(data)

    return data


def save_vt_codes(codes: Dict[str, Dict[str, List[str]]], path: str):
    """Save VT codes to JSON file in nested format."""
    # Sort by n value, then by a value
    sorted_codes = {}
    for n in sorted(int(k) for k in codes.keys()):
        n_str = str(n)
        sorted_codes[n_str] = {}
        for a in sorted(int(k) for k in codes[n_str].keys()):
            a_str = str(a)
            sorted_codes[n_str][a_str] = codes[n_str][a_str]

    with open(path, 'w') as f:
        json.dump(sorted_codes, f, indent=2)


def parse_a_values(a_str: str) -> List[int]:
    """Parse comma-separated a values like '0,1,2'."""
    return [int(x.strip()) for x in a_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='Generate VT (Varshamov-Tenengolts) codes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--min-n', type=int, default=6,
                        help='Minimum n value (default: 6)')
    parser.add_argument('--max-n', type=int, default=20,
                        help='Maximum n value (default: 20)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file (default: ../src/graphs/vt_solutions.json)')

    # a value options (mutually exclusive)
    a_group = parser.add_mutually_exclusive_group()
    a_group.add_argument('--a-values', type=str, default='0',
                         help='Comma-separated syndrome values (default: "0" for VT_0 only)')
    a_group.add_argument('--all-a', action='store_true',
                         help='Generate all a values (0 to n) for each n')

    parser.add_argument('--preview', action='store_true',
                        help='Preview what would be generated without saving')

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        script_dir = Path(__file__).parent
        output_path = script_dir / "../src/graphs/vt_solutions.json"

    output_path = output_path.resolve()

    # Parse a values
    if args.all_a:
        a_mode = "all"
        a_values_str = "all (0 to n)"
    else:
        a_mode = "specific"
        specified_a_values = parse_a_values(args.a_values)
        a_values_str = ", ".join(str(a) for a in specified_a_values)

    print("=" * 70)
    print("  VT Code Generator")
    print("=" * 70)
    print()
    print(f"Generating VT codes for n = {args.min_n} to {args.max_n}")
    print(f"Syndrome values (a): {a_values_str}")
    print(f"Output: {output_path}")
    print()

    # Load existing codes
    if output_path.exists():
        existing_codes = load_existing_vt_codes(str(output_path))
        existing_n = sorted(int(k) for k in existing_codes.keys())
        print(f"Loaded existing codes for n: {existing_n}")
        # Show existing a values for first few n
        for n in existing_n[:3]:
            n_str = str(n)
            existing_a = sorted(int(k) for k in existing_codes[n_str].keys())
            print(f"  n={n}: a values = {existing_a}")
        if len(existing_n) > 3:
            print("  ...")
    else:
        existing_codes = {}
        print("No existing file found, will create new one")
    print()

    # Generate codes for requested range
    print("Generating VT codes...")
    print("-" * 50)

    new_codes = {}
    # Deep copy existing codes
    for n_str, a_dict in existing_codes.items():
        new_codes[n_str] = dict(a_dict)

    generated_count = 0
    skipped_count = 0

    for n in range(args.min_n, args.max_n + 1):
        n_str = str(n)

        # Determine which a values to generate for this n
        if a_mode == "all":
            a_values = list(range(n + 1))  # 0 to n
        else:
            a_values = specified_a_values

        # Initialize dict for this n if needed
        if n_str not in new_codes:
            new_codes[n_str] = {}

        new_for_n = []
        existing_for_n = []

        for a in a_values:
            a_str = str(a)

            if n_str in existing_codes and a_str in existing_codes[n_str]:
                # Already have this (n, a)
                existing_for_n.append(a)
                skipped_count += 1
                continue

            # Generate
            codewords = generate_vt_code(n, a)
            new_codes[n_str][a_str] = codewords
            new_for_n.append((a, len(codewords)))
            generated_count += 1

        # Print status for this n
        if new_for_n:
            new_str = ", ".join(f"a={a}:{cnt}" for a, cnt in new_for_n)
            print(f"  n={n:2d}: NEW [{new_str}]")
        if existing_for_n and not new_for_n:
            print(f"  n={n:2d}: already exists (a={existing_for_n})")

    print()

    if generated_count == 0:
        print("No new codes to generate, all (n, a) combinations already exist")
        return 0

    print(f"Generated {generated_count} new (n, a) combinations")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing (n, a) combinations")

    # Show summary
    print()
    print("Summary of all codes:")
    print("-" * 50)
    for n_str in sorted(new_codes.keys(), key=int):
        n = int(n_str)
        a_info = []
        for a_str in sorted(new_codes[n_str].keys(), key=int):
            count = len(new_codes[n_str][a_str])
            a_info.append(f"a={a_str}:{count}")
        # Truncate if too many a values
        if len(a_info) > 5:
            a_display = ", ".join(a_info[:3]) + f", ... ({len(a_info)} total)"
        else:
            a_display = ", ".join(a_info)
        print(f"  n={n:2d}: {a_display}")

    # Save
    if args.preview:
        print()
        print("Preview mode - not saving.")
    else:
        print()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_vt_codes(new_codes, str(output_path))
        print(f"Saved to {output_path}")

    print()
    print("=" * 70)
    print("  Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
