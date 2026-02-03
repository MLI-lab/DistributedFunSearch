#!/usr/bin/env python3
"""
VT Code Overlap Analysis and Grouping

This script analyzes how the generated codebooks compare to Varshamov-Tenengolts (VT) codes.

For each function and each n value, it computes:
1. Overlap with VT_a codes (default: VT_0)
2. Overlap with the bitwise complement of VT_a codes

Usage:
    python test/vt_overlap.py <codebooks_json> [options]

Examples:
    python test/vt_overlap.py ./successful_functions/codebooks.json
    python test/vt_overlap.py ./codebooks.json --vt-a 1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.vt_utils import (
    load_vt_codes,
    bitwise_complement,
    compute_vt_complement_index,
    is_flat_vt_format,
    complement_codebook,
)


def compute_overlap(codebook: Set[str], reference: Set[str]) -> Tuple[int, int, float]:
    """
    Compute overlap between codebook and reference.

    Returns:
        - intersection_size: number of common codewords
        - reference_size: size of reference codebook
        - overlap_percentage: intersection/reference * 100
    """
    if not reference:
        return 0, 0, 0.0

    intersection = codebook & reference
    intersection_size = len(intersection)
    reference_size = len(reference)
    overlap_pct = 100.0 * intersection_size / reference_size if reference_size > 0 else 0.0

    return intersection_size, reference_size, overlap_pct


def analyze_single_function(func_data: Dict, vt_codes: Dict[int, Set[str]],
                            vt_a: int = 0, expected_n_values: List[int] = None) -> Dict:
    """
    Analyze VT overlap for a single function across all n values.

    Args:
        func_data: Function data with codebooks
        vt_codes: VT reference codes {n: set of codewords}
        vt_a: The VT syndrome value being compared against
        expected_n_values: List of n values we expect codebooks for (to track missing)

    Returns dict with overlap analysis for each n.
    """
    func_id = func_data.get('func_id', 'unknown')
    codebooks = func_data.get('codebooks', {})

    analysis = {
        'func_index': func_data.get('func_index', -1),
        'func_id': func_id,
        'priority_hash': func_data.get('priority_hash'),
        'by_n': {},
        'summary': {
            'perfect_vt_match_ns': [],
            'perfect_complement_match_ns': [],
            'partial_overlap_ns': [],
            'no_overlap_ns': [],
            'missing_codebook_ns': [],  # n values with no codebook for this function
        }
    }

    for n_str, codebook_list in codebooks.items():
        n = int(n_str)
        codebook = set(codebook_list)
        codebook_size = len(codebook)

        # Get VT reference codes
        vt_ref = vt_codes.get(n, set())
        vt_complement = complement_codebook(vt_ref) if vt_ref else set()

        # Compute overlaps
        vt_intersect, vt_size, vt_pct = compute_overlap(codebook, vt_ref)
        comp_intersect, comp_size, comp_pct = compute_overlap(codebook, vt_complement)

        # Determine match type
        if vt_size == 0:
            match_type = 'no_vt_reference'
        elif codebook_size == vt_size and vt_pct == 100.0:
            match_type = 'perfect_vt_match'
            analysis['summary']['perfect_vt_match_ns'].append(n)
        elif codebook_size == comp_size and comp_pct == 100.0:
            match_type = 'perfect_complement_match'
            analysis['summary']['perfect_complement_match_ns'].append(n)
        elif vt_pct > 0 or comp_pct > 0:
            match_type = 'partial_overlap'
            analysis['summary']['partial_overlap_ns'].append(n)
        else:
            match_type = 'no_overlap'
            analysis['summary']['no_overlap_ns'].append(n)

        # VT complement index (for reference)
        complement_vt_index = compute_vt_complement_index(n, vt_a)

        analysis['by_n'][n] = {
            'codebook_size': codebook_size,
            'vt_size': vt_size,
            'vt_overlap': vt_intersect,
            'vt_overlap_pct': round(vt_pct, 2),
            'complement_overlap': comp_intersect,
            'complement_overlap_pct': round(comp_pct, 2),
            'complement_vt_index': complement_vt_index,
            'match_type': match_type,
            'is_odd_n': n % 2 == 1,
        }

    # Track missing n values (expected but no codebook)
    if expected_n_values:
        codebook_ns = set(int(k) for k in codebooks.keys())
        for n in expected_n_values:
            if n not in codebook_ns:
                analysis['summary']['missing_codebook_ns'].append(n)

    return analysis


def group_functions_by_overlap(analyses: List[Dict], n_values: List[int]) -> Dict:
    """
    Group functions based on their VT overlap patterns.

    Categories:
    1. perfect_vt_all: 100% VT match for all n (no missing)
    2. perfect_complement_all: 100% complement match for all n (no missing)
    3. perfect_alternating: 100% match for all n, with VT on even n and complement on odd n
    4. incomplete_data: Missing codebook for one or more expected n values
    5. perfect_some: 100% match for some n, partial/none for others
    6. partial_only: Only partial overlaps
    7. no_overlap: No overlap with VT at any n
    """
    groups = {
        'perfect_vt_all': [],
        'perfect_complement_all': [],
        'perfect_alternating': [],  # VT on even n, complement on odd n
        'incomplete_data': [],  # Functions missing codebooks for some n values
        'perfect_some': [],
        'partial_only': [],
        'no_overlap': [],
    }

    for analysis in analyses:
        perfect_vt = set(analysis['summary']['perfect_vt_match_ns'])
        perfect_comp = set(analysis['summary']['perfect_complement_match_ns'])
        partial = set(analysis['summary']['partial_overlap_ns'])
        none = set(analysis['summary']['no_overlap_ns'])
        missing = set(analysis['summary'].get('missing_codebook_ns', []))

        n_set = set(n_values)
        perfect_all = perfect_vt | perfect_comp

        func_info = {
            'func_index': analysis['func_index'],
            'func_id': analysis['func_id'],
            'priority_hash': analysis['priority_hash'],
            'perfect_vt_ns': sorted(perfect_vt),
            'perfect_complement_ns': sorted(perfect_comp),
            'partial_ns': sorted(partial),
            'no_overlap_ns': sorted(none),
            'missing_ns': sorted(missing),
        }

        # Check for missing data first
        if missing:
            groups['incomplete_data'].append(func_info)
        elif perfect_vt == n_set:
            groups['perfect_vt_all'].append(func_info)
        elif perfect_comp == n_set:
            groups['perfect_complement_all'].append(func_info)
        elif perfect_all == n_set:
            # Perfect match for all n, but alternates between VT (even n) and complement (odd n)
            groups['perfect_alternating'].append(func_info)
        elif len(perfect_all) > 0:
            groups['perfect_some'].append(func_info)
        elif len(partial) > 0:
            groups['partial_only'].append(func_info)
        else:
            groups['no_overlap'].append(func_info)

    return groups


def save_grouped_functions(groups: Dict, functions: List[Dict],
                            output_dir: Path, prefix: str = "grouped"):
    """Save functions grouped by their VT overlap type."""

    # Create subdirectory for grouped functions
    grouped_dir = output_dir / "grouped_by_vt_overlap"
    grouped_dir.mkdir(parents=True, exist_ok=True)

    # Map func_index to full function data
    func_by_index = {f['func_index']: f for f in functions}

    for group_name, func_list in groups.items():
        if not func_list:
            continue

        # Save JSON with full function data
        json_path = grouped_dir / f"{group_name}.json"
        full_data = []
        for func_info in func_list:
            idx = func_info['func_index']
            if idx in func_by_index:
                full_data.append({
                    **func_by_index[idx],
                    'vt_analysis': func_info,
                })

        with open(json_path, 'w') as f:
            json.dump(full_data, f, indent=2)

        # Save Python file with function bodies
        py_path = grouped_dir / f"{group_name}.py"
        with open(py_path, 'w') as f:
            f.write(f'"""Functions with VT overlap type: {group_name}\n\n')
            f.write(f'Total functions: {len(func_list)}\n')
            f.write('"""\n\n')

            for i, func_info in enumerate(func_list):
                idx = func_info['func_index']
                if idx not in func_by_index:
                    continue

                func = func_by_index[idx]
                f.write(f"# Function {i+1} (original index: {idx})\n")
                f.write(f"# Priority hash: {func.get('priority_hash', 'N/A')}\n")
                f.write(f"# Perfect VT at n: {func_info['perfect_vt_ns']}\n")
                f.write(f"# Perfect complement at n: {func_info['perfect_complement_ns']}\n")
                f.write(f"def priority_{i+1}(node, n, s, q):\n")

                body = func['body']
                if not body.startswith('    '):
                    body = '\n'.join('    ' + line if line.strip() else line
                                    for line in body.split('\n'))
                f.write(body)
                f.write('\n\n')

        print(f"  {group_name}: {len(func_list)} functions -> {json_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze VT code overlap for generated codebooks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('codebooks_json', help='Path to codebooks JSON file')
    parser.add_argument('--vt-path', default=None,
                        help='Path to VT solutions JSON (default: auto-detect)')
    parser.add_argument('--vt-a', type=int, default=0,
                        help='VT syndrome value to compare against (default: 0 for VT_0)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: ./vt_overlap_<timestamp>/)')

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.codebooks_json)
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./vt_overlap_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find VT solutions file
    if args.vt_path:
        vt_path = Path(args.vt_path)
    else:
        # Try to find it relative to this script
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / "../src/graphs/vt_solutions.json",
            script_dir.parent / "src/graphs/vt_solutions.json",
            Path("/workspace/DistributedFunSearch/src/graphs/vt_solutions.json"),
        ]
        vt_path = None
        for candidate in candidates:
            if candidate.exists():
                vt_path = candidate
                break

        if vt_path is None:
            print("Error: Could not find vt_solutions.json")
            print("Please specify with --vt-path")
            return 1

    print("=" * 70)
    print("  VT Code Overlap Analysis")
    print("=" * 70)
    print()
    print(f"Codebooks: {input_path}")
    print(f"VT codes: {vt_path}")
    print(f"Comparing against: VT_{args.vt_a}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading codebooks...", file=sys.stderr)
    with open(input_path) as f:
        codebooks_data = json.load(f)

    print(f"Loading VT_{args.vt_a} codes...", file=sys.stderr)
    vt_codes = load_vt_codes(str(vt_path), a=args.vt_a)

    functions = codebooks_data['functions']
    n_values = codebooks_data.get('n_values', list(range(6, 17)))

    print(f"Loaded {len(functions)} functions")
    print(f"VT codes available for n: {sorted(vt_codes.keys())}")
    print(f"Analyzing n values: {n_values}")
    print()

    # Show VT code sizes and complement info
    print(f"VT_{args.vt_a} Reference Codes:")
    print("-" * 60)
    for n in sorted(vt_codes.keys()):
        comp_idx = compute_vt_complement_index(n, args.vt_a)
        odd_str = "odd" if n % 2 == 1 else "even"
        print(f"  n={n:2d} ({odd_str}): {len(vt_codes[n]):4d} codewords, "
              f"complement is VT_{comp_idx}")
    print()

    # Analyze each function
    print("Analyzing VT overlap for each function...", file=sys.stderr)
    analyses = []
    for i, func in enumerate(functions):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(functions)}...", file=sys.stderr)
        analysis = analyze_single_function(func, vt_codes, vt_a=args.vt_a,
                                           expected_n_values=n_values)
        analyses.append(analysis)

    # Group functions by overlap pattern
    print("\nGrouping functions by VT overlap pattern...", file=sys.stderr)
    groups = group_functions_by_overlap(analyses, n_values)

    # Print summary
    print("\n" + "=" * 70)
    print("  Analysis Results")
    print("=" * 70)
    print()

    print("Functions by VT overlap category:")
    print("-" * 60)
    category_descriptions = {
        'perfect_vt_all': 'VT match for all n',
        'perfect_complement_all': 'complement match for all n',
        'perfect_alternating': 'VT on even n, complement on odd n',
        'incomplete_data': 'missing codebook for some n',
        'perfect_some': 'perfect for some n only',
        'partial_only': 'partial overlap only',
        'no_overlap': 'no VT overlap',
    }
    for group_name, func_list in groups.items():
        desc = category_descriptions.get(group_name, '')
        print(f"  {group_name:25s}: {len(func_list):4d} functions  ({desc})")
    print()

    # Detailed statistics per n
    print("VT overlap by n value:")
    print("-" * 70)

    total_funcs = len(functions)
    for n in n_values:
        if n not in vt_codes:
            print(f"  n={n:2d}: No VT reference available")
            continue

        perfect_vt = sum(1 for a in analyses if n in a['summary']['perfect_vt_match_ns'])
        perfect_comp = sum(1 for a in analyses if n in a['summary']['perfect_complement_match_ns'])
        partial = sum(1 for a in analyses if n in a['summary']['partial_overlap_ns'])
        no_match = sum(1 for a in analyses if n in a['summary']['no_overlap_ns'])
        missing = sum(1 for a in analyses if n in a['summary'].get('missing_codebook_ns', []))

        accounted = perfect_vt + perfect_comp + partial + no_match + missing
        print(f"  n={n:2d}: VT={perfect_vt:3d}, Comp={perfect_comp:3d}, "
              f"Partial={partial:3d}, None={no_match:3d}, Missing={missing:3d} "
              f"(total={accounted}/{total_funcs})")

    print()

    # Save full analysis
    analysis_path = output_dir / "vt_overlap_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            'vt_a': args.vt_a,
            'n_values': n_values,
            'vt_sizes': {n: len(vt_codes.get(n, [])) for n in n_values},
            'analyses': analyses,
            'groups_summary': {k: len(v) for k, v in groups.items()},
        }, f, indent=2)
    print(f"Saved detailed analysis to {analysis_path}")

    # Save grouped functions
    print("\nSaving grouped functions...")
    save_grouped_functions(groups, functions, output_dir)

    # Save summary report
    report_path = output_dir / "vt_overlap_report.txt"
    with open(report_path, 'w') as f:
        f.write("VT Code Overlap Analysis Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("Summary by Category:\n")
        f.write("-" * 60 + "\n")
        for group_name, func_list in groups.items():
            f.write(f"  {group_name:30s}: {len(func_list):4d} functions\n")
        f.write("\n")

        f.write("Overlap by n value:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'n':>4s} | {'VT size':>8s} | {'VT match':>8s} | {'Comp match':>10s} | "
                f"{'Partial':>8s} | {'None':>8s} | {'Missing':>8s}\n")
        f.write("-" * 80 + "\n")

        for n in n_values:
            if n not in vt_codes:
                continue
            perfect_vt = sum(1 for a in analyses if n in a['summary']['perfect_vt_match_ns'])
            perfect_comp = sum(1 for a in analyses if n in a['summary']['perfect_complement_match_ns'])
            partial = sum(1 for a in analyses if n in a['summary']['partial_overlap_ns'])
            no_match = sum(1 for a in analyses if n in a['summary']['no_overlap_ns'])
            missing = sum(1 for a in analyses if n in a['summary'].get('missing_codebook_ns', []))

            f.write(f"{n:4d} | {len(vt_codes[n]):8d} | {perfect_vt:8d} | "
                    f"{perfect_comp:10d} | {partial:8d} | {no_match:8d} | {missing:8d}\n")

        f.write("\n")
        f.write(f"VT Complement Index (complement of VT_{args.vt_a} is VT_b):\n")
        f.write("-" * 60 + "\n")
        for n in n_values:
            b = compute_vt_complement_index(n, args.vt_a)
            f.write(f"  n={n:2d}: VT_{args.vt_a} complement -> VT_{b}\n")

    print(f"Saved report to {report_path}")

    print()
    print("=" * 70)
    print("  VT Overlap Analysis Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
