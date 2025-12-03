#!/usr/bin/env python3
"""
Subset Analysis for Codebooks

Analyze subset relationships between codebooks, e.g.:
- Is a 2-deletion codebook a subset of the 1-deletion VT codebook?
- What fraction of a KAMIS s=2 solution overlaps with VT s=1?

This helps answer: "Does it make sense to start from VT codes and constrain further?"

Usage:
    # Compare KAMIS s=2 results against VT s=1
    python subset_analysis.py /path/to/kamis_s2_results/ --vt-s 1

    # Compare any codebook against VT
    python subset_analysis.py /path/to/codebook.txt --vt-s 1 --n 10

    # Compare two codebook directories
    python subset_analysis.py /path/to/s2_results/ --reference /path/to/s1_results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

from codebook_loaders import (
    load_codebook,
    load_codebooks_from_directory,
    load_greedy_solutions,
    detect_format,
)
from helpers import is_flat_vt_format


def load_vt_codes(vt_path: str, a: int = 0) -> Dict[int, Set[str]]:
    """Load VT codes from JSON file."""
    with open(vt_path) as f:
        vt_data = json.load(f)

    is_nested = not is_flat_vt_format(vt_data)

    vt_codes = {}
    for n_str, value in vt_data.items():
        n = int(n_str)
        if is_nested:
            a_str = str(a)
            if a_str in value:
                vt_codes[n] = set(value[a_str])
        else:
            if a == 0:
                vt_codes[n] = set(value)

    return vt_codes


def analyze_subset(
    codebook: Set[str],
    reference: Set[str],
    codebook_name: str = "codebook",
    reference_name: str = "reference"
) -> Dict:
    """
    Analyze if codebook is a subset of reference.

    Returns:
        Dict with subset analysis results
    """
    overlap = codebook & reference
    only_in_codebook = codebook - reference
    only_in_reference = reference - codebook

    is_subset = codebook <= reference
    is_superset = codebook >= reference
    is_equal = codebook == reference

    overlap_pct = 100 * len(overlap) / len(codebook) if codebook else 0
    coverage_pct = 100 * len(overlap) / len(reference) if reference else 0

    return {
        'codebook_size': len(codebook),
        'reference_size': len(reference),
        'overlap_size': len(overlap),
        'overlap_pct': round(overlap_pct, 2),
        'coverage_pct': round(coverage_pct, 2),
        'only_in_codebook': len(only_in_codebook),
        'only_in_reference': len(only_in_reference),
        'is_subset': is_subset,
        'is_superset': is_superset,
        'is_equal': is_equal,
        'non_overlapping_codewords': sorted(only_in_codebook)[:20],  # First 20 for inspection
    }


def analyze_multi_n(
    codebooks: Dict[int, Set[str]],
    references: Dict[int, Set[str]],
    codebook_name: str = "codebook",
    reference_name: str = "reference"
) -> Tuple[Dict[int, Dict], Dict]:
    """
    Analyze subset relationships for multiple n values.

    Returns:
        (by_n_results, summary)
    """
    by_n = {}
    all_subset = True
    all_superset = True
    total_overlap = 0
    total_codebook = 0
    total_reference = 0

    n_values = sorted(set(codebooks.keys()) & set(references.keys()))

    for n in n_values:
        codebook = codebooks.get(n, set())
        reference = references.get(n, set())

        if not codebook or not reference:
            continue

        result = analyze_subset(codebook, reference, codebook_name, reference_name)
        by_n[n] = result

        if not result['is_subset']:
            all_subset = False
        if not result['is_superset']:
            all_superset = False

        total_overlap += result['overlap_size']
        total_codebook += result['codebook_size']
        total_reference += result['reference_size']

    summary = {
        'n_values_analyzed': n_values,
        'all_subset': all_subset,
        'all_superset': all_superset,
        'total_overlap': total_overlap,
        'total_codebook_size': total_codebook,
        'total_reference_size': total_reference,
        'overall_overlap_pct': round(100 * total_overlap / total_codebook, 2) if total_codebook else 0,
        'overall_coverage_pct': round(100 * total_overlap / total_reference, 2) if total_reference else 0,
    }

    return by_n, summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze subset relationships between codebooks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('codebook_path',
                        help='Path to codebook file or directory')
    parser.add_argument('--reference', '-r',
                        help='Path to reference codebook file or directory')
    parser.add_argument('--vt-path',
                        help='Path to VT solutions JSON (alternative to --reference)')
    parser.add_argument('--vt-a', type=int, default=0,
                        help='VT syndrome value (default: 0)')
    parser.add_argument('--mapping-dir',
                        help='Directory with .mapping files for KAMIS results')
    parser.add_argument('--n', type=int,
                        help='Specific n value to analyze (for single-file codebooks)')
    parser.add_argument('--n-values', type=str,
                        help='Comma-separated n values to analyze (e.g., "6,7,8,9,10")')
    parser.add_argument('--s', type=int, default=1,
                        help='s parameter for filename matching (default: 1)')
    parser.add_argument('--q', type=int, default=2,
                        help='q parameter for filename matching (default: 2)')
    parser.add_argument('--output', '-o',
                        help='Output directory for results')
    parser.add_argument('--show-non-overlapping', action='store_true',
                        help='Show non-overlapping codewords in output')

    args = parser.parse_args()

    # Parse n_values
    n_values = None
    if args.n_values:
        n_values = [int(x.strip()) for x in args.n_values.split(',')]
    elif args.n:
        n_values = [args.n]

    # Setup output
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.codebook_path).parent if Path(args.codebook_path).is_file() else Path(args.codebook_path)

    print("=" * 70)
    print("  Subset Analysis")
    print("=" * 70)
    print()

    # Load codebooks
    codebook_path = Path(args.codebook_path)
    print(f"Loading codebooks from: {codebook_path}")

    if codebook_path.is_file():
        fmt = detect_format(str(codebook_path))
        print(f"  Detected format: {fmt}")

        if args.n:
            codebooks = {args.n: load_codebook(str(codebook_path), n=args.n)}
        else:
            result = load_codebook(str(codebook_path))
            if isinstance(result, set):
                # Single codebook, need n
                if not args.n:
                    print("Error: Single codebook file requires --n parameter")
                    return 1
                codebooks = {args.n: result}
            else:
                codebooks = result
    else:
        # Directory of results
        if list(codebook_path.glob("*.result")):
            codebooks = load_codebooks_from_directory(
                str(codebook_path),
                pattern="*.result",
                mapping_dir=args.mapping_dir,
                n_values=n_values,
                s=args.s,
                q=args.q
            )
        elif list(codebook_path.glob("*_best_solution.txt")):
            codebooks = load_greedy_solutions(
                str(codebook_path),
                n_values=n_values,
                s=args.s,
                q=args.q
            )
        else:
            print(f"Error: No recognized codebook files in {codebook_path}")
            return 1

    print(f"  Loaded codebooks for n: {sorted(codebooks.keys())}")
    for n in sorted(codebooks.keys()):
        print(f"    n={n}: {len(codebooks[n])} codewords")
    print()

    # Load reference
    if args.vt_path:
        print(f"Loading VT_{args.vt_a} codes from: {args.vt_path}")
        references = load_vt_codes(args.vt_path, a=args.vt_a)
        reference_name = f"VT_{args.vt_a}"
    elif args.reference:
        ref_path = Path(args.reference)
        print(f"Loading reference from: {ref_path}")

        if ref_path.is_file():
            if args.n:
                references = {args.n: load_codebook(str(ref_path), n=args.n)}
            else:
                result = load_codebook(str(ref_path))
                if isinstance(result, set):
                    if not args.n:
                        print("Error: Single reference file requires --n parameter")
                        return 1
                    references = {args.n: result}
                else:
                    references = result
        else:
            if list(ref_path.glob("*.result")):
                references = load_codebooks_from_directory(
                    str(ref_path),
                    pattern="*.result",
                    mapping_dir=args.mapping_dir,
                    n_values=n_values,
                    s=args.s,
                    q=args.q
                )
            else:
                references = load_greedy_solutions(
                    str(ref_path),
                    n_values=n_values,
                    s=args.s,
                    q=args.q
                )
        reference_name = "reference"
    else:
        # Default: use VT codes
        script_dir = Path(__file__).parent
        vt_path = script_dir / "../src/graphs/vt_solutions.json"
        if not vt_path.exists():
            vt_path = Path("/workspace/DistributedFunSearch/src/graphs/vt_solutions.json")
        print(f"Loading VT_{args.vt_a} codes from: {vt_path}")
        references = load_vt_codes(str(vt_path), a=args.vt_a)
        reference_name = f"VT_{args.vt_a}"

    print(f"  Loaded references for n: {sorted(references.keys())}")
    print()

    # Analyze
    print("Analyzing subset relationships...")
    print("-" * 70)

    by_n, summary = analyze_multi_n(codebooks, references, "codebook", reference_name)

    # Print results
    print()
    print("Results by n:")
    print("-" * 70)
    print(f"{'n':>4} | {'Codebook':>10} | {reference_name:>10} | {'Overlap':>10} | {'Overlap%':>10} | {'Subset?':>8}")
    print("-" * 70)

    for n in sorted(by_n.keys()):
        r = by_n[n]
        subset_str = "YES" if r['is_subset'] else "NO"
        print(f"{n:>4} | {r['codebook_size']:>10} | {r['reference_size']:>10} | "
              f"{r['overlap_size']:>10} | {r['overlap_pct']:>9.1f}% | {subset_str:>8}")

        if args.show_non_overlapping and r['non_overlapping_codewords']:
            print(f"       Non-overlapping (first 10): {r['non_overlapping_codewords'][:10]}")

    print("-" * 70)
    print()

    # Summary
    print("Summary:")
    print("-" * 70)
    print(f"  n values analyzed: {summary['n_values_analyzed']}")
    print(f"  All codebooks are subsets: {summary['all_subset']}")
    print(f"  Overall overlap: {summary['total_overlap']}/{summary['total_codebook_size']} ({summary['overall_overlap_pct']:.1f}%)")

    if summary['all_subset']:
        print()
        print("  CONCLUSION: All codebooks are subsets of the reference.")
        print("  -> Starting from reference and constraining further makes sense!")
    else:
        print()
        print("  CONCLUSION: Some codebooks contain codewords NOT in the reference.")
        print("  -> May need to explore beyond the reference codebook.")

    print()
    print("=" * 70)

    # Save results
    results = {
        'codebook_source': str(args.codebook_path),
        'reference_source': str(args.vt_path or args.reference or 'vt_solutions.json'),
        'vt_a': args.vt_a if args.vt_path or (not args.reference) else None,
        'by_n': {str(k): v for k, v in by_n.items()},
        'summary': summary,
    }

    # Remove non-serializable items
    for n_key in results['by_n']:
        if 'non_overlapping_codewords' in results['by_n'][n_key]:
            results['by_n'][n_key]['non_overlapping_codewords'] = \
                results['by_n'][n_key]['non_overlapping_codewords'][:100]  # Limit

    output_file = output_dir / "subset_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
