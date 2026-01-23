#!/usr/bin/env python3
"""
Compare Any Codebook to VT Codes

Unified comparison script that works with any codebook source:
- KAMIS results (.result files)
- Greedy baseline solutions (.txt files)
- Our priority function codebooks (codebooks.json)
- Plain text files

Usage:
    # Compare KAMIS results directory
    python vt/compare.py /path/to/kamis_results/ --mapping-dir /path/to/mappings/

    # Compare greedy solutions
    python vt/compare.py /path/to/greedy_solutions/

    # Compare our priority function codebooks
    python vt/compare.py /path/to/codebooks.json

    # Compare single file
    python vt/compare.py /path/to/solution.txt --n 10

    # Compare against VT_1 instead of VT_0
    python vt/compare.py /path/to/results/ --vt-a 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.codebook_loaders import (
    load_codebook,
    load_codebooks_from_directory,
    load_greedy_solutions,
    load_json_codebook,
    detect_format,
)
from utils.vt_utils import (
    load_vt_codes,
    bitwise_complement,
    compute_vt_complement_index,
    is_flat_vt_format,
)


def compare_codebook_to_vt(
    codebook: Set[str],
    vt_codes: Set[str],
    n: int,
    vt_a: int = 0
) -> Dict:
    """
    Compare a single codebook against VT codes.

    Returns comparison results including complement analysis.
    """
    vt_complement = {bitwise_complement(cw) for cw in vt_codes}
    complement_vt_index = compute_vt_complement_index(n, vt_a)

    vt_overlap = codebook & vt_codes
    comp_overlap = codebook & vt_complement

    vt_pct = 100 * len(vt_overlap) / len(vt_codes) if vt_codes else 0
    comp_pct = 100 * len(comp_overlap) / len(vt_complement) if vt_complement else 0

    # Determine match type
    if len(codebook) == len(vt_codes) and vt_pct == 100:
        match_type = 'perfect_vt'
    elif len(codebook) == len(vt_complement) and comp_pct == 100:
        match_type = 'perfect_complement'
    elif vt_pct > 0 or comp_pct > 0:
        match_type = 'partial'
    else:
        match_type = 'no_overlap'

    return {
        'n': n,
        'codebook_size': len(codebook),
        'vt_size': len(vt_codes),
        'vt_overlap': len(vt_overlap),
        'vt_overlap_pct': round(vt_pct, 2),
        'complement_overlap': len(comp_overlap),
        'complement_overlap_pct': round(comp_pct, 2),
        'complement_vt_index': complement_vt_index,
        'match_type': match_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare any codebook to VT codes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('codebook_path',
                        help='Path to codebook file or directory')
    parser.add_argument('--vt-path',
                        help='Path to VT solutions JSON (default: auto-detect)')
    parser.add_argument('--vt-a', type=int, default=0,
                        help='VT syndrome value to compare against (default: 0)')
    parser.add_argument('--mapping-dir',
                        help='Directory with .mapping files for KAMIS results')
    parser.add_argument('--n', type=int,
                        help='Specific n value (required for single-file codebooks)')
    parser.add_argument('--n-values', type=str,
                        help='Comma-separated n values to analyze')
    parser.add_argument('--s', type=int, default=1,
                        help='s parameter for filename matching (default: 1)')
    parser.add_argument('--q', type=int, default=2,
                        help='q parameter for filename matching (default: 2)')
    parser.add_argument('--output', '-o',
                        help='Output JSON file for results')
    parser.add_argument('--source-name',
                        help='Name for this codebook source (for output)')

    args = parser.parse_args()

    # Parse n_values
    n_values = None
    if args.n_values:
        n_values = [int(x.strip()) for x in args.n_values.split(',')]

    # Find VT solutions
    if args.vt_path:
        vt_path = Path(args.vt_path)
    else:
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / "../src/graphs/vt_solutions.json",
            Path("/workspace/DistributedFunSearch/src/graphs/vt_solutions.json"),
        ]
        vt_path = None
        for c in candidates:
            if c.exists():
                vt_path = c
                break
        if vt_path is None:
            print("Error: Could not find vt_solutions.json. Use --vt-path.")
            return 1

    print("=" * 70)
    print("  Compare Codebooks to VT Codes")
    print("=" * 70)
    print()
    print(f"Codebook source: {args.codebook_path}")
    print(f"VT codes: {vt_path}")
    print(f"Comparing against: VT_{args.vt_a}")
    print()

    # Load VT codes
    print("Loading VT codes...", file=sys.stderr)
    vt_codes = load_vt_codes(str(vt_path), a=args.vt_a)
    print(f"VT codes available for n: {sorted(vt_codes.keys())}")
    print()

    # Load codebooks
    codebook_path = Path(args.codebook_path)
    print(f"Loading codebooks from: {codebook_path}", file=sys.stderr)

    codebooks: Dict[int, Set[str]] = {}
    source_name = args.source_name or codebook_path.stem

    if codebook_path.is_file():
        fmt = detect_format(str(codebook_path))
        print(f"  Detected format: {fmt}")

        result = load_codebook(str(codebook_path), n=args.n)

        if isinstance(result, set):
            if args.n is None:
                print("Error: Single codebook file requires --n parameter")
                return 1
            codebooks = {args.n: result}
        elif isinstance(result, dict):
            # Could be {n: codebook} or {func_idx: {n: codebook}}
            first_val = next(iter(result.values()))
            if isinstance(first_val, set):
                codebooks = result
            elif isinstance(first_val, dict):
                # It's our codebooks.json format - merge all functions
                print(f"  Found {len(result)} functions, merging codebooks...")
                for func_idx, func_codebooks in result.items():
                    for n, cb in func_codebooks.items():
                        if n not in codebooks:
                            codebooks[n] = set()
                        codebooks[n] |= cb
                source_name = f"{source_name} (merged)"
    else:
        # Directory
        if list(codebook_path.glob("*.result")):
            print("  Detected KAMIS results directory")
            codebooks = load_codebooks_from_directory(
                str(codebook_path),
                pattern="*.result",
                mapping_dir=args.mapping_dir,
                n_values=n_values,
                s=args.s,
                q=args.q
            )
            source_name = args.source_name or "KAMIS"
        elif list(codebook_path.glob("*_best_solution.txt")):
            print("  Detected greedy solutions directory")
            codebooks = load_greedy_solutions(
                str(codebook_path),
                n_values=n_values,
                s=args.s,
                q=args.q
            )
            source_name = args.source_name or "Greedy"
        elif list(codebook_path.glob("*.txt")):
            print("  Detected text files directory")
            for f in codebook_path.glob("*.txt"):
                # Try to extract n from filename
                name = f.stem
                for part in name.split('_'):
                    if part.startswith('n') and part[1:].isdigit():
                        n = int(part[1:])
                        if n_values is None or n in n_values:
                            codebooks[n] = load_codebook(str(f))
                        break
            source_name = args.source_name or "Text"
        else:
            print(f"Error: No recognized codebook files in {codebook_path}")
            return 1

    print(f"Loaded codebooks for n: {sorted(codebooks.keys())}")
    print()

    # Filter by n_values if specified
    if n_values:
        codebooks = {n: cb for n, cb in codebooks.items() if n in n_values}

    # Compare
    print(f"VT_{args.vt_a} Reference Codes:")
    print("-" * 60)
    for n in sorted(vt_codes.keys()):
        comp_idx = compute_vt_complement_index(n, args.vt_a)
        print(f"  n={n:2d}: {len(vt_codes[n]):4d} codewords, complement is VT_{comp_idx}")
    print()

    print("Comparison Results:")
    print("-" * 80)
    print(f"{'n':>4} | {'Size':>6} | {'VT size':>7} | {'VT overlap':>10} | {'Comp overlap':>12} | {'Match':>12}")
    print("-" * 80)

    results = []
    for n in sorted(codebooks.keys()):
        if n not in vt_codes:
            print(f"{n:>4} | {len(codebooks[n]):>6} | {'N/A':>7} | {'N/A':>10} | {'N/A':>12} | {'no VT ref':>12}")
            continue

        r = compare_codebook_to_vt(codebooks[n], vt_codes[n], n, args.vt_a)
        results.append(r)

        vt_str = f"{r['vt_overlap']}/{r['vt_size']} ({r['vt_overlap_pct']:.0f}%)"
        comp_str = f"{r['complement_overlap']} ({r['complement_overlap_pct']:.0f}%)"
        print(f"{n:>4} | {r['codebook_size']:>6} | {r['vt_size']:>7} | {vt_str:>10} | {comp_str:>12} | {r['match_type']:>12}")

    print("-" * 80)
    print()

    # Summary
    perfect_vt = sum(1 for r in results if r['match_type'] == 'perfect_vt')
    perfect_comp = sum(1 for r in results if r['match_type'] == 'perfect_complement')
    partial = sum(1 for r in results if r['match_type'] == 'partial')
    no_overlap = sum(1 for r in results if r['match_type'] == 'no_overlap')

    print("Summary:")
    print("-" * 60)
    print(f"  Source: {source_name}")
    print(f"  Perfect VT match: {perfect_vt}/{len(results)}")
    print(f"  Perfect complement match: {perfect_comp}/{len(results)}")
    print(f"  Partial overlap: {partial}/{len(results)}")
    print(f"  No overlap: {no_overlap}/{len(results)}")
    print()
    print("=" * 70)

    # Save results
    if args.output:
        output_data = {
            'source': str(args.codebook_path),
            'source_name': source_name,
            'vt_path': str(vt_path),
            'vt_a': args.vt_a,
            'results': results,
            'summary': {
                'perfect_vt': perfect_vt,
                'perfect_complement': perfect_comp,
                'partial': partial,
                'no_overlap': no_overlap,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
