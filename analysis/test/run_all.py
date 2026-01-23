#!/usr/bin/env python3
"""
Full Analysis Pipeline

This script runs the complete analysis pipeline:
1. Extract successful functions matching target signature
2. Deduplicate by priority hash
3. Evaluate on extended n values (6-16)
4. Analyze VT code overlap
5. Group functions by overlap type

Usage:
    python test/run_all.py <checkpoint_paths...> [options]

Examples:
    python test/run_all.py checkpoint.pkl --output ./my_analysis/
    python test/run_all.py /exp1/checkpoints/ /exp2/checkpoints/ --output ./combined/
    python test/run_all.py /exp1/ /exp2/ /exp3/ --latest-only
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {description} failed with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run full analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('checkpoints', nargs='+',
                        help='Checkpoint files or folders (scans .pkl files recursively)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: ./analysis_<timestamp>/)')
    parser.add_argument('--target', '-t', default=None,
                        help='Target signature (default: VT-optimal for s=1,q=2)')
    parser.add_argument('--latest-only', action='store_true',
                        help='Use only the latest checkpoint from each folder')
    parser.add_argument('--min-n', type=int, default=6,
                        help='Minimum n for evaluation (default: 6)')
    parser.add_argument('--max-n', type=int, default=16,
                        help='Maximum n for evaluation (default: 16)')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='Number of parallel workers for evaluation')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip extraction step (use existing functions)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation step (use existing codebooks)')
    parser.add_argument('--vt-path', default=None,
                        help='Path to VT solutions JSON')
    parser.add_argument('--vt-a', type=int, default=0,
                        help='VT syndrome value to compare against (default: 0)')

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    checkpoint_paths = [Path(p) for p in args.checkpoints]

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./analysis_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Full Analysis Pipeline")
    print("=" * 70)
    print()
    print(f"Checkpoint sources ({len(checkpoint_paths)}):")
    for p in checkpoint_paths:
        print(f"  - {p}")
    if args.latest_only:
        print("  (using latest checkpoint from each folder)")
    print(f"Output: {output_dir}")
    print(f"n range: {args.min_n} to {args.max_n}")
    print(f"VT comparison: VT_{args.vt_a}")
    print()

    # Step 1 & 2: Extract and deduplicate
    if not args.skip_extract:
        cmd = [
            sys.executable, str(script_dir / "extract.py"),
        ]
        # Add all checkpoint paths
        cmd.extend(str(p) for p in checkpoint_paths)
        cmd.extend(["--output", str(output_dir), "-v"])

        if args.target:
            cmd.extend(["--target", args.target])
        if args.latest_only:
            cmd.append("--latest-only")

        if not run_command(cmd, "Step 1 & 2: Extract and Deduplicate Functions"):
            return 1

    # Check that functions were extracted
    functions_json = output_dir / "successful_functions.json"
    if not functions_json.exists():
        print(f"Error: {functions_json} not found")
        print("Run without --skip-extract to create it")
        return 1

    # Step 3: Evaluate on extended inputs
    if not args.skip_eval:
        cmd = [
            sys.executable, str(script_dir / "evaluate.py"),
            str(functions_json),
            "--output", str(output_dir),
            "--min-n", str(args.min_n),
            "--max-n", str(args.max_n),
        ]
        if args.workers:
            cmd.extend(["--workers", str(args.workers)])

        if not run_command(cmd, "Step 3: Evaluate on Extended Inputs"):
            return 1

    # Check that codebooks were generated
    codebooks_json = output_dir / "codebooks.json"
    if not codebooks_json.exists():
        print(f"Error: {codebooks_json} not found")
        print("Run without --skip-eval to create it")
        return 1

    # Step 4 & 5: VT overlap analysis and grouping
    cmd = [
        sys.executable, str(script_dir / "vt_overlap.py"),
        str(codebooks_json),
        "--output", str(output_dir),
        "--vt-a", str(args.vt_a),
    ]
    if args.vt_path:
        cmd.extend(["--vt-path", args.vt_path])

    if not run_command(cmd, f"Step 4 & 5: VT_{args.vt_a} Overlap Analysis and Grouping"):
        return 1

    # Final summary
    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name:40s} ({size:,} bytes)")
        elif f.is_dir():
            count = len(list(f.glob("*")))
            print(f"  {f.name + '/':40s} ({count} files)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
