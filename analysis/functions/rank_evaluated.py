#!/usr/bin/env python3
"""
Rank evaluated functions by their gap to best known values.

Reads evaluation_results.json (output of evaluate.py) and prints:
  1. Max score per n across all functions (like max_scores_per_n.py but from evaluate output)
  2. Top K functions ranked by minimum gap to best known across all evaluated n values

Usage:
    # Basic usage (reads evaluation_results.json)
    python rank_evaluated.py ./evaluate_20260305_123456/evaluation_results.json

    # Show top 10 functions, s=2
    python rank_evaluated.py ./evaluate_*/evaluation_results.json --top 10 --s 2

    # Multiple result files
    python rank_evaluated.py results1.json results2.json --s 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.best_known import get_best_known


def load_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def rank_functions(results: list, best_known: dict, s: int) -> list:
    """Rank functions by minimum gap to best known across all n values.

    Returns list of dicts with func_id, sizes, gaps, min_gap, avg_gap.
    """
    ranked = []
    for entry in results:
        sizes = entry.get("sizes", {})
        if not sizes or entry.get("error"):
            continue

        gaps = {}
        for n_str, size in sizes.items():
            n = int(n_str)
            key = (n, s)
            if key in best_known and size > 0:
                gaps[n] = size / best_known[key]  # ratio (1.0 = optimal)

        if not gaps:
            continue

        min_gap = min(gaps.values())
        avg_gap = sum(gaps.values()) / len(gaps)

        ranked.append({
            "func_id": entry.get("func_id", "unknown"),
            "func_index": entry.get("func_index", -1),
            "signature": entry.get("signature", "?"),
            "sizes": sizes,
            "gaps": gaps,
            "min_gap": min_gap,
            "avg_gap": avg_gap,
        })

    # Sort by min_gap descending (higher ratio = closer to optimal)
    ranked.sort(key=lambda x: x["min_gap"], reverse=True)
    return ranked


def print_max_scores(results: list, best_known: dict, s: int):
    """Print max score per n across all functions."""
    max_per_n = {}
    for entry in results:
        sizes = entry.get("sizes", {})
        if entry.get("error"):
            continue
        for n_str, size in sizes.items():
            n = int(n_str)
            if size > 0:
                if n not in max_per_n or size > max_per_n[n]:
                    max_per_n[n] = size

    if not max_per_n:
        print("No valid results found.")
        return

    print("Max score per n (across all functions)")
    print(f"{'n':>6}  {'best_found':>12}  {'best_known':>12}  {'ratio':>8}  {'gap':>8}")

    for n in sorted(max_per_n.keys()):
        found = max_per_n[n]
        key = (n, s)
        known = best_known.get(key)
        if known:
            ratio = found / known
            gap = known - found
            marker = " *" if found >= known else ""
            print(f"{n:>6}  {found:>12}  {known:>12}  {ratio:>8.4f}  {gap:>8}{marker}")
        else:
            print(f"{n:>6}  {found:>12}  {'?':>12}  {'?':>8}  {'?':>8}")

    print("\n  * = matches or exceeds best known")


def print_top_functions(ranked: list, best_known: dict, s: int, top_k: int):
    """Print top K functions with best minimum gap."""
    if not ranked:
        print("\nNo valid functions to rank.")
        return

    # Collect all n values
    all_n = sorted(set(n for entry in ranked for n in entry["gaps"].keys()))

    print(f"\nTop {min(top_k, len(ranked))} functions (ranked by min ratio to best known)")

    for i, entry in enumerate(ranked[:top_k]):
        func_id = entry["func_id"]
        func_idx = entry["func_index"]
        print(f"\n#{i+1}: func_index={func_idx}, func_id={func_id}, "
              f"signature={entry['signature']}")
        print(f"    min_ratio={entry['min_gap']:.4f}, avg_ratio={entry['avg_gap']:.4f}")

        # Size table
        header = "    " + "".join(f"{'n=' + str(n):>8}" for n in all_n)
        print(header)

        # Best known row
        row_known = "    "
        for n in all_n:
            key = (n, s)
            val = best_known.get(key, "?")
            row_known += f"{val:>8}"
        print(f"  bk{row_known[4:]}")

        # Function scores row
        row_scores = "    "
        for n in all_n:
            size = entry["sizes"].get(str(n), 0)
            if size > 0:
                key = (n, s)
                known = best_known.get(key)
                if known and size >= known:
                    row_scores += f"{'*' + str(size):>8}"
                else:
                    row_scores += f"{size:>8}"
            else:
                row_scores += f"{'—':>8}"
        print(f"  sc{row_scores[4:]}")

        # Ratio row
        row_ratio = "    "
        for n in all_n:
            ratio = entry["gaps"].get(n)
            if ratio is not None:
                row_ratio += f"{ratio:>8.3f}"
            else:
                row_ratio += f"{'—':>8}"
        print(f"   %%{row_ratio[4:]}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank evaluated functions by gap to best known values."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Path(s) to evaluation_results.json from evaluate.py",
    )
    parser.add_argument("--s", type=int, default=2, help="s parameter (default: 2)")
    parser.add_argument("--top", type=int, default=5, help="Number of top functions to show (default: 5)")
    args = parser.parse_args()

    best_known = get_best_known(args.s, extended=True)

    # Load and merge all results
    all_results = []
    for path in args.results:
        p = Path(path)
        if not p.exists():
            print(f"Warning: file not found: {path}")
            continue
        results = load_results(path)
        print(f"Loaded {len(results)} functions from {path}")
        # Tag with source file
        for entry in results:
            entry["_source"] = str(p.parent.name)
        all_results.extend(results)

    if not all_results:
        print("No results loaded.")
        return

    print(f"\nTotal: {len(all_results)} functions, s={args.s}\n")

    # Max scores table
    print_max_scores(all_results, best_known, args.s)

    # Rank and show top functions
    ranked = rank_functions(all_results, best_known, args.s)
    print_top_functions(ranked, best_known, args.s, args.top)

    # Summary
    n_valid = len(ranked)
    n_error = sum(1 for r in all_results if r.get("error"))
    print(f"\nSummary: {n_valid} valid functions, {n_error} with errors, "
          f"{len(all_results) - n_valid - n_error} empty")


if __name__ == "__main__":
    main()
