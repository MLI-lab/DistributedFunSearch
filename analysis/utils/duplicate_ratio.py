"""Compute per-island duplicate ratio from w/o Dedup. checkpoint files.

For each island, counts total programs and unique hash values.
Duplicate ratio = (total - unique) / total.
Reports per-island ratios, mean across islands, and mean across runs.

Usage:
    python analysis/checkpoint/duplicate_ratio.py checkpoint1.pkl checkpoint2.pkl checkpoint3.pkl
"""

import ast
import pickle
import sys
from pathlib import Path

import numpy as np


def load_island_stats(checkpoint_path):
    """Load checkpoint and compute per-island duplicate stats."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    print(f"\n{'='*60}")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"  Iterations: {data.get('iterations', '?')}")
    print(f"  Total stored programs: {data.get('total_stored_programs', '?')}")
    print(f"  Duplicates discarded: {data.get('duplicates_discarded', '?')}")
    print(f"{'='*60}")

    island_stats = []
    for island_id, island_state in enumerate(data["islands_state"]):
        # Count all programs and their hashes across all clusters
        total_programs = 0
        all_hashes = []
        for signature_str, cluster_state in island_state["clusters"].items():
            for prog_dict in cluster_state["programs"]:
                total_programs += 1
                h = prog_dict.get("hash_value")
                if h is not None:
                    all_hashes.append(h)

        unique_hashes = len(set(all_hashes))
        programs_with_hash = len(all_hashes)
        duplicates = programs_with_hash - unique_hashes
        ratio = duplicates / total_programs if total_programs > 0 else 0.0

        island_stats.append({
            "island_id": island_id,
            "total_programs": total_programs,
            "programs_with_hash": programs_with_hash,
            "unique_hashes": unique_hashes,
            "duplicates": duplicates,
            "duplicate_ratio": ratio,
        })

        print(f"  Island {island_id:2d}: {total_programs:5d} programs, "
              f"{unique_hashes:5d} unique, {duplicates:4d} duplicates "
              f"({ratio:.1%})")

    ratios = [s["duplicate_ratio"] for s in island_stats]
    mean_ratio = np.mean(ratios)
    total_progs = sum(s["total_programs"] for s in island_stats)
    total_dupes = sum(s["duplicates"] for s in island_stats)
    global_ratio = total_dupes / total_progs if total_progs > 0 else 0.0

    print(f"\n  Mean duplicate ratio (across islands): {mean_ratio:.1%}")
    print(f"  Global duplicate ratio (pooled):       {global_ratio:.1%}")
    print(f"  Total: {total_progs} programs, {total_dupes} duplicates")

    return island_stats, mean_ratio, global_ratio


def main():
    if len(sys.argv) < 2:
        print("Usage: python duplicate_ratio.py checkpoint1.pkl [checkpoint2.pkl ...]")
        sys.exit(1)

    checkpoint_paths = sys.argv[1:]
    run_ratios = []

    for path in checkpoint_paths:
        island_stats, mean_ratio, global_ratio = load_island_stats(path)
        run_ratios.append(mean_ratio)

    if len(run_ratios) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY ACROSS {len(run_ratios)} RUNS")
        print(f"{'='*60}")
        for i, (path, ratio) in enumerate(zip(checkpoint_paths, run_ratios)):
            print(f"  Run {i+1} ({Path(path).name}): {ratio:.1%}")
        print(f"\n  Mean duplicate ratio: {np.mean(run_ratios):.1%} "
              f"(+/- {np.std(run_ratios):.1%})")


if __name__ == "__main__":
    main()
