"""Find the true max score per n across all checkpoints for given experiments.

Scans all checkpoint pkl files across all runs for each experiment config,
iterates every cluster on every island, and reports the max score achieved
for each (n, s) test key. This gives the true max per n across the entire
search trajectory, not just the best overall scoring function.

Only scans direct child run folders, not nested subdirectories like
with_reasoning/. To include reasoning runs, specify the full path
explicitly, e.g. qwen14B/code/with_reasoning.

Usage:
    python max_scores_per_n.py qwen14B/code starcoder/code
    python max_scores_per_n.py qwen8B/reflection qwen14B/reflection qwen32B/reflection
    python max_scores_per_n.py qwen14B/code qwen14B/code/with_reasoning
    python max_scores_per_n.py --base-path /mnt/disfun/checkpoints/s2 qwen14B/code

Experiment paths are relative to --base-path.
"""

import argparse
import ast
import pickle
from pathlib import Path


default_base_path = "/mnt/disfun/checkpoints/s2"

# Best known scores for s=2.
optimal_scores = {
    (7, 2): 5,
    (8, 2): 7,
    (9, 2): 11,
    (10, 2): 16,
    (11, 2): 24,
    (12, 2): 37,
}


def parse_test_key(key) -> tuple | None:
    """Parse a test key (tuple or string like '(7, 2)') into a tuple."""
    if isinstance(key, tuple):
        return key
    try:
        parsed = ast.literal_eval(key)
        if isinstance(parsed, tuple):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def load_checkpoint(pkl_path: Path) -> dict | None:
    """Load a checkpoint pkl file, return the dict or None on error."""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"    warn, could not load {pkl_path.name}: {e}")
        return None


def extract_max_scores(ckpt: dict) -> dict[tuple, int]:
    """Extract max score per test key across all clusters in a checkpoint."""
    max_scores = {}
    for island_state in ckpt.get("islands_state", []):
        for _sig, cluster in island_state.get("clusters", {}).items():
            scores_per_test = cluster.get("scores_per_test", {})
            for key, score in scores_per_test.items():
                parsed = parse_test_key(key)
                if parsed is not None:
                    if parsed not in max_scores or score > max_scores[parsed]:
                        max_scores[parsed] = score
    return max_scores


def scan_experiment(experiment_path: Path) -> dict[tuple, int]:
    """Scan all checkpoints in an experiment folder, return global max per test key.

    Only scans direct child run folders (checkpoint_run_*), not nested
    subdirectories like with_reasoning/. Use the full path explicitly
    (e.g. qwen14B/code/with_reasoning) to scan reasoning runs.
    """
    max_scores = {}

    # Only look at direct child directories that are run folders.
    run_dirs = sorted(
        d for d in experiment_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_run_")
    )

    if not run_dirs:
        print(f"  no run folders found in {experiment_path}")
        return max_scores

    # Collect pkl files per run.
    runs = {}
    total_pkls = 0
    for run_dir in run_dirs:
        pkls = sorted(run_dir.glob("*.pkl"))
        if pkls:
            runs[run_dir.name] = pkls
            total_pkls += len(pkls)

    if not runs:
        print(f"  no pkl files found in {len(run_dirs)} run folders")
        return max_scores

    print(f"  found {total_pkls} checkpoints across {len(runs)} runs")

    for run_dir, pkls in sorted(runs.items()):
        # Sort by filename (contains timestamp, so chronological order).
        pkls = sorted(pkls, key=lambda p: p.name)

        # First pass: load all checkpoints, track total_resets to find
        # which ones to scan. We need to scan clusters for:
        #   1. the checkpoint right before each reset count increase
        #      (clusters on weak islands are about to be wiped)
        #   2. the final checkpoint
        print(f"    {run_dir}: loading {len(pkls)} checkpoints, checking resets")
        loaded = []
        for pkl in pkls:
            ckpt = load_checkpoint(pkl)
            if ckpt is not None:
                loaded.append((pkl, ckpt))

        if not loaded:
            continue

        # Find checkpoints to scan based on reset transitions.
        scan_indices = set()
        scan_indices.add(len(loaded) - 1)  # always scan the last one

        for i in range(len(loaded) - 1):
            resets_now = loaded[i][1].get("total_resets", 0)
            resets_next = loaded[i + 1][1].get("total_resets", 0)
            if resets_next > resets_now:
                scan_indices.add(i)  # scan right before reset

        print(f"      {len(scan_indices)} checkpoints to scan "
              f"({len(loaded) - len(scan_indices)} skipped, no new resets)")

        for i in scan_indices:
            pkl_path, ckpt = loaded[i]
            resets = ckpt.get("total_resets", 0)
            ckpt_scores = extract_max_scores(ckpt)
            for key, score in ckpt_scores.items():
                if key not in max_scores or score > max_scores[key]:
                    max_scores[key] = score

    return max_scores


def print_results(results: dict[str, dict[tuple, int]]):
    """Print a formatted comparison table."""
    all_keys = set()
    for scores in results.values():
        all_keys.update(scores.keys())
    all_keys = sorted(all_keys)

    if not all_keys:
        print("\nno scores found.")
        return

    label_width = max(len(label) for label in results) + 2
    col_width = 8

    # Header.
    print(f"\n{'':>{label_width}}", end="")
    for key in all_keys:
        n = key[0]
        print(f"{'n=' + str(n):>{col_width}}", end="")
    print()

    # Optimal row.
    print(f"{'optimal':>{label_width}}", end="")
    for key in all_keys:
        opt = optimal_scores.get(key, "?")
        print(f"{opt:>{col_width}}", end="")
    print()

    print(f"{'':>{label_width}}" + "-" * (col_width * len(all_keys)))

    # Data rows.
    for label, scores in results.items():
        print(f"{label:>{label_width}}", end="")
        for key in all_keys:
            if key in scores:
                val = scores[key]
                opt = optimal_scores.get(key)
                marker = "*" if opt is not None and val >= opt else ""
                print(f"{str(val) + marker:>{col_width}}", end="")
            else:
                print(f"{'—':>{col_width}}", end="")
        print()

    print(f"\n  * = matches or exceeds known optimal")


def main():
    parser = argparse.ArgumentParser(
        description="Find max score per n across all checkpoints for experiments."
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment paths relative to base path, e.g. qwen14B/code starcoder/code",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=default_base_path,
        help=f"Base checkpoint directory (default: {default_base_path})",
    )
    args = parser.parse_args()

    base = Path(args.base_path)
    results = {}

    for experiment in args.experiments:
        experiment_path = base / experiment
        print(f"\n{'=' * 60}")
        print(f"experiment: {experiment}")
        print(f"path: {experiment_path}")
        print("=" * 60)

        if not experiment_path.exists():
            print(f"  error, path does not exist.")
            continue

        max_scores = scan_experiment(experiment_path)
        results[experiment] = max_scores

        # Print per experiment summary.
        for key in sorted(max_scores.keys()):
            n, s = key[0], key[1]
            opt = optimal_scores.get(key)
            marker = " (optimal)" if opt is not None and max_scores[key] >= opt else ""
            print(f"  n={n}, s={s}: {max_scores[key]}{marker}")

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("comparison table")
        print("=" * 60)

    print_results(results)


if __name__ == "__main__":
    main()
