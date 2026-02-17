#!/usr/bin/env python3
"""
Analyze description duplicates and overlap within a single run.

Computes:
- Exact duplicates (hash-based, fast)
- Near-duplicate distribution (sampled pairwise Jaccard)
- Top duplicate clusters

Usage:
    python duplicates.py checkpoint.pkl
    python duplicates.py ./checkpoints/           # Uses latest checkpoint
    python duplicates.py checkpoint.pkl --sample 2000
    python duplicates.py checkpoint.pkl --top 20
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_latest_checkpoint, load_checkpoint, extract_descriptions


def tokenize(text: str) -> Set[str]:
    """Simple word tokenization."""
    # Lowercase and split on non-alphanumeric
    import re
    words = re.findall(r'[a-z]+', text.lower())
    return set(words)


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def analyze_exact_duplicates(descriptions: List[str]) -> Tuple[Counter, int]:
    """Find exact duplicate descriptions."""
    counts = Counter(descriptions)
    num_unique = len(counts)
    return counts, num_unique


def analyze_near_duplicates(
    descriptions: List[str],
    sample_size: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Analyze near-duplicates using sampled pairwise Jaccard."""

    # Sample if needed
    if len(descriptions) > sample_size:
        random.seed(seed)
        sampled = random.sample(descriptions, sample_size)
    else:
        sampled = descriptions

    # Tokenize all sampled descriptions
    tokenized = [tokenize(d) for d in sampled]

    # Compute pairwise similarities
    similarities = []
    n = len(tokenized)
    total_pairs = n * (n - 1) // 2

    print(f"  Computing {total_pairs:,} pairwise similarities...", end=" ", flush=True)

    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_similarity(tokenized[i], tokenized[j])
            similarities.append(sim)

    print("done")

    # Bucket by similarity ranges
    buckets = {
        "100%": 0,
        "90-99%": 0,
        "80-89%": 0,
        "70-79%": 0,
        "60-69%": 0,
        "50-59%": 0,
        "40-49%": 0,
        "30-39%": 0,
        "20-29%": 0,
        "10-19%": 0,
        "0-9%": 0,
    }

    for sim in similarities:
        if sim == 1.0:
            buckets["100%"] += 1
        elif sim >= 0.9:
            buckets["90-99%"] += 1
        elif sim >= 0.8:
            buckets["80-89%"] += 1
        elif sim >= 0.7:
            buckets["70-79%"] += 1
        elif sim >= 0.6:
            buckets["60-69%"] += 1
        elif sim >= 0.5:
            buckets["50-59%"] += 1
        elif sim >= 0.4:
            buckets["40-49%"] += 1
        elif sim >= 0.3:
            buckets["30-39%"] += 1
        elif sim >= 0.2:
            buckets["20-29%"] += 1
        elif sim >= 0.1:
            buckets["10-19%"] += 1
        else:
            buckets["0-9%"] += 1

    # Convert to percentages
    bucket_pcts = {k: v / total_pairs * 100 for k, v in buckets.items()}

    return {
        "sample_size": len(sampled),
        "total_pairs": total_pairs,
        "buckets": buckets,
        "bucket_percentages": bucket_pcts,
        "similarities": similarities,
    }


def find_duplicate_clusters(
    descriptions: List[str],
    min_cluster_size: int = 5
) -> List[Tuple[str, int]]:
    """Group exact duplicates into clusters."""
    counts = Counter(descriptions)
    clusters = [(desc, count) for desc, count in counts.items() if count >= min_cluster_size]
    clusters.sort(key=lambda x: -x[1])
    return clusters


def main():
    parser = argparse.ArgumentParser(
        description='Analyze description duplicates within a run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('path', help='Checkpoint .pkl file or folder (uses latest)')
    parser.add_argument('--sample', '-s', type=int, default=1000,
                        help='Sample size for near-duplicate analysis (default: 1000)')
    parser.add_argument('--top', '-t', type=int, default=15,
                        help='Show top N duplicate clusters (default: 15)')
    parser.add_argument('--min-cluster', type=int, default=3,
                        help='Minimum cluster size to report (default: 3)')

    args = parser.parse_args()

    # Find checkpoint
    path = Path(args.path)
    checkpoint_path = find_latest_checkpoint(path)
    print(f"Loading: {checkpoint_path}")

    # Load and extract
    checkpoint = load_checkpoint(str(checkpoint_path))
    descriptions = extract_descriptions(checkpoint)
    print(f"Extracted {len(descriptions):,} descriptions")
    print()

    if not descriptions:
        print("No descriptions found")
        return 1

    # === Exact duplicates ===
    print("=" * 60)
    print("EXACT DUPLICATES")
    print("=" * 60)

    counts, num_unique = analyze_exact_duplicates(descriptions)
    num_total = len(descriptions)
    num_duplicated = num_total - num_unique
    dup_pct = num_duplicated / num_total * 100

    print(f"Total descriptions:  {num_total:,}")
    print(f"Unique descriptions: {num_unique:,}")
    print(f"Duplicates:          {num_duplicated:,} ({dup_pct:.1f}%)")
    print()

    # Duplicate distribution
    dup_counts = Counter(counts.values())
    print("Duplication distribution:")
    print(f"  {'Copies':<10} {'Descriptions':<15} {'Total instances':<15}")
    print(f"  {'-'*10} {'-'*15} {'-'*15}")
    for n_copies in sorted(dup_counts.keys()):
        n_descriptions = dup_counts[n_copies]
        total_instances = n_copies * n_descriptions
        print(f"  {n_copies:<10} {n_descriptions:<15,} {total_instances:<15,}")
    print()

    # === Top duplicate clusters ===
    print("=" * 60)
    print(f"TOP {args.top} DUPLICATE CLUSTERS")
    print("=" * 60)

    clusters = find_duplicate_clusters(descriptions, min_cluster_size=args.min_cluster)

    for i, (desc, count) in enumerate(clusters[:args.top], 1):
        print(f"{i:2}. [{count:,} copies] {desc}")

    if len(clusters) > args.top:
        print(f"    ... and {len(clusters) - args.top} more clusters")
    print()

    # === Near-duplicate analysis ===
    print("=" * 60)
    print("NEAR-DUPLICATE ANALYSIS (Jaccard similarity)")
    print("=" * 60)

    near_dup = analyze_near_duplicates(descriptions, sample_size=args.sample)

    print(f"Sample size: {near_dup['sample_size']:,} descriptions")
    print(f"Pairs analyzed: {near_dup['total_pairs']:,}")
    print()

    print("Similarity distribution:")
    print(f"  {'Range':<12} {'Pairs':<12} {'Percentage':<10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10}")

    for bucket, count in near_dup['buckets'].items():
        pct = near_dup['bucket_percentages'][bucket]
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<12} {count:<12,} {pct:>6.2f}%  {bar}")

    print()

    # Summary statistics
    sims = near_dup['similarities']
    avg_sim = sum(sims) / len(sims) if sims else 0
    high_overlap = sum(1 for s in sims if s >= 0.7) / len(sims) * 100 if sims else 0

    print(f"Average pairwise similarity: {avg_sim:.1%}")
    print(f"Pairs with ≥70% overlap: {high_overlap:.1f}%")

    return 0


if __name__ == "__main__":
    exit(main())
