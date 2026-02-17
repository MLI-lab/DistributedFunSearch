#!/usr/bin/env python3
"""
Discover strategy clusters from checkpoint descriptions.

Uses MinHash LSH for efficient similarity search, then clusters descriptions
by Jaccard similarity. Auto-generates labels from common terms in each cluster.

Usage:
    python clusters.py checkpoint.pkl
    python clusters.py ./checkpoints/                 # Uses latest checkpoint
    python clusters.py checkpoint.pkl --threshold 0.6 # Lower = bigger clusters
    python clusters.py checkpoint.pkl --top 30        # Show top 30 clusters

Algorithm:
    1. Tokenize descriptions into word sets
    2. Build MinHash signatures for each description
    3. Use LSH to find candidate similar pairs (avoids O(n²) comparison)
    4. Compute exact Jaccard for candidates, merge if ≥ threshold
    5. Extract common terms from each cluster as labels
"""

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_latest_checkpoint, load_checkpoint, extract_descriptions


# Stopwords to exclude from labels
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'only', 'same', 'than', 'too', 'very', 'just', 'also',
    'this', 'that', 'these', 'those', 'it', 'its', 'which', 'who',
    'what', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'any',
    'some', 'no', 'none', 'one', 'two', 'three', 'if', 'then', 'else',
    'more', 'most', 'less', 'least', 'much', 'many', 'few', 'little',
    'other', 'another', 'such', 'own', 'about', 'between', 'under', 'over',
    'new', 'using', 'based', 'use', 'uses', 'used',
    'heuristic', 'function', 'priority', 'approach', 'method',
    'nodes', 'node', 'graph', 'code', 'codes',
}


def tokenize(text: str) -> Set[str]:
    """Convert text to set of lowercase words."""
    words = re.findall(r'[a-z]+', text.lower())
    return set(words)


def tokenize_filtered(text: str) -> Set[str]:
    """Convert text to set of words, excluding stopwords."""
    words = re.findall(r'[a-z]+', text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


class MinHash:
    """MinHash signature generator for Jaccard similarity estimation."""

    def __init__(self, num_hashes: int = 128, seed: int = 42):
        self.num_hashes = num_hashes
        random.seed(seed)
        # Generate random hash parameters
        self.a = [random.randint(1, 2**31 - 1) for _ in range(num_hashes)]
        self.b = [random.randint(0, 2**31 - 1) for _ in range(num_hashes)]
        self.prime = 2**31 - 1  # Mersenne prime

    def signature(self, word_set: Set[str]) -> Tuple[int, ...]:
        """Generate MinHash signature for a set of words."""
        if not word_set:
            return tuple([self.prime] * self.num_hashes)

        sig = []
        for i in range(self.num_hashes):
            min_hash = float('inf')
            for word in word_set:
                h = hash(word)
                val = (self.a[i] * h + self.b[i]) % self.prime
                min_hash = min(min_hash, val)
            sig.append(min_hash)

        return tuple(sig)


class LSH:
    """Locality Sensitive Hashing for finding similar items."""

    def __init__(self, num_hashes: int = 128, num_bands: int = 16):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.buckets: List[Dict[Tuple, List[int]]] = [
            defaultdict(list) for _ in range(num_bands)
        ]

    def add(self, idx: int, signature: Tuple[int, ...]):
        """Add an item's signature to the LSH index."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            self.buckets[band_idx][band].append(idx)

    def get_candidates(self) -> Set[Tuple[int, int]]:
        """Get all candidate pairs that might be similar."""
        candidates = set()
        for band_buckets in self.buckets:
            for bucket in band_buckets.values():
                if len(bucket) > 1:
                    # All pairs in this bucket are candidates
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            pair = (min(bucket[i], bucket[j]), max(bucket[i], bucket[j]))
                            candidates.add(pair)
        return candidates


class UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        """Merge sets containing x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def get_clusters(self) -> Dict[int, List[int]]:
        """Get all clusters as {root: [members]}."""
        clusters = defaultdict(list)
        for i in range(len(self.parent)):
            clusters[self.find(i)].append(i)
        return dict(clusters)


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute exact Jaccard similarity."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def overlap_coefficient(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute overlap coefficient: |A ∩ B| / min(|A|, |B|).

    This is length-normalized: if A ⊆ B, overlap = 100%.
    Good for comparing descriptions of different lengths.
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def extract_cluster_words(descriptions: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract top words from cluster with their frequency (% of descriptions containing word)."""
    word_counts = Counter()
    for desc in descriptions:
        words = tokenize_filtered(desc)
        # Count each word once per description
        word_counts.update(words)

    # Return top words with percentage
    total = len(descriptions)
    result = []
    for word, count in word_counts.most_common(top_n):
        pct = count / total * 100
        result.append((word, pct))
    return result


def cluster_descriptions(
    descriptions: List[str],
    threshold: float = 0.50,
    num_hashes: int = 128,
    num_bands: int = 16,
    use_overlap: bool = True,
) -> Tuple[Dict[int, List[int]], int]:
    """
    Cluster descriptions by similarity.

    Args:
        use_overlap: If True, use overlap coefficient (length-normalized).
                     If False, use Jaccard similarity.

    Returns:
        clusters: Dict mapping cluster_id to list of description indices
        num_comparisons: Number of similarity comparisons made
    """
    n = len(descriptions)
    print(f"  Tokenizing {n:,} descriptions...", end=" ", flush=True)

    # Tokenize all descriptions
    token_sets = [tokenize(d) for d in descriptions]
    print("done")

    # Build MinHash signatures (still uses Jaccard approximation for LSH candidates)
    print(f"  Building MinHash signatures ({num_hashes} hashes)...", end=" ", flush=True)
    minhash = MinHash(num_hashes=num_hashes)
    signatures = [minhash.signature(ts) for ts in token_sets]
    print("done")

    # Build LSH index
    print(f"  Building LSH index ({num_bands} bands)...", end=" ", flush=True)
    lsh = LSH(num_hashes=num_hashes, num_bands=num_bands)
    for i, sig in enumerate(signatures):
        lsh.add(i, sig)
    print("done")

    # Get candidate pairs
    print("  Finding candidate pairs...", end=" ", flush=True)
    candidates = lsh.get_candidates()
    print(f"{len(candidates):,} candidates")

    # Choose similarity function
    sim_func = overlap_coefficient if use_overlap else jaccard_similarity
    sim_name = "overlap" if use_overlap else "Jaccard"

    # Compute exact similarity for candidates and cluster
    print(f"  Computing {sim_name} similarities (threshold={threshold})...", end=" ", flush=True)
    uf = UnionFind(n)
    num_comparisons = 0
    num_merges = 0

    for i, j in candidates:
        num_comparisons += 1
        sim = sim_func(token_sets[i], token_sets[j])
        if sim >= threshold:
            uf.union(i, j)
            num_merges += 1

    print(f"{num_merges:,} merges")

    # Extract clusters
    clusters = uf.get_clusters()

    return clusters, num_comparisons


def main():
    parser = argparse.ArgumentParser(
        description='Discover strategy clusters from descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('path', help='Checkpoint .pkl file or folder (uses latest)')
    parser.add_argument('--threshold', '-t', type=float, default=0.50,
                        help='Jaccard similarity threshold for clustering (default: 0.50)')
    parser.add_argument('--top', type=int, default=20,
                        help='Show top N clusters (default: 20)')
    parser.add_argument('--min-size', type=int, default=5,
                        help='Minimum cluster size to report (default: 5)')
    parser.add_argument('--num-hashes', type=int, default=128,
                        help='Number of MinHash signatures (default: 128)')
    parser.add_argument('--num-bands', type=int, default=16,
                        help='Number of LSH bands (default: 16)')
    parser.add_argument('--jaccard', action='store_true',
                        help='Use Jaccard similarity instead of overlap coefficient (default: overlap)')

    args = parser.parse_args()

    # Find and load checkpoint
    path = Path(args.path)
    checkpoint_path = find_latest_checkpoint(path)
    print(f"Loading: {checkpoint_path}")

    checkpoint = load_checkpoint(str(checkpoint_path))
    descriptions = extract_descriptions(checkpoint)
    print(f"Extracted {len(descriptions):,} descriptions")
    print()

    if not descriptions:
        print("No descriptions found")
        return 1

    # Cluster descriptions
    print("=" * 60)
    print("CLUSTERING")
    print("=" * 60)

    use_overlap = not args.jaccard
    clusters, num_comparisons = cluster_descriptions(
        descriptions,
        threshold=args.threshold,
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        use_overlap=use_overlap,
    )

    # Analyze clusters
    cluster_sizes = [(root, len(members)) for root, members in clusters.items()]
    cluster_sizes.sort(key=lambda x: -x[1])

    # Count by size
    singletons = sum(1 for _, size in cluster_sizes if size == 1)
    small_clusters = sum(1 for _, size in cluster_sizes if 2 <= size < args.min_size)
    significant_clusters = [(root, size) for root, size in cluster_sizes if size >= args.min_size]

    print()
    print("=" * 60)
    print("RESULTS")
    sim_name = "Jaccard" if args.jaccard else "Overlap coefficient"
    print("=" * 60)
    print(f"Total descriptions: {len(descriptions):,}")
    print(f"Similarity: {sim_name} ≥ {args.threshold:.0%}")
    print(f"Comparisons made: {num_comparisons:,} (vs {len(descriptions) * (len(descriptions)-1) // 2:,} brute force)")
    print()
    print(f"Unique (singletons): {singletons:,} ({singletons/len(descriptions)*100:.1f}%)")
    print(f"Small clusters (2-{args.min_size-1}): {small_clusters:,}")
    print(f"Significant clusters (≥{args.min_size}): {len(significant_clusters):,}")
    print()

    # Show top clusters
    print("=" * 60)
    print(f"TOP {min(args.top, len(significant_clusters))} STRATEGY CLUSTERS")
    print("=" * 60)
    print()

    for rank, (root, size) in enumerate(significant_clusters[:args.top], 1):
        members = clusters[root]
        member_descriptions = [descriptions[i] for i in members]

        # Extract top words with frequencies
        top_words = extract_cluster_words(member_descriptions, top_n=10)

        # Get a representative example (shortest one)
        example = min(member_descriptions, key=len)

        pct = size / len(descriptions) * 100
        print(f"Cluster {rank}: {size:,} descriptions ({pct:.2f}%)")
        print(f"  Top words: {', '.join(f'{w}({p:.0f}%)' for w, p in top_words)}")
        print(f"  Example: {example}")
        print()

    if len(significant_clusters) > args.top:
        remaining = len(significant_clusters) - args.top
        remaining_size = sum(size for _, size in significant_clusters[args.top:])
        print(f"... and {remaining} more clusters ({remaining_size:,} descriptions)")

    return 0


if __name__ == "__main__":
    exit(main())
