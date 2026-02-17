#!/usr/bin/env python3
"""
Cluster descriptions from checkpoints using MinHash LSH + Louvain community detection.

Arguments:
    path                  Checkpoint .pkl file, folder containing checkpoints,
                          or parent directory when using batch mode.
    path_b                Second checkpoint path, required when using compare mode.
    --compare, -c         Compare mode, pool descriptions from two runs and analyze.
    --batch, -b           Batch mode, process all run folders in the given directory.
    --threshold, -t       LSH and similarity threshold, default 0.5.
    --num-perm            Number of MinHash permutations, default 128.
    --cv-threshold        Coefficient of variation threshold for choosing between
                          Jaccard and Overlap similarity, default 0.4.
    --top                 Number of top clusters to display, default 20.
    --output, -o          Output directory, default is auto generated with timestamp.

Pipeline:
    1. Tokenize descriptions, remove stop words using sklearn ENGLISH_STOP_WORDS.
    2. Analyze token length distribution, choose Jaccard or Overlap based on CV.
    3. Build MinHash signatures using datasketch with num_perm permutations.
    4. LSH banding to find candidate similar pairs via datasketch MinHashLSH.
    5. Compute actual similarity for candidate pairs.
    6. Build similarity graph using NetworkX.
    7. Louvain community detection via networkx.community.louvain_communities.
    8. TF-IDF across clusters, extract top 10 words per cluster as labels.

Usage:
    python descriptions.py checkpoint.pkl
    python descriptions.py ./checkpoints/
    python descriptions.py checkpoint.pkl --threshold 0.5
    python descriptions.py checkpoint.pkl --num-perm 128
    python descriptions.py checkpoint.pkl --cv-threshold 0.4
    python descriptions.py checkpoint.pkl --top 30
    python descriptions.py checkpoint.pkl --output ./results
    python descriptions.py /mnt/checkpoints --batch
    python descriptions.py /mnt/checkpoints --batch --threshold 0.4 --output ./batch_results
    python descriptions.py run_a.pkl run_b.pkl --compare
    python descriptions.py ./run_a ./run_b --compare --threshold 0.4

Dependencies:
    pip install datasketch networkx scikit-learn numpy
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Callable, Any

import numpy as np
import networkx as nx
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_latest_checkpoint, load_checkpoint, extract_descriptions


# Extended stop words, sklearn's plus domain specific
EXTRA_STOP_WORDS = {
    'function', 'priority', 'approach', 'method', 'based', 'using',
    'node', 'nodes', 'graph', 'code', 'codes', 'set', 'sets',
    'use', 'uses', 'used', 'new', 'like', 'way', 'make', 'makes',
    'help', 'helps', 'work', 'works', 'better', 'good', 'best',
    'heuristic', 'algorithm', 'value', 'values', 'number',
}
STOP_WORDS = ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS)


def tokenize(text: str) -> Set[str]:
    """Tokenize text into set of lowercase words, removing stop words."""
    words = re.findall(r'[a-z]{3,}', text.lower())
    return {w for w in words if w not in STOP_WORDS}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def overlap_coefficient(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute overlap coefficient: |A ∩ B| / min(|A|, |B|).

    Length-normalized: if A ⊆ B, overlap = 1.0.
    Better when descriptions have varying lengths.
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b))


def analyze_length_distribution(token_sets: List[Set[str]]) -> Tuple[float, float, float, str]:
    """
    Analyze token count distribution and decide similarity metric.

    Returns:
        mean, std, cv (coefficient of variation), chosen_metric name
    """
    lengths = [len(ts) for ts in token_sets]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)

    # Coefficient of variation (relative dispersion)
    cv = std_len / mean_len if mean_len > 0 else 0.0

    return mean_len, std_len, cv, lengths


def build_minhash_signatures(
    token_sets: List[Set[str]],
    num_perm: int = 128
) -> List[MinHash]:
    """Build MinHash signatures for each token set."""
    signatures = []
    for tokens in token_sets:
        mh = MinHash(num_perm=num_perm)
        for token in tokens:
            mh.update(token.encode('utf-8'))
        signatures.append(mh)
    return signatures


def find_candidate_pairs(
    signatures: List[MinHash],
    threshold: float = 0.5,
    num_perm: int = 128
) -> Set[Tuple[int, int]]:
    """Use LSH to find candidate similar pairs."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Insert all signatures
    for i, sig in enumerate(signatures):
        lsh.insert(str(i), sig)

    # Query each signature to find similar ones
    candidates = set()
    for i, sig in enumerate(signatures):
        results = lsh.query(sig)
        for r in results:
            j = int(r)
            if i < j:  # avoid duplicates and self pairs
                candidates.add((i, j))

    return candidates


def compute_similarities(
    candidates: Set[Tuple[int, int]],
    token_sets: List[Set[str]],
    similarity_func: Callable[[Set[str], Set[str]], float],
    threshold: float
) -> List[Tuple[int, int, float]]:
    """Compute actual similarity for candidate pairs, filter by threshold."""
    edges = []
    for i, j in candidates:
        sim = similarity_func(token_sets[i], token_sets[j])
        if sim >= threshold:
            edges.append((i, j, sim))
    return edges


def build_similarity_graph(
    num_nodes: int,
    edges: List[Tuple[int, int, float]]
) -> nx.Graph:
    """Build NetworkX graph from similarity edges."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i, j, sim in edges:
        G.add_edge(i, j, weight=sim)
    return G


def detect_communities(G: nx.Graph) -> List[Set[int]]:
    """Run Louvain community detection."""
    # louvain_communities returns a list of sets
    communities = nx.community.louvain_communities(G, weight='weight', seed=42)
    return list(communities)


def extract_cluster_labels(
    communities: List[Set[int]],
    descriptions: List[str],
    top_n: int = 10
) -> List[Tuple[Set[int], List[Tuple[str, float]], str]]:
    """
    Extract TF-IDF based labels for each cluster.

    Returns:
        List of (member_indices, top_words_with_scores, combined_text_preview)
    """
    if not communities:
        return []

    # Combine descriptions for each cluster
    cluster_texts = []
    for community in communities:
        combined = ' '.join(descriptions[i] for i in community)
        cluster_texts.append(combined)

    # TF-IDF across clusters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2),  # Include bigrams for better labels
        min_df=1,
        max_df=0.95
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        # Not enough vocabulary
        return [(c, [], "") for c in communities]

    # Extract top words for each cluster
    results = []
    for idx, community in enumerate(communities):
        # Get TF-IDF scores for this cluster
        scores = tfidf_matrix[idx].toarray().flatten()

        # Get top N indices
        top_indices = scores.argsort()[-top_n:][::-1]
        top_words = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]

        # Preview of combined text
        preview = cluster_texts[idx][:200] + "..." if len(cluster_texts[idx]) > 200 else cluster_texts[idx]

        results.append((community, top_words, preview))

    return results


def analyze_checkpoint(
    checkpoint_path: Path,
    threshold: float = 0.5,
    num_perm: int = 128,
    cv_threshold: float = 0.4,
    top_clusters: int = 20,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Main analysis pipeline for a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pkl file
        threshold: LSH/similarity threshold
        num_perm: Number of MinHash permutations
        cv_threshold: Coefficient of variation threshold for metric selection
        top_clusters: Number of top clusters to display
        output_dir: Directory to save outputs (optional)

    Returns:
        Dictionary with analysis results
    """
    print(f"Loading: {checkpoint_path}")
    checkpoint = load_checkpoint(str(checkpoint_path))

    descriptions = extract_descriptions(checkpoint)
    n = len(descriptions)
    print(f"Extracted {n:,} descriptions")

    if n < 2:
        print("Not enough descriptions for clustering")
        return {"error": "insufficient_descriptions", "count": n}

    # Step 1. Tokenize
    print()
    print("=" * 60)
    print("Tokenization")
    print("=" * 60)
    print(f"Tokenizing {n:,} descriptions (removing stop words)...", end=" ", flush=True)
    token_sets = [tokenize(d) for d in descriptions]
    print("done")

    # Step 2. Analyze length distribution
    print()
    print("=" * 60)
    print("Length analysis")
    print("=" * 60)

    mean_len, std_len, cv, lengths = analyze_length_distribution(token_sets)

    print(f"Token counts per description:")
    print(f"  Mean:   {mean_len:.1f}")
    print(f"  Std:    {std_len:.1f}")
    print(f"  CV:     {cv:.3f} (coefficient of variation)")
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print()

    # Choose similarity metric
    if cv < cv_threshold:
        similarity_func = jaccard_similarity
        metric_name = "Jaccard"
        print(f"CV ({cv:.3f}) < threshold ({cv_threshold}) → using Jaccard similarity")
        print("  (descriptions have similar lengths)")
    else:
        similarity_func = overlap_coefficient
        metric_name = "Overlap coefficient"
        print(f"CV ({cv:.3f}) ≥ threshold ({cv_threshold}) → using Overlap coefficient")
        print("  (descriptions have varying lengths, using length-normalized metric)")

    # Step 3. Build MinHash signatures
    print()
    print("=" * 60)
    print("MinHash signatures")
    print("=" * 60)
    print(f"Building MinHash signatures (num_perm={num_perm})...", end=" ", flush=True)
    signatures = build_minhash_signatures(token_sets, num_perm=num_perm)
    print("done")

    # Step 4. LSH banding
    print()
    print("=" * 60)
    print("LSH candidate pairs")
    print("=" * 60)
    print(f"Finding candidate pairs (threshold={threshold})...", end=" ", flush=True)
    candidates = find_candidate_pairs(signatures, threshold=threshold, num_perm=num_perm)
    print(f"{len(candidates):,} candidates")

    brute_force = n * (n - 1) // 2
    print(f"  (vs {brute_force:,} brute force comparisons, {len(candidates)/brute_force*100:.2f}% reduction)")

    # Step 5. Compute actual similarities
    print()
    print("=" * 60)
    print("Similarity computation")
    print("=" * 60)
    print(f"Computing {metric_name} for {len(candidates):,} pairs...", end=" ", flush=True)
    edges = compute_similarities(candidates, token_sets, similarity_func, threshold)
    print(f"{len(edges):,} pairs above threshold")

    # Step 6. Build similarity graph
    print()
    print("=" * 60)
    print("Similarity graph")
    print("=" * 60)
    G = build_similarity_graph(n, edges)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Graph statistics
    connected_components = list(nx.connected_components(G))
    num_isolated = sum(1 for c in connected_components if len(c) == 1)
    print(f"Connected components: {len(connected_components):,}")
    print(f"Isolated nodes: {num_isolated:,} ({num_isolated/n*100:.1f}%)")

    # Step 7. Louvain community detection
    print()
    print("=" * 60)
    print("Louvain community detection")
    print("=" * 60)
    print("Running Louvain algorithm...", end=" ", flush=True)
    communities = detect_communities(G)
    print(f"{len(communities):,} communities")

    # Sort by size
    communities.sort(key=lambda c: -len(c))

    # Statistics
    sizes = [len(c) for c in communities]
    singletons = sum(1 for s in sizes if s == 1)
    significant = sum(1 for s in sizes if s >= 5)

    print(f"  Singletons: {singletons:,} ({singletons/len(communities)*100:.1f}%)")
    print(f"  Significant (≥5): {significant:,}")
    print(f"  Largest: {sizes[0]:,} descriptions")

    # Step 8. TF-IDF cluster labels
    print()
    print("=" * 60)
    print("TF-IDF cluster labels")
    print("=" * 60)
    print("Extracting cluster labels via TF-IDF...", end=" ", flush=True)
    labeled_clusters = extract_cluster_labels(communities, descriptions, top_n=10)
    print("done")

    # Display results
    print()
    print("=" * 60)
    print(f"Top {min(top_clusters, len(labeled_clusters))} clusters")
    print("=" * 60)
    print()

    for rank, (members, top_words, _) in enumerate(labeled_clusters[:top_clusters], 1):
        size = len(members)
        pct = size / n * 100

        # Format top words
        if top_words:
            label = ", ".join(word for word, _ in top_words[:10])
            full_label = ", ".join(f"{word}({score:.2f})" for word, score in top_words[:10])
        else:
            label = "(no distinctive terms)"
            full_label = ""

        print(f"Cluster {rank}: {size:,} descriptions ({pct:.1f}%)")
        print(f"  Label: {label}")
        if full_label:
            print(f"  Top-10: {full_label}")

        # Show example description
        example_idx = min(members)
        example = descriptions[example_idx]
        print(f"  Example: {example}")
        print()

    if len(labeled_clusters) > top_clusters:
        remaining = len(labeled_clusters) - top_clusters
        remaining_descs = sum(len(c) for c, _, _ in labeled_clusters[top_clusters:])
        print(f"... and {remaining:,} more clusters ({remaining_descs:,} descriptions)")

    # Build results dict
    results = {
        "checkpoint": str(checkpoint_path),
        "generated": datetime.now().isoformat(),
        "total_descriptions": n,
        "parameters": {
            "threshold": threshold,
            "num_perm": num_perm,
            "cv_threshold": cv_threshold,
        },
        "length_analysis": {
            "mean": mean_len,
            "std": std_len,
            "cv": cv,
            "min": min(lengths),
            "max": max(lengths),
        },
        "similarity_metric": metric_name,
        "lsh": {
            "candidate_pairs": len(candidates),
            "brute_force_pairs": brute_force,
        },
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "connected_components": len(connected_components),
            "isolated_nodes": num_isolated,
        },
        "communities": {
            "total": len(communities),
            "singletons": singletons,
            "significant": significant,
            "largest": sizes[0] if sizes else 0,
        },
        "clusters": [
            {
                "rank": i + 1,
                "size": len(members),
                "percentage": len(members) / n * 100,
                "top_words": [{"word": w, "score": s} for w, s in top_words],
                "member_indices": list(members)[:100],  # limit for JSON size
            }
            for i, (members, top_words, _) in enumerate(labeled_clusters[:50])
        ],
    }

    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results JSON
        results_file = output_dir / "cluster_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"Saved: {results_file}")

        # Save cluster summaries
        summary_file = output_dir / "cluster_summaries.txt"
        with open(summary_file, 'w') as f:
            f.write(f"# Cluster Analysis: {checkpoint_path.name}\n")
            f.write(f"# Generated: {results['generated']}\n")
            f.write(f"# Metric: {metric_name} (CV={cv:.3f})\n")
            f.write(f"# Threshold: {threshold}\n")
            f.write(f"# Total: {n:,} descriptions → {len(communities):,} clusters\n")
            f.write("=" * 70 + "\n\n")

            for rank, (members, top_words, _) in enumerate(labeled_clusters, 1):
                size = len(members)
                pct = size / n * 100
                label = ", ".join(w for w, _ in top_words[:10]) if top_words else "(no label)"

                f.write(f"Cluster {rank}: {size:,} ({pct:.1f}%) - {label}\n")

                # Write example descriptions (up to 10)
                examples = [descriptions[i] for i in list(members)[:10]]
                for ex in examples:
                    f.write(f"  • {ex}\n")
                f.write("\n")

        print(f"Saved: {summary_file}")

        # Save all descriptions with cluster assignments
        assignments_file = output_dir / "cluster_assignments.json"
        assignments = {}
        for cluster_idx, (members, top_words, _) in enumerate(labeled_clusters):
            label = ", ".join(w for w, _ in top_words[:3]) if top_words else f"cluster_{cluster_idx}"
            for member_idx in members:
                assignments[member_idx] = {
                    "cluster_id": cluster_idx,
                    "cluster_label": label,
                    "description": descriptions[member_idx][:500],
                }

        with open(assignments_file, 'w') as f:
            json.dump(assignments, f, indent=2)
        print(f"Saved: {assignments_file}")

    return results


def find_run_folders(parent_dir: Path) -> List[Path]:
    """Find all subdirectories that contain checkpoint files."""
    run_folders = []
    for item in sorted(parent_dir.iterdir()):
        if item.is_dir():
            # Check if this folder contains any .pkl files
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                run_folders.append(item)
    return run_folders


def run_batch_analysis(
    parent_dir: Path,
    output_dir: Path,
    threshold: float,
    num_perm: int,
    cv_threshold: float,
    top_clusters: int,
) -> Dict[str, Any]:
    """Run analysis on all run folders in a parent directory."""
    run_folders = find_run_folders(parent_dir)

    if not run_folders:
        print(f"No run folders with checkpoints found in {parent_dir}")
        return {"error": "no_runs_found"}

    print(f"Found {len(run_folders)} run folders in {parent_dir}")
    print()

    all_results = {}
    successful = 0
    failed = 0

    for i, run_folder in enumerate(run_folders, 1):
        print()
        print("#" * 70)
        print(f"# Run {i}/{len(run_folders)}: {run_folder.name}")
        print("#" * 70)

        try:
            checkpoint_path = find_latest_checkpoint(run_folder)
            run_output_dir = output_dir / run_folder.name

            results = analyze_checkpoint(
                checkpoint_path,
                threshold=threshold,
                num_perm=num_perm,
                cv_threshold=cv_threshold,
                top_clusters=top_clusters,
                output_dir=run_output_dir,
            )

            all_results[run_folder.name] = {
                "status": "success",
                "checkpoint": str(checkpoint_path),
                "total_descriptions": results.get("total_descriptions", 0),
                "communities": results.get("communities", {}),
                "similarity_metric": results.get("similarity_metric", "unknown"),
            }
            successful += 1

        except Exception as e:
            print(f"Error processing {run_folder.name}: {e}")
            all_results[run_folder.name] = {
                "status": "error",
                "error": str(e),
            }
            failed += 1

    # Save batch summary
    summary_file = output_dir / "batch_summary.json"
    batch_summary = {
        "parent_dir": str(parent_dir),
        "generated": datetime.now().isoformat(),
        "total_runs": len(run_folders),
        "successful": successful,
        "failed": failed,
        "parameters": {
            "threshold": threshold,
            "num_perm": num_perm,
            "cv_threshold": cv_threshold,
        },
        "runs": all_results,
    }

    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)

    print()
    print("#" * 70)
    print("# Batch summary")
    print("#" * 70)
    print(f"Total runs:  {len(run_folders)}")
    print(f"Successful:  {successful}")
    print(f"Failed:      {failed}")
    print()
    print(f"Summary saved: {summary_file}")

    return batch_summary


# Size thresholds for cluster analysis
CLUSTER_SIZE_THRESHOLDS = [5, 10, 20, 50, 100]


def compare_runs(
    path_a: Path,
    path_b: Path,
    threshold: float = 0.5,
    num_perm: int = 128,
    cv_threshold: float = 0.4,
    top_clusters: int = 10,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Compare descriptions from two runs by pooling and clustering together.

    Args:
        path_a: Path to first checkpoint or folder
        path_b: Path to second checkpoint or folder
        threshold: LSH/similarity threshold
        num_perm: Number of MinHash permutations
        cv_threshold: Coefficient of variation threshold for metric selection
        top_clusters: Number of top clusters to display per category
        output_dir: Directory to save outputs (optional)

    Returns:
        Dictionary with comparison results
    """
    # Load both checkpoints
    print("=" * 70)
    print("Loading checkpoints")
    print("=" * 70)

    checkpoint_a = find_latest_checkpoint(path_a)
    checkpoint_b = find_latest_checkpoint(path_b)

    print(f"Run A: {checkpoint_a}")
    print(f"Run B: {checkpoint_b}")
    print()

    data_a = load_checkpoint(str(checkpoint_a))
    data_b = load_checkpoint(str(checkpoint_b))

    descriptions_a = extract_descriptions(data_a)
    descriptions_b = extract_descriptions(data_b)

    n_a = len(descriptions_a)
    n_b = len(descriptions_b)
    n_total = n_a + n_b

    print(f"Run A: {n_a:,} descriptions")
    print(f"Run B: {n_b:,} descriptions")
    print(f"Total: {n_total:,} descriptions")

    if n_total < 2:
        print("Not enough descriptions for comparison")
        return {"error": "insufficient_descriptions"}

    # Pool descriptions with source tags
    # sources[i] is A or B
    pooled_descriptions = descriptions_a + descriptions_b
    sources = ['A'] * n_a + ['B'] * n_b

    # Run clustering pipeline on pooled data
    print()
    print("=" * 70)
    print("Tokenization")
    print("=" * 70)
    print(f"Tokenizing {n_total:,} pooled descriptions...", end=" ", flush=True)
    token_sets = [tokenize(d) for d in pooled_descriptions]
    print("done")

    # Length analysis
    print()
    print("=" * 70)
    print("Length analysis")
    print("=" * 70)
    mean_len, std_len, cv, _ = analyze_length_distribution(token_sets)
    print(f"Token counts: mean={mean_len:.1f}, std={std_len:.1f}, CV={cv:.3f}")

    if cv < cv_threshold:
        similarity_func = jaccard_similarity
        metric_name = "Jaccard"
        print(f"Using Jaccard similarity (CV < {cv_threshold})")
    else:
        similarity_func = overlap_coefficient
        metric_name = "Overlap"
        print(f"Using Overlap coefficient (CV >= {cv_threshold})")

    # MinHash and LSH
    print()
    print("=" * 70)
    print("MinHash LSH")
    print("=" * 70)
    print(f"Building MinHash signatures (num_perm={num_perm})...", end=" ", flush=True)
    signatures = build_minhash_signatures(token_sets, num_perm=num_perm)
    print("done")

    print(f"Finding candidate pairs (threshold={threshold})...", end=" ", flush=True)
    candidates = find_candidate_pairs(signatures, threshold=threshold, num_perm=num_perm)
    print(f"{len(candidates):,} candidates")

    # Compute similarities
    print(f"Computing similarities...", end=" ", flush=True)
    edges = compute_similarities(candidates, token_sets, similarity_func, threshold)
    print(f"{len(edges):,} edges")

    # Build graph and detect communities
    print()
    print("=" * 70)
    print("Community detection")
    print("=" * 70)
    G = build_similarity_graph(n_total, edges)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("Running Louvain...", end=" ", flush=True)
    communities = detect_communities(G)
    communities.sort(key=lambda c: -len(c))
    print(f"{len(communities):,} communities")

    # Analyze cluster composition
    print()
    print("=" * 70)
    print("Cluster composition analysis")
    print("=" * 70)

    # For each cluster, count A and B members
    cluster_stats = []
    for community in communities:
        members = list(community)
        count_a = sum(1 for m in members if sources[m] == 'A')
        count_b = sum(1 for m in members if sources[m] == 'B')
        size = len(members)

        # Classify cluster
        if count_a > 0 and count_b > 0:
            cluster_type = "shared"
        elif count_a > 0:
            cluster_type = "A-only"
        else:
            cluster_type = "B-only"

        cluster_stats.append({
            "members": members,
            "size": size,
            "count_a": count_a,
            "count_b": count_b,
            "type": cluster_type,
            "ratio_a": count_a / size if size > 0 else 0,
            "ratio_b": count_b / size if size > 0 else 0,
        })

    # Count by type
    shared_clusters = [c for c in cluster_stats if c["type"] == "shared"]
    a_only_clusters = [c for c in cluster_stats if c["type"] == "A-only"]
    b_only_clusters = [c for c in cluster_stats if c["type"] == "B-only"]

    n_shared = len(shared_clusters)
    n_a_only = len(a_only_clusters)
    n_b_only = len(b_only_clusters)

    # Descriptions in shared clusters
    a_in_shared = sum(c["count_a"] for c in shared_clusters)
    b_in_shared = sum(c["count_b"] for c in shared_clusters)
    a_in_unique = sum(c["count_a"] for c in a_only_clusters)
    b_in_unique = sum(c["count_b"] for c in b_only_clusters)

    # Coverage
    coverage_a = a_in_shared / n_a * 100 if n_a > 0 else 0
    coverage_b = b_in_shared / n_b * 100 if n_b > 0 else 0

    print()
    print("Basic counts")
    print("." * 50)
    print(f"Total clusters:        {len(communities):,}")
    print(f"Shared clusters:       {n_shared:,} ({n_shared/len(communities)*100:.1f}%)")
    print(f"A only clusters:       {n_a_only:,}")
    print(f"B only clusters:       {n_b_only:,}")

    print()
    print("Coverage")
    print("." * 50)
    print(f"{'':25} {'Run A':>12} {'Run B':>12}")
    print(f"{'Total descriptions':25} {n_a:>12,} {n_b:>12,}")
    print(f"{'In shared clusters':25} {a_in_shared:>12,} {b_in_shared:>12,}")
    print(f"{'In unique clusters':25} {a_in_unique:>12,} {b_in_unique:>12,}")
    print(f"{'Coverage by other run':25} {coverage_a:>11.1f}% {coverage_b:>11.1f}%")

    # Cluster size distribution
    print()
    print("Cluster size distribution")
    print("." * 70)
    print(f"{'Threshold':20} {'Total':>10} {'Shared':>10} {'A only':>10} {'B only':>10}")
    print("." * 70)

    size_distribution = {}
    for thresh in CLUSTER_SIZE_THRESHOLDS:
        total_ge = sum(1 for c in cluster_stats if c["size"] >= thresh)
        shared_ge = sum(1 for c in shared_clusters if c["size"] >= thresh)
        a_only_ge = sum(1 for c in a_only_clusters if c["size"] >= thresh)
        b_only_ge = sum(1 for c in b_only_clusters if c["size"] >= thresh)

        print(f"{'Clusters >= ' + str(thresh):20} {total_ge:>10,} {shared_ge:>10,} {a_only_ge:>10,} {b_only_ge:>10,}")

        size_distribution[thresh] = {
            "total": total_ge,
            "shared": shared_ge,
            "a_only": a_only_ge,
            "b_only": b_only_ge,
        }

    # Extract labels for display
    print()
    print("=" * 70)
    print("Extracting cluster labels")
    print("=" * 70)
    print("Running TF-IDF...", end=" ", flush=True)
    labeled_clusters = extract_cluster_labels(communities, pooled_descriptions, top_n=5)
    print("done")

    # Build lookup from cluster index to label and top_words
    cluster_labels = {}
    for idx, (members, top_words, _) in enumerate(labeled_clusters):
        label = ", ".join(w for w, _ in top_words[:3]) if top_words else "(no label)"
        cluster_labels[idx] = {
            "label": label,
            "top_words": top_words,
        }

    # Display top shared clusters
    print()
    print("=" * 70)
    print(f"Top {min(top_clusters, len(shared_clusters))} shared clusters")
    print("=" * 70)
    print()

    for rank, stats in enumerate(shared_clusters[:top_clusters], 1):
        idx = cluster_stats.index(stats)
        label_info = cluster_labels.get(idx, {"label": "(no label)"})
        size = stats["size"]
        pct_a = stats["ratio_a"] * 100
        pct_b = stats["ratio_b"] * 100

        print(f"{rank}. Size {size:,} (A: {stats['count_a']:,} [{pct_a:.0f}%], B: {stats['count_b']:,} [{pct_b:.0f}%])")
        print(f"   Label: {label_info['label']}")

        # Show one example from each run
        a_examples = [m for m in stats["members"] if sources[m] == 'A']
        b_examples = [m for m in stats["members"] if sources[m] == 'B']

        if a_examples:
            ex = pooled_descriptions[a_examples[0]][:100]
            print(f"   A: {ex}...")
        if b_examples:
            ex = pooled_descriptions[b_examples[0]][:100]
            print(f"   B: {ex}...")
        print()

    # Display top A only clusters
    if a_only_clusters:
        print()
        print("=" * 70)
        print(f"Top {min(top_clusters, len(a_only_clusters))} A only clusters (unique to Run A)")
        print("=" * 70)
        print()

        for rank, stats in enumerate(a_only_clusters[:top_clusters], 1):
            idx = cluster_stats.index(stats)
            label_info = cluster_labels.get(idx, {"label": "(no label)"})
            size = stats["size"]

            print(f"{rank}. Size {size:,}")
            print(f"   Label: {label_info['label']}")

            ex = pooled_descriptions[stats["members"][0]][:120]
            print(f"   Example: {ex}...")
            print()

    # Display top B only clusters
    if b_only_clusters:
        print()
        print("=" * 70)
        print(f"Top {min(top_clusters, len(b_only_clusters))} B only clusters (unique to Run B)")
        print("=" * 70)
        print()

        for rank, stats in enumerate(b_only_clusters[:top_clusters], 1):
            idx = cluster_stats.index(stats)
            label_info = cluster_labels.get(idx, {"label": "(no label)"})
            size = stats["size"]

            print(f"{rank}. Size {size:,}")
            print(f"   Label: {label_info['label']}")

            ex = pooled_descriptions[stats["members"][0]][:120]
            print(f"   Example: {ex}...")
            print()

    # Build results dict
    results = {
        "generated": datetime.now().isoformat(),
        "run_a": {
            "path": str(checkpoint_a),
            "total_descriptions": n_a,
            "in_shared_clusters": a_in_shared,
            "in_unique_clusters": a_in_unique,
            "coverage": coverage_a,
        },
        "run_b": {
            "path": str(checkpoint_b),
            "total_descriptions": n_b,
            "in_shared_clusters": b_in_shared,
            "in_unique_clusters": b_in_unique,
            "coverage": coverage_b,
        },
        "parameters": {
            "threshold": threshold,
            "num_perm": num_perm,
            "cv_threshold": cv_threshold,
            "similarity_metric": metric_name,
        },
        "clusters": {
            "total": len(communities),
            "shared": n_shared,
            "a_only": n_a_only,
            "b_only": n_b_only,
        },
        "size_distribution": size_distribution,
        "top_shared_clusters": [
            {
                "rank": i + 1,
                "size": stats["size"],
                "count_a": stats["count_a"],
                "count_b": stats["count_b"],
                "label": cluster_labels.get(cluster_stats.index(stats), {}).get("label", ""),
            }
            for i, stats in enumerate(shared_clusters[:20])
        ],
        "top_a_only_clusters": [
            {
                "rank": i + 1,
                "size": stats["size"],
                "label": cluster_labels.get(cluster_stats.index(stats), {}).get("label", ""),
            }
            for i, stats in enumerate(a_only_clusters[:20])
        ],
        "top_b_only_clusters": [
            {
                "rank": i + 1,
                "size": stats["size"],
                "label": cluster_labels.get(cluster_stats.index(stats), {}).get("label", ""),
            }
            for i, stats in enumerate(b_only_clusters[:20])
        ],
    }

    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_file = output_dir / "comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"Saved: {results_file}")

        # Save human readable summary
        summary_file = output_dir / "comparison_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Run comparison summary\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Run A: {checkpoint_a.name}\n")
            f.write(f"Run B: {checkpoint_b.name}\n")
            f.write(f"Generated: {results['generated']}\n\n")

            f.write("Coverage\n")
            f.write("." * 50 + "\n")
            f.write(f"{'':25} {'Run A':>12} {'Run B':>12}\n")
            f.write(f"{'Total descriptions':25} {n_a:>12,} {n_b:>12,}\n")
            f.write(f"{'In shared clusters':25} {a_in_shared:>12,} {b_in_shared:>12,}\n")
            f.write(f"{'In unique clusters':25} {a_in_unique:>12,} {b_in_unique:>12,}\n")
            f.write(f"{'Coverage by other run':25} {coverage_a:>11.1f}% {coverage_b:>11.1f}%\n\n")

            f.write("Cluster size distribution\n")
            f.write("." * 70 + "\n")
            f.write(f"{'Threshold':20} {'Total':>10} {'Shared':>10} {'A only':>10} {'B only':>10}\n")
            for thresh in CLUSTER_SIZE_THRESHOLDS:
                d = size_distribution[thresh]
                f.write(f"{'Clusters >= ' + str(thresh):20} {d['total']:>10,} {d['shared']:>10,} {d['a_only']:>10,} {d['b_only']:>10,}\n")

            f.write("\n\nTop shared clusters\n")
            f.write("." * 50 + "\n")
            for i, stats in enumerate(shared_clusters[:20], 1):
                idx = cluster_stats.index(stats)
                label = cluster_labels.get(idx, {}).get("label", "")
                f.write(f"{i}. Size {stats['size']:,} (A:{stats['count_a']}, B:{stats['count_b']}). {label}\n")

            f.write("\n\nTop A only clusters\n")
            f.write("." * 50 + "\n")
            for i, stats in enumerate(a_only_clusters[:20], 1):
                idx = cluster_stats.index(stats)
                label = cluster_labels.get(idx, {}).get("label", "")
                f.write(f"{i}. Size {stats['size']:,}. {label}\n")

            f.write("\n\nTop B only clusters\n")
            f.write("." * 50 + "\n")
            for i, stats in enumerate(b_only_clusters[:20], 1):
                idx = cluster_stats.index(stats)
                label = cluster_labels.get(idx, {}).get("label", "")
                f.write(f"{i}. Size {stats['size']:,}. {label}\n")

        print(f"Saved: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Cluster checkpoint descriptions using MinHash LSH + Louvain',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single checkpoint
    python descriptions.py checkpoint.pkl

    # Single folder (uses latest checkpoint)
    python descriptions.py ./checkpoints/

    # Batch mode: process all run folders in a directory
    python descriptions.py /mnt/disfun/checkpoints/s2 --batch

    # Compare two runs
    python descriptions.py run_a.pkl run_b.pkl --compare
    python descriptions.py ./run_a ./run_b --compare --threshold 0.4

    # With custom threshold
    python descriptions.py checkpoint.pkl --threshold 0.6
    python descriptions.py /path/to/runs --batch --threshold 0.4
        """
    )

    parser.add_argument('path', help='Checkpoint .pkl, folder, or parent dir (with --batch)')
    parser.add_argument('path_b', nargs='?', default=None,
                        help='Second checkpoint path (required for --compare)')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare mode: pool descriptions from two runs and analyze')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Batch mode: process all run folders in the given directory')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='LSH/similarity threshold (default: 0.5)')
    parser.add_argument('--num-perm', type=int, default=128,
                        help='Number of MinHash permutations (default: 128)')
    parser.add_argument('--cv-threshold', type=float, default=0.4,
                        help='CV threshold for Jaccard vs Overlap selection (default: 0.4)')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top clusters to display (default: 20)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: auto-generated)')

    args = parser.parse_args()

    path = Path(args.path)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.compare:
            output_dir = Path(f"./comparison_{timestamp}")
        else:
            output_dir = Path(f"./description_clusters_{timestamp}")

    # Compare mode: compare two runs
    if args.compare:
        if not args.path_b:
            print("Error: --compare requires two paths")
            print("Usage: python descriptions.py path_a path_b --compare")
            return 1

        path_b = Path(args.path_b)

        compare_runs(
            path,
            path_b,
            threshold=args.threshold,
            num_perm=args.num_perm,
            cv_threshold=args.cv_threshold,
            top_clusters=args.top,
            output_dir=output_dir,
        )

        print()
        print("=" * 60)
        print("Comparison done")
        print("=" * 60)
        print(f"Output directory: {output_dir}")

        return 0

    # Batch mode: process all run folders
    if args.batch:
        if not path.is_dir():
            print(f"Error: batch mode requires a directory, got: {path}")
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)

        run_batch_analysis(
            path,
            output_dir,
            threshold=args.threshold,
            num_perm=args.num_perm,
            cv_threshold=args.cv_threshold,
            top_clusters=args.top,
        )

        print()
        print("=" * 60)
        print("Batch done")
        print("=" * 60)
        print(f"Output directory: {output_dir}")

        return 0

    # Single mode, process one checkpoint or folder
    checkpoint_path = find_latest_checkpoint(path)

    # Run analysis
    analyze_checkpoint(
        checkpoint_path,
        threshold=args.threshold,
        num_perm=args.num_perm,
        cv_threshold=args.cv_threshold,
        top_clusters=args.top,
        output_dir=output_dir,
    )

    print()
    print("=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
