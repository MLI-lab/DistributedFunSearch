"""
Example script showing how to use analysis utilities programmatically.

This can be adapted into a Jupyter notebook for interactive analysis.
"""

import pickle
from pathlib import Path
import numpy as np


def load_checkpoint(checkpoint_path: str):
    """Load a checkpoint file."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def get_best_program_code(checkpoint_path: str, island_id: int = 0) -> str:
    """Extract best program code from a specific island."""
    checkpoint = load_checkpoint(checkpoint_path)
    program = checkpoint['best_program_per_island'][island_id]

    if program is None:
        return None

    return program.get('body', 'N/A')


def get_top_clusters(checkpoint_path: str, island_id: int = 0, top_k: int = 5):
    """Get top K clusters from an island sorted by score."""
    checkpoint = load_checkpoint(checkpoint_path)
    island_state = checkpoint['islands_state'][island_id]

    clusters = []
    for signature_str, cluster_data in island_state['clusters'].items():
        signature = eval(signature_str)
        clusters.append({
            'signature': signature,
            'score': cluster_data['score'],
            'num_programs': len(cluster_data['programs']),
            'scores_per_test': cluster_data.get('scores_per_test', {}),
            'programs': cluster_data['programs']
        })

    # Sort by score descending
    clusters.sort(key=lambda x: x['score'], reverse=True)

    return clusters[:top_k]


def compute_island_diversity(checkpoint_path: str, island_id: int = 0) -> dict:
    """Compute diversity metrics for an island."""
    checkpoint = load_checkpoint(checkpoint_path)
    island_state = checkpoint['islands_state'][island_id]

    cluster_scores = []
    cluster_sizes = []

    for cluster_data in island_state['clusters'].values():
        cluster_scores.append(cluster_data['score'])
        cluster_sizes.append(len(cluster_data['programs']))

    if not cluster_scores:
        return {
            'num_clusters': 0,
            'score_diversity': 0,
            'size_diversity': 0,
        }

    return {
        'num_clusters': len(cluster_scores),
        'score_mean': np.mean(cluster_scores),
        'score_std': np.std(cluster_scores),
        'score_diversity': np.std(cluster_scores) / (np.mean(cluster_scores) + 1e-10),
        'size_mean': np.mean(cluster_sizes),
        'size_std': np.std(cluster_sizes),
        'size_diversity': np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-10),
    }


def track_score_improvement(checkpoint_paths: list, island_id: int = 0):
    """Track score improvement across multiple checkpoints."""
    scores = []
    timestamps = []

    for path in sorted(checkpoint_paths):
        checkpoint = load_checkpoint(path)
        score = checkpoint['best_score_per_island'][island_id]
        scores.append(score)

        # Extract timestamp from filename if possible
        filename = Path(path).stem
        timestamps.append(filename)

    return timestamps, scores


def find_novel_programs(checkpoint1_path: str, checkpoint2_path: str, island_id: int = 0):
    """Find programs that exist in checkpoint2 but not in checkpoint1."""
    ckpt1 = load_checkpoint(checkpoint1_path)
    ckpt2 = load_checkpoint(checkpoint2_path)

    # Get hash values from checkpoint 1
    hash_values_1 = set()
    for cluster_data in ckpt1['islands_state'][island_id]['clusters'].values():
        for program in cluster_data['programs']:
            if 'hash_value' in program:
                hash_values_1.add(program['hash_value'])

    # Find novel programs in checkpoint 2
    novel_programs = []
    for cluster_data in ckpt2['islands_state'][island_id]['clusters'].values():
        for program in cluster_data['programs']:
            hash_val = program.get('hash_value')
            if hash_val and hash_val not in hash_values_1:
                novel_programs.append(program)

    return novel_programs


# Example usage
if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - Edit these variables
    # ========================================================================

    # Path to checkpoint file
    checkpoint_path = "Checkpoints/checkpoint_2025-01-15_10-30-00.pkl"

    # ========================================================================
    # END CONFIGURATION
    # ========================================================================

    print("=" * 80)
    print("Example Analysis")
    print("=" * 80)
    print()

    # Example 1: Get best program
    print("Example 1: Extract best program from Island 0")
    print("-" * 80)
    code = get_best_program_code(checkpoint_path, island_id=0)
    if code:
        print(code)
    else:
        print("No program found yet")
    print()

    # Example 2: Get top clusters
    print("Example 2: Top 3 clusters from Island 0")
    print("-" * 80)
    top_clusters = get_top_clusters(checkpoint_path, island_id=0, top_k=3)
    for i, cluster in enumerate(top_clusters, 1):
        print(f"Cluster {i}:")
        print(f"  Signature: {cluster['signature']}")
        print(f"  Score: {cluster['score']:.4f}")
        print(f"  Programs: {cluster['num_programs']}")
        print(f"  Scores per test: {cluster['scores_per_test']}")
        print()

    # Example 3: Compute diversity
    print("Example 3: Island 0 Diversity Metrics")
    print("-" * 80)
    diversity = compute_island_diversity(checkpoint_path, island_id=0)
    for key, value in diversity.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Example 4: Load and display basic stats
    print("Example 4: Basic Statistics")
    print("-" * 80)
    checkpoint = load_checkpoint(checkpoint_path)
    print(f"  Total Programs Stored: {checkpoint.get('total_stored_programs', 0):,}")
    print(f"  Total Prompts: {checkpoint.get('total_prompts', 0):,}")
    print(f"  CPU Time: {checkpoint['cumulative_evaluator_cpu_time'] / 3600:.2f} hours")
    print(f"  GPU Time: {checkpoint['cumulative_sampler_gpu_time'] / 3600:.2f} hours")
    print(f"  Overall Best Score: {max(checkpoint['best_score_per_island']):.4f}")
    print()

    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)
