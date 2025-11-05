"""
Plot evolution progress from multiple checkpoints.

This script loads a series of checkpoint files and creates visualizations
showing how the search evolved over time.

Usage:
    Edit the CONFIGURATION section in main() and run:
    python plot_evolution.py
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_checkpoints(checkpoint_paths: List[str]) -> List[Dict[str, Any]]:
    """Load multiple checkpoints and sort by filename (assumed to contain timestamps)."""
    checkpoints = []
    for path in checkpoint_paths:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
            checkpoint['_filename'] = Path(path).name
            checkpoints.append(checkpoint)

    # Sort by filename (which should contain timestamps)
    checkpoints.sort(key=lambda x: x['_filename'])
    return checkpoints


def plot_best_scores_over_time(checkpoints: List[Dict[str, Any]], output_dir: Path = None):
    """Plot best score evolution for each island."""
    num_islands = len(checkpoints[0]['best_score_per_island'])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract data
    checkpoint_indices = list(range(len(checkpoints)))

    for island_id in range(num_islands):
        scores = [ckpt['best_score_per_island'][island_id] for ckpt in checkpoints]
        ax.plot(checkpoint_indices, scores, marker='o', label=f'Island {island_id}', alpha=0.7)

    # Overall best
    overall_best = [max(ckpt['best_score_per_island']) for ckpt in checkpoints]
    ax.plot(checkpoint_indices, overall_best, 'k-', linewidth=2, label='Overall Best', alpha=0.9)

    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Best Score')
    ax.set_title('Evolution of Best Scores per Island')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'best_scores_evolution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'best_scores_evolution.png'}")
    else:
        plt.show()

    plt.close()


def plot_cluster_statistics(checkpoints: List[Dict[str, Any]], output_dir: Path = None):
    """Plot cluster count and size evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    checkpoint_indices = list(range(len(checkpoints)))

    # 1. Number of clusters per island
    ax = axes[0, 0]
    num_islands = len(checkpoints[0]['islands_state'])
    for island_id in range(num_islands):
        cluster_counts = [len(ckpt['islands_state'][island_id]['clusters']) for ckpt in checkpoints]
        ax.plot(checkpoint_indices, cluster_counts, marker='o', label=f'Island {island_id}', alpha=0.6)
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Clusters per Island')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Total programs per island
    ax = axes[0, 1]
    for island_id in range(num_islands):
        program_counts = [ckpt['islands_state'][island_id]['num_programs'] for ckpt in checkpoints]
        ax.plot(checkpoint_indices, program_counts, marker='o', label=f'Island {island_id}', alpha=0.6)
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Number of Programs')
    ax.set_title('Programs per Island')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Average cluster size
    ax = axes[1, 0]
    avg_cluster_sizes_per_island = []
    for island_id in range(num_islands):
        sizes_over_time = []
        for ckpt in checkpoints:
            clusters = ckpt['islands_state'][island_id]['clusters']
            if clusters:
                cluster_sizes = [len(cluster_data['programs']) for cluster_data in clusters.values()]
                sizes_over_time.append(np.mean(cluster_sizes))
            else:
                sizes_over_time.append(0)
        avg_cluster_sizes_per_island.append(sizes_over_time)
        ax.plot(checkpoint_indices, sizes_over_time, marker='o', label=f'Island {island_id}', alpha=0.6)
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Average Cluster Size')
    ax.set_title('Average Cluster Size per Island')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Total stored programs
    ax = axes[1, 1]
    total_programs = [ckpt.get('total_stored_programs', 0) for ckpt in checkpoints]
    ax.plot(checkpoint_indices, total_programs, 'b-', marker='o', linewidth=2)
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Total Programs Stored')
    ax.set_title('Cumulative Programs Stored')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'cluster_statistics.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'cluster_statistics.png'}")
    else:
        plt.show()

    plt.close()


def plot_resource_usage(checkpoints: List[Dict[str, Any]], output_dir: Path = None):
    """Plot CPU and GPU time usage."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    checkpoint_indices = list(range(len(checkpoints)))

    # CPU time
    ax = axes[0]
    cpu_times = [ckpt['cumulative_evaluator_cpu_time'] / 3600 for ckpt in checkpoints]  # Convert to hours
    ax.plot(checkpoint_indices, cpu_times, 'b-', marker='o', linewidth=2, label='CPU Time')
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('CPU Time (hours)')
    ax.set_title('Cumulative Evaluator CPU Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # GPU time
    ax = axes[1]
    gpu_times = [ckpt['cumulative_sampler_gpu_time'] / 3600 for ckpt in checkpoints]  # Convert to hours
    ax.plot(checkpoint_indices, gpu_times, 'r-', marker='o', linewidth=2, label='GPU Time')
    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('GPU Time (hours)')
    ax.set_title('Cumulative Sampler GPU Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'resource_usage.png'}")
    else:
        plt.show()

    plt.close()


def plot_token_usage(checkpoints: List[Dict[str, Any]], output_dir: Path = None):
    """Plot token usage over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    checkpoint_indices = list(range(len(checkpoints)))

    input_tokens = [ckpt.get('cumulative_input_tokens', 0) / 1000 for ckpt in checkpoints]  # Convert to thousands
    output_tokens = [ckpt.get('cumulative_output_tokens', 0) / 1000 for ckpt in checkpoints]

    ax.plot(checkpoint_indices, input_tokens, 'g-', marker='o', linewidth=2, label='Input Tokens')
    ax.plot(checkpoint_indices, output_tokens, 'orange', marker='o', linewidth=2, label='Output Tokens')

    ax.set_xlabel('Checkpoint Index')
    ax.set_ylabel('Tokens (thousands)')
    ax.set_title('Cumulative Token Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'token_usage.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'token_usage.png'}")
    else:
        plt.show()

    plt.close()


def main():
    # ========================================================================
    # CONFIGURATION - Edit these variables
    # ========================================================================

    # Checkpoint file patterns (supports wildcards)
    checkpoint_patterns = [
        "Checkpoints/checkpoint_*.pkl",
        # "experiments/run1/Checkpoints/*.pkl",  # Add more patterns as needed
    ]

    # Output directory for plots (set to None to show interactive plots)
    output_dir = "analysis_plots"  # e.g., "plots/" or None

    # ========================================================================
    # END CONFIGURATION
    # ========================================================================

    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                    DeCoSearch Evolution Plotter                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()

    # Expand wildcards and load checkpoints
    checkpoint_paths = []
    for pattern in checkpoint_patterns:
        checkpoint_paths.extend(Path().glob(pattern))

    checkpoint_paths = [str(p) for p in checkpoint_paths if p.suffix == '.pkl']

    if not checkpoint_paths:
        print("Error: No checkpoint files found")
        print(f"Searched patterns: {checkpoint_patterns}")
        return

    print(f"Found {len(checkpoint_paths)} checkpoint files")
    print()

    checkpoints = load_checkpoints(checkpoint_paths)
    print(f"Loaded {len(checkpoints)} checkpoints")
    print()

    output_path = Path(output_dir) if output_dir else None

    # Generate plots
    print("Generating plots...")
    plot_best_scores_over_time(checkpoints, output_path)
    plot_cluster_statistics(checkpoints, output_path)
    plot_resource_usage(checkpoints, output_path)
    plot_token_usage(checkpoints, output_path)

    print()
    print("=" * 80)
    print("Plotting Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
