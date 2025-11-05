"""
Checkpoint Analysis Tool for DeCoSearch

This script loads and analyzes pickled checkpoint files to inspect:
- Island states and best programs
- Cluster distributions
- Resource usage statistics
- Evolution progress

Usage:
    Edit the CONFIGURATION section in main() and run:
    python checkpoint_analyzer.py
"""

import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np
from tabulate import tabulate


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint from pickle file."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def print_checkpoint_summary(checkpoint: Dict[str, Any]):
    """Print high-level summary of checkpoint."""
    print("=" * 80)
    print("CHECKPOINT SUMMARY")
    print("=" * 80)
    print()

    # Resource usage
    print("Resource Usage:")
    print(f"  Cumulative Evaluator CPU Time: {checkpoint['cumulative_evaluator_cpu_time']:.2f} seconds")
    print(f"  Cumulative Sampler GPU Time:   {checkpoint['cumulative_sampler_gpu_time']:.2f} seconds")
    print(f"  Cumulative Input Tokens:       {checkpoint.get('cumulative_input_tokens', 0):,}")
    print(f"  Cumulative Output Tokens:      {checkpoint.get('cumulative_output_tokens', 0):,}")
    print()

    # Program statistics
    print("Program Statistics:")
    print(f"  Total Stored Programs:         {checkpoint.get('total_stored_programs', 0):,}")
    print(f"  Execution Failed:              {checkpoint.get('execution_failed', 0):,}")
    print(f"  Version Mismatch Discarded:    {checkpoint.get('version_mismatch_discarded', 0):,}")
    print(f"  Duplicates Discarded:          {checkpoint.get('duplicates_discarded', 0):,}")
    print()

    # Prompt statistics
    print("Prompt Statistics:")
    print(f"  Total Prompts:                 {checkpoint.get('total_prompts', 0):,}")
    print(f"  Duplicate Prompts:             {checkpoint.get('dublicate_prompts', 0):,}")
    print()

    # Solution tracking
    if checkpoint.get('found_optimal_solution', False):
        print("Optimal Solution:")
        print(f"  Found:                         YES")
        print(f"  Prompts Since Optimal:         {checkpoint.get('prompts_since_optimal', 0):,}")
    else:
        print("Optimal Solution:")
        print(f"  Found:                         NO")
    print()


def print_island_summary(checkpoint: Dict[str, Any]):
    """Print summary of all islands."""
    print("=" * 80)
    print("ISLAND SUMMARY")
    print("=" * 80)
    print()

    num_islands = len(checkpoint['islands_state'])
    best_scores = checkpoint['best_score_per_island']

    table_data = []
    for island_id in range(num_islands):
        island_state = checkpoint['islands_state'][island_id]
        best_score = best_scores[island_id]
        num_clusters = len(island_state['clusters'])
        num_programs = island_state['num_programs']
        version = island_state['version']

        # Calculate cluster sizes
        cluster_sizes = []
        for cluster_data in island_state['clusters'].values():
            cluster_sizes.append(len(cluster_data['programs']))

        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        max_cluster_size = np.max(cluster_sizes) if cluster_sizes else 0
        min_cluster_size = np.min(cluster_sizes) if cluster_sizes else 0

        table_data.append([
            island_id,
            f"{best_score:.4f}",
            num_clusters,
            num_programs,
            f"{avg_cluster_size:.1f}",
            max_cluster_size,
            min_cluster_size,
            version
        ])

    headers = ['Island', 'Best Score', 'Clusters', 'Programs', 'Avg Size', 'Max Size', 'Min Size', 'Version']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()


def print_island_details(checkpoint: Dict[str, Any], island_id: int):
    """Print detailed information about a specific island."""
    print("=" * 80)
    print(f"ISLAND {island_id} DETAILS")
    print("=" * 80)
    print()

    if island_id >= len(checkpoint['islands_state']):
        print(f"Error: Island {island_id} does not exist")
        return

    island_state = checkpoint['islands_state'][island_id]
    best_score = checkpoint['best_score_per_island'][island_id]
    best_scores_per_test = checkpoint['best_scores_per_test_per_island'][island_id]

    print(f"Island ID:       {island_id}")
    print(f"Version:         {island_state['version']}")
    print(f"Total Programs:  {island_state['num_programs']}")
    print(f"Best Score:      {best_score:.4f}")
    print(f"Best Scores per Test: {best_scores_per_test}")
    print()

    print("Clusters:")
    print("-" * 80)

    cluster_table = []
    for signature_str, cluster_data in island_state['clusters'].items():
        signature = eval(signature_str)
        num_programs = len(cluster_data['programs'])
        cluster_score = cluster_data['score']
        scores_per_test = cluster_data.get('scores_per_test', {})

        cluster_table.append([
            str(signature),
            f"{cluster_score:.4f}",
            num_programs,
            str(scores_per_test)
        ])

    # Sort by score descending
    cluster_table.sort(key=lambda x: float(x[1]), reverse=True)

    headers = ['Signature', 'Score', 'Programs', 'Scores per Test']
    print(tabulate(cluster_table, headers=headers, tablefmt='grid'))
    print()


def extract_best_programs(checkpoint: Dict[str, Any], output_file: str = None):
    """Extract best program from each island."""
    print("=" * 80)
    print("BEST PROGRAMS")
    print("=" * 80)
    print()

    best_programs = checkpoint['best_program_per_island']

    output_lines = []

    for island_id, program_dict in enumerate(best_programs):
        if program_dict is None:
            print(f"Island {island_id}: No program yet")
            output_lines.append(f"# Island {island_id}: No program yet\n\n")
            continue

        best_score = checkpoint['best_score_per_island'][island_id]
        best_scores_per_test = checkpoint['best_scores_per_test_per_island'][island_id]

        print(f"Island {island_id}:")
        print(f"  Score: {best_score:.4f}")
        print(f"  Scores per test: {best_scores_per_test}")
        print(f"  Function body:")
        print("-" * 80)

        # Extract function body
        body = program_dict.get('body', 'N/A')
        print(body)
        print()

        # Add to output
        output_lines.append(f"# Island {island_id}\n")
        output_lines.append(f"# Score: {best_score:.4f}\n")
        output_lines.append(f"# Scores per test: {best_scores_per_test}\n")
        output_lines.append(f"\n{body}\n")
        output_lines.append("=" * 80 + "\n\n")

    if output_file:
        with open(output_file, 'w') as f:
            f.writelines(output_lines)
        print(f"Best programs exported to: {output_file}")
        print()


def analyze_cluster_distribution(checkpoint: Dict[str, Any]):
    """Analyze distribution of clusters across islands."""
    print("=" * 80)
    print("CLUSTER DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    all_cluster_sizes = []
    all_cluster_scores = []

    for island_state in checkpoint['islands_state']:
        for cluster_data in island_state['clusters'].values():
            cluster_size = len(cluster_data['programs'])
            cluster_score = cluster_data['score']
            all_cluster_sizes.append(cluster_size)
            all_cluster_scores.append(cluster_score)

    if not all_cluster_sizes:
        print("No clusters found in checkpoint")
        return

    print("Cluster Sizes:")
    print(f"  Total Clusters:     {len(all_cluster_sizes)}")
    print(f"  Mean:               {np.mean(all_cluster_sizes):.2f}")
    print(f"  Std Dev:            {np.std(all_cluster_sizes):.2f}")
    print(f"  Min:                {np.min(all_cluster_sizes)}")
    print(f"  Max:                {np.max(all_cluster_sizes)}")
    print(f"  Median:             {np.median(all_cluster_sizes):.2f}")
    print()

    print("Cluster Scores:")
    print(f"  Mean:               {np.mean(all_cluster_scores):.4f}")
    print(f"  Std Dev:            {np.std(all_cluster_scores):.4f}")
    print(f"  Min:                {np.min(all_cluster_scores):.4f}")
    print(f"  Max:                {np.max(all_cluster_scores):.4f}")
    print(f"  Median:             {np.median(all_cluster_scores):.4f}")
    print()


def print_progress_comparison(checkpoints: Dict[str, Dict[str, Any]]):
    """Compare multiple checkpoints to show progress over time."""
    print("=" * 80)
    print("PROGRESS COMPARISON")
    print("=" * 80)
    print()

    table_data = []
    for name, checkpoint in sorted(checkpoints.items()):
        best_score = max(checkpoint['best_score_per_island'])
        total_programs = checkpoint.get('total_stored_programs', 0)
        total_prompts = checkpoint.get('total_prompts', 0)
        cpu_time = checkpoint['cumulative_evaluator_cpu_time']
        gpu_time = checkpoint['cumulative_sampler_gpu_time']

        table_data.append([
            name,
            f"{best_score:.4f}",
            total_programs,
            total_prompts,
            f"{cpu_time:.1f}",
            f"{gpu_time:.1f}"
        ])

    headers = ['Checkpoint', 'Best Score', 'Programs', 'Prompts', 'CPU (s)', 'GPU (s)']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()


def main():
    # ========================================================================
    # CONFIGURATION - Edit these variables
    # ========================================================================

    # Path to checkpoint file(s)
    checkpoint_path = "Checkpoints/checkpoint_2025-01-15_10-30-00.pkl"

    # Optional: Show details for specific island (set to None to skip)
    island_id = None  # e.g., 0 for island 0

    # Optional: Export best programs to file (set to None to skip)
    export_file = None  # e.g., "best_programs.txt"

    # Optional: Compare multiple checkpoints (set to None for single checkpoint analysis)
    compare_checkpoints = None  # e.g., ["checkpoint_1.pkl", "checkpoint_2.pkl", "checkpoint_3.pkl"]

    # ========================================================================
    # END CONFIGURATION
    # ========================================================================

    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                    DeCoSearch Checkpoint Analyzer                          ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()

    if compare_checkpoints:
        # Compare multiple checkpoints
        checkpoints = {}
        for path in compare_checkpoints:
            name = Path(path).stem
            checkpoints[name] = load_checkpoint(path)
        print_progress_comparison(checkpoints)
    else:
        # Analyze single checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return

        print(f"Loading checkpoint: {checkpoint_path}")
        print()

        checkpoint = load_checkpoint(str(checkpoint_path))

        # Print summary
        print_checkpoint_summary(checkpoint)

        # Print island summary
        print_island_summary(checkpoint)

        # Cluster distribution analysis
        analyze_cluster_distribution(checkpoint)

        # Island details if specified
        if island_id is not None:
            print_island_details(checkpoint, island_id)

        # Extract best programs if requested
        if export_file or island_id is None:  # Always show if no specific island requested
            extract_best_programs(checkpoint, export_file)

    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
