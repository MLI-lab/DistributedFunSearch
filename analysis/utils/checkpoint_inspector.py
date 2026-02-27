"""
Checkpoint Inspector for DistributedFunSearch

Analyze a single checkpoint file in detail:
- View summary stats (prompts, programs, CPU/GPU time, etc.)
- View island summaries and details
- View cluster distributions
- Extract best programs
- Search for a signature within that checkpoint

Usage:
    python checkpoint_inspector.py <checkpoint_path> [options]

Arguments:
    checkpoint              Path to checkpoint .pkl file or directory (auto-picks latest)

Options:
    --summary-only, -S      Only show summary stats, skip islands/clusters/programs
    --island, -i ID         Show details for a specific island
    --export, -e FILE       Export best programs to a file
    --list-signatures, -l   List all unique signatures in checkpoint
    --search-signature, -s  Search for a specific signature, e.g. "{(7,2,2): 4, (8,2,2): 5}"
    --keys                  Show all top-level keys in the checkpoint
    --no-programs           Skip showing best programs

Examples:
    python checkpoint_inspector.py Checkpoints/
    python checkpoint_inspector.py checkpoint.pkl -S
    python checkpoint_inspector.py checkpoint.pkl --island 0
    python checkpoint_inspector.py checkpoint.pkl --export best_programs.txt
    python checkpoint_inspector.py checkpoint.pkl --search-signature '((7,2,2), 4), ((8,2,2), 5)'
    python checkpoint_inspector.py checkpoint.pkl --list-signatures
    python checkpoint_inspector.py checkpoint.pkl --keys
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    from .checkpoint import load_checkpoint, find_latest_checkpoint
except ImportError:
    from checkpoint import load_checkpoint, find_latest_checkpoint

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: 'tabulate' not installed. Using basic formatting.")
    print("Install with: pip install tabulate")




def format_table(data, headers, tablefmt='grid'):
    """Format table with tabulate or fallback to basic formatting."""
    if HAS_TABULATE:
        return tabulate(data, headers=headers, tablefmt=tablefmt)
    else:
        # Basic fallback formatting
        col_widths = [max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)
        lines = [header_line, separator]
        for row in data:
            lines.append(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
        return "\n".join(lines)


def print_checkpoint_summary(checkpoint: Dict[str, Any]):
    """Print high-level summary of checkpoint."""
    print("=" * 80)
    print("CHECKPOINT SUMMARY")
    print("=" * 80)
    print()

    # Iteration count
    iterations = checkpoint.get('iterations', checkpoint.get('total_prompts', 0))
    print("Progress:")
    print(f"  Iterations:                    {iterations:,}")
    print()

    # Resource usage
    print("Resource Usage:")
    print(f"  Cumulative Evaluator CPU Time: {checkpoint.get('cumulative_evaluator_cpu_time', 0):.2f} seconds")
    print(f"  Cumulative Sampler GPU Time:   {checkpoint.get('cumulative_sampler_gpu_time', 0):.2f} seconds")
    print(f"  Cumulative Input Tokens:       {checkpoint.get('cumulative_input_tokens', 0):,}")
    print(f"  Cumulative Output Tokens:      {checkpoint.get('cumulative_output_tokens', 0):,}")
    cumulative_cost = checkpoint.get('cumulative_cost', 0.0)
    if cumulative_cost:
        print(f"  Cumulative Cost:               ${cumulative_cost:.2f}")
    print()

    # Program statistics
    print("Program Statistics:")
    print(f"  Total Stored Programs:         {checkpoint.get('total_stored_programs', 0):,}")
    print(f"  Execution Failed:              {checkpoint.get('execution_failed', 0):,}")
    print(f"  Version Mismatch Discarded:    {checkpoint.get('version_mismatch_discarded', 0):,}")
    print(f"  Duplicates Discarded:          {checkpoint.get('duplicates_discarded', 0):,}")
    total_resets = checkpoint.get('total_resets', 0)
    if total_resets:
        print(f"  Total Island Resets:           {total_resets:,}")
    print()

    # Prompt statistics
    duplicate_prompts = checkpoint.get('duplicate_prompts', checkpoint.get('dublicate_prompts', 0))
    parallel_prompts = checkpoint.get('parallel_prompts', 0)
    sequential_prompts = checkpoint.get('sequential_prompts', 0)
    print("Prompt Statistics:")
    print(f"  Duplicate Prompts:             {duplicate_prompts:,}")
    if iterations:
        print(f"  Duplicate Prompt Rate:         {duplicate_prompts / iterations * 100:.1f}%")
    if parallel_prompts or sequential_prompts:
        print(f"  Parallel Prompts:              {parallel_prompts:,}")
        print(f"  Sequential Prompts:            {sequential_prompts:,}")
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

    # W&B tracking info
    if checkpoint.get('wandb_run_id'):
        print("W&B Tracking:")
        print(f"  Run ID:                        {checkpoint.get('wandb_run_id')}")
        print(f"  Run Name:                      {checkpoint.get('wandb_run_name', 'N/A')}")
        print()

    # Final metrics (if computed)
    final_metrics = checkpoint.get('final_metrics')
    if final_metrics:
        print("Final Metrics (Gap to Optimal):")
        print(f"  Discovered Optimal:            {final_metrics.get('final/discovered_optimal', 'N/A')}")
        best_gap = final_metrics.get('final/best_gap_to_optimal')
        if best_gap is not None:
            print(f"  Best Gap to Optimal:           {best_gap:.4f} ({best_gap*100:.2f}%)")
        gap_99 = final_metrics.get('final/gap_at_99th_percentile')
        if gap_99 is not None:
            print(f"  Gap at 99th Percentile:        {gap_99:.4f} ({gap_99*100:.2f}%)")
        gap_95 = final_metrics.get('final/gap_at_95th_percentile')
        if gap_95 is not None:
            print(f"  Gap at 95th Percentile:        {gap_95:.4f} ({gap_95*100:.2f}%)")
        top_20 = final_metrics.get('final/top_20_mean_gap')
        if top_20 is not None:
            print(f"  Top-20 Mean Gap:               {top_20:.4f} ({top_20*100:.2f}%)")
        top_50 = final_metrics.get('final/top_50_mean_gap')
        if top_50 is not None:
            print(f"  Top-50 Mean Gap:               {top_50:.4f} ({top_50*100:.2f}%)")
        print(f"  Total Unique Signatures:       {final_metrics.get('final/total_unique_signatures', 'N/A'):,}")
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
    print(format_table(table_data, headers))
    print()


def print_island_details(checkpoint: Dict[str, Any], island_id: int):
    """Print detailed information about a specific island."""
    print("=" * 80)
    print(f"ISLAND {island_id} DETAILS")
    print("=" * 80)
    print()

    if island_id >= len(checkpoint['islands_state']):
        print(f"Error: Island {island_id} does not exist (max: {len(checkpoint['islands_state']) - 1})")
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
        signature = eval(signature_str) if isinstance(signature_str, str) else signature_str
        num_programs = len(cluster_data['programs'])
        cluster_score = cluster_data['score']
        scores_per_test = cluster_data.get('scores_per_test', {})

        # Format scores_per_test nicely
        parsed_scores = {}
        for k, v in scores_per_test.items():
            key = eval(k) if isinstance(k, str) else k
            parsed_scores[key] = v
        scores_str = ", ".join(f"{k}: {v}" for k, v in sorted(parsed_scores.items()))

        cluster_table.append([
            str(signature)[:30] + "..." if len(str(signature)) > 30 else str(signature),
            f"{cluster_score:.4f}",
            num_programs,
            scores_str[:50] + "..." if len(scores_str) > 50 else scores_str
        ])

    # Sort by score descending
    cluster_table.sort(key=lambda x: float(x[1]), reverse=True)

    headers = ['Signature', 'Score', 'Programs', 'Scores per Test']
    print(format_table(cluster_table, headers))
    print()


def extract_best_programs(checkpoint: Dict[str, Any], output_file: Optional[str] = None):
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
        print("-" * 80)

        # Show thought/thinking if present
        thought = program_dict.get('thought')
        thinking = program_dict.get('thinking')
        if thought:
            print("Thought:")
            print(thought)
            print()
        if thinking:
            print("Thinking:")
            print(thinking)
            print()

        # Extract function body
        print("Function body:")
        body = program_dict.get('body', 'N/A')
        print(body)
        print()

        # Add to output
        output_lines.append(f"# Island {island_id}\n")
        output_lines.append(f"# Score: {best_score:.4f}\n")
        output_lines.append(f"# Scores per test: {best_scores_per_test}\n")
        if thought:
            output_lines.append(f"# Thought: {thought}\n")
        if thinking:
            output_lines.append(f"# Thinking: {thinking}\n")
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


def list_all_signatures(checkpoint: Dict[str, Any]):
    """List all unique signatures in the checkpoint."""
    print("=" * 80)
    print("ALL SIGNATURES IN CHECKPOINT")
    print("=" * 80)
    print()

    all_signatures = {}
    for island_id, island_state in enumerate(checkpoint['islands_state']):
        for cluster_data in island_state['clusters'].values():
            scores_per_test = cluster_data.get('scores_per_test', {})
            parsed_scores = {}
            for k, v in scores_per_test.items():
                key = eval(k) if isinstance(k, str) else k
                parsed_scores[key] = v

            sig_str = ", ".join(f"{k}: {v}" for k, v in sorted(parsed_scores.items()))
            if sig_str not in all_signatures:
                all_signatures[sig_str] = {
                    'score': cluster_data['score'],
                    'islands': [island_id],
                    'total_programs': len(cluster_data['programs'])
                }
            else:
                all_signatures[sig_str]['islands'].append(island_id)
                all_signatures[sig_str]['total_programs'] += len(cluster_data['programs'])

    print(f"Found {len(all_signatures)} unique signatures:\n")

    # Sort by score descending
    sorted_sigs = sorted(all_signatures.items(), key=lambda x: x[1]['score'], reverse=True)

    for sig_str, info in sorted_sigs:
        print(f"  {sig_str}")
        print(f"    Score: {info['score']:.4f}, Programs: {info['total_programs']}, Islands: {info['islands']}")
        print()


def search_signature(checkpoint: Dict[str, Any], target_signature: Dict[tuple, int]):
    """
    Search for a specific signature (scores_per_test) within this checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary
        target_signature: Dictionary mapping (n, s, q) or similar tuples to scores.
                         Example: {(7, 2, 2): 4, (8, 2, 2): 5, (9, 2, 2): 8}

    Returns:
        List of matches with program details
    """
    print("=" * 80)
    print("SIGNATURE SEARCH")
    print("=" * 80)
    print()

    print(f"Searching for signature: {target_signature}")
    print()

    # Convert target to comparable format
    target_tuple = tuple(sorted(target_signature.items()))

    matches = []

    for island_id, island_state in enumerate(checkpoint['islands_state']):
        for signature_str, cluster_data in island_state['clusters'].items():
            scores_per_test = cluster_data.get('scores_per_test', {})

            # Parse scores_per_test keys (they may be strings)
            parsed_scores = {}
            for k, v in scores_per_test.items():
                key = eval(k) if isinstance(k, str) else k
                parsed_scores[key] = v

            # Check if this matches the target signature
            current_tuple = tuple(sorted(parsed_scores.items()))

            if current_tuple == target_tuple:
                programs = cluster_data.get('programs', [])
                for program_dict in programs:
                    matches.append({
                        'island_id': island_id,
                        'signature': parsed_scores,
                        'score': cluster_data['score'],
                        'program': program_dict,
                        'cluster_size': len(programs)
                    })

    if not matches:
        print("No programs found with the specified signature.")
        print()
        print("Use --list-signatures to see all available signatures in this checkpoint.")
        print()
        return []

    print(f"Found {len(matches)} program(s) with matching signature!")
    print()

    # Display the first match
    print("First Program with This Signature:")
    print("-" * 80)
    first_match = matches[0]
    print(f"  Island ID:      {first_match['island_id']}")
    print(f"  Cluster Score:  {first_match['score']:.4f}")
    print(f"  Cluster Size:   {first_match['cluster_size']} program(s)")
    print()

    program_dict = first_match['program']

    # Show thought/thinking if present
    thought = program_dict.get('thought')
    thinking = program_dict.get('thinking')
    if thought:
        print("Thought:")
        print(thought)
        print()
    if thinking:
        print("Thinking:")
        print(thinking)
        print()

    print("Function Body:")
    print("-" * 80)
    body = program_dict.get('body', 'N/A')
    print(body)
    print()

    # Show all matches summary
    if len(matches) > 1:
        print(f"All {len(matches)} Matches:")
        print("-" * 80)
        table_data = []
        for i, match in enumerate(matches):
            prog = match['program']
            body_preview = prog.get('body', '')[:50].replace('\n', ' ') + '...' if prog.get('body') else 'N/A'
            table_data.append([
                i + 1,
                match['island_id'],
                f"{match['score']:.4f}",
                body_preview
            ])

        headers = ['#', 'Island', 'Score', 'Body Preview']
        print(format_table(table_data, headers))
        print()

    return matches


def parse_signature_string(sig_str: str) -> Dict[tuple, int]:
    """
    Parse a signature string into a dictionary.

    Accepts formats like:
    - "((7,2,2), 4), ((8,2,2), 5)"
    - "{(7,2,2): 4, (8,2,2): 5}"
    - "(7,2,2):4, (8,2,2):5"
    """
    sig_str = sig_str.strip()

    # Try direct eval if it looks like a dict
    if sig_str.startswith('{'):
        try:
            return eval(sig_str)
        except:
            pass

    # Try parsing as tuple pairs
    try:
        # Handle format like "((7,2,2), 4), ((8,2,2), 5)"
        if sig_str.startswith('('):
            pairs = eval(f"[{sig_str}]")
            return {k: v for k, v in pairs}
    except:
        pass

    # Try parsing colon-separated format "(7,2,2):4, (8,2,2):5"
    try:
        result = {}
        parts = sig_str.split('),')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key_part, val_part = part.rsplit(':', 1)
                key_part = key_part.strip()
                if not key_part.endswith(')'):
                    key_part += ')'
                key = eval(key_part)
                val = int(val_part.strip())
                result[key] = val
        if result:
            return result
    except:
        pass

    raise ValueError(f"Could not parse signature string: {sig_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect a single DistributedFunSearch checkpoint file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s checkpoint.pkl
  %(prog)s checkpoint.pkl --island 0
  %(prog)s checkpoint.pkl --export best_programs.txt
  %(prog)s checkpoint.pkl --list-signatures
  %(prog)s checkpoint.pkl --search-signature "{(7,2,2): 4, (8,2,2): 5}"
        """
    )

    parser.add_argument('checkpoint', help='Path to checkpoint .pkl file')
    parser.add_argument('--island', '-i', type=int, help='Show details for specific island')
    parser.add_argument('--export', '-e', help='Export best programs to file')
    parser.add_argument('--list-signatures', '-l', action='store_true',
                        help='List all unique signatures in checkpoint')
    parser.add_argument('--search-signature', '-s',
                        help='Search for a specific signature (e.g., "{(7,2,2): 4, (8,2,2): 5}")')
    parser.add_argument('--keys', action='store_true',
                        help='Show all top-level keys in the checkpoint')
    parser.add_argument('--summary-only', '-S', action='store_true',
                        help='Only show summary stats, skip islands/clusters/programs')
    parser.add_argument('--no-programs', action='store_true',
                        help='Skip showing best programs')

    args = parser.parse_args()

    # Resolve checkpoint path (supports directories → finds latest)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path not found: {checkpoint_path}")
        return 1

    try:
        checkpoint_path = find_latest_checkpoint(checkpoint_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print()
    print("=" * 80)
    print("  DistributedFunSearch Checkpoint Inspector")
    print("=" * 80)
    print()
    print(f"Loading: {checkpoint_path}")
    print()

    checkpoint = load_checkpoint(str(checkpoint_path))

    # Show raw keys if requested
    if args.keys:
        print("Top-level checkpoint keys:")
        for key in sorted(checkpoint.keys()):
            val = checkpoint[key]
            if isinstance(val, (int, float, str, bool, type(None))):
                print(f"  {key}: {val}")
            elif isinstance(val, list):
                print(f"  {key}: list[{len(val)}]")
            elif isinstance(val, dict):
                print(f"  {key}: dict[{len(val)}]")
            else:
                print(f"  {key}: {type(val).__name__}")
        print()

    # Always show summary
    print_checkpoint_summary(checkpoint)

    if args.summary_only:
        return 0

    # Show island summary
    print_island_summary(checkpoint)

    # Cluster distribution
    analyze_cluster_distribution(checkpoint)

    # Island details if requested
    if args.island is not None:
        print_island_details(checkpoint, args.island)

    # List signatures if requested
    if args.list_signatures:
        list_all_signatures(checkpoint)

    # Search for signature if requested
    if args.search_signature:
        try:
            target_sig = parse_signature_string(args.search_signature)
            search_signature(checkpoint, target_sig)
        except ValueError as e:
            print(f"Error parsing signature: {e}")
            return 1

    # Show best programs (unless --no-programs or specific island was requested)
    if not args.no_programs and args.island is None:
        extract_best_programs(checkpoint, args.export)

    print("=" * 80)
    print("Inspection Complete")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
