"""
Checkpoint Searcher for DistributedFunSearch

Search across multiple checkpoint files in a directory to find:
- First occurrence of a specific signature
- Checkpoint closest to a target number of prompts (by-prompts)
- Checkpoint closest to a target number of stored programs (by-programs)
- When best score first exceeded a threshold
- Progress comparison across all checkpoints

Usage:
    python checkpoint_searcher.py <checkpoint_dir> <command> [options]

Examples:
    python checkpoint_searcher.py Checkpoints/ first-signature "{(7,2,2): 4, (8,2,2): 5}"
    python checkpoint_searcher.py Checkpoints/ by-prompts 50000
    python checkpoint_searcher.py Checkpoints/ by-programs 10000
    python checkpoint_searcher.py Checkpoints/ first-score-above 30.0
    python checkpoint_searcher.py Checkpoints/ compare
    python checkpoint_searcher.py Checkpoints/ list-signatures
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: 'tabulate' not installed. Using basic formatting.")


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint from pickle file."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def format_table(data, headers, tablefmt='grid'):
    """Format table with tabulate or fallback to basic formatting."""
    if HAS_TABULATE:
        return tabulate(data, headers=headers, tablefmt=tablefmt)
    else:
        col_widths = [max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)
        lines = [header_line, separator]
        for row in data:
            lines.append(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
        return "\n".join(lines)


def get_checkpoint_files(checkpoint_dir: str) -> List[Path]:
    """Get sorted list of checkpoint files in directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Directory not found: {checkpoint_dir}")

    checkpoint_files = sorted(checkpoint_path.glob("checkpoint_*.pkl"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    return checkpoint_files


def parse_signature_string(sig_str: str) -> Dict[tuple, int]:
    """
    Parse a signature string into a dictionary.

    Accepts formats like:
    - "((7,2,2), 4), ((8,2,2), 5)"
    - "{(7,2,2): 4, (8,2,2): 5}"
    - "(7,2,2):4, (8,2,2):5"
    - "(7, 2, 2):4 - (8, 2, 2):5"  (old emdash format)
    """
    import re

    sig_str = sig_str.strip()

    # Normalize old emdash format to comma format
    # "(7, 2, 2):4 - (8, 2, 2):5" -> "(7, 2, 2):4, (8, 2, 2):5"
    if ' - (' in sig_str:
        sig_str = sig_str.replace(' - ', ', ')

    # Try direct eval if it looks like a dict
    if sig_str.startswith('{'):
        try:
            return eval(sig_str)
        except:
            pass

    # Try parsing as tuple pairs
    try:
        if sig_str.startswith('('):
            pairs = eval(f"[{sig_str}]")
            return {k: v for k, v in pairs}
    except:
        pass

    # Try parsing colon-separated format "(7,2,2):4, (8,2,2):5" using regex
    # Pattern: (tuple):value
    try:
        result = {}
        # Match patterns like (7, 2, 2):4 or (7,2,2):14
        pattern = r'\(([^)]+)\)\s*:\s*(\d+)'
        matches = re.findall(pattern, sig_str)
        for tuple_content, value in matches:
            # Parse the tuple content
            key = eval(f"({tuple_content})")
            val = int(value)
            result[key] = val
        if result:
            return result
    except:
        pass

    raise ValueError(f"Could not parse signature string: {sig_str}")


def checkpoint_has_signature(checkpoint: Dict[str, Any], target_tuple: tuple) -> Tuple[bool, Optional[Dict]]:
    """Check if checkpoint contains a specific signature. Returns (found, match_info)."""
    for island_id, island_state in enumerate(checkpoint['islands_state']):
        for signature_str, cluster_data in island_state['clusters'].items():
            scores_per_test = cluster_data.get('scores_per_test', {})

            parsed_scores = {}
            for k, v in scores_per_test.items():
                key = eval(k) if isinstance(k, str) else k
                parsed_scores[key] = v

            current_tuple = tuple(sorted(parsed_scores.items()))

            if current_tuple == target_tuple:
                programs = cluster_data.get('programs', [])
                return True, {
                    'island_id': island_id,
                    'signature': parsed_scores,
                    'score': cluster_data['score'],
                    'programs': programs,
                    'cluster_size': len(programs)
                }

    return False, None


def find_first_signature(checkpoint_dir: str, target_signature: Dict[tuple, int]):
    """Find the first checkpoint where a signature appears."""
    print("=" * 80)
    print("FIND FIRST OCCURRENCE OF SIGNATURE")
    print("=" * 80)
    print()

    print(f"Target signature: {target_signature}")
    print(f"Directory: {checkpoint_dir}")
    print()

    checkpoint_files = get_checkpoint_files(checkpoint_dir)
    target_tuple = tuple(sorted(target_signature.items()))

    print(f"Scanning {len(checkpoint_files)} checkpoint files...")
    print("-" * 80)

    first_occurrence = None

    for i, ckpt_file in enumerate(checkpoint_files):
        try:
            checkpoint = load_checkpoint(str(ckpt_file))
        except Exception as e:
            print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: ERROR ({e})")
            continue

        found, match_info = checkpoint_has_signature(checkpoint, target_tuple)
        prompts = checkpoint.get('total_prompts', 0)
        programs = checkpoint.get('total_stored_programs', 0)

        if found:
            status = "FOUND"
            if first_occurrence is None:
                first_occurrence = {
                    'checkpoint_file': ckpt_file.name,
                    'checkpoint_path': str(ckpt_file),
                    'checkpoint_index': i + 1,
                    'total_checkpoints': len(checkpoint_files),
                    'total_prompts': prompts,
                    'total_stored_programs': programs,
                    'execution_failed': checkpoint.get('execution_failed', 0),
                    'duplicates_discarded': checkpoint.get('duplicates_discarded', 0),
                    'cpu_time': checkpoint.get('cumulative_evaluator_cpu_time', 0),
                    'gpu_time': checkpoint.get('cumulative_sampler_gpu_time', 0),
                    **match_info
                }
        else:
            status = "-"

        print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: {status} (prompts: {prompts:,}, programs: {programs:,})")

    print()

    if first_occurrence is None:
        print("SIGNATURE NOT FOUND in any checkpoint!")
        print()
        print("Use 'list-signatures' command to see all available signatures.")
        return None

    print("=" * 80)
    print("FIRST OCCURRENCE FOUND!")
    print("=" * 80)
    print()
    print(f"Checkpoint:           {first_occurrence['checkpoint_file']}")
    print(f"                      (#{first_occurrence['checkpoint_index']} of {first_occurrence['total_checkpoints']})")
    print()
    print("Stats at First Discovery:")
    print(f"  Total Prompts:          {first_occurrence['total_prompts']:,}")
    print(f"  Total Stored Programs:  {first_occurrence['total_stored_programs']:,}")
    print(f"  Execution Failed:       {first_occurrence['execution_failed']:,}")
    print(f"  Duplicates Discarded:   {first_occurrence['duplicates_discarded']:,}")
    print(f"  CPU Time:               {first_occurrence['cpu_time']:.2f}s")
    print(f"  GPU Time:               {first_occurrence['gpu_time']:.2f}s")
    print()
    print(f"Island ID:            {first_occurrence['island_id']}")
    print(f"Cluster Score:        {first_occurrence['score']:.4f}")
    print(f"Programs in Cluster:  {first_occurrence['cluster_size']}")
    print()

    if first_occurrence['programs']:
        print("First Program with This Signature:")
        print("-" * 80)
        program_dict = first_occurrence['programs'][0]

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

        # Show docstring if present and different from body
        docstring = program_dict.get('docstring')
        if docstring:
            print("Docstring:")
            print(docstring)
            print()

        print("Function Body:")
        print("-" * 80)
        body = program_dict.get('body', 'N/A')
        print(body)
        print()

    return first_occurrence


def find_by_prompts(checkpoint_dir: str, target_prompts: int):
    """Find checkpoint closest to target number of prompts."""
    print("=" * 80)
    print("FIND CHECKPOINT BY PROMPT COUNT")
    print("=" * 80)
    print()

    print(f"Target prompts: {target_prompts:,}")
    print(f"Directory: {checkpoint_dir}")
    print()

    checkpoint_files = get_checkpoint_files(checkpoint_dir)

    print(f"Scanning {len(checkpoint_files)} checkpoint files...")
    print("-" * 80)

    best_match = None
    best_diff = float('inf')

    for i, ckpt_file in enumerate(checkpoint_files):
        try:
            checkpoint = load_checkpoint(str(ckpt_file))
        except Exception as e:
            print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: ERROR ({e})")
            continue

        prompts = checkpoint.get('total_prompts', 0)
        diff = abs(prompts - target_prompts)

        marker = ""
        if diff < best_diff:
            best_diff = diff
            best_match = {
                'checkpoint_file': ckpt_file.name,
                'checkpoint_path': str(ckpt_file),
                'checkpoint_index': i + 1,
                'total_checkpoints': len(checkpoint_files),
                'checkpoint': checkpoint
            }
            marker = " <-- closest so far"

        print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: {prompts:,} prompts (diff: {diff:,}){marker}")

    print()

    if best_match is None:
        print("No checkpoints found!")
        return None

    checkpoint = best_match['checkpoint']
    print("=" * 80)
    print("CLOSEST CHECKPOINT FOUND")
    print("=" * 80)
    print()
    print(f"Checkpoint:           {best_match['checkpoint_file']}")
    print(f"                      (#{best_match['checkpoint_index']} of {best_match['total_checkpoints']})")
    print()
    print(f"Target Prompts:       {target_prompts:,}")
    print(f"Actual Prompts:       {checkpoint.get('total_prompts', 0):,}")
    print(f"Difference:           {best_diff:,}")
    print()
    print("Checkpoint Stats:")
    print(f"  Total Stored Programs:  {checkpoint.get('total_stored_programs', 0):,}")
    print(f"  Best Score:             {max(checkpoint['best_score_per_island']):.4f}")
    print(f"  CPU Time:               {checkpoint.get('cumulative_evaluator_cpu_time', 0):.2f}s")
    print(f"  GPU Time:               {checkpoint.get('cumulative_sampler_gpu_time', 0):.2f}s")
    print()

    return best_match


def find_by_programs(checkpoint_dir: str, target_programs: int):
    """Find checkpoint closest to target number of stored programs."""
    print("=" * 80)
    print("FIND CHECKPOINT BY PROGRAM COUNT")
    print("=" * 80)
    print()

    print(f"Target programs: {target_programs:,}")
    print(f"Directory: {checkpoint_dir}")
    print()

    checkpoint_files = get_checkpoint_files(checkpoint_dir)

    print(f"Scanning {len(checkpoint_files)} checkpoint files...")
    print("-" * 80)

    best_match = None
    best_diff = float('inf')

    for i, ckpt_file in enumerate(checkpoint_files):
        try:
            checkpoint = load_checkpoint(str(ckpt_file))
        except Exception as e:
            print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: ERROR ({e})")
            continue

        programs = checkpoint.get('total_stored_programs', 0)
        diff = abs(programs - target_programs)

        marker = ""
        if diff < best_diff:
            best_diff = diff
            best_match = {
                'checkpoint_file': ckpt_file.name,
                'checkpoint_path': str(ckpt_file),
                'checkpoint_index': i + 1,
                'total_checkpoints': len(checkpoint_files),
                'checkpoint': checkpoint
            }
            marker = " <-- closest so far"

        print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: {programs:,} programs (diff: {diff:,}){marker}")

    print()

    if best_match is None:
        print("No checkpoints found!")
        return None

    checkpoint = best_match['checkpoint']
    print("=" * 80)
    print("CLOSEST CHECKPOINT FOUND")
    print("=" * 80)
    print()
    print(f"Checkpoint:           {best_match['checkpoint_file']}")
    print(f"                      (#{best_match['checkpoint_index']} of {best_match['total_checkpoints']})")
    print()
    print(f"Target Programs:      {target_programs:,}")
    print(f"Actual Programs:      {checkpoint.get('total_stored_programs', 0):,}")
    print(f"Difference:           {best_diff:,}")
    print()
    print("Checkpoint Stats:")
    print(f"  Total Prompts:          {checkpoint.get('total_prompts', 0):,}")
    print(f"  Best Score:             {max(checkpoint['best_score_per_island']):.4f}")
    print(f"  CPU Time:               {checkpoint.get('cumulative_evaluator_cpu_time', 0):.2f}s")
    print(f"  GPU Time:               {checkpoint.get('cumulative_sampler_gpu_time', 0):.2f}s")
    print()

    return best_match


def find_first_score_above(checkpoint_dir: str, target_score: float):
    """Find first checkpoint where best score exceeded a threshold."""
    print("=" * 80)
    print("FIND FIRST CHECKPOINT WITH SCORE ABOVE THRESHOLD")
    print("=" * 80)
    print()

    print(f"Target score: {target_score}")
    print(f"Directory: {checkpoint_dir}")
    print()

    checkpoint_files = get_checkpoint_files(checkpoint_dir)

    print(f"Scanning {len(checkpoint_files)} checkpoint files...")
    print("-" * 80)

    first_occurrence = None

    for i, ckpt_file in enumerate(checkpoint_files):
        try:
            checkpoint = load_checkpoint(str(ckpt_file))
        except Exception as e:
            print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: ERROR ({e})")
            continue

        best_score = max(checkpoint['best_score_per_island'])
        prompts = checkpoint.get('total_prompts', 0)

        if best_score >= target_score:
            status = f"FOUND (score: {best_score:.4f})"
            if first_occurrence is None:
                first_occurrence = {
                    'checkpoint_file': ckpt_file.name,
                    'checkpoint_path': str(ckpt_file),
                    'checkpoint_index': i + 1,
                    'total_checkpoints': len(checkpoint_files),
                    'checkpoint': checkpoint,
                    'best_score': best_score
                }
        else:
            status = f"score: {best_score:.4f}"

        print(f"  [{i+1}/{len(checkpoint_files)}] {ckpt_file.name}: {status} (prompts: {prompts:,})")

    print()

    if first_occurrence is None:
        print(f"No checkpoint found with score >= {target_score}")
        return None

    checkpoint = first_occurrence['checkpoint']
    print("=" * 80)
    print("FIRST CHECKPOINT WITH SCORE ABOVE THRESHOLD")
    print("=" * 80)
    print()
    print(f"Checkpoint:           {first_occurrence['checkpoint_file']}")
    print(f"                      (#{first_occurrence['checkpoint_index']} of {first_occurrence['total_checkpoints']})")
    print()
    print(f"Target Score:         {target_score}")
    print(f"Actual Best Score:    {first_occurrence['best_score']:.4f}")
    print()
    print("Checkpoint Stats:")
    print(f"  Total Prompts:          {checkpoint.get('total_prompts', 0):,}")
    print(f"  Total Stored Programs:  {checkpoint.get('total_stored_programs', 0):,}")
    print(f"  CPU Time:               {checkpoint.get('cumulative_evaluator_cpu_time', 0):.2f}s")
    print(f"  GPU Time:               {checkpoint.get('cumulative_sampler_gpu_time', 0):.2f}s")
    print()

    return first_occurrence


def compare_checkpoints(checkpoint_dir: str):
    """Compare all checkpoints showing progress over time."""
    print("=" * 80)
    print("CHECKPOINT PROGRESS COMPARISON")
    print("=" * 80)
    print()

    print(f"Directory: {checkpoint_dir}")
    print()

    checkpoint_files = get_checkpoint_files(checkpoint_dir)

    print(f"Loading {len(checkpoint_files)} checkpoint files...")
    print()

    table_data = []
    for i, ckpt_file in enumerate(checkpoint_files):
        try:
            checkpoint = load_checkpoint(str(ckpt_file))
            best_score = max(checkpoint['best_score_per_island'])
            total_programs = checkpoint.get('total_stored_programs', 0)
            total_prompts = checkpoint.get('total_prompts', 0)
            cpu_time = checkpoint.get('cumulative_evaluator_cpu_time', 0)
            gpu_time = checkpoint.get('cumulative_sampler_gpu_time', 0)
            num_clusters = sum(len(island['clusters']) for island in checkpoint['islands_state'])

            table_data.append([
                i + 1,
                ckpt_file.name[:30] + "..." if len(ckpt_file.name) > 30 else ckpt_file.name,
                f"{best_score:.4f}",
                f"{total_prompts:,}",
                f"{total_programs:,}",
                num_clusters,
                f"{cpu_time:.0f}",
                f"{gpu_time:.0f}"
            ])
        except Exception as e:
            table_data.append([
                i + 1,
                ckpt_file.name[:30],
                "ERROR",
                "-",
                "-",
                "-",
                "-",
                "-"
            ])

    headers = ['#', 'Checkpoint', 'Best Score', 'Prompts', 'Programs', 'Clusters', 'CPU(s)', 'GPU(s)']
    print(format_table(table_data, headers))
    print()

    # Summary statistics
    if table_data:
        valid_scores = [float(row[2]) for row in table_data if row[2] != "ERROR"]
        if valid_scores:
            print("Summary:")
            print(f"  Initial Score:  {valid_scores[0]:.4f}")
            print(f"  Final Score:    {valid_scores[-1]:.4f}")
            print(f"  Max Score:      {max(valid_scores):.4f}")
            print()


def list_all_signatures(checkpoint_dir: str):
    """List all unique signatures found across all checkpoints."""
    print("=" * 80)
    print("ALL SIGNATURES ACROSS CHECKPOINTS")
    print("=" * 80)
    print()

    print(f"Directory: {checkpoint_dir}")
    print()

    checkpoint_files = get_checkpoint_files(checkpoint_dir)

    print(f"Scanning {len(checkpoint_files)} checkpoint files...")
    print()

    all_signatures = {}

    for i, ckpt_file in enumerate(checkpoint_files):
        try:
            checkpoint = load_checkpoint(str(ckpt_file))
        except Exception as e:
            continue

        for island_state in checkpoint['islands_state']:
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
                        'first_seen_checkpoint': ckpt_file.name,
                        'first_seen_index': i + 1
                    }

    print(f"Found {len(all_signatures)} unique signatures:\n")
    print("-" * 80)

    # Sort by score descending
    sorted_sigs = sorted(all_signatures.items(), key=lambda x: x[1]['score'], reverse=True)

    for sig_str, info in sorted_sigs:
        print(f"  {sig_str}")
        print(f"    Score: {info['score']:.4f}")
        print(f"    First seen: #{info['first_seen_index']} ({info['first_seen_checkpoint']})")
        print()

    return all_signatures


def main():
    parser = argparse.ArgumentParser(
        description='Search across multiple DistributedFunSearch checkpoint files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  first-signature SIGNATURE   Find first checkpoint containing a signature
  by-prompts COUNT            Find checkpoint closest to prompt count
  by-programs COUNT           Find checkpoint closest to program count
  first-score-above SCORE     Find first checkpoint with score >= threshold
  compare                     Compare all checkpoints (progress table)
  list-signatures             List all unique signatures across checkpoints

Examples:
  %(prog)s Checkpoints/ first-signature "{(7,2,2): 4, (8,2,2): 5}"
  %(prog)s Checkpoints/ by-prompts 50000
  %(prog)s Checkpoints/ by-programs 10000
  %(prog)s Checkpoints/ first-score-above 30.0
  %(prog)s Checkpoints/ compare
  %(prog)s Checkpoints/ list-signatures
        """
    )

    parser.add_argument('checkpoint_dir', help='Directory containing checkpoint .pkl files')
    parser.add_argument('command', choices=[
        'first-signature', 'by-prompts', 'by-programs',
        'first-score-above', 'compare', 'list-signatures'
    ], help='Search command to execute')
    parser.add_argument('argument', nargs='?', help='Argument for the command (signature, count, or score)')

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("  DistributedFunSearch Checkpoint Searcher")
    print("=" * 80)
    print()

    try:
        if args.command == 'first-signature':
            if not args.argument:
                print("Error: first-signature requires a signature argument")
                print('Example: first-signature "{(7,2,2): 4, (8,2,2): 5}"')
                return 1
            target_sig = parse_signature_string(args.argument)
            find_first_signature(args.checkpoint_dir, target_sig)

        elif args.command == 'by-prompts':
            if not args.argument:
                print("Error: by-prompts requires a count argument")
                return 1
            target_prompts = int(args.argument)
            find_by_prompts(args.checkpoint_dir, target_prompts)

        elif args.command == 'by-programs':
            if not args.argument:
                print("Error: by-programs requires a count argument")
                return 1
            target_programs = int(args.argument)
            find_by_programs(args.checkpoint_dir, target_programs)

        elif args.command == 'first-score-above':
            if not args.argument:
                print("Error: first-score-above requires a score argument")
                return 1
            target_score = float(args.argument)
            find_first_score_above(args.checkpoint_dir, target_score)

        elif args.command == 'compare':
            compare_checkpoints(args.checkpoint_dir)

        elif args.command == 'list-signatures':
            list_all_signatures(args.checkpoint_dir)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("=" * 80)
    print("Search Complete")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
