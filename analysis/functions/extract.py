#!/usr/bin/env python3
"""
Extract and deduplicate functions from checkpoint(s).

Three modes:
  exact:     Functions matching an exact score signature
  threshold: Functions where score at a given n >= some value
  top-gap:   Top-K functions by normalized gap to best known solutions

Deduplication uses the hash_value stored in each checkpoint (the semantic hash
computed during evaluation), so no re-execution is needed.

Usage:
    # Exact match (s=1, default target is VT-optimal)
    python functions/extract.py /path/to/checkpoint.pkl

    # Threshold: all functions with n=12 score >= 36
    python functions/extract.py /mnt/disfun/checkpoints/s2/ --latest-only \
        --mode threshold --threshold-n 12 --threshold-min 36 --s-value 2

    # Top-gap: best 50 by normalized gap to optimal
    python functions/extract.py /mnt/disfun/checkpoints/s2/ --latest-only \
        --mode top-gap --top-k 50 --s-value 2
"""

import ast
import argparse
import pickle
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import detect_signature


# ── Best known solutions ────────────────────────────────────────────────────

BEST_KNOWN = {
    1: {(6, 1): 10, (7, 1): 16, (8, 1): 30, (9, 1): 52, (10, 1): 94, (11, 1): 172},
    2: {(7, 2): 5, (8, 2): 7, (9, 2): 11, (10, 2): 16, (11, 2): 24, (12, 2): 37},
}

# Default target for exact mode (VT-optimal, s=1, q=2)
DEFAULT_TARGET = {
    (6, 1, 2): 10, (7, 1, 2): 16, (8, 1, 2): 30,
    (9, 1, 2): 52, (10, 1, 2): 94, (11, 1, 2): 172,
}


# ── Checkpoint loading ──────────────────────────────────────────────────────

def find_checkpoint_files(paths: List[str], latest_only: bool = False) -> List[Path]:
    """Find checkpoint .pkl files from a list of file/directory paths."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix == '.pkl':
            files.append(p)
        elif p.is_dir():
            found = list(p.glob("**/*.pkl"))
            if latest_only:
                # Group by parent dir, keep newest per dir
                by_parent = defaultdict(list)
                for f in found:
                    by_parent[f.parent].append(f)
                for parent, group in by_parent.items():
                    group.sort(key=lambda x: x.stat().st_mtime)
                    files.append(group[-1])
            else:
                files.extend(found)
    # Deduplicate, newest first
    files = list(set(files))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files


def parse_scores(raw: Dict) -> Dict[tuple, int]:
    """Parse scores_per_test to {(n, s): score}."""
    out = {}
    for k, v in raw.items():
        key = ast.literal_eval(k) if isinstance(k, str) else k
        if isinstance(key, tuple) and len(key) >= 2:
            out[(key[0], key[1])] = v
    return out


# ── Extraction ──────────────────────────────────────────────────────────────

def _record(prog: Dict, cluster: Dict, scores: Dict, island_id: int, source: str) -> Dict:
    """Build a flat function record from checkpoint data."""
    return {
        'body': prog.get('body', ''),
        'args': prog.get('args', ''),
        'name': prog.get('name', 'priority'),
        'hash_value': prog.get('hash_value'),
        'docstring': prog.get('docstring', ''),
        'description': prog.get('description') or prog.get('thought', ''),
        'scores_per_test': scores,
        'cluster_score': cluster.get('score', 0),
        'island_id': island_id,
        'source_checkpoint': source,
        'detected_signature': detect_signature(prog.get('body', ''), prog.get('args', '')),
    }


def extract_exact(checkpoint: Dict, target: Dict, source: str) -> List[Dict]:
    """Extract functions matching an exact score signature."""
    # Normalize target to (n, s) keys
    target_ns = {}
    for k, v in target.items():
        target_ns[(k[0], k[1])] = v
    target_tuple = tuple(sorted(target_ns.items()))

    results = []
    for island_id, island in enumerate(checkpoint.get('islands_state', [])):
        for _sig, cluster in island.get('clusters', {}).items():
            scores = parse_scores(cluster.get('scores_per_test', {}))
            if tuple(sorted(scores.items())) == target_tuple:
                for prog in cluster.get('programs', []):
                    if prog.get('body', '').strip():
                        results.append(_record(prog, cluster, scores, island_id, source))
    return results


def extract_threshold(checkpoint: Dict, n: int, min_score: int,
                      s: int, source: str) -> List[Dict]:
    """Extract functions where score at (n, s) >= min_score."""
    results = []
    for island_id, island in enumerate(checkpoint.get('islands_state', [])):
        for _sig, cluster in island.get('clusters', {}).items():
            scores = parse_scores(cluster.get('scores_per_test', {}))
            if scores.get((n, s), 0) >= min_score:
                for prog in cluster.get('programs', []):
                    if prog.get('body', '').strip():
                        results.append(_record(prog, cluster, scores, island_id, source))
    return results


def extract_all(checkpoint: Dict, source: str) -> List[Dict]:
    """Extract every function (for top-gap ranking later)."""
    results = []
    for island_id, island in enumerate(checkpoint.get('islands_state', [])):
        for _sig, cluster in island.get('clusters', {}).items():
            scores = parse_scores(cluster.get('scores_per_test', {}))
            for prog in cluster.get('programs', []):
                if prog.get('body', '').strip():
                    results.append(_record(prog, cluster, scores, island_id, source))
    return results


# ── Gap computation ─────────────────────────────────────────────────────────

def normalized_gap(scores: Dict[tuple, int], best: Dict[tuple, int]) -> float:
    """Average (optimal - achieved) / optimal.  0 = optimal, 1 = worst."""
    gaps = [(best[k] - scores.get(k, 0)) / best[k] for k in best]
    return sum(gaps) / len(gaps) if gaps else 1.0


# ── Deduplication ───────────────────────────────────────────────────────────

def deduplicate(functions: List[Dict]) -> List[Dict]:
    """Deduplicate by stored hash_value (falls back to body text)."""
    seen = {}
    for func in functions:
        key = func.get('hash_value') or func['body'].strip()
        if key not in seen:
            seen[key] = func
    return list(seen.values())


# ── Save ────────────────────────────────────────────────────────────────────

def save_functions(functions: List[Dict], output_dir: str, prefix: str = "successful"):
    """Save to JSON + Python + summary files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    safe = []
    for f in functions:
        fc = f.copy()
        fc['scores_per_test'] = {str(k): v for k, v in fc['scores_per_test'].items()}
        safe.append(fc)
    json_path = out / f"{prefix}_functions.json"
    with open(json_path, 'w') as fp:
        json.dump(safe, fp, indent=2, default=str)
    print(f"Saved {len(functions)} functions to {json_path}")

    # Python
    py_path = out / f"{prefix}_functions.py"
    with open(py_path, 'w') as fp:
        fp.write(f'"""Extracted priority functions ({prefix})."""\n\n')
        for i, func in enumerate(functions):
            fp.write(f"# Function {i+1}\n")
            fp.write(f"# Scores: {func['scores_per_test']}\n")
            if 'normalized_gap' in func:
                fp.write(f"# Gap: {func['normalized_gap']:.4f}\n")
            if func.get('description'):
                fp.write(f"# Description: {func['description'][:120]}\n")
            fp.write(f"# Source: {Path(func['source_checkpoint']).parent.name}\n")
            body = func['body']
            if not body.startswith('    '):
                body = '\n'.join('    ' + l if l.strip() else l for l in body.split('\n'))
            fp.write(f"def priority_{i+1}({func['args']}):\n{body}\n\n\n")
    print(f"Saved function bodies to {py_path}")

    # Summary
    summary_path = out / f"{prefix}_summary.txt"
    with open(summary_path, 'w') as fp:
        fp.write(f"Extraction Summary ({prefix})\n{'='*60}\n\n")
        fp.write(f"Total unique functions: {len(functions)}\n\n")
        by_source = defaultdict(int)
        for f in functions:
            by_source[Path(f['source_checkpoint']).parent.name] += 1
        fp.write("By source:\n")
        for src in sorted(by_source, key=lambda s: -by_source[s]):
            fp.write(f"  {by_source[src]:4d}  {src}\n")
    print(f"Saved summary to {summary_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract functions from checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('paths', nargs='+', help='Checkpoint file(s) or folder(s)')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--latest-only', action='store_true',
                        help='Only load latest checkpoint per folder')
    parser.add_argument('--verbose', '-v', action='store_true')

    parser.add_argument('--mode', choices=['exact', 'threshold', 'top-gap'], default='exact')
    parser.add_argument('--target', '-t', help='[exact] Target signature dict')
    parser.add_argument('--threshold-n', type=int, default=12, help='[threshold] n value')
    parser.add_argument('--threshold-min', type=int, default=36, help='[threshold] min score')
    parser.add_argument('--top-k', type=int, default=50, help='[top-gap] how many to keep')
    parser.add_argument('--s-value', type=int, default=1, help='s value (1 or 2)')

    args = parser.parse_args()

    # Output dir
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"./extract_{args.mode}_{ts}"

    print(f"Mode: {args.mode} | s={args.s_value} | Output: {args.output}\n")

    # Find checkpoints
    files = find_checkpoint_files(args.paths, latest_only=args.latest_only)
    if not files:
        print("No checkpoint files found!")
        return 1
    print(f"Found {len(files)} checkpoint(s)\n")

    # Extract
    all_funcs = []
    for i, path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {path.parent.name}/{path.name}...", end=" ", flush=True)
        try:
            cp = pickle.load(open(path, 'rb'))
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        if args.mode == 'exact':
            target = DEFAULT_TARGET if not args.target else eval(args.target)
            funcs = extract_exact(cp, target, str(path))
        elif args.mode == 'threshold':
            funcs = extract_threshold(cp, args.threshold_n, args.threshold_min,
                                      args.s_value, str(path))
        elif args.mode == 'top-gap':
            funcs = extract_all(cp, str(path))

        print(f"{len(funcs)} functions")
        all_funcs.extend(funcs)

    print(f"\nTotal extracted: {len(all_funcs)}")

    # Deduplicate by hash_value
    all_funcs = deduplicate(all_funcs)
    print(f"After dedup: {len(all_funcs)}")

    # For top-gap: compute gap, sort, take top-K
    if args.mode == 'top-gap':
        best = BEST_KNOWN.get(args.s_value, {})
        if not best:
            print(f"No best-known solutions for s={args.s_value}!")
            return 1
        for f in all_funcs:
            f['normalized_gap'] = normalized_gap(f['scores_per_test'], best)
        all_funcs.sort(key=lambda f: f['normalized_gap'])
        # Include ties at the cutoff
        if len(all_funcs) > args.top_k:
            cutoff_gap = all_funcs[args.top_k - 1]['normalized_gap']
            all_funcs = [f for f in all_funcs if f['normalized_gap'] <= cutoff_gap]
        print(f"Top {len(all_funcs)} by gap (best: {all_funcs[0]['normalized_gap']:.4f})")

    # Print compact table
    best = BEST_KNOWN.get(args.s_value, {})
    if best:
        ns_keys = sorted(best.keys())
        hdr = "  ".join(f"n{n}" for n, s in ns_keys)
        opt = "  ".join(f"/{best[k]:2d}" for k in ns_keys)
        print(f"\n{'#':>3} {'Gap':>6}  {hdr}  Source")
        print(f"{'':>3} {'':>6}  {opt}")
        print("-" * 70)
        for i, f in enumerate(all_funcs[:60], 1):
            sc = f['scores_per_test']
            vals = "  ".join(f"{sc.get(k, 0):3d}" for k in ns_keys)
            gap = f.get('normalized_gap', normalized_gap(sc, best))
            src = Path(f['source_checkpoint']).parent.name.replace('checkpoint_run_', '')
            print(f"{i:3d} {gap:6.4f}  {vals}  {src}")

    # Save
    prefix = {'exact': 'successful', 'threshold': f"n{args.threshold_n}_ge{args.threshold_min}",
              'top-gap': f"top{args.top_k}_gap"}[args.mode]
    save_functions(all_funcs, args.output, prefix)

    print(f"\nDone! {len(all_funcs)} functions saved to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
