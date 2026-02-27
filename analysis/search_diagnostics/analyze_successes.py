#!/usr/bin/env python3
"""Analyze successful debug samples: code metrics, strategy labeling, statistics.

Usage:
    python analyze_successes.py /path/to/debug_samples/
    python analyze_successes.py /path/to/debug_samples/ --skip-llm
    python analyze_successes.py --csv /path/to/successes.csv /path/to/debug_samples/
"""

import os
import re
import sys
import statistics
from collections import Counter

from shared import (
    read_file, list_samples, common_args, write_csv, read_csv,
    add_thinking_analysis, add_description_analysis,
    print_thinking_summary, print_description_summary, llm_call,
    tee_to_file,
)

THINKING_COLS = ["thinking_present", "thinking_length", "reasoning_score"]
DESC_COLS = ["description_present", "description_length", "description_score",
             "description_matches_code", "overambitious"]
CSV_FIELDS = ["sample_id", "code_lines", "has_helper", "num_helpers",
              "uses_numpy", "uses_nx", "strategy", "strategy_description",
              "info_type", "scores"] + THINKING_COLS + DESC_COLS


def parse_eval_scores(text):
    """Parse 3_eval_results.txt into a compact scores string."""
    if not text:
        return ""
    parts = []
    for m in re.finditer(r'\(([^)]+)\):\s*score=(\d+)', text):
        parts.append(f"({m.group(1)})={m.group(2)}")
    return " ".join(parts)


def compute_code_metrics(body_text):
    """Compute heuristic code metrics from parsed function body."""
    if not body_text:
        return {"code_lines": 0, "has_helper": False, "num_helpers": 0,
                "uses_numpy": False, "uses_nx": False}

    lines = [l for l in body_text.splitlines() if l.strip()]
    code_lines = len(lines)

    # Count nested def statements (helpers)
    helper_defs = re.findall(r'^\s+def\s+\w+\s*\(', body_text, re.MULTILINE)
    num_helpers = len(helper_defs)

    uses_numpy = bool(re.search(r'\bnp\.', body_text))
    uses_nx = bool(re.search(r'\bnx\.|G\.', body_text))

    return {
        "code_lines": code_lines,
        "has_helper": num_helpers > 0,
        "num_helpers": num_helpers,
        "uses_numpy": uses_numpy,
        "uses_nx": uses_nx,
    }


def batch_strategy_labeling(samples_with_bodies, model, batch_size=25):
    """Label strategies via LLM in batches. Use batches of functions so LLM sees multiple functiosn and identify common patterns across them. Returns dict of sample_id -> (strategy, description, info_type)."""
    results = {}
    batches = []
    for i in range(0, len(samples_with_bodies), batch_size):
        batches.append(samples_with_bodies[i:i + batch_size])

    for batch_idx, batch in enumerate(batches):
        # Build prompt with all functions in this batch
        function_listings = []
        for sample_id, body in batch:
            function_listings.append(f"--- {sample_id} ---\n{body}")
        functions_text = "\n\n".join(function_listings)

        prompt = f"""Here are {len(batch)} priority functions that were successfully evaluated
for a deletion correcting code construction problem. Each function takes
(node, G, n, s) where node is a binary string, G is the confusability
graph, n is the codeword length, and s is the number of deletions.
The function returns a float priority score used in a greedy
independent set algorithm on G.

For each function:
1. Assign a short strategy label (2-5 words) describing the core algorithmic
   idea, plus a one sentence description. Use the same label for functions
   with the same approach.
2. Assign an info_type from exactly one of:
   - graph_only: only uses graph properties (degree, neighbors, clustering,
     centrality, etc.), does not inspect the node string content
   - sequence_only: only uses the node string (count of 1s, hamming weight,
     substrings, etc.), does not use the graph G
   - both: uses graph structure and node string properties (e.g. looks at
     character composition of neighbors, string overlap between connected nodes)

{functions_text}

Output one line per function in this exact format:
sample_id | strategy_label | description | info_type
Only output the description once per unique label (on first occurrence),
leave it empty for subsequent functions with the same label."""

        try:
            response = llm_call(prompt, model)
            for line in response.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("---"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    sid = parts[0]
                    # Match sample_id to batch entries (fuzzy: LLM might abbreviate)
                    matched_id = None
                    for sample_id, _ in batch:
                        if sample_id in sid or sid in sample_id:
                            matched_id = sample_id
                            break
                    if matched_id:
                        results[matched_id] = {
                            "strategy": parts[1],
                            "strategy_description": parts[2],
                            "info_type": parts[3].lower().strip(),
                        }
        except Exception as e:
            print(f"  warning: strategy labeling failed for batch {batch_idx}: {e}")

    return results


def classify_sample(sample_id, sample_dir, verbose):
    """Compute heuristic metrics for a single success sample. Returns row dict."""
    row = {"sample_id": sample_id}

    # Code metrics
    body = read_file(os.path.join(sample_dir, "2_parsed_body.py")) or ""
    metrics = compute_code_metrics(body)
    row.update(metrics)

    # Scores from eval results
    eval_text = read_file(os.path.join(sample_dir, "3_eval_results.txt")) or ""
    row["scores"] = parse_eval_scores(eval_text)

    # Strategy fields (filled later by LLM batch)
    row["strategy"] = ""
    row["strategy_description"] = ""
    row["info_type"] = ""

    if verbose:
        print(f"  {sample_id}: {metrics['code_lines']} lines, "
              f"helpers={metrics['num_helpers']}, numpy={metrics['uses_numpy']}, "
              f"nx={metrics['uses_nx']}")

    return row, body


def print_summary(rows):
    """Print success summary."""
    total = len(rows)
    if total == 0:
        print("\nsuccess summary. total: 0 samples\n")
        return

    print(f"\nsuccess summary. total: {total} samples\n")

    code_lines = [int(r["code_lines"]) if isinstance(r["code_lines"], str) else r["code_lines"]
                  for r in rows]
    valid_lines = [c for c in code_lines if c > 0]
    if valid_lines:
        print("code structure:")
        print(f"  avg code lines: {statistics.mean(valid_lines):.1f} "
              f"(min={min(valid_lines)}, max={max(valid_lines)}, "
              f"median={statistics.median(valid_lines):.0f})")

    helpers = [r for r in rows if str(r.get("has_helper", "")).lower() == "true"
               or r.get("has_helper") is True]
    print(f"  with helper funcs: {len(helpers)} ({100*len(helpers)/total:.1f}%)")

    np_users = [r for r in rows if str(r.get("uses_numpy", "")).lower() == "true"
                or r.get("uses_numpy") is True]
    nx_users = [r for r in rows if str(r.get("uses_nx", "")).lower() == "true"
                or r.get("uses_nx") is True]
    print(f"  uses numpy: {len(np_users)} ({100*len(np_users)/total:.1f}%)")
    print(f"  uses nx/G methods: {len(nx_users)} ({100*len(nx_users)/total:.1f}%)")

    info_types = [r["info_type"] for r in rows if r.get("info_type")]
    if info_types:
        print(f"\ninformation type:")
        for it, count in Counter(info_types).most_common():
            print(f"  {it}: {count:>4} ({100*count/len(info_types):>5.1f}%)")

    strategies = [r["strategy"] for r in rows if r.get("strategy")]
    if strategies:
        strat_counts = Counter(strategies)
        print(f"\nunique strategies: {len(strat_counts)}")
        print(f"\ntop 10 strategies:")
        # Collect descriptions and info_types per strategy
        strat_info = {}
        for r in rows:
            s = r.get("strategy", "")
            if s and s not in strat_info:
                strat_info[s] = {
                    "description": r.get("strategy_description", ""),
                    "info_type": r.get("info_type", ""),
                }
        for i, (strat, count) in enumerate(strat_counts.most_common(10), 1):
            info = strat_info.get(strat, {})
            it = info.get("info_type", "")
            desc = info.get("description", "")
            pct = 100 * count / len(strategies)
            line = f"  {i}. {strat} ({count} samples, {pct:.1f}%)"
            if it:
                line += f" [{it}]"
            print(line)
            if desc:
                print(f"     {desc}")

        remaining = len(strat_counts) - 10
        if remaining > 0:
            remaining_count = sum(c for _, c in strat_counts.most_common()[10:])
            print(f"  ({remaining} other strategies with {remaining_count} samples)")


def main():
    parser = common_args("Analyze successful debug samples")
    args = parser.parse_args()

    # Summary-only mode
    if args.csv_path:
        rows = read_csv(args.csv_path)
        log_path = os.path.join(args.path, "successes.log")
        with tee_to_file(log_path):
            print_summary(rows)
            print_thinking_summary(rows)
            print_description_summary(rows)
        return

    debug_dir = args.path
    samples = list_samples(debug_dir, "success")
    if not samples:
        print(f"no success samples found in {debug_dir}")
        sys.exit(1)

    print(f"analyzing {len(samples)} success samples...")

    # Step 1: Heuristic classification
    rows = []
    bodies = []  # (sample_id, body) for strategy labeling
    for sample_id, sample_dir in samples:
        row, body = classify_sample(sample_id, sample_dir, args.verbose)
        rows.append((row, sample_dir))
        if body and body.strip():
            bodies.append((sample_id, body))

    # Step 2: Strategy labeling via LLM (batched)
    if not args.skip_llm and bodies:
        print(f"running strategy labeling on {len(bodies)} samples...")
        strategy_results = batch_strategy_labeling(bodies, args.model)
        for row, _ in rows:
            sid = row["sample_id"]
            if sid in strategy_results:
                row.update(strategy_results[sid])

    # Step 3: Thinking trace + description analysis
    final_rows = []
    for row, sample_dir in rows:
        add_thinking_analysis(row, sample_dir, args.model, args.skip_llm)
        add_description_analysis(row, sample_dir, args.model, args.skip_llm)
        final_rows.append(row)

    # Write CSV
    output_path = args.output or os.path.join(debug_dir, "successes.csv")
    write_csv(final_rows, CSV_FIELDS, output_path)

    # Print summary
    log_path = os.path.join(debug_dir, "successes.log")
    with tee_to_file(log_path):
        print_summary(final_rows)
        print_thinking_summary(final_rows)
        print_description_summary(final_rows)


if __name__ == "__main__":
    main()
