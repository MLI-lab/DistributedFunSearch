#!/usr/bin/env python3
"""Analyze parse failure debug samples: count failure reasons and show examples.

Reads the failure reason from 2_parsed_body.py (written by evaluator as
"PARSE_FAILED: <reason>") and groups samples by reason.

For each reason, shows --examples unique examples (deduplicated by normalized
raw output so you see distinct failure patterns, not n copies of the same output).

Usage:
    python analyze_parse_failures.py <debug_samples_dir> [options]

Options:
    --tokenizer NAME  HuggingFace tokenizer for token counting (default: bigcode/starcoder2-15b)
    --max-tokens N    Max new tokens the sampler model was configured with,
                      used to distinguish "hit token limit" vs "stopped early" (default: 246)
"""

import os
import sys
import re
import argparse
import statistics
from collections import defaultdict

from shared import read_file, list_samples, tee_to_file, load_tokenizer, count_tokens


def _normalize_reason(reason):
    """Normalize failure reasons to group variants together.

    E.g. all "AST repair truncated '<name>' to no body" become one bucket
    regardless of the function name.
    """
    m = re.match(r"AST repair truncated '.*?' to no body", reason)
    if m:
        return "AST repair truncated target function to no body (whole body had syntax errors)"
    return reason


def extract_failure_reason(sample_dir):
    """Extract failure reason from 2_parsed_body.py.

    The file may contain multiple PARSE_FAILED lines (one per test input).
    We take the first one since they share the same root cause (same code).
    """
    body = read_file(os.path.join(sample_dir, "2_parsed_body.py"))
    if not body:
        return "unknown (no 2_parsed_body.py)"
    body = body.strip()
    # Take first PARSE_FAILED reason (file may have multiple concatenated)
    if body.startswith("PARSE_FAILED:"):
        first_line = body.split("PARSE_FAILED:")[1].strip()
        # Trim at next PARSE_FAILED if concatenated
        if "PARSE_FAILED:" in first_line:
            first_line = first_line[:first_line.index("PARSE_FAILED:")].strip()
        return _normalize_reason(first_line)
    if body == "PARSE_FAILED":
        return "unknown (no reason recorded)"
    # File has actual code, not a parse failure
    return None


def _normalize_code(text):
    """Normalize code for deduplication: strip comments, collapse whitespace."""
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


NO_CODE_REASONS = {
    "LLM returned empty output",
    "LLM returned no code (only markdown/description tags)",
}

# Patterns that indicate actual Python code (not just prose)
_CODE_PATTERN = re.compile(
    r'\b(def |return |import |from \w+ import |class )\b|[a-zA-Z_]\w*\s*=',
    re.MULTILINE,
)


def _has_code(raw_text):
    """Check if raw LLM output contains any Python code indicators."""
    # Strip description/thinking tags first (same as evaluator does)
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    text = re.sub(r'<description>.*?</description>', '', text, flags=re.DOTALL)
    return bool(_CODE_PATTERN.search(text))


def main():
    parser = argparse.ArgumentParser(description="Analyze parse failure debug samples")
    parser.add_argument("path", help="Path to debug_samples/ directory")
    parser.add_argument("--tokenizer", default="bigcode/starcoder2-15b",
                        help="HuggingFace tokenizer for token counting "
                             "(default: bigcode/starcoder2-15b)")
    parser.add_argument("--max-tokens", type=int, default=246,
                        help="Max new tokens for the sampler model (default: 246)")
    args = parser.parse_args()

    samples = list_samples(args.path, "parse_failure")
    if not samples:
        print(f"no parse_failure samples found in {args.path}")
        sys.exit(1)

    print(f"tokenizer: {args.tokenizer}, max_tokens: {args.max_tokens}")
    tokenizer = load_tokenizer(args.tokenizer)
    if not tokenizer:
        print(f"  warning: could not load tokenizer, using chars/3.5 heuristic")

    # Group by failure reason
    by_reason = defaultdict(list)
    for sample_id, sample_dir in samples:
        reason = extract_failure_reason(sample_dir)
        if reason is None:
            continue
        raw = read_file(os.path.join(sample_dir, "1_raw_llm_output.txt")) or ""
        tokens = count_tokens(raw, tokenizer)
        by_reason[reason].append((sample_id, sample_dir, tokens, raw))

    total = sum(len(v) for v in by_reason.values())

    log_path = os.path.join(args.path, "parse_failures.log")
    with tee_to_file(log_path):
        print(f"\nparse failure analysis: {total} samples\n")

        # Distribution table with example sample IDs
        print("failure reason distribution:")
        for reason, entries in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            count = len(entries)
            pct = 100 * count / total
            examples = ", ".join(e[0] for e in entries[:3])
            print(f"  {count:>4} ({pct:>5.1f}%)  {reason}")
            print(f"         e.g. {examples}")

        # Code vs prose breakdown: how many had actual code in raw output?
        all_entries = [e for entries in by_reason.values() for e in entries]
        with_code = sum(1 for _, _, _, raw in all_entries if _has_code(raw))
        no_code = total - with_code
        print(f"\ncode vs prose (heuristic on raw LLM output):")
        print(f"  {with_code:>4} ({100*with_code/total:>5.1f}%)  had code (but too broken to parse)")
        print(f"  {no_code:>4} ({100*no_code/total:>5.1f}%)  no code at all (only prose/description)")

        # For "no code" reasons: token stats to distinguish truncation vs stopped
        for reason in NO_CODE_REASONS:
            if reason not in by_reason:
                continue
            entries = by_reason[reason]
            tokens = [e[2] for e in entries]
            hit_max = sum(1 for t in tokens if t >= args.max_tokens)
            short = len(tokens) - hit_max
            print(f"\n  [{reason}]")
            print(f"    tokens: min={min(tokens)}, median={statistics.median(tokens):.0f}, max={max(tokens)}")
            print(f"    {hit_max}/{len(tokens)} hit max_tokens ({args.max_tokens}), "
                  f"{short}/{len(tokens)} stopped early")


if __name__ == "__main__":
    main()
