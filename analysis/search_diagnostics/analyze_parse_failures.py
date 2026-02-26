#!/usr/bin/env python3
"""Analyze parse failure debug samples: count failure reasons and show examples.

Reads the failure reason from 2_parsed_body.py (written by evaluator as
"PARSE_FAILED: <reason>") and groups samples by reason.

For each reason, shows --examples unique examples (deduplicated by normalized
raw output so you see distinct failure patterns, not n copies of the same output).

Usage:
    python analyze_parse_failures.py <debug_samples_dir> [options]

Options:
    --examples N      Number of unique examples per failure reason (default: 3)
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


def extract_failure_reason(sample_dir):
    """Extract failure reason from 2_parsed_body.py."""
    body = read_file(os.path.join(sample_dir, "2_parsed_body.py"))
    if not body:
        return "unknown (no 2_parsed_body.py)"
    body = body.strip()
    if body.startswith("PARSE_FAILED:"):
        return body[len("PARSE_FAILED:"):].strip()
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


def main():
    parser = argparse.ArgumentParser(description="Analyze parse failure debug samples")
    parser.add_argument("path", help="Path to debug_samples/ directory")
    parser.add_argument("--tokenizer", default="bigcode/starcoder2-15b",
                        help="HuggingFace tokenizer for token counting "
                             "(default: bigcode/starcoder2-15b)")
    parser.add_argument("--max-tokens", type=int, default=246,
                        help="Max new tokens for the sampler model (default: 246)")
    parser.add_argument("--examples", type=int, default=3,
                        help="Number of unique examples per failure reason (default: 3)")
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

        # Distribution table
        print("failure reason distribution:")
        for reason, entries in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            count = len(entries)
            pct = 100 * count / total
            print(f"  {count:>4} ({pct:>5.1f}%)  {reason}")

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

        # Examples per reason (deduplicated by normalized code)
        for reason, entries in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            print(f"\n{'='*70}")
            print(f"reason: {reason} ({len(entries)} samples)")
            print(f"{'='*70}")
            seen = set()
            shown = 0
            for sample_id, sample_dir, tok, raw in entries:
                norm = _normalize_code(raw)
                if norm in seen:
                    continue
                seen.add(norm)
                print(f"\n--- {sample_id} ({tok} tokens) ---")
                print(raw if raw else "(empty)")
                shown += 1
                if shown >= args.examples:
                    break
            dupes = len(entries) - len(seen)
            remaining_unique = len(seen) - shown
            if dupes > 0 or remaining_unique > 0:
                parts = []
                if dupes > 0:
                    parts.append(f"{dupes} duplicates")
                if remaining_unique > 0:
                    parts.append(f"{remaining_unique} more unique")
                print(f"\n  [{', '.join(parts)} not shown]")


if __name__ == "__main__":
    main()
