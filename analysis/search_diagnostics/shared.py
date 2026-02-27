"""Shared utilities for search diagnostics analysis scripts."""

import os
import re
import csv
import sys
import argparse
import statistics
from contextlib import contextmanager

# Load .env from project root (for API keys)
try:
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# Drop unsupported params automatically (e.g. temperature=0 on reasoning models)
try:
    import litellm
    litellm.drop_params = True
except ImportError:
    pass


def load_tokenizer(model_name):
    """Try to load a huggingface tokenizer for accurate token counting."""
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def count_tokens(text, tokenizer):
    """Count tokens. Uses real tokenizer if available, otherwise chars/3.5."""
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return int(len(text) / 3.5)


def read_file(path):
    """Read file content, return None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def list_samples(debug_dir, category):
    """List all sample directories in a category folder."""
    cat_dir = os.path.join(debug_dir, category)
    if not os.path.isdir(cat_dir):
        return []
    entries = sorted(os.listdir(cat_dir))
    return [(s, os.path.join(cat_dir, s)) for s in entries
            if os.path.isdir(os.path.join(cat_dir, s))]


def extract_description(raw_output):
    """Extract text between <description>...</description> tags."""
    if not raw_output:
        return None
    match = re.search(r'<description>(.*?)</description>', raw_output, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_thinking(raw_output):
    """Remove <think>...</think> from raw output."""
    if not raw_output:
        return raw_output
    return re.sub(r'<think>.*?</think>\s*', '', raw_output, flags=re.DOTALL).strip()


def llm_call(prompt, model):
    """Make a single LLM call via litellm."""
    import litellm
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def score_thinking_trace(prompt_text, trace_text, model):
    """Score a thinking trace on 1-5 scale. Returns (score, explanation)."""
    prompt = f"""You are evaluating a thinking trace from an LLM that was given the following prompt:

---
{prompt_text}
---

Here is the thinking trace:

---
{trace_text}
---

Rate the reasoning quality on a 1-5 scale:

5 = excellent: clear problem understanding, novel/creative ideas, systematic
    reasoning that builds toward a well motivated solution
4 = coherent: sound reasoning, follows the task correctly, but no novel
    insights, applies standard/obvious approaches
3 = repetitive: on topic but keeps restating the same ideas, circular
    reasoning, doesn't converge on an approach
2 = derailed: understands the task but reasoning goes wrong, starts
    hallucinating facts, contradicts itself, or loses focus and drifts
    into unrelated topics
1 = confused: never grasps what's being asked, reasons about the wrong
    problem, can't make sense of the task, or incoherent throughout

Output exactly one line:
score | one_sentence_explanation"""

    response = llm_call(prompt, model)
    parts = response.split("|", 1)
    try:
        score = int(parts[0].strip())
        score = max(1, min(5, score))
    except (ValueError, IndexError):
        score = None
    return score


def score_description(prompt_text, raw_output_no_think, model):
    """Score a description on multiple dimensions. Returns dict."""
    prompt = f"""You are analyzing an output from an LLM that was given the following prompt:

---
{prompt_text}
---

Here is the LLM's output (description + code):

---
{raw_output_no_think}
---

The output contains a <description> tag (the LLM's natural language summary,
reflection, or plan) and the code it generated. Assess:

1. description_score (1-5):
   5 = novel/creative idea, specific enough to implement, clearly motivated
   4 = describes a concrete, feasible approach clearly (even if standard)
   3 = has a reasonable idea but too vague on details to guide implementation
   2 = superficial, mostly restates the problem in different words
   1 = just buzzwords or phrases that sound impressive without saying anything

2. description_matches_code: yes / partial / no
   Does the description match what the code actually does?
   "yes" = description accurately reflects what the code does
   "partial" = right general idea but missing key details or exaggerates
   "no" = description and code are about different approaches

3. overambitious: yes / no
   Was the described approach too complex for the LLM to implement correctly?
   (e.g. describes a multi step algorithm but code is incomplete/broken,
   mentions advanced techniques that the LLM couldn't translate to code)

Output exactly one line:
description_score | description_matches_code | overambitious"""

    response = llm_call(prompt, model)
    parts = [p.strip() for p in response.split("|")]
    result = {}
    try:
        result["description_score"] = max(1, min(5, int(parts[0])))
    except (ValueError, IndexError):
        result["description_score"] = ""
    try:
        matches = parts[1].lower()
        if "yes" in matches:
            result["description_matches_code"] = "yes"
        elif "partial" in matches:
            result["description_matches_code"] = "partial"
        elif "no" in matches:
            result["description_matches_code"] = "no"
        else:
            result["description_matches_code"] = matches
    except IndexError:
        result["description_matches_code"] = ""
    try:
        result["overambitious"] = "yes" if "yes" in parts[2].lower() else "no"
    except IndexError:
        result["overambitious"] = ""
    return result


def add_thinking_analysis(row, sample_dir, model, skip_llm=False):
    """Add thinking trace columns to row dict."""
    trace = read_file(os.path.join(sample_dir, "1b_thinking_trace.txt"))
    row["thinking_present"] = bool(trace and trace.strip())
    row["thinking_length"] = len(trace.strip()) if trace and trace.strip() else 0
    row["reasoning_score"] = ""

    if row["thinking_present"] and not skip_llm:
        prompt_text = read_file(os.path.join(sample_dir, "0_prompt.txt"))
        if prompt_text and trace:
            try:
                row["reasoning_score"] = score_thinking_trace(prompt_text, trace, model)
            except Exception as e:
                print(f"  warning: thinking trace scoring failed for {row.get('sample_id', '?')}: {e}")


def add_description_analysis(row, sample_dir, model, skip_llm=False):
    """Add description columns to row dict."""
    raw = read_file(os.path.join(sample_dir, "1_raw_llm_output.txt"))
    desc = extract_description(raw)

    row["description_present"] = desc is not None
    row["description_length"] = len(desc) if desc else 0
    row["description_score"] = ""
    row["description_matches_code"] = ""
    row["overambitious"] = ""

    if desc and not skip_llm:
        prompt_text = read_file(os.path.join(sample_dir, "0_prompt.txt"))
        if prompt_text and raw:
            raw_no_think = strip_thinking(raw)
            try:
                result = score_description(prompt_text, raw_no_think, model)
                row["description_score"] = result.get("description_score", "")
                row["description_matches_code"] = result.get("description_matches_code", "")
                row["overambitious"] = result.get("overambitious", "")
            except Exception as e:
                print(f"  warning: description scoring failed for {row.get('sample_id', '?')}: {e}")


def print_thinking_summary(rows):
    """Print thinking trace summary section."""
    total = len(rows)
    if total == 0:
        return
    present = [r for r in rows if r.get("thinking_present")]
    print(f"\nthinking trace analysis:")
    print(f"  traces present: {len(present)}/{total} ({100*len(present)/total:.1f}%)")

    if present:
        lengths = [r["thinking_length"] for r in present]
        print(f"  avg trace length: {statistics.mean(lengths):.0f} chars "
              f"(median={statistics.median(lengths):.0f}, min={min(lengths)}, max={max(lengths)})")

        scored = [r for r in present if isinstance(r.get("reasoning_score"), int)]
        if scored:
            scores = [r["reasoning_score"] for r in scored]
            print(f"\n  reasoning score distribution:")
            for s, label in [(5, "excellent"), (4, "coherent"), (3, "repetitive"),
                             (2, "derailed"), (1, "confused")]:
                count = scores.count(s)
                pct = 100 * count / len(scores)
                print(f"    {s} ({label}): {count:>4} ({pct:>5.1f}%)")
            print(f"    mean: {statistics.mean(scores):.1f}, median: {statistics.median(scores):.0f}")


def print_description_summary(rows):
    """Print description analysis summary section."""
    total = len(rows)
    if total == 0:
        return
    present = [r for r in rows if r.get("description_present")]
    print(f"\ndescription analysis:")
    print(f"  descriptions present: {len(present)}/{total} ({100*len(present)/total:.1f}%)")

    if not present:
        return

    scored = [r for r in present if isinstance(r.get("description_score"), int)]
    if not scored:
        return

    scores = [r["description_score"] for r in scored]
    print(f"\n  description score distribution:")
    for s, label in [(5, "novel"), (4, "concrete"), (3, "vague"),
                     (2, "superficial"), (1, "buzzwords")]:
        count = scores.count(s)
        pct = 100 * count / len(scores)
        print(f"    {s} ({label}): {count:>4} ({pct:>5.1f}%)")
    print(f"    mean: {statistics.mean(scores):.1f}, median: {statistics.median(scores):.0f}")

    matches = [r["description_matches_code"] for r in scored
               if r.get("description_matches_code") in ("yes", "partial", "no")]
    if matches:
        print(f"\n  description matches code:")
        for val in ["yes", "partial", "no"]:
            count = matches.count(val)
            pct = 100 * count / len(matches)
            print(f"    {val}: {count:>4} ({pct:>5.1f}%)")

    overamb = [r for r in scored if r.get("overambitious") == "yes"]
    print(f"\n  overambitious: {len(overamb)}/{len(scored)} ({100*len(overamb)/len(scored):.1f}%)")


class _TeeWriter:
    """Write to both stdout and a file."""
    def __init__(self, file, original):
        self.file = file
        self.original = original
    def write(self, text):
        self.original.write(text)
        self.file.write(text)
    def flush(self):
        self.original.flush()
        self.file.flush()


@contextmanager
def tee_to_file(path):
    """Context manager that duplicates stdout to a log file."""
    with open(path, "w") as f:
        tee = _TeeWriter(f, sys.stdout)
        old = sys.stdout
        sys.stdout = tee
        try:
            yield
        finally:
            sys.stdout = old
    print(f"log written to {path}")


def write_csv(rows, fieldnames, output_path):
    """Write rows to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv written to {output_path}")


def _coerce_row(row):
    """Convert known typed fields back from CSV strings."""
    for key in ("thinking_length", "code_lines", "num_helpers",
                "reasoning_score", "description_score", "description_length",
                "raw_token_count", "wasted_empty_tokens"):
        val = row.get(key, "")
        if val != "" and val is not None:
            try:
                row[key] = int(val)
            except (ValueError, TypeError):
                pass
    for key in ("parsed_empty_line_pct", "wasted_empty_pct"):
        val = row.get(key, "")
        if val != "" and val is not None:
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                pass
    for key in ("thinking_present", "description_present",
                "has_helper", "uses_numpy", "uses_nx",
                "has_return_raw", "has_return_parsed"):
        val = row.get(key, "")
        if isinstance(val, str):
            row[key] = val.lower() == "true"
    return row


def read_csv(path):
    """Read CSV file and return list of row dicts with typed fields."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return [_coerce_row(r) for r in reader]


def common_args(description):
    """Common CLI argument parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("path", help="Path to debug_samples/ directory")
    parser.add_argument("--csv", dest="csv_path", default=None,
                        help="Summary-only mode: analyze existing CSV file")
    parser.add_argument("--model", default="gpt-5",
                        help="LLM model for analysis (default: gpt-5)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip all LLM calls (heuristics only)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: <debug_samples>/<name>.csv)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-sample details")
    return parser
