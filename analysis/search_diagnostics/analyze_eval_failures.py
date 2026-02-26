#!/usr/bin/env python3
"""Analyze eval failure debug samples, classify errors and compute statistics.

Usage:
    python analyze_eval_failures.py /path/to/debug_samples/
    python analyze_eval_failures.py /path/to/debug_samples/ --skip-llm
    python analyze_eval_failures.py --csv /path/to/eval_failures.csv /path/to/debug_samples/

Uses gpt-5 by default for classification (--model to override).
Uses bigcode/starcoder2-15b tokenizer by default for token counting (--tokenizer to override).
Uses 246 max tokens by default for return analysis (--max-tokens to override).
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

# path to the evaluation script (included in llm prompt for context)
EVAL_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "disfun",
    "specifications", "ECC", "evaluation", "graph_fastgraph.py"
)

THINKING_COLS = ["thinking_present", "thinking_length", "reasoning_score"]
DESC_COLS = ["description_present", "description_length", "description_score",
             "description_matches_code", "overambitious"]
RETURN_COLS = ["has_return_raw", "has_return_parsed", "return_subcategory",
               "raw_token_count", "parsed_empty_line_pct",
               "wasted_empty_tokens", "wasted_empty_pct", "observed_error_type",
               "error_message"]
CSV_FIELDS = (["sample_id", "category", "detail"]
              + THINKING_COLS + DESC_COLS + RETURN_COLS)


def parse_eval_results(text):
    """Parse 3_eval_results.txt. Extract score lines, check for timeout,
    collect remaining lines as raw error text for the llm."""
    if not text:
        return {"scores": [], "is_timeout": False, "error_text": ""}

    scores = []
    other_lines = []
    for line in text.strip().splitlines():
        m = re.match(r'\(([^)]+)\):\s*score=(\d+)', line.strip())
        if m:
            scores.append((m.group(1).strip(), int(m.group(2))))
        else:
            other_lines.append(line)

    return {
        "scores": scores,
        "is_timeout": "timeout" in text.lower(),
        "error_text": "\n".join(other_lines).strip(),
    }


def _load_tokenizer(model_name):
    """Try to load a huggingface tokenizer for accurate token counting."""
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None


def _count_tokens(text, tokenizer):
    """Count tokens. Uses real tokenizer if available, otherwise chars/3.5."""
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return int(len(text) / 3.5)


def _has_return(text):
    """Check if text contains a return statement outside of comments."""
    if not text:
        return False
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("#") and re.search(r'\breturn\b', stripped):
            return True
    return False


def _empty_line_pct(text):
    """Percentage of lines that are empty or whitespace only."""
    if not text:
        return 0.0
    lines = text.splitlines()
    empty = sum(1 for l in lines if not l.strip())
    return 100.0 * empty / len(lines)


def _wasted_empty_line_tokens(text, tokenizer):
    """Count how many tokens are wasted on empty lines.

    Removes all empty/whitespace only lines from the text and compares
    token counts. Returns (wasted_tokens, wasted_pct) where pct is
    relative to original token count.
    """
    original_tokens = _count_tokens(text, tokenizer)
    lines = text.splitlines() if text else []
    stripped = "\n".join(l for l in lines if l.strip())
    stripped_tokens = _count_tokens(stripped, tokenizer)
    wasted = original_tokens - stripped_tokens
    wasted_pct = 100.0 * wasted / original_tokens if original_tokens > 0 else 0.0
    return wasted, round(wasted_pct, 1)


def _extract_error_type(eval_text):
    """Extract the python exception type from eval results text."""
    if not eval_text:
        return "unknown"
    if "timeout" in eval_text.lower():
        return "timeout"
    m = re.search(r'(\w+Error):', eval_text)
    if m:
        return m.group(1)
    if "NoneType" in eval_text:
        return "NoneType"
    return "unknown"


def _extract_error_message(eval_text):
    """Extract the error type and message, e.g. 'KeyError: 'weight''.

    Looks for the RUNTIME_ERROR line first, falls back to searching for
    any ExceptionType: message pattern.
    """
    if not eval_text:
        return "unknown"
    if "timeout" in eval_text.lower():
        return "timeout"
    m = re.search(r'RUNTIME_ERROR: (.+)', eval_text)
    if m:
        msg = m.group(1).strip()
        return msg
    m = re.search(r'COMPILE_ERROR: (.+)', eval_text)
    if m:
        return m.group(1).strip()[:120]
    return "unknown"


def analyze_return_statement(sample_dir, max_tokens, tokenizer):
    """Check whether the generated function has a return statement.

    Classifies into subcategories:
      has_return:             parsed body has return, not a return issue.
      return_lost_to_syntax_error: raw output has return but parser dropped it.
      no_return_max_tokens:   hit max_tokens before generating a return.
      no_return_short:        below max_tokens, gave up early.
    """
    raw = read_file(os.path.join(sample_dir, "1_raw_llm_output.txt")) or ""
    parsed = read_file(os.path.join(sample_dir, "2_parsed_body.py")) or ""
    eval_text = read_file(os.path.join(sample_dir, "3_eval_results.txt")) or ""

    has_ret_raw = _has_return(raw)
    has_ret_parsed = _has_return(parsed)
    token_count = _count_tokens(raw, tokenizer)
    empty_pct = _empty_line_pct(raw)
    wasted_tokens, wasted_pct = _wasted_empty_line_tokens(raw, tokenizer)

    if has_ret_parsed:
        subcat = "has_return"
    elif has_ret_raw:
        subcat = "return_lost_to_syntax_error"
    else:
        hit_max = token_count >= max_tokens
        if hit_max:
            subcat = "no_return_max_tokens"
        else:
            subcat = "no_return_short"

    return {
        "has_return_raw": has_ret_raw,
        "has_return_parsed": has_ret_parsed,
        "return_subcategory": subcat,
        "raw_token_count": token_count,
        "parsed_empty_line_pct": round(empty_pct, 1),
        "wasted_empty_tokens": wasted_tokens,
        "wasted_empty_pct": wasted_pct,
        "observed_error_type": _extract_error_type(eval_text),
        "error_message": _extract_error_message(eval_text),
    }


def classify_with_llm(function_body, error_text, model, eval_script_source):
    """Use llm to classify a non timeout eval failure."""
    prompt = f"""Here is a function that was executed in a sandbox with the following evaluation script:

---
{eval_script_source}
---

The function failed at runtime. Here is the function body and the error:

Function body:
{function_body}

Error:
{error_text}

Categorize this error into exactly one of the following categories:

1. output_type_error, function returns wrong type instead of float
   (detail: what type was returned, e.g. None, tuple, list, str)
2. wrong_variable, uses variables not in the function signature
   (detail: which variable name(s))
3. import_incorrect_call, calls a function incorrectly, e.g. log() instead
   of math.log() (detail: which function)
4. import_missing_package, imports a package not available in the sandbox
   (detail: which package)
5. hallucinated, calls a method/attribute/function that does not exist,
   e.g. non existent method on an object or undefined helper function
   (detail: which method/function)
6. incorrect_method_usage, calls a real method with wrong arguments or on
   wrong type (detail: which method)
7. wrapper_violation, uses graph library methods not exposed in the evaluation
   wrapper, e.g. raw nx.* calls (detail: which method)
8. other, anything not fitting the above (detail: brief description)

Output exactly one line:
category | detail"""

    response = llm_call(prompt, model)
    parts = [p.strip() for p in response.split("|", 1)]
    raw_cat = re.sub(r'^\d+[._]*\s*', '', parts[0]).strip() if parts else "other"
    category = raw_cat.lower().replace(" ", "_")
    detail = parts[1] if len(parts) > 1 else ""
    return category, detail


def classify_sample(sample_id, sample_dir, model, skip_llm, eval_script_source, verbose):
    """Classify a single eval failure sample. Returns row dict."""
    eval_text = read_file(os.path.join(sample_dir, "3_eval_results.txt"))
    eval_data = parse_eval_results(eval_text)
    row = {"sample_id": sample_id}

    if eval_data["is_timeout"]:
        row["category"] = "timeout"
        partial = " ".join(f"({k})={v}" for k, v in eval_data["scores"])
        row["detail"] = f"partial: {partial}" if partial else "no partial scores"
    elif skip_llm:
        row["category"] = "runtime_error"
        row["detail"] = eval_data["error_text"][:200]
    else:
        function_body = read_file(os.path.join(sample_dir, "2_parsed_body.py")) or ""
        try:
            row["category"], row["detail"] = classify_with_llm(
                function_body, eval_data["error_text"], model, eval_script_source)
        except Exception as e:
            print(f"  warning: llm classification failed for {sample_id}: {e}")
            row["category"] = "runtime_error"
            row["detail"] = eval_data["error_text"][:200]

    if verbose:
        print(f"  {sample_id}: {row['category']}, {row['detail'][:80]}")

    return row


def print_summary(rows):
    """Print eval failure summary."""
    total = len(rows)
    if total == 0:
        print("\neval failure summary. total: 0 samples\n")
        return

    print(f"\neval failure summary. total: {total} samples\n")

    cats = Counter(r["category"] for r in rows)
    print("category breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count:>4} ({100 * count / total:>5.1f}%)")

    # timeout details, score distribution per input
    timeout_rows = [r for r in rows if r["category"] == "timeout"]
    if timeout_rows:
        print(f"\ntimeout ({len(timeout_rows)} samples):")
        print("  score distribution before timeout (per input):")
        all_scores = {}
        for r in timeout_rows:
            for m in re.finditer(r'\(([^)]+)\)=(\d+)', r.get("detail", "")):
                all_scores.setdefault(m.group(1), []).append(int(m.group(2)))
        for key in sorted(all_scores):
            vals = all_scores[key]
            print(f"    ({key}): {len(vals):>3} samples, "
                  f"mean={statistics.mean(vals):.1f}, "
                  f"median={statistics.median(vals):.0f}, "
                  f"min={min(vals)}, max={max(vals)}")

    # group by exact error message across all runtime errors
    runtime_rows = [r for r in rows if r["category"] != "timeout"]
    if runtime_rows:
        msgs = Counter(r.get("error_message", "unknown") for r in runtime_rows)
        print(f"\nerror message breakdown ({len(runtime_rows)} runtime errors):")
        for msg, cnt in msgs.most_common():
            print(f"  {cnt:>4} ({_pct(cnt, len(runtime_rows)):>5.1f}%)  {msg}")



def _pct(n, total):
    return 100 * n / total if total else 0


def _print_stats(label, values):
    """Print min, max, mean, median for a list of numbers."""
    if not values:
        return
    print(f"  {label}:")
    print(f"    min={min(values):.1f}, max={max(values):.1f}, "
          f"mean={statistics.mean(values):.1f}, median={statistics.median(values):.1f}")


def print_return_analysis(rows):
    """Print return statement analysis summary.

    Checks whether the generated function body contains a return statement,
    independent of what runtime error actually fired. This catches functions
    that crash with e.g. NameError before the missing return would surface.
    """
    total = len(rows)
    if total == 0:
        return

    no_return = [r for r in rows if not r.get("has_return_parsed")]
    print(f"\nreturn statement analysis:")
    if not no_return:
        print(f"  all {total} samples have a return statement in parsed body")
        return

    n_miss = len(no_return)
    print(f"  total eval failures: {total}")
    print(f"  missing return in parsed body: {n_miss:>4} ({_pct(n_miss, total):.1f}%)")

    # subcategory breakdown
    subcats = Counter(r.get("return_subcategory", "unknown") for r in no_return)
    subcat_labels = [
        ("return_lost_to_syntax_error", "raw has return, lost to AST truncation (syntax error before it)"),
        ("no_return_max_tokens",   "hit max_tokens before generating return"),
        ("no_return_short",        "below max_tokens, gave up early"),
    ]
    print(f"\n  breakdown of {n_miss} missing return samples:")
    for subcat, label in subcat_labels:
        count = subcats.get(subcat, 0)
        print(f"    {subcat}: {count:>4} "
              f"({_pct(count, n_miss):>5.1f}% of missing, "
              f"{_pct(count, total):>5.1f}% of all)  .. {label}")

    # which error actually fired at runtime.
    # when priority() returns None the evaluator always produces:
    #   TypeError: bad operand type for unary -: 'NoneType'
    none_err = "bad operand type for unary -: 'NoneType'"
    is_none = [r for r in no_return if none_err in r.get("error_message", "")]
    is_other = [r for r in no_return if none_err not in r.get("error_message", "")]

    print(f"\n  observed error vs missing return:")
    print(f"    None was the direct cause:        {len(is_none):>4} ({_pct(len(is_none), n_miss):.1f}%)")
    none_max = sum(1 for r in is_none if r.get("return_subcategory") == "no_return_max_tokens")
    none_short = len(is_none) - none_max
    print(f"      hit max_tokens: {none_max}, below max_tokens: {none_short}")
    print(f"    other error fired first:          {len(is_other):>4} ({_pct(len(is_other), n_miss):.1f}%)")

    # token counts
    tcounts = [r["raw_token_count"] for r in no_return
               if isinstance(r.get("raw_token_count"), (int, float))]
    _print_stats(f"token counts for missing return samples ({len(tcounts)})", tcounts)

    # wasted empty line tokens for samples that hit max_tokens
    max_tok = [r for r in no_return
               if r.get("return_subcategory") == "no_return_max_tokens"]
    if max_tok:
        wasted = [r["wasted_empty_tokens"] for r in max_tok
                  if isinstance(r.get("wasted_empty_tokens"), (int, float))]
        wasted_pcts = [r["wasted_empty_pct"] for r in max_tok
                       if isinstance(r.get("wasted_empty_pct"), (int, float))]
        if wasted:
            _print_stats(f"wasted tokens on empty lines, max_tokens samples ({len(wasted)})", wasted)
        if wasted_pcts:
            _print_stats(f"wasted % on empty lines, max_tokens samples ({len(wasted_pcts)})", wasted_pcts)
            for lo in range(0, 60, 5):
                hi = lo + 5
                n = sum(1 for p in wasted_pcts if lo <= p < hi)
                if n:
                    print(f"    {lo:>2}..{hi:>3}%: {n:>3} ({_pct(n, len(wasted_pcts)):>5.1f}%)")


def consolidate_rows(rows, model):
    """Deduplicate detail strings within each category using the llm.

    Groups all details per category and asks the llm to assign matching
    descriptions to items that describe the same underlying error. Also
    normalizes category names, e.g. "8._other" becomes "other".
    """
    from collections import defaultdict

    # normalize category names first, e.g. "8._other" -> "other"
    for r in rows:
        cat = r.get("category", "")
        normalized = re.sub(r'^\d+[._]*\s*', '', cat).strip().lower().replace(" ", "_")
        if normalized != cat:
            r["category"] = normalized

    by_cat = defaultdict(list)
    for i, r in enumerate(rows):
        cat = r.get("category", "")
        if cat == "timeout":
            continue
        by_cat[cat].append((i, r.get("detail", "")))

    for cat, items in by_cat.items():
        details_block = "\n".join(f"  {i}: {d}" for i, (_, d) in enumerate(items))

        prompt = f"""Below are {len(items)} error descriptions, all from the same category.
Some of them may describe the same underlying error in different words.

{details_block}

For items that describe the same error, assign the exact same description.
Keep descriptions short but specific (max 15 words). Items that describe
genuinely different errors should keep different descriptions.

For example, these describe the same error and should get one description:
  "accessing missing edge attribute 'weight' on G[u][v] (edges are unweighted)"
  "accesses missing edge attribute 'weight' on edges, causing KeyError"
  -> both become: "accessing missing edge attribute 'weight' on unweighted edges"

And these three also describe the same error (wrong index type on graph object):
  "Indexed G.nodes with a node key (G.nodes[node]); in FastGraph G.nodes is a sequence"
  "indexed G.nodes with a node label (string) instead of an integer"
  "uses NetworkX adjacency dict G[node] with numeric index (G[node][i])"
  -> all become: "confused about graph data structure, e.g. treats G.nodes as dict (is a tuple) or G[node] as positional list (is a neighbor dict)"

Output one line per item, in order:
index | description"""

        try:
            response = llm_call(prompt, model)
        except Exception as e:
            print(f"  warning: consolidation failed for {cat}: {e}")
            continue

        for line in response.strip().splitlines():
            parts = [p.strip() for p in line.split("|", 1)]
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            if 0 <= idx < len(items):
                row_idx = items[idx][0]
                rows[row_idx]["detail"] = parts[1]

    return rows


def main():
    parser = common_args("Analyze eval failure debug samples")
    parser.add_argument("--max-tokens", type=int, default=246,
                        help="Max new tokens for the sampler model (default: 246 for StarCoder2)")
    parser.add_argument("--tokenizer", default="bigcode/starcoder2-15b",
                        help="HuggingFace tokenizer for token counting "
                             "(default: bigcode/starcoder2-15b)")
    parser.add_argument("--consolidate", action="store_true",
                        help="Re-read existing CSV and consolidate duplicate descriptions "
                             "using the llm. Requires --csv.")
    args = parser.parse_args()

    if args.csv_path:
        rows = read_csv(args.csv_path)
        if args.consolidate:
            print(f"consolidating {len(rows)} rows with {args.model}...")
            rows = consolidate_rows(rows, args.model)
            # use whatever fields exist in the rows
            all_fields = list(dict.fromkeys(
                k for r in rows for k in r.keys()))
            write_csv(rows, all_fields, args.csv_path)
        log_path = os.path.join(args.path, "eval_failures.log")
        with tee_to_file(log_path):
            print_summary(rows)
            print_return_analysis(rows)
            print_thinking_summary(rows)
            print_description_summary(rows)
        return

    samples = list_samples(args.path, "eval_failure")
    if not samples:
        print(f"no eval_failure samples found in {args.path}")
        sys.exit(1)

    print(f"analyzing {len(samples)} eval failure samples...")

    print(f"tokenizer: {args.tokenizer}, max_tokens: {args.max_tokens}")
    tokenizer = _load_tokenizer(args.tokenizer)
    if not tokenizer:
        print(f"  warning: could not load tokenizer, using chars/3.5 heuristic")

    eval_script_source = ""
    eval_path = os.path.abspath(EVAL_SCRIPT_PATH)
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_script_source = f.read()
    elif not args.skip_llm:
        print(f"warning: eval script not found at {eval_path}, llm calls will lack context")

    rows = []
    for sample_id, sample_dir in samples:
        row = classify_sample(sample_id, sample_dir, args.model, args.skip_llm,
                              eval_script_source, args.verbose)
        row.update(analyze_return_statement(sample_dir, args.max_tokens, tokenizer))
        add_thinking_analysis(row, sample_dir, args.model, args.skip_llm)
        add_description_analysis(row, sample_dir, args.model, args.skip_llm)
        rows.append(row)

    if not args.skip_llm:
        print("consolidating descriptions...")
        rows = consolidate_rows(rows, args.model)

    output_path = args.output or os.path.join(args.path, "eval_failures.csv")
    write_csv(rows, CSV_FIELDS, output_path)

    log_path = os.path.join(args.path, "eval_failures.log")
    with tee_to_file(log_path):
        print_summary(rows)
        print_return_analysis(rows)
        print_thinking_summary(rows)
        print_description_summary(rows)


if __name__ == "__main__":
    main()
