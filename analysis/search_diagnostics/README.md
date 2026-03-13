# Search Diagnostics

Analyze debug samples from a DistributedFunSearch run to understand what happened during search.

The system saves debug samples for each LLM output into three categories: `success/`, `eval_failure/`, and `parse_failure/`. Each sample is a folder containing the prompt, raw LLM output, thinking trace, parsed function body, and evaluation results.

## Scripts

| Script | What it does |
|--------|-------------|
| `analyze_eval_failures.py` | classifies why evaluation failed (timeout vs runtime error), shows partial score distributions for timeouts, uses an LLM to categorize runtime errors |
| `analyze_parse_failures.py` | classifies why the parser could not extract a function from the LLM output (syntax error, no function, multiple functions, parser bug) |
| `analyze_successes.py` | computes code metrics for successful functions and uses an LLM to label strategies in batches |
| `shared.py` | shared utilities: file I/O, LLM calls, thinking trace scoring, description analysis |

## Usage

```bash
cd analysis/search_diagnostics

# heuristics only, no LLM calls needed
python analyze_eval_failures.py /path/to/debug_samples/ --skip-llm
python analyze_parse_failures.py /path/to/debug_samples/ --skip-llm
python analyze_successes.py /path/to/debug_samples/ --skip-llm

# with LLM classification (default model: gpt-5)
python analyze_eval_failures.py /path/to/debug_samples/ --model gpt-4o-mini
python analyze_successes.py /path/to/debug_samples/ --model claude-sonnet-4-20250514

# verbose (prints per sample details)
python analyze_eval_failures.py /path/to/debug_samples/ --skip-llm -v

# summary from existing csv
python analyze_eval_failures.py --csv /path/to/eval_failures.csv /path/to/debug_samples/
```

Each script writes a CSV to the debug_samples folder and prints a summary to stdout.
