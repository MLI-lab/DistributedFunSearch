# Search Diagnostics Analysis

## Context

We want to understand the evolutionary search dynamics at different time points (early/mid/late checkpoints). When `debug_samples=True`, the evaluator writes per-sample debug data into three category folders: `success/`, `eval_failure/`, `parse_failure/`.

We write **one analysis script per category**, each producing a csv with columns `sample_id, category, detail`.

---

## Part 1: Pipeline Changes (done)

Sampler + Evaluator already modified to produce:
```
debug_samples/
    success/
        eval{PID}_{counter}_island{id}/
            0_prompt.txt, 1_raw_llm_output.txt, 1b_thinking_trace.txt,
            2_parsed_body.py, 3_eval_results.txt
    eval_failure/
        eval{PID}_{counter}_island{id}/
            (same files)
    parse_failure/
        eval{PID}_{counter}_island{id}/
            (same files, 2_parsed_body.py contains "PARSE_FAILED")
```

---

## Part 2: Analysis Scripts

### File structure

```
analysis/search_diagnostics/
    analyze_parse_failures.py     # Parse failure taxonomy
    analyze_eval_failures.py      # Eval failure categorization
    analyze_successes.py          # Success analysis (scores, function characteristics)
```

Each script:
- Takes a `debug_samples/` directory as input
- Reads its corresponding category folder
- Outputs a csv: `sample_id, category, detail`

---

### Script 1: `analyze_parse_failures.py`

Reads `debug_samples/parse_failure/`. For each sample, reads `1_raw_llm_output.txt` and classifies why parsing failed.

#### Step 1: LLM api call, parser_issue vs llm_generation_issue + diagnosis

Single api call per sample (combines detection + diagnosis to halve cost).

**Model**: configurable via `--model` cli argument, default `claude-sonnet-4-20250514`. Uses `litellm.completion()`.

**Prompt includes**:
1. The full `parse_llm_output()` function source (~230 lines from `evaluator.py:189-418`)
2. The raw LLM output (`1_raw_llm_output.txt`)

```python
prompt = f"""Here is my parser function that extracts priority function bodies from LLM output:

{parse_llm_output_source}

Here is a raw LLM output that my parser failed to extract a function from
(it returned empty body):

{raw_output}

Does this output contain a valid priority function body that my parser should
have been able to extract?

Answer in this exact format:
verdict: yes or no
explanation: <one paragraph explaining why, and if yes, which specific step
of the parser pipeline failed and what it should have done differently>
"""
```

- **yes -> `parser_issue`**: explanation goes in `detail` column (tells us what to fix in parser)
- **no -> `llm_generation_issue`**: proceed to Step 2 (heuristic classification)

**Implementation**: Uses `litellm.completion()` so the model is easily swappable (e.g. `claude-sonnet-4-20250514`, `gpt-4o`, `gemini/gemini-2.0-flash`). Model is a cli argument: `--model claude-sonnet-4-20250514`. Parse verdict from response. Run on all parse failures (likely small set, <100 samples, ~$0.10 cost).

#### Step 2: Heuristic classification of llm_generation_issue (no LLM)

Count `def` signatures in the raw output via regex:
```python
defs = re.findall(r'^\s*def\s+(\w+)\s*\(', raw_text, re.MULTILINE)
```
This works on raw text regardless of `<code>` tags, fences, or prose wrapping.

Then branch on `len(defs)`:

```
llm_generation_issue
├── no_function (len(defs) == 0)
│   Pure prose/rambling, no code at all.
│
├── one_function (len(defs) == 1)
│   Extract the code block containing the function (<code> tags or markdown fence).
│   Check for `return` statement, then try ast.parse() for further diagnosis.
│   │
│   ├── no_return_truncated
│   │   `'return ' not in code_block` + non_code_ratio <= 80%
│   │   function body itself was too long, got cut off before reaching return
│   │
│   ├── no_return_rambling
│   │   `'return ' not in code_block` + non_code_ratio > 80%
│   │   prose/description ate most of the token budget, no room left for return
│   │
│   └── syntax_error
│       `'return ' in code_block` but ast.parse(code_block) raises SyntaxError.
│       Detail column: the SyntaxError message + line number.
│       e.g. "unexpected EOF while parsing at line 12" (usually truncated)
│            "expected ':' at line 5" (actual syntax bug)
│            "invalid syntax at line 3" (general)
│
│       csv summary also includes top 5 most common SyntaxError messages
│       across all syntax_error samples (aggregated at the end of the script).
│
└── multi_function (len(defs) >= 2)
    Count priority prefixed names: priority_names = [d for d in defs if d.startswith("priority")]
    Check for solution patterns: {"evaluate", "solve", "main", "greedy"} & set(defs)
    │
    ├── full_solution
    │   Non priority defs include evaluate/solve/main/greedy.
    │   LLM misunderstood task, output entire program.
    │
    ├── multiple_versions
    │   len(priority_names) >= 2
    │   LLM generated multiple priority function variants.
    │
    ├── no_priority
    │   len(priority_names) == 0
    │   LLM generated helper functions but never wrote a priority function.
    │
    └── helpers_with_priority (len(priority_names) == 1, other defs are helpers)
        Try ast.parse() on each function separately to locate the error:
        ├── error_in_helper
        ├── error_in_priority
        └── both_ok (parser should have handled this, Step 1 LLM would flag it)
```

#### Heuristic details

- **def count**: `re.findall(r'^\s*def\s+(\w+)\s*\(', raw, re.MULTILINE)` on full raw text
- **code block extraction**: extract content from `<code>...</code>` or last ``` fence, for ast.parse() and `return` check
- **has return**: `'return ' in code_block`
- **non_code_ratio**: lines that are not code (no indent, no `def`, no `import`, no `#`) / total lines
- **near max_tokens**: `len(raw) / 3.5 > max_new_tokens * 0.9` (chars_per_token configurable)

#### csv output

Possible categories:
1. `parser_issue`, valid code, our parser failed (detail from LLM diagnosis)
2. `no_function`, no `def` at all, LLM rambled too much in text before generating code
3. `no_return_truncated`, 1 function, no return, function body too long
4. `no_return_rambling`, 1 function, no return, explanation too long to have budget for valid code
5. `syntax_error`, 1 function, has return, SyntaxError (detail = error message)
6. `full_solution`, multiple defs including evaluate/solve/main
7. `multiple_versions`, multiple `priority*` variants
8. `no_priority`, helper functions but no priority function
9. `error_in_helper`, helpers + priority, error is in a helper
10. `error_in_priority`, helpers + priority, error is in the priority function
11. `both_ok`, helpers + priority, all parse fine (parser should have caught it)

```csv
sample_id,category,detail
eval3775747_000003_island5,no_function,"0 def signatures, 847 chars, pure prose"
eval3775758_000007_island2,no_return_truncated,"1 def, no return, ~15800 tokens (99% of max), 3% non_code"
eval3775758_000009_island4,no_return_rambling,"1 def, no return, ~2100 tokens, 45% non_code"
eval3775747_000011_island1,syntax_error,"unexpected EOF while parsing at line 12"
eval3775747_000012_island3,syntax_error,"expected ':' at line 5"
eval3775790_000002_island3,parser_issue,"valid function in <code> tags, parser regex missed unclosed tag"
eval3775790_000005_island1,helpers_cropped,"2 defs [levenshtein_distance, priority], near max_tokens"
eval3775800_000004_island7,full_solution,"3 defs [priority, evaluate, main]"
```

#### Summary mode

Can run in two modes:
```bash
# Full run: classify + generate csv + print summary
python analyze_parse_failures.py logs_8B/debug_samples/

# Summary only: skip classification, just analyze existing csv
python analyze_parse_failures.py --csv logs_8B/debug_samples/parse_failures.csv
```

Summary reads the csv, counts per category, then shows relevant details for each:

```
=== Parse Failure Summary ===
Total: 47 samples

Category breakdown:
  parser_issue:         3 ( 6.4%)
  no_function:         12 (25.5%)
  no_return_truncated:  8 (17.0%)
  no_return_rambling:   5 (10.6%)
  syntax_error:        14 (29.8%)
  full_solution:        1 ( 2.1%)
  multiple_versions:    1 ( 2.1%)
  no_priority:          1 ( 2.1%)
  error_in_helper:      1 ( 2.1%)
  error_in_priority:    1 ( 2.1%)

parser_issue (3 samples):
All LLM diagnosed reasons:
  1. "parser regex missed unclosed <code> tag with nested markdown fence"
  2. "function named priority_v2 not matched by priority prefix detection"
  3. "indentation normalization failed on mixed tabs/spaces"

syntax_error (14 samples):
Top 5 SyntaxError messages:
  1. unexpected EOF while parsing  (8 samples)
  2. expected ':'                   (3 samples)
  3. invalid syntax                 (2 samples)
  4. unexpected indent              (1 sample)

error_in_helper (1 sample):
Helper function names involved:
  1. levenshtein_distance  (1 sample)
```

Only three categories get extra detail (the rest are self explanatory from their counts):
- `parser_issue`: all LLM reasons (actionable bugs to fix)
- `syntax_error`: top 5 error messages
- `error_in_helper`: which helper functions were involved

#### Thinking trace analysis

Runs unified thinking trace analysis (see [Thinking Trace Analysis](#thinking-trace-analysis-shared-across-all-scripts)).

Extra csv columns: `thinking_present,thinking_length,reasoning_score`

#### Description analysis (if `<description>` tags present)

Runs shared failure description analysis (see [Description Analysis, Failures](#description-analysis-failures-shared-for-parse-and-eval-failures)).

Extra csv columns: `description_present,description_length,description_score,description_matches_code,overambitious`

---

### Script 2: `analyze_eval_failures.py`

Reads `debug_samples/eval_failure/`. For each sample, reads `3_eval_results.txt` + `2_parsed_body.py` and categorizes the failure.

#### Step 1: Heuristic pre filter for timeout

Parse `3_eval_results.txt`, lines matching `FAILED` + `TIMEOUT` are classified as `timeout` directly (no LLM needed).
Also extract partial scores (inputs that passed before the timeout).

#### Step 2: LLM api call for all non timeout failures

One LLM call per sample (each error is independent). Api calls can run in parallel.

**Prompt includes**:
- The evaluation script (`specifications/ECC/evaluation/graph_networkx.py`) as context, this shows:
  - Available imports (`math.*`, `itertools`, `numpy`, `networkx`, `collections.Counter`)
  - Which nx methods have FastGraph wrappers (degree, neighbors, clustering, centrality, etc.)
  - The function signature (`def priority(node, G, n, s) -> float`)
- The error message + traceback from `3_eval_results.txt`
- The function body from `2_parsed_body.py`

**Prompt**:
```
Here is a function that was executed in a sandbox with the following
evaluation script:

---
{contents of graph_networkx.py}
---

The function failed at runtime. Here is the function body and the error:

Function body:
{contents of 2_parsed_body.py}

Error:
{error from 3_eval_results.txt}

Categorize this error into exactly one of the following categories:

1. output_type_error, function returns wrong type instead of float
   (detail: what type was returned, e.g. None, tuple, list, str)
2. wrong_variable, uses variables not in the function signature
   (detail: which variable names)
3. import_incorrect_call, calls a function incorrectly, e.g. log() instead
   of math.log() (detail: which function)
4. import_missing_package, imports a package not available in the sandbox
   (detail: which package)
5. hallucinated, calls a method/attribute/function that doesn't exist,
   e.g. non existent method on an object or undefined helper function
   (detail: which method/function)
6. incorrect_method_usage, calls a real method with wrong arguments or on
   wrong type (detail: which method)
7. wrapper_violation, uses graph library methods not exposed in the evaluation
   wrapper, e.g. raw nx.* calls (detail: which method)
8. other, anything not fitting the above (detail: brief description)

Output exactly one line:
category | detail
```

The LLM can also flag new categories it discovers, these go in the csv as is and show up in the summary.

#### Categories

1. `timeout`, exceeded sandbox time limit (heuristic, no LLM)
2. `output_type_error`, returns wrong type instead of float (detail: which type)
3. `wrong_variable`, uses variables not in function signature (detail: which variables)
4. `import_incorrect_call`, e.g. `log()` instead of `math.log()` (detail: which function)
5. `import_missing_package`, package not in sandbox (detail: which package)
6. `hallucinated`, method/attribute doesn't exist (detail: which method)
7. `incorrect_method_usage`, real method, wrong args/type (detail: which method)
8. `wrapper_violation`, graph library method not in wrapper (detail: which method)
9. `other`, anything else (detail: brief description)
10. *(LLM may add new categories)*

#### csv output

```csv
sample_id,category,detail
eval3775747_000006_island1,import_missing_package,"binomial, not available in sandbox"
eval3775747_000008_island0,timeout,"(11,2): exceeded 30s | partial: (7,2)=5 (8,2)=7 (9,2)=9 (10,2)=14"
eval3775747_000010_island1,timeout,"(8,2): exceeded 30s | partial: (7,2)=5"
eval3775758_000003_island2,hallucinated,"G.node_degree(), no such method on nx.Graph"
eval3775758_000007_island5,output_type_error,"missing else clause in if/elif chain"
eval3775790_000002_island3,wrong_variable,"uses 'graph' instead of 'G'"
eval3775800_000001_island6,wrapper_violation,"nx.shortest_path(), not exposed in wrapper"
```

#### Summary mode

Same as parse failures, `--csv` flag to skip classification and just analyze existing csv.

```
=== Eval Failure Summary ===
Total: 184 samples

Category breakdown:
  timeout:                150 (81.5%)
  output_type_error:        4 ( 2.2%)
  wrong_variable:           5 ( 2.7%)
  import_incorrect_call:    1 ( 0.5%)
  import_missing_package:  12 ( 6.5%)
  hallucinated:             8 ( 4.3%)
  incorrect_method_usage:   3 ( 1.6%)
  wrapper_violation:        1 ( 0.5%)
  other:                    0 ( 0.0%)

1. timeout (150 samples):
Score distribution before timeout (per n):
  n=7:  150 samples | mean=5.0  median=5  min=4  max=5
  n=8:  150 samples | mean=6.8  median=7  min=5  max=8
  n=9:  145 samples | mean=9.2  median=9  min=7  max=12
  n=10: 103 samples | mean=13.1 median=14 min=9  max=16
  n=11:  60 samples | mean=20.4 median=21 min=15 max=26
  n=12:   0 samples | (all timed out before reaching n=12)
  (only counts samples that completed that n before timing out)

2. output_type_error (4 samples):
  (just count, detail in csv)

3. wrong_variable (5 samples):
Top 5 variables:
  1. graph (should be G)          (2 samples)
  2. string (should be node)      (1 sample)
  3. num_nodes (not in sig)       (1 sample)
  4. alphabet_size (should be q)  (1 sample)

4. import_incorrect_call (1 sample):
Top calls:
  1. log() instead of math.log()  (1 sample)

5. import_missing_package (12 samples):
Top 5 packages:
  1. scipy.special (binomial)    (5 samples)
  2. numpy                        (3 samples)
  3. itertools (combinations)     (2 samples)
  4. networkx                     (1 sample)
  5. sympy                        (1 sample)

6. hallucinated (8 samples):
Top 5 methods/functions:
  1. G.node_degree()              (3 samples)
  2. G.get_neighbors_of()         (2 samples)
  3. node.hamming_weight()        (1 sample)
  4. G.adjacency_list()           (1 sample)
  5. G.shortest_path_to()         (1 sample)

7. incorrect_method_usage (3 samples):
Top 3 methods:
  1. G.neighbors() with 2 args   (2 samples)
  2. len() on generator           (1 sample)

8. wrapper_violation (0 samples)

9. other (0 samples)
```

Detail sections shown for categories where top N instances are useful:
- `timeout`: score distribution per n for inputs that completed before timeout
- `output_type_error`: just count
- `wrong_variable`: which variables
- `import_incorrect_call`: which functions called incorrectly
- `import_missing_package`: which packages
- `hallucinated`: which methods/functions
- `incorrect_method_usage`: which methods
- `wrapper_violation`, `other`: just counts

#### Thinking trace analysis

Runs unified thinking trace analysis (see [Thinking Trace Analysis](#thinking-trace-analysis-shared-across-all-scripts)).

Extra csv columns: `thinking_present,thinking_length,reasoning_score`

#### Description analysis (if `<description>` tags present)

Runs shared failure description analysis (see [Description Analysis, Failures](#description-analysis-failures-shared-for-parse-and-eval-failures)).

Extra csv columns: `description_present,description_length,description_score,description_matches_code,overambitious`

---

### Script 3: `analyze_successes.py`

Reads `debug_samples/success/`. For each sample, reads `2_parsed_body.py` for code structure stats.

#### Step 1: Code structure metrics (heuristic, no LLM)

Per sample metrics from `2_parsed_body.py`:
- **code_lines**: number of non empty lines
- **has_helper**: whether the parsed body contains a nested `def`
- **num_helpers**: count of nested `def` statements
- **uses_numpy**: whether the body references `np.`
- **uses_nx**: whether the body references `nx.` or `G.`

#### Step 2: Strategy labeling (LLM api call)

Batch all successful function bodies into LLM calls, ~20-30 per call (batched so strategy labels stay consistent across samples).

**Prompt**:
```
Here are N priority functions that were successfully evaluated for an
error correcting code construction problem. Each function takes
(node, G, n, s) and returns a float priority score used in a greedy
independent set algorithm.

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

Output as: sample_id | strategy_label | description | info_type
Only output the description once per unique label (on first occurrence).
```

This discovers what strategies the evolutionary search converges on.

#### csv output

```csv
sample_id,code_lines,has_helper,num_helpers,uses_numpy,uses_nx,strategy,strategy_description,info_type
eval3775747_000001_island2,3,False,0,False,True,inverse degree,"Prioritizes nodes with low graph degree",graph_only
eval3775747_000003_island6,12,True,1,True,True,neighbor similarity weighted,"Scores by character overlap with neighbors",both
eval3775758_000004_island4,5,False,0,False,False,hamming weight ratio,"Ranks by ratio of 1 bits in the codeword",sequence_only
```

#### Summary

```
=== Success Summary ===
Total: 163 samples

Code structure:
  Avg code lines:     8.3  (min=2, max=45, median=6)
  With helper funcs:  18 (11.0%)  avg helpers: 1.3
  Uses numpy:         23 (14.1%)
  Uses nx/G methods:  148 (90.8%)

Information type:
  graph_only:     72 (44.2%)
  sequence_only:  28 (17.2%)
  both:           63 (38.7%)

Unique strategies: 22

Top 10 strategies:
  1. inverse degree (52 samples, 31.9%) [graph_only]
     Prioritizes nodes with low graph degree (few neighbors)
  2. neighbor counting (35 samples, 21.5%) [graph_only]
     Counts number of neighbors satisfying a distance condition
  3. hamming weight ratio (22 samples, 13.5%) [sequence_only]
     Ranks by ratio of 1 bits in the codeword to degree
  4. combined degree + similarity (18 samples, 11.0%) [both]
     Combines inverse degree with character overlap scoring
  5. string overlap scoring (12 samples, 7.4%) [both]
     Scores by substring or character set overlap with neighbors
  6. clustering coefficient (8 samples, 4.9%) [graph_only]
     Uses local clustering coefficient as priority
  7-10. ...
  (12 other strategies with <=3 samples each)
```

#### Thinking trace analysis

Runs unified thinking trace analysis (see [Thinking Trace Analysis](#thinking-trace-analysis-shared-across-all-scripts)).

Extra csv columns: `thinking_present,thinking_length,reasoning_score`

#### Description analysis (if `<description>` tags present)

Runs success description analysis (see [Description Analysis, Successes](#description-analysis-successes)).

Extra csv columns: `description_present,description_length,description_score,description_matches_code`

---

## Thinking Trace Analysis (shared across all scripts)

Each script reads `1b_thinking_trace.txt` (if present) and scores the reasoning quality **blind**, the LLM does not know whether the sample succeeded or failed, to avoid biasing scores.

#### LLM scoring (one call per sample, same prompt for all categories)

**Model**: configurable via `--model`, uses `litellm.completion()`.

**Context**: Each sample has `0_prompt.txt` (the actual prompt the sampler LLM received, including few-shot examples, reflection/thought instructions, etc.). This is included so the analysis LLM sees exactly what the sampler was asked.

One LLM call per thinking trace (no batching needed since each score is independent, no cross-sample consistency required).

**Prompt** (does not mention the outcome):
```
You are evaluating a thinking trace from an LLM that was given the following
prompt:

---
{contents of 0_prompt.txt}
---

Here is the thinking trace:

---
{contents of 1b_thinking_trace.txt}
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
score | one_sentence_explanation
```

Columns added to main csv: `thinking_present,thinking_length,reasoning_score`

#### Summary section (added to each script's summary)

```
Thinking Trace Analysis:
Traces present: 120/163 (73.6%)
Avg trace length: 850 chars (median=720, min=45, max=3200)

Reasoning score distribution:
  5 (excellent):   18 (15.0%)
  4 (coherent):    42 (35.0%)
  3 (repetitive):  30 (25.0%)
  2 (derailed):    20 (16.7%)
  1 (confused):    10 ( 8.3%)
  Mean: 3.3  Median: 4
```

The interesting analysis is comparing this distribution across categories (parse_failure vs eval_failure vs success) and across runs (8B vs 14B vs 32B).

---

## Description Analysis (shared across all scripts)

All 3 scripts use the same prompt for description analysis. The descriptions can be either algorithmic summaries ("describe your new heuristic") or comparative reflections ("explain why the second outperforms the first"), depending on which prompt template was used during the search. Including `0_prompt.txt` as context lets the analysis LLM understand what kind of description was requested.

**Input per sample**: `0_prompt.txt` (the original prompt) + `1_raw_llm_output.txt` with `<think>...</think>` stripped out (so the LLM sees the description tag and the generated code together).

**Model**: configurable via `--model`, uses `litellm.completion()`.

One LLM call per sample (no batching needed since each score is independent, no cross-sample consistency required).

**Prompt**:
```
You are analyzing an output from an LLM that was given the following prompt:

---
{contents of 0_prompt.txt}
---

Here is the LLM's output (description + code):

---
{contents of 1_raw_llm_output.txt, with <think>...</think> stripped}
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
   "partial" = right general idea but missing key details or exaggerates
   "no" = description and code are about different approaches

3. overambitious: yes / no
   Was the described approach too complex for the LLM to implement correctly?
   (e.g. describes a multi step algorithm but code is incomplete/broken,
   mentions advanced techniques that the LLM couldn't translate to code)

Output exactly one line:
description_score | description_matches_code | overambitious
```

#### csv columns (added to each script's main csv)

```
...,description_present,description_length,description_score,description_matches_code,overambitious
```

#### Summary section (added to each script's summary)

```
Description Analysis:
Descriptions present: 110/163 (67.5%)

Description score distribution:
  5 (novel):       25 (22.7%)
  4 (concrete):    40 (36.4%)
  3 (vague):       28 (25.5%)
  2 (superficial): 12 (10.9%)
  1 (buzzwords):    5 ( 4.5%)
  Mean: 3.6  Median: 4

Description matches code:
  yes:     65 (59.1%)
  partial: 30 (27.3%)
  no:      15 (13.6%)

Overambitious: 18/110 (16.4%)
```

---

## Usage

```bash
# Analyze a single run
python analysis/search_diagnostics/analyze_parse_failures.py logs_8B/debug_samples/
python analysis/search_diagnostics/analyze_eval_failures.py logs_8B/debug_samples/
python analysis/search_diagnostics/analyze_successes.py logs_8B/debug_samples/

# Each produces csvs in the debug_samples directory:
#   logs_8B/debug_samples/parse_failures.csv           (main)
#   logs_8B/debug_samples/parse_failures_thinking.csv   (thinking scores)
#   logs_8B/debug_samples/eval_failures.csv
#   logs_8B/debug_samples/eval_failures_thinking.csv
#   logs_8B/debug_samples/successes.csv
#   logs_8B/debug_samples/successes_thinking.csv
```

---

## Implementation Order

1. `analyze_eval_failures.py`, simplest, pure regex parsing of 3_eval_results.txt
2. `analyze_successes.py`, straightforward score extraction
3. `analyze_parse_failures.py`, most complex, heuristics + optional LLM call for parser_issue detection
