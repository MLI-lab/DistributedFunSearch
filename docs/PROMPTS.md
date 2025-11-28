# Prompt Construction

DistributedFunSearch uses a template-based system for flexible prompt construction.

## Template System Overview

Prompts are built using templates with `{placeholder}` syntax:
1. **Template file** defines order/structure of components
2. **Placeholders** map to file paths (loaded once at init into memory)
3. **Directory placeholders** load all files at init, sample from memory at runtime
4. **Reserved placeholders** are computed dynamically (fewshot examples, version, etc.)

## File Structure

```
specifications/Deletions/
├── templates/
│   ├── funsearch.txt       # FunSearch-style code completion
│   └── eoh.txt             # EoH-style instruction-based
├── problem_descriptions/
│   └── baseline.txt        # Problem docstring
├── prompt_styles/
│   ├── funsearch.txt
│   ├── eoh_minimal.txt
│   └── eoh_full/           # Directory, sample one at runtime
│       ├── e1.txt
│       ├── e2.txt
│       ├── fs1_m1.txt      # fs{N}_ prefix overrides fewshot count
│       └── ...
├── imports/
│   └── networkx.txt
├── evaluation/
│   ├── graph_networkx.py   # Evaluation script (optional in prompt)
│   ├── graph_gt.py
│   └── no_graph.py
└── components/
    ├── fewshot_preamble.txt
    └── evaluation_preamble.txt  # "This is how the priority function gets evaluated:"
```

## Example Templates

**FunSearch-style** (`templates/funsearch.txt`):
```
{problem_description}
{evaluation_preamble}
{evaluation_script}
{prompt_style}
{imports}
{fewshot_examples}
{function_header}
```

**EoH-style** (`templates/eoh.txt`):
```
{problem_description}
{fewshot_preamble}
{fewshot_examples}
{prompt_style}
{inout_spec}
```

## Reserved Placeholders

These are computed at runtime, not loaded from files:

| Placeholder | Description |
|-------------|-------------|
| `{fewshot_examples}` | Built from sampled programs (includes scores if enabled) |
| `{num_examples}` | Count of fewshot examples |
| `{version}` | `num_examples + 1` |
| `{function_header}` | `def priority_v{version}({args}):` where args/return type extracted from `initial_functions_dir` |
| `{inout_spec}` | Input/output specification with args from `initial_functions_dir` |


## Problem Descriptions

Located in `specifications/{Task}/problem_descriptions/`:

| File | Description |
|------|-------------|
| `baseline.txt` | Generic problem description |
| `explicit_s1.txt` | States "s = 1" explicitly |
| `string_properties.txt` | Hints about string properties |

## Prompt Styles

Located in `specifications/{Task}/prompt_styles/`:

| File/Directory | Description |
|----------------|-------------|
| `None` | Pure code completion (no instructions) |
| `starcoder2.txt` | Minimal "Improve..." instruction |
| `funsearch.txt` | "Improve..." + `<code>` tags |
| `eoh_minimal.txt` | "Improve..." + `<thought>` + `<code>` |
| `extended_eoh.txt` | "Improve..." + `<thinking>` + `<thought>` + `<code>` |
| `eoh_full/` | Original EoH prompt styles (directory, samples one) |

### EoH Operators (`eoh_full/`)

From the EoH paper (Liu et al., ICML 2024):
- Source: https://github.com/FeiLiu36/EoH/blob/main/eoh/src/eoh/methods/eoh/eoh_evolution.py

| File | Description |
|------|-------------|
| `e1.txt` | Create something completely different |
| `e2.txt` | Identify patterns, add novel change |
| `fs1_m1.txt` | Targeted improvements (1 fewshot) |
| `fs1_m2.txt` | Vary parameters in fewshot example (1 fewshot) |
| `fs1_m3.txt` | Remove redundant parts (1 fewshot) |

**Fewshot Override**: Files with `fs{N}_` prefix use N examples instead of config default.

## Evaluation Code in Prompt

Optionally include the evaluation script in the prompt so the LLM can see how the priority function is used:

```python
placeholders: dict = {
    # ... other placeholders ...
    "evaluation_preamble": ".../components/evaluation_preamble.txt",
    "evaluation_script": ".../evaluation/graph_networkx.py",
}
```

When both are set:
- `evaluation_preamble` provides intro text: "This is how the priority function gets evaluated:"
- `evaluation_script` content is wrapped in \`\`\`python code 

When both are `None` (default), nothing appears in the prompt.

## Score Display

When `show_eval_scores=True`, scores are added to fewshot docstrings. Configure in `PromptConfig`:
- `display_mode`: `"absolute"` or `"relative"`
- `absolute_label` / `relative_label`: Prefix text for scores
- `best_known_solutions`: Required for relative mode

## Creating Custom Templates

1. Create template file with `{placeholder}` syntax
2. Create files for each placeholder
3. Configure paths in `PromptConfig`

