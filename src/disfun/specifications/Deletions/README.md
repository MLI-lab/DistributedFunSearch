# Deletion-Correcting Codes Specification

Find large codes where codewords have no common subsequence of length ≥ n-s.

## Problem

- **Nodes:** Binary strings of length n
- **Edges:** Two nodes connected if `LCS(node1, node2) ≥ n - s`
- **Goal:** Find maximum independent set (valid deletion-correcting code)

## Directory Structure

```
Deletions/
├── imports/                        # Import statements (shared)
│   ├── graph_tool.txt
│   ├── networkx.txt
│   └── no_graph.txt
├── initial_functions/              # Seed/baseline functions (shared)
│   ├── graph_gt/zero.txt
│   ├── graph_networkx/zero.txt
│   └── no_graph/zero.txt
├── evaluation/                     # Evaluation scripts (shared)
│   ├── graph_gt.py
│   ├── graph_networkx.py
│   └── no_graph.py
├── docstrings/                     # Docstring templates (shared)
│   ├── baseline.txt                # For single/worse function with {score}
│   ├── improved.txt                # For better function with {score}
│   ├── score_label_absolute.txt    # Label for absolute scores
│   └── score_label_relative.txt    # Label for relative scores
│
├── funsearch/                      # Baseline FunSearch strategy
│   ├── template.txt                # Main prompt template
│   ├── problem_descriptions/       # Task descriptions
│   │   ├── baseline.txt
│   │   ├── explicit_s1.txt
│   │   ├── graph_gt.txt
│   │   └── string_properties.txt
│   ├── components/
│   │   └── fewshot_preamble.txt
│   └── system_message.txt
│
├── eoh/                            # Evolution of Heuristics strategy
│   ├── styles/                     # Complete prompt templates (randomly sampled)
│   │   ├── i1.txt                  # Initialization: improve over baseline
│   │   ├── e1.txt                  # Crossover: totally different from examples
│   │   ├── e2.txt                  # Crossover: motivated from examples
│   │   ├── m1.txt                  # Mutation: modified version
│   │   ├── m2.txt                  # Mutation: different parameters
│   │   └── m3.txt                  # Mutation: simplify/generalize
│   ├── problem_descriptions/
│   │   └── (same as funsearch)
│   ├── func_desc.txt               # Function description
│   └── system_message.txt
│
└── reevo/                          # ReEvo strategy (reflection-based)
    ├── templates/
    │   ├── seed.txt                # Initialization
    │   ├── crossover.txt           # Crossover generation
    │   ├── mutation.txt            # Mutation generation
    │   ├── reflect_st.txt          # Short-term reflection
    │   ├── reflect_lt.txt          # Long-term reflection
    │   └── user_generator.txt      # Task header
    ├── problem/
    │   ├── problem_desc.txt        # Inline problem description
    │   └── func_desc.txt           # Function description
    └── system/
        ├── generator.txt           # System prompt for code generation
        └── reflector.txt           # System prompt for reflection
```

## How Prompt Building Works

Prompt building works by selecting a template and dynamically filling its placeholders with content sampled from the program database. Each strategy has a different approach:

**FunSearch** (for code completion models like StarCoder2): Uses a single template that frames the task as code completion. Few-shot examples (default 2) are sampled from the program database and shown as versioned functions (`priority_v0`, `priority_v1`), with the last version as the target for improvement. The prompt ends with a function header (`def priority_v2(...):`) for the model to complete.

**EoH** (for instruction models like GPT, Claude): Has 6 style templates, 1 initialization (i1), 2 crossover (e1, e2), and 3 mutation (m1, m2, m3). At each iteration, one style is randomly selected. Crossover styles show 2 functions and ask for something totally different (e1) or motivated from the examples (e2). Mutation styles show 1 function and ask for modifications.

**ReEvo** (for instruction models like GPT, Claude): Uses a phase-based approach with reflection:
- Crossover phase: Generate new function from 2 sampled functions, then short-term reflection comparing worse vs better code (2 LLM calls per iteration)
- Mutation phase: Mutate the best heuristic, then update long-term reflection based on all short-term reflections from crossover phase (2 LLM calls per iteration)


## Placeholder Reference

The function to be evolved is always named `priority` (hardcoded in templates, not a placeholder). All templates use placeholders that are filled at runtime:

### Sampled Code Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{better_code}` | Higher-scoring / single / seed function. |
| `{worse_code}` | Lower-scoring function for comparison. |
| `{thought}` | Sampled thought (the `<thought>` from a previously generated function). Used in eoh m1/m2. |

### Docstring Templates

Function docstrings are loaded from template files in `docstrings/`:

| File | Usage | Content |
|------|-------|---------|
| `baseline.txt` | Single function or worse function | `Returns the priority... {score}` |
| `improved.txt` | Better function | `Improved version of priority_v{version}. {score}` |
| `score_label_absolute.txt` | Label for absolute scores | `Scores (format...):` |
| `score_label_relative.txt` | Label for relative scores | `Relative to baseline...:` |

**Placeholders in docstring templates:**
- `{score}` - Replaced with formatted scores when `show_scores=True`, or removed when `show_scores=False`.
- `{version}` - Replaced with the previous version number (0, 1, 2, etc.). Used consistently across all strategies.

**Score format** depends on `score_display_mode`:
- `"absolute"`: `Scores (format...): {(6, 1, 2): 10, (7, 1, 2): 16}`
- `"relative"`: `Relative to baseline...: {(6, 1, 2): +0.0%, (7, 1, 2): +5.0%}`

**Example docstring with scores:**
```python
def priority(node, G, n, s) -> float:
    """Returns the priority with which we want to add `node` to the independent set.
    Relative to baseline (format (n, s, q): improvement%): {(6, 1, 2): +0.0%, (7, 1, 2): +5.0%}"""
```

### Content Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{problem_description}` | Task description. Used in funsearch, eoh. |
| `{problem_desc}` | Short task description (inline text). Used in reevo. |
| `{func_desc}` | Function inputs/outputs/behavior description. Used in eoh, reevo. |
| `{user_generator}` | Task header from user_generator.txt. Used in reevo. |
| `{imports}` | Import statements. Used in all strategies. |

### ReEvo Reflection Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{reflection}` | Short-term reflection result. |
| `{prior_reflection}` | Previous long-term reflection. |
| `{new_reflections}` | Newly gained short-term reflections. |
| `{initial_reflection}` | Initial hints (can be empty). |

### FunSearch-Specific Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{evaluation_preamble}` | Explanation text before evaluation script. |
| `{evaluation_script}` | The evaluation code. |
| `{function_header}` | Function signature (e.g., `def priority_v2(...):`) |

## Strategy Comparison

| Strategy | Template Selection | Fewshot | Output Format |
|----------|-------------------|---------|---------------|
| **funsearch** | Fixed template | 2  | Code only (completion) |
| **eoh** | Random from styles/ | i1: 1, e1/e2: 2, m1/m2/m3: 1 | `<thought>...</thought><code>...</code>` |
| **reevo** | Phase-based | crossover: 2, mutation: 1, seed: 1 | ` ```python ... ``` ` |

