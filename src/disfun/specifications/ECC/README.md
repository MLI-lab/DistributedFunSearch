# ECC Specification

Error correcting codes specification for DistributedFunSearch. Evolves a `priority` function that guides greedy independent set construction on code graphs.

Use this as a reference for creating your own specification directory.

## Directory structure

```
your_spec/
├── evaluation/              # Required. Scoring logic.
│   └── evaluate.py          # Must define evaluate(params, graph_dir) and priority(...)
├── initial_functions/       # Required. Seed functions to start evolution.
│   └── zero.txt             # Wrapped in <code>...</code>, optionally <description>...</description>
├── imports/                 # Required. Imports available to the LLM generated function.
│   └── base.txt             # e.g., "import numpy as np"
├── docstrings/              # Required. Docstring templates for few-shot examples.
│   ├── baseline.txt         # For the first/worse function. Supports {score} placeholder.
│   └── improved.txt         # For the better function. Supports {version} and {score}.
├── funsearch/               # Strategy specific. Only needed if using funsearch strategy.
│   ├── templates/           # Prompt templates with placeholders like {fewshot_examples}
│   ├── problem_descriptions/
│   └── system_messages/     # Optional system message for the LLM
├── eoh/                     # Only needed if using eoh strategy.
├── reevo/                   # Only needed if using reevo strategy.
└── variants.py              # Optional. Parameterizes templates for problem variants.
```

## Required files

### Evaluation script

The evaluation script is the core of your specification. It must define:

- `evaluate(params, graph_dir)` returning `(score, hash_value)`. The score is a number (higher is better). The hash is used for deduplication, return `None` to skip.
- `priority(...)` as a placeholder with `pass` as the body. The system replaces it with LLM generated code.

See `evaluation/graph_fastgraph.py` for the graph variant or `evaluation/no_graph_deletions.py` for a simpler version without graphs.

### Initial functions

At least one `.txt` file in `initial_functions/`. This seeds the database before the LLM generates anything.

```
<description>
Baseline that assigns equal priority to all nodes.
</description>

<code>
def priority(node, G, n, s) -> float:
    return 0.0
</code>
```

The function signature here determines what gets extracted as `{function_signature}` in templates. The `<description>` tag is optional.

### Imports

A `.txt` file listing imports available to the priority function. These are prepended to few-shot examples so the LLM knows what libraries it can use.

### Docstrings

Templates for the docstrings inserted into few-shot examples:

- `baseline.txt` is used for the first (or only) function shown to the LLM.
- `improved.txt` is used for subsequent, better scoring functions.

Both support `{score}` (replaced with formatted scores when `show_scores` is enabled) and `improved.txt` supports `{version}` (replaced with the version number).

## Strategy specific files

You only need files for the strategy you configure in `PromptConfig.strategy`.

### FunSearch

Needs a template and problem description. Templates use placeholders like `{fewshot_examples}`, `{function_header}`, `{imports}`, `{problem_description}`. See `funsearch/templates/` for examples.

Single turn templates send one prompt and get code back directly. They go in a single `.txt` file or a folder with `default.txt` and optionally `single.txt` (used when only one example is available). Multi turn templates split the generation into two LLM calls: `stage1.txt` asks the LLM to reason or reflect, then `stage2.txt` uses that reasoning to generate code.

### EoH

Needs style templates in `eoh/styles/`, a problem description, function description, and system message.

### ReEvo

Needs templates for each phase (`seed`, `crossover`, `mutation`, `reflect_st`, `reflect_lt`), problem and function descriptions, system messages for generator and reflector, and optionally an initial reflection.

## Variants

`variants.py` defines a `VARIANTS` dict mapping variant names to placeholder values. These replace `{string_type}`, `{edge_condition}`, etc. in templates and descriptions, allowing one specification to serve multiple problem types.

## Config

Point your config at the specification:

```python
EvaluatorConfig(
    evaluation_script_path="path/to/your_spec/evaluation/evaluate.py",
    initial_functions_dir="path/to/your_spec/initial_functions/graph",
)

PromptConfig(
    strategy="funsearch",
    spec_dir="path/to/your_spec",
    imports_file="imports/base.txt",
    funsearch_template="funsearch/templates/your_template.txt",
    funsearch_problem_desc="funsearch/problem_descriptions/your_desc.txt",
)
```
