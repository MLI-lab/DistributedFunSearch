# Prompt Construction 

DistributedFunSearch uses a modular setup where problem description, prompt format and evaluation are separated. 

Prompts are built by combining three independent components:

**1. Problem Description** (what LLM sees)
- Docstring explaining the problem and constraints
- Visible imports (graph libraries shown or hidden depending on variant)
- Uses `{version}` placeholder that gets replaced with actual version number
- Located in `specifications/{Task}/problem_descriptions/`
- Example: `baseline.txt` contains just the docstring and imports 

**2. Prompt Format** (how to structure output)
- Instructions for how the LLM should structure its response
- Whether to output code only, code with reasoning, or code with thinking
- Which XML tags to use (<thinking>, <thought>, <code>)
- Located in `specifications/{Task}/prompt_styles/`

**Evaluation Scripts** (execution logic, not shown to LLM except if included in problem description)
- Complete evaluation functions (load_graph, solve, evaluate, priority)
- Controls which parameters the priority function receives
- Located in `specifications/{Task}/evaluation/`
- Example: `deletion_codes.py` for deletion-correcting codes

These components are loaded and combined at runtime to create the full prompt sent to the LLM.

## Available Options

### Problem Descriptions

Located in `src/disfun/specifications/Deletions/problem_descriptions/`:

**baseline.txt**
- Generic problem description using graph-tool (faster than NetworkX)
- "Nodes are connected if they share a subsequence of length at least n-s"
- Shows `import graph_tool.all as gt` to LLM (allows graph operations)
- Passes graph to priority: `priority(node, G_gt, node_to_vertex, vertex_to_node, n, s)`
  - `node_to_vertex`: Maps node strings (e.g., "0110") to graph-tool vertex indices
  - `vertex_to_node`: Maps graph-tool vertex indices back to node strings
  - Needed because graph-tool uses integer indices internally

**explicit_s1.txt**
- Same as baseline but explicitly states "s = 1" in the problem statement
- Helps the model understand the specific constraint value
- Uses graph-tool, passes graph to priority function

**no_graph.txt**
- Hides graph library imports from LLM (inside `load_graph` function)
- Does not pass graph to priority function: `priority(node, n, s)`
- Forces LLM to use only string properties (no graph operations available)

**string_properties.txt**
- Like baseline (shows graph-tool import, passes graph to priority function)
- Adds explicit hint: "Consider properties of the q-ary string `node`, such as specific patterns, the number of each symbol, or other unique features, to calculate the priority."
- Guides the model to consider string-level features even when graph is available
- Priority signature: `priority(node, G_gt, node_to_vertex, vertex_to_node, n, s)`

### Prompt Formats

Located in `src/disfun/specifications/Deletions/prompt_styles/`:

**None** (Code Completion)
- No prompt style file, pure code completion
- For base models like StarCoder, Llama base (not instruction-tuned)
- Model sees problem + examples + function header, completes naturally
- Set `prompt_style_path=None` in config
- Example output:
  ```python
  return G.degree(node)
  ```

**funsearch.txt** (FunSearch)
- Code only with minimal instructions
- For instruction-tuned models like GPT, Claude
- Tells model to return only function body
- Example output:
  ```python
  return G.degree(node)
  ```

**eoh.txt** (Evolution of Heuristics)
- One-sentence algorithm description + code
- Format:
  ```
  <thought>
  Prioritize nodes with lower degree to reduce conflicts
  </thought>

  <code>
  return -G.degree(node)
  </code>
  ```

**extended_eoh.txt** (Extended EoH)
- Full chain-of-thought reasoning + one-sentence summary + code
- Format:
  ```
  <thinking>
  Analysis of graph structure...
  Why this approach works...
  Generalization considerations...
  </thinking>

  <thought>
  Prioritize nodes with lower degree
  </thought>

  <code>
  return -G.degree(node)
  </code>
  ```

## Configuration

You can configure the prompt in `config.py`:

```python
from disfun.experiments.experiment1.config import Config, EvaluatorConfig, PromptConfig

config = Config(
    # Evaluation script (how code is executed)
    evaluator=EvaluatorConfig(
        # Uses absolute path: defaults to deletion_codes.py
        # Override with: evaluation_script_path="/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/evaluation/my_eval.py"
        s_values=[1, 2],
        start_n=[6, 7],
        end_n=[10, 12]
    ),

    # Prompt construction (what LLM sees and how to format output)
    prompt=PromptConfig(
        # What content the LLM sees (absolute path)
        # Defaults to baseline.txt, override with:
        # problem_description_path="/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/problem_descriptions/explicit_s1.txt"
        # Options: "baseline.txt", "explicit_s1.txt", "no_graph.txt", "string_properties.txt"

        # How to format output (absolute path or None)
        prompt_style_path=None,  # None for code completion (StarCoder)
        # Or use: prompt_style_path="/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/prompt_styles/funsearch.txt"
        # Options: None, "funsearch.txt", "eoh.txt", "extended_eoh.txt"

        # Few-shot configuration
        fewshot_num_examples=3,
        fewshot_show_thinking=False,  # Show full reasoning (extended_eoh)
        fewshot_show_thought=True,    # Show algorithm descriptions (eoh)
        fewshot_show_code=True        # Show implementations
    )
)
```

## Score Display in Prompts

Control whether evaluation scores are shown in few-shot examples.

**Basic usage:**

```python
from disfun.config import Config, PromptConfig

config = Config(
    prompt=PromptConfig(
        show_eval_scores=True,  # Show scores (default: False)
        display_mode="absolute"  # "absolute" or "relative"
    )
)
```

**Absolute mode** shows raw scores: `(7, 2): 5, (8, 2): 7`

**Relative mode** shows percentage improvement over baseline: `(7, 2): +20%, (8, 2): +14%`
- Requires `best_known_solutions` dictionary with baseline scores
- Formula: `(Score_ours - Score_baseline) / |Score_baseline| × 100%`

## How Prompts Are Built

When the ProgramsDatabase generates a prompt:

1. **Load specification files** (once per run)
   - Load `problem_descriptions/{problem_description}.txt` - Problem context, imports, docstring
   - Load `prompt_styles/{style}.txt` - Output format instructions (empty for code completion mode)

2. **Build few-shot examples** (each prompt)
   - Sample high-performing programs from database
   - Format according to `PromptConfig.fewshot_*` settings
   - Include thinking/thought/code based on config

3. **Construct final prompt**

   The prompt is assembled in this order:

   ```
   [1. Problem description with imports]
   """
   Finds large independent set in graph G where nodes are q-ary strings of length n.
   Nodes in G are connected if they share a subsequence of length at least n-s.

   Improve the `priority_v{version}` function over its previous versions below.
   Keep the code short and comment for easy understanding.
   """

   import itertools
   import hashlib
   import numpy as np
   import graph_tool.all as gt

   [2. Few-shot examples - if any]
   # Previous implementations:
   Example 1:
   <thought>
   Algorithm description from stored program
   </thought>
   <code>
   Function body from stored program
   </code>

   Example 2:
   ...

   [3. Prompt style instructions - placed BEFORE function header]
   Return only the function body as valid Python code, without the function header.
   Do not include code block markers such as ```python or ```.

   [4. Function header to complete]
   def priority_v5(node, G_gt, node_to_vertex, vertex_to_node, n, s):
       [Model completes here]
   ```

   **For code completion models** (StarCoder, Llama base), step 3 is skipped (prompt_style=None),
   so the model sees only parts 1, 2, 4 and completes the function naturally.

   **For instruction-tuned models** (GPT, Claude), instructions appear before the function header
   to clearly frame the task while maintaining code completion context.

## Parsing LLM Output

The evaluator automatically parses LLM responses based on XML tags:

- XML tag extraction via regex
- AST parsing for code validation
- Falls back to code-only if tags missing


**Extended EoH format:**
```python
# Extracts all three components
<thinking>Full reasoning...</thinking>
<thought>Algorithm summary</thought>
<code>return value</code>

# Stored in Function object:
function.thinking = "Full reasoning..."
function.thought = "Algorithm summary"
function.body = "return value"
```

**EoH format:**
```python
# Extracts thought and code
<thought>Algorithm summary</thought>
<code>return value</code>

# Stored:
function.thinking = None
function.thought = "Algorithm summary"
function.body = "return value"
```

**FunSearch format:**
```python
# Raw code only
return value

# Stored:
function.thinking = None
function.thought = None
function.body = "return value"
```

## Creating Custom Variants

### Custom Problem Description

Create `src/disfun/specifications/Deletions/problem_descriptions/my_variant.txt`:

```python
"""
Your problem description here...
Explain the task, constraints, and objectives.
"""

import required_libraries

def helper_functions():
    # Problem-specific helpers
    pass

def priority(node, G, n, s):
    """Function to evolve."""
    return 0.0
```

Set in config:
```python
prompt=PromptConfig(
    problem_description_path="/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/problem_descriptions/my_variant.txt"
)
```

### Custom Prompt Format

Create `src/disfun/specifications/Deletions/prompt_styles/my_style.txt`:

```
Instructions for the LLM on how to format output.

Use <tags> to structure the response.

Example format:
<tag1>
Content for tag1
</tag1>

<tag2>
Content for tag2
</tag2>
```

Set in config:
```python
prompt=PromptConfig(
    prompt_style_path="/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/prompt_styles/my_style.txt"
)
```

### Custom Evaluation Logic

Create `src/disfun/specifications/Deletions/evaluation/my_eval.py` to specify how the evolved function is evaluated.

Set in config:
```python
evaluator=EvaluatorConfig(
    evaluation_script_path="/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/evaluation/my_eval.py"
)
``` 
