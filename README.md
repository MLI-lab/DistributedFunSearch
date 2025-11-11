# DistributedFunSearch

<div align="center">
  <img src="fig/overview.png" alt="DistributedFunSearch Overview" width="600">
</div>

<p>&nbsp;</p>

**DistributedFunSearch** (`disfun`) is a **multi-node distributed implementation of FunSearch** (Romera et al., 2024) that uses LLM-guided evolutionary search to discover novel algorithms. It uses RabbitMQ for asynchronous message passing and works with both API-based LLMs (e.g., GPT-4o via Azure OpenAI) and locally hosted models (defaulting to StarCoder2).

- **Independent workers**: ProgramsDatabase, Samplers, and Evaluators work independently and process tasks asynchronously to maximize throughput
- **Multi-node execution**: Distributes across multiple nodes and allows adding Samplers or Evaluators from the same or different nodes to a running experiment (see [Cluster Setup](docs/CLUSTER_SETUP.md) for SLURM/Enroot example)
- **Dynamic scaling**: Automatically spawns/terminates Samplers and Evaluators based on workload and available resources (see [Scaling Guide](docs/SCALING.md))

In each iteration:

- A few-shot prompt is constructed by sampling from the program database, which stores all previously generated functions and their metadata
- The LLM generates a new function variant
- The function is evaluated on user-defined test cases
- If the function is executable and logically distinct from previously stored ones, it is added to the program database along with its evaluation results

Our implementation includes an example application for discovering large deletion-correcting codes. For details on this specific use case, see [our paper](https://arxiv.org/abs/2504.00613).

## Quickstart

### Installation

```bash
# 1. Create and activate conda environment
conda create -n env python=3.11 pip numpy==1.26.4 -y
conda activate env

# 2. Install PyTorch (skip if using API-based LLM)
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Start RabbitMQ
sudo systemctl start rabbitmq-server

# 4. Install DistributedFunSearch
pip install .
```

See [Docker Setup](docs/DOCKER_SETUP.md) for container-based installation or [Cluster Setup](docs/CLUSTER_SETUP.md) for cluster execution.

### Run experiment

This runs the example specification for discovering deletion-correcting codes:

```bash
cd src/experiments/experiment1
python -m disfun

# Resume from checkpoint (if needed)
python -m disfun --checkpoint path/to/checkpoint.pkl
```

## Evolve your problem

Adapt DistributedFunSearch to your problem by defining a **specification file** (that specifies which function to evolve and how to evaluate it), **evaluation inputs** (what parameter values to test your evolved function on), and **evaluation outputs** (score and hash for deduplication, plus optional metrics like speed or memory).

### Create your specification

Add a new specification file to `src/disfun/specifications/` (see existing examples in `Deletions/` or `IDS/` folders).

**Specification structure:**

```python
"""
[Problem description that becomes the LLM prompt]
Improve the `your_function` function over its previous versions below.
Keep the code short and comment for easy understanding.
"""

import your_dependencies

# Helper functions and classes that define your problem
def helper_function(...):
    ...

# Evaluation entry point, must be named "evaluate"
# Called by evaluator with test inputs
def evaluate(params):
    input1, input2, input3 = params
    result, hash_value = solve(input1, input2, input3)
    return (score, hash_value)  # Score and hash for deduplication

# Main evaluation logic which uses the evolved function
def solve(input1, input2, input3):
    # Your problem-specific logic that uses the priority function
    priorities = {item: priority(item, ...) for item in items}
    # ... use priorities to construct solution ...
    return solution, hash_value

# The function that gets evolved by the LLM, must be named "priority"
def priority(item, context):
    """Returns the priority/score for the given item."""
    return 0.0  # Baseline implementation
```

**Explanation:**
- Docstring: Becomes the problem context in the LLM prompt
- Helper functions: Defines your problem (graph construction, constraints, etc.)
- `evaluate(params)`: Entry point called by evaluator
- `solve(...)`: Implements evaluation logic using the evolved function
- `priority(item, context)`: The function that the LLM evolves

The function names `evaluate` and `priority` are hardcoded in `__main__.py` (lines 284, 661). If you want to use different names, you also need to update them there.

The evaluator executes this entire script, calling `evaluate()` with evaluation inputs.

### Configure your evaluation inputs

The evolved function is tested on problem instances defined by **tuples** (the evaluation inputs). Each tuple specifies one problem instance. For example, `(n=10, s=1, q=2)` specifies a binary code of length n=10 that corrects s=1 deletion.

The `evaluate()` function in your specification receives each tuple and uses the parameters to define the problem instance. It may call the evolved function with some, all, or none of these parameters (they define the problem context, not necessarily the function inputs). 

**Configuration example:**

```python
evaluator=EvaluatorConfig(
    spec_path="src/disfun/specifications/Deletions/StarCoder2/load_graph/baseline.txt",
    s_values=[1, 2],        # Scalar or list: error correction levels
    start_n=[5, 7],         # Range start: code lengths (one per s_value)
    end_n=[10, 12],         # Range end: code lengths (one per s_value)
    q=2,                    # Scalar: alphabet size (2=binary, 4=DNA)
    timeout=90,             # Timeout per evaluation in seconds
    max_workers=2,          # Parallel CPU processes per evaluator
)
```

**Customizing:**

To change which problem instances are tested, modify the `create_evaluation_inputs()` function in `src/disfun/__main__.py`. This function generates tuples from your config parameters. Customize it by:
- Adding new parameters to `EvaluatorConfig` in `config.py`
- Changing how parameters are combined into tuples in `create_evaluation_inputs()`
- Updating your specification's `evaluate(params)` to unpack the tuple

Your specification's `evaluate(params)` receives each tuple and uses it to define the problem instance.

### Configure your evaluation outputs and scoring

**Evaluation outputs:**

The `evaluate()` function returns a tuple `(score, hash_value)` for each problem instance:
- `score`: Numeric value measuring solution quality (higher is better by default)
- `hash_value`: Optional hash for deduplication (set to `None` if not needed)

**Scoring configuration:**

```python
evaluator=EvaluatorConfig(
    # ... evaluation input parameters ...
    mode="last",  # How to aggregate scores: "last" = use largest n for each s, "average", "weighted"
)
```

**How scores are aggregated:**

1. **Per problem instance** (in `src/disfun/evaluator.py` function `extract_evaluation_result()` line 66): Extracts `test_output[0]` as score and `test_output[1]` as hash, stores in `scores_per_test` dictionary with the full problem instance tuple as key (e.g., `(n, s, q)`)

2. **Across all problem instances** (in `src/disfun/programs_database.py` function `_reduce_score()` line 82): Aggregates the `scores_per_test` dictionary into a single score that determines sampling. Extracts `(n, s)` from full tuples and aggregates based on mode set in config (`"last"` = use largest n for each s, `"average"`, `"weighted"`)

To extract additional outputs (e.g., execution time, memory usage) or change how scores are combined, modify these two functions.


### Change the LLM 

**For different open-source models:**

Edit `src/disfun/sampler.py` line ~64:
```python
checkpoint = "bigcode/starcoder2-15b"  # Any HuggingFace model ID
```

**For OpenAI models:**

In `config.py`, enable GPT mode:
```python
sampler=SamplerConfig(
    gpt=True,
    # ... other sampler config
)
```

To use a different GPT model, edit `src/disfun/gpt.py` line 22:
```python
def __init__(self, samples_per_prompt: int, model="gpt-4o-mini"):  # Change model here
```

Then export your Azure OpenAI credentials:
```bash
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_API_VERSION=<your-version>
```

## Documentation

- [Configuration Guide](docs/CONFIGURATION.md): Detailed configuration options, CLI arguments, and config blocks
- [Scaling Guide](docs/SCALING.md): Explanation on how dynamic resource scaling is implemented
- [Docker Setup](docs/DOCKER_SETUP.md): Setup with Docker containers
- [Cluster Setup](docs/CLUSTER_SETUP.md): Setup on cluster with SLURM and enroot 
