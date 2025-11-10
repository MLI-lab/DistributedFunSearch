# FunSearchMQ

<div align="center">
  <img src="fig/overview.png" alt="FunSearchMQ Overview" width="600">
</div>

<p>&nbsp;</p>

**FunSearchMQ** is a **distributed implementation of FunSearch** (Romera et al., 2024) that uses LLM-guided evolutionary search to discover novel algorithms. It supports multi-node execution via RabbitMQ for asynchronous message passing and works with both API-based LLMs (e.g., GPT-4o via Azure OpenAI) and locally hosted models (defaulting to StarCoder2).

## Features

FunSearchMQ is designed for large-scale distributed execution:

- **Multi-node execution**: Distributes across multiple nodes and allows adding workers to a running experiment from different nodes (see [Cluster Setup](docs/CLUSTER_SETUP.md) for SLURM/Enroot example)
- **Asynchronous workers**: ProgramsDatabase, Samplers, and Evaluators work independently
- **Dynamic scaling**: Automatically adjusts workers based on message load for maximum throughput (see [Scaling Guide](docs/SCALING.md))

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

# 4. Install FunSearchMQ
pip install .
```

See [Docker Setup](docs/DOCKER_SETUP.md) for container-based installation or [Cluster Setup](docs/CLUSTER_SETUP.md) for cluster execution.

### Run experiment

This runs the example specification for discovering deletion-correcting codes:

```bash
cd src/experiments/experiment1
python -m funsearchmq

# Resume from checkpoint (if needed)
python -m funsearchmq --checkpoint path/to/checkpoint.pkl
```

## Adapting to your application

FunSearchMQ can be adapted to discover algorithms for other problems by defining a new **specification file**. The specification defines the function to evolve and how to evaluate it.

### 1. Create your specification

Add a new specification file to `src/funsearchmq/specifications/` (see existing examples in `Deletions/` or `IDS/` folders).

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
- **Docstring**: Becomes the problem context in the LLM prompt
- **Helper functions**: Defines your problem (graph construction, constraints, etc.)
- **`evaluate(params)`**: Entry point called by evaluator
- **`solve(...)`**: Implements evaluation logic using the evolved function
- **`priority(item, context)`**: The function that the LLM evolves

The function names `evaluate` and `priority` are hardcoded in `__main__.py` (lines 284, 661). If you want to use different names, you also need to update them there.

The evaluator executes this entire script, calling `evaluate()` with evaluation inputs.

### 2. Configure evaluation inputs and outputs

**Evaluation inputs:**

In `config.py`, specify what inputs to test your evolved function on:

```python
evaluator=EvaluatorConfig(
    spec_path="src/funsearchmq/specifications/YourProblem/model/baseline.txt",
    # Define test input ranges, will generate all combinations
    s_values=[1, 2],        # Parameter 1 values
    start_n=[5, 7],         # Parameter 2 start values (one per s_value)
    end_n=[10, 12],         # Parameter 2 end values (one per s_value)
    q=2,                    # Parameter 3 (if needed)
    # Test inputs will be: (5,1,2), (6,1,2), ..., (10,1,2), (7,2,2), ..., (12,2,2)

    mode="last",            # How to aggregate scores: "last", "average", "weighted"
    timeout=90,             # Timeout per evaluation in seconds
    max_workers=2,          # Parallel CPU processes per evaluator
)
```

The default configuration generates `(n, s, q)` tuples for the deletion-correcting codes problem. If your problem needs a different input structure, edit `__main__.py` line 897:

```python
# Default (deletion codes):
inputs = [(n, s, config.evaluator.q) for s, start_n, end_n in zip(...) for n in range(start_n, end_n + 1)]

# Example custom (2D grid search):
inputs = [(w, h) for w in range(5, 15) for h in range(5, 15)]
```

Your specification's `evaluate(params)` then receives these custom tuples.

**Evaluation outputs:**

The `evaluate()` function returns a tuple `(score, hash_value)`:
- `score`: Numeric value measuring solution quality (higher is better by default)
- `hash_value`: Optional hash for deduplication (set to `None` if not needed)

The evaluator extracts `test_output[0]` as the score and `test_output[1]` as the hash (`evaluator.py` lines 349-351). If you need to extract more outputs or use them differently, modify `src/funsearchmq/evaluator.py` (to extract additional tuple elements) and `src/funsearchmq/programs_database.py` (to store/use them).

### 3. Set termination conditions

Define when the experiment should stop:

```python
termination=TerminationConfig(
    prompt_limit=1_000_000,                      # Stop after N prompts
    optimal_solution_programs=50_000,            # Generate N more after finding optimal
    target_solutions={(10, 1): 94, (12, 2): 30}  # Stop when these scores reached
    # Set target_solutions={} to disable early termination
)
```

### 4. Change the LLM 

**For different open-source models:**

Edit `src/funsearchmq/sampler.py` line ~64:
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

To use a different GPT model, edit `src/funsearchmq/gpt.py` line 22:
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
