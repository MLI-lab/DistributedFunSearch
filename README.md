# DistributedFunSearch

<div align="center">
  <img src="fig/overview.png" alt="DistributedFunSearch Overview" width="600">
</div>

<p>&nbsp;</p>

**DistributedFunSearch** (`disfun`) is a **multi-node distributed implementation of [FunSearch](https://github.com/google-deepmind/funsearch)** ([Romera et al., 2024](https://www.nature.com/articles/s41586-023-06924-6)) that uses LLM-guided evolutionary search to discover novel algorithms. It uses RabbitMQ for asynchronous message passing and supports local models (via vLLM) and closed-source API models (via LiteLLM) through a unified inference interface.

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

# 3. Install C compiler (required for vLLM/Triton)
# For local models only - skip if using API-based LLM
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y

# 4. Install graph-tool (required for graph-based evaluation)
conda install -c conda-forge graph-tool -y

# 5. Start RabbitMQ
sudo systemctl start rabbitmq-server

# 6. Install DistributedFunSearch
pip install . # or pip install -e . for development mode
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

You can monitor message load in real time at `http://localhost:15672` (login: guest/guest). Enable the management plugin first with `sudo rabbitmq-plugins enable rabbitmq_management`.

The default configuration uses StarCoder2-15B (local model). To optionally use API models instead, see the "Change the LLM" section.

## Evolve your problem

Adapt DistributedFunSearch to your problem by defining a **specification** (what to solve), **prompt style** (how to format LLM output), **evaluation inputs** (what parameter values to test), and **evaluation outputs** (score and hash for deduplication).

### Create your specification

DistributedFunSearch uses a **modular specification system** with two independent components:

1. **Problem description** (what to solve) - Contains problem statement, imports, helper functions, and baseline function
2. **Prompt style** (how to respond) - Defines output format (code only, code + reasoning, etc.)

**Basic setup:**

Create `src/disfun/specifications/YourTask/problem_descriptions/baseline.txt`:

```python
"""
[Problem description]
Explain your problem, constraints, and what the function should optimize.

Improve the priority function over its previous versions.
Keep the code short.
"""

import your_dependencies

# Helper functions
def helper_function(...):
    pass

# Evaluation entry point
def evaluate(params):
    input1, input2, input3 = params
    result, hash_value = solve(input1, input2, input3)
    return (score, hash_value)

# Main evaluation logic
def solve(input1, input2, input3):
    # Uses the evolved priority function
    priorities = {item: priority(item, ...) for item in items}
    # ... use priorities to construct solution ...
    return solution, hash_value

# Function to evolve
def priority(item, context):
    """Returns the priority/score for the given item."""
    return 0.0
```

**Configure in config.py:**

```python
from disfun.config import Config, EvaluatorConfig, PromptStyleConfig

config = Config(
    evaluator=EvaluatorConfig(
        spec_path="src/disfun/specifications/YourTask",
        problem_description="baseline",  # Which problem variant
    ),

    prompt_style=PromptStyleConfig(
        preset="eoh",  # "funsearch" (code only), "eoh" (thought + code), or "extended_eoh" (thinking + thought + code)
    )
)
```

**Available prompt styles:**

- `funsearch`: Code only, minimal tokens
- `eoh`: One-sentence algorithm description + code (recommended)
- `extended_eoh`: Full reasoning + summary + code (for complex problems)

**Creating variants:**

You can create multiple problem descriptions (e.g., `explicit_constraints.txt`, `with_hints.txt`) and switch between them in config. See [Prompt Construction Guide](docs/PROMPTS.md) for details on creating custom variants and understanding how prompts are built.

**Key points:**

- Function names `evaluate` and `priority` are hardcoded in `__main__.py` (lines 284, 661)
- The evaluator executes the entire specification file, calling `evaluate()` with test inputs
- Helper functions define your problem (graph construction, constraints, etc.)
- The `priority` function is what the LLM evolves

### Configure your evaluation inputs

The evolved function is tested on problem instances defined by **tuples** (the evaluation inputs). Each tuple specifies one problem instance. For example, `(n=10, s=1, q=2)` specifies a binary code of length n=10 that corrects s=1 deletion.

The `evaluate()` function in your specification receives each tuple and uses the parameters to define the problem instance. It may call the evolved function with some, all, or none of these parameters (they define the problem context, not necessarily the function inputs). 

**Configuration example:**

```python
evaluator=EvaluatorConfig(
    spec_path="src/disfun/specifications/Deletions",
    problem_description="baseline",  # Which problem variant to use
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

Edit the `model` field in `config.py` to switch between models:

**Local models:**
```python
sampler=SamplerConfig(
    model="bigcode/starcoder2-15b",
    # or: "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # or: "mistralai/Mistral-7B-Instruct-v0.2"
)
```

Each sampler loads the model on its assigned GPU using vLLM Python API.

Models are automatically downloaded to `~/.cache/huggingface/` on first use. To change the cache location, either set it in config:
```python
sampler=SamplerConfig(
    model="bigcode/starcoder2-15b",
    cache_dir="/your/custom/cache/path",  # Optional: custom cache location
)
```

Or use an environment variable:
```bash
export HF_HOME=/your/custom/cache/path
```

**API models :**
```python
sampler=SamplerConfig(
    model="gpt-4o-mini",
    # or: "claude-3-5-sonnet-20241022"
    # or: "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)
```

For API models, copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

See [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for supported API models.

**Implementation:**

The system uses:
- **Local models**: Each sampler loads the model using vLLM Python API (one model instance per GPU)
- **API models**: LiteLLM client for unified access to multiple API providers

## Documentation

- [Configuration Guide](docs/CONFIGURATION.md): Detailed configuration options, CLI arguments, and config blocks
- [Prompt Construction Guide](docs/PROMPTS.md): How prompts are built, available problem descriptions and styles, creating custom variants
- [Scaling Guide](docs/SCALING.md): Explanation on how dynamic resource scaling is implemented
- [Docker Setup](docs/DOCKER_SETUP.md): Setup with Docker containers
- [Cluster Setup](docs/CLUSTER_SETUP.md): Setup on cluster with SLURM and enroot 
