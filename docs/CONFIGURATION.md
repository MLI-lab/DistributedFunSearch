# Configuration Guide

FunSearchMQ is configured through:
1. **Command-line arguments** - Runtime options (paths, scaling, termination)
2. **Config file** - Experiment parameters (`config.py`)

## Command-Line Arguments

### General Arguments

```bash
python -m funsearchmq \
  --config-path ./config.py \           # Path to config file (default: ./config.py)
  --checkpoint ./checkpoint.pkl \       # Resume from checkpoint (restores run and continues W&B logging)
  --sandbox_base_path ./sandbox \       # Sandbox directory (default: ./sandbox)
  --log-dir ./logs                      # Log directory (default: ./logs)
```

### Resource Management

Control dynamic scaling behavior:

```bash
python -m funsearchmq \
  --check_interval 120 \                # Scaling check interval in seconds (default: 120)
  --max_evaluators 1000 \               # Max evaluators for scaling (default: 1000)
  --max_samplers 1000 \                 # Max samplers for scaling (default: 1000)
  --no-dynamic-scaling                  # Disable automatic scaling (enabled by default)
```

See [Scaling Guide](SCALING.md) for details on scaling behavior.

### Termination Conditions

Stop experiments automatically:

```bash
python -m funsearchmq \
  --prompt_limit 400000000 \            # Max prompts before stopping (default: 400,000,000)
  --optimal_solution_programs 200000 \  # Programs to generate after finding optimal (default: 200,000)
  --target_solutions '{"(7,2)": 5, "(8,2)": 7}'  # JSON dict of target solutions
```

**Example**: Stop when you find a code with independent set size ≥ 5 for (n=7, s=2):
```bash
python -m funsearchmq --target_solutions '{"(7,2)": 5}'
```

### Azure OpenAI

For API-based LLMs instead of local models:

```bash
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_API_VERSION=<your-version>
python -m funsearchmq  # Ensure config.py has sampler.gpt=True
```

## Configuration File 

Experiments are configured via `config.py` in the experiment directory (e.g., `src/experiments/experiment1/config.py`).

### Configuration Blocks

<details>
<summary><b>RabbitMQConfig</b> - Message broker connection settings</summary>

Controls connection to RabbitMQ for inter-process communication.

**Attributes:**
- `host` (str): RabbitMQ server hostname
  - Default: `'rabbitmq'` (Docker) or `'localhost'` (local)
  - For cluster: Use hostname of node running RabbitMQ
- `port` (int): Server port (default: `5672`)
- `username` (str): Authentication username (default: `'guest'`)
- `password` (str): Authentication password (default: `'guest'`)
- `vhost` (str): Virtual host for isolation between experiments (default: `'exp1'`)
  - Use `''` for default vhost
  - Use different vhosts for concurrent experiments

</details>

<details>
<summary><b>ProgramsDatabaseConfig</b> - Function storage and prompt construction</summary>

Controls how the database stores functions and constructs prompts.

**Attributes:**
- `functions_per_prompt` (int): Number of previous programs to include in few-shot prompt (default: `2`)
- `num_islands` (int): Number of islands for diversity (default: `10`)
- `reset_period` (int): Interval in seconds for resetting weakest islands (default: `None`)
- `reset_programs` (int): Number of stored programs after which weakest islands are reset (default: `1200`)
- `cluster_sampling_temperature_init` (float): Initial temperature for softmax sampling of clusters (default: `0.1`)
- `cluster_sampling_temperature_period` (int): Period of linear decay for cluster sampling temperature (default: `30000`)
- `prompts_per_batch` (int): Batch size for processing prompts from `database_queue` (default: `10`)
- `no_deduplication` (bool): Disable deduplication (default: `False`)


</details>

<details>
<summary><b>SamplerConfig</b> - LLM sampling parameters</summary>

Controls how the LLM generates new function variants.

**Attributes:**
- `prompts_per_batch` (int): Batch size for processing prompts from `sampler_queue` (default: `10`)
- `samples_per_prompt` (int): Number of continuations to generate per prompt (default: `2`)
- `temperature` (float): LLM sampling temperature (default: `0.944`)
  - Higher = more diverse, lower = more deterministic
- `temperature_period` (int): Period for dynamic temperature adjustment (default: `None`)
  - If `None`, temperature is fixed
- `max_new_tokens` (int): Maximum tokens to generate (default: `246`)
- `top_p` (float): Nucleus sampling parameter (default: `0.778`)
- `repetition_penalty` (float): Penalty for repetitive text (default: `1.222`)
  - Values > 1 discourage repetition, 1 disables it
- `gpt` (bool): Use OpenAI API instead of local model (default: `False`)
  - When `True`, GPU device assignment is disabled


</details>

<details>
<summary><b>EvaluatorConfig</b> - Function evaluation settings</summary>

Controls how generated functions are tested and scored.

**Attributes:**
- `s_values` (List[int]): Error correction parameters (default: `[2]`)
  - For deletion codes: number of deletions to correct
  - For IDS codes: number of insertions/deletions/substitutions (requires min distance 2s+1)
- `start_n` (List[int]): Shortest code length for each s (default: `[7]`)
- `end_n` (List[int]): Longest code length for each s (default: `[12]`)
- `mode` (str): Score reduction mode (default: `"last"`)
  - `"last"`: Use score from longest n
  - `"average"`: Average scores across all n
  - `"weighted"`: Weighted average
- `timeout` (int): Sandbox timeout in seconds (default: `90`)
- `max_workers` (int): Number of parallel CPU processes per evaluator (default: `2`)
- `eval_code` (bool): Include evaluation script in prompt (default: `False`)
- `include_nx` (bool): Include NetworkX in prompt (default: `True`)
- `spec_path` (str): Path to specification file (default: set by `get_spec_path()`)
- `q` (int): Alphabet size (default: `2` for binary, `4` for DNA)


**Notes:**
- Test cases are all (n, s) pairs from `start_n[i]` to `end_n[i]` for each `s_values[i]`
- Hash is computed for n=`start_n[0]` (used for deduplication)
- Increase `max_workers` for more parallelism (more CPU usage)

</details>

<details>
<summary><b>PromptConfig</b> - Prompt generation and score display</summary>

Controls how scores are displayed in few-shot prompts.

**Attributes:**
- `show_eval_scores` (bool): Include evaluation scores in docstrings (default: `True`)
- `display_mode` (str): Score display format (default: `"relative"`)
  - `"absolute"`: Show raw scores
  - `"relative"`: Show improvement vs baseline
- `best_known_solutions` (dict): Baseline scores for relative mode (default: `{(7,2): 5, (8,2): 7, ...}`)
  - Format: `{(n, s): score}`
- `absolute_label` (str): Prefix for absolute scores (default: `"Absolute scores..."`)
- `relative_label` (str): Prefix for relative scores (default: `"Performance relative to baseline..."`)

**Example (Absolute Scores):**
```python
prompt=PromptConfig(
    show_eval_scores=True,
    display_mode="absolute",
)
```

**Example (Relative Scores):**
```python
prompt=PromptConfig(
    show_eval_scores=True,
    display_mode="relative",
    best_known_solutions={
        (7, 2): 5,   # Baseline for n=7, s=2
        (8, 2): 7,   # Baseline for n=8, s=2
    },
)
```

**Notes:**
- Relative mode formula: `(score - baseline) / |baseline| × 100%`
- Scores are appended to function docstrings in prompts

</details>

<details>
<summary><b>ScalingConfig</b> - Dynamic scaling thresholds</summary>

Controls when samplers and evaluators are automatically spawned/terminated.

**Attributes:**
- `sampler_scale_up_threshold` (int): Messages in `sampler_queue` to trigger scale-up (default: `50`)
- `evaluator_scale_up_threshold` (int): Messages in `evaluator_queue` to trigger scale-up (default: `10`)
- `min_gpu_memory_gib` (int): Minimum free GPU memory (GiB) to start sampler (default: `20`)
  - **Adjust based on LLM size** (StarCoder2-15B needs ~30 GiB)
- `max_gpu_utilization` (int): Maximum GPU utilization (%) to start sampler (default: `50`)
- `min_system_memory_gib` (int): Minimum free system RAM (GiB) required (default: `30`)
- `cpu_usage_threshold` (int): Maximum CPU usage (%) for evaluator scale-up (default: `99`)
- `normalized_load_threshold` (float): Maximum load-per-core for evaluator scale-up (default: `0.99`)

**Example:**
```python
scaling=ScalingConfig(
    sampler_scale_up_threshold=50,
    evaluator_scale_up_threshold=10,
    min_gpu_memory_gib=30,     # For StarCoder2-15B
    max_gpu_utilization=50,
    min_system_memory_gib=30,
)
```

**See [Scaling Guide](SCALING.md) for detailed tuning.**

</details>

<details>
<summary><b>WandbConfig</b> - Weights & Biases logging and checkpoints</summary>

Controls W&B experiment tracking and checkpoint storage.

**Attributes:**
- `enabled` (bool): Enable W&B logging (default: `True`)
- `project` (str): W&B project name (default: `"funsearchmq"`)
- `entity` (str): W&B username or team name (default: set in config)
- `run_name` (str): Name for this run (default: `None`)
  - If `None`, auto-generates run name with timestamp: `run_YYYYMMDD_HHMMSS`
  - This name is used for both W&B run and checkpoint folder
- `log_interval` (int): Logging frequency in seconds (default: `300` = 5 minutes)
- `tags` (List[str]): Tags for this run (default: `[]`)
- `checkpoints_base_path` (str): Base directory for checkpoints (default: `"./Checkpoints"`)
  - Actual checkpoint folder: `{checkpoints_base_path}/checkpoint_{run_name}/`


**Run Name and Checkpoint Linking:**

The same run name is used for both W&B and checkpoints, ensuring they're automatically linked:

```python
wandb=WandbConfig(
    run_name=None,  # Auto-generates: run_20250108_143022
    checkpoints_base_path="/mnt/checkpoints"
)
# Results in:
# - W&B run name: run_20250108_143022
# - Checkpoint folder: /mnt/checkpoints/checkpoint_run_20250108_143022/
# - Checkpoints: checkpoint_run_20250108_143022/checkpoint_2025-01-08_14-30-22.pkl
```

**Resuming from Checkpoint:**

When resuming with `--checkpoint`:
1. Loads all experiment state (islands, scores, counters, etc.)
2. Retrieves the saved W&B run ID
3. Resumes the same W&B run (not creating a new one)
4. Continues saving checkpoints to the same folder

```bash
# Resume experiment and continue W&B logging
python -m funsearchmq --checkpoint /mnt/checkpoints/checkpoint_run_20250108_143022/checkpoint_2025-01-08_15-30-22.pkl
```

**Checkpoint Timing:**

- First checkpoint is saved after 1 hour (3600 seconds)
- Subsequent checkpoints are saved hourly
- Checkpoint directory is created when the first checkpoint is saved

</details>

<details>
<summary><b>Config</b> - Main configuration class</summary>

Top-level config that combines all sub-configs.

**Attributes:**
- `programs_database` (ProgramsDatabaseConfig): Database configuration
- `rabbitmq` (RabbitMQConfig): RabbitMQ connection
- `sampler` (SamplerConfig): LLM sampling
- `evaluator` (EvaluatorConfig): Function evaluation
- `prompt` (PromptConfig): Prompt generation
- `wandb` (WandbConfig): W&B logging
- `scaling` (ScalingConfig): Dynamic scaling
- `num_samplers` (int): Number of sampler processes (default: `2`)
  - Each sampler uses one GPU (or CPU if `gpt=True`)
- `num_evaluators` (int): Number of evaluator processes (default: `10`)
  - Each evaluator spawns `max_workers` parallel CPU processes (default: 2)
- `num_pdb` (int): Number of program databases (default: `1`)
  - Currently only supports 1


**Resource Usage:**
- **GPUs**: `num_samplers` (1 GPU per sampler, or CPU if `gpt=True`)
- **CPUs**: `num_evaluators × max_workers` (default: 10 × 2 = 20 CPU processes)

</details>

## Specification Files

The function to evolve is defined in specification files at `src/funsearchmq/specifications/`.

### Directory Structure

```
specifications/
├── Deletions/           # Deletion-correcting codes
│   ├── StarCoder2/
│   │   ├── load_graph/  # Pre-computed graphs
│   │   └── construct_graph/  # On-the-fly graph construction
│   └── gpt/
│       └── load_graph/
└── IDS/                 # Insertion/Deletion/Substitution codes
    └── StarCoder2/
        └── load_graph/
```

### Changing Specifications

Edit `get_spec_path()` in `config.py`:

```python
def get_spec_path() -> str:
    # Deletion codes (default)
    return os.path.join(decos_base, "src", "decos", "specifications",
                        "Deletions", "StarCoder2", "load_graph", "baseline.txt")

    # IDS codes (uncomment to use)
    # return os.path.join(decos_base, "src", "decos", "specifications",
    #                     "IDS", "StarCoder2", "load_graph", "baseline.txt")
```

### Graph Files

Pre-computed graphs are stored at `src/graphs/`:
- Deletion codes: `graph_d_s{s}_n{n}_q{q}.lmdb`
- IDS codes: `graph_ids_s{s}_n{n}_q{q}.lmdb`

Where `d` indicates deletion correction, `ids` indicates insertion/deletion/substitution correction, `s` is the error parameter, `n` is the code length, and `q` is the alphabet size.

To pre-compute graphs, see `src/construct_graphs/` which contains scripts and documentation for graph generation.
