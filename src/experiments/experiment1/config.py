# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License - Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing - software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND - either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration of a FunSearch experiment. Only data classes no methods

Adjusted to include RabbitMQ setup - deduplication option and set sepecification file for experiment.
"""
import dataclasses
from typing import List
import os
from datetime import datetime



@dataclasses.dataclass(frozen=True)
class RabbitMQConfig:
    """Configuration for RabbitMQ connection.

    Attributes:
      host: The hostname of the RabbitMQ server.
      port: The port of the RabbitMQ server (AMQP).
      management_port: The port for RabbitMQ management API (default 15672).
      username: Username for authentication with the RabbitMQ server.
      password: Password for authentication with the RabbitMQ server.
      vhost: Virtual host for isolation between experiments. Use '' for default vhost.
    """
    host: str = 'rabbitmq' #localhost or rabbitmq for docker or node IP address
    port: int = 5672
    management_port: int = 15672  # Management API port for monitoring
    username: str = 'guest'
    password: str = 'guest'
    vhost: str = ''  # Use '' for default vhost, or 'exp1', 'exp2', etc. for isolated experiments 
    

@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    functions_per_prompt: Number of previous programs to include in current prompt.
    num_islands: Number of islands to maintain for diversity.
    reset_period: The interval (in seconds) at which the weakest islands are reset. If None - resets occur only based on the number of stored programs.
    reset_programs: The number of stored programs after which the weakest islands are reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster sampling temperature.
    prompts_per_batch: Batch size for processing prompts received from the database_queue
    no_deduplication: Disable deduplication (default: False - set True to disable).
    save_lineage: Save evolutionary lineage HTML files and track lineage metrics (default: False).
  """
  functions_per_prompt: int = 2
  num_islands: int = 10
  reset_period: int = None
  reset_programs: int= 1200
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 30_000
  prompts_per_batch= 10
  no_deduplication: bool = False
  save_lineage: bool = False


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    prompts_per_batch: Batch size for processing prompts received from the sampler_queue
    samples_per_prompt: How many independently sampled program continuations to get for each prompt.
    temperature_period: Controls how fast the LLM's temperature decreases as more programs are registered. If None - dynamic temperature adjustment is disabled.
    temperature: Controls randomness: higher values increase diversity - lower values make outputs more deterministic.
    max_new_tokens: The maximum number of tokens the LLM can generate in response.
    top_p: Determines the range of likely tokens the model samples from - keeping only the most probable ones.
    repetition_penalty: Penalizes repetitive text; values >1 discourage repetition - while 1 disables it.
    model: Any LiteLLM-supported model: https://docs.litellm.ai/docs/providers or local model from huggingface 
    api_base: Custom API endpoint URL. Leave as None for standard cloud APIs (OpenAI - Anthropic - etc.).
    api_key: API key for authentication. Leave as None to automatically load from .env file.
    cache_dir: Optional directory for model cache (local models only). If None - defaults to ~/.cache/huggingface/
    reasoning_effort: Controls GPT-5/o3 internal chain-of-thought reasoning depth. Only works with GPT-5/o3 models.
    max_retries: Maximum number of retry attempts for failed API calls (default: 3). Only applies to API models.
  """
  prompts_per_batch= 10
  samples_per_prompt: int = 2
  temperature_period= None
  temperature: float = 0.9444444444444444
  max_new_tokens: int = 600 #246 - for reasoing set larger 
  top_p: float =  0.7777777777777778
  repetition_penalty: float = 1.222222
  reasoning_effort: str = "None"  # Set to "minimal", "low", "medium", "high", or None.
  max_retries: int = 3  # Maximum retry attempts for API calls
  model: str = "gpt-4.1-mini" #"bigcode/starcoder2-15b"  # Local model (each sampler loads on GPU) or API model (gpt-5 - claude-3-5-sonnet-20241022)
  api_base: str = None  # Leave as None for standard OpenAI. Only set for custom endpoints (vLLM server - Azure - etc.)
  api_key: str = None   # Leave as None - automatically loads OPENAI_API_KEY from /workspace/DistributedFunSearch/.env
  cache_dir: str = "/mnt/models"  # Directory for model cache (local models only). Defaults to ~/.cache/huggingface/ if None

@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
    """Configuration of the Evaluator.

    Attributes:
        evaluation_script_path: Absolute path to the evaluation script.
            Available options:
            - no_graph.py: No graph loading, on-the-fly evaluation (slow, simple)
            - graph_networkx.py: NetworkX graphs (10-100x faster than no_graph, simple API for LLMs)
            - graph_gt.py: graph-tool (10-100x faster than NetworkX, but LLMs need to convert nodes to vertex indices)
        initial_functions_dir: Directory containing initial seed functions (.txt files).
        s_values: List of error correction parameters.
        start_n: List of shortest code length for each s.
        end_n: List of longest code lengths for each s.
        mode: Mode for score reduction. Available options: 'last', 'average', 'weighted'.
        timeout: Timeout in seconds for the sandbox.
        max_workers: Number of parallel CPU processes per evaluator for evaluating functions on different inputs (default: 2).
        eval_code: Include evaluation script in prompt. (default: False - set True to enable).
        include_nx: Include the graph packages in the prompt (default: True - set False to disable).
        q: Alphabet size for the codes (default: 2 for binary). Set to 4 for DNA data storage use case (alphabet: A, C, G, T).
        graph_dir: Directory where graph files (.lmdb) are stored. Default is "src/graphs" relative to project root.
        cache_graphs: Enable in-memory graph caching to avoid reloading graphs for every evaluation (default: False). Note - each evaluator will have its own cache - so this increases RAM usage.
        cache_size_limit_gb: Only cache graphs smaller than this size in GiB (default: 2.0). Set to float('inf') to cache all graphs regardless of size.
    """
    evaluation_script_path: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/evaluation/graph_networkx.py"  # Options: no_graph.py, graph_networkx.py, graph_gt.py
    initial_functions_dir: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/initial_functions/graph_networkx"  # Use "no_graph" for on-the-fly evaluation, "graph" for graph variants
    s_values: List[int] = dataclasses.field(default_factory=lambda: [2])
    start_n: List[int] = dataclasses.field(default_factory=lambda: [7])  # Hash is computed for n==start_n[0] (automatically substituted in specification)
    end_n: List[int] = dataclasses.field(default_factory=lambda: [12])  # Match available graphs (s1_n6 through s1_n11)
    mode: str = "last"
    timeout: int = 90
    max_workers: int = 2
    eval_code: bool = False
    include_nx: bool = True
    q: int = 2  # Set to 4 for DNA data storage use case (alphabet: A, C, G, T)
    graph_dir: str = "/workspace/DistributedFunSearch/src/graphs"  # Directory containing graph files (.lmdb)
    cache_graphs: bool = False  # Enable in-memory graph caching (reduces I/O, increases RAM)
    cache_size_limit_gb: float = 2.0  # Only cache graphs smaller than this (GiB)


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    """Configuration for prompt construction.

    This combines what content the LLM sees and how it should format its response.

    Attributes:
        problem_description_path: Absolute path to problem description file.
        prompt_style_path: Absolute path to prompt style file. Set to None for code completion mode (no instructions) for base models like StarCoder.
        system_message_path: Absolute path to system message file for API models. Set to None to disable system messages.
        imports_path: Absolute path to imports file. Contains Python imports to be injected after problem description and prompt style.
                     Set to None to disable imports injection (will use imports from template instead).
                     Example: "/path/to/specifications/Deletions/imports/graph_tool.txt"
        fewshot_num_examples: Number of examples to show in few-shot prompts (sorted by score, highest first).
        fewshot_show_thinking: Whether to include <thinking> in few-shot examples.
        fewshot_show_thought: Whether to include <thought> in few-shot examples.
        fewshot_show_code: Whether to include <code> in few-shot examples.
        show_eval_scores: Whether to include evaluation scores in function docstrings (default: False).
        display_mode: How to display scores: "absolute" or "relative" (default: "absolute").
        best_known_solutions: Dictionary mapping (n, s) tuples to best-known or baseline scores.
                             Required when display_mode is "relative".
                             Example: {(6, 1): 8, (7, 1): 14, (8, 1): 26}
        absolute_label: Prefix text for absolute scores.
        relative_label: Prefix text for relative improvements (default: "Relative to baseline:").
    """
    # What content the LLM sees
    problem_description_path: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/problem_descriptions/baseline.txt"

    # How to format output
    prompt_style_path: str | None = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/prompt_styles/eoh.txt"  # Set to None for code completion models (StarCoder) - or use absolute path for instruction-tuned models
    system_message_path: str | None = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/system_messages/graph.txt"  # System message for API models (GPT - Claude) - set to None to disable. Use "graph.txt" for graph-tool variant
    imports_path: str | None = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/imports/networkx.txt"  # Imports to inject after problem description and prompt style

    # Few-shot configuration
    fewshot_num_examples: int = 2
    fewshot_show_thinking: bool = False  # Show previous model thought process
    fewshot_show_thought: bool = True    # Show previous algorithm descriptions
    fewshot_show_code: bool = True       # Show previous implementations

    # Score display
    show_eval_scores: bool = False
    display_mode: str = "relative" # "absolute" or "relative"
    best_known_solutions: dict = dataclasses.field(default_factory=lambda: {(7, 2, 2): 5, (8, 2, 2): 7, (9, 2, 2): 11, (10, 2, 2): 16, (11, 2, 2): 24, (12, 2, 2): 37})  # use e.g. VT codes for single deletion or logn +loglogn + log3 for single IDS code rate (need to transform to code sizes) (from https://arxiv.org/pdf/2312.12717)
    absolute_label: str = "Absolute scores (format (n, s, q): set_size, larger is better):"
    relative_label: str = "Performance relative to baseline (format (n, s, q): improvement%):"


@dataclasses.dataclass(frozen=True)
class WandbConfig:
    """Configuration for Weights & Biases logging.

    Attributes:
        enabled: Enable W&B logging (default: False).
        project: W&B project name.
        entity: W&B entity (username or team name).
        run_name: Name for this run (default: None, auto-generated with timestamp).
        log_interval: How often to log metrics in seconds (default: 300 = 5 minutes).
        tags: List of tags for this run.
        checkpoints_base_path: Base directory for checkpoints (default: "./Checkpoints"). Actual checkpoint folder will be: {checkpoints_base_path}/checkpoint_{run_name}/
    """
    enabled: bool = True
    project: str = "disfun"
    entity: str = "franziweindel-technical-university-of-munich"  # Set to your W&B username or team
    run_name: str = None  # Auto-generated with timestamp if None
    log_interval: int = 300  # Log every 5 minutes
    tags: List[str] = dataclasses.field(default_factory=list)
    checkpoints_base_path: str = "/mnt/disfun/checkpoints" #"./Checkpoints" # Use "./Checkpoints" for local runs


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    """File system paths for the experiment.

    Attributes:
        log_dir: Directory for logs (default: ./logs). Can be overridden by --log-dir CLI argument.
        sandbox_base_path: Directory for sandboxed code execution (default: ./sandbox).
                          Can be overridden by --sandbox_base_path CLI argument.
        backup_enabled: Enable backup of Python files before running (default: False).
                       Can be overridden by --backup CLI flag.
    """
    log_dir: str = "./logs"
    sandbox_base_path: str = "/mnt/disfun/sandbox"
    backup_enabled: bool = False


@dataclasses.dataclass(frozen=True)
class ScalingConfig:
    """Configuration for dynamic scaling of samplers and evaluators.

    Attributes:
        enabled: Enable dynamic scaling (default: True). Can be disabled with --no-dynamic-scaling CLI flag.
        check_interval: Time interval (in seconds) between consecutive scaling checks (default: 120).
                       Lower values = more responsive scaling but higher overhead.
        max_samplers: Maximum number of samplers the system can scale up to (default: 1000).
        max_evaluators: Maximum number of evaluators the system can scale up to (default: 1000).
        sampler_scale_up_threshold: Number of messages in sampler_queue to trigger scale-up (default: 50).
        evaluator_scale_up_threshold: Number of messages in evaluator_queue to trigger scale-up (default: 10).
        min_gpu_memory_gib: Minimum free GPU memory in GiB required to start a new sampler (default: 20).
                            Adjust based on your LLM size: StarCoder2-15B needs ~30 GiB - smaller models need less.
        max_gpu_utilization: Maximum GPU utilization percentage to allow starting a new sampler (default: 50).
        min_system_memory_gib: Minimum free system RAM in GiB required for scaling (default: 30).
        cpu_usage_threshold: Maximum average CPU usage percentage to allow evaluator scale-up (default: 99).
        normalized_load_threshold: Maximum normalized system load (load/cores) to allow evaluator scale-up (default: 0.99).
    """
    enabled: bool = True
    check_interval: int = 60
    max_samplers: int = 1000
    max_evaluators: int = 1000
    sampler_scale_up_threshold: int = 50
    evaluator_scale_up_threshold: int = 10
    min_gpu_memory_gib: int = 35
    max_gpu_utilization: int = 50
    min_system_memory_gib: int = 30
    cpu_usage_threshold: int = 99
    normalized_load_threshold: float = 0.99


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    """Conditions for experiment termination.

    Attributes:
        prompt_limit: Maximum number of prompts before stopping publishing (default: 400M).
                     The system will continue processing remaining queue messages.
        optimal_solution_programs: Number of additional programs to generate after finding
                                  the first optimal solution (default: 200K).
        target_solutions: Optional dict mapping (n, s_value) tuples to target scores for early termination.
                         Example: {(6,1): 10, (7,1): 16, (8,1): 30}
                         If None or empty dict, early termination based on optimal solutions is disabled.
    """
    prompt_limit: int = 400_000_000
    optimal_solution_programs: int = 200_000
    target_solutions: dict = dataclasses.field(default_factory=lambda: {
        (6, 1): 10,
        (7, 1): 16,
        (8, 1): 30,
        (9, 1): 52,
        (10, 1): 94,
        (11, 1): 172
    })


@dataclasses.dataclass
class Config:
  """Configuration of a FunSearch experiment.

  Attributes:
    programs_database: Configuration of the database.
    rabbitmq: Configuration for RabbitMQ connection.
    sampler: Configuration of the samplers.
    evaluator: Configuration of the evaluators (includes evaluation script path).
    prompt: Configuration for prompt construction (problem description, prompt style, few-shot, score display).
    wandb: Configuration for Weights & Biases logging.
    scaling: Configuration for dynamic scaling of samplers and evaluators.
    paths: Configuration for file system paths (log_dir, sandbox_base_path, backup).
    termination: Configuration for experiment termination conditions.
    num_samplers: Number of independent Samplers in the experiment.
    num_evaluators: Number of independent program Evaluators in the experiment.
    num_pdb: Number of independent program databases. Currently supports only one, but this does not create a bottleneck.
  """
  programs_database: ProgramsDatabaseConfig = dataclasses.field(default_factory=ProgramsDatabaseConfig)
  rabbitmq: RabbitMQConfig = dataclasses.field(default_factory=RabbitMQConfig)
  sampler: SamplerConfig = dataclasses.field(default_factory=SamplerConfig)
  evaluator: EvaluatorConfig = dataclasses.field(default_factory=EvaluatorConfig)
  prompt: PromptConfig = dataclasses.field(default_factory=PromptConfig)
  wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
  scaling: ScalingConfig = dataclasses.field(default_factory=ScalingConfig)
  paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)
  termination: TerminationConfig = dataclasses.field(default_factory=TerminationConfig)
  num_samplers: int = 5
  num_evaluators: int = 20
  num_pdb: int = 1
  random_seed: int = 42  # Random seed for full reproducibility (controls both prompt construction and LLM generation). If None, non-deterministic. Set to integer (e.g., 42) for reproducible experiments.




