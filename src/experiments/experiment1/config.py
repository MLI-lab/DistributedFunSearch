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

Adjusted to include RabbitMQ setup, deduplication option and prompt specification.
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
    host: str = 'rabbitmq' #localhost or rabbitmq for docker or node IP address for cluster
    port: int = 5672
    management_port: int = 15672  # Management API port for monitoring
    username: str = 'guest'
    password: str = 'guest'
    vhost: str = ''  # Use '' for default vhost, or 'exp1', 'exp2', etc. for different experiments 
    

@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    num_islands: Number of islands to maintain for diversity.
    reset_period: The interval (in seconds) at which the weakest islands are reset. If None, resets occur only based on the number of stored programs.
    reset_programs: The number of stored programs after which the weakest islands are reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster sampling temperature.
    no_deduplication: Disable deduplication (default: False, set True to disable).
    save_lineage: Track parent-child relationships between programs. When enabled, logs lineage to W&B and saves HTML visualizations showing evolutionary trees.
    initial_program_copies: Number of copies of each initial seed function to publish at startup.
  """
  num_islands: int = 10
  reset_period: int = None
  reset_programs: int= 1200
  cluster_sampling_temperature_init: float = 0.1
  cluster_sampling_temperature_period: int = 30_000
  no_deduplication: bool = False
  save_lineage: bool = False
  initial_program_copies: int = 100


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
  """Configuration of a Sampler.

  Attributes:
    prompts_per_batch: Batch size for processing prompts received from the sampler_queue.
    samples_per_prompt: How many independently sampled program continuations to get for each prompt.
    temperature_period: Controls how fast the LLM's temperature decreases as more programs are registered. If None, dynamic temperature adjustment is disabled.
    temperature: Controls randomness: higher values increase diversity, lower values make outputs more deterministic.
    max_new_tokens: The maximum number of tokens the LLM can generate in response.
    top_p: Determines the range of likely tokens the model samples from, keeping only the most probable ones.
    repetition_penalty: Penalizes repetitive text, values >1 discourage repetition, 1 disables it.
    model: Any LiteLLM-supported model: https://docs.litellm.ai/docs/providers or local model from huggingface
    model_params_billions: Number of model parameters in billions for FLOP estimation (e.g., 15 for StarCoder2-15B). Used to compute inference FLOPs as 2*N*tokens. Set to None to disable FLOP logging.
    api_base: Custom API endpoint URL. Leave as None for standard cloud APIs.
    api_key: API key for authentication. Leave as None to automatically load from .env file.
    cache_dir: Optional directory for model cache (local models only). If None, defaults to ~/.cache/huggingface/
    reasoning_effort: Controls internal chain-of-thought reasoning depth. Only works with GPT-5/o3 models.
    max_retries: Maximum number of retry attempts for failed API calls (default: 3). Only applies to API models.
    inference_timeout: Timeout in seconds for vLLM inference (default: 300). If vLLM hangs beyond this, sampler exits for restart by ResourceManager.
    gpu_memory_utilization: Fraction of GPU memory to use for vLLM (default: 0.95). Lower values leave more headroom for CUDA graphs. Try 0.90 if you get OOM during cudagraph capture.
  """
  prompts_per_batch= 50  # Reduced from 50 to prevent OOM on 45GB GPUs
  samples_per_prompt: int = 2
  temperature_period= None
  temperature: float = 0.9444444444444444
  max_new_tokens: int = 246  # for reasoing set larger
  top_p: float =  0.7777777777777778
  repetition_penalty: float = 1.222222
  reasoning_effort: str = "None"  # Set to "minimal", "low", "medium", "high", or None.
  max_retries: int = 3  
  inference_timeout: int = 300
  model: str = "bigcode/starcoder2-15b" #"bigcode/starcoder2-15b"  # Local model (each sampler loads on GPU) or API model
  model_params_billions: float = 15.0  # StarCoder2-15B has 15B params. Set to None to disable FLOP logging.
  api_base: str = None  #  Only set for custom endpoints (vLLM server, Azure, etc.)
  api_key: str = None   # Leave as None, automatically loads OPENAI_API_KEY from /workspace/DistributedFunSearch/.env
  cache_dir: str = "/mnt/models"
  gpu_memory_utilization: float = 0.85  # Fraction of GPU memory for vLLM. Lower (0.90) if OOM during cudagraph capture.
  prefetch_multiplier: int = 2  # Sampler prefetch = prompts_per_batch * prefetch_multiplier. Higher values hide network latency by buffering next batch while GPU processes current batch.

@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
    """Configuration of the Evaluator.

    Attributes:
        evaluation_script_path: Absolute path to the evaluation script.
        initial_functions_dir: Directory containing initial seed functions (.txt files).
        s_values: List of error correction parameters.
        start_n: List of shortest code length for each s.
        end_n: List of longest code lengths for each s.
        mode: Mode for score reduction. Available options: 'last', 'average', 'weighted', 'relative_difference' (to PromptConfig.best_known_solutions)
        timeout: Timeout in seconds for the sandbox.
        max_workers: Number of parallel CPU processes per evaluator for evaluating functions on different inputs (default: 2).
        q: Alphabet size for the codes (default: 2 for binary). Set to 4 for DNA data storage.
        graph_dir: Directory where graph files (.lmdb) are stored. Default is "src/graphs" relative to project root.
        cache_graphs: Enable in-memory graph caching to avoid reloading graphs for every evaluation (default: False). Each evaluator will have its own cache, so this increases RAM usage.
        cache_size_limit_gb: Only cache graphs smaller than this size in GiB (default: 2.0). Set to float('inf') to cache all graphs regardless of size.
        prefetch_count: Number of messages to buffer from RabbitMQ (default: 5). Higher values improve pipeline parallelism.
        sandbox_memory_limit_gb: Memory limit per sandbox process in GiB (default: 10). Prevents runaway generated code from consuming all RAM.
    """
    evaluation_script_path: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/evaluation/graph_networkx.py"  # Options: no_graph.py, graph_networkx.py, graph_gt.py
    initial_functions_dir: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/initial_functions/graph_networkx"  # Use "no_graph" for on-the-fly evaluation, "graph" for graph variants
    s_values: List[int] = dataclasses.field(default_factory=lambda: [1])
    start_n: List[int] = dataclasses.field(default_factory=lambda: [6])  # Hash is computed for n==start_n[0] (automatically substituted in specification)
    end_n: List[int] = dataclasses.field(default_factory=lambda: [11])  # Match available graphs (s1_n6 through s1_n11)
    mode: str = "weighted"
    timeout: int = 30
    max_workers: int = 3
    q: int = 2
    graph_dir: str = "/workspace/DistributedFunSearch/src/graphs"
    cache_graphs: bool = True  # Enable in-memory graph caching (reduces I/O, increases RAM)
    cache_size_limit_gb: float = 2.0  # Only cache graphs smaller than this (GiB)
    prefetch_count: int = 15  # Messages to buffer from RabbitMQ for pipeline parallelism
    sandbox_memory_limit_gb: float = 1.0  # Memory limit per sandbox process (GiB)


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    """Configuration for template-based prompt construction.

    Attributes:
        template_path: Path to template file defining prompt structure via {placeholder} syntax.
        placeholders: Dict mapping placeholder names to file/directory paths.
                     If path is directory, one file is sampled at runtime.
                     Reserved placeholders (fewshot_examples, num_examples, version, function_header)
                     are computed at runtime and should not be in this dict.
                     Optional: inout_spec can be a path to a template file with {function_args} and
                     {return_type} placeholders (see components/inout_spec.txt). If not provided,
                     {inout_spec} is replaced with empty string.
        system_message_path: Path to system message file for API models. Set to None to disable.
        fewshot_num_examples: Number of examples in few-shot prompts.
        fewshot_show_thinking: Include <thinking> in few-shot examples.
        fewshot_show_thought: Include <thought> in few-shot examples.
        fewshot_show_code: Include code in few-shot examples.
        show_eval_scores: Include evaluation scores in fewshot docstrings.
        display_mode: Score display format: "absolute" or "relative".
        best_known_solutions: Dict mapping (n, s) to baseline scores (for relative mode).
        absolute_label: Prefix for absolute scores.
        relative_label: Prefix for relative scores.
    """
    # Template system
    template_path: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/templates/funsearch.txt"
    placeholders: dict = dataclasses.field(default_factory=lambda: {
        "problem_description": "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/problem_descriptions/baseline.txt",
        "prompt_style": "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/prompt_styles/starcoder2.txt",  # None for code completion, or path to file/directory
        "imports": "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/imports/networkx.txt",
        "fewshot_preamble": None, 
        "evaluation_preamble": None,  
        "evaluation_script": None,  
    })
    system_message_path: str | None = None 

    # Few-shot configuration
    fewshot_num_examples: int = 2
    fewshot_show_thinking: bool = False
    fewshot_show_thought: bool = False
    fewshot_show_code: bool = True

    # Score display
    show_eval_scores: bool = False
    display_mode: str = "relative"
    best_known_solutions: dict = dataclasses.field(default_factory=lambda: {
        (6, 1, 2): 10,
        (7, 1, 2): 16,
        (8, 1, 2): 30,
        (9, 1, 2): 52,
        (10, 1, 2): 94,
        (11, 1, 2): 172
    })

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
        run_name_tag: Tag to append to run name 
        log_interval: How often to log metrics in seconds (default: 300 = 5 minutes).
        tags: List of tags for this run.
        checkpoints_base_path: Base directory for checkpoints (default: "./Checkpoints"). Actual checkpoint folder will be: {checkpoints_base_path}/checkpoint_{run_name}/
    """
    enabled: bool = True
    project: str = "disfun"
    entity: str = "franziweindel-technical-university-of-munich"  # Set to your W&B username or team
    run_name: str = None  # Auto-generated with timestamp if None
    run_name_tag: str = "weighted_seed_13_rep"  # Tag appended to run name (e.g., "gpt4o", "starcoder2")
    log_interval: int = 300  # Log every 5 minutes
    tags: List[str] = dataclasses.field(default_factory=list) # e.g. ["gpt4o", "reasoning", "deduplication"]
    checkpoints_base_path: str = "/mnt/disfun/checkpoints" 


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
        check_interval: Time interval (in seconds) between consecutive scaling checks.
        max_samplers: Maximum number of samplers the system can scale up to.
        max_evaluators: Maximum number of evaluators the system can scale up to.
        sampler_scale_up_threshold: Number of messages in sampler_queue to trigger scale-up.
        evaluator_scale_up_threshold: Number of messages in evaluator_queue to trigger scale-up.
        min_gpu_memory_gib: Minimum free GPU memory in GiB required to start a new sampler.
                            Adjust based on your LLM size: StarCoder2-15B needs ~30 GiB.
        max_gpu_utilization: Maximum GPU utilization percentage to allow starting a new sampler.
        min_system_memory_gib: Minimum free system RAM in GiB required for scaling.
        cpu_usage_threshold: Maximum average CPU usage percentage to allow evaluator scale-up.
        normalized_load_threshold: Maximum normalized system load (load/cores) to allow evaluator scale-up.
    """
    enabled: bool = False
    check_interval: int = 60
    max_samplers: int = 1000
    max_evaluators: int = 200
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
        prompt_limit: Maximum number of prompts before stopping publishing.
                     The system will continue processing remaining queue messages.
        optimal_solution_programs: Number of additional programs to generate after finding
                                  the first optimal solution.
        max_drain_time: Maximum seconds to wait for evaluators to finish after prompt_limit
                       is reached. When limit is hit, sampler_queue is purged (stopping LLM calls).
                       Set to 0 for immediate shutdown (also purges evaluator queue).
                       Set to -1 for unlimited wait (drain all evaluator work).
        target_solutions: Optional dict mapping (n, s_value) tuples to target scores for early termination.
                         Example: {(6,1): 10, (7,1): 16, (8,1): 30}
                         If None or empty dict, early termination based on optimal solutions is disabled.
    """
    prompt_limit: int = 400_000
    optimal_solution_programs: int = 20_000
    max_drain_time: int = 600  # seconds to wait for evaluators (0 = unlimited)
    target_solutions: dict = dataclasses.field(default_factory=lambda: {
        (6, 1, 2): 10,
        (7, 1, 2): 16,
        (8, 1, 2): 30,
        (9, 1, 2): 52,
        (10, 1, 2): 94,
        (11, 1, 2): 172
    })


@dataclasses.dataclass(frozen=True)
class ThroughputConfig:
    """Configuration for throughput measurement mode.

    When enabled, runs for a fixed duration and reports iterations/hour with mean ± std
    computed across measurement windows. Uses num_samplers and num_evaluators from main config.

    Attributes:
        enabled: Enable throughput mode (default: False).
        run_duration_minutes: Total measurement period (default: 60).
        window_duration_minutes: Duration of each measurement window (default: 20).
        warmup_minutes: Warm-up period before measurements start (default: 5).
        output_file: Path to JSON file for results.
    """
    enabled: bool = False
    run_duration_minutes: int = 60
    window_duration_minutes: int = 20
    warmup_minutes: int = 5
    output_file: str = "throughput_results.json"


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
    throughput: Configuration for throughput measurement mode.
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
  throughput: ThroughputConfig = dataclasses.field(default_factory=ThroughputConfig)
  num_samplers: int = 3
  num_evaluators: int = 65
  num_pdb: int = 1
  random_seed: int = 13  # Random seed for full reproducibility (controls both prompt construction and LLM generation). If None, non-deterministic.



#  export PATH="$HOME/.local/bin:$PATH"