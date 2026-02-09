# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration dataclasses for FunSearch experiments."""

import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class RabbitMQConfig:
    """RabbitMQ connection settings."""
    host: str = 'rabbitmq'  # localhost (RabbitMQ on same machine), rabbitmq (docker), or node IP (cluster)
    port: int = 5672
    management_port: int = 15672
    username: str = 'guest'
    password: str = 'guest'
    vhost: str = ''  # Empty for default vhost
    reconnect_delay: float = 5.0  # Seconds before first RabbitMQ reconnect attempt
    max_reconnect_delay: float = 60.0  # Cap for exponential backoff between RabbitMQ retries


@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
    """ProgramsDatabase settings."""
    num_islands: int = 10
    reset_programs: int = 1200  # Programs per island before weak islands reset
    cluster_sampling_temperature_init: float = 1 #0.1
    cluster_sampling_temperature_period: int = 30_000
    no_deduplication: bool = False
    save_lineage: bool = True  # Track parent/child relationships
    initial_program_copies: int = 1
    batch_size: int = 1
    batch_timeout: float = 0.01
    normalize_scores_for_sampling: bool = False  # Normalize cluster scores by baseline before softmax


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
    """Sampler settings."""
    prompts_per_batch: int = 50
    batch_timeout: float = 0.01
    samples_per_prompt: int = 2
    samples_per_prompt_mutation: int = 5  # Override for ReEvo mutation phase
    temperature_period = None  # Programs until LLM temperature decays (exploration to exploitation), None for fixed
    temperature: float = 0.9444444444444444
    max_new_tokens: int = 1024 #246 #246 for starcoder2-15b, 1024 for qwen3-8b
    top_p: float = 0.7777777777777778
    repetition_penalty: float = 1.222222
    reasoning_effort: str = None  # Only for OpenAI o1/o3/gpt-5 models
    max_retries: int = 3
    inference_timeout: int = 300
    model: str = "bigcode/starcoder2-15b"  # See https://docs.vllm.ai/en/latest/models/supported_models.html e.g., Qwen/Qwen3-8B, bigcode/starcoder2-15b
    cost_model: str = "fireworks_ai/accounts/fireworks/models/starcoder2-15b"  # fireworks_ai/accounts/fireworks/models/starcoder2-15b or fireworks_ai/accounts/fireworks/models/qwen3-8b, LiteLLM model name for pricing, see https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
    use_local_vllm: bool = True  # False for LiteLLM API calls
    use_chat_api: bool = False  # True to use vLLM chat() instead of generate(), use when providing system/user messages
    enable_thinking: bool = False  # Qwen3 thinking mode: None=model default (on), True=force on, False=force off
    model_params_billions: float = None # For FLOP estimation, None to disable
    api_base: str = None
    api_key: str = None  # None loads from .env
    cache_dir: str = "/mnt/models"
    gpu_memory_utilization: float = 0.85
    prefetch_multiplier: int = 2
    tensor_parallel_size: int | str = 1  # "auto" or explicit 1, 2, 4, 8 for multi-GPU
    enforce_eager: bool = False  # True to skip CUDA graph compilation (faster startup, required for multi-GPU)


@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
    """Evaluator settings."""
    evaluation_script_path: str = "/workspace/DistributedFunSearch/src/disfun/specifications/ECC/evaluation/graph_networkx.py"
    initial_functions_dir: str = "/workspace/DistributedFunSearch/src/disfun/specifications/ECC/initial_functions/graph_networkx"
    s_values: List[int] = dataclasses.field(default_factory=lambda: [2])  # Deletions to correct 
    start_n: List[int] = dataclasses.field(default_factory=lambda: [7])  # Shortest code length to evaluate
    end_n: List[int] = dataclasses.field(default_factory=lambda: [12])  # Longest code length to evaluate
    mode: str = "last"  # last, average, weighted, relative_difference
    timeout: int = 30
    max_workers: int = 2  # Parallel CPU processes per evaluator
    q: int = 2  # Alphabet size (2 for binary, 4 for DNA)
    graph_dir: str = "/workspace/DistributedFunSearch/src/graphs"
    graph_type: str = "deletion"  # "deletion" or "ids"
    prefetch_count: int = 15
    sandbox_memory_limit_gb: float = 1 # 1 for deletions, 20 for ids
    sandbox_debug: bool = False  # Log sandbox compile/runtime errors to stderr
    startup_delay: float = 3.0  # Seconds between evaluator starts (staggers graph loading to prevent OOM)


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    """Prompt building settings."""
    strategy: str = "funsearch"  # funsearch, eoh, reevo
    spec_dir: str = "/workspace/DistributedFunSearch/src/disfun/specifications/ECC"
    variant: str = "deletions"  # ECC variant: "deletions" or "ids"
    imports_file: str = "imports/networkx.txt"

    # FunSearch
    funsearch_template: str = "funsearch/templates/single_turn/thought.txt"
    funsearch_problem_desc: str = "funsearch/problem_descriptions/instruction.txt"
    funsearch_string_hint: str | None = None  # e.g., "funsearch/problem_descriptions/string_hint.txt"
    funsearch_system_message: str | None = "funsearch/system_messages/single_turn/thought.txt" #"funsearch/system_messages/basic.txt"
    funsearch_evaluation_preamble: str | None = None  # Include evaluation setup in prompt
    funsearch_evaluation_script: str | None = None  # Include evaluation script in prompt (shows LLM how functions are scored)
    fewshot_num_examples: int = 2
    fewshot_include_description: bool = True  # Include <description> tags in few-shot examples (if template requests description output)

    # EoH
    eoh_styles_dir: str = "eoh/styles"
    eoh_problem_desc: str = "eoh/problem_descriptions/baseline.txt"
    eoh_func_desc: str = "eoh/func_desc.txt"
    eoh_system_message: str = "eoh/system_message.txt"

    # ReEvo
    reevo_templates_dir: str = "reevo/templates"
    reevo_problem_desc: str = "reevo/problem/problem_desc.txt"
    reevo_func_desc: str = "reevo/problem/func_desc.txt"
    reevo_generator_system: str = "reevo/system/generator.txt"
    reevo_reflector_system: str = "reevo/system/reflector.txt"
    reevo_initial_reflection: str | None = "reevo/initial_reflection.txt"

    # Score display
    show_scores: bool = True
    score_display_mode: str = "relative"  # absolute, relative
    best_known_solutions: dict = dataclasses.field(default_factory=lambda: {  # Best known or upper bound scores
        (7, 2): 5,   # (n, s): score, q is constant per run, set in EvaluatorConfig
        (8, 2): 7,
        (9, 2): 11,
        (10,2): 16,
        (11, 2): 24,
        (12, 2): 40
    })

    # Docstring templates for priority functions in few-shot examples (paths relative to spec_dir)
    docstring_baseline: str = "docstrings/baseline.txt"  # For worse/single function
    docstring_improved: str = "docstrings/improved.txt"  # For better-performing function
    score_label_absolute: str = "docstrings/score_label_absolute.txt"
    score_label_relative: str = "docstrings/score_label_relative.txt"


@dataclasses.dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases settings."""
    enabled: bool = True
    project: str = "disfun_s2_qwen8b"
    entity: str = "franziweindel-technical-university-of-munich"
    run_name: str = None  # None for auto-generated
    run_name_tag: str = "evol_desc_code"
    log_interval: int = 300  # Seconds
    tags: List[str] = dataclasses.field(default_factory=list)
    checkpoints_base_path: str = "/mnt/disfun/checkpoints/s1"


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    """File system paths."""
    log_dir: str = "./logs"
    backup_enabled: bool = False
    backup_dir: str = "./code_backups"


@dataclasses.dataclass(frozen=True)
class ScalingConfig:
    """Dynamic scaling settings."""
    enabled: bool = False

    # Resource logging (monitoring only, doesn't affect scaling)
    resource_log_interval: int = 60  # Seconds between resource log entries

    # General scaling (applies to both samplers and evaluators)
    check_interval: int = 60  # Seconds between scaling checks
    max_samplers: int = 1000
    max_evaluators: int = 200
    idle_checks_before_scale_down: int = 2  # Queue must be empty this many checks before scale-down
    min_process_lifetime: int = 60  # Min seconds before process eligible for termination
    min_system_memory_gib: int = 30  # Min free system RAM to spawn any process

    # Sampler scaling (GPU processes)
    sampler_scale_up_threshold: int = 50  # sampler_queue depth to trigger scale-up
    min_gpu_memory_gib: int = 35  # Min free GPU memory to start sampler
    max_gpu_utilization: int = 50  # Max GPU util % to allow sampler scale-up
    min_gpu_util_for_scale_down: int = 80  # Skip scale-down if selected sampler's GPU util exceeds this
    sampler_termination_timeout: int = 45  # Seconds to wait for graceful shutdown (vLLM GPU cleanup)

    # Evaluator scaling (CPU processes)
    evaluator_scale_up_threshold: int = 10  # evaluator_queue depth to trigger scale-up
    cpu_usage_threshold: int = 99  # Max avg CPU % to allow evaluator scale-up
    normalized_load_threshold: float = 0.99  # Max load/cores to allow evaluator scale-up
    sample_count: int = 10  # CPU samples to average (smooths spiky readings)
    sample_interval: int = 1  # Seconds between CPU samples
    termination_timeout: int = 30  # Seconds to wait for graceful evaluator shutdown


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    """Experiment termination conditions."""
    termination_mode: str = "cost"  # "iterations" or "cost"
    iteration_limit: int = 400_000 #400_000  # Used when termination_mode="iterations"
    cost_limit: float = 120  # Max cost in USD, used when termination_mode="cost"
    stop_on_optimal: bool = False  # If True, stop early after finding optimal solution
    optimal_solution_programs: int = 20_000  # Extra iterations after optimal found (only if stop_on_optimal=True)
    target_solutions: dict = dataclasses.field(default_factory=lambda: {
        (7, 2): 5,   # (n, s): score, q is constant per run, set in EvaluatorConfig
        (8, 2): 7,
        (9, 2): 11,
        (10,2): 16,
        (11, 2): 24,
        (12, 2): 40
    })

@dataclasses.dataclass(frozen=True)
class ThroughputConfig:
    """Throughput measurement settings."""
    enabled: bool = False
    run_duration_minutes: int = 3 # excluding warmup
    window_duration_minutes: int = 1
    warmup_minutes: int = 3
    cooldown_seconds: int = 10  # Wait between sweep configs for cleanup
    output_file: str = "throughput_results.json"


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    """Parameter sweep settings for throughput measurement. Only used when running throughput sweeps (multiple configurations).
    For single-config throughput runs, set all fields to None.
    All combinations will be evaluated (Cartesian product).
    """
    evaluator_prefetch: List[int] = dataclasses.field(default_factory=lambda: [15])  # Evaluator prefetch_count values
    max_workers: List[int] = dataclasses.field(default_factory=lambda: [3])  # Evaluator max_workers values
    prompts_per_batch: List[int] = dataclasses.field(default_factory=lambda: [100])  # Sampler batch size values
    sampler_prefetch_multiplier: List[int] = dataclasses.field(default_factory=lambda: [2])  # Sampler prefetch multiplier values


@dataclasses.dataclass
class Config:
    """Main experiment configuration."""
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
    sweep: SweepConfig = dataclasses.field(default_factory=SweepConfig)
    num_samplers: int = 4
    num_evaluators: int = 60
    num_pdb: int = 1
    random_seed: int = 13  # None for non-deterministic
