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
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000
    no_deduplication: bool = False
    save_lineage: bool = False  # Track parent/child relationships
    initial_program_copies: int = 100
    batch_size: int = 10
    batch_timeout: float = 0.01


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
    """Sampler settings."""
    prompts_per_batch: int = 50
    batch_timeout: float = 0.01
    samples_per_prompt: int = 2
    samples_per_prompt_mutation: int = 5  # Override for ReEvo mutation phase
    temperature_period = None  # Programs until LLM temperature decays (exploration to exploitation), None for fixed
    temperature: float = 0.9444444444444444
    max_new_tokens: int = 246
    top_p: float = 0.7777777777777778
    repetition_penalty: float = 1.222222
    reasoning_effort: str = None  # Only for OpenAI o1/o3/gpt-5 models
    max_retries: int = 3
    inference_timeout: int = 300
    model: str = "bigcode/starcoder2-15b"
    use_local_vllm: bool = True  # False for LiteLLM API calls
    model_params_billions: float = 15.0  # For FLOP estimation, None to disable
    api_base: str = None
    api_key: str = None  # None loads from .env
    cache_dir: str = "/mnt/models"
    gpu_memory_utilization: float = 0.85
    prefetch_multiplier: int = 2


@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
    """Evaluator settings."""
    evaluation_script_path: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/evaluation/graph_networkx.py"
    initial_functions_dir: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions/initial_functions/graph_networkx"
    s_values: List[int] = dataclasses.field(default_factory=lambda: [1])  # Deletions to correct 
    start_n: List[int] = dataclasses.field(default_factory=lambda: [6])  # Shortest code length to evaluate
    end_n: List[int] = dataclasses.field(default_factory=lambda: [11])  # Longest code length to evaluate
    mode: str = "relative_difference"  # last, average, weighted, relative_difference
    timeout: int = 30
    max_workers: int = 3  # Parallel CPU processes per evaluator
    q: int = 2  # Alphabet size (2 for binary, 4 for DNA)
    graph_dir: str = "/workspace/DistributedFunSearch/src/graphs"
    prefetch_count: int = 15
    sandbox_memory_limit_gb: float = 1.0


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    """Prompt building settings."""
    strategy: str = "funsearch"  # funsearch, eoh, reevo
    spec_dir: str = "/workspace/DistributedFunSearch/src/disfun/specifications/Deletions"
    imports_file: str = "imports/networkx.txt"

    # FunSearch
    funsearch_template: str = "funsearch/template.txt"
    funsearch_problem_desc: str = "funsearch/problem_descriptions/baseline.txt"
    funsearch_system_message: str | None = None
    funsearch_evaluation_preamble: str | None = None  # Include evaluation setup in prompt
    funsearch_evaluation_script: str | None = None  # Include evaluation script in prompt (shows LLM how functions are scored)
    fewshot_num_examples: int = 2

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
    show_scores: bool = False
    score_display_mode: str = "relative"  # absolute, relative
    best_known_solutions: dict = dataclasses.field(default_factory=lambda: {  # Best known or upper bound scores
        (6, 1, 2): 10,
        (7, 1, 2): 16,
        (8, 1, 2): 30,
        (9, 1, 2): 52,
        (10, 1, 2): 94,
        (11, 1, 2): 172
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
    project: str = "disfun"
    entity: str = "franziweindel-technical-university-of-munich"
    run_name: str = None  # None for auto-generated
    run_name_tag: str = "relative_difference_seed_13"
    log_interval: int = 300  # Seconds
    tags: List[str] = dataclasses.field(default_factory=list)
    checkpoints_base_path: str = "/mnt/disfun/checkpoints"


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    """File system paths."""
    log_dir: str = "./logs"
    sandbox_base_path: str = "/mnt/disfun/sandbox"
    backup_enabled: bool = False
    backup_dir: str = "./code_backups"


@dataclasses.dataclass(frozen=True)
class ScalingConfig:
    """Dynamic scaling settings."""
    enabled: bool = False
    check_interval: int = 60  # Seconds between scaling checks
    max_samplers: int = 1000
    max_evaluators: int = 200
    sampler_scale_up_threshold: int = 50  # Queue depth to trigger scale-up
    evaluator_scale_up_threshold: int = 10
    min_gpu_memory_gib: int = 35  # Min free GPU memory to start sampler
    max_gpu_utilization: int = 50  # Max GPU util % to start sampler
    min_system_memory_gib: int = 30
    cpu_usage_threshold: int = 99  # Max CPU % for evaluator scale-up
    normalized_load_threshold: float = 0.99  # Max load/cores for evaluator scale-up
    idle_checks_before_scale_down: int = 2
    min_process_lifetime: int = 60  # Min seconds before process eligible for scale down
    termination_timeout: int = 30  # Seconds to wait for graceful shutdown (evaluators)
    sampler_termination_timeout: int = 45  # Longer timeout for samplers (vLLM GPU cleanup)
    resource_log_interval: int = 60  # Seconds between resource log entries
    sample_count: int = 10  # Samples to average for logging and scaling decisions
    sample_interval: int = 1  # Seconds between samples


@dataclasses.dataclass(frozen=True)
class TerminationConfig:
    """Experiment termination conditions."""
    iteration_limit: int = 400_000
    optimal_solution_programs: int = 20_000  # Extra iterations after optimal found
    target_solutions: dict = dataclasses.field(default_factory=lambda: {
        (6, 1, 2): 10,   # (n, s, q): target_score
        (7, 1, 2): 16,
        (8, 1, 2): 30,
        (9, 1, 2): 52,
        (10, 1, 2): 94,
        (11, 1, 2): 172
    })


@dataclasses.dataclass(frozen=True)
class ThroughputConfig:
    """Throughput measurement settings."""
    enabled: bool = False
    run_duration_minutes: int = 80
    window_duration_minutes: int = 20
    warmup_minutes: int = 15
    output_file: str = "throughput_results.json"


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
    num_samplers: int = 2
    num_evaluators: int = 40
    num_pdb: int = 1
    random_seed: int = 13  # None for non-deterministic
