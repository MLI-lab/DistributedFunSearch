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

"""Configuration of a FunSearch experiment. Only data classes no methods"""
import dataclasses
from typing import List
import os



@dataclasses.dataclass(frozen=True)
class RabbitMQConfig:
    """Configuration for RabbitMQ connection.

    Attributes:
      host: The hostname of the RabbitMQ server.
      port: The port of the RabbitMQ server.
      username: Username for authentication with the RabbitMQ server.
      password: Password for authentication with the RabbitMQ server.
    """
    host: str = 'rabbitmq'
    port: int = 5672 
    username: str = 'guest' 
    password: str = 'guest' 
    vhost = "temp_1"
    

@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    functions_per_prompt: Number of previous programs to include in current prompt.
    num_islands: Number of islands to maintain for diversity.
    reset_period: The interval (in seconds) at which the weakest islands are reset. If None, resets occur only based on the number of stored programs.  
    reset_programs: The number of stored programs after which the weakest islands are reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster sampling temperature.
    prompts_per_batch: Batch size for processing prompts received from the database_queue
    no_deduplication: Disable deduplication (default: False, set True to disable).
  """
  functions_per_prompt: int = 2
  num_islands: int = 10
  reset_period: int = None
  reset_programs: int= 1200
  cluster_sampling_temperature_init: float = 0.1 
  cluster_sampling_temperature_period: int = 30_000 
  prompts_per_batch= 10
  no_deduplication: bool = False


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    prompts_per_batch: Batch size for processing prompts received from the sampler_queue
    samples_per_prompt: How many independently sampled program continuations to obtain for each prompt.
    temperature_period: Controls how fast the LLM's temperature decreases as more programs are registered. If None, dynamic temperature adjustment is disabled.
    temperature: Controls randomness; higher values increase diversity, lower values make outputs more deterministic.
    max_new_tokens: The maximum number of tokens the LLM can generate in response.
    top_p: Determines the range of likely tokens the model samples from, keeping only the most probable ones.
    repetition_penalty: Penalizes repetitive text; values >1 discourage repetition, while 1 disables it.
    gpt: Enable GPT mode (default: False). When enabled, GPU device assignment is disabled.
  """
  prompts_per_batch= 10
  samples_per_prompt: int = 2
  temperature_period= None
  temperature: float = 0.9444444444444444
  max_new_tokens: int = 246
  top_p: float =  0.7777777777777778 
  repetition_penalty: float = 1.222222
  gpt: bool = False   
  
def get_spec_path() -> str:
    # Get the absolute directory of this file
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # Look for the substring "FunDCC" in the path
    idx = base_dir.find("FunDCC")
    if idx != -1:
        fundcc_base = base_dir[: idx + len("FunDCC")]
    else:
        fundcc_base = base_dir
    # Build the path relative to the FunDCC folder
    return os.path.join(fundcc_base, "src", "fundcc", "specifications", "StarCoder2", "load_graph", "baseline.txt")


@dataclasses.dataclass(frozen=True)
class EvaluatorConfig:
    """Configuration of a ProgramsDatabase.

    Attributes:
        s_values: List of number of deletions s.
        start_n: List of shortest code length for each s.
        end_n: List of longest code lengths for each s.
        mode: Mode for score reduction. Available options: 'last', 'average', 'weighted'. 
        timeout: Timeout in seconds for the sandbox.
        eval_code: Include evaluation script in prompt. (default: False, set True to enable).
        include_nx: Include the nx package in the prompt (default: True, set False to disable).
        spec_path: Path to the specification file used in the experiment.
    """
    s_values: List[int] = dataclasses.field(default_factory=lambda: [2])
    start_n: List[int] = dataclasses.field(default_factory=lambda: [7])
    end_n: List[int] = dataclasses.field(default_factory=lambda: [12])
    mode: str = "last"  
    timeout: int = 30 
    eval_code: bool = False
    include_nx: bool = True 
    spec_path: str = dataclasses.field(default_factory=get_spec_path)


@dataclasses.dataclass 
class Config:
  """Configuration of a FunSearch experiment.

  Attributes:
    programs_database: Configuration of the database.
    rabbitmq: Configuration for RabbitMQ connection.
    sampler: Configuration of the samplers.
    evaluator: Configuration of the evaluators.
    num_samplers: Number of independent Samplers in the experiment. 
    num_evaluators: Number of independent program Evaluators in the experiment.
    num_pdb: Number of independent program databases. Currently supports only one, but this does not create a bottleneck.
  """ 
  programs_database: ProgramsDatabaseConfig = dataclasses.field(default_factory=ProgramsDatabaseConfig)
  rabbitmq: RabbitMQConfig = dataclasses.field(default_factory=RabbitMQConfig)
  sampler: SamplerConfig = dataclasses.field(default_factory=SamplerConfig) 
  evaluator: EvaluatorConfig = dataclasses.field(default_factory=EvaluatorConfig) 
  num_samplers: int = 4
  num_evaluators: int = 40
  num_pdb: int = 1




