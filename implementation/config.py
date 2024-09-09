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

@dataclasses.dataclass(frozen=True)
class RabbitMQConfig:
    """Configuration for RabbitMQ connection.

    Attributes:
      host: The hostname of the RabbitMQ server.
      port: The port of the RabbitMQ server.
      username: Username for authentication with the RabbitMQ server.
      password: Password for authentication with the RabbitMQ server.
    """
    host: str = 'rabbitmqFW'
    port: int = 5672
    username: str = 'guest'
    password: str = 'guest'
  
@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
  """Configuration of a ProgramsDatabase.

  Attributes:
    functions_per_prompt: Number of previous programs to include in prompts.
    num_islands: Number of islands to maintain as a diversity mechanism.
    reset_period: How often (in seconds) the weakest islands should be reset.
    cluster_sampling_temperature_init: Initial temperature for softmax sampling of clusters within an island.
    cluster_sampling_temperature_period: Period of linear decay of the cluster sampling temperature.
  """
  functions_per_prompt: int = 2
  num_islands: int = 10
  reset_period: int = 4 * 60 * 60 
  cluster_sampling_temperature_init: float = 2 # changed from 0.1 to 1
  cluster_sampling_temperature_period: int = 30_000 # after 30_000 reset 
  prompts_per_batch= 10



@dataclasses.dataclass 
class Config:
  """Configuration of a FunSearch experiment.

  Attributes:
    programs_database: Configuration of the evolutionary algorithm.
    num_samplers: Number of independent Samplers in the experiment. 
    num_evaluators: Number of independent program Evaluators in the experiment..
    samples_per_prompt: How many independently sampled program continuations to obtain for each prompt.
  """ 
  # In this case, default_factory=ProgramsDatabaseConfig means that calling ProgramsDatabaseConfig() (without any arguments) will provide the default value.
  programs_database: ProgramsDatabaseConfig = dataclasses.field(default_factory=ProgramsDatabaseConfig)
  rabbitmq: RabbitMQConfig = dataclasses.field(default_factory=RabbitMQConfig)
  num_samplers: int = 1
  num_evaluators: int = 5
  num_pdb: int = 0
  samples_per_prompt: int = 4
  temperature: float = 0.2
  max_new_tokens: int = 60
  top_p: float = 0.9
  repetition_penalty: float = 1.2


