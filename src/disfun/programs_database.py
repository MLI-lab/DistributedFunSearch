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


"""Asynchronous RabbitMQ ProgramsDatabase.

Differences from the original DeepMind FunSearch version

* Works inside an async RabbitMQ loop.
* Logs cumulative evaluator CPU, sampler GPU, and I/O token counts.
* Saves and resumes from checkpoint.
* Enforces deduplication (hash-based) and version-mismatch checks.
* Stops early after an optimal solution or a prompt/solution quota.
* Implements different evaluation scoring (last, average, weighted. relative difference to a traget solution)
"""

import ast
import dataclasses
import time
import logging
import os
import signal
import numpy as np
import asyncio
import aio_pika
from typing import Any
from collections.abc import Mapping
from disfun.utils import code_manipulation
from disfun.utils import checkpointing as checkpoint_module
from disfun.utils import wandb_logging
from disfun.utils import prompt_builder
from disfun.utils.profiling import async_time_execution
from disfun.utils import rabbitmq
import json
import litellm

# Wandb import (optional)
try:
    import wandb
except ImportError:
    wandb = None


logger = logging.getLogger('main_logger')

# Type alias for per-test scores: maps test case (e.g., (n, s, q)) to achieved score
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.array(logits, dtype=np.float32)
    # Subtract the maximum for numerical stability
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits / temperature)
    probs = exp_logits / np.sum(exp_logits)
    total = probs.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        probs = probs / total
    return probs


def _group_scores_by_params(scores_dict: dict) -> dict:
    """Group scores by (s, q, ...) parameters, keyed by n.

    Input:  {(6,1,2): 10, (7,1,2): 15, (8,1,2): 22, (6,1,3): 12}
    Output: {(1,2): {6: 10, 7: 15, 8: 22}, (1,3): {6: 12}}
    """
    from collections import defaultdict
    groups = defaultdict(dict)
    for k, v in scores_dict.items():
        key = ast.literal_eval(k) if isinstance(k, str) else k
        key = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        n = key[0]
        group_key = key[1:] if len(key) > 1 else ()
        groups[group_key][n] = v
    return groups


def _reduce_score(
    scores_per_test: dict,
    mode: str,
    baseline_scores: dict | None = None
) -> float:
    """Reduce per-test scores into a single aggregate score.

    Groups by (s, q, ...), reduces each group by mode, averages across groups.
    Modes: "last" (largest n), "average", "weighted" (by n), "relative_difference" (vs baseline).
    """
    groups = _group_scores_by_params(scores_per_test)
    if not groups:
        return 0.0

    baseline_groups = _group_scores_by_params(baseline_scores) if baseline_scores else {}

    if mode == "relative_difference" and not baseline_groups:
        raise ValueError("baseline_scores required for 'relative_difference' mode")

    # Reduce each parameter group to a single score
    group_scores = []
    for group_key, n_scores in groups.items():
        if mode == "last":
            # Use score at largest n (n_scores is unordered, so find max)
            max_n = max(n_scores.keys())
            group_scores.append(n_scores[max_n])

        elif mode == "average":
            group_scores.append(sum(n_scores.values()) / len(n_scores))

        elif mode == "weighted":
            # Weight scores by n (larger n = more important)
            weighted_sum = sum(score * n for n, score in n_scores.items())
            total_weight = sum(n_scores.keys())
            group_scores.append(weighted_sum / total_weight if total_weight > 0 else 0)

        elif mode == "relative_difference":
            # Compute (actual - baseline) / baseline for each n
            baseline_n = baseline_groups.get(group_key, {})
            relative = []
            for n, actual in n_scores.items():
                base = baseline_n.get(n)
                if base is not None and base != 0:
                    relative.append((actual - base) / base)
            group_scores.append(sum(relative) / len(relative) if relative else 0)

        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'last', 'average', 'weighted', or 'relative_difference'.")

    # Average across all parameter groups
    return sum(group_scores) / len(group_scores) if group_scores else 0.0


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    All strategies use generation_prompt for the main prompt.
    ReEvo also uses reflection_prompt for the first stage.
    System messages are included for API models.
    """
    generation_prompt: str
    version_generated: int
    island_id: int
    expected_version: int = None
    reflection_prompt: str | None = None      # ReEvo first stage
    samples_per_prompt: int | None = None     # Override sampler default
    system_message: str | None = None         # System message for generation
    reflection_system_message: str | None = None  # System message for reflection (ReEvo)

    def to_dict(self):
        """Returns prompt as dict (for JSON serialization)."""
        return {
            "generation_prompt": self.generation_prompt,
            "version_generated": self.version_generated,
            "island_id": self.island_id,
            "expected_version": self.expected_version,
            "reflection_prompt": self.reflection_prompt,
            "samples_per_prompt": self.samples_per_prompt,
            "system_message": self.system_message,
            "reflection_system_message": self.reflection_system_message,
        }

    @staticmethod
    def from_dict(data):
        """Creates Prompt from dict."""
        return Prompt(**data)


class ProgramsDatabase:
    """Manages evolved programs across islands for diversity.

    Tracks lineage (program_id, parent_ids, generation) and periodically resets
    weak islands with founders from strong ones. Logs to W&B under 'lineage/'.
    """

    def __init__(
        self,
        config,
        function_to_evolve: str,
        connection_manager: rabbitmq.ConnectionManager,
        checkpoint_file: str = None,
        save_checkpoints_path: str = None,
        termination_config=None,
        best_known_solutions=None,
        wandb_config=None,
        sampler_config=None,
        evaluator_config=None,
        prompt_spec: prompt_builder.PromptSpec = None,
        run_name=None,
    ):
        self._islands = []
        self._config = config
        self._conn = connection_manager

        self._function_to_evolve = function_to_evolve
        self._best_score_per_island = [-float('inf')] * config.num_islands
        self._best_program_per_island = [None] * config.num_islands
        self._best_scores_per_test_per_island = [None] * config.num_islands
        self._total_resets = 0
        self.save_checkpoints_path = save_checkpoints_path
        self.termination_config = termination_config
        self.termination_mode = termination_config.termination_mode if termination_config else "iterations"
        self.iteration_limit = termination_config.iteration_limit if termination_config else 400_000
        self.cost_limit = termination_config.cost_limit if termination_config else None
        self.found_optimal_solution = False
        self.stop_on_optimal = termination_config.stop_on_optimal if termination_config else True
        self.optimal_solution_programs = termination_config.optimal_solution_programs if termination_config else 20_000
        self.prompts_since_optimal = 0
        self.target_signatures = termination_config.target_solutions if termination_config else None
        self._termination_limit_reached = False

        self.best_known_solutions = best_known_solutions or {}

        self.cumulative_evaluator_cpu_time = 0.0
        self.cumulative_sampler_gpu_time = 0.0

        self.cumulative_input_tokens  = 0
        self.cumulative_output_tokens = 0
        self.cumulative_cost = 0.0
        self.cost_model = sampler_config.cost_model if sampler_config and hasattr(sampler_config, 'cost_model') else None

        # Model parameters for FLOP estimation (2N FLOPs per token where N = params)
        self.model_params_billions = sampler_config.model_params_billions if sampler_config and hasattr(sampler_config, 'model_params_billions') else None

        self.duplicate_prompts = 0
        self.iterations = 0             # Number of evolutionary cycles completed (prompt, sample, evaluate, store)
        self.total_stored_programs = 0
        self.version_mismatch_discarded = 0
        self.duplicates_discarded = 0
        self.execution_failed = 0
        self.next_sampler_id = 0        # Unique ID for each sampler, used as random seed for reproducibility

        # Parallel vs sequential: tracks if database changed between consecutive prompt generations
        self.last_stored_count = 0      # total_stored_programs when previous prompt was generated
        self.parallel_prompts = 0       # Prompts generated from same database state as previous
        self.sequential_prompts = 0     # Prompts that benefited from new stored programs

        # Evolutionary lineage tracking (optional, can be disabled via config)
        self.save_lineage = config.save_lineage if hasattr(config, 'save_lineage') else False
        self.next_program_id = 1  # Counter for assigning unique program IDs
        self.lineage_log = [] if self.save_lineage else None  # Only initialize if enabled
        self._prompt_to_parents = {} if self.save_lineage else None
        self._program_id_to_generation = {}  # O(1) lookup for parent generation calculation

        # Lazy initialization of locks (will be created on first access)
        self._island_locks = None
        self._locks_initialized = False

        # Prompt building
        self.evaluator_config = evaluator_config
        self.prompt_spec = prompt_spec

        # ReEvo reflection state
        initial_reflection = getattr(prompt_spec, 'initial_reflection', '') or ''
        self.reevo_state = {
            "prior_reflection": initial_reflection,  # Current long-term reflection
            "new_reflections": [],  # Accumulated short-term reflections from crossover
        }
        self._pending_reflection_type = {}  # {(island_id, version): "short_term" | "long_term"}

        for _ in range(config.num_islands):
            island = {}
            island['clusters'] = {}
            island['version'] = 0
            island['num_programs'] = 0
            island['hash_set'] = set()  # O(1) deduplication lookup
            self._islands.append(island)

        # Store W&B config for later initialization 
        self.wandb_enabled = False
        self.wandb_config = wandb_config
        self.wandb_run_name = run_name  # Use the provided run_name (may be auto-generated)
        self.wandb_run_id = None  # Will be set after wandb.init or loaded from checkpoint

        # Load checkpoint if provided 
        self.load_checkpoint_file(checkpoint_file)

        # Build W&B config
        self.wandb_init_config = wandb_logging.build_wandb_init_config(
            config, prompt_spec, evaluator_config, sampler_config, termination_config
        )

        self._wandb_initialized = False
        self._logged_initial_prompt = False

    def load_checkpoint_file(self, checkpoint_file: str):
        logger.info(f"Checkpoint file is {checkpoint_file}")
        if checkpoint_file is not None:
            checkpoint_module.load_checkpoint(checkpoint_file, self)
        else:
            return

    async def consume_and_process(self) -> None:
        """Main consume loop with automatic connection recovery."""
        logger.info("ProgramsDatabase: consume_and_process started")

        while True:
            try:
                # Connect with retry (handles exponential backoff internally)
                if not await self._conn.connect_with_retry():
                    break  # Shutdown requested

                await self._consume_loop()
                break  # Normal exit

            except asyncio.CancelledError:
                logger.info("ProgramsDatabase: Cancelled, exiting...")
                break

            except Exception as e:
                logger.warning(f"ProgramsDatabase: {type(e).__name__}: {e}. Reconnecting...")
                await self._conn.close()

    async def _consume_loop(self):
        """Inner consume loop, processes messages from the queue."""
        batch_size = self._config.batch_size
        batch_timeout = self._config.batch_timeout

        await self._conn.channel.set_qos(prefetch_count=batch_size)

        async with self._conn.get_queue("database_queue").iterator() as stream:
            batch = []
            batch_start_time = time.time()

            async for message in stream:
                logger.debug(f"Received message: {message.body.decode()}")
                batch.append(message)
                current_time = time.time()

                if len(batch) >= batch_size or (current_time - batch_start_time) >= batch_timeout:
                    await self.process_batch(batch)
                    batch.clear()
                    batch_start_time = current_time


    @async_time_execution("Database")
    async def process_batch(self, batch: list[aio_pika.IncomingMessage]):
        try:
            tasks = [self.process_message(message) for message in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Database: Error processing message: {result}")
        except asyncio.CancelledError:
            logger.info("Process batch was cancelled.")
        except Exception as e:
            logger.error(f"Error in process_batch: {e}")

    async def process_message(self, message: aio_pika.IncomingMessage):
        try:
            self.iterations += 1
            async with message.process():
                data = json.loads(message.body.decode())

                # Update cumulative evaluator CPU and GPU times
                evaluator_cpu_time = data.get("cpu_time", 0.0)
                sampler_gpu_time = data.get("gpu_time", 0.0)
                input_tokens  = int(data.get("input_tokens", 0))
                output_tokens = int(data.get("output_tokens", 0))
                found_optimal_solution = data.get("found_optimal_solution", False)
                if found_optimal_solution and not self.found_optimal_solution:
                    self.found_optimal_solution = True  # Mark as found
                    self.prompts_since_optimal = 0  # Reset counter for additional programs

                self.cumulative_evaluator_cpu_time += evaluator_cpu_time
                self.cumulative_sampler_gpu_time += sampler_gpu_time
                self.cumulative_input_tokens += input_tokens
                self.cumulative_output_tokens += output_tokens

                # Calculate cost using LiteLLM pricing
                if self.cost_model and (input_tokens > 0 or output_tokens > 0):
                    try:
                        prompt_cost, completion_cost = litellm.cost_per_token(
                            model=self.cost_model,
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens
                        )
                        self.cumulative_cost += prompt_cost + completion_cost
                    except Exception as e:
                        logger.debug(f"Could not calculate cost: {e}")

                # Accumulate ReEvo reflections
                reflection_output = data.get("reflection_output")
                if reflection_output:
                    island_id = data.get("island_id")
                    expected_version = data.get("expected_version")
                    key = (island_id, expected_version)
                    reflection_type = self._pending_reflection_type.pop(key, None)

                    if reflection_type == "short_term":
                        self.reevo_state["new_reflections"].append(reflection_output)
                        logger.debug(f"Accumulated short-term reflection (total: {len(self.reevo_state['new_reflections'])})")
                    elif reflection_type == "long_term":
                        self.reevo_state["prior_reflection"] = reflection_output
                        self.reevo_state["new_reflections"] = []
                        logger.debug(f"Updated long-term reflection, cleared {len(self.reevo_state['new_reflections'])} short-term reflections")

                if data["new_function"] == "return":
                    await self.get_prompt()
                    self.execution_failed += 1
                    return

                try:
                    if isinstance(data["new_function"], dict):
                        program = code_manipulation.Function(**data["new_function"])
                    else:
                        program = code_manipulation.Function.deserialize(data["new_function"])
                except Exception as e:
                    logger.error(f"Failed to convert program to Function instance: {e}")
                    await self.get_prompt()
                    return

                island_id = data.get("island_id")
                parent_ids = data.get("parent_ids", [])  # Extract parent IDs for lineage tracking

                if island_id is None:
                    # Register the program to all islands
                    for i in range(len(self._islands)):
                        await self.register_program(program, i, data["scores_per_test"], data.get("expected_version", None), data.get("hash_value", None), parent_ids)
                else:
                    # Register the program to the specific island
                    await self.register_program(program, island_id, data["scores_per_test"], data.get("expected_version", None), data.get("hash_value", None), parent_ids)

                await self.get_prompt()

        except asyncio.CancelledError:
            logger.info("Process message was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Database: Error processing message: {e}")
            raise

    def _ensure_locks_initialized(self):
        """Lazily initialize island locks when first needed."""
        if not self._locks_initialized:
            self._island_locks = [asyncio.Lock() for _ in range(len(self._islands))]
            self._locks_initialized = True

    async def register_program(self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest, expected_version: int = None, hash_value: int = None, parent_ids: list[int] = None):
        # Ensure locks are initialized before use
        self._ensure_locks_initialized()

        # Check if all islands have enough programs for reset
        all_islands_sufficiently_populated = all(
            island['num_programs'] >= self._config.reset_programs
            for island in self._islands
        )
        if all_islands_sufficiently_populated:
            logger.info(f"All islands have {self._config.reset_programs}+ programs. Resetting islands.")
            try:
                await self.reset_islands()
            except Exception as e:
                logger.error(f"Error in reset islands: {e}")

        # Acquire lock for this island to prevent race conditions during deduplication check and registration
        async with self._island_locks[island_id]:
            # Proceed with program registration logic
            island = self._islands[island_id]

            if not self._config.no_deduplication and self.function_body_exists(island, hash_value):
                self.duplicates_discarded += 1
                return

            if expected_version is not None:
                current_version = island['version']
                if current_version != expected_version:
                    logger.warning(f"Island {island_id} version mismatch. Expected: {expected_version}, Actual: {current_version}")
                    self.version_mismatch_discarded += 1
                    return

            self._register_program_in_island(program, island_id, scores_per_test, hash_value, parent_ids)


    def _register_program_in_island(self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest, hash_value: int = None, parent_ids: list[int] = None):
        """Register a program in an island with lineage tracking.

        Assigns program_id, parent_ids, generation (max parent gen + 1), and timestamp.
        Logs to self.lineage_log if lineage tracking is enabled.
        """
        self.total_stored_programs += 1
        island = self._islands[island_id]
        clusters = island['clusters']
        signature = self._get_signature(scores_per_test)
        program.hash_value = hash_value

        # Calculate score once and reuse
        score = _reduce_score(scores_per_test, self.evaluator_config.mode, self.best_known_solutions)

        # Assign lineage tracking information
        if parent_ids is None:
            parent_ids = []

        program.program_id = self.next_program_id
        self.next_program_id += 1
        program.parent_ids = parent_ids

        # Calculate generation: max of parent generations + 1, or 0 if no parents
        if parent_ids:
            # O(1) lookup instead of scanning all programs
            max_parent_generation = max(
                (self._program_id_to_generation.get(pid, 0) for pid in parent_ids),
                default=0
            )
            program.generation = max_parent_generation + 1
        else:
            program.generation = 0

        # Update generation lookup dict
        self._program_id_to_generation[program.program_id] = program.generation
        program.timestamp = time.time()

        try:
            if signature not in clusters:
                logger.info(f"Creating new cluster with signature {scores_per_test}")
                cluster_data = {}
                cluster_data['score'] = score
                cluster_data['scores_per_test'] = scores_per_test
                cluster_data['programs'] = [program]
                clusters[signature] = cluster_data
            else:
                logger.info(f"Registering on cluster with signature {scores_per_test}")
                cluster_data = clusters[signature]
                cluster_data['programs'].append(program)

            island['num_programs'] += 1
            if hash_value is not None:
                island['hash_set'].add(hash_value)

            # Log lineage information for this program (only if enabled)
            if self.save_lineage:
                self.lineage_log.append({
                    'program_id': program.program_id,
                    'parent_ids': program.parent_ids,
                    'generation': program.generation,
                    'score': score,
                    'island_id': island_id,
                    'timestamp': program.timestamp,
                    'signature': signature
                })
                logger.debug(f"Logged lineage: program_id={program.program_id}, parent_ids={program.parent_ids}, generation={program.generation}, score={score}")

        except Exception as e:
            logger.error(f"Could not append program: {e}")

        try:
            # Check if the new score is higher than the current best score
            if score > self._best_score_per_island[island_id]:
                self._best_program_per_island[island_id] = program
                self._best_scores_per_test_per_island[island_id] = scores_per_test
                self._best_score_per_island[island_id] = score
                logger.info(f'Best score of island {island_id} increased to {score} with program {program} and scores {scores_per_test}')

            # If the score equals the best score, check the program signature
            elif score == self._best_score_per_island[island_id]:
                # Get the current best program's signature
                current_best_signature = self._get_signature(self._best_scores_per_test_per_island[island_id])

                # Compare signatures: if the new signature is lexicographically "larger"
                if signature > current_best_signature:
                    self._best_program_per_island[island_id] = program
                    self._best_scores_per_test_per_island[island_id] = scores_per_test
                    self._best_score_per_island[island_id] = score
                    logger.info(f'Best program of island {island_id} replaced with program {program} (signature comparison)')

        except Exception as e:
            logger.error(f"Could not update best score: {e}")

    async def reset_islands(self):
        """Reset the weakest half of islands, seeding each with a founder from a surviving island.

        Maintains diversity by clearing underperforming islands and migrating top programs.
        Founder programs get new program_id but keep parent link to original for lineage tracking.
        """
        # Ensure locks are initialized before resetting
        self._ensure_locks_initialized()

        try:
            await self._conn.get_queue("sampler_queue").purge()
            await self._conn.get_queue("evaluator_queue").purge()
        except Exception as e:
            logger.error(f"Could not purge queues: {e}")
        try:
            indices_sorted_by_score = np.argsort(self._best_score_per_island)
            num_islands_to_reset = self._config.num_islands // 2
            reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
            keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]

            if len(reset_islands_ids) == 0:
                logger.warning("No islands to reset. Skipping reset.")
                return

            for island_id in reset_islands_ids:
                async with self._island_locks[island_id]:
                    island = self._islands[island_id]
                    island['clusters'].clear()
                    island['hash_set'].clear()
                    island['version'] += 1
                    island['num_programs'] = 0

                    self._best_score_per_island[island_id] = -float('inf')
                    founder_island_id = np.random.choice(keep_islands_ids)
                    founder = self._best_program_per_island[founder_island_id]
                    founder_scores = self._best_scores_per_test_per_island[founder_island_id]
                    # Founder inherits from the original program
                    founder_parent_ids = [founder.program_id] if founder.program_id is not None else []
                    self._register_program_in_island(founder, island_id, founder_scores, None, founder_parent_ids)
                await self.get_prompt()
            self._total_resets += 1
            logger.info(f"Island reset #{self._total_resets} completed. Reset {len(reset_islands_ids)} islands.")
        except Exception as e:
            logger.error(f"Error during island reset: {e}")


    async def get_prompt(self) -> None:

        if self.found_optimal_solution and self.stop_on_optimal:
            logger.info(f"In self.found_optimal_solution: with it being equal to {self.found_optimal_solution:}")
            if self.prompts_since_optimal >= self.optimal_solution_programs:
                logger.info(f"Found an optimal solution and processed {self.optimal_solution_programs} additional programs. Stopping further publishing.")
                return  # Stop publishing once the additional limit is reached
            self.prompts_since_optimal += 1  # Track additional programs after the optimal solution
            logger.info(f"Functions processed since optimal: {self.prompts_since_optimal}")

        # Check termination based on mode (iterations or cost)
        elif self.termination_mode == "cost" and self.cost_limit is not None:
            if self.cumulative_cost >= self.cost_limit:
                if not self._termination_limit_reached:
                    self._termination_limit_reached = True
                    self._save_and_shutdown(f"Reached cost limit: ${self.cumulative_cost:.4f} >= ${self.cost_limit:.4f}")
        elif self.termination_mode == "iterations" and self.iterations >= self.iteration_limit:
            if not self._termination_limit_reached:
                self._termination_limit_reached = True
                self._save_and_shutdown(f"Reached {self.iteration_limit} iterations.")

        # Track parallel vs sequential: did database change since last prompt?
        if self.total_stored_programs == self.last_stored_count:
            self.parallel_prompts += 1
        else:
            self.sequential_prompts += 1
            self.last_stored_count = self.total_stored_programs

        island_id = np.random.randint(len(self._islands))
        island = self._islands[island_id]

        result = self._generate_prompt_for_island(island)
        expected_version = island['version']

        # Log initial prompt to W&B (once)
        if not self._logged_initial_prompt:
            logger.info(f"Initial prompt:\n{'='*80}\n{result['generation_prompt']}\n{'='*80}")
            if self.wandb_enabled and self._wandb_initialized and wandb.run:
                try:
                    wandb.run.summary["initial_prompt"] = result["generation_prompt"]
                except Exception as e:
                    logger.warning(f"Failed to log initial prompt to W&B: {e}")
            self._logged_initial_prompt = True

        prompt = Prompt(
            generation_prompt=result["generation_prompt"],
            version_generated=result["version_generated"],
            island_id=island_id,
            expected_version=expected_version,
            reflection_prompt=result["reflection_prompt"],
            samples_per_prompt=result["samples_per_prompt"],
            system_message=result.get("system_message"),
            reflection_system_message=result.get("reflection_system_message"),
        )

        # Track reflection type for ReEvo (to know how to accumulate when result comes back)
        template_name = result.get("template_name")
        if template_name in ("crossover",):
            self._pending_reflection_type[(island_id, expected_version)] = "short_term"
        elif template_name in ("mutation",):
            self._pending_reflection_type[(island_id, expected_version)] = "long_term"

        message_data = {
            "prompt": prompt.to_dict(),
            "total_registered_programs": island['num_programs'],
            "flag": result["flag_duplicate"],  # Signals duplicate few-shot examples for debugging
            "parent_ids": result["parent_ids"],
        }

        try:
            serialized_message = json.dumps(message_data)
            await self._conn.channel.default_exchange.publish(
                aio_pika.Message(body=serialized_message.encode()),
                routing_key='sampler_queue'
            )
        except Exception as e:

            logger.error(f"Database: Error during prompt preparation or message sending: {e}")


    def _generate_prompt_for_island(self, island, multiple=False) -> dict:
        """Generate a prompt by sampling programs from island clusters.

        Returns:
            dict with keys:
                - generation_prompt: str | None
                - reflection_prompt: str | None (ReEvo only)
                - samples_per_prompt: int | None
                - flag_duplicate: bool
                - version_generated: int
                - parent_ids: list[int]
        """
        clusters = island['clusters']
        signatures = list(clusters.keys())

        empty_result = {
            "generation_prompt": None,
            "reflection_prompt": None,
            "samples_per_prompt": None,
            "flag_duplicate": False,
            "version_generated": 0,
            "parent_ids": [],
        }

        if not signatures:
            logger.warning("No clusters in island. Skipping prompt generation.")
            return empty_result

        template_name, num_programs_needed = prompt_builder.select_template(
            self.prompt_spec, state=self.reevo_state
        )
        # Get samples_per_prompt from template requirements (may be None)
        template_req = self.prompt_spec.template_requirements.get(template_name)
        samples_per_prompt = template_req.samples_per_prompt if template_req else None

        # Compute sampling probabilities with temperature-based softmax
        valid_sigs, probs = self._compute_cluster_probabilities(island, signatures)
        if not valid_sigs:
            return empty_result

        # Sample clusters (use all if fewer than needed)
        num_to_sample = min(len(valid_sigs), num_programs_needed)
        if num_to_sample < num_programs_needed:
            logger.warning(f"Only {num_to_sample} cluster(s) available, need {num_programs_needed}. Using {num_to_sample} few-shot example(s).")

        indices = np.random.choice(len(valid_sigs), size=num_to_sample, p=probs, replace=False)
        selected_sigs = [valid_sigs[i] for i in indices]

        # Sample one program from each selected cluster
        sampled_programs = []
        parent_ids = []
        for sig in selected_sigs:
            cluster = clusters[sig]
            if not cluster['programs']:
                continue
            program = self.sample_program(cluster)
            sampled_programs.append((program, cluster.get('scores_per_test', {})))
            if program.program_id is not None:
                parent_ids.append(program.program_id)

        if not sampled_programs:
            return empty_result

        # Check for duplicate few-shot examples (same hash = low diversity)
        flag_duplicate = False
        if len(sampled_programs) == 2:
            h0, h1 = sampled_programs[0][0].hash_value, sampled_programs[1][0].hash_value
            if h0 is not None and h0 == h1:
                flag_duplicate = True
                self.duplicate_prompts += 1

        result = {
            "flag_duplicate": flag_duplicate,
            "version_generated": len(sampled_programs),
            "parent_ids": parent_ids,
            "samples_per_prompt": samples_per_prompt,
            "template_name": template_name,  # For ReEvo reflection tracking
        }

        # Build prompt(s) based on strategy
        if self.prompt_spec.strategy == prompt_builder.PromptStrategy.REEVO:
            # ReEvo: reflection + generation prompts
            refl_prompt, gen_prompt = prompt_builder.build_reevo_prompts(
                self.prompt_spec, template_name, sampled_programs, state=self.reevo_state
            )
            result["reflection_prompt"] = refl_prompt
            result["generation_prompt"] = gen_prompt
            result["system_message"] = self.prompt_spec.system_message
            result["reflection_system_message"] = self.prompt_spec.reflector_system_message
        else:
            # FunSearch/EoH: single generation prompt
            result["generation_prompt"] = prompt_builder.build_prompt(
                self.prompt_spec, template_name, sampled_programs, state=self.reevo_state
            )
            result["reflection_prompt"] = None
            result["system_message"] = self.prompt_spec.system_message
            result["reflection_system_message"] = None

        return result

    def _compute_cluster_probabilities(self, island, signatures):
        """Compute sampling probabilities for clusters using temperature-scaled softmax.

        Temperature decays cyclically: high (explore) -> low (exploit) -> reset.
        Filters out near-zero probabilities to avoid numerical issues in sampling.
        """
        if not signatures:
            return [], np.array([])

        clusters = island['clusters']
        scores = np.array([clusters[s]['score'] for s in signatures])

        # Temperature decays from init to ~0 over each period, then resets
        period = self._config.cluster_sampling_temperature_period
        progress = (island['num_programs'] % period) / period
        temp = self._config.cluster_sampling_temperature_init * (1 - progress)
        temp = max(temp, 0.01)  # Floor to avoid numerical issues

        try:
            probs = _softmax(scores, temp)
        except Exception as e:
            logger.warning(f"Softmax failed: {e}, using uniform")
            return signatures, np.ones(len(signatures)) / len(signatures)

        # Filter out near-zero probabilities
        valid_mask = probs > 1e-6
        if not valid_mask.any():
            return signatures, np.ones(len(signatures)) / len(signatures)

        valid_sigs = [s for s, m in zip(signatures, valid_mask) if m]
        valid_probs = probs[valid_mask]
        valid_probs /= valid_probs.sum()  # Renormalize

        return valid_sigs, valid_probs

    def function_body_exists(self, island, hash_value: int) -> bool:
        """O(1) check if a program with this hash exists in the island."""
        assert hash_value is not None, "Error: No hash value computed! Check that hash value condition in the specification script is set to match start_n."
        return hash_value in island['hash_set']

    def _get_signature(self, scores_per_test):
        """Get signature for tie-breaking when aggregate scores are equal.

        Returns tuple of scores sorted by keys in descending order (largest n first).
        This prioritizes harder test cases (larger n) in tie-breaking.
        """
        if all(isinstance(k, str) for k in scores_per_test.keys()):
            scores_per_test = {ast.literal_eval(k): v for k, v in scores_per_test.items()}

        def ensure_hashable(val):
            if isinstance(val, list):
                return tuple(val)
            return val

        # Reverse order: larger n compared first (harder cases prioritized)
        return tuple(ensure_hashable(scores_per_test[k]) for k in sorted(scores_per_test.keys(), reverse=True))

    def sample_program(self, cluster_data, temperature=1.0):
        """Samples a program from the cluster, favoring shorter programs."""
        programs = cluster_data['programs']
        if not programs:
            raise ValueError("Cluster contains no programs to sample.")

        lengths = np.array([len(str(program)) for program in programs])  # Program lengths
        if lengths.max() == lengths.min():
            probabilities = np.ones(len(programs)) / len(programs)  # Uniform sampling if all lengths are identical
        else:
            # Normalize lengths as negative values to favor shorter programs
            normalized_lengths = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-6)
            probabilities = _softmax(-normalized_lengths, temperature)  # Softmax over negative lengths
        # Sample a program based on the probabilities
        sampled_index = np.random.choice(len(programs), p=probabilities)
        return programs[sampled_index]

    def _save_and_shutdown(self, reason: str):
        """Save checkpoint and trigger graceful shutdown."""
        logger.info(f"{reason} Saving checkpoint and shutting down...")
        checkpoint_module.save_checkpoint(self)
        os.kill(os.getpid(), signal.SIGTERM)

