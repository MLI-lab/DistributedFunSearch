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

* Works inside an async RabbitMQ loop (`consume_and_process` - `get_prompt`).
* Logs cumulative evaluator CPU - sampler GPU - and I/O token counts.
* Saves and resumes from checkpoint.
* Enforces deduplication (hash-based) and version-mismatch checks.
* Stops early after an optimal solution or a prompt/solution quota.
* Implements different evaluation scoring (last - average - weighted - relative difference to a traget solution)
"""

import ast
import copy
import dataclasses
import time
import logging
import re
import os
import signal
import numpy as np
import asyncio
import random
import aiohttp
from typing import Any
from collections.abc import Mapping, Sequence
from disfun import code_manipulation
from disfun import specification_loader
from disfun import checkpoint as checkpoint_module
from disfun import wandb_logging
import json
import aio_pika

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


logger = logging.getLogger('main_logger')

Signature = tuple[float, ...]
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


def _reduce_score(scores_per_test: dict, mode: str = "last", start_n: list = [6], end_n: list = [11], s_values: list = [1], target_signatures=None) -> float:
    """
    Reduces per-test scores into a single score based on the specified mode.
    Extracts (n - s) from full problem instance tuples and aggregates for each s in s_values.

    Available modes:
    - "last": Uses the score for the largest n (end_n) for each s value.
    - "average": Averages scores across all n values for each s - then averages across s values.
    - "weighted": Weights scores by n to prioritize larger n-values.
    - "relative_difference": Uses relative difference (actual - target) / target to normalize across targets.

    Args:
        scores_per_test (dict): Dictionary mapping problem instance tuples to scores.
                               Keys can be full tuples like (n, s, q) or (n, s, q, k, ...).
                               The first two elements are used as (n, s) for aggregation.
        mode (str): Scoring method to use.
        start_n (list): Start values for n per s-value.
        end_n (list): End values for n per s-value.
        s_values (list): List of s-values to consider.
        target_signatures (dict, optional): Dictionary of target sizes for each (n - s).

    Returns:
        float: Final reduced score.
    """
    try:
        # Convert string keys to tuples and extract (n - s) from full problem instance tuples
        parsed_scores = {}
        for k, v in scores_per_test.items():
            key = ast.literal_eval(k) if isinstance(k, str) else k
            # Extract (n - s) from full tuple: take first two elements
            ns_key = tuple(key[:2]) if isinstance(key, tuple) and len(key) >= 2 else key
            parsed_scores[ns_key] = v
    except Exception as e:
        raise ValueError(f"Failed to parse scores_per_test keys: {e}") from e

    if not (len(start_n) == len(end_n) == len(s_values)):
        raise ValueError("The number of elements in start_n, end_n, and s_values must match.")

    if mode == "relative_difference" and target_signatures is None:
        raise ValueError("target_signatures must be provided for 'relative_difference' mode.")

    per_s_scores = []

    for s, s_start_n, s_end_n in zip(s_values, start_n, end_n, strict=True):
        all_dimensions = [(n, s) for n in range(s_start_n, s_end_n + 1)]

        if mode == "last":
            per_s_scores.append(parsed_scores.get(all_dimensions[-1], 0))

        elif mode == "average":
            complete_scores = [parsed_scores.get(dim, 0) for dim in all_dimensions]
            per_s_scores.append(sum(complete_scores) / len(complete_scores) if complete_scores else 0)

        elif mode == "weighted":
            weights = [dim[0] for dim in all_dimensions]
            weighted_sum = sum(parsed_scores.get(dim, 0) * w for dim, w in zip(all_dimensions, weights, strict=True))
            total_weight = sum(weights)
            per_s_scores.append(weighted_sum / total_weight if total_weight > 0 else 0)

        elif mode == "relative_difference":
            relative_scores = []
            for dim in all_dimensions:
                actual = parsed_scores.get(dim, 0)
                target = target_signatures.get(dim, None)
                if target is not None:
                    relative_scores.append((actual - target) / target)
            per_s_scores.append(sum(relative_scores) / len(relative_scores) if relative_scores else 0)

        else:
            raise ValueError("Invalid mode. Available modes are 'last', 'average', 'weighted', and 'relative_difference'.")

    return sum(per_s_scores) / len(per_s_scores) if per_s_scores else 0


def _format_scores_for_prompt(
    scores_per_test: dict,
    display_mode: str,
    best_known_solutions: dict,
    absolute_label: str,
    relative_label: str
) -> str:
    """
    Formats scores for inclusion in function docstrings.

    Args:
        scores_per_test: Dictionary mapping (n,s) to achieved scores.
        display_mode: Either "absolute" or "relative".
        best_known_solutions: Dictionary mapping (n,s) to baseline scores.
        absolute_label: Prefix text for absolute scores.
        relative_label: Prefix text for relative improvements.

    Returns:
        Formatted string like "Absolute scores: {(6,1): 8, (7,1): 14}" or
                            "Relative to baseline: {(6,1): +0.0%, (7,1): +7.1%}".
    """
    parsed_scores = {}
    for k, v in scores_per_test.items():
        key = eval(k) if isinstance(k, str) else k
        parsed_scores[key] = v

    if display_mode == "absolute":
        items = [f"{k}: {v}" for k, v in sorted(parsed_scores.items())]
        return f"{absolute_label} {{{', '.join(items)}}}"

    elif display_mode == "relative":
        improvements = []
        for dim in sorted(parsed_scores.keys()):
            score_ours = parsed_scores.get(dim, 0)
            score_baseline = best_known_solutions.get(dim, None)

            if score_baseline is not None and score_baseline != 0:
                rel_improvement = ((score_ours - score_baseline) / abs(score_baseline)) * 100
                improvements.append(f"{dim}: {rel_improvement:+.1f}%")
            else:
                improvements.append(f"{dim}: {score_ours}")

        return f"{relative_label} {{{', '.join(improvements)}}}"

    return ""


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase - to be sent to Samplers."""
    code: str
    version_generated: int
    island_id: int
    expected_version: int = None

    def to_dict(self):
        """Returns prompt as dict (for JSON serialization)."""
        return {
            "code": self.code,
            "version_generated": self.version_generated,
            "island_id": self.island_id,
            "expected_version": self.expected_version,
        }

    @staticmethod
    def from_dict(data):
        """Creates Prompt from dict."""
        return Prompt(**data)


class ProgramsDatabase:
    """A collection of programs - organized as islands.

    The ProgramsDatabase maintains a population of evolved programs across multiple islands
    for diversity. It implements evolutionary lineage tracking to record parent-child
    relationships between programs.

    Evolutionary Lineage Tracking:
    -----------------------------
    Each program is assigned:
    - program_id: Unique identifier (auto-incrementing)
    - parent_ids: List of program IDs from the few-shot prompt that generated it
    - generation: Evolutionary depth (0 for baseline - max(parent_generations) + 1 for offspring)
    - timestamp: Creation time

    Special Handling for Island Resets:
    -----------------------------------
    When weak islands are reset - founder programs (best programs from surviving islands)
    are copied to the reset island. These founder programs inherit lineage from their
    source program: the new founder's parent_ids contains the original program's program_id,
    maintaining the evolutionary chain across island boundaries.

    All lineage information is logged to self.lineage_log and tracked in W&B metrics
    under the 'lineage/' namespace.
    """

    def __init__(
        self,
        connection: aio_pika.RobustConnection,
        channel: aio_pika.RobustChannel,
        database_queue: aio_pika.Queue,
        sampler_queue: aio_pika.Queue,
        evaluator_queue: aio_pika.Queue,
        config,
        function_to_evolve: str,
        checkpoint_file: str = None,
        save_checkpoints_path: str=None,
        mode: str=None,
        start_n=[6],
        end_n=[11],
        s_values=[1],
        no_deduplication=False,
        prompt_limit=400_000,
        optimal_solution_programs=20_000,
        max_drain_time=600,
        target_signatures=None,
        show_eval_scores=False,
        display_mode="absolute",
        best_known_solutions=None,
        absolute_label="Absolute scores:",
        relative_label="Relative to baseline:",
        q=2,
        wandb_config=None,
        sampler_config=None,
        evaluator_config=None,
        prompt_config=None,
        run_name=None,
        rabbitmq_config=None
    ):
        self._islands = []
        self._config = config

        # Use shared connection manager for RabbitMQ
        from disfun import process_utils
        self._conn_manager = process_utils.RabbitMQConnectionManager(
            config=rabbitmq_config,
            component_name="ProgramsDatabase",
            queue_names=["database_queue", "sampler_queue", "evaluator_queue"],
            logger=logger,
            timeout=300
        )
        # Initialize with passed-in connection
        self._conn_manager.connection = connection
        self._conn_manager.channel = channel
        if database_queue:
            self._conn_manager.queues["database_queue"] = database_queue
        if sampler_queue:
            self._conn_manager.queues["sampler_queue"] = sampler_queue
        if evaluator_queue:
            self._conn_manager.queues["evaluator_queue"] = evaluator_queue

        self.samples_per_batch = config.prompts_per_batch
        self._function_to_evolve = function_to_evolve
        self._best_score_per_island = [-float('inf')] * config.num_islands
        self._best_program_per_island = [None] * config.num_islands
        self._best_scores_per_test_per_island = [None] * config.num_islands
        self._last_reset_time = time.time()
        self._total_resets = 0
        self.save_checkpoints_path = save_checkpoints_path
        self.mode=mode
        self.start_n= start_n
        self.end_n = end_n
        self.s_values= s_values
        self.no_deduplication = no_deduplication
        self.prompt_limit = prompt_limit
        self.found_optimal_solution = False
        self.optimal_solution_programs = optimal_solution_programs
        self.prompts_since_optimal = 0
        self.target_signatures=target_signatures
        self.max_drain_time = max_drain_time
        self._prompt_limit_reached = False
        self._drain_start_time = None

        self.show_eval_scores = show_eval_scores
        self.display_mode = display_mode
        self.best_known_solutions = best_known_solutions or {}
        self.absolute_label = absolute_label
        self.relative_label = relative_label
        self.q = q

        if self.display_mode == "relative" and not self.best_known_solutions:
            logger.warning("display_mode='relative' requires best_known_solutions - falling back to 'absolute'")
            self.display_mode = "absolute"

        self.cumulative_evaluator_cpu_time = 0.0
        self.cumulative_sampler_gpu_time = 0.0

        self.cumulative_input_tokens  = 0
        self.cumulative_output_tokens = 0

        # Model parameters for FLOP estimation (2N FLOPs per token where N = params)
        self.model_params_billions = sampler_config.model_params_billions if sampler_config and hasattr(sampler_config, 'model_params_billions') else None

        self.duplicate_prompts=0
        self.total_prompts=0 # equals total processed messages as each message stored triggers a prompt
        self.total_stored_programs = 0
        self.version_mismatch_discarded = 0
        self.duplicates_discarded=0
        self.execution_failed = 0
        self.next_sampler_id = 0  # Counter for unique sampler IDs (saved to checkpoint for reproducibility)

        # Evolutionary lineage tracking (optional - can be disabled via config)
        self.save_lineage = config.save_lineage if hasattr(config, 'save_lineage') else False
        self.next_program_id = 1  # Counter for assigning unique program IDs
        self.lineage_log = [] if self.save_lineage else None  # Only initialize if enabled
        self._prompt_to_parents = {} if self.save_lineage else None
        self._program_id_to_generation = {}  # O(1) lookup for parent generation calculation

        # Lazy initialization of locks (will be created on first access)
        self._island_locks = None
        self._locks_initialized = False

        # Template-based prompt system
        self.evaluator_config = evaluator_config
        self.prompt_config = prompt_config
        self._template_str = None
        self._placeholder_contents = None
        self._function_args = None
        self._inout_spec = None
        self._function_header_template = None
        self._template_loaded = False
        self._logged_initial_prompt = False

        for _ in range(config.num_islands):
            island = {}
            island['clusters'] = {}
            island['version'] = 0
            island['num_programs'] = 0
            island['hash_set'] = set()  # O(1) deduplication lookup
            self._islands.append(island)

        # Store W&B config for later initialization (defer to avoid blocking)
        # IMPORTANT: Initialize these BEFORE loading checkpoint so checkpoint values aren't overwritten
        self.wandb_enabled = False
        self.wandb_config = wandb_config
        self.wandb_run_name = run_name  # Use the provided run_name (may be auto-generated)
        self.wandb_run_id = None  # Will be set after wandb.init or loaded from checkpoint

        # Load checkpoint if provided (this may overwrite wandb_run_id)
        self.load_checkpoint_file(checkpoint_file)
        # Build comprehensive config for W&B
        self.wandb_init_config = {
            # ProgramsDatabase config
            "num_islands": config.num_islands,
            "fewshot_num_examples": prompt_config.fewshot_num_examples if prompt_config else 2,
            "reset_period": config.reset_period,
            "reset_programs": config.reset_programs,
            "cluster_sampling_temperature_init": config.cluster_sampling_temperature_init,
            "cluster_sampling_temperature_period": config.cluster_sampling_temperature_period,
            "prompts_per_batch_database": config.prompts_per_batch,
            "no_deduplication": config.no_deduplication,
            # Evaluator config
            "mode": mode,
            "start_n": str(start_n),
            "end_n": str(end_n),
            "s_values": str(s_values),
            "q": q,
            # Prompt config
            "show_eval_scores": show_eval_scores,
            "display_mode": display_mode,
            # Limits
            "prompt_limit": prompt_limit,
            "optimal_solution_programs": optimal_solution_programs,
            "target_signatures": str(target_signatures) if target_signatures else None,
        }

        # Add evaluator config if provided
        if evaluator_config:
            self.wandb_init_config.update({
                "timeout": evaluator_config.timeout,
                "max_workers": evaluator_config.max_workers,
            })

        # Add sampler config if provided
        if sampler_config:
            self.wandb_init_config.update({
                "samples_per_prompt": sampler_config.samples_per_prompt,
                "temperature": sampler_config.temperature,
                "temperature_period": sampler_config.temperature_period,
                "max_new_tokens": sampler_config.max_new_tokens,
                "top_p": sampler_config.top_p,
                "repetition_penalty": sampler_config.repetition_penalty,
                "model": sampler_config.model,
                "prompts_per_batch_sampler": sampler_config.prompts_per_batch,
                "model_params_billions": getattr(sampler_config, 'model_params_billions', None),
            })

        self._wandb_initialized = False

        # Load template system (once at init)
        self._load_template_system()

    def _load_template_system(self):
        """Load template and placeholder contents once at init."""
        if self._template_loaded or self.prompt_config is None:
            return

        try:
            from pathlib import Path

            # Load template
            self._template_str = specification_loader.load_template(self.prompt_config.template_path)

            # Load placeholder contents (files and directories)
            self._placeholder_contents = specification_loader.load_placeholder_contents(
                self.prompt_config.placeholders
            )

            # Extract function signature from initial function
            initial_func_path = next(Path(self.evaluator_config.initial_functions_dir).glob("*.txt"))
            self._function_args, self._return_type = specification_loader.extract_function_signature(str(initial_func_path))

            # Load inout_spec template from components if provided, otherwise empty
            inout_spec_path = self.prompt_config.placeholders.get("inout_spec")
            if inout_spec_path and Path(inout_spec_path).exists():
                inout_spec_template = Path(inout_spec_path).read_text().strip()
                self._inout_spec = inout_spec_template.format(
                    function_args=self._function_args,
                    return_type=self._return_type
                )
            else:
                self._inout_spec = ""

            # Pre-compute function_header template ({version} and {prev_version} replaced per-prompt)
            self._function_header_template = f"def {self._function_to_evolve}_v{{version}}({self._function_args}) -> {self._return_type}:\n    \"\"\"Improved version of `{self._function_to_evolve}_v{{prev_version}}`.\"\"\""

            self._template_loaded = True
            logger.info(f"Loaded template system: template={self.prompt_config.template_path}, args={self._function_args}")

        except Exception as e:
            logger.error(f"Failed to load template system: {e}")
            self._template_loaded = True  # Prevent retry

    def load_checkpoint_file(self, checkpoint_file: str):
        logger.info(f"Checkpoint file is {checkpoint_file}")
        if checkpoint_file is not None:
            checkpoint_module.load_checkpoint(checkpoint_file, self)
        else:
            return

    def serialize_checkpoint(self) -> dict:
        """Serializes the necessary state of the database for checkpointing."""
        return checkpoint_module.serialize_checkpoint(self)

    async def periodic_checkpoint(self):
        """Periodically save checkpoints."""
        await checkpoint_module.periodic_checkpoint(self)


    def _compute_wandb_metrics(self) -> dict:
        """Compute metrics for Weights & Biases logging."""
        return wandb_logging.compute_wandb_metrics(self)

    def _get_program_by_id(self, program_id: int):
        """Find a program by its ID across all islands."""
        return wandb_logging.get_program_by_id(self, program_id)

    def _trace_lineage(self, program_id: int, max_depth: int = 100):
        """Trace the full evolutionary lineage of a program."""
        return wandb_logging.trace_lineage(self, program_id, max_depth)

    def _generate_lineage_html(self, program_id: int, island_id: int):
        """Generate an HTML visualization of a program's evolutionary lineage."""
        return wandb_logging.generate_lineage_html(self, program_id, island_id)

    def _generate_lineage_tree_diagram(self, program_id: int, island_id: int):
        """Generate a simple tree diagram showing the genealogy structure."""
        return wandb_logging.generate_lineage_tree_diagram(self, program_id, island_id)

    def _log_top_programs_table(self):
        """Log a W&B table with the best program from each island and their lineage."""
        wandb_logging.log_top_programs_table(self)

    async def _initialize_wandb(self):
        """Initialize W&B asynchronously (called once on first logging attempt)."""
        await wandb_logging.initialize_wandb(self)

    async def periodic_wandb_logging(self):
        """Periodically log metrics to Weights & Biases."""
        await wandb_logging.periodic_wandb_logging(self)

    def finish_wandb_run(self):
        """Explicitly finish the W&B run when the experiment is truly complete."""
        wandb_logging.finish_wandb_run(self)

    # Properties for backward compatibility - delegate to connection manager
    @property
    def connection(self):
        return self._conn_manager.connection

    @connection.setter
    def connection(self, value):
        self._conn_manager.connection = value

    @property
    def channel(self):
        return self._conn_manager.channel

    @channel.setter
    def channel(self, value):
        self._conn_manager.channel = value

    @property
    def database_queue(self):
        return self._conn_manager.get_queue("database_queue")

    @database_queue.setter
    def database_queue(self, value):
        self._conn_manager.queues["database_queue"] = value

    @property
    def sampler_queue(self):
        return self._conn_manager.get_queue("sampler_queue")

    @sampler_queue.setter
    def sampler_queue(self, value):
        self._conn_manager.queues["sampler_queue"] = value

    @property
    def evaluator_queue(self):
        return self._conn_manager.get_queue("evaluator_queue")

    @evaluator_queue.setter
    def evaluator_queue(self, value):
        self._conn_manager.queues["evaluator_queue"] = value

    async def _close_connection(self):
        """Delegate to shared connection manager."""
        await self._conn_manager.close()

    async def _ensure_connection(self):
        """Delegate to shared connection manager."""
        return await self._conn_manager.ensure_connection()

    async def consume_and_process(self) -> None:
        """Main consume loop with automatic connection recovery.

        Uses the same reconnection pattern as Sampler and Evaluator for consistency.
        """
        batch_size = 10
        batch_timeout = 0.01
        reconnect_delay = 5.0
        max_reconnect_delay = 60.0

        logger.info("ProgramsDatabase: consume_and_process started")

        async def _consume_loop():
            """Inner consume loop - processes messages from the queue."""
            await self.channel.set_qos(prefetch_count=batch_size)

            async with self.database_queue.iterator() as stream:
                batch = []
                batch_start_time = time.time()

                async for message in stream:
                    logger.debug(f"Received message: {message.body.decode()}")
                    batch.append(message)
                    current_time = time.time()

                    # Check if the batch should be processed
                    if len(batch) >= batch_size or (current_time - batch_start_time) >= batch_timeout:
                        await self.process_batch(batch)
                        batch.clear()
                        batch_start_time = current_time

        # Main reconnection loop
        while True:
            try:
                # Ensure connection is alive (reconnect if needed)
                connected = await self._ensure_connection()
                if not connected:
                    logger.error(f"ProgramsDatabase: Failed to establish connection, retrying in {reconnect_delay:.1f}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    continue

                # Reset delay on successful connection
                reconnect_delay = 5.0

                # Run the consume loop
                await _consume_loop()

                # If consume loop exits normally, break
                break

            except asyncio.CancelledError:
                logger.info("ProgramsDatabase: Cancelled, exiting...")
                break

            except (aio_pika.exceptions.AMQPConnectionError,
                    aio_pika.exceptions.ChannelClosed,
                    aio_pika.exceptions.ChannelInvalidStateError,
                    ConnectionError,
                    OSError) as e:
                # Connection lost - attempt to reconnect
                logger.warning(
                    f"ProgramsDatabase: Connection error: {e}. "
                    f"Reconnecting in {reconnect_delay:.1f}s..."
                )
                await self._close_connection()
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                continue

            except Exception as e:
                # Unexpected error - log and retry
                logger.error(
                    f"ProgramsDatabase: Unexpected error: {e}. "
                    f"Reconnecting in {reconnect_delay:.1f}s...",
                    exc_info=True
                )
                await self._close_connection()
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                continue


    #@async_time_execution
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
            self.total_prompts += 1
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

                self.cumulative_input_tokens  += input_tokens
                self.cumulative_output_tokens += output_tokens

                logger.debug(f"Updated cumulative CPU time: {self.cumulative_evaluator_cpu_time:.2f} seconds")
                logger.debug(f"Updated cumulative GPU time: {self.cumulative_sampler_gpu_time:.2f} seconds")

                if data["new_function"] == "return":
                    await self.get_prompt()
                    self.execution_failed += 1
                    logger.debug("Received 'return' for new_function. Skipping registration.")
                    return

                try:
                    if isinstance(data["new_function"], dict):
                        program = code_manipulation.Function(**data["new_function"])
                    else:
                        program = code_manipulation.Function.deserialize(data["new_function"])
                except Exception as e:
                    logger.error(f"Failed to convert program to Function instance: {e}")
                    await self.get_prompt()

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

        # Check if reset period is defined
        if self._config.reset_period is not None:
            # Only check the timing if reset_period is not None
            if (time.time() - self._last_reset_time > self._config.reset_period):
                all_islands_sufficiently_populated = all(island['num_programs'] >= self._config.reset_programs for island in self._islands)

                if all_islands_sufficiently_populated:
                    logger.info(f"Reset period exceeded and islands have {self._config.reset_programs} or more programs, resetting islands.")
                    self._last_reset_time = time.time()
                    try:
                        await self.reset_islands()
                    except Exception as e:
                        logger.error(f"Error in reset islands: {e}")
                else:
                    logger.warning("Reset period exceeded, but not all islands have enough programs. Skipping reset for now.")
        else:
            # If reset_period is None - only check population
            all_islands_sufficiently_populated = all(island['num_programs'] >= self._config.reset_programs for island in self._islands)
            if all_islands_sufficiently_populated:
                logger.info("Reset period not defined, but all islands have enough programs. Proceeding to reset islands.")
                try:
                    await self.reset_islands()
                except Exception as e:
                    logger.error(f"Error in reset islands: {e}")
            else:
                logger.debug("Reset period not defined, but not all islands have enough programs. Skipping reset for now.")

        # Acquire lock for this island to prevent race conditions during deduplication check and registration
        async with self._island_locks[island_id]:
            # Proceed with program registration logic
            island = self._islands[island_id]

            if not self.no_deduplication and self.function_body_exists(island, hash_value):
                self.duplicates_discarded += 1
                logger.debug("Program with identical body already exists in island. Skipping registration.")
                return

            if expected_version is not None:
                current_version = island['version']
                if current_version != expected_version:
                    logger.warning(f"Island {island_id} version mismatch. Expected: {expected_version}, Actual: {current_version}")
                    self.version_mismatch_discarded += 1
                    return

            self._register_program_in_island(program, island_id, scores_per_test, hash_value, parent_ids)


    def _register_program_in_island(self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest, hash_value: int = None, parent_ids: list[int] = None):
        """Register a program in an island and assign evolutionary lineage.

        Args:
            program: The Function object to register
            island_id: Target island ID
            scores_per_test: Dictionary of test scores
            hash_value: Hash of program's output for deduplication
            parent_ids: List of program IDs that were in the few-shot prompt (default: [])

        Lineage Assignment:
        ------------------
        Each registered program receives:
        - program_id: Unique auto-incrementing identifier
        - parent_ids: Programs from the few-shot prompt that generated this program
        - generation: max(parent_generations) + 1 - or 0 if no parents (baseline)
        - timestamp: Current time

        The lineage information is logged to self.lineage_log for tracking evolutionary trajectories.
        """
        self.total_stored_programs += 1
        island = self._islands[island_id]
        clusters = island['clusters']
        signature = self._get_signature(scores_per_test)
        program.hash_value = hash_value

        # Calculate score once and reuse
        score = _reduce_score(scores_per_test, self.mode, self.start_n, self.end_n, self.s_values, self.target_signatures)

        # Assign lineage tracking information
        if parent_ids is None:
            parent_ids = []

        program.program_id = self.next_program_id
        self.next_program_id += 1
        program.parent_ids = parent_ids

        # Calculate generation: max of parent generations + 1 - or 0 if no parents
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

            # If the score is equal to the best score - check the program signature
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
        """Reset the weakest half of islands with founders from the best islands.

        This method maintains diversity by periodically resetting underperforming islands.
        The weakest islands (by best score) are cleared - and each is seeded with the best
        program from a randomly selected surviving island.

        Lineage Tracking During Resets:
        --------------------------------
        Founder programs maintain evolutionary continuity across island boundaries.
        When a program is copied as a founder to a reset island - it receives a new program_id
        but its parent_ids contains the original program's program_id. This creates an
        evolutionary link showing the program was "migrated" from another island rather than
        evolved from a prompt.
        """
        # Ensure locks are initialized before resetting
        self._ensure_locks_initialized()

        try:
            await self.sampler_queue.purge()
            await self.evaluator_queue.purge()
        except Exception as e:
            logger.error(f"Could not remove all messages from the queue: {e}")
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

        if self.found_optimal_solution:
            logger.info(f"In self.found_optimal_solution: with it being equal to {self.found_optimal_solution:}")
            if self.prompts_since_optimal >= self.optimal_solution_programs:
                logger.info(f"Found an optimal solution and processed {self.optimal_solution_programs} additional programs. Stopping further publishing.")
                return  # Stop publishing once the additional limit is reached
            self.prompts_since_optimal += 1  # Track additional programs after the optimal solution
            logger.info(f"Functions processed since optimal: {self.prompts_since_optimal}")

        elif self.total_prompts >= self.prompt_limit:
            if not self._prompt_limit_reached:
                self._prompt_limit_reached = True
                self._drain_start_time = time.time()
                await self._handle_prompt_limit_reached()
            return

        logger.debug(f"len(self._islands) {len(self._islands)}")
        island_id = np.random.randint(len(self._islands))
        logger.debug(f"Island id is {island_id}")
        island = self._islands[island_id]

        code, flag_duplicate, version_generated, parent_ids = self._generate_prompt_for_island(island)
        expected_version = island['version']

        # Log initial prompt to W&B (once)
        if not self._logged_initial_prompt and code:
            logger.info(f"Initial prompt:\n{'='*80}\n{code}\n{'='*80}")
            if self.wandb_enabled and self._wandb_initialized and wandb.run:
                try:
                    wandb.run.summary["initial_prompt"] = code
                except Exception as e:
                    logger.warning(f"Failed to log initial prompt to W&B: {e}")
            self._logged_initial_prompt = True

        prompt = Prompt(code, version_generated, island_id, expected_version)
        message_data = {
            "prompt": prompt.to_dict(),
            "total_registered_programs": island['num_programs'],
            "flag":flag_duplicate,
            "parent_ids": parent_ids  # Include parent IDs for lineage tracking
        }

        try:
            serialized_message = json.dumps(message_data)
            await self.channel.default_exchange.publish(
                aio_pika.Message(body=serialized_message.encode()),
                routing_key='sampler_queue'
            )
            logger.debug("Database: Successfully published prompt to sampler with total registered programs.")
        except Exception as e:

            logger.error(f"Database: Error during prompt preparation or message sending: {e}")


    def _generate_prompt_for_island(self, island, multiple=False) -> tuple[str | None, int, int, list[int]]:
        """Generate a prompt for an island.

        Returns:
            tuple: (prompt - flag_duplicate - version_generated - parent_ids)
        """
        clusters = island['clusters']
        signatures = list(clusters.keys())
        fewshot_num_examples = self.prompt_config.fewshot_num_examples if self.prompt_config else 2
        if not signatures:
            logger.warning(f"No clusters found in island {island}. Skipping prompt generation.")
            return None, False, 0, []

        def compute_valid_signatures_and_probabilities(signatures, exclude_signature=None):
            """Helper function to compute valid signatures and probabilities."""
            filtered_signatures = [s for s in signatures if s != exclude_signature] if exclude_signature else signatures
            cluster_scores = np.array([clusters[s]['score'] for s in filtered_signatures])
            period = self._config.cluster_sampling_temperature_period
            temperature = self._config.cluster_sampling_temperature_init * (1 - (island['num_programs'] % period) / period)
            while True:
                try:
                    probabilities = _softmax(cluster_scores, temperature)
                    logger.debug(f"Probabilities are {probabilities}")
                except Exception as e:
                    logger.error(f"Cannot compute softmax: {e}")
                    break

                valid_indices = np.where(probabilities > 1e-6)[0]
                valid_probabilities = probabilities[valid_indices]
                valid_signatures = [filtered_signatures[i] for i in valid_indices]
                logger.debug(f"Valid sig are {valid_signatures}")

                if len(valid_signatures) > 0:
                    return valid_signatures, valid_probabilities

                # Reduce temperature if no valid signatures are found
                temperature *= 0.9
                if temperature < 1e-6:
                    logger.warning("Temperature reduced below threshold. Falling back to uniform sampling.")
                    break

            # Fallback: uniform sampling
            logger.warning("Using uniform sampling as fallback.")
            valid_signatures = filtered_signatures
            valid_probabilities = np.ones(len(filtered_signatures)) / len(filtered_signatures)
            return valid_signatures, valid_probabilities

        # Compute valid signatures and probabilities.
        valid_signatures, valid_probabilities = compute_valid_signatures_and_probabilities(signatures)
        sampled_programs = []
        sampled_signatures = set()
        parent_ids = []  # Track parent program IDs
        logger.debug(f"Length of valid sig: {len(valid_signatures)}")

        # If only one valid signature is available - sample from it once.
        if len(valid_signatures) == 1:
            selected_signature = valid_signatures[0]
            cluster = clusters[selected_signature]
            cluster_programs = cluster['programs']
            logger.debug(f"Selected signature: {selected_signature} with programs {cluster_programs}")
            sampled_signatures.add(selected_signature)
            if len(cluster_programs) >= 1:
                program = self.sample_program(cluster)
                scores = cluster.get('scores_per_test', {})
                sampled_programs.append((program, scores))
                # Track parent ID
                if program.program_id is not None:
                    parent_ids.append(program.program_id)
                version_generated = 1
                prompt, flag = self._generate_prompt(sampled_programs)
                return prompt, flag, version_generated, parent_ids
            else:
                logger.warning("Single valid cluster has no programs. Skipping prompt generation.")
                return None, False, 0, []

        # If there are multiple valid signatures:
        # Determine the number of clusters to sample.
        if len(valid_signatures) >= fewshot_num_examples:
            logger.debug("Sampling from multiple valid clusters.")
            # Sample exactly fewshot_num_examples clusters without replacement.
            cluster_indices = np.random.choice(
                len(valid_signatures),
                size=fewshot_num_examples,
                p=valid_probabilities,
                replace=False
            )
            sampled_signatures.update([valid_signatures[i] for i in cluster_indices])
        else:
            # If fewer than desired valid clusters are available - use all available ones.
            logger.warning("Fewer valid clusters than fewshot_num_examples; using all available clusters.")
            sampled_signatures.update(valid_signatures)
            # Optionally - you could recalculate probabilities excluding these and sample additional ones if desired.

        # Sample one program from each selected cluster.
        for signature in sampled_signatures:
            cluster = clusters[signature]
            cluster_programs = cluster['programs']
            if not cluster_programs:
                logger.warning(f"Cluster {signature} has no programs. Skipping.")
                continue
            program = self.sample_program(cluster)
            scores = cluster.get('scores_per_test', {})
            sampled_programs.append((program, scores))
            # Track parent ID
            if program.program_id is not None:
                parent_ids.append(program.program_id)

        # Sort sampled programs by the corresponding cluster's score.
        sorted_programs = sorted(sampled_programs, key=lambda p: clusters[next(iter(sampled_signatures))]['score'])
        version_generated = len(sorted_programs)
        prompt, flag_duplicate = self._generate_prompt(sorted_programs)
        return prompt, flag_duplicate, version_generated, parent_ids

    def _format_scores(self, scores: dict) -> str:
        """Format scores for inclusion in fewshot docstrings."""
        return _format_scores_for_prompt(
            scores,
            self.display_mode,
            self.best_known_solutions,
            self.absolute_label,
            self.relative_label
        )

    def _generate_prompt(self, implementations_with_scores: Sequence[tuple]) -> str:
        """Generate prompt using template system."""
        if not self._template_loaded or self._template_str is None:
            logger.error("Template system not loaded")
            return None, False

        implementations = [impl for impl, _ in implementations_with_scores]
        scores_list = [scores for _, scores in implementations_with_scores]

        # Version functions for fewshot display
        implementations = copy.deepcopy(implementations)
        for i, impl in enumerate(implementations):
            impl.name = f'{self._function_to_evolve}_v{i}'
            if i > 0:
                impl.docstring = f'Improved version of `{self._function_to_evolve}_v{i - 1}`. {{score}}'

        num_examples = len(implementations)
        version = num_examples  # Next version after v0, v1, ... is vN where N = num_examples

        # Determine fewshot count (may be overridden by prompt_style)
        fewshot_override = None
        prompt = self._template_str

        # Fill static placeholders from pre-loaded contents
        for name, (content, style_dict) in self._placeholder_contents.items():
            if style_dict is not None:
                # Directory: sample from in-memory dict
                _, (content, fewshot_override) = random.choice(list(style_dict.items()))
            # For evaluation_script: strip priority function and wrap in code fences
            if name == "evaluation_script" and content:
                content = specification_loader.strip_function_from_code(content, "priority")
                content = f"```python\n{content}\n```"
            prompt = prompt.replace(f"{{{name}}}", content or "")

        # Determine actual fewshot count
        num_fewshot = fewshot_override if fewshot_override is not None else self.prompt_config.fewshot_num_examples
        fewshot_programs = list(zip(implementations[:num_fewshot], scores_list[:num_fewshot], strict=True))

        # Build fewshot examples (handles score display internally)
        fewshot_examples = specification_loader.build_fewshot_examples(
            fewshot_programs,
            self.prompt_config,
            self._format_scores if self.prompt_config.show_eval_scores else None
        )

        # Fill reserved dynamic placeholders
        prompt = prompt.replace("{fewshot_examples}", fewshot_examples)
        prompt = prompt.replace("{num_examples}", str(num_fewshot))
        prompt = prompt.replace("{version}", str(version))
        function_header = self._function_header_template.replace("{version}", str(version)).replace("{prev_version}", str(version - 1))
        prompt = prompt.replace("{function_header}", function_header)
        prompt = prompt.replace("{inout_spec}", self._inout_spec)

        # Merge adjacent docstrings: """\n""" or """ """ becomes single continuation
        # Match """, optional whitespace/newlines, then """ and merge them
        prompt = re.sub(r'"""\s*"""', '', prompt)

        # Clean up multiple consecutive blank lines from empty placeholders
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)

        # Check for duplicates
        duplicate_prompt = False
        if len(implementations) == 2 and implementations[0].hash_value == implementations[1].hash_value:
            duplicate_prompt = True
            self.duplicate_prompts += 1

        logger.debug(f"Template prompt constructed: {len(prompt)} characters")
        return prompt.rstrip('\n'), duplicate_prompt

    def function_body_exists(self, island, hash_value: int) -> bool:
        """O(1) check if a program with this hash exists in the island."""
        assert hash_value is not None, "Error: No hash value computed! Check that hash value condition in the specification script is set to match start_n."
        return hash_value in island['hash_set']

    def _get_signature(self, scores_per_test):
        if all(isinstance(k, str) for k in scores_per_test.keys()):
            scores_per_test = {ast.literal_eval(k): v for k, v in scores_per_test.items()}

        def ensure_hashable(val):
            if isinstance(val, list):
                return tuple(val)
            return val

        return tuple(ensure_hashable(scores_per_test[k]) for k in sorted(scores_per_test.keys()))

    def sample_program(self, cluster_data, temperature=1.0):
        """Samples a program from the cluster - favoring shorter programs."""
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

    async def _handle_prompt_limit_reached(self):
        """Handle prompt limit: purge queues and either shutdown or start drain watcher."""
        if self.max_drain_time == 0:
            logger.info(f"Reached {self.prompt_limit} prompts (max_drain_time=0). Purging all queues...")
            try:
                await self.sampler_queue.purge()
                await self.evaluator_queue.purge()
            except Exception as e:
                logger.error(f"Failed to purge queues: {e}")
            self._save_and_shutdown("All queues purged.")
        else:
            logger.info(f"Reached {self.prompt_limit} prompts. Purging sampler queue, draining evaluators (max_drain_time={self.max_drain_time}s)...")
            try:
                await self.sampler_queue.purge()
            except Exception as e:
                logger.error(f"Failed to purge sampler queue: {e}")
            asyncio.create_task(self._watch_for_drain())

    async def _watch_for_drain(self):
        """Monitor queues and trigger shutdown when drained or timeout expires."""
        empty_count = 0
        while True:
            await asyncio.sleep(10)
            elapsed = time.time() - self._drain_start_time

            if self.max_drain_time > 0 and elapsed >= self.max_drain_time:
                self._save_and_shutdown(f"Drain timeout ({self.max_drain_time}s) reached.")
                return

            try:
                if await self._check_queues_empty():
                    empty_count += 1
                    logger.info(f"All queues empty ({empty_count}/3) after {elapsed:.1f}s")
                    if empty_count >= 3:
                        self._save_and_shutdown(f"All queues drained after {elapsed:.1f}s.")
                        return
                else:
                    empty_count = 0
            except Exception as e:
                logger.error(f"Drain watcher error: {e}")
                empty_count = 0

    async def _check_queues_empty(self) -> bool:
        """Check if all queues are empty via RabbitMQ management API."""
        try:
            cfg = self._conn_manager._config.rabbitmq
            vhost = '%2F' if not cfg.vhost else cfg.vhost
            timeout = aiohttp.ClientTimeout(total=5)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                for queue_name in ['sampler_queue', 'evaluator_queue', 'database_queue']:
                    url = f"http://{cfg.host}:{cfg.management_port}/api/queues/{vhost}/{queue_name}"
                    async with session.get(url, auth=aiohttp.BasicAuth(cfg.username, cfg.password)) as resp:
                        if resp.status == 200 and (await resp.json()).get('messages', 0) > 0:
                            return False
            return True
        except Exception as e:
            logger.error(f"Error checking queue status: {e}")
            return False
