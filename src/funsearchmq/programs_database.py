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


"""Asynchronous RabbitMQ ProgramsDatabase.

Differences from the original DeepMind FunSearch version

* Works inside an async RabbitMQ loop (`consume_and_process`, `get_prompt`).
* Logs cumulative evaluator CPU, sampler GPU, and I/O token counts.
* Saves and resumes from checkpoint.
* Enforces deduplication (hash-based) and version-mismatch checks.
* Stops early after an optimal solution or a prompt/solution quota.
* Implements different evaluation scoring (last, average, weighted, relative difference to a traget solution)
"""

import copy
import dataclasses
import time
import logging
import numpy as np
import scipy
import asyncio
import pickle
import threading
import gc
import os
import multiprocessing
from typing import Mapping, Any, List, Sequence, Optional
from funsearchmq import code_manipulation
import json
import aio_pika
import re
from logging.handlers import RotatingFileHandler
import psutil
from logging import FileHandler
import datetime
from funsearchmq.profiling import async_time_execution

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



import json

def _reduce_score(scores_per_test: dict, mode: str = "last", start_n: list = [6], end_n: list = [11], s_values: list = [1], target_signatures=None) -> float:
    """
    Reduces per-test scores into a single score based on the specified mode.
    Generates (n, s) pairs for each s in s_values, where n is in [start_n, end_n].
    Available modes:
    - "last": Uses the score of the largest n for each s.
    - "average": Averages scores across all (n, s) pairs.
    - "weighted": Weighs scores by n to prioritize larger n-values.
    - "relative_difference": Uses relative difference (actual - target) / target to normalize across targets.

    Args:
        scores_per_test (dict): Dictionary mapping (n, s) tuples to scores.
        mode (str): Scoring method to use.
        start_n (list): Start values for n per s-value.
        end_n (list): End values for n per s-value.
        s_values (list): List of s-values to consider.
        target_signatures (dict, optional): Dictionary of target sizes for each (n, s).

    Returns:
        float: Final reduced score.
    """
    try:
        # Convert string keys in scores_per_test to (int, int) tuples
        parsed_scores = {eval(k): v for k, v in scores_per_test.items()}
    except Exception as e:
        raise ValueError(f"Failed to parse scores_per_test keys: {e}")

    if not (len(start_n) == len(end_n) == len(s_values)):
        raise ValueError("The number of elements in start_n, end_n, and s_values must match.")

    if mode == "relative_difference" and target_signatures is None:
        raise ValueError("target_signatures must be provided for 'relative_difference' mode.")

    per_s_scores = []

    for s, s_start_n, s_end_n in zip(s_values, start_n, end_n):
        all_dimensions = [(n, s) for n in range(s_start_n, s_end_n + 1)]

        if mode == "last":
            per_s_scores.append(parsed_scores.get(all_dimensions[-1], 0))

        elif mode == "average":
            complete_scores = [parsed_scores.get(dim, 0) for dim in all_dimensions]
            per_s_scores.append(sum(complete_scores) / len(complete_scores) if complete_scores else 0)

        elif mode == "weighted":
            weights = [dim[0] for dim in all_dimensions]
            weighted_sum = sum(parsed_scores.get(dim, 0) * w for dim, w in zip(all_dimensions, weights))
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


def _get_q_description(q: int) -> str:
    """
    Returns a descriptive string for the alphabet size.

    Args:
        q: Alphabet size (2 for binary, 4 for quaternary, etc.)

    Returns:
        String like "binary" for q=2, "quaternary" for q=4, or "{q}-ary" for other values.
    """
    if q == 2:
        return "binary"
    elif q == 4:
        return "quaternary"
    else:
        return f"{q}-ary"


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers."""
    code: str
    version_generated: int
    island_id: int
    expected_version: int = None

    def serialize(self):
        """Serializes the object to a JSON string."""
        return json.dumps({
            "code": self.code,
            "version_generated": self.version_generated,
            "island_id": self.island_id,
            "expected_version": self.expected_version,
        })

    @staticmethod
    def deserialize(serialized_str: str):
        """Deserializes the JSON string back to a Prompt object."""
        data = json.loads(serialized_str)
        return Prompt(**data)


class ProgramsDatabase:
    """A collection of programs, organized as islands.

    The ProgramsDatabase maintains a population of evolved programs across multiple islands
    for diversity. It implements evolutionary lineage tracking to record parent-child
    relationships between programs.

    Evolutionary Lineage Tracking:
    -----------------------------
    Each program is assigned:
    - program_id: Unique identifier (auto-incrementing)
    - parent_ids: List of program IDs from the few-shot prompt that generated it
    - generation: Evolutionary depth (0 for baseline, max(parent_generations) + 1 for offspring)
    - timestamp: Creation time

    Special Handling for Island Resets:
    -----------------------------------
    When weak islands are reset, founder programs (best programs from surviving islands)
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
        template: code_manipulation.Program,
        function_to_evolve: str,
        checkpoint_file: str = None,
        save_checkpoints_path: str=None,
        mode: str=None,
        eval_code=False,
        include_nx=True,
        start_n=[6],
        end_n=[11],
        s_values=[1],
        no_deduplication=False,
        prompt_limit=400_000,
        optimal_solution_programs=20_000,
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
        run_name=None
    ):
        self._islands = [] 
        self.connection = connection
        self.channel = channel
        self.database_queue = database_queue
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self._config = config
        self._template = template
        self.samples_per_batch = config.prompts_per_batch
        self._function_to_evolve = function_to_evolve
        self._best_score_per_island = [-float('inf')] * config.num_islands
        self._best_program_per_island = [None] * config.num_islands
        self._best_scores_per_test_per_island = [None] * config.num_islands
        self._last_reset_time = time.time()
        self.save_checkpoints_path = save_checkpoints_path  
        self.mode=mode
        self.eval_code = eval_code
        self.include_nx = include_nx
        self.start_n= start_n
        self.end_n = end_n
        self.s_values= s_values
        self.no_deduplication = no_deduplication
        self.prompt_limit = prompt_limit
        self.found_optimal_solution = False
        self.optimal_solution_programs = optimal_solution_programs
        self.prompts_since_optimal = 0
        self.target_signatures=target_signatures

        self.show_eval_scores = show_eval_scores
        self.display_mode = display_mode
        self.best_known_solutions = best_known_solutions or {}
        self.absolute_label = absolute_label
        self.relative_label = relative_label
        self.q = q

        if self.display_mode == "relative" and not self.best_known_solutions:
            logger.warning("display_mode='relative' requires best_known_solutions, falling back to 'absolute'")
            self.display_mode = "absolute"

        self.cumulative_evaluator_cpu_time = 0.0
        self.cumulative_sampler_gpu_time = 0.0

        self.cumulative_input_tokens  = 0
        self.cumulative_output_tokens = 0         

        self.dublicate_prompts=0
        self.total_prompts=0 # equals total processed messages as each message stored triggers a prompt
        self.total_stored_programs = 0
        self.version_mismatch_discarded = 0
        self.duplicates_discarded=0
        self.execution_failed = 0

        # Evolutionary lineage tracking (optional, can be disabled via config)
        self.save_lineage = config.save_lineage if hasattr(config, 'save_lineage') else False
        self.next_program_id = 1  # Counter for assigning unique program IDs
        self.lineage_log = [] if self.save_lineage else None  # Only initialize if enabled
        self._prompt_to_parents = {} if self.save_lineage else None

        # Lazy initialization of locks (will be created on first access)
        self._island_locks = None
        self._locks_initialized = False

        for _ in range(config.num_islands):
            island = {}
            island['clusters'] = {}
            island['version'] = 0
            island['num_programs'] = 0
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
            "functions_per_prompt": config.functions_per_prompt,
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
            "eval_code": eval_code,
            "include_nx": include_nx,
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
                "gpt": sampler_config.gpt,
                "prompts_per_batch_sampler": sampler_config.prompts_per_batch,
            })

        self._wandb_initialized = False

    def load_checkpoint_file(self, checkpoint_file: str):
        logger.info(f"Checkpoint file is {checkpoint_file}")
        if checkpoint_file is not None:
            self.load_checkpoint(checkpoint_file)
        else:
            return

    def load_checkpoint(self, checkpoint_file: str) -> None:
        """
        Loads the state from a checkpoint file.
        """
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)


        self.cumulative_evaluator_cpu_time = checkpoint_data.get("cumulative_evaluator_cpu_time", 0.0)
        self.cumulative_sampler_gpu_time = checkpoint_data.get("cumulative_sampler_gpu_time", 0.0)

        self.cumulative_input_tokens  = checkpoint_data.get("cumulative_input_tokens",  0)
        self.cumulative_output_tokens = checkpoint_data.get("cumulative_output_tokens", 0)

        self.total_prompts=checkpoint_data.get("total_prompts", 0)
        self.dublicate_prompts = checkpoint_data.get("dublicate_prompts", 0)

        self.total_stored_programs = checkpoint_data.get("total_stored_programs",0)
        self.execution_failed= checkpoint_data.get("execution_failed",0)
        self.version_mismatch_discarded = checkpoint_data.get("version_mismatch_discarded", 0) 
        self.duplicates_discarded=checkpoint_data.get("duplicates_discarded", 0) 

        self.found_optimal_solution = checkpoint_data.get("found_optimal_solution", False)  # Restore flag
        logger.info(f"Optimal solution was found in prev checkpoint {self.found_optimal_solution}")
        self.prompts_since_optimal = checkpoint_data.get("prompts_since_optimal", 0)  # Restore flag
        logger.info(f"Prompts_since_optimal are {self.prompts_since_optimal}")

        # Load W&B run ID and name if they exist in checkpoint
        self.wandb_run_id = checkpoint_data.get("wandb_run_id", None)
        if self.wandb_run_id:
            logger.info(f"Will resume W&B run: {self.wandb_run_id}")

        # Load W&B run name from checkpoint (for checkpoint directory continuity)
        checkpoint_run_name = checkpoint_data.get("wandb_run_name", None)
        if checkpoint_run_name:
            self.wandb_run_name = checkpoint_run_name
            logger.info(f"Restored run name from checkpoint: {checkpoint_run_name}")

        for i, score in enumerate(checkpoint_data["best_score_per_island"]):
            self._best_score_per_island[i] = score

        self._best_program_per_island = [
            code_manipulation.Function.from_dict(program) if program else None
            for program in checkpoint_data["best_program_per_island"]
        ]

        self._best_scores_per_test_per_island = checkpoint_data["best_scores_per_test_per_island"]
        self._last_reset_time = checkpoint_data["last_reset_time"]

        # Restore islands
        for island_id, island_state in enumerate(checkpoint_data["islands_state"]):
            logger.debug(f"Loading state for island id {island_id}")
            island = self._islands[island_id]
            self._load_island_state(island, island_state)
        logger.info("Checkpoint loaded successfully.")

    def _load_island_state(self, island, island_state):
        """
        Loads the state of a single island.
        """
        island['clusters'].clear()  # clear current clusters in the island if any
        for signature_str, cluster_state in island_state["clusters"].items():
            signature = eval(signature_str)
            if isinstance(signature, list):
                signature = tuple(signature)
            cluster_data = {}
            cluster_data['score'] = cluster_state['score']
            cluster_data['scores_per_test'] = cluster_state.get('scores_per_test', {})
            cluster_data['programs'] = [
                code_manipulation.Function.from_dict(prog_dict)
                for prog_dict in cluster_state['programs']
            ]
            island['clusters'][signature] = cluster_data

        island['version'] = island_state['version']
        island['num_programs'] = island_state['num_programs']


    def serialize_checkpoint(self) -> dict:
        """
        Serializes the necessary state of the database for checkpointing.
        """
        checkpoint_data = {
            "cumulative_evaluator_cpu_time": self.cumulative_evaluator_cpu_time,
            "cumulative_sampler_gpu_time": self.cumulative_sampler_gpu_time,
            "cumulative_input_tokens":  self.cumulative_input_tokens,
            "cumulative_output_tokens": self.cumulative_output_tokens,
            "best_score_per_island": list(self._best_score_per_island),
            "best_program_per_island": [program.to_dict() if program else None for program in self._best_program_per_island],
            "best_scores_per_test_per_island": list(self._best_scores_per_test_per_island),
            "last_reset_time": self._last_reset_time,
            "total_prompts": self.total_prompts,
            "dublicate_prompts": self.dublicate_prompts,
            "perc_duplicate_prompts": (self.dublicate_prompts / self.total_prompts if self.total_prompts else 0),
            "total_stored_programs": self.total_stored_programs,
            "execution_failed": self.execution_failed,
            "version_mismatch_discarded": self.version_mismatch_discarded,
            "duplicates_discarded": self.duplicates_discarded,
            "found_optimal_solution": self.found_optimal_solution,
            "prompts_since_optimal":self.prompts_since_optimal,
            "wandb_run_id": self.wandb_run_id,  # Save W&B run ID for resumption
            "wandb_run_name": self.wandb_run_name,  # Save run name for checkpoint directory continuity
            "islands_state": []
        }


        for island_id, island in enumerate(self._islands):
            island_state = self._serialize_island_state(island)
            checkpoint_data["islands_state"].append(island_state)

        return checkpoint_data

    def _serialize_island_state(self, island):
        """
        Serializes the state of a single island.
        """
        clusters_state = {}
        for signature, cluster_data in island['clusters'].items():
            clusters_state[str(signature)] = self._serialize_cluster_state(cluster_data)

        island_state = {
            "clusters": clusters_state,
            "version": island['version'],
            "num_programs": island['num_programs']
        }

        return island_state

    def _serialize_cluster_state(self, cluster_data):
        """
        Serializes the state of a single cluster.
        """
        programs_serialized = [program.to_dict() for program in cluster_data['programs']]
        cluster_state = {
            "score": cluster_data['score'],
            "programs": programs_serialized,
            "scores_per_test": cluster_data.get('scores_per_test', {}),
        }
        return cluster_state


    async def periodic_checkpoint(self):
        checkpoint_interval = 3600 
        while True:
            await asyncio.sleep(checkpoint_interval) 
            try: 
                current_pid = os.getpid()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                checkpoint_dir = os.path.join(os.getcwd(), self.save_checkpoints_path)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                filepath = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.pkl")
                data = self.serialize_checkpoint()
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
                logger.info("Checkpoint has been saved.")
            except Exception as e:
                logger.error(f"Error in saving checkpoint file {e}")


    def _compute_wandb_metrics(self) -> dict:
        """Compute metrics for Weights & Biases logging."""
        metrics = {}

        # 1. Best score per island (overall and detailed per test input)
        for island_id, score in enumerate(self._best_score_per_island):
            metrics[f"island_{island_id}/best_score"] = score

            # Log detailed scores for each evaluation input (n, s)
            scores_per_test = self._best_scores_per_test_per_island[island_id]
            if scores_per_test is not None:
                for test_key, test_score in scores_per_test.items():
                    # test_key is (n, s) tuple
                    if isinstance(test_key, tuple) and len(test_key) >= 2:
                        n, s = test_key[0], test_key[1]
                        metrics[f"island_{island_id}/score_n{n}_s{s}"] = test_score
                    else:
                        metrics[f"island_{island_id}/score_{test_key}"] = test_score

        # 2. Overall best score across all islands
        metrics["overall/best_score"] = max(self._best_score_per_island)

        # Log overall best detailed scores
        best_island_id = np.argmax(self._best_score_per_island)
        best_scores_per_test = self._best_scores_per_test_per_island[best_island_id]
        if best_scores_per_test is not None:
            for test_key, test_score in best_scores_per_test.items():
                if isinstance(test_key, tuple) and len(test_key) >= 2:
                    n, s = test_key[0], test_key[1]
                    metrics[f"overall/score_n{n}_s{s}"] = test_score
                else:
                    metrics[f"overall/score_{test_key}"] = test_score

        # 3. CPU and GPU times
        metrics["resources/cumulative_cpu_time"] = self.cumulative_evaluator_cpu_time
        metrics["resources/cumulative_gpu_time"] = self.cumulative_sampler_gpu_time

        # 4. Token counts
        metrics["tokens/cumulative_input"] = self.cumulative_input_tokens
        metrics["tokens/cumulative_output"] = self.cumulative_output_tokens

        # 5. Number of clusters per island and cluster sizes
        cluster_sizes_all = []
        for island_id, island in enumerate(self._islands):
            num_clusters = len(island['clusters'])
            metrics[f"island_{island_id}/num_clusters"] = num_clusters
            metrics[f"island_{island_id}/num_programs"] = island['num_programs']

            # Get cluster sizes for this island
            cluster_sizes = [len(cluster_data['programs'])
                            for cluster_data in island['clusters'].values()]

            if cluster_sizes:
                metrics[f"island_{island_id}/avg_cluster_size"] = np.mean(cluster_sizes)
                metrics[f"island_{island_id}/max_cluster_size"] = np.max(cluster_sizes)
                metrics[f"island_{island_id}/min_cluster_size"] = np.min(cluster_sizes)
                cluster_sizes_all.extend(cluster_sizes)

        # 6. Overall cluster statistics
        if cluster_sizes_all:
            metrics["clusters/overall_avg_size"] = np.mean(cluster_sizes_all)
            metrics["clusters/overall_max_size"] = np.max(cluster_sizes_all)
            metrics["clusters/overall_min_size"] = np.min(cluster_sizes_all)
            metrics["clusters/total_count"] = sum(len(island['clusters']) for island in self._islands)

        # 7. Program statistics
        metrics["programs/total_stored"] = self.total_stored_programs
        metrics["programs/execution_failed"] = self.execution_failed
        metrics["programs/version_mismatch_discarded"] = self.version_mismatch_discarded
        metrics["programs/duplicates_discarded"] = self.duplicates_discarded

        # 8. Prompt statistics
        metrics["prompts/total"] = self.total_prompts
        metrics["prompts/duplicate"] = self.dublicate_prompts

        # 9. Optimal solution tracking
        if self.found_optimal_solution:
            metrics["solution/found_optimal"] = 1
            metrics["solution/prompts_since_optimal"] = self.prompts_since_optimal
        else:
            metrics["solution/found_optimal"] = 0

        # 10. Evolutionary lineage tracking (only if enabled)
        if self.save_lineage and self.lineage_log:
            # Basic statistics
            generations = [entry['generation'] for entry in self.lineage_log]
            metrics["lineage/max_generation"] = max(generations)
            metrics["lineage/avg_generation"] = np.mean(generations)
            metrics["lineage/total_programs_tracked"] = len(self.lineage_log)

            # Recent lineage activity (last 100 programs)
            recent_lineage = self.lineage_log[-100:]
            recent_generations = [entry['generation'] for entry in recent_lineage]
            metrics["lineage/recent_avg_generation"] = np.mean(recent_generations)
            metrics["lineage/recent_max_generation"] = max(recent_generations)

            # Count programs by generation
            generation_counts = {}
            for gen in generations:
                generation_counts[gen] = generation_counts.get(gen, 0) + 1

            # Log generation distribution (up to generation 10)
            for gen in range(min(11, max(generations) + 1)):
                metrics[f"lineage/generation_{gen}_count"] = generation_counts.get(gen, 0)

            # Parent count statistics
            parent_counts = [len(entry['parent_ids']) for entry in self.lineage_log]
            metrics["lineage/avg_parents_per_program"] = np.mean(parent_counts)
            metrics["lineage/programs_with_no_parents"] = sum(1 for count in parent_counts if count == 0)

        return metrics

    def _get_program_by_id(self, program_id: int):
        """Find a program by its ID across all islands."""
        for island in self._islands:
            for cluster in island['clusters'].values():
                for program in cluster.get('programs', []):
                    if program.program_id == program_id:
                        return program
        return None

    def _trace_lineage(self, program_id: int, max_depth: int = 100):
        """Trace the full evolutionary lineage of a program.

        Returns a list of dictionaries, each containing:
        - program: The Function object
        - generation: Generation number
        - score: Program's score
        - scores_per_test: Detailed scores
        - parent_ids: List of parent IDs
        """
        lineage = []
        visited = set()
        current_ids = [program_id]
        depth = 0

        while current_ids and depth < max_depth:
            next_ids = []
            for pid in current_ids:
                if pid in visited:
                    continue
                visited.add(pid)

                # Find the program
                program = self._get_program_by_id(pid)
                if program is None:
                    # Try to find in lineage_log
                    for entry in self.lineage_log:
                        if entry['program_id'] == pid:
                            lineage.append({
                                'program_id': pid,
                                'program': None,  # Program no longer in memory
                                'generation': entry['generation'],
                                'score': entry['score'],
                                'parent_ids': entry['parent_ids'],
                                'timestamp': entry.get('timestamp'),
                            })
                            next_ids.extend(entry['parent_ids'])
                            break
                    continue

                # Find the program's scores from lineage_log
                program_entry = None
                for entry in self.lineage_log:
                    if entry['program_id'] == pid:
                        program_entry = entry
                        break

                if program_entry:
                    lineage.append({
                        'program_id': pid,
                        'program': program,
                        'generation': program.generation,
                        'score': program_entry['score'],
                        'parent_ids': program.parent_ids or [],
                        'timestamp': program.timestamp,
                    })
                    next_ids.extend(program.parent_ids or [])

            current_ids = next_ids
            depth += 1

        return lineage

    def _generate_lineage_html(self, program_id: int, island_id: int):
        """Generate an HTML visualization of a program's evolutionary lineage."""
        lineage = self._trace_lineage(program_id)

        if not lineage:
            return "<html><body><h1>No lineage found</h1></body></html>"

        # Sort lineage by generation (newest to oldest)
        lineage.sort(key=lambda x: x['generation'], reverse=True)

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Evolutionary Lineage - Program {program_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; }}
        .program-card {{
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .program-card:hover {{ border-color: #4CAF50; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .generation {{ background: #4CAF50; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
        .score {{ background: #2196F3; color: white; padding: 5px 10px; border-radius: 4px; }}
        .parent-ids {{ color: #666; font-size: 14px; margin: 5px 0; }}
        .code-block {{
            background: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        .current {{ border-color: #FF9800; border-width: 3px; }}
        .arrow {{ text-align: center; color: #999; font-size: 24px; margin: 10px 0; }}
        .metadata {{ color: #666; font-size: 12px; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>Evolutionary Lineage - Island {island_id}, Program {program_id}</h1>
    <p>Showing {num_ancestors} programs in the evolutionary chain (newest to oldest)</p>
""".format(program_id=program_id, island_id=island_id, num_ancestors=len(lineage))

        for i, entry in enumerate(lineage):
            is_current = (i == 0)
            parent_str = ", ".join(str(p) for p in entry['parent_ids']) if entry['parent_ids'] else "None (baseline)"

            code = ""
            if entry['program'] is not None:
                code = str(entry['program'])
            else:
                code = "[Program no longer in memory]"

            html += """
    <div class="program-card{current}">
        <div class="header">
            <h3>Program ID: {pid}</h3>
            <div>
                <span class="generation">Gen {gen}</span>
                <span class="score">Score: {score:.2f}</span>
            </div>
        </div>
        <div class="parent-ids"><strong>Parents:</strong> {parents}</div>
        <div class="code-block">{code}</div>
        <div class="metadata">Timestamp: {timestamp}</div>
    </div>
""".format(
                current=" current" if is_current else "",
                pid=entry['program_id'],
                gen=entry['generation'],
                score=entry['score'],
                parents=parent_str,
                code=code.replace('<', '&lt;').replace('>', '&gt;'),
                timestamp=entry.get('timestamp', 'N/A')
            )

            if i < len(lineage) - 1:
                html += '    <div class="arrow">↓ evolved from ↓</div>\n'

        html += """
</body>
</html>
"""
        return html

    def _generate_lineage_tree_diagram(self, program_id: int, island_id: int):
        """Generate a simple tree diagram showing the genealogy structure."""
        lineage = self._trace_lineage(program_id)

        if not lineage:
            return "<html><body><h1>No lineage found</h1></body></html>"

        # Build parent-child relationships
        nodes = {}
        for entry in lineage:
            nodes[entry['program_id']] = entry

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Lineage Tree - Program {program_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; }}
        .tree {{ margin: 20px; }}
        .node {{
            background: white;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 10px;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .node:hover {{ background: #e8f5e9; }}
        .current {{ border-color: #FF9800; border-width: 3px; background: #fff3e0; }}
        .baseline {{ border-color: #9C27B0; background: #f3e5f5; }}
        .node-id {{ font-weight: bold; color: #333; }}
        .node-gen {{ color: #666; font-size: 12px; }}
        .node-score {{ color: #2196F3; font-weight: bold; }}
        .arrow {{ color: #999; margin: 0 10px; }}
        .generation-group {{ margin: 20px 0; padding: 15px; background: white; border-radius: 8px; }}
        .gen-label {{ font-weight: bold; color: #4CAF50; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>Lineage Tree - Island {island_id}, Program {program_id}</h1>
    <p>Genealogy structure showing {num_programs} programs across {num_generations} generations</p>
""".format(
            program_id=program_id,
            island_id=island_id,
            num_programs=len(lineage),
            num_generations=max((e['generation'] for e in lineage), default=0) + 1
        )

        # Group by generation
        by_generation = {}
        for entry in lineage:
            gen = entry['generation']
            if gen not in by_generation:
                by_generation[gen] = []
            by_generation[gen].append(entry)

        # Display from newest to oldest generation
        for gen in sorted(by_generation.keys(), reverse=True):
            html += f'    <div class="generation-group">\n'
            html += f'        <div class="gen-label">Generation {gen}</div>\n'
            html += '        <div style="display: flex; flex-wrap: wrap; align-items: center;">\n'

            for entry in by_generation[gen]:
                is_current = (entry['program_id'] == program_id)
                is_baseline = (entry['generation'] == 0)
                node_class = "current" if is_current else ("baseline" if is_baseline else "")

                parent_str = ", ".join(str(p) for p in entry['parent_ids']) if entry['parent_ids'] else "None"

                html += f"""            <div class="node {node_class}">
                <div class="node-id">ID: {entry['program_id']}</div>
                <div class="node-gen">Gen: {entry['generation']}</div>
                <div class="node-score">Score: {entry['score']:.2f}</div>
                <div style="font-size: 11px; color: #888;">Parents: {parent_str}</div>
            </div>\n"""

            html += '        </div>\n'
            html += '    </div>\n'

        html += """
    <div style="margin-top: 30px; padding: 15px; background: white; border-radius: 8px;">
        <h3>Legend</h3>
        <p><span class="node current" style="display: inline-block; margin: 5px;">Current Program</span> - The program you're viewing</p>
        <p><span class="node baseline" style="display: inline-block; margin: 5px;">Baseline</span> - Initial programs (Generation 0)</p>
        <p><span class="node" style="display: inline-block; margin: 5px;">Ancestor</span> - Programs in the evolutionary chain</p>
    </div>
</body>
</html>
"""
        return html

    def _log_top_programs_table(self):
        """Log a W&B table with the best program from each island and their lineage."""
        if not self.wandb_enabled or not WANDB_AVAILABLE:
            return

        table_data = []

        for island_id in range(len(self._islands)):
            program = self._best_program_per_island[island_id]
            if program is None or program.program_id is None:
                continue

            score = self._best_score_per_island[island_id]
            scores_per_test = self._best_scores_per_test_per_island[island_id]

            # Get lineage (only if enabled)
            lineage = self._trace_lineage(program.program_id) if self.save_lineage else []

            # Format detailed scores
            scores_str = ", ".join(f"{k}:{v}" for k, v in scores_per_test.items()) if scores_per_test else "N/A"

            # Full code (not truncated)
            full_code = str(program)

            # Generate and save HTML lineage (only if enabled)
            if self.save_lineage:
                try:
                    import os
                    os.makedirs(self.save_checkpoints_path, exist_ok=True)

                    # Generate detailed lineage HTML with code
                    html_content = self._generate_lineage_html(program.program_id, island_id)
                    html_filename = f"lineage_detailed_island{island_id}_program{program.program_id}.html"
                    html_path = f"{self.save_checkpoints_path}/{html_filename}"
                    with open(html_path, 'w') as f:
                        f.write(html_content)

                    # Generate tree diagram HTML (structure only, no code)
                    tree_content = self._generate_lineage_tree_diagram(program.program_id, island_id)
                    tree_filename = f"lineage_tree_island{island_id}_program{program.program_id}.html"
                    tree_path = f"{self.save_checkpoints_path}/{tree_filename}"
                    with open(tree_path, 'w') as f:
                        f.write(tree_content)

                    # Log both as W&B artifacts (only if W&B is enabled and initialized)
                    if self.wandb_enabled and self._wandb_initialized:
                        try:
                            artifact = wandb.Artifact(
                                name=f"lineage_island{island_id}_step{self.total_prompts}",
                                type="lineage_visualization",
                                description=f"Evolutionary lineage for island {island_id}, program {program.program_id}"
                            )
                            artifact.add_file(html_path, name="detailed_with_code.html")
                            artifact.add_file(tree_path, name="tree_diagram.html")
                            wandb.log_artifact(artifact)
                            lineage_link = f"See artifact: lineage_island{island_id}_step{self.total_prompts}"
                        except Exception as e:
                            logger.warning(f"Failed to upload lineage artifact to W&B: {e}")
                            lineage_link = f"Local files: {html_filename}, {tree_filename}"
                    else:
                        lineage_link = f"Local files: {html_filename}, {tree_filename}"
                except Exception as e:
                    logger.error(f"Error generating lineage HTML: {e}")
                    lineage_link = "Error generating lineage"
            else:
                lineage_link = "Disabled"

            table_data.append([
                island_id,
                program.program_id,
                program.generation,
                score,
                scores_str,
                full_code,
                len(lineage),
                lineage_link
            ])

        if table_data:
            table = wandb.Table(
                columns=["Island", "Program ID", "Generation", "Score", "Detailed Scores", "Full Code", "Lineage Depth", "Lineage Visualization"],
                data=table_data
            )
            wandb.log({"top_programs": table})
            logger.info(f"Logged top programs table with {len(table_data)} entries")


    async def _initialize_wandb(self):
        """Initialize W&B asynchronously (called once on first logging attempt)."""
        if self._wandb_initialized:
            return

        if not self.wandb_config or not self.wandb_config.enabled:
            return

        if not WANDB_AVAILABLE:
            logger.warning("W&B logging enabled but wandb not installed. Run: pip install wandb")
            return

        try:
            # Run wandb.init in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()

            # Check if we're resuming from a checkpoint with an existing run ID
            if self.wandb_run_id:
                # Try to resume existing run with strict mode
                expected_run_id = self.wandb_run_id
                logger.info(f"Attempting to resume W&B run with ID: {expected_run_id}")
                logger.info(f"W&B project: {self.wandb_config.project}, entity: {self.wandb_config.entity}")

                resume_failed = False
                resume_error = None

                try:
                    await loop.run_in_executor(
                        None,
                        lambda: wandb.init(
                            project=self.wandb_config.project,
                            entity=self.wandb_config.entity,
                            id=expected_run_id,
                            resume="must",  # Fail if run doesn't exist or can't be resumed
                            tags=self.wandb_config.tags,
                            config=self.wandb_init_config,
                            settings=wandb.Settings(
                                console='off',  # Don't capture console output
                                _disable_stats=False,
                                _disable_meta=False,
                            )
                        )
                    )

                    # Verify that we actually resumed the expected run
                    if wandb.run and wandb.run.id == expected_run_id:
                        logger.info(f"Successfully resumed W&B run: {expected_run_id}")
                        logger.info(f"W&B run URL: {wandb.run.url}")
                    else:
                        # This shouldn't happen with resume="must", but check anyway
                        logger.warning(f"Unexpected: W&B run ID mismatch. Expected {expected_run_id}, got {wandb.run.id if wandb.run else 'None'}")

                except Exception as e:
                    resume_failed = True
                    resume_error = e
                    logger.error(f"Failed to resume W&B run {expected_run_id}: {type(e).__name__}: {e}")

                    # Provide specific guidance based on error type
                    error_msg = str(e).lower()
                    if "not found" in error_msg or "does not exist" in error_msg:
                        logger.error(f"Reason: Run {expected_run_id} does not exist in project '{self.wandb_config.project}'")
                        logger.error("Possible causes: Run was deleted, wrong project/entity, or run ID is incorrect")
                    elif "finished" in error_msg or "completed" in error_msg:
                        logger.error(f"Reason: Run {expected_run_id} is already marked as finished/completed")
                        logger.error("Suggestion: Check W&B dashboard to verify run status")
                    elif "permission" in error_msg or "access" in error_msg:
                        logger.error(f"Reason: No permission to access run {expected_run_id}")
                        logger.error("Suggestion: Verify entity/project permissions and API key")
                    else:
                        logger.error(f"Reason: Unknown error - {e}")

                    # Close any partial W&B connection
                    if wandb.run:
                        wandb.finish()

                    logger.info("Creating a new W&B run instead...")

                # If resume failed, create a new run
                if resume_failed:
                    await loop.run_in_executor(
                        None,
                        lambda: wandb.init(
                            project=self.wandb_config.project,
                            entity=self.wandb_config.entity,
                            name=self.wandb_run_name,
                            tags=self.wandb_config.tags,
                            config=self.wandb_init_config,
                            settings=wandb.Settings(
                                console='off',
                                _disable_stats=False,
                                _disable_meta=False,
                            )
                        )
                    )
                    if wandb.run:
                        logger.info(f"Created new W&B run: {wandb.run.id}")
                        logger.info(f"New run URL: {wandb.run.url}")
                        logger.warning(f"Note: This is a NEW run, not a resumption of {expected_run_id}")

            else:
                # No checkpoint run ID - start fresh run
                logger.info("No previous W&B run ID found. Creating a new run...")
                await loop.run_in_executor(
                    None,
                    lambda: wandb.init(
                        project=self.wandb_config.project,
                        entity=self.wandb_config.entity,
                        name=self.wandb_run_name,
                        tags=self.wandb_config.tags,
                        config=self.wandb_init_config,
                        settings=wandb.Settings(
                            console='off',  # Don't capture console output
                            _disable_stats=False,
                            _disable_meta=False,
                        )
                    )
                )
                if wandb.run:
                    logger.info(f"Created new W&B run: {wandb.run.id}")
                    logger.info(f"New run URL: {wandb.run.url}")

            self.wandb_enabled = True
            self.wandb_log_interval = self.wandb_config.log_interval
            self._wandb_initialized = True

            # Store the run ID (either new or resumed)
            if wandb.run:
                self.wandb_run_id = wandb.run.id
                logger.info(f"W&B logging initialized: {wandb.run.url}")
            else:
                logger.warning("W&B run object is None after initialization")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.wandb_enabled = False

    async def periodic_wandb_logging(self):
        """Periodically log metrics to Weights & Biases."""
        # Initialize W&B asynchronously on first run
        await self._initialize_wandb()

        if not self.wandb_enabled:
            return

        try:
            while True:
                await asyncio.sleep(self.wandb_log_interval)
                try:
                    metrics = self._compute_wandb_metrics()
                    wandb.log(metrics)
                    logger.debug(f"Logged {len(metrics)} metrics to W&B")

                    # Log top programs table with lineage
                    self._log_top_programs_table()
                except Exception as e:
                    logger.error(f"Error logging to W&B: {e}")
        except asyncio.CancelledError:
            logger.info("W&B logging task cancelled. NOT finishing run to allow resumption from checkpoint.")
            # Do NOT call wandb.finish() here - leave the run "running" so it can be resumed
            # If the run is truly complete, the user should manually finish it in W&B UI
            # or call wandb.finish() explicitly when termination conditions are met
            if self.wandb_enabled and wandb.run is not None:
                logger.info(f"W&B run {wandb.run.id} left in resumable state. Resume with: --checkpoint <path>")
            raise

    def finish_wandb_run(self):
        """
        Explicitly finish the W&B run when the experiment is truly complete.
        Call this manually when you know the run should be marked as finished (not resumable).
        """
        if self.wandb_enabled and wandb.run is not None:
            try:
                logger.info(f"Finishing W&B run {wandb.run.id}...")
                wandb.finish()
                logger.info("W&B run finished successfully")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")

    async def consume_and_process(self) -> None:
        """ Continuously consumes messages in batches from the database queue and processes them. """
        from funsearchmq import process_utils

        batch_size = 10
        batch_timeout = 0.01

        logger.info(f"Consume_and_process started")

        async def _consume_loop():
            """Inner consume loop - will be wrapped with reconnection logic."""
            await self.channel.set_qos(prefetch_count=batch_size)

            async with self.database_queue.iterator() as stream:
                batch = []
                batch_start_time = time.time()

                try:
                    async for message in stream:
                        logger.debug(f"Received message: {message.body.decode()}")
                        batch.append(message)
                        current_time = time.time()

                        # Check if the batch should be processed
                        if len(batch) >= batch_size or (current_time - batch_start_time) >= batch_timeout:
                            await self.process_batch(batch)
                            batch.clear()
                            batch_start_time = current_time

                except asyncio.CancelledError:
                    logger.info("Database task was canceled. Processing any remaining batch.")
                    if batch:
                        await self.process_batch(batch)
                    raise  # Re-raise to ensure proper cancellation

        # Wrap consume loop with automatic reconnection
        await process_utils.with_reconnection(
            _consume_loop,
            logger,
            component_name="ProgramsDatabase"
        )


    #@async_time_execution
    async def process_batch(self, batch: List[aio_pika.IncomingMessage]):
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
            # If reset_period is None, only check population
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

            if not self.no_deduplication and self.function_body_exists(island['clusters'], hash_value):
                self.duplicates_discarded += 1
                logger.debug(f"Program with identical body already exists in island. Skipping registration.")
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
        - generation: max(parent_generations) + 1, or 0 if no parents (baseline)
        - timestamp: Current time

        The lineage information is logged to self.lineage_log for tracking evolutionary trajectories.
        """
        self.total_stored_programs += 1
        island = self._islands[island_id]
        clusters = island['clusters']
        signature = self._get_signature(scores_per_test)
        program.hash_value = hash_value

        # Assign lineage tracking information
        if parent_ids is None:
            parent_ids = []

        program.program_id = self.next_program_id
        self.next_program_id += 1
        program.parent_ids = parent_ids

        # Calculate generation: max of parent generations + 1, or 0 if no parents
        if parent_ids:
            # Find maximum generation among parents
            max_parent_generation = 0
            for cluster in clusters.values():
                for p in cluster.get('programs', []):
                    if p.program_id in parent_ids and p.generation is not None:
                        max_parent_generation = max(max_parent_generation, p.generation)
            program.generation = max_parent_generation + 1
        else:
            program.generation = 0

        program.timestamp = time.time()

        try:
            if signature not in clusters:
                logger.info(f"Creating new cluster with signature {scores_per_test}")
                cluster_data = {}
                cluster_data['score'] = _reduce_score(scores_per_test, self.mode, self.start_n, self.end_n, self.s_values, self.target_signatures)
                cluster_data['scores_per_test'] = scores_per_test
                cluster_data['programs'] = [program]
                clusters[signature] = cluster_data
            else:
                logger.info(f"Registering on cluster with signature {scores_per_test}")
                cluster_data = clusters[signature]
                cluster_data['programs'].append(program)
        
            island['num_programs'] += 1

            # Log lineage information for this program (only if enabled)
            if self.save_lineage:
                score = _reduce_score(scores_per_test, self.mode, self.start_n, self.end_n, self.s_values, self.target_signatures)
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
            # Calculate the score for the new program
            score = _reduce_score(scores_per_test, self.mode, self.start_n, self.end_n, self.s_values, self.target_signatures)
        
            # Check if the new score is higher than the current best score
            if score > self._best_score_per_island[island_id]:
                self._best_program_per_island[island_id] = program
                self._best_scores_per_test_per_island[island_id] = scores_per_test
                self._best_score_per_island[island_id] = score
                logger.info(f'Best score of island {island_id} increased to {score} with program {program} and scores {scores_per_test}')
        
            # If the score is equal to the best score, check the program signature
            elif score == self._best_score_per_island[island_id]:
                # Get the current best program's signature
                current_best_program = self._best_program_per_island[island_id]
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
        The weakest islands (by best score) are cleared, and each is seeded with the best
        program from a randomly selected surviving island.

        Lineage Tracking During Resets:
        --------------------------------
        Founder programs maintain evolutionary continuity across island boundaries.
        When a program is copied as a founder to a reset island, it receives a new program_id
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
            logger.info(f"Reached the limit of {self.prompt_limit} prompts. Stopping further publishing, but continue processing remaining queue messages.")
            return  # Stop further publishing once the limit is reached

        logger.debug(f"len(self._islands) {len(self._islands)}")
        island_id = np.random.randint(len(self._islands))
        logger.debug(f"Island id is {island_id}")
        island = self._islands[island_id]

        code, flag_duplicate, version_generated, parent_ids = self._generate_prompt_for_island(island)
        expected_version = island['version']

        prompt = Prompt(code, version_generated, island_id, expected_version)
        message_data = {
            "prompt": prompt.serialize(),
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


    def _generate_prompt_for_island(self, island, multiple=False) -> tuple[Optional[str], int, int, list[int]]:
        """Generate a prompt for an island.

        Returns:
            tuple: (prompt, flag_duplicate, version_generated, parent_ids)
        """
        clusters = island['clusters']
        signatures = list(clusters.keys())
        functions_per_prompt = self._config.functions_per_prompt
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

        # If only one valid signature is available, sample from it once.
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
        if len(valid_signatures) >= functions_per_prompt:
            logger.debug("Sampling from multiple valid clusters.")
            # Sample exactly functions_per_prompt clusters without replacement.
            cluster_indices = np.random.choice(
                len(valid_signatures),
                size=functions_per_prompt,
                p=valid_probabilities,
                replace=False
            )
            sampled_signatures.update([valid_signatures[i] for i in cluster_indices])
        else:
            # If fewer than desired valid clusters are available, use all available ones.
            logger.warning("Fewer valid clusters than functions_per_prompt; using all available clusters.")
            sampled_signatures.update(valid_signatures)
            # Optionally, you could recalculate probabilities excluding these and sample additional ones if desired.
    
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

    def _generate_prompt(self, implementations_with_scores: Sequence[tuple]) -> str:
        logger.debug(f"Type of `implementations_with_scores`: {type(implementations_with_scores)}")

        implementations = [impl for impl, _ in implementations_with_scores]
        scores_list = [scores for _, scores in implementations_with_scores]

        for i, implementation in enumerate(implementations):
            logger.debug(f"Implementation {i}: Type: {type(implementation)}, Attributes: {dir(implementation)}")
            logger.debug(f"Implementation {i}: Content: {implementation}")

        implementations = copy.deepcopy(implementations)

        versioned_functions = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name

            # Add scores to the docstring of evaluated implementations
            if self.show_eval_scores and scores_list[i]:
                score_text = _format_scores_for_prompt(
                    scores_list[i],
                    self.display_mode,
                    self.best_known_solutions,
                    self.absolute_label,
                    self.relative_label
                )

                if i >= 1:
                    # For i >= 1, use "Improved version" docstring
                    base_docstring = f'Improved version of `{self._function_to_evolve}_v{i - 1}`.'
                    implementation.docstring = f'{base_docstring} {score_text}'
                else:
                    # For i == 0, append scores to existing docstring
                    if implementation.docstring:
                        implementation.docstring = f'{implementation.docstring.strip()} {score_text}'
                    else:
                        implementation.docstring = score_text
            elif i >= 1:
                # No scores, but still update docstring for i >= 1
                implementation.docstring = f'Improved version of `{self._function_to_evolve}_v{i - 1}`.'
            try:
                implementation_str = code_manipulation.rename_function_calls(
                    str(implementation), self._function_to_evolve, new_function_name
                )
                versioned_functions.append(code_manipulation.text_to_function(implementation_str))
            except Exception as e:
                logger.error(f"Error in converting text to function: {e}")

        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'

        try:
            # Create docstring for the template - just the basic "Improved version" text
            template_docstring = f'Improved version of `{self._function_to_evolve}_v{next_version - 1}`.'

            header = dataclasses.replace(
                implementations[-1],
                name=new_function_name,
                body='',
                docstring=template_docstring
            )
            versioned_functions.append(header)
        except Exception as e:
            logger.error(f"Error in creating header: {e}")

        if hasattr(self._template, 'preface'):
            preface = getattr(self._template, 'preface', '')

            # Remove all existing imports
            import_pattern = r"(?m)^import .*|from .* import .*"
            preface_cleaned = re.sub(import_pattern, "", preface).strip()

            # Replace generic "q-ary" with actual alphabet size description
            q_description = _get_q_description(self.q)
            preface_cleaned = preface_cleaned.replace("q-ary", q_description)

            # Define required imports
            imports = ["import numpy as np"]
            if self.include_nx:
                imports.append("import networkx as nx")

            # If the preface starts with a docstring, leave it intact
            if preface_cleaned.startswith('"""'):
                docstring_end = preface_cleaned.index('"""', 3) + 3
                initial_docstring = preface_cleaned[:docstring_end]
                remaining_preface = preface_cleaned[docstring_end:].strip()
            else:
                initial_docstring = ""
                remaining_preface = preface_cleaned

            # Construct the new preface with specified newline rules
            sections = []
            if initial_docstring:
                sections.append(initial_docstring.strip())
            if remaining_preface:
                sections.append(remaining_preface.strip())
            sections.extend(imports)
            sections.append("")  # Add a blank line after imports

            # Join sections, ensuring appropriate newlines
            preface = "\n".join(filter(None, sections))+ "\n" + "\n"
            self._template = dataclasses.replace(self._template, preface=preface)

        try:
            if self.eval_code:
                # hashing logic in eval script is excluded for constructing prompt
                spec_path = '/Funsearch/implementation/specifications_construct/without_hash.txt'
                with open(spec_path, 'r') as file:
                    specification = file.read()
                template_no_hash= code_manipulation.text_to_program(specification)
                # Use the first two functions from the template, followed by versioned functions
                first_two_functions = template_no_hash.functions[:4]
                new_functions_list = first_two_functions + versioned_functions
            else:
                # Use only versioned functions
                new_functions_list = versioned_functions

            prompt = dataclasses.replace(self._template, functions=new_functions_list)

            prompt_str = str(prompt)

            logger.debug(f"Final prompt after class removal: {prompt_str}")

            # Write to a file if two programs have the same hash value
            duplicate_prompt = False
            if len(implementations) == 2 and implementations[0].hash_value == implementations[1].hash_value:
                duplicate_prompt = True
                self.dublicate_prompts += 1
                try:
                    with open("duplicate_prompt.txt", "a") as f:
                        f.write(prompt_str)
                    logger.info("Duplicate prompt written to 'duplicate_prompt.txt'.")
                except Exception as e:
                    logger.error(f"Failed to write duplicate prompt to file: {e}")

            return prompt_str.rstrip('\n'), duplicate_prompt
        except Exception as e:
            logger.error(f"Error in replacing prompt: {e}")
            return None, False


    def function_body_exists(self, clusters, hash_value: int) -> bool:
        assert hash_value is not None, "Error: No hash value computed! Check that hash value condition in the specification script is set to match start_n."

        for cluster in clusters.values():
            programs = cluster['programs']
            for program in programs:
                if program.hash_value == hash_value:
                    return True
        return False

    def _get_signature(self, scores_per_test):
        if all(isinstance(k, str) for k in scores_per_test.keys()):
            scores_per_test = {eval(k): v for k, v in scores_per_test.items()}

        def ensure_hashable(val):
            if isinstance(val, list):
                return tuple(val)
            return val

        return tuple(ensure_hashable(scores_per_test[k]) for k in sorted(scores_per_test.keys()))

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
            probabilities = self._softmax(-normalized_lengths, temperature=temperature)  # Softmax over negative lengths
        # Sample a program based on the probabilities
        sampled_index = np.random.choice(len(programs), p=probabilities)
        return programs[sampled_index]


    def _softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Tempered softmax for sampling."""
        logits = np.array(logits, dtype=np.float32)
        exp_logits = np.exp(logits / temperature)
        return exp_logits / exp_logits.sum()