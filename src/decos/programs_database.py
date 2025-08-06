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
* Saves / reloads *hourly pickle checkpoints (resume after crash or restart).
* Enforces deduplication (hash-based) and version-mismatch checks.
* Stops early after an optimal solution or a prompt/solution quota.
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
from decos import code_manipulation
import json
import aio_pika
import re
from logging.handlers import RotatingFileHandler
import psutil
from logging import FileHandler
import datetime
from decos.profiling import async_time_execution


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

def _reduce_score(scores_per_test: dict, mode: str = "last", start_n: list = [6], end_n: list = [11], s_values: list = [1], TARGET_SIGNATURES=None) -> float:
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
        TARGET_SIGNATURES (dict, optional): Dictionary of target sizes for each (n, s).

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

    if mode == "relative_difference" and TARGET_SIGNATURES is None:
        raise ValueError("TARGET_SIGNATURES must be provided for 'relative_difference' mode.")

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
                target = TARGET_SIGNATURES.get(dim, None)
                if target is not None:
                    relative_scores.append((actual - target) / target)
            per_s_scores.append(sum(relative_scores) / len(relative_scores) if relative_scores else 0)

        else:
            raise ValueError("Invalid mode. Available modes are 'last', 'average', 'weighted', and 'relative_difference'.")

    return sum(per_s_scores) / len(per_s_scores) if per_s_scores else 0


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
    """A collection of programs, organized as islands."""

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
        TARGET_SIGNATURES=None
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
        self.TARGET_SIGNATURES=TARGET_SIGNATURES


        self.cumulative_evaluator_cpu_time = 0.0  # Track total CPU time from evaluators
        self.cumulative_sampler_gpu_time = 0.0  # Track total GPU time

        self.cumulative_input_tokens  = 0         
        self.cumulative_output_tokens = 0         

        self.dublicate_prompts=0
        self.total_prompts=0 # equals total processed messages as each message stored triggers a prompt 
        self.total_stored_programs = 0
        self.version_mismatch_discarded = 0
        self.duplicates_discarded=0
        self.execution_failed = 0
        for _ in range(config.num_islands):
            island = {}
            island['clusters'] = {}
            island['version'] = 0
            island['num_programs'] = 0
            self._islands.append(island)

        # Load checkpoint if provided
        self.load_checkpoint_file(checkpoint_file)

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


    async def consume_and_process(self) -> None:
        """ Continuously consumes messages in batches from the database queue and processes them. """
        batch_size = 10
        batch = []
        batch_timeout = 0.01

        logger.info(f"Consume_and_process started")

        try:
            await self.channel.set_qos(prefetch_count=batch_size)
            async with self.database_queue.iterator() as stream:
                batch_start_time = time.time()
                while True:
                    try:
                        async for message in stream:
                            logger.debug(f"Received message: {message.body.decode()}")
                            batch.append(message)
                            current_time = time.time()

                            # Check if the batch should be processed
                            if len(batch) >= batch_size or (current_time - batch_start_time) >= batch_timeout:
                                await self.process_batch(batch)
                                batch.clear()
                                batch_start_time = current_time  # Reset timer after batch processing

                    except asyncio.CancelledError:
                        logger.info("Database task was canceled. Processing any remaining batch.")
                        if batch:
                            await self.process_batch(batch)
                        raise  # Re-raise the CancelledError to ensure proper cancellation

                    except Exception as e:
                        logger.error(f"Error during message consumption: {e}")

        except asyncio.CancelledError:
            logger.info("Database consume_and_process was canceled.")

        except Exception as e:
            logger.error(f"Error initializing the database consume_and_process: {e}")


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


                if island_id is None:
                    # Register the program to all islands
                    for i in range(len(self._islands)):
                        await self.register_program(program, i, data["scores_per_test"], data.get("expected_version", None), data.get("hash_value", None))
                else:
                    # Register the program to the specific island
                    await self.register_program(program, island_id, data["scores_per_test"], data.get("expected_version", None), data.get("hash_value", None))

                await self.get_prompt()

        except asyncio.CancelledError:
            logger.info("Process message was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Database: Error processing message: {e}")
            raise

    async def register_program(self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest, expected_version: int = None, hash_value: int = None):
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

        self._register_program_in_island(program, island_id, scores_per_test, hash_value)


    def _register_program_in_island(self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest, hash_value: int = None):
        self.total_stored_programs += 1
        island = self._islands[island_id]
        clusters = island['clusters']
        signature = self._get_signature(scores_per_test)
        program.hash_value = hash_value

        try: 
            if signature not in clusters:
                logger.info(f"Creating new cluster with signature {scores_per_test}")
                cluster_data = {}
                cluster_data['score'] = _reduce_score(scores_per_test, self.mode, self.start_n, self.end_n, self.s_values, self.TARGET_SIGNATURES)
                cluster_data['programs'] = [program]
                clusters[signature] = cluster_data
            else:
                logger.info(f"Registering on cluster with signature {scores_per_test}")
                cluster_data = clusters[signature]
                cluster_data['programs'].append(program)
        
            island['num_programs'] += 1
    
        except Exception as e: 
            logger.error(f"Could not append program: {e}")
    
        try: 
            # Calculate the score for the new program
            score = _reduce_score(scores_per_test, self.mode, self.start_n, self.end_n, self.s_values, self.TARGET_SIGNATURES)
        
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
                island = self._islands[island_id]
                island['clusters'].clear()
                island['version'] += 1
                island['num_programs'] = 0

                self._best_score_per_island[island_id] = -float('inf')
                founder_island_id = np.random.choice(keep_islands_ids)
                founder = self._best_program_per_island[founder_island_id]
                founder_scores = self._best_scores_per_test_per_island[founder_island_id]
                self._register_program_in_island(founder, island_id, founder_scores)
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

        code, flag_duplicate, version_generated = self._generate_prompt_for_island(island)
        expected_version = island['version']

        prompt = Prompt(code, version_generated, island_id, expected_version)
        message_data = {
            "prompt": prompt.serialize(),
            "total_registered_programs": island['num_programs'], 
            "flag":flag_duplicate
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


    def _generate_prompt_for_island(self, island, multiple=False) -> tuple[Optional[str], int, int]:
        clusters = island['clusters']
        signatures = list(clusters.keys())
        functions_per_prompt = self._config.functions_per_prompt
        if not signatures:
            logger.warning(f"No clusters found in island {island}. Skipping prompt generation.")
            return None, False, 0

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
        logger.debug(f"Length of valid sig: {len(valid_signatures)}")

        # If only one valid signature is available, sample from it once.
        if len(valid_signatures) == 1:
            selected_signature = valid_signatures[0]
            cluster_programs = clusters[selected_signature]['programs']
            logger.debug(f"Selected signature: {selected_signature} with programs {cluster_programs}")
            sampled_signatures.add(selected_signature)
            if len(cluster_programs) >= 1:
                # Only sample one program from this single cluster.
                sampled_programs.append(self.sample_program(clusters[selected_signature]))
                version_generated = 1
                prompt, flag = self._generate_prompt(sampled_programs)
                return prompt, flag, version_generated
            else:
                logger.warning("Single valid cluster has no programs. Skipping prompt generation.")
                return None, False, 0

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
            cluster_programs = clusters[signature]['programs']
            if not cluster_programs:
                logger.warning(f"Cluster {signature} has no programs. Skipping.")
                continue
            sampled_programs.append(self.sample_program(clusters[signature]))
    
        # Sort sampled programs by the corresponding cluster's score.
        sorted_programs = sorted(sampled_programs, key=lambda p: clusters[next(iter(sampled_signatures))]['score'])
        version_generated = len(sorted_programs)  # Number of programs sampled becomes the version.
        prompt, flag_duplicate = self._generate_prompt(sorted_programs)
        return prompt, flag_duplicate, version_generated

    def _generate_prompt(self, implementations: Sequence[code_manipulation.Function]) -> str:
        logger.debug(f"Type of `implementations`: {type(implementations)}")
        for i, implementation in enumerate(implementations):
            logger.debug(f"Implementation {i}: Type: {type(implementation)}, Attributes: {dir(implementation)}")
            logger.debug(f"Implementation {i}: Content: {implementation}")
        implementations = copy.deepcopy(implementations)

        versioned_functions = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            if i >= 1:
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
            header = dataclasses.replace(
                implementations[-1],
                name=new_function_name,
                body='',
                docstring=f'Improved version of `{self._function_to_evolve}_v{next_version - 1}`.'
            )
            versioned_functions.append(header)
        except Exception as e:
            logger.error(f"Error in creating header: {e}")

        if hasattr(self._template, 'preface'):
            preface = getattr(self._template, 'preface', '')

            # Remove all existing imports
            import_pattern = r"(?m)^import .*|from .* import .*"
            preface_cleaned = re.sub(import_pattern, "", preface).strip()

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