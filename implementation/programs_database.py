"""A programs database that implements the evolutionary algorithm."""

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
import code_manipulation
import json
import aio_pika
import re
from logging.handlers import RotatingFileHandler
import psutil
from logging import FileHandler
import datetime
from profiling import async_time_execution


logger = logging.getLogger('main_logger')

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits` representing the cluster scores."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Adjust the maximum probability to ensure the sum is exactly 1
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[:index]) - np.sum(result[index+1:])
    return result


def _reduce_score(scores_per_test: dict, mode: str = 'last') -> float:
    """
    Reduces per-test scores into a single score based on the specified mode.
    """
    n_dimensions = 6  # Number of expected dimensions
    all_dimensions = list(range(n_dimensions))

    if mode == 'last':
        return scores_per_test[list(scores_per_test.keys())[-1]]
    elif mode == 'average':
        # Ensure all dimensions are present, setting missing ones to 0
        complete_scores = {d: scores_per_test.get(d, 0) for d in all_dimensions}
        # Calculate the average score
        average_score = sum(complete_scores.values()) / len(complete_scores)
        return average_score
    else:
        raise ValueError("Invalid mode. Available modes are 'last' and 'average'.")


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
        save_checkpoints_path: str=None
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
        self.registered_programs = 0
        self.total_programs = 0
        self.execution_failed = 0
        self._best_score_per_island = [-float('inf')] * config.num_islands
        self._best_program_per_island = [None] * config.num_islands
        self._best_scores_per_test_per_island = [None] * config.num_islands
        self._last_reset_time = time.time()
        self.save_checkpoints_path = save_checkpoints_path 

        # Initialize islands as shared dictionaries
        for _ in range(config.num_islands):
            island = {}
            island['clusters'] = {}
            island['version'] = 0
            island['num_programs'] = 0
            self._islands.append(island)

        # Load checkpoint if provided
        self.load_checkpoint_file(checkpoint_file)

    def load_checkpoint_file(self, checkpoint_file: str):
        if checkpoint_file is not None:
            self.load_checkpoint(checkpoint_file)
        else:
            return

    def load_checkpoint(self, checkpoint_file: str) -> None:
        """
        Loads the state from a checkpoint file.
        """
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return

        self.registered_programs = checkpoint_data["registered_programs"]
        self.total_programs= checkpoint_data["total_programs"]
        self.execution_failed= checkpoint_data["execution_failed"]

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
            logger.debug(f"Loading state fpr island id {island_id}")
            island = self._islands[island_id]
            self._load_island_state(island, island_state)
        logger.info("Checkpoint loaded successfully.")
        

    def _load_island_state(self, island, island_state):
        """
        Loads the state of a single island.
        """
        island['clusters'].clear() # clear current clusters in the island if any
        for signature_str, cluster_state in island_state["clusters"].items():
            signature = eval(signature_str) # convert str version of signature back into its original tuple form, so that it can be used as a key in the island’s dictionary of clusters.
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
            "registered_programs": self.registered_programs,
            "total_programs": self.total_programs,
            "execution_failed": self.execution_failed,
            "best_score_per_island": list(self._best_score_per_island),
            "best_program_per_island": [program.to_dict() if program else None for program in self._best_program_per_island],
            "best_scores_per_test_per_island": list(self._best_scores_per_test_per_island),
            "last_reset_time": self._last_reset_time,
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
        checkpoint_interval = 3600  #3600  # 1 hour
        while True:
            await asyncio.sleep(checkpoint_interval)  # Non-blocking sleep
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
        batch_timeout = 0.05

        try:
            async with self.channel:
                await self.channel.set_qos(prefetch_count=batch_size)
                async with self.database_queue.iterator() as stream:
                    batch_start_time = time.time()

                    while True:
                        try:
                            async for message in stream:
                                batch.append(message)
                                current_time = time.time()

                                if len(batch) >= batch_size or (current_time - batch_start_time) >= batch_timeout:
                                    await self.process_batch(batch)
                                    batch = []
                                    batch_start_time = current_time

                        except asyncio.CancelledError:
                            logger.info("Database task was canceled. Processing any remaining batch.")
                            if batch:
                                await self.process_batch(batch)

                        except Exception as e:
                            logger.error(f"Error during message consumption: {e}")   

        except asyncio.CancelledError:
            logger.info("Database consume_and_process was canceled.")

        except Exception as e:
            logger.error(f"Error initializing the database consume_and_process: {e}")

    @async_time_execution
    async def process_batch(self, batch: List[aio_pika.IncomingMessage]):
        try:
            tasks = [self.process_message(message) for message in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing message: {result}")
        except asyncio.CancelledError:
            logger.info("Process batch was cancelled.")
        except Exception as e:
            logger.error(f"Error in process_batch: {e}")

    async def process_message(self, message: aio_pika.IncomingMessage):
        try:
            async with message.process():
                data = json.loads(message.body.decode())

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

                self.total_programs += 1

                await self.get_prompt()

        except asyncio.CancelledError:
            logger.info("Process message was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Error processing message: {e}")
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

        if program.body is None:
            logger.debug("Program body is None. Skipping registration.")
            return

        if self.function_body_exists(island['clusters'], hash_value):
            logger.debug("Program with identical body already exists in island. Skipping registration.")
            return

        if expected_version is not None:
            current_version = island['version']
            if current_version != expected_version:
                logger.warning(f"Island {island_id} version mismatch. Expected: {expected_version}, Actual: {current_version}")
                return

        self._register_program_in_island(program, island_id, scores_per_test, hash_value)
        self.registered_programs += 1


    def _register_program_in_island(self, program: code_manipulation.Function, island_id: int, scores_per_test: ScoresPerTest, hash_value: int = None):
        island = self._islands[island_id]
        clusters = island['clusters']
        signature = self._get_signature(scores_per_test)
        program.hash_value = hash_value

        try: 
            if signature not in clusters:
                cluster_data = {}
                cluster_data['score'] = _reduce_score(scores_per_test)
                cluster_data['programs'] = [program]
                clusters[signature] = cluster_data
            else:
                cluster_data = clusters[signature]
                cluster_data['programs'].append(program)
            
            island['num_programs'] += 1
        
        except Exception as e: 
            logger.error(f"Could not append program: {e}")
    
        try: 
            score = _reduce_score(scores_per_test)
            if score > self._best_score_per_island[island_id]:
                self._best_program_per_island[island_id] = program
                self._best_scores_per_test_per_island[island_id] = scores_per_test
                self._best_score_per_island[island_id] = score
                logger.info(f'Best score of island {island_id} increased to {score} with program {program} and scores {scores_per_test}')
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
        logger.debug(f"len(self._islands) {len(self._islands)}")
        island_id = np.random.randint(len(self._islands))
        logger.debug(f"Island id is {island_id}")
        island = self._islands[island_id]

        code, version_generated = self._generate_prompt_for_island(island)
        expected_version = island['version']

        prompt = Prompt(code, version_generated, island_id, expected_version)
        try:
            serialized_prompt = prompt.serialize()
            await self.channel.default_exchange.publish(
                aio_pika.Message(body=serialized_prompt.encode()),
                routing_key='sampler_queue'
            )
            logger.debug("Database: Successfully published prompt to sampler.")
        except Exception as e:
            logger.error(f"Database: Error during prompt preparation or message sending: {e}")

    def _generate_prompt_for_island(self, island) -> tuple[Optional[str], int]:
        clusters = island['clusters']
        signatures = list(clusters.keys())
        logger.debug(f"Island {island}. Cluster signitures are {list(clusters.keys())} ")
        if not signatures:
            logger.warning(f"No clusters found in island {island}. Skipping prompt generation.")
            return None, 0

        cluster_scores = np.array([clusters[signature]['score'] for signature in signatures])
        period = self._config.cluster_sampling_temperature_period
        temperature = self._config.cluster_sampling_temperature_init * (1 - (island['num_programs'] % period) / period)
        threshold = 1e-6

        while True:
            try:
                probabilities = _softmax(cluster_scores, temperature)
                logger.debug(f"probabilities are {probabilities}")
            except Exception as e:
                logger.error(f"Cannot compute softmax: {e}")
                break  # Fall back to uniform sampling below

            valid_indices = np.where(probabilities > threshold)[0]
            valid_probabilities = probabilities[valid_indices]
            valid_signatures = [signatures[i] for i in valid_indices]

            if len(valid_signatures) > 0:
                break  # Proceed with valid probabilities and signatures

            # Reduce temperature if no valid signatures are found
            temperature *= 0.9
            if temperature < 1e-6:
                logger.warning("Temperature reduced below threshold. Falling back to uniform sampling.")
                break

        # Fallback: uniform sampling if no valid probabilities
        if not valid_signatures:
            logger.warning("Using uniform sampling as fallback.")
            valid_signatures = signatures
            valid_probabilities = np.ones(len(signatures)) / len(signatures)

        valid_probabilities /= valid_probabilities.sum()
        functions_per_prompt = min(len(valid_signatures), self._config.functions_per_prompt)

        try:
            idx = np.random.choice(len(valid_signatures), size=functions_per_prompt, p=valid_probabilities, replace=False)
        except ValueError as e:
            logger.error(f"Sampling error: {e}")
            return None, 0

        chosen_signatures = [valid_signatures[i] for i in idx]
        implementations = [self.sample_program(clusters[signature]) for signature in chosen_signatures]  # Simplified
        scores = [clusters[signature]['score'] for signature in chosen_signatures]

        sorted_implementations = [implementations[i] for i in np.argsort(scores)]
        version_generated = len(sorted_implementations) + 1

        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(self, implementations: Sequence[code_manipulation.Function], eval_code: Optional[bool] = False) -> str:
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

        if not isinstance(self._template, code_manipulation.Program):
            try:
                self._template = code_manipulation.text_to_program(self._template)
            except Exception as e:
                logger.error(f"Error converting text to Program: {e}")
                return None

        if hasattr(self._template, 'preface'):
            preface = getattr(self._template, 'preface', '')
            pattern = re.escape(self._function_to_evolve) + r'_v\d+'
            new_function_version = f'{self._function_to_evolve}_v{next_version}'

            if re.search(pattern, preface):
                preface = re.sub(pattern, new_function_version, preface)
                self._template = dataclasses.replace(self._template, preface=preface)

            # Remove all existing imports
            import_pattern = r"(?m)^import .*|from .* import .*"
            preface_cleaned = re.sub(import_pattern, "", preface).strip()

            # Define imports explicitly
            import_numpy = "import numpy as np"
            import_networkx = "import networkx as nx"

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
            sections.append(import_numpy)
            sections.append(import_networkx)
            sections.append("")  # Add a blank line after imports

            # Join sections, ensuring appropriate newlines
            preface = "\n".join(filter(None, sections))+ "\n" + "\n"
            self._template = dataclasses.replace(self._template, preface=preface)

        try:
            if eval_code:
                # Use the first two functions from the template, followed by versioned functions
                first_two_functions = self._template.functions[:2]
                new_functions_list = first_two_functions + versioned_functions
            else:
                # Use only versioned functions
                new_functions_list = versioned_functions

            prompt = dataclasses.replace(self._template, functions=new_functions_list)
            logger.debug(f"Final prompt is:\n{prompt}")
            return str(prompt).rstrip('\n')
        except Exception as e:
            logger.error(f"Error in replacing prompt: {e}")
            return None


    def function_body_exists(self, clusters, hash_value: int) -> bool:
        for cluster in clusters.values():
            programs = cluster['programs']
            for program in programs:
                if program.hash_value == hash_value:
                    return True
        return False

    def _get_signature(self, scores_per_test):
        """Converts scores_per_test to a tuple signature."""
        if all(isinstance(k, str) for k in scores_per_test.keys()):
            scores_per_test = {eval(k): v for k, v in scores_per_test.items()}
        return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))

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