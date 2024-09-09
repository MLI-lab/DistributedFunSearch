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
from multiprocessing import Manager
from multiprocessing.managers import BaseManager, NamespaceProxy, DictProxy
from typing import Mapping, Any, List, Sequence
import code_manipulation
import config as config_lib
import json
import aio_pika
import re
from profiling import async_time_execution, async_track_memory


logger = logging.getLogger('main_logger')

class CustomManager(BaseManager):
    pass

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
  # Take index of maximum probability and adjusts this value so that the sum of all probabilities is exactly 1.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result


def _reduce_score(scores_per_test: dict, mode: str = 'last') -> float:
    """
    Reduces per-test scores into a single score based on the specified mode.
    - 'last': Returns the score corresponding to the last test entry in the provided mapping.
    - 'average': Computes the average score of all tests in the mapping.
    If any dimensions are missing (6, 7, 8, 9, 10, 11), they are added with a score of 0.
    """
    n_dimensions = 6  # Define the number of expected dimensions (e.g., 6)
    all_dimensions = list(range(n_dimensions))  # Create a list of dimensions [0, 1, 2, ..., n_dimensions-1]

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
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Island from which the example priority functions where sampled. New generated prompt is registered to same island_id.
  """
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


def create_island_proxy(manager, template, function_to_evolve, functions_per_prompt, cluster_sampling_temperature_init, cluster_sampling_temperature_period):
    """ Creates IslandProxy instances using a custom manager """
    return manager.Island(
        template=template,
        function_to_evolve=function_to_evolve,
        functions_per_prompt=functions_per_prompt,
        cluster_sampling_temperature_init=cluster_sampling_temperature_init,
        cluster_sampling_temperature_period=cluster_sampling_temperature_period,
    )

class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        manager: multiprocessing.managers.SyncManager,
        mang:  multiprocessing.managers.BaseManager,
        connection: aio_pika.RobustConnection,
        channel: aio_pika.RobustChannel,
        database_queue: aio_pika.Queue,
        sampler_queue: aio_pika.Queue,
        evaluator_queue: aio_pika.Queue,
        config: config_lib.ProgramsDatabaseConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
    ): 
        self.manager=manager
        self.mang=mang
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
        self.execution_failed=0
        self._best_score_per_island = [-float('inf')] * config.num_islands
        self._best_program_per_island = [None] * config.num_islands
        self._best_scores_per_test_per_island = [None] * config.num_islands
        self._last_reset_time = time.time()
        self._islands: list[Island] = mang.list()
        try: 
            for _ in range(config.num_islands):
                self._islands.append(create_island_proxy(self.manager, self._template, self._function_to_evolve, self._config.functions_per_prompt, self._config.cluster_sampling_temperature_init, self._config.cluster_sampling_temperature_period))

        except Exception as e: 
            logger.error(f"Cannot fill manager list because {e}")

    def serialize_checkpoint(self) -> dict:
        """
        Serializes the necessary state of the database for checkpointing.
        """
        checkpoint_data = {
            "registered_programs": self.registered_programs,
            "total_programs": self.total_programs,
            "execution_failed": self.execution_failed,
            "best_score_per_island": self._best_score_per_island,
            "best_program_per_island": [str(program) if program else None for program in self._best_program_per_island],
            "best_scores_per_test_per_island": self._best_scores_per_test_per_island,
            "last_reset_time": self._last_reset_time,
            "islands_state": []
        }

        for island in self._islands:
            island_data = {
                "version": island.version,
                "clusters": {}
            }
            clusters = island.get_clusters() # use prox method to get clusters
            for signature, cluster in clusters.items():
                island_data["clusters"][signature] = {
                    "score": cluster.get_score(),
                    "programs": [str(program) for program in cluster.get_programs()]
                }
            checkpoint_data["islands_state"].append(island_data)

        return checkpoint_data


    async def consume_and_process(self) -> None:
        """ Consumes messages in batches from database queue and sends to be processed """
        batch_size = 10  
        batch = []  
        batch_timeout = 0.1  # Timeout in seconds to force batch processing when no more messages are coming in. 

        async with self.channel:
            await self.channel.set_qos(prefetch_count=10)
            async with self.database_queue.iterator() as stream:
                batch_start_time = time.time()
                try:
                    async for message in stream:
                        batch.append(message)
                        current_time = time.time()
                        if len(batch) >= batch_size or (current_time - batch_start_time) >= batch_timeout:
                            await self.process_batch(batch)
                            batch = []  
                            batch_start_time = current_time 
                except asyncio.CancelledError:
                    logger.warning("Evaluator raised error and executing self.shutdown")
                except Exception as e:
                    logger.error(f"Error processing messages: {e}")

    #@async_time_execution
    #@async_track_memory
    async def process_batch(self, batch: List[aio_pika.IncomingMessage]):
        """
        Processes a batch of messages asynchronously.
        Each message in the batch is handled concurrently, allowing other tasks to run while one task is waiting for I/O operations (e.g., reading a file or waiting for a network response).
        """ 
        tasks = [self.process_message(message) for message in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing message: {result}")

    async def process_message(self, message: aio_pika.IncomingMessage):
        """ 
        Processes messages by registering syntactically correct programs and fetching new prompts.
        """
        async with message.process(): # asynch with context manager to acknowledge messages automatically once block of code is complete. 
            data = json.loads(message.body.decode()) # decode() to convert from bytes to a string, json.loads() from json string into a Python dictionary.

            if data["new_function"] == "return":
                await self.get_prompt()
                self.execution_failed+= 1
                logger.debug("Received None for new_function. Skipping registration.")
                return
            try:
                if isinstance(data["new_function"], dict):
                    program = code_manipulation.Function(**data["new_function"])
                else:
                    program = code_manipulation.Function.deserialize(data["new_function"])
            except Exception as e:
                logger.error(f"Failed to deserialize program: {e}")
                await self.get_prompt()
                return

            island_id = data["island_id"]
            scores_per_test = data["scores_per_test"]
            expected_version = data.get("expected_version", None)
            try:
                await self.register_program(program, island_id, scores_per_test, expected_version)
                self.total_programs+= 1
            except Exception as e:
                logger.error(f"Database: Error in register program {e}")
            try:
                await self.get_prompt()
            except Exception as e:
                logger.error(f"Database: Error in issuing a new prompt after registering new function {e}")

    async def register_program(self, program: code_manipulation.Function, island_id: int | None, scores_per_test: ScoresPerTest, expected_version: int = None):
        """
        Performs checks and registers a program.
        """

        # Reset islands if the reset period has been exceeded and at least 50 programs have been registered
        logger.info(f"Difference between resetting times is {time.time() - self._last_reset_time} and config time is {self._config.reset_period}")
        if (time.time() - self._last_reset_time > self._config.reset_period):
            # Check if all islands have at least 50 programs
            all_islands_sufficiently_populated = all(island.get_num_programs() >= self._config.reset_programs for island in self._islands)

            if all_islands_sufficiently_populated:
                logger.info("Reset period exceeded and islands have 50 or more programs, resetting islands.")
                self._last_reset_time = time.time()
                try:
                    await self.reset_islands()  # Reset islands only if both conditions are satisfied
                except Exception as e:
                    logger.error(f"Error in reset island {e}")
            else:
                logger.info("Reset period exceeded, but not all islands have 50 programs. Skipping reset for now.")

        # Threshold for cluster size to check if function body exists.
        cluster_check_threshold = 20
            
        # Do not register program in island if body is None or identical body already exists.
        if island_id is not None:
            island = self._islands[island_id]
            if program.body is None:
                logger.debug("Program body is None. Skipping registration.")
                return
            if island.cluster_length() < cluster_check_threshold:
                if island.function_body_exists(program.clean_body()):
                    logger.debug("Program with identical body already exists in island. Skipping registration.")
                    return

        # Register the program to all islands if island_id is None 
        if island_id is None:
            for island_id in range(len(self._islands)):
                try:
                    logger.debug(f"Before register_program_in_island")
                    await self._register_program_in_island(program, island_id, scores_per_test)
                except Exception as e: 
                    logger.error(f"Could not call self._register_program_in_island because {e}")

        # Register program in island if version of island did not change, otherwise do not register and skip this program.
        else:
            if expected_version is not None:
                current_version = self._islands[island_id].version
                logger.debug(f"Checking version for island {island_id}: expected {expected_version}, current {current_version}")
                if current_version != expected_version:
                    logger.warning(f"Island {island_id} version mismatch. Expected: {expected_version}, Actual: {current_version}")
                    return
            try: 
                await self._register_program_in_island(program, island_id, scores_per_test)
                self.registered_programs += 1
            except Exception as e: 
                logger.error(f"Could not call self._register_program_in_island because {e}")

    async def _register_program_in_island(
        self,
        program: code_manipulation.Function,
        island_id: int,
        scores_per_test: ScoresPerTest,
    ) -> None: 
        try:
            self._islands[island_id].register_program(program, scores_per_test)
        except Exception as e:
            logger.error(f"Could not call register_program because of {e}")
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]: #
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logger.info(f'Best score of island {island_id} increased to {score} with program {program} and scores {scores_per_test}')

        program_details = str(program)
        logger.debug(f'Program added to island %d: %s', island_id, program_details)

    async def reset_islands(self):
        logger.info("Resetting islands based on configuration and scores.")
        # Empty queues to avoid processing messages from islands with old versions, i.e., prior to reset. 
        try: 
            await self.sampler_queue.purge()
            message_count = self.sampler_queue.declaration_result.message_count
            logger.info(f" Sampler queue emptied, there are {message_count} messages in the queue. ")
            await self.evaluator_queue.purge()
            message_count_e = self.evaluator_queue.declaration_result.message_count
            logger.info(f" Evaluator queue emptied, there are {message_count_e} messages in the queue. ")
        except Exception as e: 
            logger.error(f"Could not remove all messages from the queue: {e}")
        try:
            if len(self._best_score_per_island) == 0:
                logger.error("Best score per island is empty. Cannot reset islands.")
                return
            indices_sorted_by_score = np.argsort(self._best_score_per_island + np.random.randn(len(self._best_score_per_island)) * 1e-6) # add small random noise to ensure all islands have slightly different score for strict ordering to be possible.
            num_islands_to_reset = self._config.num_islands // 2 # resetting half the islands
            reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
            keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
            if len(reset_islands_ids) == 0:
                logger.warning("No islands to reset. Skipping reset.")
                return
            for island_id in reset_islands_ids:
                current_version = self._islands[island_id].version
                try: 
                    self._islands[island_id] = create_island_proxy(self.manager, self._template, self._function_to_evolve, self._config.functions_per_prompt,self._config.cluster_sampling_temperature_init, self._config.cluster_sampling_temperature_period)
                except Exception as e: 
                    logger.error(f"Cannot use create isand proxy because of {e}")    
                self._islands[island_id].version = current_version + 1
                self._best_score_per_island[island_id] = -float('inf')
                founder_island_id = np.random.choice(keep_islands_ids)
                founder = self._best_program_per_island[founder_island_id]
                founder_scores = self._best_scores_per_test_per_island[founder_island_id]
                await self._register_program_in_island(founder, island_id, founder_scores)
                logger.info("Reset islands sucessfully")
                # Fetch new prompt after reset to start loop again ( as evaluator and sampler queue are now empty)
                # Not really necessary as probably workers busy with processing messages 
                await self.get_prompt() 
        except Exception as e:
            logger.error(f"Error during island reset: {e}")


    async def get_prompt(self) -> None:
        """
        Asynchronously returns a prompt containing implementations from one chosen island and sends it to the sampler queue.
        Await keyword pauses the execution of the get_prompt function until the message is published to the queue. 
        During this pause, other tasks can run. The actual I/O operation here is sending the message to the sampler_queue via aio_pika.        
        """
        island_id = np.random.randint(len(self._islands))
        try: 
            # Code is the string that is the prompz
            code, version_generated = self._islands[island_id].get_prompt()
        except Exception as e: 
            logger.error(f"Cannot call get prompt, code on island {e}")
        expected_version = self._islands[island_id].version
        logger.info(f"Code is {code} and version generated {version_generated}")
        try: 
            prompt = Prompt(code, version_generated, island_id, expected_version)
        except Exception as e: 
            logger.error(f"Error here: Prompt(code, version_generated, island_id, expected_version) code error message is: {e}")

        try:
            serialized_prompt = prompt.serialize()
            await self.channel.default_exchange.publish(
                aio_pika.Message(body=serialized_prompt.encode()),
                routing_key='sampler_queue'
            )
            logger.debug("Database: Successfully published prompt to sampler.")
        except Exception as e:
            logger.error(f"Database: Error during prompt preparation or message sending: {e}")


def create_new_manager():
    CustomManager.register('Cluster', Cluster)
    CustomManager.register('ClusterProxy', ClusterProxy)
    manager = CustomManager()
    manager.start()  # This starts the server process
    return manager


def create_cluster_proxy(manager, score, program):
    """ Factory method to create cluster proxies using the island-specific manager. """
    logger = logging.getLogger('main_logger')  # Get the main logger
    logger.info(f"Creating cluster proxy with score {score} and program {program}")
    return manager.Cluster(score, program)



class Island:
    def __init__(
        self, 
        template: str, 
        function_to_evolve: str,
        functions_per_prompt: str, 
        cluster_sampling_temperature_init: float , 
        cluster_sampling_temperature_period: int, 
    ):
        self.manager=create_new_manager()  # Each IslandProxy gets its own manager
        self.mang = Manager()
        self._clusters = self.mang.dict()
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._functions_per_prompt = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self._num_programs = 0
        self.version = 0
        self.scores_per_test=None

    def get_num_programs(self) -> int:
        return self._num_programs

    def get_clusters(self):
        """Return the clusters dictionary."""
        return self._clusters

    def cluster_length(self):
        return len(self._clusters)

    def _get_signature(self, scores_per_test):
        """ Converts string tuple keys to actual tuples, sorts them, and retrieves corresponding values. """
        # Converting string keys to tuples if they are not already tuples
        if all(isinstance(k, str) for k in scores_per_test.keys()):
            scores_per_test = {eval(k): v for k, v in scores_per_test.items()}

        # Sorting keys which are now tuples and creating a signature tuple
        return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


    def _reduce_score(self, scores_per_test: dict, mode: str = 'last') -> float:
        """
        Reduces per-test scores into a single score based on the specified mode.
        - 'last': Returns the score corresponding to the last test entry in the provided mapping.
        - 'average': Computes the average score of all tests in the mapping.
        If any dimensions are missing (6, 7, 8, 9, 10, 11), they are added with a score of 0.
        """
        n_dimensions = 6  # Define the number of expected dimensions (e.g., 6)
        all_dimensions = list(range(n_dimensions))  # Create a list of dimensions [0, 1, 2, ..., n_dimensions-1]
        print(f"Scores per test are: {scores_per_test}")

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


    def _softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Returns the tempered softmax of 1D finite `logits` representing the cluster scores."""
        if not np.all(np.isfinite(logits)):
            non_finites = set(logits[~np.isfinite(logits)])
            raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
        if not np.issubdtype(logits.dtype, np.floating):
            logits = np.array(logits, dtype=np.float32)
        result = scipy.special.softmax(logits / temperature, axis=-1)
        # Take index of maximum probability and adjusts this value so that the sum of all probabilities is exactly 1.
        index = np.argmax(result)
        result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
        return result

    def register_program(
        self,
        program: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        try: 
            signature = self._get_signature(scores_per_test)
        except Exception as e: 
            print(f"Error in  signature = self._get_signature(scores_per_test) in Island due to {e}")
        if signature not in self._clusters:
            score = self._reduce_score(scores_per_test)
            self._clusters[signature] = create_cluster_proxy(self.manager, score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1



    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        signatures = list(self._clusters.keys())  # clusters is a manager dict that exposes its keys directly
        cluster_scores = np.array([self._clusters[signature].get_score() for signature in signatures])
        print(f"Cluster scores are {cluster_scores}")

        # Initialize the temperature
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (1 - (self._num_programs % period) / period)
        threshold = 1e-6  # Probability threshold to filter valid clusters

        while True:
            try:
                probabilities = self._softmax(cluster_scores, temperature)
                print(f"Probabilities at temperature {temperature} are {probabilities}")
            except Exception as e:
                print(f"Cannot call softmax because cluster scores are {cluster_scores}")
                return None, 0

            # Filter out near-zero probabilities
            valid_indices = np.where(probabilities > threshold)[0]
            valid_probabilities = probabilities[valid_indices]
            valid_signatures = [signatures[i] for i in valid_indices]

            if len(valid_signatures) > 0:
                # If we have valid signatures, break out of the loop
                break

            # If no valid signatures, adjust temperature (make it smaller to increase peakiness)
            print(f"No valid clusters at temperature {temperature}. Decreasing temperature.")
            temperature *= 0.9  # Decrease temperature by 10% each time

            # Optionally, set a lower limit to avoid an infinite loop
            if temperature < 1e-6:
                print("Temperature too low, returning None.")
                return None, 0

        # Normalize the remaining valid probabilities
        valid_probabilities = valid_probabilities / valid_probabilities.sum()

        # Adjust functions_per_prompt to non-zero probabilities
        functions_per_prompt = min(len(valid_signatures), self._functions_per_prompt)

        # Proceed with sampling based on available non-zero probabilities
        try:
            idx = np.random.choice(len(valid_signatures), size=functions_per_prompt, p=valid_probabilities, replace=False)
        except ValueError as e:
            print(f"Error in sampling with np.random.choice: {e}")
            return None, 0

        # Select and sort implementations based on their scores
        chosen_signatures = [valid_signatures[i] for i in idx]
        implementations = [self._clusters[signature].sample_program() for signature in chosen_signatures]
        scores = [self._clusters[signature].get_score() for signature in chosen_signatures]

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1

        return self._generate_prompt(sorted_implementations), version_generated


    def _generate_prompt(
        self,
        implementations: Sequence[code_manipulation.Function]) -> str:

        implementations = copy.deepcopy(implementations)
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'

        try: 
            header = dataclasses.replace(
                implementations[-1],
                name=new_function_name,
                body='',
                docstring=('Improved version of '
                        f'`{self._function_to_evolve}_v{next_version - 1}`.'),
            )
            versioned_functions.append(header)
        except Exception as e: 
            print(f"Error in using replace for header {e}")

        if not isinstance(self._template, code_manipulation.Program):
            try:
                self._template = code_manipulation.text_to_program(self._template)
            except Exception as e:
                print(f"Error in converting text to Program: {e}")
                return None

        # Check if `preface` contains `self._function_to_evolve` and replace it if found
        if hasattr(self._template, 'preface'):
            preface = getattr(self._template, 'preface', '')
    
            # Create a regex pattern to match the function name followed by `_v{number}`
            pattern = re.escape(self._function_to_evolve) + r'_v\d+'
    
            # Define the new versioned function name
            new_function_version = f'{self._function_to_evolve}_v{next_version}'
    
            # Replace the matched pattern (e.g., function_v1) with the new version
            if re.search(pattern, preface):
                preface = re.sub(pattern, new_function_version, preface)
                self.logger.debug(f"Replaced with {new_function_version} in preface.")
        
                # Update the template with the modified preface
                self._template = dataclasses.replace(self._template, preface=preface)
            else:
                self.logger.info(f"The preface does not contain the function name {self._function_to_evolve}.")
        else:
            self.logger.info(f"The template does not have a preface attribute.")

        try:
            prompt = dataclasses.replace(self._template, functions=versioned_functions)
        except Exception as e:
            self.logger.error(f"Error in prompt replace: {e}")

        final_prompt = str(prompt).rstrip('\n')

        return final_prompt




    def function_body_exists(self, cleaned_body: str) -> bool:

        for cluster in self._clusters.values():
            try:
                programs = cluster.get_programs()  # Retrieve all programs from the cluster
                for program in programs:
                    if program.clean_body()== cleaned_body:
                        self.logger.debug(f"Sucessfully compared without error")
                        return True
            except Exception as e:
                self.logger.error(f"Error accessing programs in cluster: {e}")
        return False


class IslandProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__dict__', 'register_program', 'get_prompt', 'function_body_exists', '_get_signature', 'cluster_length', 'get_clusters', 'get_num_programs')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the NamespaceProxy with arguments from Manager
        self.logger = logging.getLogger('main_logger')  # Get the logger for this proxy

    def get_num_programs(self):
        return self._callmethod('get_num_programs')

    def register_program(self, program, scores_per_test):
        self._callmethod('register_program', (program, scores_per_test))

    def get_prompt(self):
        return self._callmethod('get_prompt')

    def function_body_exists(self, function_body):
        return self._callmethod('function_body_exists', (function_body,))

    def _get_signature(self, scores_per_test):
        return self._callmethod('_get_signature', (scores_per_test,))

    def cluster_length(self):
        return self._callmethod('cluster_length')

    def get_clusters(self):
        return self._callmethod('get_clusters')



class Cluster:
    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs = [implementation]
        self._lengths = [len(str(implementation))]

    def get_score(self) -> float:
        return self._score

    def get_programs(self):
        """Returns the list of programs in the cluster."""
        return self._programs

    def register_program(self, program: code_manipulation.Function) -> None:
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (max(self._lengths) + 1e-6)
        probabilities = self._softmax(-normalized_lengths, 1.0)
        return np.random.choice(self._programs, p=probabilities)

    def _softmax(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        if not np.all(np.isfinite(logits)):
            raise ValueError('Non-finite logits')
        if not np.issubdtype(logits.dtype, np.floating):
            logits = np.array(logits, dtype=np.float32)
        result = scipy.special.softmax(logits / temperature, axis=-1)
        max_idx = np.argmax(result)
        result[max_idx] = 1 - np.sum(result[0:max_idx]) - np.sum(result[max_idx+1:])
        return result



class ClusterProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__dict__', 'register_program', 'sample_program', 'get_score', 'get_programs')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the NamespaceProxy with arguments from Manager
        self.logger = logging.getLogger('main_logger')  # Get the logger for this proxy

    def register_program(self, program):
        self._callmethod('register_program', (program,))

    def sample_program(self):
        return self._callmethod('sample_program')

    def get_score(self):
        return self._callmethod('get_score')

    def get_programs(self):
        return self._callmethod('get_programs')









