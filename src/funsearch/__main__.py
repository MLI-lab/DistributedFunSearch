import asyncio
import logging
from logging import FileHandler
import json
import aio_pika
from yarl import URL
import torch.multiprocessing as mp
import time
import os
import signal
import sys
import pickle
from funsearch import programs_database
from funsearch import sampler
from funsearch import code_manipulation
from multiprocessing import Manager
import copy
import psutil
import GPUtil
import pynvml
from typing import Sequence, Any
import datetime
from funsearch import evaluator
import signal
import sys
from funsearch import gpt
import asyncio
import aio_pika
from multiprocessing import current_process
import argparse
import glob
import shutil
from funsearch.scaling_utils import ResourceManager
import importlib.util
import time 

def load_config(config_path):
    """
    Dynamically load a configuration module from a specified file path.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if not hasattr(config_module, "Config"):
        raise ValueError(f"The configuration file at {config_path} must define a 'Config' class.")
    
    return config_module.Config()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def backup_python_files(src, dest, exclude_dirs=[]):
    """
    Recursively copies all Python files in src to dest.
    If dest or any subdirectory does not exist, they are created.
    """
    for file_path in glob.glob(os.path.join(src, '**', '*.py'), recursive=True):
        if "/code_backup/" in file_path:
            continue
        if any([file_path.startswith(dir) for dir in exclude_dirs]):
            continue
        new_path = f"{dest}/{file_path.replace('./', '')}"
        dirname = os.path.dirname(new_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy(file_path, new_path)


class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config, log_dir, TARGET_SIGNATURES):
        self.template = code_manipulation.text_to_program(specification)
        self.template_pdb = code_manipulation.text_to_program(specification, remove_classes=True)
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger(log_dir)
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        if self.config.sampler.gpt: 
            self.resource_manager = ResourceManager(log_dir=log_dir, cpu_only=True)
        else: 
            self.resource_manager = ResourceManager(log_dir=log_dir)
        self.process_to_device_map = {}
        self.TARGET_SIGNATURES = TARGET_SIGNATURES

    def initialize_logger(self, log_dir):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)

        # Create the log directory for the experiment
        os.makedirs(log_dir, exist_ok=True)

        log_file_path = os.path.join(log_dir, 'funsearch.log')
        handler = FileHandler(log_file_path, mode='w') 
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger

    async def publish_initial_program_with_retry(self, amqp_url, initial_program_data, max_retries=5, delay=5):
        attempt = 0
        while attempt < max_retries:
            try:
                sampler_connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                )
                sampler_channel = await sampler_connection.channel()

                # Ensure the evaluator_queue is declared
                await sampler_channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )

                await sampler_channel.default_exchange.publish(
                    aio_pika.Message(body=initial_program_data.encode()),
                    routing_key='evaluator_queue'
                )
                self.logger.info("Published initial program")
                await sampler_channel.close()
                await sampler_connection.close()
                return  # Exit the function after successful publish
            except Exception as e:
                attempt += 1
                self.logger.error(f"Attempt {attempt} failed to publish initial program: {e}")
                if attempt < max_retries:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error("Max retries reached. Failed to publish initial program.")
                    raise e  # Re-raise the exception after max retries

    async def log_tasks(self):
        """
        Periodically logs the following information every 60 seconds:
        - Total number of active asyncio tasks.
        - Task details such as name, coroutine function, and current state.
        - If a task is in the "PENDING" state, logs the stack frames of the task to help diagnose blocking issues.
        Usage:
            asyncio.create_task(self.log_tasks())
        """
        while True:
            tasks = asyncio.all_tasks()
            self.logger.debug(f"Currently {len(tasks)} tasks running:")
            for task in tasks:
                coro_name = task.get_coro().__name__ if task.get_coro() else "Unknown"
                self.logger.debug(f"Task: {task.get_name()}, Function: {coro_name}, Status: {task._state}")
            
                # Log the stack of the task if it's pending or blocked
                if task._state == "PENDING":
                    stack = task.get_stack()
                    for frame in stack:
                        self.logger.debug(f"Pending Task Frame: {frame}")
            await asyncio.sleep(60)

    async def main_task(self, save_checkpoints_path, enable_scaling=True, checkpoint_file=None):
        try: 
            amqp_url = URL(
                f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/{self.config.rabbitmq.vhost}' #{self.config.rabbitmq.vhost}
            ).update_query(heartbeat=480000)
            connection = await aio_pika.connect_robust(amqp_url)
        except Exception as e:
            try:
                self.logger.info("No vhost configured, connecting without.")
                amqp_url = URL(
                    f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/' #{self.config.rabbitmq.vhost}
                ).update_query(heartbeat=480000)
                connection = await aio_pika.connect_robust(amqp_url)
            except Exception as e: 
                self.logger.info("Cannot connect to rabbitmq. Change config file.")
    
        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}")

        # Initialize function to evolve
        function_to_evolve = 'priority'

        if checkpoint_file is None:
            initial_program_data = json.dumps({
                "sample": self.template.get_function(function_to_evolve).body,
                "island_id": None,
                "version_generated": None,
                "expected_version": 0
            })

        try:
            # Create connections and declare queues
            sampler_connection = await aio_pika.connect_robust(amqp_url, timeout=300)
            self.sampler_channel = await sampler_connection.channel()

            database_connection = await aio_pika.connect_robust(amqp_url, timeout=300)
            self.database_channel = await database_connection.channel()

            evaluator_queue = await self.sampler_channel.declare_queue(
                "evaluator_queue", durable=False, auto_delete=True,
                arguments={'x-consumer-timeout': 360000000}
            )
            sampler_queue = await self.sampler_channel.declare_queue(
                "sampler_queue", durable=False, auto_delete=True,
                arguments={'x-consumer-timeout': 360000000}
            )
            database_queue = await self.database_channel.declare_queue(
                "database_queue", durable=False, auto_delete=True,
                arguments={'x-consumer-timeout': 360000000}
            )
            try:
                # Now create the database instance
                database = programs_database.ProgramsDatabase(
                    database_connection, self.database_channel, database_queue,
                    sampler_queue, evaluator_queue, self.config.programs_database,
                    self.template_pdb, function_to_evolve, checkpoint_file, save_checkpoints_path,
                    mode=self.config.evaluator.mode, eval_code=self.config.evaluator.eval_code, include_nx=self.config.evaluator.include_nx,
                    start_n=self.config.evaluator.start_n, end_n=self.config.evaluator.end_n, s_values=self.config.evaluator.s_values, no_deduplication=self.config.programs_database.no_deduplication, prompt_limit=args.prompt_limit, optimal_solution_programs=args.optimal_solution_programs
                )
                database_task = asyncio.create_task(database.consume_and_process())
            except Exception as e: 
                self.logger.error(f"Exception in database as {e}")

            checkpoint_task = asyncio.create_task(database.periodic_checkpoint())

            # Start consumers before publishing
            try:
                self.start_initial_processes(function_to_evolve, amqp_url, checkpoint_file)
                self.logger.info("Initial processes started successfully.")
            except Exception as e:
                self.logger.error(f"Failed to start initial processes: {e}")

            # Publish the initial program with retry logic
            while True:
                sampler_queue = await self.sampler_channel.declare_queue("sampler_queue", passive=True)
                consumer_count = sampler_queue.declaration_result.consumer_count
                self.logger.info(f"consumer_count is {consumer_count} while config num_samplers is {self.config.num_samplers}")

                if consumer_count > self.config.num_samplers - 1 and checkpoint_file is None:
                    await self.publish_initial_program_with_retry(amqp_url, initial_program_data)
                    break
                elif consumer_count > self.config.num_samplers - 1:
                    await database.get_prompt()
                    self.logger.info(f"Loading from checkpoint: {checkpoint_file}")
                    break
                else:
                    self.logger.info("No consumers yet on sampler_queue. Retrying in 10 seconds...")
                    await asyncio.sleep(10)

            # Start resource logging
            resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=60))

            self.tasks = [database_task, checkpoint_task, resource_logging_task]

            if enable_scaling:
                try: 
                    scaling_task = asyncio.create_task(
                        self.resource_manager.run_scaling_loop(
                            evaluator_queue=evaluator_queue, 
                            sampler_queue=sampler_queue,
                            evaluator_processes=self.evaluator_processes,
                            sampler_processes=self.sampler_processes,
                            evaluator_function=self.evaluator_process,
                            sampler_function=self.sampler_process,
                            evaluator_args=(self.template, self.inputs, amqp_url, self.TARGET_SIGNATURES),
                            sampler_args=(amqp_url,),
                            max_evaluators=args.max_evaluators,
                            max_samplers=args.max_samplers,
                            check_interval=args.check_interval,
                        )
                    )
                    self.tasks.append(scaling_task)
                except Exception as e: 
                    self.logger.error(f"Error enabling scaling {e}")

            self.channels = [self.database_channel, self.sampler_channel]
            self.queues = ["database_queue", "sampler_queue", "evaluator_queue"]

            # Run all tasks concurrently
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")


    def start_initial_processes(self, function_to_evolve, amqp_url, checkpoint_file):
        amqp_url = str(amqp_url)

        # If self.config.sampler.gpt is True, just start samplers without GPU device assignment
        if self.config.sampler.gpt:
            self.logger.info("GPT mode enabled. Starting sampler processes without GPU device assignment.")
            for i in range(self.config.num_samplers):
                device = None
                try:
                    proc = mp.Process(target=self.sampler_process, args=(amqp_url, device), name=f"Sampler-{i}")
                    proc.start()
                    self.logger.debug(f"Started Sampler Process {i} (GPT mode) with PID: {proc.pid}")
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device
                except Exception as e:
                    self.logger.error(f"Error starting sampler {i}: {e}")
                    continue
        else:
            assigned_gpus = set()
            # Use the ResourceManager's assign_gpu_device method for consistent GPU assignment.
            for i in range(self.config.num_samplers):
                try: 
                    assignment = self.resource_manager.assign_gpu_device(assigned_gpus=assigned_gpus)
                except Exception as e: 
                    self.logger.error(f"Cannot start sampler {i}: No suitable GPU available and error {e}.")

                if assignment is None:
                    self.logger.error("No suitable GPU available for sampler. Skipping or failing gracefully.")
                    continue
                else: 
                    host_gpu, device = assignment
                    assigned_gpus.add(device)
                self.logger.info(f"Assigning sampler {i} to GPU {device} (host GPU: {host_gpu})")
                try:
                    proc = mp.Process(target=self.sampler_process, args=(amqp_url, device), name=f"Sampler-{i}")
                    proc.start()
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device
                    self.logger.debug(f"Process-to-Device Map: {self.process_to_device_map}")
                except Exception as e:
                    self.logger.error(f"Failed to start sampler {i} due to error: {e}")
                    continue

        # Start initial evaluator processes
        for i in range(self.config.num_evaluators):
            proc = mp.Process(target=self.evaluator_process, args=(self.template, self.inputs, amqp_url, self.TARGET_SIGNATURES), name=f"Evaluator-{i}")
            proc.start()
            self.logger.debug(f"Started Evaluator Process {i} with PID: {proc.pid}")
            self.evaluator_processes.append(proc)



    def sampler_process(self, amqp_url, device=None):
        local_id = current_process().pid  # Use process ID as a local identifier
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection = None
        channel = None
        sampler_task = None

        async def graceful_shutdown(loop, connection, channel, sampler_task):
            self.logger.info(f"Sampler {local_id}: Initiating graceful shutdown...")
            if sampler_task:
                try:
                    await asyncio.wait_for(sampler_task, timeout=60)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Sampler {local_id}: Task timed out. Cancelling...")
                    sampler_task.cancel()
                    try: 
                        await sampler_task  # Ensure task cancellation completes
                    except Exception as e:
                        self.logger.warning(f"Evaluator {local_id}: Error after task cancellation: {e}")
                except Exception as e:  
                    self.logger.info(f"Sampler {local_id}: Error during task closure: {e}")

            if channel:
                try:
                    await channel.close()
                except Exception as e:
                    self.logger.info(f"Sampler {local_id}: Error closing channel: {e}")

            if connection:
                try:
                    await connection.close()
                except Exception as e:
                    self.logger.info(f"Sampler {local_id}: Error closing connection: {e}")

            self.logger.info(f"Evaluator {local_id}: Graceful shutdown complete.")
            loop.stop()  # Stop the event loop from within this coroutine



        async def run_sampler():
            nonlocal connection, channel, sampler_task 
            try:
                self.logger.debug(f"Sampler {local_id}: Starting connection to RabbitMQ.")
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    #client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                self.logger.debug(f"Sampler {local_id}: Connected to RabbitMQ.")
                channel = await connection.channel()
                self.logger.debug(f"Sampler {local_id}: Channel established.")

                sampler_queue = await channel.declare_queue(
                    "sampler_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared sampler_queue.")

                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared evaluator_queue.")

                try:
                    if self.config.sampler.gpt:
                        self.logger.debug("Before initialization")
                        sampler_instance = gpt.Sampler(
                            connection, channel, sampler_queue, evaluator_queue, self.config.sampler)
                        self.logger.debug(f"Sampler {local_id}: Initialized Sampler instance.")
                    else: 
                        sampler_instance = sampler.Sampler(
                            connection, channel, sampler_queue, evaluator_queue, self.config.sampler, device)
                        self.logger.debug(f"Sampler {local_id}: Initialized Sampler instance.")
                except Exception as e: 
                    self.logger.error(f"Could not start Sampler instance {e}")
                    return 

                sampler_task = asyncio.create_task(sampler_instance.consume_and_process())
                await sampler_task

            except asyncio.CancelledError:
                print(f"Sampler {local_id}: Process was cancelled.")
            except Exception as e:
                print(f"Sampler {local_id} encountered an error: {e}")
            finally:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                self.logger.debug(f"Sampler {local_id}: Connection closed.")

        def shutdown_callback():
            self.logger.info(f"Sampler {local_id}: Received shutdown signal, scheduling graceful shutdown.")
            # Schedule the coroutine on the existing loop:
            asyncio.create_task(graceful_shutdown(loop, connection, channel, sampler_task))

        loop.add_signal_handler(signal.SIGTERM, shutdown_callback)
        loop.add_signal_handler(signal.SIGINT, shutdown_callback)

        try:
            loop.run_until_complete(run_sampler())
        finally:
            loop.close()
            self.logger.info(f"Sampler {local_id}: Event loop closed.")


    def evaluator_process(self, template, inputs, amqp_url, TARGET_SIGNATURES):
        import funsearch.evaluator
        import signal
        import asyncio

        local_id = mp.current_process().pid  # Use process ID as a local identifier

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # We’ll store these in outer-scope variables so both run_evaluator()
        # and graceful_shutdown() can access them.
        connection = None
        channel = None
        evaluator_task = None

        async def graceful_shutdown(loop, connection, channel, evaluator_task):
            """Gracefully shut down the evaluator task, AMQP channel, and connection."""
            self.logger.info(f"Evaluator {local_id}: Initiating graceful shutdown...")

            if evaluator_task:
                try:
                    # Give the task time to finish cleanly
                    await asyncio.wait_for(evaluator_task, timeout=60)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Evaluator {local_id}: Task timed out. Cancelling...")
                    evaluator_task.cancel()
                    try:
                        await evaluator_task  # Ensure cancellation completes
                    except Exception as e:
                        self.logger.warning(f"Evaluator {local_id}: Error after task cancellation: {e}")
                except Exception as e: 
                    self.logger.info(f"Evaluator {local_id}: Error during task closure: {e}")

            if channel:
                try:
                    await channel.close()
                except Exception as e:
                    self.logger.info(f"Evaluator {local_id}: Error closing channel: {e}")

            if connection:
                try:
                    await connection.close()
                except Exception as e:
                    self.logger.info(f"Evaluator {local_id}: Error closing connection: {e}")

            self.logger.info(f"Evaluator {local_id}: Graceful shutdown complete.")
            loop.stop()  # Stop the event loop from within this coroutine

        async def run_evaluator():
            """Main async entry: connects to AMQP, starts the evaluator task, etc."""
            nonlocal connection, channel, evaluator_task
        
            try:
                connection = await aio_pika.connect_robust(amqp_url, timeout=300)
                channel = await connection.channel()

                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )
                database_queue = await channel.declare_queue(
                    "database_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )

                evaluator_instance = evaluator.Evaluator(
                    connection, channel, evaluator_queue, database_queue, 
                    self.template, 'priority', 'evaluate', inputs, args.sandbox_base_path,
                    timeout_seconds=self.config.evaluator.timeout, 
                    local_id=local_id, 
                    TARGET_SIGNATURES=self.TARGET_SIGNATURES
                )

                # Create the evaluator task.
                evaluator_task = asyncio.create_task(evaluator_instance.consume_and_process())

                # Wait for the evaluator task to finish (or get cancelled).
                await evaluator_task

            except asyncio.CancelledError:
                self.logger.info(f"Evaluator {local_id}: Process was cancelled.")
            except Exception as e:
                self.logger.info(f"Evaluator {local_id}: Error occurred: {e}")
            finally:
                # In case we didn’t go through graceful_shutdown yet, close everything
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                self.logger.debug(f"Evaluator {local_id}: Connection/Channel closed.")

        def shutdown_callback():
            self.logger.info(f"Evaluator {local_id}: Received shutdown signal, scheduling graceful shutdown.")
            # Schedule the coroutine on the existing loop:
            asyncio.create_task(graceful_shutdown(loop, connection, channel, evaluator_task))

        loop.add_signal_handler(signal.SIGTERM, shutdown_callback)
        loop.add_signal_handler(signal.SIGINT, shutdown_callback)

        try:
            loop.run_until_complete(run_evaluator())
        finally:
            loop.close()
            self.logger.info(f"Evaluator {local_id}: Event loop closed.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run FunSearch experiment.")

######################################### General setting related arguments #######################################

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Enable backup of Python files before running the task.",
    )

    parser.add_argument(
        "--save_checkpoints_path",
        type=str,
        default=os.path.join(os.getcwd(), "Checkpoints"),
        help="Path to where the checkpoints should be written. Defaults to './Checkpoints'.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file. Defaults to None if not provided.",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(os.getcwd(), "config.py"),  # Set default to 'config.py' in the current directory
        help="Path to the configuration file (Python script containing the experiment config). Defaults to './config.py'.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(os.getcwd(), "logs"),
        help="Directory where logs will be stored. Defaults to './logs'.",
    )

    parser.add_argument(
        "--sandbox_base_path",
        type=str,
        default=os.path.join(os.getcwd(), "sandbox"),
        help="Path to the sandbox directory. Defaults to './sandbox'.",
    )

########################################## Resources related arguments #############################################

    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling of evaluators and samplers (enabled by default).",
    )

    parser.add_argument(
        "--check_interval",
        type=int,
        default=120,
        help="Time interval (in seconds) between consecutive scaling checks for evaluators and samplers. Defaults to 120s (2 minutes)."
    )

    parser.add_argument(
        "--max_evaluators",
        type=int,
        default=1000,
        help="Maximum evaluators the system can scale up to. Adjust based on resource availability. Default no hard limit and based on dynamic resource checks."
    )

    parser.add_argument(
        "--max_samplers",
        type=int,
        default=1000, 
        help="Maximum samplers the system can scale up to. Adjust based on resource availability. Default no hard limit and based on dynamic resource checks."
    )

########################## Termination related arguments ###########################################

    parser.add_argument(
        "--prompt_limit",
        type=int,
        default=400_000, 
        help="Maximum number of prompts that can be generated before stopping further publishing. The system will continue processing remaining queue messages. Adjust based on computational constraints."
    )

    parser.add_argument(
        "--optimal_solution_programs",
        type=int,
        default=20_000,
        help="Number of additional programs to generate after the first optimal solution is found. Once this limit is reached, further publishing stops, but remaining queue messages continue processing."
    )

    parser.add_argument(
        "--target_solutions",
        type=str,
        default='{"(6, 1)": 10, "(7, 1)": 16, "(8, 1)": 30, "(9, 1)": 52, "(10, 1)": 94, "(11, 1)": 172}',  
        help="JSON string specifying target solutions for (n, s_value). Example: '{\"(6, 1)\": 8, \"(7, 1)\": 14, \"(8, 1)\": 25}'"
    )

    args = parser.parse_args()

    # Convert JSON string to dictionary
    try:
        if args.target_solutions:
            TARGET_SIGNATURES = json.loads(args.target_solutions)
            TARGET_SIGNATURES = {eval(k): v for k, v in TARGET_SIGNATURES.items()}  # Convert string keys to tuples
        else:
            TARGET_SIGNATURES=args.target_solutions
            
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for --target_solutions. Example: '{\"(6, 1)\": 8, \"(7, 1)\": 14, \"(8, 1)\": 25}'.")

    # Invert the logic: dynamic scaling is True by default unless explicitly disabled
    enable_dynamic_scaling = not args.no_dynamic_scaling

    if args.backup:
        # Define the source directory (current working directory)
        src_dir = os.getcwd()

        # Define the base directory for backups
        backup_base_dir = '/mnt/hdd_pool/userdata/franziska/code_backups'
        os.makedirs(backup_base_dir, exist_ok=True)

        # Create a timestamped backup directory
        backup_dir = os.path.join(backup_base_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)

        # Call the backup function
        backup_python_files(src=src_dir, dest=backup_dir)

        # Log the backup operation
        print(f"Backup completed. Python files saved to: {backup_dir}")

    # Configure separate logger for time and memory logging
    time_memory_logger = logging.getLogger('time_memory_logger')
    time_memory_logger.setLevel(logging.INFO)

    os.makedirs(args.log_dir, exist_ok=True) 

    time_memory_log_file = os.path.join(args.log_dir, 'time_memory.log')  
    file_handler = logging.FileHandler(time_memory_log_file, mode='w')  
    file_handler.setLevel(logging.INFO)

    # Define a simple log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    time_memory_logger.addHandler(file_handler)

    async def main():
        config = load_config(args.config_path)

        # Load the specification from the provided path or default
        spec_path = config.evaluator.spec_path
        try:
            with open(spec_path, 'r') as file:
                specification = file.read()
            if not isinstance(specification, str) or not specification.strip():
                raise ValueError("Specification must be a non-empty string.")
        except FileNotFoundError:
            print(f"Error: Specification file not found at {spec_path}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error in specification: {e}")
            sys.exit(1)

        if not (len(config.evaluator.s_values) == len(config.evaluator.start_n) == len(config.evaluator.end_n)):
            raise ValueError("The number of elements in --s-values, --start-n, and --end-n must match.")

        inputs = [(n, s) for s, start_n, end_n in zip(config.evaluator.s_values, config.evaluator.start_n, config.evaluator.end_n) for n in range(start_n, end_n + 1)]
 
        # Initialize the task manager
        task_manager = TaskManager(specification=specification, inputs=inputs, config=config, log_dir=args.log_dir, TARGET_SIGNATURES=TARGET_SIGNATURES )

        task = asyncio.create_task(
            task_manager.main_task(
                save_checkpoints_path=args.save_checkpoints_path,
                enable_scaling=enable_dynamic_scaling,                
                checkpoint_file=args.checkpoint  
            )
        )
        await task  # Ensure the task is awaited

    # Top-level call to asyncio.run() to start the event loop
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error in asyncio.run(main()): {e}")
