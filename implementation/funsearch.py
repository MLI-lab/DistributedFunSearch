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
import programs_database
import sampler
import code_manipulation
from multiprocessing import Manager
import copy
import psutil
import GPUtil
import pynvml
from typing import Sequence, Any
import datetime
import evaluator
import signal
import sys
import asyncio
import aio_pika
from multiprocessing import current_process
import argparse
import glob
import shutil
from scaling_utils import ResourceManager
import importlib.util

def load_config(config_path):
    """
    Dynamically load a configuration module from a specified path.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
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
    def __init__(self, specification: str, inputs: Sequence[Any], config, log_dir):
        self.template = code_manipulation.text_to_program(specification)
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
        self.resource_manager = ResourceManager(log_dir=log_dir)

    def initialize_logger(self, log_dir):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)

        # Create the log directory for the experiment
        os.makedirs(log_dir, exist_ok=True)

        log_file_path = os.path.join(log_dir, 'funsearch.log')
        handler = FileHandler(log_file_path, mode='w')  # Create a file handler
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
                    "evaluator_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )

                await sampler_channel.default_exchange.publish(
                    aio_pika.Message(body=initial_program_data.encode()),
                    routing_key='evaluator_queue'
                )
                self.logger.debug("Published initial program")
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


    async def scaling_controller(self, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)
        check_interval_eval = 120  # seconds
        check_interval_sam = 120
        max_evaluators = 110 # (should not be more than num_cores-rest of processes load)/2 as each eval uses two cpus 
        min_evaluators = 1
        max_samplers = 16
        min_samplers = 1
        evaluator_threshold = 5
        sampler_threshold = 15
        initial_sleep_duration = 120  # seconds
        await asyncio.sleep(initial_sleep_duration)

        # Create a connection and channels for getting queue metrics
        try:
            connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
            )
            channel = await connection.channel()
        except Exception as e:
            self.logger.error(f"Error connecting to RabbitMQ: {e}")
            return

        while True:
            try:
                load_avg_1, load_avg_5, _ = os.getloadavg()
                num_cores = len(os.sched_getaffinity(0))  # Available CPU cores
                if load_avg_1 > num_cores or load_avg_5 > num_cores:
                    self.logger.warning(f"5-minute average load ({load_avg_5:.2f}) exceeds available cores ({num_cores}). Scaling down processes.")
    
                    # Continue terminating processes until load is below threshold
                    while load_avg_1 > num_cores and len(self.evaluator_processes) > min_evaluators:
                        self.resource_manager.terminate_process(self.evaluator_processes, 'Evaluator', immediate=True)
                        self.logger.info(f"Terminated one evaluator process. Remaining evaluators: {len(self.evaluator_processes)}")
        
                        # Wait for a minute to allow the system load to adjust
                        await asyncio.sleep(90)
        
                        # Recheck system load
                        load_avg_1, load_avg_5, _ = os.getloadavg()
                        self.logger.info(f"Rechecking System Load (1m): {load_avg_1:.2f}, (5m): {load_avg_5:.2f}, Available Cores: {num_cores}")

                    if load_avg_5 <= num_cores:
                        self.logger.info(f"System load is now within acceptable limits (5m: {load_avg_5:.2f}).")
                    elif len(self.evaluator_processes) <= min_evaluators:
                        self.logger.warning(f"Minimum number of evaluators reached ({min_evaluators}), but system load is still high (5m: {load_avg_5:.2f}).")

                # Collect metrics from queues
                evaluator_message_count = await self.resource_manager.get_queue_message_count(channel, "evaluator_queue")
                sampler_message_count = await self.resource_manager.get_queue_message_count(channel, "sampler_queue")

                # Adjust evaluator processes
                await self.resource_manager.adjust_processes(
                    evaluator_message_count, evaluator_threshold,
                    self.evaluator_processes, self.evaluator_process,
                    args=(self.template, self.inputs, amqp_url),
                    max_processes=max_evaluators, min_processes=min_evaluators,
                    process_name='Evaluator'
                )
                # Adjust sampler processes
                await self.resource_manager.adjust_processes(
                    sampler_message_count, sampler_threshold,
                    self.sampler_processes, self.sampler_process,
                    args=(amqp_url,),
                    max_processes=max_samplers, min_processes=min_samplers,
                    process_name='Sampler'
                )
            except Exception as e:
                print(f"Scaling controller encountered an error: {e}")
            await asyncio.sleep(120)  # Non-blocking sleep

    async def main_task(self, save_checkpoints_path, enable_scaling=True, checkpoint_file=None):
        amqp_url = URL(
            f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/{self.config.rabbitmq.vhost}' # Add virtual host for LRZ {self.config.rabbitmq.vhost}'
        ).update_query(heartbeat=480000)
        pid = os.getpid()

        self.logger.info(f"Main_task is running in process with PID: {pid}")

        # Initialize the template and initial program data
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
            sampler_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
            )
            self.sampler_channel = await sampler_connection.channel()

            database_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
            )
            self.database_channel = await database_connection.channel()

            # Declare queues before starting consumers
            evaluator_queue = await self.sampler_channel.declare_queue(
                "evaluator_queue", durable=False, auto_delete=False,
                arguments={'x-consumer-timeout': 360000000}
            )
            sampler_queue = await self.sampler_channel.declare_queue(
                "sampler_queue", durable=False, auto_delete=False,
                arguments={'x-consumer-timeout': 360000000}
            )
            database_queue = await self.database_channel.declare_queue(
                "database_queue", durable=False, auto_delete=False,
                arguments={'x-consumer-timeout': 360000000}
            )
            
            # Start database
            database = programs_database.ProgramsDatabase(
                database_connection, self.database_channel, database_queue, sampler_queue, evaluator_queue, self.config.programs_database, self.template, function_to_evolve, checkpoint_file, save_checkpoints_path,
            )

            database_task = asyncio.create_task(database.consume_and_process())
            checkpoint_task = asyncio.create_task(database.periodic_checkpoint())

            # Start consumers before publishing
            try:
                self.start_initial_processes(function_to_evolve, amqp_url, checkpoint_file)
                self.logger.info("Initial processes started successfully.")
            except Exception as e:
                print(f"Failed to start initial processes: {e}")


            # Publish the initial program with retry logic
            while True: 
                sampler_queue = await self.sampler_channel.declare_queue("sampler_queue", passive=True)
                consumer_count = sampler_queue.declaration_result.consumer_count

                # Check if there is a consumer attached
                if consumer_count > 0 and checkpoint_file is None:
                    await self.publish_initial_program_with_retry(amqp_url, initial_program_data)
                    break  # Exit the loop once the program is published
                elif consumer_count > 0:
                    await database.get_prompt()
                    self.logger.info(f"Loading from checkpoint: {checkpoint_file}")
                    break  # Exit the loop once the prompt is retrieved
                else:
                    # If no consumer is attached, sleep and check again
                    self.logger.info(f"No consumers yet on sampler_queue. Retrying in 10 seconds...")
                    await asyncio.sleep(10)  # Wait 10 seconds before checking again

            # Dynamically scale workers based on message load and resources and log resource availability
            resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=200))
    
            self.tasks = [database_task, checkpoint_task, resource_logging_task]

            if enable_scaling:
                scaling_task = asyncio.create_task(self.scaling_controller(function_to_evolve, amqp_url))
                self.tasks.append(scaling_task)
            self.channels = [self.database_channel, self.sampler_channel]
            self.queues = [["database_queue"], ["sampler_queue"], ["evaluator_queue"]]

            # Use asyncio.gather to run tasks concurrently
            await asyncio.gather(*self.tasks)

        except Exception as e:
            print(f"Exception occurred in main_task: {e}")


    def start_initial_processes(self, function_to_evolve, amqp_url, checkpoint_file):
        amqp_url = str(amqp_url)

        # Get a list of visible GPUs as remapped by CUDA_VISIBLE_DEVICES (inside the container)
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]  # Host-visible device IDs

        gpus = GPUtil.getGPUs()

        # Create a mapping of host-visible GPU IDs (integers in visible_devices) to container-visible device indices
        id_to_container_index = {visible_devices[i]: i for i in range(len(visible_devices))}

        # Initialize GPU memory usage and utilization tracking (host-visible GPU IDs)
        gpu_memory_info = {gpu.id: (gpu.memoryFree, gpu.memoryUtil * 100) for gpu in gpus if gpu.id in visible_devices}

        self.logger.info(f"Found visible GPUs with initial free memory and utilization: {gpu_memory_info}")

        # Start initial sampler processes
        for i in range(self.config.num_samplers):
            suitable_gpu_id = None
            combined_memory = 0
            combined_gpus = []

            # Check if any single GPU has >= 32 GiB of memory free and < 50% utilization
            for gpu_id, (free_memory, utilization) in gpu_memory_info.items():
                if free_memory > 30000 and utilization < 110: 
                    suitable_gpu_id = gpu_id
                    break
                elif utilization < 100:
                    combined_memory += free_memory
                    combined_gpus.append(gpu_id)

            # If a single GPU was found with sufficient memory
            if suitable_gpu_id is not None:
                container_index = id_to_container_index[suitable_gpu_id]
                device = f"cuda:{container_index}"
                # Adjust memory tracking (simplistic estimation)
                gpu_memory_info[suitable_gpu_id] = (gpu_memory_info[suitable_gpu_id][0] - 32768, gpu_memory_info[suitable_gpu_id][1])
            elif combined_memory >= 32768:  # If combined memory from multiple GPUs is >= 20 GiB
                device = None  # Use None to indicate that multiple GPUs will be used
                self.logger.info(f"Using combination of GPUs: {combined_gpus} with total memory: {combined_memory} MiB")
            else:
                self.logger.error(f"Cannot start sampler {i}: Not enough available GPU memory.")
                continue  # Skip this sampler if no GPU has sufficient memory

            self.logger.info(f"Assigning sampler {i} to device {device if device else 'combined GPUs (None)'}")
            try: 
                proc = mp.Process(target=self.sampler_process, args=(amqp_url, device), name=f"Sampler-{i}")
                proc.start()
                self.logger.debug(f"Started Sampler Process {i} on {device} with PID: {proc.pid}")
                self.sampler_processes.append(proc)
                self.process_to_device_map[proc.pid] = device
            except Exception as e: 
                continue

        # Start initial evaluator processes
        for i in range(self.config.num_evaluators):
            proc = mp.Process(target=self.evaluator_process, args=(self.template, self.inputs, amqp_url), name=f"Evaluator-{i}")
            proc.start()
            self.logger.debug(f"Started Evaluator Process {i} with PID: {proc.pid}")
            self.evaluator_processes.append(proc)


    def sampler_process(self, amqp_url, device):
        local_id = current_process().pid  # Use process ID as a local identifier
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Initialize these variables at a higher scope to be accessible in signal_handler
        connection = None
        channel = None
        sampler_task = None

        async def graceful_shutdown(loop, connection, channel, sampler_task):
            self.logger.info(f"Sampler {local_id}: Initiating graceful shutdown...")

            if sampler_task:
                try:
                    await asyncio.wait_for(sampler_task, timeout=10)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Sampler {local_id}: Task timed out. Cancelling...")
                    sampler_task.cancel()
                    await sampler_task  # Ensure task cancellation completes

            if channel:
                await channel.close()
            if connection:
                await connection.close()

            loop.stop()
            self.logger.info(f"Sampler {local_id}: Graceful shutdown complete.")

        def signal_handler(sig, frame):
            self.logger.info(f"Sampler process {local_id} received signal {sig}. Initiating shutdown.")
            loop.create_task(graceful_shutdown(loop, connection, channel, sampler_task))

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        async def run_sampler():
            nonlocal connection, channel, sampler_task  # Access the outer-scoped variables
            try:
                self.logger.debug(f"Sampler {local_id}: Starting connection to RabbitMQ.")
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                self.logger.debug(f"Sampler {local_id}: Connected to RabbitMQ.")
                channel = await connection.channel()
                self.logger.debug(f"Sampler {local_id}: Channel established.")

                sampler_queue = await channel.declare_queue(
                    "sampler_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared sampler_queue.")

                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared evaluator_queue.")
                try:
                    sampler_instance = sampler.Sampler(
                        connection, channel, sampler_queue, evaluator_queue, self.config, device)
                    self.logger.debug(f"Sampler {local_id}: Initialized Sampler instance.")
                except Exception as e: 
                    self.logger.error(f"Could not start Sampler instance")
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

        try:
            loop.run_until_complete(run_sampler())
        except Exception as e:
            print(f"Sampler process {local_id}: Exception occurred: {e}")
        finally:
            loop.close()
            print(f"Sampler process {local_id} has been closed gracefully.")


    def evaluator_process(self, template, inputs, amqp_url):
        import evaluator
        local_id = mp.current_process().pid  # Use process ID as a local identifier

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Initialize these variables at a higher scope to be accessible in signal_handler
        connection = None
        channel = None
        evaluator_task = None

        async def graceful_shutdown(loop, connection, channel, evaluator_task):
            self.logger.info(f"Evaluator {local_id}: Initiating graceful shutdown...")

            if evaluator_task:
                try:
                    await asyncio.wait_for(evaluator_task, timeout=60)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Evaluator {local_id}: Task timed out. Cancelling...")
                    evaluator_task.cancel()
                    await evaluator_task  # Ensure task cancellation completes
                except Exception as e: 
                    print(f"Evaluator {local_id}: Error during task cancellation: {e}")

            if channel:
                try:
                    await channel.close()
                except Exception as e:
                   print(f"Evaluator {local_id}: Error closing channel: {e}")
            
            if connection:
                try:
                    await connection.close()
                except Exception as e:
                    print(f"Evaluator {local_id}: Error closing connection: {e}")

            loop.stop()
            self.logger.info(f"Evaluator {local_id}: Graceful shutdown complete.")

        async def run_evaluator():
            nonlocal connection, channel, evaluator_task  # Access the outer-scoped variables
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                channel = await connection.channel()
                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                database_queue = await channel.declare_queue(
                    "database_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )

                evaluator_instance = evaluator.Evaluator(
                    connection, channel, evaluator_queue, database_queue, 
                    self.template, 'priority', 'evaluate', inputs, '/workspace/sandboxstorage/', 
                    timeout_seconds=300, local_id=local_id
                )
                evaluator_task = asyncio.create_task(evaluator_instance.consume_and_process())

                await evaluator_task
            except asyncio.CancelledError:
                print(f"Evaluator {local_id}: Process was cancelled.")
            except Exception as e:
                print(f"Evaluator {local_id}: Error occurred: {e}")
            finally:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                self.logger.debug(f"Evaluator {local_id} connection closed.")

        def signal_handler(sig, frame):
            self.logger.info(f"Evaluator process {local_id} received signal {sig}. Initiating shutdown.")
            loop.create_task(graceful_shutdown(loop, connection, channel, evaluator_task))

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            loop.run_until_complete(run_evaluator())
        except Exception as e:
            print(f"Evaluator process {local_id}: Exception occurred: {e}")
        finally:
            loop.close()
            print(f"Evaluator process {local_id} has been closed gracefully.")



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run funsearch with backup, dynamic scaling, and optional specification file.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Enable backup of Python files before running the task.",
    )
    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling of evaluators and samplers (enabled by default).",
    )
    parser.add_argument(
        "--spec-path",
        type=str, 
        default='/Funsearch/implementation/specifications/baseline.txt',
        help="Path to the specification file. Defaults to '/Funsearch/implementation/specifications/baseline.txt'.",
    )

    parser.add_argument(
        "--save_checkpoints_path",
        type=str,
        default=os.path.join(os.getcwd(), 'Checkpoints'),
        help="Path to where the checkpoints should be written to. Defaults to Checkpoints.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file. Defaults to None if not provided.",
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Name of the configuration file (without .py extension). Defaults to 'config'.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where logs will be stored. Defaults to 'logs'."
    )

    args = parser.parse_args()

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

    os.makedirs(args.log_dir, exist_ok=True)  # Ensure the logs folder exists

    time_memory_log_file = os.path.join(args.log_dir, 'time_memory.log')  # Path to the log file
    file_handler = logging.FileHandler(time_memory_log_file, mode='w')  # Create a file handler
    file_handler.setLevel(logging.INFO)

    # Define a simple log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    time_memory_logger.addHandler(file_handler)

    async def main():
        # Load the specification from the provided path or default
        spec_path = args.spec_path
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

        inputs = [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]
        
        config = load_config(args.config_name)
        # Initialize the task manager
        task_manager = TaskManager(specification=specification, inputs=inputs, config=config, log_dir=args.log_dir )

        task = asyncio.create_task(
            task_manager.main_task(
                enable_scaling=enable_dynamic_scaling,
                save_checkpoints_path=args.save_checkpoints_path,
                checkpoint_file=args.checkpoint  # Corrected from args.checkpoint_file to args.checkpoint
            )
        )
        await task  # Ensure the task is awaited

    # Top-level call to asyncio.run() to start the event loop
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error in asyncio.run(main()): {e}")
