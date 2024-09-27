import asyncio
import logging
from logging.handlers import RotatingFileHandler
import json
import aio_pika
from yarl import URL
import torch.multiprocessing as mp
import threading
import time
import os
import signal
import sys
import pickle
import config as config_lib
import programs_database
import llama_grid
import code_manipulation
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
import copy
import psutil
import GPUtil
from typing import Sequence, Any
import datetime
import evaluator
import signal
import sys
import asyncio
import aio_pika
from multiprocessing import current_process
import tracemalloc

tracemalloc.start() # to track memory usage and compare memory usage over time to see where the largest memory allocations are coming from 

def take_snapshot():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[Top 10]")
    for stat in top_stats[:10]:
        print(stat)
    return snapshot

def compare_snapshots(snapshot_before, snapshot_after):
    print("[Snapshot Difference]")
    for line in snapshot_after.compare_to(snapshot_before, 'lineno'):
        print(line)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomManager(BaseManager):
    pass


def save_checkpoint(main_database):
    current_pid = os.getpid()
    current_thread = threading.current_thread().name
    thread_id = threading.get_ident()
    print(f"save_checkpoint: Process ID: {current_pid}, Thread: {current_thread}, Thread ID: {thread_id}")
    # Gets the current time and formats it as a string 'YYYY-MM-DD_HH-MM-SS'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(os.getcwd(), "Checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.pkl")
    data = main_database.serialize_checkpoint()
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"Checkpoint saved at: {filepath}")


class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config: config_lib.Config, mang, manager):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.mang = mang
        self.manager = manager
        self.logger = self.initialize_logger()
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None

    def initialize_logger(self):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.DEBUG)
        log_file_path = os.path.join(os.getcwd(), 'funsearch.log')
        handler = RotatingFileHandler(log_file_path, maxBytes=100 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    async def scaling_controller(self, template, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)
        current_pid = os.getpid()
        current_thread = threading.current_thread().name
        thread_id = threading.get_ident()
        check_interval_eval = 120  # seconds
        check_interval_sam = 100
        check_interval_db= 1080
        max_evaluators = 25
        min_evaluators = 1
        max_samplers = 4
        min_samplers = 1
        max_databases = 5
        min_databases = 0
        evaluator_threshold = 5
        sampler_threshold = 5
        database_threshold = 20
        initial_sleep_duration = 60  # seconds
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
                # Collect metrics from queues
                evaluator_message_count = await self.get_queue_message_count(channel, "evaluator_queue")
                sampler_message_count = await self.get_queue_message_count(channel, "sampler_queue")
                database_message_count = await self.get_queue_message_count(channel, "database_queue")

                # Adjust evaluator processes
                await self.adjust_processes(
                    evaluator_message_count, evaluator_threshold,
                    self.evaluator_processes, self.evaluator_process,
                    args=(template, self.inputs, amqp_url),
                    max_processes=max_evaluators, min_processes=min_evaluators,
                    process_name='Evaluator'
                )

                # Adjust sampler processes
                await self.adjust_processes(
                    sampler_message_count, sampler_threshold,
                    self.sampler_processes, self.sampler_process,
                    args=(amqp_url,),
                    max_processes=max_samplers, min_processes=min_samplers,
                    process_name='Sampler'
                )

                # Adjust database processes
                await self.adjust_processes(
                    database_message_count, database_threshold,
                    self.database_processes, self.database_process,
                    args=(self.config.programs_database, template, function_to_evolve, amqp_url),
                    max_processes=max_databases, min_processes=min_databases,
                    process_name='Database'
                )

            except Exception as e:
                self.logger.error(f"Scaling controller encountered an error: {e}")


            await asyncio.sleep(120)  # Non-blocking sleep

    async def get_queue_message_count(self, channel, queue_name):
        try:
            queue = await channel.declare_queue(queue_name, passive=True)
            message_count = queue.declaration_result.message_count
            return message_count
        except Exception as e:
            self.logger.error(f"Error getting message count for queue {queue_name}: {e}")
            return 0

    async def adjust_processes(self, message_count, threshold, processes, target_fnc, args, max_processes, min_processes, process_name):
        num_processes = len(processes)
        self.logger.debug(f"Adjusting {process_name}: message_count={message_count}, threshold={threshold}, num_processes={num_processes}, min_processes={min_processes}")

        if message_count > threshold and num_processes < max_processes:
            # Scale up
            self.start_process(target_fnc, args, processes, process_name)
            current_processes = len(processes)
            self.logger.info(f"Scaled up {process_name} processes to {current_processes}")

        elif message_count < threshold and num_processes > min_processes:
            # Scale down
            self.terminate_process(processes, process_name)
            current_processes = len(processes)
            self.logger.info(f"Scaled down {process_name} processes to {current_processes}")
        else: 
            self.logger.info(f"No scaling action needed for {process_name}. Current processes: {num_processes}, Message count: {message_count}")
            return 


    def start_process(self, target_fnc, args, processes, process_name):
        current_pid = os.getpid()
        current_thread = threading.current_thread().name
        thread_id = threading.get_ident()
        self.logger.info(f"start_process: Process ID: {current_pid}, Thread: {current_thread}, Thread ID: {thread_id}")

        # CPU check for evaluator processes
        if target_fnc == self.evaluator_process:
            cpu_affinity = os.sched_getaffinity(0)  # Get CPUs available to the container
            cpu_usage = psutil.cpu_percent(percpu=True)  # Get usage for all system CPUs
            container_cpu_usage = [cpu_usage[i] for i in cpu_affinity]

            # Count how many of the available CPUs are under 50% usage
            available_cpus_with_low_usage = sum(1 for usage in container_cpu_usage if usage < 50)
            self.logger.info(f"Available CPUs with <50% usage (in container): {available_cpus_with_low_usage}")

            # Scale up only if more than 3 CPUs have less than 50% usage
            if available_cpus_with_low_usage <= 4:
                self.logger.info(f"Cannot scale up {process_name}: Not enough available CPU resources.")
                return  # Exit the function if not enough CPU resources
            self.logger.info(f"Args for evaluator are {args}")

        # GPU check for sampler processes
        if target_fnc == self.sampler_process:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]  # Ensure non-empty strings and convert to int

            gpus = GPUtil.getGPUs()

            # Create a mapping of host-visible GPU IDs (integers in visible_devices) to container-visible device indices
            id_to_container_index = {visible_devices[i]: i for i in range(len(visible_devices))}

            # Initialize GPU memory usage tracking (host-visible GPU IDs)
            gpu_memory_usage = {gpu.id: gpu.memoryFree for gpu in gpus if gpu.id in visible_devices}

            # Find a suitable GPU with enough free memory
            suitable_gpu_id = None
            for gpu_id, free_memory in gpu_memory_usage.items():
                if free_memory > 1000:  # Check if more than 1000 MIB is available
                    suitable_gpu_id = gpu_id
                    break

            # Map to container-visible device (like cuda:0 or cuda:1)
            if suitable_gpu_id is not None:
                container_index = id_to_container_index[suitable_gpu_id]  # Get container-visible index
                device = f"cuda:{container_index}"
                # Adjust memory tracking (simplistic estimation)
                gpu_memory_usage[suitable_gpu_id] -= 17000
            else:
                self.logger.error(f"Cannot start {process_name}: Not enough available GPU memory.")
                return  # Skip this process if no GPU has sufficient memory

            self.logger.info(f"Assigning {process_name} to device {device}")
            args += (device,)  # Append the GPU device to args

        # Start the process
        proc = mp.Process(target=target_fnc, args=args, name=f"{process_name}-{len(processes)}")
        proc.start()
        processes.append(proc)
        self.logger.info(f"Started {process_name} process with PID: {proc.pid} on device {args[-1]}")


    def get_process_with_zero_or_lowest_cpu(self, processes):
        least_busy_process = None
        min_cpu = float('inf')
    
        # First pass: Check for a process with 0.0% CPU usage
        for proc in processes:
            try:
                p = psutil.Process(proc.pid)
                cpu_usage = p.cpu_percent(interval=1)
                self.logger.debug(f"Process PID {proc.pid} CPU usage: {cpu_usage}%")
                if cpu_usage == 0.0:
                    self.logger.info(f"Process with PID {proc.pid} has 0.0% CPU usage.")
                    return proc
                # Track the least busy process in case no process with 0.0% usage is found
                if cpu_usage < min_cpu:
                    min_cpu = cpu_usage
                    least_busy_process = proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.logger.warning(f"Failed to access CPU usage for process PID {proc.pid}. It might have finished or access was denied.")
                continue

        if least_busy_process:
            self.logger.info(f"No process with 0.0% CPU usage found. Returning least busy process with PID {least_busy_process.pid} having CPU usage {min_cpu}%.")
        else:
            self.logger.info("No suitable process found.")
    
        return least_busy_process


    def terminate_process(self, processes, process_name):
        if processes:
            # Try to get the least busy process
            least_busy_process = self.get_process_with_zero_or_lowest_cpu(processes)
            self.logger.info(f"least_busy_process is {least_busy_process}")
            if least_busy_process is None:
                # If all processes are busy, pop the last process
                least_busy_process = processes.pop()
                self.logger.info("All processes busy terminating last process")
            else:
                # Remove the chosen process from the list
                processes.remove(least_busy_process)

            if least_busy_process.is_alive():
                self.logger.info(f"Initiating termination for {process_name} process with PID: {least_busy_process.pid}")
                least_busy_process.terminate()
                least_busy_process.join(timeout=10)  # Wait for it to fully terminate
                if least_busy_process.is_alive():
                    self.logger.warning(f"{process_name} process with PID: {least_busy_process.pid} is still alive after timeout, forcing kill.")
                    least_busy_process.kill()
                self.logger.info(f"{process_name} process with PID: {least_busy_process.pid} terminated successfully.")
        else:
            self.logger.warning(f"No {process_name} processes to terminate.")

    async def periodic_checkpoint(self, main_database):
        current_pid = os.getpid()
        current_thread = threading.current_thread().name
        thread_id = threading.get_ident()
        self.logger.info(f"periodic_checkpoint: Process ID: {current_pid}, Thread: {current_thread}, Thread ID: {thread_id}")
        checkpoint_interval = 3600  # 1 hour
        while True:
            await asyncio.sleep(checkpoint_interval)  # Non-blocking sleep
            save_checkpoint(main_database)
            self.logger.info("Checkpoint has been saved.")

    async def main_task(self):
        amqp_url = URL(
            f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/'
        ).update_query(heartbeat=180000)
        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}")
        try:
            # Create connections for the samplers and database
            sampler_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
            )

            database_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
            )

            # Channels on separate connections
            self.sampler_channel = await sampler_connection.channel()
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

            template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'

            # Initialize the designated ProgramsDatabase instance for checkpointing
            main_database = programs_database.ProgramsDatabase(
                self.manager, self.mang, database_connection, self.database_channel, database_queue,
                sampler_queue, evaluator_queue, self.config.programs_database, template, function_to_evolve
            )

            main_database_task = asyncio.create_task(main_database.consume_and_process())
            # Schedule periodic_checkpoint and scaling_controller as async tasks
            periodic_checkpoint_task = asyncio.create_task(self.periodic_checkpoint(main_database))
            scaling_controller_task = asyncio.create_task(self.scaling_controller(template, function_to_evolve, amqp_url))

            # Start initial processes
            self.start_initial_processes(template, function_to_evolve, amqp_url)
            self.tasks = [main_database_task, periodic_checkpoint_task, scaling_controller_task]
            self.channels = [self.database_channel, self.sampler_channel]
            self.queues = [["database_queue"], ["sampler_queue"], ["evaluator_queue"]]

            initial_program_data = json.dumps({
                "sample": template.get_function(function_to_evolve).body,
                "island_id": None,
                "version_generated": None,
                "expected_version": 0
            })
            await self.sampler_channel.default_exchange.publish(
                aio_pika.Message(body=initial_program_data.encode()),
                routing_key='evaluator_queue'
            )
            self.logger.debug("Published initial program")

            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")
            main_database.shutdown()


    def start_initial_processes(self, template, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)

        # Get a list of visible GPUs as remapped by CUDA_VISIBLE_DEVICES (inside the container)
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]  # Host-visible device IDs

        gpus = GPUtil.getGPUs()

        # Create a mapping of host-visible GPU IDs (integers in visible_devices) to container-visible device indices
        id_to_container_index = {visible_devices[i]: i for i in range(len(visible_devices))}

        # Initialize GPU memory usage tracking (host-visible GPU IDs)
        gpu_memory_usage = {gpu.id: gpu.memoryFree for gpu in gpus if gpu.id in visible_devices}

        self.logger.info(f"Found visible GPUs with initial free memory: {gpu_memory_usage}")

        # Start initial sampler processes
        for i in range(self.config.num_samplers):
            # Find a suitable GPU with enough free memory
            suitable_gpu_id = None
            for gpu_id, free_memory in gpu_memory_usage.items():
                if free_memory > 17000:  # Check if more than 17000MiB is available
                    suitable_gpu_id = gpu_id
                    break

            # Map to container-visible device (like cuda:0 or cuda:1)
            if suitable_gpu_id is not None:
                container_index = id_to_container_index[suitable_gpu_id]  # Get container-visible index
                device = f"cuda:{container_index}"
                # Adjust memory tracking (simplistic estimation)
                gpu_memory_usage[suitable_gpu_id] -= 17000
            else:
                self.logger.error(f"Cannot start sampler {i}: Not enough available GPU memory.")
                continue  # Skip this sampler if no GPU has sufficient memory

            self.logger.info(f"Assigning sampler {i} to device {device}")

            proc = mp.Process(target=self.sampler_process, args=(amqp_url, device), name=f"Sampler-{i}")
            proc.start()
            self.logger.info(f"Started Sampler Process {i} on {device} with PID: {proc.pid}")
            self.sampler_processes.append(proc)

        # Start initial evaluator and database processes as previously done
        for i in range(self.config.num_evaluators):
            proc = mp.Process(target=self.evaluator_process, args=(template, self.inputs, amqp_url), name=f"Evaluator-{i}")
            proc.start()
            self.logger.info(f"Started Evaluator Process {i} with PID: {proc.pid}")
            self.evaluator_processes.append(proc)

        for i in range(self.config.num_pdb - 1):
            proc = mp.Process(target=self.database_process, args=(self.config.programs_database, template, function_to_evolve, amqp_url), name=f"Database-{i}")
            proc.start()
            self.logger.info(f"Started Database Process {i} with PID: {proc.pid}")
            self.database_processes.append(proc)

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
                    "sampler_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared sampler_queue.")

                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared evaluator_queue.")

                sampler_instance = llama_grid.Sampler(
                    connection, channel, sampler_queue, evaluator_queue, self.config, device
                )
                self.logger.debug(f"Sampler {local_id}: Initialized Sampler instance.")

                sampler_task = asyncio.create_task(sampler_instance.consume_and_process())
                await sampler_task
            except asyncio.CancelledError:
                self.logger.info(f"Sampler {local_id}: Process was cancelled.")
            except Exception as e:
                self.logger.error(f"Sampler {local_id} encountered an error: {e}")
            finally:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                self.logger.debug(f"Sampler {local_id}: Connection closed.")

        try:
            loop.run_until_complete(run_sampler())
        except Exception as e:
            self.logger.info(f"Sampler process {local_id}: Exception occurred: {e}")
        finally:
            loop.close()
            self.logger.debug(f"Sampler process {local_id} has been closed gracefully.")


    def database_process(self, config, template, function_to_evolve, amqp_url):
        local_id = mp.current_process().pid
        # Initialize these variables at a higher scope to be accessible in signal_handler
        connection = None
        channel = None
        database_task = None

        async def graceful_shutdown(loop, connection, channel, task):
            logger.info(f"Database process {local_id}: Initiating graceful shutdown...")

            if task:
                logger.info("Waiting for the current database task to finish...")
                try:
                    await asyncio.wait_for(task, timeout=10)  # Timeout in case the task takes too long
                except asyncio.TimeoutError:
                    logger.warning("Database task took too long. Cancelling...")
                    task.cancel()
                    await task  # Ensure the task is cancelled

            if channel:
                await channel.close()
            if connection:
                await connection.close()

            loop.stop()  # Stop the event loop after cleanup
            logger.info(f"Database process {local_id} has been shut down gracefully.")

        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}. Initiating graceful shutdown.")
            loop.create_task(graceful_shutdown(loop, connection, channel, database_task))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.add_signal_handler(signal.SIGINT, signal_handler, loop)
        loop.add_signal_handler(signal.SIGTERM, signal_handler, loop)

        async def run_database():
            nonlocal connection, channel, database_task  # Access the outer-scoped variables
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                channel = await connection.channel()
                database_queue = await channel.declare_queue("database_queue", durable=False, auto_delete=True, arguments={'x-consumer-timeout': 360000000})
                sampler_queue = await channel.declare_queue("sampler_queue", durable=False, auto_delete=True, arguments={'x-consumer-timeout': 360000000})
                evaluator_queue = await channel.declare_queue("evaluator_queue", durable=False, auto_delete=True, arguments={'x-consumer-timeout': 360000000})

                database_instance = programs_database.ProgramsDatabase(
                    self.manager, self.mang, connection, channel, database_queue, sampler_queue, evaluator_queue, config, template, function_to_evolve
                )
                database_task = asyncio.create_task(database_instance.consume_and_process())
                await database_task  # Await the database task
            except Exception as e:
                self.logger.info(f"Database process {local_id} encountered an error: {e}")
            finally:
                pass  # Cleanup will occur in graceful_shutdown
        try:
            loop.run_until_complete(run_database())
        except Exception as e:
            self.logger.info(f"Database process {local_id}: Exception occurred: {e}")
        finally:
            loop.close()
            self.logger.debug(f"Database process {local_id} has been closed gracefully.")

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
                    self.logger.error(f"Evaluator {local_id}: Error during task cancellation: {e}")

            if channel:
                try:
                    await channel.close()
                except Exception as e:
                    self.logger.error(f"Evaluator {local_id}: Error closing channel: {e}")
            
            if connection:
                try:
                    await connection.close()
                except Exception as e:
                    self.logger.error(f"Evaluator {local_id}: Error closing connection: {e}")

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
                    "evaluator_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )
                database_queue = await channel.declare_queue(
                    "database_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )

                evaluator_instance = evaluator.Evaluator(
                    connection, channel, evaluator_queue, database_queue, 
                    template, 'priority', 'evaluate', inputs, 'sandboxstorage', 
                    timeout_seconds=600, local_id=local_id
                )
                evaluator_task = asyncio.create_task(evaluator_instance.consume_and_process())
                await evaluator_task
            except asyncio.CancelledError:
                self.logger.info(f"Evaluator {local_id}: Process was cancelled.")
            except Exception as e:
                self.logger.error(f"Evaluator {local_id}: Error occurred: {e}")
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
            self.logger.info(f"Evaluator process {local_id}: Exception occurred: {e}")
        finally:
            loop.close()
            self.logger.debug(f"Evaluator process {local_id} has been closed gracefully.")



    async def run(self):
        try:
            await self.main_task()
        except Exception as e:
            self.logger.error(f"Main operation error occurred: {e}.")


def initialize_task_manager(config=None):
    if config is None:
        config = config_lib.Config()
    # Load the specification file from the current working directory
    with open(os.path.join(os.getcwd(), 'specification_instruct.txt'), 'r') as file:
        specification = file.read()

    mang = Manager()
    manager = CustomManager()

    # Register the classes before the manager is started
    manager.register('Island', programs_database.Island, programs_database.IslandProxy)
    manager.register('Cluster', programs_database.Cluster, programs_database.ClusterProxy)
    manager.start()

    # Create the TaskManager instance
    task_manager = TaskManager(specification, [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)], config, mang, manager)

    return task_manager


if __name__ == "__main__":
    # Initialize TaskManager using the defined function
    task_manager = initialize_task_manager()

    async def main():
        # Create a task to run the TaskManager
        task = asyncio.create_task(task_manager.run())
        await task  # Wait for the task to complete

    # Run the main function inside the asyncio event loop
    asyncio.run(main())