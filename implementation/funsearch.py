import asyncio
import logging
from logging.handlers import RotatingFileHandler
from logging import FileHandler
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
    


os.environ["TOKENIZERS_PARALLELISM"] = "false"



class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config: config_lib.Config):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger()
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        self.process_to_device_map = {}  # Dictionary to track process-to-device mapping
        # Initialize pynvml only once during TaskManager initialization
        try:
            pynvml.nvmlInit()
            self.logger.info("Successfully initialized NVML for GPU monitoring.")
        except pynvml.NVMLError as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            raise

    async def log_resource_stats_periodically(self, interval=300):
        """Log available CPU and GPU memory/utilization every `interval` seconds."""
        while True:
            try:
                # Log CPU usage
                cpu_affinity = os.sched_getaffinity(0)  # Get CPUs available to the current process/container
                cpu_usage = psutil.cpu_percent(interval=None, percpu=True)  # Get usage for all system CPUs
                available_cpu_usage = [cpu_usage[i] for i in cpu_affinity]  # Filter for CPUs available to the process
                avg_cpu_usage = sum(available_cpu_usage) / len(available_cpu_usage)  # Calculate the average CPU usage
                self.logger.info(f"Available CPUs: {len(cpu_affinity)}, Average CPU Usage: {avg_cpu_usage:.2f}%")

                # Log GPU usage
                device_count = pynvml.nvmlDeviceGetCount()  # Get the number of available GPUs
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    free_memory_mib = memory_info.free / 1024**2  # Convert bytes to MiB
                    total_memory_mib = memory_info.total / 1024**2
                    gpu_utilization = utilization.gpu  # GPU utilization percentage
                    self.logger.info(f"GPU {i}: Free Memory = {free_memory_mib:.2f} MiB / {total_memory_mib:.2f} MiB, GPU Utilization = {gpu_utilization}%")

            except psutil.Error as e:
                self.logger.error(f"Failed to query CPU information: {e}")
            except pynvml.NVMLError as e:
                self.logger.error(f"Failed to query GPU information: {e}")
            finally:
                await asyncio.sleep(interval)  # Wait for the specified interval before checking again

    def get_process_with_zero_or_lowest_gpu(self, processes, gpu_utilization_threshold=10):
        """Find a process to terminate based on GPU utilization."""
        try:
            for proc in processes:
                try:
                    # Try to extract the GPU device from the process arguments (e.g., "cuda:0")
                    device = self.process_to_device_map.get(proc.pid)  # Look up device in the dictionary
                    if device and device != 'cuda':
                        # Extract the GPU index from the device string, e.g., "cuda:0" -> 0
                        gpu_index = int(device.split(":")[1])
                    
                        # Get GPU utilization percentage using pynvml
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # Get GPU utilization

                        self.logger.debug(f"Process PID {proc.pid} GPU utilization: {gpu_utilization}%")

                        # If GPU utilization is below the threshold, select this process for termination
                        if gpu_utilization < gpu_utilization_threshold:
                            self.logger.info(f"Process with PID {proc.pid} is using GPU {gpu_index} with utilization {gpu_utilization}%, below threshold {gpu_utilization_threshold}%.")
                            return proc
                    else:
                        self.logger.info(f"Process PID {proc.pid} does not have a GPU device argument.")
                except pynvml.NVMLError as e:
                    self.logger.warning(f"Failed to get GPU utilization for process PID {proc.pid}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error checking GPU utilization for process PID {proc.pid}: {e}")
                    continue

            # If no GPU-based process has utilization below the threshold, return None
            self.logger.info("No suitable GPU-based process found with GPU utilization below threshold.")
            return None

        except Exception as e:
            self.logger.error(f"Error occurred while checking GPU utilization: {e}")
            # Fall back to checking CPU usage if an error occurs
            self.logger.info("Falling back to checking CPU utilization.")
            return self.get_process_with_zero_or_lowest_cpu(processes)

    def initialize_logger(self):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.DEBUG)
        log_file_path = os.path.join(os.getcwd(), 'funsearch.log')
        # Use FileHandler instead of RotatingFileHandler
        handler = FileHandler(log_file_path)
        # Optional: If you want to set a file mode to append to the log instead of overwriting it
        # handler = FileHandler(log_file_path, mode='a')  # 'a' for append new log messages, 'w' for overwrite new log messages when the logger is newely initialized
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
        check_interval_sam = 120
        max_evaluators = 120
        min_evaluators = 1
        max_samplers = 4
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
                # Collect metrics from queues
                evaluator_message_count = await self.get_queue_message_count(channel, "evaluator_queue")
                sampler_message_count = await self.get_queue_message_count(channel, "sampler_queue")

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
            if current_processes > num_processes:
                self.logger.info(f"Scaled up {process_name} processes to {current_processes}")
            else:
                self.logger.info(f"Could not scale up {process_name} processes; still at {current_processes}")

        elif message_count < threshold and num_processes > min_processes:
            # Scale down
            self.terminate_process(processes, process_name)
            current_processes = len(processes)
            if current_processes < num_processes:
                self.logger.info(f"Scaled down {process_name} processes to {current_processes}")
            else: 
                self.logger.info(f"Could not scale down {process_name} processes; still at {current_processes}")
        else: 
            self.logger.info(f"No scaling action needed for {process_name}. Current processes: {num_processes}, Message count: {message_count}")
            return 


    def start_process(self, target_fnc, args, processes, process_name):
        current_pid = os.getpid()
        current_thread = threading.current_thread().name
        thread_id = threading.get_ident()

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

        # GPU check for sampler processes
        if target_fnc == self.sampler_process:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]  # Ensure non-empty strings and convert to int

            # Initialize GPU memory info and utilization
            gpu_memory_info = {}

            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError as e:
                self.logger.error(f"Failed to initialize NVML: {e}")
                device = 'cuda'
                self.logger.warning(f"Proceeding with device=cuda for {process_name}.")
                args += (device,)
                # Start the process
                proc = mp.Process(target=target_fnc, args=args, name=f"{process_name}-{len(processes)}")
                proc.start()
                processes.append(proc)
                return

            for dev_id in visible_devices:
                try:
                    handle = None
                    if isinstance(dev_id, int):  # Check if dev_id is an integer (regular GPU)
                        dev_int = int(dev_id)
                        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_int)
                        # Create mapping for regular GPU device indices
                        id_to_container_index = {dev: idx for idx, dev in enumerate(visible_devices) if isinstance(dev, int)}
                    else:
                        handle = pynvml.nvmlDeviceGetHandleByUUID(dev_id)

                except (ValueError, pynvml.NVMLError) as e:
                    self.logger.error(f"Error getting handle for device ID {dev_id}: {e}")
                    continue

                if handle:
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        free_memory_mib = memory_info.free / 1024**2  # Convert bytes to MiB
                        gpu_utilization = utilization.gpu  # Percentage
                        gpu_memory_info[dev_id] = (free_memory_mib, gpu_utilization)
                    except pynvml.NVMLError as e:
                        self.logger.error(f"Error querying memory for device {dev_id}: {e}")
                        gpu_memory_info[dev_id] = (None, None)  # Set to None on error

            suitable_gpu_id = None
            combined_memory = 0
            combined_gpus = []

            # Check if any single GPU has >= 20 GiB of memory free and < 50% utilization
            for dev_id, (free_memory, utilization) in gpu_memory_info.items():
                if free_memory is None and utilization is None:
                    self.logger.warning(f"Memory information could not be queried for device {dev_id}. Skipping this device.")
                    continue  # Skip this device and continue checking others
                if free_memory > 32768 and utilization < 200: #need more than 32 GiB for starcoder too fit
                    suitable_gpu_id = dev_id
                    break
                elif utilization < 200:
                    combined_memory += free_memory if free_memory else 0
                    combined_gpus.append(dev_id)

            # Assign device based on availability
            if suitable_gpu_id is not None and isinstance(suitable_gpu_id, int): 
                container_index = id_to_container_index[suitable_gpu_id]
                device = f"cuda:{container_index}"
            elif combined_memory >= 32768:
                device = 'cuda'  # Use multiple GPUs
                self.logger.info(f"Using combination of GPUs: {combined_gpus} with total memory: {combined_memory} MiB")
            elif free_memory is None and utilization is None: # When no memory and utilization could be queried 
                device='cuda'
                self.logger.warning(f"Memory information could not be queried for device {dev_id}.")
            else:
                self.logger.warning(f"Not enough GPU memory available for {process_name}, no scaling.")
                return

            self.logger.info(f"Assigning {process_name} to device {device if device else 'combined GPUs (cuda)'}")

            args += (device,)  # Append the device to args

        # Start the process
        proc = mp.Process(target=target_fnc, args=args, name=f"{process_name}-{len(processes)}")
        proc.start()
        processes.append(proc)
        # Store the process PID and device in the map only for sampler processes
        if target_fnc == self.sampler_process:
            self.process_to_device_map[proc.pid] = device


    def get_process_with_zero_or_lowest_cpu(self, processes, cpu_utilization_threshold=20):
        """Find a process to terminate based on CPU utilization."""
        for proc in processes:
            try:
                p = psutil.Process(proc.pid)
                cpu_usage = p.cpu_percent(interval=1)
                self.logger.debug(f"Process PID {proc.pid} CPU utilization: {cpu_usage}%")

                # If CPU utilization is below the threshold, select this process for termination
                if cpu_usage < cpu_utilization_threshold:
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.logger.warning(f"Failed to access CPU usage for process PID {proc.pid}. It might have finished or access was denied.")
                continue

        # If no process meets the threshold, return None
        self.logger.info(f"No process with CPU utilization below {cpu_utilization_threshold}% found.")
        return None

    def terminate_process(self, processes, process_name):
        if processes:
            # Try to get the least busy process
            if process_name.startswith('Evaluator'):
                least_busy_process = self.get_process_with_zero_or_lowest_cpu(processes)
            elif process_name.startswith('Sampler'): 
                least_busy_process = self.get_process_with_zero_or_lowest_gpu(processes)
            else: 
                self.logger.info(f"No Sampler or Evaluator process is {process_name}")
                return 
            self.logger.info(f"least_busy_process is {least_busy_process}")
            if least_busy_process is None:
                return
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


    async def main_task(self, checkpoint_file=None):
        #logger_coroutine = asyncio.create_task(self.log_tasks())
        amqp_url = URL(
            f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/'
        ).update_query(heartbeat=180000)
        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}")
        resource_logging_task = asyncio.create_task(self.log_resource_stats_periodically(interval=200))

        # Initialize the template and initial program data
        template = code_manipulation.text_to_program(self.specification)
        function_to_evolve = 'priority'
        if checkpoint_file is None: 
            initial_program_data = json.dumps({
                "sample": template.get_function(function_to_evolve).body,
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
                database_connection, self.database_channel, database_queue, sampler_queue, evaluator_queue, self.config.programs_database, template, function_to_evolve, checkpoint_file
            )

            database_task = asyncio.create_task(database.consume_and_process())
            checkpoint_task = asyncio.create_task(database.periodic_checkpoint())

            # Start consumers before publishing
            try:
                self.start_initial_processes(template, function_to_evolve, amqp_url, checkpoint_file)
                self.logger.info("Initial processes started successfully.")
            except Exception as e:
                self.logger.error(f"Failed to start initial processes: {e}")


            # Publish the initial program with retry logic
            if checkpoint_file is None:
                await asyncio.sleep(30)  # Delay to ensure the sampler process is ready
                await self.publish_initial_program_with_retry(amqp_url, initial_program_data)
            else: 
                await asyncio.sleep(90)  # Delay to ensure the sampler process is ready
                await database.get_prompt()
                self.logger.info(f"Loading from checkpoint: {checkpoint_file}")

            scaling_controller_task = asyncio.create_task(self.scaling_controller(template, function_to_evolve, amqp_url))
    
            self.tasks = [database_task, scaling_controller_task, checkpoint_task, resource_logging_task]
            self.channels = [self.database_channel, self.sampler_channel]
            self.queues = [["database_queue"], ["sampler_queue"], ["evaluator_queue"]]

            # Use asyncio.gather to run tasks concurrently and catch exceptions
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")


    def start_initial_processes(self, template, function_to_evolve, amqp_url, checkpoint_file):
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

            # Check if any single GPU has >= 20 GiB of memory free and < 50% utilization
            for gpu_id, (free_memory, utilization) in gpu_memory_info.items():
                if free_memory > 17000 and utilization < 200:  # Check for at least 20 GiB and < 50% utilization
                    suitable_gpu_id = gpu_id
                    break
                elif utilization < 200:
                    combined_memory += free_memory
                    combined_gpus.append(gpu_id)

            # If a single GPU was found with sufficient memory
            if suitable_gpu_id is not None:
                container_index = id_to_container_index[suitable_gpu_id]
                device = f"cuda:{container_index}"
                # Adjust memory tracking (simplistic estimation)
                gpu_memory_info[suitable_gpu_id] = (gpu_memory_info[suitable_gpu_id][0] - 17000, gpu_memory_info[suitable_gpu_id][1])
            elif combined_memory >= 17000:  # If combined memory from multiple GPUs is >= 20 GiB
                device = None  # Use None to indicate that multiple GPUs will be used
                self.logger.info(f"Using combination of GPUs: {combined_gpus} with total memory: {combined_memory} MiB")
            else:
                self.logger.error(f"Cannot start sampler {i}: Not enough available GPU memory.")
                continue  # Skip this sampler if no GPU has sufficient memory

            self.logger.info(f"Assigning sampler {i} to device {device if device else 'combined GPUs (None)'}")

            proc = mp.Process(target=self.sampler_process, args=(amqp_url, device), name=f"Sampler-{i}")
            proc.start()
            self.logger.debug(f"Started Sampler Process {i} on {device} with PID: {proc.pid}")
            self.sampler_processes.append(proc)
            # Store the process PID and device in the process_to_device_map
            self.process_to_device_map[proc.pid] = device

        # Start initial evaluator processes
        for i in range(self.config.num_evaluators):
            proc = mp.Process(target=self.evaluator_process, args=(template, self.inputs, amqp_url), name=f"Evaluator-{i}")
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
                    "evaluator_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                database_queue = await channel.declare_queue(
                    "database_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )

                evaluator_instance = evaluator.Evaluator(
                    connection, channel, evaluator_queue, database_queue, 
                    template, 'priority', 'evaluate', inputs, 'sandboxstorage', 
                    timeout_seconds=800, local_id=local_id
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



if __name__ == "__main__":
    async def main():
        # Initialize configuration
        config = config_lib.Config()

        # Load the specification
        spec_path = os.path.join(os.getcwd(), 'specification.txt')
        with open(spec_path, 'r') as file:
            specification = file.read()

        inputs = [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]

        # Assuming task_manager is already initialized somewhere
        task_manager = TaskManager(specification=specification, inputs=inputs, config=config)

        # Start the main task
        task = asyncio.create_task(task_manager.main_task('/franziska/implementation/Checkpoints/checkpoint_2024-10-22_08-05-11.pkl'))  # Provide checkpoint file if needed

        # Await the task to run it
        await task

    # Top-level call to asyncio.run() to start the event loop
    asyncio.run(main())


