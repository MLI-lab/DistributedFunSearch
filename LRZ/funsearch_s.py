import asyncio
import logging
from logging.handlers import RotatingFileHandler
from logging import FileHandler
import json
import argparse
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
import sampler
import code_manipulation
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
import socket
import pynvml
import argparse



os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        return f"Error fetching IP address: {e}"



class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config: config_lib.Config, check_interval_sam):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger()
        self.check_interval_sam = check_interval_sam
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        self.process_to_device_map = {}

    def initialize_logger(self):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)
        log_file_path = os.path.join(os.getcwd(), 'sampler.log')
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
        max_samplers = 4
        min_samplers = 1
        sampler_threshold = 15
        initial_sleep_duration = 300  # seconds
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
                sampler_message_count = await self.get_queue_message_count(channel, "sampler_queue")

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


            await asyncio.sleep(self.check_interval_sam)  # Non-blocking sleep

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


    def start_process(self, target_fnc, args, processes, process_name, memory_threshold=32768):
        current_pid = os.getpid()
        current_thread = threading.current_thread().name
        thread_id = threading.get_ident()
        self.logger.info(f"start_process: Process ID: {current_pid}, Thread: {current_thread}, Thread ID: {thread_id}")

        # GPU check for sampler processes
        if target_fnc == self.sampler_process:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]  # Ensure non-empty strings and convert to int

            gpus = GPUtil.getGPUs()

            # Create a mapping of host-visible GPU IDs (integers in visible_devices) to container-visible device indices
            id_to_container_index = {visible_devices[i]: i for i in range(len(visible_devices))}

            # Check if a single GPU has enough free memory
            suitable_gpu_id = None
            for gpu in gpus:
                if gpu.id in visible_devices and gpu.memoryFree > memory_threshold:
                    suitable_gpu_id = gpu.id
                    break

            if suitable_gpu_id is not None:
                # Use the selected GPU if it has enough free memory
                container_index = id_to_container_index[suitable_gpu_id]  # Get container-visible index
                device = f"cuda:{container_index}"
                self.logger.info(f"Assigning {process_name} to device {device} with {gpus[suitable_gpu_id].memoryFree} MiB available.")
            else:
                # If no single GPU has enough memory, check if the combined memory is enough
                total_free_memory = sum(gpu.memoryFree for gpu in gpus if gpu.id in visible_devices)
                if total_free_memory > memory_threshold:
                    # If the combined memory is sufficient, use all available GPUs (multi-GPU setup)
                    device = "cuda"
                    self.logger.info(f"Assigning {process_name} to device {device} with total combined memory {total_free_memory} MiB available.")
                else:
                    # Not enough memory, log details and skip process start
                    memory_info = ", ".join([f"GPU {gpu.id}: {gpu.memoryFree} MiB free" for gpu in gpus if gpu.id in visible_devices])
                    self.logger.info(f"Cannot start {process_name}: Not enough GPU memory. Combined memory available: {total_free_memory} MiB. Details: {memory_info}")
                    return

            # Append the GPU device to args
            args += (device,)

        # Start the process
        try: 
            proc = mp.Process(target=target_fnc, args=args, name=f"{process_name}-{len(processes)}")
            proc.start()
            processes.append(proc)
            self.logger.info(f"Started {process_name} process with PID: {proc.pid} on device {args[-1]}")
            # Store the process PID and device in the map only for sampler processes
            if target_fnc == self.sampler_process:
                self.process_to_device_map[proc.pid] = device
        except Exception as e: 
            return


    def get_process_with_zero_or_lowest_cpu(self, processes, cpu_utilization_threshold=20):
        """Find a process to terminate based on CPU utilization."""
        # Check each process for CPU usage below the threshold
        for proc in processes:
            try:
                p = psutil.Process(proc.pid)
                cpu_usage = p.cpu_percent(interval=1)
                self.logger.debug(f"Process PID {proc.pid} CPU utilization: {cpu_usage}%")

                # If CPU utilization is below the threshold, select this process for termination
                if cpu_usage < cpu_utilization_threshold:
                    self.logger.info(f"Process with PID {proc.pid} has CPU utilization {cpu_usage}%, below threshold {cpu_utilization_threshold}%.")
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.logger.warning(f"Failed to access CPU usage for process PID {proc.pid}. It might have finished or access was denied.")
                continue

        # If no process meets the threshold, return None
        self.logger.info(f"No process with CPU utilization below {cpu_utilization_threshold}% found.")
        return None


    def get_process_to_terminate_based_on_gpu(self, processes, gpu_utilization_threshold=10):
        """Find a process to terminate based on GPU utilization."""
        try:
            for proc in processes:
                try:
                    # Try to extract the GPU device from the process arguments (e.g., "cuda:0")
                    device = self.process_to_device_map.get(proc.pid)
                    if device and device != 'cuda':
                        # Extract the GPU index from the device string, e.g., "cuda:0" -> 0
                        gpu_index = int(device.split(":")[1])
                    
                        # Get GPU utilization percentage
                        gpus = GPUtil.getGPUs()
                        gpu = next((gpu for gpu in gpus if gpu.id == gpu_index), None)
                        if gpu is not None:
                            gpu_utilization = gpu.load * 100  # Convert from fraction to percentage
                            self.logger.debug(f"Process PID {proc.pid} GPU utilization: {gpu_utilization}%")

                            # If GPU utilization is below the threshold, select this process for termination
                            if gpu_utilization < gpu_utilization_threshold:
                                self.logger.info(f"Process with PID {proc.pid} is using GPU {gpu_index} with utilization {gpu_utilization}%, below threshold {gpu_utilization_threshold}%.")
                                return proc
                        else:
                            self.logger.warning(f"GPU with index {gpu_index} not found.")
                    else:
                        self.logger.info(f"Process PID {proc.pid} does not have a GPU device argument.")
                except Exception as e:
                    self.logger.warning(f"Failed to check GPU utilization for process PID {proc.pid}: {e}")
                    continue

            # If no GPU-based process has utilization below the threshold, return None
            self.logger.info("No suitable GPU-based process found with GPU utilization below threshold.")
            return None

        except Exception as e:
            self.logger.error(f"Error occurred while checking GPU utilization: {e}")
            # Fall back to checking CPU usage if an error occurs
            self.logger.info("Falling back to checking CPU utilization.")
            return self.get_process_with_zero_or_lowest_cpu(processes)


    def terminate_process(self, processes, process_name):
        if processes:
            # Try to get the least busy process
            least_busy_process = self.get_process_to_terminate_based_on_gpu(processes)
            self.logger.info(f"least_busy_process is {least_busy_process}")
            if least_busy_process is None:
                # If all processes are busy, pop the last process
                # least_busy_process = processes.pop()
                self.logger.info("All processes busy, no termination.")
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

    async def main_task(self):
        amqp_url = URL(
            f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/{self.config.rabbitmq.vhost}'
        ).update_query(heartbeat=180000)
        pid = os.getpid()
        ip_address = get_ip_address()
        self.logger.info(f"Main_task is running in process with PID: {pid} and on node {ip_address}")
        try:

            template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'

            scaling_controller_task = asyncio.create_task(self.scaling_controller(template, function_to_evolve, amqp_url))

            # Start initial processes
            self.start_initial_processes(template, function_to_evolve, amqp_url)
            self.tasks = [scaling_controller_task]

            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")


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
                if free_memory > 32768:  # Check if more than 17000MiB is available
                    suitable_gpu_id = gpu_id
                    break

            # Map to container-visible device (like cuda:0 or cuda:1)
            if suitable_gpu_id is not None:
                container_index = id_to_container_index[suitable_gpu_id]  # Get container-visible index
                device = f"cuda:{container_index}"
                # Adjust memory tracking (simplistic estimation)
                gpu_memory_usage[suitable_gpu_id] -= 32768
            else:
                self.logger.error(f"Cannot start sampler {i}: Not enough available GPU memory.")
                continue  # Skip this sampler if no GPU has sufficient memory

            self.logger.info(f"Assigning sampler {i} to device {device}")
            try: 
                proc = mp.Process(target=self.sampler_process, args=(amqp_url, device), name=f"Sampler-{i}")
                proc.start()
                self.logger.info(f"Started Sampler Process {i} on {device} with PID: {proc.pid}")
                self.sampler_processes.append(proc)
                # Store the process PID and device in the process_to_device_map
                self.process_to_device_map[proc.pid] = device
            except Exception as e: 
                continue


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

                sampler_instance = sampler.Sampler(
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

    async def run(self):
        try:
            await self.main_task()
        except Exception as e:
            self.logger.error(f"Main operation error occurred: {e}.")


def initialize_task_manager(config=None, check_interval_sam=200):
    if config is None:
        config = config_lib.Config()
    # Load the specification file from the current working directory
    with open(os.path.join(os.getcwd(), 'specification.txt'), 'r') as file:
        specification = file.read()


    # Create the TaskManager instance
    task_manager = TaskManager(specification, [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)], config,check_interval_sam)

    return task_manager


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the TaskManager with configurable scaling interval.")
    parser.add_argument("--check_interval_sam", type=int, default=200, help="Interval in seconds for scaling sampler processes.")
    args = parser.parse_args()

    # Initialize TaskManager using the parsed arguments
    task_manager = initialize_task_manager(check_interval_sam=args.check_interval_sam)

    async def main():
        task = asyncio.create_task(task_manager.run())
        await task

    asyncio.run(main())