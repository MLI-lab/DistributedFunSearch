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
import socket 
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
    def __init__(self, specification: str, inputs: Sequence[Any], config: config_lib.Config, check_interval_eval):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger()
        self.check_interval_eval = check_interval_eval
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None

    def initialize_logger(self):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)
        log_file_path = os.path.join(os.getcwd(), 'eval.log')
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
        max_evaluators = 200
        min_evaluators = 1
        evaluator_threshold = 5
        initial_sleep_duration = 200  # seconds
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

                # Adjust evaluator processes
                await self.adjust_processes(
                    evaluator_message_count, evaluator_threshold,
                    self.evaluator_processes, self.evaluator_process,
                    args=(template, self.inputs, amqp_url),
                    max_processes=max_evaluators, min_processes=min_evaluators,
                    process_name='Evaluator'
                )

            except Exception as e:
                self.logger.error(f"Scaling controller encountered an error: {e}")

            await asyncio.sleep(self.check_interval_eval)  # Non-blocking sleep

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

            # Schedule periodic_checkpoint and scaling_controller as async tasks
            scaling_controller_task = asyncio.create_task(self.scaling_controller(template, function_to_evolve, amqp_url))

            # Start initial processes
            self.start_initial_processes(template, function_to_evolve, amqp_url)
            self.tasks = [scaling_controller_task]

            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")


    def start_initial_processes(self, template, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)

        # Start initial evaluator and database processes as previously done
        for i in range(self.config.num_evaluators):
            proc = mp.Process(target=self.evaluator_process, args=(template, self.inputs, amqp_url), name=f"Evaluator-{i}")
            proc.start()
            self.logger.info(f"Started Evaluator Process {i} with PID: {proc.pid}")
            self.evaluator_processes.append(proc)
            
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
                    template, 'priority', 'evaluate', inputs, '/workspace/sandboxstorage/', 
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


def initialize_task_manager(config=None, check_interval_eval=200):
    if config is None:
        config = config_lib.Config()
    # Load the specification file from the current working directory
    with open(os.path.join(os.getcwd(), 'specification.txt'), 'r') as file:
        specification = file.read()

    # Create the TaskManager instance
    task_manager = TaskManager(specification, [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)], config, check_interval_eval)

    return task_manager


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the TaskManager with configurable scaling interval.")
    parser.add_argument("--check_interval_eval", type=int, default=200, help="Interval in seconds for scaling evaluator processes.")
    args = parser.parse_args()

    # Initialize TaskManager using the parsed arguments
    task_manager = initialize_task_manager(check_interval_eval=args.check_interval_eval)

    async def main():
        task = asyncio.create_task(task_manager.run())
        await task

    asyncio.run(main())