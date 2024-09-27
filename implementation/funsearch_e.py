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
import starcoder
import code_manipulation
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
import copy
import psutil 
import GPUtil 
from typing import Sequence, Any
import threading
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomManager(BaseManager):
    pass

class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config: config_lib.Config, mang, manager):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.mang = mang
        self.manager = manager
        self.logger = self.initialize_logger()
        self.shared_id = self.mang.Value('i', 0)
        self.shared_lock = self.mang.Lock()
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes= []
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


    async def adjust_consumers(self, channel, queue_name, target_fnc, processes, *args, max_consumers=25, min_consumers=1, sleep=300, sleep_after_scale=80, sleep_after_rsfull=1200, threshold=5):
        while True:
            queue = await channel.declare_queue(queue_name, durable=False, auto_delete=True)
            message_count = queue.declaration_result.message_count
            consumer_count = queue.declaration_result.consumer_count

            # Scale up only if message count is above the threshold and consumer count is below the max
            if message_count > threshold and consumer_count < max_consumers:
                # CPU check for evaluator_queue
                if queue_name == "evaluator_queue":
                    cpu_affinity = os.sched_getaffinity(0)  # CPUs available to the container (e.g., {1, 2, 10, 12})
                    cpu_usage = psutil.cpu_percent(percpu=True)  # Gets the usage for all system CPUs
                    container_cpu_usage = [cpu_usage[i] for i in cpu_affinity]

                    # Count how many of the available CPUs are under 60% usage
                    available_cpus_with_low_usage = sum(1 for usage in container_cpu_usage if usage < 60)
                    self.logger.info(f"Available CPUs with <60% usage (in container): {available_cpus_with_low_usage}")

                    # Scale up only if more than 4 CPUs have less than 60% usage
                    if available_cpus_with_low_usage <= 4:
                        self.logger.info(f"Cannot scale up {target_fnc}: Not enough available CPU resources.")
                        await asyncio.sleep(sleep_after_rsfull)
                        continue

                # GPU check for sampler_queue
                if queue_name == "sampler_queue":
                    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                    if visible_devices and visible_devices[0]:
                        visible_devices = [int(dev.strip()) for dev in visible_devices]

                    gpus = GPUtil.getGPUs()
                    filtered_gpus = [gpu for gpu in gpus if gpu.id in visible_devices]
                    total_available_gpu_memory = sum(gpu.memoryFree for gpu in filtered_gpus)
                    self.logger.info(f"Total available GPU memory (in container): {total_available_gpu_memory / 1024:.2f} GB")

                    # Scale up only if the total available GPU memory is more than 38 GB
                    if total_available_gpu_memory < 38 * 1024:
                        self.logger.info(f"Cannot scale up {target_fnc}: Not enough GPU memory resources.")
                        await asyncio.sleep(sleep_after_rsfull)
                        continue

                # If resource checks passed, scale up
                try:
                    if self.shared_lock.acquire(timeout=10):  # Acquire lock with timeout
                        try:
                            proc = mp.Process(target=target_fnc, args=args)
                            self.logger.info(f"Scaling up on {target_fnc} ...")
                            proc.start()
                            processes.append(proc)
                        finally:
                            self.shared_lock.release()  # Always release lock after scaling
                    else:
                        self.logger.warning("Failed to acquire lock within 10 seconds for scaling up")
                except Exception as e:
                    self.logger.error(f"Exception while scaling up: {e}")

                await asyncio.sleep(sleep_after_scale)
                continue

            # Scale down if message count is below the threshold and there are more consumers than the minimum
            if message_count < threshold and consumer_count > min_consumers:
                try:
                    if self.shared_lock.acquire(timeout=10):  # Acquire lock with timeout
                        try:
                            if processes:
                                proc = processes.pop()
                                proc.terminate()
                                proc.join()  # Wait for it to fully terminate
                                self.logger.info(f"Scaled down on {target_fnc}, process {proc.pid} terminated.")
                        finally:
                            self.shared_lock.release()  # Always release lock after scaling
                    else:
                        self.logger.warning("Failed to acquire lock within 10 seconds for scaling down")
                except Exception as e:
                    self.logger.error(f"Exception while scaling down: {e}")

                await asyncio.sleep(sleep_after_scale)
                continue

            # Sleep before checking again
            await asyncio.sleep(sleep)


    def schedule_consumer_adjustments(self, template, function_to_evolve, amqp_url): # Runs in the background
        asyncio.create_task(self.adjust_consumers(self.sampler_channel, "evaluator_queue", self.evaluator_process, self.evaluator_processes, template, self.inputs, amqp_url, max_consumers=80, min_consumers=1, sleep=200, sleep_after_scale=120, sleep_after_rsfull=1200, threshold=5))

    async def main_task(self):
        amqp_url = URL(f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/').update_query(heartbeat=180000)
        pid = os.getpid()
        self.logger.info(f"main_task is running in process with PID: {pid}")
        try:
            # Create connections for the samplers and database
            sampler_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
                client_properties={"connection_attempts": 3, "retry_delay": 5},
                reconnect_interval=5
            )   


            # Channels on separate connections
            self.sampler_channel = await sampler_connection.channel()

            template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'
        
            self.schedule_consumer_adjustments(template, function_to_evolve, amqp_url) 
                
            # Initialize evaluator instances in separate processes
            for i in range(self.config.num_evaluators):
                proc = mp.Process(target=self.evaluator_process, args=(template, self.inputs, amqp_url), name=f"Evaluator-{i}")
                proc.start()
                self.logger.info(f"Started Evaluator Process {i} with PID: {proc.pid}")
                with self.shared_lock:
                    self.evaluator_processes.append(proc)
            
        except Exception as e:
            self.logger.info(f"Exception occurred in evaluator_process: {e}")

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
                    await asyncio.wait_for(evaluator_task, timeout=10)
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
                    client_properties={"connection_attempts": 3, "retry_delay": 5},
                    reconnect_interval=5
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
            self.logger.info(f"Main operation error occured: {e}.")

def initialize_task_manager(config=None):
    if config is None:
        config = config_lib.Config()
    # Load the specification file from the current working directory
    with open(os.path.join(os.getcwd(), 'specification.txt'), 'r') as file:
        specification = file.read()

    mang = Manager()
    manager = CustomManager()
    
    # Register the classes before the manager is started
    manager.register('Island', programs_database.Island, programs_database.IslandProxy)
    manager.register('Cluster', programs_database.Cluster, programs_database.ClusterProxy)
    manager.start()
    
    # Create the TaskManager instance
    task_manager = TaskManager(specification, [(6,1), (7,1), (8,1), (9,1), (10,1), (11,1)], config, mang, manager)
    
    return task_manager




if __name__ == "__main__":
    # Initialize TaskManager using the defined function
    task_manager = initialize_task_manager()

    # Run the TaskManager's async run method
    asyncio.run(task_manager.run())
