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
import gpt
import code_manipulation
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
import copy
import psutil # for checking CPU utilization.
import GPUtil # for checking GPU memory usage.
from typing import Sequence, Any
import threading

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomManager(BaseManager):
    pass

def save_checkpoint(main_database):
    timestamp = int(time.time())  # Gets the current time as an integer timestamp
    filepath = os.path.join(os.getcwd(), f"checkpoint_{timestamp}.pkl")  # Creates a file name with the timestamp
    data = main_database.serialize_checkpoint()
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config: config_lib.Config):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger()
        self.shared_id = mang.Value('i', 0)
        self.shared_lock = mang.Lock()
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


    async def adjust_consumers(self, channel, queue_name, target_fnc, processes, *args, max_consumers=80, min_consumers=1, threshold=5):
        while True:
            queue = await channel.declare_queue(queue_name, durable=False, auto_delete=True)
            message_count = queue.declaration_result.message_count
            consumer_count = queue.declaration_result.consumer_count

            # Scale up only if message count is above the threshold and consumer count is below the max
            if message_count > threshold and consumer_count < max_consumers:
                # CPU check for evaluator_queue
                if queue_name == "evaluator_queue":
                    cpu_affinity = os.sched_getaffinity(0)  # CPUs available to the container (e.g., {1, 2, 10, 12})
                    self.logger.debug(f"cpu_affinity is {cpu_affinity}")
                    cpu_usage = psutil.cpu_percent(percpu=True)  # Gets the usage for all system CPUs
                    container_cpu_usage = [cpu_usage[i] for i in cpu_affinity]

                    # Count how many of the available CPUs are under 50% usage
                    available_cpus_with_low_usage = sum(1 for usage in container_cpu_usage if usage < 50)
                    self.logger.info(f"Available CPUs with <50% usage (in container): {available_cpus_with_low_usage}")

                    # Scale up only if more than 3 CPUs have less than 50% usage
                    if available_cpus_with_low_usage <= 4:
                        self.logger.info(f"Cannot scale up {target_fnc}: Not enough available CPU resources.")
                        await asyncio.sleep(60)
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
                        await asyncio.sleep(60)
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

                await asyncio.sleep(60)
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

                await asyncio.sleep(60)
                continue

            # Sleep for 10 minutes before checking again
            await asyncio.sleep(120)


    async def periodic_checkpoint(self, main_database):
        checkpoint_interval = 600  # 10 min
        while True:
            await asyncio.sleep(checkpoint_interval)  # Non-blocking sleep
            save_checkpoint(main_database)  # Directly save the checkpoint
            self.logger.info("Checkpoint has been saved.")

    def start_periodic_checkpoint_thread(self, main_database):
        asyncio.create_task(self.periodic_checkpoint(main_database))


    async def main_task(self):
        amqp_url = URL(f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/').update_query(heartbeat=2800)
        try:
            # Create connections for the samplers and database
            sampler_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
                client_properties={"connection_attempts": 3, "retry_delay": 5},
                reconnect_interval=5
            )   

            database_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
                client_properties={"connection_attempts": 3, "retry_delay": 5},
                reconnect_interval=5
            )   

            # Channels on separate connections
            self.sampler_channel = await sampler_connection.channel()
            self.database_channel = await database_connection.channel()

            evaluator_queue = await self.sampler_channel.declare_queue("evaluator_queue", durable=False, auto_delete=True)
            sampler_queue = await self.sampler_channel.declare_queue("sampler_queue", durable=False, auto_delete=True)
            database_queue = await self.database_channel.declare_queue("database_queue", durable=False, auto_delete=True)

            template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'

        
            # Initialize the designated ProgramsDatabase instance for checkpointing
            main_database = programs_database.ProgramsDatabase(
                manager, mang, database_connection, self.database_channel, database_queue,
                sampler_queue, evaluator_queue, self.config.programs_database, template, function_to_evolve
            )

            main_database_task = asyncio.create_task(main_database.consume_and_process())
            self.start_periodic_checkpoint_thread(main_database)
                
            # Initialize sampler instances in separate processes
            for _ in range(self.config.num_samplers):
                proc = mp.Process(target=self.sampler_process, args=(amqp_url,))
                proc.start()
                with self.shared_lock:
                    self.sampler_processes.append(proc)

            # Initialize evaluator instances in separate processes
            for _ in range(self.config.num_evaluators):
                proc = mp.Process(target=self.evaluator_process, args=(template, self.inputs, amqp_url))
                proc.start()
                with self.shared_lock:
                    self.evaluator_processes.append(proc)

            # Initialize database instances in separate processes
            for _ in range(self.config.num_pdb - 1):  
                proc = mp.Process(target=self.database_process, args=(self.config.programs_database, template, function_to_evolve, amqp_url))
                proc.start()
                with self.shared_lock:
                    self.database_processes.append(proc)  # Maintain a list of database processes

            adjust_eval_consumers_task = asyncio.create_task(
                self.adjust_consumers(
                    self.sampler_channel, "evaluator_queue", self.evaluator_process, 
                    self.evaluator_processes, template, self.inputs, amqp_url,
                    max_consumers=80, min_consumers=1, threshold=5
                )
            )

            adjust_sampler_consumers_task = asyncio.create_task(
                self.adjust_consumers(
                    self.sampler_channel, "sampler_queue", self.sampler_process, 
                    self.sampler_processes, amqp_url,
                    max_consumers=5, min_consumers=1, threshold=5
                )
            )

            adjust_db_consumers_task = asyncio.create_task(
                self.adjust_consumers(
                    self.database_channel, "database_queue", self.database_process, 
                    self.database_processes, self.config.programs_database, 
                    template, function_to_evolve, amqp_url,
                    max_consumers=5, min_consumers=0, threshold=5
                )
            )
            self.tasks = [ main_database_task, adjust_eval_consumers_task, adjust_sampler_consumers_task, adjust_db_consumers_task]
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
            self.logger.info(f"Exception occurred in evaluator_process: {e}")

    def sampler_process(self, amqp_url):
        local_id = mp.current_process().pid  # Use process ID as a local identifier

        # Signal handler to close connections and exit gracefully
        def signal_handler(sig, frame):
            if 'connection' in locals() and connection:
                loop.run_until_complete(connection.close())
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        async def run_sampler():
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 3, "retry_delay": 5},
                    reconnect_interval=5
                )
                channel = await connection.channel()
                sampler_queue = await channel.declare_queue("sampler_queue", durable=False, auto_delete=True)
                evaluator_queue = await channel.declare_queue("evaluator_queue", durable=False, auto_delete=True)

                sampler_instance = gpt.Sampler(connection, channel, sampler_queue, evaluator_queue, self.config)
                sampler_task = asyncio.create_task(sampler_instance.consume_and_process())
                await sampler_task
            except Exception as e:
                self.logger.info(f"Database process {local_id} encountered an error: {e}")
            finally:
                if connection:
                    await connection.close()
                    self.logger.debug(f"Database process {local_id} connection closed.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_sampler())


    def database_process(self, config, template, function_to_evolve, amqp_url):
        local_id = mp.current_process().pid  # Use process ID as a local identifier

        # Signal handler to close connections and exit gracefully
        def signal_handler(sig, frame):
            if 'connection' in locals() and connection:
                loop.run_until_complete(connection.close())
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        async def run_database():
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 3, "retry_delay": 5},
                    reconnect_interval=5
                )
                channel = await connection.channel()
                database_queue = await channel.declare_queue("database_queue", durable=False, auto_delete=True)
                sampler_queue = await channel.declare_queue("sampler_queue", durable=False, auto_delete=True)
                evaluator_queue = await channel.declare_queue("evaluator_queue", durable=False, auto_delete=True)

                database_instance = programs_database.ProgramsDatabase(
                    manager, mang,  connection, channel, database_queue, sampler_queue, evaluator_queue, config, template, function_to_evolve
                )
                database_task = asyncio.create_task(database_instance.consume_and_process())
                await database_task
            except Exception as e:
                self.logger.info(f"Database process {local_id} encountered an error: {e}")
            finally:
                if connection:
                    await connection.close()
                    self.logger.debug(f"Database process {local_id} connection closed.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_database())


    def evaluator_process(self, template, inputs, amqp_url):
        import evaluator
        local_id = mp.current_process().pid  # Use process ID as a local identifier

        # Signal handler to close connections and exit gracefully
        def signal_handler(sig, frame):
            if 'connection' in locals() and connection:
                loop.run_until_complete(connection.close())
            sys.exit(0)

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        async def run_evaluator():
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 3, "retry_delay": 5},
                    reconnect_interval=5
                )
                channel = await connection.channel()
                evaluator_queue = await channel.declare_queue("evaluator_queue", durable=False, auto_delete=True)
                database_queue = await channel.declare_queue("database_queue", durable=False, auto_delete=True)

                evaluator_instance = evaluator.Evaluator(
                    connection, channel, evaluator_queue, database_queue, template,'priority', 'evaluate', inputs, 'sandboxstorage', timeout_seconds=600, local_id=local_id
                )
                evaluator_task = asyncio.create_task(evaluator_instance.consume_and_process())
                await evaluator_task
            except Exception as e:
                self.logger.info(f"Exception occurred in evaluator_process: {e}")
            finally:
                if connection:
                    await connection.close()
                    self.logger.debug(f"Evaluator {local_id} connection closed.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_evaluator())

    async def run(self):
        try:
            await self.main_task()
        except Exception as e:
            self.logger.info(f"Main operation error occured: {e}.")

if __name__ == "__main__":
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
    task_manager = TaskManager(specification, [(6,1), (7,1), (8,1), (9,1), (10,1), (11,1)], config)
    asyncio.run(task_manager.run())
