import asyncio
import logging
from logging import FileHandler
from multiprocessing import current_process
import argparse
import aio_pika
from yarl import URL
import torch.multiprocessing as mp
import os
import signal
import sys
import time
from typing import Sequence, Any
from decos.scaling_utils import ResourceManager
from decos import sampler
from decos import process_utils
import importlib
import socket
from decos import gpt


def load_config(config_path):
    """
    Dynamically load a configuration .py file from a specified path.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        return f"Error fetching IP address: {e}"

class TaskManager:
    def __init__(self, config, check_interval, log_dir):
        self.config = config
        self.logger = self.initialize_logger(log_dir)
        self.sampler_processes = []
        self.tasks = []
        if self.config.sampler.gpt: 
            self.resource_manager = ResourceManager(log_dir=log_dir, cpu_only=True)
        else: 
            self.resource_manager = ResourceManager(log_dir=log_dir)
        self.process_to_device_map = {}

    def initialize_logger(self, log_dir):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)
        os.makedirs(log_dir, exist_ok=True)
        hostname = socket.gethostname()
        log_file_name = f'sampler_{hostname}.log'
        log_file_path = os.path.join(log_dir, log_file_name)
        handler = FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    async def main_task(self, enable_scaling=True):
        """
        Main async entry point. Establishes queue connections, starts initial processes,
        and optionally starts a scaling loop from ResourceManager.
        """
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
                self.logger.info(f"Cannot connect to rabbitmq. Change config file: {e}")

        resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=60))
        self.tasks = [resource_logging_task]

        pid = os.getpid()
        self.logger.info(f"main_task is running in process PID: {pid}")

        try:
            # Start initial sampler processes
            self.start_initial_processes(amqp_url)

            # Create a connection/channel that ResourceManager can use to monitor queue sizes
            self.logger.info("Creating connection for scaling logic...")
            connection = await aio_pika.connect_robust(str(amqp_url), timeout=300)
            channel = await connection.channel()

            # Declare the sampler queue (the queue we want to scale on)
            sampler_queue = await process_utils.declare_standard_queue(channel, "sampler_queue")
            self.logger.info("sampler_queue declared for scaling logic.")

            if enable_scaling:
                scaling_task = asyncio.create_task(
                    self.resource_manager.run_scaling_loop(
                        evaluator_queue=None, 
                        sampler_queue=sampler_queue,
                        evaluator_processes=None,
                        sampler_processes=self.sampler_processes,
                        evaluator_function=None,
                        sampler_function=self.sampler_process,
                        evaluator_args=None,
                        sampler_args=(amqp_url,),
                        max_evaluators=None,
                        max_samplers=args.max_samplers,
                        check_interval=args.check_interval,
                    )
                )
                self.tasks.append(scaling_task)

            # Wait on all tasks
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception in main_task: {e}")

    def start_initial_processes(self, amqp_url):
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
                    assignment = self.resource_manager.assign_gpu_device(min_free_memory_gib=20, max_utilization=50, assigned_gpus=assigned_gpus)
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

    def sampler_process(self, amqp_url, device=None):
        from decos import sampler, gpt 
        local_id = current_process().pid
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
                    await sampler_task
                except Exception as e:
                    self.logger.error(f"Sampler {local_id}: Error during task cancellation: {e}")
            if channel:
                try:
                    await channel.close()
                except Exception as e:
                    self.logger.error(f"Sampler {local_id}: Error closing channel: {e}")
            if connection:
                try:
                    await connection.close()
                except Exception as e:
                    self.logger.error(f"Sampler {local_id}: Error closing connection: {e}")
            loop.stop()
            self.logger.info(f"Sampler {local_id}: Graceful shutdown complete.")

        async def run_sampler():
            nonlocal connection, channel, sampler_task
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                channel = await connection.channel()

                sampler_queue = await process_utils.declare_standard_queue(channel, "sampler_queue")
                evaluator_queue = await process_utils.declare_standard_queue(channel, "evaluator_queue")

                try:
                    if self.config.sampler.gpt: 
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
                self.logger.info(f"Sampler {local_id}: Cancelled.")
            except Exception as e:
                self.logger.error(f"Sampler {local_id} error: {e}")
            finally:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                self.logger.debug(f"Sampler {local_id}: Connection closed.")

        process_utils.setup_signal_handlers(
            loop, "Sampler", local_id, self.logger,
            lambda: graceful_shutdown(loop, connection, channel, sampler_task)
        )

        try:
            loop.run_until_complete(run_sampler())
        except Exception as e:
            self.logger.info(f"Sampler process {local_id}: Exception occurred: {e}")
        finally:
            loop.close()
            self.logger.debug(f"Sampler process {local_id} has been closed gracefully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TaskManager with configurable scaling interval.")

######################################### General setting related arguments #######################################
    parser.add_argument(
        "--check_interval", 
        type=int, 
        default=200,
        help="Time interval (in seconds) between consecutive scaling checks for evaluators and samplers. Defaults to 200s."
        )

    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling (enabled by default)."
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where logs will be stored (default: logs)."
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(os.getcwd(), "config.py"),  # Set default to 'config.py' in the current directory
        help="Path to the configuration file (Python script containing the experiment config). Defaults to './config.py'.",
    )

########################################## Resources related arguments #############################################

    parser.add_argument(
        "--max_samplers",
        type=int,
        default=1000, 
        help="Maximum samplers the system can scale up to. Adjust based on resource availability. Default no hard limit and based on dynamic resource checks."
    )

    args = parser.parse_args()

    # By default, scaling is enabled unless --no-dynamic-scaling is passed
    enable_dynamic_scaling = not args.no_dynamic_scaling

    async def main():
        config = load_config(args.config_path)
        task_manager = TaskManager(
            config=config,
            check_interval=args.check_interval,
            log_dir=args.log_dir
        )
        task = asyncio.create_task(
            task_manager.main_task(enable_scaling=enable_dynamic_scaling)
        )
        await task

    asyncio.run(main())