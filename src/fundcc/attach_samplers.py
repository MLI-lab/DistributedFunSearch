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
from fundcc.scaling_utils import ResourceManager
from fundcc import sampler
import importlib
import socket
from fundcc import gpt


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
        self.evaluator_processes = []
        self.database_processes = []
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
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
                self.logger.info("Cannot connect to rabbitmq. Change config file.")

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
            sampler_queue = await channel.declare_queue(
                "sampler_queue",
                durable=False,
                auto_delete=True,
                arguments={'x-consumer-timeout': 360000000}
            )
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
            self.logger.info("Waiting on tasks...")
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception in main_task: {e}")

    def start_initial_processes(self, amqp_url):
        """
        Starts a fixed number of initial sampler processes,
        each assigned to a GPU if available.
        """
        amqp_url = str(amqp_url)
        assigned_gpus = set()

        for i in range(self.config.num_samplers):
            try: 
                assignment = self.resource_manager.assign_gpu_device(assigned_gpus=assigned_gpus)
            except Exception as e: 
                self.logger.error(f"Cannot start sampler {i}: No suitable GPU available and error {e}.")
            
            if assignment is not None:
                host_gpu, device = assignment
                assigned_gpus.add(device)
                self.logger.info(f"Assigning sampler {i} to GPU {device} (host GPU: {host_gpu})")
            try:
                proc = mp.Process(target=self.sampler_process,args=(amqp_url, device),name=f"Sampler-{i}")
                proc.start()
                self.sampler_processes.append(proc)
                self.process_to_device_map[proc.pid] = device
                self.logger.debug(f"Process-to-Device Map:: {self.process_to_device_map}")
            except Exception as e:
                self.logger.error(f"Failed to start sampler {i} due to error: {e}")
                continue

    def sampler_process(self, amqp_url, device=None):
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
                    await asyncio.wait_for(sampler_task, timeout=10)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Sampler {local_id}: Task timed out. Cancelling...")
                    sampler_task.cancel()
                    await sampler_task
            if channel:
                await channel.close()
            if connection:
                await connection.close()
            loop.stop()
            self.logger.info(f"Sampler {local_id}: Graceful shutdown complete.")

        def signal_handler(sig, frame):
            self.logger.info(f"Sampler {local_id} received signal {sig}. Initiating shutdown.")
            loop.create_task(graceful_shutdown(loop, connection, channel, sampler_task))

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        async def run_sampler():
            nonlocal connection, channel, sampler_task
            try:
                self.logger.debug(f"Sampler {local_id}: Connecting to RabbitMQ.")
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    #client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                channel = await connection.channel()
                self.logger.debug(f"Sampler {local_id}: Channel established.")

                sampler_queue = await channel.declare_queue(
                    "sampler_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )
                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=True,
                    arguments={'x-consumer-timeout': 360000000}
                )

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
                self.logger.info(f"Sampler {local_id}: Cancelled.")
            except Exception as e:
                self.logger.error(f"Sampler {local_id} error: {e}")
            finally:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                self.logger.debug(f"Sampler {local_id}: Connection closed.")

        try:
            loop.run_until_complete(run_sampler())
        except Exception as e:
            self.logger.info(f"Sampler {local_id} exception: {e}")
        finally:
            loop.close()
            self.logger.debug(f"Sampler {local_id} ended gracefully.")


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
