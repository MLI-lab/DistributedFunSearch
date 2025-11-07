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
from funsearchmq.scaling_utils import ResourceManager
from funsearchmq import sampler
from funsearchmq import process_utils
from funsearchmq.process_entry import sampler_process_entry, load_config
import socket
from funsearchmq import gpt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        return f"Error fetching IP address: {e}"

class TaskManager:
    def __init__(self, config, check_interval, log_dir, config_path):
        self.config = config
        self.config_path = config_path  # Store for spawn compatibility
        self.log_dir = log_dir  # Store for spawn compatibility
        self.logger = self.initialize_logger(log_dir)
        self.sampler_processes = []
        self.tasks = []
        if self.config.sampler.gpt:
            self.resource_manager = ResourceManager(log_dir=log_dir, cpu_only=True, scaling_config=self.config.scaling)
        else:
            self.resource_manager = ResourceManager(log_dir=log_dir, scaling_config=self.config.scaling)
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
                        sampler_entry_function=sampler_process_entry,
                        evaluator_entry_function=None,
                        config_path=self.config_path,
                        log_dir=self.log_dir,
                        template=None,
                        inputs=None,
                        target_signatures=None,
                        sandbox_base_path=None,
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
            ctx = mp.get_context('spawn')
            for i in range(self.config.num_samplers):
                device = None
                try:
                    proc = ctx.Process(target=sampler_process_entry, args=(self.config_path, device, self.log_dir), name=f"Sampler-{i}")
                    proc.start()
                    self.logger.debug(f"Started Sampler Process {i} (GPT mode) with PID: {proc.pid}")
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device
                except Exception as e:
                    self.logger.error(f"Error starting sampler {i}: {e}")
                    continue
        else:
            assigned_gpus = set()
            ctx = mp.get_context('spawn')
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
                    proc = ctx.Process(target=sampler_process_entry, args=(self.config_path, device, self.log_dir), name=f"Sampler-{i}")
                    proc.start()
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device
                    self.logger.debug(f"Process-to-Device Map: {self.process_to_device_map}")
                except Exception as e:
                    self.logger.error(f"Failed to start sampler {i} due to error: {e}")
                    continue



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
            log_dir=args.log_dir,
            config_path=args.config_path
        )
        task = asyncio.create_task(
            task_manager.main_task(enable_scaling=enable_dynamic_scaling)
        )
        await task

    asyncio.run(main())