import asyncio
import logging
from logging import FileHandler
import json
import aio_pika
import os
import signal
import sys
import torch.multiprocessing as mp
import socket
import argparse
from typing import Sequence, Any
import datetime
from scaling_utils import ResourceManager
from yarl import URL
import code_manipulation
import importlib
import socket


def load_config(config_path):
    """
    Dynamically load a configuration module from a specified path.
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
    def __init__(self, specification: str, inputs: Sequence[Any], config, check_interval_eval, log_dir):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger(log_dir)
        self.check_interval_eval = check_interval_eval
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        self.resource_manager = ResourceManager(log_dir=log_dir, cpu_only=True)

    def initialize_logger(self, log_dir):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)

        os.makedirs(log_dir, exist_ok=True)  # Ensure the logs folder exists

        # Use the hostname as part of the log file name for uniqueness
        hostname = socket.gethostname()
        log_file_name = f'eval_{hostname}.log'
        log_file_path = os.path.join(log_dir, log_file_name)

        handler = FileHandler(log_file_path, mode='w')  # Create a file handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger


    async def scaling_controller(self, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)
        check_interval_sam = 120
        max_evaluators = 110 # (should not be more than num_cores-rest of processes load)/2 as each eval uses two cpus 
        min_evaluators = 1
        evaluator_threshold = 5
        initial_sleep_duration = 120  # seconds
        await asyncio.sleep(initial_sleep_duration)

        # Create a connection and channels for getting queue metrics
        try:
            connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=80,
            )
            channel = await connection.channel()
        except Exception as e:
            self.logger.error(f"Error connecting to RabbitMQ: {e}")
            return

        while True:
            try:
                load_avg_1, load_avg_5, _ = os.getloadavg()
                num_cores = len(os.sched_getaffinity(0))  # Available CPU cores
                if load_avg_5 > num_cores or load_avg_1 > num_cores:
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

                # Adjust evaluator processes
                await self.resource_manager.adjust_processes(
                    evaluator_message_count, evaluator_threshold,
                    self.evaluator_processes, self.evaluator_process,
                    args=(self.template, self.inputs, amqp_url),
                    max_processes=max_evaluators, min_processes=min_evaluators,
                    process_name='Evaluator'
                )

            except Exception as e:
                self.logger.error(f"Scaling controller encountered an error: {e}")
            await asyncio.sleep(self.check_interval_eval)  # Non-blocking sleep



    async def main_task(self,  enable_scaling=True):
        amqp_url = URL(
            f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/{self.config.rabbitmq.vhost}'
        ).update_query(heartbeat=480000)        
        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}.")
        try:

            self.template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'

            # Start initial processes
            self.start_initial_processes(self.template, function_to_evolve, amqp_url)
            resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=200))

            self.tasks = [resource_logging_task]
            if enable_scaling:
                scaling_task = asyncio.create_task(self.scaling_controller(function_to_evolve, amqp_url))
                self.tasks.append(scaling_task)
    
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")


    def start_initial_processes(self, template, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)

        # Start initial evaluator and database processes as previously done
        for i in range(self.config.num_evaluators):
            proc = mp.Process(target=self.evaluator_process, args=(self.template, self.inputs, amqp_url), name=f"Evaluator-{i}")
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
                    timeout=80,
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



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the TaskManager with configurable scaling interval.")
    parser.add_argument("--check_interval_eval", type=int, default=200, help="Interval in seconds for scaling evaluator processes.")
    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling of evaluators and samplers (enabled by default).",
    )
    parser.add_argument(
        "--spec-path",
        type=str,
        default=os.path.join(os.getcwd(), 'implementation/specifications/baseline.txt'),
        help="Path to the specification file. Defaults to 'implementation/specifications/baseline.txt'.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where logs will be stored. Defaults to 'logs'."
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Name of the configuration file (without .py extension). Defaults to 'config'.",
    )
    args = parser.parse_args()

    
    # Invert the logic: dynamic scaling is True by default unless explicitly disabled
    enable_dynamic_scaling = not args.no_dynamic_scaling
    
    async def main():
        # Initialize configuration
        config = load_config(args.config_name)

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

        # Initialize the task manager
        task_manager = TaskManager(
            specification=specification,
            inputs=inputs,
            config=config,
            check_interval_eval=args.check_interval_eval, 
            log_dir=args.log_dir
        )

        # Start the main task
        task = asyncio.create_task(
            task_manager.main_task(
                enable_scaling=enable_dynamic_scaling,
            )
        )

        # Await tasks to run them
        await task

    # Top-level call to asyncio.run() to start the event loop
    asyncio.run(main())