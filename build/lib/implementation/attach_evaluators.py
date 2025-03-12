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

def load_config(config_path):
    """
    Load a configuration .py file from a specified path.
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
    def __init__(self, specification: str, inputs: Sequence[Any], config, check_interval, log_dir, TARGET_SIGNATURES):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger(log_dir)
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []  
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        self.resource_manager = ResourceManager(log_dir=log_dir, cpu_only=True)
        self.TARGET_SIGNATURES= TARGET_SIGNATURES

    def initialize_logger(self, log_dir):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)
        os.makedirs(log_dir, exist_ok=True)
        hostname = socket.gethostname()
        log_file_name = f'eval_{hostname}.log'
        log_file_path = os.path.join(log_dir, log_file_name)
        handler = FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    async def main_task(self, enable_scaling=True):
        amqp_url = URL(
            f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}'
            f'@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/{self.config.rabbitmq.vhost}'
        ).update_query(heartbeat=480000)
        resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=60))
        self.tasks = [resource_logging_task]

        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}.")
        try:
            self.template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'

            # Start initial evaluator processes
            self.start_initial_processes(self.template, function_to_evolve, amqp_url)

            # Create a connection/channel for scaling metrics
            self.logger.info("Creating connection for scaling logic...")
            connection = await aio_pika.connect_robust(str(amqp_url), timeout=300)
            channel = await connection.channel()

            # Declare the evaluator queue for scaling
            evaluator_queue = await channel.declare_queue(
                "evaluator_queue",
                durable=False,
                auto_delete=False,
                arguments={'x-consumer-timeout': 360000000}
            )
            self.logger.info("evaluator_queue declared for scaling logic.")

            if enable_scaling:
                scaling_task = asyncio.create_task(
                    self.resource_manager.run_scaling_loop(
                        evaluator_queue=evaluator_queue, 
                        sampler_queue=None,
                        evaluator_processes=self.evaluator_processes,
                        sampler_processes=None,
                        evaluator_function=self.evaluator_process,
                        sampler_function=None,
                        evaluator_args=(self.template, self.inputs, amqp_url, self.TARGET_SIGNATURES),
                        sampler_args=None,
                        max_evaluators=args.max_evaluators,
                        max_samplers=None,
                        check_interval=args.check_interval,
                    )
                )
                self.tasks.append(scaling_task)

            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")

    def start_initial_processes(self, template, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)
        # Start initial evaluator processes
        for i in range(self.config.num_evaluators):
            proc = mp.Process(
                target=self.evaluator_process,
                args=(template, self.inputs, amqp_url),
                name=f"Evaluator-{i}"
            )
            proc.start()
            self.logger.info(f"Started Evaluator Process {i} with PID: {proc.pid}")
            self.evaluator_processes.append(proc)

    def evaluator_process(self, template, inputs, amqp_url):
        import evaluator  # Import evaluator module dynamically
        local_id = mp.current_process().pid  # Use process ID as local identifier
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

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
                    await evaluator_task
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
            nonlocal connection, channel, evaluator_task
            try:
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                channel = await connection.channel()
                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue",
                    durable=False,
                    auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                database_queue = await channel.declare_queue(
                    "database_queue",
                    durable=False,
                    auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                evaluator_instance = evaluator.Evaluator(
                    connection, channel, evaluator_queue, database_queue,
                    self.template, 'priority', 'evaluate', inputs, '/workspace/sandboxstorage/',
                    timeout_seconds=args.timeout, local_id=local_id, TARGET_SIGNATURES=self.TARGET_SIGNATURES
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
                self.logger.debug(f"Evaluator {local_id}: Connection closed.")

        def signal_handler(sig, frame):
            self.logger.info(f"Evaluator process {local_id} received signal {sig}. Initiating shutdown.")
            loop.create_task(graceful_shutdown(loop, connection, channel, evaluator_task))

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
    parser = argparse.ArgumentParser(description="Run the TaskManager for evaluators with configurable scaling interval.")

    parser.add_argument(
        "--s-values",
        type=int,
        nargs="+",
        default=[1],  # Default is single deletion (s=1)
        help="List of s values for deletions (default: [1]). Example: '--s-values 1 2' for single and two deletions.",
    )

    parser.add_argument(
        "--start-n",
        type=int,
        nargs="+",
        default=[6],  # Default range start is 6
        help="List of start values for each s (default: [6]). Example: '--start-n 4 6' for different starts per s.",
    )

    parser.add_argument(
        "--end-n",
        type=int,
        nargs="+",
        default=[11],  # Default range end is 11
        help="List of end values for each s (default: [11]). Example: '--end-n 11 12' for different ends per s.",
    )

    parser.add_argument(
        "--check_interval", 
        type=int, 
        default=600,
        help="Time interval (in seconds) between consecutive scaling checks for evaluators and samplers. Defaults to 600s."
        )
        
    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling of evaluators (enabled by default)."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for the sandbox. Default is 10min."
    )

    parser.add_argument(
        "--spec-path",
        type=str,
        default=os.path.join(os.getcwd(), 'implementation/specifications/baseline.txt'),
        help="Path to the specification file. Defaults to 'implementation/specifications/baseline.txt'."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where logs will be stored. Defaults to 'logs'."
    )

    parser.add_argument(
        "--max_evaluators",
        type=int,
        default=1000,
        help="Maximum evaluators the system can scale up to. Adjust based on resource availability. Default no hard limit and based on dynamic resource checks."
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Name of the configuration file (without .py extension). Defaults to 'config'."
    )

    parser.add_argument(
        "--target_solutions",
        type=str,
        default='{"(6, 1)": 8, "(7, 1)": 14, "(8, 1)": 25, "(9, 1)": 42, "(10, 1)": 71, "(11, 1)": 125}',  
        help="JSON string specifying target solutions for (n, s_value). Example: '{\"(6, 1)\": 8, \"(7, 1)\": 14, \"(8, 1)\": 25}'"
    )
    args = parser.parse_args()

    if not (len(args.s_values) == len(args.start_n) == len(args.end_n)):
        raise ValueError("The number of elements in --s-values, --start-n, and --end-n must match.")


    # Convert JSON string to dictionary
    try:
        TARGET_SIGNATURES = json.loads(args.target_solutions)
        TARGET_SIGNATURES = {eval(k): v for k, v in TARGET_SIGNATURES.items()}  # Convert string keys to tuples
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for --target_solutions. Example: '{\"(6, 1)\": 8, \"(7, 1)\": 14, \"(8, 1)\": 25}'")


    # Dynamic scaling is enabled unless --no-dynamic-scaling is passed.
    enable_dynamic_scaling = not args.no_dynamic_scaling

    async def main():
        config = load_config(args.config_name)
        # Load the specification from the provided path.
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


        inputs = [(n, s) for s, start_n, end_n in zip(args.s_values, args.start_n, args.end_n) for n in range(start_n, end_n + 1)]

        task_manager = TaskManager(
            specification=specification,
            inputs=inputs,
            config=config,
            check_interval=args.check_interval,
            log_dir=args.log_dir, 
            TARGET_SIGNATURES= TARGET_SIGNATURES
        )

        task = asyncio.create_task(
            task_manager.main_task(enable_scaling=enable_dynamic_scaling)
        )
        await task

    asyncio.run(main())
