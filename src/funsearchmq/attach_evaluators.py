import asyncio
import logging
from logging import FileHandler
import json
import os
import signal
import sys
import torch.multiprocessing as mp
import socket
import argparse
from typing import Sequence, Any
import datetime
from funsearchmq.scaling_utils import ResourceManager
from funsearchmq import process_utils
from funsearchmq.process_entry import evaluator_process_entry, load_config
from funsearchmq import code_manipulation

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        return f"Error fetching IP address: {e}"

class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config, log_dir, target_signatures, config_path, sandbox_base_path):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.config_path = config_path  # Store for spawn compatibility
        self.log_dir = log_dir  # Store for spawn compatibility
        self.sandbox_base_path = sandbox_base_path  # Store for spawn compatibility
        self.log_filename = None  # Will store the shared log filename
        self.logger = self.initialize_logger(log_dir)
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        self.resource_manager = ResourceManager(log_dir=log_dir, cpu_only=True, scaling_config=self.config.scaling)
        self.target_signatures= target_signatures

    def initialize_logger(self, log_dir):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)
        os.makedirs(log_dir, exist_ok=True)
        pid = os.getpid()
        # Create PID-based log file that will be shared with child processes
        self.log_filename = f'attach_evaluators_pid{pid}.log'
        log_file_path = os.path.join(log_dir, self.log_filename)
        handler = FileHandler(log_file_path, mode='a')  # Changed to append mode for child processes
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    async def main_task(self, enable_scaling=True):
        resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=60))
        self.tasks = [resource_logging_task]

        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}.")
        try:
            self.template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'

            # Start initial evaluator processes
            self.start_initial_processes(self.template, function_to_evolve)

            # Create a connection/channel for scaling metrics using utility function
            self.logger.info("Creating connection for scaling logic...")
            connection = await process_utils.create_rabbitmq_connection(
                self.config, timeout=300
            )
            channel = await connection.channel()

            # Declare the evaluator queue for scaling
            evaluator_queue = await process_utils.declare_standard_queue(channel, "evaluator_queue")
            self.logger.info("evaluator_queue declared for scaling logic.")

            if enable_scaling:
                scaling_task = asyncio.create_task(
                    self.resource_manager.run_scaling_loop(
                        evaluator_queue=evaluator_queue,
                        sampler_queue=None,
                        evaluator_processes=self.evaluator_processes,
                        sampler_processes=None,
                        sampler_entry_function=None,
                        evaluator_entry_function=evaluator_process_entry,
                        config_path=self.config_path,
                        log_dir=self.log_dir,
                        template=self.template,
                        inputs=self.inputs,
                        target_signatures=self.target_signatures,
                        sandbox_base_path=self.sandbox_base_path,
                        max_evaluators=args.max_evaluators,
                        max_samplers=None,
                        check_interval=self.config.scaling.check_interval if hasattr(self.config, 'scaling') and self.config.scaling else args.check_interval,
                        log_filename=self.log_filename,
                    )
                )
                self.tasks.append(scaling_task)

            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")

    def start_initial_processes(self, template, function_to_evolve):
        # Start initial evaluator processes
        ctx = mp.get_context('fork')  # Use fork for evaluators
        for i in range(self.config.num_evaluators):
            proc = ctx.Process(
                target=evaluator_process_entry,
                # Pass log filename so child processes write to same file
                args=(self.config_path, template, self.inputs, self.target_signatures, self.log_dir, self.sandbox_base_path, self.log_filename),
                name=f"Evaluator-{i}"
            )
            proc.start()
            self.logger.info(f"Started Evaluator Process {i} with PID: {proc.pid}")
            self.evaluator_processes.append(proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TaskManager for evaluators with configurable scaling interval.")

######################################### General setting related arguments #######################################

    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(os.getcwd(), "logs"),
        help="Directory where logs will be stored. Defaults to './logs'.",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(os.getcwd(), "config.py"),  # Set default to 'config.py' in the current directory
        help="Path to the configuration file (Python script containing the experiment config). Defaults to './config.py'.",
    )

    parser.add_argument(
        "--sandbox_base_path",
        type=str,
        default=os.path.join(os.getcwd(), "sandbox"),
        help="Path to the sandbox directory. Defaults to './sandbox'.",
    )

########################################## Resources related arguments #############################################

    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling of evaluators (enabled by default)."
    )

    parser.add_argument(
        "--check_interval",
        type=int,
        default=120,
        help="Time interval (in seconds) between consecutive scaling checks for evaluators and samplers. Defaults to 120s (2 minutes)."
    )

    parser.add_argument(
        "--max_evaluators",
        type=int,
        default=1000,
        help="Maximum evaluators the system can scale up to. Adjust based on resource availability. Default no hard limit and based on dynamic resource checks."
    )

    parser.add_argument(
        "--max_samplers",
        type=int,
        default=1000, 
        help="Maximum samplers the system can scale up to. Adjust based on resource availability. Default no hard limit and based on dynamic resource checks."
    )

########################## Termination related arguments ###########################################

    parser.add_argument(
        "--target_solutions",
        type=str,
        default='{"(6, 1)": 10, "(7, 1)": 16, "(8, 1)": 30, "(9, 1)": 52, "(10, 1)": 94, "(11, 1)": 172}',  
        help="JSON string specifying target solutions for (n, s_value). Example: '{\"(6, 1)\": 8, \"(7, 1)\": 14, \"(8, 1)\": 25}'"
    )

    args = parser.parse_args()

    # Convert JSON string to dictionary
    try:
        if args.target_solutions:
            target_signatures = json.loads(args.target_solutions)
            target_signatures = {eval(k): v for k, v in target_signatures.items()}  # Convert string keys to tuples
        else:
            target_signatures=args.target_solutions
            
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for --target_solutions. Example: '{\"(6, 1)\": 8, \"(7, 1)\": 14, \"(8, 1)\": 25}'.")

    # Dynamic scaling is enabled unless --no-dynamic-scaling is passed.
    enable_dynamic_scaling = not args.no_dynamic_scaling

    async def main():
        config = load_config(args.config_path)

        # Load the specification from the provided path or default
        spec_path = config.evaluator.spec_path
        try:
            with open(spec_path, 'r') as file:
                specification = file.read()
            if not isinstance(specification, str) or not specification.strip():
                raise ValueError("Specification must be a non-empty string.")

            # Substitute start_n placeholder with actual value from config
            # This allows hash computation to use the correct n value without manual sync
            actual_start_n = config.evaluator.start_n[0]  # Get first start_n value
            specification = specification.replace("n == start_n", f"n == {actual_start_n}")

        except FileNotFoundError:
            print(f"Error: Specification file not found at {spec_path}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error in specification: {e}")
            sys.exit(1)

        if not (len(config.evaluator.s_values) == len(config.evaluator.start_n) == len(config.evaluator.end_n)):
            raise ValueError("The number of elements in --s-values, --start-n, and --end-n must match.")

        inputs = [(n, s, config.evaluator.q) for s, start_n, end_n in zip(config.evaluator.s_values, config.evaluator.start_n, config.evaluator.end_n) for n in range(start_n, end_n + 1)]

        task_manager = TaskManager(
            specification=specification,
            inputs=inputs,
            config=config,
            log_dir=args.log_dir,
            target_signatures=target_signatures,
            config_path=args.config_path,
            sandbox_base_path=args.sandbox_base_path
        )

        task = asyncio.create_task(
            task_manager.main_task(enable_scaling=enable_dynamic_scaling)
        )
        await task

    # Top-level call to asyncio.run() to start the event loop
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error in asyncio.run(main()): {e}")