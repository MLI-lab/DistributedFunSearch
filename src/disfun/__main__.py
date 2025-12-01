# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License - Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing - software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This is a distributed implementation of FunSearch, adapted from DeepMind's original single-threaded version.
It uses RabbitMQ and asyncio for asynchronous message passing, enabling parallel evaluation and sampling across multiple processes and nodes.

In initialize_logger set logger.setLevel(logging.INFO) to logger.setLevel(logging.DEBUG) for more detailed logs including prompt sent to LLM.
"""

import os
import re
import sys
import time
import glob
import random

import json
import shutil
import signal
import pickle
import argparse
import logging
import asyncio
import datetime
from typing import Any
from collections.abc import Sequence

import torch.multiprocessing as mp
import aio_pika
import psutil

from disfun import (
    programs_database,
    code_manipulation,
    process_utils,
)
from disfun.scaling_utils import ResourceManager
from disfun.process_entry import sampler_process_entry, evaluator_process_entry

# Disable multi-threaded tokenization.
# Our prompts are short and we run many parallel processes so single-threaded tokenization is faster
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be called before any multiprocessing to avoid CUDA context conflicts
# Required to prevent fork+threading deadlocks when dynamically scaling samplers
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set



def backup_python_files(src, dest, exclude_dirs=[]):
    """
    Recursively copies all Python files in src to dest.
    If dest or any subdirectory does not exist. they are created.
    """
    for file_path in glob.glob(os.path.join(src, '**', '*.py'), recursive=True):
        if "/code_backup/" in file_path:
            continue
        if any([file_path.startswith(dir) for dir in exclude_dirs]):
            continue
        new_path = f"{dest}/{file_path.replace('./', '')}"
        dirname = os.path.dirname(new_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy(file_path, new_path)


def create_evaluation_inputs(config):
    """
    Generate evaluation input tuples from config parameters.

    Each tuple defines one problem instance. Customize this function to change
    how config parameters are combined into evaluation inputs.

    Args:
        config: Configuration object with evaluator settings

    Returns:
        List of tuples - where each tuple contains problem-specific parameters

    Example:
        For config with s_values=[1 - 2] - start_n=[5 - 7] - end_n=[6 - 8] - q=2,
        generates: [(5,1,2) - (6,1,2) - (7,2,2) - (8,2,2)]
    """
    inputs = [
        (n, s, config.evaluator.q)
        for s, start_n, end_n in zip(
            config.evaluator.s_values,
            config.evaluator.start_n,
            config.evaluator.end_n,
            strict=True
        )
        for n in range(start_n, end_n + 1)
    ]
    return inputs


class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config, log_dir, target_signatures, config_path, sandbox_base_path):
        self.template = code_manipulation.text_to_program(specification)
        self.inputs = inputs
        self.config = config
        self.config_path = config_path  # Store for spawn compatibility
        self.log_dir = log_dir  # Store for spawn compatibility
        self.sandbox_base_path = sandbox_base_path  # Store for spawn compatibility
        # Create PID-based log filename for main process
        pid = os.getpid()
        self.log_filename = f'main_pid{pid}.log'
        self.logger = process_utils.initialize_logger(log_dir, self.log_filename, process_type="Main")
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        self.sampler_connection = None
        self.database_connection = None
        self.sampler_channel = None
        self.database_channel = None
        # ResourceManager handles both GPU (for vLLM) and API-based samplers
        self.resource_manager = ResourceManager(log_dir=log_dir, scaling_config=self.config.scaling)
        self.process_to_device_map = {}
        self.target_signatures = target_signatures

    async def publish_initial_program_with_retry(self, initial_program_data, max_retries=5, delay=5):
        attempt = 0
        while attempt < max_retries:
            try:
                sampler_connection = await process_utils.create_rabbitmq_connection(
                    self.config, timeout=300
                )
                sampler_channel = await sampler_connection.channel()

                # Ensure the evaluator_queue is declared
                await sampler_channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )

                await sampler_channel.default_exchange.publish(
                    aio_pika.Message(body=initial_program_data.encode()),
                    routing_key='evaluator_queue'
                )
                self.logger.info("Published initial program")
                await sampler_channel.close()
                await sampler_connection.close()
                return  # Exit the function after successful publish
            except Exception as e:
                attempt += 1
                self.logger.error(f"Attempt {attempt} failed to publish initial program: {e}")
                if attempt < max_retries:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error("Max retries reached. Failed to publish initial program.")
                    raise e  # Re-raise the exception after max retries

    async def log_tasks(self):
        """
        Periodically logs the following information every 60 seconds:
        - Total number of active asyncio tasks.
        - Task details such as name - coroutine function - and current state.
        - If a task is in the "PENDING" state - logs the stack frames of the task to help diagnose blocking issues.
        Usage:
            asyncio.create_task(self.log_tasks())
        """
        while True:
            tasks = asyncio.all_tasks()
            self.logger.debug(f"Currently {len(tasks)} tasks running:")
            for task in tasks:
                coro_name = task.get_coro().__name__ if task.get_coro() else "Unknown"
                self.logger.debug(f"Task: {task.get_name()}, Function: {coro_name}, Status: {task._state}")

                # Log the stack of the task if it's pending or blocked
                if task._state == "PENDING":
                    stack = task.get_stack()
                    for frame in stack:
                        self.logger.debug(f"Pending Task Frame: {frame}")
            await asyncio.sleep(60)

    async def main_task(self, enable_scaling=True, checkpoint_file=None):
        # Determine run name: prefer checkpoint > config > auto-generate
        run_name = None

        # If resuming from checkpoint, try to load the original run name
        if checkpoint_file:
            # e.g., /path/checkpoint_run_20251109_120115/checkpoint_*.pkl to run_20251109_120115
            path_match = re.search(r'/checkpoint_(run_\d{8}_\d{6})/', checkpoint_file)
            if path_match:
                run_name = path_match.group(1)
                self.logger.info(f"Extracted run name from checkpoint path: {run_name}")
            else:
                # Load run name from checkpoint file
                try:
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                        run_name = checkpoint_data.get('wandb_run_name', None)
                        if run_name:
                            self.logger.info(f"Loaded run name from checkpoint: {run_name}")
                except Exception as e:
                    self.logger.warning(f"Could not extract run name from checkpoint: {e}")

        # Fall back to config or auto-generate if not found in checkpoint
        if not run_name:
            if self.config.wandb.run_name is None:
                run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                # Append tag if configured
                if self.config.wandb.run_name_tag:
                    run_name = f"{run_name}_{self.config.wandb.run_name_tag}"
                self.logger.info(f"Auto-generated run name: {run_name}")
            else:
                run_name = self.config.wandb.run_name
                self.logger.info(f"Using configured run name: {run_name}")

        # Construct checkpoint path: {base_path}/checkpoint_{run_name}/
        checkpoints_base_path = self.config.wandb.checkpoints_base_path
        save_checkpoints_path = os.path.join(checkpoints_base_path, f"checkpoint_{run_name}")
        self.logger.info(f"Checkpoints will be saved to: {save_checkpoints_path}")

        # Initialize random seed for reproducibility (if not resuming from checkpoint)
        # This controls: (1) prompt construction (cluster/program sampling), (2) LLM generation
        if checkpoint_file is None and self.config.random_seed is not None:
            import numpy as np
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            self.logger.info(f"Initialized random seed: {self.config.random_seed} (controls prompt construction + LLM generation)")
        elif checkpoint_file:
            self.logger.info("Random state will be restored from checkpoint")

        try:
            # Test RabbitMQ connectivity before proceeding
            test_connection = await process_utils.create_rabbitmq_connection(
                self.config, timeout=300
            )
            await test_connection.close()
        except Exception as e:
            self.logger.error(f"Cannot connect to RabbitMQ: {e}")
            self.logger.info("Cannot connect to rabbitmq. Change config file.")
            raise

        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}")

        # Initialize function to evolve
        function_to_evolve = 'priority'

        # Load all initial functions from the initial_functions directory
        initial_programs = []
        if checkpoint_file is None:
            from pathlib import Path
            initial_functions_dir = Path(self.config.evaluator.initial_functions_dir)

            # Check if we should strip tags from initial functions
            prompt_style_path = self.config.prompt.prompt_style_path if hasattr(self.config.prompt, 'prompt_style_path') else None
            strip_tags = prompt_style_path is None

            if initial_functions_dir.exists() and initial_functions_dir.is_dir():
                for txt_file in sorted(initial_functions_dir.glob("*.txt")):
                    try:
                        function_body = txt_file.read_text().strip()

                        # If no prompt_style, strip tags from initial functions
                        if strip_tags:
                            function_body = re.sub(r'<thinking>.*?</thinking>\s*', '', function_body, flags=re.DOTALL)
                            function_body = re.sub(r'<thought>.*?</thought>\s*', '', function_body, flags=re.DOTALL)
                            function_body = re.sub(r'<code>(.*?)</code>', r'\1', function_body, flags=re.DOTALL)
                            function_body = function_body.strip()
                            self.logger.info(f"Stripped tags from initial function: {txt_file.name}")

                        initial_program_data = json.dumps({
                            "sample": function_body,
                            "island_id": None,
                            "version_generated": None,
                            "expected_version": 0
                        })
                        initial_programs.append(initial_program_data)
                        self.logger.info(f"Loaded initial function from: {txt_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load initial function from {txt_file}: {e}")

                if not initial_programs:
                    self.logger.warning(f"No initial functions found in {initial_functions_dir}")
                else:
                    self.logger.info(f"Loaded {len(initial_programs)} initial function(s)")
            else:
                self.logger.warning(f"Initial functions directory not found: {initial_functions_dir}")

        try:
            # Create connections and declare queues
            self.sampler_connection = await process_utils.create_rabbitmq_connection(
                self.config, timeout=300
            )
            self.sampler_channel = await self.sampler_connection.channel()

            self.database_connection = await process_utils.create_rabbitmq_connection(
                self.config, timeout=300
            )
            self.database_channel = await self.database_connection.channel()

            evaluator_queue = await process_utils.declare_standard_queue(self.sampler_channel, "evaluator_queue")
            sampler_queue = await process_utils.declare_standard_queue(self.sampler_channel, "sampler_queue")
            database_queue = await process_utils.declare_standard_queue(self.database_channel, "database_queue")
            try:
                # Now create the database instance
                database = programs_database.ProgramsDatabase(
                    self.database_connection, self.database_channel, database_queue,
                    sampler_queue, evaluator_queue, self.config.programs_database,
                    function_to_evolve, checkpoint_file, save_checkpoints_path,
                    mode=self.config.evaluator.mode,
                    start_n=self.config.evaluator.start_n, end_n=self.config.evaluator.end_n, s_values=self.config.evaluator.s_values, no_deduplication=self.config.programs_database.no_deduplication, prompt_limit=self.config.termination.prompt_limit if hasattr(self.config, 'termination') and self.config.termination else 400_000_000, optimal_solution_programs=self.config.termination.optimal_solution_programs if hasattr(self.config, 'termination') and self.config.termination else 200000, target_signatures=self.target_signatures,
                    show_eval_scores=self.config.prompt.show_eval_scores, display_mode=self.config.prompt.display_mode, best_known_solutions=self.config.prompt.best_known_solutions, absolute_label=self.config.prompt.absolute_label, relative_label=self.config.prompt.relative_label, q=self.config.evaluator.q,
                    wandb_config=self.config.wandb,
                    sampler_config=self.config.sampler,
                    evaluator_config=self.config.evaluator,
                    prompt_config=self.config.prompt,
                    run_name=run_name,
                    rabbitmq_config=self.config
                )
                database_task = asyncio.create_task(database.consume_and_process())
            except Exception as e:
                self.logger.error(f"Exception in database as {e}")

            checkpoint_task = asyncio.create_task(database.periodic_checkpoint())
            wandb_logging_task = asyncio.create_task(database.periodic_wandb_logging())

            # Start consumers before publishing
            try:
                # Get starting sampler ID from database (loaded from checkpoint if resuming)
                # This ensures samplers get unique IDs that don't repeat seeds from previous runs
                starting_sampler_id = database.next_sampler_id if checkpoint_file else 0
                if checkpoint_file:
                    self.logger.info(f"Resuming from checkpoint: starting sampler IDs at {starting_sampler_id}")

                self.start_initial_processes(function_to_evolve, checkpoint_file, starting_sampler_id)
                self.logger.info("Initial processes started successfully.")

                # Sync next_sampler_id from database to resource manager
                # This ensures dynamically scaled samplers get unique IDs that don't collide
                if database.next_sampler_id > self.resource_manager.next_sampler_id:
                    self.resource_manager.next_sampler_id = database.next_sampler_id
                    self.logger.info(f"Synced next_sampler_id to resource manager: {database.next_sampler_id}")

                # Link resource_manager to database so sampler ID counter stays in sync for checkpointing
                self.resource_manager.database = database
            except Exception as e:
                self.logger.error(f"Failed to start initial processes: {e}")

            # Wait for at least one sampler to be ready before publishing
            # Note: For local models (vLLM), sampler won't register as consumer until model is loaded (~5-10 min)
            self.logger.info("Waiting for at least one sampler to connect (this may take several minutes for local models)...")
            while True:
                # Call declare() to get fresh queue info from RabbitMQ
                queue_info = await sampler_queue.declare()
                consumer_count = queue_info.consumer_count
                self.logger.info(f"consumer_count is {consumer_count} while config num_samplers is {self.config.num_samplers}")

                if consumer_count >= 1 and checkpoint_file is None:
                    # Publish initial programs (multiple copies for warm start)
                    num_copies = getattr(self.config.programs_database, 'initial_program_copies', 1)
                    all_publications = []
                    for _ in range(num_copies):
                        all_publications.extend(initial_programs)

                    if all_publications:
                        await asyncio.gather(*[
                            self.publish_initial_program_with_retry(prog_data)
                            for prog_data in all_publications
                        ])
                        self.logger.info(f"Published {len(all_publications)} initial programs ({len(initial_programs)} unique x {num_copies} copies)")
                    break
                elif consumer_count >= 1:
                    await database.get_prompt()
                    self.logger.info(f"Loading from checkpoint: {checkpoint_file}")
                    break
                else:
                    self.logger.info("No consumers yet on sampler_queue. Retrying in 10 seconds...")
                    await asyncio.sleep(10)

            # Start resource logging
            resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=60))

            self.tasks = [database_task, checkpoint_task, wandb_logging_task, resource_logging_task]

            if enable_scaling:
                try:
                    scaling_task = asyncio.create_task(
                        self.resource_manager.run_scaling_loop(
                            evaluator_queue=evaluator_queue,
                            sampler_queue=sampler_queue,
                            evaluator_processes=self.evaluator_processes,
                            sampler_processes=self.sampler_processes,
                            sampler_entry_function=sampler_process_entry,
                            evaluator_entry_function=evaluator_process_entry,
                            config=self.config,
                            config_path=self.config_path,
                            log_dir=self.log_dir,
                            template=self.template,
                            inputs=self.inputs,
                            target_signatures=self.target_signatures,
                            sandbox_base_path=self.sandbox_base_path,
                            max_evaluators=self.config.scaling.max_evaluators if hasattr(self.config, 'scaling') and self.config.scaling else 1000,
                            max_samplers=self.config.scaling.max_samplers if hasattr(self.config, 'scaling') and self.config.scaling else 1000,
                            check_interval=self.config.scaling.check_interval if hasattr(self.config, 'scaling') and self.config.scaling else 120,
                            log_filename=self.log_filename,
                        )
                    )
                    self.tasks.append(scaling_task)
                except Exception as e:
                    self.logger.error(f"Error enabling scaling {e}")

            self.channels = [self.database_channel, self.sampler_channel]
            self.queues = ["database_queue", "sampler_queue", "evaluator_queue"]

            # Run all tasks concurrently
            await asyncio.gather(*self.tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")


    def start_initial_processes(self, function_to_evolve, checkpoint_file, starting_sampler_id=0):
        from disfun.sampler import is_local_model
        import traceback

        try:
            # Check if using local model (needs GPU) or API model (CPU only)
            use_local = is_local_model(self.config.sampler.model)
        except Exception as e:
            self.logger.error(f"Error checking model type: {e}")
            self.logger.error(traceback.format_exc())
            raise

        # Track the next sampler ID to assign (starts from checkpoint value if resuming)
        next_sampler_id = starting_sampler_id

        if use_local:
            # LOCAL MODEL: Each sampler loads model on assigned GPU
            self.logger.info(f"Starting {self.config.num_samplers} sampler(s) with LOCAL model: {self.config.sampler.model}")
            self.logger.info("Each sampler will load the model on its assigned GPU")

            assigned_gpus = set()
            ctx = mp.get_context('spawn')

            for i in range(self.config.num_samplers):
                try:
                    assignment = self.resource_manager.assign_gpu_device(
                        min_free_memory_gib=20,
                        max_utilization=50,
                        assigned_gpus=assigned_gpus
                    )
                except Exception as e:
                    self.logger.error(f"Cannot start sampler {i}: No suitable GPU available. Error: {e}")
                    continue

                if assignment is None:
                    self.logger.error("No suitable GPU available for sampler. Skipping.")
                    continue

                host_gpu, device = assignment
                assigned_gpus.add(device)
                sampler_id = next_sampler_id
                next_sampler_id += 1
                self.logger.info(f"Assigning sampler (ID={sampler_id}) to GPU {device} (host GPU: {host_gpu})")

                try:
                    proc = ctx.Process(
                        target=sampler_process_entry,
                        args=(self.config_path, device, self.log_dir, self.log_filename, sampler_id),
                        name=f"Sampler-{sampler_id}"
                    )
                    proc.start()
                    self.logger.info(f"Started Sampler Process (ID={sampler_id}) with PID: {proc.pid} on GPU {device}")
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device

                    # Delay between starting samplers to avoid model loading race conditions
                    if i < self.config.num_samplers - 1:
                        self.logger.info("Waiting 10 seconds before starting next sampler...")
                        time.sleep(10)
                except Exception as e:
                    self.logger.error(f"Failed to start sampler (ID={sampler_id}): {e}")
                    continue
        else:
            # API MODEL: Samplers are CPU processes
            self.logger.info(f"Starting {self.config.num_samplers} sampler(s) with API model: {self.config.sampler.model}")
            self.logger.info("Samplers will use API (no GPU assignment needed)")

            ctx = mp.get_context('spawn')
            for _ in range(self.config.num_samplers):
                device = None  # No GPU for API models
                sampler_id = next_sampler_id
                next_sampler_id += 1
                try:
                    proc = ctx.Process(
                        target=sampler_process_entry,
                        args=(self.config_path, device, self.log_dir, self.log_filename, sampler_id),
                        name=f"Sampler-{sampler_id}"
                    )
                    proc.start()
                    self.logger.info(f"Started Sampler Process (ID={sampler_id}) with PID: {proc.pid}")
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device
                except Exception as e:
                    self.logger.error(f"Error starting sampler (ID={sampler_id}): {e}")
                    continue

        # Initialize ResourceManager's next_sampler_id to continue from where we left off
        # This ensures dynamically scaled samplers get unique IDs that don't collide with initial ones
        self.resource_manager.next_sampler_id = next_sampler_id
        self.logger.info(f"Initialized next_sampler_id to {self.resource_manager.next_sampler_id} for dynamic scaling")

        # Start initial evaluator processes
        ctx = mp.get_context('fork')  # Use fork for evaluators (no model loading - no deadlock risk)
        for i in range(self.config.num_evaluators):
            proc = ctx.Process(
                target=evaluator_process_entry,
                # Pass log filename so child processes write to same file
                args=(self.config_path, self.template, self.inputs, self.target_signatures, self.log_dir, self.sandbox_base_path, self.log_filename),
                name=f"Evaluator-{i}"
            )
            proc.start()
            self.logger.debug(f"Started Evaluator Process {i} with PID: {proc.pid}")
            self.evaluator_processes.append(proc)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run FunSearch experiment.")

######################################### General setting related arguments #######################################

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Enable backup of Python files before running the task. Note: Also configurable via config.paths.backup_enabled.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file. Defaults to None if not provided.",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(os.getcwd(), "config.py"),  # Set default to 'config.py' in the current directory
        help="Path to the configuration file (Python script containing the experiment config). Defaults to './config.py'.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(os.getcwd(), "logs"),
        help="Directory where logs will be stored. Defaults to './logs'. Note: CLI arg takes precedence over config.paths.log_dir.",
    )

    parser.add_argument(
        "--sandbox_base_path",
        type=str,
        default=os.path.join(os.getcwd(), "sandbox"),
        help="Path to the sandbox directory. Defaults to './sandbox'. Note: CLI arg takes precedence over config.paths.sandbox_base_path.",
    )

########################################## Resources related arguments #############################################

    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling of evaluators and samplers (enabled by default). Note: Also configurable via config.scaling.enabled.",
    )

    parser.add_argument(
        "--check_interval",
        type=int,
        default=120,
        help="Time interval (in seconds) between consecutive scaling checks for evaluators and samplers. "
             "Defaults to 120s (2 minutes). Note: config.scaling.check_interval takes precedence over this CLI argument."
    )

    parser.add_argument(
        "--max_evaluators",
        type=int,
        default=1000,
        help="Maximum evaluators the system can scale up to. Adjust based on resource availability. "
             "Note: config.scaling.max_evaluators takes precedence if set to non-default value."
    )

    parser.add_argument(
        "--max_samplers",
        type=int,
        default=1000,
        help="Maximum samplers the system can scale up to. Adjust based on resource availability. "
             "Note: config.scaling.max_samplers takes precedence if set to non-default value."
    )

########################## Termination related arguments ###########################################

    parser.add_argument(
        "--prompt_limit",
        type=int,
        default=400_000_000,
        help="Maximum number of prompts that can be generated before stopping further publishing. "
             "The system will continue processing remaining queue messages. "
             "Note: config.termination.prompt_limit takes precedence if set to non-default value."
    )

    parser.add_argument(
        "--optimal_solution_programs",
        type=int,
        default=200_000,
        help="Number of additional programs to generate after the first optimal solution is found. "
             "Once this limit is reached further publishing stops but remaining queue messages continue processing. "
             "Note: config.termination.optimal_solution_programs takes precedence if set to non-default value."
    )

    parser.add_argument(
        "--target_solutions",
        type=str,
        default='{"(6, 1)": 10, "(7, 1)": 16, "(8, 1)": 30, "(9, 1)": 52, "(10, 1)": 94, "(11, 1)": 172}',
        help="JSON string specifying target solutions for (n, s_value) to terminate search early when reached. "
             "Set to empty dict to disable early termination based on optimal solutions."
    )

    args = parser.parse_args()

    # Load config first to get defaults
    config = process_utils.load_config(args.config_path)

    # Merge CLI args with config values (CLI takes precedence)
    # Paths: CLI overrides config
    log_dir = args.log_dir if args.log_dir != os.path.join(os.getcwd(), "logs") else config.paths.log_dir
    sandbox_base_path = args.sandbox_base_path if args.sandbox_base_path != os.path.join(os.getcwd(), "sandbox") else config.paths.sandbox_base_path
    backup_enabled = args.backup or config.paths.backup_enabled

    # Scaling: CLI overrides config
    enable_dynamic_scaling = not args.no_dynamic_scaling if args.no_dynamic_scaling else config.scaling.enabled
    max_evaluators = args.max_evaluators if args.max_evaluators != 1000 else config.scaling.max_evaluators
    max_samplers = args.max_samplers if args.max_samplers != 1000 else config.scaling.max_samplers

    # Termination: CLI overrides config
    prompt_limit = args.prompt_limit if args.prompt_limit != 400_000_000 else config.termination.prompt_limit
    optimal_solution_programs = args.optimal_solution_programs if args.optimal_solution_programs != 200_000 else config.termination.optimal_solution_programs

    # Target solutions: CLI overrides config
    try:
        if args.target_solutions != '{"(6, 1)": 10, "(7, 1)": 16, "(8, 1)": 30, "(9, 1)": 52, "(10, 1)": 94, "(11, 1)": 172}':
            # CLI arg was provided and is different from default
            target_signatures = json.loads(args.target_solutions)
            target_signatures = {eval(k): v for k, v in target_signatures.items()}  # Convert string keys to tuples
        else:
            # Use config value
            target_signatures = config.termination.target_solutions if config.termination.target_solutions else None

    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format for --target_solutions. Example: '{\"(6, 1)\": 8, \"(7, 1)\": 14, \"(8, 1)\": 25}'.") from e

    if backup_enabled:
        # Define the source directory (current working directory)
        src_dir = os.getcwd()

        # Define the base directory for backups
        backup_base_dir = '/mnt/hdd_pool/userdata/franziska/code_backups'
        os.makedirs(backup_base_dir, exist_ok=True)

        # Create a timestamped backup directory
        backup_dir = os.path.join(backup_base_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)

        # Call the backup function
        backup_python_files(src=src_dir, dest=backup_dir)

        # Log the backup operation
        print(f"Backup completed. Python files saved to: {backup_dir}")

    # Configure separate logger for time and memory logging
    time_memory_logger = logging.getLogger('time_memory_logger')
    time_memory_logger.setLevel(logging.INFO)

    os.makedirs(log_dir, exist_ok=True)

    time_memory_log_file = os.path.join(log_dir, 'time_memory.log')
    file_handler = logging.FileHandler(time_memory_log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    time_memory_logger.addHandler(file_handler)

    async def main():
        # Load the evaluation script from the absolute path
        from pathlib import Path

        eval_script_path = Path(config.evaluator.evaluation_script_path)

        try:
            with open(eval_script_path) as file:
                specification = file.read()
            if not isinstance(specification, str) or not specification.strip():
                raise ValueError("Specification must be a non-empty string.")

            # Substitute start_n placeholder with actual value from config
            # This allows hash computation to use the correct n value without manual sync
            actual_start_n = config.evaluator.start_n[0]  # Get first start_n value
            specification = specification.replace("n == start_n", f"n == {actual_start_n}")

        except FileNotFoundError:
            print(f"Error: Evaluation script not found at {eval_script_path}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error in evaluation script: {e}")
            sys.exit(1)

        if not (len(config.evaluator.s_values) == len(config.evaluator.start_n) == len(config.evaluator.end_n)):
            raise ValueError("The number of elements in --s-values, --start-n, and --end-n must match.")

        # Generate evaluation input tuples
        inputs = create_evaluation_inputs(config)

        # Initialize the task manager
        task_manager = TaskManager(
            specification=specification,
            inputs=inputs,
            config=config,
            log_dir=log_dir,
            target_signatures=target_signatures,
            config_path=args.config_path,
            sandbox_base_path=sandbox_base_path
        )
        main.task_manager = task_manager
        task = asyncio.create_task(
            task_manager.main_task(
                enable_scaling=enable_dynamic_scaling,
                checkpoint_file=args.checkpoint
            )
        )
        await task  # Ensure the task is awaited

    # helper for graceful-shutdown
    async def _shutdown(loop, signame):
        print(f"\nReceived {signame}. Shutting down gracefully…")

        # Cancel all async tasks first to stop scaling loop and other background tasks
        print("Cancelling all async tasks (database, scaling, etc.)...")
        for task in main.task_manager.tasks:
            if not task.done():
                task.cancel()

        # Wait briefly for tasks to acknowledge cancellation
        await asyncio.sleep(0.5)

        # Terminate child processes and give them a grace period
        children = (main.task_manager.evaluator_processes +
                    main.task_manager.sampler_processes +
                    main.task_manager.database_processes)

        print(f"Shutting down {len(children)} child processes...")
        for p in children:
            if p.is_alive():
                print(f"Sending SIGTERM to process {p.pid} ({p.name})")
                p.terminate()          # SIGTERM first

        # Phase 1: Give child processes 30 seconds to close connections gracefully
        print("Phase 1: Waiting up to 30s for graceful shutdown...")
        deadline = time.time() + 30
        last_count = len(children)
        while any(p.is_alive() for p in children) and time.time() < deadline:
            alive = [p for p in children if p.is_alive()]
            if len(alive) != last_count:
                print(f"  {len(alive)} processes still alive...")
                last_count = len(alive)
            await asyncio.sleep(0.5)

        # Phase 2: Hard cleanup - immediately force kill everything including descendants
        still_alive = [p for p in children if p.is_alive()]
        if still_alive:
            print(f"\nPhase 2: Hard cleanup, Force killing {len(still_alive)} remaining processes immediately")

            # First - recursively find and kill all descendants
            for p in still_alive:
                try:
                    parent = psutil.Process(p.pid)
                    descendants = parent.children(recursive=True)

                    # Kill descendants first (bottom-up)
                    for child in descendants:
                        try:
                            print(f"SIGKILL child process {child.pid}")
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    # Then kill the parent
                    print(f"SIGKILL process {p.pid} ({p.name})")
                    p.kill()
                except (psutil.NoSuchProcess, Exception) as e:
                    print(f"Error killing {p.pid}: {e}")

            # Reap zombies quickly without waiting
            print("Reaping killed processes...")
            for p in still_alive:
                try:
                    p.join(timeout=0.1)  # Very short timeout - just reap zombies
                except Exception as e:
                    print(f"Error joining process {p.pid}: {e}")

            print(f"Hard cleanup complete. {len(still_alive)} processes forcefully terminated.")
        else:
            print("All child processes terminated gracefully within 30s.")

        # Join all terminated processes to prevent zombies
        print("Cleaning up all terminated processes...")
        for p in children:
            if not p.is_alive():
                try:
                    p.join(timeout=0.1)  # Reap any remaining zombies
                except Exception:
                    pass

        # Close connections gracefully
        try:
            if main.task_manager.sampler_connection:
                await main.task_manager.sampler_connection.close()
        except Exception as e:
            print(f"Error closing sampler connection: {e}")

        try:
            if main.task_manager.database_connection:
                await main.task_manager.database_connection.close()
        except Exception as e:
            print(f"Error closing database connection: {e}")

        try:
            if main.task_manager.connection:
                await main.task_manager.connection.close()
        except Exception as e:
            print(f"Error closing main connection: {e}")

        # Clean up ResourceManager state
        if hasattr(main.task_manager, 'resource_manager') and main.task_manager.resource_manager:
            main.task_manager.resource_manager.cleanup()

        # Delete queues - wait for consumers to disconnect first to avoid race condition
        print("Attempting to delete RabbitMQ queues...")
        try:
            cleanup_connection = await process_utils.create_rabbitmq_connection(
                main.task_manager.config, timeout=10
            )
            cleanup_channel = await cleanup_connection.channel()

            for queue_name in ['evaluator_queue', 'sampler_queue', 'database_queue']:
                try:
                    # First check queue status
                    queue = await cleanup_channel.declare_queue(
                        queue_name, durable=False, auto_delete=False, passive=True
                    )
                    # Wait briefly if there are still consumers (they should be disconnecting)
                    if queue.declaration_result.consumer_count > 0:
                        print(f"Queue {queue_name} has {queue.declaration_result.consumer_count} consumers, waiting...")
                        await asyncio.sleep(2)

                    # Delete - prefer if_unused=True but fall back to force delete
                    try:
                        await queue.delete(if_unused=True, if_empty=False)
                    except Exception:
                        await queue.delete(if_unused=False, if_empty=False)
                    print(f"Deleted queue: {queue_name}")
                except aio_pika.exceptions.ChannelNotFoundEntity:
                    print(f"Queue {queue_name} does not exist, skipping")
                except Exception as e:
                    print(f"Could not delete queue {queue_name}: {e}")

            await cleanup_channel.close()
            await cleanup_connection.close()
            print("Queue cleanup completed.")
        except Exception as e:
            print(f"Warning: Could not perform queue cleanup: {e}")

        # Wait for RabbitMQ internal tasks to complete
        await asyncio.sleep(2.0)

        # Cancel all still-pending asyncio tasks
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        # Wait for tasks to complete cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            # Give extra time for cleanup to complete
            await asyncio.sleep(1.0)

        # NOW stop the loop after all cleanup is done
        loop.stop()


    # run the loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Use list to allow modification in nested function
    shutdown_state = {'task': None}

    def handle_shutdown(signame):
        if shutdown_state['task'] is None:
            shutdown_state['task'] = asyncio.create_task(_shutdown(loop, signame))

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda s=sig: handle_shutdown(s.name)
        )

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        # If shutdown task was created, wait for it to complete
        shutdown_task = shutdown_state['task']
        if shutdown_task and not shutdown_task.done():
            try:
                print("Waiting for shutdown to complete...")
                loop.run_until_complete(shutdown_task)
            except Exception as e:
                print(f"Error during shutdown: {e}")

        # Now cancel any remaining tasks
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    if task is not shutdown_task:  # Don't cancel shutdown task
                        task.cancel()
                loop.run_until_complete(asyncio.wait(pending, timeout=5.0))
        except Exception as e:
            print(f"Error during final cleanup: {e}")

        loop.close()
        sys.exit(0)

