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

"""Main entry point for DistributedFunSearch experiments.

This is a distributed implementation of FunSearch, adapted from DeepMind's original
single-threaded version. It uses RabbitMQ and asyncio for asynchronous message passing,
enabling parallel evaluation and sampling across multiple processes and nodes.

Runs ProgramsDatabase in the main process and spawns Sampler/Evaluator as child
processes. Child process entry points live in startup.py because multiprocessing
with 'spawn' context can only pickle functions from importable modules.
"""

import os

# Load environment variables from .env file (for API keys like OPENAI_API_KEY)
# Must be called early, before other imports that may read env vars
from dotenv import load_dotenv
load_dotenv()
import re
import sys
import ast
import time
import glob
import random
import dataclasses

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
import atexit

from disfun import programs_database
from disfun.utils import rabbitmq
from disfun.sandbox import cleanup_orphaned_sandbox_processes
from disfun.utils import code_manipulation, prompt_builder, wandb_logging
from disfun.utils import checkpointing as checkpoint
from disfun.utils.resource_manager import ResourceManager, ScalingContext
from disfun.startup import sampler_process_entry, evaluator_process_entry, load_config, initialize_logger, load_initial_programs, get_gpu_count

# Disable multi-threaded tokenization.
# Our prompts are short and we run many parallel processes so single-threaded tokenization is faster
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _cleanup_sandbox_processes():
    """No-op. Fork-based sandbox children are reaped automatically by parent.

    Kept for API compatibility.
    """
    cleanup_orphaned_sandbox_processes(max_age_seconds=0)


def _cleanup_orphaned_vllm_workers():
    """Kill orphaned multiprocessing.spawn workers."""
    for proc in psutil.process_iter(['cmdline']):
        try:
            if proc.info['cmdline'] and 'multiprocessing.spawn' in ' '.join(proc.info['cmdline']):
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            pass


atexit.register(_cleanup_sandbox_processes)
atexit.register(_cleanup_orphaned_vllm_workers)


# Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be called before any multiprocessing to avoid CUDA context conflicts
# Required to prevent fork+threading deadlocks when dynamically scaling samplers
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


def backup_python_files(src, dest, exclude_dirs=[]):
    """Recursively copy all Python files in src to dest."""
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run FunSearch experiment.")

    # General settings
    parser.add_argument(
        "--backup", action="store_true",
        help="Enable backup of Python files before running.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file to resume from.")
    parser.add_argument(
        "--config-path", type=str,
        default=os.path.join(os.getcwd(), "config.py"),
        help="Path to configuration file. Defaults to './config.py'.")
    parser.add_argument(
        "--log-dir", type=str,
        default=os.path.join(os.getcwd(), "logs"),
        help="Directory for logs. Defaults to './logs'.")
    # Resource scaling
    parser.add_argument(
        "--no-dynamic-scaling", action="store_true",
        help="Disable dynamic scaling of evaluators and samplers.")
    parser.add_argument(
        "--check_interval", type=int, default=120,
        help="Seconds between scaling checks. Defaults to 120.")
    parser.add_argument(
        "--max_evaluators", type=int, default=1000,
        help="Maximum evaluators to scale up to.")
    parser.add_argument(
        "--max_samplers", type=int, default=1000,
        help="Maximum samplers to scale up to.")

    # Termination
    parser.add_argument(
        "--iteration_limit", type=int, default=400_000_000,
        help="Maximum iterations before shutdown.")
    parser.add_argument(
        "--optimal_solution_programs", type=int, default=200_000,
        help="Programs to generate after first optimal solution.")
    parser.add_argument(
        "--target_solutions", type=str,
        default='{"(6, 1)": 10, "(7, 1)": 16, "(8, 1)": 30, "(9, 1)": 52, "(10, 1)": 94, "(11, 1)": 172}',
        help="JSON dict of (n, s) target solutions for early termination.")
    parser.add_argument(
        "--stop_on_optimal", action="store_true", dest="stop_on_optimal", default=None,
        help="Stop early after finding optimal solution.")
    parser.add_argument(
        "--no_stop_on_optimal", action="store_false", dest="stop_on_optimal",
        help="Continue running even after finding optimal solution.")

    # Attach mode
    parser.add_argument(
        "--attach", type=str, choices=["evaluators", "samplers"], default=None,
        help="Attach workers to running experiment instead of starting full experiment.")

    # Worker counts (override config)
    parser.add_argument(
        "--num_evaluators", type=int, default=None,
        help="Number of evaluators to start (overrides config).")
    parser.add_argument(
        "--num_samplers", type=int, default=None,
        help="Number of samplers to start (overrides config).")

    return parser.parse_args()


def merge_config_with_args(args, config):
    """Merge CLI arguments with config. CLI overrides config when explicitly set."""
    # Paths
    default_log_dir = os.path.join(os.getcwd(), "logs")
    log_dir = args.log_dir if args.log_dir != default_log_dir else config.paths.log_dir
    backup_enabled = args.backup or config.paths.backup_enabled

    # Scaling
    enable_dynamic_scaling = not args.no_dynamic_scaling if args.no_dynamic_scaling else config.scaling.enabled

    # Termination values
    iteration_limit = args.iteration_limit if args.iteration_limit != 400_000_000 else config.termination.iteration_limit
    optimal_solution_programs = args.optimal_solution_programs if args.optimal_solution_programs != 200_000 else config.termination.optimal_solution_programs
    stop_on_optimal = args.stop_on_optimal if args.stop_on_optimal is not None else config.termination.stop_on_optimal

    # Target solutions
    default_targets = '{"(6, 1)": 10, "(7, 1)": 16, "(8, 1)": 30, "(9, 1)": 52, "(10, 1)": 94, "(11, 1)": 172}'
    if args.target_solutions != default_targets:
        try:
            target_signatures = json.loads(args.target_solutions)
            target_signatures = {ast.literal_eval(k): v for k, v in target_signatures.items()}
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON for --target_solutions") from e
    else:
        target_signatures = config.termination.target_solutions or None

    termination_config = dataclasses.replace(
        config.termination,
        iteration_limit=iteration_limit,
        stop_on_optimal=stop_on_optimal,
        optimal_solution_programs=optimal_solution_programs,
        target_solutions=target_signatures
    )

    # Evaluation inputs - graphs are preloaded by each evaluator during init
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

    return {
        'log_dir': log_dir,
        'backup_enabled': backup_enabled,
        'enable_dynamic_scaling': enable_dynamic_scaling,
        'termination_config': termination_config,
        'target_signatures': target_signatures,
        'inputs': inputs,
    }


async def terminate_child_processes(task_manager):
    """Terminate child processes (evaluators and samplers) with graceful shutdown then force kill."""
    children = task_manager.evaluator_processes + task_manager.sampler_processes
    if not children:
        return

    # Capture descendants before terminating parents (they become orphans after)
    all_descendants = []
    for p in children:
        if p.is_alive():
            try:
                all_descendants.extend(psutil.Process(p.pid).children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    print(f"Terminating {len(children)} processes (+{len(all_descendants)} descendants)...")
    for p in children:
        if p.is_alive():
            p.terminate()

    # Wait up to 30s for graceful shutdown
    deadline = time.time() + 30
    while any(p.is_alive() for p in children) and time.time() < deadline:
        await asyncio.sleep(0.5)

    # Force kill anything still alive
    still_alive = [p for p in children if p.is_alive()]
    still_alive_descendants = [d for d in all_descendants if d.is_running()]

    for d in still_alive_descendants:
        try:
            d.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    for p in still_alive:
        try:
            p.kill()
        except Exception:
            pass

    # Reap zombies
    for p in children:
        try:
            p.join(timeout=0.1)
        except Exception:
            pass

    if still_alive:
        print(f"Force killed {len(still_alive)} processes")
    else:
        print("All processes terminated gracefully")


async def cleanup_rabbitmq(task_manager):
    """Close RabbitMQ connections and delete queues."""
    try:
        if task_manager.rabbitmq_manager:
            await task_manager.rabbitmq_manager.close_all()
    except Exception as e:
        print(f"Error closing RabbitMQManager: {e}")

    try:
        if task_manager.main_connection:
            await task_manager.main_connection.close()
    except Exception as e:
        print(f"Error closing main connection: {e}")

    if hasattr(task_manager, 'resource_manager') and task_manager.resource_manager:
        task_manager.resource_manager.cleanup()

    # Delete queues
    try:
        conn = await rabbitmq.create_connection(task_manager.config, timeout=10)
        channel = await conn.channel()

        for queue_name in ['evaluator_queue', 'sampler_queue', 'database_queue']:
            try:
                queue = await channel.declare_queue(queue_name, durable=False, auto_delete=False, passive=True)
                if queue.declaration_result.consumer_count > 0:
                    await asyncio.sleep(2)
                await queue.delete(if_unused=False, if_empty=False)
                print(f"Deleted queue: {queue_name}")
            except aio_pika.exceptions.ChannelNotFoundEntity:
                pass
            except Exception as e:
                print(f"Could not delete {queue_name}: {e}")

        await channel.close()
        await conn.close()
    except Exception as e:
        print(f"Queue cleanup failed: {e}")


class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config, log_dir, termination_config, config_path, attach_mode=None):
        self.template = code_manipulation.text_to_program(specification)
        self.inputs = inputs
        self.config = config
        self.config_path = config_path
        self.log_dir = log_dir
        self.attach_mode = attach_mode  # None=full, "evaluators"=evaluators only, "samplers"=samplers only

        pid = os.getpid()
        log_prefix = f'attach_{attach_mode}' if attach_mode else 'main'
        self.log_filename = f'{log_prefix}_pid{pid}.log'
        self.logger = initialize_logger(log_dir, self.log_filename, process_type="Main")

        self.evaluator_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.main_connection = None
        self.main_channel = None
        self.rabbitmq_manager = None
        self.resource_manager = ResourceManager(log_dir=log_dir, scaling_config=self.config.scaling)
        self.process_to_device_map = {}
        self.termination_config = termination_config
        self.shutting_down = False

    async def publish_initial_program_with_retry(self, initial_program_data, max_retries=5, delay=5):
        attempt = 0
        while attempt < max_retries:
            try:
                sampler_connection = await rabbitmq.create_connection(
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

    async def monitor_evaluator_health(self, check_interval=60):
        """Monitor evaluator processes and respawn any that have crashed.

        This runs even when dynamic scaling is disabled, ensuring crashed
        evaluators are restarted to maintain throughput.
        """
        startup_delay = getattr(self.config.evaluator, 'startup_delay', 0)
        ctx = mp.get_context('spawn')

        while not self.shutting_down:
            await asyncio.sleep(check_interval)

            if self.shutting_down:
                break

            # Check each evaluator process
            dead_evaluators = []
            for i, proc in enumerate(self.evaluator_processes):
                if not proc.is_alive():
                    dead_evaluators.append((i, proc))

            if not dead_evaluators:
                continue

            for _, proc in dead_evaluators:
                ec = proc.exitcode
                if ec is None:
                    reason = "unknown (still running?)"
                elif ec < 0:
                    reason = f"killed by {signal.Signals(-ec).name} (unhandled)"
                elif ec == 0:
                    reason = "normal exit (consume loop finished or reconnects exhausted)"
                elif ec == 143:
                    reason = "received SIGTERM (handled gracefully)"
                elif ec == 130:
                    reason = "received SIGINT (handled gracefully)"
                else:
                    reason = f"exitcode {ec}"
                self.logger.warning(f"Dead evaluator PID={proc.pid}: {reason}")
            self.logger.warning(f"Detected {len(dead_evaluators)} dead evaluator(s), respawning...")

            for i, old_proc in dead_evaluators:
                if self.shutting_down:
                    break

                try:
                    proc = ctx.Process(
                        target=evaluator_process_entry,
                        args=(self.config_path, self.template, self.inputs,
                              self.termination_config.target_solutions,
                              self.log_dir, self.log_filename),
                        name=f"Evaluator-respawn-{old_proc.pid}"
                    )
                    proc.start()

                    # Update tracking
                    self.evaluator_processes[i] = proc
                    self.logger.info(f"Respawned evaluator PID={proc.pid} (was {old_proc.pid})")

                    # Stagger restarts to avoid memory spike during graph loading
                    if startup_delay > 0 and len(dead_evaluators) > 1:
                        await asyncio.sleep(startup_delay)

                except Exception as e:
                    self.logger.error(f"Failed to respawn evaluator: {e}")

    async def monitor_sampler_health(self, check_interval=60):
        """Monitor sampler processes and respawn any that have crashed.

        This runs even when dynamic scaling is disabled, ensuring crashed
        samplers are restarted to maintain throughput.
        """
        while not self.shutting_down:
            await asyncio.sleep(check_interval)

            if self.shutting_down:
                break

            # Check each sampler process
            dead_samplers = []
            for i, proc in enumerate(self.sampler_processes):
                if not proc.is_alive():
                    device = self.process_to_device_map.get(proc.pid)
                    dead_samplers.append((i, proc, device))

            if not dead_samplers:
                continue

            self.logger.warning(f"Detected {len(dead_samplers)} dead sampler(s), respawning...")

            use_local = self.config.sampler.use_local_vllm
            ctx = mp.get_context('spawn')

            for i, old_proc, device in dead_samplers:
                if self.shutting_down:
                    break

                # Get next sampler ID from resource manager
                sampler_id = self.resource_manager.next_sampler_id
                self.resource_manager.next_sampler_id += 1

                try:
                    proc = ctx.Process(
                        target=sampler_process_entry,
                        args=(self.config_path, device, self.log_dir, self.log_filename, sampler_id),
                        name=f"Sampler-{sampler_id}"
                    )
                    proc.start()

                    # Update tracking
                    self.sampler_processes[i] = proc
                    if device:
                        self.process_to_device_map[proc.pid] = device
                    # Clean up old mapping
                    if old_proc.pid in self.process_to_device_map:
                        del self.process_to_device_map[old_proc.pid]

                    self.logger.info(f"Respawned sampler (ID={sampler_id}) with PID={proc.pid} on device {device}")

                    # Stagger restarts to avoid simultaneous model loading
                    if use_local and len(dead_samplers) > 1:
                        await asyncio.sleep(90)

                except Exception as e:
                    self.logger.error(f"Failed to respawn sampler: {e}")

    def _load_initial_programs(self):
        """Load initial functions from the initial_functions directory."""
        return load_initial_programs(
            self.config.evaluator.initial_functions_dir,
            logger=self.logger
        )

    def _determine_run_name(self, checkpoint_file):
        """Determine run name from checkpoint, config, or auto-generate."""
        # Try to extract from checkpoint path first
        if checkpoint_file:
            path_match = re.search(r'/checkpoint_(run_\d{8}_\d{6})/', checkpoint_file)
            if path_match:
                self.logger.info(f"Extracted run name from checkpoint path: {path_match.group(1)}")
                return path_match.group(1)

            # Try loading from checkpoint file
            try:
                with open(checkpoint_file, 'rb') as f:
                    run_name = pickle.load(f).get('wandb_run_name')
                    if run_name:
                        self.logger.info(f"Loaded run name from checkpoint: {run_name}")
                        return run_name
            except Exception as e:
                self.logger.warning(f"Could not extract run name from checkpoint: {e}")

        # Use config or auto-generate
        if self.config.wandb.run_name:
            self.logger.info(f"Using configured run name: {self.config.wandb.run_name}")
            return self.config.wandb.run_name

        run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.config.wandb.run_name_tag:
            run_name = f"{run_name}_{self.config.wandb.run_name_tag}"
        self.logger.info(f"Auto-generated run name: {run_name}")
        return run_name

    def _create_scaling_context(self, evaluator_queue, sampler_queue, for_attach_mode=False):
        """Create ScalingContext for dynamic scaling."""
        if for_attach_mode:
            is_eval = self.attach_mode == "evaluators"
            return ScalingContext(
                config=self.config,
                config_path=self.config_path,
                log_dir=self.log_dir,
                log_filename=self.log_filename,
                evaluator_queue=evaluator_queue if is_eval else None,
                sampler_queue=None if is_eval else sampler_queue,
                evaluator_processes=self.evaluator_processes if is_eval else [],
                sampler_processes=[] if is_eval else self.sampler_processes,
                sampler_entry_function=None if is_eval else sampler_process_entry,
                evaluator_entry_function=evaluator_process_entry if is_eval else None,
                template=self.template if is_eval else None,
                inputs=self.inputs if is_eval else None,
                target_signatures=self.termination_config.target_solutions if is_eval else None,
                max_evaluators=self.config.scaling.max_evaluators if is_eval else 0,
                max_samplers=0 if is_eval else self.config.scaling.max_samplers,
                check_interval=self.config.scaling.check_interval,
            )

        scaling = self.config.scaling if hasattr(self.config, 'scaling') and self.config.scaling else None
        return ScalingContext(
            config=self.config,
            config_path=self.config_path,
            log_dir=self.log_dir,
            log_filename=self.log_filename,
            evaluator_queue=evaluator_queue,
            sampler_queue=sampler_queue,
            evaluator_processes=self.evaluator_processes,
            sampler_processes=self.sampler_processes,
            sampler_entry_function=sampler_process_entry,
            evaluator_entry_function=evaluator_process_entry,
            template=self.template,
            inputs=self.inputs,
            target_signatures=self.termination_config.target_solutions,
            max_evaluators=scaling.max_evaluators if scaling else 1000,
            max_samplers=scaling.max_samplers if scaling else 1000,
            check_interval=scaling.check_interval if scaling else 120,
        )

    async def _run_attach_mode(self, evaluator_queue, sampler_queue, enable_scaling):
        """Run in attach mode: only start workers, no database/checkpoint/wandb."""
        self.logger.info(f"Attach mode: starting {self.attach_mode} only")
        self.start_initial_processes(starting_sampler_id=0)
        self.logger.info(f"Started {self.attach_mode} processes")

        self.tasks = [asyncio.create_task(self.resource_manager.log_resource_stats_periodically())]

        # Respawn dead evaluators/samplers (runs regardless of scaling setting)
        if self.attach_mode != "samplers":
            self.tasks.append(asyncio.create_task(self.monitor_evaluator_health(check_interval=60)))
        if self.attach_mode != "evaluators":
            self.tasks.append(asyncio.create_task(self.monitor_sampler_health(check_interval=60)))

        # Publish resource stats to main node for W&B logging
        resource_stats_interval = getattr(self.config.scaling, 'resource_log_interval', 60)
        self.tasks.append(asyncio.create_task(
            wandb_logging.publish_resource_stats_periodically(self.config, resource_stats_interval)
        ))

        if enable_scaling:
            scaling_ctx = self._create_scaling_context(evaluator_queue, sampler_queue, for_attach_mode=True)
            self.tasks.append(asyncio.create_task(self.resource_manager.run_scaling_loop(scaling_ctx)))

        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _wait_for_sampler_and_publish(self, sampler_queue, database, initial_programs, checkpoint_file):
        """Wait for at least one sampler to connect, then publish initial programs or load checkpoint."""
        self.logger.info("Waiting for at least one sampler to connect...")
        while True:
            queue_info = await sampler_queue.declare()
            consumer_count = queue_info.consumer_count
            self.logger.info(f"consumer_count is {consumer_count} while config num_samplers is {self.config.num_samplers}")

            if consumer_count >= 1 and checkpoint_file is None:
                num_copies = getattr(self.config.programs_database, 'initial_program_copies', 1)
                all_publications = initial_programs * num_copies
                if all_publications:
                    await asyncio.gather(*[
                        self.publish_initial_program_with_retry(prog_data)
                        for prog_data in all_publications
                    ])
                    self.logger.info(f"Published {len(all_publications)} initial programs ({len(initial_programs)} unique x {num_copies} copies)")
                break
            elif consumer_count >= 1:
                # Wait for database connection before calling get_prompt
                while database._conn.channel is None:
                    self.logger.info("Waiting for database RabbitMQ connection...")
                    await asyncio.sleep(0.1)
                await database.get_prompt()
                self.logger.info(f"Loading from checkpoint: {checkpoint_file}")
                break
            else:
                self.logger.info("No consumers yet on sampler_queue. Retrying in 10 seconds...")
                await asyncio.sleep(10)

    async def _run_full_experiment(self, evaluator_queue, sampler_queue, enable_scaling, checkpoint_file, run_name, save_checkpoints_path, initial_programs):
        """Run full experiment with database, checkpoint, wandb, and all workers."""
        function_to_evolve = 'priority'

        prompt_spec = prompt_builder.load_prompt_spec_from_config(self.config)
        self.logger.info(f"Loaded prompt spec: strategy={prompt_spec.strategy.value}, templates={list(prompt_spec.templates.keys())}")

        db_connection_manager = self.rabbitmq_manager.get_connection(
            "ProgramsDatabase",
            ["database_queue", "sampler_queue", "evaluator_queue"]
        )

        database = programs_database.ProgramsDatabase(
            config=self.config.programs_database,
            function_to_evolve=function_to_evolve,
            connection_manager=db_connection_manager,
            checkpoint_file=checkpoint_file,
            save_checkpoints_path=save_checkpoints_path,
            termination_config=self.termination_config,
            best_known_solutions=self.config.prompt.best_known_solutions,
            wandb_config=self.config.wandb,
            sampler_config=self.config.sampler,
            evaluator_config=self.config.evaluator,
            prompt_spec=prompt_spec,
            run_name=run_name,
        )

        database_task = asyncio.create_task(database.consume_and_process())
        checkpoint_task = asyncio.create_task(checkpoint.periodic_checkpoint(database))
        wandb_logging_task = asyncio.create_task(
            wandb_logging.periodic_wandb_logging(database, self.rabbitmq_manager)
        )

        # Start workers
        starting_sampler_id = database.next_sampler_id if checkpoint_file else 0
        if checkpoint_file:
            self.logger.info(f"Resuming from checkpoint: starting sampler IDs at {starting_sampler_id}")
        self.start_initial_processes(starting_sampler_id)
        self.logger.info("Initial processes started successfully.")

        # Sync sampler ID counter
        if database.next_sampler_id > self.resource_manager.next_sampler_id:
            self.resource_manager.next_sampler_id = database.next_sampler_id
        self.resource_manager.database = database

        # Wait for sampler and publish
        await self._wait_for_sampler_and_publish(sampler_queue, database, initial_programs, checkpoint_file)

        # Start background tasks
        self.tasks = [
            database_task, checkpoint_task, wandb_logging_task,
            asyncio.create_task(self.resource_manager.log_resource_stats_periodically()),
            asyncio.create_task(self.monitor_sampler_health(check_interval=60)),
            asyncio.create_task(self.monitor_evaluator_health(check_interval=60)),
            # Consume resource stats from remote attach nodes for W&B aggregation
            asyncio.create_task(wandb_logging.consume_resource_stats(self.config)),
        ]

        if enable_scaling:
            try:
                scaling_ctx = self._create_scaling_context(evaluator_queue, sampler_queue)
                self.tasks.append(asyncio.create_task(self.resource_manager.run_scaling_loop(scaling_ctx)))
            except Exception as e:
                self.logger.error(f"Error enabling scaling: {e}")

        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def main_task(self, enable_scaling=True, checkpoint_file=None):
        """Main orchestration: setup RabbitMQ, then run attach mode or full experiment."""
        run_name = self._determine_run_name(checkpoint_file)
        save_checkpoints_path = os.path.join(
            self.config.wandb.checkpoints_base_path, f"checkpoint_{run_name}"
        )
        self.logger.info(f"Checkpoints will be saved to: {save_checkpoints_path}")

        # Initialize random seed
        if checkpoint_file is None and self.config.random_seed is not None:
            import numpy as np
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            self.logger.info(f"Initialized random seed: {self.config.random_seed}")
        elif checkpoint_file:
            self.logger.info("Random state will be restored from checkpoint")

        # Test RabbitMQ connectivity
        try:
            test_connection = await rabbitmq.create_connection(self.config, timeout=300)
            await test_connection.close()
        except Exception as e:
            self.logger.error(f"Cannot connect to RabbitMQ: {e}")
            raise

        self.logger.info(f"Main_task running with PID: {os.getpid()}")
        initial_programs = self._load_initial_programs() if checkpoint_file is None else []

        try:
            # Setup RabbitMQ
            self.rabbitmq_manager = rabbitmq.RabbitMQManager(self.config, logger=self.logger)
            self.logger.info("Created central RabbitMQManager")

            self.main_connection = await rabbitmq.create_connection(self.config, timeout=300)
            self.main_channel = await self.main_connection.channel()
            evaluator_queue = await rabbitmq.declare_queue(self.main_channel, "evaluator_queue")
            sampler_queue = await rabbitmq.declare_queue(self.main_channel, "sampler_queue")

            if self.attach_mode:
                await self._run_attach_mode(evaluator_queue, sampler_queue, enable_scaling)
            else:
                await self._run_full_experiment(
                    evaluator_queue, sampler_queue, enable_scaling,
                    checkpoint_file, run_name, save_checkpoints_path, initial_programs
                )

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")


    def start_initial_processes(self, starting_sampler_id=0):
        ctx = mp.get_context('spawn')
        next_sampler_id = starting_sampler_id

        # Start samplers (skip if attach_mode == "evaluators")
        if self.attach_mode != "evaluators":
            use_local = self.config.sampler.use_local_vllm

            if use_local:
                tp_size = int(self.config.sampler.tensor_parallel_size)
                gpus_needed = self.config.num_samplers * tp_size
                total_gpus = get_gpu_count()

                if gpus_needed > total_gpus:
                    raise RuntimeError(
                        f"Not enough GPUs: need {gpus_needed} ({self.config.num_samplers} samplers × {tp_size} tp), "
                        f"but only {total_gpus} available."
                    )

                self.logger.info(
                    f"Starting {self.config.num_samplers} sampler(s) with LOCAL model: {self.config.sampler.model} "
                    f"(tp={tp_size}, using {gpus_needed}/{total_gpus} GPUs)"
                )

                for i in range(self.config.num_samplers):
                    sampler_id = next_sampler_id
                    next_sampler_id += 1
                    base_gpu = sampler_id * tp_size

                    # GPU assignment handled in startup.py based on sampler_id
                    proc = ctx.Process(
                        target=sampler_process_entry,
                        args=(self.config_path, None, self.log_dir, self.log_filename, sampler_id),
                        name=f"Sampler-{sampler_id}"
                    )
                    proc.start()
                    self.logger.info(f"Started Sampler (ID={sampler_id}) PID={proc.pid} on GPUs {base_gpu}-{base_gpu + tp_size - 1}")
                    self.sampler_processes.append(proc)
                    gpu_ids = ",".join(str(base_gpu + g) for g in range(tp_size))
                    self.process_to_device_map[proc.pid] = gpu_ids  # Track GPU(s) for respawn

                    if i < self.config.num_samplers - 1:
                        time.sleep(90)
            else:
                self.logger.info(f"Starting {self.config.num_samplers} sampler(s) with API model: {self.config.sampler.model}")

                for _ in range(self.config.num_samplers):
                    sampler_id = next_sampler_id
                    next_sampler_id += 1
                    proc = ctx.Process(
                        target=sampler_process_entry,
                        args=(self.config_path, None, self.log_dir, self.log_filename, sampler_id),
                        name=f"Sampler-{sampler_id}"
                    )
                    proc.start()
                    self.logger.info(f"Started Sampler (ID={sampler_id}) PID={proc.pid}")
                    self.sampler_processes.append(proc)

            self.resource_manager.next_sampler_id = next_sampler_id

        # Start evaluators (skip if attach_mode == "samplers")
        if self.attach_mode == "samplers":
            return

        startup_delay = getattr(self.config.evaluator, 'startup_delay', 0)
        for i in range(self.config.num_evaluators):
            proc = ctx.Process(
                target=evaluator_process_entry,
                args=(self.config_path, self.template, self.inputs, self.termination_config.target_solutions, self.log_dir, self.log_filename),
                name=f"Evaluator-{i}"
            )
            proc.start()
            self.logger.debug(f"Started Evaluator {i} PID={proc.pid}")
            self.evaluator_processes.append(proc)

            # Stagger evaluator starts to avoid memory spike during graph loading
            if startup_delay > 0 and i < self.config.num_evaluators - 1:
                time.sleep(startup_delay)


if __name__ == "__main__":
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config_path)

    # Merge CLI args with config (CLI overrides config when explicitly set)
    merged = merge_config_with_args(args, config)

    # Override worker counts if specified
    if args.num_evaluators is not None:
        config.num_evaluators = args.num_evaluators
    if args.num_samplers is not None:
        config.num_samplers = args.num_samplers

    log_dir = merged['log_dir']
    enable_dynamic_scaling = merged['enable_dynamic_scaling']
    termination_config = merged['termination_config']
    target_signatures = merged['target_signatures']
    inputs = merged['inputs']

    # Optional backup
    if merged['backup_enabled']:
        backup_base_dir = config.paths.backup_dir
        os.makedirs(backup_base_dir, exist_ok=True)
        backup_dir = os.path.join(backup_base_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)
        backup_python_files(src=os.getcwd(), dest=backup_dir)
        print(f"Backup completed: {backup_dir}")

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

        # Check if throughput measurement mode is enabled
        if hasattr(config, 'throughput') and config.throughput.enabled:
            from disfun.throughput import run_throughput, run_sweep, _generate_sweep_combinations

            # Check if sweep mode (has sweep parameters configured)
            has_sweep = hasattr(config, 'sweep') and config.sweep and _generate_sweep_combinations(config)

            if has_sweep:
                print("\n" + "=" * 60)
                print("THROUGHPUT SWEEP MODE")
                print("=" * 60)
                await run_sweep(
                    config=config,
                    config_path=args.config_path,
                    log_dir=log_dir,
                    specification=specification,
                    inputs=inputs,
                    target_signatures=target_signatures,
                )
            else:
                print("\n" + "=" * 60)
                print("THROUGHPUT MEASUREMENT MODE")
                print("=" * 60)
                print(f"Samplers: {config.num_samplers}, Evaluators: {config.num_evaluators}")
                print(f"Duration: {config.throughput.warmup_minutes} min warmup + {config.throughput.run_duration_minutes} min measurement")
                print("=" * 60 + "\n")

                await run_throughput(
                    config=config,
                    config_path=args.config_path,
                    log_dir=log_dir,
                    specification=specification,
                    inputs=inputs,
                    target_signatures=target_signatures,
                )
            return

        # Initialize the task manager
        task_manager = TaskManager(
            specification=specification,
            inputs=inputs,
            config=config,
            log_dir=log_dir,
            termination_config=termination_config,
            config_path=args.config_path,
            attach_mode=args.attach,
        )
        main.task_manager = task_manager
        task = asyncio.create_task(
            task_manager.main_task(
                enable_scaling=enable_dynamic_scaling,
                checkpoint_file=args.checkpoint
            )
        )
        await task  # Ensure the task is awaited

    async def _shutdown(loop, signame):
        """Graceful shutdown handler."""
        print(f"\nReceived {signame}. Shutting down...")

        if not hasattr(main, 'task_manager') or main.task_manager is None:
            loop.stop()
            return

        task_manager = main.task_manager
        task_manager.shutting_down = True

        # Cancel async tasks
        for task in task_manager.tasks:
            if not task.done():
                task.cancel()
        await asyncio.sleep(0.5)

        # Terminate child processes
        await terminate_child_processes(task_manager)

        # Kill orphaned sandbox processes
        _cleanup_sandbox_processes()

        # Kill orphaned vLLM workers
        _cleanup_orphaned_vllm_workers()

        # Clean up RabbitMQ
        await cleanup_rabbitmq(task_manager)

        # Cancel remaining asyncio tasks
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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

