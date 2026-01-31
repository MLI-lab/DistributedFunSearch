"""Throughput measurement for DistributedFunSearch."""

import os
import json
import time
import asyncio
import dataclasses
from datetime import datetime

import numpy as np
import aio_pika
import wandb
import torch.multiprocessing as mp

from disfun.utils import code_manipulation, prompt_builder
from disfun.sandbox import cleanup_orphaned_sandbox_processes
from disfun.utils.wandb_logging import convert_tuple_keys
from disfun.utils.resource_manager import ResourceManager
from disfun import programs_database
from disfun.utils import rabbitmq
from disfun.startup import sampler_process_entry, evaluator_process_entry, initialize_logger, load_initial_programs


class ThroughputRunner:
    """Measures throughput (iterations/hour) for a single configuration."""

    def __init__(self, config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures, checkpoint_path=None):
        self.config = config
        self.throughput_config = config.throughput
        self.config_path = config_path
        self.log_dir = log_dir
        self.sandbox_base_path = sandbox_base_path
        self.inputs = inputs
        self.target_signatures = target_signatures
        self.template = code_manipulation.text_to_program(specification)
        self.checkpoint_path = checkpoint_path

        self.log_filename = f"throughput_pid{os.getpid()}.log"
        self.logger = initialize_logger(log_dir, self.log_filename, process_type="Throughput")

        self.sampler_processes = []
        self.evaluator_processes = []
        self.connections = []
        self.database = None

    async def get_queue_depths(self):
        """Get current message counts for all queues."""
        depths = {"sampler_queue": 0, "evaluator_queue": 0, "database_queue": 0}
        try:
            conn = await rabbitmq.create_connection(self.config, timeout=5)
            ch = await conn.channel()
            for q_name in depths.keys():
                try:
                    queue = await ch.declare_queue(q_name, durable=False, auto_delete=False, passive=True)
                    depths[q_name] = queue.declaration_result.message_count
                except Exception:
                    pass  # Queue might not exist yet
            await ch.close()
            await conn.close()
        except Exception:
            pass  # Connection failed, return zeros
        return depths

    async def check_queue_state(self):
        """Check and log state of all queues before starting. Returns True if clean, False if dirty."""
        self.logger.info("Checking queue state before starting...")
        is_clean = True
        try:
            conn = await rabbitmq.create_connection(self.config, timeout=10)
            ch = await conn.channel()
            for q_name in ["evaluator_queue", "sampler_queue", "database_queue"]:
                try:
                    queue = await ch.declare_queue(q_name, durable=False, auto_delete=False, passive=True)
                    msg_count = queue.declaration_result.message_count
                    consumer_count = queue.declaration_result.consumer_count
                    if msg_count > 0 or consumer_count > 0:
                        self.logger.warning(f"Queue '{q_name}' not empty: {msg_count} messages, {consumer_count} consumers")
                        is_clean = False
                    else:
                        self.logger.info(f"Queue '{q_name}' exists (0 messages, 0 consumers)")
                except Exception as e:
                    err_str = str(e)
                    if "NOT_FOUND" in err_str or "timeout" in err_str.lower() or "closed" in err_str.lower():
                        self.logger.info(f"Queue '{q_name}' does not exist (clean)")
                    else:
                        self.logger.warning(f"Error checking queue '{q_name}': {e}")
            await ch.close()
            await conn.close()
        except Exception as e:
            self.logger.warning(f"Could not check queue state: {e}")

        if is_clean:
            self.logger.info("All queues clean")
        else:
            self.logger.warning("Queues not empty, previous cleanup may have failed")
        return is_clean

    async def run(self):
        """Run throughput measurement."""
        start = datetime.now()
        tc = self.throughput_config
        wc = self.config.wandb

        self.logger.info(f"Throughput: {self.config.num_samplers} samplers, {self.config.num_evaluators} evaluators")
        self.logger.info(f"Duration: {tc.warmup_minutes}min warmup + {tc.run_duration_minutes}min measurement")

        # Check queue state before starting, clean up if dirty
        await self.check_queue_state()

        # Init W&B
        run_name = wc.run_name or f"throughput_{wc.run_name_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if wc.enabled:
            # Serialize entire config to W&B
            wandb_config = dataclasses.asdict(self.config)
            # Convert tuple keys to strings (JSON doesn't support tuple keys)
            wandb_config = convert_tuple_keys(wandb_config)
            # Add evaluation script content (helpers available to LLM-generated code)
            eval_script_path = getattr(self.config.evaluator, 'evaluation_script_path', None)
            if eval_script_path:
                try:
                    with open(eval_script_path, 'r') as f:
                        wandb_config["evaluation_script"] = f.read()
                except Exception as e:
                    self.logger.warning(f"Failed to read evaluation script for W&B logging: {e}")
            wandb.init(
                project=wc.project, entity=wc.entity, name=run_name,
                tags=["throughput"] + list(wc.tags),
                config=wandb_config,
            )

        # Disable scaling for measurement
        modified_config = dataclasses.replace(self.config, scaling=dataclasses.replace(self.config.scaling, enabled=False))

        # Create ResourceManager for resource tracking
        self.resource_manager = ResourceManager(log_dir=self.log_dir, scaling_config=modified_config.scaling)

        # Set up RabbitMQ for main process (queue monitoring)
        self.logger.info(f"Connecting to RabbitMQ at {modified_config.rabbitmq.host}:{modified_config.rabbitmq.port} vhost={modified_config.rabbitmq.vhost}")
        try:
            conn = await rabbitmq.create_connection(modified_config, timeout=300)
            self.connections = [conn]
            self.logger.info("RabbitMQ connection established successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

        ch = await conn.channel()
        eval_q = await rabbitmq.declare_queue(ch, "evaluator_queue")
        sampler_q = await rabbitmq.declare_queue(ch, "sampler_queue")

        # Load prompt specification
        prompt_spec = prompt_builder.load_prompt_spec_from_config(modified_config)

        # Create termination config for throughput mode (time-based, not iteration-based)
        throughput_termination = dataclasses.replace(
            modified_config.termination,
            iteration_limit=999_999_999,  # High value to prevent iteration-based termination
            stop_on_optimal=False,  # Disable early stopping on optimal
            target_solutions=None  # Disable target-based termination
        )

        # Create connection manager for ProgramsDatabase
        db_connection_manager = rabbitmq.ConnectionManager(
            config=modified_config,
            component_name="ProgramsDatabase",
            queue_names=["database_queue", "sampler_queue", "evaluator_queue"],
            timeout=300
        )

        # Create database with connection manager (W&B disabled for throughput)
        self.database = programs_database.ProgramsDatabase(
            config=modified_config.programs_database,
            function_to_evolve="priority",
            connection_manager=db_connection_manager,
            checkpoint_file=self.checkpoint_path,
            save_checkpoints_path=os.path.join(wc.checkpoints_base_path, f"checkpoint_{run_name}"),
            termination_config=throughput_termination,
            best_known_solutions=modified_config.prompt.best_known_solutions,
            wandb_config=dataclasses.replace(wc, enabled=False),
            sampler_config=modified_config.sampler,
            evaluator_config=modified_config.evaluator,
            prompt_spec=prompt_spec,
            run_name=run_name,
        )

        db_task = asyncio.create_task(self.database.consume_and_process())
        self._start_processes(modified_config)

        # Start resource monitoring task (interval from scaling_config.resource_log_interval)
        resource_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically())

        # Wait for sampler connection
        self.logger.info("Waiting for samplers to connect...")
        while (await sampler_q.declare()).consumer_count < 1:
            await asyncio.sleep(5)

        # Publish initial programs or sample from checkpoint
        copies = getattr(modified_config.programs_database, "initial_program_copies", 1)
        if self.checkpoint_path and self.database.total_stored_programs > 0:
            # Checkpoint loaded. Sample prompts from database to prime the pipeline.
            self.logger.info(f"Checkpoint loaded with {self.database.total_stored_programs} programs. Sampling {copies} prompts to prime pipeline.")
            for _ in range(copies):
                # Trigger prompt generation from the loaded database
                await self.database.get_prompt()
        else:
            # No checkpoint, use initial programs
            initial_progs = self._load_initial_programs(modified_config)
            if initial_progs:
                self.logger.info(f"Publishing {len(initial_progs) * copies} initial programs to evaluator queue.")
                for prog in initial_progs * copies:
                    await ch.default_exchange.publish(aio_pika.Message(body=prog.encode()), routing_key="evaluator_queue")

        # Warm-up
        self.logger.info(f"Warm-up: {tc.warmup_minutes} min")
        await asyncio.sleep(tc.warmup_minutes * 60)

        # Collect window metrics
        window_iters = []
        window_gpu_utils = []
        window_cpu_utils = []
        window_sampler_q_depths = []
        window_evaluator_q_depths = []
        window_database_q_depths = []
        num_windows = tc.run_duration_minutes // tc.window_duration_minutes

        self.logger.info(f"Measurement: {num_windows} windows of {tc.window_duration_minutes} min")
        for w in range(num_windows):
            start_prompts = self.database.iterations

            # Collect resource samples during this window
            gpu_samples = []
            cpu_samples = []
            sampler_q_samples = []
            evaluator_q_samples = []
            database_q_samples = []
            window_seconds = tc.window_duration_minutes * 60
            sample_interval = 10  # Sample every 10 seconds
            num_samples = window_seconds // sample_interval

            for _ in range(num_samples):
                # Collect GPU utilization
                gpu_utils = await self.resource_manager.async_get_gpu_usage()
                if gpu_utils:
                    mean_gpu = sum(gpu_utils.values()) / len(gpu_utils)
                    gpu_samples.append(mean_gpu)

                # Collect CPU utilization
                cpu_util = await self.resource_manager.async_get_cpu_usage()
                cpu_samples.append(cpu_util)

                # Collect queue depths
                depths = await self.get_queue_depths()
                sampler_q_samples.append(depths["sampler_queue"])
                evaluator_q_samples.append(depths["evaluator_queue"])
                database_q_samples.append(depths["database_queue"])

                await asyncio.sleep(sample_interval)

            window_iters.append(self.database.iterations - start_prompts)

            # Calculate mean utilization for this window
            mean_gpu_util = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
            mean_cpu_util = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            mean_sampler_q = sum(sampler_q_samples) / len(sampler_q_samples) if sampler_q_samples else 0
            mean_evaluator_q = sum(evaluator_q_samples) / len(evaluator_q_samples) if evaluator_q_samples else 0
            mean_database_q = sum(database_q_samples) / len(database_q_samples) if database_q_samples else 0
            window_gpu_utils.append(mean_gpu_util)
            window_cpu_utils.append(mean_cpu_util)
            window_sampler_q_depths.append(mean_sampler_q)
            window_evaluator_q_depths.append(mean_evaluator_q)
            window_database_q_depths.append(mean_database_q)

            # Calculate per-hour rate for this window
            factor = 60 / tc.window_duration_minutes
            window_per_hour = window_iters[-1] * factor

            self.logger.info(f"Window {w+1}/{num_windows}: {window_iters[-1]} iterations ({window_per_hour:.0f}/hr), GPU: {mean_gpu_util:.1f}%, CPU: {mean_cpu_util:.1f}%, Queues: S={mean_sampler_q:.0f} E={mean_evaluator_q:.0f} D={mean_database_q:.0f}")

            # Log to W&B per window
            if wc.enabled:
                wandb.log({
                    "window": w + 1,
                    "window_iterations": window_iters[-1],
                    "window_iterations_per_hour": window_per_hour,
                    "window_gpu_utilization": mean_gpu_util,
                    "window_cpu_utilization": mean_cpu_util,
                    "window_sampler_queue_depth": mean_sampler_q,
                    "window_evaluator_queue_depth": mean_evaluator_q,
                    "window_database_queue_depth": mean_database_q,
                })

        # Cancel background tasks
        resource_task.cancel()
        db_task.cancel()
        try:
            await resource_task
        except asyncio.CancelledError:
            pass
        try:
            await db_task
        except asyncio.CancelledError:
            pass

        # Calculate stats (extrapolate to per-hour)
        factor = 60 / tc.window_duration_minutes
        per_hour = [w * factor for w in window_iters]

        results = {
            "num_samplers": self.config.num_samplers, "num_evaluators": self.config.num_evaluators,
            "model": self.config.sampler.model, "window_iterations_raw": window_iters,
            "window_iterations_per_hour": per_hour, "iterations_per_hour_mean": float(np.mean(per_hour)),
            "iterations_per_hour_std": float(np.std(per_hour)), "total_iterations": sum(window_iters),
            "warmup_minutes": tc.warmup_minutes, "run_duration_minutes": tc.run_duration_minutes,
            "window_duration_minutes": tc.window_duration_minutes, "timestamp": start.isoformat(),
            "total_duration_seconds": (datetime.now() - start).total_seconds(),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_programs": self.database.total_stored_programs if self.checkpoint_path else 0,
            "gpu_utilization_mean": float(np.mean(window_gpu_utils)) if window_gpu_utils else 0,
            "gpu_utilization_std": float(np.std(window_gpu_utils)) if window_gpu_utils else 0,
            "cpu_utilization_mean": float(np.mean(window_cpu_utils)) if window_cpu_utils else 0,
            "cpu_utilization_std": float(np.std(window_cpu_utils)) if window_cpu_utils else 0,
            # Queue depth stats per window
            "window_sampler_queue_depths": window_sampler_q_depths,
            "window_evaluator_queue_depths": window_evaluator_q_depths,
            "window_database_queue_depths": window_database_q_depths,
            "sampler_queue_depth_mean": float(np.mean(window_sampler_q_depths)) if window_sampler_q_depths else 0,
            "evaluator_queue_depth_mean": float(np.mean(window_evaluator_q_depths)) if window_evaluator_q_depths else 0,
            "database_queue_depth_mean": float(np.mean(window_database_q_depths)) if window_database_q_depths else 0,
        }

        # Log summary to W&B
        if wc.enabled:
            wandb.summary.update({
                "iterations_per_hour_mean": results["iterations_per_hour_mean"],
                "iterations_per_hour_std": results["iterations_per_hour_std"],
                "total_iterations": results["total_iterations"],
                "gpu_utilization_mean": results["gpu_utilization_mean"],
                "gpu_utilization_std": results["gpu_utilization_std"],
                "cpu_utilization_mean": results["cpu_utilization_mean"],
                "cpu_utilization_std": results["cpu_utilization_std"],
                "sampler_queue_depth_mean": results["sampler_queue_depth_mean"],
                "evaluator_queue_depth_mean": results["evaluator_queue_depth_mean"],
                "database_queue_depth_mean": results["database_queue_depth_mean"],
            })

        return results

    def _start_processes(self, config):
        """Start sampler and evaluator processes."""
        ctx = mp.get_context("spawn")
        use_local = config.sampler.use_local_vllm
        rm = ResourceManager(log_dir=self.log_dir, scaling_config=config.scaling)

        assigned_gpus = set()
        for i in range(config.num_samplers):
            device = None
            if use_local:
                assignment = rm.assign_gpu_device(min_free_memory_gib=20, max_utilization=50, assigned_gpus=assigned_gpus)
                if not assignment:
                    self.logger.error(f"No GPU for sampler {i}")
                    continue
                _, device = assignment
                assigned_gpus.add(device)

            proc = ctx.Process(target=sampler_process_entry, args=(self.config_path, device, self.log_dir, self.log_filename, i))
            proc.start()
            self.sampler_processes.append(proc)
            if use_local and i < config.num_samplers - 1:
                time.sleep(10)

        for i in range(config.num_evaluators):
            proc = ctx.Process(target=evaluator_process_entry,
                args=(self.config_path, self.template, self.inputs, self.target_signatures, self.log_dir, self.sandbox_base_path, self.log_filename))
            proc.start()
            self.evaluator_processes.append(proc)

    def _load_initial_programs(self, config):
        """Load initial programs from directory."""
        return load_initial_programs(
            config.evaluator.initial_functions_dir,
            logger=self.logger
        )

    async def cleanup(self):
        """Clean up processes and queues."""
        import psutil
        self.logger.info("Starting cleanup...")
        all_procs = self.sampler_processes + self.evaluator_processes

        # Collect all descendant PIDs before terminating (multiprocessing spawns helpers)
        all_pids = set()
        for p in all_procs:
            try:
                if p.pid:
                    all_pids.add(p.pid)
                    # Get all children recursively
                    try:
                        parent = psutil.Process(p.pid)
                        for child in parent.children(recursive=True):
                            all_pids.add(child.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception:
                pass

        # Terminate processes
        alive_count = sum(1 for p in all_procs if p.is_alive())
        if alive_count:
            self.logger.info(f"Terminating {alive_count} processes ({len(all_pids)} total PIDs including children)...")
        for p in all_procs:
            if p.is_alive():
                p.terminate()

        deadline = time.time() + 30
        while any(p.is_alive() for p in all_procs) and time.time() < deadline:
            await asyncio.sleep(0.5)

        # Force kill remaining tracked processes
        still_alive = [p for p in all_procs if p.is_alive()]
        if still_alive:
            self.logger.warning(f"Force killing {len(still_alive)} processes")
            for p in still_alive:
                p.kill()

        for p in all_procs:
            try:
                p.join(timeout=1)
            except Exception:
                pass

        # Kill any orphaned child processes that weren't terminated
        for pid in all_pids:
            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    self.logger.info(f"Killing orphaned process {pid}")
                    proc.kill()
                    proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass

        # Close connections
        for conn in self.connections:
            try:
                await conn.close()
            except Exception:
                pass

        # Delete queues
        try:
            conn = await rabbitmq.create_connection(self.config, timeout=10)
            ch = await conn.channel()
            for q in ["evaluator_queue", "sampler_queue", "database_queue"]:
                try:
                    queue = await ch.declare_queue(q, durable=False, auto_delete=False, passive=True)
                    msg_count = queue.declaration_result.message_count
                    consumer_count = queue.declaration_result.consumer_count
                    self.logger.info(f"Deleting queue {q} ({msg_count} msgs, {consumer_count} consumers)")
                    await queue.purge()
                    await queue.delete(if_unused=False, if_empty=False)
                except Exception as e:
                    if "NOT_FOUND" not in str(e):
                        self.logger.warning(f"Error cleaning queue {q}: {e}")
            await ch.close()
            await conn.close()
        except Exception as e:
            self.logger.warning(f"Queue cleanup failed: {e}")

        # Kill orphaned sandbox processes
        cleanup_orphaned_sandbox_processes(self.logger)

        if self.config.wandb.enabled:
            wandb.finish()

        self.logger.info("Cleanup complete")

    def save_results(self, results):
        """Save results and print summary."""
        path = os.path.join(self.log_dir, self.throughput_config.output_file)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}\nThroughput Results\n{'='*60}")
        print(f"Samplers: {results['num_samplers']}, Evaluators: {results['num_evaluators']}")
        print(f"Model: {results['model']}")
        print(f"Duration: {results['warmup_minutes']}min warmup + {results['run_duration_minutes']}min measurement")
        print(f"Windows: {len(results['window_iterations_raw'])} x {results['window_duration_minutes']}min")
        print(f"-" * 60)
        print(f"Iterations/hour: {results['iterations_per_hour_mean']:.0f} ± {results['iterations_per_hour_std']:.0f}")
        print(f"Total iterations: {results['total_iterations']}")
        print(f"GPU utilization: {results['gpu_utilization_mean']:.1f}% ± {results['gpu_utilization_std']:.1f}%")
        print(f"CPU utilization: {results['cpu_utilization_mean']:.1f}% ± {results['cpu_utilization_std']:.1f}%")
        print(f"-" * 60)
        print(f"Queue depths (mean): Sampler={results['sampler_queue_depth_mean']:.0f}, Evaluator={results['evaluator_queue_depth_mean']:.0f}, Database={results['database_queue_depth_mean']:.0f}")
        print(f"Queue depths per window:")
        for i, (s, e, d) in enumerate(zip(results['window_sampler_queue_depths'], results['window_evaluator_queue_depths'], results['window_database_queue_depths'])):
            print(f"  Window {i+1}: S={s:.0f}, E={e:.0f}, D={d:.0f}")
        print(f"{'='*60}\nSaved: {path}")


async def run_throughput(config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures, checkpoint_path=None):
    """Main entry point for throughput measurement."""
    runner = ThroughputRunner(config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures, checkpoint_path)
    try:
        results = await runner.run()
        runner.save_results(results)
        return results
    except Exception as e:
        import traceback
        runner.logger.error(f"Throughput measurement failed: {e}")
        runner.logger.error(traceback.format_exc())
        print(f"\n*** Throughput error: {e} ***", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        await runner.cleanup()


# Sweep functionality

def _generate_sweep_combinations(config):
    """Generate all parameter combinations from config.sweep."""
    from itertools import product

    if not hasattr(config, 'sweep') or config.sweep is None:
        return []

    sweep = config.sweep
    sweep_params = {}

    if hasattr(sweep, 'prompts_per_batch') and sweep.prompts_per_batch:
        sweep_params["prompts_per_batch"] = list(sweep.prompts_per_batch)
    if hasattr(sweep, 'evaluator_prefetch') and sweep.evaluator_prefetch:
        sweep_params["evaluator_prefetch"] = list(sweep.evaluator_prefetch)
    if hasattr(sweep, 'max_workers') and sweep.max_workers:
        sweep_params["max_workers"] = list(sweep.max_workers)
    if hasattr(sweep, 'sampler_prefetch_multiplier') and sweep.sampler_prefetch_multiplier:
        sweep_params["sampler_prefetch_multiplier"] = list(sweep.sampler_prefetch_multiplier)

    if not sweep_params:
        return []

    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _apply_sweep_params(base_config, sweep_params):
    """Create modified config with sweep parameters applied."""
    import cloudpickle

    sampler_kwargs = {}
    if "prompts_per_batch" in sweep_params:
        sampler_kwargs["prompts_per_batch"] = sweep_params["prompts_per_batch"]
    if "sampler_prefetch_multiplier" in sweep_params:
        sampler_kwargs["prefetch_multiplier"] = sweep_params["sampler_prefetch_multiplier"]

    modified_sampler = dataclasses.replace(base_config.sampler, **sampler_kwargs) if sampler_kwargs else base_config.sampler

    evaluator_kwargs = {}
    if "evaluator_prefetch" in sweep_params:
        evaluator_kwargs["prefetch_count"] = sweep_params["evaluator_prefetch"]
    if "max_workers" in sweep_params:
        evaluator_kwargs["max_workers"] = sweep_params["max_workers"]

    modified_evaluator = dataclasses.replace(base_config.evaluator, **evaluator_kwargs) if evaluator_kwargs else base_config.evaluator

    # Update W&B run name
    sweep_tag = "_".join(f"{k[:2]}{v}" for k, v in sweep_params.items())
    modified_wandb = dataclasses.replace(
        base_config.wandb,
        run_name=f"sweep_{sweep_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_name_tag=f"sweep_{sweep_tag}"
    )

    return dataclasses.replace(
        base_config,
        sampler=modified_sampler,
        evaluator=modified_evaluator,
        wandb=modified_wandb
    )


async def run_sweep(config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures, checkpoint_path=None, start_idx=0):
    """Run parameter sweep across all combinations in config.sweep."""
    import cloudpickle

    combinations = _generate_sweep_combinations(config)
    if not combinations:
        print("No sweep parameters configured. Run single measurement instead.")
        return await run_throughput(config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures, checkpoint_path)

    total = len(combinations)
    tc = config.throughput

    print(f"\n{'#'*70}")
    print(f"Throughput Sweep")
    print(f"{'#'*70}")
    print(f"Samplers: {config.num_samplers}, Evaluators: {config.num_evaluators}")
    print(f"Total configurations: {total}, starting from: {start_idx + 1}")
    print(f"Duration per config: {tc.warmup_minutes}min warmup + {tc.run_duration_minutes}min measurement")
    print(f"{'#'*70}\n")

    results_file = os.path.join(log_dir, f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    all_results = []

    for idx, sweep_params in enumerate(combinations[start_idx:], start=start_idx):
        print(f"\n{'='*70}")
        print(f"Configuration {idx + 1}/{total}: {sweep_params}")
        print(f"{'='*70}\n")

        modified_config = _apply_sweep_params(config, sweep_params)

        # Save modified config for subprocess
        sweep_tag = "_".join(f"{k[:2]}{v}" for k, v in sweep_params.items())
        config_log_dir = os.path.join(log_dir, f"sweep_{sweep_tag}")
        os.makedirs(config_log_dir, exist_ok=True)
        sweep_config_path = os.path.join(config_log_dir, "config.pkl")
        with open(sweep_config_path, 'wb') as f:
            cloudpickle.dump(modified_config, f)

        try:
            result = await run_throughput(
                modified_config, sweep_config_path, config_log_dir, sandbox_base_path,
                specification, inputs, target_signatures, checkpoint_path
            )
            result["sweep_params"] = sweep_params
            result["config_idx"] = idx
            all_results.append(result)

            print(f"Config {idx + 1} result: {result['iterations_per_hour_mean']:.0f} +/- {result['iterations_per_hour_std']:.0f} iter/hr")

        except Exception as e:
            print(f"ERROR in config {idx + 1}: {e}")
            all_results.append({"sweep_params": sweep_params, "config_idx": idx, "error": str(e)})

        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Cooldown between configs
        if idx < total - 1 and hasattr(tc, 'cooldown_seconds'):
            print(f"Cooldown: {tc.cooldown_seconds}s...")
            await asyncio.sleep(tc.cooldown_seconds)

    # Print summary
    print(f"\n{'#'*70}")
    print("Sweep Complete")
    print(f"{'#'*70}")
    valid = [r for r in all_results if "error" not in r and r.get("iterations_per_hour_mean", 0) > 0]
    if valid:
        sorted_results = sorted(valid, key=lambda x: x["iterations_per_hour_mean"], reverse=True)
        print("Top 5 configurations:")
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {r['iterations_per_hour_mean']:.0f} iter/hr | {r['sweep_params']}")
    print(f"Results saved: {results_file}")

    return all_results
