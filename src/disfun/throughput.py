"""Throughput measurement for DistributedFunSearch."""

import os
import re
import json
import time
import asyncio
import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import aio_pika
import torch.multiprocessing as mp

from disfun import code_manipulation, process_utils, programs_database
from disfun.scaling_utils import ResourceManager
from disfun.process_entry import sampler_process_entry, evaluator_process_entry


class ThroughputRunner:
    """Measures throughput (iterations/hour) for a single configuration."""

    def __init__(self, config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures):
        self.config = config
        self.throughput_config = config.throughput
        self.config_path = config_path
        self.log_dir = log_dir
        self.sandbox_base_path = sandbox_base_path
        self.inputs = inputs
        self.target_signatures = target_signatures
        self.template = code_manipulation.text_to_program(specification)

        self.log_filename = f"throughput_pid{os.getpid()}.log"
        self.logger = process_utils.initialize_logger(log_dir, self.log_filename, process_type="Throughput")

        self.sampler_processes = []
        self.evaluator_processes = []
        self.connections = []
        self.database = None

    async def run(self):
        """Run throughput measurement."""
        start = datetime.now()
        tc = self.throughput_config

        self.logger.info(f"Throughput measurement: {self.config.num_samplers} samplers, {self.config.num_evaluators} evaluators")
        self.logger.info(f"Duration: {tc.warmup_minutes} min warmup + {tc.run_duration_minutes} min measurement")

        # Disable scaling for measurement
        modified_config = dataclasses.replace(
            self.config,
            scaling=dataclasses.replace(self.config.scaling, enabled=False),
        )

        # Set up RabbitMQ
        conn1 = await process_utils.create_rabbitmq_connection(modified_config, timeout=300)
        conn2 = await process_utils.create_rabbitmq_connection(modified_config, timeout=300)
        self.connections = [conn1, conn2]

        ch1, ch2 = await conn1.channel(), await conn2.channel()
        eval_q = await process_utils.declare_standard_queue(ch1, "evaluator_queue")
        sampler_q = await process_utils.declare_standard_queue(ch1, "sampler_queue")
        db_q = await process_utils.declare_standard_queue(ch2, "database_queue")

        # Create database with W&B disabled
        run_name = f"throughput_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.database = programs_database.ProgramsDatabase(
            conn2, ch2, db_q, sampler_q, eval_q,
            modified_config.programs_database, "priority", None,
            os.path.join(modified_config.wandb.checkpoints_base_path, f"checkpoint_{run_name}"),
            mode=modified_config.evaluator.mode,
            start_n=modified_config.evaluator.start_n, end_n=modified_config.evaluator.end_n,
            s_values=modified_config.evaluator.s_values,
            no_deduplication=modified_config.programs_database.no_deduplication,
            prompt_limit=999_999_999, optimal_solution_programs=999_999_999, max_drain_time=0,
            target_signatures=None,
            show_eval_scores=modified_config.prompt.show_eval_scores,
            display_mode=modified_config.prompt.display_mode,
            best_known_solutions=modified_config.prompt.best_known_solutions,
            absolute_label=modified_config.prompt.absolute_label,
            relative_label=modified_config.prompt.relative_label,
            q=modified_config.evaluator.q,
            wandb_config=dataclasses.replace(modified_config.wandb, enabled=False),
            sampler_config=modified_config.sampler,
            evaluator_config=modified_config.evaluator,
            prompt_config=modified_config.prompt,
            run_name=run_name, rabbitmq_config=modified_config,
        )

        db_task = asyncio.create_task(self.database.consume_and_process())
        self._start_processes(modified_config)

        # Wait for sampler connection
        self.logger.info("Waiting for samplers to connect...")
        while (await sampler_q.declare()).consumer_count < 1:
            await asyncio.sleep(5)

        # Publish initial programs
        initial_progs = self._load_initial_programs(modified_config)
        if initial_progs:
            copies = getattr(modified_config.programs_database, "initial_program_copies", 1)
            for prog in initial_progs * copies:
                await ch1.default_exchange.publish(aio_pika.Message(body=prog.encode()), routing_key="evaluator_queue")

        # Warm-up
        self.logger.info(f"Warm-up: {tc.warmup_minutes} min")
        await asyncio.sleep(tc.warmup_minutes * 60)

        # Collect window metrics
        window_iters = []
        num_windows = tc.run_duration_minutes // tc.window_duration_minutes

        self.logger.info(f"Starting measurement: {num_windows} windows of {tc.window_duration_minutes} min")
        for w in range(num_windows):
            start_prompts = self.database.total_prompts
            await asyncio.sleep(tc.window_duration_minutes * 60)
            window_iters.append(self.database.total_prompts - start_prompts)
            self.logger.info(f"Window {w+1}/{num_windows}: {window_iters[-1]} iterations")

        db_task.cancel()
        try:
            await db_task
        except asyncio.CancelledError:
            pass

        # Calculate stats (extrapolate to per-hour)
        factor = 60 / tc.window_duration_minutes
        per_hour = [w * factor for w in window_iters]

        return {
            "num_samplers": self.config.num_samplers,
            "num_evaluators": self.config.num_evaluators,
            "model": self.config.sampler.model,
            "window_iterations_raw": window_iters,
            "window_iterations_per_hour": per_hour,
            "iterations_per_hour_mean": float(np.mean(per_hour)),
            "iterations_per_hour_std": float(np.std(per_hour)),
            "total_iterations": sum(window_iters),
            "warmup_minutes": tc.warmup_minutes,
            "run_duration_minutes": tc.run_duration_minutes,
            "window_duration_minutes": tc.window_duration_minutes,
            "timestamp": start.isoformat(),
            "total_duration_seconds": (datetime.now() - start).total_seconds(),
        }

    def _start_processes(self, config):
        """Start sampler and evaluator processes."""
        from disfun.sampler import is_local_model

        ctx = mp.get_context("spawn")
        use_local = is_local_model(config.sampler.model)
        rm = ResourceManager(log_dir=self.log_dir, scaling_config=config.scaling)

        # Start samplers
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

        # Start evaluators
        for i in range(config.num_evaluators):
            proc = ctx.Process(target=evaluator_process_entry,
                args=(self.config_path, self.template, self.inputs, self.target_signatures, self.log_dir, self.sandbox_base_path, self.log_filename))
            proc.start()
            self.evaluator_processes.append(proc)

    def _load_initial_programs(self, config):
        """Load initial programs from directory."""
        programs = []
        init_dir = Path(config.evaluator.initial_functions_dir)
        if not init_dir.exists():
            return programs

        for f in sorted(init_dir.glob("*.txt")):
            try:
                body = f.read_text().strip()
                body = re.sub(r"<thinking>.*?</thinking>\s*", "", body, flags=re.DOTALL)
                body = re.sub(r"<thought>.*?</thought>\s*", "", body, flags=re.DOTALL)
                body = re.sub(r"<code>(.*?)</code>", r"\1", body, flags=re.DOTALL)
                programs.append(json.dumps({"sample": body.strip(), "island_id": None, "version_generated": None, "expected_version": 0}))
            except Exception:
                pass
        return programs

    async def cleanup(self):
        """Clean up processes and queues."""
        all_procs = self.sampler_processes + self.evaluator_processes

        for p in all_procs:
            if p.is_alive():
                p.terminate()

        deadline = time.time() + 30
        while any(p.is_alive() for p in all_procs) and time.time() < deadline:
            await asyncio.sleep(0.5)

        for p in all_procs:
            if p.is_alive():
                p.kill()
            try:
                p.join(timeout=1)
            except Exception:
                pass

        for conn in self.connections:
            try:
                await conn.close()
            except Exception:
                pass

        # Delete queues
        try:
            conn = await process_utils.create_rabbitmq_connection(self.config, timeout=10)
            ch = await conn.channel()
            for q in ["evaluator_queue", "sampler_queue", "database_queue"]:
                try:
                    queue = await ch.declare_queue(q, durable=False, auto_delete=False, passive=True)
                    await queue.purge()
                    await queue.delete(if_unused=False, if_empty=False)
                except Exception:
                    pass
            await ch.close()
            await conn.close()
        except Exception:
            pass

        # Kill orphaned sandbox processes
        for proc in psutil.process_iter(["cmdline"]):
            try:
                if proc.info.get("cmdline") and "container_main.py" in " ".join(proc.info["cmdline"]):
                    proc.kill()
            except Exception:
                pass

    def save_results(self, results):
        """Save results and print summary."""
        path = os.path.join(self.log_dir, self.throughput_config.output_file)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"THROUGHPUT RESULTS")
        print(f"{'='*60}")
        print(f"Samplers: {results['num_samplers']}, Evaluators: {results['num_evaluators']}")
        print(f"Model: {results['model']}")
        print(f"Duration: {results['warmup_minutes']} min warmup + {results['run_duration_minutes']} min measurement")
        print(f"Windows: {len(results['window_iterations_raw'])} x {results['window_duration_minutes']} min")
        print(f"-" * 60)
        print(f"Iterations per hour: {results['iterations_per_hour_mean']:.0f} ± {results['iterations_per_hour_std']:.0f}")
        print(f"Total iterations: {results['total_iterations']}")
        print(f"{'='*60}")
        print(f"Saved: {path}")


async def run_throughput(config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures):
    """Main entry point for throughput measurement."""
    runner = ThroughputRunner(config, config_path, log_dir, sandbox_base_path, specification, inputs, target_signatures)
    try:
        results = await runner.run()
        runner.save_results(results)
        return results
    finally:
        await runner.cleanup()
