import os
import psutil
import pynvml
import logging
import asyncio
import multiprocessing
import torch.multiprocessing as torch_mp
from dataclasses import dataclass
import socket
import statistics
import time
import traceback
from typing import Any, Callable, Optional

from disfun.utils.rabbitmq import get_queue_stats
from disfun.startup import create_logger


@dataclass
class ScalingContext:
    """Bundles all parameters needed for the scaling loop."""
    config: Any
    config_path: str
    log_dir: str
    log_filename: Optional[str] = None
    # Queues
    evaluator_queue: Any = None
    sampler_queue: Any = None
    # Process lists (mutable, will be modified)
    evaluator_processes: list = None
    sampler_processes: list = None
    # Entry functions
    sampler_entry_function: Callable = None
    evaluator_entry_function: Callable = None
    # Evaluator-specific
    template: Any = None
    inputs: Any = None
    target_signatures: Any = None
    # Limits
    max_evaluators: int = 0
    min_evaluators: int = 1
    max_samplers: int = 0
    min_samplers: int = 1
    check_interval: int = 120

    def __post_init__(self):
        if self.evaluator_processes is None:
            self.evaluator_processes = []
        if self.sampler_processes is None:
            self.sampler_processes = []


class ResourceManager:
    def __init__(self, log_dir=None, resource_logger=None, cpu_only=False, scaling_config=None):
        """Initialize the ResourceManager asynchronously.

        Args:
            log_dir: Directory for log files
            resource_logger: Pre-configured logger instance
            cpu_only: Whether to run in CPU-only mode
            scaling_config: ScalingConfig instance with scaling thresholds
        """
        self.hostname = socket.gethostname()
        self.cpu_only = cpu_only
        self.process_to_device_map = {}  # pid -> device
        self.process_start_times = {}    # pid -> start time (for minimum lifetime check)
        self.scaling_config = scaling_config
        self.next_sampler_id = 0  # Monotonically increasing counter for unique sampler IDs (never reused)
        self.database = None  # Reference to ProgramsDatabase for syncing next_sampler_id to checkpoint
        # Scale-down idle tracking
        self._evaluator_idle_checks = 0
        self._sampler_idle_checks = 0
        if resource_logger is None:
            if log_dir is None:
                raise ValueError("Either resource_logger or log_dir must be provided")
            self.resource_logger = self._initialize_resource_logger(log_dir)
        else:
            self.resource_logger = resource_logger
        if not self.cpu_only:
            try:
                self._initialize_nvml()
            except Exception as e:
                self.resource_logger.warning(f"Failed to initialize NVML: {e}")
                self.cpu_only = True
                self.resource_logger.info("Switching to CPU-only mode.")


    def _initialize_nvml(self):
        """Initialize NVML for GPU monitoring."""
        pynvml.nvmlInit()

    def _free_device(self, pid):
        """Free device associated with a process (called when process dies)."""
        self.process_start_times.pop(pid, None)
        if pid in self.process_to_device_map:
            device = self.process_to_device_map.pop(pid)
            self.resource_logger.info(f"Freed device {device} from dead process (PID: {pid})")

    async def has_enough_system_memory(self):
        """Check if system has enough free memory based on scaling_config."""
        mem = await asyncio.to_thread(psutil.virtual_memory)
        free_gib = mem.available / (1024**3)
        return free_gib >= self.scaling_config.min_system_memory_gib

    def _initialize_resource_logger(self, log_dir):
        """Sets up a file-based logger using centralized create_logger."""
        pid = os.getpid()
        log_file_name = f"resources_{self.hostname}_pid{pid}.log"
        logger = create_logger(f'resource_logger_{pid}', log_dir, log_file_name)
        logger.info(f"Resource logger initialized for PID {pid}")
        return logger


    async def _collect_resource_sample(self) -> dict:
        """Collect a single sample of all resource metrics."""
        sample = {
            'cpu': await self.async_get_cpu_usage(),
            'io_wait': await asyncio.to_thread(lambda: psutil.cpu_times_percent(interval=1).iowait),
            'load': (await asyncio.to_thread(os.getloadavg))[0],
            'ctx_switches': await asyncio.to_thread(lambda: psutil.cpu_stats().ctx_switches),
            'mem': (await asyncio.to_thread(psutil.virtual_memory)).percent,
            'swap': (await asyncio.to_thread(psutil.swap_memory)).percent,
            'd_state': await asyncio.to_thread(
                lambda: len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'D'])
            ),
        }

        # Disk I/O (can be None in Docker)
        disk_io = await asyncio.to_thread(psutil.disk_io_counters)
        if disk_io:
            sample['disk_read'] = disk_io.read_bytes / 1e6
            sample['disk_write'] = disk_io.write_bytes / 1e6
        else:
            sample['disk_read'] = 0
            sample['disk_write'] = 0

        # GPU utilization
        if not self.cpu_only:
            sample['gpu'] = await self.async_get_gpu_usage()

        return sample

    def _format_resource_stats(self, samples: list[dict]) -> str:
        """Format collected samples into a log message."""
        def avg(key):
            return statistics.mean(s[key] for s in samples)

        msg = (
            f"Avg CPU: {avg('cpu'):.2f}%, Load: {avg('load'):.2f}, IO Wait: {avg('io_wait'):.2f}%, "
            f"Ctx Switches: {avg('ctx_switches'):.0f}, Disk Read/Write: {avg('disk_read'):.2f}/{avg('disk_write'):.2f} MB, "
            f"Mem: {avg('mem'):.2f}%, Swap: {avg('swap'):.2f}%, D State: {avg('d_state'):.0f}"
        )

        # Add GPU stats if available
        if not self.cpu_only and samples and 'gpu' in samples[0]:
            gpu_avgs = {}
            for s in samples:
                for gpu_idx, util in s.get('gpu', {}).items():
                    gpu_avgs.setdefault(gpu_idx, []).append(util)
            if gpu_avgs:
                gpu_parts = [f"GPU {i}: {statistics.mean(v):.2f}%" for i, v in sorted(gpu_avgs.items())]
                msg += f", {', '.join(gpu_parts)}"

        return msg

    async def log_resource_stats_periodically(self):
        """Log system resource usage periodically, averaging samples."""
        log_interval = self.scaling_config.resource_log_interval
        sample_count = self.scaling_config.sample_count
        sample_interval = self.scaling_config.sample_interval

        while True:
            try:
                samples = []
                for _ in range(sample_count):
                    samples.append(await self._collect_resource_sample())
                    await asyncio.sleep(sample_interval)

                self.resource_logger.info(self._format_resource_stats(samples))

            except Exception as e:
                self.resource_logger.error(f"Error logging resource stats: {e}")

            await asyncio.sleep(log_interval)

    async def async_get_cpu_usage(self):
        """Retrieves CPU usage asynchronously."""
        return await asyncio.to_thread(psutil.cpu_percent, interval=1)

    async def async_get_gpu_usage(self):
        """Retrieves GPU utilization asynchronously for all available GPUs.

        Returns:
            dict: Mapping of GPU index to utilization percentage, e.g. {0: 45.2, 1: 12.3}.
                  Returns empty dict if cpu_only mode.
        """
        if self.cpu_only:
            return {}

        gpu_utils = {}
        try:
            device_count = await asyncio.to_thread(pynvml.nvmlDeviceGetCount)
            for gpu_index in range(device_count):
                try:
                    handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, gpu_index)
                    utilization = await asyncio.to_thread(pynvml.nvmlDeviceGetUtilizationRates, handle)
                    gpu_utils[gpu_index] = utilization.gpu
                except Exception as e:
                    self.resource_logger.warning(f"GPU {gpu_index} monitoring failed: {e}")
                    gpu_utils[gpu_index] = 0
        except Exception as e:
            self.resource_logger.warning(f"GPU monitoring failed: {e}")
            return {}

        return gpu_utils


    def _cleanup_dead_processes(self, processes: list, process_type: str, free_device: bool = False):
        """Remove dead processes from tracking list."""
        dead = [p for p in processes if not p.is_alive()]
        for p in dead:
            processes.remove(p)
            self.resource_logger.warning(f"Removed dead {process_type} (PID: {p.pid}) from tracking")
            if free_device:
                self._free_device(p.pid)

    async def _try_scale_up_evaluator(self, ctx: ScalingContext, message_count: int) -> bool:
        """Try to scale up evaluators. Returns True if scaling action was taken."""
        can_scale = await self.can_scale_evaluator()
        threshold = self.scaling_config.evaluator_scale_up_threshold

        if message_count > threshold and len(ctx.evaluator_processes) < ctx.max_evaluators and can_scale:
            self.resource_logger.info(f"Scaling up evaluator (queue depth: {message_count})")
            self._start_evaluator(ctx)
            return True
        return False

    async def _try_scale_up_sampler(self, ctx: ScalingContext, message_count: int) -> bool:
        """Try to scale up samplers. Returns True if scaling action was taken."""
        assignment = await self.can_scale_up_samplers()
        if self.cpu_only:
            assignment = await self.can_scale_evaluator()

        threshold = self.scaling_config.sampler_scale_up_threshold

        if (message_count > threshold and
            len(ctx.sampler_processes) < ctx.max_samplers and
            assignment and
            await self.has_enough_system_memory()):
            self.resource_logger.info(f"Scaling up sampler (queue depth: {message_count})")
            started = self._start_sampler(ctx, assignment)
            if not started:
                self.resource_logger.info("No available GPU found. Skipping sampler scale-up.")
            return True
        return False

    async def _try_scale_down(self, processes: list, process_type: str,
                               message_count: int, min_count: int, idle_checks: int) -> tuple[bool, int]:
        """Try to scale down if queue is empty. Returns (action_taken, new_idle_checks)."""
        required_idle_checks = self.scaling_config.idle_checks_before_scale_down
        if message_count == 0 and len(processes) > min_count:
            idle_checks += 1
            if idle_checks >= required_idle_checks:
                # Select candidate for termination
                idx, gpu_util = self._select_process_to_terminate(processes, process_type)
                if idx is None:
                    self.resource_logger.info(f"No {process_type} eligible for termination (all too young)")
                    return False, idle_checks

                # For samplers, skip if the selected GPU is still busy (likely finishing work)
                if process_type == "Sampler" and gpu_util is not None:
                    threshold = self.scaling_config.min_gpu_util_for_scale_down
                    if gpu_util > threshold:
                        self.resource_logger.info(
                            f"Skipping Sampler scale-down: selected GPU util {gpu_util:.1f}% > {threshold}%"
                        )
                        return False, idle_checks  # Keep idle_checks, retry next interval

                self.resource_logger.info(f"{process_type} queue empty for {idle_checks} checks, terminating")
                await self.terminate_process(processes, process_type, idx=idx)
                return True, 0
            return False, idle_checks
        return False, 0  # Reset idle checks on non-empty queue

    async def run_scaling_loop(self, ctx: ScalingContext):
        """Scales evaluator and sampler processes dynamically based on queue depth and system resources."""
        self.resource_logger.info("Starting scaling loop")
        self.config = ctx.config

        try:
            while True:
                try:
                    # Clean up dead processes
                    self._cleanup_dead_processes(ctx.sampler_processes, "sampler", free_device=True)
                    self._cleanup_dead_processes(ctx.evaluator_processes, "evaluator")

                    # Get queue depths
                    eval_count, _ = await self.get_queue_message_count(ctx.evaluator_queue) if ctx.evaluator_queue else (0, 0)
                    sampler_count, _ = await self.get_queue_message_count(ctx.sampler_queue) if ctx.sampler_queue else (0, 0)
                    self.resource_logger.info(f"Queue depths: evaluator={eval_count}, sampler={sampler_count}")

                    # Scale evaluators
                    eval_scaled = False
                    if ctx.evaluator_queue and ctx.max_evaluators > 0:
                        eval_scaled = await self._try_scale_up_evaluator(ctx, eval_count)
                        if not eval_scaled:
                            eval_scaled, self._evaluator_idle_checks = await self._try_scale_down(
                                ctx.evaluator_processes, "Evaluator", eval_count,
                                ctx.min_evaluators, self._evaluator_idle_checks
                            )
                        else:
                            self._evaluator_idle_checks = 0

                    # Scale samplers
                    sampler_scaled = False
                    if ctx.sampler_queue and ctx.max_samplers > 0:
                        sampler_scaled = await self._try_scale_up_sampler(ctx, sampler_count)
                        if not sampler_scaled:
                            sampler_scaled, self._sampler_idle_checks = await self._try_scale_down(
                                ctx.sampler_processes, "Sampler", sampler_count,
                                ctx.min_samplers, self._sampler_idle_checks
                            )
                        else:
                            self._sampler_idle_checks = 0

                    if not eval_scaled and not sampler_scaled:
                        self.resource_logger.info("No scaling action taken.")

                except Exception as e:
                    self.resource_logger.error(f"Scaling loop error: {e}")

                await asyncio.sleep(ctx.check_interval)

        except asyncio.CancelledError:
            self.resource_logger.info("Scaling loop cancelled")
            raise

    def _start_evaluator(self, ctx: ScalingContext):
        """Start an evaluator process using ScalingContext."""
        mp_ctx = multiprocessing.get_context('spawn')
        proc = mp_ctx.Process(
            target=ctx.evaluator_entry_function,
            args=(ctx.config_path, ctx.template, ctx.inputs, ctx.target_signatures,
                  ctx.log_dir, ctx.log_filename, True),
            name=f"Evaluator-{len(ctx.evaluator_processes)}"
        )
        proc.start()
        ctx.evaluator_processes.append(proc)
        self.resource_logger.info(f"Started Evaluator (PID: {proc.pid})")

    def _start_sampler(self, ctx: ScalingContext, assignment) -> bool:
        """Start a sampler process using ScalingContext. Returns True if started."""
        mp_ctx = torch_mp.get_context('spawn')
        sampler_id = self.next_sampler_id
        self.next_sampler_id += 1
        if self.database is not None:
            self.database.next_sampler_id = self.next_sampler_id

        # Determine device based on assignment
        if assignment is True:  # CPU-only mode
            device = None
        elif assignment is not None:
            _, device = assignment  # (host_gpu, container_device)
        else:
            return False

        proc = mp_ctx.Process(
            target=ctx.sampler_entry_function,
            args=(ctx.config_path, device, ctx.log_dir, ctx.log_filename, sampler_id, True),
            name=f"Sampler-{sampler_id}"
        )
        proc.start()
        ctx.sampler_processes.append(proc)
        self.process_to_device_map[proc.pid] = device
        self.process_start_times[proc.pid] = time.time()

        if device:
            self.resource_logger.info(f"Started Sampler (PID: {proc.pid}, ID: {sampler_id}) on {device}")
        else:
            self.resource_logger.info(f"Started Sampler (PID: {proc.pid}, ID: {sampler_id}) in CPU-only mode")
        return True

    async def get_smoothed_cpu_usage(self):
        """Collect CPU usage samples and return averages for scaling decisions."""
        sample_count = self.scaling_config.sample_count
        sample_interval = self.scaling_config.sample_interval
        samples = []
        for _ in range(sample_count):
            usage = await asyncio.to_thread(psutil.cpu_percent, sample_interval, True)
            avg_sample = sum(usage) / len(usage) if usage else 0
            samples.append(avg_sample)
        return samples

    async def can_scale_up_samplers(self):
        """Returns a GPU assignment tuple (host_gpu, container_device) if scaling is possible, else None."""
        if self.cpu_only:
            return None

        min_memory = self.scaling_config.min_gpu_memory_gib
        max_util = self.scaling_config.max_gpu_utilization
        return self.assign_gpu_device(min_free_memory_gib=min_memory, max_utilization=max_util)

    async def can_scale_evaluator(self):
        """Check if CPU usage and load allow scaling up evaluators."""
        cpu_threshold = self.scaling_config.cpu_usage_threshold
        load_threshold = self.scaling_config.normalized_load_threshold

        smoothed_usage = await self.get_smoothed_cpu_usage()
        avg_cpu_usage = sum(smoothed_usage) / len(smoothed_usage) if smoothed_usage else 0

        load_avg = await asyncio.to_thread(os.getloadavg)
        available_cores = len(os.sched_getaffinity(0))
        normalized_load = load_avg[0] / available_cores if available_cores > 0 else load_avg[0]

        self.resource_logger.info(
            f"{self.hostname}: CPU: {avg_cpu_usage:.2f}%, Load: {normalized_load:.2f} (per core)"
        )

        return avg_cpu_usage < cpu_threshold and normalized_load < load_threshold

    def assign_gpu_device(self, min_free_memory_gib, max_utilization, assigned_gpus=None):
        """Assigns a GPU with sufficient free memory and low utilization."""
        if self.cpu_only:
            return None

        try:
            # Get visible GPUs
            visible_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_str:
                try:
                    visible_devices = [int(x.strip()) for x in visible_str.split(",") if x.strip()]
                except ValueError:
                    self.resource_logger.error("Failed to parse CUDA_VISIBLE_DEVICES.")
                    return None
            else:
                visible_devices = list(range(pynvml.nvmlDeviceGetCount()))

            # Map host GPU index to container-visible index
            id_to_container_index = {visible_devices[i]: i for i in range(len(visible_devices))}

            # Use assigned_gpus passed from the caller, otherwise fallback to existing assignments
            if assigned_gpus is None:
                assigned_gpus = set(self.process_to_device_map.values())

            available_gpus = []

            for host_gpu in visible_devices:
                container_device = f"cuda:{id_to_container_index[host_gpu]}"
            
                if container_device in assigned_gpus:
                    continue  # Skip GPUs that are already assigned in this loop

                handle = pynvml.nvmlDeviceGetHandleByIndex(host_gpu)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory_gib = memory_info.free / (1024 ** 3)

                if util.gpu < max_utilization and free_memory_gib >= min_free_memory_gib:
                    container_index = id_to_container_index[host_gpu]
                    available_gpus.append((host_gpu, container_index, free_memory_gib, util.gpu))

            if not available_gpus:
                return None

            # Sort GPUs by most free memory and lowest utilization
            available_gpus.sort(key=lambda x: (-x[2], x[3]))

            # Pick the best available GPU
            best_gpu = available_gpus[0]
            host_gpu, container_index, _, _ = best_gpu
            container_device = f"cuda:{container_index}"

            # Reserve GPU in assigned_gpus before returning
            assigned_gpus.add(container_device)


            self.resource_logger.info(
                f"Assigning GPU {host_gpu} (container {container_device}): Free {best_gpu[2]:.2f} GiB, Utilization {best_gpu[3]}%"
            )

            return host_gpu, container_device
        except Exception as e:
            self.resource_logger.error(f"Error in assign_gpu_device: {e}")
            return None


    def _select_process_to_terminate(self, processes, process_name):
        """Select best process to terminate: lowest GPU utilization, respecting minimum lifetime.

        Returns:
            tuple: (index, gpu_util) where gpu_util is the utilization % of the selected
                   sampler's GPU, or None for evaluators/CPU-only mode.
        """
        min_lifetime = self.scaling_config.min_process_lifetime
        now = time.time()
        eligible = []

        for i, proc in enumerate(processes):
            pid = proc.pid
            start_time = self.process_start_times.get(pid, 0)
            if now - start_time < min_lifetime:
                continue
            eligible.append((i, proc))

        if not eligible:
            return None, None

        # For samplers with GPUs, prefer terminating the one on lowest-utilization GPU
        if process_name == "Sampler" and not self.cpu_only:
            best_idx = None
            lowest_util = float('inf')

            for i, proc in eligible:
                device = self.process_to_device_map.get(proc.pid)
                if device and device.startswith("cuda:"):
                    try:
                        gpu_idx = int(device.split(":")[1])
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        if util < lowest_util:
                            lowest_util = util
                            best_idx = i
                    except Exception:
                        pass

            if best_idx is not None:
                return best_idx, lowest_util

        # Fallback: return oldest eligible process (no GPU util for evaluators)
        return eligible[0][0], None

    def cleanup(self):
        """Clean up ResourceManager state (call during shutdown)."""
        self.resource_logger.info("ResourceManager: Cleaning up state...")
        self.process_to_device_map.clear()
        self.process_start_times.clear()
        if not self.cpu_only:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self.resource_logger.info("ResourceManager: Cleanup complete")

    async def terminate_process(self, processes, process_name, idx=None):
        """Terminates a process and its children (vLLM may spawn subprocesses holding GPU memory).

        Args:
            processes: List of processes to select from
            process_name: "Sampler" or "Evaluator" for logging
            idx: Optional index to terminate. If None, selects automatically.
        """
        if not processes:
            return

        if idx is None:
            idx, _ = self._select_process_to_terminate(processes, process_name)
            if idx is None:
                self.resource_logger.info(f"No {process_name} eligible for termination (all too young)")
                return

        proc = processes.pop(idx)
        pid = proc.pid

        # Get child PIDs before terminating (vLLM may spawn subprocesses)
        child_pids = []
        try:
            child_pids = [c.pid for c in psutil.Process(pid).children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Samplers get longer timeout for GPU cleanup
        if process_name == "Sampler":
            timeout = self.scaling_config.sampler_termination_timeout
        else:
            timeout = self.scaling_config.termination_timeout

        proc.terminate()
        self.resource_logger.info(f"Terminating {process_name} (PID: {pid})...")

        await asyncio.to_thread(proc.join, timeout)

        if proc.is_alive():
            self.resource_logger.warning(f"{process_name} (PID: {pid}) force killing")
            proc.kill()
            await asyncio.to_thread(proc.join)

        # Kill orphaned children (may hold GPU memory)
        for child_pid in child_pids:
            try:
                psutil.Process(child_pid).kill()
                self.resource_logger.info(f"Killed orphaned child {child_pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Cleanup tracking maps
        self.process_start_times.pop(pid, None)
        if pid in self.process_to_device_map:
            device = self.process_to_device_map.pop(pid)
            self.resource_logger.info(f"Freed {device} (PID: {pid})")


    async def get_queue_message_count(self, queue):
        """Retrieves message and consumer counts for a queue via RabbitMQ HTTP API.

        Returns:
            tuple: (message_count, consumer_count), both integers.
        """
        if queue is None:
            return 0, 0

        messages, consumers = await get_queue_stats(self.config, queue.name)
        self.resource_logger.info(f"Queue '{queue.name}': messages={messages}, consumers={consumers}")
        return messages, consumers

