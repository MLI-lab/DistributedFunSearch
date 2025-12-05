import os
import psutil
import pynvml
import logging
import asyncio
import torch.multiprocessing as mp
from logging import FileHandler
import socket
import statistics
import aiohttp
import time
import traceback


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
        # Counter for detecting disconnected samplers (consecutive checks with 0 consumers)
        self.sampler_zero_consumer_count = 0
        # Flag to skip zero-consumer check during initial startup (model loading takes 5-10 min)
        self.samplers_ever_connected = False
        # Time-based tracking for faster disconnection detection
        self.last_sampler_activity_time = None
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

    def _all_samplers_busy(self, utilization_threshold=50):
        """Check if all assigned sampler GPUs have high utilization.

        Returns True if all samplers are actively using their GPUs,
        meaning we should not scale down even if queue is empty.
        """
        if self.cpu_only or not self.process_to_device_map:
            return False

        try:
            for pid, device in self.process_to_device_map.items():
                if device is None or not device.startswith("cuda:"):
                    continue
                gpu_idx = int(device.split(":")[1])
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                if util < utilization_threshold:
                    # At least one sampler has low GPU utilization
                    return False
            # All samplers have high utilization
            return True
        except Exception as e:
            self.resource_logger.warning(f"Error checking sampler GPU utilization: {e}")
            return False

    async def has_enough_system_memory(self, min_free_gib=None):
        """Check if system has enough free memory.

        Args:
            min_free_gib: Minimum free memory in GiB. If None - uses scaling_config value (default: 30)
        """
        if min_free_gib is None:
            min_free_gib = self.scaling_config.min_system_memory_gib if self.scaling_config else 30
        mem = await asyncio.to_thread(psutil.virtual_memory)
        free_gib = mem.available / (1024**3)
        return free_gib >= min_free_gib

    def _initialize_resource_logger(self, log_dir):
        """Sets up a file-based logger."""
        pid = os.getpid()
        log_file_name = f"resources_{self.hostname}_pid{pid}.log"
        log_file_path = os.path.join(log_dir, log_file_name)

        logger = logging.getLogger(f'resource_logger_{pid}')
        logger.setLevel(logging.DEBUG)
        os.makedirs(log_dir, exist_ok=True)
        handler = FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        logger.info(f"Resource logger initialized for PID {pid}. Log file: {log_file_path}")
        return logger


    async def log_resource_stats_periodically(self, interval=60, sample_duration=10, sample_interval=1):
        """
        Logs system resource usage periodically - averaging values over `sample_duration` seconds.
        - interval: Time between log entries (seconds).
        - sample_duration: Time window over which to collect samples.
        - sample_interval: Time between samples within the window.
        """
        while True:
            try:
                num_samples = max(1, sample_duration // sample_interval)

                # Collect samples
                cpu_samples = []
                io_wait_samples = []
                load_samples = []
                d_state_samples = []
                disk_read_samples = []
                disk_write_samples = []
                ctx_switch_samples = []
                mem_samples = []
                swap_samples = []

                if not self.cpu_only:
                    gpu_samples = {}  # Dict of lists: {gpu_index: [sample1 - sample2 - ...]}

                for _ in range(num_samples):
                    # CPU usage
                    cpu_samples.append(await self.async_get_cpu_usage())

                    # I/O wait
                    io_wait_samples.append(await asyncio.to_thread(lambda: psutil.cpu_times_percent(interval=1).iowait))

                    # Load averages
                    load_avg = await asyncio.to_thread(os.getloadavg)
                    load_samples.append(load_avg[0])  # 1-min load average

                    # Context switches
                    ctx_switch_samples.append(await asyncio.to_thread(lambda: psutil.cpu_stats().ctx_switches))

                    # Disk I/O
                    disk_io = await asyncio.to_thread(psutil.disk_io_counters)
                    if disk_io:  # Can be None on some systems (e.g. - Docker containers)
                        disk_read_samples.append(disk_io.read_bytes / 1e6)  # Convert to MB
                        disk_write_samples.append(disk_io.write_bytes / 1e6)
                    else:
                        disk_read_samples.append(0)
                        disk_write_samples.append(0)

                    # Memory & swap
                    memory = await asyncio.to_thread(psutil.virtual_memory)
                    swap = await asyncio.to_thread(psutil.swap_memory)
                    mem_samples.append(memory.percent)
                    swap_samples.append(swap.percent)

                    # D-state processes (blocked on I/O)
                    d_state_samples.append(
                        await asyncio.to_thread(lambda: len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'D']))
                    )

                    # GPU utilization (if available)
                    if not self.cpu_only:
                        try:
                            gpu_utils = await self.async_get_gpu_usage()
                            for gpu_index, util in gpu_utils.items():
                                if gpu_index not in gpu_samples:
                                    gpu_samples[gpu_index] = []
                                gpu_samples[gpu_index].append(util)
                        except Exception as e:
                            self.resource_logger.warning(f"GPU monitoring failed: {e}")

                    await asyncio.sleep(sample_interval)

                # Compute averages
                avg_cpu = statistics.mean(cpu_samples)
                avg_io_wait = statistics.mean(io_wait_samples)
                avg_load = statistics.mean(load_samples)
                avg_ctx_switch = statistics.mean(ctx_switch_samples)
                avg_disk_read = statistics.mean(disk_read_samples)
                avg_disk_write = statistics.mean(disk_write_samples)
                avg_mem = statistics.mean(mem_samples)
                avg_swap = statistics.mean(swap_samples)
                avg_d_state = statistics.mean(d_state_samples)

                log_message = (
                    f"Avg CPU: {avg_cpu:.2f}% - Load: {avg_load:.2f} - I/O Wait: {avg_io_wait:.2f}% - "
                    f"Ctx Switches: {avg_ctx_switch} - Disk Read/Write: {avg_disk_read:.2f}/{avg_disk_write:.2f} MB - "
                    f"Mem Usage: {avg_mem:.2f}% - Swap: {avg_swap:.2f}% - D-State Processes: {avg_d_state}"
                )

                # Include GPU if applicable
                if not self.cpu_only and gpu_samples:
                    gpu_stats = []
                    for gpu_index in sorted(gpu_samples.keys()):
                        avg_gpu = statistics.mean(gpu_samples[gpu_index]) if gpu_samples[gpu_index] else 0
                        gpu_stats.append(f"GPU {gpu_index}: {avg_gpu:.2f}%")
                    log_message += f" - {' - '.join(gpu_stats)}"

                self.resource_logger.info(log_message)

            except Exception as e:
                self.resource_logger.error(f"Error logging resource stats: {e}")

            await asyncio.sleep(interval)

    async def async_get_cpu_usage(self):
        """Retrieves CPU usage asynchronously."""
        return await asyncio.to_thread(psutil.cpu_percent, interval=1)

    async def async_get_gpu_usage(self):
        """Retrieves GPU utilization asynchronously for all available GPUs.

        Returns:
            dict: Mapping of GPU index to utilization percentage {0: 45.2 - 1: 12.3 - ...}
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


    async def run_scaling_loop(self, evaluator_queue=None, sampler_queue=None,
                               evaluator_processes=None, sampler_processes=None,
                               sampler_entry_function=None, evaluator_entry_function=None,
                               config=None, config_path=None, log_dir=None, template=None, inputs=None, target_signatures=None,
                               sandbox_base_path=None, max_evaluators=10000, min_evaluators=1,
                               max_samplers=1000, min_samplers=1, check_interval=120, log_filename=None):
        """Scales evaluator and sampler processes dynamically based on queue sizes and system resources."""
        self.resource_logger.info("Starting scaling loop")
        self.config = config  # Store config for use in get_queue_message_count
        evaluator_processes = evaluator_processes or []
        sampler_processes = sampler_processes or []
        max_evaluators = max_evaluators if max_evaluators is not None else 0

        try:
            while True:
                try:
                    # Clean up dead processes from lists (crashed/killed processes)
                    if sampler_processes:
                        dead_samplers = [p for p in sampler_processes if not p.is_alive()]
                        for p in dead_samplers:
                            sampler_processes.remove(p)
                            self.resource_logger.warning(f"Removed dead sampler process (PID: {p.pid}) from tracking")
                            self._free_device(p.pid)
                    if evaluator_processes:
                        dead_evaluators = [p for p in evaluator_processes if not p.is_alive()]
                        for p in dead_evaluators:
                            evaluator_processes.remove(p)
                            self.resource_logger.warning(f"Removed dead evaluator process (PID: {p.pid}) from tracking")

                    # Get message counts and consumer counts for both queues
                    evaluator_message_count, evaluator_consumer_count = await self.get_queue_message_count(evaluator_queue) if evaluator_queue else (0, 0)
                    sampler_message_count, sampler_consumer_count = await self.get_queue_message_count(sampler_queue) if sampler_queue else (0, 0)
                    self.resource_logger.info(f"Message counts: evaluator={evaluator_message_count}, sampler={sampler_message_count}")

                    # Scale Evaluators
                    evaluator_scaled = False
                    if evaluator_queue and max_evaluators > 0:
                        can_scale_eval = await self.can_scale_evaluator()
                        evaluator_threshold = self.scaling_config.evaluator_scale_up_threshold if self.scaling_config else 10
                        if evaluator_message_count > evaluator_threshold and len(evaluator_processes) < max_evaluators and can_scale_eval:
                            self.resource_logger.info(f"Can scale evaluators with messages in queue {evaluator_message_count}")
                            self.start_evaluator_process(evaluator_entry_function, config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, evaluator_processes, "Evaluator", log_filename)
                            evaluator_scaled = True
                            self._evaluator_idle_checks = 0
                        elif evaluator_message_count == 0 and len(evaluator_processes) > min_evaluators:
                            # Require 2 consecutive idle checks before scaling down (avoid killing during in-flight processing)
                            self._evaluator_idle_checks += 1
                            if self._evaluator_idle_checks >= 2:
                                self.resource_logger.info(f"Queue empty for {self._evaluator_idle_checks} checks, terminating evaluator")
                                await self.terminate_process(evaluator_processes, "Evaluator")
                                evaluator_scaled = True
                                self._evaluator_idle_checks = 0
                        else:
                            self._evaluator_idle_checks = 0  # Reset on non-empty queue

                    # Scale Samplers
                    sampler_scaled = False
                    if sampler_queue and max_samplers > 0:
                        assignment = await self.can_scale_up_samplers()
                        if self.cpu_only:
                            assignment = await self.can_scale_evaluator()  # if we are in cpu only mode also check cpu load for samplers
                        self.resource_logger.info(f"Assignment is {assignment}")
                        sampler_threshold = self.scaling_config.sampler_scale_up_threshold if self.scaling_config else 50
                        current_time = time.time()

                        # Track if samplers have ever connected (to distinguish startup from disconnection)
                        if sampler_consumer_count > 0:
                            self.samplers_ever_connected = True
                            self.sampler_zero_consumer_count = 0
                            self.last_sampler_activity_time = current_time

                        # CRITICAL: Detect disconnected samplers and spawn replacements
                        # Triggers on EITHER:
                        # 1. 2+ consecutive checks with 0 consumers, OR
                        # 2. Messages waiting for >2 minutes with 0 consumers (time-based)
                        if self.samplers_ever_connected and sampler_consumer_count == 0 and sampler_message_count > 0:
                            self.sampler_zero_consumer_count += 1
                            time_since_activity = (current_time - self.last_sampler_activity_time
                                                   if self.last_sampler_activity_time else float('inf'))

                            should_spawn = (
                                self.sampler_zero_consumer_count >= 2 or
                                time_since_activity > 120  # 2 minutes timeout
                            )

                            if should_spawn:
                                self.resource_logger.warning(
                                    f"ALERT: sampler_queue has {sampler_message_count} messages but 0 consumers "
                                    f"({self.sampler_zero_consumer_count} checks, {time_since_activity:.0f}s since activity). "
                                    f"Spawning replacement sampler..."
                                )
                                if assignment and await self.has_enough_system_memory():
                                    started = self.start_sampler_process(
                                        sampler_entry_function, config_path, log_dir, sampler_processes,
                                        "Sampler", assignment=assignment, log_filename=log_filename
                                    )
                                    if started:
                                        self.resource_logger.info("Successfully spawned replacement sampler.")
                                        sampler_scaled = True
                                    else:
                                        self.resource_logger.warning("Failed to spawn replacement sampler (no GPU available).")
                                else:
                                    self.resource_logger.warning("Cannot spawn replacement sampler (resources unavailable).")

                        # Normal scaling logic
                        if not sampler_scaled:  # Only if we didn't already spawn a replacement
                            if sampler_message_count > sampler_threshold and len(sampler_processes) < max_samplers and assignment and await self.has_enough_system_memory():
                                self.resource_logger.info(f"Can scale samplers with messages in queue  {sampler_message_count}")
                                started = self.start_sampler_process(sampler_entry_function, config_path, log_dir, sampler_processes, "Sampler", assignment=assignment, log_filename=log_filename)
                                if not started:
                                    self.resource_logger.info("No available GPU found. Skipping sampler scale-up.")
                                sampler_scaled = True
                                self._sampler_idle_checks = 0
                            elif sampler_message_count == 0 and len(sampler_processes) > min_samplers:
                                # Require 2 consecutive idle checks before scaling down
                                self._sampler_idle_checks += 1
                                if self._sampler_idle_checks >= 2:
                                    # Skip scale down if all samplers are actively using GPUs
                                    all_busy = await asyncio.to_thread(self._all_samplers_busy)
                                    if all_busy:
                                        self.resource_logger.info("Queue empty but all samplers have high GPU utilization. Skipping scale down.")
                                        self._sampler_idle_checks = 0
                                    else:
                                        self.resource_logger.info(f"Sampler queue empty for {self._sampler_idle_checks} checks, terminating sampler")
                                        await self.terminate_process(sampler_processes, "Sampler")
                                        sampler_scaled = True
                                        self._sampler_idle_checks = 0
                            else:
                                self._sampler_idle_checks = 0  # Reset on non-empty queue

                    # If nothing was scaled - log that scaling was skipped
                    if not evaluator_scaled and not sampler_scaled:
                        self.resource_logger.info("No scaling action taken in this iteration.")

                except Exception as e:
                    self.resource_logger.error(f"Scaling loop encountered an error: {e}")

                await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            self.resource_logger.info("Scaling loop cancelled, stopping gracefully...")
            raise  # Re-raise to properly propagate cancellation

    def start_evaluator_process(self, entry_function, config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, processes, process_name, log_filename):
        """Starts a new evaluator process using 'fork' multiprocessing context.

        Uses fork because evaluators don't load ML models and only execute functions
        in sandboxed subprocesses. Fork is faster than spawn for CPU-bound workloads.
        """
        ctx = mp.get_context('fork')
        proc = ctx.Process(
            target=entry_function,
            args=(config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, log_filename, True),  # use_parent_log=True
            name=f"{process_name}-{len(processes)}"
        )
        proc.start()
        processes.append(proc)
        self.resource_logger.info(f"Started {process_name} process (PID: {proc.pid})")

    async def get_smoothed_cpu_usage(self, duration=10, interval=1):
        """
        Asynchronously collect CPU usage samples over 'duration' seconds at
        'interval' second intervals and return a list of average CPU usage values.
        """
        samples = []
        iterations = int(duration / interval)
        for _ in range(iterations):
            # psutil.cpu_percent will block for 'interval' seconds; offload it.
            usage = await asyncio.to_thread(psutil.cpu_percent, interval, True)
            # Compute the average usage across cores for this sample
            avg_sample = sum(usage) / len(usage) if usage else 0
            samples.append(avg_sample)
        return samples

    async def can_scale_up_samplers(self):
        """
        Returns a GPU assignment tuple (host_gpu - container_device) if we can
        scale up samplers - or None if we cannot.
        """
        if self.cpu_only:
            # No GPUs available at all
            return None

        # See if any GPU is free enough - using config values
        min_memory = self.scaling_config.min_gpu_memory_gib if self.scaling_config else 20
        max_util = self.scaling_config.max_gpu_utilization if self.scaling_config else 50
        assignment = await asyncio.to_thread(
            self.assign_gpu_device, min_free_memory_gib=min_memory, max_utilization=max_util
        )
        return assignment

    async def can_scale_evaluator(self, cpu_usage_threshold=None, normalized_load_threshold=None, duration=10, interval=1):
        """
        Determine if it's safe to scale up evaluators based on CPU usage and system load.

        Args:
            cpu_usage_threshold: Maximum allowed average CPU usage percentage. If None - uses scaling_config value (default: 99).
            normalized_load_threshold: Maximum allowed 1-minute load (load average divided by available cores). If None - uses scaling_config value (default: 0.99).
            duration: Duration in seconds to smooth CPU usage samples (default: 10).
            interval: Interval in seconds between CPU usage samples (default: 1).

        Returns:
            True if both CPU usage and normalized load are below their respective thresholds.
        """
        # Use config values if parameters not provided
        if cpu_usage_threshold is None:
            cpu_usage_threshold = self.scaling_config.cpu_usage_threshold if self.scaling_config else 99
        if normalized_load_threshold is None:
            normalized_load_threshold = self.scaling_config.normalized_load_threshold if self.scaling_config else 0.99

        # Get smoothed CPU usage over the specified duration.
        smoothed_usage = await self.get_smoothed_cpu_usage(duration, interval)
        avg_cpu_usage = sum(smoothed_usage) / len(smoothed_usage) if smoothed_usage else 0

        # Get the 1-minute load average and normalize by available cores.
        load_avg = await asyncio.to_thread(os.getloadavg)
        load_avg_1 = load_avg[0]
        available_cores = len(await asyncio.to_thread(os.sched_getaffinity, 0))
        normalized_load = load_avg_1 / available_cores if available_cores > 0 else load_avg_1

        self.resource_logger.info(
            f"{self.hostname}: Smoothed Avg CPU Usage: {avg_cpu_usage:.2f}% | "
            f"Normalized Load: {normalized_load:.2f} (Load per core)"
        )

        # Return True only if both CPU metrics are below threshold and a GPU is available.
        return (avg_cpu_usage < cpu_usage_threshold) and (normalized_load < normalized_load_threshold)


    def start_sampler_process(self, entry_function, config_path, log_dir, processes, process_name, assignment, log_filename=None):
        """Starts a new sampler process using 'spawn' multiprocessing context.

        Uses spawn to avoid fork+threading deadlocks when loading ML models (StarCoder2/GPT).
        Spawn creates a clean process without inheriting thread state from parent.
        """
        ctx = mp.get_context('spawn')
        # Use monotonically increasing counter for unique sampler IDs (never reused, even after termination)
        sampler_id = self.next_sampler_id
        self.next_sampler_id += 1
        # Sync to database for checkpointing
        if self.database is not None:
            self.database.next_sampler_id = self.next_sampler_id

        if assignment is True:  # CPU-only mode - no GPU assignment
            proc = ctx.Process(
                target=entry_function,
                args=(config_path, None, log_dir, log_filename, sampler_id, True),  # use_parent_log=True
                name=f"{process_name}-{sampler_id}"
            )
            proc.start()
            processes.append(proc)
            self.resource_logger.info(f"Started {process_name} process (PID: {proc.pid}, ID: {sampler_id}) in CPU-only mode.")
            self.process_to_device_map[proc.pid] = None
            self.process_start_times[proc.pid] = time.time()
            return True
        elif assignment is not None:
            # GPU assignment is available
            host_gpu, container_device = assignment
            proc = ctx.Process(
                target=entry_function,
                args=(config_path, container_device, log_dir, log_filename, sampler_id, True),  # use_parent_log=True
                name=f"{process_name}-{sampler_id}"
            )
            proc.start()
            processes.append(proc)
            self.resource_logger.info(f"Started {process_name} process (PID: {proc.pid}, ID: {sampler_id}) on GPU {container_device} (host GPU: {host_gpu})")
            self.process_to_device_map[proc.pid] = container_device
            self.process_start_times[proc.pid] = time.time()
            return True
        else:
            return False

    def assign_gpu_device(self, min_free_memory_gib=50, max_utilization=20, assigned_gpus=None):
        """
        Assigns a GPU that has sufficient free memory and low utilization.
        Ensures that samplers are distributed across different GPUs.
        Tracks assigned GPUs in real-time within a single initialization cycle.
        """
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

            # Use assigned_gpus passed from the caller - otherwise fallback to existing assignments
            if assigned_gpus is None:
                assigned_gpus = set(self.process_to_device_map.values())

            available_gpus = []

            for host_gpu in visible_devices:
                container_device = f"cuda:{id_to_container_index[host_gpu]}"

                handle = pynvml.nvmlDeviceGetHandleByIndex(host_gpu)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory_gib = memory_info.free / (1024 ** 3)

                # If GPU looks free but bookkeeping says it's assigned, log a warning
                if util.gpu < max_utilization and container_device in assigned_gpus:
                    pid = next((p for p, d in self.process_to_device_map.items() if d == container_device), None)
                    self.resource_logger.warning(
                        f"GPU {container_device} appears idle (util {util.gpu}%) but assigned to PID {pid}. May need manual check."
                    )

                if container_device in assigned_gpus:
                    continue  # Skip GPUs that are already assigned

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
                f"Assigning GPU {host_gpu} (container {container_device}): Free {best_gpu[2]:.2f} GiB - Utilization {best_gpu[3]}%"
            )

            return host_gpu, container_device
        except Exception as e:
            self.resource_logger.error(f"Error in assign_gpu_device: {e}")
            return None


    def _select_process_to_terminate(self, processes, process_name, min_lifetime=60):
        """Select best process to terminate: lowest GPU utilization, respecting minimum lifetime.

        Args:
            processes: List of process objects
            process_name: "Sampler" or "Evaluator"
            min_lifetime: Minimum seconds a process must run before termination (default: 60)

        Returns:
            Index of process to terminate, or None if no eligible process found
        """
        now = time.time()
        eligible = []

        for i, proc in enumerate(processes):
            pid = proc.pid
            # Skip processes that haven't run long enough (still initializing)
            start_time = self.process_start_times.get(pid, 0)
            if now - start_time < min_lifetime:
                continue
            eligible.append((i, proc))

        if not eligible:
            return None

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
                return best_idx

        # Fallback: return oldest eligible process
        return eligible[0][0]

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

    async def terminate_process(self, processes, process_name, timeout=30):
        """Terminates a process and its children (vLLM may spawn subprocesses holding GPU memory)."""
        if not processes:
            return

        # Select best process to terminate (respects min lifetime, prefers low GPU util)
        idx = await asyncio.to_thread(self._select_process_to_terminate, processes, process_name)
        if idx is None:
            self.resource_logger.info(f"No {process_name} eligible for termination (all too young)")
            return

        proc = processes.pop(idx)
        pid = proc.pid

        # Get child PIDs before terminating (vLLM may spawn subprocesses)
        child_pids = []
        try:
            child_pids = await asyncio.to_thread(
                lambda: [c.pid for c in psutil.Process(pid).children(recursive=True)]
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Terminate with longer timeout for samplers (GPU cleanup)
        effective_timeout = 45 if process_name == "Sampler" else timeout
        proc.terminate()
        self.resource_logger.info(f"Terminating {process_name} (PID: {pid})...")

        await asyncio.to_thread(proc.join, effective_timeout)

        if proc.is_alive():
            self.resource_logger.warning(f"{process_name} (PID: {pid}) force killing")
            proc.kill()
            await asyncio.to_thread(proc.join)

        # Kill orphaned children (may hold GPU memory)
        for child_pid in child_pids:
            try:
                await asyncio.to_thread(psutil.Process(child_pid).kill)
                self.resource_logger.info(f"Killed orphaned child {child_pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Cleanup tracking maps
        self.process_start_times.pop(pid, None)
        if pid in self.process_to_device_map:
            device = self.process_to_device_map.pop(pid)
            self.resource_logger.info(f"Freed {device} (PID: {pid})")


    async def get_queue_message_count(self, queue):
        """
        Retrieves the current number of messages and consumers in the queue using RabbitMQ HTTP management API.

        Returns:
            tuple: (message_count, consumer_count) - both integers
        """
        if queue is None:
            return 0, 0

        try:
            # Get RabbitMQ connection details from config
            rabbitmq_host = self.config.rabbitmq.host
            rabbitmq_port = self.config.rabbitmq.management_port
            rabbitmq_user = self.config.rabbitmq.username
            rabbitmq_pass = self.config.rabbitmq.password
            # URL-encode vhost: empty string or '/' becomes '%2F'
            rabbitmq_vhost = '%2F' if not self.config.rabbitmq.vhost else self.config.rabbitmq.vhost

            url = f"http://{rabbitmq_host}:{rabbitmq_port}/api/queues/{rabbitmq_vhost}/{queue.name}"

            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout for cluster networks
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, auth=aiohttp.BasicAuth(rabbitmq_user, rabbitmq_pass)) as response:
                    if response.status == 200:
                        data = await response.json()
                        count = data.get('messages', 0)
                        consumer_count = data.get('consumers', 0)
                        self.resource_logger.info(f"Queue '{queue.name}': messages={count}, consumers={consumer_count}")
                        return count, consumer_count
                    else:
                        self.resource_logger.error(f"Failed to get queue stats from management API: {response.status}")
                        return 0, 0
        except Exception:
            self.resource_logger.error(
                f"Error getting message count for queue '{queue.name}':\n{traceback.format_exc()}"
            )
            return 0, 0

    async def get_rabbitmq_stats(self, config):
        """
        Get RabbitMQ server statistics via management API.

        Returns dict with:
        - memory_used_mb: Memory used by RabbitMQ in MB
        - memory_limit_mb: Memory limit in MB
        - memory_percent: Memory usage percentage
        - connection_count: Number of active connections
        - node_running: Whether the node is running
        - fd_used: File descriptors used
        - fd_total: Total file descriptors available
        """
        stats = {
            'memory_used_mb': None,
            'memory_limit_mb': None,
            'memory_percent': None,
            'connection_count': None,
            'node_running': None,
            'fd_used': None,
            'fd_total': None,
            'error': None
        }

        # Get management port from config (default 15672)
        management_port = getattr(config.rabbitmq, 'management_port', 15672)
        api_url = f"http://{config.rabbitmq.host}:{management_port}/api"

        try:
            auth = aiohttp.BasicAuth(config.rabbitmq.username, config.rabbitmq.password)
            timeout = aiohttp.ClientTimeout(total=5)

            async with aiohttp.ClientSession(auth=auth, timeout=timeout) as session:
                # Get node overview
                async with session.get(f"{api_url}/overview") as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        # Memory stats
                        if 'queue_totals' in data:
                            stats['total_messages'] = data['queue_totals'].get('messages', 0)

                        # Get node-specific stats
                        node_name = data.get('node', '')
                        if node_name:
                            async with session.get(f"{api_url}/nodes/{node_name}") as node_resp:
                                if node_resp.status == 200:
                                    node_data = await node_resp.json()

                                    # Memory
                                    mem_used = node_data.get('mem_used', 0)
                                    mem_limit = node_data.get('mem_limit', 0)
                                    stats['memory_used_mb'] = mem_used / (1024 * 1024)
                                    stats['memory_limit_mb'] = mem_limit / (1024 * 1024)
                                    if mem_limit > 0:
                                        stats['memory_percent'] = (mem_used / mem_limit) * 100

                                    # File descriptors
                                    stats['fd_used'] = node_data.get('fd_used', 0)
                                    stats['fd_total'] = node_data.get('fd_total', 0)

                                    # Node status
                                    stats['node_running'] = node_data.get('running', False)

                # Get connection count
                async with session.get(f"{api_url}/connections") as conn_resp:
                    if conn_resp.status == 200:
                        connections = await conn_resp.json()
                        stats['connection_count'] = len(connections)

        except aiohttp.ClientConnectorError:
            stats['error'] = "Cannot connect to RabbitMQ management API (is it enabled?)"
        except TimeoutError:
            stats['error'] = "Timeout connecting to RabbitMQ management API"
        except Exception as e:
            stats['error'] = f"Error fetching RabbitMQ stats: {e}"

        return stats

    async def check_rabbitmq_health(self, config, log_details=True):
        """
        Check RabbitMQ health and log warnings if issues detected.

        Returns:
        - 'healthy': No issues detected
        - 'warning': Some concerns but not critical
        - 'critical': Critical issues detected
        - 'unknown': Cannot determine health
        """
        stats = await self.get_rabbitmq_stats(config)

        if stats['error']:
            self.resource_logger.warning(f"RabbitMQ health check: {stats['error']}")
            return 'unknown'

        health_status = 'healthy'
        issues = []

        # Check memory usage
        if stats['memory_percent'] is not None:
            if stats['memory_percent'] > 90:
                issues.append(f"Memory usage critical: {stats['memory_percent']:.1f}%")
                health_status = 'critical'
            elif stats['memory_percent'] > 75:
                issues.append(f"Memory usage high: {stats['memory_percent']:.1f}%")
                if health_status != 'critical':
                    health_status = 'warning'

        # Check connection count (warn if very high)
        if stats['connection_count'] is not None and stats['connection_count'] > 1000:
            issues.append(f"High connection count: {stats['connection_count']}")
            if health_status != 'critical':
                health_status = 'warning'

        # Check file descriptors
        if stats['fd_used'] and stats['fd_total']:
            fd_percent = (stats['fd_used'] / stats['fd_total']) * 100
            if fd_percent > 90:
                issues.append(f"File descriptors critical: {fd_percent:.1f}%")
                health_status = 'critical'
            elif fd_percent > 75:
                issues.append(f"File descriptors high: {fd_percent:.1f}%")
                if health_status != 'critical':
                    health_status = 'warning'

        # Check if node is running
        if stats['node_running'] is False:
            issues.append("RabbitMQ node is not running!")
            health_status = 'critical'

        # Log results
        if log_details or health_status != 'healthy':
            if issues:
                self.resource_logger.warning(f"RabbitMQ health: {health_status.upper()}, {'; '.join(issues)}")
            else:
                self.resource_logger.info(
                    f"RabbitMQ health: {health_status.upper()} - "
                    f"Memory: {stats['memory_used_mb']:.0f}/{stats['memory_limit_mb']:.0f} MB "
                    f"({stats['memory_percent']:.1f}%) - "
                    f"Connections: {stats['connection_count']} - "
                    f"FD: {stats['fd_used']}/{stats['fd_total']}"
                )

        return health_status

    async def measure_rabbitmq_latency(self, config):
        """
        Measure network latency to RabbitMQ server by pinging the management API.

        Returns latency in milliseconds - or None if unreachable.
        """
        management_port = getattr(config.rabbitmq, 'management_port', 15672)
        api_url = f"http://{config.rabbitmq.host}:{management_port}/api/overview"

        try:
            auth = aiohttp.BasicAuth(config.rabbitmq.username, config.rabbitmq.password)
            timeout = aiohttp.ClientTimeout(total=5)

            start_time = time.time()
            async with aiohttp.ClientSession(auth=auth, timeout=timeout) as session:
                async with session.get(api_url) as resp:
                    await resp.read()  # Ensure full response is received
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            return latency_ms

        except Exception as e:
            self.resource_logger.warning(f"Cannot measure RabbitMQ latency: {e}")
            return None
