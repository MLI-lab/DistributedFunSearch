import os
import psutil
import pynvml
import logging
import asyncio
import torch.multiprocessing as mp
from logging import FileHandler
import socket
import statistics





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
        self.process_to_device_map = {}
        self.scaling_config = scaling_config
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


    async def has_enough_system_memory(self, min_free_gib=None):
        """Check if system has enough free memory.

        Args:
            min_free_gib: Minimum free memory in GiB. If None, uses scaling_config value (default: 30)
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
        Logs system resource usage periodically, averaging values over `sample_duration` seconds.
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
                    gpu_samples = []
                
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
                    disk_read_samples.append(disk_io.read_bytes / 1e6)  # Convert to MB
                    disk_write_samples.append(disk_io.write_bytes / 1e6)

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
                            gpu_samples.append(await self.async_get_gpu_usage())
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
                    f"Avg CPU: {avg_cpu:.2f}%, Load: {avg_load:.2f}, I/O Wait: {avg_io_wait:.2f}%, "
                    f"Ctx Switches: {avg_ctx_switch}, Disk Read/Write: {avg_disk_read:.2f}/{avg_disk_write:.2f} MB, "
                    f"Mem Usage: {avg_mem:.2f}%, Swap: {avg_swap:.2f}%, D-State Processes: {avg_d_state}"
                )

                # Include GPU if applicable
                if not self.cpu_only:
                    avg_gpu = statistics.mean(gpu_samples) if gpu_samples else 0
                    log_message += f", GPU Usage: {avg_gpu:.2f}%"

                self.resource_logger.info(log_message)

            except Exception as e:
                self.resource_logger.error(f"Error logging resource stats: {e}")

            await asyncio.sleep(interval)

    async def async_get_cpu_usage(self):
        """Retrieves CPU usage asynchronously."""
        return await asyncio.to_thread(psutil.cpu_percent, interval=1)

    async def async_get_gpu_usage(self):
        """Retrieves GPU utilization asynchronously."""
        if self.cpu_only:
            return 0
        try:
            handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, 0)
            utilization = await asyncio.to_thread(pynvml.nvmlDeviceGetUtilizationRates, handle)
            return utilization.gpu
        except Exception as e:
            self.resource_logger.warning(f"GPU monitoring failed: {e}")
            return 0


    async def run_scaling_loop(self, evaluator_queue=None, sampler_queue=None, evaluator_processes=None,
                               sampler_processes=None, sampler_entry_function=None, evaluator_entry_function=None,
                               config_path=None, log_dir=None, template=None, inputs=None, target_signatures=None,
                               sandbox_base_path=None, max_evaluators=10000, min_evaluators=1,
                               max_samplers=1000, min_samplers=1, check_interval=120):
        """Scales evaluator and sampler processes dynamically based on queue sizes and system resources."""
        self.resource_logger.info("Starting scaling loop")
        evaluator_processes = evaluator_processes or []
        sampler_processes = sampler_processes or []
        max_evaluators = max_evaluators if max_evaluators is not None else 0

        try:
            while True:
                try:
                    evaluator_message_count = await self.get_queue_message_count(evaluator_queue) if evaluator_queue else 0
                    sampler_message_count = await self.get_queue_message_count(sampler_queue) if sampler_queue else 0
                    self.resource_logger.info(f"Message counts are {evaluator_message_count} and {sampler_message_count}")
                    # Scale Evaluators
                    evaluator_scaled = False
                    if evaluator_queue and max_evaluators > 0:
                        can_scale_eval = await self.can_scale_evaluator()
                        evaluator_threshold = self.scaling_config.evaluator_scale_up_threshold if self.scaling_config else 10
                        if evaluator_message_count > evaluator_threshold and len(evaluator_processes) < max_evaluators and can_scale_eval:
                            self.resource_logger.info(f"Can scale evaluators with messages in queue {evaluator_message_count}")
                            self.start_evaluator_process(evaluator_entry_function, config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, evaluator_processes, "Evaluator")
                            evaluator_scaled = True
                        elif evaluator_message_count == 0 and len(evaluator_processes) > min_evaluators:
                            self.resource_logger.info(f"Zero messages in the queue and not last Evaluator, terminating ...")
                            await self.terminate_process(evaluator_processes, "Evaluator")
                            evaluator_scaled = True

                    # Scale Samplers
                    sampler_scaled = False
                    if sampler_queue and max_samplers > 0:
                        assignment = await self.can_scale_up_samplers()
                        if self.cpu_only:
                             assignment = await self.can_scale_evaluator() # if we are in cpu only mode also check cpu load for samplers
                        self.resource_logger.info(f"Assignment is {assignment}")
                        sampler_threshold = self.scaling_config.sampler_scale_up_threshold if self.scaling_config else 50
                        if sampler_message_count > sampler_threshold and len(sampler_processes) < max_samplers and assignment and await self.has_enough_system_memory():
                            self.resource_logger.info(f"Can scale samplers with messages in queue  {sampler_message_count}")
                            started = self.start_sampler_process(sampler_entry_function, config_path, log_dir, sampler_processes, "Sampler", assignment=assignment)
                            if not started:
                                self.resource_logger.info("No available GPU found. Skipping sampler scale-up.")
                            sampler_scaled = True
                        elif sampler_message_count == 0 and len(sampler_processes) > min_samplers:
                            self.resource_logger.info(f"Can terminate a sampler with messages in queue {sampler_message_count}")
                            await self.terminate_process(sampler_processes, "Sampler")
                            sampler_scaled = True

                    # If nothing was scaled, log that scaling was skipped
                    if not evaluator_scaled and not sampler_scaled:
                        self.resource_logger.info("No scaling action taken in this iteration.")

                except Exception as e:
                    self.resource_logger.error(f"Scaling loop encountered an error: {e}")

                await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            self.resource_logger.info("Scaling loop cancelled, stopping gracefully...")
            raise  # Re-raise to properly propagate cancellation

    def start_evaluator_process(self, entry_function, config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, processes, process_name):
        """Starts a new evaluator process using 'fork' multiprocessing context.

        Uses fork because evaluators don't load ML models and only execute functions
        in sandboxed subprocesses. Fork is faster and has no threading deadlock risk.
        """
        ctx = mp.get_context('fork')
        proc = ctx.Process(
            target=entry_function,
            args=(config_path, template, inputs, target_signatures, log_dir, sandbox_base_path),
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
        Returns a GPU assignment tuple (host_gpu, container_device) if we can
        scale up samplers, or None if we cannot.
        """
        if self.cpu_only:
            # No GPUs available at all
            return None

        # See if any GPU is free enough, using config values
        min_memory = self.scaling_config.min_gpu_memory_gib if self.scaling_config else 20
        max_util = self.scaling_config.max_gpu_utilization if self.scaling_config else 50
        assignment = self.assign_gpu_device(min_free_memory_gib=min_memory, max_utilization=max_util)
        return assignment  

    async def can_scale_evaluator(self, cpu_usage_threshold=None, normalized_load_threshold=None, duration=10, interval=1):
        """
        Determine if it's safe to scale up evaluators based on CPU usage and system load.

        Args:
            cpu_usage_threshold: Maximum allowed average CPU usage percentage. If None, uses scaling_config value (default: 99).
            normalized_load_threshold: Maximum allowed 1-minute load (load average divided by available cores). If None, uses scaling_config value (default: 0.99).
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
        available_cores = len(os.sched_getaffinity(0))
        normalized_load = load_avg_1 / available_cores if available_cores > 0 else load_avg_1

        self.resource_logger.info(
            f"{self.hostname}: Smoothed Avg CPU Usage: {avg_cpu_usage:.2f}% | "
            f"Normalized Load: {normalized_load:.2f} (Load per core)"
        )

        # Return True only if both CPU metrics are below threshold and a GPU is available.
        return (avg_cpu_usage < cpu_usage_threshold) and (normalized_load < normalized_load_threshold)


    def start_sampler_process(self, entry_function, config_path, log_dir, processes, process_name, assignment):
        """Starts a new sampler process using 'spawn' multiprocessing context.

        Uses spawn to avoid fork+threading deadlocks when loading ML models (StarCoder2/GPT).
        Spawn creates a clean process without inheriting thread state from parent.
        """
        ctx = mp.get_context('spawn')
        if assignment is True:  # CPU-only mode, no GPU assignment
            proc = ctx.Process(
                target=entry_function,
                args=(config_path, None, log_dir),  # No GPU device
                name=f"{process_name}-{len(processes)}"
            )
            proc.start()
            processes.append(proc)
            self.resource_logger.info(f"Started {process_name} process (PID: {proc.pid}) in CPU-only mode.")
            self.process_to_device_map[proc.pid] = None
            return True
        elif assignment is not None:
            # GPU assignment is available
            host_gpu, container_device = assignment
            proc = ctx.Process(
                target=entry_function,
                args=(config_path, container_device, log_dir),
                name=f"{process_name}-{len(processes)}"
            )
            proc.start()
            processes.append(proc)
            self.resource_logger.info(f"Started {process_name} process (PID: {proc.pid}) on GPU {container_device} (host GPU: {host_gpu})")
            self.process_to_device_map[proc.pid] = container_device
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
            #id_to_container_index = {host_id: container_id for container_id, host_id in enumerate(visible_devices)}
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


    async def terminate_process(self, processes, process_name, timeout=30):
        """Terminates a running process and ensures it fully exits (async to avoid blocking)."""
        if processes:
            proc = processes.pop(0)
            pid = proc.pid
            proc.terminate()
            self.resource_logger.info(f"Sent SIGTERM to {process_name} process (PID: {pid}). Waiting for termination...")

            # Use asyncio.to_thread to avoid blocking the event loop
            await asyncio.to_thread(proc.join, timeout)

            if proc.is_alive():  # If still running, force kill
                self.resource_logger.warning(f"{process_name} process (PID: {pid}) did not terminate in {timeout}s. Sending SIGKILL.")
                proc.kill()
                await asyncio.to_thread(proc.join)

            # Clean up GPU assignment map if this was a sampler process
            if pid in self.process_to_device_map:
                device = self.process_to_device_map.pop(pid)
                self.resource_logger.info(f"Freed GPU assignment for PID {pid}: {device}")

            self.resource_logger.info(f"Terminated {process_name} process (PID: {pid})")


    async def get_queue_message_count(self, queue):
        """
        Retrieves the current number of messages in the queue
        by passively re-declaring `queue.name` on `queue.channel`.
        """
        if queue is None:
            return 0

        try:
            declared_queue = await queue.channel.declare_queue(queue.name, passive=True)
            return declared_queue.declaration_result.message_count
        except Exception as e:
            self.resource_logger.error(
                f"Error getting message count for queue '{queue.name}': {e}"
            )
            return 0


