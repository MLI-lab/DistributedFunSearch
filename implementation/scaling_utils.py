import os  
import psutil 
import pynvml  
import logging 
import asyncio  
import threading  
import torch.multiprocessing as mp  
from typing import Sequence, Any
import logging
from logging import FileHandler
import socket  # Import socket for hostname retrieval
import datetime

class ResourceManager:
    def __init__(self, log_dir=None, resource_logger=None, cpu_only=False):
        """
        Initialize the ResourceManager.

        Args:
            log_dir (str): Directory to store logs.
            resource_logger (logging.Logger): Existing logger instance. If None, a new one is created.
            cpu_only (bool): If True, operate in CPU-only mode and skip all GPU-related initialization and operations.
        """
        self.hostname = socket.gethostname()  # Get the hostname of the machine

        if resource_logger is None:
            if log_dir is None:
                raise ValueError("Either resource_logger or log_dir must be provided")
            self.resource_logger = self._initialize_resource_logger(log_dir)
        else:
            self.resource_logger = resource_logger

        self.cpu_only = cpu_only
        self.process_to_device_map = {}

        if not self.cpu_only:
            try:
                self._initialize_nvml()
            except Exception as e:
                self.resource_logger.warning(f"{self.hostname}: Failed to initialize NVML: {e}")
                self.cpu_only = True  # Fallback to CPU-only mode if GPU initialization fails
                self.resource_logger.info(f"{self.hostname}: Switching to CPU-only mode.")

    def _initialize_nvml(self):
        """
        Initialize NVML for GPU monitoring.
        Raises an exception if NVML cannot be initialized.
        """
        pynvml.nvmlInit()
        self.resource_logger.debug("Successfully initialized NVML for GPU monitoring.")

    def _initialize_resource_logger(self, log_dir):
        """
        Initialize a logger with a unique log file name for each process.
        The log file name includes the process ID (PID) or other unique identifiers.
        """
        pid = os.getpid()  # Get the current process ID
        log_file_name = f"resources_{self.hostname}_pid{pid}.log"
        log_file_path = os.path.join(log_dir, log_file_name)

        logger = logging.getLogger(f'resource_logger_{pid}')
        logger.setLevel(logging.DEBUG)

        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Configure file handler with the unique log file
        handler = FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        logger.info(f"Resource logger initialized for PID {pid}. Log file: {log_file_path}")
        return logger


    async def log_resource_stats_periodically(self, interval=100):
        """Log available CPU, GPU, RAM, and system load/utilization every `interval` seconds."""
        while True:
            try:
                # Log CPU usage
                cpu_affinity = os.sched_getaffinity(0)
                cpu_usage = psutil.cpu_percent(interval=None, percpu=True)
                available_cpu_usage = [cpu_usage[i] for i in cpu_affinity]
                avg_cpu_usage = sum(available_cpu_usage) / len(available_cpu_usage)
                self.resource_logger.info(f"{self.hostname}: Available CPUs: {len(cpu_affinity)}, Average CPU Usage: {avg_cpu_usage:.2f}%")

                if not self.cpu_only:
                    # Log GPU usage
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                        free_memory_mib = memory_info.free / 1024**2
                        total_memory_mib = memory_info.total / 1024**2
                        gpu_utilization = utilization.gpu
                        self.resource_logger.info(f"{self.hostname}: GPU {i}: Free Memory = {free_memory_mib:.2f} MiB / {total_memory_mib:.2f} MiB, GPU Utilization = {gpu_utilization}%")

                # Log RAM usage
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                rss_mib = memory_info.rss / 1024**2
                vms_mib = memory_info.vms / 1024**2
                self.resource_logger.info(f"{self.hostname}: Memory Usage: RSS = {rss_mib:.2f} MiB, VMS = {vms_mib:.2f} MiB")

                # Log system load
                load_avg_1, load_avg_5, load_avg_15 = os.getloadavg()
                num_cores = len(cpu_affinity)
                self.resource_logger.info(f"{self.hostname}: System Load (1m, 5m, 15m): {load_avg_1:.2f}, {load_avg_5:.2f}, {load_avg_15:.2f}, Load/Cores Ratio (1m): {load_avg_1:.2f}/{num_cores} ({load_avg_1 / num_cores:.2f})")

            except psutil.Error as e:
                self.resource_logger.error(f"{self.hostname}: Failed to query CPU or RAM information: {e}")
            except pynvml.NVMLError as e:
                if not self.cpu_only:
                    self.resource_logger.error(f"{self.hostname}: Failed to query GPU information: {e}")
            finally:
                await asyncio.sleep(interval)  # Wait for the specified interval before checking again


    def start_process(self, target_fnc, args, processes, process_name):
        current_pid = os.getpid()
        current_thread = threading.current_thread().name
        thread_id = threading.get_ident()


        # CPU check for evaluator processes
        if process_name == 'Evaluator':
            cpu_affinity = os.sched_getaffinity(0)  # Get CPUs available to the container
            cpu_usage = psutil.cpu_percent(percpu=True)  # Get usage for all system CPUs
            container_cpu_usage = [cpu_usage[i] for i in cpu_affinity]

            # Count how many of the available CPUs are under 50% usage
            available_cpus_with_low_usage = sum(1 for usage in container_cpu_usage if usage < 50)
            self.resource_logger.info(f"Available CPUs with <50% usage (in container): {available_cpus_with_low_usage}")

            # Scale up only if more than 4 CPUs have less than 50% usage
            if available_cpus_with_low_usage <= 4:
                self.resource_logger.info(f"Cannot scale up {process_name}: Not enough available CPU resources.")
                return  # Exit the function if not enough CPU resources

        # GPU check for sampler processes
        if process_name == 'Sampler':
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]  # Ensure non-empty strings and convert to int

            # Initialize GPU memory info and utilization
            gpu_memory_info = {}

            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError as e:
                self.resource_logger.error(f"Failed to initialize NVML: {e}")
                device = 'cpu' if self.cpu_only else 'cuda'
                self.resource_logger.warning(f"Proceeding with device={device} for {process_name}.")
                args += (device,)
                try:
                    # Start the process
                    proc = mp.Process(target=target_fnc, args=args, name=f"{process_name}-{len(processes)}")
                    proc.start()
                    processes.append(proc)
                    return
                except Exception as e:
                    self.resource_logger.error(f"Could not start process: {e}")
                    return

            for dev_id in visible_devices:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    free_memory_mib = memory_info.free / 1024**2  # Convert bytes to MiB
                    gpu_utilization = utilization.gpu  # Percentage
                    gpu_memory_info[dev_id] = (free_memory_mib, gpu_utilization)
                except pynvml.NVMLError as e:
                    self.resource_logger.error(f"Error querying memory for device {dev_id}: {e}")
                    gpu_memory_info[dev_id] = (None, None)  # Set to None on error

            suitable_gpu_id = None
            combined_memory = 0
            combined_gpus = []

            # Check if any single GPU has >= 32 GiB of memory free and < 50% utilization
            for dev_id, (free_memory, utilization) in gpu_memory_info.items():
                if free_memory is None and utilization is None:
                    self.resource_logger.warning(f"Memory information could not be queried for device {dev_id}. Skipping this device.")
                    continue  # Skip this device and continue checking others
                if free_memory > 32768 and utilization < 50:  # Need more than 32 GiB for large models
                    suitable_gpu_id = dev_id
                    self.resource_logger.info(f"Device {dev_id} has sufficient free memory ({free_memory} MiB) and low utilization ({utilization}%).")
                    break
                elif utilization < 50:
                    combined_memory += free_memory if free_memory else 0
                    combined_gpus.append(dev_id)

            # Assign device based on availability
            if suitable_gpu_id is not None:
                device = f"cuda:{suitable_gpu_id}"
            elif combined_memory >= 32768:
                device = 'cuda'  # Use multiple GPUs
                self.resource_logger.info(f"Using combination of GPUs: {combined_gpus} with total memory: {combined_memory} MiB")
            else:
                self.resource_logger.warning(f"Not enough GPU memory available for {process_name}. Skipping process start.")
                return

            self.resource_logger.info(f"Assigning {process_name} to device {device if device else 'combined GPUs (cuda)'}")
            args += (device,)  # Append the device to args

        # Start the process
        try:
            proc = mp.Process(target=target_fnc, args=args, name=f"{process_name}-{len(processes)}")
            proc.start()
            processes.append(proc)
            # Store the process PID and device in the map only for sampler processes
            if process_name == 'Sampler':
                self.process_to_device_map[proc.pid] = device
        except Exception as e:
            self.resource_logger.error(f"Could not start process because {e}.")


    def start_process(self, target_fnc, args, processes, process_name):
        """
        Start a new process. If in CPU-only mode, skip GPU checks.
        """
        if not self.cpu_only and process_name == 'Sampler':
            # GPU-specific checks here
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]

            if not visible_devices:
                self.resource_logger.warning(f"No GPUs available for {process_name}. Skipping process start.")
                return

        # Start the process
        proc = mp.Process(target=target_fnc, args=args, name=f"{process_name}-{len(processes)}")
        proc.start()
        processes.append(proc)

    # Other methods can also use the `self.cpu_only` flag to skip GPU-specific operations as needed.


    async def get_queue_message_count(self, channel, queue_name):
        try:
            queue = await channel.declare_queue(queue_name, passive=True)
            message_count = queue.declaration_result.message_count
            return message_count
        except Exception as e:
            self.resource_logger(f"Error getting message count for queue {queue_name}: {e}")
            return 0

    async def adjust_processes(self, message_count, threshold, processes, target_fnc, args, max_processes, min_processes, process_name):
        num_processes = len(processes)
        self.resource_logger.debug(f"Adjusting {process_name}: message_count={message_count}, threshold={threshold}, num_processes={num_processes}, min_processes={min_processes}")

        if message_count > threshold and num_processes < max_processes:
            # Scale up
            self.start_process(target_fnc, args, processes, process_name)
            current_processes = len(processes)
            if current_processes > num_processes:
                self.resource_logger.info(f"Scaled up {process_name} processes to {current_processes}")
            else:
                self.resource_logger.info(f"Could not scale up {process_name} processes; still at {current_processes}")

        elif message_count < threshold and num_processes > min_processes:
            # Scale down
            self.terminate_process(processes, process_name)
            current_processes = len(processes)
            if current_processes < num_processes:
                self.resource_logger.info(f"Scaled down {process_name} processes to {current_processes}")
            else: 
                self.resource_logger.info(f"Could not scale down {process_name} processes; still at {current_processes}")
        else: 
            self.resource_logger.info(f"No scaling action needed for {process_name}. Current processes: {num_processes}, Message count: {message_count}")
            return 


    def terminate_process(self, processes, process_name, immediate=False):
        if processes:
            if immediate:
                process_to_terminate = processes.pop(0)
                self.resource_logger.info(f"Immediately terminating {process_name} process with PID: {process_to_terminate.pid}")
            else:
                # Try to get the least busy process
                if process_name.startswith('Evaluator'):
                    least_busy_process = self.get_process_with_zero_or_lowest_cpu(processes)
                elif process_name.startswith('Sampler'): 
                    least_busy_process = self.get_process_with_zero_or_lowest_gpu(processes)
                else: 
                    self.resource_logger.info(f"No Sampler or Evaluator process is {process_name}")
                    return 
                self.resource_logger.info(f"least_busy_process is {least_busy_process}")
                if least_busy_process is None:
                    return
                else:
                    # Remove the chosen process from the list
                    processes.remove(least_busy_process)

                if least_busy_process.is_alive():
                    self.resource_logger.info(f"Initiating termination for {process_name} process with PID: {least_busy_process.pid}")
                    least_busy_process.terminate()
                    least_busy_process.join(timeout=10)  # Wait for it to fully terminate
                    if least_busy_process.is_alive():
                        self.resource_logger.warning(f"{process_name} process with PID: {least_busy_process.pid} is still alive after timeout, forcing kill.")
                        least_busy_process.kill()
                    self.resource_logger.info(f"{process_name} process with PID: {least_busy_process.pid} terminated successfully.")
        else:
            self.resource_logger.warning(f"No {process_name} processes to terminate.")

    def get_process_with_zero_or_lowest_cpu(self, processes, cpu_utilization_threshold=20):
        """Find a process to terminate based on CPU utilization."""
        for proc in processes:
            try:
                p = psutil.Process(proc.pid)
                cpu_usage = p.cpu_percent(interval=1)
                self.resource_logger.debug(f"Process PID {proc.pid} CPU utilization: {cpu_usage}%")

                # If CPU utilization is below the threshold, select this process for termination
                if cpu_usage < cpu_utilization_threshold:
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.resource_logger.warning(f"Failed to access CPU usage for process PID {proc.pid}. It might have finished or access was denied.")
                continue

        # If no process meets the threshold, return None
        self.resource_logger.info(f"No process with CPU utilization below {cpu_utilization_threshold}% found.")
        return None


    def get_process_with_zero_or_lowest_gpu(self, processes, gpu_utilization_threshold=10):
        """Find a process to terminate based on GPU utilization."""
        try:
            for proc in processes:
                try:
                    # Try to extract the GPU device from the process arguments 
                    device = self.process_to_device_map.get(proc.pid)
                    if device and device != 'cuda':
                        # Extract the GPU index from the device string, e.g., "cuda:0" -> 0
                        gpu_index = int(device.split(":")[1])
                    
                        # Get GPU utilization percentage using pynvml
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  

                        # If GPU utilization is below the threshold, select this process for termination
                        if gpu_utilization < gpu_utilization_threshold:
                            self.resource_logger.info(f"Process with PID {proc.pid} is using GPU {gpu_index} with utilization {gpu_utilization}%, below threshold {gpu_utilization_threshold}%.")
                            return proc
                    else:
                        self.resource_logger.info(f"Process PID {proc.pid} does not have a GPU device argument.")
                except pynvml.NVMLError as e:
                    self.resource_logger.warning(f"Failed to get GPU utilization for process PID {proc.pid}: {e}")
                    continue
                except Exception as e:
                    self.resource_logger.warning(f"Error checking GPU utilization for process PID {proc.pid}: {e}")
                    continue

            # If no GPU-based process has utilization below the threshold, return None
            self.resource_logger.info("No GPU-based process found with GPU utilization below threshold.")
            return None

        except Exception as e:
            self.resource_logger.error(f"Error occurred while checking GPU utilization: {e}, falling back to cpu based check.")
            return self.get_process_with_zero_or_lowest_cpu(processes)