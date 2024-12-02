import asyncio
import logging
from logging import FileHandler
from multiprocessing import current_process
import argparse
import aio_pika
from yarl import URL
import torch.multiprocessing as mp
import os
import signal
import sys
import GPUtil
from typing import Sequence, Any
from scaling_utils import ResourceManager
from yarl import URL
import code_manipulation
import sampler
import importlib

def load_config(config_path):
    """
    Dynamically load a configuration module from a specified path.
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
    def __init__(self, specification: str, inputs: Sequence[Any], config, check_interval_sam, log_dir):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger(log_dir)
        self.check_interval_sam = check_interval_sam
        self.evaluator_processes = []
        self.database_processes = []
        self.sampler_processes = []
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None
        self.resource_manager = ResourceManager(log_dir=log_dir)

    def initialize_logger(self, log_dir):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)

        log_file_path = os.path.join(log_dir, 'sampler.log')  # Path to the log file

        handler = FileHandler(log_file_path, mode='w')  # Create a file handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger


    async def scaling_controller(self, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)
        check_interval_sam = 120
        max_samplers = 16
        min_samplers = 1
        sampler_threshold = 15
        initial_sleep_duration = 120  
        await asyncio.sleep(initial_sleep_duration)

        # Create a connection and channels for getting queue metrics
        try:
            connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
            )
            channel = await connection.channel()
        except Exception as e:
            self.logger.error(f"Error connecting to RabbitMQ: {e}")
            return

        while True:
            try:
                sampler_message_count = await self.resource_manager.get_queue_message_count(channel, "sampler_queue")

                # Adjust sampler processes
                await self.resource_manager.adjust_processes(
                    sampler_message_count, sampler_threshold,
                    self.sampler_processes, self.sampler_process,
                    args=(amqp_url,),
                    max_processes=max_samplers, min_processes=min_samplers,
                    process_name='Sampler'
                )
            except Exception as e:
                self.logger.error(f"Scaling controller encountered an error: {e}")
            await asyncio.sleep(120)  # Non-blocking sleep


    async def main_task(self,  enable_scaling=True):
        amqp_url = URL(
            f'amqp://{self.config.rabbitmq.username}:{self.config.rabbitmq.password}@{self.config.rabbitmq.host}:{self.config.rabbitmq.port}/{self.config.rabbitmq.vhost}'
        ).update_query(heartbeat=480000)
        pid = os.getpid()
        self.logger.info(f"Main_task is running in process with PID: {pid}")
        try:

            self.template = code_manipulation.text_to_program(self.specification)
            function_to_evolve = 'priority'

            self.start_initial_processes(function_to_evolve, amqp_url)
            resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=200))

            self.tasks = [resource_logging_task]
            if enable_scaling:
                scaling_task = asyncio.create_task(self.scaling_controller(function_to_evolve, amqp_url))
                self.tasks.append(scaling_task)
    
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception occurred in main_task: {e}")

    def start_initial_processes(self, function_to_evolve, amqp_url):
        amqp_url = str(amqp_url)

        # Get a list of visible GPUs as remapped by CUDA_VISIBLE_DEVICES (inside the container)
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        visible_devices = [int(dev.strip()) for dev in visible_devices if dev.strip()]  # Host-visible device IDs

        gpus = GPUtil.getGPUs()

        # Create a mapping of host-visible GPU IDs (integers in visible_devices) to container-visible device indices
        id_to_container_index = {visible_devices[i]: i for i in range(len(visible_devices))}

        # Initialize GPU memory usage and utilization tracking (host-visible GPU IDs)
        gpu_memory_info = {gpu.id: (gpu.memoryFree, gpu.memoryUtil * 100) for gpu in gpus if gpu.id in visible_devices}

        self.logger.info(f"Found visible GPUs with initial free memory and utilization: {gpu_memory_info}")

        # Start initial sampler processes
        for i in range(self.config.num_samplers):
            suitable_gpu_id = None
            combined_memory = 0
            combined_gpus = []

            # Check if any single GPU has >= 32 GiB of memory free and < 50% utilization
            for gpu_id, (free_memory, utilization) in gpu_memory_info.items():
                if free_memory > 30000 and utilization < 110: 
                    suitable_gpu_id = gpu_id
                    break
                elif utilization < 100:
                    combined_memory += free_memory
                    combined_gpus.append(gpu_id)

            # If a single GPU was found with sufficient memory
            if suitable_gpu_id is not None:
                container_index = id_to_container_index[suitable_gpu_id]
                device = f"cuda:{container_index}"
                # Adjust memory tracking (simplistic estimation)
                gpu_memory_info[suitable_gpu_id] = (gpu_memory_info[suitable_gpu_id][0] - 32768, gpu_memory_info[suitable_gpu_id][1])
            elif combined_memory >= 32768:  # If combined memory from multiple GPUs is >= 20 GiB
                device = None  # Use None to indicate that multiple GPUs will be used
                self.logger.info(f"Using combination of GPUs: {combined_gpus} with total memory: {combined_memory} MiB")
            else:
                self.logger.error(f"Cannot start sampler {i}: Not enough available GPU memory.")
                continue  # Skip this sampler if no GPU has sufficient memory

            self.logger.info(f"Assigning sampler {i} to device {device if device else 'combined GPUs (None)'}")
            try: 
                proc = mp.Process(target=self.sampler_process, args=(amqp_url, device), name=f"Sampler-{i}")
                proc.start()
                self.logger.debug(f"Started Sampler Process {i} on {device} with PID: {proc.pid}")
                self.sampler_processes.append(proc)
                self.process_to_device_map[proc.pid] = device
            except Exception as e: 
                continue



    def sampler_process(self, amqp_url, device):

        local_id = current_process().pid  # Use process ID as a local identifier

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Initialize these variables at a higher scope to be accessible in signal_handler
        connection = None
        channel = None
        sampler_task = None

        async def graceful_shutdown(loop, connection, channel, sampler_task):
            self.logger.info(f"Sampler {local_id}: Initiating graceful shutdown...")

            if sampler_task:
                try:
                    await asyncio.wait_for(sampler_task, timeout=10)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Sampler {local_id}: Task timed out. Cancelling...")
                    sampler_task.cancel()
                    await sampler_task  # Ensure task cancellation completes

            if channel:
                await channel.close()
            if connection:
                await connection.close()

            loop.stop()
            self.logger.info(f"Sampler {local_id}: Graceful shutdown complete.")

        def signal_handler(sig, frame):
            self.logger.info(f"Sampler process {local_id} received signal {sig}. Initiating shutdown.")
            loop.create_task(graceful_shutdown(loop, connection, channel, sampler_task))

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        async def run_sampler():
            nonlocal connection, channel, sampler_task  # Access the outer-scoped variables
            try:
                self.logger.debug(f"Sampler {local_id}: Starting connection to RabbitMQ.")
                connection = await aio_pika.connect_robust(
                    amqp_url,
                    timeout=300,
                    client_properties={"connection_attempts": 1, "retry_delay": 0}
                )
                self.logger.debug(f"Sampler {local_id}: Connected to RabbitMQ.")
                channel = await connection.channel()
                self.logger.debug(f"Sampler {local_id}: Channel established.")

                sampler_queue = await channel.declare_queue(
                    "sampler_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared sampler_queue.")

                evaluator_queue = await channel.declare_queue(
                    "evaluator_queue", durable=False, auto_delete=False,
                    arguments={'x-consumer-timeout': 360000000}
                )
                self.logger.debug(f"Sampler {local_id}: Declared evaluator_queue.")

                sampler_instance = sampler.Sampler(
                    connection, channel, sampler_queue, evaluator_queue, self.config, device
                )
                self.logger.debug(f"Sampler {local_id}: Initialized Sampler instance.")

                sampler_task = asyncio.create_task(sampler_instance.consume_and_process())
                await sampler_task
            except asyncio.CancelledError:
                self.logger.info(f"Sampler {local_id}: Process was cancelled.")
            except Exception as e:
                self.logger.error(f"Sampler {local_id} encountered an error: {e}")
            finally:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                self.logger.debug(f"Sampler {local_id}: Connection closed.")

        try:
            loop.run_until_complete(run_sampler())
        except Exception as e:
            self.logger.info(f"Sampler process {local_id}: Exception occurred: {e}")
        finally:
            loop.close()
            self.logger.debug(f"Sampler process {local_id} has been closed gracefully.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the TaskManager with configurable scaling interval.")
    parser.add_argument("--check_interval_sam", type=int, default=200, help="Interval in seconds for scaling sampler processes.")
    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling of evaluators and samplers (enabled by default).",
    )
    parser.add_argument(
        "--spec-path",
        type=str,
        default=os.path.join(os.getcwd(), 'implementation/specifications/baseline.txt'),
        help="Path to the specification file. Defaults to 'implementation/specifications/baseline.txt'.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where logs will be stored. Defaults to 'logs'."
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Name of the configuration file (without .py extension). Defaults to 'config'.",
    )

    args = parser.parse_args()

    
    # Invert the logic: dynamic scaling is True by default unless explicitly disabled
    enable_dynamic_scaling = not args.no_dynamic_scaling
    
    async def main():
        # Initialize configuration
        config = load_config(args.config_name)

        # Load the specification from the provided path or default
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

        inputs = [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]

        # Initialize the task manager
        task_manager = TaskManager(
            specification=specification,
            inputs=inputs,
            config=config,
            check_interval_sam=args.check_interval_sam, 
            log_dir=args.log_dir
        )

        # Start the main task
        task = asyncio.create_task(
            task_manager.main_task(
                enable_scaling=enable_dynamic_scaling,
            )
        )

        # Await tasks to run them
        await task

    # Top-level call to asyncio.run() to start the event loop
    asyncio.run(main())