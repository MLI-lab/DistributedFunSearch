import asyncio
import argparse
import torch.multiprocessing as mp
import os
import sys
from disfun.scaling_utils import ResourceManager
from disfun import sampler, process_utils
from disfun.process_entry import sampler_process_entry

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TaskManager:
    def __init__(self, config, check_interval, log_dir, config_path):
        self.config = config
        self.config_path = config_path  # Store for spawn compatibility
        self.log_dir = log_dir  # Store for spawn compatibility

        # Initialize logger and store filename for child processes
        pid = os.getpid()
        self.log_filename = f'attach_samplers_pid{pid}.log'
        self.logger = process_utils.initialize_logger(log_dir, self.log_filename)

        self.sampler_processes = []
        self.tasks = []
        self.resource_manager = ResourceManager(log_dir=log_dir, scaling_config=self.config.scaling)
        self.process_to_device_map = {}

    async def main_task(self, enable_scaling=True):
        """
        Main async entry point. Establishes queue connections - starts initial processes,
        and optionally starts a scaling loop from ResourceManager.
        """
        resource_logging_task = asyncio.create_task(self.resource_manager.log_resource_stats_periodically(interval=60))
        self.tasks = [resource_logging_task]

        pid = os.getpid()
        self.logger.info(f"main_task is running in process PID: {pid}")

        try:
            # Start initial sampler processes
            self.start_initial_processes()

            # Create a connection/channel using utility function
            self.logger.info("Creating connection for scaling logic...")
            connection = await process_utils.create_rabbitmq_connection(
                self.config, timeout=300
            )
            channel = await connection.channel()

            # Declare the sampler queue (the queue we want to scale on)
            sampler_queue = await process_utils.declare_standard_queue(channel, "sampler_queue")
            self.logger.info("sampler_queue declared for scaling logic.")

            if enable_scaling:
                scaling_task = asyncio.create_task(
                    self.resource_manager.run_scaling_loop(
                        evaluator_queue=None,
                        sampler_queue=sampler_queue,
                        evaluator_processes=None,
                        sampler_processes=self.sampler_processes,
                        sampler_entry_function=sampler_process_entry,
                        evaluator_entry_function=None,
                        config_path=self.config_path,
                        log_dir=self.log_dir,
                        template=None,
                        inputs=None,
                        target_signatures=None,
                        sandbox_base_path=None,
                        max_evaluators=None,
                        max_samplers=args.max_samplers,
                        check_interval=self.config.scaling.check_interval if hasattr(self.config, 'scaling') and self.config.scaling else args.check_interval,
                        log_filename=self.log_filename,
                    )
                )
                self.tasks.append(scaling_task)

            # Wait on all tasks
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Exception in main_task: {e}")

    def start_initial_processes(self):
        from disfun.sampler import is_local_model

        # Check if using local model (needs GPU) or API model (CPU only)
        use_local = is_local_model(self.config.sampler.model)

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
                self.logger.info(f"Assigning sampler {i} to GPU {device} (host GPU: {host_gpu})")

                try:
                    proc = ctx.Process(
                        target=sampler_process_entry,
                        args=(self.config_path, device, self.log_dir, self.log_filename),
                        name=f"Sampler-{i}"
                    )
                    proc.start()
                    self.logger.info(f"Started Sampler Process {i} with PID: {proc.pid} on GPU {device}")
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device
                except Exception as e:
                    self.logger.error(f"Failed to start sampler {i}: {e}")
                    continue
        else:
            # API MODEL: Samplers are CPU processes
            self.logger.info(f"Starting {self.config.num_samplers} sampler(s) with API model: {self.config.sampler.model}")
            self.logger.info("Samplers will use API (no GPU assignment needed)")

            ctx = mp.get_context('spawn')
            for i in range(self.config.num_samplers):
                device = None
                try:
                    proc = ctx.Process(
                        target=sampler_process_entry,
                        args=(self.config_path, device, self.log_dir, self.log_filename),
                        name=f"Sampler-{i}"
                    )
                    proc.start()
                    self.logger.info(f"Started Sampler Process {i} with PID: {proc.pid}")
                    self.sampler_processes.append(proc)
                    self.process_to_device_map[proc.pid] = device
                except Exception as e:
                    self.logger.error(f"Error starting sampler {i}: {e}")
                    continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TaskManager with configurable scaling interval.")

######################################### General setting related arguments #######################################
    parser.add_argument(
        "--check_interval",
        type=int,
        default=200,
        help="Time interval (in seconds) between consecutive scaling checks for evaluators and samplers. Defaults to 200s."
        )

    parser.add_argument(
        "--no-dynamic-scaling",
        action="store_true",
        help="Disable dynamic scaling (enabled by default)."
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory where logs will be stored (default: logs)."
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.join(os.getcwd(), "config.py"),  # Set default to 'config.py' in the current directory
        help="Path to the configuration file (Python script containing the experiment config). Defaults to './config.py'.",
    )

########################################## Resources related arguments #############################################

    parser.add_argument(
        "--max_samplers",
        type=int,
        default=1000,
        help="Maximum samplers the system can scale up to. Adjust based on resource availability. Default no hard limit and based on dynamic resource checks."
    )

    args = parser.parse_args()

    # By default - scaling is enabled unless --no-dynamic-scaling is passed
    enable_dynamic_scaling = not args.no_dynamic_scaling

    async def main():
        config = process_utils.load_config(args.config_path)
        # Prefer config.scaling.check_interval over CLI argument
        check_interval = config.scaling.check_interval if hasattr(config, 'scaling') and config.scaling else args.check_interval
        task_manager = TaskManager(
            config=config,
            check_interval=check_interval,
            log_dir=args.log_dir,
            config_path=args.config_path
        )
        task = asyncio.create_task(
            task_manager.main_task(enable_scaling=enable_dynamic_scaling)
        )
        await task

    asyncio.run(main())