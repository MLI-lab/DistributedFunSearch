import asyncio
import logging
from logging.handlers import RotatingFileHandler
import json
import aio_pika
from yarl import URL
from collections.abc import Sequence
from typing import Any
import threading
import time
import os
import signal
import sys
import pickle
import config as config_lib
import programs_database
import sampler2
import code_manipulation

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TaskManager:
    def __init__(self, specification: str, inputs: Sequence[Any], config: config_lib.Config):
        self.specification = specification
        self.inputs = inputs
        self.config = config
        self.logger = self.initialize_logger()
        self.tasks = []
        self.channels = []
        self.queues = []
        self.connection = None

    def initialize_logger(self):
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.INFO)
        log_file_path = os.path.join(os.getcwd(), 'task_sampler.log')
        handler = RotatingFileHandler(log_file_path, maxBytes=100 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    async def main_task(self):
        amqp_url = URL('amqp://guest:guest@localhost:5673/').update_query(heartbeat=2800)
        try:
            # Create connections for the sampler
            sampler_connection = await aio_pika.connect_robust(
                amqp_url,
                timeout=300,
                client_properties={"connection_attempts": 3, "retry_delay": 5},
                reconnect_interval=5
            )   

            # Channel on the connection
            self.sampler_channel = await sampler_connection.channel()

            sampler_queue = await self.sampler_channel.declare_queue("sampler_queue", durable=False, auto_delete=True)

        
            # Initialize the sampler instance and run it
            sampler_instance = sampler2.Sampler(sampler_connection, self.sampler_channel, sampler_queue, None, self.config)
            sampler_task = asyncio.create_task(sampler_instance.consume_and_process())

            self.tasks = [sampler_task]
            self.channels = [self.sampler_channel]

            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.info(f"Error in main_task: {e}")

    async def run(self):
        try:
            await self.main_task()
        except Exception as e:
            self.logger.info(f"Main operation error occured: {e}.")

if __name__ == "__main__":

    config = config_lib.Config()

    # Initialize the task manager
    task_manager = TaskManager(specification, [(6,1), (7,1), (8,1), (9,1), (10,1), (11,1)], config)

    # Run the main task
    asyncio.run(task_manager.run())