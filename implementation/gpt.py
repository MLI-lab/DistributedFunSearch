from openai import OpenAI
import asyncio
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.inference import prepare_pippy
import numpy as np
import logging
import aio_pika
import asyncio
import json
import programs_database
from typing import List
from profiling import sync_time_execution, sync_track_memory, async_track_memory, async_time_execution


import openai
import os
import logging

logger = logging.getLogger('main_logger')

class LLM_model:
    def __init__(self, samples_per_prompt: int, model="gpt-4o-mini"):
        self.samples_per_prompt = samples_per_prompt
        self.model = model
        self.client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

    def draw_sample(self, prompt: str):  # No async needed
        try:
            # Using synchronous ChatCompletion.create
            response = self.client.chat.completions.create(
                model=self.model,  # Your gpt-4o-mini model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in Python programming."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n=self.samples_per_prompt
            )
            logger.info(f"response is {response}")
            # Correctly access 'message.content' in the response
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.error(f"API error during draw_sample: {str(e)}")
            return []

class Sampler:
    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config):
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self.config = config
        self._llm = LLM_model(samples_per_prompt=self.config.samples_per_prompt)

    async def consume_and_process(self):
        async with self.sampler_queue.iterator() as stream:
            async for message in stream:
                async with message.process():
                    try:
                        prompt = programs_database.Prompt.deserialize(message.body.decode())
                        responses = self._llm.draw_sample(prompt.code)  # Removed 'await' since 'draw_sample' is synchronous
                        for response in responses:
                            message_data = {
                                "sample": response,
                                "island_id": prompt.island_id,
                                "version_generated": prompt.version_generated,
                                "expected_version": prompt.expected_version
                            }
                            serialized_message = json.dumps(message_data)
                            await self.channel.default_exchange.publish(
                                aio_pika.Message(body=serialized_message.encode()),
                                routing_key='evaluator_queue'
                            )
                            logger.debug("Successfully published prompt to evaluator_queue")
                    except Exception as e:
                        logger.error(f"Error processing and sending message: {str(e)}")

if __name__ == "__main__":
    # Configuration and connection setup must be done here
    pass
