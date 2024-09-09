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
        # Initialize counters for tracking usage
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0  # Track total cost

    def calculate_cost(self, prompt_tokens, completion_tokens):
        # Rates for gpt-4o-mini
        prompt_rate = 0.150 / 1_000_000  # $0.150 per 1M input tokens
        completion_rate = 0.600 / 1_000_000  # $0.600 per 1M output tokens

        # Calculate cost for this request
        prompt_cost = prompt_tokens * prompt_rate
        completion_cost = completion_tokens * completion_rate

        # Return total cost
        return prompt_cost + completion_cost

    def draw_sample(self, prompt: str):  # No async needed
        try:
            # Using synchronous ChatCompletion.create
            response = self.client.chat.completions.create(
                model=self.model,  # Your gpt-4o-mini model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in Python programming. Only complete the provided code, without any explanations, comments, or additional text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n=self.samples_per_prompt
            )

            # Extract usage details from the response as attributes
            usage = response.usage
            self.total_requests += 1
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens

            # Calculate the cost for this request
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            self.total_cost += cost

            # Log the response, tokens, and cost
            logger.info(f"Response: {response}")
            logger.info(f"Tokens used in this request: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            logger.info(f"Cost for this request: ${cost:.6f}")
            logger.info(f"Total cost so far: ${self.total_cost:.6f}")
            logger.info(f"Total requests so far: {self.total_requests}")
            logger.info(f"Total tokens used so far: prompt={self.total_prompt_tokens}, completion={self.total_completion_tokens}, total={self.total_tokens}")

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
