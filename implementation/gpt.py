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

import openai
import os
import logging

logger = logging.getLogger('main_logger')

class LLM_model:
    def __init__(self, samples_per_prompt: int, model="gpt-4o-mini"):
        self.samples_per_prompt = samples_per_prompt
        self.model = model

        # Load configuration from environment variables
        openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')  # Azure OpenAI API key
        openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')  # Azure OpenAI endpoint
        openai.api_type = 'azure'
        openai.api_version = '2024-08-01-preview'  # Ensure this matches your Azure OpenAI deployment

        # Initialize counters for tracking usage
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0  # Track total cost

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost based on prompt and completion tokens.
        Rates are defined for the specific model being used.
        """
        # Rates for gpt-4o-mini (adjust if your model has different rates)
        prompt_rate = 0.150 / 1_000_000  # $0.150 per 1M input tokens
        completion_rate = 0.600 / 1_000_000  # $0.600 per 1M output tokens

        # Calculate cost for this request
        prompt_cost = prompt_tokens * prompt_rate
        completion_cost = completion_tokens * completion_rate

        # Return total cost
        return prompt_cost + completion_cost

    def draw_sample(self, prompt: str) -> list:
        """
        Generate a sample response from the LLM based on the provided prompt.
        """
        try:
            # Using the updated ChatCompletion API format with 'deployment_id'
            response = openai.ChatCompletion.create(
                deployment_id=self.model,  # Updated parameter for openai>=1.0.0
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in Python programming."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )

            # Extract usage details from the response
            usage = response['usage']
            self.total_requests += 1
            self.total_prompt_tokens += usage.get('prompt_tokens', 0)
            self.total_completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)

            # Calculate the cost for this request
            cost = self.calculate_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
            self.total_cost += cost

            # Log the response, tokens, and cost
            logger.info(f"Response: {response}")
            logger.info(f"Tokens used in this request: prompt={usage.get('prompt_tokens', 0)}, "
                        f"completion={usage.get('completion_tokens', 0)}, total={usage.get('total_tokens', 0)}")
            logger.info(f"Cost for this request: ${cost:.6f}")
            logger.info(f"Total cost so far: ${self.total_cost:.6f}")
            logger.info(f"Total requests so far: {self.total_requests}")
            logger.info(f"Total tokens used so far: prompt={self.total_prompt_tokens}, "
                        f"completion={self.total_completion_tokens}, total={self.total_tokens}")

            # Return the list of message content from the response choices
            return [choice['message']['content'] for choice in response['choices']]

        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Unexpected error during draw_sample: {str(e)}")
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
    pass