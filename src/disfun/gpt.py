import asyncio
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.inference import prepare_pippy
import numpy as np
import logging
import aio_pika
import asyncio
import json
from disfun import programs_database
from typing import List
from disfun.profiling import sync_time_execution, sync_track_memory, async_track_memory, async_time_execution
from openai import AzureOpenAI
import os
import logging

logger = logging.getLogger('main_logger')


class LLM_model:
    def __init__(self, samples_per_prompt: int, model="gpt-4o-mini"):
        self.samples_per_prompt = samples_per_prompt
        self.model = model
        logger.debug("In LLM")

        # Initialize the Azure OpenAI client
        try: 
            self.client = AzureOpenAI(
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2023-07-01-preview'),  # Default API version if not set
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),  # Azure OpenAI endpoint, e.g., "https://<resource-name>.openai.azure.com/"
                api_key=os.getenv('AZURE_OPENAI_API_KEY')  # Azure OpenAI API key
            )
        except Exception as e: 
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")

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
            # Using the updated client for Azure OpenAI with `model`
            response = self.client.chat.completions.create(
                model=self.model,  
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in Python programming."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n= self.samples_per_prompt
            )

            # Log the entire response
            logger.debug(f"Full response: {response}")

            # Extract usage details from the response
            usage = response.usage # contains information on token usage for the completion request to compute cost
            self.total_requests += 1
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens

            # Calculate the cost for this request
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            self.total_cost += cost

            # Log the response, tokens, and cost
            logger.debug(f"Tokens used in this request: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
            logger.debug(f"Cost for this request: ${cost:.6f}")
            logger.debug(f"Total cost so far: ${self.total_cost:.6f}")
            logger.debug(f"Total requests so far: {self.total_requests}")
            logger.debug(f"Total tokens used so far: prompt={self.total_prompt_tokens}, completion={self.total_completion_tokens}, total={self.total_tokens}")

            # Retrieve the generated text from the response
            generated_responses = [choice.message.content for choice in response.choices]
            logger.debug(f"Generated response from gpt mini is {generated_responses}")
        
            # Return the list of message content (the generated text)
            return generated_responses

        except Exception as e:
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
        self.prefetch_count = 10

    async def consume_and_process(self):
        from disfun import process_utils

        async def _consume_loop():
            """Inner consume loop - will be wrapped with reconnection logic."""
            await self.channel.set_qos(prefetch_count=self.prefetch_count)

            async with self.sampler_queue.iterator() as stream:
                async for message in stream:
                    async with message.process():
                        try:
                            gpu_time = 0
                            data = json.loads(message.body.decode())
                            prompt_data = data["prompt"]
                            prompt = programs_database.Prompt.deserialize(prompt_data)
                            total_registered_programs = data.get("total_registered_programs", 0)
                            parent_ids = data.get("parent_ids", [])
                            responses = self._llm.draw_sample(prompt.code)
                            logger.debug(f"responses is {responses}")

                            for response in responses:
                                message_data = {
                                    "sample": response,
                                    "island_id": prompt.island_id,
                                    "version_generated": prompt.version_generated,
                                    "expected_version": prompt.expected_version,
                                    "gpu_time": gpu_time,
                                    "parent_ids": parent_ids,
                                }
                                serialized_message = json.dumps(message_data)
                                await self.channel.default_exchange.publish(
                                    aio_pika.Message(body=serialized_message.encode()),
                                    routing_key='evaluator_queue'
                                )
                                logger.debug("Successfully published prompt to evaluator_queue")
                        except Exception as e:
                            logger.error(f"Error processing and sending message: {str(e)}")

        # Wrap consume loop with automatic reconnection
        await process_utils.with_reconnection(
            _consume_loop,
            logger,
            component_name="GPT Sampler"
        )

if __name__ == "__main__":
    pass
