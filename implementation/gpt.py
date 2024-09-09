import openai
import asyncio
import aio_pika
from programs_database import Prompt
import json
import logging
from typing import List

logger = logging.getLogger('main_logger')

import openai

class LLM_model:
    """Language model that predicts continuation of provided source code using GPT-4o Mini."""

    def __init__(self, samples_per_prompt: int, api_key: str, model="gpt-4o-mini") -> None:
        self.samples_per_prompt = samples_per_prompt
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key

    async def draw_sample(self, prompt: str) -> list:
        """Returns multiple predicted continuations for a given prompt using GPT-4o Mini."""
        try:
            response = await openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in Python programming."},
                    {"role": "user", "content": prompt}
                ],
                n=self.samples_per_prompt  # Generate specified number of samples per prompt
            )
            logger.info(f"Repsonse looks like {response}")
            return [choice['content'] for choice in response.choices]
        except Exception as e:
            logger.error(f"API error during draw_sample: {str(e)}")
            return []

class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config):
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self.config=config
        self._llm = LLM_model(samples_per_prompt= self.config.samples_per_prompt, api_key=config.api_key)
        

    async def consume_and_process(self) -> None:
        async with self.sampler_queue.iterator() as stream:
            async for message in stream:
                async with message.process():
                    try:
                        prompt = Prompt.deserialize(message.body.decode())
                        responses = await self._llm.draw_sample(prompt.code)
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
