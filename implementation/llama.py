import random
from transformers import LlamaForCausalLM, AutoTokenizer  # Change to AutoTokenizer
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

logger = logging.getLogger('main_logger')

class LLM_model:
    """Language model that predicts continuation of provided source code."""

    def __init__(self, samples_per_prompt: int, device="cuda", checkpoint="implementation/llama3-demo/llama3/Meta-Llama-3-8B-Instruct") -> None:  # Update path for Llama 3.1 Instruct 8B
        self._samples_per_prompt = samples_per_prompt
        self.device = device
        self.checkpoint = checkpoint
        
        # Define tuples for sampling
        temperature_top_p_tuples = [
            (0.94445, 0.7778), (1.1667, 0.64445), (0.944445, 0.8222),
            (1.05, 0.6) ]
        rep_penalty_max_new_tokens_tuples = [
            (1.222, 246), (1.1, 140), (1.222, 260), (1.2, 100), (1.222223, 300)
        ]

        # Sample from tuples
        temperature, top_p = random.choice(temperature_top_p_tuples)
        repetition_penalty, max_new_tokens = random.choice(rep_penalty_max_new_tokens_tuples)
        
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

        # Use AutoTokenizer for Llama 3.1
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token}) 
        
        try: 
            # Adjust dtype to use bfloat16 or float16 for inference
            self.model = LlamaForCausalLM.from_pretrained(self.checkpoint, device_map="auto", torch_dtype=torch.float32)  # Use torch.bfloat16 for efficiency
        except Exception as e: 
            logger.error(f"Could not load model because: {e}")

        self.generate_kwargs = dict(
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
        )

    def draw_batch_samples(self, prompts: list) -> list:
        """Returns multiple predicted continuations for each prompt in a list of prompts."""
        try:
            self.tokenizer.padding_side = 'left'

            # Tokenize prompts with truncation and padding
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
            input_length = inputs.input_ids.shape[1]
            batch_size = inputs.input_ids.shape[0]
            logger.debug(f"Prompts being processed by LLM is {batch_size}")

            all_samples = []

            # Generate multiple outputs for each prompt
            for _ in range(self._samples_per_prompt):
                try:
                    outputs = self.model.generate(**inputs, **self.generate_kwargs, pad_token_id=self.tokenizer.eos_token_id)
                except Exception as e:
                    logger.error(f"Could not generate prompts because {e}")
                logger.debug(f"LLM: output dims is {outputs.shape}")
                try:
                    generated_tokens = outputs[:, input_length:]
                    decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    all_samples.append(decoded_texts)
                except Exception as e:
                    logger.error(f"Could not decode text because: {e}")

            # Transpose the results to group samples by prompt
            grouped_samples = list(map(list, zip(*all_samples)))
            logger.debug(f"Grouped samples are {grouped_samples}")

            return grouped_samples

        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            return []

class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config):
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self._config = config
        self.samples_per_prompt = self._config.samples_per_prompt  
        self.samples_per_batch = self._config.programs_database.prompts_per_batch  
        self._llm = LLM_model(self.samples_per_prompt)

    async def consume_and_process(self) -> None:
        try:
            await self.channel.set_qos(prefetch_count=10)
            async with self.sampler_queue.iterator() as stream:
                batch = []
                batch_timeout = 0.4
                batch_start_time = asyncio.get_event_loop().time() 
                try:
                    async for message in stream:
                        batch.append(message)
                        current_time = asyncio.get_event_loop().time()
                        if len(batch) >= self.samples_per_batch or (current_time - batch_start_time) > batch_timeout:
                            await self.process_batch_s(batch)
                            batch = []  # Reset batch after processing
                            batch_start_time = asyncio.get_event_loop().time() 
                except asyncio.CancelledError:
                    logger.info("Sampler task was canceled.")
                    raise  
                except Exception as e:
                    logger.error(f"Exception in consume_and_process: {e}")
        except asyncio.CancelledError:
            logger.info("Consume_and_process was canceled.")
        except Exception as e:
            logger.error(f"Error setting up the channel or iterator: {e}")

    async def process_batch_s(self, batch: List[aio_pika.IncomingMessage]):
        prompts = []
        metadata = []

        for message in batch:
            try:
                async with message.process():
                    prompt = programs_database.Prompt.deserialize(message.body.decode())
                    try: 
                        logger.info(f"Prompt is {prompt}")
                        prompts.append(prompt.code)
                    except Exception as e:
                        logger.error(f"Sampler error cannot print prompt or append {e}")
                    metadata.append({
                        "island_id": prompt.island_id,
                        "version_generated": prompt.version_generated,
                        "expected_version": prompt.expected_version,
                    })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue
        try:
            samples_list = self._llm.draw_batch_samples(prompts)
        except Exception as e:
            logger.error(f"Could not prompt LLM because: {e}")
            return

        for samples, meta in zip(samples_list, metadata):
            for sample in samples:
                message_data = {
                    "sample": sample,
                    "island_id": meta["island_id"],
                    "version_generated": meta["version_generated"],
                    "expected_version": meta["expected_version"]
                }
                serialized_message = json.dumps(message_data)
                try:
                    await self.channel.default_exchange.publish(
                        aio_pika.Message(body=serialized_message.encode()),
                        routing_key='evaluator_queue'
                    )
                    logger.debug("Successfully published prompt to evaluator_queue")
                except Exception as e:
                    logger.error(f"Sampler: Exception in published prompt to evaluator_queue {e}.")
