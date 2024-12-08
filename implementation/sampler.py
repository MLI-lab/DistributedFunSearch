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
import os
from profiling import sync_time_execution


logger = logging.getLogger('main_logger')

class LLM_model:
    """Language model that predicts continuation of provided source code."""
    def __init__(self, samples_per_prompt: int, temperature, top_p, repetition_penalty, max_new_tokens, device="cuda", checkpoint="bigcode/starcoder2-15b") -> None:
        self._samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.checkpoint = checkpoint
        # Set the cache directory
        self.cache_dir = "/workspace/models/"
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        if device == "cuda" or device is None:
            self.device_map = "auto"
            self.device = device
            logger.info(f"Using all available GPUs with device_map='auto'.")
        else:
            # Use the specified single GPU
            self.device = device if isinstance(device, str) else f"cuda:{device}"
            self.device_map = None
            logger.info(f"Attempting to load model on device: {self.device}")

        try:
            # Load the tokenizer with local_files_only=True
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
        except Exception as e:
            logger.error(f"Could not load tokenizer from cache because: {e}")
            raise

        # Add padding token to process prompts in batches
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        try:
            # Load the model with local_files_only=True
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                local_files_only=False,
                device_map=self.device_map
            ).to(self.device)
            logger.info(f"Sucessfully loaded model.")
        except Exception as e:
            logger.error(f"Could not load model from cache because: {e}")
            raise

        self.generate_kwargs = dict(
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
        )
    #@sync_track_memory
    @sync_time_execution
    def draw_batch_samples(self, prompts: list) -> list:
        """Returns multiple predicted continuations for each prompt in a list of prompts."""
        try:
            self.tokenizer.padding_side = 'left'
            #max_total_length = 512  # Define max length based on model capacity
        
            # Store original lengths of each prompt without truncation
            original_lengths = [len(self.tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]

            # Tokenize prompts with truncation and padding
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)#, max_length=max_total_length).to(self.device)
            input_length = inputs.input_ids.shape[1]
            logger.debug(f"LLM: input dims is {inputs.input_ids.shape}")
            batch_size = inputs.input_ids.shape[0]
            logger.info(f"Prompts being processed by LLM is {batch_size}")

            # Check if any prompt was truncated
            #for i, prompt in enumerate(prompts):
            #    if original_lengths[i] > max_total_length:
            #        logger.info(f"Truncated prompt: {prompt[:50]}... (original length: {original_lengths[i]} tokens, truncated to {max_total_length} tokens)")

            all_samples = []

            # Generate multiple outputs for each prompt by passing the batch _samples_per_prompt times
            for _ in range(self._samples_per_prompt):
                try:
                    outputs = self.model.generate(**inputs, **self.generate_kwargs, pad_token_id=self.tokenizer.eos_token_id)  # [batch_size, generated_length]
                except Exception as e:
                    logger.error(f"Could not generate prompts because {e}")
                    continue  # Skip to the next iteration if generation fails

                logger.debug(f"LLM: output dims is {outputs.shape}")
                try:
                    generated_tokens = outputs[:, input_length:]  # Extract only the new tokens
                    decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    all_samples.append(decoded_texts)  # list of lists [samples_per_prompt, batch_size]
                except Exception as e:
                    logger.error(f"Could not decode text because: {e}")

            # Transpose the results to group samples by prompt
            grouped_samples = list(map(list, zip(*all_samples)))

            return grouped_samples

        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            return []


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config, device):
        print(f"In Sampler with device {device}")
        self.device=device
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self._config = config
        self.samples_per_prompt = self._config.samples_per_prompt  
        self.samples_per_batch = self._config.programs_database.prompts_per_batch  # Access nested config for batch size
        self._llm = LLM_model(self.samples_per_prompt, self._config.temperature, self._config.top_p, self._config.repetition_penalty, self._config.max_new_tokens, self.device)

    async def consume_and_process(self) -> None:
        try:
            await self.channel.set_qos(prefetch_count=10)
            async with self.sampler_queue.iterator() as stream:
                batch = []
                batch_timeout=0.01
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
                    logger.debug("Sampler task was canceled.")
                    raise  
                except Exception as e:
                    logger.error(f"Exception in consume_and_process: {e}")
        except asyncio.CancelledError:
            logger.error("Consume_and_process was canceled.")
        except Exception as e:
            logger.error(f"Error setting up the channel or iterator: {e}")

    #@async_time_execution
    #@async_track_memory
    async def process_batch_s(self, batch: List[aio_pika.IncomingMessage]):
        prompts = []
        metadata = []

        # Collect prompts and metadata from each message
        for message in batch:
            try:
                async with message.process():
                    prompt = programs_database.Prompt.deserialize(message.body.decode())
                    try: 
                        if prompt.code is not None:  # Only append if the code is not None
                            prompts.append(prompt.code)
                            metadata.append({
                                "island_id": prompt.island_id,
                                "version_generated": prompt.version_generated,
                                "expected_version": prompt.expected_version,
                            })
                        else:
                            logger.warning(f"Prompt with island_id {prompt.island_id} has no code and will be skipped.")
                    except Exception as e:
                        logger.error(f"Sampler error cannot print prompt or append {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

        # Check if there are any valid prompts left after filtering
        if not prompts:
            logger.warning("No valid prompts found in batch. Skipping batch processing.")
            return

        try:
            # Generate samples for the valid prompts
            samples_list = self._llm.draw_batch_samples(prompts)
        except Exception as e:
            logger.error(f"Could not prompt LLM because: {e}")
            return

        # Publish results
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
                    logger.error(f"Sampler: Exception in publishing prompt to evaluator_queue {e}.")
