# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Asynchronous RabbitMQ Sampler.

Differences from the original DeepMind FunSearch version

* Implements placeholder to inferences LLM (StarCoder-2 15B).
* Dynamic batching based on message load. Messages are collected for up to 10 milliseconds;
  if at least 10 prompts arrive within that window we batch 10, otherwise we batch the smaller number that arrived.
* Dynamically adjusts the sampling temperature based on how many programs
  have been stored. Encourages exploration early (higher temperature) and
  shifts toward exploitation (greedy decoding) after a configurable number
  of new programs. Once this threshold is reached, temperature is reset
  and the process repeats.
* Tracks GPU runtime and token counts (input/output) for each sample.
  GPU time is measured for the entire batch and then evenly distributed across all samples
  to ensure accurate resource tracking.
* CPU/GPU fallback: if CUDA is unavailable the model logs a warning
  and runs on CPU rather than crashing.
* When a prompt is flagged as functionally identical to a previous one, all samples
  are logged to `duplicate_samples.txt` for manual inspection and debugging.
"""


import random
import os
import json
import logging
import asyncio
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import aio_pika
import numpy as np

from funsearchmq import programs_database
from funsearchmq.profiling import async_time_execution

logger = logging.getLogger('main_logger')


class LLM_model:
    """Language model that generates continuation of provided source code."""
    
    def __init__(
            self,
            samples_per_prompt: int,
            temperature,
            top_p,
            repetition_penalty,
            max_new_tokens,
            device="cuda",   # can be "cuda", None, "cpu", "cuda:0", etc.
            checkpoint="bigcode/starcoder2-15b",
    ) -> None:
        self.gpu_time = 0.0
        self._samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.checkpoint = checkpoint
        self.previous_total_registered_programs = 0

        # Set cache directory and environment variable
        try:
            self.cache_dir = "/mnt/models/"
            os.makedirs(self.cache_dir, exist_ok=True)
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        except Exception as e:
            logger.error(
                f"Warning: Could not create cache directory {self.cache_dir}, "
                f"falling back to default cache location instead. Error: {e}"
            )

        # Decide how to handle the device mapping
        if device == "cuda" or device is None:
            # Let HF handle the distribution across all GPUs automatically
            self.device_map = "auto"
            self.device = None
            logger.info("Using device_map='auto' (all available GPUs).")
        else:
            self.device_map = None

            if isinstance(device, int):
                self.device = f"cuda:{device}"
            else:
                self.device = device  # e.g. "cuda:0" or "cpu"

            if self.device == "cpu":
                logger.warning("No CUDA GPU available. Falling back to CPU.")

            logger.info(f"Using explicit device='{self.device}', device_map=None")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
        except Exception as e:
            logger.error(f"Could not load tokenizer from cache because: {e}")
            raise

        # Ensure we have a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        # Load model
        try:
            if self.device_map == "auto":
                # Let HF do the device placement
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16, 
                    local_files_only=False,
                    device_map="auto",
                )
            else:
                # Load on CPU/GPU as requested, then .to() if relevant
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16,
                    local_files_only=False,
                    device_map=None,
                )
                # Move to user-specified device
                self.model.to(self.device)

            logger.info("Successfully loaded model.")
        except Exception as e:
            logger.error(f"Could not load model from cache because: {e}")
            raise

        self.generate_kwargs = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True,
        }

    def adjust_temperature(self, total_registered_programs: int, temperature_period: int):
        if temperature_period is not None:
            effective = total_registered_programs - self.previous_total_registered_programs
            new_temp = max(0, self.temperature * (1 - effective / temperature_period))
            if new_temp > 0:
                self.generate_kwargs.update({
                    "do_sample": True,
                    "temperature": max(0.1, new_temp),
                    "top_p": self.top_p,
                })
            else:
                self.generate_kwargs["do_sample"] = False
                self.generate_kwargs.pop("temperature", None)
                self.generate_kwargs.pop("top_p", None)
            self.previous_total_registered_programs = total_registered_programs
            logger.debug(
                f"Adjusted LLM temperature to {new_temp} "
                f"based on {total_registered_programs} registered programs."
            )

    def draw_batch_samples(
            self,
            prompts: List[str],
            total_registered_programs: int = 0,
            temperature_period: int = 10000
    ) -> List[List[str]]:
        if temperature_period is not None:
            try:
                self.adjust_temperature(total_registered_programs, temperature_period)
            except Exception as e:
                logger.error(f"Error adjusting temperature: {e}")

        try:
            self.tokenizer.padding_side = 'left'
            # Tokenize once for the whole batch (on CPU by default)
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False
            )

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            input_token_counts = [len(ids) for ids in inputs["input_ids"]]


            input_length = inputs["input_ids"].shape[1]
            logger.info(f"LLM: input dims {inputs['input_ids'].shape}")

            all_samples = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            all_output_token_counts = []

            for _ in range(self._samples_per_prompt):
                try:
                    outputs = self.model.generate(
                        **inputs,
                        **self.generate_kwargs,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    continue

                logger.debug(f"LLM: output dims {outputs.shape}")
                try:
                    generated_tokens = outputs[:, input_length:]
                    decoded_texts = self.tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )
                    all_samples.append(decoded_texts)
                except Exception as e:
                    logger.error(f"Decoding failed: {e}")

                all_output_token_counts.append([t.numel() for t in generated_tokens])        

            end_event.record()
            torch.cuda.synchronize()
            self.gpu_time = start_event.elapsed_time(end_event) / 1000.0
            logger.debug(f"GPU sampling time: {self.gpu_time:.2f} sec")
            # Transpose so outer index = prompt
            output_token_counts = list(map(list, zip(*all_output_token_counts)))

            # Group the samples so that for each prompt we have a list of generated completions
            grouped_samples = list(map(list, zip(*all_samples)))
            return grouped_samples, input_token_counts, output_token_counts

        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            return [], [], []

    def cleanup(self):
        """Release GPU memory and clean up model resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("LLM_model: GPU memory cleaned up")
        except Exception as e:
            logger.error(f"LLM_model: Error during cleanup: {e}")


class Sampler:
    """Node that samples program continuations and sends them for analysis."""
    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config, device):
        self.device = device
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self._config = config
        self.temperature_period = self._config.temperature_period
        self.samples_per_prompt = self._config.samples_per_prompt
        self.samples_per_batch = self._config.prompts_per_batch

        try:
            self._llm = LLM_model(
                samples_per_prompt=self.samples_per_prompt,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                repetition_penalty=self._config.repetition_penalty,
                max_new_tokens=self._config.max_new_tokens,
                device=self.device,   # Could be "cuda", None, "cpu", or "cuda:0"
                checkpoint="bigcode/starcoder2-15b"
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Optionally raise

    async def consume_and_process(self) -> None:
        from funsearchmq import process_utils

        async def _consume_loop():
            """Inner consume loop - will be wrapped with reconnection logic."""
            logger.info(f"Sampler on device {self.device}: Setting QoS prefetch_count=10...")
            await self.channel.set_qos(prefetch_count=10)
            logger.info(f"Sampler on device {self.device}: Starting to consume messages from sampler_queue...")

            async with self.sampler_queue.iterator() as stream:
                logger.info(f"Sampler on device {self.device}: Successfully registered as consumer, now listening for messages...")
                batch = []
                batch_timeout = 0.01
                batch_start_time = asyncio.get_event_loop().time()

                async for message in stream:
                    batch.append(message)
                    current_time = asyncio.get_event_loop().time()
                    # If we hit batch size or time threshold, process the batch
                    if (len(batch) >= self.samples_per_batch or (current_time - batch_start_time) > batch_timeout):
                        await self.process_batch_s(batch)
                        batch = []
                        batch_start_time = asyncio.get_event_loop().time()

        # Wrap consume loop with automatic reconnection
        await process_utils.with_reconnection(
            _consume_loop,
            logger,
            component_name=f"Sampler on device {self.device}"
        )

    async def process_batch_s(self, batch: List[aio_pika.IncomingMessage]):
        prompts = []
        metadata = []
        flags = []
        for message in batch:
            try:
                async with message.process():
                    data = json.loads(message.body.decode())
                    prompt_data = data["prompt"]
                    total_registered_programs = data.get("total_registered_programs", 0)
                    flag = data.get("flag", False) # sampler gets from database a flag if prompt has few shot examples thar are functionally identically
                    flags.append(flag)
                    prompt = programs_database.Prompt.deserialize(prompt_data)

                    if prompt.code is not None:
                        prompts.append(prompt.code)
                        metadata.append({
                            "island_id": prompt.island_id,
                            "version_generated": prompt.version_generated,
                            "expected_version": prompt.expected_version,
                            "parent_ids": data.get("parent_ids", []),  # Extract parent IDs for lineage tracking
                        })
                    else:
                        logger.warning(f"Skipping prompt with island_id {prompt.island_id}: Prompt is empty.")
                        
            except Exception as e:
                logger.error(f"Sampler: Error processing message: {e}")
                total_registered_programs = 0  
                continue

        if not prompts:
            logger.warning("No valid prompts in batch; skipping processing.")
            return

        # Get the completions from the LLM
        try:
            samples_list, input_token_counts, output_token_counts = self._llm.draw_batch_samples(prompts, total_registered_programs, self.temperature_period)
            gpu_time = self._llm.gpu_time
        except Exception as e:
            logger.error(f"LLM sampling failed: {e}")
            return

        # Calculate total samples generated in this batch to properly distribute GPU time
        total_samples = sum(len(samples) for samples in samples_list)
        gpu_time_per_sample = gpu_time / total_samples if total_samples > 0 else 0.0
        logger.debug(f"Batch GPU time: {gpu_time:.2f}s for {total_samples} samples = {gpu_time_per_sample:.4f}s per sample")

        # Publish results to the evaluator queue
        for prompt_idx, (samples, meta, flag) in enumerate(zip(samples_list,metadata, flags)):
            # log duplicated-prompt runs for manual inspection of output+prompt
            if flag:
                try:
                    with open("duplicate_samples.txt", "a") as f:
                        f.write(f"Prompt Metadata:\n{meta}\n")
                        for idx, sample in enumerate(samples):
                            f.write(f"Output {idx + 1}:\n{sample}\n{'-'*50}\n")
                    logger.info("Logged duplicate prompt and outputs to "
                                "'duplicate_samples.txt'.")
                except Exception as e:
                    logger.error(f"Error logging duplicate data: {e}")

            # Send every sample to the evaluator queue
            for sample_idx, sample in enumerate(samples):
                message_data = {
                    "sample":             sample,
                    "island_id":          meta["island_id"],
                    "version_generated":  meta["version_generated"],
                    "expected_version":   meta["expected_version"],
                    "gpu_time":           gpu_time_per_sample,
                    "input_tokens":       input_token_counts[prompt_idx],
                    "output_tokens":      output_token_counts[prompt_idx][sample_idx],
                    "parent_ids":         meta.get("parent_ids", []),  # Pass parent IDs for lineage tracking
                }
                serialized_message = json.dumps(message_data)

                try:
                    await self.channel.default_exchange.publish(
                        aio_pika.Message(body=serialized_message.encode()),
                        routing_key="evaluator_queue",
                    )
                    logger.debug("Published sample to evaluator_queue.")
                except Exception as e:
                    logger.error(f"Error publishing sample: {e}")

    def cleanup(self):
        """Release LLM resources and GPU memory."""
        import gc
        try:
            if hasattr(self, '_llm'):
                self._llm.cleanup()
                del self._llm
            gc.collect()
            logger.info("Sampler: Cleanup completed")
        except Exception as e:
            logger.error(f"Sampler: Error during cleanup: {e}")
