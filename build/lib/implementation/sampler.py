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
import programs_database
from profiling import async_time_execution

logger = logging.getLogger('main_logger')


class LLM_model:
    """Language model that predicts continuation of provided source code."""
    def __init__(self, samples_per_prompt: int, temperature, top_p, repetition_penalty, max_new_tokens, 
                 device="cuda", checkpoint="bigcode/starcoder2-15b") -> None:
        self.gpu_time = 0.0
        self._samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.checkpoint = checkpoint
        self.previous_total_registered_programs = 0

        # Set cache directory and environment variable
        self.cache_dir = "/workspace/models/"
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir

        if device == "cuda" or device is None:
            self.device_map = "auto"
            self.device = device
            logger.info("Using all available GPUs with device_map='auto'.")
        else:
            self.device = device if isinstance(device, str) else f"cuda:{device}"
            self.device_map = None
            logger.info(f"Attempting to load model on device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
        except Exception as e:
            logger.error(f"Could not load tokenizer from cache because: {e}")
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                local_files_only=False,
                device_map=self.device_map
            ).to(self.device)
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
            logger.debug(f"Adjusted LLM temperature to {new_temp} based on {total_registered_programs} registered programs.")

    def draw_batch_samples(self, prompts: List[str], total_registered_programs: int = 0, temperature_period: int = 10000) -> List[List[str]]:
        if temperature_period is not None:
            try:
                self.adjust_temperature(total_registered_programs, temperature_period)
            except Exception as e:
                logger.error(f"Error adjusting temperature: {e}")
        try:
            self.tokenizer.padding_side = 'left'
            # Tokenize once for the whole batch
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
            input_length = inputs.input_ids.shape[1]
            logger.info(f"LLM: input dims {inputs.input_ids.shape}")

            all_samples = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            for _ in range(self._samples_per_prompt):
                try:
                    outputs = self.model.generate(**inputs, **self.generate_kwargs, pad_token_id=self.tokenizer.eos_token_id)
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    continue

                logger.debug(f"LLM: output dims {outputs.shape}")
                try:
                    generated_tokens = outputs[:, input_length:]
                    decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    all_samples.append(decoded_texts)
                except Exception as e:
                    logger.error(f"Decoding failed: {e}")

            end_event.record()
            torch.cuda.synchronize()
            self.gpu_time = start_event.elapsed_time(end_event) / 1000.0
            logger.debug(f"GPU sampling time: {self.gpu_time:.2f} sec")
            grouped_samples = list(map(list, zip(*all_samples)))
            return grouped_samples

        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            return []


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
            self._llm = LLM_model(self.samples_per_prompt, self._config.temperature, self._config.top_p, 
                                  self._config.repetition_penalty, self._config.max_new_tokens, self.device, 
                                  "bigcode/starcoder2-15b")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")

    async def consume_and_process(self) -> None:
        try:
            await self.channel.set_qos(prefetch_count=10)
            async with self.sampler_queue.iterator() as stream:
                batch = []
                batch_timeout = 0.01
                batch_start_time = asyncio.get_event_loop().time()
                try:
                    async for message in stream:
                        batch.append(message)
                        current_time = asyncio.get_event_loop().time()
                        if len(batch) >= self.samples_per_batch or (current_time - batch_start_time) > batch_timeout:
                            await self.process_batch_s(batch)
                            batch = []
                            batch_start_time = asyncio.get_event_loop().time()
                except asyncio.CancelledError:
                    logger.debug("Sampler task canceled.")
                    raise
                except Exception as e:
                    logger.error(f"Error in consume_and_process loop: {e}")
        except asyncio.CancelledError:
            logger.error("consume_and_process canceled.")
        except Exception as e:
            logger.error(f"Channel/iterator setup error: {e}")

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
                    flag = data.get("flag", False)
                    flags.append(flag)
                    prompt = programs_database.Prompt.deserialize(prompt_data)
                    if prompt.code is not None:
                        prompts.append(prompt.code)
                        metadata.append({
                            "island_id": prompt.island_id,
                            "version_generated": prompt.version_generated,
                            "expected_version": prompt.expected_version,
                        })
                    else:
                        logger.warning(f"Skipping prompt with island_id {prompt.island_id}: no code.")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

        if not prompts:
            logger.warning("No valid prompts in batch; skipping processing.")
            return

        try:
            samples_list = self._llm.draw_batch_samples(prompts, total_registered_programs, self.temperature_period)
            gpu_time = self._llm.gpu_time
        except Exception as e:
            logger.error(f"LLM sampling failed: {e}")
            return

        for samples, meta, flag in zip(samples_list, metadata, flags):
            if flag:
                try:
                    with open("duplicate_samples.txt", "a") as f:
                        f.write(f"Prompt Metadata:\n{meta}\n")
                        for idx, sample in enumerate(samples):
                            f.write(f"Output {idx + 1}:\n{sample}\n{'-'*50}\n")
                    logger.info("Logged duplicate prompt and outputs to 'duplicate_samples.txt'.")
                except Exception as e:
                    logger.error(f"Error logging duplicate data: {e}")

            for sample in samples:
                message_data = {
                    "sample": sample,
                    "island_id": meta["island_id"],
                    "version_generated": meta["version_generated"],
                    "expected_version": meta["expected_version"],
                    "gpu_time": gpu_time,
                }
                serialized_message = json.dumps(message_data)
                try:
                    await self.channel.default_exchange.publish(
                        aio_pika.Message(body=serialized_message.encode()),
                        routing_key='evaluator_queue'
                    )
                    logger.debug("Published sample to evaluator_queue.")
                except Exception as e:
                    logger.error(f"Error publishing sample: {e}")
