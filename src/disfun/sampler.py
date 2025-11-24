# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License - Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing - software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND - either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unified LiteLLM-based Sampler with vLLM support.

This sampler replaces both the original sampler.py (HuggingFace transformers)
and gpt.py (Azure OpenAI) with a single unified implementation using LiteLLM.

Key features:
* Supports 100+ LLM providers through LiteLLM (OpenAI - Anthropic - Together AI - local vLLM - etc.)
* Dynamic batching based on message load (10ms window - up to 10 prompts per batch)
* Dynamic temperature adjustment based on stored program count
* Token tracking for both input and output
* Time tracking (GPU time for local models - API latency for cloud models)
* vLLM backend support for 10-20x faster local inference vs HuggingFace transformers
"""

import random
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import time

import litellm
import aio_pika
from dotenv import load_dotenv

from disfun import programs_database
from disfun.profiling import async_time_execution

logger = logging.getLogger('main_logger')

# Load environment variables from .env file
load_dotenv()

# Setup dedicated cost logger
cost_logger = logging.getLogger('cost_logger')
cost_logger.setLevel(logging.INFO)

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Try to import vLLM for local models
try:
    from vllm import LLM as vLLM_Engine, SamplingParams
    import torch
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. Only API-based models will work.")


def is_local_model(model: str) -> bool:
    """Determine if model should use local vLLM vs API."""
    # API models from various providers
    api_prefixes = [
        "gpt-", "claude-", "anthropic/", "together_ai/", "replicate/",
        "openai/", "azure/", "openrouter/", "huggingface/"
    ]
    return not any(model.startswith(prefix) for prefix in api_prefixes)


class LLM_model:
    """Unified language model interface.

    Supports two modes:
    1. Local models: vLLM Python API (each sampler loads model on assigned GPU)
    2. API models: LiteLLM client (GPT - Claude - etc. via HTTP)
    """

    def __init__(
            self,
            samples_per_prompt: int,
            temperature: float,
            top_p: float,
            repetition_penalty: float,
            max_new_tokens: int,
            model: str,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            device: Optional[int] = None,  # GPU device for local models
            reasoning_effort: Optional[str] = None,  # For GPT-5/o3 models
            system_message: Optional[str] = None,  # System message for API models
            max_retries: int = 3,  # Maximum retry attempts for API calls
            random_seed: Optional[int] = None  # Random seed for reproducible generation
    ) -> None:
        self.inference_time = 0.0
        self._samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.device = device
        self.reasoning_effort = reasoning_effort
        self.system_message = system_message
        self.max_retries = max_retries
        self.random_seed = random_seed
        self.previous_total_registered_programs = 0

        # Cost tracking for API models
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        # Logging flag for first prompt/output
        self._logged_first_prompt = False

        # Determine if using local vLLM or API
        self.use_local_vllm = is_local_model(model)

        if self.use_local_vllm:
            # LOCAL MODE: Use vLLM Python API
            if not VLLM_AVAILABLE:
                raise RuntimeError(
                    f"Model '{model}' appears to be a local model - but vLLM is not installed. "
                    f"Install with: pip install vllm"
                )

            logger.info(f"Initializing LOCAL vLLM model: {model} on device {device}")

            # Set CUDA device if specified
            if device is not None:
                # Extract numeric device ID (handle both "cuda:0" and 0 formats)
                if isinstance(device, str):
                    device_id = device.split(":")[-1] if ":" in device else device
                else:
                    device_id = str(device)

                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
                logger.info(f"Set CUDA_VISIBLE_DEVICES={device_id}")
            else:
                logger.info("Using auto device assignment for vLLM")

            # Initialize vLLM engine (loads model on GPU)
            try:
                self.vllm_engine = vLLM_Engine(
                    model=model,
                    tensor_parallel_size=1,  # Single GPU per sampler
                    dtype="float16",
                    gpu_memory_utilization=0.90,
                    trust_remote_code=True,
                )
                logger.info(f"vLLM engine initialized successfully on GPU {device}")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM engine: {e}")
                raise

            # vLLM sampling params
            sampling_params_kwargs = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_new_tokens,
                "repetition_penalty": self.repetition_penalty,
                "n": self._samples_per_prompt,
            }

            # Add random seed for reproducible generation
            if self.random_seed is not None:
                sampling_params_kwargs["seed"] = self.random_seed

            self.sampling_params = SamplingParams(**sampling_params_kwargs)

        else:
            # API MODE: Use LiteLLM
            logger.info(f"Initializing API model via LiteLLM: {model}")

            if api_base:
                litellm.api_base = api_base
            if api_key:
                litellm.api_key = api_key

            # LiteLLM generation kwargs
            self.generate_kwargs = {
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
                "top_p": self.top_p,
                "frequency_penalty": max(0, self.repetition_penalty - 1.0),
                "n": self._samples_per_prompt,
            }

            # Add random seed for reproducible generation
            if self.random_seed is not None:
                self.generate_kwargs["seed"] = self.random_seed

            # Add reasoning_effort only for models that support it (o1 - o3 - o3-mini - gpt-5)
            if self.reasoning_effort:
                model_lower = self.model.lower()
                supports_reasoning = any(x in model_lower for x in ['o1', 'o3', 'gpt-5'])
                # Exclude o1-mini which doesn't support reasoning_effort
                if 'o1-mini' in model_lower:
                    supports_reasoning = False

                if supports_reasoning:
                    self.generate_kwargs["reasoning_effort"] = self.reasoning_effort
                    logger.info(f"Enabled reasoning_effort={self.reasoning_effort} for {self.model}")
                else:
                    logger.warning(f"Model {self.model} does not support reasoning_effort parameter, skipping this setting")

            # Log retry configuration for API models
            logger.info(f"API retry configuration: max_retries={self.max_retries} (exponential backoff)")

        logger.info(f"Model initialized: mode={'LOCAL_VLLM' if self.use_local_vllm else 'API'} - "
                   f"temp={temperature}, top_p={top_p}, max_tokens={max_new_tokens}")

    def adjust_temperature(self, total_registered_programs: int, temperature_period: int):
        """Dynamically adjust temperature based on stored program count."""
        if temperature_period is not None:
            effective = total_registered_programs - self.previous_total_registered_programs
            new_temp = max(0, self.temperature * (1 - effective / temperature_period))

            if self.use_local_vllm:
                # Update vLLM sampling params
                if new_temp > 0:
                    self.sampling_params.temperature = max(0.1, new_temp)
                    self.sampling_params.top_p = self.top_p
                else:
                    # Greedy decoding
                    self.sampling_params.temperature = 0.0
                    self.sampling_params.top_p = 1.0
            else:
                # Update LiteLLM kwargs
                if new_temp > 0:
                    self.generate_kwargs["temperature"] = max(0.1, new_temp)
                    self.generate_kwargs["top_p"] = self.top_p
                else:
                    self.generate_kwargs["temperature"] = 0.0
                    self.generate_kwargs.pop("top_p", None)

            self.previous_total_registered_programs = total_registered_programs
            logger.debug(
                f"Adjusted temperature to {new_temp} "
                f"based on {total_registered_programs} registered programs."
            )

    async def draw_batch_samples(
            self,
            prompts: List[str],
            total_registered_programs: int = 0,
            temperature_period: int = 10000
    ) -> tuple[List[List[str]], List[int], List[List[int]]]:
        """Generate samples for a batch of prompts.

        For API models - all prompts in the batch are processed in parallel using
        asyncio.gather - significantly reducing total latency compared to sequential calls.

        Args:
            prompts: List of prompt strings
            total_registered_programs: Current count of stored programs
            temperature_period: Period for temperature decay

        Returns:
            Tuple of (grouped_samples, input_token_counts, output_token_counts)
        """
        if temperature_period is not None:
            try:
                self.adjust_temperature(total_registered_programs, temperature_period)
            except Exception as e:
                logger.error(f"Error adjusting temperature: {e}")

        if self.use_local_vllm:
            return self._draw_batch_vllm(prompts)
        else:
            return await self._draw_batch_api(prompts)

    def _draw_batch_vllm(self, prompts: List[str]) -> tuple[List[List[str]], List[int], List[List[int]]]:
        """Generate samples using local vLLM engine."""
        try:
            start_time = time.time()

            logger.info(f"vLLM: Processing batch of {len(prompts)} prompts")

            # Log first prompt ever at INFO level - subsequent at DEBUG
            if prompts:
                if not self._logged_first_prompt:
                    logger.info(f"vLLM: First prompt:\n{'='*80}\n{prompts[0]}\n{'='*80}")
                else:
                    logger.debug(f"vLLM: First prompt in batch:\n{'='*80}\n{prompts[0]}\n{'='*80}")

            # vLLM batch generation
            outputs = self.vllm_engine.generate(prompts, self.sampling_params)

            all_samples = []
            input_token_counts = []
            all_output_token_counts = []

            for idx, output in enumerate(outputs):
                # Extract samples for this prompt
                samples = [o.text for o in output.outputs]
                all_samples.append(samples)

                # Log first output ever at INFO level - subsequent at DEBUG
                if idx == 0 and samples:
                    if not self._logged_first_prompt:
                        logger.info(f"vLLM: First model output:\n{'='*80}\n{samples[0]}\n{'='*80}")
                        self._logged_first_prompt = True  # Set flag after logging first prompt and output
                    else:
                        logger.debug(f"vLLM: First model output:\n{'='*80}\n{samples[0]}\n{'='*80}")

                # Token counts
                input_tokens = len(output.prompt_token_ids)
                output_tokens = [len(o.token_ids) for o in output.outputs]

                input_token_counts.append(input_tokens)
                all_output_token_counts.append(output_tokens)

            end_time = time.time()
            self.inference_time = end_time - start_time
            logger.debug(f"vLLM inference time: {self.inference_time:.2f} sec")

            return all_samples, input_token_counts, all_output_token_counts

        except Exception as e:
            logger.error(f"Error during vLLM batch generation: {e}")
            return [], [], []

    async def _draw_batch_api(self, prompts: List[str]) -> tuple[List[List[str]], List[int], List[List[int]]]:
        """Generate samples using LiteLLM API client with parallel async calls."""
        try:
            start_time = time.time()

            # Construct messages with optional system message
            messages_batch = []
            for prompt in prompts:
                messages = []
                if self.system_message:
                    messages.append({"role": "system", "content": self.system_message})
                messages.append({"role": "user", "content": prompt})
                messages_batch.append(messages)

            logger.info(f"LiteLLM API: Processing batch of {len(prompts)} prompts in parallel")

            # Log first prompt ever at INFO level - subsequent at DEBUG
            if prompts:
                if not self._logged_first_prompt:
                    logger.info(f"LiteLLM: First prompt:\n{'='*80}\n{prompts[0]}\n{'='*80}")
                else:
                    logger.debug(f"LiteLLM: First prompt in batch:\n{'='*80}\n{prompts[0]}\n{'='*80}")

            # Create async tasks for all API calls (parallel execution)
            async def call_api(messages, idx):
                """Make a single async API call."""
                try:
                    response = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        num_retries=self.max_retries,
                        **self.generate_kwargs
                    )
                    return (idx, response, None)
                except Exception as e:
                    # Extract detailed error information
                    error_msg = str(e)

                    # Try to get more details from the exception
                    details = []
                    if hasattr(e, 'status_code'):
                        details.append(f"status={e.status_code}")
                    if hasattr(e, 'llm_provider'):
                        details.append(f"provider={e.llm_provider}")

                    if details:
                        logger.error(f"LiteLLM API completion failed for prompt {idx}, {error_msg}, {' - '.join(details)}")
                    else:
                        logger.error(f"LiteLLM API completion failed for prompt {idx}, {error_msg}")
                    return (idx, None, e)

            # Execute all API calls in parallel using asyncio.gather
            tasks = [call_api(messages, idx) for idx, messages in enumerate(messages_batch)]
            results = await asyncio.gather(*tasks)

            # Process results in original order
            input_token_counts = []
            all_output_token_counts = []
            all_samples = []

            for idx, response, error in results:
                if error or response is None:
                    all_samples.append([])
                    input_token_counts.append(0)
                    all_output_token_counts.append([])
                    continue

                samples = [choice.message.content for choice in response.choices]
                all_samples.append(samples)

                # Log first output ever at INFO level - subsequent at DEBUG
                for sample_idx, sample in enumerate(samples):
                    if not self._logged_first_prompt and idx == 0 and sample_idx == 0:
                        logger.info(f"LiteLLM: First model output:\n{'='*80}\n{sample}\n{'='*80}")
                        self._logged_first_prompt = True  # Set flag after logging first prompt and output
                    else:
                        logger.debug(f"LiteLLM: Prompt {idx+1}/{len(results)}, Sample {sample_idx+1}/{len(samples)} output:\n{'='*80}\n{sample}\n{'='*80}")

                usage = response.usage
                input_token_counts.append(usage.prompt_tokens)

                completion_tokens_per_sample = usage.completion_tokens // self._samples_per_prompt
                output_token_counts_for_prompt = [completion_tokens_per_sample] * self._samples_per_prompt
                all_output_token_counts.append(output_token_counts_for_prompt)

                # Calculate cost using LiteLLM's built-in cost tracking
                try:
                    cost = litellm.completion_cost(completion_response=response)
                    self.total_cost += cost
                    self.total_input_tokens += usage.prompt_tokens
                    self.total_output_tokens += usage.completion_tokens
                    self.request_count += 1

                    logger.info(f"LiteLLM API: Request #{self.request_count} cost: ${cost:.6f} "
                              f"({usage.prompt_tokens} in + {usage.completion_tokens} out) | "
                              f"Session total: ${self.total_cost:.4f} "
                              f"({self.total_input_tokens + self.total_output_tokens:,} tokens)")

                    # Log to dedicated cost log file (with PID prefix for tracking across rotations)
                    cost_logger.info(f"[PID {os.getpid()}] model={self.model}, req={self.request_count}, "
                                   f"cost=${cost:.6f} - in={usage.prompt_tokens} - out={usage.completion_tokens} - "
                                   f"session_total=${self.total_cost:.6f}")
                except Exception as cost_err:
                    logger.warning(f"Could not calculate cost: {cost_err}")

            end_time = time.time()
            self.inference_time = end_time - start_time
            logger.debug(f"API inference time: {self.inference_time:.2f} sec")

            return all_samples, input_token_counts, all_output_token_counts

        except Exception as e:
            logger.error(f"Error during API batch generation: {e}")
            return [], [], []

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.use_local_vllm and hasattr(self, 'vllm_engine'):
                # vLLM cleanup
                del self.vllm_engine
                if VLLM_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("vLLM engine cleaned up")
            logger.info("LLM_model: Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class Sampler:
    """Node that samples program continuations and sends them for evaluation."""

    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config, device=None, log_dir=None, system_message=None, random_seed=None):
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self._config = config
        self.device = device
        self.temperature_period = self._config.temperature_period
        self.samples_per_prompt = self._config.samples_per_prompt
        self.samples_per_batch = self._config.prompts_per_batch
        self.log_dir = log_dir
        self.system_message = system_message
        self._logged_first_prompt = False  # Flag to log first prompt and output once

        try:
            self._llm = LLM_model(
                samples_per_prompt=self.samples_per_prompt,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                repetition_penalty=self._config.repetition_penalty,
                max_new_tokens=self._config.max_new_tokens,
                model=self._config.model,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                device=device,
                reasoning_effort=self._config.reasoning_effort,
                system_message=self.system_message,
                max_retries=self._config.max_retries,
                random_seed=random_seed,
            )
            mode = "LOCAL_VLLM" if self._llm.use_local_vllm else "API"
            logger.info(f"Sampler initialized: mode={mode}, model={self._config.model}, device={device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    async def consume_and_process(self) -> None:
        from disfun import process_utils

        async def _consume_loop():
            """Inner consume loop - will be wrapped with reconnection logic."""
            logger.info(f"Sampler ({self._config.model}): Setting QoS prefetch_count=10...")
            await self.channel.set_qos(prefetch_count=10)

            logger.info(f"Sampler ({self._config.model}): Starting iterator to consume messages...")
            async with self.sampler_queue.iterator() as stream:
                batch = []
                batch_timeout = 0.01  # 10ms window
                timeout_task = None

                async def process_batch_with_timeout():
                    """Process batch after timeout expires."""
                    nonlocal batch, timeout_task
                    await asyncio.sleep(batch_timeout)
                    logger.debug(f"Sampler: Timeout fired, batch size = {len(batch)}")
                    if batch:
                        # Copy batch to avoid race condition where main loop modifies it
                        batch_to_process = batch.copy()
                        batch.clear()
                        logger.debug(f"Sampler: Timeout task processing {len(batch_to_process)} messages")
                        await self.process_batch_s(batch_to_process)
                        logger.debug(f"Sampler: Timeout task finished processing")

                    # Clear timeout_task reference
                    timeout_task = None

                    # If messages arrived during processing - create new timeout task
                    if batch:
                        logger.debug(f"Sampler: Messages arrived during processing ({len(batch)}), creating new timeout task")
                        timeout_task = asyncio.create_task(process_batch_with_timeout())
                    else:
                        logger.debug(f"Sampler: Timeout task completed, no pending messages")

                logger.info(f"Sampler ({self._config.model}): Consumer registered, now listening for messages...")
                async for message in stream:
                    logger.debug(f"Sampler: Received message, current batch size = {len(batch)}, timeout_task = {timeout_task}")
                    batch.append(message)

                    # If we don't have a timeout task running - start one
                    if timeout_task is None:
                        logger.debug(f"Sampler: Creating new timeout task for batch size {len(batch)}")
                        timeout_task = asyncio.create_task(process_batch_with_timeout())

                    # If batch is full - cancel timeout and process immediately
                    if len(batch) >= self.samples_per_batch:
                        logger.debug(f"Sampler: Batch full ({len(batch)} messages), cancelling timeout and processing immediately")
                        if timeout_task:
                            timeout_task.cancel()
                            try:
                                await timeout_task  # Wait for cancellation to complete
                            except asyncio.CancelledError:
                                pass
                            timeout_task = None
                        batch_to_process = batch.copy()
                        batch.clear()
                        await self.process_batch_s(batch_to_process)
                        logger.debug(f"Sampler: Batch processing completed")

        # Wrap consume loop with automatic reconnection
        await process_utils.with_reconnection(
            _consume_loop,
            logger,
            component_name=f"Sampler ({self._config.model})"
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
                    flag = data.get("flag", False)
                    flags.append(flag)
                    prompt = programs_database.Prompt.deserialize(prompt_data)

                    if prompt.code is not None:
                        prompts.append(prompt.code)
                        metadata.append({
                            "island_id": prompt.island_id,
                            "version_generated": prompt.version_generated,
                            "expected_version": prompt.expected_version,
                            "parent_ids": data.get("parent_ids", []),
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
            samples_list, input_token_counts, output_token_counts = await self._llm.draw_batch_samples(
                prompts, total_registered_programs, self.temperature_period
            )
            inference_time = self._llm.inference_time
        except Exception as e:
            logger.error(f"LLM sampling failed: {e}")
            return

        # Calculate total samples generated in this batch to properly distribute inference time
        total_samples = sum(len(samples) for samples in samples_list)
        time_per_sample = inference_time / total_samples if total_samples > 0 else 0.0
        logger.debug(f"Batch inference time: {inference_time:.2f}s for {total_samples} samples = {time_per_sample:.4f}s per sample")

        # Publish results to the evaluator queue
        for prompt_idx, (samples, meta, flag) in enumerate(zip(samples_list, metadata, flags)):
            # Log duplicated-prompt runs for manual inspection
            if flag:
                try:
                    with open("duplicate_samples.txt", "a") as f:
                        f.write(f"Prompt Metadata:\n{meta}\n")
                        for idx, sample in enumerate(samples):
                            f.write(f"Output {idx + 1}:\n{sample}\n{'-'*50}\n")
                    logger.info("Logged duplicate prompt and outputs to 'duplicate_samples.txt'.")
                except Exception as e:
                    logger.error(f"Error logging duplicate data: {e}")

            # Send every sample to the evaluator queue
            for sample_idx, sample in enumerate(samples):
                message_data = {
                    "sample": sample,
                    "island_id": meta["island_id"],
                    "version_generated": meta["version_generated"],
                    "expected_version": meta["expected_version"],
                    "gpu_time": time_per_sample,  # For API models this is API latency, for vLLM it's GPU time
                    "input_tokens": input_token_counts[prompt_idx],
                    "output_tokens": output_token_counts[prompt_idx][sample_idx],
                    "parent_ids": meta.get("parent_ids", []),
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
        """Release LLM resources."""
        import gc
        try:
            if hasattr(self, '_llm'):
                self._llm.cleanup()
                del self._llm
            gc.collect()
            logger.info("Sampler: Cleanup completed")
        except Exception as e:
            logger.error(f"Sampler: Error during cleanup: {e}")
