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

Key features:
* Supports 100+ LLM providers through LiteLLM (OpenAI, Anthropic, Together AI, local vLLM, etc.)
* Dynamic batching based on message load 
* Dynamic temperature adjustment based on stored program count
* Token tracking for both input and output
* Time tracking (GPU time for local models, API latency for cloud models)
* vLLM support for local inference
"""

import gc
import json
import logging
import asyncio
import os
import sys
import time
from typing import List, Optional

import aio_pika
import litellm

from disfun import programs_database

logger = logging.getLogger('main_logger')

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


class LLM_model:
    """Unified language model interface.

    Supports two modes:
    1. Local models: vLLM Python API (each sampler loads model on assigned GPU)
    2. API models: LiteLLM client (GPT, Claude, etc. via HTTP)
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
            max_retries: int = 3,  # Maximum retry attempts for API calls
            random_seed: Optional[int] = None,  # Random seed for reproducible generation
            gpu_memory_utilization: float = 0.95,  # Fraction of GPU memory for vLLM
            use_local_vllm: bool = True,  # True=local vLLM, False=LiteLLM API
            use_chat_api: bool = False,  # True=vLLM chat(), False=vLLM generate()
            enable_thinking: Optional[bool] = None,  # Qwen3 thinking mode control
            cost_model: Optional[str] = None,  # LiteLLM model name for pricing lookup
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
        self.max_retries = max_retries
        self.random_seed = random_seed
        self.generation_counter = 0  # Counter to increment seed for each generation call
        self.previous_total_registered_programs = 0  # For dynamic temperature decay based on new programs stored

        # Cost tracking for API models
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        # Logging flag for first generation prompt/output
        self._logged_first_generation = False

        # Local vLLM or LiteLLM API mode
        self.use_local_vllm = use_local_vllm
        self.use_chat_api = use_chat_api
        self.enable_thinking = enable_thinking
        self.cost_model = cost_model

        if self.use_local_vllm:
            if not VLLM_AVAILABLE:
                raise RuntimeError(f"use_local_vllm=True but vLLM is not installed. Install with: pip install vllm")

            logger.info(f"Initializing vLLM model: {model} on device {device}")
            try:
                self.vllm_engine = vLLM_Engine(
                    model=model,
                    tensor_parallel_size=1,
                    dtype="float16",
                    gpu_memory_utilization=gpu_memory_utilization,
                    trust_remote_code=True,
                    enforce_eager=False,  # Enables CUDA graphs for faster repeated inference
                )
                # Run dummy generation to compile CUDA graphs upfront (avoids slow first inference)
                warmup_params = SamplingParams(temperature=0.1, max_tokens=16, n=1)
                _ = self.vllm_engine.generate(["def hello():\n    "], warmup_params)
                logger.info(f"vLLM initialized on GPU {device}")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM: {e}")
                raise

            # vLLM sampling params (seed added at generation time with counter for reproducibility)
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                n=self._samples_per_prompt,
            )

        else:
            # LiteLLM API mode
            logger.info(f"Initializing API model via LiteLLM: {model}")

            if api_base:
                litellm.api_base = api_base
            if api_key:
                litellm.api_key = api_key

            # LiteLLM generation kwargs (seed added at generation time with counter for reproducibility)
            self.generate_kwargs = {
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
                "top_p": self.top_p,
                "frequency_penalty": max(0, self.repetition_penalty - 1.0),  # OpenAI calls it frequency_penalty, convert from repetition_penalty
                "n": self._samples_per_prompt,
            }

            # Add reasoning_effort if set (OpenAI o1/o3/gpt-5 only, API will error if unsupported)
            if self.reasoning_effort:
                self.generate_kwargs["reasoning_effort"] = self.reasoning_effort

            # Log retry configuration for API models
            logger.info(f"API retry configuration: max_retries={self.max_retries} (exponential backoff)")

        mode_str = 'LOCAL_VLLM' if self.use_local_vllm else 'API'
        if self.use_local_vllm and self.use_chat_api:
            mode_str += f"_CHAT (enable_thinking={self.enable_thinking})"
        logger.info(f"Model initialized: mode={mode_str}, temp={temperature}, top_p={top_p}, max_tokens={max_new_tokens}")

    def adjust_temperature(self, total_registered_programs: int, temperature_period: int):
        """Dynamically adjust temperature based on stored program count."""
        if temperature_period is not None:
            effective = total_registered_programs - self.previous_total_registered_programs
            new_temp = max(0, self.temperature * (1 - effective / temperature_period))

            # Update temperature (top_p stays unchanged)
            if self.use_local_vllm:
                self.sampling_params.temperature = new_temp
            else:
                self.generate_kwargs["temperature"] = new_temp

            self.previous_total_registered_programs = total_registered_programs
            logger.debug(
                f"Adjusted temperature to {new_temp} "
                f"based on {total_registered_programs} registered programs."
            )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost using LiteLLM pricing for the configured cost_model."""
        if not self.cost_model:
            return 0.0
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self.cost_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens
            )
            return prompt_cost + completion_cost
        except Exception as e:
            logger.warning(f"Could not calculate cost for {self.cost_model}: {e}")
            return 0.0

    async def draw_batch_samples(
            self,
            prompts: List[str],
            total_registered_programs: int = 0,
            temperature_period: int = 10000,
            samples_per_prompt: int | None = None,
            system_message: str | None = None,
    ) -> tuple[List[List[str]], List[int], List[List[int]]]:
        """Generate samples for a batch of prompts.

        For API models, all prompts in the batch are processed in parallel using
        asyncio.gather, reducing total latency compared to sequential calls.

        Args:
            prompts: List of prompt strings
            total_registered_programs: Current count of stored programs
            temperature_period: Period for temperature decay
            samples_per_prompt: Outputs per prompt (defaults to config value)
            system_message: Override default system message (API models only)

        Returns:
            Tuple of (grouped_samples, input_token_counts, output_token_counts)
        """
        if temperature_period is not None:
            try:
                self.adjust_temperature(total_registered_programs, temperature_period)
            except Exception as e:
                logger.error(f"Error adjusting temperature: {e}")

        n = samples_per_prompt if samples_per_prompt is not None else self._samples_per_prompt

        if self.use_local_vllm:
            if self.use_chat_api:
                return await self._draw_batch_vllm_chat(prompts, n, system_message=system_message)
            return await self._draw_batch_vllm(prompts, n)
        else:
            return await self._draw_batch_api(prompts, n, system_message=system_message)

    async def _draw_batch_vllm(self, prompts: List[str], n: int) -> tuple[List[List[str]], List[int], List[List[int]]]:
        """Generate samples using local vLLM engine.

        The vLLM generate() call is run in a thread pool to avoid blocking
        the asyncio event loop, which would prevent RabbitMQ heartbeats from
        being processed and cause connection timeouts.

        Note: system_message is not used for vLLM completion models.
        """
        try:
            start_time = time.time()

            logger.info(f"vLLM: Processing batch of {len(prompts)} prompts (n={n})")

            # Log first prompt at info level, subsequent at debug
            if prompts:
                if not self._logged_first_generation:
                    logger.info(f"vLLM: First prompt:\n{'='*80}\n{prompts[0]}\n{'='*80}")
                else:
                    logger.debug(f"vLLM: First prompt in batch:\n{'='*80}\n{prompts[0]}\n{'='*80}")

            # Create sampling params with specified n (samples_per_prompt)
            sampling_params = SamplingParams(
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                max_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                n=n,
            )
            if self.random_seed is not None:
                sampling_params.seed = self.random_seed + self.generation_counter
                self.generation_counter += 1
                logger.debug(f"vLLM: Using seed {sampling_params.seed} for generation #{self.generation_counter}")

            # vLLM batch generation, run in thread pool to avoid blocking event loop
            outputs = await asyncio.to_thread(
                self.vllm_engine.generate, prompts, sampling_params
            )

            all_samples = []
            input_token_counts = []
            all_output_token_counts = []

            for idx, output in enumerate(outputs):
                # Extract samples for this prompt
                samples = [o.text for o in output.outputs]
                all_samples.append(samples)

                # Log first output at info level, subsequent at debug
                if idx == 0 and samples:
                    if not self._logged_first_generation:
                        logger.info(f"vLLM: First model output:\n{'='*80}\n{samples[0]}\n{'='*80}")
                        self._logged_first_generation = True  # Set flag after logging first prompt and output
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

            # Calculate cost using LiteLLM pricing if cost_model is configured
            if self.cost_model:
                total_input = sum(input_token_counts)
                total_output = sum(sum(ot) for ot in all_output_token_counts)
                cost = self._calculate_cost(total_input, total_output)
                self.total_cost += cost
                self.total_input_tokens += total_input
                self.total_output_tokens += total_output
                self.request_count += 1
                logger.info(f"vLLM: Request #{self.request_count} cost: ${cost:.6f} "
                          f"({total_input} in + {total_output} out) | "
                          f"Session total: ${self.total_cost:.4f}")
                cost_logger.info(f"[PID {os.getpid()}] model={self.cost_model}, req={self.request_count}, "
                               f"cost=${cost:.6f}, in={total_input}, out={total_output}, "
                               f"session_total=${self.total_cost:.6f}")

            return all_samples, input_token_counts, all_output_token_counts

        except Exception as e:
            logger.error(f"vLLM error, restarting sampler: {e}")
            self.cleanup()
            sys.exit(1)

    async def _draw_batch_vllm_chat(self, prompts: List[str], n: int, system_message: str | None = None) -> tuple[List[List[str]], List[int], List[List[int]]]:
        """Generate samples using vLLM chat() API with chat_template_kwargs support.

        Required for models like Qwen3 where thinking mode is controlled via chat template.
        Uses llm.chat() instead of llm.generate() to apply the chat template.
        """
        try:
            start_time = time.time()

            logger.info(f"vLLM chat: Processing batch of {len(prompts)} prompts (n={n})")

            # Log first prompt
            if prompts:
                if not self._logged_first_generation:
                    logger.info(f"vLLM chat: First prompt:\n{'='*80}\n{prompts[0]}\n{'='*80}")
                else:
                    logger.debug(f"vLLM chat: First prompt in batch:\n{'='*80}\n{prompts[0]}\n{'='*80}")

            # Build messages for each prompt
            messages_batch = []
            for prompt in prompts:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                messages_batch.append(messages)

            # Create sampling params
            sampling_params = SamplingParams(
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                max_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                n=n,
            )
            if self.random_seed is not None:
                sampling_params.seed = self.random_seed + self.generation_counter
                self.generation_counter += 1

            # Build chat_template_kwargs for thinking mode control
            chat_template_kwargs = {}
            if self.enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = self.enable_thinking

            # vLLM chat generation, run in thread pool to avoid blocking event loop
            outputs = await asyncio.to_thread(
                self.vllm_engine.chat,
                messages_batch,
                sampling_params,
                chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
            )

            all_samples = []
            input_token_counts = []
            all_output_token_counts = []

            for idx, output in enumerate(outputs):
                samples = [o.text for o in output.outputs]
                all_samples.append(samples)

                # Log first output
                if idx == 0 and samples:
                    if not self._logged_first_generation:
                        logger.info(f"vLLM chat: First model output:\n{'='*80}\n{samples[0]}\n{'='*80}")
                        self._logged_first_generation = True
                    else:
                        logger.debug(f"vLLM chat: First model output:\n{'='*80}\n{samples[0]}\n{'='*80}")

                input_tokens = len(output.prompt_token_ids)
                output_tokens = [len(o.token_ids) for o in output.outputs]

                input_token_counts.append(input_tokens)
                all_output_token_counts.append(output_tokens)

            end_time = time.time()
            self.inference_time = end_time - start_time
            logger.debug(f"vLLM chat inference time: {self.inference_time:.2f} sec")

            # Calculate cost using LiteLLM pricing if cost_model is configured
            if self.cost_model:
                total_input = sum(input_token_counts)
                total_output = sum(sum(ot) for ot in all_output_token_counts)
                cost = self._calculate_cost(total_input, total_output)
                self.total_cost += cost
                self.total_input_tokens += total_input
                self.total_output_tokens += total_output
                self.request_count += 1
                logger.info(f"vLLM chat: Request #{self.request_count} cost: ${cost:.6f} "
                          f"({total_input} in + {total_output} out) | "
                          f"Session total: ${self.total_cost:.4f}")
                cost_logger.info(f"[PID {os.getpid()}] model={self.cost_model}, req={self.request_count}, "
                               f"cost=${cost:.6f}, in={total_input}, out={total_output}, "
                               f"session_total=${self.total_cost:.6f}")

            return all_samples, input_token_counts, all_output_token_counts

        except Exception as e:
            logger.error(f"vLLM chat error, restarting sampler: {e}")
            self.cleanup()
            sys.exit(1)

    async def _draw_batch_api(self, prompts: List[str], n: int, system_message: str | None = None) -> tuple[List[List[str]], List[int], List[List[int]]]:
        """Generate samples using LiteLLM API client with parallel async calls."""
        try:
            start_time = time.time()

            # Construct messages with optional system message
            messages_batch = []
            for prompt in prompts:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                messages_batch.append(messages)

            logger.info(f"LiteLLM API: Processing batch of {len(prompts)} prompts in parallel (n={n})")

            # Log first prompt at info level, subsequent at debug
            if prompts:
                if not self._logged_first_generation:
                    logger.info(f"LiteLLM: First prompt:\n{'='*80}\n{prompts[0]}\n{'='*80}")
                else:
                    logger.debug(f"LiteLLM: First prompt in batch:\n{'='*80}\n{prompts[0]}\n{'='*80}")

            # Build generation kwargs with specified n (samples_per_prompt)
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["n"] = n
            if self.random_seed is not None:
                generate_kwargs["seed"] = self.random_seed + self.generation_counter
                self.generation_counter += 1
                logger.debug(f"LiteLLM: Using seed {generate_kwargs['seed']} for generation #{self.generation_counter}")

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
                        **generate_kwargs
                    )
                    return (idx, response, None)
                except Exception as e:
                    logger.error(f"LiteLLM API error for prompt {idx}: {e}")
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

                # Log first output at info level, subsequent at debug
                for sample_idx, sample in enumerate(samples):
                    if not self._logged_first_generation and idx == 0 and sample_idx == 0:
                        logger.info(f"LiteLLM: First model output:\n{'='*80}\n{sample}\n{'='*80}")
                        self._logged_first_generation = True  # Set flag after logging first prompt and output
                    else:
                        logger.debug(f"LiteLLM: Prompt {idx+1}/{len(results)}, Sample {sample_idx+1}/{len(samples)} output:\n{'='*80}\n{sample}\n{'='*80}")

                usage = response.usage
                input_token_counts.append(usage.prompt_tokens)

                completion_tokens_per_sample = usage.completion_tokens // n
                output_token_counts_for_prompt = [completion_tokens_per_sample] * n
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
                                   f"cost=${cost:.6f}, in={usage.prompt_tokens}, out={usage.completion_tokens}, "
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
        """Clean up vLLM GPU memory."""
        if not (self.use_local_vllm and getattr(self, 'vllm_engine', None)):
            return

        logger.info("LLM_model: Cleaning up vLLM...")

        try:
            # Destroy distributed state (needed for multi-GPU setups)
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except ImportError:
                pass

            self.vllm_engine = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("LLM_model: Cleanup done")
        except Exception as e:
            logger.error(f"LLM_model: Cleanup error: {e}")


class Sampler:
    """Samples program continuations and sends them for evaluation."""

    def __init__(self, config, rabbitmq_config, device=None, log_dir=None, random_seed=None):
        self._config = config
        self.device = device

        from disfun.utils import rabbitmq
        self._conn = rabbitmq.ConnectionManager(
            config=rabbitmq_config,
            component_name=f"Sampler ({config.model})",
            queue_names=["sampler_queue", "evaluator_queue"],
            logger=logger
        )
        self.temperature_period = self._config.temperature_period
        self.samples_per_prompt = self._config.samples_per_prompt
        self.samples_per_batch = self._config.prompts_per_batch
        self.log_dir = log_dir
        self._logged_first_reflection = False  # Flag to log first reflection output once

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
                max_retries=self._config.max_retries,
                random_seed=random_seed,
                gpu_memory_utilization=getattr(self._config, 'gpu_memory_utilization', 0.95),
                use_local_vllm=self._config.use_local_vllm,
                use_chat_api=getattr(self._config, 'use_chat_api', False),
                enable_thinking=getattr(self._config, 'enable_thinking', None),
                cost_model=getattr(self._config, 'cost_model', None),
            )
            mode = "LOCAL_VLLM" if self._llm.use_local_vllm else "API"
            logger.info(f"Sampler initialized: mode={mode}, model={self._config.model}, device={device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    async def consume_and_process(self) -> None:
        """Main consume loop with automatic connection recovery."""

        while True:
            try:
                # Connect with retry (handles exponential backoff internally)
                if not await self._conn.connect_with_retry():
                    break  # Shutdown requested

                await self._consume_loop()
                break  # Normal exit

            except asyncio.CancelledError:
                logger.info(f"Sampler ({self._config.model}): Cancelled, exiting...")
                break

            except Exception as e:
                logger.warning(f"Sampler ({self._config.model}): {type(e).__name__}: {e}. Reconnecting...")
                await self._conn.close()

    async def _consume_loop(self):
        """Inner consume loop, processes messages from the queue."""
        batch_size = self.samples_per_batch
        batch_timeout = getattr(self._config, 'batch_timeout', 0.01)

        # Prefetch multiple batches to hide network latency
        prefetch_multiplier = getattr(self._config, 'prefetch_multiplier', 1)
        prefetch = batch_size * prefetch_multiplier
        await self._conn.channel.set_qos(prefetch_count=prefetch)

        logger.info(f"Sampler ({self._config.model}): Listening (batch_size={batch_size}, prefetch={prefetch})")

        async with self._conn.get_queue("sampler_queue").iterator() as stream:
            batch = []
            batch_start_time = time.time()

            async for message in stream:
                batch.append(message)

                if len(batch) >= batch_size or (time.time() - batch_start_time) >= batch_timeout:
                    await self.process_batch(batch)
                    batch.clear()
                    batch_start_time = time.time()

    async def process_batch(self, batch: List[aio_pika.IncomingMessage]):
        """Process a batch of prompts with optional reflection stage.

        Flow:
        1. Parse messages, extract generation_prompt and optional reflection_prompt
        2. If any have reflection_prompt: batch those first, fill {reflection} placeholders
        3. Batch all generation prompts together
        4. Publish results
        """
        generation_prompts = []
        reflection_data = []  # (index, prompt) for prompts needing reflection
        metadata = []
        flags = []
        total_registered_programs = 0
        batch_samples_per_prompt = None  # Use first prompt's value for the batch
        batch_system_message = None  # Use first prompt's value for the batch
        batch_reflection_system_message = None

        for i, message in enumerate(batch):
            try:
                async with message.process():
                    data = json.loads(message.body.decode())
                    prompt_data = data["prompt"]
                    total_registered_programs = data.get("total_registered_programs", 0)
                    flag = data.get("flag", False)
                    prompt = programs_database.Prompt.from_dict(prompt_data)

                    if not prompt.generation_prompt:
                        logger.warning(f"Skipping prompt with island_id {prompt.island_id}: empty prompt")
                        continue

                    generation_prompts.append(prompt.generation_prompt)
                    flags.append(flag)
                    metadata.append({
                        "island_id": prompt.island_id,
                        "version_generated": prompt.version_generated,
                        "expected_version": prompt.expected_version,
                        "parent_ids": data.get("parent_ids", []),
                    })

                    # Track batch-level settings (use first prompt's values)
                    if batch_samples_per_prompt is None:
                        batch_samples_per_prompt = prompt.samples_per_prompt
                    if batch_system_message is None:
                        batch_system_message = prompt.system_message
                    if batch_reflection_system_message is None:
                        batch_reflection_system_message = prompt.reflection_system_message

                    # Track reflection prompts if present (ReEvo only)
                    if prompt.reflection_prompt:
                        reflection_data.append((
                            len(generation_prompts) - 1,  # Index in generation_prompts
                            prompt.reflection_prompt,
                        ))

            except Exception as e:
                logger.error(f"Sampler: Error processing message: {e}")
                continue

        if not generation_prompts:
            logger.warning("No valid prompts in batch; skipping.")
            return

        # Batch reflection prompts (if any)
        reflection_input_tokens = [0] * len(generation_prompts)
        reflection_outputs = [None] * len(generation_prompts)  # Track reflection outputs per prompt
        total_inference_time = 0.0

        if reflection_data:
            indices, refl_prompts = zip(*reflection_data)
            logger.debug(f"Running batched reflection for {len(refl_prompts)} prompts")

            # Use draw_batch_samples with samples_per_prompt=1 for reflections
            refl_samples, refl_in_tokens, _ = await self._llm.draw_batch_samples(
                list(refl_prompts), samples_per_prompt=1,
                system_message=batch_reflection_system_message
            )
            total_inference_time += self._llm.inference_time

            # Flatten: each refl_samples[i] is a list with one element
            refl_outputs = [samples[0] if samples else "" for samples in refl_samples]

            # Log first reflection
            if not self._logged_first_reflection and refl_outputs:
                logger.info(f"First reflection output:\n{'='*80}\n{refl_outputs[0]}\n{'='*80}")
                self._logged_first_reflection = True

            # Fill {reflection} placeholders and store reflection outputs
            for j, idx in enumerate(indices):
                generation_prompts[idx] = generation_prompts[idx].replace("{reflection}", refl_outputs[j])
                reflection_input_tokens[idx] = refl_in_tokens[j]
                reflection_outputs[idx] = refl_outputs[j]  # Store for sending to database

        # Batch all generation prompts
        try:
            samples_list, gen_input_tokens, output_token_counts = await self._llm.draw_batch_samples(
                generation_prompts, total_registered_programs, self.temperature_period,
                samples_per_prompt=batch_samples_per_prompt,
                system_message=batch_system_message,
            )
            total_inference_time += self._llm.inference_time
        except Exception as e:
            logger.error(f"LLM sampling failed: {e}")
            return

        # Combine input tokens (reflection + generation)
        input_token_counts = [
            reflection_input_tokens[i] + (gen_input_tokens[i] if gen_input_tokens else 0)
            for i in range(len(generation_prompts))
        ]

        # Calculate total samples generated in this batch to properly distribute inference time
        total_samples = sum(len(samples) for samples in samples_list)
        time_per_sample = total_inference_time / total_samples if total_samples > 0 else 0.0
        logger.debug(f"Batch inference time: {total_inference_time:.2f}s for {total_samples} samples = {time_per_sample:.4f}s per sample")

        # Collect all messages for batch publishing
        messages_to_publish = []
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

            for sample_idx, sample in enumerate(samples):
                output_tokens = 0
                if output_token_counts and prompt_idx < len(output_token_counts):
                    if sample_idx < len(output_token_counts[prompt_idx]):
                        output_tokens = output_token_counts[prompt_idx][sample_idx]

                message_data = {
                    "sample": sample,
                    "island_id": meta["island_id"],
                    "version_generated": meta["version_generated"],
                    "expected_version": meta["expected_version"],
                    "gpu_time": time_per_sample,
                    "input_tokens": input_token_counts[prompt_idx] if sample_idx == 0 else 0,
                    "output_tokens": output_tokens,
                    "parent_ids": meta.get("parent_ids", []),
                    # Include reflection output for ReEvo (only on first sample to avoid duplicates)
                    "reflection_output": reflection_outputs[prompt_idx] if sample_idx == 0 else None,
                }
                messages_to_publish.append(aio_pika.Message(body=json.dumps(message_data).encode()))

        # Batch publish all messages concurrently
        if messages_to_publish:
            try:
                await asyncio.gather(*[
                    self._conn.channel.default_exchange.publish(msg, routing_key="evaluator_queue")
                    for msg in messages_to_publish
                ])
                logger.debug(f"Batch published {len(messages_to_publish)} samples to evaluator_queue.")
            except Exception as e:
                logger.error(f"Error batch publishing samples: {e}")

    async def shutdown(self):
        """Graceful shutdown: close connection and release model."""
        await self._conn.close()
        self.cleanup()

    def cleanup(self):
        """Release LLM resources (sync cleanup)."""
        try:
            if hasattr(self, '_llm'):
                self._llm.cleanup()
                del self._llm
            gc.collect()
            logger.info("Sampler: Cleanup completed")
        except Exception as e:
            logger.error(f"Sampler: Error during cleanup: {e}")
