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
and gpt.py (Azure OpenAI) with a single unified implemeOntation using LiteLLM.

Key features:
* Supports 100+ LLM providers through LiteLLM (OpenAI, Anthropic, Together AI, local vLLM, etc.)
* Dynamic batching based on message load (10ms window, batch size from config.prompts_per_batch)
* Dynamic temperature adjustment based on stored program count
* Token tracking for both input and output
* Time tracking (GPU time for local models - API latency for cloud models)
* vLLM backend support for 10-20x faster local inference vs HuggingFace transformers
"""

import os
import json
import logging
import asyncio
import time

import litellm
import aio_pika
from dotenv import load_dotenv

from disfun import programs_database

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


class VLLMTimeoutError(Exception):
    """Raised when vLLM inference times out.

    This exception signals that the sampler process should exit
    so ResourceManager can respawn it with a fresh vLLM instance.
    """
    pass


class VLLMRepeatedFailureError(Exception):
    """Raised when vLLM fails repeatedly.

    This exception signals that vLLM is in a bad state and the sampler
    process should exit so ResourceManager can respawn it.
    """
    pass


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
            api_base: str | None = None,
            api_key: str | None = None,
            device: int | None = None,  # GPU device for local models
            reasoning_effort: str | None = None,  # For GPT-5/o3 models
            system_message: str | None = None,  # System message for API models
            max_retries: int = 3,  # Maximum retry attempts for API calls
            random_seed: int | None = None,  # Random seed for reproducible generation
            inference_timeout: int = 300  # Timeout in seconds for vLLM inference
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
        self.inference_timeout = inference_timeout
        self.generation_counter = 0  # Counter to increment seed for each generation call
        self.previous_total_registered_programs = 0

        # Cost tracking for API models
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        # Consecutive failure tracking for vLLM health monitoring
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5  # Exit process after this many consecutive failures

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

            # Note: CUDA_VISIBLE_DEVICES must be set BEFORE importing vLLM (done in process_entry.py)
            # Initialize vLLM engine (loads model on GPU)
            # enforce_eager=True disables CUDA graphs, which helps with memory cleanup (vllm#3874)
            try:
                self.vllm_engine = vLLM_Engine(
                    model=model,
                    tensor_parallel_size=1,  # Single GPU per sampler
                    dtype="float16",
                    gpu_memory_utilization=0.90,
                    trust_remote_code=True,
                    enforce_eager=True,  # Helps with GPU memory release on cleanup
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
            prompts: list[str],
            total_registered_programs: int = 0,
            temperature_period: int = 10000
    ) -> tuple[list[list[str]], list[int], list[list[int]]]:
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
            return await self._draw_batch_vllm(prompts)
        else:
            return await self._draw_batch_api(prompts)

    async def _draw_batch_vllm(self, prompts: list[str]) -> tuple[list[list[str]], list[int], list[list[int]]]:
        """Generate samples using local vLLM engine.

        The vLLM generate() call is run in a thread pool to avoid blocking
        the asyncio event loop, which would prevent RabbitMQ heartbeats from
        being processed and cause connection timeouts.
        """
        try:
            start_time = time.time()

            logger.info(f"vLLM: Processing batch of {len(prompts)} prompts")

            # Log first prompt ever at INFO level - subsequent at DEBUG
            if prompts:
                if not self._logged_first_prompt:
                    logger.info(f"vLLM: First prompt:\n{'='*80}\n{prompts[0]}\n{'='*80}")
                else:
                    logger.debug(f"vLLM: First prompt in batch:\n{'='*80}\n{prompts[0]}\n{'='*80}")

            # Update seed for this generation call to ensure different outputs
            # while maintaining reproducibility (same base_seed + counter = same sequence)
            if self.random_seed is not None:
                current_seed = self.random_seed + self.generation_counter
                self.sampling_params.seed = current_seed
                self.generation_counter += 1
                logger.debug(f"vLLM: Using seed {current_seed} for generation #{self.generation_counter}")

            # vLLM batch generation - run in thread pool to avoid blocking event loop
            # This allows RabbitMQ heartbeats to be processed during long inference
            # Wrap with timeout to detect hung vLLM instances
            try:
                outputs = await asyncio.wait_for(
                    asyncio.to_thread(self.vllm_engine.generate, prompts, self.sampling_params),
                    timeout=self.inference_timeout
                )
            except TimeoutError as e:
                logger.error(
                    f"vLLM inference timed out after {self.inference_timeout}s for {len(prompts)} prompts."
                )
                raise VLLMTimeoutError(
                    f"vLLM inference exceeded {self.inference_timeout}s timeout"
                ) from e

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

            # Reset consecutive failure counter on success
            self.consecutive_failures = 0

            return all_samples, input_token_counts, all_output_token_counts

        except VLLMTimeoutError:
            # Re-raise timeout errors to be handled by consume_and_process
            raise

        except Exception as e:
            # Log full traceback to understand what's actually failing
            logger.error(
                f"Error during vLLM batch generation (failure {self.consecutive_failures + 1}/"
                f"{self.max_consecutive_failures}): {type(e).__name__}: {e}",
                exc_info=True
            )

            self.consecutive_failures += 1

            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.error(
                    f"vLLM has failed {self.consecutive_failures} times consecutively."
                )
                raise VLLMRepeatedFailureError(
                    f"vLLM failed {self.consecutive_failures} times consecutively"
                ) from e

            return [], [], []

    async def _draw_batch_api(self, prompts: list[str]) -> tuple[list[list[str]], list[int], list[list[int]]]:
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

            # Update seed for this generation call to ensure different outputs
            # while maintaining reproducibility (same base_seed + counter = same sequence)
            if self.random_seed is not None:
                current_seed = self.random_seed + self.generation_counter
                self.generate_kwargs["seed"] = current_seed
                self.generation_counter += 1
                logger.debug(f"LiteLLM: Using seed {current_seed} for generation #{self.generation_counter}")

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
        """Clean up vLLM GPU memory (see vllm#1908 for known issues)."""
        if not (self.use_local_vllm and getattr(self, 'vllm_engine', None)):
            return

        import gc
        logger.info("LLM_model: Cleaning up vLLM...")

        try:
            # 1. Destroy model parallel state (critical for memory release)
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except ImportError:
                pass

            # 2. Delete internal components in order
            if hasattr(self.vllm_engine, 'llm_engine'):
                if hasattr(self.vllm_engine.llm_engine, 'model_executor'):
                    if hasattr(self.vllm_engine.llm_engine.model_executor, 'driver_worker'):
                        self.vllm_engine.llm_engine.model_executor.driver_worker = None
                    self.vllm_engine.llm_engine.model_executor = None
                self.vllm_engine.llm_engine = None

            # 3. Delete engine
            self.vllm_engine = None

            # 4. GC + CUDA cleanup
            gc.collect()
            if VLLM_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("LLM_model: Cleanup done")
        except Exception as e:
            logger.error(f"LLM_model: Cleanup error: {e}")


class Sampler:
    """Node that samples program continuations and sends them for evaluation."""

    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config, device=None, log_dir=None, system_message=None, random_seed=None, rabbitmq_config=None):
        self._config = config
        self._shutdown_requested = False
        self._reconnect_delay = 5.0  # Initial reconnect delay in seconds
        self._max_reconnect_delay = 60.0  # Maximum reconnect delay
        self.device = device

        # Use shared connection manager for RabbitMQ
        from disfun import process_utils
        self._conn_manager = process_utils.RabbitMQConnectionManager(
            config=rabbitmq_config,
            component_name=f"Sampler ({config.model})",
            queue_names=["sampler_queue", "evaluator_queue"],
            logger=logger,
            timeout=300
        )
        # Initialize with passed-in connection (may be None for deferred connection)
        self._conn_manager.connection = connection
        self._conn_manager.channel = channel
        if sampler_queue:
            self._conn_manager.queues["sampler_queue"] = sampler_queue
        if evaluator_queue:
            self._conn_manager.queues["evaluator_queue"] = evaluator_queue
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
                inference_timeout=getattr(self._config, 'inference_timeout', 300),
            )
            mode = "LOCAL_VLLM" if self._llm.use_local_vllm else "API"
            logger.info(f"Sampler initialized: mode={mode}, model={self._config.model}, device={device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    # Properties for backward compatibility - delegate to connection manager
    @property
    def connection(self):
        return self._conn_manager.connection

    @connection.setter
    def connection(self, value):
        self._conn_manager.connection = value

    @property
    def channel(self):
        return self._conn_manager.channel

    @channel.setter
    def channel(self, value):
        self._conn_manager.channel = value

    @property
    def sampler_queue(self):
        return self._conn_manager.get_queue("sampler_queue")

    @sampler_queue.setter
    def sampler_queue(self, value):
        self._conn_manager.queues["sampler_queue"] = value

    @property
    def evaluator_queue(self):
        return self._conn_manager.get_queue("evaluator_queue")

    @evaluator_queue.setter
    def evaluator_queue(self, value):
        self._conn_manager.queues["evaluator_queue"] = value

    async def _ensure_connection(self):
        """Delegate to shared connection manager."""
        return await self._conn_manager.ensure_connection()

    async def _close_connection(self):
        """Delegate to shared connection manager."""
        await self._conn_manager.close()

    async def consume_and_process(self) -> None:
        """Main consume loop with automatic connection recovery.

        This method implements a robust consume loop that:
        1. Checks shutdown flag before any reconnection attempt
        2. Ensures connection is alive before consuming
        3. Catches connection errors and re-establishes the connection
        4. Uses exponential backoff for reconnection attempts
        """

        async def _consume_loop():
            """Inner consume loop - processes messages from the queue."""
            # Set prefetch to match batch size so RabbitMQ delivers enough messages for full batches
            prefetch = self.samples_per_batch
            logger.info(f"Sampler ({self._config.model}): Setting QoS prefetch_count={prefetch}...")
            await self.channel.set_qos(prefetch_count=prefetch)

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
                        logger.debug("Sampler: Timeout task finished processing")

                    # Clear timeout_task reference
                    timeout_task = None

                    # If messages arrived during processing - create new timeout task
                    if batch:
                        logger.debug(f"Sampler: Messages arrived during processing ({len(batch)}), creating new timeout task")
                        timeout_task = asyncio.create_task(process_batch_with_timeout())
                    else:
                        logger.debug("Sampler: Timeout task completed, no pending messages")

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
                        logger.debug("Sampler: Batch processing completed")

        # Main reconnection loop
        reconnect_delay = self._reconnect_delay

        while True:
            # Check shutdown flag at top of loop
            if self._shutdown_requested:
                logger.info(f"Sampler ({self._config.model}): Shutdown requested, exiting consume loop")
                break

            try:
                # Ensure connection is alive (reconnect if needed)
                if self._conn_manager.config is not None:
                    connected = await self._ensure_connection()
                    if not connected:
                        if self._shutdown_requested:
                            break
                        logger.error(f"Sampler ({self._config.model}): Failed to establish connection, retrying in {reconnect_delay:.1f}s...")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, self._max_reconnect_delay)
                        continue

                # Reset delay on successful connection
                reconnect_delay = self._reconnect_delay

                # Run the consume loop
                await _consume_loop()

                # If consume loop exits normally, break
                break

            except asyncio.CancelledError:
                # Shutdown requested - exit cleanly without reconnection
                logger.info(f"Sampler ({self._config.model}): Cancelled, exiting...")
                break

            except (VLLMTimeoutError, VLLMRepeatedFailureError) as e:
                # vLLM is dead - exit immediately for clean restart
                # Don't try reinit: it rarely helps and a fresh process is more reliable
                logger.error(
                    f"Sampler ({self._config.model}): {e}. "
                    f"Force-exiting process for clean restart by ResourceManager..."
                )
                # Try to close connection with timeout - don't block forever
                try:
                    await asyncio.wait_for(self._close_connection(), timeout=5.0)
                except (asyncio.TimeoutError, Exception) as close_err:
                    logger.warning(f"Connection close failed/timed out: {close_err}")

                # Use os._exit() to force immediate termination
                # sys.exit() can hang waiting for the stuck vLLM thread to finish
                import os
                logger.error("Calling os._exit(1) to force process termination...")
                os._exit(1)

            except (aio_pika.exceptions.AMQPConnectionError,
                    aio_pika.exceptions.ChannelClosed,
                    aio_pika.exceptions.ChannelInvalidStateError,
                    ConnectionError,
                    OSError) as e:
                # Check shutdown flag before attempting reconnection
                if self._shutdown_requested:
                    logger.info(f"Sampler ({self._config.model}): Connection error during shutdown, exiting")
                    break

                # Connection lost - attempt to reconnect
                logger.warning(
                    f"Sampler ({self._config.model}): Connection error: {e}. "
                    f"Reconnecting in {reconnect_delay:.1f}s..."
                )
                await self._close_connection()
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, self._max_reconnect_delay)
                continue

            except Exception as e:
                # Check shutdown flag before attempting reconnection
                if self._shutdown_requested:
                    logger.info(f"Sampler ({self._config.model}): Error during shutdown, exiting")
                    break

                # Unexpected error - log and retry
                logger.error(
                    f"Sampler ({self._config.model}): Unexpected error: {e}. "
                    f"Reconnecting in {reconnect_delay:.1f}s...",
                    exc_info=True
                )
                await self._close_connection()
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, self._max_reconnect_delay)
                continue

    async def process_batch_s(self, batch: list[aio_pika.IncomingMessage]):
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
                    prompt = programs_database.Prompt.deserialize(prompt_data)

                    if prompt.code is not None:
                        prompts.append(prompt.code)
                        metadata.append({
                            "island_id": prompt.island_id,
                            "version_generated": prompt.version_generated,
                            "expected_version": prompt.expected_version,
                            "parent_ids": data.get("parent_ids", []),
                        })
                        flags.append(flag)  # Only append flag when prompt is valid
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
        except (VLLMTimeoutError, VLLMRepeatedFailureError):
            # Fatal errors - vLLM engine is dead, need process restart
            raise
        except Exception as e:
            # Transient errors - skip this batch, continue with next
            logger.error(f"LLM sampling failed: {e}")
            return

        # Calculate total samples generated in this batch to properly distribute inference time
        total_samples = sum(len(samples) for samples in samples_list)
        time_per_sample = inference_time / total_samples if total_samples > 0 else 0.0
        logger.debug(f"Batch inference time: {inference_time:.2f}s for {total_samples} samples = {time_per_sample:.4f}s per sample")

        # Publish results to the evaluator queue
        for prompt_idx, (samples, meta, flag) in enumerate(zip(samples_list, metadata, flags, strict=True)):
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
                    "input_tokens": input_token_counts[prompt_idx] if sample_idx == 0 else 0,  # Input processed once per prompt
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

    def request_shutdown(self):
        """Signal that shutdown is requested - stops reconnection attempts."""
        self._shutdown_requested = True

    async def async_cleanup(self):
        """Async cleanup - close RabbitMQ connections."""
        try:
            await self._close_connection()
            logger.info("Sampler: RabbitMQ connections closed")
        except Exception as e:
            logger.error(f"Sampler: Error closing connections: {e}")

    def cleanup(self):
        """Release LLM resources (sync cleanup)."""
        import gc
        try:
            if hasattr(self, '_llm'):
                self._llm.cleanup()
                del self._llm
            gc.collect()
            logger.info("Sampler: Cleanup completed")
        except Exception as e:
            logger.error(f"Sampler: Error during cleanup: {e}")
