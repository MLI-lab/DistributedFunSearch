import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import logging
import aio_pika
import asyncio
import json
import programs_database
from typing import List
import os

logger = logging.getLogger('main_logger')

class LLM_model:
    """Language model that predicts continuation of provided source code."""

    def __init__(self, samples_per_prompt: int, temperature, top_p, repetition_penalty, max_new_tokens, device="cuda", checkpoint="meta-llama/Meta-Llama-3.1-8B-Instruct") -> None:
        self._samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

        # Check if CUDA is available and validate the device
        if torch.cuda.is_available():
            available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            logger.info(f"Available devices: {available_devices}")

            # Assign the device based on user input
            self.device = device if isinstance(device, str) else f"cuda:{device}"  # Ensuring it's a valid string like 'cuda:0'
            logger.info(f"Attempting to load model on device: {self.device}")

            # Verify the device exists in available devices
            if self.device not in available_devices:
                raise ValueError(f"Invalid device specified: {self.device}. Available devices are: {available_devices}")
        else:
            self.device = "cpu"  # Fallback to CPU if no GPU is available
            logger.warning("CUDA is not available. Falling back to CPU.")

        self.checkpoint = checkpoint
        current_directory = os.getcwd()
        sub_dir = "ChachingFace"
        self.cache_dir = os.path.join(current_directory, sub_dir)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, cache_dir=self.cache_dir)
        
        # Add padding token to process prompts in batches
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})  # Set EOS token as padding token
        
        try:
            # Load the model and move it to the specified device
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, 
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16  # Use FP16 precision for faster performance on GPUs
            ).to(self.device)  # Move the model to the specified device
            
        except Exception as e:
            logger.error(f"Could not load model because: {e}")
            self.device="cuda"
            # Load the model and move it to the specified device
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, 
                cache_dir=self.cache_dir,
                device_map="auto",
                torch_dtype=torch.float16  # Use FP16 precision for faster performance on GPUs
            )

        # Setup generation parameters
        self.generate_kwargs = dict(
            temperature=self.temperature,  # Lower values make outputs more deterministic, higher values increase diversity.
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,  # Selects the top tokens whose cumulative probability exceeds the threshold.
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
            logger.debug(f"LLM: input dims is {inputs.input_ids.shape}")
            batch_size = inputs.input_ids.shape[0]
            logger.debug(f"Prompts being processed by LLM is {batch_size}")

            all_samples = []

            # Generate multiple outputs for each prompt
            for _ in range(self._samples_per_prompt):
                try:
                    outputs = self.model.generate(**inputs, **self.generate_kwargs, pad_token_id=self.tokenizer.eos_token_id)  # [batch_size, generated_length]
                except Exception as e:
                    logger.error(f"Could not generate prompts because {e}")
                logger.debug(f"LLM: output dims is {outputs.shape}")
                
                try:
                    generated_tokens = outputs[:, input_length:]  # Extract only the new tokens
                    decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    all_samples.append(decoded_texts)  # List of lists [samples_per_prompt, batch_size]
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

    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config, device):
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self._config = config
        self.samples_per_prompt = self._config.samples_per_prompt  
        self.samples_per_batch = self._config.programs_database.prompts_per_batch  # Access nested config for batch size
        self.device=device
        self._llm = LLM_model(self.samples_per_prompt, self._config.temperature, self._config.top_p, self._config.repetition_penalty, self._config.max_new_tokens, self.device)
        self.current_task = None

    async def consume_and_process(self) -> None:
        try:
            await self.channel.set_qos(prefetch_count=10)
            batch = []
            batch_timeout = 0.2  # Timeout for batch processing
            batch_start_time = asyncio.get_event_loop().time()

            async with self.sampler_queue.iterator() as stream:
                async for message in stream:
                    batch.append(message)
                    current_time = asyncio.get_event_loop().time()

                    # Process batch if size is reached or time exceeded
                    if len(batch) >= self.samples_per_batch or (current_time - batch_start_time) > batch_timeout:
                        self.current_task = asyncio.create_task(self.process_batch_s(batch))
                        try:
                            await asyncio.wait_for(self.current_task, timeout=300)  
                        except asyncio.TimeoutError:
                            logger.error("Current task took too long to complete, timing out and proceeding.")
                            self.current_task.cancel()  # Cancel the task
                            await self.current_task  # Ensure the cancellation completes
                        except Exception as e:
                            logger.error(f"Error occurred while processing batch: {e}")
                            raise

                        batch = []  # Reset batch after processing
                        batch_start_time = asyncio.get_event_loop().time()

        except asyncio.CancelledError:
            logger.info("Sampler task was canceled.")
            if batch:
                logger.info("Processing remaining batch before shutting down.")
                # Ensure pending batch is processed
                self.current_task = asyncio.create_task(self.process_batch_s(batch))
                try:
                    await asyncio.wait_for(self.current_task, timeout=300)  # Timeout for the final batch
                except asyncio.TimeoutError:
                    logger.error("Final batch took too long to process, cancelling task.")
                    self.current_task.cancel()
                    await self.current_task
        except Exception as e:
            logger.error(f"Exception in consume_and_process: {e}")
            raise
        finally:
            try:
                # Call shutdown with a timeout
                await asyncio.wait_for(self.shutdown(), timeout=100)
            except asyncio.TimeoutError:
                logger.error("Shutdown took too long and timed out.")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

    async def shutdown(self):
        if self.current_task:
            logger.info("Waiting for the current task to complete before shutdown.")
            try:
                await asyncio.wait_for(self.current_task, timeout=300)  # Wait 5 min for the task to finish
            except asyncio.TimeoutError:
                logger.warning("Current task did not finish in time. Proceeding with shutdown.")
        logger.info("Sampler has been shut down gracefully.")

    async def process_batch_s(self, batch: List[aio_pika.IncomingMessage]):
        prompts = []
        metadata = []
        messages_to_process = []

        try:
            # Collect prompts and metadata
            for message in batch:
                try:
                    prompt = programs_database.Prompt.deserialize(message.body.decode())
                    logger.info(f"Prompt is {prompt}")
                    if prompt.code is not None:
                        prompts.append(prompt.code)
                        metadata.append({
                            "island_id": prompt.island_id,
                            "version_generated": prompt.version_generated,
                            "expected_version": prompt.expected_version,
                        })
                        messages_to_process.append(message)
                    else:
                        logger.warning(f"Encountered None prompt in message {message}. Publishing None and acknowledging it.")

                        # Publish message with sample=None for the None prompt
                        message_data = {
                            "sample": None,
                            "island_id": prompt.island_id if prompt else None,
                            "version_generated": prompt.version_generated if prompt else None,
                            "expected_version": prompt.expected_version if prompt else None
                        }
                        serialized_message = json.dumps(message_data)
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=serialized_message.encode(),
                            ),
                            routing_key='evaluator_queue'
                        )
                    
                        # Acknowledge the message, no need to requeue
                        await message.ack()

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await message.reject(requeue=False)  # Reject malformed messages without requeuing

            if not messages_to_process:
                logger.warning("No valid prompts found in batch. Publishing message with sample=None.")
                return  # Exit after processing None prompts

            # Process valid prompts with the LLM
            samples_list = self._llm.draw_batch_samples(prompts)

            # Publish results for valid prompts
            for samples, meta in zip(samples_list, metadata):
                for sample in samples:
                    message_data = {
                        "sample": sample,
                        "island_id": meta["island_id"],
                        "version_generated": meta["version_generated"],
                        "expected_version": meta["expected_version"]
                    }
                    serialized_message = json.dumps(message_data)
                    await self.channel.default_exchange.publish(
                        aio_pika.Message(
                            body=serialized_message.encode(),
                        ),
                        routing_key='evaluator_queue'
                    )
            logger.debug("Successfully published valid prompts to evaluator_queue")

            # Acknowledge messages after successful processing and publishing
            for message in messages_to_process:
                await message.ack()

        except Exception as e:
            logger.error(f"Error in process_batch_s: {e}")
            raise  # Re-raise the exception to allow for further handling
        except asyncio.CancelledError:
            logger.info("Process batch was cancelled.")

