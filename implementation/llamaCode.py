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

    def __init__(self, samples_per_prompt: int, device="cuda", checkpoint="meta-llama/CodeLlama-13b-Python-hf") -> None:
        self._samples_per_prompt = samples_per_prompt
        self.device = device
        self.checkpoint = checkpoint
        current_directory = os.getcwd()
        sub_dir = "ChachingFace"
        self.cache_dir = os.path.join(current_directory, sub_dir)

        if device == "cuda":
            # Use all available GPUs with device_map="auto"
            self.device_map = "auto"
            self.device = device
            logger.info(f"Using all available GPUs with device_map='auto'.")
        else:
            # Use the specified single GPU
            self.device = device if isinstance(device, str) else f"cuda:{device}"
            self.device_map = None
            if self.device not in available_devices:
                raise ValueError(f"Invalid device specified: {self.device}. Available devices are: {available_devices}")
            logger.info(f"Attempting to load model on device: {self.device}")
        
        # Login to Hugging Face Hub (only when LLM_model is instantiated)
        huggingface_token = "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"  # Replace with your Hugging Face token
        login(token=huggingface_token)

        # Define tuples for sampling
        temperature_top_p_tuples = [
            (0.94445, 0.7778), (1.1667, 0.64445), (0.944445, 0.8222),
            (1.05, 0.6)
        ]

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

        # Load Llama tokenizer and model with FP16 precision
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, cache_dir=self.cache_dir)
        
        # Add padding token to process prompts in batches 
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})  # sets the EOS token as padding token
        
        try:
            # Load the model with the appropriate device map and move it to the specified device
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16  # Use FP16 precision for faster performance on GPUs
            ).to(self.device) if self.device_map is None else AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,  # Use FP16 precision for faster performance on GPUs
                device_map="auto"
            )
        except Exception as e: 
            logger.error(f"Could not download model because: {e}")
            return #Stop executing if model loading fails

        self.generate_kwargs = dict(
            temperature=self.temperature,  # Lower values (<0.5) make outputs more deterministic, higher values increase diversity.
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
            logger.debug(f"Grouped samples are {grouped_samples}")

            return grouped_samples

        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            return []


class Sampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(self, connection, channel, sampler_queue, evaluator_queue, config, d):
        self.d=d
        self.connection = connection
        self.channel = channel
        self.sampler_queue = sampler_queue
        self.evaluator_queue = evaluator_queue
        self._config = config
        self.samples_per_prompt = self._config.samples_per_prompt  
        self.samples_per_batch = self._config.programs_database.prompts_per_batch  # Access nested config for batch size
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

        # Collect prompts and metadata from each message
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
                    logger.error(f"Sampler: Exception in published prompt to evaluator_queue {e}.")
