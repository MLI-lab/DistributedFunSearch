"""
Entry point functions for spawned sampler and evaluator processes.

These functions must be in a separate module (not __main__) to be pickle-able
when using multiprocessing with spawn context.
"""

import os
import sys
import asyncio
from multiprocessing import current_process


def sampler_process_entry(config_path, device, log_dir, log_filename, sampler_id=0, use_parent_log=False):
    """Standalone sampler process entry point (spawn-compatible).

    Args:
        sampler_id: Unique ID for this sampler (0, 1, 2, ...) used for seed offset
        use_parent_log: If True, log to parent's log file instead of shared samplers.log
    """
    # Set CUDA_VISIBLE_DEVICES BEFORE importing anything that touches CUDA/PyTorch
    # (vLLM, torch, process_utils all cache device info on import)
    if device is not None:
        if isinstance(device, str):
            device_id = device.split(":")[-1] if ":" in device else device
        else:
            device_id = str(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"Sampler process: Set CUDA_VISIBLE_DEVICES={device_id}")
        # After setting CUDA_VISIBLE_DEVICES to a single GPU, that GPU becomes cuda:0
        # in the subprocess's view. Update the device string accordingly.
        device = "cuda:0"

    # Import process_utils AFTER setting CUDA_VISIBLE_DEVICES (it imports torch)
    from disfun import process_utils

    # Load config FIRST to get cache_dir setting
    config = process_utils.load_config(config_path)

    # Set HF_HOME BEFORE importing any libraries that use it
    if hasattr(config.sampler, 'cache_dir') and config.sampler.cache_dir:
        os.environ["HF_HOME"] = config.sampler.cache_dir
        print(f"Sampler process: Set HF_HOME to {config.sampler.cache_dir}")

    # NOW import sampler module (vLLM will use the env vars we just set)
    from disfun import sampler

    # Initialize logger in child process
    # If use_parent_log=True, log to parent's file; otherwise log to shared samplers.log
    logger = process_utils.initialize_logger(log_dir, log_filename, process_type="Sampler", use_custom_log_file=use_parent_log)

    # Load system message for API models (if configured)
    system_message = None
    if hasattr(config, 'prompt') and hasattr(config.prompt, 'system_message_path'):
        system_message_path = config.prompt.system_message_path
        if system_message_path:
            from pathlib import Path
            system_message_file = Path(system_message_path)
            if system_message_file.exists():
                system_message = system_message_file.read_text().strip()
                logger.info(f"Loaded system message from {system_message_path}")
            else:
                logger.warning(f"System message file not found at {system_message_path}")

    local_id = current_process().pid
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connection = None
    channel = None
    sampler_task = None
    sampler_instance = None
    cleanup_done = {'done': False}  # Use dict for shared state

    async def run_sampler():
        nonlocal connection, channel, sampler_task, sampler_instance, cleanup_done
        try:
            # IMPORTANT: Initialize model BEFORE connecting to RabbitMQ
            # Model loading can take 10+ minutes and blocks the event loop,
            # which would cause RabbitMQ heartbeat timeouts if connected first.
            logger.info(f"Sampler {local_id}: Initializing model {config.sampler.model} on device {device}...")
            logger.info(f"Sampler {local_id}: (RabbitMQ connection will be established AFTER model loads)")

            # Calculate unique seed for this sampler
            sampler_seed = None
            if config.random_seed is not None:
                sampler_seed = config.random_seed + sampler_id * 1_000_000
                logger.info(f"Sampler {local_id}: Using base seed {sampler_seed} (config.random_seed={config.random_seed} + sampler_id={sampler_id} * 1M)")

            # Create sampler with model but without RabbitMQ connection yet
            # Pass None for connection/channel - they'll be set via _ensure_connection()
            try:
                sampler_instance = sampler.Sampler(
                    None, None, None, None, config.sampler, device=device, log_dir=log_dir,
                    system_message=system_message, random_seed=sampler_seed, rabbitmq_config=config
                )
                logger.info(f"Sampler {local_id}: Model loaded successfully.")
            except Exception as e:
                logger.error(f"Sampler {local_id}: Could not initialize model: {e}", exc_info=True)
                return

            # NOW connect to RabbitMQ (model is loaded, heartbeats will work)
            logger.info(f"Sampler {local_id}: Connecting to RabbitMQ...")
            connection = await process_utils.create_rabbitmq_connection(config)
            channel = await connection.channel()
            sampler_queue = await process_utils.declare_standard_queue(channel, "sampler_queue")
            evaluator_queue = await process_utils.declare_standard_queue(channel, "evaluator_queue")
            logger.info(f"Sampler {local_id}: Connected to RabbitMQ successfully.")

            # Update sampler with connection
            sampler_instance.connection = connection
            sampler_instance.channel = channel
            sampler_instance.sampler_queue = sampler_queue
            sampler_instance.evaluator_queue = evaluator_queue

            logger.info(f"Sampler {local_id}: Starting consume_and_process task...")
            sampler_task = asyncio.create_task(sampler_instance.consume_and_process())
            logger.info(f"Sampler {local_id}: consume_and_process task created, now awaiting...")
            await sampler_task

        except asyncio.CancelledError:
            logger.info(f"Sampler {local_id}: Cancelled.")
        except Exception as e:
            logger.error(f"Sampler {local_id}: Error: {e}")
        finally:
            # Cleanup if not already done by signal handler
            if not cleanup_done['done']:
                cleanup_done['done'] = True
                if sampler_instance:
                    # Use sampler's async_cleanup to close its current connections
                    # (which may have been reconnected and differ from our initial refs)
                    await sampler_instance.async_cleanup()
                    sampler_instance.cleanup()
                else:
                    # Fallback: close our original references if sampler never initialized
                    if channel:
                        try:
                            await channel.close()
                        except Exception:
                            pass
                    if connection:
                        try:
                            await connection.close()
                        except Exception:
                            pass

    process_utils.setup_signal_handlers(
        loop, "Sampler", local_id, logger,
        lambda: process_utils.graceful_shutdown(
            "Sampler", local_id, logger, loop, connection, channel,
            sampler_task, sampler_instance, cleanup_done
        )
    )

    try:
        loop.run_until_complete(run_sampler())
    finally:
        loop.close()
        sys.exit(0)


def evaluator_process_entry(config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, log_filename, use_parent_log=False):
    """Standalone evaluator process entry point (spawn-compatible).

    Args:
        use_parent_log: If True, log to parent's log file instead of shared evaluators.log
    """
    from disfun import process_utils
    import disfun.evaluator as evaluator_module

    # Reload config and logger in child process
    config = process_utils.load_config(config_path)
    # If use_parent_log=True, log to parent's file; otherwise log to shared evaluators.log
    logger = process_utils.initialize_logger(log_dir, log_filename, process_type="Evaluator", use_custom_log_file=use_parent_log)

    local_id = current_process().pid
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connection = None
    channel = None
    evaluator_task = None
    evaluator_instance = None
    cleanup_done = {'done': False}  # Use dict for shared state

    async def run_evaluator():
        nonlocal connection, channel, evaluator_task, evaluator_instance, cleanup_done

        try:
            connection = await process_utils.create_rabbitmq_connection(
                config, timeout=300
            )
            channel = await connection.channel()

            evaluator_queue = await process_utils.declare_standard_queue(channel, "evaluator_queue")
            database_queue = await process_utils.declare_standard_queue(channel, "database_queue")

            evaluator_instance = evaluator_module.Evaluator(
                connection, channel, evaluator_queue, database_queue,
                template, 'priority', 'evaluate', inputs, sandbox_base_path,
                timeout_seconds=config.evaluator.timeout,
                local_id=local_id,
                target_signatures=target_signatures,
                max_workers=config.evaluator.max_workers,
                graph_dir=config.evaluator.graph_dir,
                cache_graphs=config.evaluator.cache_graphs,
                cache_size_limit_gb=config.evaluator.cache_size_limit_gb,
                rabbitmq_config=config  # Pass config for reconnection
            )

            evaluator_task = asyncio.create_task(evaluator_instance.consume_and_process())
            await evaluator_task

        except asyncio.CancelledError:
            logger.info(f"Evaluator {local_id}: Process was cancelled.")
        except Exception as e:
            logger.info(f"Evaluator {local_id}: Error occurred: {e}")
        finally:
            if not cleanup_done['done']:
                try:
                    if channel:
                        await channel.close()
                    if connection:
                        await connection.close()
                except Exception:
                    pass

    # Set up signal handlers with shared graceful_shutdown
    process_utils.setup_signal_handlers(
        loop, "Evaluator", local_id, logger,
        lambda: process_utils.graceful_shutdown(
            "Evaluator", local_id, logger, loop, connection, channel,
            evaluator_task, evaluator_instance, cleanup_done
        )
    )

    try:
        loop.run_until_complete(run_evaluator())
    finally:
        loop.close()
        logger.info(f"Evaluator {local_id}: Event loop closed.")
        sys.exit(0)
