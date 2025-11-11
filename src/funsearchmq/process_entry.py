"""
Entry point functions for spawned sampler and evaluator processes.

These functions must be in a separate module (not __main__) to be pickle-able
when using multiprocessing with spawn context.
"""

import os
import sys
import asyncio
import logging
from logging import FileHandler
from multiprocessing import current_process

def load_config(config_path):
    """Dynamically load a configuration module from a given file path."""
    import importlib.util

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, "Config"):
        raise ValueError(f"The configuration file at {config_path} must define a 'Config' class.")

    return config_module.Config()


def initialize_process_logger(log_dir, log_filename):
    """Initialize logger for child process (spawn-compatible).

    Args:
        log_dir: Directory containing the log file
        log_filename: Name of the log file to write to (shared with parent process)
    """
    log_file_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    # Use append mode so multiple processes can write to the same file
    handler = FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def sampler_process_entry(config_path, device, log_dir, log_filename):
    """Standalone sampler process entry point (spawn-compatible)."""
    from funsearchmq import sampler, gpt, process_utils

    # Reload config and logger in child process
    config = load_config(config_path)
    logger = initialize_process_logger(log_dir, log_filename)

    local_id = current_process().pid
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connection = None
    channel = None
    sampler_task = None
    sampler_instance = None
    cleanup_done = False

    async def graceful_shutdown(loop, connection, channel, sampler_task, sampler_instance):
        nonlocal cleanup_done
        if cleanup_done:
            return

        logger.info(f"Sampler {local_id}: Initiating graceful shutdown...")

        if sampler_task and not sampler_task.done():
            logger.info(f"Sampler {local_id}: Cancelling consume task...")
            sampler_task.cancel()
            try:
                await asyncio.wait_for(sampler_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.warning(f"Sampler {local_id}: Error cancelling task: {e}")

        if sampler_instance:
            try:
                logger.info(f"Sampler {local_id}: Cleaning up LLM resources...")
                sampler_instance.cleanup()
                del sampler_instance
            except Exception as e:
                logger.error(f"Sampler {local_id}: Error during sampler cleanup: {e}")

        if channel:
            try:
                await channel.close()
            except Exception as e:
                logger.info(f"Sampler {local_id}: Channel already closed during shutdown")

        if connection:
            try:
                await connection.close()
            except Exception as e:
                logger.info(f"Sampler {local_id}: Connection already closed during shutdown")

        cleanup_done = True
        logger.info(f"Sampler {local_id}: Graceful shutdown complete.")
        loop.stop()

    async def run_sampler():
        nonlocal connection, channel, sampler_task, sampler_instance, cleanup_done
        try:
            logger.info(f"Sampler {local_id}: Starting connection to RabbitMQ on device {device}...")
            connection = await process_utils.create_rabbitmq_connection(
                config, timeout=300
            )
            logger.info(f"Sampler {local_id}: Connected to RabbitMQ successfully.")
            channel = await connection.channel()
            logger.info(f"Sampler {local_id}: Channel established.")

            sampler_queue = await process_utils.declare_standard_queue(channel, "sampler_queue")
            logger.info(f"Sampler {local_id}: Declared sampler_queue.")

            evaluator_queue = await process_utils.declare_standard_queue(channel, "evaluator_queue")
            logger.info(f"Sampler {local_id}: Declared evaluator_queue.")

            try:
                if config.sampler.gpt:
                    logger.info(f"Sampler {local_id}: Initializing GPT sampler...")
                    sampler_instance = gpt.Sampler(
                        connection, channel, sampler_queue, evaluator_queue, config.sampler)
                    logger.info(f"Sampler {local_id}: GPT Sampler instance initialized successfully.")
                else:
                    logger.info(f"Sampler {local_id}: Initializing LLM sampler on device {device}...")
                    sampler_instance = sampler.Sampler(
                        connection, channel, sampler_queue, evaluator_queue, config.sampler, device)
                    logger.info(f"Sampler {local_id}: LLM Sampler instance initialized successfully on device {device}.")
            except Exception as e:
                logger.error(f"Sampler {local_id}: Could not start Sampler instance - {e}", exc_info=True)
                return

            logger.info(f"Sampler {local_id}: Starting consume_and_process task...")
            sampler_task = asyncio.create_task(sampler_instance.consume_and_process())
            logger.info(f"Sampler {local_id}: consume_and_process task created, now awaiting...")
            await sampler_task

        except asyncio.CancelledError:
            print(f"Sampler {local_id}: Process was cancelled.")
        except Exception as e:
            print(f"Sampler {local_id} encountered an error: {e}")
        finally:
            if not cleanup_done:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                logger.debug(f"Sampler {local_id}: Connection closed.")

    # Set up signal handlers
    process_utils.setup_signal_handlers(
        loop, "Sampler", local_id, logger,
        lambda: graceful_shutdown(loop, connection, channel, sampler_task, sampler_instance)
    )

    try:
        loop.run_until_complete(run_sampler())
    finally:
        loop.close()
        logger.info(f"Sampler {local_id}: Event loop closed.")
        sys.exit(0)


def evaluator_process_entry(config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, log_filename):
    """Standalone evaluator process entry point (spawn-compatible)."""
    import funsearchmq.evaluator as evaluator_module
    from funsearchmq import process_utils

    # Reload config and logger in child process
    config = load_config(config_path)
    logger = initialize_process_logger(log_dir, log_filename)

    local_id = current_process().pid
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connection = None
    channel = None
    evaluator_task = None
    evaluator_instance = None
    cleanup_done = False

    async def graceful_shutdown(loop, connection, channel, evaluator_task, evaluator_instance):
        nonlocal cleanup_done
        if cleanup_done:
            return

        logger.info(f"Evaluator {local_id}: Initiating graceful shutdown...")

        if evaluator_task and not evaluator_task.done():
            logger.info(f"Evaluator {local_id}: Cancelling consume task...")
            evaluator_task.cancel()
            try:
                await asyncio.wait_for(evaluator_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.warning(f"Evaluator {local_id}: Error cancelling task: {e}")

        if evaluator_instance:
            try:
                logger.info(f"Evaluator {local_id}: Shutting down evaluator instance...")
                await evaluator_instance.shutdown()
            except Exception as e:
                logger.error(f"Evaluator {local_id}: Error during evaluator shutdown: {e}")

        if channel:
            try:
                await channel.close()
            except Exception as e:
                logger.info(f"Evaluator {local_id}: Channel already closed during shutdown")

        if connection:
            try:
                await connection.close()
            except Exception as e:
                logger.info(f"Evaluator {local_id}: Connection already closed during shutdown")

        cleanup_done = True
        logger.info(f"Evaluator {local_id}: Graceful shutdown complete.")
        loop.stop()

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
                max_workers=config.evaluator.max_workers
            )

            evaluator_task = asyncio.create_task(evaluator_instance.consume_and_process())
            await evaluator_task

        except asyncio.CancelledError:
            logger.info(f"Evaluator {local_id}: Process was cancelled.")
        except Exception as e:
            logger.info(f"Evaluator {local_id}: Error occurred: {e}")
        finally:
            if not cleanup_done:
                if channel:
                    await channel.close()
                if connection:
                    await connection.close()
                logger.debug(f"Evaluator {local_id}: Connection/Channel closed.")

    # Set up signal handlers
    process_utils.setup_signal_handlers(
        loop, "Evaluator", local_id, logger,
        lambda: graceful_shutdown(loop, connection, channel, evaluator_task, evaluator_instance)
    )

    try:
        loop.run_until_complete(run_evaluator())
    finally:
        loop.close()
        logger.info(f"Evaluator {local_id}: Event loop closed.")
        sys.exit(0)
