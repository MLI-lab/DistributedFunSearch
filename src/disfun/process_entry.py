"""
Entry point functions for spawned sampler and evaluator processes.

These functions must be in a separate module (not __main__) to be pickle-able
when using multiprocessing with spawn context.
"""

import os
import sys
import asyncio
from multiprocessing import current_process
from disfun import process_utils


def sampler_process_entry(config_path, device, log_dir, log_filename):
    """Standalone sampler process entry point (spawn-compatible)."""
    # Load config FIRST to get cache_dir setting
    config = process_utils.load_config(config_path)

    # Set HF_HOME BEFORE importing any libraries that use it
    if hasattr(config.sampler, 'cache_dir') and config.sampler.cache_dir:
        os.environ["HF_HOME"] = config.sampler.cache_dir
        print(f"Sampler process: Set HF_HOME to {config.sampler.cache_dir}")

    # NOW import sampler module (vLLM will use the HF_HOME we just set)
    from disfun import sampler

    # Initialize logger in child process (with separate sampler log)
    logger = process_utils.initialize_logger(log_dir, log_filename, process_type="Sampler")

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
                logger.info(f"Sampler {local_id}: Initializing sampler with model {config.sampler.model} on device {device}...")
                sampler_instance = sampler.Sampler(
                    connection, channel, sampler_queue, evaluator_queue, config.sampler, device=device, log_dir=log_dir, system_message=system_message, random_seed=config.random_seed
                )
                logger.info(f"Sampler {local_id}: Sampler instance initialized successfully.")
            except Exception as e:
                logger.error(f"Sampler {local_id}: Could not start Sampler instance, {e}", exc_info=True)
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
        logger.info(f"Sampler {local_id}: Event loop closed.")
        sys.exit(0)


def evaluator_process_entry(config_path, template, inputs, target_signatures, log_dir, sandbox_base_path, log_filename):
    """Standalone evaluator process entry point (spawn-compatible)."""
    import disfun.evaluator as evaluator_module

    # Reload config and logger in child process (with separate evaluator log)
    config = process_utils.load_config(config_path)
    logger = process_utils.initialize_logger(log_dir, log_filename, process_type="Evaluator")

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
                cache_size_limit_gb=config.evaluator.cache_size_limit_gb
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
