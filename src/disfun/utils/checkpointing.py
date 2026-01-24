"""Checkpoint save/load functionality for ProgramsDatabase.

These functions take a ProgramsDatabase instance and access its internals directly.
This is intentional. They are extensions of ProgramsDatabase, not independent modules.
"""

import ast
import asyncio
import datetime
import logging
import os
import pickle
import random

import numpy as np

from disfun.utils import code_manipulation

logger = logging.getLogger('main_logger')


def load_checkpoint(checkpoint_file: str, database) -> None:
    """Load state from a checkpoint file into the database.

    Args:
        checkpoint_file: Path to the checkpoint pickle file
        database: ProgramsDatabase instance to load state into
    """
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)

    # Restore counters
    database.cumulative_evaluator_cpu_time = checkpoint_data.get("cumulative_evaluator_cpu_time", 0.0)
    database.cumulative_sampler_gpu_time = checkpoint_data.get("cumulative_sampler_gpu_time", 0.0)
    database.cumulative_input_tokens = checkpoint_data.get("cumulative_input_tokens", 0)
    database.cumulative_output_tokens = checkpoint_data.get("cumulative_output_tokens", 0)
    database.iterations = checkpoint_data.get("iterations", 0)
    database.duplicate_prompts = checkpoint_data.get("duplicate_prompts", 0)
    database.total_stored_programs = checkpoint_data.get("total_stored_programs", 0)
    database.execution_failed = checkpoint_data.get("execution_failed", 0)
    database.version_mismatch_discarded = checkpoint_data.get("version_mismatch_discarded", 0)
    database.duplicates_discarded = checkpoint_data.get("duplicates_discarded", 0)

    # Restore parallel vs sequential prompt metrics
    database.last_stored_count = checkpoint_data.get("last_stored_count", checkpoint_data.get("last_prompt_version", 0))
    database.parallel_prompts = checkpoint_data.get("parallel_prompts", 0)
    database.sequential_prompts = checkpoint_data.get("sequential_prompts", 0)

    # Restore optimal solution tracking
    database.found_optimal_solution = checkpoint_data.get("found_optimal_solution", False)
    logger.info(f"Optimal solution was found in prev checkpoint {database.found_optimal_solution}")
    database.prompts_since_optimal = checkpoint_data.get("prompts_since_optimal", 0)
    logger.info(f"Prompts_since_optimal are {database.prompts_since_optimal}")

    # Load W&B run ID and name if they exist in checkpoint
    database.wandb_run_id = checkpoint_data.get("wandb_run_id", None)
    if database.wandb_run_id:
        logger.info(f"Will resume W&B run: {database.wandb_run_id}")

    checkpoint_run_name = checkpoint_data.get("wandb_run_name", None)
    if checkpoint_run_name:
        database.wandb_run_name = checkpoint_run_name
        logger.info(f"Restored run name from checkpoint: {checkpoint_run_name}")

    # Restore random states for reproducibility
    numpy_random_state = checkpoint_data.get("numpy_random_state", None)
    if numpy_random_state:
        np.random.set_state(numpy_random_state)
        logger.info("Restored numpy random state from checkpoint")

    python_random_state = checkpoint_data.get("python_random_state", None)
    if python_random_state:
        random.setstate(python_random_state)
        logger.info("Restored Python random state from checkpoint")

    # Restore sampler ID counter for reproducibility
    database.next_sampler_id = checkpoint_data.get("next_sampler_id", 0)
    logger.info(f"Restored next_sampler_id from checkpoint: {database.next_sampler_id}")

    # Restore ReEvo reflection state
    reevo_state = checkpoint_data.get("reevo_state")
    if reevo_state:
        database.reevo_state = reevo_state
        logger.info(f"Restored ReEvo state: {len(reevo_state.get('new_reflections', []))} short-term reflections")

    pending_reflection_type = checkpoint_data.get("_pending_reflection_type")
    if pending_reflection_type:
        # Convert string keys back to tuples
        database._pending_reflection_type = {
            tuple(map(int, k.strip("()").split(", "))): v
            for k, v in pending_reflection_type.items()
        }
        logger.info(f"Restored {len(database._pending_reflection_type)} pending reflection types")

    # Note: final_metrics is saved to checkpoint but NOT restored here.
    # It's only computed at termination and used for post-hoc analysis (e.g., inspector.py).

    # Restore best scores
    for i, score in enumerate(checkpoint_data["best_score_per_island"]):
        database._best_score_per_island[i] = score

    database._best_program_per_island = [
        code_manipulation.Function.from_dict(program) if program else None
        for program in checkpoint_data["best_program_per_island"]
    ]

    database._best_scores_per_test_per_island = checkpoint_data["best_scores_per_test_per_island"]
    database._total_resets = checkpoint_data.get("total_resets", 0)

    # Restore islands and rebuild generation dict
    database._program_id_to_generation = {}
    for island_id, island_state in enumerate(checkpoint_data["islands_state"]):
        logger.debug(f"Loading state for island id {island_id}")
        island = database._islands[island_id]
        _load_island_state(island, island_state, database._program_id_to_generation)

    logger.info(f"Checkpoint loaded successfully. Rebuilt generation lookup for {len(database._program_id_to_generation)} programs.")


def _load_island_state(island, island_state, generation_dict=None):
    """Load the state of a single island.

    Args:
        island: Island dict to load into
        island_state: Serialized island state
        generation_dict: Optional dict to populate with program_id -> generation mappings
    """
    island['clusters'].clear()
    island['hash_set'] = set()  # Rebuild hash set from loaded programs
    for signature_str, cluster_state in island_state["clusters"].items():
        signature = ast.literal_eval(signature_str)
        if isinstance(signature, list):
            signature = tuple(signature)
        programs = [
            code_manipulation.Function.from_dict(prog_dict)
            for prog_dict in cluster_state['programs']
        ]
        cluster_data = {
            'score': cluster_state['score'],
            'scores_per_test': cluster_state.get('scores_per_test', {}),
            'programs': programs
        }
        island['clusters'][signature] = cluster_data
        # Rebuild hash set and generation dict
        for prog in programs:
            if prog.hash_value is not None:
                island['hash_set'].add(prog.hash_value)
            if generation_dict is not None and prog.program_id is not None:
                generation_dict[prog.program_id] = prog.generation if prog.generation is not None else 0

    island['version'] = island_state['version']
    island['num_programs'] = island_state['num_programs']


def serialize_checkpoint(database) -> dict:
    """Serialize the database state for checkpointing.

    Args:
        database: ProgramsDatabase instance to serialize

    Returns:
        Dictionary containing all checkpoint data
    """
    checkpoint_data = {
        "cumulative_evaluator_cpu_time": database.cumulative_evaluator_cpu_time,
        "cumulative_sampler_gpu_time": database.cumulative_sampler_gpu_time,
        "cumulative_input_tokens": database.cumulative_input_tokens,
        "cumulative_output_tokens": database.cumulative_output_tokens,
        "best_score_per_island": list(database._best_score_per_island),
        "best_program_per_island": [
            program.to_dict() if program else None
            for program in database._best_program_per_island
        ],
        "best_scores_per_test_per_island": list(database._best_scores_per_test_per_island),
        "total_resets": database._total_resets,
        "iterations": database.iterations,
        "duplicate_prompts": database.duplicate_prompts,
        "perc_duplicate_prompts": (database.duplicate_prompts / database.iterations if database.iterations else 0),
        "total_stored_programs": database.total_stored_programs,
        "execution_failed": database.execution_failed,
        "version_mismatch_discarded": database.version_mismatch_discarded,
        "duplicates_discarded": database.duplicates_discarded,
        # Parallel vs sequential prompt metrics
        "last_stored_count": database.last_stored_count,
        "parallel_prompts": database.parallel_prompts,
        "sequential_prompts": database.sequential_prompts,
        "found_optimal_solution": database.found_optimal_solution,
        "prompts_since_optimal": database.prompts_since_optimal,
        "wandb_run_id": database.wandb_run_id,
        "wandb_run_name": database.wandb_run_name,
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "next_sampler_id": database.next_sampler_id,
        # ReEvo reflection state
        "reevo_state": database.reevo_state,
        "_pending_reflection_type": {str(k): v for k, v in database._pending_reflection_type.items()},
        # Final metrics (computed at end of search)
        "final_metrics": getattr(database, 'final_metrics', None),
        "islands_state": []
    }

    for island in database._islands:
        island_state = _serialize_island_state(island)
        checkpoint_data["islands_state"].append(island_state)

    return checkpoint_data


def _serialize_island_state(island) -> dict:
    """Serialize the state of a single island."""
    clusters_state = {}
    for signature, cluster_data in island['clusters'].items():
        clusters_state[str(signature)] = _serialize_cluster_state(cluster_data)

    return {
        "clusters": clusters_state,
        "version": island['version'],
        "num_programs": island['num_programs']
    }


def _serialize_cluster_state(cluster_data) -> dict:
    """Serialize the state of a single cluster."""
    return {
        "score": cluster_data['score'],
        "programs": [program.to_dict() for program in cluster_data['programs']],
        "scores_per_test": cluster_data.get('scores_per_test', {}),
    }


async def periodic_checkpoint(database, checkpoint_interval: int = 3600):
    """Periodically save checkpoints.

    Args:
        database: ProgramsDatabase instance to checkpoint
        checkpoint_interval: Seconds between checkpoints (default: 3600 = 1 hour)
    """
    while True:
        await asyncio.sleep(checkpoint_interval)
        try:
            save_checkpoint(database)
        except Exception as e:
            logger.error(f"Error in saving checkpoint file {e}")


def save_checkpoint(database):
    """Save a checkpoint immediately.

    Args:
        database: ProgramsDatabase instance to checkpoint
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(os.getcwd(), database.save_checkpoints_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.pkl")
    data = serialize_checkpoint(database)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Checkpoint saved to {filepath}")
