"""Weights & Biases logging functionality for ProgramsDatabase.

These functions take a ProgramsDatabase instance and access its internals directly.
This is intentional. They are extensions of ProgramsDatabase, not independent modules.
"""

import os
import logging
import asyncio
import numpy as np

# Wandb import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger('main_logger')


def convert_tuple_keys(obj):
    """
    Recursively convert tuple keys to strings for JSON serialization.

    Useful when passing dataclass configs to wandb.init(), since JSON
    doesn't support tuple keys (e.g., {(7, 2): 5} -> {"(7, 2)": 5}).

    Args:
        obj: Dict, list, or other object to process

    Returns:
        Object with tuple keys converted to strings
    """
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: convert_tuple_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuple_keys(item) for item in obj]
    return obj


def build_wandb_init_config(
    config,
    prompt_spec,
    evaluator_config,
    sampler_config,
    termination_config=None
) -> dict:
    """Build config dict for W&B initialization.

    Collects settings from various config objects into a single dict for logging.
    """
    wandb_config = {
        # ProgramsDatabase config
        "num_islands": config.num_islands,
        "fewshot_num_examples": prompt_spec.fewshot_num_examples if prompt_spec else 2,
        "reset_programs": config.reset_programs,
        "cluster_sampling_temperature_init": config.cluster_sampling_temperature_init,
        "cluster_sampling_temperature_period": config.cluster_sampling_temperature_period,
        "no_deduplication": config.no_deduplication,
        # Evaluator config
        "mode": evaluator_config.mode if evaluator_config else None,
        "q": evaluator_config.q if evaluator_config else None,
        # Termination config
        "iteration_limit": termination_config.iteration_limit if termination_config else None,
        "optimal_solution_programs": termination_config.optimal_solution_programs if termination_config else None,
        "target_signatures": str(termination_config.target_solutions) if termination_config and termination_config.target_solutions else None,
    }

    if evaluator_config:
        wandb_config.update({
            "timeout": evaluator_config.timeout,
            "max_workers": evaluator_config.max_workers,
            "start_n": str(getattr(evaluator_config, 'start_n', None)),
            "end_n": str(getattr(evaluator_config, 'end_n', None)),
            "s_values": str(getattr(evaluator_config, 's_values', None)),
        })
        eval_script_path = getattr(evaluator_config, 'evaluation_script_path', None)
        if eval_script_path:
            try:
                with open(eval_script_path, 'r') as f:
                    wandb_config["evaluation_script"] = f.read()
                wandb_config["evaluation_script_path"] = eval_script_path
            except Exception as e:
                logger.warning(f"Failed to read evaluation script for W&B logging: {e}")

    if sampler_config:
        wandb_config.update({
            "samples_per_prompt": sampler_config.samples_per_prompt,
            "temperature": sampler_config.temperature,
            "temperature_period": sampler_config.temperature_period,
            "max_new_tokens": sampler_config.max_new_tokens,
            "top_p": sampler_config.top_p,
            "repetition_penalty": sampler_config.repetition_penalty,
            "model": sampler_config.model,
            "prompts_per_batch_sampler": sampler_config.prompts_per_batch,
            "model_params_billions": getattr(sampler_config, 'model_params_billions', None),
        })

    return wandb_config


async def compute_rabbitmq_metrics(rabbitmq_manager) -> dict:
    """Compute RabbitMQ metrics for Weights & Biases logging.

    Args:
        rabbitmq_manager: RabbitMQManager instance for queue metrics

    Returns:
        Dictionary of queue metrics to log
    """
    metrics = {}

    if rabbitmq_manager is None:
        return metrics

    try:
        rmq_metrics = await rabbitmq_manager.get_metrics()

        # Queue depths
        metrics["rabbitmq/sampler_queue_depth"] = rmq_metrics.get("sampler_queue_depth", 0)
        metrics["rabbitmq/evaluator_queue_depth"] = rmq_metrics.get("evaluator_queue_depth", 0)
        metrics["rabbitmq/database_queue_depth"] = rmq_metrics.get("database_queue_depth", 0)
        metrics["rabbitmq/total_messages"] = rmq_metrics.get("total_messages", 0)

        # Queue consumers
        queue_consumers = rmq_metrics.get("queue_consumers", {})
        metrics["rabbitmq/sampler_consumers"] = queue_consumers.get("sampler_queue", 0)
        metrics["rabbitmq/evaluator_consumers"] = queue_consumers.get("evaluator_queue", 0)
        metrics["rabbitmq/database_consumers"] = queue_consumers.get("database_queue", 0)

        # Connection status (1 = connected, 0 = disconnected)
        connections = rmq_metrics.get("connections", {})
        for component, is_connected in connections.items():
            metrics[f"rabbitmq/connected_{component.lower()}"] = 1 if is_connected else 0

    except Exception as e:
        logger.debug(f"Failed to get RabbitMQ metrics: {e}")

    return metrics


def compute_wandb_metrics(database) -> dict:
    """Compute metrics for Weights & Biases logging.

    Args:
        database: ProgramsDatabase instance to extract metrics from

    Returns:
        Dictionary of metrics to log
    """
    metrics = {}

    # 1. Best score per island (overall and detailed per test input)
    for island_id, score in enumerate(database._best_score_per_island):
        metrics[f"island_{island_id}/best_score"] = score

        # Log detailed scores for each evaluation input (n, s)
        scores_per_test = database._best_scores_per_test_per_island[island_id]
        if scores_per_test is not None:
            for test_key, test_score in scores_per_test.items():
                # test_key is (n, s) tuple
                if isinstance(test_key, tuple) and len(test_key) >= 2:
                    n, s = test_key[0], test_key[1]
                    metrics[f"island_{island_id}/score_n{n}_s{s}"] = test_score
                else:
                    metrics[f"island_{island_id}/score_{test_key}"] = test_score

    # 2. Overall best score across all islands
    metrics["overall/best_score"] = max(database._best_score_per_island)

    # Log overall best detailed scores
    best_island_id = np.argmax(database._best_score_per_island)
    best_scores_per_test = database._best_scores_per_test_per_island[best_island_id]
    if best_scores_per_test is not None:
        for test_key, test_score in best_scores_per_test.items():
            if isinstance(test_key, tuple) and len(test_key) >= 2:
                n, s = test_key[0], test_key[1]
                metrics[f"overall/score_n{n}_s{s}"] = test_score
            else:
                metrics[f"overall/score_{test_key}"] = test_score

    # 3. CPU and GPU times
    metrics["resources/cumulative_cpu_time"] = database.cumulative_evaluator_cpu_time
    metrics["resources/cumulative_gpu_time"] = database.cumulative_sampler_gpu_time

    # 4. Token counts
    metrics["tokens/cumulative_input"] = database.cumulative_input_tokens
    metrics["tokens/cumulative_output"] = database.cumulative_output_tokens
    metrics["tokens/cumulative_total"] = database.cumulative_input_tokens + database.cumulative_output_tokens

    # 4b. FLOP estimation (2N FLOPs per token, where N = model parameters)
    # This approximation comes from the forward pass requiring ~2N multiply-adds per token
    if database.model_params_billions is not None:
        total_tokens = database.cumulative_input_tokens + database.cumulative_output_tokens
        model_params = database.model_params_billions * 1e9  # Convert to actual parameter count
        cumulative_flops = 2 * model_params * total_tokens
        metrics["compute/cumulative_flops"] = cumulative_flops
        metrics["compute/cumulative_pflops"] = cumulative_flops / 1e15  # PetaFLOPs for readability

    # 5. Number of clusters per island and cluster sizes
    cluster_sizes_all = []
    for island_id, island in enumerate(database._islands):
        num_clusters = len(island['clusters'])
        metrics[f"island_{island_id}/num_clusters"] = num_clusters
        metrics[f"island_{island_id}/num_programs"] = island['num_programs']

        # Get cluster sizes for this island
        cluster_sizes = [len(cluster_data['programs'])
                        for cluster_data in island['clusters'].values()]

        if cluster_sizes:
            metrics[f"island_{island_id}/avg_cluster_size"] = np.mean(cluster_sizes)
            metrics[f"island_{island_id}/max_cluster_size"] = np.max(cluster_sizes)
            metrics[f"island_{island_id}/min_cluster_size"] = np.min(cluster_sizes)
            cluster_sizes_all.extend(cluster_sizes)

    # 6. Overall cluster statistics
    if cluster_sizes_all:
        metrics["clusters/overall_avg_size"] = np.mean(cluster_sizes_all)
        metrics["clusters/overall_max_size"] = np.max(cluster_sizes_all)
        metrics["clusters/overall_min_size"] = np.min(cluster_sizes_all)
        metrics["clusters/total_count"] = sum(len(island['clusters']) for island in database._islands)

    # 7. Program statistics
    metrics["programs/total_stored"] = database.total_stored_programs
    metrics["programs/execution_failed"] = database.execution_failed
    metrics["programs/version_mismatch_discarded"] = database.version_mismatch_discarded
    metrics["programs/duplicates_discarded"] = database.duplicates_discarded

    # 8. Prompt statistics
    metrics["iterations/total"] = database.iterations
    metrics["prompts/duplicate"] = database.duplicate_prompts

    # 9. Island reset tracking
    metrics["evolution/total_resets"] = database._total_resets

    # 10. Parallel vs sequential prompts (did DB change between consecutive prompts?)
    metrics["parallelism/database_version"] = database.database_version
    metrics["parallelism/parallel_prompts"] = database.parallel_prompts
    metrics["parallelism/sequential_prompts"] = database.sequential_prompts
    total_prompts = database.parallel_prompts + database.sequential_prompts
    if total_prompts > 0:
        metrics["parallelism/parallel_ratio"] = database.parallel_prompts / total_prompts

    # 11. Optimal solution tracking
    if database.found_optimal_solution:
        metrics["solution/found_optimal"] = 1
        metrics["solution/prompts_since_optimal"] = database.prompts_since_optimal
    else:
        metrics["solution/found_optimal"] = 0

    # 12. Evolutionary lineage tracking (only if enabled)
    if database.save_lineage and database.lineage_log:
        # Basic statistics
        generations = [entry['generation'] for entry in database.lineage_log]
        metrics["lineage/max_generation"] = max(generations)
        metrics["lineage/avg_generation"] = np.mean(generations)
        metrics["lineage/total_programs_tracked"] = len(database.lineage_log)

        # Recent lineage activity (last 100 programs)
        recent_lineage = database.lineage_log[-100:]
        recent_generations = [entry['generation'] for entry in recent_lineage]
        metrics["lineage/recent_avg_generation"] = np.mean(recent_generations)
        metrics["lineage/recent_max_generation"] = max(recent_generations)

        # Count programs by generation
        generation_counts = {}
        for gen in generations:
            generation_counts[gen] = generation_counts.get(gen, 0) + 1

        # Log generation distribution (up to generation 10)
        for gen in range(min(11, max(generations) + 1)):
            metrics[f"lineage/generation_{gen}_count"] = generation_counts.get(gen, 0)

        # Parent count statistics
        parent_counts = [len(entry['parent_ids']) for entry in database.lineage_log]
        metrics["lineage/avg_parents_per_program"] = np.mean(parent_counts)
        metrics["lineage/programs_with_no_parents"] = sum(1 for count in parent_counts if count == 0)

    return metrics


def get_program_by_id(database, program_id: int):
    """Find a program by its ID across all islands.

    Args:
        database: ProgramsDatabase instance
        program_id: The program ID to search for

    Returns:
        The program if found, None otherwise
    """
    for island in database._islands:
        for cluster in island['clusters'].values():
            for program in cluster.get('programs', []):
                if program.program_id == program_id:
                    return program
    return None


def trace_lineage(database, program_id: int, max_depth: int = 100):
    """Trace the full evolutionary lineage of a program.

    Args:
        database: ProgramsDatabase instance
        program_id: ID of the program to trace
        max_depth: Maximum depth to trace

    Returns:
        List of dictionaries containing program, generation, score,
        scores_per_test, and parent_ids for each ancestor.
    """
    lineage = []
    visited = set()
    current_ids = [program_id]
    depth = 0

    while current_ids and depth < max_depth:
        next_ids = []
        for pid in current_ids:
            if pid in visited:
                continue
            visited.add(pid)

            # Find the program
            program = get_program_by_id(database, pid)
            if program is None:
                # Try to find in lineage_log
                for entry in database.lineage_log:
                    if entry['program_id'] == pid:
                        lineage.append({
                            'program_id': pid,
                            'program': None,  # Program no longer in memory
                            'generation': entry['generation'],
                            'score': entry['score'],
                            'parent_ids': entry['parent_ids'],
                            'timestamp': entry.get('timestamp'),
                        })
                        next_ids.extend(entry['parent_ids'])
                        break
                continue

            # Find the program's scores from lineage_log
            program_entry = None
            for entry in database.lineage_log:
                if entry['program_id'] == pid:
                    program_entry = entry
                    break

            if program_entry:
                lineage.append({
                    'program_id': pid,
                    'program': program,
                    'generation': program.generation,
                    'score': program_entry['score'],
                    'parent_ids': program.parent_ids or [],
                    'timestamp': program.timestamp,
                })
                next_ids.extend(program.parent_ids or [])

        current_ids = next_ids
        depth += 1

    return lineage


def generate_lineage_html(database, program_id: int, island_id: int):
    """Generate an HTML visualization of a program's evolutionary lineage.

    Args:
        database: ProgramsDatabase instance
        program_id: ID of the program
        island_id: Island the program belongs to

    Returns:
        HTML string with the visualization
    """
    lineage = trace_lineage(database, program_id)

    if not lineage:
        return "<html><body><h1>No lineage found</h1></body></html>"

    # Sort lineage by generation (newest to oldest)
    lineage.sort(key=lambda x: x['generation'], reverse=True)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Evolutionary Lineage - Program {program_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; }}
        .program-card {{
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .program-card:hover {{ border-color: #4CAF50; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .generation {{ background: #4CAF50; color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
        .score {{ background: #2196F3; color: white; padding: 5px 10px; border-radius: 4px; }}
        .parent-ids {{ color: #666; font-size: 14px; margin: 5px 0; }}
        .code-block {{
            background: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        .current {{ border-color: #FF9800; border-width: 3px; }}
        .arrow {{ text-align: center; color: #999; font-size: 24px; margin: 10px 0; }}
        .metadata {{ color: #666; font-size: 12px; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>Evolutionary Lineage - Island {island_id} - Program {program_id}</h1>
    <p>Showing {len(lineage)} programs in the evolutionary chain (newest to oldest)</p>
"""

    for i, entry in enumerate(lineage):
        is_current = (i == 0)
        parent_str = ", ".join(str(p) for p in entry['parent_ids']) if entry['parent_ids'] else "None (baseline)"

        code = ""
        if entry['program'] is not None:
            code = str(entry['program'])
        else:
            code = "[Program no longer in memory]"

        html += """
    <div class="program-card{current}">
        <div class="header">
            <h3>Program ID: {pid}</h3>
            <div>
                <span class="generation">Gen {gen}</span>
                <span class="score">Score: {score:.2f}</span>
            </div>
        </div>
        <div class="parent-ids"><strong>Parents:</strong> {parents}</div>
        <div class="code-block">{code}</div>
        <div class="metadata">Timestamp: {timestamp}</div>
    </div>
""".format(
            current=" current" if is_current else "",
            pid=entry['program_id'],
            gen=entry['generation'],
            score=entry['score'],
            parents=parent_str,
            code=code.replace('<', '&lt;').replace('>', '&gt;'),
            timestamp=entry.get('timestamp', 'N/A')
        )

        if i < len(lineage) - 1:
            html += '    <div class="arrow">↓ evolved from ↓</div>\n'

    html += """
</body>
</html>
"""
    return html


def generate_lineage_tree_diagram(database, program_id: int, island_id: int):
    """Generate a simple tree diagram showing the genealogy structure.

    Args:
        database: ProgramsDatabase instance
        program_id: ID of the program
        island_id: Island the program belongs to

    Returns:
        HTML string with the tree diagram
    """
    lineage = trace_lineage(database, program_id)

    if not lineage:
        return "<html><body><h1>No lineage found</h1></body></html>"

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Lineage Tree - Program {program_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; }}
        .tree {{ margin: 20px; }}
        .node {{
            background: white;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 10px;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .node:hover {{ background: #e8f5e9; }}
        .current {{ border-color: #FF9800; border-width: 3px; background: #fff3e0; }}
        .baseline {{ border-color: #9C27B0; background: #f3e5f5; }}
        .node-id {{ font-weight: bold; color: #333; }}
        .node-gen {{ color: #666; font-size: 12px; }}
        .node-score {{ color: #2196F3; font-weight: bold; }}
        .arrow {{ color: #999; margin: 0 10px; }}
        .generation-group {{ margin: 20px 0; padding: 15px; background: white; border-radius: 8px; }}
        .gen-label {{ font-weight: bold; color: #4CAF50; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>Lineage Tree - Island {island_id} - Program {program_id}</h1>
    <p>Genealogy structure showing {num_programs} programs across {num_generations} generations</p>
""".format(
        program_id=program_id,
        island_id=island_id,
        num_programs=len(lineage),
        num_generations=max((e['generation'] for e in lineage), default=0) + 1
    )

    # Group by generation
    by_generation = {}
    for entry in lineage:
        gen = entry['generation']
        if gen not in by_generation:
            by_generation[gen] = []
        by_generation[gen].append(entry)

    # Display from newest to oldest generation
    for gen in sorted(by_generation.keys(), reverse=True):
        html += '    <div class="generation-group">\n'
        html += f'        <div class="gen-label">Generation {gen}</div>\n'
        html += '        <div style="display: flex; flex-wrap: wrap; align-items: center;">\n'

        for entry in by_generation[gen]:
            is_current = (entry['program_id'] == program_id)
            is_baseline = (entry['generation'] == 0)
            node_class = "current" if is_current else ("baseline" if is_baseline else "")

            parent_str = ", ".join(str(p) for p in entry['parent_ids']) if entry['parent_ids'] else "None"

            html += f"""            <div class="node {node_class}">
                <div class="node-id">ID: {entry['program_id']}</div>
                <div class="node-gen">Gen: {entry['generation']}</div>
                <div class="node-score">Score: {entry['score']:.2f}</div>
                <div style="font-size: 11px; color: #888;">Parents: {parent_str}</div>
            </div>\n"""

        html += '        </div>\n'
        html += '    </div>\n'

    html += """
    <div style="margin-top: 30px; padding: 15px; background: white; border-radius: 8px;">
        <h3>Legend</h3>
        <p><span class="node current" style="display: inline-block; margin: 5px;">Current Program</span> - The program you're viewing</p>
        <p><span class="node baseline" style="display: inline-block; margin: 5px;">Baseline</span> - Initial programs (Generation 0)</p>
        <p><span class="node" style="display: inline-block; margin: 5px;">Ancestor</span> - Programs in the evolutionary chain</p>
    </div>
</body>
</html>
"""
    return html


def log_top_programs_table(database):
    """Log a W&B table with the best program from each island and their lineage.

    Args:
        database: ProgramsDatabase instance
    """
    if not database.wandb_enabled or not WANDB_AVAILABLE:
        return

    table_data = []

    for island_id in range(len(database._islands)):
        program = database._best_program_per_island[island_id]
        if program is None or program.program_id is None:
            continue

        score = database._best_score_per_island[island_id]
        scores_per_test = database._best_scores_per_test_per_island[island_id]

        # Get lineage (only if enabled)
        lineage = trace_lineage(database, program.program_id) if database.save_lineage else []

        # Format detailed scores
        scores_str = ", ".join(f"{k}:{v}" for k, v in scores_per_test.items()) if scores_per_test else "N/A"

        # Full code (not truncated)
        full_code = str(program)

        # Generate and save HTML lineage (only if enabled)
        if database.save_lineage:
            try:
                os.makedirs(database.save_checkpoints_path, exist_ok=True)

                # Generate detailed lineage HTML with code
                html_content = generate_lineage_html(database, program.program_id, island_id)
                html_filename = f"lineage_detailed_island{island_id}_program{program.program_id}.html"
                html_path = f"{database.save_checkpoints_path}/{html_filename}"
                with open(html_path, 'w') as f:
                    f.write(html_content)

                # Generate tree diagram HTML (structure only, no code)
                tree_content = generate_lineage_tree_diagram(database, program.program_id, island_id)
                tree_filename = f"lineage_tree_island{island_id}_program{program.program_id}.html"
                tree_path = f"{database.save_checkpoints_path}/{tree_filename}"
                with open(tree_path, 'w') as f:
                    f.write(tree_content)

                # Log both as W&B artifacts (only if W&B is enabled and initialized)
                if database.wandb_enabled and database._wandb_initialized:
                    try:
                        artifact = wandb.Artifact(
                            name=f"lineage_island{island_id}_step{database.iterations}",
                            type="lineage_visualization",
                            description=f"Evolutionary lineage for island {island_id}, program {program.program_id}"
                        )
                        artifact.add_file(html_path, name="detailed_with_code.html")
                        artifact.add_file(tree_path, name="tree_diagram.html")
                        wandb.log_artifact(artifact)
                        lineage_link = f"See artifact: lineage_island{island_id}_step{database.iterations}"
                    except Exception as e:
                        logger.warning(f"Failed to upload lineage artifact to W&B: {e}")
                        lineage_link = f"Local files: {html_filename}, {tree_filename}"
                else:
                    lineage_link = f"Local files: {html_filename}, {tree_filename}"
            except Exception as e:
                logger.error(f"Error generating lineage HTML: {e}")
                lineage_link = "Error generating lineage"
        else:
            lineage_link = "Disabled"

        table_data.append([
            island_id,
            program.program_id,
            program.generation,
            score,
            scores_str,
            full_code,
            len(lineage),
            lineage_link
        ])

    if table_data:
        table = wandb.Table(
            columns=["Island", "Program ID", "Generation", "Score", "Detailed Scores", "Full Code", "Lineage Depth", "Lineage Visualization"],
            data=table_data
        )
        wandb.log({"top_programs": table})
        logger.info(f"Logged top programs table with {len(table_data)} entries")


async def initialize_wandb(database):
    """Initialize W&B asynchronously (called once on first logging attempt).

    Args:
        database: ProgramsDatabase instance with wandb_config
    """
    if database._wandb_initialized:
        return

    if not database.wandb_config or not database.wandb_config.enabled:
        return

    if not WANDB_AVAILABLE:
        logger.warning("W&B logging enabled but wandb not installed. Run: pip install wandb")
        return

    try:
        # Run wandb.init in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()

        # Check if we're resuming from a checkpoint with an existing run ID
        if database.wandb_run_id:
            # Try to resume existing run with strict mode
            expected_run_id = database.wandb_run_id
            logger.info(f"Attempting to resume W&B run with ID: {expected_run_id}")
            logger.info(f"W&B project: {database.wandb_config.project}, entity: {database.wandb_config.entity}")

            resume_failed = False

            try:
                await loop.run_in_executor(
                    None,
                    lambda: wandb.init(
                        project=database.wandb_config.project,
                        entity=database.wandb_config.entity,
                        id=expected_run_id,
                        resume="must",  # Fail if run doesn't exist or can't be resumed
                        tags=database.wandb_config.tags,
                        config=database.wandb_init_config,
                        settings=wandb.Settings(
                            console='off',  # Don't capture console output
                            _disable_stats=False,
                            _disable_meta=False,
                        )
                    )
                )

                # Verify that we actually resumed the expected run
                if wandb.run and wandb.run.id == expected_run_id:
                    logger.info(f"Successfully resumed W&B run: {expected_run_id}")
                    logger.info(f"W&B run URL: {wandb.run.url}")
                else:
                    # This shouldn't happen with resume="must", but check anyway
                    logger.warning(f"Unexpected: W&B run ID mismatch. Expected {expected_run_id}, got {wandb.run.id if wandb.run else 'None'}")

            except Exception as e:
                resume_failed = True
                logger.error(f"Failed to resume W&B run {expected_run_id}: {type(e).__name__}: {e}")

                # Provide specific guidance based on error type
                error_msg = str(e).lower()
                if "not found" in error_msg or "does not exist" in error_msg:
                    logger.error(f"Reason: Run {expected_run_id} does not exist in project '{database.wandb_config.project}'")
                    logger.error("Possible causes: Run was deleted, wrong project/entity, or run ID is incorrect")
                elif "finished" in error_msg or "completed" in error_msg:
                    logger.error(f"Reason: Run {expected_run_id} is already marked as finished/completed")
                    logger.error("Suggestion: Check W&B dashboard to verify run status")
                elif "permission" in error_msg or "access" in error_msg:
                    logger.error(f"Reason: No permission to access run {expected_run_id}")
                    logger.error("Suggestion: Verify entity/project permissions and API key")
                else:
                    logger.error(f"Reason: Unknown error, {e}")

                # Close any partial W&B connection
                if wandb.run:
                    wandb.finish()

                logger.info("Creating a new W&B run instead...")

            # If resume failed, create a new run
            if resume_failed:
                await loop.run_in_executor(
                    None,
                    lambda: wandb.init(
                        project=database.wandb_config.project,
                        entity=database.wandb_config.entity,
                        name=database.wandb_run_name,
                        tags=database.wandb_config.tags,
                        config=database.wandb_init_config,
                        settings=wandb.Settings(
                            console='off',
                            _disable_stats=False,
                            _disable_meta=False,
                        )
                    )
                )
                if wandb.run:
                    logger.info(f"Created new W&B run: {wandb.run.id}")
                    logger.info(f"New run URL: {wandb.run.url}")
                    logger.warning(f"Note: This is a new run, not a resumption of {expected_run_id}")

        else:
            # No checkpoint run ID, start fresh run
            logger.info("No previous W&B run ID found. Creating a new run...")
            await loop.run_in_executor(
                None,
                lambda: wandb.init(
                    project=database.wandb_config.project,
                    entity=database.wandb_config.entity,
                    name=database.wandb_run_name,
                    tags=database.wandb_config.tags,
                    config=database.wandb_init_config,
                    settings=wandb.Settings(
                        console='off',  # Don't capture console output
                        _disable_stats=False,
                        _disable_meta=False,
                    )
                )
            )
            if wandb.run:
                logger.info(f"Created new W&B run: {wandb.run.id}")
                logger.info(f"New run URL: {wandb.run.url}")

        database.wandb_enabled = True
        database.wandb_log_interval = database.wandb_config.log_interval
        database._wandb_initialized = True

        # Store the run ID (either new or resumed)
        if wandb.run:
            database.wandb_run_id = wandb.run.id
            logger.info(f"W&B logging initialized: {wandb.run.url}")
        else:
            logger.warning("W&B run object is None after initialization")
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        database.wandb_enabled = False


async def periodic_wandb_logging(database, rabbitmq_manager=None):
    """Periodically log metrics to Weights & Biases.

    Args:
        database: ProgramsDatabase instance
        rabbitmq_manager: Optional RabbitMQManager for queue metrics
    """
    # Initialize W&B asynchronously on first run
    await initialize_wandb(database)

    if not database.wandb_enabled:
        return

    try:
        while True:
            await asyncio.sleep(database.wandb_log_interval)
            try:
                # Get database metrics (sync)
                metrics = compute_wandb_metrics(database)

                # Get RabbitMQ metrics (async)
                rmq_metrics = await compute_rabbitmq_metrics(rabbitmq_manager)
                metrics.update(rmq_metrics)

                wandb.log(metrics)
                logger.debug(f"Logged {len(metrics)} metrics to W&B")

                # Log top programs table with lineage
                log_top_programs_table(database)
            except Exception as e:
                logger.error(f"Error logging to W&B: {e}")
    except asyncio.CancelledError:
        logger.info("W&B logging task cancelled. Leaving run unfinished to allow resumption from checkpoint.")
        # Intentionally skip wandb.finish() here, leave the run "running" so it can be resumed
        # If the run is truly complete, the user should manually finish it in W&B UI
        # or call wandb.finish() explicitly when termination conditions are met
        if database.wandb_enabled and wandb.run is not None:
            logger.info(f"W&B run {wandb.run.id} left in resumable state. Resume with: --checkpoint <path>")
        raise


def finish_wandb_run(database):
    """Explicitly finish the W&B run when the experiment is truly complete.

    Call this manually when you know the run should be marked as finished (not resumable).

    Args:
        database: ProgramsDatabase instance
    """
    if database.wandb_enabled and wandb.run is not None:
        try:
            logger.info(f"Finishing W&B run {wandb.run.id}...")
            wandb.finish()
            logger.info("W&B run finished successfully")
        except Exception as e:
            logger.error(f"Error finishing W&B run: {e}")
