"""Throughput sweep for parameter optimization.

Runs throughput measurements across a grid of parameters to find optimal configuration.
All settings are configured in config.py.

Usage:
    python -m disfun.throughput_sweep --config_path ./config.py

Configuration (in config.py):
    - num_samplers, num_evaluators: Worker counts
    - ThroughputConfig: warmup_minutes, run_duration_minutes, window_duration_minutes, cooldown_seconds
    - SweepConfig: prompts_per_batch, evaluator_prefetch, max_workers, sampler_prefetch_multiplier (parameter grids)
    - PathsConfig: log_dir, sandbox_base_path

Examples:
    # Run sweep with settings from config.py
    python -m disfun.throughput_sweep --config_path ./config.py

    # Resume from config 20 after interruption
    python -m disfun.throughput_sweep --config_path ./config.py --start_idx 20
"""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import datetime
from itertools import product


def create_sweep_config(config):
    """Get parameter grid from config.sweep."""
    sweep_params = {
        "prompts_per_batch": list(config.sweep.prompts_per_batch),
        "evaluator_prefetch": list(config.sweep.evaluator_prefetch),
        "max_workers": list(config.sweep.max_workers),
    }
    # Add sampler_prefetch_multiplier if present in sweep config
    if hasattr(config.sweep, 'sampler_prefetch_multiplier'):
        sweep_params["sampler_prefetch_multiplier"] = list(config.sweep.sampler_prefetch_multiplier)
    return sweep_params


def generate_sweep_combinations(sweep_config):
    """Generate all combinations of sweep parameters."""
    keys = list(sweep_config.keys())
    values = [sweep_config[k] for k in keys]

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


async def run_single_config(base_config, sweep_params, config_idx, total_configs, log_dir, sandbox_base_path, checkpoint_path=None):
    """Run throughput measurement for a single parameter configuration."""
    from disfun.throughput import run_throughput
    from disfun import code_manipulation

    # Create modified config with sweep parameters
    sampler_kwargs = {"prompts_per_batch": sweep_params["prompts_per_batch"]}
    if "sampler_prefetch_multiplier" in sweep_params:
        sampler_kwargs["prefetch_multiplier"] = sweep_params["sampler_prefetch_multiplier"]
    modified_sampler = dataclasses.replace(base_config.sampler, **sampler_kwargs)

    modified_evaluator = dataclasses.replace(
        base_config.evaluator,
        prefetch_count=sweep_params["evaluator_prefetch"],
        max_workers=sweep_params["max_workers"]
    )

    # Update W&B run name to include sweep params
    sweep_tag = f"pb{sweep_params['prompts_per_batch']}_pf{sweep_params['evaluator_prefetch']}_mw{sweep_params['max_workers']}"
    if "sampler_prefetch_multiplier" in sweep_params:
        sweep_tag += f"_spm{sweep_params['sampler_prefetch_multiplier']}"
    modified_wandb = dataclasses.replace(
        base_config.wandb,
        run_name=f"sweep_{sweep_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_name_tag=f"sweep_{sweep_tag}"
    )

    # Create the modified config
    config = dataclasses.replace(
        base_config,
        sampler=modified_sampler,
        evaluator=modified_evaluator,
        wandb=modified_wandb
    )

    print(f"\n{'='*70}")
    print(f"Configuration {config_idx + 1}/{total_configs}")
    print(f"  prompts_per_batch: {sweep_params['prompts_per_batch']}")
    print(f"  evaluator_prefetch: {sweep_params['evaluator_prefetch']}")
    print(f"  max_workers: {sweep_params['max_workers']}")
    if "sampler_prefetch_multiplier" in sweep_params:
        print(f"  sampler_prefetch_multiplier: {sweep_params['sampler_prefetch_multiplier']}")
    print(f"{'='*70}\n")

    # Load specification
    spec_path = config.evaluator.evaluation_script_path
    with open(spec_path, 'r') as f:
        specification = f.read()

    # Build inputs from config
    inputs = []
    for s_idx, s in enumerate(config.evaluator.s_values):
        start = config.evaluator.start_n[s_idx] if s_idx < len(config.evaluator.start_n) else config.evaluator.start_n[0]
        end = config.evaluator.end_n[s_idx] if s_idx < len(config.evaluator.end_n) else config.evaluator.end_n[0]
        q = config.evaluator.q
        for n in range(start, end + 1):
            inputs.append((n, s, q))

    # Target signatures (optional)
    target_signatures = config.termination.target_solutions if config.termination.target_solutions else None

    # Create config-specific log dir
    config_log_dir = os.path.join(log_dir, f"sweep_{sweep_tag}")
    os.makedirs(config_log_dir, exist_ok=True)

    # Save config to temp file for process spawning
    # Use cloudpickle to handle dynamically loaded classes
    config_path = os.path.join(config_log_dir, "config.pkl")
    import cloudpickle
    with open(config_path, 'wb') as f:
        cloudpickle.dump(config, f)

    try:
        results = await run_throughput(
            config=config,
            config_path=config_path,
            log_dir=config_log_dir,
            sandbox_base_path=sandbox_base_path,
            specification=specification,
            inputs=inputs,
            target_signatures=target_signatures,
            checkpoint_path=checkpoint_path
        )

        # Add sweep params to results
        results["sweep_params"] = sweep_params
        results["config_idx"] = config_idx

        return results

    except Exception as e:
        print(f"ERROR in config {config_idx + 1}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "sweep_params": sweep_params,
            "config_idx": config_idx,
            "error": str(e),
            "iterations_per_hour_mean": 0,
            "iterations_per_hour_std": 0,
        }


async def run_sweep(config_path: str, start_idx: int = 0, checkpoint_path: str = None):
    """Run the full parameter sweep."""
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(config_path)))

    # Load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    base_config = config_module.config

    # Enable throughput mode
    modified_throughput = dataclasses.replace(base_config.throughput, enabled=True)
    base_config = dataclasses.replace(base_config, throughput=modified_throughput)

    # Get timing and sweep params from config
    tc = base_config.throughput
    sweep_config = create_sweep_config(base_config)

    combinations = generate_sweep_combinations(sweep_config)
    total_configs = len(combinations)

    print(f"\n{'#'*70}")
    print(f"THROUGHPUT SWEEP")
    print(f"{'#'*70}")
    print(f"Ratio: {base_config.num_samplers} samplers : {base_config.num_evaluators} evaluators")
    print(f"Total configurations: {total_configs}")
    print(f"Starting from config: {start_idx + 1}")
    print(f"Checkpoint: {checkpoint_path if checkpoint_path else 'None (fresh start)'}")
    print(f"Duration per config: {tc.warmup_minutes}min warmup + {tc.run_duration_minutes}min measurement + {tc.cooldown_seconds}s cooldown")
    print(f"Estimated total time: {(total_configs - start_idx) * (tc.warmup_minutes + tc.run_duration_minutes + tc.cooldown_seconds/60) / 60:.1f} hours")
    print(f"Queue state check: enabled (will clean dirty queues before each config)")
    print(f"\nSweep parameters:")
    for key, values in sweep_config.items():
        print(f"  {key}: {values}")
    print(f"{'#'*70}\n")

    # Setup directories
    log_dir = base_config.paths.log_dir
    sandbox_base_path = base_config.paths.sandbox_base_path
    os.makedirs(log_dir, exist_ok=True)

    # Results file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(log_dir, f"sweep_results_{base_config.num_samplers}s_{base_config.num_evaluators}e_{timestamp}.json")

    all_results = []

    for idx, sweep_params in enumerate(combinations[start_idx:], start=start_idx):
        try:
            result = await run_single_config(
                base_config=base_config,
                sweep_params=sweep_params,
                config_idx=idx,
                total_configs=total_configs,
                log_dir=log_dir,
                sandbox_base_path=sandbox_base_path,
                checkpoint_path=checkpoint_path
            )
            all_results.append(result)

            # Save intermediate results after each config
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            throughput = result.get('iterations_per_hour_mean', 0)
            std = result.get('iterations_per_hour_std', 0)
            print(f"\nConfig {idx + 1} result: {throughput:.0f} +/- {std:.0f} iter/hour")

            # Cooldown between configs
            if idx < total_configs - 1:
                print(f"Cooldown: {tc.cooldown_seconds}s before next config...")
                await asyncio.sleep(tc.cooldown_seconds)

        except KeyboardInterrupt:
            print(f"\n\nSweep interrupted at config {idx + 1}. Results saved to {results_file}")
            break
        except Exception as e:
            print(f"Error in config {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "sweep_params": sweep_params,
                "config_idx": idx,
                "error": str(e)
            })

    # Final summary
    print(f"\n{'#'*70}")
    print("SWEEP COMPLETE")
    print(f"{'#'*70}")
    print(f"Results saved to: {results_file}")

    # Sort by throughput and show top configs
    valid_results = [r for r in all_results if "error" not in r and r.get("iterations_per_hour_mean", 0) > 0]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: x["iterations_per_hour_mean"], reverse=True)
        print(f"\nTop 5 configurations:")
        for i, r in enumerate(sorted_results[:5]):
            p = r["sweep_params"]
            mean = r['iterations_per_hour_mean']
            std = r['iterations_per_hour_std']
            spm_str = f", spm={p['sampler_prefetch_multiplier']}" if 'sampler_prefetch_multiplier' in p else ""
            print(f"  {i+1}. {mean:.0f} +/- {std:.0f} iter/hr | "
                  f"pb={p['prompts_per_batch']}, pf={p['evaluator_prefetch']}, mw={p['max_workers']}{spm_str}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run throughput parameter sweep. All settings come from config.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sweep with settings from config.py
  python -m disfun.throughput_sweep --config_path ./config.py

  # Resume from config 20 after interruption
  python -m disfun.throughput_sweep --config_path ./config.py --start_idx 20

Configuration (in config.py):
  - num_samplers, num_evaluators: Worker counts
  - ThroughputConfig: warmup_minutes, run_duration_minutes, window_duration_minutes, cooldown_seconds
  - SweepConfig: prompts_per_batch, evaluator_prefetch, max_workers, sampler_prefetch_multiplier (parameter grids)
  - PathsConfig: log_dir, sandbox_base_path
        """
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to config.py")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start from this config index (0-based, for resuming)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file to load. Starts from steady-state instead of empty database.")

    args = parser.parse_args()

    asyncio.run(run_sweep(args.config_path, args.start_idx, args.checkpoint))


if __name__ == "__main__":
    main()
