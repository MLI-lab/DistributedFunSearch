#!/usr/bin/env python3
"""
KaMIS baseline wrapper for maximum independent set computation.

Runs online_mis and/or redumis algorithms on metis-format graphs.
Supports multiple runs with different seeds and configurable timeouts.

Requires metis format graphs. Convert lmdb graphs first:
    python convert_lmdb_to_metis.py --n-values 6,7,8 --q 4 --graph-type ids


Usage Examples:
    # Binary deletion codes with online_mis (1 minute timeout)
    python kamis_baseline.py --n-values 6,7,8,9,10 --timeout 60 --algorithm online_mis


Arguments:
    --n-values        Comma-separated n values (required)
    --s               Number of errors to correct (default: 1)
    --q               Alphabet size: 2=binary, 4=quaternary (default: 2)
    --graph-type      Graph type: 'deletions' or 'ids' (default: deletions)
    --graph-dir       Directory with metis graphs (default: /mnt/Graphs)
    --algorithm       Algorithm: 'online_mis', 'redumis', or 'both' (default: online_mis)
    --timeout         Timeout per run in seconds (default: 60, 86400 for 24h)
    --runs            Number of runs per algorithm (default: 3)
    --seeds           Comma-separated seeds for each run (default: 0,100000,200000)
    --redumis-config  ReduMIS config: 'standard' or 'social' (default: standard)
    --output, -o      Output directory for results (default: ./kamis_results)
    --log             Log file path (default: <output>/kamis.log)

"""

import argparse
import os
import sys
import json
import subprocess
import signal
import time
import logging
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.graph_paths import DEFAULT_GRAPH_DIR, get_metis_path as get_metis_path_central

KAMIS_DIR = Path(__file__).parent / "KaMIS"
DEPLOY_DIR = KAMIS_DIR / "deploy"
ONLINE_MIS_BIN = DEPLOY_DIR / "online_mis"
REDUMIS_BIN = DEPLOY_DIR / "redumis"
RESULTS_BASE_DIR = Path(__file__).parent / "results"


def setup_logging(log_file: str):
    """Set up logging to both console and file with timestamps."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console: simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # File: with timestamps
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log(msg: str):
    """Print to both console and log file."""
    logging.info(msg)


def generate_output_dir(s: int, q: int, graph_type: str, algorithm: str) -> str:
    """Generate output directory name with parameters and timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    algo_str = algorithm if algorithm != 'both' else 'both'
    dir_name = f"kamis_s{s}_q{q}_{graph_type}_{algo_str}_{timestamp}"
    return str(RESULTS_BASE_DIR / dir_name)


def check_kamis_compiled():
    """Check if KaMIS binaries are compiled. Exit with instructions if not."""
    if not DEPLOY_DIR.exists():
        log("\n" + "="*60)
        log("ERROR: KaMIS binaries not found!")
        log("="*60)
        log("\nPlease compile KaMIS first:")
        log(f"  cd {KAMIS_DIR}")
        log("  ./compile_withcmake.sh")
        log("\nThis will create the deploy/ directory with binaries.")
        log("="*60 + "\n")
        sys.exit(1)

    if not ONLINE_MIS_BIN.exists():
        log(f"\n[ERROR] online_mis binary not found at: {ONLINE_MIS_BIN}")
        sys.exit(1)

    if not REDUMIS_BIN.exists():
        log(f"\n[ERROR] redumis binary not found at: {REDUMIS_BIN}")
        sys.exit(1)

    log(f"[INFO] KaMIS binaries found at: {DEPLOY_DIR}")


def load_node_mapping(mapping_file: str) -> Dict:
    """Load node ID mapping from JSON file."""
    with open(mapping_file, "r") as f:
        return json.load(f)


def get_solution_size(output_file: str) -> int:
    """
    Get the size of a KaMIS solution (fast, no mapping needed).

    Args:
        output_file: Path to KaMIS output file (one 0/1 per line)

    Returns:
        Number of nodes in the independent set
    """
    if not os.path.exists(output_file):
        return 0

    count = 0
    with open(output_file, "r") as f:
        for line in f:
            if line.strip() == "1":
                count += 1
    return count


def parse_kamis_output(output_file: str, mapping_file: Optional[str] = None) -> Tuple[List, int]:
    """
    Parse KaMIS output file and return independent set.

    Args:
        output_file: Path to KaMIS output file (one 0/1 per line)
        mapping_file: Optional path to mapping file (to get original node IDs)

    Returns:
        (independent_set_nodes, solution_size)
    """
    if not os.path.exists(output_file):
        return [], 0

    # Fast path: if we don't need the actual nodes, just count
    # (This is the common case when we only care about size)

    with open(output_file, "r") as f:
        lines = f.readlines()

    # Parse: 1 means node is in set, 0 means not
    independent_set_int = []
    for node_id, line in enumerate(lines, start=1):
        if line.strip() == "1":
            independent_set_int.append(node_id)

    # Convert back to original string IDs if mapping provided
    if mapping_file and os.path.exists(mapping_file):
        mapping = load_node_mapping(mapping_file)
        if "int_to_str" in mapping:
            int_to_str = mapping["int_to_str"]
            independent_set_str = [int_to_str[str(node_id)] for node_id in independent_set_int]
            return independent_set_str, len(independent_set_str)

    return independent_set_int, len(independent_set_int)


def run_kamis_algorithm(
    algorithm: str,
    metis_file: str,
    output_file: str,
    timeout: float,
    seed: int,
    config: Optional[str] = None
) -> Tuple[int, float, bool]:
    """
    Run a KaMIS algorithm with timeout.

    Args:
        algorithm: "online_mis" or "redumis"
        metis_file: Path to METIS graph file
        output_file: Path to save solution
        timeout: Time limit in seconds
        seed: Random seed
        config: Config for redumis ("standard" or "social")

    Returns:
        (solution_size, runtime, success)
    """
    # Build command
    if algorithm == "online_mis":
        cmd = [
            str(ONLINE_MIS_BIN),
            metis_file,
            f"--time_limit={timeout}",
            f"--seed={seed}",
            f"--output={output_file}",
            "--console_log"
        ]
    elif algorithm == "redumis":
        config = config or "standard"
        cmd = [
            str(REDUMIS_BIN),
            metis_file,
            f"--config={config}",
            f"--time_limit={timeout}",
            f"--seed={seed}",
            f"--output={output_file}",
            "--console_log"
        ]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Run command with Popen for better timeout control
    log(f"    [DEBUG] Command: {' '.join(str(c) for c in cmd)}")
    log(f"    [DEBUG] Graph file exists: {os.path.exists(metis_file)}")
    start_time = time.time()
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )

        # KaMIS needs extra time to write output after hitting limit
        subprocess_timeout = timeout + 600

        try:
            stdout, stderr = process.communicate(timeout=subprocess_timeout)
            runtime = time.time() - start_time
            success = process.returncode == 0

            if process.returncode != 0:
                log(f"    [DEBUG] Return code: {process.returncode}")
                # Interpret signal-based crashes
                if process.returncode < 0:
                    sig_num = -process.returncode
                    sig_names = {
                        4: "SIGILL (illegal instruction - CPU incompatibility?)",
                        6: "SIGABRT (abort - assertion failed?)",
                        7: "SIGBUS (bus error - memory alignment?)",
                        8: "SIGFPE (floating point exception)",
                        9: "SIGKILL (killed - OOM killer?)",
                        11: "SIGSEGV (segmentation fault - memory access error)",
                        15: "SIGTERM (terminated)",
                    }
                    sig_name = sig_names.get(sig_num, f"signal {sig_num}")
                    log(f"    [DEBUG] Crash signal: {sig_name}")
            if stderr and stderr.strip():
                log(f"    [DEBUG] stderr: {stderr.strip()[:1000]}")
            if not success and stdout and stdout.strip():
                log(f"    [DEBUG] stdout (last 2000 chars): ...{stdout.strip()[-2000:]}")

            # Parse solution
            if os.path.exists(output_file):
                solution_size = get_solution_size(output_file)
                if solution_size > 0:
                    success = True
            else:
                solution_size = 0
                success = False

            return solution_size, runtime, success

        except subprocess.TimeoutExpired:
            runtime = time.time() - start_time
            log(f"    [WARNING] Process exceeded subprocess timeout ({runtime:.1f}s)")
            log(f"    [INFO] Sending SIGTERM for graceful cleanup")

            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            # Wait for termination
            grace_period = 120
            for i in range(grace_period):
                time.sleep(1)
                if process.poll() is not None:
                    log(f"    [INFO] Process terminated after {i+1}s")
                    break
            else:
                log(f"    [WARNING] Process didn't terminate gracefully, sending SIGKILL")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

            runtime = time.time() - start_time

            if os.path.exists(output_file):
                solution_size = get_solution_size(output_file)
                if solution_size > 0:
                    log(f"    [INFO] Found solution of size {solution_size} despite timeout")
                    return solution_size, runtime, True

            return 0, runtime, False

    except Exception as e:
        runtime = time.time() - start_time
        log(f"    [ERROR] {e}")
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, AttributeError):
                pass
        return 0, runtime, False


def get_metis_path(graph_dir: str, graph_type: str, s: int, n: int, q: int) -> str:
    """Get METIS file path using centralized graph_paths module."""
    return get_metis_path_central(graph_type, s, n, q, base_dir=graph_dir)


def save_results_json(results: List[Dict], output_dir: str, s: int, q: int, graph_type: str):
    """Save results to JSON file (called incrementally)."""
    results_file = os.path.join(output_dir, "kamis_results.json")
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "s": s,
            "q": q,
            "graph_type": graph_type,
        },
        "results": results,
    }
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    return results_file


def run_kamis_baseline(
    n_values: List[int],
    s: int = 1,
    q: int = 2,
    graph_type: str = "deletions",
    graph_dir: str = DEFAULT_GRAPH_DIR,
    algorithms: List[str] = ["online_mis"],
    timeout: float = 60.0,
    num_runs: int = 3,
    base_seeds: List[int] = [0, 100000, 200000],
    redumis_config: str = "standard",
    output_dir: str = "./kamis_results"
) -> List[Dict]:
    """
    Run KaMIS baselines on multiple graphs.

    Args:
        n_values: List of code lengths to evaluate
        s, q: Problem parameters
        graph_type: "deletions" or "ids"
        graph_dir: Directory with METIS graph files
        algorithms: List of algorithms to run ["online_mis", "redumis"]
        timeout: Time limit per run in seconds
        num_runs: Number of independent runs per algorithm
        base_seeds: Seeds for each run
        redumis_config: Config for redumis ("standard" or "social")
        output_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    check_kamis_compiled()

    if len(base_seeds) != num_runs:
        raise ValueError(f"base_seeds must have {num_runs} elements, got {len(base_seeds)}")

    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for each algorithm
    for algo in algorithms:
        os.makedirs(os.path.join(output_dir, algo), exist_ok=True)

    all_results = []
    start_time_total = datetime.now()

    # Calculate total work for progress tracking
    total_tasks = len(n_values) * len(algorithms)
    current_task = 0

    log("\n" + "="*70)
    log("  KaMIS Baseline Evaluation")
    log("="*70)
    log(f"  Started: {start_time_total.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  n values: {n_values}")
    log(f"  Parameters: s={s}, q={q}, type={graph_type}")
    log(f"  Algorithms: {', '.join(algorithms)}")
    log(f"  Runs per (n, algorithm): {num_runs}")
    log(f"  Timeout per run: {timeout}s ({timeout/3600:.1f}h)")
    log(f"  Seeds: {base_seeds}")
    log(f"  Output directory: {output_dir}")
    log(f"  Total tasks: {total_tasks} (n × algorithm pairs)")
    log("="*70 + "\n")

    for algorithm in algorithms:
        algo_dir = os.path.join(output_dir, algorithm)

        log(f"\n{'='*70}")
        log(f"  ALGORITHM: {algorithm.upper()}")
        log(f"  Config: {redumis_config if algorithm == 'redumis' else 'N/A'}")
        log("="*70)

        for n in n_values:
            current_task += 1
            metis_file = get_metis_path(graph_dir, graph_type, s, n, q)

            if not os.path.exists(metis_file):
                log(f"\n[{current_task}/{total_tasks}] n={n} | {algorithm} | SKIPPED (METIS not found)")
                log(f"    File: {metis_file}")
                log(f"    Run: python convert_lmdb_to_metis.py --n-values {n} --q {q} --graph-type {graph_type}")
                continue

            graph_name = os.path.basename(metis_file).replace(".metis", "")

            log(f"\n[{current_task}/{total_tasks}] n={n} | {algorithm}")
            log(f"    Graph: {graph_name}")

            sizes_per_run = []
            runtimes_per_run = []

            for run_idx, seed in enumerate(base_seeds):
                log(f"    [RUN {run_idx+1}/{num_runs}] Seed: {seed}", )

                # Save result files in algorithm subdirectory
                result_file = os.path.join(
                    algo_dir,
                    f"n{n}_seed{seed}.result"
                )

                solution_size, runtime, success = run_kamis_algorithm(
                    algorithm=algorithm,
                    metis_file=metis_file,
                    output_file=result_file,
                    timeout=timeout,
                    seed=seed,
                    config=redumis_config if algorithm == "redumis" else None
                )

                sizes_per_run.append(solution_size)
                runtimes_per_run.append(runtime)

                status = "OK" if success else "FAIL"
                log(f"      [{status}] Size: {solution_size}, Time: {runtime:.1f}s")

            # Statistics
            best_size = max(sizes_per_run) if sizes_per_run else 0
            mean_size = mean(sizes_per_run) if sizes_per_run else 0
            std_size = stdev(sizes_per_run) if len(sizes_per_run) > 1 else 0
            mean_runtime = mean(runtimes_per_run) if runtimes_per_run else 0
            best_count = sizes_per_run.count(best_size) if best_size > 0 else 0

            log(f"    [SUMMARY] Best: {best_size}, Mean: {mean_size:.1f}±{std_size:.1f}, "
                f"Best found: {best_count}/{num_runs}, Avg time: {mean_runtime:.1f}s")

            result_entry = {
                "n": n,
                "s": s,
                "q": q,
                "graph": graph_name,
                "algorithm": algorithm,
                "config": redumis_config if algorithm == "redumis" else "N/A",
                "timeout": timeout,
                "num_runs": num_runs,
                "best": best_size,
                "mean": round(mean_size, 2),
                "std": round(std_size, 2),
                "sizes": sizes_per_run,
                "runtimes": [round(r, 2) for r in runtimes_per_run],
                "mean_runtime": round(mean_runtime, 2),
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result_entry)

            # Save results incrementally after each (algorithm, n) pair
            results_file = save_results_json(all_results, output_dir, s, q, graph_type)
            log(f"    [SAVED] Results updated: {results_file}")

    # Final summary
    end_time_total = datetime.now()
    elapsed = end_time_total - start_time_total

    log("\n" + "="*70)
    log("  FINAL SUMMARY")
    log("="*70)
    log(f"  Completed: {end_time_total.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total elapsed: {elapsed}")
    log("")
    log(f"  {'n':>4} | {'Algorithm':>12} | {'Best':>6} | {'Mean±Std':>12} | {'Time':>8}")
    log(f"  {'-'*4}-+-{'-'*12}-+-{'-'*6}-+-{'-'*12}-+-{'-'*8}")
    for result in all_results:
        log(f"  {result['n']:>4} | {result['algorithm']:>12} | {result['best']:>6} | "
            f"{result['mean']:>5.1f}±{result['std']:<5.1f} | {result['mean_runtime']:>6.1f}s")
    log("="*70)
    log(f"\n  Results saved to: {output_dir}")
    log(f"  - kamis_results.json (all results)")
    for algo in algorithms:
        log(f"  - {algo}/ (solution files)")
    log("="*70 + "\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run KaMIS baselines on conflict graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--n-values', type=str, required=True,
                        help='Comma-separated n values (e.g., "6,7,8,9,10")')
    parser.add_argument('--s', type=int, default=1,
                        help='Number of deletions to correct (default: 1)')
    parser.add_argument('--q', type=int, default=2,
                        help='Alphabet size (default: 2 for binary)')
    parser.add_argument('--graph-type', choices=['deletions', 'ids'], default='deletions',
                        help='Graph type (default: deletions)')
    parser.add_argument('--graph-dir', default=DEFAULT_GRAPH_DIR,
                        help=f'Directory with METIS graphs (default: {DEFAULT_GRAPH_DIR})')
    parser.add_argument('--algorithm', choices=['online_mis', 'redumis', 'both'],
                        default='online_mis',
                        help='Algorithm to run (default: online_mis)')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Timeout per run in seconds (default: 60)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per algorithm (default: 3)')
    parser.add_argument('--seeds', type=str, default='0,100000,200000',
                        help='Comma-separated seeds for each run')
    parser.add_argument('--redumis-config', choices=['standard', 'social'],
                        default='standard',
                        help='ReduMIS config (default: standard)')
    parser.add_argument('--output', '-o', default='auto',
                        help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--log', default=None,
                        help='Log file path (default: <output>/kamis.log)')

    args = parser.parse_args()

    # Parse arguments
    n_values = [int(x.strip()) for x in args.n_values.split(',')]
    seeds = [int(x.strip()) for x in args.seeds.split(',')]

    if args.algorithm == 'both':
        algorithms = ['online_mis', 'redumis']
    else:
        algorithms = [args.algorithm]

    if len(seeds) != args.runs:
        print(f"Warning: {len(seeds)} seeds provided for {args.runs} runs. Adjusting runs to match seeds.")
        args.runs = len(seeds)

    # Determine output directory
    if args.output == 'auto':
        output_dir = generate_output_dir(args.s, args.q, args.graph_type, args.algorithm)
    else:
        output_dir = args.output

    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    log_file = args.log or os.path.join(output_dir, "kamis.log")
    setup_logging(log_file)
    log(f"[INFO] Logging to: {log_file}\n")

    # Run baselines
    run_kamis_baseline(
        n_values=n_values,
        s=args.s,
        q=args.q,
        graph_type=args.graph_type,
        graph_dir=args.graph_dir,
        algorithms=algorithms,
        timeout=args.timeout,
        num_runs=args.runs,
        base_seeds=seeds,
        redumis_config=args.redumis_config,
        output_dir=output_dir
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
