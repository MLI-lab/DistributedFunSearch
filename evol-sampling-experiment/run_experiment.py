#!/usr/bin/env python3
# run_experiment.py
"""
Main entry point for the evolutionary sampling experiment.

Usage:
    # Step 1: Filter for hard problems (problems where model gets 0/64 correct)
    python run_experiment.py --mode filter --output data/hard_problems.json

    # Step 2: Run comparison with N=16
    python run_experiment.py --mode compare --problems data/hard_problems.json --output results/comparison_n16.json

    # Step 3: Run comparison with multiple N values
    python run_experiment.py --mode compare_multi --problems data/hard_problems.json --output results/comparison_multi.json

    # Step 4: Analyze results
    python run_experiment.py --mode analyze --results results/comparison_n16.json
"""

import argparse
import json
import os
import sys

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_NAME,
    DIFFICULTY_FILTER,
    MAX_PROBLEMS,
    FILTER_N_SAMPLES,
    FILTER_THRESHOLD,
    N_VALUES_TO_TEST,
    TEMPERATURE,
    MAX_TOKENS,
)
from data.loader import load_math_dataset, load_numina_dataset
from model.inference import load_model
from experiments.filter_hard_problems import find_hard_problems
from experiments.compare_methods import run_comparison, run_multi_n_comparison


def save_json(data: dict, path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {path}")


def load_json(path: str) -> dict:
    """Load data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def print_summary(results: dict):
    """Print a nice summary of results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    cfg = results.get("config", {})
    n_total = cfg.get("n_total", "?")
    n_problems = cfg.get("n_problems", len(results.get("detailed_results", [])))

    print(f"\nN = {n_total} samples per method")
    print(f"Problems tested: {n_problems}")

    print(f"\n--- Success Rates ---")
    print(f"IID sampling:   {results['iid_solved']} solved ({results['iid_solve_rate']:.1%})")
    print(f"Evol sampling:  {results['evol_solved']} solved ({results['evol_solve_rate']:.1%})")

    print(f"\n--- Key Comparison ---")
    print(f"Solved by EVOL only (not IID): {results['evol_only_solved']}  <- KEY METRIC")
    print(f"Solved by IID only (not EVOL): {results['iid_only_solved']}")
    print(f"Solved by both:                {results['both_solved']}")
    print(f"Solved by neither:             {results['neither_solved']}")

    print(f"\n--- Token Usage (Compute Fairness) ---")
    print(f"IID total tokens:  {results['total_iid_tokens']:,}")
    print(f"Evol total tokens: {results['total_evol_tokens']:,}")
    print(f"Evol overhead:     {results['token_overhead_ratio']:.2f}x")

    print(f"\n--- Where did Evol successes come from? ---")
    print(f"From seeds (gen 0):        {results['evol_successes_from_seeds']}")
    print(f"From refinements (gen 1+): {results['evol_successes_from_refinements']}")
    if results.get("successes_by_generation"):
        print(f"By generation: {results['successes_by_generation']}")

    print("\n" + "=" * 60)

    # Interpretation
    evol_advantage = results["evol_only_solved"] - results["iid_only_solved"]
    if evol_advantage > 0:
        print(f"POSITIVE SIGNAL: Evol solved {evol_advantage} more unique problems than IID!")
        if results["evol_successes_from_refinements"] > 0:
            print(f"Refinements contributed {results['evol_successes_from_refinements']} successes")
        print("\n-> Worth investigating further with full RL training")
    elif evol_advantage == 0:
        print("= No clear difference between methods")
    else:
        print(f"X IID actually solved {-evol_advantage} more unique problems")

    print("=" * 60 + "\n")


def mode_filter(args):
    """Run Phase 1: Filter for hard problems."""
    print("=" * 60)
    print("PHASE 1: Filtering for Hard Problems")
    print("=" * 60)

    # Load dataset based on selection
    if args.dataset == "numina":
        print(f"\nLoading NuminaMath dataset (sources={args.numina_sources}, max={args.max_problems})...")
        problems = load_numina_dataset(
            sources=args.numina_sources,
            max_problems=args.max_problems,
        )
    else:  # math-hard
        print(f"\nLoading MATH-Hard dataset (max={args.max_problems})...")
        problems = load_math_dataset(max_problems=args.max_problems)

    print(f"Loaded {len(problems)} problems")

    print(f"\nLoading model: {args.model}...")
    model = load_model(args.model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    print(f"\nFiltering for hard problems (0/{args.filter_n_samples} correct)...")
    hard_problems = find_hard_problems(
        problems,
        model,
        n_samples=args.filter_n_samples,
        threshold=FILTER_THRESHOLD,
        save_path=args.output,
        save_every=50,
    )

    save_json(hard_problems, args.output)
    print(f"\nSaved {len(hard_problems)} hard problems to {args.output}")


def mode_compare(args):
    """Run Phase 2: Compare methods on hard problems."""
    print("=" * 60)
    print("PHASE 2: Comparing IID vs Evolutionary Sampling")
    print("=" * 60)

    problems = load_json(args.problems)
    print(f"Loaded {len(problems)} hard problems from {args.problems}")

    print(f"\nLoading model: {args.model}...")
    model = load_model(args.model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    print(f"\nRunning comparison with N={args.n}...")
    results = run_comparison(problems, model, n_total=args.n, save_path=args.output)

    print_summary(results)


def mode_compare_multi(args):
    """Run comparison with multiple N values."""
    print("=" * 60)
    print("PHASE 2b: Comparing Methods Across Multiple N Values")
    print("=" * 60)

    problems = load_json(args.problems)
    print(f"Loaded {len(problems)} hard problems from {args.problems}")

    print(f"\nLoading model: {args.model}...")
    model = load_model(args.model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    # Determine output directory
    output_dir = os.path.dirname(args.output) if args.output else "results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning comparison for N values: {N_VALUES_TO_TEST}")
    all_results = run_multi_n_comparison(
        problems,
        model,
        n_values=N_VALUES_TO_TEST,
        save_dir=output_dir,
    )

    # Save combined results
    save_json(
        {str(k): {kk: vv for kk, vv in v.items() if kk != "detailed_results"}
         for k, v in all_results.items()},
        args.output or os.path.join(output_dir, "comparison_multi_summary.json"),
    )

    # Print summary for each N
    for n, results in all_results.items():
        print(f"\n{'='*60}")
        print(f"N = {n}")
        print_summary(results)


def mode_analyze(args):
    """Analyze existing results."""
    print("=" * 60)
    print("Analyzing Results")
    print("=" * 60)

    results = load_json(args.results)
    print_summary(results)

    # Additional analysis if detailed results available
    if "detailed_results" in results:
        details = results["detailed_results"]

        # Find interesting cases
        evol_only = [d for d in details if d.get("evol_any_correct") and not d.get("iid_any_correct")]
        iid_only = [d for d in details if d.get("iid_any_correct") and not d.get("evol_any_correct")]

        if evol_only:
            print("\n--- Problems Solved by EVOL Only ---")
            for d in evol_only[:5]:  # Show first 5
                print(f"  Problem: {d['problem'][:80]}...")
                print(f"  Answer: {d['answer']}")
                print(f"  Correct at generations: {d.get('evol_correct_generations', [])}")
                print()

        if iid_only:
            print("\n--- Problems Solved by IID Only ---")
            for d in iid_only[:5]:
                print(f"  Problem: {d['problem'][:80]}...")
                print(f"  Answer: {d['answer']}")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary Sampling Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter for hard problems
  python run_experiment.py --mode filter --output data/hard_problems.json

  # Compare methods with N=16
  python run_experiment.py --mode compare --problems data/hard_problems.json --output results/comparison.json

  # Compare across multiple N values
  python run_experiment.py --mode compare_multi --problems data/hard_problems.json --output results/multi.json

  # Analyze results
  python run_experiment.py --mode analyze --results results/comparison.json
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["filter", "compare", "compare_multi", "analyze"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument(
        "--problems",
        type=str,
        help="Path to problems JSON (for compare modes)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path",
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to results JSON (for analyze mode)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=16,
        help="N for compare mode (default: 16)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help=f"Max problems to process (default: {MAX_PROBLEMS})",
    )
    parser.add_argument(
        "--filter-n-samples",
        type=int,
        default=None,
        help=f"Samples per problem for filtering (default: {FILTER_N_SAMPLES})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["math-hard", "numina"],
        default="math-hard",
        help="Dataset to use: math-hard (lighteval/MATH-Hard) or numina (AI-MO/NuminaMath-1.5)",
    )
    parser.add_argument(
        "--numina-sources",
        type=str,
        default="amc_aime,olympiads",
        help="Comma-separated NuminaMath sources (default: amc_aime,olympiads)",
    )

    args = parser.parse_args()

    # Apply defaults from config if not specified
    args.model = args.model or MODEL_NAME
    args.max_problems = args.max_problems or MAX_PROBLEMS
    args.filter_n_samples = args.filter_n_samples or FILTER_N_SAMPLES
    args.numina_sources = args.numina_sources.split(",") if args.numina_sources else None

    # Validate arguments
    if args.mode == "filter":
        if not args.output:
            args.output = "data/hard_problems.json"

    elif args.mode in ["compare", "compare_multi"]:
        if not args.problems:
            parser.error("--problems is required for compare modes")
        if not args.output:
            args.output = f"results/comparison_n{args.n}.json"

    elif args.mode == "analyze":
        if not args.results:
            parser.error("--results is required for analyze mode")

    # Run the selected mode
    if args.mode == "filter":
        mode_filter(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "compare_multi":
        mode_compare_multi(args)
    elif args.mode == "analyze":
        mode_analyze(args)


if __name__ == "__main__":
    main()
