# experiments/compare_methods.py
"""
Phase 2: Compare IID vs Evolutionary sampling on hard problems.

This is the core experiment that tests whether evolutionary/refinement
sampling can solve problems that IID sampling cannot.
"""

import json
import os
from typing import Optional
from tqdm import tqdm

from sampling.iid_sampling import iid_sample
from sampling.evol_sampling import evolutionary_sample
from evaluation.verifier import evaluate_solutions
from model.inference import ModelWrapper
from config import get_evol_config


def run_comparison(
    problems: list[dict],
    model: ModelWrapper,
    n_total: int = 16,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compare IID vs evolutionary sampling on hard problems.

    Args:
        problems: List of hard problems (pre-filtered)
        model: ModelWrapper instance
        n_total: Total number of samples per method

    Returns:
        Aggregated results with detailed metrics
    """
    n_seeds, refinement_steps = get_evol_config(n_total)
    print(f"Config: N={n_total}, seeds={n_seeds}, refinements={refinement_steps}")
    print(f"IID: {n_total} samples")
    print(f"Evol: {n_seeds} seeds × {1 + refinement_steps} generations = {n_total} samples")

    results = []

    for idx, problem in enumerate(tqdm(problems, desc=f"Comparing methods (N={n_total})")):
        # IID baseline
        iid_result = iid_sample(problem["problem"], n_total, model)
        iid_rewards = evaluate_solutions(iid_result["solutions"], problem["answer"])

        # Evolutionary method
        evol_result = evolutionary_sample(
            problem["problem"],
            n_seeds,
            refinement_steps,
            model,
        )
        evol_rewards = evaluate_solutions(evol_result["all_solutions"], problem["answer"])

        # Track which generation found correct answers (if any)
        evol_correct_generations = []
        for reward, meta in zip(evol_rewards, evol_result["metadata"]):
            if reward:
                evol_correct_generations.append(meta["generation"])

        result = {
            "problem": problem["problem"],
            "answer": problem["answer"],
            "problem_idx": problem.get("idx", idx),
            "iid": {
                "solutions": iid_result["solutions"],
                "rewards": iid_rewards,
                "any_correct": any(iid_rewards),
                "n_correct": sum(iid_rewards),
                "total_tokens": iid_result["total_tokens"],
                "input_tokens": iid_result["total_input_tokens"],
                "output_tokens": iid_result["total_output_tokens"],
            },
            "evol": {
                "solutions": evol_result["all_solutions"],
                "lineages": evol_result["lineages"],
                "metadata": evol_result["metadata"],
                "rewards": evol_rewards,
                "any_correct": any(evol_rewards),
                "n_correct": sum(evol_rewards),
                "total_tokens": evol_result["total_tokens"],
                "input_tokens": evol_result["total_input_tokens"],
                "output_tokens": evol_result["total_output_tokens"],
                "correct_generations": evol_correct_generations,
                "seed_successes": sum(
                    r for r, m in zip(evol_rewards, evol_result["metadata"])
                    if m["is_seed"]
                ),
                "refinement_successes": sum(
                    r for r, m in zip(evol_rewards, evol_result["metadata"])
                    if not m["is_seed"]
                ),
            },
        }
        results.append(result)

    aggregated = aggregate_results(results, n_total)

    if save_path:
        _save_results(aggregated, save_path)

    return aggregated


def aggregate_results(results: list[dict], n_total: int) -> dict:
    """Compute summary statistics."""
    n_problems = len(results)

    if n_problems == 0:
        return {"error": "No problems to analyze"}

    iid_solved = sum(r["iid"]["any_correct"] for r in results)
    evol_solved = sum(r["evol"]["any_correct"] for r in results)

    # Key metrics
    evol_only_solved = sum(
        r["evol"]["any_correct"] and not r["iid"]["any_correct"]
        for r in results
    )
    iid_only_solved = sum(
        r["iid"]["any_correct"] and not r["evol"]["any_correct"]
        for r in results
    )
    both_solved = sum(
        r["iid"]["any_correct"] and r["evol"]["any_correct"]
        for r in results
    )
    neither_solved = sum(
        not r["iid"]["any_correct"] and not r["evol"]["any_correct"]
        for r in results
    )

    # Token usage
    total_iid_tokens = sum(r["iid"]["total_tokens"] for r in results)
    total_evol_tokens = sum(r["evol"]["total_tokens"] for r in results)

    # Which generation found answers?
    generation_successes = {}
    for r in results:
        for gen in r["evol"]["correct_generations"]:
            generation_successes[gen] = generation_successes.get(gen, 0) + 1

    # Successes by source
    evol_successes_from_seeds = sum(r["evol"]["seed_successes"] for r in results)
    evol_successes_from_refinements = sum(r["evol"]["refinement_successes"] for r in results)

    return {
        "config": {
            "n_total": n_total,
            "n_problems": n_problems,
        },

        # Success metrics
        "iid_solved": iid_solved,
        "iid_solve_rate": iid_solved / n_problems,
        "evol_solved": evol_solved,
        "evol_solve_rate": evol_solved / n_problems,

        # Key comparison
        "evol_only_solved": evol_only_solved,
        "iid_only_solved": iid_only_solved,
        "both_solved": both_solved,
        "neither_solved": neither_solved,

        # Token usage
        "total_iid_tokens": total_iid_tokens,
        "total_evol_tokens": total_evol_tokens,
        "token_overhead_ratio": (
            total_evol_tokens / total_iid_tokens if total_iid_tokens > 0 else 0
        ),

        # Where did evol successes come from?
        "evol_successes_from_seeds": evol_successes_from_seeds,
        "evol_successes_from_refinements": evol_successes_from_refinements,
        "successes_by_generation": generation_successes,

        # Full results for detailed analysis
        "detailed_results": results,
    }


def run_multi_n_comparison(
    problems: list[dict],
    model: ModelWrapper,
    n_values: list[int],
    save_dir: Optional[str] = None,
) -> dict:
    """
    Run comparison for multiple N values.

    This tests whether the benefit of evolutionary sampling disappears
    as N gets larger (more IID samples might eventually find correct answers).
    """
    all_results = {}

    for n in n_values:
        print(f"\n{'='*50}")
        print(f"Running comparison with N={n}")
        print(f"{'='*50}")

        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"comparison_n{n}.json")

        results = run_comparison(problems, model, n_total=n, save_path=save_path)
        all_results[n] = results

        # Quick summary
        print(f"N={n}: IID solved {results['iid_solved']}, Evol solved {results['evol_solved']}")
        print(f"  Evol-only: {results['evol_only_solved']}, IID-only: {results['iid_only_solved']}")

    return all_results


def _save_results(results: dict, save_path: str):
    """Save results to JSON files."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # Save summary (without full solution text)
    summary = {k: v for k, v in results.items() if k != "detailed_results"}
    summary_path = save_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save slim details (without full solution text to save space)
    slim_details = []
    for r in results.get("detailed_results", []):
        slim_details.append({
            "problem": r["problem"][:200] + "..." if len(r["problem"]) > 200 else r["problem"],
            "answer": r["answer"],
            "problem_idx": r.get("problem_idx"),
            "iid_any_correct": r["iid"]["any_correct"],
            "iid_n_correct": r["iid"]["n_correct"],
            "evol_any_correct": r["evol"]["any_correct"],
            "evol_n_correct": r["evol"]["n_correct"],
            "evol_correct_generations": r["evol"]["correct_generations"],
        })

    full_results = {**summary, "detailed_results": slim_details}
    with open(save_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    # Quick test with mock data
    print("Testing compare_methods...")

    class MockModel:
        def __init__(self):
            self.call_count = 0

        def generate_batch(self, prompts):
            results = []
            for _ in prompts:
                self.call_count += 1
                # Alternate between correct and wrong
                ans = "4" if self.call_count % 3 == 0 else "42"
                results.append({
                    "response": f"\\boxed{{{ans}}}",
                    "input_tokens": 10,
                    "output_tokens": 20,
                })
            return results

        def generate_refinement(self, problem, previous):
            self.call_count += 1
            # Refinements more likely to be correct
            ans = "4" if self.call_count % 2 == 0 else "42"
            return {
                "response": f"\\boxed{{{ans}}}",
                "input_tokens": 50,
                "output_tokens": 20,
            }

        def _make_solution_prompt(self, problem):
            return f"Solve: {problem}"

    mock_model = MockModel()
    test_problems = [
        {"problem": "What is 2+2?", "answer": "4"},
        {"problem": "What is 3+3?", "answer": "6"},
    ]

    results = run_comparison(test_problems, mock_model, n_total=16)
    print(f"\nIID solved: {results['iid_solved']}")
    print(f"Evol solved: {results['evol_solved']}")
    print(f"Evol-only: {results['evol_only_solved']}")
