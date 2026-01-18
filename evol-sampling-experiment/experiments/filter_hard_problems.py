# experiments/filter_hard_problems.py
"""
Phase 1: Find problems where the model gets 0% success.

These are the problems where standard RL would get no learning signal,
and where we hypothesize evolutionary sampling might help.
"""

import json
import os
from typing import Optional
from tqdm import tqdm

from sampling.iid_sampling import iid_sample
from evaluation.verifier import evaluate_solutions
from model.inference import ModelWrapper


def find_hard_problems(
    problems: list[dict],
    model: ModelWrapper,
    n_samples: int = 64,
    threshold: int = 0,
    save_path: Optional[str] = None,
    save_every: int = 50,
) -> list[dict]:
    """
    Find problems where model gets <= threshold correct out of n_samples.

    This constructs the "hard problem regime" where:
    - Standard RL would get no learning signal (all rewards = 0)
    - We want to test if evolutionary sampling can still find solutions

    Args:
        problems: List of problem dicts with "problem" and "answer" keys
        model: ModelWrapper instance
        n_samples: Number of samples to try per problem (default 64)
        threshold: Keep problems with this many or fewer successes (default 0)
        save_path: Optional path to save intermediate results
        save_every: Save intermediate results every N problems

    Returns:
        List of hard problems with filtering stats
    """
    hard_problems = []
    all_stats = []  # Track stats for all problems (for analysis)

    for i, problem in enumerate(tqdm(problems, desc="Filtering for hard problems")):
        result = iid_sample(problem["problem"], n_samples, model)
        rewards = evaluate_solutions(result["solutions"], problem["answer"])
        n_correct = sum(rewards)
        pass_rate = n_correct / n_samples

        stats = {
            "problem_idx": i,
            "n_samples": n_samples,
            "n_correct": n_correct,
            "pass_rate": pass_rate,
        }
        all_stats.append(stats)

        if n_correct <= threshold:
            hard_problems.append({
                **problem,
                "filter_stats": stats,
            })

        # Progress update and intermediate save
        if (i + 1) % save_every == 0:
            print(
                f"Progress: {i+1}/{len(problems)}, "
                f"found {len(hard_problems)} hard problems so far"
            )
            if save_path:
                _save_intermediate(hard_problems, all_stats, save_path, i + 1)

    # Final summary
    print(f"\n{'='*50}")
    print("Filtering complete!")
    print(f"Total problems: {len(problems)}")
    print(f"Hard problems (≤{threshold}/{n_samples} correct): {len(hard_problems)}")
    print(f"Percentage hard: {len(hard_problems)/len(problems):.1%}")
    print(f"{'='*50}\n")

    # Save final results
    if save_path:
        _save_intermediate(hard_problems, all_stats, save_path, len(problems))

    return hard_problems


def _save_intermediate(
    hard_problems: list[dict],
    all_stats: list[dict],
    save_path: str,
    n_processed: int,
):
    """Save intermediate results during filtering."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # Save hard problems
    with open(save_path, "w") as f:
        json.dump(hard_problems, f, indent=2)

    # Save all stats for analysis
    stats_path = save_path.replace(".json", "_all_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "n_processed": n_processed,
            "n_hard": len(hard_problems),
            "stats": all_stats,
        }, f, indent=2)


def analyze_difficulty_distribution(stats: list[dict]) -> dict:
    """Analyze the distribution of pass rates across problems."""
    from collections import Counter

    # Bucket pass rates
    buckets = Counter()
    for s in stats:
        if s["pass_rate"] == 0:
            buckets["0%"] += 1
        elif s["pass_rate"] < 0.1:
            buckets["1-10%"] += 1
        elif s["pass_rate"] < 0.25:
            buckets["10-25%"] += 1
        elif s["pass_rate"] < 0.5:
            buckets["25-50%"] += 1
        else:
            buckets[">50%"] += 1

    return dict(buckets)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing filter_hard_problems...")

    # Mock model for testing
    class MockModel:
        def generate_batch(self, prompts):
            return [{"response": "\\boxed{42}", "input_tokens": 10, "output_tokens": 20}
                    for _ in prompts]

        def _make_solution_prompt(self, problem):
            return f"Solve: {problem}"

    mock_model = MockModel()
    test_problems = [
        {"problem": "What is 2+2?", "answer": "4"},
        {"problem": "What is 3+3?", "answer": "42"},  # Model always says 42
    ]

    hard = find_hard_problems(test_problems, mock_model, n_samples=4, threshold=0)
    print(f"Found {len(hard)} hard problems")
