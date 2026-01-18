# data/loader.py
"""Load and preprocess math datasets from HuggingFace."""

import re
import sys
import os
from typing import Optional
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET


def extract_answer_from_solution(solution: str) -> str:
    """Extract the final answer from a solution string.

    MATH dataset solutions typically end with \\boxed{answer}.
    Handles nested braces properly.
    """
    # Find the last \boxed{...} in the solution
    # Handle nested braces by matching balanced braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, solution)
    if matches:
        return matches[-1].strip()
    return solution.strip()  # Fallback: return whole solution


def load_math_dataset(
    difficulty: Optional[list[str]] = None,
    max_problems: Optional[int] = None,
    split: str = "train",
    dataset_name: Optional[str] = None,
) -> list[dict]:
    """
    Load MATH dataset from HuggingFace.

    Args:
        difficulty: List of difficulty levels to include (e.g., ["Level 5"])
        max_problems: Maximum number of problems to load
        split: Dataset split to use ("train" or "test")
        dataset_name: Override dataset name (default: from config)

    Returns:
        List of problem dicts:
        {
            "problem": str,      # The problem text
            "answer": str,       # Ground truth answer (extracted from solution)
            "solution": str,     # Full solution text
            "level": str,        # Difficulty level
            "type": str,         # Problem type (algebra, geometry, etc.)
            "idx": int,          # Original index in dataset
        }
    """
    ds_name = dataset_name or DATASET
    print(f"Loading MATH dataset from {ds_name} (split={split})...")
    dataset = load_dataset(ds_name, split=split)

    problems = []
    for idx, item in enumerate(dataset):
        # Filter by difficulty if specified
        if difficulty and item["level"] not in difficulty:
            continue

        # Extract answer from solution
        answer = extract_answer_from_solution(item["solution"])

        prob = {
            "problem": item["problem"],
            "answer": answer,
            "solution": item["solution"],
            "level": item["level"],
            "type": item["type"],
            "idx": idx,
        }

        problems.append(prob)

        if max_problems and len(problems) >= max_problems:
            break

    print(f"Loaded {len(problems)} problems" +
          (f" (filtered to {difficulty})" if difficulty else ""))

    return problems


def load_numina_dataset(
    sources: Optional[list[str]] = None,
    max_problems: Optional[int] = None,
    exclude_proof: bool = True,
    exclude_notfound: bool = True,
) -> list[dict]:
    """
    Load NuminaMath-1.5 dataset from HuggingFace.

    Args:
        sources: List of sources to include (e.g., ["amc_aime", "olympiads"])
                 If None, loads all sources
        max_problems: Maximum number of problems to load
        exclude_proof: Skip problems with answer="proof"
        exclude_notfound: Skip problems with answer="notfound"

    Returns:
        List of problem dicts with standardized format
    """
    print(f"Loading NuminaMath-1.5 dataset...")
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train")

    problems = []
    for idx, item in enumerate(dataset):
        # Filter by source if specified
        if sources and item["source"] not in sources:
            continue

        # Skip invalid problems
        if not item.get("problem_is_valid", True):
            continue

        answer = item.get("answer", "")

        # Skip proof problems if requested
        if exclude_proof and answer == "proof":
            continue

        # Skip notfound problems if requested
        if exclude_notfound and answer == "notfound":
            continue

        prob = {
            "problem": item["problem"],
            "answer": answer,
            "solution": item.get("solution", ""),
            "level": f"source:{item['source']}",  # Use source as pseudo-level
            "type": item.get("problem_type", "unknown"),
            "source": item["source"],
            "idx": idx,
        }

        problems.append(prob)

        if max_problems and len(problems) >= max_problems:
            break

    source_str = f" (filtered to {sources})" if sources else ""
    print(f"Loaded {len(problems)} problems{source_str}")

    return problems


def get_problem_stats(problems: list[dict]) -> dict:
    """Get statistics about the loaded problems."""
    from collections import Counter

    levels = Counter(p.get("level", "unknown") for p in problems)
    types = Counter(p.get("type", "unknown") for p in problems)
    sources = Counter(p.get("source", "unknown") for p in problems)

    return {
        "total": len(problems),
        "by_level": dict(levels),
        "by_type": dict(types),
        "by_source": dict(sources),
    }


if __name__ == "__main__":
    # Quick test - MATH
    print("=" * 50)
    print("Testing MATH-Hard dataset")
    print("=" * 50)
    problems = load_math_dataset(max_problems=5)
    print(f"\nLoaded {len(problems)} problems")
    print(f"Example: {problems[0]['problem'][:100]}...")
    print(f"Answer: {problems[0]['answer']}")

    # Quick test - NuminaMath
    print("\n" + "=" * 50)
    print("Testing NuminaMath dataset (amc_aime + olympiads)")
    print("=" * 50)
    problems = load_numina_dataset(sources=["amc_aime", "olympiads"], max_problems=5)
    print(f"\nLoaded {len(problems)} problems")
    if problems:
        print(f"Example: {problems[0]['problem'][:100]}...")
        print(f"Answer: {problems[0]['answer']}")
        print(f"Source: {problems[0]['source']}")
