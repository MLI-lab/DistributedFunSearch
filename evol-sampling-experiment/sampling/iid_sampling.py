# sampling/iid_sampling.py
"""Standard IID sampling baseline."""

from model.inference import ModelWrapper


def iid_sample(problem: str, n_samples: int, model: ModelWrapper) -> dict:
    """
    Generate n_samples independent solutions for a problem.

    This is the standard approach used in GRPO and other RL methods:
    sample N independent responses and compute relative rewards.

    Args:
        problem: The math problem text
        n_samples: Number of independent samples to generate
        model: ModelWrapper instance

    Returns:
        {
            "solutions": list[str],          # The generated solutions
            "total_input_tokens": int,       # Total input tokens used
            "total_output_tokens": int,      # Total output tokens generated
            "total_tokens": int              # Total tokens (input + output)
        }
    """
    # Generate prompts (all identical for IID sampling)
    prompts = [model._make_solution_prompt(problem)] * n_samples

    # Batch generate for efficiency
    results = model.generate_batch(prompts)

    solutions = [r["response"] for r in results]
    total_input = sum(r["input_tokens"] for r in results)
    total_output = sum(r["output_tokens"] for r in results)

    return {
        "solutions": solutions,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


if __name__ == "__main__":
    # Quick test
    from model.inference import load_model

    print("Testing IID sampling...")
    model = load_model("Qwen/Qwen2.5-0.5B-Instruct")

    result = iid_sample("What is 2 + 2?", n_samples=3, model=model)
    print(f"Generated {len(result['solutions'])} solutions")
    print(f"Total tokens: {result['total_tokens']}")
    for i, sol in enumerate(result["solutions"]):
        print(f"\nSolution {i+1}:")
        print(sol[:200] + "..." if len(sol) > 200 else sol)
