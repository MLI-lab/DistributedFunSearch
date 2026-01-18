# sampling/evol_sampling.py
"""Evolutionary/refinement-based sampling."""

from model.inference import ModelWrapper


def evolutionary_sample(
    problem: str,
    n_seeds: int,
    refinement_steps: int,
    model: ModelWrapper,
) -> dict:
    """
    Generate solutions using evolutionary/refinement approach.

    Process:
    1. Generate n_seeds initial solutions (can batch)
    2. For each seed, iteratively refine refinement_steps times
    3. Return all solutions (seeds + all refinements)

    Total solutions = n_seeds * (1 + refinement_steps)

    The hypothesis is that iterative refinement can find correct solutions
    that random IID sampling cannot reach, especially on hard problems.

    Args:
        problem: The math problem text
        n_seeds: Number of initial seed solutions
        refinement_steps: Number of refinement iterations per seed
        model: ModelWrapper instance

    Returns:
        {
            "all_solutions": list[str],    # All solutions (seeds + refinements)
            "lineages": list[list[str]],   # Each lineage: [seed, ref1, ref2, ...]
            "metadata": list[dict],        # Info about each solution
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_tokens": int
        }
    """
    lineages = []
    all_solutions = []
    metadata = []
    total_input = 0
    total_output = 0

    # Generate seeds (batch for efficiency)
    seed_prompts = [model._make_solution_prompt(problem)] * n_seeds
    seed_results = model.generate_batch(seed_prompts)

    for seed_idx, seed_result in enumerate(seed_results):
        seed = seed_result["response"]
        total_input += seed_result["input_tokens"]
        total_output += seed_result["output_tokens"]

        lineage = [seed]
        all_solutions.append(seed)
        metadata.append({
            "is_seed": True,
            "lineage_idx": seed_idx,
            "generation": 0,
        })

        # Iteratively refine this seed
        current = seed
        for gen in range(refinement_steps):
            result = model.generate_refinement(problem, current)
            refined = result["response"]
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]

            lineage.append(refined)
            all_solutions.append(refined)
            metadata.append({
                "is_seed": False,
                "lineage_idx": seed_idx,
                "generation": gen + 1,
            })
            current = refined

        lineages.append(lineage)

    return {
        "all_solutions": all_solutions,
        "lineages": lineages,
        "metadata": metadata,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


def get_solution_info(metadata: list[dict], idx: int) -> str:
    """Get human-readable info about a solution."""
    m = metadata[idx]
    if m["is_seed"]:
        return f"Seed {m['lineage_idx']}"
    else:
        return f"Lineage {m['lineage_idx']}, Gen {m['generation']}"


if __name__ == "__main__":
    # Quick test
    from model.inference import load_model

    print("Testing evolutionary sampling...")
    model = load_model("Qwen/Qwen2.5-0.5B-Instruct")

    # Test with 2 seeds, 1 refinement each = 4 total solutions
    result = evolutionary_sample(
        "What is 2 + 2?",
        n_seeds=2,
        refinement_steps=1,
        model=model,
    )

    print(f"Generated {len(result['all_solutions'])} solutions")
    print(f"Lineages: {len(result['lineages'])}")
    print(f"Total tokens: {result['total_tokens']}")

    for i, (sol, meta) in enumerate(zip(result["all_solutions"], result["metadata"])):
        info = get_solution_info(result["metadata"], i)
        print(f"\n{info}:")
        print(sol[:150] + "..." if len(sol) > 150 else sol)
