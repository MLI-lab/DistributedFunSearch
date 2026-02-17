#!/usr/bin/env python3
"""
Extract sentence templates from algorithm descriptions using an LLM.

This script samples descriptions from a checkpoint and asks an LLM to identify
the common template pattern(s) they follow.

Arguments:
    path                  Checkpoint .pkl file or folder containing checkpoints.
    --samples, -n         Number of descriptions to sample, default 100.
    --model, -m           LLM model to use, default gpt-4o-mini.
    --output, -o          Output file for results, default prints to stdout.
    --seed                Random seed for sampling, default 42.

Usage:
    python templates.py checkpoint.pkl
    python templates.py checkpoint.pkl --samples 200
    python templates.py checkpoint.pkl --model gpt-4o-mini
    python templates.py ./checkpoints/run_folder/

Dependencies:
    pip install litellm
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Load .env file from project root
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on environment variables

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import find_latest_checkpoint, load_checkpoint, extract_descriptions


FEW_SHOT_PROMPT = '''Task: Find the most general sentence template from these algorithm descriptions.

Instructions:
1. It is likely that one single template fits all or nearly all descriptions.
2. Only output additional templates if they are fundamentally different in structure.
3. Prefer one general template over multiple specific ones.
4. Treat synonyms as the same slot (e.g. optimize, maximize, enhance, improve all map to [goal]).
5. Ignore minor variations like extra adjectives, clauses, or different conjunctions.

Example input:
1. "The cat quickly sat on the mat."
2. "The dog sat on the old rug."
3. "The small bird sat carefully on the branch."
4. "A horse ran through the field."

Good output (one general template):
Template: "[subject] [action] [location]."
  Coverage: 4/4 (100%)
  Slots:
    [subject]: animal names with optional article
    [action]: movement or position verbs
    [location]: places or surfaces

Bad output (too granular, unnecessary splits):
Template 1: "The [animal] sat on the [surface]." (3/4)
Template 2: "A [animal] ran through the [place]." (1/4)

---

Now find the most general template from these {n} algorithm descriptions:

{descriptions}

Output:
1. The single most general template that covers most or all descriptions.
2. For each slot, list the actual words/phrases that fill it (e.g., [goal]: optimize, maximize, enhance, improve).
3. Only add a second template if some descriptions have a fundamentally different structure.
'''


def build_prompt(descriptions: list[str]) -> str:
    """Build the template extraction prompt with sampled descriptions."""
    numbered = "\n".join(f'{i+1}. "{d}"' for i, d in enumerate(descriptions))
    return FEW_SHOT_PROMPT.format(n=len(descriptions), descriptions=numbered)


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM with the prompt and return response."""
    if not LITELLM_AVAILABLE:
        raise ImportError("litellm is required. Install with: pip install litellm")

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # low temperature for consistent extraction
    )
    return response.choices[0].message.content


def extract_templates(
    checkpoint_path: Path,
    sample_size: int = 100,
    model: str = "gpt-4o-mini",
    seed: int = 42,
) -> dict:
    """
    Extract templates from checkpoint descriptions.

    Args:
        checkpoint_path: Path to checkpoint file or folder
        sample_size: Number of descriptions to sample
        model: LLM model to use
        seed: Random seed for reproducibility

    Returns:
        Dictionary with prompt, response, and metadata
    """
    # Load checkpoint
    print(f"Loading: {checkpoint_path}")
    checkpoint = load_checkpoint(str(checkpoint_path))

    # Extract descriptions
    descriptions = extract_descriptions(checkpoint)
    total = len(descriptions)
    print(f"Total descriptions: {total:,}")

    if total < sample_size:
        print(f"Warning: only {total} descriptions available, using all")
        sample_size = total

    # Sample randomly
    random.seed(seed)
    sampled = random.sample(descriptions, sample_size)
    print(f"Sampled: {sample_size} descriptions (seed={seed})")

    # Build prompt
    prompt = build_prompt(sampled)
    print(f"Prompt length: {len(prompt):,} characters")
    print()

    # Call LLM
    print(f"Calling {model}...")
    print("=" * 60)
    response = call_llm(prompt, model=model)
    print(response)
    print("=" * 60)

    return {
        "checkpoint": str(checkpoint_path),
        "total_descriptions": total,
        "sample_size": sample_size,
        "seed": seed,
        "model": model,
        "prompt": prompt,
        "response": response,
        "sampled_descriptions": sampled,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract sentence templates from algorithm descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("path", help="Checkpoint .pkl file or folder")
    parser.add_argument("--samples", "-n", type=int, default=100,
                        help="Number of descriptions to sample (default: 100)")
    parser.add_argument("--model", "-m", default="gpt-4o-mini",
                        help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file for full results JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")

    args = parser.parse_args()

    if not LITELLM_AVAILABLE:
        print("Error: litellm is required. Install with: pip install litellm")
        return 1

    # Find checkpoint
    path = Path(args.path)
    checkpoint_path = find_latest_checkpoint(path)

    # Run extraction
    results = extract_templates(
        checkpoint_path,
        sample_size=args.samples,
        model=args.model,
        seed=args.seed,
    )

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"Saved full results to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
