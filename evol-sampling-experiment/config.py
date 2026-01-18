# config.py
# All hyperparameters for the evolutionary sampling experiment

# Model settings
# Recommended models (GPU memory required):
#   - Qwen/Qwen2.5-7B-Instruct (~15GB) - default, good balance
#   - Qwen/Qwen2.5-3B-Instruct (~7GB) - smaller but capable
#   - Qwen/Qwen2.5-1.5B-Instruct (~3GB) - fast testing
#   - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (~15GB) - excellent at reasoning
#   - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (~3GB) - smaller reasoning model
#   - microsoft/Phi-3-mini-4k-instruct (~8GB) - good balance
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# Dataset settings
# Options:
#   - "lighteval/MATH-Hard" - Pre-filtered Level 5 only (recommended)
#   - "DigitalLearningGmbH/MATH-lighteval" - Full dataset, needs filtering
DATASET = "lighteval/MATH-Hard"
DIFFICULTY_FILTER = ["Level 5"]  # Only used if not using MATH-Hard
MAX_PROBLEMS = 500

# Filtering settings (Phase 1)
# Only needed if using full MATH dataset, not MATH-Hard
FILTER_N_SAMPLES = 64        # Samples to identify hard problems
FILTER_THRESHOLD = 0         # Keep problems with 0 successes

# Comparison settings (Phase 2)
# We'll test multiple N values to see if benefit disappears at large N
N_VALUES_TO_TEST = [16, 32, 64]  # Start with 16, then expand

# For N=16:
# IID: 16 samples
# Evol: 4 seeds × 4 generations (1 seed + 3 refinements) = 16 samples

# For N=64:
# IID: 64 samples
# Evol: 8 seeds × 8 generations = 64 samples


def get_evol_config(n_total: int) -> tuple[int, int]:
    """Return (n_seeds, refinement_steps) for a given total N.

    The evolutionary approach uses sqrt(N) seeds, each refined until we have N total.
    n_seeds * (1 + refinement_steps) = n_total
    """
    import math
    n_seeds = int(math.sqrt(n_total))
    refinement_steps = (n_total // n_seeds) - 1  # -1 because seed counts as gen 0
    # Verify: n_seeds * (1 + refinement_steps) == n_total
    actual_total = n_seeds * (1 + refinement_steps)
    assert actual_total == n_total, f"Config mismatch: {actual_total} != {n_total}"
    return n_seeds, refinement_steps
