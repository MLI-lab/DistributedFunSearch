# Evolutionary Sampling Experiment

Testing whether evolutionary/iterative refinement can find correct solutions on hard math problems where standard IID sampling fails.

## Research Question

In RL methods like GRPO, we sample N responses per prompt and compute relative rewards. On very hard problems, if all N samples fail, there's no learning signal.

**Hypothesis:** Instead of N independent samples, using √N seeds with iterative refinement might reach correct solutions that random IID sampling cannot.

## Quick Start

```bash
cd evol-sampling-experiment
pip install -r requirements.txt

# Step 1: Filter for hard problems (0/64 correct)
# Uses lighteval/MATH-Hard by default (pre-filtered Level 5 problems)
python run_experiment.py --mode filter --output data/hard_problems.json

# Step 2: Compare methods with N=16
python run_experiment.py --mode compare --problems data/hard_problems.json --n 16 --output results/comparison_n16.json

# Step 3: Analyze results
python run_experiment.py --mode analyze --results results/comparison_n16.json
```

## Command-Line Options

```bash
python run_experiment.py --mode <mode> [options]

Modes:
  filter        - Find problems where model gets 0% correct
  compare       - Compare IID vs Evol sampling on hard problems
  compare_multi - Compare across multiple N values
  analyze       - Analyze existing results

Options:
  --model MODEL           Model to use (default: Qwen/Qwen2.5-7B-Instruct)
  --max-problems N        Max problems to process (default: 500)
  --filter-n-samples N    Samples per problem for filtering (default: 64)
  --n N                   N for compare mode (default: 16)
  --problems PATH         Path to hard problems JSON (for compare modes)
  --output PATH           Output path
  --results PATH          Path to results JSON (for analyze mode)
```

## Recommended Models

| Model | GPU Memory | Notes |
|-------|------------|-------|
| `Qwen/Qwen2.5-7B-Instruct` | ~15GB | Default, good balance |
| `Qwen/Qwen2.5-3B-Instruct` | ~7GB | Smaller but capable |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3GB | Fast testing |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~15GB | Excellent reasoning |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | ~3GB | Smaller reasoning model |
| `microsoft/Phi-3-mini-4k-instruct` | ~8GB | Good balance |

Example with different model:
```bash
python run_experiment.py --mode filter --model Qwen/Qwen2.5-3B-Instruct --output data/hard_problems.json
```

## Experimental Design

### Phase 1: Construct the "hard problem" regime
- Uses `lighteval/MATH-Hard` dataset (pre-filtered Level 5 problems)
- Sample N=64 IID responses from base model
- Keep only problems with 0/64 correct
- These are problems where standard RL gets no signal

### Phase 2: Compare methods
- **IID:** 16 independent samples
- **Evol:** 4 seeds × 4 generations (1 seed + 3 refinements) = 16 samples

### Phase 3 (optional): Test multiple N values
```bash
python run_experiment.py --mode compare_multi --problems data/hard_problems.json --output results/multi.json
```

## Key Metrics

| Metric | Meaning |
|--------|---------|
| `evol_only_solved` | Problems solved by evol but not IID **(KEY METRIC)** |
| `iid_only_solved` | Problems solved by IID but not evol |
| `evol_successes_from_refinements` | Did iterative refinement actually help? |
| `token_overhead_ratio` | Compute cost of evol vs IID |

## Interpreting Results

- If `evol_only_solved > iid_only_solved` and refinements contributed → hypothesis holds, worth pursuing full RL training
- If no difference → save compute, stick with IID
- Track token overhead for compute fairness analysis

## Project Structure

```
evol-sampling-experiment/
├── config.py                    # Hyperparameters
├── run_experiment.py            # Main entry point
├── data/
│   └── loader.py                # MATH dataset loading
├── model/
│   └── inference.py             # vLLM/HuggingFace wrapper
├── evaluation/
│   └── verifier.py              # Binary answer verification
├── sampling/
│   ├── iid_sampling.py          # Standard N independent samples
│   └── evol_sampling.py         # Evolutionary refinement sampling
├── experiments/
│   ├── filter_hard_problems.py  # Phase 1: Find hard problems
│   └── compare_methods.py       # Phase 2: Method comparison
└── results/                     # Output directory
```

## Configuration

Edit `config.py` to adjust:
- `MODEL_NAME` - Which model to use (default: Qwen/Qwen2.5-7B-Instruct)
- `DATASET` - Dataset to use (default: lighteval/MATH-Hard)
- `MAX_PROBLEMS` - Number of problems to test (default: 500)
- `FILTER_N_SAMPLES` - Samples for filtering (default: 64)
- `N_VALUES_TO_TEST` - N values for multi-comparison (default: [16, 32, 64])

## GPU Memory Notes

- vLLM requires ~40GB free memory by default (uses 90% utilization)
- If vLLM fails, falls back to HuggingFace (slower but uses less memory)
- Use `CUDA_VISIBLE_DEVICES=X` to select a specific GPU
- For testing with limited GPU memory, use smaller models like Qwen2.5-1.5B-Instruct
