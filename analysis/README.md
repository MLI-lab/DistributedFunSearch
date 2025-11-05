# DeCoSearch Analysis Tools

This directory contains tools for analyzing checkpoint files generated during DeCoSearch experiments.

## Tools

### 1. Checkpoint Analyzer (`checkpoint_analyzer.py`)

Load and inspect checkpoint files to view island states, clusters, best programs, and statistics.

**Features:**
- View checkpoint summary (resource usage, program stats)
- Display island summary table
- Show detailed island information
- Extract best programs from all islands
- Analyze cluster distribution
- Compare multiple checkpoints

**Usage:**

Edit the CONFIGURATION section in `main()` to set:
- `checkpoint_path`: Path to your checkpoint file
- `island_id`: Specific island to analyze (or None)
- `export_file`: Output file for best programs (or None)
- `compare_checkpoints`: List of checkpoints to compare (or None)

Then run:
```bash
python checkpoint_analyzer.py
```

### 2. Evolution Plotter (`plot_evolution.py`)

Create visualizations showing how the search evolved across multiple checkpoints.

**Features:**
- Plot best scores over time per island
- Show cluster count and size evolution
- Display resource usage (CPU/GPU) trends
- Visualize token usage

**Usage:**

Edit the CONFIGURATION section in `main()` to set:
- `checkpoint_patterns`: List of file patterns (supports wildcards)
- `output_dir`: Directory to save plots (or None for interactive)

Then run:
```bash
python plot_evolution.py
```

**Generated Plots:**
1. `best_scores_evolution.png` - Best score per island over time
2. `cluster_statistics.png` - Cluster counts, sizes, and program counts
3. `resource_usage.png` - CPU and GPU time usage
4. `token_usage.png` - Input/output token consumption

## Installation

The analysis tools use dependencies that are included in the main DeCoSearch package:
- `matplotlib` (for plotting)
- `tabulate` (for table formatting)
- `numpy` (for computations)

These are automatically installed when you run `pip install .` from the DeCoSearch root directory.

## Checkpoint File Structure

Checkpoints are pickled dictionaries containing:

```python
{
    # Resource usage
    'cumulative_evaluator_cpu_time': float,
    'cumulative_sampler_gpu_time': float,
    'cumulative_input_tokens': int,
    'cumulative_output_tokens': int,

    # Island data
    'best_score_per_island': List[float],
    'best_program_per_island': List[Dict],
    'best_scores_per_test_per_island': List[Dict],
    'islands_state': List[Dict],

    # Program statistics
    'total_stored_programs': int,
    'execution_failed': int,
    'version_mismatch_discarded': int,
    'duplicates_discarded': int,

    # Prompt statistics
    'total_prompts': int,
    'dublicate_prompts': int,

    # Solution tracking
    'found_optimal_solution': bool,
    'prompts_since_optimal': int,

    # Metadata
    'last_reset_time': float,
}
```

Each island state contains:
```python
{
    'clusters': {
        '<signature>': {
            'score': float,
            'scores_per_test': Dict,
            'programs': List[Dict],  # Serialized Function objects
        }
    },
    'version': int,
    'num_programs': int,
}
```

## Examples

### Checkpoint Inspection

**In `checkpoint_analyzer.py`:**
```python
checkpoint_path = "Checkpoints/checkpoint_2025-01-15_10-30-00.pkl"
island_id = None  # View all islands
export_file = "best_programs.txt"  # Export best programs
```
Then run: `python checkpoint_analyzer.py`

### Track Experiment Progress

**In `checkpoint_analyzer.py`:**
```python
compare_checkpoints = [
    "Checkpoints/checkpoint_2025-01-15_08-00-00.pkl",
    "Checkpoints/checkpoint_2025-01-15_09-00-00.pkl",
    "Checkpoints/checkpoint_2025-01-15_10-00-00.pkl"
]
```

**In `plot_evolution.py`:**
```python
checkpoint_patterns = ["Checkpoints/checkpoint_*.pkl"]
output_dir = "progress_plots"
```
Then run both scripts.

### Detailed Island Analysis

**In `checkpoint_analyzer.py`:**
```python
checkpoint_path = "Checkpoints/latest.pkl"
island_id = 3  # Focus on island 3
export_file = None
```
This shows:
- Island version and program count
- All clusters with their signatures and scores
- Programs per cluster
- Detailed scores per test case


4. **Comparing Experiments:** Use W&B for live tracking, checkpoints for offline analysis

5. **Extracting Code:** Best programs are serialized as dictionaries with 'body' field:
   ```python
   import pickle
   with open('checkpoint.pkl', 'rb') as f:
       ckpt = pickle.load(f)
   best_code = ckpt['best_program_per_island'][0]['body']
   print(best_code)
   ```

