# DistributedFunSearch Analysis Tools

This directory contains tools for analyzing checkpoint files, evaluating priority functions, and comparing codebooks from various sources (KAMIS, greedy baselines, our priority functions) against VT codes.

## Quick Reference

| Script | Purpose |
|--------|---------|
| `extract_successful_functions.py` | Extract priority functions from checkpoints |
| `evaluate_extended.py` | Evaluate functions on larger n values |
| `vt_overlap_analysis.py` | Analyze VT overlap for our codebooks |
| `run_full_analysis.py` | Run complete analysis pipeline |
| `generate_vt_codes.py` | Generate VT codes for any n, a |
| `profile_evaluations.py` | Profile evaluation variants |
| `checkpoint_inspector.py` | Inspect single checkpoint |
| `checkpoint_searcher.py` | Search across checkpoints |
| `compare_to_vt.py` | Compare ANY codebook to VT |
| `subset_analysis.py` | Check subset relationships |
| `baselines/random_greedy.py` | Random greedy baseline |

## Tools

### 1. Extract Successful Functions (`extract_successful_functions.py`)

Extract and deduplicate priority functions from checkpoints that achieve a target signature (scores per test).

**Features:**
- Extract functions matching exact target signature
- Supports multiple files/folders, scans all checkpoints across experiments
- Deduplication based on priority values (across all sources)
- Auto-detects function signature type (no_graph, graph_gt, graph_networkx)
- `--latest-only` option to use only the newest checkpoint per folder

**Usage:**
```bash
# Single checkpoint file
python analysis/extract_successful_functions.py /path/to/checkpoint.pkl \
    --output ./results/

# Single folder (scans all .pkl files recursively)
python analysis/extract_successful_functions.py /path/to/checkpoints/ \
    --output ./results/

# Multiple folders (combine experiments)
python analysis/extract_successful_functions.py /exp1/checkpoints/ /exp2/checkpoints/ \
    --output ./results/

# Mix of files and folders
python analysis/extract_successful_functions.py /exp1/ /exp2/latest.pkl \
    --output ./results/

# Use only latest checkpoint from each folder
python analysis/extract_successful_functions.py /exp1/ /exp2/ /exp3/ \
    --latest-only --output ./results/

# Custom target signature
python analysis/extract_successful_functions.py /path/to/checkpoints/ \
    --target "{(6,1,2):10,(7,1,2):16,(8,1,2):30,(9,1,2):52,(10,1,2):94,(11,1,2):172}"

# Skip deduplication (faster)
python analysis/extract_successful_functions.py /path/to/checkpoints/ \
    --output ./results/ --no-dedup
```

**Outputs:**
- `successful_functions.json` - All function metadata
- `successful_functions.py` - Function bodies as Python code
- `successful_summary.txt` - Summary statistics

---

### 2. Evaluate on Extended Inputs (`evaluate_extended.py`)

Evaluate extracted priority functions on larger code lengths n to test generalization.

**Features:**
- Auto-detects function signature (no_graph, graph_gt, graph_networkx)
- Parallel evaluation across multiple workers
- **Per-(function, n) checkpointing** - crash-safe, saves after each (func, n) evaluation
- **Resume from checkpoint** - continue from exactly where it left off, even mid-function
- **Real-time progress logging** - each (func, n) size logged as it's computed

**Usage:**
```bash
# Evaluate on n=6 to n=16
python analysis/evaluate_extended.py ./results/successful_functions.json \
    --max-n 16

# Use pre-computed graphs (faster for graph_gt functions)
python analysis/evaluate_extended.py ./results/successful_functions.json \
    --graph-dir /path/to/graphs/ --max-n 16

# Limit workers and functions
python analysis/evaluate_extended.py ./results/successful_functions.json \
    --workers 8 --limit 100

# Resume from checkpoint after crash
python analysis/evaluate_extended.py ./results/successful_functions.json --resume

# Save checkpoint every 10 (func, n) evaluations (less I/O overhead)
python analysis/evaluate_extended.py ./results/successful_functions.json \
    --checkpoint-interval 10

# Disable checkpointing (faster, no crash recovery)
python analysis/evaluate_extended.py ./results/successful_functions.json \
    --no-checkpoint
```

**Outputs:**
- `evaluation_results.json` - Full evaluation results
- `evaluation_summary.txt` - Summary table
- `codebooks.json` - Generated codebooks for VT analysis
- `evaluation_checkpoint.json` - Checkpoint file (for resuming)
- `evaluation_progress.log` - Real-time progress log with sizes

---

### 3. VT Overlap Analysis (`vt_overlap_analysis.py`)

Compare generated codebooks against VT (Varshamov-Tenengolts) codes.

**Features:**
- Compare with VT_a codes and their bitwise complements (default: a=0)
- Handle VT complement symmetry: complement(VT_a) = VT_b where b = (n(n+1)/2 - a) mod (n+1)
- Group functions by overlap type (100% VT, 100% complement, partial, none)
- Supports both flat and nested VT code formats

**Usage:**
```bash
# Compare against VT_0 (default)
python analysis/vt_overlap_analysis.py ./results/codebooks.json

# Compare against VT_1
python analysis/vt_overlap_analysis.py ./results/codebooks.json --vt-a 1

# Custom VT codes path
python analysis/vt_overlap_analysis.py ./results/codebooks.json \
    --vt-path /path/to/vt_solutions.json
```

**Outputs:**
- `vt_overlap_analysis.json` - Detailed overlap analysis (includes which VT_a was used)
- `vt_overlap_report.txt` - Summary report
- `grouped_by_vt_overlap/` - Functions grouped by overlap type

---

### 4. Full Pipeline (`run_full_analysis.py`)

Run the complete analysis pipeline in one command. Supports multiple checkpoint files/folders.

**Usage:**
```bash
# Single checkpoint
python analysis/run_full_analysis.py /path/to/checkpoint.pkl \
    --output ./results/ --max-n 16

# Multiple folders (combines experiments)
python analysis/run_full_analysis.py /exp1/checkpoints/ /exp2/checkpoints/ \
    --output ./combined_results/

# Latest checkpoint from each folder
python analysis/run_full_analysis.py /exp1/ /exp2/ /exp3/ --latest-only

# Compare against VT_1 instead of VT_0
python analysis/run_full_analysis.py /path/to/checkpoints/ --vt-a 1
```

---

### 5. Generate VT Codes (`generate_vt_codes.py`)

Generate VT codes for specified n and syndrome (a) values. Updates `vt_solutions.json`.

**Output format (nested structure):**
```json
{
  "6": {
    "0": ["000000", "001011", ...],
    "1": ["000001", "001010", ...]
  },
  "7": {
    "0": [...],
    "1": [...]
  }
}
```

**Usage:**
```bash
# Generate VT_0 only (default)
python analysis/generate_vt_codes.py --max-n 25

# Generate VT_0 and VT_1
python analysis/generate_vt_codes.py --max-n 20 --a-values 0,1

# Generate all a values (0 to n for each n)
python analysis/generate_vt_codes.py --max-n 15 --all-a

# Preview without saving
python analysis/generate_vt_codes.py --max-n 25 --all-a --preview

# Custom output path
python analysis/generate_vt_codes.py --max-n 25 --output ./my_vt_codes.json
```

Note: Only generates (n, a) combinations not already in the file. Auto-migrates legacy flat format to nested.

---

### 6. Profile Evaluation Variants (`profile_evaluations.py`)

Profile and compare the three evaluation options:
- `no_graph`: On-the-fly neighbor computation
- `graph_networkx`: NetworkX graphs with simple API
- `graph_gt`: graph-tool with fast C++ backend

**Usage:**

Edit the configuration section at the top of the script:
```python
PROFILE_NO_GRAPH = True
PROFILE_NETWORKX = True
PROFILE_GRAPH_TOOL = True

GRAPH_DIR = "/path/to/graphs"
SPEC_PATH = "/path/to/spec.txt"

S_VALUES = [1]
N_VALUES = [6, 7, 8, 9, 10, 11]
Q = 2
```

Then run:
```bash
python analysis/profile_evaluations.py
```

---

### 7. Checkpoint Inspector (`checkpoint_inspector.py`)

Inspect a single checkpoint file in detail.

**Usage:**
```bash
# Full inspection
python analysis/checkpoint_inspector.py /path/to/checkpoint.pkl

# Focus on specific island
python analysis/checkpoint_inspector.py /path/to/checkpoint.pkl --island 0

# List all signatures
python analysis/checkpoint_inspector.py /path/to/checkpoint.pkl --list-signatures

# Search for specific signature
python analysis/checkpoint_inspector.py /path/to/checkpoint.pkl \
    --search-signature "{(7,1,2): 16, (8,1,2): 30}"
```

---

### 8. Checkpoint Searcher (`checkpoint_searcher.py`)

Search across multiple checkpoint files in a directory.

**Usage:**
```bash
# Find first occurrence of signature
python analysis/checkpoint_searcher.py /path/to/checkpoints/ \
    first-signature "{(6,1,2):10,(7,1,2):16}"

# Find checkpoint by prompt count
python analysis/checkpoint_searcher.py /path/to/checkpoints/ by-prompts 50000

# Compare all checkpoints
python analysis/checkpoint_searcher.py /path/to/checkpoints/ compare

# List all signatures
python analysis/checkpoint_searcher.py /path/to/checkpoints/ list-signatures
```

---

### 9. Compare Any Codebook to VT (`compare_to_vt.py`)

Unified comparison script that works with any codebook source against VT codes.

**Supports:**
- KAMIS results (.result files)
- Greedy baseline solutions (.txt files)
- Our priority function codebooks (codebooks.json)
- Plain text files (one codeword per line)

**Usage:**
```bash
# Compare KAMIS results directory
python compare_to_vt.py /path/to/kamis_results/ --mapping-dir /path/to/mappings/

# Compare greedy solutions
python compare_to_vt.py /path/to/greedy_solutions/

# Compare our priority function codebooks
python compare_to_vt.py /path/to/codebooks.json

# Compare single file
python compare_to_vt.py /path/to/solution.txt --n 10

# Compare against VT_1 instead of VT_0
python compare_to_vt.py /path/to/results/ --vt-a 1
```

**Outputs:**
- Comparison table showing overlap with VT and complement
- Match type classification (perfect_vt, perfect_complement, partial, no_overlap)
- Optional JSON output with `--output`

---

### 10. Subset Analysis (`subset_analysis.py`)

Analyze subset relationships between codebooks, e.g., "Is a 2-deletion codebook a subset of the 1-deletion VT codebook?"

**Usage:**
```bash
# Compare KAMIS s=2 results against VT s=1
python subset_analysis.py /path/to/kamis_s2_results/ --vt-path /path/to/vt_s1.json

# Compare any codebook against VT
python subset_analysis.py /path/to/codebook.txt --n 10

# Compare two codebook directories
python subset_analysis.py /path/to/s2_results/ --reference /path/to/s1_results/

# Show non-overlapping codewords
python subset_analysis.py /path/to/codebook/ --show-non-overlapping
```

**Outputs:**
- Overlap statistics per n value
- Subset/superset determination
- `subset_analysis.json` - Detailed results including non-overlapping codewords

---

### 11. Random Greedy Baseline (`baselines/random_greedy.py`)

Run multiple trials with different random seeds, building maximal independent sets by shuffling node order and greedily selecting nodes.

**Usage:**
```bash
# Run for specific n values
python baselines/random_greedy.py --n-values 6,7,8,9,10 --trials 1000

# Use pre-computed graphs (faster)
python baselines/random_greedy.py --graph-dir /path/to/graphs --n-values 10,11,12

# Build graphs on-the-fly (slower)
python baselines/random_greedy.py --n-values 6,7,8 --trials 100
```

**Outputs:**
- `greedy_s{s}_n{n}_q{q}_best.txt` - Best solution (one codeword per line)
- `greedy_summary.json` - Statistics (best, mean, std, frequency of max)

---

### Codebook Loaders (`codebook_loaders.py`)

Unified codebook loading module with auto-detection. Not meant to be run directly.

**Features:**
- Auto-detects format from file extension and content
- Supports: KAMIS (.result), text (.txt), JSON (list/dict/codebooks.json)
- Loads from files or directories
- Finds best solution when multiple seeds exist

**Functions:**
- `load_codebook(path, mapping_file, n)` - Load any format
- `load_codebooks_from_directory(dir, pattern, ...)` - Load all from directory
- `load_greedy_solutions(dir, ...)` - Load greedy baseline outputs
- `detect_format(path)` - Auto-detect file format

---

### Shared Helpers (`helpers.py`)

Common utility functions used by multiple scripts. Not meant to be run directly.

**Contents:**
- Signature detection (`detect_signature`)
- LCS computation (`lcs_length`, `are_neighbors`)
- Graph building (`build_graph_networkx`, `build_graph_gt`)
- VT utilities (`compute_vt_syndrome`, `compute_vt_complement_index`, `bitwise_complement`, `is_flat_vt_format`)
- Constants (`SIGNATURE_NO_GRAPH`, `SIGNATURE_GRAPH_GT`, `SIGNATURE_GRAPH_NETWORKX`, `COMMON_IMPORTS`)

---

## Function Signature Types

The evaluation scripts automatically detect which signature type a priority function uses:

| Signature | Function Definition | When Used |
|-----------|-------------------|-----------|
| `no_graph` | `priority(node, n, s, q)` | Functions that compute priorities from node string only |
| `graph_gt` | `priority(node, G_gt, node_to_vertex, vertex_to_node, n, s)` | Functions using graph-tool graphs |
| `graph_networkx` | `priority(node, G, n, s)` | Functions using NetworkX graphs |

---


## Checkpoint File Structure

Checkpoints are pickled dictionaries containing:

```python
{
    'best_score_per_island': List[float],
    'best_program_per_island': List[Dict],
    'best_scores_per_test_per_island': List[Dict],
    'islands_state': List[Dict],  # Contains clusters with programs
    'total_stored_programs': int,
    'total_prompts': int,
    'cumulative_evaluator_cpu_time': float,
    'cumulative_sampler_gpu_time': float,
    # ... more fields
}
```

Each island's cluster contains:
```python
{
    'score': float,
    'scores_per_test': Dict,  # e.g., {'(6, 1, 2)': 10, '(7, 1, 2)': 16}
    'programs': List[Dict],   # Each has 'body', 'args', etc.
}
```
