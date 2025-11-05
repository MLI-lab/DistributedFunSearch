# Deletion Code Specifications

This directory contains specifications for finding error-correcting codes that can correct deletions using longest common subsequence (LCS).

## Problem Formulation

**Objective:** Find large binary codes where all codewords share a common subsequence of length at most n - s - 1

**Why?** A code where any two codewords have LCS < n - s ensures that after s deletions, the original codeword can be uniquely recovered.

**Example:** For s=1, we need LCS ≤ n-2, meaning any codeword can be recovered even after 1 deletion.

## Graph Formulation

- **Nodes:** All binary strings of length n
- **Edges:** Two nodes are connected if `LCS(node1, node2) ≥ n - s`
- **Independent Set:** A set of nodes with no edges between them = valid deletion-correcting code

## Directory Structure

```
Deletions/
├── README.md (this file)
├── gpt/
│   ├── load_graph/
│   │   ├── baseline.txt       # Loads pre-computed graphs from LMDB
│   │   ├── prompt_*.txt       # Advanced prompts for GPT models
│   │   └── test.py            # Test script to verify functionality
│   └── construct_graph/
│       └── baseline.txt       # Constructs graphs on-the-fly
└── StarCoder2/
    ├── load_graph/
    │   ├── baseline.txt       # Loads pre-computed graphs from LMDB
    │   └── prompt_*.txt       # Advanced prompts for StarCoder2
    └── construct_graph/
        └── baseline.txt       # Constructs graphs on-the-fly
```

## Usage

### Option 1: Using Pre-computed Graphs (Recommended for n ≥ 8)

1. **Graphs should already exist** in `src/graphs/` as `graph_s{s}_n{n}.lmdb` files

2. **Configure your experiment** by editing `src/experiments/experiment1/config.py`:

   Ensure the `get_spec_path()` function returns the Deletions specification:
   ```python
   def get_spec_path() -> str:
       # ... (path construction code)

       # For StarCoder2:
       return os.path.join(decos_base, "src", "decos", "specifications",
                          "Deletions", "StarCoder2", "load_graph", "baseline.txt")

       # OR for GPT:
       # return os.path.join(decos_base, "src", "decos", "specifications",
       #                    "Deletions", "gpt", "load_graph", "baseline.txt")
   ```

3. **Configure test parameters:**
   ```python
   evaluator = EvaluatorConfig(
       s_values=[1, 2],      # Number of deletions to correct
       start_n=[6, 7],       # Starting code length for each s
       end_n=[10, 12],       # Ending code length for each s
       # ... other config options
   )
   ```

4. **Run experiment:**
   ```bash
   cd src/experiments/experiment1
   python -m decos
   ```

### Option 2: Constructing Graphs On-the-Fly (For small n ≤ 7)

Use this for small values of n where graph construction is fast enough:

```python
def get_spec_path() -> str:
    # ... (path construction code)

    # Use construct_graph instead of load_graph:
    return os.path.join(decos_base, "src", "decos", "specifications",
                       "Deletions", "StarCoder2", "construct_graph", "baseline.txt")
```

**Warning:** Graph construction has O(2^(2n)) complexity with dynamic programming for LCS computation.

### Using Advanced Prompts

The `load_graph` directories contain multiple prompt versions (`prompt_1.txt`, `prompt_3.txt`, etc.) that have been evolved through the system. To use an advanced prompt:

```python
# Change from baseline.txt to prompt_5.txt (or any other prompt)
return os.path.join(decos_base, "src", "decos", "specifications",
                   "Deletions", "gpt", "load_graph", "prompt_5.txt")
```

## Key Differences from IDS Codes

| Aspect | Deletion Codes | IDS Codes |
|--------|---------------|-----------|
| Distance Metric | Longest Common Subsequence | Edit Distance (Levenshtein) |
| Edge Condition | LCS(seq1, seq2) ≥ n - s | edit_distance(seq1, seq2) < 2s + 1 |
| Min Distance | Sequences must differ in ≥ s positions | Pairwise distance ≥ 2s + 1 |
| Graph File Format | `graph_s{s}_n{n}.lmdb` | `graph_ids_s{s}_n{n}.lmdb` |
| Error Types | Deletions only | Insertions, Deletions, Substitutions |
| Dependencies | No extra packages needed | Requires `python-Levenshtein` |

## Testing

Test the GPT construct_graph specification:

```bash
cd src/decos/specifications/Deletions/gpt/load_graph
python test.py
```

Expected output:
```
Testing Deletion-correcting code specification...

Test 1: n=6, s=1 (LCS threshold: 5)
  Independent set size: 8
  Hash value: <some_hash>
```

## Performance Notes

- **Graph construction time** scales as O(2^(2n) × n²) due to LCS computation for all pairs
- **Memory usage**: ~40MB for n=10, s=1
- **Pre-computed graphs** are available in `src/graphs/` for common (n, s) pairs

## Known Optimal Solutions

For reference, here are some known optimal code sizes for deletion-correcting codes:

| n | s | Optimal Size | Notes |
|---|---|--------------|-------|
| 6 | 1 | 8 | VT (Varshamov-Tenengolts) code |
| 7 | 1 | 16 | |
| 8 | 1 | 20 | |
| 9 | 1 | 40 | |
| 10 | 1 | 72 | |

For more optimal solutions, see `src/graphs/vt_solutions.json`

## Implementation Details

The specifications use:
- **Dynamic programming** for efficient LCS computation
- **Space-optimized DP** using only two rows instead of full n×n matrix
- **Early termination** when LCS threshold is reached

## LLM Models Supported

- **StarCoder2-15B**: Local inference, requires GPU with ~20GB VRAM
- **GPT (Azure OpenAI)**: API-based inference, no local GPU required

To switch between models, set `gpt=True` or `gpt=False` in `SamplerConfig` in your config file.

## Future Enhancements

Potential improvements:
1. Add test scripts for StarCoder2 specifications
2. Create more advanced prompts through evolutionary search
3. Implement GPU-accelerated LCS computation for very large n
4. Add support for non-binary alphabets
5. Optimize memory usage for large graph storage
