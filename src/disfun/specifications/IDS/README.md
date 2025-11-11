# IDS (Insertion/Deletion/Substitution) Code Specifications

This directory contains specifications for finding error-correcting codes that can correct insertions, deletions, AND substitutions using edit distance (Levenshtein distance).

## Problem Formulation

**Objective:** Find large binary codes where all codewords have pairwise edit distance ≥ 2s + 1

**Why?** A code with minimum distance d = 2s + 1 can correct up to s errors (insertions, deletions, or substitutions).

**Example:** For s=1, we need minimum distance 3, meaning any codeword can be recovered even after 1 insertion, deletion, or substitution.

## Graph Formulation

- **Nodes:** All binary strings of length n
- **Edges:** Two nodes are connected if `edit_distance(node1, node2) < 2s + 1`
- **Independent Set:** A set of nodes with no edges between them = valid code with sufficient minimum distance

## Directory Structure

```
IDS/
├── README.md (this file)
└── StarCoder2/
    ├── load_graph/
    │   └── baseline.txt      # Loads pre-computed graphs from LMDB
    └── construct_graph/
        ├── baseline.txt       # Constructs graphs on-the-fly
        └── test.py            # Test script to verify functionality
```

## Usage

### Option 1: Using Pre-computed Graphs (Recommended for n ≥ 8)

1. **Construct graphs first:**
   ```bash
   cd src/construct_graphs
   pip install python-Levenshtein
   python construct_ids_graphs.py
   ```
   This creates `graph_ids_s{s}_n{n}.lmdb` files in `src/graphs/`

2. **Configure your experiment:**
   ```python
   from decos.config import EvaluatorConfig

   evaluator = EvaluatorConfig(
       spec_path="src/decos/specifications/IDS/StarCoder2/load_graph/baseline.txt",
       s_values=[1, 2],      # Test for s=1 and s=2
       start_n=6,
       end_n=10,
       # ... other config options
   )
   ```

3. **Run experiment:**
   ```bash
   cd src/experiments/your_experiment
   python -m decos --config-path config.py
   ```

### Option 2: Constructing Graphs On-the-Fly (For small n ≤ 7)

Use this for small values of n where graph construction is fast enough:

```python
evaluator = EvaluatorConfig(
    spec_path="src/decos/specifications/IDS/StarCoder2/construct_graph/baseline.txt",
    s_values=[1],
    start_n=6,
    end_n=7,
    # ... other config options
)
```

**Warning:** Graph construction has O(2^(2n)) complexity. For n=10, this means ~1 million pairwise comparisons per graph!

## Key Differences from Deletion-Only Codes

| Aspect | Deletion Codes | IDS Codes |
|--------|---------------|-----------|
| Distance Metric | Longest Common Subsequence | Edit Distance (Levenshtein) |
| Edge Condition | LCS(seq1, seq2) ≥ n - s | edit_distance(seq1, seq2) < 2s + 1 |
| Min Distance | n - s | 2s + 1 |
| Graph File Format | `graph_s{s}_n{n}.lmdb` | `graph_ids_s{s}_n{n}.lmdb` |
| Error Types | Deletions only | Insertions, Deletions, Substitutions |

## Testing

Test the construct_graph specification:

```bash
cd src/decos/specifications/IDS/StarCoder2/construct_graph
python test.py
```

Expected output:
```
Testing IDS construct_graph specification...

Test 1: n=6, s=1 (min distance required: 3)
  Independent set size: 8
  Hash value: <some_hash>
  Verifying independent set validity...
  Minimum pairwise distance in set: 3
  Required distance: 3
  Valid code (can correct 1 errors)
```

## Performance Notes

- **Graph construction time** scales as O(2^(2n)) due to computing all pairwise edit distances
- **Memory usage** is similar to deletion-only codes (~40MB for n=10, s=1)
- **Pre-computing graphs** is highly recommended for n ≥ 8

## Known Optimal Solutions

For reference, here are some known optimal code sizes for IDS codes:

| n | s | d=2s+1 | Optimal Size | Notes |
|---|---|--------|--------------|-------|
| 6 | 1 | 3 | 8 | |
| 7 | 1 | 3 | 16 | |
| 8 | 1 | 3 | 20 | |

(Add more as you discover them!)

## Implementation Details

The specifications use the `python-Levenshtein` package for fast edit distance computation. This package:
- Is written in C for performance
- Implements the Wagner-Fischer algorithm
- Has been added to `pyproject.toml` dependencies

## Future Enhancements

Potential improvements:
1. Add GPU-accelerated edit distance computation for very large n
2. Implement early termination if distance exceeds threshold
3. Add support for non-binary alphabets
4. Create specifications for GPT models
