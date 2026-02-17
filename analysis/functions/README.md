# Function Evaluation & Generalization Pipeline

This folder contains scripts for extracting successful functions from checkpoints, evaluating their generalization to longer code lengths, and optionally analyzing overlap with Varshamov-Tenengolts (VT) codes (only relevant for single-deletion codes, s=1).

### Step 1: Extract Successful Functions (`extract.py`)

Extracts all priority functions from checkpoint(s) that achieve a target signature, then deduplicates them based on the priority values they assign (semantic deduplication via output hash).

```bash
# Single checkpoint
python extract.py checkpoint.pkl

# Multiple checkpoints or folders
python extract.py /exp1/checkpoints/ /exp2/checkpoints/ --output ./results/

# Custom target signature
python extract.py checkpoint.pkl --target '((6,1,2),10),((7,1,2),16)'
```

**Output:** `successful_functions.json` containing deduplicated functions with metadata.

### Step 2: Evaluate on Extended Inputs (`evaluate.py`)

Takes the extracted functions and evaluates them on larger inputs (n=6 through n=16 by default). Tests generalization, i.e. whether functions that work well on small n also work on larger n.

```bash
# Evaluate functions
python evaluate.py ./successful_functions/successful_functions.json --max-n 16

# Resume from checkpoint (crash-safe)
python evaluate.py ./successful_functions/successful_functions.json --resume

# Parallel evaluation
python evaluate.py ./successful_functions.json --workers 10
```

Features: incremental checkpoint saving (crash safe), auto detects function signature (no graph, graph tool, networkx), real time logging of codebook sizes.

**Output:** `codebooks.json` containing the independent sets found for each (function, n) pair.

### Step 3: VT Code Overlap Analysis (`vt_overlap.py`)

Analyzes how the generated codebooks compare to Varshamov-Tenengolts (VT) codes, the optimal single deletion correcting codes.

```bash
# Analyze VT overlap
python vt_overlap.py ./codebooks.json

# Compare against specific VT_a
python vt_overlap.py ./codebooks.json --vt-a 1
```

Output: overlap analysis showing percentage overlap with VT_a codes, percentage overlap with complement of VT_a codes, and grouping of functions by overlap pattern.

### Full Pipeline (`run_all.py`)

Runs the complete pipeline in one command:

```bash
# Full analysis (including VT overlap for s=1)
python run_all.py checkpoint.pkl --output ./my_analysis/

# Skip VT analysis (for s>1 where VT codes are not relevant)
python run_all.py checkpoint.pkl --skip-vt

# Multiple experiments
python run_all.py /exp1/ /exp2/ /exp3/ --latest-only

# Skip steps if already done
python run_all.py checkpoint.pkl --skip-extract --skip-eval
```

## Function Signatures

The scripts auto-detect which type of priority function is used:

| Signature | Function Interface | Graph Library |
|-----------|-------------------|---------------|
| `no_graph` | `priority(node, n, s, q)` | None (LCS-based) |
| `graph_networkx` | `priority(node, G, n, s)` | NetworkX |
| `graph_gt` | `priority(node, G_gt, node_to_vertex, vertex_to_node, n, s)` | graph-tool |

## Output Structure

```
output_dir/
├── successful_functions.json    # Extracted & deduplicated functions
├── codebooks.json              # Evaluation results (codebooks per n)
├── codebooks_checkpoint.json   # Incremental checkpoint
├── vt_analysis.json           # VT overlap analysis
└── logs/                      # Evaluation logs
```
