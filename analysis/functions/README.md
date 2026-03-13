# Function Evaluation and Generalization Pipeline

Scripts for extracting successful functions from checkpoints, evaluating their generalization to longer code lengths, and optionally analyzing overlap with Varshamov Tenengolts (VT) codes (only relevant for single deletion codes, s=1).

### Step 1: Extract Successful Functions (`extract.py`)

Extracts priority functions from checkpoint(s) that achieve a target signature, then deduplicates them based on the priority values they assign (semantic deduplication via output hash).

```bash
# Single checkpoint
python extract.py checkpoint.pkl

# Multiple checkpoints or folders
python extract.py /exp1/checkpoints/ /exp2/checkpoints/ --output ./results/

# Custom target signature
python extract.py checkpoint.pkl --target '((6,1,2),10),((7,1,2),16)'
```

Output: `successful_functions.json` containing deduplicated functions with metadata.

### Step 2: Evaluate on Extended Inputs (`evaluate.py`)

Takes the extracted functions and evaluates them on larger inputs (n=6 through n=16 by default). Tests generalization, i.e. whether functions that work well on small n also work on larger n.

```bash
# Evaluate functions
python evaluate.py ./successful_functions/successful_functions.json --max-n 16

# Resume from checkpoint
python evaluate.py ./successful_functions/successful_functions.json --resume

# Parallel evaluation
python evaluate.py ./successful_functions.json --workers 10
```

Saves checkpoints, auto detects function signature (no graph, graph tool, networkx), real time logging of codebook sizes.

Output: `evaluation_results.json` with codebook sizes for each (function, n) pair, and optionally `codebooks.json` with full codebook contents.

### Step 3: Rank Functions (`rank_evaluated.py`)

Ranks the evaluated functions by how close they get to best known solutions.

```bash
python rank_evaluated.py ./evaluate_output/evaluation_results.json --s 2 --top 10
```

Output: table of max scores per n and top K functions ranked by ratio to best known.

### Step 4: VT Code Overlap Analysis (`vt_overlap.py`)

Analyzes how the generated codebooks compare to Varshamov Tenengolts (VT) codes, the optimal single deletion correcting codes. Only relevant for s=1.

```bash
# Analyze VT overlap
python vt_overlap.py ./codebooks.json

# Compare against specific VT_a
python vt_overlap.py ./codebooks.json --vt-a 1
```

Output: overlap analysis showing percentage overlap with VT_a codes and grouping of functions by overlap pattern.

## Function Signatures

The scripts auto detect which type of priority function is used:

| Signature | Function Interface | Graph Library |
|-----------|-------------------|---------------|
| `no_graph` | `priority(node, n, s, q)` | None (LCS based) |
| `graph_networkx` | `priority(node, G, n, s)` | NetworkX |

## Output Structure

```
output_dir/
├── successful_functions.json    # Extracted and deduplicated functions
├── evaluation_results.json      # Sizes per (function, n)
├── codebooks.json               # Full codebook contents (optional)
├── evaluation_checkpoint.json   # Incremental checkpoint
└── evaluation_progress.log      # Progress log
```
