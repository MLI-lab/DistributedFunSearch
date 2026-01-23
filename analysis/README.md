# DistributedFunSearch Analysis Tools

Tools for analyzing checkpoint files, testing discovered priority functions on larger code lengths (n) and deletion parameters than used during the search, and comparing the resulting codebooks against VT codes.

## Quick Start

### Run the Full Test Pipeline

```bash
cd /workspace/DistributedFunSearch/analysis

# Single checkpoint
python test/run_all.py /path/to/checkpoint.pkl --output ./my_results/

# Multiple folders
python test/run_all.py /exp1/checkpoints/ /exp2/checkpoints/ --output ./combined/

# Use latest checkpoint from each folder
python test/run_all.py /exp1/ /exp2/ --latest-only
```

### Run Steps Individually

```bash
# Step 1: Extract and deduplicate successful functions from checkpoints
python test/extract.py /path/to/checkpoints/ --output ./results/

# Step 2: Evaluate on extended code lengths (e.g. n=6 to n=16)
python test/evaluate.py ./results/successful_functions.json --max-n 16

# Step 3: Compare codebooks to VT codes and group by overlap pattern
# Default compares to VT_0, use --vt-a to compare against any VT_a
# Groups: perfect_vt_all, perfect_complement_all, perfect_alternating,
#         incomplete_data (missing codebooks for some n), perfect_some,
#         partial_only, no_overlap
python test/vt_overlap.py ./results/codebooks.json
python test/vt_overlap.py ./results/codebooks.json --vt-a 1  # compare to VT_1
```

## Checkpoint Tools (`checkpoint/`)

Inspect single checkpoints or search across multiple checkpoint files.

```bash
python checkpoint/inspector.py /path/to/checkpoint.pkl
python checkpoint/inspector.py /path/to/checkpoint.pkl --island 0
python checkpoint/searcher.py /path/to/checkpoints/ first-signature "{(6,1,2):10,(7,1,2):16}"
python checkpoint/searcher.py /path/to/checkpoints/ compare
```

## VT Code Tools (`vt/`)

Generate VT codes for any syndrome value a, compare codebooks against VT_a, or analyze subset relationships.

```bash
python vt/generate.py --max-n 25                     # generate VT_0 codes
python vt/generate.py --max-n 25 --a-values 0,1,2   # generate VT_0, VT_1, VT_2
python vt/generate.py --max-n 25 --all-a            # generate all VT_a (a=0 to n)
python vt/compare.py /path/to/codebooks.json        # compare to VT_0 (default)
python vt/compare.py /path/to/codebooks.json --vt-a 1  # compare to VT_1
python vt/subset.py /path/to/codebook.txt --n 10
```

## Baselines (`baselines/`)

Random greedy independent set baseline.

```bash
python baselines/random_greedy.py --n-values 6,7,8,9,10 --trials 1000
```
