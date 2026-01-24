# KaMIS Baseline

KaMIS (Karlsruhe Maximum Independent Set) solver for finding maximum independent sets on conflict graphs.

## Workflow

```
LMDB Graph          METIS Graph           KaMIS Result
(our format)   -->  (integer IDs)    -->  (0/1 per node)

convert_lmdb_to_metis.py    kamis_baseline.py    get size
```

## Step 1: Convert LMDB to METIS (one-time)

```bash
python convert_lmdb_to_metis.py --n-values 10,11,12
```

Converts our LMDB graphs to METIS format. Skips if METIS file already exists.

Options:
- `--force` - Overwrite existing files
- `--ultra-efficient` - Low memory mode for n >= 20

## Step 2: Run KaMIS

```bash
python kamis_baseline.py --n-values 10,11,12 --algorithm redumis --timeout 3600
```

Algorithms:
- `online_mis` - Fast greedy + local search
- `redumis` - Slower, higher quality (branch-and-bound)
- `both` - Run both

Options:
- `--timeout` - Time limit per run in seconds
- `--runs` - Number of runs with different seeds (default: 3)
- `--output` - Output directory for results

## Step 3: Get Results

Results are saved as `.result` files (one 0/1 per line) and summarized in `kamis_results.json`.

To get solution size:
```bash
grep -c "^1$" solution.result
```

Or in Python:
```python
from analysis.baselines.kamis import get_kamis_solution_size
size = get_kamis_solution_size("solution.result")
```

## Complete Example

```bash
# Convert (skips existing)
python convert_lmdb_to_metis.py --n-values 6,7,8,9,10

# Run baseline
python kamis_baseline.py \
    --n-values 6,7,8,9,10 \
    --algorithm redumis \
    --timeout 60 \
    --output ./results

# Results printed and saved to ./results/kamis_results.json
```

## Requirements

KaMIS binaries must be compiled first:
```bash
cd KaMIS
./compile_withcmake.sh
```

This creates binaries in `KaMIS/deploy/`.
