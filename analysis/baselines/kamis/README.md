# KaMIS Baseline

KaMIS (Karlsruhe Maximum Independent Set) solver for finding maximum independent sets on conflict graphs.

## Setup

### 1. Get KaMIS submodule

After cloning this repository, fetch KaMIS:
```bash
git submodule update --init
```

### 2. Compile KaMIS

Requires: cmake, make, g++

```bash
apt-get update && apt-get install -y cmake make g++  
cd analysis/baselines/kamis/KaMIS
chmod +x compile_withcmake.sh 
./compile_withcmake.sh
```

This creates executables in `KaMIS/deploy/` (redumis, online_mis, etc.)

### Large graphs (n=11+ quaternary IDS)

The default KaMIS build uses a 32-bit signed guard check that rejects graphs with more than ~1.07 billion (undirected) edges. For large graphs like `graph_ids_s1_n11_q4` (1.5B edges), this check was patched in three files to use `unsigned int` limits (max ~2.15B undirected edges / ~4.3B directed):

- `KaMIS/extern/KaHIP/lib/io/graph_io.cpp`
- `KaMIS/lib/mis/kernel/ParFastKer/fast_reductions/extern/KaHIP/lib/io/graph_io.cpp`
- `KaMIS/lib/mis/kernel/ParFastKer/LinearTime/MIS_sigmod_pub/Graph.cpp`

These patches are already applied in this repository.

### Cluster (SLURM + enroot)

The enroot container needs `libgomp1` (OpenMP runtime) to run KaMIS. The `run_kamis_baseline.sh` script handles this via `--container-writable` and `apt-get install libgomp1`.

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

