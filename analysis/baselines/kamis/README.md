# KaMIS Baseline

KaMIS (Karlsruhe Maximum Independent Set) solver for finding maximum independent sets on conflict graphs.

## Setup

### 1. Get KaMIS and apply 64-bit patch

```bash
cd analysis/baselines/kamis

# Clone KaMIS (or use: git submodule update --init --recursive)
git clone https://github.com/KarlsruheMIS/KaMIS.git
cd KaMIS
git submodule update --init --recursive

# Apply 64-bit patch for large graphs
git apply ../kamis_64bit_changes.patch
```

The warning `unable to rmdir 'mmwis/extern/KaHIP': Directory not empty` is harmless.

### 2. Compile KaMIS

Requires: cmake, make, g++

```bash
cd KaMIS
mkdir build && cd build
cmake ../ -DPORTABLE=ON -D64BITMODE=ON
make -j 10
cd ..
mkdir -p deploy
cp build/online_mis build/redumis build/graphchecker build/sort_adjacencies deploy/
```

Flags:
- `-DPORTABLE=ON`: Disables `-march=native` so binary works on any CPU (required for cluster)
- `-D64BITMODE=ON`: Enables 64-bit edge IDs for graphs with >2B edges

This creates executables in `KaMIS/deploy/` (redumis, online_mis, etc.)

### What the 64-bit patch does

The `kamis_64bit_changes.patch` modifies KaMIS to:
1. Use `uint64_t` for EdgeID when `MODE64BITEDGES` is defined (supports >2B edges)
2. Add a `PORTABLE` cmake option to disable `-march=native`
3. Fix type mismatches in KaHIP files

Without the patch and `-D64BITMODE=ON`, you'll get: `The graph is too large. Currently only 32bit supported!`

### Cluster (SLURM + enroot)

The enroot container needs build tools (cmake, make, g++) and OpenMP runtime (libgomp1). These should be installed in the container image.

The `run_kamis_baseline.sh` script automatically compiles KaMIS on the cluster node before running.


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

Results are saved as `.result` files (one 0/1 per line) and summarized in `kamis_results.json`. The final MIS size is printed to the terminal.

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

