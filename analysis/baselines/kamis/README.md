# KaMIS Baseline

KaMIS (Karlsruhe Maximum Independent Set) solver for finding maximum independent sets on conflict graphs.

## Setup

### 1. Get KaMIS

```bash
cd analysis/baselines/kamis
git clone https://github.com/KarlsruheMIS/KaMIS.git
cd KaMIS
git submodule update --init --recursive
```

### 2. Compile

Requires cmake, make, g++.

```bash
cd KaMIS
mkdir build && cd build
cmake ../
make -j 10
cd ..
mkdir -p deploy
cp build/online_mis build/redumis build/graphchecker build/sort_adjacencies deploy/
```

## Step 1: Convert LMDB to METIS (one time)

```bash
python convert_lmdb_to_metis.py --n-values 10,11,12
```

Converts our LMDB graphs to METIS format. Skips if METIS file already exists.

Options:
- `--force` overwrite existing files
- `--ultra-efficient` low memory mode for n >= 20

## Step 2: Run KaMIS

```bash
python kamis_baseline.py --n-values 10,11,12 --algorithm redumis --timeout 3600
```

Algorithms:
- `online_mis` fast greedy + local search
- `redumis` slower, higher quality (branch and bound)
- `both` run both

Options:
- `--timeout` time limit per run in seconds
- `--runs` number of runs with different seeds (default: 3)
- `--output` output directory for results

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
