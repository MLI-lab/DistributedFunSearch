# Graph Construction Scripts

Scripts for constructing graphs used in DistributedFunSearch experiments.

---

## Deletion-Correcting Code Graphs

**Script:** `construct_deletions_graphs.py`

Constructs graphs for finding codes that can correct deletions.

### Graph Structure

| Property | Description |
|----------|-------------|
| Nodes | q-ary strings of length n (q=2 for binary, q=4 for DNA) |
| Edges | Two nodes connected if they share a common subsequence of length >= n-s |
| Solution | An independent set represents a valid deletion-correcting code |

### Usage

1. Edit the `params` list and `q` value in `__main__`:

```python
q = 4  # 2 for binary, 4 for DNA (quaternary)
params = [
    (6, 1),  # n=6, s=1 (single deletion correction)
    (7, 1),
    # ... add more as needed
]
```

2. Run the script:

```bash
cd src/construct_graphs
python construct_deletions_graphs.py
```

### Output

Graphs are saved with automatic directory structure:

```
{base_dir}/deletion/{alphabet}/s{s}/graph_d_s{s}_n{n}_q{q}.lmdb
```

Where `{alphabet}` is `binary` (q=2), `quaternary` (q=4), or `q{n}` for other values.

**Examples:**
```
/mnt/Graphs/deletion/binary/s1/graph_d_s1_n7_q2.lmdb
/mnt/Graphs/deletion/quaternary/s1/graph_d_s1_n7_q4.lmdb
src/graphs/deletion/binary/s2/graph_d_s2_n10_q2.lmdb
```

**Changing output directory:** Use `--output` to specify a different base directory:

```python
# Default: saves to src/graphs/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../graphs")

# Custom: save to external storage
OUTPUT_DIR = "/mnt/large_storage/graphs"
```

> **Note:** Construction can be slow for large n or q due to computing pairwise LCS for all q^n sequences.

---

## IDS (Insertion/Deletion/Substitution) Code Graphs

**Script:** `construct_ids_graphs.py`

Constructs graphs for finding codes that can correct insertions, deletions, and substitutions.

### Graph Structure

| Property | Description |
|----------|-------------|
| Nodes | q-ary strings of length n (q=2 for binary, q=4 for DNA) |
| Edges | Two nodes connected if `edit_distance(node1, node2) < 2s + 1` |
| Solution | An independent set represents a valid code with min distance >= 2s + 1 |

### Usage

1. Edit the `params` list and `q` value in `__main__`:

```python
q = 4  # 2 for binary, 4 for DNA (quaternary)
params = [
    (6, 1),  # n=6, s=1 (requires min distance 3)
    (7, 1),
    # ... add more as needed
]
```

2. Run the script:

```bash
cd src/construct_graphs
python construct_ids_graphs.py
```

### Output

Graphs are saved with automatic directory structure:

```
{base_dir}/ids/{alphabet}/s{s}/graph_ids_s{s}_n{n}_q{q}.lmdb
```

Where `{alphabet}` is `binary` (q=2), `quaternary` (q=4), or `q{n}` for other values.

**Examples:**
```
/mnt/Graphs/ids/binary/s1/graph_ids_s1_n8_q2.lmdb
/mnt/Graphs/ids/quaternary/s2/graph_ids_s2_n10_q4.lmdb
src/graphs/ids/binary/s1/graph_ids_s1_n7_q2.lmdb
```

**Changing output directory:** For large graphs (n >= 10), you may want to save to a different location with more storage. Edit `OUTPUT_DIR` in `__main__`:

```python
# Default: saves to src/graphs/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../graphs")

# Custom: save to external storage
OUTPUT_DIR = "/mnt/large_storage/graphs"
```

> **Note:** Construction can be slow for large n or q due to computing pairwise edit distances for all q^n sequences.

---

## Checkpointing

Both scripts support resuming from checkpoints after interruption:

- Checkpoints saved after each worker completes
- Default location: `<OUTPUT_DIR>/checkpoints/checkpoint_<type>_s{s}_n{n}_q{q}.pkl`
- Auto-detected on restart: script resumes from last checkpoint
- Delete checkpoint file to force fresh start

**Changing checkpoint location:** Checkpoints are stored relative to `OUTPUT_DIR`. To use a different location, change `OUTPUT_DIR` (see above) or modify `_get_checkpoint_path()`:

```python
def _get_checkpoint_path(output_dir, n, s, q):
    # Default: checkpoints inside output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    # Custom: use fast local SSD for checkpoints
    # checkpoint_dir = "/tmp/graph_checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"checkpoint_<type>_s{s}_n{n}_q{q}.pkl")
```

---

## Memory Tracking

Both scripts include memory monitoring:

| Phase | Description |
|-------|-------------|
| Before | Shows estimated upper bound (Hamming ball formula) |
| During | Samples memory every 0.5s (main process + all workers) |
| After | Reports actual peak memory vs estimate |

For SLURM jobs, verify actual usage:

```bash
sacct -j <job_id> --format=JobID,MaxRSS,ReqMem,Elapsed
```
