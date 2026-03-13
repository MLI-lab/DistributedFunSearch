# Graph Construction Scripts

Scripts for constructing graphs used in DistributedFunSearch experiments.

---

## Combined Script: `construct_graphs.py`

Constructs graphs for either deletion-correcting codes or IDS (Insertion/Deletion/Substitution) codes, selected via `--type`.

### Graph Types

| Type | Edge Condition | Solution |
|------|---------------|----------|
| `deletion` | Two nodes connected if they share a common subsequence of length >= n-s | Independent set = valid deletion-correcting code |
| `ids` | Two nodes connected if `edit_distance(node1, node2) < 2s + 1` | Independent set = valid code with min distance >= 2s + 1 |

In both cases, nodes are q-ary strings of length n (q=2 for binary, q=4 for DNA).

### Usage

```bash
# Build deletion graphs for n=10,11,12 with s=1, binary alphabet
python construct_graphs.py --type deletion --n 10,11,12 --s 1 --q 2

# Build IDS graphs for n=8,9,10 with s=1, quaternary alphabet (DNA)
python construct_graphs.py --type ids --n 8,9,10 --s 1 --q 4

# Build graph for n=15 with s=2, 100 workers, custom output
python construct_graphs.py --type deletion --n 15 --s 2 --workers 100 --output /mnt/Graphs

# Force streaming mode for memory-limited systems
python construct_graphs.py --type ids --n 18 --s 1 --q 4 --stream
```

Default alphabet size: q=2 for deletion, q=4 for ids.

### Output

Graphs are saved with automatic directory structure:

```
{base_dir}/{type}/{alphabet}/s{s}/graph_{prefix}_s{s}_n{n}_q{q}.lmdb
```

Where `{type}` is `deletion` or `ids`, `{prefix}` is `d` or `ids`, and `{alphabet}` is `binary` (q=2), `quaternary` (q=4), or `q{q}` for other values.

**Examples:**
```
/mnt/Graphs/deletion/binary/s1/graph_d_s1_n7_q2.lmdb
/mnt/Graphs/ids/quaternary/s1/graph_ids_s1_n8_q4.lmdb
```

**Changing output directory:** Use `--output` to specify a different base directory.

> **Note:** Construction can be slow for large n or q due to computing pairwise distances for all q^n sequences.

---

## SLURM Shell Script: `construct_graph.sh`

Wraps `construct_graphs.py` for SLURM job submission. Configure via environment variables:

```bash
GRAPH_TYPE=deletion N_VALUES=10,11,12 S_VALUE=1 Q_VALUE=2 WORKERS=90 sbatch construct_graph.sh
```

---

## Checkpointing

The script supports resuming from checkpoints after interruption:

- Checkpoints saved after each worker completes
- Default location: `<OUTPUT_DIR>/checkpoints/checkpoint_<prefix>_s{s}_n{n}_q{q}.pkl`
- Auto-detected on restart: script resumes from last checkpoint
- Delete checkpoint file to force fresh start

---

## Memory Tracking

The script includes memory monitoring:

| Phase | Description |
|-------|-------------|
| Before | Shows estimated upper bound (Hamming ball formula, IDS only) |
| During | Samples memory every 0.5s (main process + all workers) |
| After | Reports actual peak memory vs estimate |

For SLURM jobs, verify actual usage:

```bash
sacct -j <job_id> --format=JobID,MaxRSS,ReqMem,Elapsed
```
