# Graph Construction Scripts

This directory contains standalone scripts for constructing graphs used in DeCoSearch experiments.

## Deletion-Correcting Code Graphs

### construct_deletions_graphs.py

Constructs graphs for finding codes that can correct deletions.

**Graph structure:**
- Nodes: q-ary strings of length n (q=2 for binary, q=4 for DNA)
- Edges: Two nodes are connected if they share a common subsequence of length >= n-s
- An independent set in this graph represents a valid deletion-correcting code

**Usage:**

1. Install required dependencies:
   ```bash
   pip install tqdm
   ```

2. Edit the `params` list and `q` value in the `__main__` block to specify which (n, s) pairs and alphabet size you want:
   ```python
   q = 4  # 2 for binary, 4 for DNA (quaternary)
   params = [
       (6, 1),  # n=6, s=1 (single deletion correction)
       (7, 1),
       # ... add more as needed
   ]
   ```

3. Run the script:
   ```bash
   cd src/construct_graphs
   python construct_deletions_graphs.py
   ```

4. Graphs will be saved to `src/graphs/` as LMDB databases with naming format:
   ```
   graph_d_s{s}_n{n}_q{q}.lmdb
   ```

**Example output:**
- `graph_d_s1_n6_q2.lmdb`: Binary code with n=6, s=1
- `graph_d_s1_n7_q4.lmdb`: DNA code with n=7, s=1

**Note:** Graph construction can be slow for large n or q values due to computing pairwise longest common subsequences (LCS) for all q^n sequences.

## IDS (Insertion/Deletion/Substitution) Code Graphs

### construct_ids_graphs.py

Constructs graphs for finding codes that can correct insertions, deletions, and substitutions.

**Graph structure:**
- Nodes: q-ary strings of length n (q=2 for binary, q=4 for DNA)
- Edges: Two nodes are connected if `edit_distance(node1, node2) < 2s + 1`
- An independent set in this graph represents a valid code with minimum distance `>= 2s + 1`

**Usage:**

1. Install required dependencies:
   ```bash
   pip install python-Levenshtein tqdm
   ```

2. Edit the `params` list and `q` value in the `__main__` block to specify which (n, s) pairs and alphabet size you want:
   ```python
   q = 4  # 2 for binary, 4 for DNA (quaternary)
   params = [
       (6, 1),  # n=6, s=1 (requires min distance 3)
       (7, 1),
       # ... add more as needed
   ]
   ```

3. Run the script:
   ```bash
   cd src/construct_graphs
   python construct_ids_graphs.py
   ```

4. Graphs will be saved to `src/graphs/` as LMDB databases with naming format:
   ```
   graph_ids_s{s}_n{n}_q{q}.lmdb
   ```

**Example output:**
- `graph_ids_s1_n6_q2.lmdb`: Binary code with n=6, s=1 (min distance 3)
- `graph_ids_s2_n10_q4.lmdb`: DNA code with n=10, s=2 (min distance 5)

**Note:** Graph construction can be slow for large n or q values due to computing pairwise edit distances for all q^n sequences.

## Using the Graphs in Experiments

To use IDS graphs in your experiments:

1. **First, construct the graphs** (see instructions above)

2. **Update your config** to use IDS specification:
   Edit `src/experiments/experiment1/config.py` and modify the `get_spec_path()` function:
   ```python
   def get_spec_path() -> str:
       # ... (docstring and path construction code)

       # Comment out the Deletions line:
       # return os.path.join(decos_base, "src", "decos", "specifications", "Deletions", "StarCoder2", "load_graph", "baseline.txt")

       # Uncomment the IDS line:
       return os.path.join(decos_base, "src", "decos", "specifications", "IDS", "StarCoder2", "load_graph", "baseline.txt")
   ```

3. **Configure your test parameters**:
   The config file already has the `EvaluatorConfig` set up. You can adjust it if needed:
   ```python
   evaluator = EvaluatorConfig(
       s_values=[1, 2],      # Number of errors to correct
       start_n=[6, 6],       # Starting code length for each s
       end_n=[10, 12],       # Ending code length for each s
       # ... other config options
   )
   ```

4. **Run your experiment**:
   ```bash
   cd src/experiments/experiment1
   python -m decos
   ```

**Note:** The config uses a helper function `get_spec_path()` that automatically constructs the full path. Simply comment/uncomment the appropriate return statement to switch between Deletions and IDS specifications.
