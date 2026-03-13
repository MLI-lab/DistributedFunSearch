# Analysis

Analysis for DistributedFunSearch experiments.

## Folders

**`functions/`** extracts successful priority functions from checkpoints, evaluates them on extended code lengths (beyond what the search used), and optionally compares the resulting codebooks against VT codes.

**`baselines/`** contains baseline algorithms we compare against: lower bounds (random greedy, Gurobi exact MIS, VT codes, Helberg-Ferreira, DoDo) and upper bounds (Kulkarni-Kiyavash, LP relaxation).

**`search_diagnostics/`** analyzes debug samples from a single run to understand what happened during search: why did samples fail to parse, what runtime errors occurred, what strategies did the LLM use in successful samples.

**`reasoning/`** compares reasoning vs non-reasoning runs: code complexity from checkpoints (SLOC, cyclomatic complexity, helper functions, library usage), iteration ratios from W&B, and the reasoning models report.

**`vt/`** generates Varshamov-Tenengolts codes, compares codebooks against them, and analyzes subset relationships.

**`utils/`** shared utility modules used across analysis scripts (checkpoint loading, codebook loaders, graph paths, function signature detection, graph sparsity, duplicate ratio).
