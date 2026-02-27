# Baselines

Baseline algorithms for comparing against DistributedFunSearch results.

## Lower bounds (`lower_bounds/`)

These construct actual codes or find independent sets, giving lower bounds on achievable code sizes.

| Script | What it does |
|--------|-------------|
| `random_greedy.py` | runs random greedy independent set trials on confusability graphs |
| `gurobi_mis.py` | solves maximum independent set exactly via integer linear programming (also gives upper bound when it times out) |
| `vt_code_size.py` | computes VT code sizes using Euler's totient function |
| `dodo_ids_sizes.py` | code sizes from the DoDo paper (Table 1) |
| `helberg.py` | Helberg-Ferreira construction for multiple insertion/deletion correcting codes |

## Upper bounds (`upper_bounds/`)

These compute theoretical limits on how large a code can be.

| Script | What it does |
|--------|-------------|
| `kulkarni_kiyavash_upper_bound.py` | nonasymptotic upper bounds from Kulkarni and Kiyavash (2013) |
| `upper_bound_LP.py` | LP relaxation of the independent set problem for tighter bounds |

## KaMIS solver (`kamis/`)

Wrapper around the KaMIS maximum independent set solver (compiled C++ binary). See `kamis/README.md`.

## HPC scripts (`hpc_scripts/`)

Shell scripts for running baselines on a SLURM cluster. Also contains the Gurobi license file.
