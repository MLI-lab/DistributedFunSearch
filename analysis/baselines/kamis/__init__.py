"""
KaMIS baseline for maximum independent set computation.

This module provides wrappers around the KaMIS (Karlsruhe Maximum Independent Set)
algorithms for computing maximum independent sets on conflict graphs.

Scripts:
- convert_lmdb_to_metis.py: Convert LMDB graphs to METIS format
- kamis_baseline.py: Run KaMIS algorithms (online_mis, redumis)
- load_solution.py: Load and inspect KaMIS solutions

The KaMIS binaries should be compiled in the KaMIS/ subdirectory:
    cd KaMIS && ./compile_withcmake.sh

Usage:
    # Convert LMDB to METIS (skips if already exists)
    python -m analysis.baselines.kamis.convert_lmdb_to_metis --n-values 6,7,8

    # Run KaMIS baseline
    python -m analysis.baselines.kamis.kamis_baseline --n-values 6,7,8 --algorithm online_mis

    # Load and inspect solution
    python -m analysis.baselines.kamis.load_solution path/to/solution.result
"""

from .convert_lmdb_to_metis import (
    convert_lmdb_to_metis,
    convert_lmdb_to_metis_inmemory,
    convert_lmdb_to_metis_streaming,
    get_metis_path,
    get_lmdb_path,
)

from .kamis_baseline import (
    run_kamis_baseline,
    run_kamis_algorithm,
    parse_kamis_output,
)

from .load_solution import (
    load_solution,
    load_kamis_result,
    load_text_solution,
    get_kamis_solution_size,
)

__all__ = [
    'convert_lmdb_to_metis',
    'convert_lmdb_to_metis_inmemory',
    'convert_lmdb_to_metis_streaming',
    'get_metis_path',
    'get_lmdb_path',
    'run_kamis_baseline',
    'run_kamis_algorithm',
    'parse_kamis_output',
    'load_solution',
    'load_kamis_result',
    'load_text_solution',
]
