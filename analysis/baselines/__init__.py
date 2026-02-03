"""
Baseline algorithms for comparison.

Available baselines:
- random_greedy: Random greedy independent set baseline
- kamis: KaMIS (Maximum Independent Set) algorithms
- vt_code_size: VT code size calculator
- dodo_ids_sizes: DoDo IDS code sizes from paper
- helberg: Helberg-Ferreira multiple insertion/deletion correcting codes
- kulkarni_kiyavash_upper_bound: Nonasymptotic upper bounds for deletion correcting codes

Usage:
    # Random greedy baseline
    python -m analysis.baselines.random_greedy --n-values 6,7,8 --trials 1000

    # KaMIS baseline
    python -m analysis.baselines.kamis.kamis_baseline --n-values 6,7,8 --algorithm online_mis

    # VT code sizes
    python -m analysis.baselines.vt_code_size --min-n 6 --max-n 20

    # DoDo IDS sizes
    python -m analysis.baselines.dodo_ids_sizes --min-n 6 --max-n 12

    # Helberg-Ferreira codes
    python -m analysis.baselines.helberg

    # Kulkarni-Kiyavash upper bounds
    python -m analysis.baselines.kulkarni_kiyavash_upper_bound --table --max-n 15
"""

from .vt_code_size import vt0_size, euler_phi, odd_divisors
from .dodo_ids_sizes import dodo_rate, code_size_from_rate, get_dodo_baseline, DODO_TABLE_1_SIZES
from .kulkarni_kiyavash_upper_bound import (
    compute_upper_bound,
    upper_bound_single_deletion,
    upper_bound_multiple_deletions,
    deletion_set_size,
)

__all__ = [
    'vt0_size',
    'euler_phi',
    'odd_divisors',
    'dodo_rate',
    'code_size_from_rate',
    'get_dodo_baseline',
    'DODO_TABLE_1_SIZES',
    # Kulkarni-Kiyavash upper bounds
    'compute_upper_bound',
    'upper_bound_single_deletion',
    'upper_bound_multiple_deletions',
    'deletion_set_size',
]
