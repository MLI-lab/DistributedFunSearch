"""
Baseline algorithms for comparison.

Available baselines:
- random_greedy: Random greedy independent set baseline
- kamis: KaMIS (Maximum Independent Set) algorithms
- vt_code_size: VT code size calculator
- dodo_ids_sizes: DoDo IDS code sizes from paper

Usage:
    # Random greedy baseline
    python -m analysis.baselines.random_greedy --n-values 6,7,8 --trials 1000

    # KaMIS baseline
    python -m analysis.baselines.kamis.kamis_baseline --n-values 6,7,8 --algorithm online_mis

    # VT code sizes
    python -m analysis.baselines.vt_code_size --min-n 6 --max-n 20

    # DoDo IDS sizes
    python -m analysis.baselines.dodo_ids_sizes --min-n 6 --max-n 12
"""

from .vt_code_size import vt0_size, euler_phi, odd_divisors
from .dodo_ids_sizes import dodo_rate, code_size_from_rate, get_dodo_baseline, DODO_TABLE_1_SIZES

__all__ = [
    'vt0_size',
    'euler_phi',
    'odd_divisors',
    'dodo_rate',
    'code_size_from_rate',
    'get_dodo_baseline',
    'DODO_TABLE_1_SIZES',
]
