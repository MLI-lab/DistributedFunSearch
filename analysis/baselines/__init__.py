"""
Baseline algorithms for comparison.

Lower bounds (constructive):
- lower_bounds.random_greedy: random greedy independent set
- lower_bounds.gurobi_mis: exact MIS via integer linear programming
- lower_bounds.vt_code_size: VT code size calculator
- lower_bounds.dodo_ids_sizes: DoDo IDS code sizes from paper
- lower_bounds.helberg: Helberg-Ferreira multiple insertion/deletion correcting codes
- kamis: KaMIS maximum independent set solver

Upper bounds (theoretical):
- upper_bounds.kulkarni_kiyavash_upper_bound: nonasymptotic upper bounds
- upper_bounds.upper_bound_LP: LP relaxation upper bounds
"""

from .lower_bounds.vt_code_size import vt0_size, euler_phi, odd_divisors
from .lower_bounds.dodo_ids_sizes import dodo_rate, code_size_from_rate, get_dodo_baseline, DODO_TABLE_1_SIZES
from .upper_bounds.kulkarni_kiyavash_upper_bound import (
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
    'compute_upper_bound',
    'upper_bound_single_deletion',
    'upper_bound_multiple_deletions',
    'deletion_set_size',
]
