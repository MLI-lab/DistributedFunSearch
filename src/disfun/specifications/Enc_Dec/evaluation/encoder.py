"""Encoder evaluation for deletion-correcting codes.

This module implements evaluation functions to check the validity of encoders
parameterized by weight functions and moduli.

Parameterization:
- Encoder: r constraints, each with weight function f_j and modulus m_j

For decoder validation, use bruteforce_decoder.py instead.
"""

from itertools import product, combinations
from typing import List, Tuple, Callable, Dict, Any, Optional


def compute_syndrome(bits: List[int], weight_funcs: List[Callable[[int], int]],
                     moduli: List[int]) -> Tuple[int, ...]:
    """Compute syndrome tuple for a bit vector.

    Args:
        bits: Binary vector [b_0, b_1, ..., b_{n-1}]
        weight_funcs: List of r weight functions f_j(i) -> int (0-indexed)
        moduli: List of r moduli m_j

    Returns:
        Tuple (sigma_0, sigma_1, ..., sigma_{r-1}) where
        sigma_j = sum(bits[i] * f_j(i) for i in range(n)) mod m_j
    """
    n = len(bits)
    syndrome = []
    for f_j, m_j in zip(weight_funcs, moduli):
        sigma_j = sum(bits[i] * f_j(i) for i in range(n)) % m_j
        syndrome.append(sigma_j)
    return tuple(syndrome)


def check_encoder_validity(n: int, parity_positions: List[int],
                           weight_funcs: List[Callable[[int], int]],
                           moduli: List[int]) -> Dict[str, Any]:
    """Check if all syndrome values are achievable via parity bit assignments.

    For a valid encoder, the parity bits must be able to achieve any target
    syndrome. We enumerate all 2^|P| parity assignments and check if they
    produce unique syndromes that cover the required space.

    Args:
        n: Codeword length
        parity_positions: List of parity bit positions (0-indexed)
        weight_funcs: List of r weight functions f_j(i) -> int
        moduli: List of r moduli m_j

    Returns:
        {
            'valid': True if encoder can achieve all syndrome values,
            'unique_fraction': Fraction of unique syndromes,
            'total_assignments': Total parity assignments (2^|P|),
            'unique_syndromes': Number of unique syndromes achieved,
            'required_syndromes': Product of moduli (target space size),
            'wasted_assignments': How many assignments collided
        }
    """
    p = len(parity_positions)
    total_assignments = 2 ** p

    seen_syndromes: set = set()

    for assignment in product([0, 1], repeat=p):
        bits = [0] * n
        for pos, bit in zip(parity_positions, assignment):
            bits[pos] = bit
        syndrome = compute_syndrome(bits, weight_funcs, moduli)
        seen_syndromes.add(syndrome)

    unique_syndromes = len(seen_syndromes)
    required_syndromes = 1
    for m in moduli:
        required_syndromes *= m

    wasted_assignments = total_assignments - unique_syndromes

    # Valid if parity positions can achieve ALL syndrome values
    # i.e., we cover the entire syndrome space [0, m_1) x [0, m_2) x ...
    valid = unique_syndromes >= required_syndromes

    return {
        'valid': valid,
        'unique_fraction': unique_syndromes / total_assignments,
        'coverage_fraction': unique_syndromes / required_syndromes,
        'total_assignments': total_assignments,
        'unique_syndromes': unique_syndromes,
        'required_syndromes': required_syndromes,
        'wasted_assignments': wasted_assignments
    }


def vt_code(n: int) -> Tuple[List[Callable[[int], int]], List[int]]:
    """Return Varshamov-Tenengolts code parameters.

    VT codes use a single constraint with:
    - f(i) = i + 1 (for 0-indexed positions, equivalent to 1-indexed f(i) = i)
    - m = n + 1

    Args:
        n: Codeword length

    Returns:
        (weight_funcs, moduli) tuple for use with check_encoder_validity
    """
    weight_funcs = [lambda i: i + 1]
    moduli = [n + 1]
    return weight_funcs, moduli


def helberg_ferreira_code(n: int, s: int) -> Tuple[List[Callable[[int], int]], List[int]]:
    """Return Helberg-Ferreira code parameters for s-deletion correction.

    From "On Multiple Insertion/Deletion Correcting Codes" (Helberg & Ferreira, 2002).

    The construction uses a single constraint with Fibonacci-like weights:
        v_i = 1 + v_{i-1} + v_{i-2} + ... + v_{i-s}  for i > 0
        v_i = 0                                       for i <= 0

    Modulus:
        u = 1 + v_n + v_{n-1} + ... + v_{n-s+1}

    Args:
        n: Codeword length
        s: Number of deletions to correct

    Returns:
        (weight_funcs, moduli) tuple for use with check_encoder_validity
    """
    v = [0] * (n + s + 1)

    for i in range(1, n + 1):
        v[s + i - 1] = 1 + sum(v[s + i - 1 - j] for j in range(1, s + 1))

    weights = [v[s + i - 1] for i in range(1, n + 1)]
    u = 1 + sum(v[s + n - 1 - j] for j in range(s))

    def weight_func(i: int) -> int:
        return weights[i]

    return [weight_func], [u]


def find_valid_parity_positions(
    n: int,
    num_parity: int,
    weight_funcs: List[Callable[[int], int]],
    moduli: List[int],
    max_combinations: int = 100000
) -> Optional[List[int]]:
    """Search for parity positions that give a valid encoder.

    For small n, exhaustively searches all C(n, num_parity) combinations.
    Returns the first valid set found, or None if none exist.

    Args:
        n: Codeword length
        num_parity: Number of parity bits
        weight_funcs: List of weight functions
        moduli: List of moduli
        max_combinations: Max combinations to try before giving up

    Returns:
        List of valid parity positions, or None if not found
    """
    from math import comb

    total_combinations = comb(n, num_parity)
    if total_combinations > max_combinations:
        return None  # Too many to search exhaustively

    best_positions = None
    best_unique = 0

    for positions in combinations(range(n), num_parity):
        result = check_encoder_validity(n, list(positions), weight_funcs, moduli)
        if result['valid']:
            return list(positions)  # Found valid encoder
        if result['unique_syndromes'] > best_unique:
            best_unique = result['unique_syndromes']
            best_positions = list(positions)

    return best_positions  # Return best found even if not valid


def find_minimum_parity_positions(
    n: int,
    weight_funcs: List[Callable[[int], int]],
    moduli: List[int],
    max_parity: int = 20,
    max_combinations: int = 100000
) -> Tuple[Optional[List[int]], int]:
    """Find minimum number of parity bits needed for valid encoder.

    Searches from 1 parity bit up to max_parity, returning first valid set.

    Args:
        n: Codeword length
        weight_funcs: List of weight functions
        moduli: List of moduli
        max_parity: Maximum parity bits to try
        max_combinations: Max combinations per parity count

    Returns:
        (parity_positions, num_parity) or (None, -1) if not found
    """
    from math import comb

    for num_parity in range(1, min(max_parity + 1, n)):
        if comb(n, num_parity) > max_combinations:
            continue  # Skip if too many combinations

        positions = find_valid_parity_positions(
            n, num_parity, weight_funcs, moduli, max_combinations
        )
        if positions is not None:
            # Verify it's actually valid
            result = check_encoder_validity(n, positions, weight_funcs, moduli)
            if result['valid']:
                return positions, num_parity

    return None, -1
