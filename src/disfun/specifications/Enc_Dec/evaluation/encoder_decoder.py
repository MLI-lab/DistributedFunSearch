"""Encoder/Decoder evaluation for deletion-correcting codes.

This module implements evaluation functions to check the validity of encoder/decoder
pairs parameterized by weight functions and moduli, as described in the paper.

Parameterization:
- Encoder: r constraints, each with weight function f_j and modulus m_j
- Decoder: Matrix A with columns (f_0(i), ..., f_{r-1}(i))^T

VT codes are a special case with r=1, f(i)=i, m=n+1.
"""

from itertools import combinations, product
from collections import defaultdict, Counter
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
            'num_collisions': Number of syndromes that appeared more than once
        }
    """
    p = len(parity_positions)
    total_assignments = 2 ** p

    # Just use set - fastest (only hashing, no counting)
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

    # Valid if no wasted assignments (all unique)
    valid = wasted_assignments == 0

    return {
        'valid': valid,
        'unique_fraction': unique_syndromes / total_assignments,
        'total_assignments': total_assignments,
        'unique_syndromes': unique_syndromes,
        'required_syndromes': required_syndromes,
        'wasted_assignments': wasted_assignments
    }


def check_decoder_validity(n: int, s: int, weight_funcs: List[Callable[[int], int]],
                           moduli: List[int], return_details: bool = True) -> Dict[str, Any]:
    """Check if all deletion patterns lead to distinct syndrome differences.

    Patterns that delete only zeros (all bits=0) are skipped. Only patterns with at least
    one deleted 1-bit are checked for collisions.

    Args:
        n: Codeword length
        s: Number of deletions
        weight_funcs: List of r weight functions f_j(i) -> int
        moduli: List of r moduli m_j
        return_details: If True, return full collision details (slower)

    Returns:
        {
            'valid': True if no collisions among non-zero patterns,
            'unique_fraction': Fraction of unique Av values,
            'total_patterns': Patterns checked (excludes zero-only),
            'unique_syndromes': Number of unique Av values,
            'wasted_patterns': How many patterns collided,
            'collisions': (only if return_details=True) collision details
        }
    """
    total_patterns = 0

    if return_details:
        # Slower: store patterns for collision details
        av_to_patterns: Dict[Tuple[int, ...], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = defaultdict(list)

        for positions in combinations(range(n), s):
            for bits in product([0, 1], repeat=s):
                # Skip zero-only patterns (handled by shift correction)
                if sum(bits) == 0:
                    continue

                total_patterns += 1
                v = [0] * n
                for pos, bit in zip(positions, bits):
                    v[pos] = bit
                Av = compute_syndrome(v, weight_funcs, moduli)
                av_to_patterns[Av].append((positions, bits))

        unique_syndromes = len(av_to_patterns)
        collisions = [(Av, patterns) for Av, patterns in av_to_patterns.items() if len(patterns) > 1]
        collisions.sort(key=lambda x: x[0])
        wasted = total_patterns - unique_syndromes

        return {
            'valid': wasted == 0,
            'unique_fraction': unique_syndromes / total_patterns if total_patterns > 0 else 1.0,
            'total_patterns': total_patterns,
            'unique_syndromes': unique_syndromes,
            'wasted_patterns': wasted,
            'collisions': collisions
        }
    else:
        # Fast path: just use set
        seen_syndromes: set = set()

        for positions in combinations(range(n), s):
            for bits in product([0, 1], repeat=s):
                # Skip zero-only patterns (handled by shift correction)
                if sum(bits) == 0:
                    continue

                total_patterns += 1
                v = [0] * n
                for pos, bit in zip(positions, bits):
                    v[pos] = bit
                Av = compute_syndrome(v, weight_funcs, moduli)
                seen_syndromes.add(Av)

        unique_syndromes = len(seen_syndromes)
        wasted = total_patterns - unique_syndromes

        return {
            'valid': wasted == 0,
            'unique_fraction': unique_syndromes / total_patterns if total_patterns > 0 else 1.0,
            'total_patterns': total_patterns,
            'unique_syndromes': unique_syndromes,
            'wasted_patterns': wasted
        }


def vt_code(n: int) -> Tuple[List[Callable[[int], int]], List[int]]:
    """Return Varshamov-Tenengolts code parameters.

    VT codes use a single constraint with:
    - f(i) = i + 1 (for 0-indexed positions, equivalent to 1-indexed f(i) = i)
    - m = n + 1

    The +1 offset is crucial: with f(0) = 0, deleting any bit at position 0
    would give Av=0, colliding with zero-only patterns. With f(i) = i + 1,
    position 0 has weight 1, ensuring all 1-bit deletions are distinguishable.

    Args:
        n: Codeword length

    Returns:
        (weight_funcs, moduli) tuple for use with check_* functions
    """
    # f(i) = i + 1 so that position 0 has weight 1 (not 0)
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

    For s=1: reduces to VT codes (v_i = i, u = n+1)
    For s=2: Fibonacci-like weights

    Args:
        n: Codeword length
        s: Number of deletions to correct

    Returns:
        (weight_funcs, moduli) tuple for use with check_* functions
    """
    # Compute weights using recurrence: v_i = 1 + sum of previous s terms
    # Use 1-indexed internally, then shift to 0-indexed for output
    v = [0] * (n + s + 1)  # v[0..s-1] = 0 (base cases for i <= 0)

    for i in range(1, n + 1):
        # v_i = 1 + v_{i-1} + v_{i-2} + ... + v_{i-s}
        # In our array: index i corresponds to position i (1-indexed)
        v[s + i - 1] = 1 + sum(v[s + i - 1 - j] for j in range(1, s + 1))

    # Extract weights for positions 1..n (1-indexed in paper)
    weights = [v[s + i - 1] for i in range(1, n + 1)]

    # Compute modulus: u = 1 + v_n + v_{n-1} + ... + v_{n-s+1}
    u = 1 + sum(v[s + n - 1 - j] for j in range(s))

    # Create weight function (0-indexed: position i in code maps to weights[i])
    def weight_func(i: int) -> int:
        return weights[i]

    return [weight_func], [u]


if __name__ == '__main__':
    # VT codes for s=1, n=6..15
    print("=" * 60)
    print("VT CODES (s=1)")
    print("=" * 60)
    for n in range(6, 16):
        wf, mod = vt_code(n)
        r = check_decoder_validity(n, 1, wf, mod, return_details=True)
        status = "PASS" if r['valid'] else "FAIL"
        print(f"n={n:2d}: [{status}] {r['unique_syndromes']}/{r['total_patterns']} unique")
        if not r['valid'] and r['collisions']:
            for Av, patterns in r['collisions'][:2]:
                print(f"       Av={Av}: {patterns[:3]}")

    # Helberg-Ferreira for s=2
    print("\n" + "=" * 60)
    print("HELBERG-FERREIRA (s=2)")
    print("Format: {p0=1, p2=0} means deleted positions 0,2 with values 1,0")
    print("Each line = patterns that give same Av (collide)")
    print("=" * 60)
    for n in range(6, 12):
        wf, mod = helberg_ferreira_code(n, 2)
        weights = [wf[0](i) for i in range(n)]
        r = check_decoder_validity(n, 2, wf, mod, return_details=True)
        status = "PASS" if r['valid'] else "FAIL"
        print(f"n={n:2d}: [{status}] {r['unique_syndromes']}/{r['total_patterns']} unique, mod={mod[0]}")
        print(f"       weights={weights}")
        if not r['valid'] and r['collisions']:
            print(f"       Patterns with same Av:")
            for _, patterns in r['collisions']:
                # Format: {pos0=val0, pos1=val1}
                pats = ", ".join("{" + ", ".join(f"p{p}={b}" for p, b in zip(pos, bits)) + "}" for pos, bits in patterns)
                print(f"         {pats}")
