"""Brute-force decoder validation for deletion-correcting codes.

This module implements the evaluation procedure described in the paper:
1. Enumerate all codewords satisfying encoder constraints
2. Partition by Hamming weight
3. For each codeword, generate all deletion patterns (|D| <= s)
4. Compute signature Φ(x,D) = (received_word, syndrome_difference)
5. Detect collisions via hash table

For n > 22, use the C++ extension for performance.
"""

from itertools import combinations
from typing import List, Tuple, Callable, Dict, Any, Optional
import numpy as np

# Try to import C++ extension
try:
    from ._cpp import bruteforce_cpp
    _CPP_AVAILABLE = True
except ImportError:
    try:
        # Try direct import (for when running from _cpp directory)
        import bruteforce_cpp
        _CPP_AVAILABLE = True
    except ImportError:
        _CPP_AVAILABLE = False


def _int_to_bits(x: int, n: int) -> List[int]:
    """Convert integer to binary list of length n (MSB first)."""
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


def _bits_to_int(bits: List[int]) -> int:
    """Convert binary list to integer."""
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def _popcount(x: int) -> int:
    """Count number of 1 bits in x."""
    return bin(x).count('1')


def _delete_positions(bits: List[int], positions: Tuple[int, ...]) -> List[int]:
    """Delete bits at specified positions, return received word."""
    pos_set = set(positions)
    return [b for i, b in enumerate(bits) if i not in pos_set]


def _compute_syndrome_difference(
    bits: List[int],
    positions: Tuple[int, ...],
    weights: np.ndarray,
    moduli: np.ndarray
) -> Tuple[int, ...]:
    """Compute syndrome difference Δ(x, D) for deletion pattern D.

    The syndrome difference is:
    Δ_j(x,D) = Σ_{i in D} x_i * f_j(i) - shift_correction_j

    where shift_correction accounts for position shifts after deletions.

    For simplicity, we compute:
    Δ_j = (original_syndrome - syndrome_of_received_with_shifted_positions) mod m_j
    """
    n = len(bits)
    r = len(moduli)

    # Original syndrome contribution from deleted positions
    # Δ_j = Σ_{i in D} x_i * f_j(i)
    # Plus correction for position shifts
    delta = []
    for j in range(r):
        # Contribution from deleted bits
        contrib = sum(bits[i] * weights[j, i] for i in positions)

        # Shift correction: for remaining positions, their effective weight changes
        # After deleting positions D, position i > max(D∩{0..i-1}) shifts left
        pos_set = set(positions)
        shift_correction = 0
        for i in range(n):
            if i not in pos_set:
                # How many deleted positions are before i?
                shift = sum(1 for p in positions if p < i)
                # Original weight - shifted weight
                shifted_pos = i - shift
                if shifted_pos < n - len(positions):
                    shift_correction += bits[i] * (weights[j, i] - weights[j, shifted_pos])

        delta_j = (contrib + shift_correction) % moduli[j]
        delta.append(int(delta_j))

    return tuple(delta)


def validate_decoder_bruteforce_python(
    n: int,
    s: int,
    weights: np.ndarray,
    moduli: np.ndarray,
    target_syndromes: np.ndarray,
    return_collisions: bool = False
) -> Dict[str, Any]:
    """Pure Python implementation of brute-force decoder validation.

    Args:
        n: Codeword length
        s: Maximum number of deletions
        weights: r x n weight matrix where weights[j, i] = f_j(i)
        moduli: Array of r moduli
        target_syndromes: Array of r target syndrome values (σ_j)
        return_collisions: If True, return collision details

    Returns:
        Dictionary with validation results
    """
    r = len(moduli)

    # Step 1: Enumerate all codewords satisfying encoder constraints
    codewords = []
    codeword_ints = []

    for x in range(2 ** n):
        bits = _int_to_bits(x, n)
        bits_arr = np.array(bits, dtype=np.int64)

        # Check if this vector satisfies encoder constraints
        syndromes = (bits_arr @ weights.T) % moduli
        if np.all(syndromes == target_syndromes):
            codewords.append(bits)
            codeword_ints.append(x)

    codebook_size = len(codewords)

    if codebook_size == 0:
        return {
            'valid': True,
            'codebook_size': 0,
            'total_signatures': 0,
            'unique_signatures': 0,
            'collision_count': 0,
            'score': 1.0,
            'collisions': [] if return_collisions else None
        }

    # Step 2: Generate all deletion patterns (exactly s deletions)
    deletion_patterns = list(combinations(range(n), s))

    num_patterns = len(deletion_patterns)

    # Step 3: Partition codewords by Hamming weight
    weight_groups: Dict[int, List[int]] = {}
    for idx, x in enumerate(codeword_ints):
        w = _popcount(x)
        if w not in weight_groups:
            weight_groups[w] = []
        weight_groups[w].append(idx)

    # Step 4: Compute signatures and detect collisions
    # Signature: (received_word_int, received_len, delta)
    # We only need to check collisions within weight windows
    seen: Dict[Tuple, Tuple[int, int]] = {}  # signature -> (codeword_idx, pattern_idx)
    collisions = []
    total_signatures = 0
    collision_count = 0

    for cw_idx, bits in enumerate(codewords):
        for pat_idx, positions in enumerate(deletion_patterns):
            total_signatures += 1

            # Compute received word
            received = _delete_positions(bits, positions)
            received_int = _bits_to_int(received)
            received_len = len(received)

            # Compute syndrome difference
            delta = _compute_syndrome_difference(bits, positions, weights, moduli)

            # Create signature
            signature = (received_int, received_len, delta)

            if signature in seen:
                prev_cw_idx, prev_pat_idx = seen[signature]
                if prev_cw_idx != cw_idx:
                    collision_count += 1
                    if return_collisions:
                        collisions.append({
                            'codeword1': codeword_ints[prev_cw_idx],
                            'pattern1': deletion_patterns[prev_pat_idx],
                            'codeword2': codeword_ints[cw_idx],
                            'pattern2': positions,
                            'signature': signature
                        })
            else:
                seen[signature] = (cw_idx, pat_idx)

    unique_signatures = len(seen)
    valid = collision_count == 0
    score = unique_signatures / total_signatures if total_signatures > 0 else 1.0

    return {
        'valid': valid,
        'codebook': codeword_ints,  # List of codeword integers for hashing
        'codebook_size': codebook_size,
        'total_signatures': total_signatures,
        'unique_signatures': unique_signatures,
        'collision_count': collision_count,
        'score': score,
        'collisions': collisions if return_collisions else None
    }


def validate_decoder_bruteforce(
    n: int,
    s: int,
    weight_funcs: List[Callable[[int], int]],
    moduli: List[int],
    target_syndromes: List[int],
    return_collisions: bool = False,
    use_cpp: bool = True
) -> Dict[str, Any]:
    """Brute-force decoder validation.

    This function validates whether a set of encoder constraints defines a valid
    s-deletion-correcting code by checking for signature collisions.

    Args:
        n: Codeword length (blocklength)
        s: Maximum number of deletions to correct
        weight_funcs: List of r weight functions f_j(i) -> int (0-indexed)
        moduli: List of r moduli m_j
        target_syndromes: List of r target syndrome values σ_j
        return_collisions: If True, return collision details (slower)
        use_cpp: If True and available, use C++ extension

    Returns:
        Dictionary containing:
        - 'valid': True if no collisions (code is valid)
        - 'codebook_size': Number of valid codewords
        - 'total_signatures': Total number of (codeword, deletion_pattern) pairs
        - 'unique_signatures': Number of unique signatures
        - 'collision_count': Number of signature collisions
        - 'score': Fraction of unique signatures (1.0 = valid)
        - 'collisions': List of collision details (if return_collisions=True)
    """
    r = len(weight_funcs)
    assert len(moduli) == r, "Number of moduli must match number of weight functions"
    assert len(target_syndromes) == r, "Number of target syndromes must match number of weight functions"

    # Convert weight functions to numpy array
    weights = np.array([[f(i) for i in range(n)] for f in weight_funcs], dtype=np.int64)
    moduli_arr = np.array(moduli, dtype=np.int32)
    targets_arr = np.array(target_syndromes, dtype=np.int32)

    # Use C++ extension if available and requested
    if use_cpp and _CPP_AVAILABLE:
        result = bruteforce_cpp.validate_decoder_bruteforce(
            n, s, weights, moduli_arr, targets_arr, return_collisions
        )
        # Convert collision format from C++ to match Python format
        if return_collisions and result['collisions'] is not None:
            result['collisions'] = [
                {
                    'codeword1': coll['codeword1'],
                    'pattern1': tuple(coll['pattern1']),
                    'codeword2': coll['codeword2'],
                    'pattern2': tuple(coll['pattern2']),
                } for coll in result['collisions']
            ]
        return result

    # Fall back to pure Python
    return validate_decoder_bruteforce_python(
        n, s, weights, moduli_arr, targets_arr, return_collisions
    )


def vt_code_params(n: int) -> Tuple[List[Callable], List[int], List[int]]:
    """Return VT code parameters for testing.

    VT codes use f(i) = i + 1, m = n + 1, σ = 0.
    """
    weight_funcs = [lambda i: i + 1]
    moduli = [n + 1]
    target_syndromes = [0]
    return weight_funcs, moduli, target_syndromes


def helberg_ferreira_params(n: int, s: int) -> Tuple[List[Callable], List[int], List[int]]:
    """Return Helberg-Ferreira code parameters for testing.

    Uses Fibonacci-like weights for s-deletion correction.
    """
    # Compute weights using recurrence
    v = [0] * (n + s + 1)
    for i in range(1, n + 1):
        v[s + i - 1] = 1 + sum(v[s + i - 1 - j] for j in range(1, s + 1))

    weights = [v[s + i - 1] for i in range(1, n + 1)]
    u = 1 + sum(v[s + n - 1 - j] for j in range(s))

    def weight_func(i: int) -> int:
        return weights[i]

    return [weight_func], [u], [0]


if __name__ == '__main__':
    import time

    print("=" * 70)
    print("BRUTE-FORCE DECODER VALIDATION")
    print(f"C++ extension available: {_CPP_AVAILABLE}")
    print("=" * 70)

    # Test VT codes (should pass for s=1, fail for s>=2)
    print("\n--- VT Codes (should PASS for s=1) ---")
    for n in [8, 10, 12]:
        wf, mod, targets = vt_code_params(n)
        start = time.time()
        result = validate_decoder_bruteforce(n, 1, wf, mod, targets, use_cpp=False)
        elapsed = time.time() - start
        status = "PASS" if result['valid'] else "FAIL"
        print(f"n={n:2d}, s=1: [{status}] codebook={result['codebook_size']}, "
              f"signatures={result['total_signatures']}, score={result['score']:.4f}, "
              f"time={elapsed:.3f}s")

    print("\n--- VT Codes (should FAIL for s=2) ---")
    for n in [8, 10, 12]:
        wf, mod, targets = vt_code_params(n)
        start = time.time()
        result = validate_decoder_bruteforce(n, 2, wf, mod, targets, return_collisions=True, use_cpp=False)
        elapsed = time.time() - start
        status = "PASS" if result['valid'] else "FAIL"
        print(f"n={n:2d}, s=2: [{status}] codebook={result['codebook_size']}, "
              f"collisions={result['collision_count']}, score={result['score']:.4f}, "
              f"time={elapsed:.3f}s")
        if result['collisions'] and len(result['collisions']) > 0:
            coll = result['collisions'][0]
            print(f"  Example collision: cw1={bin(coll['codeword1'])}, pat1={coll['pattern1']}, "
                  f"cw2={bin(coll['codeword2'])}, pat2={coll['pattern2']}")

    print("\n--- Helberg-Ferreira Codes (should PASS for s=2) ---")
    for n in [8, 10, 12]:
        wf, mod, targets = helberg_ferreira_params(n, 2)
        start = time.time()
        result = validate_decoder_bruteforce(n, 2, wf, mod, targets, use_cpp=False)
        elapsed = time.time() - start
        status = "PASS" if result['valid'] else "FAIL"
        print(f"n={n:2d}, s=2: [{status}] codebook={result['codebook_size']}, "
              f"signatures={result['total_signatures']}, score={result['score']:.4f}, "
              f"time={elapsed:.3f}s")

    # C++ extension performance comparison
    if _CPP_AVAILABLE:
        print("\n" + "=" * 70)
        print("C++ EXTENSION PERFORMANCE")
        print("=" * 70)

        print("\n--- Python vs C++ Comparison ---")
        for n in [14, 16]:
            wf, mod, targets = vt_code_params(n)

            # Python
            start = time.time()
            result_py = validate_decoder_bruteforce(n, 1, wf, mod, targets, use_cpp=False)
            py_time = time.time() - start

            # C++
            start = time.time()
            result_cpp = validate_decoder_bruteforce(n, 1, wf, mod, targets, use_cpp=True)
            cpp_time = time.time() - start

            speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
            print(f"n={n:2d}: Python={py_time:.3f}s, C++={cpp_time:.3f}s, speedup={speedup:.1f}x")

        print("\n--- C++ Large n Benchmark ---")
        for n in [18, 20, 22]:
            wf, mod, targets = vt_code_params(n)
            start = time.time()
            result = validate_decoder_bruteforce(n, 1, wf, mod, targets, use_cpp=True)
            elapsed = time.time() - start
            status = "PASS" if result['valid'] else "FAIL"
            print(f"n={n:2d}, s=1: [{status}] codebook={result['codebook_size']:7d}, "
                  f"signatures={result['total_signatures']:10d}, time={elapsed:.3f}s")

        print("\n--- C++ s=2 Benchmark ---")
        for n in [16, 18, 20]:
            wf, mod, targets = vt_code_params(n)
            start = time.time()
            result = validate_decoder_bruteforce(n, 2, wf, mod, targets, use_cpp=True)
            elapsed = time.time() - start
            print(f"n={n:2d}, s=2: codebook={result['codebook_size']:7d}, "
                  f"signatures={result['total_signatures']:10d}, collisions={result['collision_count']:8d}, "
                  f"time={elapsed:.3f}s")
    else:
        print("\n--- Performance Benchmark (Pure Python) ---")
        for n in [14, 16, 18]:
            wf, mod, targets = vt_code_params(n)
            start = time.time()
            result = validate_decoder_bruteforce(n, 1, wf, mod, targets, use_cpp=False)
            elapsed = time.time() - start
            print(f"n={n:2d}: codebook={result['codebook_size']}, "
                  f"signatures={result['total_signatures']}, time={elapsed:.2f}s")
            if elapsed > 30:
                print("  (stopping benchmark - too slow)")
                break
