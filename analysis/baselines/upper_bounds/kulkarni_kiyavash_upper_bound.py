#!/usr/bin/env python3
"""
Kulkarni-Kiyavash Upper Bounds for Deletion Correcting Codes

Implementation based on:
Kulkarni, A. A., & Kiyavash, N. (2013). Nonasymptotic upper bounds for deletion correcting codes. IEEE Transactions on Information Theory, 59(8), 5115-5130.

Combined with deletion set size formulas from:
Mercier, H., Khabbazian, M., & Bhargava, V. K. (2008). On the number of subsequences when deleting symbols from a string. IEEE transactions on information theory, 54(7), 3279-3285.

Usage:
    python kulkarni_kiyavash_upper_bound.py
    python kulkarni_kiyavash_upper_bound.py --n 10 --s 2
    python kulkarni_kiyavash_upper_bound.py --table --max-n 15

Note: For s > 1, only binary alphabet (q=2) is supported since the closed-form
formulas for |D_s| are only exact for binary strings. For s=1, any alphabet works.

Supported values: s = 1, 2, 3 (closed form formulas from Mercier et al.)
  s=1: Lemma 3 (Levenshtein's result), works for any alphabet
  s=2: Lemma 4, binary only
  s=3: Lemma 5, binary only
"""

import argparse
import math
from typing import List, Tuple, Dict, Optional
from itertools import product
from functools import lru_cache


# =============================================================================
# String Analysis Utilities
# =============================================================================

def count_runs(string: Tuple[int, ...]) -> int:
    """
    Count the number of runs in a string.
    A run is a maximal contiguous subsequence of identical symbols.

    Args:
        string: Tuple of symbols (e.g., (0, 1, 1, 0, 0))

    Returns:
        Number of runs r(x)
    """
    if len(string) == 0:
        return 0

    runs = 1
    for i in range(1, len(string)):
        if string[i] != string[i-1]:
            runs += 1
    return runs


def get_run_lengths(string: Tuple[int, ...]) -> List[int]:
    """
    Get the lengths of all runs in a string.

    Args:
        string: Tuple of symbols

    Returns:
        List of run lengths
    """
    if len(string) == 0:
        return []

    lengths = []
    current_length = 1

    for i in range(1, len(string)):
        if string[i] == string[i-1]:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1
    lengths.append(current_length)

    return lengths


def count_runs_of_length_k(string: Tuple[int, ...], k: int) -> int:
    """
    Count runs of exactly length k in a string.

    Args:
        string: Tuple of symbols
        k: Target run length

    Returns:
        Number of runs of length k (r_k)
    """
    return sum(1 for length in get_run_lengths(string) if length == k)


def count_sandwiches_length_1(string: Tuple[int, ...]) -> int:
    """
    Count sandwiches of length 1 in a string.
    A sandwich of length 1 is a single symbol squeezed between two runs of
    the same symbol (which is different from the sandwiched symbol).

    Example: In "010", the middle '1' is a sandwich of length 1 (squeezed between 0s).
    In "012", there is no sandwich (0 and 2 are different symbols).

    Args:
        string: Tuple of symbols

    Returns:
        Number of sandwiches of length 1 (s_1)
    """
    if len(string) < 3:
        return 0

    # Get runs as (symbol, length) pairs
    runs = []
    current_symbol = string[0]
    current_length = 1

    for i in range(1, len(string)):
        if string[i] == current_symbol:
            current_length += 1
        else:
            runs.append((current_symbol, current_length))
            current_symbol = string[i]
            current_length = 1
    runs.append((current_symbol, current_length))

    # A sandwich of length 1 occurs when:
    # * We have a run of length 1 (the sandwich content)
    # * The adjacent runs on both sides have the same symbol (the squeezing symbol)
    # * The sandwich symbol is different from the squeezing symbol

    count = 0
    for i in range(1, len(runs) - 1):
        sandwich_symbol, sandwich_len = runs[i]
        left_symbol, _ = runs[i - 1]
        right_symbol, _ = runs[i + 1]

        # Check if this is a sandwich of length 1
        if sandwich_len == 1 and left_symbol == right_symbol:
            count += 1

    return count


def count_sandwiches_length_2(string: Tuple[int, ...]) -> int:
    """
    Count sandwiches of length 2 in a string.
    A sandwich of length 2 is a sequence of 2 symbols (one run of length 2)
    squeezed between two runs of the same symbol.

    Args:
        string: Tuple of symbols

    Returns:
        Number of sandwiches of length 2 (s_2)
    """
    if len(string) < 4:
        return 0

    # Get runs as (symbol, length) pairs
    runs = []
    current_symbol = string[0]
    current_length = 1

    for i in range(1, len(string)):
        if string[i] == current_symbol:
            current_length += 1
        else:
            runs.append((current_symbol, current_length))
            current_symbol = string[i]
            current_length = 1
    runs.append((current_symbol, current_length))

    count = 0
    for i in range(1, len(runs) - 1):
        sandwich_symbol, sandwich_len = runs[i]
        left_symbol, _ = runs[i - 1]
        right_symbol, _ = runs[i + 1]

        # Check if this is a sandwich of length 2
        if sandwich_len == 2 and left_symbol == right_symbol:
            count += 1

    return count


def get_sandwich_info(string: Tuple[int, ...]) -> Dict:
    """
    Get detailed sandwich information for formulas requiring squeezing run lengths.

    A sandwich is a sequence squeezed between two runs of the same symbol.

    Returns dictionary with:
      s_1: total sandwiches of length 1
      s_1_11: sandwiches of length 1 with both squeezing runs of length 1
      s_1_12: sandwiches of length 1 with squeezing runs of length 1 and >=2
      s_1_22: sandwiches of length 1 with both squeezing runs of length >= 2
      Similar for other lengths
    """
    info = {
        's_1': 0, 's_1_11': 0, 's_1_12': 0, 's_1_22': 0,
        's_2': 0, 's_2_11': 0, 's_2_12': 0, 's_2_22': 0,
        's_3': 0, 's_4': 0,
        's_11': 0,  # Consecutive sandwiches of length 1
        's_12': 0,  # Consecutive sandwiches of length 1 and 2
        's_1x1': 0,  # Sandwiches of length 1 separated by a single symbol
    }

    if len(string) < 3:
        return info

    # Get runs as (symbol, length) pairs
    runs = []
    current_symbol = string[0]
    current_length = 1
    for i in range(1, len(string)):
        if string[i] == current_symbol:
            current_length += 1
        else:
            runs.append((current_symbol, current_length))
            current_symbol = string[i]
            current_length = 1
    runs.append((current_symbol, current_length))

    n_runs = len(runs)

    # Find true sandwiches: sequences between runs of the same symbol
    sandwiches = []  # (position, length, left_squeeze_len, right_squeeze_len)
    for i in range(1, n_runs - 1):
        left_sym, left_len = runs[i - 1]
        _, sand_len = runs[i]
        right_sym, right_len = runs[i + 1]
        if left_sym == right_sym:  # only if squeezing symbols match
            sandwiches.append((i, sand_len, left_len, right_len))

    for _, sand_len, left_sq, right_sq in sandwiches:
        q1 = min(left_sq, right_sq)
        q2 = max(left_sq, right_sq)

        if sand_len == 1:
            info['s_1'] += 1
            if q1 == 1 and q2 == 1:
                info['s_1_11'] += 1
            elif q1 == 1:
                info['s_1_12'] += 1
            else:
                info['s_1_22'] += 1
        elif sand_len == 2:
            info['s_2'] += 1
            if q1 == 1 and q2 == 1:
                info['s_2_11'] += 1
            elif q1 == 1:
                info['s_2_12'] += 1
            else:
                info['s_2_22'] += 1
        elif sand_len == 3:
            info['s_3'] += 1
        elif sand_len == 4:
            info['s_4'] += 1

    # Count consecutive sandwiches of length 1 (adjacent run indices, diff = 1)
    for i in range(len(sandwiches) - 1):
        if sandwiches[i][1] == 1 and sandwiches[i + 1][1] == 1:
            if sandwiches[i + 1][0] == sandwiches[i][0] + 1:
                info['s_11'] += 1

    # Count sandwiches of length 1 separated by one run (diff = 2)
    for i in range(len(sandwiches) - 1):
        if sandwiches[i][1] == 1 and sandwiches[i + 1][1] == 1:
            if sandwiches[i + 1][0] == sandwiches[i][0] + 2:
                info['s_1x1'] += 1

    return info


# =============================================================================
# Deletion Set Size Formulas (Mercier et al. 2008)
# =============================================================================

def binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def deletion_set_size_d1(string: Tuple[int, ...]) -> int:
    """
    |D_1(u)| = r (number of runs)

    Levenshtein's result: the number of distinct subsequences when deleting
    one symbol equals the number of runs.
    """
    return count_runs(string)


def deletion_set_size_d2(string: Tuple[int, ...]) -> int:
    """
    |D_2(u)| = C(r+1, 2) - r_1 - s_1

    From Lemma 4 in Mercier et al.

    Notation:
        r: total number of runs in the string
           Example: "01100" has 3 runs: [0], [11], [00]

        r_1: number of runs of length exactly 1
             Example: "01100" has r_1=1 (only the first [0] has length 1)

        s_1: number of sandwiches of length 1
             A sandwich is a run that sits between two adjacent runs of the same symbol.
             Note: the formulas use adjacent runs only, not the general sandwich definition.
             Example: "010" has s_1=1 (the [1] sits between two adjacent [0] runs)
             Example: "0110" has s_1=0 (the [11] sits between [0] runs, but length is 2)
             Example: "012" has s_1=0 (the [1] is between [0] and [2], different symbols)
    """
    r = count_runs(string)
    r_1 = count_runs_of_length_k(string, 1)
    s_1 = count_sandwiches_length_1(string)

    return binomial(r + 1, 2) - r_1 - s_1


def deletion_set_size_d3(string: Tuple[int, ...]) -> int:
    """
    |D_3(u)| = C(r+2, 3) - r_2 - r_1*r - s_2
               - s_1^{11}*(r-3) - s_1^{12}*(r-2) - s_1^{22}*(r-1)

    From Lemma 5 in Mercier et al. This formula is exact for binary strings.

    Notation:
        r: total number of runs
           Example: "011000" has 3 runs: [0], [11], [000]

        r_k: number of runs of length exactly k
             Example: "011000" has r_1=1, r_2=1, r_3=1

        s_k: number of sandwiches of length k
             A sandwich is a run between two adjacent runs of the same symbol.
             Note: the formulas use adjacent runs only, not the general sandwich definition.
             Example: "0110" has s_2=1 (the [11] is between two adjacent [0] runs)
             Example: "01110" has s_3=1 (the [111] is between two adjacent [0] runs)

        s_1^{ij}: sandwiches of length 1, categorized by the lengths of the
                  two surrounding (squeezing) runs. From the paper's notation:
                  * 1 (no bar) means the squeezing run has length exactly 1
                  * 2 (with bar, written as 2̄ in the paper) means length >= 2
                  So: s_1^{11} = both exactly 1
                      s_1^{12} = one exactly 1, one at least 2
                      s_1^{22} = both at least 2

             Example: "010" has s_1^{11}=1
                      The [1] is sandwiched between [0] (length 1) and [0] (length 1)

             Example: "0100" has s_1^{12}=1
                      The [1] is sandwiched between [0] (length 1) and [00] (length 2)

             Example: "00100" has s_1^{22}=1
                      The [1] is sandwiched between [00] (length 2) and [00] (length 2)
    """
    r = count_runs(string)
    r_1 = count_runs_of_length_k(string, 1)
    r_2 = count_runs_of_length_k(string, 2)

    sand_info = get_sandwich_info(string)
    s_2 = sand_info['s_2']
    s_1_11 = sand_info['s_1_11']
    s_1_12 = sand_info['s_1_12']
    s_1_22 = sand_info['s_1_22']

    result = binomial(r + 2, 3) - r_2 - r_1 * r
    result -= s_2
    result -= s_1_11 * (r - 3)
    result -= s_1_12 * (r - 2)
    result -= s_1_22 * (r - 1)

    return max(1, result)  # Ensure at least 1


def deletion_set_size(string: Tuple[int, ...], s: int) -> int:
    """
    Compute |D_s(x)|, the size of the deletion set for s deletions.

    Uses closed form formulas from Mercier et al. (2008):
        Lemma 3: |D_1| = r
        Lemma 4: |D_2| = C(r+1,2) - r_1 - s_1
        Lemma 5: |D_3| = C(r+2,3) - r_2 - r_1*r - s_2
                        - s_1^{11}*(r-3) - s_1^{12}*(r-2) - s_1^{22}*(r-1)

    Note: These formulas are exact for BINARY strings (alphabet {0,1}).
    For non-binary alphabets, use the DP algorithm in Section IV of the paper.

    Args:
        string: Tuple of binary symbols (0s and 1s)
        s: Number of deletions (must be 1, 2, or 3)

    Returns:
        Size of the deletion set for s deletions

    Raises:
        ValueError: If s > 3 or string is not binary
    """
    n = len(string)

    # Verify binary alphabet
    if any(x not in (0, 1) for x in string):
        raise ValueError("Deletion set size formulas only valid for binary strings (alphabet {0,1})")

    if s == 0:
        return 1
    if s >= n:
        return 1

    if s == 1:
        return deletion_set_size_d1(string)
    if s == 2:
        return deletion_set_size_d2(string)
    if s == 3:
        return deletion_set_size_d3(string)

    raise ValueError(f"Deletion set size computation only supported for s <= 3, got s={s}")


# =============================================================================
# Upper Bound Computations (Kulkarni-Kiyavash 2013)
# =============================================================================

def upper_bound_single_deletion(n: int, q: int = 2) -> float:
    """
    Closed form upper bound for single deletion correcting codes.

    Theorem 3.1: |C*_{q,1,n}| <= (q^n - q) / ((q-1)(n-1))

    Args:
        n: String length
        q: Alphabet size (default: 2 for binary)

    Returns:
        Upper bound on code size
    """
    if n <= 1:
        return float(q ** n)

    return (q ** n - q) / ((q - 1) * (n - 1))


def upper_bound_multiple_deletions(n: int, s: int, q: int = 2,
                                    verbose: bool = False) -> float:
    """
    Upper bound for multiple deletion correcting codes (s deletions).

    Theorem 4.1: |C*_{q,s,n}| <= sum_{x in F_q^{n-s}} 1/|D_s(x)|

    Note: Only supports binary alphabet (q=2) for s > 1 since the closed-form
    formulas for |D_s| are only exact for binary strings.

    Args:
        n: String length
        s: Number of deletions (must be 1, 2, or 3)
        q: Alphabet size (must be 2 for s > 1)
        verbose: Print progress

    Returns:
        Upper bound on code size
    """
    if s <= 0:
        return float(q ** n)

    if s == 1:
        return upper_bound_single_deletion(n, q)

    if s > 3:
        raise ValueError(f"Upper bound computation only supported for s <= 3, got s={s}")

    if q != 2:
        raise ValueError(f"For s > 1, only binary alphabet (q=2) is supported, got q={q}")

    if n <= s:
        return 1.0

    # Length of strings in F_q^{n-s}
    string_length = n - s
    total_strings = q ** string_length

    if verbose:
        print(f"Computing upper bound: n={n}, s={s}, q={q}")
        print(f"Enumerating {total_strings} strings of length {string_length}")

    upper_bound = 0.0

    # Enumerate all binary strings of length n-s
    for i, string in enumerate(product(range(q), repeat=string_length)):
        d_s = deletion_set_size(string, s)
        upper_bound += 1.0 / d_s

        if verbose and (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1}/{total_strings} strings processed")

    return upper_bound


def compute_upper_bound(n: int, s: int, q: int = 2, verbose: bool = False) -> float:
    """
    Compute the Kulkarni-Kiyavash upper bound for deletion correcting codes.

    Args:
        n: String length
        s: Number of deletions (must be 1, 2, or 3)
        q: Alphabet size
        verbose: Print progress

    Returns:
        Upper bound on code size
    """
    if s == 1:
        return upper_bound_single_deletion(n, q)
    else:
        return upper_bound_multiple_deletions(n, s, q, verbose)


# =============================================================================
# Comparison and Analysis
# =============================================================================

def levenshtein_asymptotic_bound(n: int, s: int, q: int = 2) -> float:
    """
    Levenshtein's asymptotic bound (for comparison).

    For s deletions: |C*| ~ (s! * q^n) / (n^s)
    """
    return math.factorial(s) * (q ** n) / (n ** s)


def compute_comparison_table(max_n: int, s_values: List[int], q: int = 2,
                              verbose: bool = True) -> Dict:
    """
    Compute comparison table of upper bounds for various n and s.

    Args:
        max_n: Maximum string length
        s_values: List of s values to compute
        q: Alphabet size
        verbose: Print progress

    Returns:
        Dictionary of results
    """
    results = {}

    for s in s_values:
        if s > 5:
            print(f"Skipping s={s} (formulas only available for s <= 5)")
            continue

        if verbose:
            print(f"\nComputing bounds for s={s}:")

        min_n = s + 2  # Minimum viable n

        for n in range(min_n, max_n + 1):
            # Check if computation is feasible
            if s > 1 and q ** (n - s) > 10_000_000:
                if verbose:
                    print(f"  n={n}: Skipping (too many strings: {q**(n-s)})")
                continue

            try:
                bound = compute_upper_bound(n, s, q, verbose=False)
                lev_bound = levenshtein_asymptotic_bound(n, s, q)

                results[(n, s)] = {
                    'kulkarni_kiyavash': bound,
                    'levenshtein_asymptotic': lev_bound,
                    'ratio': bound / lev_bound if lev_bound > 0 else float('inf')
                }

                if verbose:
                    print(f"  n={n:2d}: KK={bound:12.2f}, Lev={lev_bound:12.2f}, "
                          f"ratio={bound/lev_bound:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  n={n}: Error - {e}")

    return results


def print_upper_bound_table(max_n: int, s_values: List[int], q: int = 2):
    """Print a formatted table of upper bounds."""
    print("\n" + "=" * 80)
    print(f"KULKARNI-KIYAVASH UPPER BOUNDS FOR {q}-ARY DELETION CORRECTING CODES")
    print("=" * 80)

    # Filter to supported s values
    s_values = [s for s in s_values if s <= 5]

    header = f"{'n':^6}"
    for s in s_values:
        header += f" | {'s='+str(s):^15}"
    print(header)
    print("-" * len(header))

    for n in range(4, max_n + 1):
        row = f"{n:^6}"
        for s in s_values:
            if n <= s + 1:
                row += f" | {'-':^15}"
            elif s > 1 and q ** (n - s) > 1_000_000:
                row += f" | {'(too large)':^15}"
            else:
                try:
                    bound = compute_upper_bound(n, s, q)
                    row += f" | {bound:^15.2f}"
                except Exception:
                    row += f" | {'error':^15}"
        print(row)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute Kulkarni-Kiyavash upper bounds for deletion correcting codes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--n', type=int, default=None,
                        help='String length n')
    parser.add_argument('--s', type=int, default=1,
                        help='Number of deletions s (default: 1, max: 5)')
    parser.add_argument('--q', type=int, default=2,
                        help='Alphabet size q (default: 2 for binary)')
    parser.add_argument('--table', action='store_true',
                        help='Print comparison table')
    parser.add_argument('--max-n', type=int, default=15,
                        help='Maximum n for table (default: 15)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    if args.s > 5:
        print("Error: s must be <= 5 (formulas only available for s <= 5)")
        return

    if args.table:
        # Print comparison table
        s_values = [1, 2, 3, 4, 5]
        print_upper_bound_table(args.max_n, s_values, args.q)

        print("\n\nDetailed comparison with Levenshtein's asymptotic bound:")
        compute_comparison_table(args.max_n, [1, 2, 3], args.q, verbose=args.verbose)

    elif args.n is not None:
        # Compute single bound
        bound = compute_upper_bound(args.n, args.s, args.q, verbose=args.verbose)

        if args.json:
            import json
            result = {
                'n': args.n,
                's': args.s,
                'q': args.q,
                'upper_bound': bound,
                'upper_bound_floor': int(bound)
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\nKulkarni-Kiyavash Upper Bound")
            print(f"{'='*40}")
            print(f"Parameters: n={args.n}, s={args.s}, q={args.q}")
            print(f"Upper bound: {bound:.6f}")
            print(f"Floor:       {int(bound)}")

            # Compare with Levenshtein's asymptotic bound
            lev = levenshtein_asymptotic_bound(args.n, args.s, args.q)
            print(f"\nLevenshtein asymptotic: {lev:.6f}")
            print(f"Ratio (KK/Lev): {bound/lev:.6f}")
    else:
        # Default: show example computations
        print("=" * 70)
        print("KULKARNI-KIYAVASH UPPER BOUNDS FOR DELETION CORRECTING CODES")
        print("=" * 70)

        print("\n[Single Deletion (s=1), Closed Form]")
        print("-" * 50)
        print(f"Formula: |C*| <= (q^n - q) / ((q-1)(n-1))")
        print()
        for n in range(6, 21):
            bound = upper_bound_single_deletion(n, q=2)
            print(f"  n={n:2d}, q=2: upper bound = {bound:12.2f}")

        print("\n[Multiple Deletions (s=2), Summation]")
        print("-" * 50)
        print("Formula: |C*| <= sum_{x} 1/|D_s(x)|")
        print()
        for n in range(6, 16):
            bound = compute_upper_bound(n, s=2, q=2)
            print(f"  n={n:2d}, s=2, q=2: upper bound = {bound:12.2f}")

        print("\n[Comparison Table]")
        print_upper_bound_table(12, [1, 2, 3], q=2)


if __name__ == "__main__":
    main()
