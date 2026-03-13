#!/usr/bin/env python3
"""
DoDo IDS Code Sizes Baseline

Computes theoretical code sizes from the DoDo paper:
"DoDo: Explicit Constructions of Insertion/Deletion/Substitution Codes"

Note: Table 1 in the paper lists actual code sizes found which are less than
what the formula for redundancy gives. Use Table 1 values as baseline:

Single edit error correction (s=1):
    (7,1): 275
    (8,1): 900
    (9,1): 3011
    (10,1): 10414
    (11,1): 36368

Usage:
    python dodo_ids_sizes.py
    python dodo_ids_sizes.py --min-n 6 --max-n 15
"""

import argparse
import math


# Actual code sizes from Table 1 in DoDo paper (single edit correction)
DODO_TABLE_1_SIZES = {
    (7, 1): 275,
    (8, 1): 900,
    (9, 1): 3011,
    (10, 1): 10414,
    (11, 1): 36368,
}


def dodo_rate(n: int) -> float:
    """
    Compute rate r(n) from DoDo paper formula.

    r(n) = 1 - [log2(n) + log2(log2(n)) + log2(3)] / (2n)

    The redundancy is in bits, divided by 2 to convert to base 4 symbols.

    Args:
        n: Code length

    Returns:
        Rate r(n) as a float between 0 and 1
    """
    redund_bits = math.log2(n) + math.log2(math.log2(n)) + math.log2(3)
    return 1.0 - redund_bits / (2.0 * n)


def code_size_from_rate(n: int, rate: float) -> float:
    """
    Compute code size from rate.

    |C(n)| = 4^(rate * n)

    Args:
        n: Code length
        rate: Rate r(n)

    Returns:
        Code size as a float
    """
    return 4 ** (rate * n)


def get_dodo_baseline(n: int, s: int = 1) -> int:
    """
    Get DoDo baseline code size for given parameters.

    Uses Table 1 values if available, otherwise computes from formula.

    Args:
        n: Code length
        s: Number of errors to correct (default: 1)

    Returns:
        Code size (integer)
    """
    if (n, s) in DODO_TABLE_1_SIZES:
        return DODO_TABLE_1_SIZES[(n, s)]
    else:
        rate = dodo_rate(n)
        return int(round(code_size_from_rate(n, rate)))


def main():
    parser = argparse.ArgumentParser(
        description='Compute DoDo IDS code sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--min-n', type=int, default=6,
                        help='Minimum n value (default: 6)')
    parser.add_argument('--max-n', type=int, default=12,
                        help='Maximum n value (default: 12)')
    parser.add_argument('--show-table1', action='store_true',
                        help='Show Table 1 values from paper')

    args = parser.parse_args()

    if args.show_table1:
        print("DoDo Paper Table 1 - Single Edit Correction (s=1)")
        print("=" * 40)
        for (n, s), size in sorted(DODO_TABLE_1_SIZES.items()):
            print(f"  n={n}, s={s}: {size}")
        print()

    print(f"{'n':>3} | {'rate r(n)':>10} | {'|C(n)| (formula)':>18} | {'Table 1':>10}")
    print("-" * 55)

    for n in range(args.min_n, args.max_n + 1):
        r = dodo_rate(n)
        c_float = code_size_from_rate(n, r)
        c_int = int(round(c_float))
        table1_val = DODO_TABLE_1_SIZES.get((n, 1), "-")
        print(f"{n:>3} | {r:>10.4f} | {c_int:>18d} | {str(table1_val):>10}")


if __name__ == "__main__":
    main()
