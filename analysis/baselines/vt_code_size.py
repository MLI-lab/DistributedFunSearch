#!/usr/bin/env python3
"""
VT Code Size Calculator

Computes the theoretical size of VT_0(n) codes using Euler's totient function.

The formula is:
    |VT_0(n)| = (1 / (2(n+1))) * sum_{d | (n+1), d odd} phi(d) * 2^((n+1)/d)

Where phi(d) is Euler's totient function.

Usage:
    python vt_code_size.py
    python vt_code_size.py --min-n 6 --max-n 20
"""

import argparse
import math


def euler_phi(n: int) -> int:
    """
    Compute Euler's totient function phi(n).

    phi(n) = count of integers k in [1, n] where gcd(k, n) = 1

    Args:
        n: Positive integer

    Returns:
        phi(n)
    """
    result = 0
    for i in range(1, n + 1):
        if math.gcd(i, n) == 1:
            result += 1
    return result


def odd_divisors(n: int) -> list:
    """
    Returns the list of odd divisors of n.

    Args:
        n: Positive integer

    Returns:
        List of odd divisors of n
    """
    return [d for d in range(1, n + 1) if n % d == 0 and d % 2 == 1]


def vt0_size(n: int) -> int:
    """
    Calculates |VT_0(n)|.

    Formula:
        |VT_0(n)| = (1 / (2(n+1))) * sum_{d | (n+1), d odd} phi(d) * 2^((n+1)/d)

    Args:
        n: Code length

    Returns:
        Size of VT_0(n) code
    """
    n_plus_1 = n + 1
    odd_divs = odd_divisors(n_plus_1)
    summation = sum(euler_phi(d) * (2 ** (n_plus_1 // d)) for d in odd_divs)
    return summation // (2 * n_plus_1)


def main():
    parser = argparse.ArgumentParser(
        description='Compute VT_0(n) code sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--min-n', type=int, default=6,
                        help='Minimum n value (default: 6)')
    parser.add_argument('--max-n', type=int, default=20,
                        help='Maximum n value (default: 20)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    if args.json:
        import json
        result = {n: vt0_size(n) for n in range(args.min_n, args.max_n + 1)}
        print(json.dumps(result, indent=2))
    else:
        print(f"{'n':>3} | {'|VT_0(n)|':>12}")
        print("-" * 20)
        for n in range(args.min_n, args.max_n + 1):
            print(f"{n:>3} | {vt0_size(n):>12}")


if __name__ == "__main__":
    main()
