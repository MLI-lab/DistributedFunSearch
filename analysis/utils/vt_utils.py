#!/usr/bin/env python3
"""
VT Code Utilities

Consolidated utilities for working with Varshamov-Tenengolts (VT) codes.
These functions were previously duplicated across multiple analysis scripts.
"""

import json
from typing import Dict, Set


def load_vt_codes(vt_path: str, a: int = 0) -> Dict[int, Set[str]]:
    """
    Load VT codes from JSON file.

    Handles both formats:
    - Flat (legacy): {"6": ["codewords..."], "7": [...]}
    - Nested: {"6": {"0": ["codewords..."], "1": [...]}, "7": {...}}

    Args:
        vt_path: Path to vt_solutions.json
        a: Syndrome value to load (default 0 for VT_0)

    Returns:
        Dict mapping n -> set of codewords
    """
    with open(vt_path) as f:
        vt_data = json.load(f)

    # Detect format
    is_nested = not is_flat_vt_format(vt_data)

    # Convert to dict with int keys and set values
    vt_codes = {}
    for n_str, value in vt_data.items():
        n = int(n_str)
        if is_nested:
            # Nested format: value is dict of {a_str: codewords}
            a_str = str(a)
            if a_str in value:
                vt_codes[n] = set(value[a_str])
        else:
            # Flat format: value is list of codewords (assumed to be VT_0)
            if a == 0:
                vt_codes[n] = set(value)

    return vt_codes


def is_flat_vt_format(data: dict) -> bool:
    """
    Check if VT data is in old flat format (n -> codewords) vs nested (n -> a -> codewords).

    Args:
        data: VT codes dictionary

    Returns:
        True if flat format, False if nested
    """
    if not data:
        return False
    first_value = next(iter(data.values()))
    return isinstance(first_value, list)


def compute_vt_syndrome(codeword: str) -> int:
    """
    Compute VT syndrome: sum_{i=1}^{n} i * x_i

    Args:
        codeword: Binary string like "01101"

    Returns:
        The weighted position sum
    """
    return sum((i + 1) * int(bit) for i, bit in enumerate(codeword))


def compute_vt_complement_index(n: int, a: int = 0) -> int:
    """
    Compute the VT index of the complement codebook.

    For VT_a, the complement is VT_b where b = (n(n+1)/2 - a) mod (n+1).

    This is because:
    - syndrome(complement(x)) = n(n+1)/2 - syndrome(x)
    - If x in VT_a, then syndrome(x) ≡ a (mod n+1)
    - So syndrome(complement(x)) ≡ n(n+1)/2 - a (mod n+1)

    Args:
        n: Code length
        a: Syndrome value (default 0)

    Returns:
        The complement VT index b
    """
    return (n * (n + 1) // 2 - a) % (n + 1)


def bitwise_complement(codeword: str) -> str:
    """Compute bitwise complement of a binary string."""
    return ''.join('1' if c == '0' else '0' for c in codeword)


def complement_codebook(codebook: Set[str]) -> Set[str]:
    """Compute the bitwise complement of an entire codebook."""
    return {bitwise_complement(cw) for cw in codebook}
