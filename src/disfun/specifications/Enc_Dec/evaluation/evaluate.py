"""Combined encoder/decoder evaluation for deletion-correcting codes.

Evaluation pipeline:
1. Encoder check (fast, O(n^s)): enumerate parity assignments, check unique syndromes
2. Decoder check (always runs): check signature collisions

Returns dict:
    encoder_score: float in [0, 1], 1.0 = valid encoder
    decoder_score: float in [0, 1], 1.0 = valid decoder
    hash: str, hash of (codebook, parity_positions) for deduplication
    collisions: list of collision dicts (up to 10)

Hashing: Based on codebook + parity positions, so different priority functions
that produce the same codebook are deduplicated correctly.
"""

import sys
import hashlib
from typing import List, Tuple, Callable, Dict, Any
from math import log2, ceil

from .encoder import check_encoder_validity
from .bruteforce_decoder import validate_decoder_bruteforce


def priority(n: int, s: int) -> Tuple[List[Callable], List[int], List[int]]:
    """Returns (weight_funcs, moduli, parity_positions) for the encoder. a{score}"""
    pass


def evaluate(params) -> Dict[str, Any]:
    """Evaluate encoder/decoder for given (n, s) parameters.

    Args:
        params: Tuple (n, s) - codeword length and number of deletions

    Returns:
        dict with keys: encoder_score, decoder_score, hash, collisions
    """
    n, s = params
    return solve(n, s)


def solve(n: int, s: int) -> Dict[str, Any]:
    """Run encoder/decoder evaluation pipeline.

    Returns:
        {
            'encoder_score': float,  # 0-1, 1.0 = valid encoder
            'decoder_score': float,  # 0-1, 1.0 = valid decoder
            'hash': str or None,  # hash for deduplication
            'collisions': list  # collision details, up to 10
        }
    """
    # Generate encoding parameters
    try:
        weight_funcs, moduli, parity_positions = priority(n, s)
    except Exception as e:
        print(f"priority function failed: {e}", file=sys.stderr)
        return {'encoder_score': 0.0, 'decoder_score': None, 'hash': None, 'collisions': []}

    # Validate inputs
    r = len(weight_funcs)
    if len(moduli) != r:
        print(f"Moduli count mismatch: {len(moduli)} != {r}", file=sys.stderr)
        return {'encoder_score': 0.0, 'decoder_score': None, 'hash': None, 'collisions': []}

    if len(parity_positions) == 0:
        print("No parity positions specified", file=sys.stderr)
        return {'encoder_score': 0.0, 'decoder_score': None, 'hash': None, 'collisions': []}

    # Check redundancy constraint: |P| <= s * ceil(log2(n))
    max_parity = s * ceil(log2(n)) if n > 1 else s
    if len(parity_positions) > max_parity:
        print(f"Redundancy too high: |P|={len(parity_positions)}, max={max_parity}", file=sys.stderr)
        return {'encoder_score': 0.0, 'decoder_score': None, 'hash': None, 'collisions': []}

    # Check parity positions are valid indices
    if any(p < 0 or p >= n for p in parity_positions):
        print(f"Invalid parity positions: must be in [0, {n-1}]", file=sys.stderr)
        return {'encoder_score': 0.0, 'decoder_score': None, 'hash': None, 'collisions': []}

    if len(set(parity_positions)) != len(parity_positions):
        print("Duplicate parity positions", file=sys.stderr)
        return {'encoder_score': 0.0, 'decoder_score': None, 'hash': None, 'collisions': []}

    # Step 1: Encoder check (fast)
    encoder_result = check_encoder_validity(n, parity_positions, weight_funcs, moduli)
    encoder_score = encoder_result['unique_fraction']  # 1.0 = valid encoder

    # Step 2: Decoder check (always run, even if encoder invalid - gives more signal)
    targets = [0] * r  # Fixed to zero
    decoder_result = validate_decoder_bruteforce(
        n, s, weight_funcs, moduli, targets, return_collisions=True
    )

    # Get codebook from decoder result (already computed, no duplication)
    codebook = decoder_result['codebook']
    hash_value = hash_encoding(n, s, codebook, parity_positions)

    collision_count = decoder_result['collision_count']
    total_sigs = decoder_result['total_signatures']

    # Decoder score: 1 - collision_rate (1.0 = valid decoder)
    collision_rate = collision_count / total_sigs if total_sigs > 0 else 0.0
    decoder_score = 1.0 - collision_rate

    # Format collisions as list of dicts with bit strings
    collisions_list = []
    raw_collisions = decoder_result.get('collisions', [])
    for coll in raw_collisions[:10]:
        collisions_list.append({
            'cw1': format(coll['codeword1'], f'0{n}b'),
            'cw2': format(coll['codeword2'], f'0{n}b'),
            'del1': list(coll['pattern1']),
            'del2': list(coll['pattern2'])
        })

    # Print summary
    print(f"Encoder score: {encoder_score:.4f}", file=sys.stderr)
    print(f"Decoder score: {decoder_score:.6f}", file=sys.stderr)
    print(f"Hash: {hash_value[:16]}...", file=sys.stderr)
    if collision_count > 0:
        print(f"Collisions: {collision_count} total, returning first {len(collisions_list)}", file=sys.stderr)

    return {
        'encoder_score': encoder_score,
        'decoder_score': decoder_score,
        'hash': hash_value,
        'collisions': collisions_list
    }


def hash_encoding(n: int, s: int, codebook: List[int],
                  parity_positions: List[int]) -> str:
    """Generate hash based on codebook + parity positions for deduplication.

    Two different priority functions that produce the same codebook and use
    the same parity positions are functionally equivalent and should hash identically.
    """
    # Codebook should already be sorted (enumerated in order)
    parts = [
        f"n={n},s={s}",
        f"codebook={codebook}",
        f"parity={sorted(parity_positions)}"
    ]
    encoding_str = "|".join(parts)
    return hashlib.sha256(encoding_str.encode()).hexdigest()
