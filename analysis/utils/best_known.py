"""
Centralized best-known (optimal) code sizes for deletion-correcting codes.

All analysis scripts should import from here instead of defining their own copies.

Usage:
    from utils.best_known import get_best_known, get_all_optimal

    # Training range for s=2
    best = get_best_known(s=2)          # {(7,2):5, (8,2):7, ...}

    # Extended range for s=2 (includes n=13..18)
    extended = get_best_known(s=2, extended=True)

    # Full dict for any s value
    all_opt = get_all_optimal(s=2)      # same as extended=True
"""

# Best known / optimal code sizes
#
# Training range: values used as targets during search (what the evolutionary
# loop is optimized against).
#
# Extended range: larger n values for out-of-distribution evaluation.

BEST_KNOWN = {
    1: {
        # Training range (s=1, q=2 binary deletion-correcting codes)
        (6, 1): 10,
        (7, 1): 16,
        (8, 1): 30,
        (9, 1): 52,
        (10, 1): 94,
        (11, 1): 172,
    },
    2: {
        # Training range (s=2, q=2)
        (7, 2): 5,
        (8, 2): 7,
        (9, 2): 11,
        (10, 2): 16,
        (11, 2): 24,
        (12, 2): 37,
    },
    3: {
        # Training range (s=3, q=2)
        (8, 3): 4,
        (9, 3): 5,
        (10, 3): 6,
        (11, 3): 8,
        (12, 3): 12,
        (13, 3): 16,
    },
    "ids": {
        # Training range: quaternary (q=4) single-edit-correcting codes (IDS)
        # Best known from MIS solvers (ReduMIS/OnlineMIS)
        (6, 1): 113,
        (7, 1): 352,
        (8, 1): 1130,
        (9, 1): 3717,
    },
    "s1_2": {
        # Joint training range for s=1 (n=9-11) + s=2 (n=10-12)
        (9, 1): 52,
        (10, 1): 94,
        (11, 1): 172,
        (10, 2): 16,
        (11, 2): 24,
        (12, 2): 37,
    },
}

BEST_KNOWN_EXTENDED = {
    1: {
        # Extended range (s=1, q=2)
        (12, 1): 316,
        (13, 1): 586,
        (14, 1): 1096,
        (15, 1): 2048,
        (16, 1): 3856,
        (17, 1): 7286,
        (18, 1): 13798,
        (19, 1): 26216,
        (20, 1): 49940,
    },
    2: {
        # Extended range (s=2, q=2)
        (6, 2): 4,
        (13, 2): 56,
        (14, 2): 87,
        (15, 2): 136,
        (16, 2): 215,
        (17, 2): 349,
        (18, 2): 567,
    },
    3: {
        # Extended range (s=3, q=2)
        (6, 3): 2,
        (7, 3): 2,
        (14, 3): 22,
        (15, 3): 30,
        (16, 3): 42,
        (17, 3): 60,
        (18, 3): 71,
    },
    "ids": {
        # Extended range: quaternary (q=4) single-edit-correcting codes (IDS)
        (10, 1): 12489,
        (11, 1): 42080,
    },
    "s1_2": {
        # Extended range for joint s=1,2 search
        # s=1 held-out: n=6-8, 12-18
        (6, 1): 10, (7, 1): 16, (8, 1): 30,
        (12, 1): 316, (13, 1): 586, (14, 1): 1096,
        (15, 1): 2048, (16, 1): 3856, (17, 1): 7286, (18, 1): 13798,
        (19, 1): 26216, (20, 1): 49940,
        # s=2 held-out: n=6-9, 13-18
        (6, 2): 4, (7, 2): 5, (8, 2): 7, (9, 2): 11,
        (13, 2): 56, (14, 2): 87, (15, 2): 136,
        (16, 2): 215, (17, 2): 349, (18, 2): 567,
        # s=3 entirely unseen: n=6-18
        (6, 3): 2, (7, 3): 2, (8, 3): 4, (9, 3): 5,
        (10, 3): 6, (11, 3): 8, (12, 3): 12, (13, 3): 16,
        (14, 3): 22, (15, 3): 30, (16, 3): 42, (17, 3): 60, (18, 3): 71,
    },
}


def get_best_known(s, extended: bool = False) -> dict[tuple, int]:
    """Get best-known code sizes for a given s value.

    Args:
        s: Number of deletions (1, 2, ...).
        extended: If True, include extended-range n values too.

    Returns:
        Dict mapping (n, s) -> optimal code size.
    """
    result = dict(BEST_KNOWN.get(s, {}))
    if extended:
        result.update(BEST_KNOWN_EXTENDED.get(s, {}))
    return result


def get_all_optimal(s) -> dict[tuple, int]:
    """Get all known optimal values (training + extended) for a given s value."""
    return get_best_known(s, extended=True)


def get_training_n_range(s: int) -> list[int]:
    """Get sorted list of n values in the training range for a given s."""
    return sorted(k[0] for k in BEST_KNOWN.get(s, {}))


def get_extended_n_range(s: int) -> list[int]:
    """Get sorted list of n values in the extended range for a given s."""
    return sorted(k[0] for k in BEST_KNOWN_EXTENDED.get(s, {}))
