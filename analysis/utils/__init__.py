"""
Analysis library modules.

This package contains shared utilities for the analysis scripts:
- helpers: Function signature detection, LCS computation, graph building
- codebook_loaders: Unified codebook loading from various formats
- vt_utils: VT code utilities (loading, syndrome computation, etc.)
"""

from .helpers import (
    # Signature types
    SIGNATURE_NO_GRAPH,
    SIGNATURE_GRAPH_GT,
    SIGNATURE_GRAPH_NETWORKX,
    # Common imports
    COMMON_IMPORTS,
    # Functions
    detect_signature,
    lcs_length,
    are_neighbors,
    build_graph_networkx,
    build_graph_gt,
    load_graph_from_lmdb,
)

from .vt_utils import (
    load_vt_codes,
    is_flat_vt_format,
    compute_vt_syndrome,
    compute_vt_complement_index,
    bitwise_complement,
    complement_codebook,
)

from .codebook_loaders import (
    load_codebook,
    load_codebooks_from_directory,
    load_greedy_solutions,
    detect_format,
)
