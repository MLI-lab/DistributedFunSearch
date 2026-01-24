"""
Graph Path Configuration

Centralized paths for graph files. Structure:
    /mnt/Graphs/deletion/binary/s1/graph_d_s1_n10_q2.lmdb

Usage:
    from utils.graph_paths import get_graph_path, DEFAULT_GRAPH_DIR
    path = get_graph_path("deletion", s=1, n=10, q=2)
"""

import os

DEFAULT_GRAPH_DIR = "/mnt/Graphs"


def _alphabet_name(q: int) -> str:
    return "binary" if q == 2 else "quaternary" if q == 4 else f"q{q}"


def _prefix(graph_type: str) -> str:
    return "graph_d" if graph_type in ("deletion", "deletions", "d") else "graph_ids"


def get_lmdb_path(graph_type: str, s: int, n: int, q: int, base_dir: str = None) -> str:
    """Get LMDB graph path: deletion/binary/s1/graph_d_s1_n10_q2.lmdb"""
    base = base_dir or DEFAULT_GRAPH_DIR
    subdir = "deletion" if graph_type in ("deletion", "deletions", "d") else "ids"
    return os.path.join(base, subdir, _alphabet_name(q), f"s{s}", f"{_prefix(graph_type)}_s{s}_n{n}_q{q}.lmdb")


def get_metis_path(graph_type: str, s: int, n: int, q: int, base_dir: str = None) -> str:
    """Get METIS graph path."""
    return get_lmdb_path(graph_type, s, n, q, base_dir).replace(".lmdb", ".metis")


def get_mapping_path(graph_type: str, s: int, n: int, q: int, base_dir: str = None) -> str:
    """Get mapping file path."""
    return get_lmdb_path(graph_type, s, n, q, base_dir).replace(".lmdb", ".mapping")


def get_graph_path(graph_type: str, s: int, n: int, q: int, graph_dir: str = None) -> str:
    """Get graph path if it exists, else None."""
    path = get_lmdb_path(graph_type, s, n, q, graph_dir)
    return path if os.path.exists(path) else None


def get_solutions_dir(solver: str = "greedy", base_dir: str = None) -> str:
    """Get solutions directory: /mnt/Graphs/solutions/greedy/"""
    return os.path.join(base_dir or DEFAULT_GRAPH_DIR, "solutions", solver)


def get_checkpoints_dir(base_dir: str = None) -> str:
    """Get checkpoints directory."""
    return os.path.join(base_dir or DEFAULT_GRAPH_DIR, "checkpoints")
