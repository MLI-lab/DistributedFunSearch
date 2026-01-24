#!/usr/bin/env python3
"""
Unified Codebook Loaders

Load codebooks from various sources:
- Our priority function evaluation results (codebooks.json)
- KAMIS .result files
- Plain text files (one codeword per line)
- Greedy baseline solutions

Auto-detects format based on file extension and content.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Union

from .graph_paths import get_mapping_path


def load_kamis_result(result_file: str, mapping_file: str) -> Set[str]:
    """
    Load KAMIS result file (.result format).

    Args:
        result_file: Path to .result file (one 0/1 per line)
        mapping_file: Path to .mapping file (maps integer IDs to codeword strings)

    Returns:
        Set of codeword strings in the independent set
    """
    # Load mapping
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    int_to_str = mapping["int_to_str"]

    # Load result
    with open(result_file, "r") as f:
        lines = f.readlines()

    # Parse: 1 means node is in set
    codebook = set()
    for node_id, line in enumerate(lines, start=1):
        if line.strip() == "1":
            codebook.add(int_to_str[str(node_id)])

    return codebook


def load_text_codebook(text_file: str) -> Set[str]:
    """
    Load codebook from plain text file (one codeword per line).

    Args:
        text_file: Path to text file

    Returns:
        Set of codeword strings
    """
    with open(text_file, "r") as f:
        codebook = {line.strip() for line in f if line.strip()}
    return codebook


def load_json_codebook(json_file: str, n: Optional[int] = None) -> Union[Set[str], Dict[int, Set[str]]]:
    """
    Load codebook from JSON file.

    Handles multiple formats:
    1. List of codewords: ["000", "001", ...]
    2. Dict with n as key: {"6": ["000", ...], "7": [...]}
    3. Our codebooks.json format with functions

    Args:
        json_file: Path to JSON file
        n: Optional n value to extract (for dict format)

    Returns:
        Set of codewords, or Dict[n, Set[codewords]] if n not specified
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    # Case 1: Simple list of codewords
    if isinstance(data, list):
        # Check if it's a list of strings (codewords) or list of dicts (functions)
        if len(data) > 0 and isinstance(data[0], str):
            return set(data)
        # It's our codebooks.json format with functions - handled below

    # Case 2: Dict with n as keys (like vt_solutions.json)
    if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
        result = {}
        for n_str, codewords in data.items():
            if isinstance(codewords, list):
                result[int(n_str)] = set(codewords)
            elif isinstance(codewords, dict):
                # Nested format like {"6": {"0": [...]}} - take first a value
                first_a = next(iter(codewords.values()))
                result[int(n_str)] = set(first_a)

        if n is not None:
            return result.get(n, set())
        return result

    # Case 3: Our codebooks.json format
    if isinstance(data, dict) and 'functions' in data:
        # Return dict mapping func_index -> {n: codebook}
        result = {}
        for func in data['functions']:
            func_idx = func.get('func_index', 0)
            codebooks = func.get('codebooks', {})
            result[func_idx] = {int(k): set(v) for k, v in codebooks.items()}
        return result

    raise ValueError(f"Unknown JSON format in {json_file}")


def find_mapping_file(result_file: str) -> Optional[str]:
    """Find the corresponding .mapping file for a KAMIS .result file."""
    import re

    # Try same directory with .mapping extension
    base = result_file.rsplit('.', 1)[0]

    # Remove seed suffix if present (e.g., _online_mis_seed0)
    for suffix in ['_online_mis_seed0', '_online_mis_seed100000', '_online_mis_seed200000',
                   '_redumis_seed0', '_redumis_seed100000', '_redumis_seed200000']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    mapping_file = base + '.mapping'
    if os.path.exists(mapping_file):
        return mapping_file

    # Try parent directory
    parent_dir = os.path.dirname(result_file)
    graph_name = os.path.basename(base)

    # Look in sibling directories
    for sibling in ['kamis_graphs', 'graphs', '..']:
        candidate = os.path.join(parent_dir, sibling, graph_name + '.mapping')
        if os.path.exists(candidate):
            return candidate

    # Try using centralized graph paths
    # Parse graph parameters from filename (e.g., graph_d_s1_n10_q2)
    match = re.search(r'graph_(d|ids)_s(\d+)_n(\d+)_q(\d+)', graph_name)
    if match:
        graph_type = "deletion" if match.group(1) == "d" else "ids"
        s, n, q = int(match.group(2)), int(match.group(3)), int(match.group(4))
        mapping_path = get_mapping_path(graph_type, s, n, q)
        if os.path.exists(mapping_path):
            return mapping_path

    return None


def detect_format(file_path: str) -> str:
    """
    Auto-detect codebook file format.

    Returns:
        One of: 'kamis', 'text', 'json_list', 'json_dict', 'codebooks_json'
    """
    ext = Path(file_path).suffix.lower()

    if ext == '.result':
        return 'kamis'

    if ext == '.txt':
        return 'text'

    if ext == '.json':
        # Need to peek at content
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return 'text'  # Fallback

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], str):
                return 'json_list'
            elif len(data) > 0 and isinstance(data[0], dict):
                return 'codebooks_json'
        elif isinstance(data, dict):
            if 'functions' in data:
                return 'codebooks_json'
            return 'json_dict'

    # Default to text
    return 'text'


def load_codebook(
    file_path: str,
    mapping_file: Optional[str] = None,
    n: Optional[int] = None
) -> Union[Set[str], Dict[int, Set[str]]]:
    """
    Load codebook from any supported format (auto-detected).

    Args:
        file_path: Path to codebook file
        mapping_file: Optional mapping file for KAMIS results (auto-detected if not provided)
        n: Optional n value to extract (for formats with multiple n values)

    Returns:
        Set of codewords, or Dict[n, Set[codewords]] for multi-n formats
    """
    fmt = detect_format(file_path)

    if fmt == 'kamis':
        if mapping_file is None:
            mapping_file = find_mapping_file(file_path)
        if mapping_file is None:
            raise ValueError(f"Could not find mapping file for KAMIS result: {file_path}")
        return load_kamis_result(file_path, mapping_file)

    elif fmt == 'text':
        return load_text_codebook(file_path)

    elif fmt in ('json_list', 'json_dict', 'codebooks_json'):
        return load_json_codebook(file_path, n)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def load_codebooks_from_directory(
    directory: str,
    pattern: str = "*.result",
    mapping_dir: Optional[str] = None,
    n_values: Optional[List[int]] = None,
    s: int = 1,
    q: int = 2
) -> Dict[int, Set[str]]:
    """
    Load codebooks from a directory of result files.

    Useful for loading KAMIS or greedy results organized by n value.

    Args:
        directory: Directory containing result files
        pattern: Glob pattern for result files
        mapping_dir: Directory containing .mapping files
        n_values: List of n values to load (None = all found)
        s: s parameter (for filename matching)
        q: q parameter (for filename matching)

    Returns:
        Dict mapping n -> codebook (best solution if multiple seeds)
    """
    from glob import glob

    directory = Path(directory)
    codebooks = {}

    # Find all result files
    result_files = list(directory.glob(pattern))

    # Group by n value
    by_n = {}
    for f in result_files:
        # Extract n from filename like graph_d_s1_n10_q2_online_mis_seed0.result
        name = f.stem
        for part in name.split('_'):
            if part.startswith('n') and part[1:].isdigit():
                n = int(part[1:])
                if n_values is None or n in n_values:
                    if n not in by_n:
                        by_n[n] = []
                    by_n[n].append(f)
                break

    # Load best (largest) codebook for each n
    for n, files in by_n.items():
        best_codebook = set()
        for f in files:
            try:
                mapping_file = None
                if mapping_dir:
                    # Construct mapping filename
                    base_name = f.stem
                    for suffix in ['_online_mis_seed0', '_online_mis_seed100000', '_online_mis_seed200000',
                                   '_redumis_seed0', '_redumis_seed100000', '_redumis_seed200000']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    mapping_file = os.path.join(mapping_dir, base_name + '.mapping')

                codebook = load_codebook(str(f), mapping_file)
                if len(codebook) > len(best_codebook):
                    best_codebook = codebook
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
                continue

        if best_codebook:
            codebooks[n] = best_codebook

    return codebooks


def load_greedy_solutions(
    directory: str,
    n_values: Optional[List[int]] = None,
    s: int = 1,
    q: int = 2,
    graph_type: str = "deletions"
) -> Dict[int, Set[str]]:
    """
    Load greedy baseline solutions from directory.

    Expects files like: graph_d_s1_n10_q2_best_solution.txt

    Args:
        directory: Directory containing solution files
        n_values: List of n values to load (None = all found)
        s, q: Parameters for filename matching
        graph_type: "deletions" or "ids"

    Returns:
        Dict mapping n -> codebook
    """
    directory = Path(directory)
    prefix = "graph_d" if graph_type == "deletions" else "graph_ids"

    codebooks = {}
    for f in directory.glob(f"{prefix}_s{s}_n*_q{q}_best_solution.txt"):
        # Extract n from filename
        name = f.stem
        for part in name.split('_'):
            if part.startswith('n') and part[1:].isdigit():
                n = int(part[1:])
                if n_values is None or n in n_values:
                    codebooks[n] = load_text_codebook(str(f))
                break

    return codebooks
