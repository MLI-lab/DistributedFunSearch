"""Shared utilities for checkpoint analysis scripts."""

import pickle
from pathlib import Path
from typing import Dict, List, Any


def find_latest_checkpoint(path: Path) -> Path:
    """Find the latest checkpoint file in a directory or return the file itself."""
    if path.is_file() and path.suffix == '.pkl':
        return path
    elif path.is_dir():
        checkpoints = list(path.glob("*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(f"No .pkl files found in {path}")
        # Sort by modification time, return latest
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        return checkpoints[-1]
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def find_checkpoints(path: Path) -> List[Path]:
    """Find all checkpoint files, sorted by modification time."""
    if path.is_file() and path.suffix == '.pkl':
        return [path]
    elif path.is_dir():
        checkpoints = list(path.glob("*.pkl"))
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        return checkpoints
    return []


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_descriptions(checkpoint: Dict[str, Any]) -> List[str]:
    """Extract all description/thought strings from checkpoint."""
    descriptions = []

    for island_state in checkpoint.get('islands_state', []):
        for cluster_data in island_state.get('clusters', {}).values():
            for program in cluster_data.get('programs', []):
                desc = program.get('thought') or program.get('description') or ''
                if desc.strip():
                    descriptions.append(desc.strip())

    return descriptions
