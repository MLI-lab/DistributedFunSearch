"""Test sandbox memory limits using a simplified evaluation in a forked child.

Binary searches for the minimum memory limit that allows the sandbox to
run a small networkx evaluation without hitting RLIMIT_AS.

Usage:
    cd src/experiments/experiment1
    python /workspace/DistributedFunSearch/src/disfun/tests/test_sandbox_memory.py
"""

import os
import sys
import resource

# Add parent to path so we can import disfun modules
sys.path.insert(0, '/workspace/DistributedFunSearch/src')

from disfun.sandbox import ExternalProcessSandbox


# Minimal test program using the actual evaluation script
TEST_PROGRAM = '''
import hashlib
import math
import itertools
from collections import Counter
import numpy as np
import networkx as nx
import random

def priority(node, G, n, s) -> float:
    """Trivial priority function for testing."""
    return 1.0

def evaluate(params, graph_dir):
    n, s, q, G = params

    # Seed random (this triggers numpy.random import)
    random.seed(1)
    np.random.seed(1)

    # Compute priorities
    priorities = {node: priority(node, G, n, s) for node in G.nodes}

    # Sort and build independent set
    nodes_sorted = sorted(G.nodes, key=lambda x: (-priorities[x], x))
    independent_set = set()
    removed = set()
    for node in nodes_sorted:
        if node in removed:
            continue
        independent_set.add(node)
        removed.add(node)
        removed.update(G.neighbors(node))

    return (len(independent_set), None)
'''


def get_memory_info():
    """Get current process memory in MB."""
    import psutil
    p = psutil.Process()
    info = p.memory_info()
    return {
        'rss_mb': info.rss / 1024**2,
        'vms_mb': info.vms / 1024**2,
    }


def test_with_limit(limit_gb: float) -> tuple[bool, str, float]:
    """Test sandbox with given memory limit.

    Returns: (success, error_or_result, cpu_time)
    """
    sandbox = ExternalProcessSandbox(
        timeout_secs=30,
        graph_dir=None,
        memory_limit_gb=limit_gb,
    )

    # Create a small test graph
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([('0', '1'), ('1', '2'), ('2', '3')])

    test_input = (4, 2, 2, G)  # n=4, s=2, q=2, G

    result, success, cpu_time = sandbox.run(
        program=TEST_PROGRAM,
        function_to_run='evaluate',
        test_input=test_input,
    )

    if success:
        return True, f"result={result}", cpu_time
    else:
        return False, "sandbox failed", cpu_time


def main():
    print("=" * 60)
    print("Sandbox Memory Limit Test")
    print("=" * 60)

    # Show current memory before any tests
    mem = get_memory_info()
    print(f"\nParent process memory: RSS={mem['rss_mb']:.1f}MB, VMS={mem['vms_mb']:.1f}MB")

    # Note: the sandbox pre-loads numpy.random in compile_code() to avoid
    # lazy-load mmap failures in forked children under memory limits.
    print("\nRunning sandbox memory limit tests:")

    print("\n" + "-" * 60)
    print("Testing memory limits:")
    print("-" * 60)

    for limit_gb in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        success, msg, cpu = test_with_limit(limit_gb)
        status = "OK" if success else "FAIL"
        print(f"  {limit_gb:.2f} GB: {status:4s}  ({msg})")

        if success and limit_gb >= 1.0:
            # Found working limit, do binary search for minimum
            break

    # Binary search for minimum
    print("\n" + "-" * 60)
    print("Binary search for minimum limit:")
    print("-" * 60)

    low, high = 0.25, 4.0

    # First verify high works
    success, _, _ = test_with_limit(high)
    if not success:
        print(f"  ERROR: Even {high} GB fails!")
        return

    while high - low > 0.05:
        mid = (low + high) / 2
        success, _, _ = test_with_limit(mid)
        if success:
            high = mid
            print(f"  {mid:.2f} GB: OK")
        else:
            low = mid
            print(f"  {mid:.2f} GB: FAIL")

    print("\n" + "=" * 60)
    print(f"MINIMUM WORKING LIMIT: ~{high:.2f} GB")
    print(f"RECOMMENDED SETTING:   {max(2.0, high + 0.5):.1f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
