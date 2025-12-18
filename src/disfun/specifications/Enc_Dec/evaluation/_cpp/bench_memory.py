#!/usr/bin/env python3
"""Benchmark with memory tracking (each test in separate subprocess)."""
import os
import subprocess
import sys
import json

def format_time(t):
    if t < 60:
        return f"{t:.2f}s"
    elif t < 3600:
        return f"{t/60:.1f}m"
    else:
        return f"{t/3600:.1f}h"

# Script to run in subprocess (gets accurate per-test memory)
BENCH_SCRIPT = '''
import os
os.environ['OMP_NUM_THREADS'] = '6'
import time
import resource
import json
import numpy as np
import bruteforce_cpp

n, s = {n}, {s}
weights = np.array([[i + 1 for i in range(n)]], dtype=np.int64)
moduli = np.array([n + 1], dtype=np.int32)
targets = np.array([0], dtype=np.int32)

start = time.perf_counter()
result = bruteforce_cpp.validate_decoder_bruteforce(n, s, weights, moduli, targets, False)
elapsed = time.perf_counter() - start
peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(json.dumps({{
    "codebook_size": result["codebook_size"],
    "total_signatures": result["total_signatures"],
    "unique_signatures": result["unique_signatures"],
    "collision_count": result["collision_count"],
    "time": elapsed,
    "peak_mem": peak_mem
}}))
'''

def bench(n, s):
    """Run benchmark in subprocess for accurate memory measurement."""
    script = BENCH_SCRIPT.format(n=n, s=s)
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)

from concurrent.futures import ThreadPoolExecutor, as_completed

def run_s_benchmarks(s):
    """Run all benchmarks for a given s value."""
    if s == 2:
        n_values = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    elif s == 3:
        n_values = [10, 12, 14, 16, 18, 20, 22, 24, 26]
    elif s == 4:
        n_values = [10, 12, 14, 16, 18, 20, 22, 24]
    else:  # s == 5
        n_values = [10, 12, 14, 16, 18, 20, 22]

    results = []
    for n in n_values:
        try:
            r = bench(n, s)
            r['n'] = n
            r['s'] = s
            results.append(r)
        except Exception as e:
            results.append({'n': n, 's': s, 'error': str(e)})
            break
    return s, results

print("Benchmark with 6 threads per test (VT codes)")
print("Running s=2,3,4,5 in parallel...")
print()

# Run all s values in parallel
all_results = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_s_benchmarks, s): s for s in [2, 3, 4, 5]}
    for future in as_completed(futures):
        s, results = future.result()
        all_results[s] = results
        print(f"s={s} completed")

print()

# Print results in order
for s in [2, 3, 4, 5]:
    print(f"### s = {s}")
    print()
    print(f"| n | Codebook | Total Sigs | Unique Sigs | Collisions | Time | Mem |")
    print(f"|---|----------|------------|-------------|------------|------|-----|")

    for r in all_results[s]:
        if 'error' in r:
            print(f"| {r['n']} | error: {r['error']} |")
        else:
            print(f"| {r['n']} | {r['codebook_size']:,} | {r['total_signatures']:,} | {r['unique_signatures']:,} | {r['collision_count']:,} | {format_time(r['time'])} | {r['peak_mem']:.0f} MB |")

    print()
