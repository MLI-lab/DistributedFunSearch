#!/usr/bin/env python3
"""Benchmark with specific thread count."""
import sys
import os
import time
import numpy as np

# Set thread count BEFORE importing
num_threads = int(sys.argv[1])
n = int(sys.argv[2])
s = int(sys.argv[3])

os.environ['OMP_NUM_THREADS'] = str(num_threads)

import bruteforce_cpp

weights = np.array([[i + 1 for i in range(n)]], dtype=np.int64)
moduli = np.array([n + 1], dtype=np.int32)
targets = np.array([0], dtype=np.int32)

# Warmup
bruteforce_cpp.validate_decoder_bruteforce(n, s, weights, moduli, targets, False)

# Benchmark (3 runs)
times = []
for _ in range(3):
    start = time.perf_counter()
    result = bruteforce_cpp.validate_decoder_bruteforce(n, s, weights, moduli, targets, False)
    times.append(time.perf_counter() - start)

avg_time = sum(times) / len(times)
print(f"{avg_time:.3f},{result['codebook_size']},{result['collision_count']},{result['unique_signatures']}")
