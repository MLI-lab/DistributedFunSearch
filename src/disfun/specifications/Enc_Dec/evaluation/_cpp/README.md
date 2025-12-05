# Brute-Force Decoder Validation - C++ Extension

High-performance C++ implementation for validating deletion-correcting code decoders.

## Overview

This module validates whether a decoder can uniquely identify codewords after up to `s` deletions by:
1. Enumerating all valid codewords (satisfying encoder constraints)
2. Generating all deletion patterns (|D| ≤ s)
3. Computing signatures Φ(x,D) = (received_word, syndrome_difference)
4. Detecting collisions between different codewords

## Building

```bash
cd src/disfun/specifications/Enc_Dec/evaluation/_cpp
python setup.py build_ext --inplace
```

Or manually:
```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp \
    -include timespec_compat.h \
    -DVERSION_INFO=\"2.0.0\" \
    -I$(python -c "import pybind11; print(pybind11.get_include())") \
    -I$(python -c "import sysconfig; print(sysconfig.get_path('include'))") \
    bruteforce.cpp -o bruteforce_cpp.cpython-310-x86_64-linux-gnu.so
```

## Usage

```python
import numpy as np
from bruteforce_cpp import validate_decoder_bruteforce

n, s = 24, 2  # blocklength, max deletions
r = 2         # number of constraints

# Weight matrix (r x n) and moduli
weights = np.array([[1, 2, 3, ...], [1, 1, 1, ...]], dtype=np.int64)
moduli = np.array([127, 131], dtype=np.int32)
targets = np.array([0, 0], dtype=np.int32)  # target syndromes

result = validate_decoder_bruteforce(n, s, weights, moduli, targets, return_collisions=True)

print(f"Valid: {result['valid']}")
print(f"Codebook size: {result['codebook_size']}")
print(f"Score: {result['score']:.4f}")  # fraction of unique signatures
```

## Performance

Benchmarks on 128-core server (n=26, s=2, ~4K codewords, ~1.4M signatures):

| Version | Time | Notes |
|---------|------|-------|
| Pure Python | ~15s | Fallback for small n |
| v2 (default) | 0.095s | **~150x speedup** |

### Scaling with problem size (s=2)

| n | Codebook | Signatures | Unique Sigs | Time | Memory |
|---|----------|------------|-------------|------|--------|
| 20 | 51 | 10K | 3K | 0.06s | <1 MB |
| 22 | 268 | 68K | 20K | 0.03s | 1 MB |
| 24 | 1,007 | 303K | 87K | 0.04s | 6 MB |
| 26 | 4,075 | 1.4M | 414K | 0.13s | 28 MB |
| 28 | 16,481 | 6.7M | 1.8M | 0.64s | 125 MB |
| 30 | 64,349 | 30M | 8.5M | 3.25s | 582 MB |

## Feasibility Limits

### Practical limits (< 1 min, < 8 GB RAM)

| Deletions (s) | Max n | Time | Memory | Total Signatures |
|---------------|-------|------|--------|------------------|
| s = 2 | **30** | ~4s | ~600 MB | ~30M |
| s = 3 | **28** | ~4s | ~560 MB | ~60M |
| s = 4 | **26** | ~3s | ~314 MB | ~73M |

### Extended limits (< 5 min, < 32 GB RAM)

| Deletions (s) | Max n | Time | Memory | Total Signatures |
|---------------|-------|------|--------|------------------|
| s = 2 | **32** | ~2s* | ~18 MB* | ~260K* |
| s = 3 | **30** | ~20s | ~2.7 GB | ~290M |
| s = 4 | **28** | ~25s | ~1.4 GB | ~400M |

*n=32 values depend heavily on constraints (random constraints may be very restrictive)

### Detailed scaling by s

**s = 2 deletions:**
| n | Patterns | Total Sigs | Time | Memory |
|---|----------|------------|------|--------|
| 26 | 352 | 1.4M | 0.13s | 28 MB |
| 28 | 407 | 6.7M | 0.64s | 125 MB |
| 30 | 466 | 30M | 3.25s | 582 MB |

**s = 3 deletions:**
| n | Patterns | Total Sigs | Time | Memory |
|---|----------|------------|------|--------|
| 24 | 2,325 | 2.3M | 0.13s | 24 MB |
| 26 | 2,952 | 12M | 0.65s | 123 MB |
| 28 | 3,683 | 61M | 3.6s | 558 MB |
| 30 | 4,526 | 291M | 19s | 2.7 GB |

**s = 4 deletions:**
| n | Patterns | Total Sigs | Time | Memory |
|---|----------|------------|------|--------|
| 24 | 12,951 | 13M | 0.79s | 62 MB |
| 26 | 17,902 | 73M | 3.1s | 314 MB |
| 28 | 24,158 | 398M | 25s | 1.4 GB |
| 30 | 31,931 | 2B | 138s | 6.6 GB |

## Implementation Versions

### v2 (Default) - `bruteforce.cpp`

Best for s ≥ 2 (the common case).

**Algorithm:**
1. Parallel codeword enumeration with thread-local buffers
2. Parallel signature generation with thread-local storage
3. Sequential merge into global hash table for collision detection

**Optimizations:**
- Precomputed weight differences for fast syndrome computation
- Fixed-size signature struct (no heap allocation per signature)
- MurmurHash-inspired hash function
- Bitmask-based deletion patterns (Gosper's hack)

### v4 - `bruteforce_v4.cpp`

Best for s = 1 (1.9x faster than v2).

**Key insight:** Signatures can only collide if `received_word` is identical. Partition by `received_word % 1024` for lock-free parallel collision detection.

**Algorithm:**
1. Parallel signature generation into partitioned thread-local buffers
2. Parallel collision detection per partition (no synchronization needed)

**Trade-off:** Partitioning overhead exceeds lock-free benefits for s ≥ 2.

### v3 - `bruteforce_v3.cpp`

Experimental weight-based partitioning. Slower than v2 due to multiple hash table lookups.

## Benchmark: v2 vs v4

| s | Patterns/codeword | v2 | v4 | Winner |
|---|-------------------|----|----|--------|
| 1 | 27 | 0.053s | 0.028s | **v4 (1.9x)** |
| 2 | 352 | 0.095s | 0.116s | **v2** |
| 3 | 2,952 | 0.527s | 0.549s | **v2** |

## Files

| File | Description |
|------|-------------|
| `bruteforce.cpp` | Active source (copy of v2) |
| `bruteforce_v2.cpp` | Sequential merge version |
| `bruteforce_v3.cpp` | Weight-partitioned version |
| `bruteforce_v4.cpp` | Lock-free partitioned version |
| `setup.py` | pybind11 build configuration |
| `timespec_compat.h` | Workaround for conda gcc + glibc |

## Switching Versions

To use v4 for s=1 workloads:
```bash
cp bruteforce_v4.cpp bruteforce.cpp
python setup.py build_ext --inplace
```

## Algorithm Details

### Signature Computation

For codeword `x` and deletion pattern `D`:
```
Φ(x, D) = (received_word, Δ_0, Δ_1, ..., Δ_{r-1})
```

Where:
- `received_word` = bits of x with positions in D removed
- `Δ_j` = syndrome difference due to deletions (mod m_j)

### Collision Detection

Two (codeword, pattern) pairs collide if they produce identical signatures:
```
Φ(x₁, D₁) = Φ(x₂, D₂) with x₁ ≠ x₂
```

A valid decoder has zero collisions.

### Weight Partitioning Insight

Codewords with Hamming weight difference > s can never collide because:
- After ≤ s deletions, received words differ in weight by > 0
- Different weight → different received_word → no collision

This is implicitly captured by partitioning on `received_word`.
