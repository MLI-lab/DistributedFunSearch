# Enc_Dec: Encoder/Decoder Evaluation for Deletion-Correcting Codes

This specification evaluates encoder/decoder validity for deletion-correcting codes parameterized by weight functions and moduli.

## Background

### Problem

Given an n-bit codeword, can we uniquely recover the original after exactly s deletions?

### Parameterization

**Encoder:**
- Uses r constraints, each with weight function f_j(i) and modulus m_j
- Parity positions P ⊆ {0, ..., n-1}
- Constraint: Σ x_i · f_j(i) ≡ target_j (mod m_j)

**Decoder (Brute-Force Validation):**
- Enumerates all valid codewords satisfying encoder constraints
- For each codeword and deletion pattern, computes signature: (received_word, syndrome_difference)
- **Valid** if no two different codewords produce the same signature

### VT Codes

Varshamov-Tenengolts codes are a special case:
- r = 1 (single constraint)
- f(i) = i + 1 (1-indexed equivalent)
- m = n + 1, target = 0

VT codes can correct **single deletions** (s=1) but fail for s≥2.

## Search Space

We search for encoder/decoder parameters that achieve valid s-deletion correction with minimal redundancy.

### What We Optimize

The evolutionary search generates a `priority` function that returns:

```python
def priority(n: int, s: int) -> Tuple[List[Callable], List[int], List[int]]:
    """Returns encoder parameters for n-bit codewords with s deletions.

    Args:
        n: Codeword length
        s: Number of deletions to correct

    Returns:
        weight_funcs: List of r weight functions f_j(i) -> int (0-indexed)
        moduli: List of r moduli m_j
        parity_positions: List of |P| parity bit positions
    """
```

Target syndromes are fixed to zero (σ_j = 0 for all j).

### Redundancy

The redundancy is |P| (number of parity bits). The known lower bound is:

```
|P| ≥ s · log₂(n)
```

We search for valid encodings with |P| ≤ s · ⌈log₂(n)⌉. The evaluation enforces this upper bound and rejects encodings with higher redundancy.

### Encoder Validity

Given parity positions P, the parity bits must solve:

```
Σ_{p∈P} b_p · f_j(p) ≡ d_j (mod m_j)
```

where d_j is the difference between target and message syndrome (a subset-sum problem over Z_{m_1} × ... × Z_{m_r}).

We verify encoder validity by enumerating all 2^|P| parity assignments and checking that all produce **unique syndromes**. The score is:

```
encoder_score = unique_syndromes / 2^|P|
```

A score of 1.0 means valid encoder. Cost is O(2^|P|) = O(n^s).

### Decoder Validity

Only checked if encoder is valid. Uses brute-force collision detection (see below).

### Evaluation Pipeline

```
1. Encoder check (fast, O(n^s)):
   - Enumerate 2^|P| parity assignments
   - Score = unique_syndromes / 2^|P|
   - If score < 1.0 → invalid, skip decoder check

2. Decoder check (expensive, O(|C| × C(n,s))):
   - Only if encoder valid
   - Enumerate codebook, check signature collisions
   - Valid if collision_count = 0
```

## Algorithm to check unique decoding 

### Step 1: Enumerate valid codewords (parallel)

Iterate through all 2^n possible bit vectors, keep only those satisfying encoder constraint:

### Step 2: Generate deletion patterns

All C(n,s) ways to choose s positions to delete. Uses Gosper's hack for efficient enumeration.

### Step 3: Compute signatures (parallel)

For each (codeword, deletion_pattern) pair:
- `received_word` = bits remaining after deletion (length n-s)
- `delta_j` = `original_syndrome_j - received_syndrome_j (mod m_j)` for each constraint j
- `signature` = (received_word, delta_0, ..., delta_{r-1})

### Step 4: Detect collisions (parallel, streaming)

For each signature `(received_word, delta)`:

1. Compute partition ID = `hash(received_word, received_syndrome) % num_partitions`
2. Lock partition, lookup signature in hash table:
   - **Not found** → insert `{signature: codeword_idx}`
   - **Found, same codeword** → ok (different deletion patterns, same result)
   - **Found, different codeword** → **collision** (decoder fails)

A naive approach would first generate all signatures, store them in a list, then check for duplicates. That requires O(total signatures) = O(|C| × C(n,s)) memory. Instead, we insert directly into hash tables as signatures are computed—duplicates are detected on-the-fly and not stored, so memory is O(unique signatures).

**Parallelism:** Multiple threads process different codewords simultaneously. Each thread inserts into whichever partition its signature hashes to, with one mutex per partition to avoid races.

## Building the C++ Extension

Required for n > 16. Provides 30-130x speedup over Python.

```bash
cd evaluation/_cpp
pip install pybind11
CC=gcc CXX=g++ python setup.py build_ext --inplace
```

## Usage

```python
from evaluation.bruteforce_decoder import validate_decoder_bruteforce

n, s = 16, 1
weight_funcs = [lambda i: i + 1]  # VT code (0-indexed, so i+1 gives weights 1,2,...,n)
moduli = [n + 1]
targets = [0]

result = validate_decoder_bruteforce(n, s, weight_funcs, moduli, targets)
# result['valid'], result['codebook_size'], result['collision_count'], result['score']
```

Supports multiple constraints, pass lists of weight functions/moduli/targets. Default limit is r ≤ 8 (fixed-size array for performance), increase `MAX_CONSTRAINTS` in `bruteforce.cpp` if needed.

Automatically uses C++ extension if available (30-130x faster). Run `evaluation/_cpp/bench_memory.py` to benchmark.

## Performance

Benchmarked on 6 CPU threads (VT codes). Memory is proportional to unique signatures.

### s = 2

| n | Codebook | Total Sigs | Unique Sigs | Collisions | Time | Mem |
|---|----------|------------|-------------|------------|------|-----|
| 10 | 94 | 4,230 | 256 | 3,306 | 0.00s | 34 MB |
| 12 | 316 | 20,856 | 1,024 | 17,266 | 0.00s | 34 MB |
| 14 | 1,096 | 99,736 | 4,096 | 82,453 | 0.01s | 34 MB |
| 16 | 3,856 | 462,720 | 16,384 | 408,785 | 0.02s | 37 MB |
| 18 | 13,798 | 2,111,094 | 65,536 | 1,882,519 | 0.06s | 50 MB |
| 20 | 49,940 | 9,488,600 | 262,144 | 8,511,058 | 0.31s | 103 MB |
| 22 | 182,362 | 42,125,622 | 1,048,576 | 38,221,897 | 1.73s | 313 MB |
| 24 | 671,092 | 185,221,392 | 4,194,304 | 169,614,905 | 8.27s | 1151 MB |
| 26 | 2,485,534 | 807,798,550 | 16,777,216 | 744,835,619 | 33.87s | 4504 MB |
| 28 | 9,256,396 | 3,498,917,688 | 67,108,864 | 3,250,250,458 | 3.3m | 8997 MB |

--> for evolving feasible till n=23 (memory)

### s = 3

| n | Codebook | Total Sigs | Unique Sigs | Collisions | Time | Mem |
|---|----------|------------|-------------|------------|------|-----|
| 10 | 94 | 11,280 | 128 | 10,395 | 0.00s | 34 MB |
| 12 | 316 | 69,520 | 512 | 65,709 | 0.00s | 34 MB |
| 14 | 1,096 | 398,944 | 2,048 | 384,307 | 0.01s | 34 MB |
| 16 | 3,856 | 2,159,360 | 8,192 | 2,109,819 | 0.04s | 35 MB |
| 18 | 13,798 | 11,259,168 | 32,768 | 11,039,687 | 0.19s | 42 MB |
| 20 | 49,940 | 56,931,600 | 131,072 | 56,078,858 | 0.96s | 68 MB |
| 22 | 182,362 | 280,837,480 | 524,288 | 277,219,176 | 5.78s | 174 MB |
| 24 | 671,092 | 1,358,290,208 | 2,097,152 | 1,343,651,652 | 30.20s | 594 MB |
| 26 | 2,485,534 | 6,462,388,400 | 8,388,608 | 6,403,286,036 | 2.5m | 2274 MB |

--> for evolving feasible till n=24


### s = 4

| n | Codebook | Total Sigs | Unique Sigs | Collisions | Time | Mem |
|---|----------|------------|-------------|------------|------|-----|
| 10 | 94 | 19,740 | 64 | 18,879 | 0.00s | 34 MB |
| 12 | 316 | 156,420 | 256 | 152,583 | 0.00s | 34 MB |
| 14 | 1,096 | 1,097,096 | 1,024 | 1,085,352 | 0.02s | 34 MB |
| 16 | 3,856 | 7,017,920 | 4,096 | 6,978,718 | 0.10s | 35 MB |
| 18 | 13,798 | 42,221,880 | 16,384 | 42,017,872 | 0.55s | 38 MB |
| 20 | 49,940 | 241,959,300 | 65,536 | 241,211,166 | 3.77s | 52 MB |
| 22 | 182,362 | 1,333,978,030 | 262,144 | 1,330,557,425 | 23.21s | 104 MB |
| 24 | 671,092 | 7,131,023,592 | 1,048,576 | 7,116,923,904 | 2.2m | 316 MB |
| 26 | 2,485,534 | 37,158,733,300 | 4,194,304 | 37,101,342,704 | 13.1m | 1159 MB |
| 28 | - | - | - | - | >1h | - |

--> for evolving feasible till n=22


### s = 5

| n | Codebook | Total Sigs | Unique Sigs | Collisions | Time | Mem |
|---|----------|------------|-------------|------------|------|-----|
| 10 | 94 | 23,688 | 32 | 22,851 | 0.00s | 34 MB |
| 12 | 316 | 250,272 | 128 | 247,435 | 0.01s | 34 MB |
| 14 | 1,096 | 2,194,192 | 512 | 2,185,495 | 0.04s | 34 MB |
| 16 | 3,856 | 16,843,008 | 2,048 | 16,807,835 | 0.25s | 34 MB |
| 18 | 13,798 | 118,221,264 | 8,192 | 118,040,534 | 1.72s | 37 MB |
| 20 | 49,940 | 774,269,760 | 32,768 | 773,661,514 | 11.37s | 43 MB |
| 22 | 182,362 | 4,802,320,908 | 131,072 | 4,799,174,142 | 1.2m | 70 MB |
| 24 | - | - | - | - | ~10m | - |
| 26 | - | - | - | - | >1h | - |

--> for evolving feasible till n=20

### Controlling thread count

Set the `OMP_NUM_THREADS` environment variable:

```bash
OMP_NUM_THREADS=6 python your_script.py
```

## Implementation Details

### Memory-Efficient Design

Instead of storing all signatures then checking for collisions, the implementation inserts directly into hash tables as signatures are generated. This reduces memory from O(|C| × C(n,s)) to O(unique signatures).

### Partitioning for Parallel Collision Detection

Signatures are partitioned by hash for parallel processing:
- Each partition has its own hash table and mutex
- Threads lock only the partition they need
- `num_partitions = min(2^(n-s), 2^24)` (capped at 16M for ~1GB overhead)

Partition key uses both `received_word` and `received_syndrome` for better load distribution (1.3-1.8x faster than `received_word` alone).

### Collision Detection Within Partitions

```
Hash table:
    Key:   signature (received_word, delta)
    Value: codeword_idx

Algorithm:
    for each (signature, codeword_idx) in partition:
        if signature in hash_table:
            stored_codeword_idx = hash_table[signature]
            if stored_codeword_idx != codeword_idx:
                # Different codeword, same signature = decoding failure
                collision_count += 1
            # else: same codeword, different pattern = ok
        else:
            hash_table[signature] = codeword_idx
```

## File Structure

```
Enc_Dec/
├── evaluation/
│   ├── evaluate.py            # Combined encoder/decoder evaluation
│   ├── encoder.py             # Encoder validity check
│   ├── bruteforce_decoder.py  # Decoder collision detection
│   └── _cpp/                  # C++ extension (required for large n)
│       ├── bruteforce.cpp     # Main implementation
│       ├── setup.py           # pybind11 build config
│       ├── bench_memory.py    # Benchmark script
│       └── timespec_compat.h  # conda gcc compatibility
├── initial_functions/         # Known constructions for seeding evolution
│   ├── vt.txt                 # Varshamov-Tenengolts (s=1)
│   ├── helberg_ferreira.txt   # Helberg-Ferreira (any s)
│   └── cumsum_power.txt       # Generalized VT with power sums
├── imports/
│   └── base.txt               # Common imports for evolved functions
└── README.md
```

## Initial Functions

Known constructions available for seeding evolution:

| File | Construction | Description |
|------|--------------|-------------|
| `vt.txt` | VT code | f(i) = i+1, m = n+1. Valid for s=1 only |
| `helberg_ferreira.txt` | Helberg-Ferreira | Fibonacci-like weights. Valid for any s |
| `cumsum_power.txt` | Generalized VT | Cumulative power sums m^(e)_i = Σ j^e |

## Complexity

- **Codeword enumeration**: O(2^n)
- **Total signatures**: O(|C| × Σ_{d=0}^s C(n,d))
- **Memory**: O(unique signatures) due to streaming insertion

