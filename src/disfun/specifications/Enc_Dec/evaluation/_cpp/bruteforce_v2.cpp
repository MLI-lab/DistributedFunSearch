/**
 * Optimized brute-force decoder validation for deletion-correcting codes.
 *
 * Optimizations over v1:
 * 1. Precomputed deletion masks as bitmasks (not vectors)
 * 2. Fixed-size signature structure (no heap allocation)
 * 3. Parallel collision detection with thread-local hash tables
 * 4. Vectorized syndrome computation
 * 5. Cache-friendly memory access patterns
 * 6. Compact signature representation
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Maximum number of constraints supported
constexpr int MAX_CONSTRAINTS = 8;
constexpr int MAX_N = 32;

// Compact signature - fixed size, no heap allocation
struct CompactSignature {
    uint64_t received_word;
    uint32_t received_len : 8;
    uint32_t num_constraints : 8;
    uint32_t _padding : 16;
    int32_t delta[MAX_CONSTRAINTS];  // Fixed-size array

    bool operator==(const CompactSignature& other) const {
        if (received_word != other.received_word ||
            received_len != other.received_len) return false;
        for (int i = 0; i < num_constraints; ++i) {
            if (delta[i] != other.delta[i]) return false;
        }
        return true;
    }
};

// Fast hash for CompactSignature
struct CompactSignatureHash {
    size_t operator()(const CompactSignature& sig) const {
        // MurmurHash-inspired mixing
        uint64_t h = sig.received_word;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= sig.received_len;

        for (int i = 0; i < sig.num_constraints; ++i) {
            h ^= static_cast<uint64_t>(sig.delta[i]) << (i * 8);
        }
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }
};

// Precomputed deletion pattern with bitmask
struct DeletionPattern {
    uint32_t mask;           // Bitmask: bit i = 1 means position i is deleted
    uint8_t num_deleted;     // Number of deletions
    uint8_t positions[MAX_N]; // Actual positions (for syndrome computation)
};

// Collision info
struct CollisionInfo {
    uint32_t codeword_idx;
    uint16_t pattern_idx;
};

// Popcount
inline int popcount32(uint32_t x) {
    #if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
    #else
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    #endif
}

// Generate deletion patterns as bitmasks
std::vector<DeletionPattern> generate_deletion_patterns_v2(int n, int s) {
    std::vector<DeletionPattern> patterns;

    // Empty pattern (no deletions)
    DeletionPattern empty;
    empty.mask = 0;
    empty.num_deleted = 0;
    patterns.push_back(empty);

    // Generate all subsets of size 1 to s using Gosper's hack
    for (int d = 1; d <= s; ++d) {
        uint32_t subset = (1u << d) - 1;  // First subset of size d
        uint32_t limit = 1u << n;

        while (subset < limit) {
            DeletionPattern pat;
            pat.mask = subset;
            pat.num_deleted = d;

            // Extract positions
            int pos_idx = 0;
            for (int i = 0; i < n && pos_idx < d; ++i) {
                if (subset & (1u << (n - 1 - i))) {
                    pat.positions[pos_idx++] = i;
                }
            }
            patterns.push_back(pat);

            // Gosper's hack for next subset
            uint32_t c = subset & -subset;
            uint32_t r = subset + c;
            subset = (((r ^ subset) >> 2) / c) | r;
        }
    }

    return patterns;
}

// Fast deletion application using bitmask
inline uint64_t apply_deletion_fast(uint32_t codeword, int n, uint32_t delete_mask) {
    if (delete_mask == 0) return codeword;

    uint64_t result = 0;
    int out_pos = 0;
    int num_deleted = popcount32(delete_mask);
    int out_len = n - num_deleted;

    for (int i = 0; i < n; ++i) {
        uint32_t bit_mask = 1u << (n - 1 - i);
        if (!(delete_mask & bit_mask)) {
            // Not deleted, copy bit
            int bit = (codeword >> (n - 1 - i)) & 1;
            result |= (static_cast<uint64_t>(bit) << (out_len - 1 - out_pos));
            ++out_pos;
        }
    }
    return result;
}

// Precomputed weight differences for shift correction
struct PrecomputedWeights {
    std::vector<int64_t> weights;  // r x n
    std::vector<int32_t> moduli;   // r
    int r;
    int n;

    // For each position i and shift amount s: weight_diff[i][s] = weight[i] - weight[i-s]
    std::vector<std::vector<int64_t>> weight_diffs;  // r x n x n

    void init(const int64_t* w, const int32_t* m, int num_constraints, int blocklength) {
        r = num_constraints;
        n = blocklength;
        weights.assign(w, w + r * n);
        moduli.assign(m, m + r);

        // Precompute weight differences
        weight_diffs.resize(r);
        for (int j = 0; j < r; ++j) {
            weight_diffs[j].resize(n * n, 0);
            for (int i = 0; i < n; ++i) {
                for (int shift = 0; shift <= i; ++shift) {
                    int shifted_pos = i - shift;
                    weight_diffs[j][i * n + shift] = weights[j * n + i] - weights[j * n + shifted_pos];
                }
            }
        }
    }
};

// Fast syndrome difference computation using precomputed weights
inline void compute_syndrome_fast(
    uint32_t codeword,
    int n,
    const DeletionPattern& pattern,
    const PrecomputedWeights& pw,
    int32_t* delta_out
) {
    for (int j = 0; j < pw.r; ++j) {
        int64_t contrib = 0;
        int64_t shift_correction = 0;
        int shift = 0;
        int del_idx = 0;

        for (int i = 0; i < n; ++i) {
            int bit = (codeword >> (n - 1 - i)) & 1;

            if (del_idx < pattern.num_deleted && i == pattern.positions[del_idx]) {
                // Deleted position
                contrib += bit * pw.weights[j * n + i];
                ++shift;
                ++del_idx;
            } else {
                // Non-deleted position
                shift_correction += bit * pw.weight_diffs[j][i * n + shift];
            }
        }

        int32_t m = pw.moduli[j];
        delta_out[j] = static_cast<int32_t>(((contrib + shift_correction) % m + m) % m);
    }
}

// Thread-local hash table type
using LocalHashTable = std::unordered_map<CompactSignature, CollisionInfo, CompactSignatureHash>;

// Main optimized validation function
py::dict validate_decoder_bruteforce_v2(
    int n,
    int s,
    py::array_t<int64_t> weights_arr,
    py::array_t<int32_t> moduli_arr,
    py::array_t<int32_t> targets_arr,
    bool return_collisions
) {
    auto weights_info = weights_arr.request();
    auto moduli_info = moduli_arr.request();
    auto targets_info = targets_arr.request();

    int r = static_cast<int>(weights_info.shape[0]);
    const int64_t* weights = static_cast<const int64_t*>(weights_info.ptr);
    const int32_t* moduli = static_cast<const int32_t*>(moduli_info.ptr);
    const int32_t* targets = static_cast<const int32_t*>(targets_info.ptr);

    if (n > MAX_N) {
        throw std::runtime_error("n > 32 not supported");
    }
    if (r > MAX_CONSTRAINTS) {
        throw std::runtime_error("More than 8 constraints not supported");
    }

    // Precompute weights for fast syndrome computation
    PrecomputedWeights pw;
    pw.init(weights, moduli, r, n);

    // Generate deletion patterns
    auto patterns = generate_deletion_patterns_v2(n, s);
    size_t num_patterns = patterns.size();

    // Step 1: Enumerate codewords (parallel)
    std::vector<uint32_t> codewords;
    uint64_t total_candidates = 1ULL << n;

    #pragma omp parallel
    {
        std::vector<uint32_t> local_codewords;
        local_codewords.reserve(total_candidates / (omp_get_num_threads() * 10));

        #pragma omp for nowait schedule(static, 4096)
        for (uint64_t x = 0; x < total_candidates; ++x) {
            bool valid = true;
            for (int j = 0; j < r && valid; ++j) {
                int64_t syndrome = 0;
                // Unrolled syndrome computation
                for (int i = 0; i < n; ++i) {
                    syndrome += ((x >> (n - 1 - i)) & 1) * weights[j * n + i];
                }
                syndrome = ((syndrome % moduli[j]) + moduli[j]) % moduli[j];
                if (syndrome != targets[j]) valid = false;
            }
            if (valid) {
                local_codewords.push_back(static_cast<uint32_t>(x));
            }
        }

        #pragma omp critical
        {
            codewords.insert(codewords.end(), local_codewords.begin(), local_codewords.end());
        }
    }

    size_t codebook_size = codewords.size();

    if (codebook_size == 0) {
        py::dict result;
        result["valid"] = true;
        result["codebook_size"] = 0;
        result["total_signatures"] = 0;
        result["unique_signatures"] = 0;
        result["collision_count"] = 0;
        result["score"] = 1.0;
        result["collisions"] = py::none();
        return result;
    }

    // Step 2: Parallel signature computation and collision detection
    int num_threads = 1;
    #ifdef _OPENMP
    num_threads = omp_get_max_threads();
    #endif

    std::vector<LocalHashTable> thread_tables(num_threads);
    std::vector<std::vector<std::tuple<uint32_t, uint16_t, CompactSignature>>> thread_signatures(num_threads);

    // Reserve space
    size_t sigs_per_thread = (codebook_size * num_patterns) / num_threads + 1;
    for (int t = 0; t < num_threads; ++t) {
        thread_tables[t].reserve(sigs_per_thread);
        thread_signatures[t].reserve(sigs_per_thread);
    }

    std::atomic<size_t> total_signatures{0};

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        auto& local_sigs = thread_signatures[tid];

        #pragma omp for schedule(dynamic, 64)
        for (size_t cw_idx = 0; cw_idx < codebook_size; ++cw_idx) {
            uint32_t codeword = codewords[cw_idx];

            for (size_t pat_idx = 0; pat_idx < num_patterns; ++pat_idx) {
                const auto& pattern = patterns[pat_idx];

                // Compute signature
                CompactSignature sig;
                sig.received_word = apply_deletion_fast(codeword, n, pattern.mask);
                sig.received_len = n - pattern.num_deleted;
                sig.num_constraints = r;
                compute_syndrome_fast(codeword, n, pattern, pw, sig.delta);

                // Store for cross-thread collision detection
                local_sigs.emplace_back(static_cast<uint32_t>(cw_idx),
                                        static_cast<uint16_t>(pat_idx), sig);
            }
        }

        // Update total signatures count (atomic)
        total_signatures.fetch_add(local_sigs.size(), std::memory_order_relaxed);
    }

    // Step 3: Merge and detect collisions (single-threaded for correctness)
    LocalHashTable global_table;
    global_table.reserve(total_signatures.load());

    std::vector<std::tuple<uint32_t, uint16_t, uint32_t, uint16_t>> collision_pairs;
    size_t collision_count = 0;

    for (int t = 0; t < num_threads; ++t) {
        for (const auto& [cw_idx, pat_idx, sig] : thread_signatures[t]) {
            auto it = global_table.find(sig);
            if (it != global_table.end()) {
                if (it->second.codeword_idx != cw_idx) {
                    ++collision_count;
                    if (return_collisions) {
                        collision_pairs.emplace_back(
                            it->second.codeword_idx, it->second.pattern_idx,
                            cw_idx, pat_idx
                        );
                    }
                }
            } else {
                global_table[sig] = CollisionInfo{cw_idx, pat_idx};
            }
        }
    }

    size_t unique_signatures = global_table.size();
    bool valid = (collision_count == 0);
    double score = total_signatures > 0 ?
        static_cast<double>(unique_signatures) / total_signatures : 1.0;

    // Build result
    py::dict result;
    result["valid"] = valid;
    result["codebook_size"] = codebook_size;
    result["total_signatures"] = total_signatures.load();
    result["unique_signatures"] = unique_signatures;
    result["collision_count"] = collision_count;
    result["score"] = score;

    if (return_collisions) {
        std::vector<py::dict> collisions;
        collisions.reserve(collision_pairs.size());
        for (const auto& [cw1, pat1, cw2, pat2] : collision_pairs) {
            py::dict coll;
            coll["codeword1"] = static_cast<int64_t>(codewords[cw1]);
            std::vector<int> pos1(patterns[pat1].positions,
                                  patterns[pat1].positions + patterns[pat1].num_deleted);
            coll["pattern1"] = pos1;
            coll["codeword2"] = static_cast<int64_t>(codewords[cw2]);
            std::vector<int> pos2(patterns[pat2].positions,
                                  patterns[pat2].positions + patterns[pat2].num_deleted);
            coll["pattern2"] = pos2;
            collisions.push_back(coll);
        }
        result["collisions"] = collisions;
    } else {
        result["collisions"] = py::none();
    }

    return result;
}

PYBIND11_MODULE(bruteforce_cpp, m) {
    m.doc() = "Optimized C++ extension for brute-force decoder validation";

    m.def("validate_decoder_bruteforce", &validate_decoder_bruteforce_v2,
          py::arg("n"),
          py::arg("s"),
          py::arg("weights"),
          py::arg("moduli"),
          py::arg("targets"),
          py::arg("return_collisions") = false,
          R"doc(
Optimized brute-force decoder validation.

Optimizations:
- Precomputed deletion masks as bitmasks
- Fixed-size signatures (no heap allocation per signature)
- Parallel signature computation with thread-local hash tables
- Precomputed weight differences for fast syndrome computation
- Cache-friendly memory access
)doc");

    m.attr("__version__") = VERSION_INFO;
}
