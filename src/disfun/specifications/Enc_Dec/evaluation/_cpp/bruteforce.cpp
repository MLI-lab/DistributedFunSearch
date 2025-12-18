/**
 * Memory-efficient brute-force decoder validation.
 *
 * Instead of storing all signatures and then checking for collisions,
 * this version inserts directly into hash tables as signatures are generated.
 * This reduces memory from O(|C| * C(n,s)) to O(unique signatures).
 *
 * Uses mutex per partition for thread-safe insertion.
 */

#include "timespec_compat.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

constexpr int MAX_CONSTRAINTS = 8;
constexpr int MAX_N = 32;
constexpr size_t MAX_PARTITIONS = 16777216;  // 2^24, ~1GB memory limit

struct CompactSignature {
    uint64_t received_word;
    uint8_t num_constraints;
    int32_t delta[MAX_CONSTRAINTS];

    bool operator==(const CompactSignature& other) const {
        if (received_word != other.received_word) return false;
        for (int i = 0; i < num_constraints; ++i) {
            if (delta[i] != other.delta[i]) return false;
        }
        return true;
    }
};

struct CompactSignatureHash {
    size_t operator()(const CompactSignature& sig) const {
        uint64_t h = sig.received_word;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        for (int i = 0; i < sig.num_constraints; ++i) {
            h ^= static_cast<uint64_t>(sig.delta[i]) << (i * 8);
        }
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }
};

struct DeletionPattern {
    uint32_t mask;
    uint8_t num_deleted;
    uint8_t positions[MAX_N];
};

struct CollisionInfo {
    uint32_t codeword_idx;
    uint16_t pattern_idx;
};

inline int popcount32(uint32_t x) {
    #if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
    #else
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    #endif
}

std::vector<DeletionPattern> generate_deletion_patterns(int n, int s) {
    std::vector<DeletionPattern> patterns;
    uint32_t subset = (1u << s) - 1;
    uint64_t limit = 1ULL << n;
    while (subset < limit) {
        DeletionPattern pat;
        pat.mask = subset;
        pat.num_deleted = s;
        int pos_idx = 0;
        for (int i = 0; i < n && pos_idx < s; ++i) {
            if (subset & (1u << (n - 1 - i))) {
                pat.positions[pos_idx++] = i;
            }
        }
        patterns.push_back(pat);
        uint32_t c = subset & -subset;
        uint32_t r = subset + c;
        if (r < subset) break;
        if (c == 0) break;
        subset = (((r ^ subset) >> 2) / c) | r;
    }
    return patterns;
}

inline uint64_t apply_deletion_fast(uint32_t codeword, int n, uint32_t delete_mask) {
    if (delete_mask == 0) return codeword;
    uint64_t result = 0;
    int out_pos = 0;
    int num_deleted = popcount32(delete_mask);
    int out_len = n - num_deleted;
    for (int i = 0; i < n; ++i) {
        uint32_t bit_mask = 1u << (n - 1 - i);
        if (!(delete_mask & bit_mask)) {
            int bit = (codeword >> (n - 1 - i)) & 1;
            result |= (static_cast<uint64_t>(bit) << (out_len - 1 - out_pos));
            ++out_pos;
        }
    }
    return result;
}

struct SimpleWeights {
    std::vector<int64_t> weights;
    std::vector<int32_t> moduli;
    int r, n;

    void init(const int64_t* w, const int32_t* m, int num_constraints, int blocklength) {
        r = num_constraints;
        n = blocklength;
        weights.assign(w, w + r * n);
        moduli.assign(m, m + r);
    }
};

inline int64_t compute_syndrome_simple(
    uint64_t word,
    int len,
    int constraint_idx,
    const SimpleWeights& sw
) {
    int64_t syndrome = 0;
    for (int i = 0; i < len; ++i) {
        int bit = (word >> (len - 1 - i)) & 1;
        syndrome += bit * sw.weights[constraint_idx * sw.n + i];
    }
    return ((syndrome % sw.moduli[constraint_idx]) + sw.moduli[constraint_idx]) % sw.moduli[constraint_idx];
}

// Compute delta and optionally return received_syndrome for partition hashing
inline void compute_delta_simple(
    uint32_t codeword,
    uint64_t received_word,
    int n,
    int received_len,
    const SimpleWeights& sw,
    int32_t* delta_out,
    int32_t* received_syndrome_out = nullptr  // Optional: for partition hashing
) {
    for (int j = 0; j < sw.r; ++j) {
        int64_t original = compute_syndrome_simple(codeword, n, j, sw);
        int64_t received = compute_syndrome_simple(received_word, received_len, j, sw);
        int32_t m = sw.moduli[j];
        delta_out[j] = static_cast<int32_t>(((original - received) % m + m) % m);
        if (received_syndrome_out) {
            received_syndrome_out[j] = static_cast<int32_t>(received);
        }
    }
}

using PartitionHashTable = std::unordered_map<CompactSignature, CollisionInfo, CompactSignatureHash>;

inline size_t compute_num_partitions(int n, int s) {
    int received_len = n - s;
    if (received_len >= 24) {
        return MAX_PARTITIONS;
    }
    return 1ULL << received_len;
}

// Partition by hash of (received_word, received_syndrome) for better load distribution.
// Benchmarks show this is faster than received_word alone due to reduced lock contention.
inline size_t get_partition(uint64_t received_word, const int32_t* received_syndrome,
                            int num_constraints, size_t num_partitions) {
    uint64_t h = received_word;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;

    // Mix in received_syndrome values
    for (int i = 0; i < num_constraints; ++i) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(received_syndrome[i])) << ((i * 13) % 32);
    }

    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return h % num_partitions;
}

// Partition with its own mutex and hash table
struct LockedPartition {
    std::mutex mtx;
    PartitionHashTable hash_table;
    size_t collision_count = 0;
    std::vector<std::tuple<uint32_t, uint16_t, uint32_t, uint16_t>> collision_pairs;
};

py::dict validate_decoder_bruteforce_lowmem(
    int n, int s,
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

    if (n > MAX_N) throw std::runtime_error("n > 32 not supported");
    if (r > MAX_CONSTRAINTS) throw std::runtime_error("More than 8 constraints not supported");

    SimpleWeights sw;
    sw.init(weights, moduli, r, n);

    auto patterns = generate_deletion_patterns(n, s);
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

    // Step 2: Create partitions with locks
    size_t num_partitions = compute_num_partitions(n, s);
    std::vector<LockedPartition> partitions(num_partitions);

    std::atomic<size_t> total_signatures{0};
    int received_len = n - s;

    // Step 3: Generate signatures and insert directly into hash tables
    #pragma omp parallel
    {
        size_t local_count = 0;
        int32_t received_syndrome[MAX_CONSTRAINTS];  // Thread-local buffer for partition hashing

        #pragma omp for schedule(dynamic, 64)
        for (size_t cw_idx = 0; cw_idx < codebook_size; ++cw_idx) {
            uint32_t codeword = codewords[cw_idx];

            for (size_t pat_idx = 0; pat_idx < num_patterns; ++pat_idx) {
                const auto& pattern = patterns[pat_idx];
                ++local_count;

                CompactSignature sig;
                sig.received_word = apply_deletion_fast(codeword, n, pattern.mask);
                sig.num_constraints = r;

                // Compute delta and received_syndrome (for partition hashing)
                compute_delta_simple(codeword, sig.received_word, n, received_len, sw,
                                     sig.delta, received_syndrome);

                size_t part_id = get_partition(sig.received_word, received_syndrome,
                                               r, num_partitions);

                // Lock this partition and insert
                auto& part = partitions[part_id];
                {
                    std::lock_guard<std::mutex> lock(part.mtx);

                    auto it = part.hash_table.find(sig);
                    if (it != part.hash_table.end()) {
                        if (it->second.codeword_idx != cw_idx) {
                            ++part.collision_count;
                            if (return_collisions) {
                                part.collision_pairs.emplace_back(
                                    it->second.codeword_idx, it->second.pattern_idx,
                                    static_cast<uint32_t>(cw_idx), static_cast<uint16_t>(pat_idx)
                                );
                            }
                        }
                    } else {
                        part.hash_table[sig] = CollisionInfo{
                            static_cast<uint32_t>(cw_idx),
                            static_cast<uint16_t>(pat_idx)
                        };
                    }
                }
            }
        }

        total_signatures.fetch_add(local_count, std::memory_order_relaxed);
    }

    // Step 4: Aggregate results
    size_t unique_signatures = 0;
    size_t collision_count = 0;
    for (size_t i = 0; i < num_partitions; ++i) {
        unique_signatures += partitions[i].hash_table.size();
        collision_count += partitions[i].collision_count;
    }

    bool valid = (collision_count == 0);
    double score = total_signatures > 0 ?
        static_cast<double>(unique_signatures) / total_signatures : 1.0;

    py::dict result;
    result["valid"] = valid;
    result["codebook_size"] = codebook_size;
    result["total_signatures"] = total_signatures.load();
    result["unique_signatures"] = unique_signatures;
    result["collision_count"] = collision_count;
    result["score"] = score;

    if (return_collisions) {
        std::vector<py::dict> collisions;
        for (size_t part_id = 0; part_id < num_partitions; ++part_id) {
            for (const auto& [cw1, pat1, cw2, pat2] : partitions[part_id].collision_pairs) {
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
        }
        result["collisions"] = collisions;
    } else {
        result["collisions"] = py::none();
    }

    return result;
}

PYBIND11_MODULE(bruteforce_cpp, m) {
    m.doc() = "Memory-efficient brute-force decoder validation";

    m.def("validate_decoder_bruteforce", &validate_decoder_bruteforce_lowmem,
          py::arg("n"), py::arg("s"),
          py::arg("weights"), py::arg("moduli"), py::arg("targets"),
          py::arg("return_collisions") = false,
          R"doc(
Memory-efficient brute-force decoder validation.

Inserts signatures directly into hash tables instead of storing all signatures.
Memory usage: O(unique signatures) instead of O(total signatures).
Uses mutex per partition for thread-safe insertion.
)doc");

    m.attr("__version__") = "1.0.0";
}
