/**
 * Optimized brute-force decoder validation v4.
 *
 * Key insight: Signatures can only collide if received_word is IDENTICAL.
 * Therefore, we can partition by received_word into completely independent
 * hash tables - NO LOCKING NEEDED!
 *
 * This is much faster than v2's sequential merge or v3's locked hash table.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

constexpr int MAX_CONSTRAINTS = 8;
constexpr int MAX_N = 32;
constexpr int NUM_PARTITIONS = 1024;  // Partition by received_word % NUM_PARTITIONS

struct CompactSignature {
    uint64_t received_word;
    uint8_t received_len;
    uint8_t num_constraints;
    int32_t delta[MAX_CONSTRAINTS];

    bool operator==(const CompactSignature& other) const {
        if (received_word != other.received_word || received_len != other.received_len) return false;
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
        h ^= sig.received_len;
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
    DeletionPattern empty;
    empty.mask = 0;
    empty.num_deleted = 0;
    patterns.push_back(empty);

    for (int d = 1; d <= s; ++d) {
        uint32_t subset = (1u << d) - 1;
        uint64_t limit = 1ULL << n;  // Use 64-bit to handle n=32
        while (subset < limit) {
            DeletionPattern pat;
            pat.mask = subset;
            pat.num_deleted = d;
            int pos_idx = 0;
            for (int i = 0; i < n && pos_idx < d; ++i) {
                if (subset & (1u << (n - 1 - i))) {
                    pat.positions[pos_idx++] = i;
                }
            }
            patterns.push_back(pat);
            // Gosper's hack for next subset (with overflow protection)
            uint32_t c = subset & -subset;
            uint32_t r = subset + c;
            if (r < subset) break;  // Overflow detected
            if (c == 0) break;      // Safety check
            subset = (((r ^ subset) >> 2) / c) | r;
        }
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

struct PrecomputedWeights {
    std::vector<int64_t> weights;
    std::vector<int32_t> moduli;
    int r, n;
    std::vector<std::vector<int64_t>> weight_diffs;

    void init(const int64_t* w, const int32_t* m, int num_constraints, int blocklength) {
        r = num_constraints;
        n = blocklength;
        weights.assign(w, w + r * n);
        moduli.assign(m, m + r);
        weight_diffs.resize(r);
        for (int j = 0; j < r; ++j) {
            weight_diffs[j].resize(n * n, 0);
            for (int i = 0; i < n; ++i) {
                for (int shift = 0; shift <= i; ++shift) {
                    weight_diffs[j][i * n + shift] = weights[j * n + i] - weights[j * n + (i - shift)];
                }
            }
        }
    }
};

inline void compute_syndrome_fast(
    uint32_t codeword, int n,
    const DeletionPattern& pattern,
    const PrecomputedWeights& pw,
    int32_t* delta_out
) {
    for (int j = 0; j < pw.r; ++j) {
        int64_t contrib = 0, shift_correction = 0;
        int shift = 0, del_idx = 0;
        for (int i = 0; i < n; ++i) {
            int bit = (codeword >> (n - 1 - i)) & 1;
            if (del_idx < pattern.num_deleted && i == pattern.positions[del_idx]) {
                contrib += bit * pw.weights[j * n + i];
                ++shift; ++del_idx;
            } else {
                shift_correction += bit * pw.weight_diffs[j][i * n + shift];
            }
        }
        int32_t m = pw.moduli[j];
        delta_out[j] = static_cast<int32_t>(((contrib + shift_correction) % m + m) % m);
    }
}

using PartitionHashTable = std::unordered_map<CompactSignature, CollisionInfo, CompactSignatureHash>;

// Get partition ID based on received_word - signatures with different received_word NEVER collide
inline int get_partition(uint64_t received_word) {
    // Use the lower bits of received_word for partitioning
    // This ensures signatures with the same received_word go to the same partition
    return static_cast<int>(received_word % NUM_PARTITIONS);
}

py::dict validate_decoder_bruteforce_v4(
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

    PrecomputedWeights pw;
    pw.init(weights, moduli, r, n);

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

    // Step 2: Generate signatures partitioned by received_word
    // Each partition is completely independent - NO LOCKING NEEDED!

    // Structure to hold signatures for each partition
    struct PartitionData {
        std::vector<std::tuple<CompactSignature, uint32_t, uint16_t>> signatures;
    };

    std::vector<std::vector<PartitionData>> thread_partitions;
    int num_threads = 1;
    #ifdef _OPENMP
    num_threads = omp_get_max_threads();
    #endif
    thread_partitions.resize(num_threads);
    for (auto& tp : thread_partitions) {
        tp.resize(NUM_PARTITIONS);
    }

    std::atomic<size_t> total_signatures{0};

    // Phase 1: Generate and partition signatures (fully parallel)
    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        auto& my_partitions = thread_partitions[tid];
        size_t local_count = 0;

        #pragma omp for schedule(dynamic, 64)
        for (size_t cw_idx = 0; cw_idx < codebook_size; ++cw_idx) {
            uint32_t codeword = codewords[cw_idx];

            for (size_t pat_idx = 0; pat_idx < num_patterns; ++pat_idx) {
                const auto& pattern = patterns[pat_idx];
                ++local_count;

                CompactSignature sig;
                sig.received_word = apply_deletion_fast(codeword, n, pattern.mask);
                sig.received_len = n - pattern.num_deleted;
                sig.num_constraints = r;
                compute_syndrome_fast(codeword, n, pattern, pw, sig.delta);

                int part_id = get_partition(sig.received_word);
                my_partitions[part_id].signatures.emplace_back(
                    sig, static_cast<uint32_t>(cw_idx), static_cast<uint16_t>(pat_idx)
                );
            }
        }

        total_signatures.fetch_add(local_count, std::memory_order_relaxed);
    }

    // Phase 2: Process each partition independently (fully parallel, NO LOCKS!)
    std::vector<std::atomic<size_t>> partition_unique(NUM_PARTITIONS);
    std::vector<std::atomic<size_t>> partition_collisions(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        partition_unique[i] = 0;
        partition_collisions[i] = 0;
    }

    std::vector<std::vector<std::tuple<uint32_t, uint16_t, uint32_t, uint16_t>>> partition_collision_pairs(NUM_PARTITIONS);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int part_id = 0; part_id < NUM_PARTITIONS; ++part_id) {
        PartitionHashTable hash_table;
        size_t local_collisions = 0;

        // Collect all signatures for this partition from all threads
        for (int tid = 0; tid < num_threads; ++tid) {
            for (const auto& [sig, cw_idx, pat_idx] : thread_partitions[tid][part_id].signatures) {
                auto it = hash_table.find(sig);
                if (it != hash_table.end()) {
                    if (it->second.codeword_idx != cw_idx) {
                        ++local_collisions;
                        if (return_collisions) {
                            partition_collision_pairs[part_id].emplace_back(
                                it->second.codeword_idx, it->second.pattern_idx,
                                cw_idx, pat_idx
                            );
                        }
                    }
                } else {
                    hash_table[sig] = CollisionInfo{cw_idx, pat_idx};
                }
            }
        }

        partition_unique[part_id] = hash_table.size();
        partition_collisions[part_id] = local_collisions;
    }

    // Aggregate results
    size_t unique_signatures = 0;
    size_t collision_count = 0;
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        unique_signatures += partition_unique[i].load();
        collision_count += partition_collisions[i].load();
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
        for (int part_id = 0; part_id < NUM_PARTITIONS; ++part_id) {
            for (const auto& [cw1, pat1, cw2, pat2] : partition_collision_pairs[part_id]) {
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
    m.doc() = "Fully parallel C++ extension v4 - lock-free partitioned hash tables";

    m.def("validate_decoder_bruteforce", &validate_decoder_bruteforce_v4,
          py::arg("n"), py::arg("s"),
          py::arg("weights"), py::arg("moduli"), py::arg("targets"),
          py::arg("return_collisions") = false,
          R"doc(
Lock-free parallel brute-force decoder validation.

Key insight: Signatures can only collide if received_word is IDENTICAL.
We partition by received_word % 1024, so each partition is completely
independent - NO LOCKING NEEDED!

Phase 1: Parallel signature generation into partitioned buffers
Phase 2: Parallel collision detection per partition (no synchronization)
)doc");

    m.attr("__version__") = VERSION_INFO;
}
