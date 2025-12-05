/**
 * Brute-force decoder validation for deletion-correcting codes.
 *
 * This C++ extension provides high-performance validation for large n (> 22).
 * It uses:
 * - Bitwise operations for codeword representation
 * - Precomputed deletion patterns
 * - Efficient hash table for collision detection
 * - OpenMP for parallelization
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Signature structure for hash table
struct Signature {
    uint64_t received_word;   // Received word as integer (up to 64 bits)
    uint8_t received_len;     // Length of received word
    std::vector<int32_t> delta; // Syndrome differences

    bool operator==(const Signature& other) const {
        return received_word == other.received_word &&
               received_len == other.received_len &&
               delta == other.delta;
    }
};

// Custom hash function for Signature
struct SignatureHash {
    size_t operator()(const Signature& sig) const {
        // FNV-1a hash
        size_t h = 14695981039346656037ULL;
        h ^= sig.received_word;
        h *= 1099511628211ULL;
        h ^= sig.received_len;
        h *= 1099511628211ULL;
        for (int32_t d : sig.delta) {
            h ^= static_cast<size_t>(d);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

// Collision info - stores which codeword/pattern caused the collision
struct CollisionInfo {
    uint32_t codeword_idx;
    uint16_t pattern_idx;
};

// Count bits set in an integer (popcount)
inline int popcount(uint64_t x) {
    #if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
    #else
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
    #endif
}

// Generate all deletion patterns for given n and s
std::vector<std::vector<int>> generate_deletion_patterns(int n, int s) {
    std::vector<std::vector<int>> patterns;

    // Pattern with no deletions (empty)
    patterns.push_back({});

    // Generate combinations for each deletion count d = 1, 2, ..., s
    for (int d = 1; d <= s; ++d) {
        std::vector<int> indices(d);
        std::iota(indices.begin(), indices.end(), 0); // Initialize to 0, 1, ..., d-1

        while (true) {
            patterns.push_back(indices);

            // Generate next combination
            int i = d - 1;
            while (i >= 0 && indices[i] == n - d + i) {
                --i;
            }
            if (i < 0) break;
            ++indices[i];
            for (int j = i + 1; j < d; ++j) {
                indices[j] = indices[j - 1] + 1;
            }
        }
    }

    return patterns;
}

// Apply deletion pattern to codeword, return received word as integer
uint64_t apply_deletion(uint64_t codeword, int n, const std::vector<int>& positions) {
    if (positions.empty()) {
        return codeword;
    }

    uint64_t received = 0;
    int recv_idx = 0;
    size_t pos_idx = 0;

    for (int i = 0; i < n; ++i) {
        if (pos_idx < positions.size() && i == positions[pos_idx]) {
            // Skip this position (deleted)
            ++pos_idx;
        } else {
            // Keep this bit
            int bit = (codeword >> (n - 1 - i)) & 1;
            received |= (static_cast<uint64_t>(bit) << (n - 1 - static_cast<int>(positions.size()) - recv_idx));
            ++recv_idx;
        }
    }

    return received;
}

// Compute syndrome difference for a deletion pattern
std::vector<int32_t> compute_syndrome_difference(
    uint64_t codeword,
    int n,
    const std::vector<int>& positions,
    const int64_t* weights,  // r x n row-major
    const int32_t* moduli,
    int r
) {
    std::vector<int32_t> delta(r);

    for (int j = 0; j < r; ++j) {
        int64_t contrib = 0;
        int64_t shift_correction = 0;

        // Convert codeword to bits array for easier processing
        std::vector<int> bits(n);
        for (int i = 0; i < n; ++i) {
            bits[i] = (codeword >> (n - 1 - i)) & 1;
        }

        // Contribution from deleted bits
        for (int pos : positions) {
            contrib += bits[pos] * weights[j * n + pos];
        }

        // Shift correction: for remaining positions, compute weight difference
        std::vector<bool> is_deleted(n, false);
        for (int pos : positions) {
            is_deleted[pos] = true;
        }

        int shift = 0;
        for (int i = 0; i < n; ++i) {
            if (is_deleted[i]) {
                ++shift;
            } else {
                int shifted_pos = i - shift;
                if (shifted_pos >= 0 && shifted_pos < n - static_cast<int>(positions.size())) {
                    shift_correction += bits[i] * (weights[j * n + i] - weights[j * n + shifted_pos]);
                }
            }
        }

        int32_t m = moduli[j];
        delta[j] = static_cast<int32_t>(((contrib + shift_correction) % m + m) % m);
    }

    return delta;
}

// Main validation function
py::dict validate_decoder_bruteforce(
    int n,
    int s,
    py::array_t<int64_t> weights_arr,
    py::array_t<int32_t> moduli_arr,
    py::array_t<int32_t> targets_arr,
    bool return_collisions
) {
    // Get array info
    auto weights_info = weights_arr.request();
    auto moduli_info = moduli_arr.request();
    auto targets_info = targets_arr.request();

    int r = static_cast<int>(weights_info.shape[0]);
    const int64_t* weights = static_cast<const int64_t*>(weights_info.ptr);
    const int32_t* moduli = static_cast<const int32_t*>(moduli_info.ptr);
    const int32_t* targets = static_cast<const int32_t*>(targets_info.ptr);

    // Limit n to 32 bits for now (can extend to 64 if needed)
    if (n > 32) {
        throw std::runtime_error("n > 32 not supported in current implementation");
    }

    // Step 1: Generate deletion patterns
    auto deletion_patterns = generate_deletion_patterns(n, s);
    size_t num_patterns = deletion_patterns.size();

    // Step 2: Enumerate codewords satisfying constraints
    std::vector<uint32_t> codewords;
    uint64_t total_candidates = 1ULL << n;

    #pragma omp parallel
    {
        std::vector<uint32_t> local_codewords;

        #pragma omp for nowait
        for (uint64_t x = 0; x < total_candidates; ++x) {
            // Check if x satisfies all constraints
            bool valid = true;
            for (int j = 0; j < r && valid; ++j) {
                int64_t syndrome = 0;
                for (int i = 0; i < n; ++i) {
                    int bit = (x >> (n - 1 - i)) & 1;
                    syndrome += bit * weights[j * n + i];
                }
                syndrome = ((syndrome % moduli[j]) + moduli[j]) % moduli[j];
                if (syndrome != targets[j]) {
                    valid = false;
                }
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

    // Step 3: Compute signatures and detect collisions
    std::unordered_map<Signature, CollisionInfo, SignatureHash> seen;
    std::vector<py::dict> collisions;
    size_t total_signatures = 0;
    size_t collision_count = 0;

    // Sequential processing for collision detection (hash table not thread-safe easily)
    for (size_t cw_idx = 0; cw_idx < codebook_size; ++cw_idx) {
        uint32_t codeword = codewords[cw_idx];

        for (size_t pat_idx = 0; pat_idx < num_patterns; ++pat_idx) {
            ++total_signatures;

            const auto& positions = deletion_patterns[pat_idx];

            // Compute received word
            uint64_t received = apply_deletion(codeword, n, positions);
            uint8_t received_len = static_cast<uint8_t>(n - positions.size());

            // Compute syndrome difference
            auto delta = compute_syndrome_difference(codeword, n, positions, weights, moduli, r);

            // Create signature
            Signature sig{received, received_len, std::move(delta)};

            auto it = seen.find(sig);
            if (it != seen.end()) {
                uint32_t prev_cw_idx = it->second.codeword_idx;
                if (prev_cw_idx != cw_idx) {
                    ++collision_count;
                    if (return_collisions) {
                        py::dict coll;
                        coll["codeword1"] = static_cast<int64_t>(codewords[prev_cw_idx]);
                        coll["pattern1"] = deletion_patterns[it->second.pattern_idx];
                        coll["codeword2"] = static_cast<int64_t>(codeword);
                        coll["pattern2"] = positions;
                        collisions.push_back(coll);
                    }
                }
            } else {
                CollisionInfo info{static_cast<uint32_t>(cw_idx), static_cast<uint16_t>(pat_idx)};
                seen[sig] = info;
            }
        }
    }

    size_t unique_signatures = seen.size();
    bool valid = (collision_count == 0);
    double score = total_signatures > 0 ? static_cast<double>(unique_signatures) / total_signatures : 1.0;

    py::dict result;
    result["valid"] = valid;
    result["codebook_size"] = codebook_size;
    result["total_signatures"] = total_signatures;
    result["unique_signatures"] = unique_signatures;
    result["collision_count"] = collision_count;
    result["score"] = score;

    if (return_collisions) {
        result["collisions"] = collisions;
    } else {
        result["collisions"] = py::none();
    }

    return result;
}

PYBIND11_MODULE(bruteforce_cpp, m) {
    m.doc() = "C++ extension for brute-force decoder validation";

    m.def("validate_decoder_bruteforce", &validate_decoder_bruteforce,
          py::arg("n"),
          py::arg("s"),
          py::arg("weights"),
          py::arg("moduli"),
          py::arg("targets"),
          py::arg("return_collisions") = false,
          R"doc(
Brute-force decoder validation for deletion-correcting codes.

Args:
    n: Codeword length (blocklength)
    s: Maximum number of deletions
    weights: r x n weight matrix (numpy array, int64)
    moduli: r moduli (numpy array, int32)
    targets: r target syndromes (numpy array, int32)
    return_collisions: If True, return collision details

Returns:
    Dictionary with validation results
)doc");

    m.attr("__version__") = VERSION_INFO;
}
