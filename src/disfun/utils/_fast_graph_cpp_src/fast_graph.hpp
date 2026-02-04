#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>

namespace fastgraph {

/**
 * Memory-efficient graph using CSR (Compressed Sparse Row) format.
 *
 * CSR uses two arrays:
 * - indptr[i] = start index of neighbors for node i
 * - indices[indptr[i]:indptr[i+1]] = neighbor indices for node i
 *
 * Memory: O(nodes + edges) instead of O(nodes * avg_degree) for adjacency list
 */
class FastGraphCpp {
public:
    FastGraphCpp() = default;

    // Build graph from node list and edge list
    void build(const std::vector<std::string>& nodes,
               const std::vector<std::pair<std::string, std::string>>& edges);

    // Load from LMDB database (auto-selects single-pass or streaming based on size)
    void load_from_lmdb(const std::string& path);

    // Single-pass: fast but high memory (for graphs < 500M edges)
    void load_from_lmdb_single_pass(const std::string& path);

    // Streaming: slow (3 passes) but low memory (for huge graphs)
    void load_from_lmdb_streaming(const std::string& path);

    // Node access
    const std::vector<std::string>& nodes() const { return nodes_; }
    size_t number_of_nodes() const { return nodes_.size(); }
    size_t number_of_edges() const { return num_edges_; }

    // Neighbor access - returns vector of neighbor node strings
    std::vector<std::string> neighbors(const std::string& node) const;

    // Degree access
    size_t degree(const std::string& node) const;

    // Get all degrees as a map
    std::unordered_map<std::string, size_t> degrees() const;

    // Check if node exists
    bool has_node(const std::string& node) const;

    // Fast greedy independent set using priorities
    // priorities[node] = priority value (higher = picked first)
    std::vector<std::string> greedy_independent_set(
        const std::unordered_map<std::string, double>& priorities) const;

    // Random greedy independent set (for baseline comparison)
    // Shuffles nodes using seed, then greedy selection
    std::vector<std::string> random_greedy_independent_set(uint64_t seed) const;

    // Run multiple random trials and return sizes (for statistical analysis)
    // Much faster than calling random_greedy_independent_set in a Python loop
    std::vector<size_t> random_greedy_trials(size_t num_trials, uint64_t base_seed) const;

private:
    // Node data
    std::vector<std::string> nodes_;
    std::unordered_map<std::string, uint32_t> node_to_idx_;

    // CSR format adjacency
    std::vector<uint32_t> indptr_;   // Size: nodes + 1
    std::vector<uint32_t> indices_;  // Size: 2 * edges (undirected)

    size_t num_edges_ = 0;
};

} // namespace fastgraph
