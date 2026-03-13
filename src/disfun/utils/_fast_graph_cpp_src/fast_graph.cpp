#include "fast_graph.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <lmdb.h>
#include <cstring>

// Simple JSON array parser for neighbor lists like ["node1","node2"]
static std::vector<std::string> parse_json_array(const char* data, size_t len) {
    std::vector<std::string> result;
    if (len < 2 || data[0] != '[') return result;

    size_t i = 1;
    while (i < len) {
        // Skip whitespace and commas
        while (i < len && (data[i] == ' ' || data[i] == ',' || data[i] == '\n')) i++;
        if (i >= len || data[i] == ']') break;

        // Expect opening quote
        if (data[i] != '"') break;
        i++;

        // Read string until closing quote
        size_t start = i;
        while (i < len && data[i] != '"') i++;
        if (i >= len) break;

        result.emplace_back(data + start, i - start);
        i++; // Skip closing quote
    }
    return result;
}

namespace fastgraph {

void FastGraphCpp::build(const std::vector<std::string>& nodes,
                         const std::vector<std::pair<std::string, std::string>>& edges) {
    nodes_ = nodes;
    num_edges_ = edges.size();

    // Build node to index mapping
    node_to_idx_.clear();
    node_to_idx_.reserve(nodes.size());
    for (uint32_t i = 0; i < nodes.size(); i++) {
        node_to_idx_[nodes[i]] = i;
    }

    // Count neighbors per node
    std::vector<uint32_t> neighbor_counts(nodes.size(), 0);
    for (const auto& edge : edges) {
        auto it1 = node_to_idx_.find(edge.first);
        auto it2 = node_to_idx_.find(edge.second);
        if (it1 != node_to_idx_.end() && it2 != node_to_idx_.end()) {
            neighbor_counts[it1->second]++;
            neighbor_counts[it2->second]++;
        }
    }

    // Build indptr (cumulative sum of neighbor counts)
    indptr_.resize(nodes.size() + 1);
    indptr_[0] = 0;
    for (size_t i = 0; i < nodes.size(); i++) {
        indptr_[i + 1] = indptr_[i] + neighbor_counts[i];
    }

    // Allocate indices array
    indices_.resize(indptr_.back());

    // Fill indices (use neighbor_counts as write position tracker)
    std::fill(neighbor_counts.begin(), neighbor_counts.end(), 0);
    for (const auto& edge : edges) {
        uint32_t idx1 = node_to_idx_[edge.first];
        uint32_t idx2 = node_to_idx_[edge.second];

        indices_[indptr_[idx1] + neighbor_counts[idx1]++] = idx2;
        indices_[indptr_[idx2] + neighbor_counts[idx2]++] = idx1;
    }

    // Sort neighbors for each node (optional, for consistent iteration order)
    for (size_t i = 0; i < nodes.size(); i++) {
        std::sort(indices_.begin() + indptr_[i], indices_.begin() + indptr_[i + 1]);
    }
}

// Helper to open LMDB and get cursor
static void open_lmdb(const std::string& path, MDB_env*& env, MDB_txn*& txn,
                      MDB_dbi& dbi, MDB_cursor*& cursor) {
    int rc = mdb_env_create(&env);
    if (rc) throw std::runtime_error("mdb_env_create failed");

    rc = mdb_env_set_mapsize(env, 100ULL * 1024 * 1024 * 1024); // 100GB max
    if (rc) throw std::runtime_error("mdb_env_set_mapsize failed");

    rc = mdb_env_open(env, path.c_str(), MDB_RDONLY | MDB_NOLOCK, 0644);
    if (rc) {
        mdb_env_close(env);
        throw std::runtime_error("mdb_env_open failed: " + path + " (" + mdb_strerror(rc) + ")");
    }

    rc = mdb_txn_begin(env, nullptr, MDB_RDONLY, &txn);
    if (rc) {
        mdb_env_close(env);
        throw std::runtime_error("mdb_txn_begin failed: " + path + " (" + mdb_strerror(rc) + ")");
    }

    rc = mdb_dbi_open(txn, nullptr, 0, &dbi);
    if (rc) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        throw std::runtime_error("mdb_dbi_open failed");
    }

    rc = mdb_cursor_open(txn, dbi, &cursor);
    if (rc) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        throw std::runtime_error("mdb_cursor_open failed");
    }
}

static void close_lmdb(MDB_env* env, MDB_txn* txn, MDB_cursor* cursor) {
    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    mdb_env_close(env);
}

// Single-pass loader: fast but uses more memory (stores edge pairs temporarily)
void FastGraphCpp::load_from_lmdb_single_pass(const std::string& path) {
    MDB_env* env;
    MDB_dbi dbi;
    MDB_txn* txn;
    MDB_cursor* cursor;
    MDB_val key, value;
    int rc;

    open_lmdb(path, env, txn, dbi, cursor);

    std::vector<std::string> nodes;
    std::vector<std::pair<std::string, std::string>> edges;

    while ((rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT)) == 0) {
        std::string node(static_cast<char*>(key.mv_data), key.mv_size);
        nodes.push_back(node);

        auto neighbors = parse_json_array(static_cast<char*>(value.mv_data), value.mv_size);
        for (const auto& neighbor : neighbors) {
            if (node < neighbor) {
                edges.emplace_back(node, neighbor);
            }
        }
    }

    close_lmdb(env, txn, cursor);
    build(nodes, edges);
}

// Streaming loader: slower (3 passes) but low memory for huge graphs
void FastGraphCpp::load_from_lmdb_streaming(const std::string& path) {
    MDB_env* env;
    MDB_dbi dbi;
    MDB_txn* txn;
    MDB_cursor* cursor;
    MDB_val key, value;
    int rc;

    // Pass 1: Collect nodes and build node_to_idx
    open_lmdb(path, env, txn, dbi, cursor);

    nodes_.clear();
    node_to_idx_.clear();

    while ((rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT)) == 0) {
        std::string node(static_cast<char*>(key.mv_data), key.mv_size);
        node_to_idx_[node] = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back(std::move(node));
    }

    close_lmdb(env, txn, cursor);

    size_t num_nodes = nodes_.size();

    // Pass 2: Count neighbors per node
    open_lmdb(path, env, txn, dbi, cursor);

    std::vector<uint32_t> neighbor_counts(num_nodes, 0);
    num_edges_ = 0;

    while ((rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT)) == 0) {
        std::string node(static_cast<char*>(key.mv_data), key.mv_size);
        uint32_t node_idx = node_to_idx_[node];

        auto neighbors = parse_json_array(static_cast<char*>(value.mv_data), value.mv_size);
        for (const auto& neighbor : neighbors) {
            auto it = node_to_idx_.find(neighbor);
            if (it != node_to_idx_.end()) {
                neighbor_counts[node_idx]++;
                if (node < neighbor) {
                    num_edges_++;
                }
            }
        }
    }

    close_lmdb(env, txn, cursor);

    // Build indptr
    indptr_.resize(num_nodes + 1);
    indptr_[0] = 0;
    for (size_t i = 0; i < num_nodes; i++) {
        indptr_[i + 1] = indptr_[i] + neighbor_counts[i];
    }

    indices_.resize(indptr_.back());

    // Pass 3: Fill indices
    open_lmdb(path, env, txn, dbi, cursor);

    std::fill(neighbor_counts.begin(), neighbor_counts.end(), 0);

    while ((rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT)) == 0) {
        std::string node(static_cast<char*>(key.mv_data), key.mv_size);
        uint32_t node_idx = node_to_idx_[node];

        auto neighbors = parse_json_array(static_cast<char*>(value.mv_data), value.mv_size);
        for (const auto& neighbor : neighbors) {
            auto it = node_to_idx_.find(neighbor);
            if (it != node_to_idx_.end()) {
                uint32_t neighbor_idx = it->second;
                indices_[indptr_[node_idx] + neighbor_counts[node_idx]++] = neighbor_idx;
            }
        }
    }

    close_lmdb(env, txn, cursor);

    // Sort neighbors for consistent iteration order
    for (size_t i = 0; i < num_nodes; i++) {
        std::sort(indices_.begin() + indptr_[i], indices_.begin() + indptr_[i + 1]);
    }
}

void FastGraphCpp::load_from_lmdb(const std::string& path) {
    MDB_env* env;
    MDB_stat stat;
    int rc;

    // Check LMDB size to decide loading strategy without full scan
    rc = mdb_env_create(&env);
    if (rc) throw std::runtime_error("mdb_env_create failed");

    rc = mdb_env_set_mapsize(env, 100ULL * 1024 * 1024 * 1024);
    if (rc) throw std::runtime_error("mdb_env_set_mapsize failed");

    rc = mdb_env_open(env, path.c_str(), MDB_RDONLY | MDB_NOLOCK, 0644);
    if (rc) {
        mdb_env_close(env);
        throw std::runtime_error("mdb_env_open failed: " + path + " (" + mdb_strerror(rc) + ")");
    }

    // Get database statistics
    MDB_txn* txn;
    MDB_dbi dbi;
    rc = mdb_txn_begin(env, nullptr, MDB_RDONLY, &txn);
    if (rc) {
        mdb_env_close(env);
        throw std::runtime_error("mdb_txn_begin failed");
    }

    rc = mdb_dbi_open(txn, nullptr, 0, &dbi);
    if (rc) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        throw std::runtime_error("mdb_dbi_open failed");
    }

    rc = mdb_stat(txn, dbi, &stat);
    mdb_txn_abort(txn);
    mdb_env_close(env);

    if (rc) {
        throw std::runtime_error("mdb_stat failed");
    }

    // Use streaming only for huge graphs (> 2M nodes, like IDS n=11 with 4.2M nodes)
    // IDS n=10 has 1M nodes, n=11 has 4.2M nodes
    if (stat.ms_entries > 2000000) {
        load_from_lmdb_streaming(path);
    } else {
        load_from_lmdb_single_pass(path);
    }
}

std::vector<std::string> FastGraphCpp::neighbors(const std::string& node) const {
    auto it = node_to_idx_.find(node);
    if (it == node_to_idx_.end()) {
        return {};
    }

    uint32_t idx = it->second;
    std::vector<std::string> result;
    result.reserve(indptr_[idx + 1] - indptr_[idx]);

    for (uint32_t i = indptr_[idx]; i < indptr_[idx + 1]; i++) {
        result.push_back(nodes_[indices_[i]]);
    }
    return result;
}

size_t FastGraphCpp::degree(const std::string& node) const {
    auto it = node_to_idx_.find(node);
    if (it == node_to_idx_.end()) {
        return 0;
    }
    uint32_t idx = it->second;
    return indptr_[idx + 1] - indptr_[idx];
}

std::unordered_map<std::string, size_t> FastGraphCpp::degrees() const {
    std::unordered_map<std::string, size_t> result;
    result.reserve(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); i++) {
        result[nodes_[i]] = indptr_[i + 1] - indptr_[i];
    }
    return result;
}

bool FastGraphCpp::has_node(const std::string& node) const {
    return node_to_idx_.find(node) != node_to_idx_.end();
}

std::vector<std::string> FastGraphCpp::greedy_independent_set(
    const std::unordered_map<std::string, double>& priorities) const {

    size_t n = nodes_.size();
    if (n == 0) return {};

    // Create sorted indices by priority (descending), with lexicographic tie-breaking
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        auto it_a = priorities.find(nodes_[a]);
        auto it_b = priorities.find(nodes_[b]);
        double pa = (it_a != priorities.end()) ? it_a->second : 0.0;
        double pb = (it_b != priorities.end()) ? it_b->second : 0.0;
        if (pa != pb) return pa > pb; // Higher priority first
        return nodes_[a] < nodes_[b]; // Lexicographic tie-break
    });

    // Greedy selection with removed tracking
    std::vector<bool> removed(n, false);
    std::vector<std::string> independent_set;
    independent_set.reserve(n / 10); // Rough estimate

    for (uint32_t idx : order) {
        if (removed[idx]) continue;

        independent_set.push_back(nodes_[idx]);
        removed[idx] = true;

        // Mark all neighbors as removed
        for (uint32_t i = indptr_[idx]; i < indptr_[idx + 1]; i++) {
            removed[indices_[i]] = true;
        }
    }

    return independent_set;
}

std::vector<std::string> FastGraphCpp::random_greedy_independent_set(uint64_t seed) const {
    size_t n = nodes_.size();
    if (n == 0) return {};

    // Create shuffled order using Fisher-Yates with simple LCG
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    // Simple LCG random number generator (fast, deterministic)
    uint64_t state = seed;
    auto next_rand = [&state]() -> uint64_t {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    };

    // Fisher-Yates shuffle
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = next_rand() % (i + 1);
        std::swap(order[i], order[j]);
    }

    // Greedy selection
    std::vector<bool> removed(n, false);
    std::vector<std::string> independent_set;
    independent_set.reserve(n / 10);

    for (uint32_t idx : order) {
        if (removed[idx]) continue;

        independent_set.push_back(nodes_[idx]);
        removed[idx] = true;

        for (uint32_t i = indptr_[idx]; i < indptr_[idx + 1]; i++) {
            removed[indices_[i]] = true;
        }
    }

    return independent_set;
}

std::vector<size_t> FastGraphCpp::random_greedy_trials(size_t num_trials, uint64_t base_seed) const {
    size_t n = nodes_.size();
    if (n == 0) return std::vector<size_t>(num_trials, 0);

    std::vector<size_t> sizes(num_trials);

    // Pre-allocate reusable buffers
    std::vector<uint32_t> order(n);
    std::vector<bool> removed(n);

    for (size_t trial = 0; trial < num_trials; trial++) {
        // Reset order
        std::iota(order.begin(), order.end(), 0);

        // LCG with trial-specific seed
        uint64_t state = base_seed + trial;
        auto next_rand = [&state]() -> uint64_t {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return state;
        };

        // Fisher-Yates shuffle
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = next_rand() % (i + 1);
            std::swap(order[i], order[j]);
        }

        // Reset removed flags
        std::fill(removed.begin(), removed.end(), false);

        // Greedy selection (just count, don't store nodes)
        size_t count = 0;
        for (uint32_t idx : order) {
            if (removed[idx]) continue;

            count++;
            removed[idx] = true;

            for (uint32_t i = indptr_[idx]; i < indptr_[idx + 1]; i++) {
                removed[indices_[i]] = true;
            }
        }

        sizes[trial] = count;
    }

    return sizes;
}

} // namespace fastgraph
