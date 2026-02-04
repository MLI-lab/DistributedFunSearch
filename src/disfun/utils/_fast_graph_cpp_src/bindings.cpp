#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fast_graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_graph_cpp, m) {
    m.doc() = "Fast graph implementation with CSR storage and C++ greedy solver";

    py::class_<fastgraph::FastGraphCpp>(m, "FastGraphCpp")
        .def(py::init<>())

        // Load from LMDB
        .def("load_from_lmdb", &fastgraph::FastGraphCpp::load_from_lmdb,
             py::arg("path"),
             "Load graph from LMDB database file")

        // Node access - return as tuple for consistency with Python FastGraph
        .def_property_readonly("nodes",
            [](const fastgraph::FastGraphCpp& g) {
                return py::tuple(py::cast(g.nodes()));
            },
            "Return all nodes as tuple")

        .def("number_of_nodes", &fastgraph::FastGraphCpp::number_of_nodes,
             "Return number of nodes")

        .def("number_of_edges", &fastgraph::FastGraphCpp::number_of_edges,
             "Return number of edges")

        // Neighbor access - return as tuple for consistency
        .def("neighbors",
            [](const fastgraph::FastGraphCpp& g, const std::string& node) {
                return py::tuple(py::cast(g.neighbors(node)));
            },
            py::arg("node"),
            "Return neighbors of node as tuple")

        // Degree access
        .def("degree",
            [](const fastgraph::FastGraphCpp& g, py::object node_or_none) {
                if (node_or_none.is_none()) {
                    return py::cast(g.degrees());
                }
                return py::cast(g.degree(node_or_none.cast<std::string>()));
            },
            py::arg("node") = py::none(),
            "Return degree of node, or dict of all degrees if node is None")

        // Node membership
        .def("has_node", &fastgraph::FastGraphCpp::has_node,
             py::arg("node"),
             "Check if node exists in graph")

        .def("__contains__", &fastgraph::FastGraphCpp::has_node,
             "Support 'node in G' syntax")

        .def("__len__", &fastgraph::FastGraphCpp::number_of_nodes,
             "Return number of nodes")

        .def("__iter__",
            [](const fastgraph::FastGraphCpp& g) {
                return py::make_iterator(g.nodes().begin(), g.nodes().end());
            },
            py::keep_alive<0, 1>(),
            "Iterate over nodes")

        // Fast greedy independent set
        .def("greedy_independent_set", &fastgraph::FastGraphCpp::greedy_independent_set,
             py::arg("priorities"),
             "Compute greedy independent set given priority dict. Returns list of nodes.")

        // Random greedy independent set (for baselines)
        .def("random_greedy_independent_set", &fastgraph::FastGraphCpp::random_greedy_independent_set,
             py::arg("seed"),
             "Compute random greedy independent set. Shuffles nodes using seed, then greedy.")

        // Run many random trials efficiently (for statistical analysis)
        .def("random_greedy_trials", &fastgraph::FastGraphCpp::random_greedy_trials,
             py::arg("num_trials"),
             py::arg("base_seed") = 42,
             "Run many random greedy trials and return list of sizes. Much faster than Python loop.");

    // Module-level function to load graph
    m.def("load_graph_from_lmdb",
        [](const std::string& path) {
            auto g = std::make_unique<fastgraph::FastGraphCpp>();
            g->load_from_lmdb(path);
            return g;
        },
        py::arg("path"),
        "Load graph from LMDB database and return FastGraphCpp instance");
}
