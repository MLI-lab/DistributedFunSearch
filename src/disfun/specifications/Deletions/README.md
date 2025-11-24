# Deletion-Correcting Codes Specification

Find large codes where codewords have no common subsequence of length ≥ n-s.

## Problem

- **Nodes:** Binary strings of length n
- **Edges:** Two nodes connected if `LCS(node1, node2) ≥ n - s`
- **Goal:** Find maximum independent set (valid deletion-correcting code)

## Directory Structure

```
Deletions/
├── problem_descriptions/       # Problem statement (no imports)
│   ├── baseline.txt           # Generic description
│   ├── explicit_s1.txt        # Explicit s=1 case
│   ├── no_graph.txt           # String-only variant
│   └── string_properties.txt  # With string pattern hints
├── imports/                    # Import statements (NEW)
│   ├── graph_tool.txt         # graph-tool imports
│   ├── networkx.txt           # NetworkX imports
│   └── no_graph.txt           # No graph library
├── prompt_styles/              # Output format instructions
│   ├── funsearch.txt          # Code only
│   ├── eoh.txt                # Thought + code
│   └── extended_eoh.txt       # Thinking + thought + code
├── system_messages/            # System prompts for API models
│   ├── graph.txt              # For graph variants
│   └── no_graph.txt           # For string-only variant
├── initial_functions/          # Seed functions
│   ├── graph_gt/zero.txt      # graph-tool baseline
│   ├── graph_networkx/zero.txt # NetworkX baseline
│   └── no_graph/zero.txt      # String-only baseline
└── evaluation/                 # Execution + scoring
    ├── graph_gt.py            # graph-tool (10-100x faster than NetworkX)
    ├── graph_networkx.py      # NetworkX (easier for LLMs)
    └── no_graph.py            # On-the-fly evaluation (slow)
```
