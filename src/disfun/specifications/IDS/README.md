# Insertion/Deletion/Substitution-Correcting Codes Specification

Find large codes where codewords have pairwise edit distance ≥ 2s+1.

## Problem

- **Nodes:** q-ary strings of length n
- **Edges:** Two nodes connected if `edit_distance(node1, node2) < 2s + 1`
- **Goal:** Find maximum independent set (valid IDS-correcting code)

## Directory Structure

```
IDS/
├── problem_descriptions/       # Problem statement (no imports)
│   ├── baseline.txt           # Generic description
│   ├── explicit_s1.txt        # Explicit s=1 case
│   └── string_properties.txt  # With string pattern hints
├── imports/                    # Import statements
│   ├── graph_tool.txt         # graph-tool imports
│   ├── networkx.txt           # NetworkX imports
│   └── no_graph.txt           # Levenshtein + NetworkX
├── prompt_styles/              # Output format instructions
│   ├── funsearch.txt          # Code only
│   ├── eoh.txt                # Thought + code
│   └── extended_eoh.txt       # Thinking + thought + code
├── system_messages/            # System prompts for API models
│   └── graph.txt              # For all variants
├── initial_functions/          # Seed functions
│   ├── graph_gt/zero.txt      # graph-tool baseline
│   ├── graph_networkx/zero.txt # NetworkX baseline
│   └── no_graph/zero.txt      # String-only baseline
└── evaluation/                 # Execution + scoring
    ├── graph_gt.py            # graph-tool (10-100x faster than NetworkX)
    ├── graph_networkx.py      # NetworkX (easier for LLMs)
    └── no_graph.py            # On-the-fly with Levenshtein distance
```

## Key Differences from Deletions

- **Distance metric:** Edit distance (Levenshtein) instead of LCS
- **Threshold:** `edit_distance < 2s + 1` (vs `LCS >= n-s`)
- **Graph files:** `graph_ids_s{s}_n{n}_q{q}.lmdb` (vs `graph_d_s{s}_n{n}_q{q}.lmdb`)
- **Library:** Uses `Levenshtein` for on-the-fly distance computation
