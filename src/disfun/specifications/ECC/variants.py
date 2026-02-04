"""Problem variants for Error Correcting Codes.

Each variant defines placeholders for problem descriptions:
- string_type: "binary" or "q-ary"
- edge_condition: How nodes are connected in the graph
- error_type: Type of errors the code corrects
"""

DELETIONS = {
    "string_type": "binary",
    "edge_condition": "Nodes are connected if they share a subsequence of length at least n-s, where s is the number of deletions the code should correct",
    "error_type": "deletion",
}

IDS = {
    "string_type": "q-ary",
    "edge_condition": "Nodes are connected if their edit distance < 2s+1, where s is the number of errors the code should correct",
    "error_type": "insertion/deletion/substitution",
}

# Alias for convenience
VARIANTS = {
    "deletions": DELETIONS,
    "ids": IDS,
}
