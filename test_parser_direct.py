#!/usr/bin/env python3
"""Direct test of parse_funsearch_continuation"""

import sys
import os

# Add the openevolve package to path
sys.path.insert(0, '/workspace/DistributedFunSearch/src/baselines/implementation_baselines/openevolve_comparison/openevolve')

print("Importing parse_funsearch_continuation...")
from openevolve.utils.code_utils import parse_funsearch_continuation

# Test data from actual log
llm_response = """    degree = G.degree(node)
    return degree / (math.comb(n, s) if s < n else 1)"""

parent_code = """import networkx as nx
from typing import Set
import math

def priority(node, G, n, s):
    return 0.0
"""

print("=" * 80)
print("Testing parse_funsearch_continuation with body-only LLM response")
print("=" * 80)
print(f"\nLLM response:\n{repr(llm_response)}\n")

result = parse_funsearch_continuation(llm_response, parent_code, "priority")

if result:
    print("="*80)
    print("SUCCESS! Parser returned:")
    print("="*80)
    print(result)
    print("\n" + "="*80)

    # Try to compile it
    try:
        compile(result, '<string>', 'exec')
        print("✓ Code compiles successfully!")
    except SyntaxError as e:
        print(f"✗ Syntax Error: {e}")
        print(f"   Line {e.lineno}: {repr(e.text)}")

    # Try to exec it
    try:
        namespace = {}
        exec(result, namespace)
        if 'priority' in namespace:
            print("✓ Code executes and priority function exists!")
        else:
            print("✗ priority function not found in namespace")
    except Exception as e:
        print(f"✗ Execution error: {e}")
else:
    print("✗ Parser returned None!")
    sys.exit(1)
