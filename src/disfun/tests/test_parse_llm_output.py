#!/usr/bin/env python3
"""
Test script for parse_llm_output function in evaluator.py

Usage:
    cd src/disfun
    python -m tests.test_parse_llm_output

Test groups:
    1. Raw code (no wrapper, fallback tier)
    2. <code> tags (tier 1 extraction)
    3. Markdown fences (tier 2 extraction)
    4. Template integration
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import parse_llm_output


def print_test(name: str, raw_input: str):
    """Run a single test and print results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    print(f"\nINPUT:")
    for i, line in enumerate(raw_input.split('\n'), 1):
        print(f"{i:3}: {line}")

    body, description = parse_llm_output(raw_input)

    print(f"\nPARSED OUTPUT:")
    print(f"description: {description}")
    print(f"\nbody ({len(body)} chars):")
    if body:
        for i, line in enumerate(body.rstrip('\n').split('\n'), 1):
            print(f"{i:3}: {line}")
    else:
        print("  (empty)")


def print_section(title: str):
    """Print a section header."""
    print(f"\n\n{'#'*60}")
    print(f"# {title}")
    print(f"{'#'*60}")


# ============================================================
# GROUP 1: Raw code (no wrapper, fallback tier)
# ============================================================

def test_raw_simple_function():
    """Simple function with basic logic, no wrapper."""
    raw = """def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    return degree + len(node)"""
    print_test("Raw: Simple Function", raw)


def test_raw_nested_indentation():
    """Function with multiple indentation levels."""
    raw = """def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    total = 0
    for neighbor in G_gt.vertex(v).out_neighbors():
        if int(neighbor) > v:
            total += 1
        else:
            total -= 1
    return total"""
    print_test("Raw: Nested Indentation", raw)


def test_raw_helper_inside():
    """Function with helper function defined inside."""
    raw = """def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    def hamming_weight(s):
        return sum(1 for c in s if c == '1')

    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    return degree * hamming_weight(node)"""
    print_test("Raw: Helper Inside Function", raw)


def test_raw_imports_with_priority():
    """Imports at module level with priority function."""
    raw = """import numpy as np
from collections import defaultdict

def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return np.sqrt(v)"""
    print_test("Raw: Imports with Priority", raw)


def test_raw_body_indented():
    """Body code already indented (completion style)."""
    raw = """    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    return degree * 2"""
    print_test("Raw: Indented Body Code", raw)


def test_raw_body_with_helper():
    """Indented body followed by module level helper."""
    raw = """    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    return degree * helper(v)

def helper(x):
    return x * 2"""
    print_test("Raw: Body with Helper", raw)


def test_raw_body_with_imports():
    """Imports at module level with indented body code."""
    raw = """import numpy as np

    v = node_to_vertex[node]
    return np.sqrt(v)"""
    print_test("Raw: Imports with Body", raw)


def test_raw_body_with_nested_helper():
    """Body code with indented helper (already nested)."""
    raw = """    def helper(x):
        return x * 2

    v = node_to_vertex[node]
    return helper(v.out_degree())"""
    print_test("Raw: Body with Nested Helper", raw)


# ============================================================
# GROUP 2: <code> tags (tier 1 extraction)
# ============================================================

def test_code_tags_simple():
    """Simple function wrapped in code tags."""
    raw = """<code>
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return G_gt.vertex(v).out_degree()
</code>"""
    print_test("Code Tags: Simple Function", raw)


def test_code_tags_multiple_functions():
    """Multiple functions, should extract priority and helper."""
    raw = """<code>
def helper(x):
    return x * 2

def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    return helper(degree)

def unused():
    pass
</code>"""
    print_test("Code Tags: Multiple Functions", raw)


def test_code_tags_body_before_func():
    """Body code before function definitions."""
    raw = """<code>
    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    return degree * 2

def helper(x):
    return x * 2
</code>"""
    print_test("Code Tags: Body Before Functions", raw)


def test_code_tags_with_description():
    """Code tags with description tags."""
    raw = """<description>
Using inverse degree combined with hamming weight.
</description>

<code>
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    weight = sum(1 for c in node if c == '1')
    return -degree + weight
</code>"""
    print_test("Code Tags: With Description", raw)


def test_code_tags_with_nested_fence():
    """Markdown fence nested inside code tags."""
    raw = """<code>
```python
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return G_gt.vertex(v).out_degree()
```
</code>"""
    print_test("Code Tags: Nested Fence Inside", raw)


# ============================================================
# GROUP 3: Markdown fences (tier 2 extraction)
# ============================================================

def test_fence_simple():
    """Simple function in markdown fence."""
    raw = """```python
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return G_gt.vertex(v).out_degree()
```"""
    print_test("Fence: Simple Function", raw)


def test_fence_multiple_blocks():
    """Multiple code blocks, should take the LAST one."""
    raw = """Here's a bad approach:
```python
return 0  # wrong
```

Here's the correct solution:
```python
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return G_gt.vertex(v).out_degree()
```"""
    print_test("Fence: Multiple Blocks (takes last)", raw)


def test_fence_with_think_block():
    """Content before fence (like Qwen3 think blocks)."""
    raw = """<think>
Let me analyze this problem. The key insight is that we should
prioritize nodes with higher degree because they have more connections.
</think>

```python
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return G_gt.vertex(v).out_degree()
```"""
    print_test("Fence: Think Block Before", raw)


def test_fence_plain_no_language():
    """Plain fence without language specifier."""
    raw = """```
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return G_gt.vertex(v).out_degree()
```"""
    print_test("Fence: Plain (no language)", raw)


# ============================================================
# GROUP 4: Template integration
# ============================================================

def test_template_integration():
    """Test that parsed body integrates correctly into a template."""
    from evaluator import _sample_to_program
    from disfun.utils.code_manipulation import text_to_program

    template_code = """import graph_tool
from graph_tool import Graph

def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    return 0

def evaluate(n, s):
    return build_graph(n, s)
"""
    template = text_to_program(template_code)

    llm_output = """<code>
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    degree = G_gt.vertex(v).out_degree()
    return -degree + len(node)
</code>"""

    evolved_function, program_str, description = _sample_to_program(
        llm_output, None, template, 'priority'
    )

    print(f"\n{'='*60}")
    print("TEST: Template Integration")
    print(f"{'='*60}")
    print(f"\nTEMPLATE (before integration):")
    for i, line in enumerate(str(template).split('\n'), 1):
        print(f"{i:3}: {line}")
    print(f"\nLLM OUTPUT:")
    for i, line in enumerate(llm_output.split('\n'), 1):
        print(f"{i:3}: {line}")
    print(f"\nINTEGRATED PROGRAM (after integration):")
    for i, line in enumerate(program_str.split('\n'), 1):
        print(f"{i:3}: {line}")


def main():
    print("\n# parse_llm_output Test Suite\n")

    # Group 1: Raw code
    print_section("GROUP 1: Raw Code (no wrapper)")
    test_raw_simple_function()
    test_raw_nested_indentation()
    test_raw_helper_inside()
    test_raw_imports_with_priority()
    test_raw_body_indented()
    test_raw_body_with_helper()
    test_raw_body_with_imports()
    test_raw_body_with_nested_helper()

    # Group 2: <code> tags
    print_section("GROUP 2: <code> Tags")
    test_code_tags_simple()
    test_code_tags_multiple_functions()
    test_code_tags_body_before_func()
    test_code_tags_with_description()
    test_code_tags_with_nested_fence()

    # Group 3: Markdown fences
    print_section("GROUP 3: Markdown Fences")
    test_fence_simple()
    test_fence_multiple_blocks()
    test_fence_with_think_block()
    test_fence_plain_no_language()

    # Group 4: Template integration
    print_section("GROUP 4: Template Integration")
    test_template_integration()

    print(f"\n{'='*60}")
    print("Completed 18 tests")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
