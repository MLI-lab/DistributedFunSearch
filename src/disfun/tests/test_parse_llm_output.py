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

from disfun.utils.code_manipulation import parse_llm_output


def print_test(name: str, raw_input: str):
    """Run a single test and print results."""
    print(f"\n--- {name} ---")

    print(f"\ninput.")
    for i, line in enumerate(raw_input.split('\n'), 1):
        print(f"{i:3}: {line}")

    body, description, thinking_trace, failure_reason = parse_llm_output(raw_input)

    print(f"\nparsed output.")
    print(f"description, {description}")
    if failure_reason:
        print(f"failure_reason, {failure_reason}")
    if thinking_trace:
        print(f"thinking_trace, {thinking_trace[:200]}{'...' if len(thinking_trace) > 200 else ''}")
    print(f"\nbody ({len(body)} chars).")
    if body:
        for i, line in enumerate(body.rstrip('\n').split('\n'), 1):
            print(f"{i:3}: {line}")
    else:
        print("  (empty)")


def print_section(title: str):
    """Print a section header."""
    print(f"\n\n--- {title} ---")


# Group 1, raw code (no wrapper, fallback tier).

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


# Group 2, <code> tags (tier 1 extraction).

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


# Group 3, markdown fences (tier 2 extraction).

def test_fence_simple():
    """Simple function in markdown fence."""
    raw = """```python
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    v = node_to_vertex[node]
    return G_gt.vertex(v).out_degree()
```"""
    print_test("Fence: Simple Function", raw)


def test_fence_multiple_blocks():
    """Multiple code blocks, should take the last one."""
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


# Group 4, template integration.

def test_template_integration():
    """Test that parsed body integrates correctly into a template."""
    from disfun.utils.code_manipulation import sample_to_program, text_to_program

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

    evolved_function, program_str, description, thinking_trace, failure_reason = sample_to_program(
        llm_output, None, template, 'priority'
    )

    print(f"\n--- Template Integration ---")
    print(f"\ntemplate (before integration).")
    for i, line in enumerate(str(template).split('\n'), 1):
        print(f"{i:3}: {line}")
    print(f"\nllm output.")
    for i, line in enumerate(llm_output.split('\n'), 1):
        print(f"{i:3}: {line}")
    print(f"\nintegrated program (after integration).")
    for i, line in enumerate(program_str.split('\n'), 1):
        print(f"{i:3}: {line}")


# Group 5, real-world LLM failure patterns (derived from debug_samples/eval_1268124).

def assert_body_compiles(body, test_name):
    """Verify that parsed body can be placed inside a function and compiled."""
    if not body:
        print(f"  fail, empty body.")
        return False
    wrapped = f"def priority(node, G, n, s):\n{body}"
    try:
        compile(wrapped, '<test>', 'exec')
        print(f"  pass, body compiles.")
        return True
    except SyntaxError as e:
        print(f"  fail, {e}")
        return False


def test_multiline_def_signature():
    """Multi-line def with continuation args (from sample 000001_island8).
    Bug: parser included continuation lines in body, causing IndentationError."""
    raw = """<code>
```python
def priority(node : str,
             G    : object,
             n    : int,
             s    : int) -> float:
    v = sum(1 for c in node if c == '1')
    degree = len(list(G.neighbors(node)))
    if degree == 0:
        return float('inf')
    return v / (degree + 1)
```
</code>"""
    print_test("Multi-line def signature", raw)
    body, _, _, _ = parse_llm_output(raw)
    ok = assert_body_compiles(body, "multiline_def")
    # Body must NOT contain continuation lines from the def signature
    assert 'G    : object' not in body, "fail, def continuation leaked into body."
    print(f"  check, no def continuation in body, {'pass' if 'G    : object' not in body else 'fail'}.")


def test_mixed_indentation_3space():
    """LLM uses 3-space indent (from sample 000151_island2).
    Bug: 3-space body mixed with 4-space injected imports -> IndentationError."""
    raw = """<code>
```python
def priority_new(node, G, n, s):
   v = sum(1 for c in node if c == '1')
   degree = len(list(G.neighbors(node)))
   if degree == 0:
      return float('inf')
   return v / (degree + 1)
```
</code>"""
    print_test("Mixed indentation (3-space)", raw)
    body, _, _, _ = parse_llm_output(raw)
    ok = assert_body_compiles(body, "3space_indent")
    # Verify body was normalized to 4-space
    first_line = next((l for l in body.splitlines() if l.strip()), '')
    indent = len(first_line) - len(first_line.lstrip())
    print(f"  check, normalized to 4-space indent, {'pass' if indent == 4 else f'fail (got {indent})'}.")


def test_helper_before_priority_different_indent():
    """Helper function + priority with non-standard indent.
    Verifies helper is captured and indentation is consistent."""
    raw = """<code>
```python
def hamming_weight(s):
   return sum(1 for c in s if c == '1')

def priority_new(node, G, n, s):
   hw = hamming_weight(node)
   degree = len(list(G.neighbors(node)))
   return hw - degree
```
</code>"""
    print_test("Helper before priority (3-space indent)", raw)
    body, _, _, _ = parse_llm_output(raw)
    ok = assert_body_compiles(body, "helper_different_indent")
    assert 'hamming_weight' in body, "fail, helper function missing from body."
    print(f"  check, helper included, {'pass' if 'hamming_weight' in body else 'fail'}.")


def test_code_tags_with_double_fence():
    """<code> wrapping ```python blocks (most common pattern: 60% of samples).
    Verifies nested fence stripping works correctly."""
    raw = """<code>
```python
def priority(node, G, n, s):
    degree = len(list(G.neighbors(node)))
    ones = sum(1 for c in node if c == '1')
    return -degree + ones * 0.5
```
</code>"""
    print_test("Code tags with double fence", raw)
    body, _, _, _ = parse_llm_output(raw)
    ok = assert_body_compiles(body, "double_fence")
    assert '```' not in body, "fail, fence markers leaked into body."
    print(f"  check, no fence markers in body, {'pass' if '```' not in body else 'fail'}.")


def test_except_pass_swallows_none_return():
    """Function with try/except: pass (from sample 000006).
    Parser handles this correctly already — documents the behavior."""
    raw = """<code>
```python
def priority(node, G, n, s):
    try:
        degree = len(list(G.neighbors(node)))
        ones = sum(1 for c in node if c == '1')
        return -degree + ones
    except:
        pass
```
</code>"""
    print_test("Try/except pass (returns None)", raw)
    body, _, _, _ = parse_llm_output(raw)
    ok = assert_body_compiles(body, "except_pass")


def test_priority_new_renamed():
    """LLM names function priority_new (63% of samples).
    Verify recursive calls are renamed to 'priority'."""
    raw = """<code>
```python
def priority_new(node, G, n, s):
    degree = len(list(G.neighbors(node)))
    if degree == 0:
        return priority_new(node, G, n, s - 1) if s > 0 else 0
    return -degree
```
</code>"""
    print_test("Priority_new renamed to priority", raw)
    body, _, _, _ = parse_llm_output(raw)
    ok = assert_body_compiles(body, "priority_new_rename")
    assert 'priority_new' not in body, "fail, priority_new not renamed."
    assert 'priority(' in body, "fail, recursive call not renamed to priority."
    print(f"  check, renamed to priority, {'pass' if 'priority_new' not in body and 'priority(' in body else 'fail'}.")


def test_combinations_import_inside_body():
    """LLM does 'from itertools import combinations' inside function body.
    Verifies that inline import is preserved in output."""
    raw = """<code>
```python
def priority_new(node, G, n, s):
    from itertools import combinations
    neighbors = list(G.neighbors(node))
    triangle_count = 0
    for u, v in combinations(neighbors, 2):
        if G.has_edge(u, v):
            triangle_count += 1
    return -len(neighbors) + triangle_count * 0.1
```
</code>"""
    print_test("Combinations import inside body", raw)
    body, _, _, _ = parse_llm_output(raw)
    ok = assert_body_compiles(body, "combinations_import")
    assert 'combinations' in body, "fail, combinations reference missing from body."
    print(f"  check, combinations in body, {'pass' if 'combinations' in body else 'fail'}.")


def main():
    print("\n# parse_llm_output Test Suite\n")

    # Group 1: Raw code
    print_section("Group 1, raw code (no wrapper)")
    test_raw_simple_function()
    test_raw_nested_indentation()
    test_raw_helper_inside()
    test_raw_imports_with_priority()
    test_raw_body_indented()
    test_raw_body_with_helper()
    test_raw_body_with_imports()
    test_raw_body_with_nested_helper()

    # Group 2, <code> tags.
    print_section("Group 2, <code> tags")
    test_code_tags_simple()
    test_code_tags_multiple_functions()
    test_code_tags_body_before_func()
    test_code_tags_with_description()
    test_code_tags_with_nested_fence()

    # Group 3, markdown fences.
    print_section("Group 3, markdown fences")
    test_fence_simple()
    test_fence_multiple_blocks()
    test_fence_with_think_block()
    test_fence_plain_no_language()

    # Group 4, template integration.
    print_section("Group 4, template integration")
    test_template_integration()

    # Group 5, real-world LLM failure patterns.
    print_section("Group 5, real-world LLM failure patterns")
    test_multiline_def_signature()
    test_mixed_indentation_3space()
    test_helper_before_priority_different_indent()
    test_code_tags_with_double_fence()
    test_except_pass_swallows_none_return()
    test_priority_new_renamed()
    test_combinations_import_inside_body()

    print(f"\nCompleted 25 tests.\n")


if __name__ == "__main__":
    main()
