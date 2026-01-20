#!/usr/bin/env python3
"""Test script for prompt_builder.py

Usage:
    python -m disfun.tests.test_prompt_builder              # Test all 
    python -m disfun.tests.test_prompt_builder funsearch    # FunSearch only
    python -m disfun.tests.test_prompt_builder eoh          # EoH only
    python -m disfun.tests.test_prompt_builder reevo        # ReEvo only
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from disfun.utils import prompt_builder
from disfun.utils.prompt_builder import PromptStrategy, PromptSpec


# =============================================================================
# Test Data
# =============================================================================

@dataclass
class MockFunction:
    """Mock Function object for testing."""
    name: str = "priority"
    args: str = "node, G, n, s"
    return_type: str = "float"
    body: str = "    return 1.0"
    docstring: str = ""
    thought: str = ""


# Graph signature (node, G, n, s)
WORSE_FUNC = MockFunction(
    body='    return len(node)',
    thought="I tried using string length as a simple heuristic.",
)

BETTER_FUNC = MockFunction(
    body='    ones = node.count("1")\n    degree = G.degree(node)\n    return ones * 10 - degree',
    thought="Counting ones and penalizing high degree nodes should help.",
)

# No-graph signature (node, n, s, q)
WORSE_FUNC_NO_GRAPH = MockFunction(
    args="node, n, s, q",
    body='    return len(node)',
    thought="I tried using string length as a simple heuristic.",
)

BETTER_FUNC_NO_GRAPH = MockFunction(
    args="node, n, s, q",
    body='    ones = node.count("1")\n    return ones * 10',
    thought="Counting ones should help find nodes with fewer common subsequences.",
)

WORSE_SCORES = {"(6, 1)": 8, "(7, 1)": 12, "(8, 1)": 20}
BETTER_SCORES = {"(6, 1)": 10, "(7, 1)": 16, "(8, 1)": 28}
BEST_KNOWN = {(6, 1): 10, (7, 1): 16, (8, 1): 30}

PROGRAMS = [(WORSE_FUNC, WORSE_SCORES), (BETTER_FUNC, BETTER_SCORES)]
PROGRAMS_NO_GRAPH = [(WORSE_FUNC_NO_GRAPH, WORSE_SCORES), (BETTER_FUNC_NO_GRAPH, BETTER_SCORES)]


# =============================================================================
# Helpers
# =============================================================================

def header(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def get_spec_dir() -> Path:
    return Path(__file__).parent.parent / "specifications" / "Deletions"


def load_spec(strategy: PromptStrategy, **kwargs) -> PromptSpec:
    """Load spec with sensible defaults."""
    defaults = {
        "spec_dir": str(get_spec_dir()),
        "imports_file": "imports/networkx.txt",
        "initial_functions_dir": "initial_functions/graph_networkx",
        "fewshot_num_examples": 2,
        "fewshot_include_thought": True,
        "funsearch_template": "funsearch/templates/completion.txt",
        "funsearch_problem_desc": "funsearch/problem_descriptions/completion.txt",
        "show_scores": False,
        "score_display_mode": "relative",
        "best_known_solutions": BEST_KNOWN,
    }
    defaults.update(kwargs)
    return prompt_builder.load_specification(strategy=strategy, **defaults)


def print_prompt(prompt: str, max_chars: int | None = None):
    """Print prompt with optional truncation."""
    if max_chars and len(prompt) > max_chars:
        print(f"\n[Prompt ({len(prompt)} chars, showing first {max_chars})]\n{prompt[:max_chars]}\n...\n[End]\n")
    else:
        print(f"\n[Prompt ({len(prompt)} chars)]\n{prompt}\n[End]\n")


# =============================================================================
# FunSearch Tests
# =============================================================================

def test_funsearch():
    """Test all FunSearch variants in logical order."""
    header("FUNSEARCH TESTS")

    # 1. Basic completion (no tags)
    print("\n--- 1. Basic completion ---")
    spec = load_spec(PromptStrategy.FUNSEARCH)
    prompt = prompt_builder.build_prompt(spec, "funsearch", PROGRAMS)
    print(f"Template: completion.txt")
    print(f"Function: {spec.function_args} -> {spec.return_type}")
    assert "{function_header}" not in prompt
    assert "<code>" not in prompt, "Completion should not have <code> tags"
    print_prompt(prompt)

    # 2. Completion with evaluation script
    print("\n--- 2. Completion with evaluation script ---")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template="funsearch/templates/completion_with_eval.txt",
        funsearch_evaluation_script="evaluation/prompt_context/graph.txt",
    )
    prompt = prompt_builder.build_prompt(spec, "funsearch", PROGRAMS)
    print(f"Template: completion_with_eval.txt")
    assert "def solve(" in prompt, "Should include solve function"
    print_prompt(prompt)

    # 3. Completion with scores in docstring
    print("\n--- 3. Completion with scores ---")
    spec = load_spec(PromptStrategy.FUNSEARCH, show_scores=True)
    prompt = prompt_builder.build_prompt(spec, "funsearch", PROGRAMS)
    print(f"show_scores=True")
    assert "Gap to best known" in prompt or "%" in prompt, "Should have score info"
    print_prompt(prompt)

    # 4. Instruction templates
    templates = [
        ("instruction_basic", False),
        ("instruction_thought", True),
        ("instruction_reflection", True),
    ]

    for name, has_thought in templates:
        print(f"\n--- 4. Instruction: {name} ---")
        spec = load_spec(
            PromptStrategy.FUNSEARCH,
            funsearch_template=f"funsearch/templates/{name}.txt",
            funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
            funsearch_system_message="funsearch/system_message.txt",
        )
        prompt = prompt_builder.build_prompt(spec, "funsearch", PROGRAMS)
        fewshot = prompt.split("Improve on")[0]

        print(f"Has <thought>: {has_thought}")
        print(f"System message: {spec.system_message[:50]}..." if spec.system_message else "None")

        assert "<code>" in prompt and fewshot.count("<code>") >= 2
        if has_thought:
            assert "<thought>" in prompt and fewshot.count("<thought>") >= 2
        print_prompt(prompt)

    # 5. fewshot_include_thought=False (thought tags only in format instruction, not in examples)
    print("\n--- 5. fewshot_include_thought=False ---")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template="funsearch/templates/instruction_thought.txt",
        funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
        fewshot_include_thought=False,
    )
    prompt = prompt_builder.build_prompt(spec, "funsearch", PROGRAMS)
    fewshot = prompt.split("Improve on")[0]
    format_section = prompt.split("Improve on")[1]

    print(f"Fewshot <thought> count: {fewshot.count('<thought>')} (should be 0)")
    print(f"Format has <thought>: {'<thought>' in format_section}")

    assert fewshot.count("<thought>") == 0, "Fewshots should not have <thought>"
    assert fewshot.count("<code>") >= 2, "Fewshots should have <code>"
    assert "<thought>" in format_section, "Format should still request <thought>"
    print_prompt(prompt)

    print("\nFunSearch tests passed!")


# =============================================================================
# EoH Tests
# =============================================================================

def test_eoh():
    """Test all EoH templates."""
    header("EOH TESTS")

    spec = load_spec(PromptStrategy.EOH)
    print(f"Templates: {list(spec.templates.keys())}")
    print(f"System message: {'Yes' if spec.system_message else 'No'}")

    order = ['i1', 'e1', 'e2', 'm1', 'm2', 'm3']
    for name in [t for t in order if t in spec.templates]:
        reqs = spec.template_requirements[name]
        programs = PROGRAMS if reqs.num_programs == 2 else [PROGRAMS[1]]
        prompt = prompt_builder.build_prompt(spec, name, programs)

        print(f"\n--- {name} ({reqs.num_programs} program(s)) ---")
        print_prompt(prompt)

    print("\nEoH tests passed!")


# =============================================================================
# ReEvo Tests
# =============================================================================

def test_reevo():
    """Test all ReEvo templates."""
    header("REEVO TESTS")

    spec = load_spec(PromptStrategy.REEVO)
    print(f"Templates: {list(spec.templates.keys())}")
    print(f"Generator system: {'Yes' if spec.system_message else 'No'}")
    print(f"Reflector system: {'Yes' if spec.reflector_system_message else 'No'}")
    print(f"Initial reflection: {'Yes' if spec.initial_reflection else 'No'}")

    state = {
        "reflection": "Better functions use bit counting.",
        "prior_reflection": "Initial approaches using only degree were not effective.",
        "new_reflections": "Bit patterns matter. Penalizing neighbors helps.",
        "initial_reflection": spec.initial_reflection,
    }

    order = ['seed', 'reflect_st', 'crossover', 'reflect_lt', 'mutation']
    for name in [t for t in order if t in spec.templates]:
        reqs = spec.template_requirements[name]
        programs = PROGRAMS if reqs.num_programs == 2 else [PROGRAMS[1]]
        prompt = prompt_builder.build_prompt(spec, name, programs, state=state)

        print(f"\n--- {name} ({reqs.num_programs} program(s), reflection={reqs.needs_reflection}) ---")
        print_prompt(prompt)

    print("\nReEvo tests passed!")


# =============================================================================
# Main
# =============================================================================

def test_all():
    """Run all tests in logical order."""
    test_funsearch()
    test_eoh()
    test_reevo()
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)


def main():
    args = sys.argv[1:]
    commands = {
        "funsearch": test_funsearch,
        "eoh": test_eoh,
        "reevo": test_reevo,
    }

    if not args:
        test_all()
    elif args[0] in commands:
        commands[args[0]]()
    else:
        print(f"Unknown: {args[0]}\n{__doc__}")
        sys.exit(1)


if __name__ == "__main__":
    main()
