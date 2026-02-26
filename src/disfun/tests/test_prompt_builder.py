#!/usr/bin/env python3
"""Test script for prompt_builder.py

Usage:
    python -m disfun.tests.test_prompt_builder                  # Test all
    python -m disfun.tests.test_prompt_builder funsearch        # FunSearch only
    python -m disfun.tests.test_prompt_builder eoh              # EoH only
    python -m disfun.tests.test_prompt_builder reevo            # ReEvo only
    python -m disfun.tests.test_prompt_builder variants         # Variant tests only
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
    description: str = ""


# Graph signature (node, G, n, s)
WORSE_FUNC = MockFunction(
    body='    return len(node)',
    description="I tried using string length as a simple heuristic.",
)

BETTER_FUNC = MockFunction(
    body='    ones = node.count("1")\n    degree = G.degree(node)\n    return ones * 10 - degree',
    description="Counting ones and penalizing high degree nodes should help.",
)

# No-graph signature (node, n, s, q)
WORSE_FUNC_NO_GRAPH = MockFunction(
    args="node, n, s, q",
    body='    return len(node)',
    description="I tried using string length as a simple heuristic.",
)

BETTER_FUNC_NO_GRAPH = MockFunction(
    args="node, n, s, q",
    body='    ones = node.count("1")\n    return ones * 10',
    description="Counting ones should help find nodes with fewer common subsequences.",
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
    """Get ECC spec directory path."""
    return Path(__file__).parent.parent / "specifications" / "ECC"


def load_spec(strategy: PromptStrategy, variant: str = "deletions", **kwargs) -> PromptSpec:
    """Load spec with sensible defaults.

    Args:
        strategy: Prompt strategy to use
        variant: ECC variant name ("deletions" or "ids")
        **kwargs: Override any default parameters
    """
    defaults = {
        "spec_dir": str(get_spec_dir()),
        "variant": variant,
        "imports_file": "imports/networkx.txt",
        "initial_functions_dir": "initial_functions/graph",
        "fewshot_num_examples": 2,
        "fewshot_include_description": True,
        "funsearch_template": "funsearch/templates/completion.txt",
        "funsearch_problem_desc": "funsearch/problem_descriptions/completion.txt",
        "show_scores": False,
        "score_display_mode": "relative",
        "best_known_solutions": BEST_KNOWN,
    }
    defaults.update(kwargs)
    return prompt_builder.load_specification(strategy=strategy, **defaults)


def print_prompt(prompt: str):
    """Print full prompt."""
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

    # 4. Single-turn instruction templates with matching system messages
    single_turn_configs = [
        # (template_path, system_message, expected_code_tags, expected_desc_tags)
        ("single_turn/basic.txt", "single_turn/basic.txt", True, False),
        ("single_turn/thought.txt", "single_turn/thought.txt", True, True),
        ("single_turn/reflection", "single_turn/reflection.txt", True, True),  # folder
    ]

    for template_path, sys_msg, expect_code, expect_desc in single_turn_configs:
        name = template_path.split("/")[-1].replace(".txt", "")
        print(f"\n--- 4. Single-turn: {name} ---")
        spec = load_spec(
            PromptStrategy.FUNSEARCH,
            funsearch_template=f"funsearch/templates/{template_path}",
            funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
            funsearch_system_message=f"funsearch/system_messages/{sys_msg}",
        )
        print(f"is_multi_turn: {spec.is_multi_turn}")
        print(f"System message: {spec.system_message}")
        assert not spec.is_multi_turn, "Single-turn should not be multi-turn"

        # Test with build_funsearch_prompts
        refl, gen = prompt_builder.build_funsearch_prompts(spec, PROGRAMS)
        assert refl is None, "Single-turn should have no reflection prompt"
        assert "Improve on" in gen, "Should have improve instruction"

        # Verify tag detection from system message
        has_code = "<code>" in gen
        has_desc = "<description>" in gen
        assert has_code == expect_code, f"Expected code_tags={expect_code}, got {has_code}"
        assert has_desc == expect_desc, f"Expected desc_tags={expect_desc}, got {has_desc}"
        print(f"- Tags correct: <code>={has_code}, <description>={has_desc}")
        print_prompt(gen)

    # 5. Single-turn reflection with 1 vs 2 programs
    print("\n--- 5. Single-turn reflection: 1 vs 2 programs ---")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template="funsearch/templates/single_turn/reflection",
        funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
    )
    _, gen_2 = prompt_builder.build_funsearch_prompts(spec, PROGRAMS)
    _, gen_1 = prompt_builder.build_funsearch_prompts(spec, [PROGRAMS[0]])
    assert "compare" in gen_2.lower(), "2 programs should use compare"
    assert "analyze" in gen_1.lower(), "1 program should use analyze"
    print("- 2 programs: uses default template (compare)")
    print("- 1 program: uses single template (analyze)")

    # 6. Multi-turn thought
    print("\n--- 6. Multi-turn thought ---")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template="funsearch/templates/multi_turn/thought",
        funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
    )
    print(f"is_multi_turn: {spec.is_multi_turn}")
    print(f"Templates: {list(spec.templates.keys())}")
    assert spec.is_multi_turn, "Multi-turn should be detected"

    refl, gen = prompt_builder.build_funsearch_prompts(spec, PROGRAMS)
    assert refl is not None, "Multi-turn should have reflection prompt"
    assert "{reflection}" in gen, "Stage2 should keep {reflection} placeholder"
    assert "describe" in refl.lower(), "Stage1 should ask to describe"
    print("- Stage1 asks to describe heuristic")
    print("- Stage2 has {reflection} placeholder")
    print_prompt(refl)

    # 7. Multi-turn reflection
    print("\n--- 7. Multi-turn reflection ---")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template="funsearch/templates/multi_turn/reflection",
        funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
    )
    print(f"is_multi_turn: {spec.is_multi_turn}")
    print(f"Templates: {list(spec.templates.keys())}")

    # Test with 2 programs
    refl, gen = prompt_builder.build_funsearch_prompts(spec, PROGRAMS)
    assert "compare" in refl.lower(), "Stage1 should ask to compare"
    print("- 2 programs: stage1 asks to compare")

    # Test with 1 program (should use _single variants)
    refl, gen = prompt_builder.build_funsearch_prompts(spec, [PROGRAMS[0]])
    assert "analyze" in refl.lower(), "Stage1 single should ask to analyze"
    print("- 1 program: stage1_single asks to analyze")

    # 8. Multi-turn system messages
    print("\n--- 8. Multi-turn system messages ---")
    # Note: Multi-turn has separate system messages for stage1 and stage2
    # Stage1: plain text output (no tags)
    # Stage2: code output with <code> tags
    sys_msg_dir = get_spec_dir() / "funsearch" / "system_messages" / "multi_turn"

    for variant in ["thought", "reflection"]:
        stage1_path = sys_msg_dir / variant / "stage1.txt"
        stage2_path = sys_msg_dir / variant / "stage2.txt"
        assert stage1_path.exists(), f"Missing {stage1_path}"
        assert stage2_path.exists(), f"Missing {stage2_path}"

        stage1_msg = stage1_path.read_text()
        stage2_msg = stage2_path.read_text()

        # Stage1 should NOT have code tags (it's for reasoning)
        assert "<code>" not in stage1_msg, f"Stage1 {variant} should not have <code>"
        # Stage2 should have code tags (for code output)
        assert "<code>" in stage2_msg, f"Stage2 {variant} should have <code>"
        print(f"- multi_turn/{variant}: stage1 no tags, stage2 has <code>")


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


# =============================================================================
# Variant Tests (ECC folder with placeholders)
# =============================================================================

def test_variants():
    """Test ECC folder with deletions and ids variants."""
    header("VARIANT TESTS")

    # Test deletions variant
    print("\n--- 1. Deletions variant ---")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        variant="deletions",
        funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
    )
    _, prompt = prompt_builder.build_funsearch_prompts(spec, PROGRAMS)

    # Verify placeholders are replaced with deletions values
    assert "binary" in prompt, "Should have 'binary' for string_type"
    assert "deletion" in prompt, "Should have 'deletion' for error_type"
    assert "subsequence" in prompt, "Should have 'subsequence' in edge_condition"
    assert "{string_type}" not in prompt, "Placeholder should be replaced"
    assert "{error_type}" not in prompt, "Placeholder should be replaced"
    assert "{edge_condition}" not in prompt, "Placeholder should be replaced"
    print("- Deletions placeholders correctly substituted")
    print(f"- Problem description preview: {spec.problem_description[:200]}...")

    # Test ids variant
    print("\n--- 2. IDS variant ---")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        variant="ids",
        funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
    )
    _, prompt = prompt_builder.build_funsearch_prompts(spec, PROGRAMS)

    # Verify placeholders are replaced with IDS values
    assert "q-ary" in prompt, "Should have 'q-ary' for string_type"
    assert "edit distance" in prompt, "Should have 'edit distance' in edge_condition"
    assert "{string_type}" not in prompt, "Placeholder should be replaced"
    assert "{error_type}" not in prompt, "Placeholder should be replaced"
    assert "{edge_condition}" not in prompt, "Placeholder should be replaced"
    print("- IDS placeholders correctly substituted")
    print(f"- Problem description preview: {spec.problem_description[:200]}...")

    # Test EoH with deletions variant
    print("\n--- 3. EoH with deletions variant ---")
    spec = load_spec(PromptStrategy.EOH, variant="deletions")
    prompt = prompt_builder.build_prompt(spec, "i1", PROGRAMS)
    assert "binary" in prompt or "deletion" in prompt, "EoH should have variant substituted"
    assert "deletion" in spec.system_message, "System message should have variant"
    print("- EoH system message has variant placeholders substituted")
    print(f"- System message: {spec.system_message}")

    # Test ReEvo with ids variant
    print("\n--- 4. ReEvo with ids variant ---")
    spec = load_spec(PromptStrategy.REEVO, variant="ids")
    assert "insertion/deletion/substitution" in spec.system_message, "Generator system should have IDS error type"
    assert "q-ary" in spec.initial_reflection, "Initial reflection should have q-ary"
    print("- ReEvo system message has IDS error type")
    print(f"- Generator system: {spec.system_message}")
    print(f"- Initial reflection: {spec.initial_reflection}")

    # Test error handling for invalid variant
    print("\n--- 5. Error handling ---")
    try:
        load_spec(PromptStrategy.FUNSEARCH, variant="invalid_variant")
        assert False, "Should raise ValueError for invalid variant"
    except ValueError as e:
        print(f"- Correctly raises ValueError: {e}")


# =============================================================================
# Main
# =============================================================================

def test_all():
    """Run all tests in logical order."""
    test_funsearch()
    test_eoh()
    test_reevo()
    test_variants()


def main():
    args = sys.argv[1:]
    commands = {
        "funsearch": test_funsearch,
        "eoh": test_eoh,
        "reevo": test_reevo,
        "variants": test_variants,
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
