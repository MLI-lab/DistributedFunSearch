#!/usr/bin/env python3
"""Test script for prompt_builder.py

Usage:
    python -m disfun.tests.test_prompt_builder                         # Test all strategies and templates
    python -m disfun.tests.test_prompt_builder summary                 # Print summary of all templates
    python -m disfun.tests.test_prompt_builder scores                  # Test score display in prompts
    python -m disfun.tests.test_prompt_builder funsearch               # Test FunSearch completion template
    python -m disfun.tests.test_prompt_builder funsearch_instruction   # Test FunSearch instruction templates
    python -m disfun.tests.test_prompt_builder fewshot_tags            # Test fewshot_include_thinking/thought config
    python -m disfun.tests.test_prompt_builder eoh                     # Test all EoH templates
    python -m disfun.tests.test_prompt_builder eoh e1                  # Test specific EoH template
    python -m disfun.tests.test_prompt_builder reevo                   # Test all ReEvo templates
    python -m disfun.tests.test_prompt_builder reevo crossover         # Test specific ReEvo template
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from disfun.utils import prompt_builder
from disfun.utils.prompt_builder import PromptStrategy, PromptSpec


@dataclass
class MockFunction:
    """Mock Function object for testing."""
    name: str = "priority"
    args: str = "node, G, n, s"
    return_type: str = "float"
    body: str = "    return 1.0"
    docstring: str = ""
    thought: str = ""
    thinking: str = ""


# Two sample functions with different scores
WORSE_FUNCTION = MockFunction(
    name="priority",
    args="node, G, n, s",
    return_type="float",
    body='    return len(node)',
    docstring="A simple baseline that returns string length.",
    thought="I tried using string length as a simple heuristic.",
    thinking="The simplest approach is to use string length as priority."
)

BETTER_FUNCTION = MockFunction(
    name="priority",
    args="node, G, n, s",
    return_type="float",
    body='    ones = node.count("1")\n    degree = G.degree(node)\n    return ones * 10 - degree',
    docstring="Improved version considering bit count and graph structure.",
    thought="Counting ones and penalizing high degree nodes should help find independent sets.",
    thinking="Looking at the problem, nodes with more 1s might have fewer common subsequences. Also, high degree nodes have many neighbors, so avoiding them could help build larger independent sets."
)

# Sample scores in (n, s) format, q is constant per run
WORSE_SCORES = {
    "(6, 1)": 8,
    "(7, 1)": 12,
    "(8, 1)": 20,
}

BETTER_SCORES = {
    "(6, 1)": 10,
    "(7, 1)": 16,
    "(8, 1)": 28,
}

# Best known solutions for relative score display
BEST_KNOWN_SOLUTIONS = {
    (6, 1): 10,
    (7, 1): 16,
    (8, 1): 30,
}


def get_spec_dir() -> Path:
    """Get the specifications directory path."""
    return Path(__file__).parent.parent / "specifications" / "Deletions"


def load_spec(
    strategy: PromptStrategy,
    show_scores: bool = False,
    funsearch_template: str = "funsearch/templates/completion.txt",
    funsearch_problem_desc: str = "funsearch/problem_descriptions/completion.txt",
    funsearch_system_message: str = None,
    fewshot_include_thinking: bool = True,
    fewshot_include_thought: bool = True,
) -> PromptSpec:
    """Load spec for given strategy."""
    spec_dir = get_spec_dir()

    return prompt_builder.load_specification(
        strategy=strategy,
        spec_dir=str(spec_dir),
        imports_file="imports/networkx.txt",
        initial_functions_dir="initial_functions/graph_networkx",
        fewshot_num_examples=2,
        fewshot_include_thinking=fewshot_include_thinking,
        fewshot_include_thought=fewshot_include_thought,
        funsearch_template=funsearch_template,
        funsearch_problem_desc=funsearch_problem_desc,
        funsearch_system_message=funsearch_system_message,
        show_scores=show_scores,
        score_display_mode="relative",
        best_known_solutions=BEST_KNOWN_SOLUTIONS,
    )


def test_funsearch():
    """Test FunSearch completion template."""
    print("\n" + "=" * 80)
    print("Testing FunSearch completion")
    print("=" * 80)

    spec = load_spec(PromptStrategy.FUNSEARCH)
    print(f"Loaded spec: {len(spec.templates)} template(s)")
    print(f"Function args: {spec.function_args}")
    print(f"Return type: {spec.return_type}")

    programs = [
        (WORSE_FUNCTION, WORSE_SCORES),
        (BETTER_FUNCTION, BETTER_SCORES),
    ]

    template_name, num_needed = prompt_builder.select_template(spec)
    print(f"Template: {template_name}, programs needed: {num_needed}")

    prompt = prompt_builder.build_prompt(spec, template_name, programs)

    print(f"\n[Prompt ({len(prompt)} chars)]")
    print(prompt)
    print("[End prompt]\n")

    # Verify completion format (should have function header, no tags)
    assert "{function_header}" not in prompt, "Placeholder not filled"
    assert "<code>" not in prompt, "Completion template should not have <code> tags"
    print("Completion format verified.")

    return prompt


def test_funsearch_instruction(template_name: str = None):
    """Test FunSearch instruction templates."""
    print("\n" + "=" * 80)
    print("Testing FunSearch instruction templates")
    print("=" * 80)

    # Define instruction templates to test
    instruction_templates = {
        "instruction_basic": {
            "template": "funsearch/templates/instruction_basic.txt",
            "has_thinking": False,
            "has_thought": False,
            "has_code": True,
        },
        "instruction_thought": {
            "template": "funsearch/templates/instruction_thought.txt",
            "has_thinking": False,
            "has_thought": True,
            "has_code": True,
        },
        "instruction_reasoning": {
            "template": "funsearch/templates/instruction_reasoning.txt",
            "has_thinking": True,
            "has_thought": True,
            "has_code": True,
        },
    }

    templates_to_test = [template_name] if template_name else list(instruction_templates.keys())

    programs = [
        (WORSE_FUNCTION, WORSE_SCORES),
        (BETTER_FUNCTION, BETTER_SCORES),
    ]

    for tname in templates_to_test:
        if tname not in instruction_templates:
            print(f"Template '{tname}' not found. Available: {list(instruction_templates.keys())}")
            continue

        config = instruction_templates[tname]
        print(f"\n[Testing {tname}]")

        spec = load_spec(
            PromptStrategy.FUNSEARCH,
            funsearch_template=config["template"],
            funsearch_problem_desc="funsearch/problem_descriptions/instruction.txt",
            funsearch_system_message="funsearch/system_message.txt",
        )

        prompt = prompt_builder.build_prompt(spec, "funsearch", programs)

        print(f"Prompt length: {len(prompt)} chars")
        print(f"System message: {spec.system_message[:50]}..." if spec.system_message else "No system message")

        # Verify tag presence in prompt template
        if config["has_code"]:
            assert "<code>" in prompt, f"{tname} should have <code> in format block"
            print("Has <code> tag in format: Yes")
        if config["has_thought"]:
            assert "<thought>" in prompt, f"{tname} should have <thought> in format block"
            print("Has <thought> tag in format: Yes")
        if config["has_thinking"]:
            assert "<thinking>" in prompt, f"{tname} should have <thinking> in format block"
            print("Has <thinking> tag in format: Yes")

        # Verify few shot examples have appropriate tags
        fewshot_section = prompt.split("Improve on")[0]  # Get section before instruction
        if config["has_code"]:
            assert fewshot_section.count("<code>") >= 2, "Few shot examples should have <code> tags"
            print("Few shot examples wrapped in <code>: Yes")
        if config["has_thought"]:
            assert fewshot_section.count("<thought>") >= 2, "Few shot examples should have <thought> tags"
            print("Few shot examples have <thought>: Yes")
        if config["has_thinking"]:
            assert fewshot_section.count("<thinking>") >= 2, "Few shot examples should have <thinking> tags"
            print("Few shot examples have <thinking>: Yes")

        # Verify no function_header placeholder (instruction mode)
        assert "{function_header}" not in prompt, "Instruction template should not use function_header"

        print(f"\n[Prompt]")
        print(prompt)
        print("[End prompt]\n")


def test_fewshot_tags():
    """Test fewshot_include_thinking and fewshot_include_thought config options."""
    print("\n" + "=" * 80)
    print("Testing fewshot tag config options")
    print("=" * 80)

    programs = [
        (WORSE_FUNCTION, WORSE_SCORES),
        (BETTER_FUNCTION, BETTER_SCORES),
    ]

    # Use instruction_reasoning template which has <thinking>, <thought>, <code>
    template = "funsearch/templates/instruction_reasoning.txt"
    problem_desc = "funsearch/problem_descriptions/instruction.txt"

    # Test 1: Default (both True), few shots should have thinking and thought
    print("\n[Test 1: fewshot_include_thinking=True, fewshot_include_thought=True (default)]")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template=template,
        funsearch_problem_desc=problem_desc,
        fewshot_include_thinking=True,
        fewshot_include_thought=True,
    )
    prompt = prompt_builder.build_prompt(spec, "funsearch", programs)
    fewshot_section = prompt.split("Improve on")[0]

    thinking_count = fewshot_section.count("<thinking>")
    thought_count = fewshot_section.count("<thought>")
    code_count = fewshot_section.count("<code>")

    print(f"Fewshot <thinking> count: {thinking_count}")
    print(f"Fewshot <thought> count: {thought_count}")
    print(f"Fewshot <code> count: {code_count}")

    assert thinking_count >= 2, "Default: few shots should have <thinking> tags"
    assert thought_count >= 2, "Default: few shots should have <thought> tags"
    assert code_count >= 2, "Default: few shots should have <code> tags"
    print("Passed: Fewshots include thinking and thought tags")

    # Test 2: Both False, few shots should not have thinking or thought
    print("\n[Test 2: fewshot_include_thinking=False, fewshot_include_thought=False]")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template=template,
        funsearch_problem_desc=problem_desc,
        fewshot_include_thinking=False,
        fewshot_include_thought=False,
    )
    prompt = prompt_builder.build_prompt(spec, "funsearch", programs)
    fewshot_section = prompt.split("Improve on")[0]

    thinking_count = fewshot_section.count("<thinking>")
    thought_count = fewshot_section.count("<thought>")
    code_count = fewshot_section.count("<code>")

    print(f"Fewshot <thinking> count: {thinking_count}")
    print(f"Fewshot <thought> count: {thought_count}")
    print(f"Fewshot <code> count: {code_count}")

    assert thinking_count == 0, "Fewshots should not have <thinking> tags when disabled"
    assert thought_count == 0, "Fewshots should not have <thought> tags when disabled"
    assert code_count >= 2, "Fewshots should still have <code> tags"
    print("Passed: Fewshots exclude thinking and thought tags when disabled")

    # Verify the format instruction still has the tags
    format_section = prompt.split("Improve on")[1] if "Improve on" in prompt else prompt
    assert "<thinking>" in format_section, "Format instruction should still request <thinking>"
    assert "<thought>" in format_section, "Format instruction should still request <thought>"
    print("Passed: Format instruction still requests thinking/thought output")

    # Test 3: Only thinking disabled
    print("\n[Test 3: fewshot_include_thinking=False, fewshot_include_thought=True]")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template=template,
        funsearch_problem_desc=problem_desc,
        fewshot_include_thinking=False,
        fewshot_include_thought=True,
    )
    prompt = prompt_builder.build_prompt(spec, "funsearch", programs)
    fewshot_section = prompt.split("Improve on")[0]

    thinking_count = fewshot_section.count("<thinking>")
    thought_count = fewshot_section.count("<thought>")

    print(f"Fewshot <thinking> count: {thinking_count}")
    print(f"Fewshot <thought> count: {thought_count}")

    assert thinking_count == 0, "Fewshots should not have <thinking> when disabled"
    assert thought_count >= 2, "Fewshots should have <thought> when enabled"
    print("Passed: Mixed config works correctly")

    # Test 4: Only thought disabled
    print("\n[Test 4: fewshot_include_thinking=True, fewshot_include_thought=False]")
    spec = load_spec(
        PromptStrategy.FUNSEARCH,
        funsearch_template=template,
        funsearch_problem_desc=problem_desc,
        fewshot_include_thinking=True,
        fewshot_include_thought=False,
    )
    prompt = prompt_builder.build_prompt(spec, "funsearch", programs)
    fewshot_section = prompt.split("Improve on")[0]

    thinking_count = fewshot_section.count("<thinking>")
    thought_count = fewshot_section.count("<thought>")

    print(f"Fewshot <thinking> count: {thinking_count}")
    print(f"Fewshot <thought> count: {thought_count}")

    assert thinking_count >= 2, "Fewshots should have <thinking> when enabled"
    assert thought_count == 0, "Fewshots should not have <thought> when disabled"
    print("Passed: Mixed config works correctly")

    print("\n" + "=" * 80)
    print("All fewshot tag config tests passed")
    print("=" * 80)


def test_eoh(template_name: str = None):
    """Test EoH prompt building."""
    print("\n" + "=" * 80)
    print(f"Testing EoH" + (f" template {template_name}" if template_name else ", all templates"))
    print("=" * 80)

    spec = load_spec(PromptStrategy.EOH)
    print(f"Loaded spec: {len(spec.templates)} template(s): {list(spec.templates.keys())}")

    eoh_order = ['i1', 'e1', 'e2', 'm1', 'm2', 'm3']
    templates_to_test = [template_name] if template_name else [t for t in eoh_order if t in spec.templates]

    for tname in templates_to_test:
        if tname not in spec.templates:
            print(f"Template '{tname}' not found. Available: {list(spec.templates.keys())}")
            continue

        reqs = spec.template_requirements[tname]
        print(f"\n[Template: {tname} (needs {reqs.num_programs} program(s))]")

        if reqs.num_programs == 2:
            programs = [
                (WORSE_FUNCTION, WORSE_SCORES),
                (BETTER_FUNCTION, BETTER_SCORES),
            ]
        else:
            programs = [(BETTER_FUNCTION, BETTER_SCORES)]

        prompt = prompt_builder.build_prompt(spec, tname, programs)

        print(f"\n[Prompt ({len(prompt)} chars)]")
        print(prompt)
        print("[End prompt]\n")

    print("\n[System messages]")
    print(f"Generator: {spec.system_message}" if spec.system_message else "Generator: None")

    return spec


def test_reevo(template_name: str = None):
    """Test ReEvo prompt building."""
    print("\n" + "=" * 80)
    print(f"Testing ReEvo" + (f" template {template_name}" if template_name else ", all templates"))
    print("=" * 80)

    spec = load_spec(PromptStrategy.REEVO)
    print(f"Loaded spec: {len(spec.templates)} template(s): {list(spec.templates.keys())}")
    print(f"Initial reflection: '{spec.initial_reflection[:50]}...'" if spec.initial_reflection else "No initial reflection")

    reevo_order = ['seed', 'reflect_st', 'crossover', 'reflect_lt', 'mutation']
    templates_to_test = [template_name] if template_name else [t for t in reevo_order if t in spec.templates]

    reevo_state = {
        "reflection": "Better functions use bit counting.",
        "prior_reflection": "Initial approaches using only degree were not effective.",
        "new_reflections": "Bit patterns matter more than graph structure. Penalizing neighbors helps.",
        "initial_reflection": spec.initial_reflection,
    }

    for tname in templates_to_test:
        if tname not in spec.templates:
            print(f"Template '{tname}' not found. Available: {list(spec.templates.keys())}")
            continue

        reqs = spec.template_requirements[tname]
        print(f"\n[Template: {tname} (needs {reqs.num_programs} program(s), reflection={reqs.needs_reflection})]")

        if reqs.num_programs == 2:
            programs = [
                (WORSE_FUNCTION, WORSE_SCORES),
                (BETTER_FUNCTION, BETTER_SCORES),
            ]
        else:
            programs = [(BETTER_FUNCTION, BETTER_SCORES)]

        prompt = prompt_builder.build_prompt(spec, tname, programs, state=reevo_state)

        print(f"\n[Prompt ({len(prompt)} chars)]")
        print(prompt)
        print("[End prompt]\n")

    print("\n[System messages]")
    print(f"Generator: {spec.system_message}" if spec.system_message else "Generator: None")
    print(f"Reflector: {spec.reflector_system_message}" if spec.reflector_system_message else "Reflector: None")

    return spec


def test_all():
    """Test all strategies."""
    test_funsearch()
    test_funsearch_instruction()
    test_fewshot_tags()
    test_eoh()
    test_reevo()


def summary():
    """Print summary of all templates without full prompts."""
    print("\n" + "=" * 80)
    print("Prompt builder summary")
    print("=" * 80)

    template_orders = {
        PromptStrategy.FUNSEARCH: ['funsearch'],
        PromptStrategy.EOH: ['i1', 'e1', 'e2', 'm1', 'm2', 'm3'],
        PromptStrategy.REEVO: ['seed', 'reflect_st', 'crossover', 'reflect_lt', 'mutation'],
    }

    for strategy in [PromptStrategy.FUNSEARCH, PromptStrategy.EOH, PromptStrategy.REEVO]:
        spec = load_spec(strategy)
        print(f"\n{strategy.value.upper()}:")
        print(f"  System message: {'Yes' if spec.system_message else 'No'}")
        if strategy == PromptStrategy.EOH:
            print(f"  Few shot format: Tags detected from template (<thinking>, <thought>, <code>)")
        if strategy == PromptStrategy.REEVO:
            print(f"  Reflector system message: {'Yes' if spec.reflector_system_message else 'No'}")
            print(f"  Initial reflection: {'Yes' if spec.initial_reflection else 'No'}")
        print(f"  Templates:")
        ordered_templates = [t for t in template_orders[strategy] if t in spec.template_requirements]
        for name in ordered_templates:
            reqs = spec.template_requirements[name]
            print(f"    {name}: {reqs.num_programs} program(s), reflection={reqs.needs_reflection}")

    # Show FunSearch instruction templates
    print("\nFunSearch instruction templates:")
    instruction_templates = [
        ("instruction_basic", "No reasoning, just <code>"),
        ("instruction_thought", "<thought> + <code>"),
        ("instruction_reasoning", "<thinking> + <thought> + <code>"),
    ]
    for name, desc in instruction_templates:
        print(f"    {name}: {desc}")


def test_scores():
    """Test score display in prompts."""
    print("\n" + "=" * 80)
    print("Testing score display")
    print("=" * 80)

    programs = [
        (WORSE_FUNCTION, WORSE_SCORES),
        (BETTER_FUNCTION, BETTER_SCORES),
    ]

    for strategy in [PromptStrategy.FUNSEARCH, PromptStrategy.EOH, PromptStrategy.REEVO]:
        print(f"\n[{strategy.value} with show_scores=True]")
        spec = load_spec(strategy, show_scores=True)

        if strategy == PromptStrategy.FUNSEARCH:
            template_name = "funsearch"
        elif strategy == PromptStrategy.EOH:
            template_name = "e1"
        else:
            template_name = "reflect_st"

        prompt = prompt_builder.build_prompt(spec, template_name, programs)
        print(f"Prompt length: {len(prompt)} chars")
        print(f"Contains 'Relative to best known': {'Relative to best known' in prompt}")
        print(f"\n[Prompt (first 2000 chars)]")
        print(prompt[:2000])
        print("[End]\n")


def main():
    """Main entry point."""
    args = sys.argv[1:]

    if not args:
        print("Testing all strategies...")
        test_all()
    elif args[0] == "summary":
        summary()
    elif args[0] == "scores":
        test_scores()
    elif args[0] == "funsearch":
        test_funsearch()
    elif args[0] == "funsearch_instruction":
        template = args[1] if len(args) > 1 else None
        test_funsearch_instruction(template)
    elif args[0] == "fewshot_tags":
        test_fewshot_tags()
    elif args[0] == "eoh":
        template = args[1] if len(args) > 1 else None
        test_eoh(template)
    elif args[0] == "reevo":
        template = args[1] if len(args) > 1 else None
        test_reevo(template)
    else:
        print(f"Unknown command: {args[0]}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
