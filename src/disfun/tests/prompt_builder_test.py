#!/usr/bin/env python3
"""Test script for prompt_builder.py

Usage:
    python -m disfun.prompt_builder_test                    # Test all strategies and templates
    python -m disfun.prompt_builder_test summary            # Print summary of all templates
    python -m disfun.prompt_builder_test scores             # Test score display in prompts
    python -m disfun.prompt_builder_test funsearch          # Test FunSearch only
    python -m disfun.prompt_builder_test eoh                # Test all EoH templates
    python -m disfun.prompt_builder_test eoh e1             # Test specific EoH template
    python -m disfun.prompt_builder_test reevo              # Test all ReEvo templates
    python -m disfun.prompt_builder_test reevo crossover    # Test specific ReEvo template
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from disfun.utils import prompt_builder
from disfun.utils.prompt_builder import PromptStrategy, PromptSpec


# Sample functions for testing
@dataclass
class MockFunction:
    """Mock Function object for testing."""
    name: str = "priority"
    args: str = "node, G, n, s"
    return_type: str = "float"
    body: str = "    return 1.0"
    docstring: str = ""
    thought: str = ""


# Two sample functions with different scores
WORSE_FUNCTION = MockFunction(
    name="priority",
    args="node, G, n, s",
    return_type="float",
    body='    return len(node)',
    docstring="A simple baseline that returns string length.",
    thought="I tried using string length as a simple heuristic."
)

BETTER_FUNCTION = MockFunction(
    name="priority",
    args="node, G, n, s",
    return_type="float",
    body='    ones = node.count("1")\n    degree = G.degree(node)\n    return ones * 10 - degree',
    docstring="Improved version considering bit count and graph structure.",
    thought="Counting ones and penalizing high-degree nodes should help find independent sets."
)

# Sample scores
WORSE_SCORES = {
    "(6, 1, 2)": 8,
    "(7, 1, 2)": 12,
    "(8, 1, 2)": 20,
}

BETTER_SCORES = {
    "(6, 1, 2)": 10,
    "(7, 1, 2)": 16,
    "(8, 1, 2)": 28,
}

# Best known solutions for relative score display
BEST_KNOWN_SOLUTIONS = {
    (6, 1, 2): 10,
    (7, 1, 2): 16,
    (8, 1, 2): 30,
}


def load_spec(strategy: PromptStrategy, show_scores: bool = False) -> PromptSpec:
    """Load spec for given strategy."""
    spec_dir = Path(__file__).parent / "specifications" / "Deletions"

    return prompt_builder.load_specification(
        strategy=strategy,
        spec_dir=str(spec_dir),
        imports_file="imports/networkx.txt",
        initial_functions_dir="initial_functions/graph_networkx",
        fewshot_num_examples=2,
        # Score display options
        show_scores=show_scores,
        score_display_mode="relative",
        best_known_solutions=BEST_KNOWN_SOLUTIONS,
        # Docstring templates (using defaults from spec_dir)
    )


def test_funsearch():
    """Test FunSearch prompt building."""
    print("\n" + "=" * 80)
    print("TESTING FUNSEARCH")
    print("=" * 80)

    spec = load_spec(PromptStrategy.FUNSEARCH)
    print(f"Loaded spec: {len(spec.templates)} template(s)")
    print(f"Function args: {spec.function_args}")
    print(f"Return type: {spec.return_type}")

    # FunSearch uses fewshot examples (2 programs)
    programs = [
        (WORSE_FUNCTION, WORSE_SCORES),
        (BETTER_FUNCTION, BETTER_SCORES),
    ]

    template_name, num_needed = prompt_builder.select_template(spec)
    print(f"Template: {template_name}, programs needed: {num_needed}")

    prompt = prompt_builder.build_prompt(spec, template_name, programs)

    print(f"\n--- PROMPT ({len(prompt)} chars) ---")
    print(prompt)
    print("--- END PROMPT ---\n")

    return prompt


def test_eoh(template_name: str = None):
    """Test EoH prompt building."""
    print("\n" + "=" * 80)
    print(f"TESTING EOH" + (f" - {template_name}" if template_name else " - ALL TEMPLATES"))
    print("=" * 80)

    spec = load_spec(PromptStrategy.EOH)
    print(f"Loaded spec: {len(spec.templates)} template(s): {list(spec.templates.keys())}")

    # Order: initialization first, then exploration, then mutation
    eoh_order = ['i1', 'e1', 'e2', 'm1', 'm2', 'm3']
    templates_to_test = [template_name] if template_name else [t for t in eoh_order if t in spec.templates]

    for tname in templates_to_test:
        if tname not in spec.templates:
            print(f"Template '{tname}' not found. Available: {list(spec.templates.keys())}")
            continue

        reqs = spec.template_requirements[tname]
        print(f"\n--- Template: {tname} (needs {reqs.num_programs} program(s), thought={reqs.needs_thought}) ---")

        # Select programs based on requirements
        if reqs.num_programs == 2:
            programs = [
                (WORSE_FUNCTION, WORSE_SCORES),
                (BETTER_FUNCTION, BETTER_SCORES),
            ]
        else:
            programs = [(BETTER_FUNCTION, BETTER_SCORES)]

        prompt = prompt_builder.build_prompt(spec, tname, programs)

        print(f"\n--- PROMPT ({len(prompt)} chars) ---")
        print(prompt)
        print("--- END PROMPT ---\n")

    # Also show system messages
    print("\n--- SYSTEM MESSAGES ---")
    print(f"Generator: {spec.system_message}" if spec.system_message else "Generator: None")

    return spec


def test_reevo(template_name: str = None):
    """Test ReEvo prompt building."""
    print("\n" + "=" * 80)
    print(f"TESTING REEVO" + (f" - {template_name}" if template_name else " - ALL TEMPLATES"))
    print("=" * 80)

    spec = load_spec(PromptStrategy.REEVO)
    print(f"Loaded spec: {len(spec.templates)} template(s): {list(spec.templates.keys())}")
    print(f"Initial reflection: '{spec.initial_reflection[:50]}...'" if spec.initial_reflection else "No initial reflection")

    # Order: seed, short-term reflection, crossover, long-term reflection, mutation
    reevo_order = ['seed', 'reflect_st', 'crossover', 'reflect_lt', 'mutation']
    templates_to_test = [template_name] if template_name else [t for t in reevo_order if t in spec.templates]

    # Sample ReEvo state with reflections
    reevo_state = {
        "reflection": "Better functions use bit counting.",
        "prior_reflection": "Initial approaches using only degree were not effective.",
        "new_reflections": "- Bit patterns matter more than graph structure\n- Penalizing neighbors helps",
        "initial_reflection": spec.initial_reflection,
    }

    for tname in templates_to_test:
        if tname not in spec.templates:
            print(f"Template '{tname}' not found. Available: {list(spec.templates.keys())}")
            continue

        reqs = spec.template_requirements[tname]
        print(f"\n--- Template: {tname} (needs {reqs.num_programs} program(s), reflection={reqs.needs_reflection}) ---")

        # Select programs based on requirements
        if reqs.num_programs == 2:
            programs = [
                (WORSE_FUNCTION, WORSE_SCORES),
                (BETTER_FUNCTION, BETTER_SCORES),
            ]
        else:
            programs = [(BETTER_FUNCTION, BETTER_SCORES)]

        prompt = prompt_builder.build_prompt(spec, tname, programs, state=reevo_state)

        print(f"\n--- PROMPT ({len(prompt)} chars) ---")
        print(prompt)
        print("--- END PROMPT ---\n")

    # Also show system messages
    print("\n--- SYSTEM MESSAGES ---")
    print(f"Generator: {spec.system_message}" if spec.system_message else "Generator: None")
    print(f"Reflector: {spec.reflector_system_message}" if spec.reflector_system_message else "Reflector: None")

    return spec


def test_all():
    """Test all strategies."""
    test_funsearch()
    test_eoh()
    test_reevo()


def summary():
    """Print summary of all templates without full prompts."""
    print("\n" + "=" * 80)
    print("PROMPT BUILDER SUMMARY")
    print("=" * 80)

    # Define template ordering for each strategy
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
            print(f"  Few-shot format: <thought>...</thought><code>...</code>")
        if strategy == PromptStrategy.REEVO:
            print(f"  Reflector system message: {'Yes' if spec.reflector_system_message else 'No'}")
            print(f"  Initial reflection: {'Yes' if spec.initial_reflection else 'No'}")
        print(f"  Templates:")
        ordered_templates = [t for t in template_orders[strategy] if t in spec.template_requirements]
        for name in ordered_templates:
            reqs = spec.template_requirements[name]
            print(f"    - {name}: {reqs.num_programs} program(s), reflection={reqs.needs_reflection}")


def test_scores():
    """Test score display in prompts."""
    print("\n" + "=" * 80)
    print("TESTING SCORE DISPLAY")
    print("=" * 80)

    programs = [
        (WORSE_FUNCTION, WORSE_SCORES),
        (BETTER_FUNCTION, BETTER_SCORES),
    ]

    for strategy in [PromptStrategy.FUNSEARCH, PromptStrategy.EOH, PromptStrategy.REEVO]:
        print(f"\n--- {strategy.value.upper()} with show_scores=True ---")
        spec = load_spec(strategy, show_scores=True)

        # Pick a template that uses both worse_code and better_code
        if strategy == PromptStrategy.FUNSEARCH:
            template_name = "funsearch"
        elif strategy == PromptStrategy.EOH:
            template_name = "e1"  # Uses both worse and better
        else:
            template_name = "reflect_st"  # ReEvo reflection uses both

        prompt = prompt_builder.build_prompt(spec, template_name, programs)
        print(f"Prompt length: {len(prompt)} chars")
        print(f"Contains 'Relative to best known': {'Relative to best known' in prompt}")
        print(f"\n--- PROMPT (first 2000 chars) ---")
        print(prompt[:2000])
        print("--- END ---\n")


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
    elif args[0] == "eoh":
        template = args[1] if len(args) > 1 else None
        test_eoh(template)
    elif args[0] == "reevo":
        template = args[1] if len(args) > 1 else None
        test_reevo(template)
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: python -m disfun.prompt_builder_test [summary|scores|funsearch|eoh|reevo] [template_name]")
        sys.exit(1)


if __name__ == "__main__":
    main()
