"""Helper module for loading modular specification files."""

from pathlib import Path
import logging
import os

logger = logging.getLogger('main_logger')


def load_specification_files(problem_description_path: str, prompt_style_path: str | None, system_message_path: str | None = None, imports_path: str | None = None) -> tuple[str, str, str, str]:
    """Load problem description - prompt style - system message - and imports files using absolute paths.

    Args:
        problem_description_path: Absolute path to problem description file
        prompt_style_path: Absolute path to prompt style file - or None for code completion mode (no instructions)
        system_message_path: Absolute path to system message file for API models - or None to disable
        imports_path: Absolute path to imports file - or None to skip imports injection

    Returns:
        tuple: (problem_desc_content, prompt_style_content, system_message_content, imports_content)
    """
    # Load problem description
    problem_file = Path(problem_description_path)
    if not problem_file.exists():
        logger.warning(f"Problem description not found at {problem_file}")
        problem_desc_content = ""
    else:
        problem_desc_content = problem_file.read_text()
        logger.debug(f"Loaded problem description from {problem_file}")

    # Load prompt style (if specified)
    if prompt_style_path is None:
        logger.debug("No prompt style specified, using code completion mode (no instructions)")
        prompt_style_content = ""
    else:
        prompt_style_file = Path(prompt_style_path)
        if not prompt_style_file.exists():
            logger.warning(f"Prompt style not found at {prompt_style_file}, using empty style (code completion mode)")
            prompt_style_content = ""
        else:
            prompt_style_content = prompt_style_file.read_text().strip()
            logger.debug(f"Loaded prompt style from {prompt_style_file}")

    # Load system message (if specified)
    if system_message_path is None:
        logger.debug("No system message specified, API models will not use a system message")
        system_message_content = ""
    else:
        system_message_file = Path(system_message_path)
        if not system_message_file.exists():
            logger.warning(f"System message not found at {system_message_file}, API models will not use a system message")
            system_message_content = ""
        else:
            system_message_content = system_message_file.read_text().strip()
            logger.debug(f"Loaded system message from {system_message_file}")

    # Load imports (if specified)
    if imports_path is None:
        logger.debug("No imports path specified, using template-based imports (if any)")
        imports_content = ""
    else:
        imports_file = Path(imports_path)
        if not imports_file.exists():
            logger.warning(f"Imports file not found at {imports_file}, no imports will be injected")
            imports_content = ""
        else:
            imports_content = imports_file.read_text().strip()
            logger.debug(f"Loaded imports from {imports_file}")

    return problem_desc_content, prompt_style_content, system_message_content, imports_content


def build_fewshot_examples(sampled_programs: list, prompt_config) -> str:
    """Build few-shot examples based on prompt configuration.

    Args:
        sampled_programs: List of (program - scores) tuples
        prompt_config: PromptConfig object with fewshot settings

    Returns:
        str: Formatted few-shot examples (complete functions without labels or tags)
    """
    if not sampled_programs:
        return ""

    examples = []
    for i, (program, scores) in enumerate(sampled_programs, 1):
        example_parts = []

        # Add thinking (if enabled and present)
        if prompt_config.fewshot_show_thinking and hasattr(program, 'thinking') and program.thinking:
            example_parts.append(f"<thinking>\n{program.thinking}\n</thinking>")

        # Add thought (if enabled and present)
        if prompt_config.fewshot_show_thought and hasattr(program, 'thought') and program.thought:
            example_parts.append(f"<thought>\n{program.thought}\n</thought>")

        # Add code (if enabled) - show full function with signature
        if prompt_config.fewshot_show_code:
            # Construct complete function (program.args is already a string - not a list)
            function_str = f"def {program.name}({program.args}):"
            if program.docstring:
                # Wrap docstring in triple quotes if not already present
                if not program.docstring.strip().startswith('"""'):
                    function_str += f'\n    """{program.docstring}"""'
                else:
                    function_str += f"\n{program.docstring}"
            if program.body:
                function_str += f"\n{program.body}"

            # Don't wrap in <code> tags - let the prompt style instruction guide the LLM
            # The evaluator handles responses with or without <code> tags
            example_parts.append(function_str)

        if example_parts:
            examples.append("\n\n".join(example_parts))

    return "\n\n".join(examples)
