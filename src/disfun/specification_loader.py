"""Helper module for loading template-based specification files."""

from pathlib import Path
import ast
import logging
import re

logger = logging.getLogger('main_logger')


def load_template(template_path: str) -> str:
    """Load template file content."""
    return Path(template_path).read_text()


def load_placeholder_contents(placeholders: dict[str, str]) -> dict[str, tuple[str | None, dict | None]]:
    """Load content for each placeholder (once at init).

    Args:
        placeholders: Dict mapping placeholder names to file/directory paths.

    Returns:
        Dict: {name: (content_if_file, dict_if_directory)}
            - If file: (content_string, None)
            - If directory: (None, {filename: (content, fewshot_override)})
    """
    result = {}
    fs_pattern = re.compile(r'^fs(\d+)_')

    for name, path in placeholders.items():
        if path is None:
            result[name] = ("", None)
        elif Path(path).is_dir():
            # Load all .txt files from directory
            styles = {}
            for txt_file in Path(path).glob("*.txt"):
                filename = txt_file.stem
                content = txt_file.read_text().strip()
                match = fs_pattern.match(filename)
                fewshot_override = int(match.group(1)) if match else None
                styles[filename] = (content, fewshot_override)
            result[name] = (None, styles) if styles else ("", None)
        elif Path(path).exists():
            # Don't strip imports so trailing newline creates blank line before functions
            content = Path(path).read_text()
            if name != "imports":
                content = content.strip()
            result[name] = (content, None)
        else:
            logger.warning(f"Placeholder path not found: {path}")
            result[name] = ("", None)

    return result


def extract_function_signature(initial_function_path: str) -> tuple[str, str]:
    """Extract argument list and return type from initial function file using AST.

    Handles files with <code>...</code> tags by extracting the code block first.

    Returns:
        tuple: (args_string, return_type_string)
    """
    content = Path(initial_function_path).read_text()

    # Extract code from <code> tags if present
    code_match = re.search(r'<code>(.*?)</code>', content, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()

    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = ", ".join(arg.arg for arg in node.args.args)
            ret_type = ast.unparse(node.returns) if node.returns else "float"
            return args, ret_type
    raise ValueError(f"No function found in {initial_function_path}")


def strip_function_from_code(code: str, function_name: str) -> str:
    """Remove a function definition from code using AST.

    Args:
        code: Python source code as string.
        function_name: Name of function to remove.

    Returns:
        Code with the function removed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Find the function to remove
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start_line = node.lineno
            end_line = node.end_lineno
            lines = code.splitlines()
            # Remove lines (1-indexed to 0-indexed)
            new_lines = lines[:start_line - 1] + lines[end_line:]
            # Clean up extra blank lines
            result = "\n".join(new_lines)
            # Remove multiple consecutive blank lines
            while "\n\n\n" in result:
                result = result.replace("\n\n\n", "\n\n")
            return result.strip()

    return code


def build_fewshot_examples(sampled_programs: list, prompt_config, format_scores_fn=None) -> str:
    """Build few-shot examples with optional score display.

    Args:
        sampled_programs: List of (program, scores) tuples
        prompt_config: PromptConfig with fewshot and score display settings
        format_scores_fn: Optional function to format scores for docstrings

    Returns:
        str: Formatted few-shot examples
    """
    if not sampled_programs:
        return ""

    examples = []
    for program, scores in sampled_programs:
        parts = []

        # Add thinking/thought if enabled
        if prompt_config.fewshot_show_thinking and getattr(program, 'thinking', None):
            parts.append(f"<thinking>\n{program.thinking}\n</thinking>")
        if prompt_config.fewshot_show_thought and getattr(program, 'thought', None):
            parts.append(f"<thought>\n{program.thought}\n</thought>")

        # Add code if enabled
        if prompt_config.fewshot_show_code:
            func_str = f"def {program.name}({program.args}):"

            # Build docstring with optional scores
            docstring = program.docstring or ""
            if prompt_config.show_eval_scores and scores and format_scores_fn:
                score_text = format_scores_fn(scores)
                docstring = docstring.replace("{score}", score_text).strip()
            else:
                # Remove {score} placeholder when not showing scores
                docstring = docstring.replace("{score}", "")
                # Remove lines that are empty or whitespace-only, then strip
                lines = [line.rstrip() for line in docstring.split('\n') if line.strip()]
                docstring = '\n'.join(lines)

            if docstring:
                if not docstring.startswith('"""'):
                    func_str += f'\n    """{docstring}"""'
                else:
                    func_str += f"\n{docstring}"

            if program.body:
                func_str += f"\n{program.body}"

            parts.append(func_str)

        if parts:
            examples.append("\n\n".join(parts))

    return "\n\n".join(examples)
