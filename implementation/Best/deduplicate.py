import ast
import astunparse
import re

# Input and output paths
combined_file = "preprocessed_programs.py"
deduplicated_output = "best_programs.py"
error_log = "deduplication_errors.log"

# Normalize variable names and remove comments/docstrings
def normalize_function_code(code):
    try:
        tree = ast.parse(code)

        # Replace variable names with placeholders
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                node.arg = "var"
            elif isinstance(node, ast.Name):
                node.id = "var"
            elif isinstance(node, ast.FunctionDef):
                # Remove docstrings
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    node.body = node.body[1:]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""  # Remove string constants

        normalized_code = astunparse.unparse(tree)
        normalized_code = re.sub(r"#.*", "", normalized_code)  # Remove inline comments
        return normalized_code.strip()
    except SyntaxError as e:
        raise ValueError(f"Normalization failed: {e}")

# Deduplicate programs
def deduplicate_programs(input_file, output_file):
    unique_signatures = set()
    deduplicated_programs = []
    total_programs = 0  # Counter for total programs

    with open(input_file, "r") as f:
        programs = f.read().split("\ndef ")

    with open(output_file, "w") as output, open(error_log, "w") as log:
        output.write("# Deduplicated programs\n\n")

        for program in programs:
            if not program.strip():
                continue

            total_programs += 1  # Increment total program count
            function_code = f"def {program.strip()}"
            try:
                normalized_code = normalize_function_code(function_code)
                if normalized_code not in unique_signatures:
                    unique_signatures.add(normalized_code)
                    deduplicated_programs.append(function_code)
            except ValueError as e:
                # Log raw program causing issues
                log.write(f"Error deduplicating program: {e}\nProgram:\n{function_code}\n{'=' * 50}\n")

        for deduplicated in deduplicated_programs:
            output.write(deduplicated + "\n\n")

    print(f"Total programs before deduplication: {total_programs}")
    print(f"Total programs after deduplication: {len(deduplicated_programs)}")
    print(f"Deduplicated programs written to {output_file}")
    print(f"Errors logged to {error_log}")

# Run deduplication
deduplicate_programs(combined_file, deduplicated_output)
