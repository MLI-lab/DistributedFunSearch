import os
import ast
import subprocess


def extract_priority_functions(file_path):
    """
    Extract all `priority` functions from the specified file.
    """
    with open(file_path, 'r') as f:
        source = f.read()
    tree = ast.parse(source, filename=file_path)

    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'priority':
            func_code = ast.get_source_segment(source, node)
            functions.append(func_code)
    return functions


def replace_priority_function(script_path, priority_function_code):
    """
    Replace the existing `priority` function in the `compare.py` script.
    """
    with open(script_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source, filename=script_path)
    new_body = []
    found = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'priority':
            # Replace this function
            new_node = ast.parse(priority_function_code).body[0]
            new_body.append(new_node)
            found = True
        else:
            new_body.append(node)
    if not found:
        # No existing priority function found, append the new one
        new_node = ast.parse(priority_function_code).body[0]
        new_body.append(new_node)

    # Include type_ignores attribute
    module_node = ast.Module(body=new_body, type_ignores=getattr(tree, 'type_ignores', []))

    # Generate the new source code
    try:
        new_source = ast.unparse(module_node)
    except AttributeError:
        # For Python versions < 3.9, use astor library
        try:
            import astor
        except ImportError:
            print("The 'astor' library is required for this script. Please install it using 'pip install astor'.")
            exit(1)
        new_source = astor.to_source(module_node)

    with open(script_path, 'w') as f:
        f.write(new_source)


def run_evaluation(script_path, params=(6, 1)):
    """
    Run the `compare.py` script and capture its output.
    """
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    stdout = result.stdout.strip()
    try:
        # Extract the independent set
        for line in stdout.split('\n'):
            if line.startswith('Independent set:'):
                independent_set = eval(line.split(':', 1)[1].strip())  # Extract the list
                return set(independent_set)  # Convert to set for comparison
        raise ValueError("Independent set not found in output")
    except Exception as e:
        print(f"Error parsing output: {e}")
        return set()


def calculate_overlap(vt_solution, new_solution):
    """
    Calculate overlap and non-overlap between two solutions.
    """
    overlap = len(vt_solution.intersection(new_solution))
    total_new = len(new_solution)
    non_overlap = total_new - overlap
    return overlap, non_overlap, total_new


def main(programs_path, vt_priority_path, evaluate_script_path, output_file, complete_overlap_file, partial_overlap_file):
    """
    Compare VT solutions with priority functions in the program file.
    """
    # Step 1: Run VT codes priority function
    print("Running evaluation with VT codes...")
    with open(vt_priority_path, 'r') as f:
        vt_code = f.read()
    replace_priority_function(evaluate_script_path, vt_code)

    vt_solution = run_evaluation(evaluate_script_path)
    if not vt_solution:
        print("Error: VT solution is empty. Check the VT priority function or evaluation script.")
        return

    print(f"VT solution generated with {len(vt_solution)} sequences.")

    # Step 2: Extract priority functions from the provided program file
    functions = extract_priority_functions(programs_path)

    # Ensure the output files exist and clear them
    open(output_file, 'w').close()
    open(complete_overlap_file, 'w').close()
    open(partial_overlap_file, 'w').close()

    # Initialize counters
    complete_overlap_count = 0
    partial_overlap_count = 0

    # Step 3: Iterate through each priority function
    for i, function in enumerate(functions):
        print(f"Running evaluation for priority function {i + 1}/{len(functions)}...")

        # Replace the priority function in the evaluation script
        replace_priority_function(evaluate_script_path, function)

        # Step 4: Run the evaluation script
        new_solution = run_evaluation(evaluate_script_path)

        # Step 5: Calculate overlap and non-overlap
        overlap, non_overlap, total_new = calculate_overlap(vt_solution, new_solution)

        # Step 6: Write result to the output file immediately
        with open(output_file, 'a') as f:
            f.write(f"Function {i + 1}:\n")
            f.write(f"Total Sequences: {total_new}\n")
            f.write(f"Overlap: {overlap}/{len(vt_solution)} sequences\n")
            f.write(f"Non-Overlap: {non_overlap} sequences\n\n")

        # Step 7: Categorize functions based on overlap
        if overlap == len(vt_solution) and non_overlap == 0:
            complete_overlap_count += 1
            with open(complete_overlap_file, 'a') as f:
                f.write(f"Function {i + 1}:\n{function}\n\n")
            print(f"Function {i + 1} has complete overlap.")
        else:
            partial_overlap_count += 1
            with open(partial_overlap_file, 'a') as f:
                f.write(f"Function {i + 1}:\n{function}\n\n")
            print(f"Function {i + 1} has partial overlap.")

        print(f"Finished evaluation for function {i + 1}. Overlap: {overlap}/{len(vt_solution)} sequences. "
              f"Non-Overlap: {non_overlap} sequences. Total: {total_new} sequences.")

    # Print summary
    print(f"\nSummary:")
    print(f"Functions with complete overlap: {complete_overlap_count}")
    print(f"Functions with partial overlap: {partial_overlap_count}")

    # Save summary to output file
    with open(output_file, 'a') as f:
        f.write(f"\nSummary:\n")
        f.write(f"Functions with complete overlap: {complete_overlap_count}\n")
        f.write(f"Functions with partial overlap: {partial_overlap_count}\n")


if __name__ == "__main__":
    # Path to the `unique_priority_programs.py` file containing all priority functions
    programs_path = "unique_priority_programs.py"
    # Path to the VT priority function file
    vt_priority_path = "VT_code.py"
    # Path to the evaluation script (`compare.py`)
    evaluate_script_path = "compare.py"
    # Output file to save results
    output_file = "overlap_results.txt"
    # Output file for functions with complete overlap
    complete_overlap_file = "complete_overlap_functions.txt"
    # Output file for functions with partial overlap
    partial_overlap_file = "partial_overlap_functions.txt"

    main(programs_path, vt_priority_path, evaluate_script_path, output_file, complete_overlap_file, partial_overlap_file)
