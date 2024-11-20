import os
import subprocess
import time
import ast

def extract_priority_functions(file_path):
    """
    Extract all `priority` functions from the `best_programs.py` file.
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
    Replace the existing `priority` function in the `evaluate.py` script.
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

def run_evaluation(script_path):
    """
    Run the `evaluate.py` script and capture its output.
    """
    start_time = time.time()
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    return result.stdout, elapsed_time

def main(best_programs_path, evaluate_script_path, output_file):
    """
    Replace `priority` functions iteratively and run the `evaluate.py` script.
    """
    # Step 1: Extract all priority functions
    functions = extract_priority_functions(best_programs_path)

    # Ensure the output file exists and clear it
    open(output_file, 'w').close()

    # Step 2: Iterate through each priority function
    for i, function in enumerate(functions):
        print(f"Running evaluation for priority function {i + 1}/{len(functions)}...")

        # Replace the priority function in the evaluation script
        replace_priority_function(evaluate_script_path, function)

        # Step 3: Run the evaluation script
        try:
            output, elapsed_time = run_evaluation(evaluate_script_path)
        except Exception as e:
            print(f"Error while running evaluation for function {i + 1}: {e}")
            continue

        # Step 4: Write result to the output file immediately
        with open(output_file, 'a') as f:
            f.write(f"Function {i + 1}:\n")
            f.write(output)
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n\n")
        
        print(f"Finished evaluation for function {i + 1}. Time: {elapsed_time:.2f}s")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Path to the `best_programs.py` file containing all priority functions
    best_programs_path = "best_programs.py"
    # Path to the evaluation script (`evaluate.py`)
    evaluate_script_path = "evaluate.py"
    # Output file to save results
    output_file = "evaluation_results.txt"

    main(best_programs_path, evaluate_script_path, output_file)
