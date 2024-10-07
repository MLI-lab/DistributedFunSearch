import json
import re

def update_config_file_no_backrefs(config_file, params):
    """Update the config file with the provided parameters."""
    # Open the configuration file and read its content
    with open(config_file, 'r') as file:
        content = file.read()

    # Loop through the parameters and replace the corresponding values in the file
    for key, value in params.items():
        # If the value is a string, wrap it in quotes
        if isinstance(value, str):
            value = f'"{value}"'
        else:
            value = str(value)

        # Construct the pattern to match the key-value pair in the config file
        # Ensure that only the value part gets replaced, not the type part
        pattern = rf'({key}\s*:\s*[^\s=]+)\s*=\s*[^\n,]+'
        replacement = rf'\1 = {value}'

        # Replace the old value with the new value in the content
        content = re.sub(pattern, replacement, content)

    # Write the updated content back to the configuration file
    with open(config_file, 'w') as file:
        file.write(content)

# Add this block so that the script can be run both as a module and standalone
if __name__ == "__main__":
    import sys
    # The config file path is passed as the first argument, and parameters as the second
    config_file = sys.argv[1]
    params = json.loads(sys.argv[2])  # The second argument is the parameters in JSON format
    
    # Call the function to update the configuration file
    update_config_file_no_backrefs(config_file, params)

    # For verification, print out the final content of the updated file
    with open(config_file, 'r') as file:
        final_updated_content = file.read()
