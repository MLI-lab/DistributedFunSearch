import re
import sys

def update_config_file(config_file, host_value):
    """Update the RabbitMQ host in the config file based on the assigned SLURM node."""
    try:
        with open(config_file, 'r') as file:
            content = file.read()
    except Exception as e:
        print(f"Couldnt not read content file {e}")

    # Replace the `host` attribute in the configuration with hostname str of assigned node
    # Use word boundary \b to match only 'host', not 'vhost'
    pattern = r"(\bhost\s*:\s*str\s*=\s*)['\"].*?['\"]|\bhost\s*:\s*str\s*=\s*''"
    replacement = rf"\1'{host_value}'"
    updated_content = re.sub(pattern, replacement, content)


    with open(config_file, 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":
    config_file_path = sys.argv[1]
    new_host_value = sys.argv[2]
    update_config_file(config_file_path, new_host_value)
