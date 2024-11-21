import re
import sys

def update_config_file(config_file, host_value):
    """Update the RabbitMQ host in the config file."""
    try: 
        with open(config_file, 'r') as file:
            content = file.read()
    except Exception as e: 
        print(f"Couldnt not read content file {e}")

    # Replace the `host` attribute in the configuration
    pattern = r"(host\s*:\s*str\s*=\s*)['\"].*?['\"]|host\s*:\s*str\s*=\s*''"
    replacement = rf"\1'{host_value}'"
    updated_content = re.sub(pattern, replacement, content)

    # Debugging: print out the updated content to verify
    #print("Updated config content:")
    #print(updated_content)

    with open(config_file, 'w') as file:
        file.write(updated_content)

if __name__ == "__main__":
    config_file_path = sys.argv[1]
    new_host_value = sys.argv[2]
    update_config_file(config_file_path, new_host_value)
