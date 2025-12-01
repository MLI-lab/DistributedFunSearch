import re
import sys

def update_config_file(config_file, host_value, vhost=None, port=None, management_port=None):
    """Update the RabbitMQ settings in the config file based on the assigned SLURM node.

    Args:
        config_file: Path to the config.py file
        host_value: RabbitMQ hostname
        vhost: Virtual host name (optional)
        port: AMQP port (optional)
        management_port: Management API port (optional)
    """
    try:
        with open(config_file) as file:
            content = file.read()
    except Exception as e:
        print(f"Could not read config file: {e}")
        return

    # Replace the `host` attribute in the configuration with hostname str of assigned node
    # Use word boundary \b to match only 'host' not 'vhost'
    pattern = r"(\bhost\s*:\s*str\s*=\s*)['\"].*?['\"]"
    replacement = rf"\1'{host_value}'"
    content = re.sub(pattern, replacement, content)

    # Update vhost if provided
    if vhost is not None:
        pattern = r"(vhost\s*:\s*str\s*=\s*)['\"].*?['\"]"
        replacement = rf"\1'{vhost}'"
        content = re.sub(pattern, replacement, content)

    # Update port if provided
    if port is not None:
        pattern = r"(\bport\s*:\s*int\s*=\s*)\d+"
        replacement = rf"\g<1>{port}"
        content = re.sub(pattern, replacement, content)

    # Update management_port if provided
    if management_port is not None:
        pattern = r"(management_port\s*:\s*int\s*=\s*)\d+"
        replacement = rf"\g<1>{management_port}"
        content = re.sub(pattern, replacement, content)

    with open(config_file, 'w') as file:
        file.write(content)

    print(f"Updated config: host={host_value}, vhost={vhost}, port={port}, management_port={management_port}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python update_config_file.py <config_file> <host> [vhost] [port] [management_port]")
        sys.exit(1)

    config_file_path = sys.argv[1]
    new_host_value = sys.argv[2]
    new_vhost = sys.argv[3] if len(sys.argv) > 3 else None
    new_port = int(sys.argv[4]) if len(sys.argv) > 4 else None
    new_management_port = int(sys.argv[5]) if len(sys.argv) > 5 else None

    update_config_file(config_file_path, new_host_value, new_vhost, new_port, new_management_port)
