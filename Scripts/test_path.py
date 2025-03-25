import json
import os
import platform
import subprocess
import re


def resolve_placeholders(config):
    """Resolves ${...} placeholders in dictionary values."""
    pattern = re.compile(r"\$\{(.*?)\}")

    # Keep resolving until all variables are expanded
    unresolved = True
    while unresolved:
        unresolved = False
        for key, value in config.items():
            if isinstance(value, str):
                matches = pattern.findall(value)
                for match in matches:
                    if match in config:
                        value = value.replace("${" + match + "}", config[match])
                        config[key] = value
                        unresolved = True  # Continue until all placeholders are resolved
    return config


def load_config(config_file="config.json"):
    """Loads the configuration and selects paths based on the operating system."""
    with open(config_file, "r") as file:
        config = json.load(file)

    system = platform.system()

    if system == "Windows":
        paths = config["win_paths"]
    elif system == "Darwin":  # macOS
        paths = config["mac_paths"]
    else:
        raise ValueError("Unsupported operating system: " + system)

    return resolve_placeholders(paths)


def launch_blender(blender_path):
    """Launch Blender if the executable exists."""
    if os.path.exists(blender_path):
        print(f"Launching Blender from: {blender_path}")
        subprocess.Popen([blender_path])
    else:
        print(f"Blender path not found: {blender_path}")


if __name__ == "__main__":
    config = load_config()

    # Example use of resolved paths
    blender_path = config["blender_executable_path"]
    textures_path = config["textures_folder_path"]
    output_path = config.get("dataset_Refined_folder_path", "Unknown")

    print(f"Blender Path: {blender_path}")
    print(f"Textures Folder Path: {textures_path}")
    print(f"Output Path: {output_path}")

    # Launch Blender
    launch_blender(blender_path)