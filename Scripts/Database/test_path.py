
#import bpy
import json
import os
import glob
import platform

#Check if this works at all
def load_config(config_file="config.json"):
    """Loads the configuration and selects paths based on the operating system."""
    with open(config_file, "r") as file:
        config = json.load(file)

    system = platform.system()
    
    if system == "Windows":
        return config["win_paths"]
    elif system == "Darwin":  # macOS
        return config["mac_paths"]
    else:
        raise ValueError("Unsupported operating system: " + system)


if __name__ == "__main__":
    # Usage example
    config = load_config()

    blender_path = config["blender_path"]
    output_path = config["dataset_output"]

    print(f"Blender Path: {blender_path}")
    print(f"Output Path: {output_path}")
