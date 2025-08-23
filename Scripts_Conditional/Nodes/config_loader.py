# config_loader.py

import json
import os
import platform
import re


def resolve_placeholders(config):
    pattern = re.compile(r"\$\{(.*?)\}")
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
                        unresolved = True
    return config


def load_config(config_file=None):
    if config_file is None:
        # Use the config.json file in the same directory as this script
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

    with open(config_file, "r") as file:
        config = json.load(file)

    system = platform.system()

    if system == "Windows":
        paths = config["win_paths"]
    elif system == "Darwin":
        paths = config["mac_paths"]
    else:
        raise ValueError("Unsupported operating system: " + system)

    return resolve_placeholders(paths)
