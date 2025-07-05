# generate_node_type_to_params.py

import json
import re
from collections import defaultdict

def parse_vector(value):
    """Returns True if the string represents a vector or Euler tuple."""
    if isinstance(value, str):
        return bool(re.match(r"<(Vector|Euler)", value))
    return False

def extract_param_keys_from_node(node):
    keys = set()

    # ── Handle inputs ─────────────────────────────────────
    for inp in node.get("inputs", []):
        if inp.get("is_linked"):
            continue
        val = inp.get("value")

        if isinstance(val, (int, float)):
            keys.add(inp["name"])
        elif isinstance(val, bool):
            keys.add(inp["name"])
        elif isinstance(val, str) and parse_vector(val):
            # Assume vector-like: add .0, .1, .2
            keys.update(f"{inp['name']}.{i}" for i in range(3))

    # ── Handle parameters ─────────────────────────────────
    for k, v in node.get("parameters", {}).items():
        if isinstance(v, (int, float, str, bool)):
            keys.add(k)

    return keys

def build_node_type_to_params(merged_path, output_path):
    with open(merged_path, "r") as f:
        data = json.load(f)

    node_type_to_params = defaultdict(set)

    for mat in data["materials"].values():
        for node in mat.get("nodes", []):
            node_type = node["type"]
            keys = extract_param_keys_from_node(node)
            node_type_to_params[node_type].update(keys)

    # Convert sets to sorted lists
    node_type_to_params = {k: sorted(list(v)) for k, v in node_type_to_params.items()}

    with open(output_path, "w") as f:
        json.dump(node_type_to_params, f, indent=2)
    print(f"[✓] Saved node_type_to_params to: {output_path}")


# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
if __name__ == "__main__":
    MERGED_JSON = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/merged_dataset.json"
    OUTPUT_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_params.json"
    build_node_type_to_params(MERGED_JSON, OUTPUT_PATH)