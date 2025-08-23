import json
import logging
from collections import defaultdict
from config_loader import load_config

# Setup
config = load_config()
dataset_refined_folder = config["dataset_refined_folder_path"]
dataset_auxiliary_folder = config["dataset_auxiliary_folder_path"]
merged_dataset_path = dataset_refined_folder + "/merged_dataset.json"
output_path = dataset_auxiliary_folder + "/node_type_to_sockets.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def generate_node_type_to_sockets(merged_path, output_path):
    with open(merged_path, "r") as f:
        raw_data = json.load(f)

    result = defaultdict(lambda: {"inputs": set(), "outputs": set()})
    material_count = 0
    node_count = 0

    for mat_name, mat_data in raw_data.get("materials", {}).items():
        material_count += 1
        for node in mat_data.get("nodes", []):
            node_type = node.get("type")
            if not node_type:
                continue

            for sock in node.get("inputs", []):
                result[node_type]["inputs"].add(sock["name"])
            for sock in node.get("outputs", []):
                result[node_type]["outputs"].add(sock["name"])

            node_count += 1

    # Convert sets to sorted lists
    result_clean = {
        node_type: {
            "inputs": sorted(list(sockets["inputs"])),
            "outputs": sorted(list(sockets["outputs"]))
        }
        for node_type, sockets in result.items()
    }

    with open(output_path, "w") as f:
        json.dump(result_clean, f, indent=2)

    logging.info(f"Processed {material_count} materials, {node_count} nodes.")
    logging.info(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_node_type_to_sockets(merged_dataset_path, output_path)