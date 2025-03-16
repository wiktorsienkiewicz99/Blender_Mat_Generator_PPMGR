import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
database_path = r"C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\merged_database.json"
node_to_id_path = r"C:\Users\hyperbook\Desktop\PPMGR\node_to_id.json"
id_to_node_path = r"C:\Users\hyperbook\Desktop\PPMGR\id_to_node.json"

def generate_node_to_id(input_file, node_to_id_file, id_to_node_file):
    logging.info("Loading training data...")
    with open(input_file, "r") as file:
        raw_data = json.load(file)

    logging.info("Extracting unique node types...")
    node_types = set()

    # Collect all unique node types
    for material_name, nodes in raw_data.get("nodes", {}).items():
        for node in nodes:
            if "type" in node:
                node_types.add(node["type"])

    # Create node-to-ID and ID-to-node mappings
    node_to_id = {node_type: idx + 1 for idx, node_type in enumerate(sorted(node_types))}  # Start IDs from 1
    id_to_node = {v: k for k, v in node_to_id.items()}

    # Save mappings to JSON files
    logging.info(f"Saving node-to-ID mapping to '{node_to_id_file}'...")
    with open(node_to_id_file, "w") as file:
        json.dump(node_to_id, file, indent=4)

    logging.info(f"Saving ID-to-node mapping to '{id_to_node_file}'...")
    with open(id_to_node_file, "w") as file:
        json.dump(id_to_node, file, indent=4)

    logging.info("Mappings generated successfully!")
    return node_to_id, id_to_node

# Run the script
if __name__ == "__main__":
    generate_node_to_id(database_path, node_to_id_path, id_to_node_path)
