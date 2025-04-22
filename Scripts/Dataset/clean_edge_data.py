import json
import logging
from config_loader import load_config

# Load paths from config
config = load_config()
dataset_refined_folder = config["dataset_refined_folder_path"]
dataset_auxiliary_folder = config["dataset_auxiliary_folder_path"]

merged_dataset_path = dataset_refined_folder + "/merged_dataset.json"
node_to_id_path = dataset_auxiliary_folder + "/node_to_id.json"
socket_to_id_path = dataset_auxiliary_folder + "/socket_to_id.json"
cleaned_graph_data_path = dataset_refined_folder + "/cleaned_graph_dataset.json"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def generate_cleaned_graph_dataset(merged_path, node_to_id_path, socket_to_id_path, output_path):
    logging.info("Loading files...")
    with open(merged_path, "r") as f:
        raw_data = json.load(f)

    with open(node_to_id_path, "r") as f:
        node_to_id = json.load(f)

    with open(socket_to_id_path, "r") as f:
        socket_to_id = json.load(f)

    samples = []

    logging.info("Processing materials...")
    for mat_name, mat_data in raw_data.get("materials", {}).items():
        nodes = mat_data.get("nodes", [])
        edges = mat_data.get("edges", [])

        if not nodes or not edges:
            continue

        node_name_to_type = {}     # name → type
        node_name_to_type_id = {}  # name → global node ID
        node_types = []

        for node in nodes:
            node_name = node.get("name")
            node_type = node.get("type")
            if not node_name or not node_type:
                continue

            node_type_id = node_to_id.get(node_type)
            if node_type_id is None:
                logging.warning(f"Unknown node type '{node_type}' in '{mat_name}'. Skipping.")
                break

            node_name_to_type[node_name] = node_type
            node_name_to_type_id[node_name] = node_type_id
            node_types.append(node_type)

        else:  # Only continue if no break happened (i.e., all node types known)

            # Build full node sequence with IDs
            node_ids = [node_to_id[t] for t in node_types]

            formatted_edges = []
            for edge in edges:
                from_name = edge.get("from_node")
                to_name = edge.get("to_node")
                from_socket = edge.get("from_socket")
                to_socket = edge.get("to_socket")

                if from_name not in node_name_to_type_id or to_name not in node_name_to_type_id:
                    continue

                from_node_id = node_name_to_type_id[from_name]
                to_node_id = node_name_to_type_id[to_name]

                from_socket_id = socket_to_id.get(from_socket)
                to_socket_id = socket_to_id.get(to_socket)

                if from_socket_id is None or to_socket_id is None:
                    continue

                formatted_edges.append([
                    from_node_id,
                    from_socket_id,
                    to_node_id,
                    to_socket_id
                ])

            samples.append({
                "material_name": mat_name,
                "nodes": node_ids,
                "node_types": node_types,
                "edges": formatted_edges
            })

    logging.info(f"Saving cleaned dataset with {len(samples)} materials to '{output_path}'...")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    logging.info("Cleaned dataset saved successfully!")

# Run
if __name__ == "__main__":
    generate_cleaned_graph_dataset(
        merged_dataset_path,
        node_to_id_path,
        socket_to_id_path,
        cleaned_graph_data_path
    )