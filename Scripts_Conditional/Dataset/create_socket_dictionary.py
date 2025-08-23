import json
import logging
from config_loader import load_config

config = load_config()

dataset_refined_folder = config["dataset_refined_folder_path"]
dataset_auxiliary_folder = config["dataset_auxiliary_folder_path"]

# Paths
merged_dataset_path = dataset_refined_folder + "/merged_dataset.json"
socket_to_id_path = dataset_auxiliary_folder + "/socket_to_id.json"
id_to_socket_path = dataset_auxiliary_folder + "/id_to_socket.json"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def generate_socket_to_id(input_file, socket_to_id_file, id_to_socket_file):
    logging.info("Loading merged dataset...")
    with open(input_file, "r") as file:
        raw_data = json.load(file)

    logging.info("Extracting unique socket names from edges...")
    socket_names = set()

    for material_name, material_data in raw_data.get("materials", {}).items():
        for edge in material_data.get("edges", []):
            if "from_socket" in edge:
                socket_names.add(edge["from_socket"])
            if "to_socket" in edge:
                socket_names.add(edge["to_socket"])

    # Create mappings
    socket_to_id = {name: idx for idx, name in enumerate(sorted(socket_names))}
    id_to_socket = {v: k for k, v in socket_to_id.items()}

    # Save to files
    logging.info(f"Saving socket-to-ID mapping to '{socket_to_id_file}'...")
    with open(socket_to_id_file, "w") as file:
        json.dump(socket_to_id, file, indent=4)

    logging.info(f"Saving ID-to-socket mapping to '{id_to_socket_file}'...")
    with open(id_to_socket_file, "w") as file:
        json.dump(id_to_socket, file, indent=4)

    logging.info("Socket mappings generated successfully!")
    return socket_to_id, id_to_socket


# Run the script
if __name__ == "__main__":
    generate_socket_to_id(merged_dataset_path, socket_to_id_path, id_to_socket_path)