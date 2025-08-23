import os
import json
from config_loader import load_config

config = load_config()

# Load paths from configuration file
dataset_refined_folder = config["dataset_refined_folder_path"]
dataset_raw_folder = config["dataset_raw_folder_path"]

# Local file path
merged_dataset_path = dataset_refined_folder + "/merged_dataset.json"


def merge_json_files(input_folder, output_file):
    merged_data = {
        "materials": {}
    }

    # Iterate through all JSON files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_folder, file_name)

            # Load the JSON data
            with open(file_path, "r") as file:
                data = json.load(file)

                # Merge materials
                for material_name, material_data in data.items():
                    if material_name not in merged_data["materials"]:
                        merged_data["materials"][material_name] = {
                            "nodes": [],
                            "edges": []
                        }

                    # Merge nodes
                    merged_data["materials"][material_name]["nodes"].extend(material_data.get("nodes", []))

                    # Merge edges
                    merged_data["materials"][material_name]["edges"].extend(material_data.get("edges", []))

    # Save the merged data into a new JSON file
    with open(output_file, "w") as output:
        json.dump(merged_data, output, indent=4)

merge_json_files(dataset_raw_folder, merged_dataset_path)

print(f"All JSON files from '{dataset_raw_folder}' have been merged into '{merged_dataset_path}'.")
