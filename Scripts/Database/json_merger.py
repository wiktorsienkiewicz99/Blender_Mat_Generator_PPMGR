import os
import json

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

# Define the input folder and output file
input_folder = r"C:\Users\hyperbook\Desktop\PPMGR\Projects\Results"  # Replace with the path to your folder
output_file = r"C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\merged_database.json"      # Replace with your desired output file name

merge_json_files(input_folder, output_file)

print(f"All JSON files from '{input_folder}' have been merged into '{output_file}'.")
