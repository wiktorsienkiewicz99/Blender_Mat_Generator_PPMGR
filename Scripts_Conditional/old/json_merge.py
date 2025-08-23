import json
import glob
import os

def merge_json_files(input_folder, output_file):
    """Merge all JSON files in the input folder into a single JSON file."""
    merged_data = {
        'nodes': [],
        'edges': []
    }
    
    # Find all JSON files in the input folder
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Append nodes and edges from each file to the merged data
            merged_data['nodes'].extend(data.get('nodes', []))
            merged_data['edges'].extend(data.get('edges', []))
    
    # Save the merged data to the output file
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"Merged JSON saved to {output_file}")

# Define the folder containing JSON files and the output merged file path
input_folder = os.path.abspath("path/to/your/json/folder")  # Folder with individual JSON files
output_file = os.path.abspath("path/to/your/output/folder/merged_materials.json")  # Path for the merged JSON file

# Run the merge function
merge_json_files(input_folder, output_file)
