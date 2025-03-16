"""
### Script Summary

This script extracts only the edge information from a JSON database containing materials, nodes, and edges. It saves the cleaned data with only the "edges" for each material to a new JSON file.

#### Key Components:
1. **Paths:**
   - `merged_database_path`: Path to the input JSON file containing the original database.
   - `cleaned_edge_database_path`: Path to the output JSON file for saving the cleaned edges.

2. **Functionality:**
   - **`extract_edges_only`:**
     - Reads the input JSON file.
     - Extracts the "edges" information for each material.
     - Logs a message if no edges are found for a material.
     - Saves the extracted edges to a new JSON file.

3. **Execution:**
   - The script is executed as a standalone module.
   - Outputs the cleaned data to the specified output file.

#### Example Output:
The output JSON will have the following structure:
```json
{
    "materials": {
        "Material_1": {
            "edges": [
                {
                    "from_node": "Node1",
                    "from_socket": "Color",
                    "to_node": "Node2",
                    "to_socket": "Base Color"
                }
            ]
        },
        "Material_2": {
            "edges": []
        }
    }
}
```
"""


import json
import os

# Paths
merged_database_path = r"C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\merged_database.json"
cleaned_edge_database_path = r"C:\Users\hyperbook\Desktop\PPMGR\cleaned_edge_database.json"

def extract_edges_only(input_file, output_file):
    """Extract only edges from the merged database and save to a new file."""
    with open(input_file, "r") as file:
        data = json.load(file)

    cleaned_data = {"materials": {}}

    for material_name, material_data in data.get("materials", {}).items():
        # Extract edges
        edges = material_data.get("edges", [])

        if edges:  # Ensure edges are not empty
            cleaned_data["materials"][material_name] = {"edges": edges}
        else:
            print(f"No edges found in material: {material_name}")

    # Save the cleaned edge data into a new JSON file
    with open(output_file, "w") as file:
        json.dump(cleaned_data, file, indent=4)

    print(f"Edges extracted and saved to '{output_file}'.")

# Run the extraction
if __name__ == "__main__":
    extract_edges_only(merged_database_path, cleaned_edge_database_path)
