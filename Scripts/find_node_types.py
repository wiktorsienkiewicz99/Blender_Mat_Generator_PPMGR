import json

def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_unique_node_types(json_data):
    # Check if the data has a 'materials' key
    if 'materials' not in json_data:
        print("No 'materials' key found in the JSON data")
        return set()
    
    materials = json_data['materials']
    node_types = set()
    
    # Iterate through all materials
    for material_name, material_data in materials.items():
        if 'nodes' not in material_data:
            continue
        
        # Iterate through all nodes in the material
        for node in material_data['nodes']:
            if 'type' in node:
                node_types.add(node['type'])
    
    return node_types

# Path to the merged dataset
file_path = 'Dataset/Refined/merged_dataset.json'

# Read the JSON data
try:
    data = read_json_from_file(file_path)
    
    # Find unique node types
    unique_node_types = find_unique_node_types(data)
    
    # Print the unique node types
    print(f"Found {len(unique_node_types)} unique node types:")
    for node_type in sorted(unique_node_types):
        print(f"- {node_type}")
    
except Exception as e:
    print(f"Error: {e}")