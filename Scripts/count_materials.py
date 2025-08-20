import json

def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def count_materials(json_data):
    # Check if the data has a 'materials' key
    if 'materials' in json_data:
        materials = json_data['materials']
        return len(materials)
    else:
        print("No 'materials' key found in the JSON data")
        return 0

# Path to the merged dataset
file_path = 'Dataset/Refined/merged_dataset.json'

# Read the JSON data
try:
    data = read_json_from_file(file_path)
    
    # Count the materials
    material_count = count_materials(data)
    
    print(f"Number of materials in {file_path}: {material_count}")
except Exception as e:
    print(f"Error: {e}")