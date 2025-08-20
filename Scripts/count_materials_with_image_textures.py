import json

def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def count_materials_with_image_textures(json_data):
    # Check if the data has a 'materials' key
    if 'materials' not in json_data:
        print("No 'materials' key found in the JSON data")
        return 0
    
    materials = json_data['materials']
    materials_with_image_textures = 0
    material_names_with_image_textures = []
    
    # Iterate through all materials
    for material_name, material_data in materials.items():
        if 'nodes' not in material_data:
            continue
        
        has_image_texture = False
        
        # Iterate through all nodes in the material
        for node in material_data['nodes']:
            if 'type' in node and node['type'] == 'TEX_IMAGE':
                has_image_texture = True
                break
        
        if has_image_texture:
            materials_with_image_textures += 1
            material_names_with_image_textures.append(material_name)
    
    return materials_with_image_textures, material_names_with_image_textures

# Path to the merged dataset
file_path = 'Dataset/Refined/merged_dataset.json'

# Read the JSON data
try:
    data = read_json_from_file(file_path)
    
    # Count materials with image textures
    count, material_names = count_materials_with_image_textures(data)
    
    # Print the count
    print(f"Number of materials with at least one image texture node: {count}")
    
    # Print the total number of materials for reference
    total_materials = len(data['materials']) if 'materials' in data else 0
    print(f"Total number of materials: {total_materials}")
    
    # Print the percentage
    if total_materials > 0:
        percentage = (count / total_materials) * 100
        print(f"Percentage of materials with image textures: {percentage:.2f}%")
    
    # Print a few example material names with image textures (if any)
    if material_names:
        print("\nExamples of materials with image textures:")
        for name in material_names[:5]:  # Print up to 5 examples
            print(f"- {name}")
        if len(material_names) > 5:
            print(f"... and {len(material_names) - 5} more")
    
except Exception as e:
    print(f"Error: {e}")