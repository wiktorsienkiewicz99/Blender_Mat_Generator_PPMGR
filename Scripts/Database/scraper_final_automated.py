"""
### Script Summary

This script processes `.blend` files in a specified directory to extract material node and edge data, including ungrouped internal node group data. The results are saved as JSON files for further use.
/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Database/run_scraper.sh
#### Key Components:
1. **Functions**:
   - **`get_socket_value(socket)`**:
     - Retrieves the default value of a given socket.
     - Handles various data types, such as floats, integers, strings, and lists.
   - **`get_node_parameters(node)`**:
     - Extracts all node-specific parameters, including dropdown values and socket default values.
     - Handles specific parameters for various node types (e.g., color ramp, image texture settings).
   - **`ungroup_node_tree(node_tree, parent_node_name)`**:
     - Flattens a group node tree by exposing its internal nodes and links.
     - Processes group nodes recursively to extract ungrouped nodes and edges.
   - **`extract_material_data()`**:
     - Processes all materials in the Blender file.
     - Extracts node and edge data, including ungrouped group nodes.
   - **`save_material_data_to_json(material_data, output_path)`**:
     - Saves the extracted material data (nodes and edges) to a JSON file.
   - **`process_blend_files_in_directory(blend_folder, output_directory)`**:
     - Iterates over all `.blend` files in a directory.
     - Extracts material data for each file and saves it as a JSON file in the specified output directory.

2. **Paths**:
   - `blend_folder`: Directory containing the `.blend` files to process.
   - `output_directory`: Directory where the processed material data is saved as JSON files.

3. **Execution**:
   - The script iterates through `.blend` files in the input folder.
   - It extracts and saves ungrouped node and edge data for each material into separate JSON files.

#### Example Output:
The output JSON file contains material data with ungrouped nodes and edges:
```json
{
    "Material_1": {
        "nodes": [
            {
                "name": "Image Texture",
                "type": "TEX_IMAGE",
                "parameters": {
                    "Image Path": "path/to/image.png",
                    "Color Space": "sRGB"
                },
                "inputs": [],
                "outputs": [{"name": "Color"}]
            }
        ],
        "edges": [
            {
                "from_node": "Image Texture",
                "from_type": "TEX_IMAGE",
                "from_socket": "Color",
                "to_node": "Material Output",
                "to_type": "OUTPUT_MATERIAL",
                "to_socket": "Surface"
            }
        ]
    }
}
```

#### Key Features:
- **Handles Node Groups**: Recursively processes nodes within groups and includes their connections.
- **Extracts Detailed Parameters**: Saves comprehensive node parameter data for further analysis or reconstruction.
- **Edge Details**: Captures connections between nodes, including socket names and node types.

This script is especially useful for processing Blender materials into a JSON-based representation for external use, such as visualization or re-importation into other tools.
"""


import bpy
import json
import os
import glob
import platform

#Check if this works at all
def load_config(config_file="config.json"):
    """Loads the configuration and selects paths based on the operating system."""
    with open(config_file, "r") as file:
        config = json.load(file)

    system = platform.system()
    
    if system == "Windows":
        return config["win_paths"]
    elif system == "Darwin":  # macOS
        return config["mac_paths"]
    else:
        raise ValueError("Unsupported operating system: " + system)

def get_socket_value(socket):
    """Get the default value of a socket, if present."""
    if hasattr(socket, 'default_value'):
        if isinstance(socket.default_value, (float, int, str)):
            return socket.default_value
        elif hasattr(socket.default_value, 'to_list'):
            return socket.default_value.to_list()
        else:
            return str(socket.default_value)
    return "No default value"

def get_node_parameters(node):
    """Extract all parameters from a node including dropdown values and socket default values."""
    parameters = {}
    if hasattr(node, 'blend_type'):  # e.g., for Mix nodes
        parameters["Blend Type"] = node.blend_type
    if hasattr(node, 'operation'):  # e.g., for Math nodes
        parameters["Operation"] = node.operation
    if hasattr(node, 'color_ramp'):  # e.g., for ColorRamp nodes
        parameters["ColorRamp"] = [
            {"Position": el.position, "Color": el.color[:]}
            for el in node.color_ramp.elements
        ]
    if hasattr(node, 'data_type'):
        parameters["Data Type"] = node.data_type
    if hasattr(node, 'vector_type'):
        parameters["Vector Type"] = node.vector_type
    if hasattr(node, 'space'):
        parameters["Space"] = node.space
    if node.type == 'TEX_IMAGE':
        if node.image is not None:  # Check if image is loaded
            parameters["Image Path"] = bpy.path.abspath(node.image.filepath)
            parameters["Image Name"] = node.image.name
            parameters["Source"] = node.image.source
            parameters["Color Space"] = node.image.colorspace_settings.name
        else:
            parameters["Image"] = "No image loaded"
    if hasattr(node, 'projection'):
        parameters["Projection"] = node.projection
    if hasattr(node, 'interpolation'):
        parameters["Interpolation"] = node.interpolation
    if hasattr(node, 'extension'):
        parameters["Extension"] = node.extension
    if hasattr(node, 'use_alpha'):
        parameters["Alpha"] = node.use_alpha
    
    return parameters

def ungroup_node_tree(node_tree, parent_node_name=""):
    """Flattens a group node tree by exposing its internal nodes and links."""
    ungrouped_nodes = []
    ungrouped_edges = []

    if node_tree is None:
        return ungrouped_nodes, ungrouped_edges

    for node in node_tree.nodes:
        if node.type in ["FRAME", "REROUTE", "NOTE", "GROUP_INPUT", "GROUP_OUTPUT"]:
            continue

        # Create a unique name for the node
        node_name = f"{parent_node_name}.{node.name}" if parent_node_name else node.name

        # Extract node details
        ungrouped_nodes.append({
            'name': node_name,
            'type': node.type,
            'parameters': get_node_parameters(node),
            'inputs': [{'name': sock.name, 'is_linked': sock.is_linked, 'value': get_socket_value(sock) if not sock.is_linked else None} for sock in node.inputs],
            'outputs': [{'name': sock.name} for sock in node.outputs]
        })

        if node.type == 'GROUP' and hasattr(node, 'node_tree'):
            group_nodes, group_edges = ungroup_node_tree(node.node_tree, node_name)
            ungrouped_nodes.extend(group_nodes)
            ungrouped_edges.extend(group_edges)

    # Extract edges
    for link in node_tree.links:
        from_node = link.from_node
        to_node = link.to_node

        ungrouped_edges.append({
            'from_node': from_node.name,
            'from_type': from_node.type,
            'from_socket': link.from_socket.name,
            'to_node': to_node.name,
            'to_type': to_node.type,
            'to_socket': link.to_socket.name
        })

    return ungrouped_nodes, ungrouped_edges

def extract_material_data():
    """Extracts nodes and edges from all materials, ungrouping node groups."""
    material_data = {}

    for material in bpy.data.materials:
        if not material.use_nodes:
            continue

        nodes, edges = ungroup_node_tree(material.node_tree)
        material_data[material.name] = {'nodes': nodes, 'edges': edges}

    return material_data

def save_material_data_to_json(material_data, output_path):
    """Save material data to a JSON file."""
    with open(output_path, 'w') as file:
        json.dump(material_data, file, indent=4)
    print(f"Material data saved to {output_path}")

def process_blend_files_in_directory(blend_folder, output_directory):
    """Processes all .blend files in the specified folder."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    blend_files = glob.glob(os.path.join(blend_folder, "*.blend"))

    for blend_file in blend_files:
        print(f"Processing {blend_file}...")
        bpy.ops.wm.open_mainfile(filepath=blend_file)

        material_data = extract_material_data()
        output_file = os.path.join(output_directory, os.path.basename(blend_file).replace(".blend", "_materials.json"))

        save_material_data_to_json(material_data, output_file)

        bpy.ops.wm.read_factory_settings(use_empty=True)

if __name__ == "__main__":
    blend_folder = bpy.path.abspath("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects")  # Replace with actual path
    output_directory = bpy.path.abspath("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects/Results")  # Replace with actual path
    process_blend_files_in_directory(blend_folder, output_directory)
