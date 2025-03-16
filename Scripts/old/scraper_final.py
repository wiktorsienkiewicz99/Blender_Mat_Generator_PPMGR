import bpy
import json
import os

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
    
    return parameters

def extract_material_nodes():
    """Extracts all material nodes and their properties from the project."""
    material_data = {}

    # Iterate through all materials in the Blender project
    for material in bpy.data.materials:
        if material.use_nodes:
            nodes_info = []
            # Extract node information for materials that use nodes
            for node in material.node_tree.nodes:
                # Extract inputs and their values
                inputs_info = []
                for input_socket in node.inputs:
                    input_info = {
                        'name': input_socket.name,
                        'type': input_socket.type,
                        'is_linked': input_socket.is_linked,
                        'value': get_socket_value(input_socket) if not input_socket.is_linked else None
                    }
                    inputs_info.append(input_info)

                # Extract outputs
                outputs_info = [output.name for output in node.outputs]
                
                # Extract node-specific parameters
                node_parameters = get_node_parameters(node)
                
                # Collect node information
                node_info = {
                    'name': node.name,
                    'type': node.type,
                    'inputs': inputs_info,
                    'outputs': outputs_info,
                    'parameters': node_parameters  # Add node-specific parameters
                }
                nodes_info.append(node_info)
                
            # Add material and its nodes to the dictionary
            material_data[material.name] = nodes_info

    return material_data

def extract_material_edges():
    """Extracts all edges (connections) between nodes in the material node tree."""
    edges_data = {}

    # Iterate through all materials in the Blender project
    for material in bpy.data.materials:
        if material.use_nodes:
            edges_info = []
            # Iterate over all the links (edges) between nodes
            for link in material.node_tree.links:
                edge_info = {
                    'from_node': link.from_node.name,
                    'from_socket': link.from_socket.name,
                    'to_node': link.to_node.name,
                    'to_socket': link.to_socket.name
                }
                edges_info.append(edge_info)

            # Add material and its edges to the dictionary
            edges_data[material.name] = edges_info

    return edges_data



def save_combined_material_data_to_json(nodes_data, edges_data, output_path):
    """Save both material nodes and edges data to a single JSON file."""
    combined_data = {
        'nodes': nodes_data,
        'edges': edges_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined material data (nodes and edges) saved to {output_path}")

# Specify the output directory and file name
output_directory = bpy.path.abspath("//Results/database")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_file = os.path.join(output_directory, "combined_materials_data.json")

# Extract material nodes and edges
nodes_data = extract_material_nodes()
edges_data = extract_material_edges()

# Save combined data (nodes and edges) to one JSON file
save_combined_material_data_to_json(nodes_data, edges_data, output_file)
