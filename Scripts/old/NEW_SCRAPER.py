import bpy
import json
import os

def get_socket_default_value(socket):
    """Get the default value of a socket, if present."""
    if hasattr(socket, 'default_value'):
        if isinstance(socket.default_value, (float, int, str)):
            return socket.default_value
        elif hasattr(socket.default_value, 'to_list'):
            return socket.default_value.to_list()
        else:
            return str(socket.default_value)
    return "No default value"

def get_linked_socket_info(socket):
    """Retrieve linked information for a given socket."""
    links_info = []
    for link in socket.links:
        links_info.append({
            "Target Node": link.to_node.name,
            "Target Socket": link.to_socket.name
        })
    return links_info if links_info else "No connections"

def extract_edges_from_node(node):
    """Extract connections (edges) from the node's output sockets."""
    edges = {}
    for output_socket in node.outputs:
        if output_socket.is_linked:
            edges[output_socket.name] = get_linked_socket_info(output_socket)
        else:
            edges[output_socket.name] = "No connections"
    return edges

def get_node_parameters(node):
    """Extract all parameters from a node including dropdown values and socket default values."""
    parameters = {}

    for input_socket in node.inputs:
        if input_socket.is_linked:
            parameters[input_socket.name] = {
                "Linked To": get_linked_socket_info(input_socket)
            }
        else:
            parameters[input_socket.name] = get_socket_default_value(input_socket)
    
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

def extract_material_data(material):
    """Extract nodes, parameters, and edges for a given material separately."""
    material_data = {
        "Material": material.name,
        "Nodes": [],
        "Edges": [],
        "Parameters": []
    }
    
    if material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            # Extract node info
            material_data["Nodes"].append({
                "Node": node.name,
                "Type": node.type
            })
            
            # Extract node parameters
            node_params = get_node_parameters(node)
            for param_name, param_value in node_params.items():
                material_data["Parameters"].append({
                    "Node": node.name,
                    "Parameter": param_name,
                    "Value": param_value
                })
            
            # Extract edges
            node_edges = extract_edges_from_node(node)
            for edge_name, edge_info in node_edges.items():
                material_data["Edges"].append({
                    "Node": node.name,
                    "Edge": edge_name,
                    "Connections": edge_info
                })
    
    return material_data

def extract_all_materials():
    """Extract all material data from the current Blender project, separating nodes, edges, and parameters."""
    all_materials_data = []
    
    for material in bpy.data.materials:
        material_data = extract_material_data(material)
        all_materials_data.append(material_data)
    
    return all_materials_data

def save_material_data_to_json(materials_data, output_path):
    """Save extracted material data to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(materials_data, f, indent=4)
    print(f"Material data saved to {output_path}")

# Specify the output directory and file name
output_directory = bpy.path.abspath("//Results/materials_output/")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_file = os.path.join(output_directory, "materials_data_separated.json")

# Extract materials and save the data
materials_data = extract_all_materials()
save_material_data_to_json(materials_data, output_file)
