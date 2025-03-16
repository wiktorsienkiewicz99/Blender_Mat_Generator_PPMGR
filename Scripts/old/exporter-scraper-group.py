import bpy
import os
import json

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

def get_linked_socket_info(socket, group_map=None):
    """Retrieve linked information for a given socket, resolving group nodes if necessary."""
    links_info = []
    for link in socket.links:
        target_node = link.to_node
        target_socket = link.to_socket

        # If the target node is in the group_map, resolve it
        if group_map and target_node.name in group_map:
            target_node_name = group_map[target_node.name]
        else:
            target_node_name = target_node.name

        links_info.append({
            "Target Node": target_node_name,
            "Target Socket": target_socket.name
        })
    return links_info if links_info else "No connections"

def break_group_node(node, group_map):
    """Recursively break down group nodes into individual nodes, fully flattening the group."""
    group_nodes_info = []
    if node.node_tree is None:
        return group_nodes_info  # Return empty list if there's no node tree

    # Iterate through each node inside the group
    for inner_node in node.node_tree.nodes:
        if inner_node.type in ['GROUP_INPUT', 'GROUP_OUTPUT']:
            # Skip Group Input and Group Output nodes but maintain their links for rewiring
            continue
        node_info = get_node_info(inner_node, group_map)
        if node_info:
            group_map[inner_node.name] = inner_node.name  # Update map to handle links properly
            group_nodes_info.append(node_info)

    return group_nodes_info

def get_node_info(node, group_map=None):
    """Extract detailed information about each node, excluding 'GROUP_INPUT' and 'GROUP_OUTPUT'."""
    if node.type in ['GROUP_INPUT', 'GROUP_OUTPUT', 'FRAME']:
        return None  # Ignore these nodes

    node_info = {
        "Node": node.name,
        "Type": node.type,
        "Parameters": {},
        "Outputs": {}
    }

    # Capture input sockets (handle linked and unlinked sockets)
    for input_socket in node.inputs:
        if input_socket.is_linked:
            node_info["Parameters"][input_socket.name] = {
                "Linked To": get_linked_socket_info(input_socket, group_map)
            }
        else:
            value = get_socket_default_value(input_socket)
            node_info["Parameters"][input_socket.name] = value

    # Capture output sockets (handle linked and unlinked sockets)
    for output_socket in node.outputs:
        if output_socket.is_linked:
            node_info["Outputs"][output_socket.name] = get_linked_socket_info(output_socket, group_map)
        else:
            node_info["Outputs"][output_socket.name] = "No connections"

    # Handle specific node types with extra attributes (e.g., Texture, Shader)
    if node.type == 'TEX_IMAGE':
        if node.image:
            node_info["Parameters"]["Image Path"] = bpy.path.abspath(node.image.filepath)
            node_info["Parameters"]["Image Name"] = node.image.name
            node_info["Parameters"]["Source"] = node.image.source
            node_info["Parameters"]["Color Space"] = node.image.colorspace_settings.name
        else:
            node_info["Parameters"]["Image"] = "No image loaded"
        node_info["Parameters"]["Projection"] = node.projection
        node_info["Parameters"]["Interpolation"] = node.interpolation
        node_info["Parameters"]["Extension"] = node.extension
        if hasattr(node, 'use_alpha'):
            node_info["Parameters"]["Alpha"] = node.use_alpha

    # Handle specific node types
    if node.type == 'MIX':
        if hasattr(node, 'data_type'):
            node_info["Parameters"]["Data Type"] = node.data_type
        if hasattr(node, 'blend_type'):
            node_info["Parameters"]["Blend Type"] = node.blend_type

    if node.type == 'MAPPING':
        if hasattr(node, 'vector_type'):
            node_info["Parameters"]["Vector Type"] = node.vector_type

    if node.type == 'DISPLACEMENT':
        if hasattr(node, 'space'):
            node_info["Parameters"]["Space"] = node.space

    if node.type == 'NORMAL_MAP':
        if hasattr(node, 'space'):
            node_info["Parameters"]["Space"] = node.space

    # Recursively handle group nodes and flatten them
    if node.type == 'GROUP':
        group_nodes = break_group_node(node, group_map)
        return group_nodes  # Return the flattened list of internal nodes

    return node_info

# Extract all material data
def extract_material_data():
    materials_data = []
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            material_nodes = []
            group_map = {}  # To track connections inside group nodes
            for node in mat.node_tree.nodes:
                node_data = get_node_info(node, group_map)
                if node_data:
                    # If the node is a flattened group (list of nodes), extend the material_nodes list
                    if isinstance(node_data, list):
                        material_nodes.extend(node_data)
                    else:
                        material_nodes.append(node_data)
            
            materials_data.append({
                "Material": mat.name,
                "Node Tree": material_nodes
            })
    return materials_data

# File path setup and save
def save_material_data():
    materials_data = extract_material_data()

    blender_file_name = bpy.path.basename(bpy.data.filepath)
    base_name = os.path.splitext(blender_file_name)[0]

    output_directory = bpy.path.abspath("//Results/database")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_path = os.path.join(output_directory, f"{base_name}_materials_database_test_group.json")

    # Save the extracted data to a JSON file
    try:
        with open(file_path, 'w') as file:
            json.dump(materials_data, file, indent=4)
        print(f"Materials data successfully written to {file_path}")
    except IOError as e:
        print(f"Error writing data to file: {e}")

# Run the data extraction
save_material_data()
