import bpy
import json
import os
import glob

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

def extract_edges_from_group(node_tree, edges_info, parent_node_name=""):
    """Extracts edges from a group node and adds them to the edges_info list."""
    if node_tree is None:
        # If the group node has no internal node tree, skip it
        return
    
    for link in node_tree.links:
        from_node_name = parent_node_name + "." + link.from_node.name if parent_node_name else link.from_node.name
        to_node_name = parent_node_name + "." + link.to_node.name if parent_node_name else link.to_node.name

        edge_info = {
            'from_node': from_node_name,
            'from_socket': link.from_socket.name,
            'to_node': to_node_name,
            'to_socket': link.to_socket.name
        }
        edges_info.append(edge_info)


def extract_material_edges():
    """Extracts all edges (connections) between nodes in the material node tree."""
    edges_data = {}

    # Iterate through all materials in the Blender project
    for material in bpy.data.materials:
        if material.use_nodes:
            edges_info = []
            # Iterate over all the links (edges) between nodes
            for link in material.node_tree.links:
                if link.from_node.type in ["GROUP_INPUT", "GROUP_OUTPUT"] or link.to_node.type in ["GROUP_INPUT", "GROUP_OUTPUT"]:
                    continue  # Skip group input/output

                if link.from_node.type == 'GROUP' and hasattr(link.from_node, 'node_tree'):
                    # Handle edges inside a group (from node group)
                    extract_edges_from_group(link.from_node.node_tree, edges_info, link.from_node.name)
                if link.to_node.type == 'GROUP' and hasattr(link.to_node, 'node_tree'):
                    # Handle edges inside a group (to node group)
                    extract_edges_from_group(link.to_node.node_tree, edges_info, link.to_node.name)

                # Standard edge for non-group nodes
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


def extract_nodes_from_group(node_tree, nodes_info, parent_node_name=""):
    """Extracts nodes from a group node and adds them to the nodes_info list."""
    if node_tree is None:
        # If the group node has no internal node tree, skip it
        return

    for node in node_tree.nodes:
        if node.type in ["FRAME", "REROUTE", "NOTE", "GROUP_INPUT", "GROUP_OUTPUT"]:
            # Skip these node types
            continue

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

        # Create a unique name for the node by prepending the group name
        node_name = parent_node_name + "." + node.name if parent_node_name else node.name

        # Collect node information
        node_info = {
            'name': node_name,
            'type': node.type,
            'inputs': inputs_info,
            'outputs': outputs_info,
            'parameters': node_parameters
        }
        nodes_info.append(node_info)

        # If the node is a group, recurse into it
        if node.type == 'GROUP' and hasattr(node, 'node_tree'):
            extract_nodes_from_group(node.node_tree, nodes_info, node_name)


def extract_material_nodes():
    """Extracts all material nodes and their properties from the project."""
    material_data = {}

    # Iterate through all materials in the Blender project
    for material in bpy.data.materials:
        if material.use_nodes:
            nodes_info = []
            # Extract node information for materials that use nodes
            for node in material.node_tree.nodes:
                if node.type in ["FRAME", "REROUTE", "NOTE", "GROUP_INPUT", "GROUP_OUTPUT"]:
                    # Skip these node types
                    continue

                if node.type == 'GROUP' and hasattr(node, 'node_tree'):
                    # If it's a node group, extract the internal nodes
                    extract_nodes_from_group(node.node_tree, nodes_info, node.name)
                else:
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
                        'parameters': node_parameters
                    }
                    nodes_info.append(node_info)

     
            material_data[material.name] = nodes_info

    return material_data


def save_combined_material_data_to_json(nodes_data, edges_data, output_path):
    """Save both material nodes and edges data to a single JSON file."""
    combined_data = {
        'nodes': nodes_data,
        'edges': edges_data
    }

    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined material data (nodes and edges) saved to {output_path}")


def process_blend_files_in_directory(blend_folder, output_directory):
    """Processes all .blend files in the specified folder."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

 
    blend_files = glob.glob(os.path.join(blend_folder, "*.blend"))

    for blend_file in blend_files:
        # Open the .blend file
        print(f"Processing {blend_file}...")
        bpy.ops.wm.open_mainfile(filepath=blend_file)

        # Extract material nodes and edges
        nodes_data = extract_material_nodes()
        edges_data = extract_material_edges()

        output_file = os.path.join(output_directory, os.path.basename(blend_file).replace(".blend", "_materials.json"))

        save_combined_material_data_to_json(nodes_data, edges_data, output_file)

        bpy.ops.wm.read_factory_settings(use_empty=True)

# Main function to process all Blender files
if __name__ == "__main__":

    blend_folder = bpy.path.abspath("C:/Users/hyperbook/Desktop/PPMGR/Projects")  # Replace with actual path
    output_directory = bpy.path.abspath("C:/Users/hyperbook/Desktop/PPMGR/Projects/Results")  # Replace with actual path
    process_blend_files_in_directory(blend_folder, output_directory)
