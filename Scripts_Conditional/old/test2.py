import bpy

def get_parameter_value(param):
    """Helper function to get the value of a parameter."""
    if isinstance(param, bpy.types.NodeSocket):
        if hasattr(param, 'default_value'):
            return param.default_value
        else:
            return "No default value"
    return param

def print_material_nodes():
    # Iterate over all materials in the Blender scene
    for material in bpy.data.materials:
        # Check if the material uses nodes
        if material.use_nodes:
            print(f"Material: {material.name}")
            # Access the material's node tree
            node_tree = material.node_tree
            # Iterate over all nodes in the node tree
            for node in node_tree.nodes:
                print(f"  Node Type: {node.type}")
                # Print parameters of the node
                print(f"    Parameters:")
                for param_name in dir(node):
                    if not param_name.startswith("__") and not callable(getattr(node, param_name)):
                        param_value = getattr(node, param_name)
                        param_value = get_parameter_value(param_value)
                        if isinstance(param_value, (int, float, str, bool, tuple, list)):
                            print(f"      {param_name}: {param_value}")
                        elif isinstance(param_value, bpy.types.ID):
                            print(f"      {param_name}: {param_value.name}")
                        else:
                            print(f"      {param_name}: (nested values or complex type)")

            # Iterate over all links (edges) in the node tree
            print(f"    Edges:")
            for link in node_tree.links:
                print(f"      {link.from_node.name} ({link.from_socket.name}) --> {link.to_node.name} ({link.to_socket.name})")
        else:
            print(f"Material: {material.name} (no nodes)")

# Run the function to print nodes of all materials
print_material_nodes()
