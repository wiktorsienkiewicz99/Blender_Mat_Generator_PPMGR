#!/usr/bin/env python3
"""
Blender script to import a predicted material graph.

This script creates a material in Blender based on a predicted material graph stored in a JSON file.
It can be run from the command line using Blender's Python API.

Usage:
    blender [blender_options] --python import_predicted_material.py [-- [material_name] [graph_json_path]]

Arguments:
    material_name (optional): Name of the material to create (default: "Predicted_GNN_Material")
    graph_json_path (optional): Path to the predicted material graph JSON file
                               (default: "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Generated/predicted_material_graph.json")

Examples:
    # Run with default arguments
    /Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender --python /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Blender/import_predicted_material.py

    # Run with custom material name
    /Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender --python /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Blender/import_predicted_material.py -- "Custom_Material"

    # Run with custom material name and graph JSON path
    /Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender --python /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Blender/import_predicted_material.py -- "Custom_Material" "/path/to/graph.json"

    # Run in background mode (no UI)
    /Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender --background --python /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Blender/import_predicted_material.py

Notes:
    - The script will create a preview sphere with the material applied if one doesn't already exist.
    - The script will exit with a non-zero status code if there's an error.
"""
import bpy
import json
import sys
import os
import argparse
#NODE WRANGLER
# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary"

NODE_MAP_PATH = os.path.join(BASE_PATH, "id_to_node.json")
SOCKET_MAP_PATH = os.path.join(BASE_PATH, "id_to_socket.json")
NODE_TYPE_MAP_PATH = os.path.join(BASE_PATH, "node_type_map.py")

DEFAULT_GRAPH_JSON_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Generated/predicted_material_graph.json"
DEFAULT_MATERIAL_NAME = "Predicted_GNN_Material"

GRID_X = 260
GRID_Y = -180

# Parse command line arguments
def parse_arguments():
    # Get all args after "--" (Blender passes its own args before that)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Import a predicted material graph into Blender")
    parser.add_argument("material_name", nargs="?", default=DEFAULT_MATERIAL_NAME, 
                        help="Name of the material to create")
    parser.add_argument("graph_json_path", nargs="?", default=DEFAULT_GRAPH_JSON_PATH,
                        help="Path to the predicted material graph JSON file")

    # Parse only known args to avoid errors with Blender's args
    args, _ = parser.parse_known_args(argv)
    return args

# Get arguments
args = parse_arguments()
MATERIAL_NAME = args.material_name
GRAPH_JSON_PATH = args.graph_json_path

# ─────────────────────────────────────────────
# LOAD NODE_TYPE_MAP
# ─────────────────────────────────────────────
NODE_TYPE_MAP = {}
if os.path.exists(NODE_TYPE_MAP_PATH):
    exec(open(NODE_TYPE_MAP_PATH).read(), {}, NODE_TYPE_MAP)
    NODE_TYPE_MAP = NODE_TYPE_MAP.get("NODE_TYPE_MAP", {})
else:
    print("MISSING node_type_map.py not found!")

# ─────────────────────────────────────────────
# LOAD MAPPINGS AND GRAPH DATA
# ─────────────────────────────────────────────
try:
    print(f"Loading node map from: {NODE_MAP_PATH}")
    with open(NODE_MAP_PATH, "r") as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}
    print(f"Loaded {len(id_to_node)} node mappings")

    print(f"Loading socket map from: {SOCKET_MAP_PATH}")
    with open(SOCKET_MAP_PATH, "r") as f:
        id_to_socket = {int(k): v for k, v in json.load(f).items()}
    print(f"Loaded {len(id_to_socket)} socket mappings")

    print(f"Loading graph data from: {GRAPH_JSON_PATH}")
    with open(GRAPH_JSON_PATH, "r") as f:
        graph_data = json.load(f)

    node_sequence = graph_data["node_sequence"]
    edges = graph_data["edges"]

    # Check if parameters exist in the graph data
    if "parameters" not in graph_data:
        print("WARNING: No parameters found in graph data. Nodes will use default parameter values.")

    # Check if textures exist in the graph data
    has_textures = "textures" in graph_data and "paths" in graph_data["textures"]
    if has_textures:
        print(f"Found texture paths in graph data: {list(graph_data['textures']['paths'].keys())}")
    else:
        print("No textures found in graph data. IMAGE TEX nodes will use default textures.")

    print(f"Loaded graph with {len(node_sequence)} nodes and {len(edges)} edges")
except FileNotFoundError as e:
    print(f"ERROR: File not found: {e.filename}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in file: {e}")
    sys.exit(1)
except KeyError as e:
    print(f"ERROR: Missing key in graph data: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error loading data: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# BUILD MATERIAL IN BLENDER
# ─────────────────────────────────────────────
def create_material_from_prediction(name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create nodes based on node_sequence
    node_objects = []
    for i, node_id in enumerate(node_sequence):
        short_type = id_to_node.get(node_id, "VALUE")
        blender_type = NODE_TYPE_MAP.get(short_type, "ShaderNodeValue")

        try:
            new_node = nodes.new(type=blender_type)
        except:
            print(f"WARNING: Failed to create node type '{blender_type}', fallback to ShaderNodeValue")
            new_node = nodes.new(type="ShaderNodeValue")

        new_node.location = (i * GRID_X, (i % 3) * GRID_Y)
        new_node.label = f"{short_type}_{i}"
        node_objects.append(new_node)

    # Apply parameters to nodes if available
    if "parameters" in graph_data:
        print("\nApplying parameters to nodes:")
        for idx, params_data in graph_data["parameters"].items():
            idx = int(idx)  # Convert string index to int
            if idx < len(node_objects):
                node = node_objects[idx]
                node_type = params_data.get("node_type", "")
                params = params_data.get("params", {})

                print(f"  Node {idx} [{node_type}]:")
                for param_name, param_value in params.items():
                    try:
                        # Find the input socket or property with this name
                        if param_name in node.inputs:
                            # Input socket parameter
                            socket = node.inputs[param_name]
                            if hasattr(socket, "default_value"):
                                # Handle different socket types
                                if isinstance(socket.default_value, float):
                                    socket.default_value = float(param_value)
                                elif isinstance(socket.default_value, int):
                                    socket.default_value = int(param_value)
                                elif isinstance(socket.default_value, bool):
                                    socket.default_value = bool(param_value)
                                elif isinstance(socket.default_value, str):
                                    socket.default_value = str(param_value)
                                elif hasattr(socket.default_value, "__len__"):
                                    # Vector, Color, etc.
                                    if len(socket.default_value) == 4 and isinstance(param_value, (int, float)):
                                        # Color with alpha - set all channels to same value
                                        socket.default_value = [float(param_value)] * 4
                                    elif len(socket.default_value) == 3 and isinstance(param_value, (int, float)):
                                        # Vector3 or RGB - set all channels to same value
                                        socket.default_value = [float(param_value)] * 3
                                    elif len(socket.default_value) == 2 and isinstance(param_value, (int, float)):
                                        # Vector2 - set all channels to same value
                                        socket.default_value = [float(param_value)] * 2
                                print(f"    {param_name} = {socket.default_value}")
                        elif param_name == "Image" and node.bl_idname == "ShaderNodeTexImage":
                            # Special handling for image textures
                            try:
                                # Load the image
                                image_path = param_value
                                image_name = os.path.basename(image_path)

                                # Check if the image is already loaded
                                if image_name in bpy.data.images:
                                    image = bpy.data.images[image_name]
                                else:
                                    # Load the image
                                    image = bpy.data.images.load(image_path)

                                # Assign the image to the node
                                node.image = image
                                print(f"    Loaded image: {image_path}")
                            except Exception as e:
                                print(f"    WARNING: Failed to load image {param_value}: {e}")
                        elif hasattr(node, param_name):
                            # Node property
                            setattr(node, param_name, param_value)
                            print(f"    {param_name} = {param_value}")
                        else:
                            # Try to find a property with a similar name
                            for prop_name in dir(node):
                                if prop_name.lower() == param_name.lower() or prop_name.lower().replace("_", "") == param_name.lower().replace("_", ""):
                                    setattr(node, prop_name, param_value)
                                    print(f"    {prop_name} = {param_value}")
                                    break
                    except Exception as e:
                        print(f"    WARNING: Failed to set parameter {param_name}: {e}")

    # Try to find output material node and auto-connect
    output_nodes = [n for n in node_objects if n.bl_idname == "ShaderNodeOutputMaterial"]
    if output_nodes:
        output_node = output_nodes[0]
        try:
            # Connect last non-output node to surface
            for n in reversed(node_objects):
                if n != output_node and n.outputs:
                    links.new(n.outputs[0], output_node.inputs.get("Surface"))
                    break
        except Exception as e:
            print(f"WARNING: Cannot auto-connect to Output node: {e}")

    # Track Value nodes and their connections
    value_node_connections = {}

    # First pass: identify Value nodes and their connections
    for edge in edges:
        try:
            src_idx = edge["src_idx"]
            dst_idx = edge["dst_idx"]
            src_socket_id = edge["src_socket"]
            dst_socket_id = edge["dst_socket"]

            # Check if source is a Value node
            if src_idx < len(node_sequence) and id_to_node.get(node_sequence[src_idx], "") == "VALUE":
                src_node = node_objects[src_idx]
                dst_node = node_objects[dst_idx]

                dst_socket_name = id_to_socket.get(dst_socket_id, "Color")
                dst_sock = next((s for s in dst_node.inputs if s.name == dst_socket_name), None)

                if dst_sock and hasattr(src_node, "outputs") and len(src_node.outputs) > 0:
                    # Store this connection for later processing
                    if src_idx not in value_node_connections:
                        value_node_connections[src_idx] = []

                    value_node_connections[src_idx].append({
                        "dst_node": dst_node,
                        "dst_socket": dst_sock,
                        "dst_socket_name": dst_socket_name
                    })
        except Exception as e:
            print(f"WARNING: Failed to process Value node connection: {e}")

    # Second pass: apply Value node values directly to destination sockets
    print("\nApplying Value node values directly to parameters:")
    for src_idx, connections in value_node_connections.items():
        try:
            value_node = node_objects[src_idx]

            # Get the value from the Value node
            if hasattr(value_node, "outputs") and len(value_node.outputs) > 0 and hasattr(value_node.outputs[0], "default_value"):
                value = value_node.outputs[0].default_value

                print(f"  Value node {src_idx} = {value}")

                # Apply this value to all connected sockets
                for conn in connections:
                    dst_node = conn["dst_node"]
                    dst_sock = conn["dst_socket"]
                    dst_socket_name = conn["dst_socket_name"]

                    if hasattr(dst_sock, "default_value"):
                        # Handle different socket types
                        if isinstance(dst_sock.default_value, float):
                            dst_sock.default_value = float(value)
                        elif isinstance(dst_sock.default_value, int):
                            dst_sock.default_value = int(value)
                        elif isinstance(dst_sock.default_value, bool):
                            dst_sock.default_value = bool(value)
                        elif hasattr(dst_sock.default_value, "__len__"):
                            # Vector, Color, etc.
                            if len(dst_sock.default_value) == 4 and isinstance(value, (int, float)):
                                # Color with alpha - set all channels to same value
                                dst_sock.default_value = [float(value)] * 4
                            elif len(dst_sock.default_value) == 3 and isinstance(value, (int, float)):
                                # Vector3 or RGB - set all channels to same value
                                dst_sock.default_value = [float(value)] * 3
                            elif len(dst_sock.default_value) == 2 and isinstance(value, (int, float)):
                                # Vector2 - set all channels to same value
                                dst_sock.default_value = [float(value)] * 2

                        print(f"    Applied to {dst_node.name}.{dst_socket_name} = {dst_sock.default_value}")
        except Exception as e:
            print(f"WARNING: Failed to apply Value node {src_idx}: {e}")

    # Add edges (excluding Value node connections that we've handled)
    for edge in edges:
        try:
            src_idx = edge["src_idx"]
            dst_idx = edge["dst_idx"]

            # Skip Value node connections that we've already handled
            if src_idx in value_node_connections:
                continue

            src_socket_id = edge["src_socket"]
            dst_socket_id = edge["dst_socket"]

            src_node = node_objects[src_idx]
            dst_node = node_objects[dst_idx]

            src_socket_name = id_to_socket.get(src_socket_id, "Color")
            dst_socket_name = id_to_socket.get(dst_socket_id, "Color")

            src_sock = next((s for s in src_node.outputs if s.name == src_socket_name), None)
            dst_sock = next((s for s in dst_node.inputs if s.name == dst_socket_name), None)

            if src_sock and dst_sock:
                links.new(src_sock, dst_sock)
            else:
                print(f"WARNING: Socket not found: {src_socket_name} → {dst_socket_name}")
        except Exception as e:
            print(f"WARNING: Failed to link edge: {e}")

    print(f"Created material '{name}' with {len(node_objects)} nodes and {len(edges)} edges.")
    return mat


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def main():
    try:
        print(f"\n{'='*50}")
        print(f"CREATING MATERIAL: {MATERIAL_NAME}")
        print(f"FROM GRAPH: {GRAPH_JSON_PATH}")
        print(f"{'='*50}\n")

        # Create the material
        material = create_material_from_prediction(MATERIAL_NAME)

        # Create a simple object to apply the material to (optional)
        if not bpy.data.objects.get("Material_Preview_Sphere"):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
            preview_obj = bpy.context.active_object
            preview_obj.name = "Material_Preview_Sphere"

            # Apply the material
            if preview_obj.data.materials:
                preview_obj.data.materials[0] = material
            else:
                preview_obj.data.materials.append(material)

            print(f"Created preview sphere with material '{MATERIAL_NAME}'")

        print(f"\n{'='*50}")
        print(f"MATERIAL CREATION SUCCESSFUL")
        print(f"{'='*50}\n")

        return 0
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR: Failed to create material: {e}")
        print(f"{'='*50}\n")
        return 1

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    exit_code = main()
    # In Blender scripts, sys.exit() might not work as expected
    # So we just print the exit code
    if exit_code != 0:
        print(f"Script failed with exit code {exit_code}")
    else:
        print("Script completed successfully")
