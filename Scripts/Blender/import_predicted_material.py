#/Volumes/ProgramFiles/Apps/Blender_426.app/Contents/MacOS/Blender --python /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Blender/import_predicted_material.py
import bpy
import json
import sys
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary"

NODE_MAP_PATH = os.path.join(BASE_PATH, "id_to_node.json")
SOCKET_MAP_PATH = os.path.join(BASE_PATH, "id_to_socket.json")
NODE_TYPE_MAP_PATH = os.path.join(BASE_PATH, "node_type_map.py")

GRAPH_JSON_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Generated/predicted_material_graph.json"
MATERIAL_NAME = "Predicted_GNN_Material"

GRID_X = 260
GRID_Y = -180

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
with open(NODE_MAP_PATH, "r") as f:
    id_to_node = {int(k): v for k, v in json.load(f).items()}

with open(SOCKET_MAP_PATH, "r") as f:
    id_to_socket = {int(k): v for k, v in json.load(f).items()}

with open(GRAPH_JSON_PATH, "r") as f:
    graph_data = json.load(f)

node_sequence = graph_data["node_sequence"]
edges = graph_data["edges"]

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

    # Add edges
    for edge in edges:
        try:
            src_idx = edge["src_idx"]
            dst_idx = edge["dst_idx"]
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
# RUN
# ─────────────────────────────────────────────
create_material_from_prediction(MATERIAL_NAME)